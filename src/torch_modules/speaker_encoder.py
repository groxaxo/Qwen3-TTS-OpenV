import torch
import torch.nn as nn
import torch.nn.functional as F


def _reflect_pad_conv1d(x, weight, bias, dilation=1):
    kernel_size = weight.shape[2]
    effective_k = dilation * (kernel_size - 1)
    pad_left = effective_k // 2
    pad_right = effective_k - pad_left
    x = F.pad(x, (pad_left, pad_right), mode="reflect")
    return F.conv1d(x, weight.to(x.dtype), bias.to(x.dtype) if bias is not None else None, dilation=dilation)


class TDNNBlock(nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor, dilation: int = 1):
        super().__init__()
        self.dilation = int(dilation)
        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(_reflect_pad_conv1d(x, self.weight, self.bias, dilation=self.dilation))


class SEBlock(nn.Module):
    def __init__(self, conv1_w, conv1_b, conv2_w, conv2_b):
        super().__init__()
        self.register_buffer("conv1_w", conv1_w)
        self.register_buffer("conv1_b", conv1_b)
        self.register_buffer("conv2_w", conv2_w)
        self.register_buffer("conv2_b", conv2_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_mean = x.mean(dim=2, keepdim=True)
        x_mean = F.relu(_reflect_pad_conv1d(x_mean, self.conv1_w, self.conv1_b))
        x_mean = torch.sigmoid(_reflect_pad_conv1d(x_mean, self.conv2_w, self.conv2_b))
        return x * x_mean


class Res2NetBlock(nn.Module):
    def __init__(self, branches: nn.ModuleList, scale: int):
        super().__init__()
        self.branches = branches
        self.scale = int(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunks = torch.chunk(x, self.scale, dim=1)
        outputs = []
        output_part = None
        for i, chunk in enumerate(chunks):
            if i == 0:
                output_part = chunk
            elif i == 1:
                output_part = self.branches[i - 1](chunk)
            else:
                output_part = self.branches[i - 1](chunk + output_part)
            outputs.append(output_part)
        return torch.cat(outputs, dim=1)


class SERes2NetBlock(nn.Module):
    def __init__(self, tdnn1: TDNNBlock, res2net: Res2NetBlock, tdnn2: TDNNBlock, se: SEBlock):
        super().__init__()
        self.tdnn1 = tdnn1
        self.res2net = res2net
        self.tdnn2 = tdnn2
        self.se = se

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.tdnn1(x)
        x = self.res2net(x)
        x = self.tdnn2(x)
        x = self.se(x)
        return x + residual


class AttentiveStatPooling(nn.Module):
    def __init__(self, tdnn: TDNNBlock, conv_w: torch.Tensor, conv_b: torch.Tensor):
        super().__init__()
        self.tdnn = tdnn
        self.register_buffer("conv_w", conv_w)
        self.register_buffer("conv_b", conv_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eps = 1e-12
        batch, channels, timesteps = x.shape
        mask = torch.ones(batch, 1, timesteps, dtype=x.dtype, device=x.device)
        total = mask.sum(dim=2, keepdim=True)

        mean = (mask * x).sum(dim=2) / total.squeeze(2)
        std = torch.sqrt(((mask * (x - mean.unsqueeze(2))).pow(2)).sum(dim=2) / total.squeeze(2) + eps)

        mean_expanded = mean.unsqueeze(2).expand(-1, -1, timesteps)
        std_expanded = std.unsqueeze(2).expand(-1, -1, timesteps)
        attn_input = torch.cat([x, mean_expanded, std_expanded], dim=1)

        attn = self.tdnn(attn_input)
        attn = torch.tanh(attn)
        attn = _reflect_pad_conv1d(attn, self.conv_w, self.conv_b)
        attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(attn, dim=2)

        weighted_mean = (attn * x).sum(dim=2)
        weighted_std = torch.sqrt(((attn * (x - weighted_mean.unsqueeze(2))).pow(2)).sum(dim=2) + eps)
        return torch.cat([weighted_mean, weighted_std], dim=1).unsqueeze(2)


class SpeakerEncoder(nn.Module):
    """nn.Module wrapper for the ECAPA-TDNN speaker encoder."""

    def __init__(self, weights: dict, config: dict):
        super().__init__()

        self.enc_channels = config.get("enc_channels", [512, 512, 512, 512, 1536])
        self.enc_dilations = config.get("enc_dilations", [1, 2, 3, 4, 1])
        self.scale = config.get("enc_res2net_scale", 8)
        self._weight_dtype = weights["fc.weight"].dtype

        self.initial_tdnn = TDNNBlock(
            weights["blocks.0.conv.weight"],
            weights["blocks.0.conv.bias"],
            dilation=self.enc_dilations[0],
        )

        se_res2net_blocks = []
        for i in range(1, len(self.enc_channels) - 1):
            base = f"blocks.{i}"
            tdnn1 = TDNNBlock(weights[f"{base}.tdnn1.conv.weight"], weights[f"{base}.tdnn1.conv.bias"])
            res2net = Res2NetBlock(
                nn.ModuleList(
                    [
                        TDNNBlock(
                            weights[f"{base}.res2net_block.blocks.{r}.conv.weight"],
                            weights[f"{base}.res2net_block.blocks.{r}.conv.bias"],
                            dilation=self.enc_dilations[i],
                        )
                        for r in range(self.scale - 1)
                    ]
                ),
                scale=self.scale,
            )
            tdnn2 = TDNNBlock(weights[f"{base}.tdnn2.conv.weight"], weights[f"{base}.tdnn2.conv.bias"])
            se = SEBlock(
                weights[f"{base}.se_block.conv1.weight"],
                weights[f"{base}.se_block.conv1.bias"],
                weights[f"{base}.se_block.conv2.weight"],
                weights[f"{base}.se_block.conv2.bias"],
            )
            se_res2net_blocks.append(SERes2NetBlock(tdnn1, res2net, tdnn2, se))
        self.se_res2net_blocks = nn.ModuleList(se_res2net_blocks)

        self.mfa = TDNNBlock(weights["mfa.conv.weight"], weights["mfa.conv.bias"], dilation=self.enc_dilations[-1])
        self.asp = AttentiveStatPooling(
            TDNNBlock(weights["asp.tdnn.conv.weight"], weights["asp.tdnn.conv.bias"]),
            weights["asp.conv.weight"],
            weights["asp.conv.bias"],
        )
        self.register_buffer("fc_w", weights["fc.weight"])
        self.register_buffer("fc_b", weights["fc.bias"])

    def forward(self, mels: torch.Tensor) -> torch.Tensor:
        x = mels.transpose(1, 2).to(self._weight_dtype)

        hidden_list = []
        x = self.initial_tdnn(x)
        hidden_list.append(x)

        for block in self.se_res2net_blocks:
            x = block(x)
            hidden_list.append(x)

        x = torch.cat(hidden_list[1:], dim=1)
        x = self.mfa(x)
        x = self.asp(x)
        x = _reflect_pad_conv1d(x, self.fc_w, self.fc_b)
        return x.squeeze(-1)
