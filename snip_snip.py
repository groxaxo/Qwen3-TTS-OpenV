import os
import tempfile
import wave
from typing import Dict

import gradio as gr
import librosa
import matplotlib.pyplot as plt
import numpy as np

def _normalize_audio(data: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(data)) if data.size else 0.0
    if peak <= 0:
        return data.astype(np.float32, copy=False)
    return (data / peak).astype(np.float32, copy=False)

def _audio_for_gradio(data: np.ndarray) -> np.ndarray:
    if data.ndim == 1:
        return data.astype(np.float32, copy=False)
    return data.T.astype(np.float32, copy=False)


def _write_wav_pcm16(path: str, sr: int, data: np.ndarray) -> None:
    if data.ndim == 1:
        pcm = np.clip(data, -1.0, 1.0)
        pcm = (pcm * 32767.0).astype(np.int16)
        channels = 1
    else:
        pcm = np.clip(data, -1.0, 1.0)
        pcm = (pcm * 32767.0).astype(np.int16).T
        channels = data.shape[0]

    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sr)
        wav_file.writeframes(pcm.tobytes())


def _collect_metadata(source_path: str, sr: int, data: np.ndarray) -> Dict[str, object]:
    channels = 1 if data.ndim == 1 else data.shape[0]
    sample_count = data.shape[-1]
    duration = sample_count / sr if sr else 0.0
    rms = float(np.sqrt(np.mean(np.square(data)))) if data.size else 0.0
    peak = float(np.max(np.abs(data))) if data.size else 0.0

    return {
        "source_file": os.path.basename(source_path),
        "source_extension": os.path.splitext(source_path)[1].lower() or "unknown",
        "source_size_bytes": os.path.getsize(source_path),
        "sample_rate_hz": sr,
        "channels": channels,
        "samples_per_channel": sample_count,
        "duration_seconds": round(duration, 4),
        "dtype": str(data.dtype),
        "rms": round(rms, 6),
        "peak_abs": round(peak, 6),
    }


def _build_spectrogram(data: np.ndarray, sr: int) -> np.ndarray:
    mono = data if data.ndim == 1 else np.mean(data, axis=0)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.specgram(mono, NFFT=1024, Fs=sr, noverlap=512, cmap="magma")
    ax.set_title("Spectrogram")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    fig.tight_layout()
    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    return rgb


def load_audio(audio_path: str):
    if not audio_path:
        return (
            gr.update(maximum=10, value=0),
            gr.update(maximum=10, value=1),
            None,
            None,
            None,
            None,
        )

    data, sr = librosa.load(audio_path, sr=None, mono=False)
    if data.ndim > 2:
        raise gr.Error("Only mono/stereo files are supported.")

    data = _normalize_audio(np.asarray(data, dtype=np.float32))
    metadata = _collect_metadata(audio_path, sr, data)
    duration = metadata["duration_seconds"]
    spectrogram = _build_spectrogram(data, sr)

    return (
        gr.update(maximum=duration, value=0),
        gr.update(maximum=duration, value=duration),
        gr.update(value=0, maximum=duration),
        gr.update(value=duration, maximum=duration),
        metadata,
        spectrogram,
        (sr, _audio_for_gradio(data)),
        {"sr": sr, "data": data, "source_path": audio_path},
    )


def sync_slider_to_inputs(start_time: float, end_time: float):
    return start_time, end_time


def sync_inputs_to_slider(start_time: float, end_time: float):
    return start_time, end_time


def slice_audio(audio_state: Dict[str, object], start_time: float, end_time: float):
    if not audio_state:
        raise gr.Error("Please upload an audio file first.")
    if start_time >= end_time:
        raise gr.Error("Start time must be less than end time.")

    sr = int(audio_state["sr"])
    data = np.asarray(audio_state["data"], dtype=np.float32)
    source_path = str(audio_state["source_path"])

    start_idx = int(start_time * sr)
    end_idx = int(end_time * sr)
    sliced = data[start_idx:end_idx] if data.ndim == 1 else data[:, start_idx:end_idx]
    if sliced.size == 0:
        raise gr.Error("Selected range produced an empty slice.")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
        wav_path = temp_wav.name
    _write_wav_pcm16(wav_path, sr, sliced)

    metadata = _collect_metadata(source_path, sr, sliced)
    metadata["slice_range_seconds"] = [round(start_time, 4), round(end_time, 4)]
    metadata["wav_export_path"] = wav_path
    spectrogram = _build_spectrogram(sliced, sr)

    return (sr, _audio_for_gradio(sliced)), wav_path, metadata, spectrogram


with gr.Blocks(title="Audio Slicer + Analyzer") as demo:
    gr.Markdown("# Audio Slicer + Analyzer")
    gr.Markdown(
        "Upload audio (WAV/MP3/FLAC/OGG/M4A), inspect metadata + spectrogram, "
        "slice any segment, and export the sliced result as WAV."
    )

    audio_state = gr.State()

    with gr.Row():
        with gr.Column():
            input_audio = gr.Audio(
                label="Upload Audio (common formats)",
                type="filepath",
            )
            with gr.Row():
                start_slider = gr.Slider(
                    minimum=0,
                    maximum=10,
                    value=0,
                    step=0.01,
                    label="Trim Start (seconds)",
                )
                end_slider = gr.Slider(
                    minimum=0,
                    maximum=10,
                    value=1,
                    step=0.01,
                    label="Trim End (seconds)",
                )
            with gr.Row():
                start_input = gr.Number(
                    value=0,
                    minimum=0,
                    maximum=10,
                    precision=3,
                    label="Trim Start Input",
                )
                end_input = gr.Number(
                    value=1,
                    minimum=0,
                    maximum=10,
                    precision=3,
                    label="Trim End Input",
                )
            slice_btn = gr.Button("Slice + Export WAV", variant="primary")

        with gr.Column():
            preview_audio = gr.Audio(label="Loaded Audio Preview", interactive=False)
            output_audio = gr.Audio(label="Sliced Audio Preview", interactive=False)
            wav_file = gr.File(label="WAV Export")

    with gr.Row():
        source_meta = gr.JSON(label="Source Audio Metadata")
        sliced_meta = gr.JSON(label="Sliced Audio Metadata")

    with gr.Row():
        source_spec = gr.Image(label="Source Spectrogram", type="numpy")
        sliced_spec = gr.Image(label="Sliced Spectrogram", type="numpy")

    input_audio.change(
        fn=load_audio,
        inputs=input_audio,
        outputs=[
            start_slider,
            end_slider,
            start_input,
            end_input,
            source_meta,
            source_spec,
            preview_audio,
            audio_state,
        ],
    )

    start_slider.change(
        fn=sync_slider_to_inputs,
        inputs=[start_slider, end_slider],
        outputs=[start_input, end_input],
    )
    end_slider.change(
        fn=sync_slider_to_inputs,
        inputs=[start_slider, end_slider],
        outputs=[start_input, end_input],
    )
    start_input.change(
        fn=sync_inputs_to_slider,
        inputs=[start_input, end_input],
        outputs=[start_slider, end_slider],
    )
    end_input.change(
        fn=sync_inputs_to_slider,
        inputs=[start_input, end_input],
        outputs=[start_slider, end_slider],
    )

    slice_btn.click(
        fn=slice_audio,
        inputs=[audio_state, start_input, end_input],
        outputs=[output_audio, wav_file, sliced_meta, sliced_spec],
    )

if __name__ == "__main__":
    demo.launch()
