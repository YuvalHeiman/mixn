"""Audio I/O utilities.

- Listing input audio files
 - Loading stereo audio (WAV via soundfile, MP3 via ffmpeg CLI)
- Saving mastered output as WAV (safe cross-platform default)

Notes
-----
- We support .wav and .mp3 inputs. For output, we save WAV by default.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import json
import shutil
import subprocess
import numpy as np
import soundfile as sf

SUPPORTED_EXTS = {".wav", ".mp3"}
INPUT_DIR = Path("input_audio")
OUTPUT_DIR = Path("output_audio")


def ensure_directories() -> None:
    """Ensure input/output directories exist."""
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def list_audio_files() -> List[Path]:
    """List supported audio files in the `input_audio/` directory.

    Returns a sorted list of `Path`s.
    """
    ensure_directories()
    files = [p for p in INPUT_DIR.glob("*") if p.suffix.lower() in SUPPORTED_EXTS]
    return sorted(files)


def load_audio_stereo(file_path: Path) -> Tuple[np.ndarray, int]:
    """Load an audio file preserving native sample rate and channels.

    Strategy
    --------
    - Try soundfile first (works well for WAV and some compressed formats).
    - If that fails (common for MP3), fall back to pydub/ffmpeg.

    Returns
    -------
    audio : np.ndarray
        Stereo-like array shaped (channels, samples). Mono files are returned
        as shape (1, samples) for consistency.
    sr : int
        Sample rate.
    """
    # Prefer soundfile for WAV and formats it supports well
    try:
        data, sr = sf.read(str(file_path), always_2d=True, dtype="float32")
        # soundfile returns (samples, channels)
        return data.T, int(sr)
    except Exception:
        # If it fails (common for MP3), try ffmpeg decode
        pass

    # Ensure ffmpeg exists
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found in PATH. Install it (e.g., 'brew install ffmpeg') to read MP3 files."
        )

    # Probe sample rate and channels with ffprobe
    try:
        info = _ffprobe_stream_info(file_path)
        sr = info.get("sample_rate")
        ch = info.get("channels")
        if not sr or not ch:
            raise RuntimeError("Unable to determine audio stream parameters via ffprobe.")
        audio = _ffmpeg_decode_f32(file_path, sr, ch)
        return audio, sr
    except Exception as e:
        raise RuntimeError(f"Failed to decode audio via ffmpeg: {e}")


def _ffprobe_stream_info(file_path: Path) -> dict:
    """Return {'sample_rate': int, 'channels': int} using ffprobe JSON output."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate,channels",
        "-of",
        "json",
        str(file_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(res.stdout)
    streams = data.get("streams", [])
    if not streams:
        return {"sample_rate": None, "channels": None}
    s = streams[0]
    sr = int(s.get("sample_rate", 0)) if s.get("sample_rate") else None
    ch = int(s.get("channels", 0)) if s.get("channels") else None
    return {"sample_rate": sr, "channels": ch}


def _ffmpeg_decode_f32(file_path: Path, sr: int, channels: int) -> np.ndarray:
    """Decode audio to float32 PCM using ffmpeg and return (channels, samples)."""
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(file_path),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        str(channels),
        "-ar",
        str(sr),
        "pipe:1",
    ]
    res = subprocess.run(cmd, capture_output=True, check=True)
    raw = res.stdout
    if not raw:
        raise RuntimeError("ffmpeg returned no audio data")
    audio = np.frombuffer(raw, dtype=np.float32)
    if channels > 1:
        audio = audio.reshape((-1, channels)).T  # (channels, samples)
    else:
        audio = audio.reshape((1, -1))
    return audio


def build_output_path(original: Path) -> Path:
    """Compute the mastered output path in `output_audio/`.

    Always writes WAV to keep things simple and robust.
    """
    stem_name = original.stem
    out_name = f"{stem_name}_mastered.wav"
    return OUTPUT_DIR / out_name


def save_audio(path: Path, audio: np.ndarray, sr: int) -> None:
    """Save audio to disk as WAV.

    Parameters
    ----------
    path : Path
        Destination file path (should end with .wav).
    audio : np.ndarray
        Expected shape (channels, samples). If (samples,) is passed, it will
        be converted to mono (1, samples).
    sr : int
        Sample rate.
    """
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
    # soundfile expects shape (samples, channels)
    data = audio.T.astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, sr)
