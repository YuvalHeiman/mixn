"""DSP utilities: simple EQ, compression, limiting, LUFS normalization.

This module intentionally keeps implementations approachable and well-commented.
They are not meant to rival professional plugins, but to provide educational,
useful processing for a terminal app.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from scipy.signal import butter, sosfiltfilt

try:  # optional for loudness normalization
    import pyloudnorm as pyln
except Exception:  # pragma: no cover
    pyln = None  # type: ignore

from .types import AdjustOption


# ================================
# Basic filters
# ================================

def _butter_sos(filter_type: str, cutoff_hz: float | List[float], sr: int, order: int = 4):
    nyq = 0.5 * sr
    if isinstance(cutoff_hz, list):
        wn = [c / nyq for c in cutoff_hz]
    else:
        wn = cutoff_hz / nyq
    sos = butter(order, wn, btype=filter_type, output="sos")
    return sos


def lowpass(x: np.ndarray, sr: int, cutoff_hz: float, order: int = 4) -> np.ndarray:
    sos = _butter_sos("lowpass", cutoff_hz, sr, order)
    return sosfiltfilt(sos, x)


def highpass(x: np.ndarray, sr: int, cutoff_hz: float, order: int = 4) -> np.ndarray:
    sos = _butter_sos("highpass", cutoff_hz, sr, order)
    return sosfiltfilt(sos, x)


def bandpass(x: np.ndarray, sr: int, low_hz: float, high_hz: float, order: int = 4) -> np.ndarray:
    sos = _butter_sos("bandpass", [low_hz, high_hz], sr, order)
    return sosfiltfilt(sos, x)


# ================================
# Frequency-domain EQ (Gaussian peaking + soft shelves)
# ================================

def _build_eq_curve(freqs: np.ndarray, sr: int, bands: List[Dict[str, float]]) -> np.ndarray:
    """Construct a smooth EQ gain curve over the positive frequency bins.

    Each band dict should have keys:
        - type: "peak" | "low_shelf" | "high_shelf"
        - freq: center/cutoff frequency in Hz
        - gain_db: positive to boost, negative to cut
        - q: optional, controls bandwidth for "peak" (default 1.0)
    """
    curve_db = np.zeros_like(freqs)
    for b in bands:
        btype = b.get("type", "peak")
        f0 = float(b.get("freq", 1000.0))
        gain_db = float(b.get("gain_db", 0.0))
        q = float(b.get("q", 1.0))
        if gain_db == 0.0:
            continue
        if btype == "peak":
            # Gaussian in linear frequency with width from Q
            sigma = (f0 / max(q, 1e-6))
            sigma = max(sigma, 1.0)
            curve_db += gain_db * np.exp(-0.5 * ((freqs - f0) / sigma) ** 2)
        elif btype == "low_shelf":
            # Soft shelf using a smooth step below f0
            n = 4.0
            w = 1.0 / (1.0 + (freqs / max(f0, 1.0)) ** n)
            curve_db += gain_db * w
        elif btype == "high_shelf":
            n = 4.0
            w = 1.0 / (1.0 + (max(f0, 1.0) / np.maximum(freqs, 1.0)) ** n)
            curve_db += gain_db * w
    curve = 10 ** (curve_db / 20.0)
    return curve


def apply_eq_fft(x: np.ndarray, sr: int, bands: List[Dict[str, float]]) -> np.ndarray:
    """Apply a simple EQ in the frequency domain to a 1D signal.

    Parameters
    ----------
    x : np.ndarray
        Mono signal.
    sr : int
        Sample rate.
    bands : List[Dict[str, float]]
        List of band dicts as described in `_build_eq_curve`.
    """
    if len(bands) == 0:
        return x
    n = x.shape[-1]
    # Real FFT frequency bins
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    X = np.fft.rfft(x)
    curve = _build_eq_curve(freqs, sr, bands)
    Y = X * curve
    y = np.fft.irfft(Y, n=n)
    return y.astype(np.float32, copy=False)


# ================================
# Simple compressor and limiter
# ================================

def _db(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(x), eps))


def _rms_envelope(x: np.ndarray, sr: int, window_ms: float = 10.0) -> np.ndarray:
    """Compute a simple RMS envelope using a Hann window convolution."""
    win_len = max(1, int(sr * window_ms / 1000.0))
    if win_len % 2 == 0:
        win_len += 1
    window = np.hanning(win_len)
    window = window / np.sum(window)
    # Power average then sqrt
    power = np.convolve(x**2, window, mode="same")
    return np.sqrt(power + 1e-12)


def compress(
    x: np.ndarray,
    sr: int,
    threshold_db: float = -18.0,
    ratio: float = 2.0,
    attack_ms: float = 10.0,
    release_ms: float = 100.0,
) -> np.ndarray:
    """Very simple feed-forward compressor on a mono signal.

    Uses an RMS detector and static curve to compute gain reduction.
    """
    env = _rms_envelope(x, sr, window_ms=attack_ms)
    level_db = _db(env)
    over_db = np.maximum(level_db - threshold_db, 0.0)
    gain_db = -over_db * (1.0 - 1.0 / max(ratio, 1.0))
    # Smooth gain with release window
    rel_env = _rms_envelope(gain_db, sr, window_ms=release_ms)
    gain = 10 ** (rel_env / 20.0)
    y = x * gain
    return y.astype(np.float32, copy=False)


def limiter(x: np.ndarray, sr: int, ceiling_db: float = -1.0) -> np.ndarray:
    """Hard-ish limiter via high-ratio compression near the ceiling."""
    # Compute instantaneous dBFS level using RMS with short window
    env = _rms_envelope(x, sr, window_ms=5.0)
    level_db = _db(env)
    threshold_db = ceiling_db
    over_db = np.maximum(level_db - threshold_db, 0.0)
    # High ratio
    gain_db = -over_db * 0.99
    gain = 10 ** (gain_db / 20.0)
    y = x * gain
    # safety clip
    y = np.clip(y, -1.0, 1.0)
    return y.astype(np.float32, copy=False)


# ================================
# Loudness normalization
# ================================

def normalize_to_lufs(stereo: np.ndarray, sr: int, target_lufs: float = -14.0) -> np.ndarray:
    """Normalize a stereo signal to a target LUFS if pyloudnorm is available.

    If pyloudnorm is not installed, returns input unchanged and relies on the
    limiter to control peaks.
    """
    if stereo.ndim == 1:
        stereo = stereo[np.newaxis, :]
    if pyln is None:
        return stereo
    meter = pyln.Meter(sr)
    mono = np.mean(stereo, axis=0)
    loudness = meter.integrated_loudness(mono)
    gain_db = float(target_lufs - loudness)
    gain = 10 ** (gain_db / 20.0)
    return (stereo * gain).astype(np.float32, copy=False)


# ================================
# Stem/master application helpers
# ================================

def apply_option_to_stem(stem: np.ndarray, sr: int, option: AdjustOption) -> np.ndarray:
    """Apply a single `AdjustOption` to a stereo stem.

    The implementation dispatches based on option.category and its params.
    """
    if stem.ndim == 1:
        stem = stem[np.newaxis, :]
    out = np.zeros_like(stem)
    for ch in range(stem.shape[0]):
        x = stem[ch]
        if option.category == "eq":
            bands = []
            # Map generic params to single- or multi-band EQ
            # Expect params: type, freq, gain_db, q (optional)
            band = {
                "type": str(option.params.get("type", "peak")),
                "freq": float(option.params.get("freq", 1000.0)),
                "gain_db": float(option.params.get("gain_db", 0.0)),
                "q": float(option.params.get("q", 1.0)),
            }
            bands.append(band)
            y = apply_eq_fft(x, sr, bands)
        elif option.category == "compression":
            y = compress(
                x,
                sr,
                threshold_db=float(option.params.get("threshold_db", -18.0)),
                ratio=float(option.params.get("ratio", 2.0)),
                attack_ms=float(option.params.get("attack_ms", 10.0)),
                release_ms=float(option.params.get("release_ms", 100.0)),
            )
        else:
            # Limiting/normalization are master-stage; pass-through here.
            y = x
        out[ch] = y
    return out


def apply_master_options(mix: np.ndarray, sr: int, options: List[AdjustOption]) -> np.ndarray:
    """Apply master-bus options in a reasonable order: normalization then limiting."""
    if mix.ndim == 1:
        mix = mix[np.newaxis, :]
    # Normalize first (sets overall level), then limit to catch peaks.
    for opt in options:
        if opt.category == "normalization":
            target = float(opt.params.get("target_lufs", -14.0))
            mix = normalize_to_lufs(mix, sr, target_lufs=target)
    for opt in options:
        if opt.category == "limiting":
            ceiling = float(opt.params.get("ceiling_db", -1.0))
            out = np.zeros_like(mix)
            for ch in range(mix.shape[0]):
                out[ch] = limiter(mix[ch], sr, ceiling_db=ceiling)
            mix = out
    return mix
