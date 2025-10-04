"""Audio analysis utilities: simple stem separation and suggestion generation.

This module uses lightweight, filter-based heuristics to approximate stems,
computes basic metrics, and proposes human-readable adjustments.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from . import dsp
from .types import AdjustOption, AnalysisResult, StemName


# ================================
# Stem separation (heuristic)
# ================================

def separate_stems_stereo(y: np.ndarray, sr: int) -> Dict[StemName, np.ndarray]:
    """Approximate stems from stereo input using simple filters (no librosa).

    Heuristic approach:
    - bass  = lowpass(x, 150 Hz)
    - vocals = bandpass(x, 150–6000 Hz) with a low-frequency roll-off
    - drums = highpass(x, 200 Hz) (captures a lot of percussive energy)
    - other = residual (x - bass - vocals - drums)

    Returns a dict mapping stem name -> (channels, samples) arrays.
    """
    if y.ndim == 1:
        y = y[np.newaxis, :]
    C, _ = y.shape
    vocals = np.zeros_like(y)
    drums = np.zeros_like(y)
    bass = np.zeros_like(y)
    other = np.zeros_like(y)

    for ch in range(C):
        x = y[ch]
        # Bass
        b = dsp.lowpass(x, sr, cutoff_hz=150.0, order=4)
        # Vocals (mid band) with LF removal
        mid = dsp.bandpass(x, sr, low_hz=150.0, high_hz=6000.0, order=4)
        v = mid - dsp.lowpass(mid, sr, cutoff_hz=180.0, order=4)
        # Drums (highpassed content)
        d = dsp.highpass(x, sr, cutoff_hz=200.0, order=4)
        # Other = residual
        o = x - (b + v + d)

        bass[ch] = b
        vocals[ch] = v
        drums[ch] = d
        other[ch] = o

    return {"vocals": vocals, "drums": drums, "bass": bass, "other": other}


# ================================
# Metrics
# ================================

def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64) + 1e-12))


def _peak(x: np.ndarray) -> float:
    return float(np.max(np.abs(x)))


def _crest_factor_db(x: np.ndarray) -> float:
    r = _rms(x)
    p = _peak(x)
    if r == 0.0:
        return 60.0
    return 20.0 * np.log10(p / r + 1e-12)


def _spectral_centroid_hz(x: np.ndarray, sr: int) -> float:
    """Compute spectral centroid without librosa.

    Uses a Hann window and full-signal rFFT magnitude barycenter.
    """
    if x.ndim == 2:
        x = np.mean(x, axis=0)
    # Limit to max 10 seconds to control cost
    max_samples = min(len(x), int(sr * 10))
    x = x[:max_samples]
    if len(x) == 0:
        return 0.0
    # Window and rFFT
    window = np.hanning(len(x)).astype(np.float64)
    X = np.fft.rfft(x * window)
    mag = np.abs(X)
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sr)
    denom = np.sum(mag) + 1e-12
    centroid = float(np.sum(freqs * mag) / denom)
    return centroid


def compute_metrics(stems: Dict[StemName, np.ndarray], sr: int) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    for name, audio in stems.items():
        mono = np.mean(audio, axis=0)
        metrics[name] = {
            "rms": _rms(mono),
            "peak": _peak(mono),
            "crest_db": _crest_factor_db(mono),
            "centroid_hz": _spectral_centroid_hz(mono, sr),
        }
    return metrics


# ================================
# Suggestions
# ================================

def generate_suggestions(stems_metrics: Dict[str, Dict[str, float]], sr: int, mix_lufs: float | None) -> Dict[str, List[AdjustOption]]:
    suggestions: Dict[str, List[AdjustOption]] = {name: [] for name in ["vocals", "drums", "bass", "other"]}

    # Vocals
    voc_centroid = stems_metrics.get("vocals", {}).get("centroid_hz", 2000.0)
    if voc_centroid < 2000.0:
        suggestions["vocals"].append(
            AdjustOption(
                id="vocals_eq_clarity",
                label="Boost clarity (EQ +3 dB @ 3 kHz)",
                category="eq",
                params={"type": "peak", "freq": 3000.0, "gain_db": 3.0, "q": 1.0},
            )
        )
    else:
        suggestions["vocals"].append(
            AdjustOption(
                id="vocals_eq_deharsh",
                label="Reduce harshness (EQ -2 dB @ 6 kHz)",
                category="eq",
                params={"type": "peak", "freq": 6000.0, "gain_db": -2.0, "q": 1.0},
            )
        )
    suggestions["vocals"].append(
        AdjustOption(
            id="vocals_comp_gentle",
            label="Gentle compression (2:1, -18 dB)",
            category="compression",
            params={"threshold_db": -18.0, "ratio": 2.0, "attack_ms": 10.0, "release_ms": 100.0},
        )
    )

    # Drums
    suggestions["drums"].append(
        AdjustOption(
            id="drums_comp_punch",
            label="Add punch (3:1, -12 dB)",
            category="compression",
            params={"threshold_db": -12.0, "ratio": 3.0, "attack_ms": 10.0, "release_ms": 120.0},
        )
    )
    suggestions["drums"].append(
        AdjustOption(
            id="drums_eq_tame_cymbals",
            label="Tame cymbals (High-shelf -2 dB @ 8 kHz)",
            category="eq",
            params={"type": "high_shelf", "freq": 8000.0, "gain_db": -2.0, "q": 1.0},
        )
    )

    # Bass
    crest_bass = stems_metrics.get("bass", {}).get("crest_db", 20.0)
    if crest_bass > 14.0:
        suggestions["bass"].append(
            AdjustOption(
                id="bass_comp_control",
                label="Control dynamics (3:1, -16 dB)",
                category="compression",
                params={"threshold_db": -16.0, "ratio": 3.0, "attack_ms": 15.0, "release_ms": 120.0},
            )
        )
    suggestions["bass"].append(
        AdjustOption(
            id="bass_eq_tighten",
            label="Tighten lows (Low-shelf -2 dB @ 50 Hz)",
            category="eq",
            params={"type": "low_shelf", "freq": 50.0, "gain_db": -2.0, "q": 0.7},
        )
    )
    suggestions["bass"].append(
        AdjustOption(
            id="bass_eq_presence",
            label="Add presence (Peak +2 dB @ 120 Hz)",
            category="eq",
            params={"type": "peak", "freq": 120.0, "gain_db": 2.0, "q": 1.0},
        )
    )

    # Other
    suggestions["other"].append(
        AdjustOption(
            id="other_comp_glue",
            label="Glue (1.5:1, -18 dB)",
            category="compression",
            params={"threshold_db": -18.0, "ratio": 1.5, "attack_ms": 20.0, "release_ms": 200.0},
        )
    )

    # Master bus suggestions
    master_opts: List[AdjustOption] = []
    if mix_lufs is None or mix_lufs < -16.0:
        master_opts.append(
            AdjustOption(
                id="master_norm_-14lufs",
                label="Normalize to -14 LUFS",
                category="normalization",
                params={"target_lufs": -14.0},
            )
        )
    master_opts.append(
        AdjustOption(
            id="master_limiter_-1dbtp",
            label="Apply limiter (-1 dBTP ceiling)",
            category="limiting",
            params={"ceiling_db": -1.0},
        )
    )
    suggestions["master"] = master_opts

    return suggestions


def analyze_audio(file_path: str, y: np.ndarray, sr: int) -> AnalysisResult:
    """Run the overall analysis pipeline.

    Parameters
    ----------
    file_path: str
        Path to the selected audio file (for reporting context).
    y: np.ndarray
        Stereo-like array (channels, samples).
    sr: int
        Sample rate.
    """
    stems = separate_stems_stereo(y, sr)
    metrics = compute_metrics(stems, sr)

    # Optional LUFS for the full mix to inform master suggestions
    mix_lufs = None
    try:
        import pyloudnorm as pyln  # local import to keep optional

        meter = pyln.Meter(sr)
        mono = np.mean(y, axis=0)
        mix_lufs = float(meter.integrated_loudness(mono))
    except Exception:
        mix_lufs = None

    suggestions = generate_suggestions(metrics, sr, mix_lufs)

    return AnalysisResult(
        file_path=file_path,
        sr=sr,
        stems=stems,
        metrics=metrics,
        suggestions=suggestions,
    )
