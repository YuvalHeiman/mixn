from __future__ import annotations

import numpy as np

from audio_processing.dsp import DeEsser, MultiBandProcessor, bandpass


def _band_energy_fft(x: np.ndarray, sr: int, low: float, high: float) -> float:
    n = len(x)
    X = np.fft.rfft(x * np.hanning(n))
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    mask = (freqs >= low) & (freqs <= high)
    power = np.mean(np.abs(X[mask]) ** 2) + 1e-12
    return float(power)


def _frame_rms(x: np.ndarray, frame: int) -> np.ndarray:
    n = len(x)
    if frame <= 1:
        frame = 1024
    hop = frame // 2
    # Pad to avoid edge bias
    xpad = np.pad(x, (frame // 2, frame // 2), mode="reflect")
    out = []
    for i in range(0, len(xpad) - frame + 1, hop):
        w = xpad[i : i + frame]
        out.append(np.sqrt(np.mean(w * w) + 1e-12))
    return np.asarray(out)


def test_deesser_reduces_sibilant_band_energy():
    sr = 44100
    dur = 2.0
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)

    # Base tone ("vocal" body)
    base = 0.2 * np.sin(2 * np.pi * 200.0 * t)

    # Sibilant bursts around 8 kHz
    sib_carrier = np.sin(2 * np.pi * 8000.0 * t)
    burst_mask = (np.sin(2 * np.pi * 5.0 * t) > 0.7).astype(np.float32)  # ~narrow pulses
    sib = 0.6 * sib_carrier * burst_mask

    x = base + sib

    de = DeEsser()
    y = de.process(x, sr)

    in_energy = _band_energy_fft(x, sr, 5000.0, 10000.0)
    out_energy = _band_energy_fft(y, sr, 5000.0, 10000.0)

    # Expect a noticeable reduction (> 3 dB => power halved ~0.5)
    assert out_energy < in_energy * 0.7


def test_multiband_compressor_reduces_midband_variation():
    sr = 44100
    dur = 2.0
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)

    # 1 kHz tone with varying amplitude (simulate dynamic mid content)
    amp_env = 0.1 + 0.8 * (0.5 * (1 + np.sin(2 * np.pi * 1.0 * t)))  # 0.1..0.9 slow LFO
    x = amp_env * np.sin(2 * np.pi * 1000.0 * t)

    # Use a single mid band to avoid overlap interactions
    bands = [
        {
            "low_hz": 500.0,
            "high_hz": 2000.0,
            "threshold_db": -28.0,
            "ratio": 3.0,
            "attack_ms": 10.0,
            "release_ms": 120.0,
            "makeup_db": 0.0,
        }
    ]
    mb = MultiBandProcessor(bands=bands, mode="compressor", filter_order=4)

    y = mb.process(x, sr)

    # Isolate mid band for measurement
    xin = bandpass(x, sr, 500.0, 2000.0, order=4)
    xout = bandpass(y, sr, 500.0, 2000.0, order=4)

    rms_in = _frame_rms(xin, 2048)
    rms_out = _frame_rms(xout, 2048)

    # Expect reduced variation (std dev) after compression
    assert np.std(rms_out) < np.std(rms_in) * 0.8


def test_dynamic_eq_upward_boost_when_below_threshold():
    sr = 44100
    dur = 1.5
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)

    # Very low-level 100 Hz tone (below threshold)
    x = 0.02 * np.sin(2 * np.pi * 100.0 * t)

    # Single low band with upward dynamics enabled
    bands = [
        {
            "low_hz": 20.0,
            "high_hz": 150.0,
            "threshold_db": -24.0,
            "ratio": 2.0,
            "attack_ms": 20.0,
            "release_ms": 200.0,
            "makeup_db": 0.0,
            "upward_max_boost_db": 6.0,
            "upward_ratio": 2.0,
        }
    ]
    mb = MultiBandProcessor(bands=bands, mode="dynamic_eq", filter_order=4)

    y = mb.process(x, sr)

    # Low-band energy should increase due to upward action (cap at 6 dB)
    in_rms = float(np.sqrt(np.mean(x * x) + 1e-12))
    out_rms = float(np.sqrt(np.mean(y * y) + 1e-12))
    assert out_rms > in_rms * 1.5
