"""DSP utilities: simple EQ, compression, limiting, LUFS normalization.

This module intentionally keeps implementations approachable and well-commented.
They are not meant to rival professional plugins, but to provide educational,
useful processing for a terminal app.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import butter, sosfiltfilt, stft, istft

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

    # Special handling: De-Esser should consider stereo linking, so process whole stem
    if option.category == "deesser":
        try:
            # Build config in a tolerant way
            cfg = dict(option.params)
            fr = cfg.get("freq_range")
            if isinstance(fr, (list, tuple)) and len(fr) == 2:
                cfg["freq_range"] = (float(fr[0]), float(fr[1]))
            de = DeEsser.from_config(cfg) if hasattr(DeEsser, "from_config") else DeEsser()
        except Exception:
            de = DeEsser()
        return de.process(stem, sr)

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
            # Master-stage categories are handled elsewhere; pass-through here.
            y = x
        out[ch] = y
    return out


def apply_master_options(mix: np.ndarray, sr: int, options: List[AdjustOption]) -> np.ndarray:
    """Apply master-bus options in a reasonable order: normalization then limiting."""
    if mix.ndim == 1:
        mix = mix[np.newaxis, :]
    # Multiband dynamics first
    for opt in options:
        if opt.category == "multiband":
            try:
                mcfg = dict(opt.params)
                mb = MultiBandProcessor.from_config(mcfg) if hasattr(MultiBandProcessor, "from_config") else MultiBandProcessor()
            except Exception:
                mb = MultiBandProcessor()
            mix = mb.process(mix, sr)
    # Normalize next (sets overall level)
    for opt in options:
        if opt.category == "normalization":
            target = float(opt.params.get("target_lufs", -14.0))
            mix = normalize_to_lufs(mix, sr, target_lufs=target)
    # Limiting last to catch peaks
    for opt in options:
        if opt.category == "limiting":
            ceiling = float(opt.params.get("ceiling_db", -1.0))
            out = np.zeros_like(mix)
            for ch in range(mix.shape[0]):
                out[ch] = limiter(mix[ch], sr, ceiling_db=ceiling)
            mix = out
    return mix


# ================================
# Advanced processing: De-Esser
# ================================

class DeEsser:
    """Frequency-selective dynamics to tame sibilance.

    Detection uses STFT energy in a sibilant band compared to full-band energy.
    A downward compression curve is applied to the sibilant band only.

    Parameters
    ----------
    freq_range : tuple(float, float)
        Sibilant band in Hz, default (5000, 10000).
    threshold_db : float
        Threshold on (sibilant_level_db - fullband_level_db). Default -20 dB.
        Example: -20 dB means if the sibilant band is within 20 dB of the full-band
        energy, reduction engages.
    ratio : float
        Downward compression ratio when above threshold. Default 4.0.
    attack_ms, release_ms : float
        Envelope times applied to the detector in the time domain of STFT frames.
    lookahead_ms : float
        Frames of anticipation; shifts gain earlier to catch fast transients.
    listen : bool
        If True, returns the solo sidechain (sibilant band) instead of applying
        reduction—useful for tuning.
    stereo_link : bool
        If True (default), compute a single detector from the mono sum and apply the
        same reduction to both channels. If False, detect/process each channel.
    n_fft : int
        STFT window size. Default 1024.
    hop_length : Optional[int]
        STFT hop length. Default n_fft // 4.
    win_length : Optional[int]
        Window length. Default n_fft.
    """

    def __init__(
        self,
        freq_range: Tuple[float, float] = (5000.0, 10000.0),
        threshold_db: float = -20.0,
        ratio: float = 4.0,
        attack_ms: float = 5.0,
        release_ms: float = 50.0,
        lookahead_ms: float = 2.0,
        listen: bool = False,
        stereo_link: bool = True,
        n_fft: int = 1024,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
    ) -> None:
        self.freq_range = freq_range
        self.threshold_db = float(threshold_db)
        self.ratio = float(max(ratio, 1.0))
        self.attack_ms = float(max(attack_ms, 0.0))
        self.release_ms = float(max(release_ms, 0.0))
        self.lookahead_ms = float(max(lookahead_ms, 0.0))
        self.listen = bool(listen)
        self.stereo_link = bool(stereo_link)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length) if hop_length is not None else int(n_fft // 4)
        self.win_length = int(win_length) if win_length is not None else int(n_fft)

    # ---------- Public API ----------
    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if audio.ndim == 1:
            return self._process_mono(audio, sr)
        # stereo or multi-channel
        C, N = audio.shape
        out = np.zeros_like(audio)
        if self.stereo_link and C >= 2:
            # Build detector from mono sum, then apply to each channel's STFT
            mono = np.mean(audio, axis=0)
            f, t, Z = self._stft(mono, sr)
            gain_frames = self._compute_gain_frames(f, Z, sr)
            for ch in range(C):
                _, _, Zc = self._stft(audio[ch], sr)
                Yc = self._apply_gain_to_band(f, Zc, gain_frames)
                yc = self._istft(f, t, Yc, sr, length=N)
                out[ch] = yc
        else:
            for ch in range(C):
                out[ch] = self._process_mono(audio[ch], sr)
        return out.astype(np.float32, copy=False)

    @staticmethod
    def from_config(cfg: Dict[str, float | int | bool | Tuple[float, float]]) -> "DeEsser":
        return DeEsser(
            freq_range=tuple(cfg.get("freq_range", (5000.0, 10000.0))) if isinstance(cfg.get("freq_range", None), (list, tuple)) else (5000.0, 10000.0),
            threshold_db=float(cfg.get("threshold_db", -20.0)),
            ratio=float(cfg.get("ratio", 4.0)),
            attack_ms=float(cfg.get("attack_ms", 5.0)),
            release_ms=float(cfg.get("release_ms", 50.0)),
            lookahead_ms=float(cfg.get("lookahead_ms", 2.0)),
            listen=bool(cfg.get("listen", False)),
            stereo_link=bool(cfg.get("stereo_link", True)),
            n_fft=int(cfg.get("n_fft", 1024)),
            hop_length=int(cfg.get("hop_length", 0)) or None,
            win_length=int(cfg.get("win_length", 0)) or None,
        )

    # ---------- Internals ----------
    def _stft(self, x: np.ndarray, sr: int):
        f, t, Z = stft(
            x,
            fs=sr,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
            nfft=self.n_fft,
            window="hann",
            boundary="zeros",
            padded=True,
        )
        return f, t, Z

    def _istft(self, f: np.ndarray, t: np.ndarray, Z: np.ndarray, sr: int, length: Optional[int] = None) -> np.ndarray:
        _, x_rec = istft(
            Z,
            fs=sr,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
            nfft=self.n_fft,
            window="hann",
            input_onesided=True,
        )
        if length is not None and len(x_rec) != length:
            # Trim or pad for exact length
            if len(x_rec) > length:
                x_rec = x_rec[:length]
            else:
                pad = np.zeros(length - len(x_rec), dtype=x_rec.dtype)
                x_rec = np.concatenate([x_rec, pad])
        return x_rec.astype(np.float32, copy=False)

    def _process_mono(self, x: np.ndarray, sr: int) -> np.ndarray:
        n = len(x)
        f, t, Z = self._stft(x, sr)
        if self.listen:
            # Return the sidechain (sibilant band) solo
            Y = np.zeros_like(Z)
            mask = (f >= self.freq_range[0]) & (f <= self.freq_range[1])
            Y[mask, :] = Z[mask, :]
            y = self._istft(f, t, Y, sr, length=n)
            return y
        gain_frames = self._compute_gain_frames(f, Z, sr)
        Y = self._apply_gain_to_band(f, Z, gain_frames)
        y = self._istft(f, t, Y, sr, length=n)
        return y

    def _compute_gain_frames(self, f: np.ndarray, Z: np.ndarray, sr: int) -> np.ndarray:
        eps = 1e-12
        mask = (f >= self.freq_range[0]) & (f <= self.freq_range[1])
        power_full = np.mean(np.abs(Z) ** 2, axis=0) + eps
        power_sib = np.mean(np.abs(Z[mask, :]) ** 2, axis=0) + eps
        # Relative band level in dB
        rel_db = 10.0 * np.log10(power_sib / power_full)
        # Amount above threshold (in dB)
        over_db = np.maximum(rel_db - self.threshold_db, 0.0)
        # Smooth over frames using one-pole attack/release
        hop_s = self.hop_length / float(sr)
        att_hops = max(self.attack_ms / 1000.0 / hop_s, 1e-6)
        rel_hops = max(self.release_ms / 1000.0 / hop_s, 1e-6)
        smoothed_over = _one_pole_smooth(over_db, att_hops, rel_hops)
        # Static curve: downward compression
        gain_red_db = -smoothed_over * (1.0 - 1.0 / self.ratio)
        gain_lin = 10.0 ** (gain_red_db / 20.0)
        # Lookahead: shift gain earlier
        look_frames = int(round(self.lookahead_ms / 1000.0 / hop_s))
        if look_frames > 0 and len(gain_lin) > 1:
            shifted = np.empty_like(gain_lin)
            shifted[:-look_frames] = gain_lin[look_frames:]
            shifted[-look_frames:] = gain_lin[-1]
            gain_lin = shifted
        return gain_lin

    def _apply_gain_to_band(self, f: np.ndarray, Z: np.ndarray, gain_frames: np.ndarray) -> np.ndarray:
        Y = np.array(Z, copy=True)
        mask = (f >= self.freq_range[0]) & (f <= self.freq_range[1])
        if np.any(mask):
            Y[mask, :] *= gain_frames[np.newaxis, :]
        return Y


def _one_pole_smooth(x: np.ndarray, attack_hops: float, release_hops: float) -> np.ndarray:
    """One-pole smoothing with separate attack/release time constants in 'hops'."""
    if len(x) == 0:
        return x
    a_att = float(np.exp(-1.0 / max(attack_hops, 1e-6)))
    a_rel = float(np.exp(-1.0 / max(release_hops, 1e-6)))
    y = np.empty_like(x, dtype=float)
    y[0] = x[0]
    for i in range(1, len(x)):
        if x[i] > y[i - 1]:
            y[i] = a_att * y[i - 1] + (1.0 - a_att) * x[i]
        else:
            y[i] = a_rel * y[i - 1] + (1.0 - a_rel) * x[i]
    return y


# ================================
# Advanced processing: Multi-band Processor (Compressor / Dynamic EQ)
# ================================

class MultiBandProcessor:
    """Split into frequency bands, apply per-band dynamics, and recombine.

    Parameters
    ----------
    bands : Optional[List[Dict]]
        Each band dict supports keys:
          - low_hz: float (inclusive), optional
          - high_hz: float (inclusive), optional
          - threshold_db: float
          - ratio: float
          - attack_ms: float
          - release_ms: float
          - makeup_db: float
          - upward_max_boost_db: float (only used in dynamic_eq mode)
          - upward_ratio: float (only used in dynamic_eq mode)
        If None, a 5-band default is created.
    mode : str
        "compressor" or "dynamic_eq".
    """

    def __init__(
        self,
        bands: Optional[List[Dict[str, float]]] = None,
        mode: str = "compressor",
        filter_order: int = 4,
    ) -> None:
        self.mode = str(mode)
        self.filter_order = int(filter_order)
        self.bands = bands if bands is not None else self._default_bands()

    # ---------- Public API ----------
    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if audio.ndim == 1:
            return self._process_mono(audio, sr)
        C, N = audio.shape
        out = np.zeros_like(audio)
        for ch in range(C):
            out[ch] = self._process_mono(audio[ch], sr)
        return out.astype(np.float32, copy=False)

    @staticmethod
    def from_config(cfg: Dict[str, object]) -> "MultiBandProcessor":
        bands = cfg.get("bands")
        mode = cfg.get("mode", "compressor")
        order = int(cfg.get("filter_order", 4)) if isinstance(cfg, dict) else 4
        return MultiBandProcessor(bands=bands if isinstance(bands, list) else None, mode=str(mode), filter_order=order)

    # ---------- Internals ----------
    def _default_bands(self) -> List[Dict[str, float]]:
        return [
            {"low_hz": 20.0, "high_hz": 120.0, "threshold_db": -30.0, "ratio": 2.0, "attack_ms": 15.0, "release_ms": 150.0, "makeup_db": 0.0, "upward_max_boost_db": 2.0, "upward_ratio": 2.0},
            {"low_hz": 120.0, "high_hz": 500.0, "threshold_db": -28.0, "ratio": 2.0, "attack_ms": 15.0, "release_ms": 150.0, "makeup_db": 0.0, "upward_max_boost_db": 2.0, "upward_ratio": 2.0},
            {"low_hz": 500.0, "high_hz": 2000.0, "threshold_db": -24.0, "ratio": 2.0, "attack_ms": 10.0, "release_ms": 120.0, "makeup_db": 0.0, "upward_max_boost_db": 1.5, "upward_ratio": 2.0},
            {"low_hz": 2000.0, "high_hz": 6000.0, "threshold_db": -24.0, "ratio": 2.5, "attack_ms": 8.0, "release_ms": 100.0, "makeup_db": 0.0, "upward_max_boost_db": 1.0, "upward_ratio": 2.5},
            {"low_hz": 6000.0, "high_hz": 20000.0, "threshold_db": -26.0, "ratio": 3.0, "attack_ms": 5.0, "release_ms": 80.0, "makeup_db": 0.0, "upward_max_boost_db": 0.0, "upward_ratio": 2.0},
        ]

    def _band_filter(self, x: np.ndarray, sr: int, low_hz: Optional[float], high_hz: Optional[float]) -> np.ndarray:
        if low_hz is None and high_hz is None:
            return x
        if low_hz is None and high_hz is not None:
            return lowpass(x, sr, cutoff_hz=high_hz, order=self.filter_order)
        if low_hz is not None and high_hz is None:
            return highpass(x, sr, cutoff_hz=low_hz, order=self.filter_order)
        # Both bounds present -> bandpass
        return bandpass(x, sr, low_hz=float(low_hz), high_hz=float(high_hz), order=self.filter_order)

    def _process_mono(self, x: np.ndarray, sr: int) -> np.ndarray:
        n = len(x)
        bands_out: List[np.ndarray] = []
        for b in self.bands:
            low = float(b.get("low_hz", 0.0)) if b.get("low_hz") is not None else None  # type: ignore[assignment]
            high = float(b.get("high_hz", 0.0)) if b.get("high_hz") is not None else None  # type: ignore[assignment]
            # Clamp to Nyquist
            if low is not None:
                low = max(0.0, min(low, sr * 0.5 - 1.0))
            if high is not None:
                high = max(10.0, min(high, sr * 0.5 - 1.0))

            xb = self._band_filter(x, sr, low, high)

            thr = float(b.get("threshold_db", -24.0))
            ratio = float(max(b.get("ratio", 2.0), 1.0))
            att = float(max(b.get("attack_ms", 10.0), 0.0))
            rel = float(max(b.get("release_ms", 120.0), 0.0))
            makeup = float(b.get("makeup_db", 0.0))
            up_max = float(b.get("upward_max_boost_db", 0.0))
            up_ratio = float(max(b.get("upward_ratio", 2.0), 1.0))

            # Detector
            env = _rms_envelope(xb, sr, window_ms=max(att, 5.0))
            lvl_db = _db(env)

            over_db = np.maximum(lvl_db - thr, 0.0)
            # Downward reduction
            down_red_db = -over_db * (1.0 - 1.0 / ratio)

            if self.mode == "dynamic_eq" and up_max > 0.0:
                below_db = np.maximum(thr - lvl_db, 0.0)
                up_boost_db = np.minimum(below_db / up_ratio, up_max)
            else:
                up_boost_db = 0.0

            target_db = down_red_db + up_boost_db
            # Smooth target gain in dB
            hop_s = 1.0 / sr  # approximate per-sample hops
            att_hops = max(att / 1000.0 / hop_s, 1e-6)
            rel_hops = max(rel / 1000.0 / hop_s, 1e-6)
            smoothed_db = _one_pole_smooth(target_db, att_hops, rel_hops)

            gain = 10.0 ** ((smoothed_db + makeup) / 20.0)
            yb = xb * gain.astype(xb.dtype, copy=False)
            bands_out.append(yb)

        y = np.sum(np.stack(bands_out, axis=0), axis=0)
        return y.astype(np.float32, copy=False)

