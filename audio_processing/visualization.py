"""Visualization utilities for showing before/after waveforms and change maps.

This module generates:
- A PNG report with before/after waveforms and highlighted change sections.
- A JSON change dictionary with time intervals and per-band change summaries.

Dependencies: numpy, scipy (for filters used via dsp), matplotlib (optional at runtime but
recommended). If matplotlib is missing, plotting is skipped but the JSON changes are still
computed and returned.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import json
import numpy as np

from . import dsp


@dataclass
class BandSpec:
    name: str
    low_hz: Optional[float]
    high_hz: Optional[float]


DEFAULT_BANDS: List[BandSpec] = [
    BandSpec("low", 20.0, 120.0),
    BandSpec("low_mid", 120.0, 500.0),
    BandSpec("mid", 500.0, 2000.0),
    BandSpec("high_mid", 2000.0, 6000.0),
    BandSpec("high", 6000.0, 20000.0),
]
SIBILANT_BAND = BandSpec("sibilant", 5000.0, 10000.0)


def _mono(x: np.ndarray) -> np.ndarray:
    return np.mean(x, axis=0) if x.ndim == 2 else x


def _frame_rms_envelope(x: np.ndarray, sr: int, frame_ms: float = 20.0, hop_ms: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Return (times_sec, rms_per_frame)."""
    if hop_ms is None:
        hop_ms = frame_ms / 2.0
    frame = max(16, int(sr * frame_ms / 1000.0))
    hop = max(8, int(sr * hop_ms / 1000.0))
    if frame % 2 == 1:
        frame += 1
    n = len(x)
    times = []
    vals = []
    for start in range(0, n - frame + 1, hop):
        chunk = x[start : start + frame]
        rms = float(np.sqrt(np.mean(chunk * chunk) + 1e-12))
        t = (start + frame / 2) / sr
        times.append(t)
        vals.append(rms)
    return np.asarray(times, dtype=float), np.asarray(vals, dtype=float)


def _band_signal(x: np.ndarray, sr: int, band: BandSpec) -> np.ndarray:
    if band.low_hz is None and band.high_hz is None:
        return x
    if band.low_hz is None:
        return dsp.lowpass(x, sr, cutoff_hz=float(band.high_hz), order=4)
    if band.high_hz is None:
        return dsp.highpass(x, sr, cutoff_hz=float(band.low_hz), order=4)
    return dsp.bandpass(x, sr, low_hz=float(band.low_hz), high_hz=float(band.high_hz), order=4)


def _events_from_delta(times: np.ndarray, delta_db: np.ndarray, threshold_db: float, min_ms: float) -> List[Tuple[float, float, float]]:
    """Compress consecutive frames where delta_db <= -threshold_db into (start, end, avg_delta_db)."""
    if len(times) == 0:
        return []
    changed = delta_db <= -abs(threshold_db)
    events: List[Tuple[float, float, float]] = []
    i = 0
    n = len(times)
    while i < n:
        if not changed[i]:
            i += 1
            continue
        j = i
        s = times[i]
        deltas: List[float] = []
        while j < n and changed[j]:
            deltas.append(delta_db[j])
            j += 1
        e = times[min(j - 1, n - 1)]
        if (e - s) * 1000.0 >= min_ms:
            avg_delta = float(np.mean(deltas)) if deltas else float(delta_db[i])
            events.append((s, e, avg_delta))
        i = j
    return events


def _events_from_abs_delta(times: np.ndarray, delta_db: np.ndarray, threshold_db: float, min_ms: float) -> List[Tuple[float, float, float]]:
    """Like _events_from_delta but triggers on absolute magnitude (boosts or cuts)."""
    if len(times) == 0:
        return []
    changed = np.abs(delta_db) >= abs(threshold_db)
    events: List[Tuple[float, float, float]] = []
    i = 0
    n = len(times)
    while i < n:
        if not changed[i]:
            i += 1
            continue
        j = i
        s = times[i]
        deltas: List[float] = []
        while j < n and changed[j]:
            deltas.append(delta_db[j])
            j += 1
        e = times[min(j - 1, n - 1)]
        if (e - s) * 1000.0 >= min_ms:
            avg_delta = float(np.mean(deltas)) if deltas else float(delta_db[i])
            events.append((s, e, avg_delta))
        i = j
    return events


def compute_change_map(
    before: np.ndarray,
    after: np.ndarray,
    sr: int,
    frame_ms: float = 20.0,
    sibilant_thresh_db: float = 1.5,
    band_thresh_db: float = 1.0,
    min_event_ms: float = 30.0,
    bands: Optional[List[BandSpec]] = None,
) -> Dict:
    """Compute a change dictionary between before and after signals.

    Returns a dict with keys:
      - summary: overall stats
      - events: list of {type, band, start_sec, end_sec, avg_delta_db}
    """
    x0 = _mono(before)
    x1 = _mono(after)
    n = min(len(x0), len(x1))
    x0 = x0[:n]
    x1 = x1[:n]

    bands = bands or DEFAULT_BANDS

    # Full-band envelope difference (not used for events but useful for summary)
    t_full, rms0 = _frame_rms_envelope(x0, sr, frame_ms)
    _, rms1 = _frame_rms_envelope(x1, sr, frame_ms)
    delta_full_db = 20.0 * np.log10((rms1 + 1e-12) / (rms0 + 1e-12))

    events: List[Dict] = []

    # Sibilant events (approximate De-Esser activity)
    s0 = _band_signal(x0, sr, SIBILANT_BAND)
    s1 = _band_signal(x1, sr, SIBILANT_BAND)
    t_sib, sib0 = _frame_rms_envelope(s0, sr, frame_ms)
    _, sib1 = _frame_rms_envelope(s1, sr, frame_ms)
    sib_delta_db = 20.0 * np.log10((sib1 + 1e-12) / (sib0 + 1e-12))
    for s, e, avg in _events_from_delta(t_sib, sib_delta_db, sibilant_thresh_db, min_event_ms):
        events.append({
            "type": "de_esser",
            "band": SIBILANT_BAND.name,
            "start_sec": round(float(s), 3),
            "end_sec": round(float(e), 3),
            "avg_delta_db": round(float(avg), 2)
        })

    # Per-band events (approximate compression/dynamic EQ) — detect boosts and cuts
    for b in bands:
        b0 = _band_signal(x0, sr, b)
        b1 = _band_signal(x1, sr, b)
        t_b, e0 = _frame_rms_envelope(b0, sr, frame_ms)
        _, e1 = _frame_rms_envelope(b1, sr, frame_ms)
        delta_b_db = 20.0 * np.log10((e1 + 1e-12) / (e0 + 1e-12))
        for s, e, avg in _events_from_abs_delta(t_b, delta_b_db, band_thresh_db, min_event_ms):
            events.append({
                "type": "band_gain_change",
                "band": b.name,
                "start_sec": round(float(s), 3),
                "end_sec": round(float(e), 3),
                "avg_delta_db": round(float(avg), 2)
            })

    # Summary
    summary = {
        "duration_sec": round(n / sr, 3),
        "fullband_delta_db_mean": round(float(np.mean(delta_full_db)), 2) if len(delta_full_db) else 0.0,
        "num_deesser_events": sum(1 for ev in events if ev["type"] == "de_esser"),
        "num_band_events": sum(1 for ev in events if ev["type"] == "band_gain_change"),
    }

    return {"summary": summary, "events": events}


def _save_json(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _plot_report(
    before: np.ndarray,
    after: np.ndarray,
    sr: int,
    change_map: Dict,
    png_path: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return  # plotting optional

    x0 = _mono(before)
    x1 = _mono(after)
    n = min(len(x0), len(x1))
    x0 = x0[:n]
    x1 = x1[:n]
    t = np.arange(n) / sr

    fig, ax = plt.subplots(1, 1, figsize=(13, 6), constrained_layout=True)

    # Plot overlapped waveforms
    ax.plot(t, x0, color="#444444", linewidth=0.6, label="Before")
    # Light purple, semi-transparent for After
    ax.plot(t, x1, color="#b19cd9", alpha=0.6, linewidth=0.8, label="After (processed)")
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Before vs After (mono overlay)")

    # Highlight events with shaded spans
    color_map = {
        "de_esser": "#e45756",       # red
        "band_gain_change": "#54a24b",  # green
    }
    events = change_map.get("events", [])
    for ev in events:
        c = color_map.get(ev.get("type", ""), "#ffbf0f")
        ax.axvspan(float(ev.get("start_sec", 0.0)), float(ev.get("end_sec", 0.0)), color=c, alpha=0.15)

    # Annotate top-N events by magnitude (most negative delta first)
    try:
        events_sorted = sorted(events, key=lambda e: float(e.get("avg_delta_db", 0.0)))
    except Exception:
        events_sorted = events
    topN = events_sorted[:12]
    y_top = 0.92  # relative position for annotation arrows
    ylim = ax.get_ylim()
    y_annot = ylim[0] + (ylim[1] - ylim[0]) * y_top
    for ev in topN:
        t0 = float(ev.get("start_sec", 0.0))
        t1 = float(ev.get("end_sec", 0.0))
        tmid = 0.5 * (t0 + t1)
        label = f"{ev.get('type','')}: {ev.get('band','')} {float(ev.get('avg_delta_db',0.0)):.1f} dB"
        ax.annotate(
            label,
            xy=(tmid, 0.0), xycoords=("data", "data"),
            xytext=(tmid, y_annot), textcoords=("data", "data"),
            ha="center", va="bottom",
            fontsize=8, color="#333333",
            arrowprops=dict(arrowstyle="-|>", color="#666666", lw=0.8, shrinkA=0, shrinkB=0),
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#888888", alpha=0.8),
        )

    # Legend and explanatory textbox
    ax.legend(loc="upper left")
    summ = change_map.get("summary", {})
    exp_lines = [
        "Overlay: Before (dark gray), After (light purple, semi-transparent)",
        f"Events: De-Esser={summ.get('num_deesser_events',0)}, Band changes={summ.get('num_band_events',0)}",
        "Highlighted spans show where processing reduced energy.",
        "Arrows label the most significant changes (time and dB).",
    ]
    fig.text(0.01, 0.01, "\n".join(exp_lines), fontsize=9, color="#333333",
             bbox=dict(boxstyle="round,pad=0.35", fc="#f7f7f7", ec="#cccccc"))

    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def create_waveform_change_report(
    before: np.ndarray,
    after: np.ndarray,
    sr: int,
    out_audio_path: Path,
    frame_ms: float = 20.0,
    sibilant_thresh_db: float = 1.5,
    band_thresh_db: float = 1.0,
    min_event_ms: float = 30.0,
) -> Tuple[Dict, Path, Path]:
    """Generate a waveform PNG and a JSON change dictionary next to the output audio.

    Returns (change_map, png_path, json_path).
    """
    change_map = compute_change_map(
        before=before,
        after=after,
        sr=sr,
        frame_ms=frame_ms,
        sibilant_thresh_db=sibilant_thresh_db,
        band_thresh_db=band_thresh_db,
        min_event_ms=min_event_ms,
    )

    parent = out_audio_path.parent
    stem = out_audio_path.stem
    png_path = parent / f"{stem}_waveform_report.png"
    json_path = parent / f"{stem}_changes.json"

    _plot_report(before, after, sr, change_map, png_path)
    _save_json(change_map, json_path)

    return change_map, png_path, json_path
