"""Shared datatypes for the mixing/mastering CLI app.

These dataclasses and type aliases help keep interfaces clear between
modules. This file should remain very lightweight and dependency-free.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

StemName = Literal["vocals", "drums", "bass", "other"]
AdjustmentCategory = Literal[
    "eq",
    "compression",
    "limiting",
    "normalization",
    "deesser",
    "multiband",
]


@dataclass
class AdjustOption:
    """Represents a single adjustment option a user can pick for a stem or master.

    Attributes:
        id: Stable identifier for the option (e.g., "vocals_eq_clarity").
        label: Human-readable description shown to the user.
        category: The processing category ("eq", "compression", "limiting", "normalization").
        params: Implementation-specific parameters for the DSP stage.
    """

    id: str
    label: str
    category: AdjustmentCategory
    params: Dict[str, object]


@dataclass
class AnalysisResult:
    """Container for analysis outputs.

    Attributes:
        file_path: The original audio file path.
        sr: Sample rate of the loaded audio.
        stems: Dict mapping stem names to stereo audio arrays of shape (channels, samples).
        metrics: Dict mapping stem names to computed metrics (RMS, crest factor, centroid, etc.).
        suggestions: Dict mapping stem names (and "master") to lists of suggested `AdjustOption`s.
    """

    file_path: str
    sr: int
    stems: Dict[StemName, "np.ndarray"]  # type: ignore[name-defined]
    metrics: Dict[str, Dict[str, float]]
    suggestions: Dict[str, List[AdjustOption]]


@dataclass
class AdjustmentChoice:
    """Represents the user-chosen option for a given stem (or master)."""

    stem: str  # can be one of stems or "master"
    option: Optional[AdjustOption]


# Lazy import typing helper to avoid importing numpy at module import time in types.
class _NumpyProtocol:
    ...

try:  # pragma: no cover - only for typing hints
    import numpy as np  # noqa: F401
except Exception:  # pragma: no cover
    pass
