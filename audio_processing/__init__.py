"""Audio processing package for the terminal mixing/mastering app.

This package groups together modular components for:
- I/O utilities (listing, loading, and saving audio)
- DSP utilities (EQ, compression, limiting, normalization)
- Analysis (basic stem separation heuristics and metrics)

The code is intentionally simple, clear, and well-commented, to be accessible
for junior developers.
"""

__all__ = [
    "io_utils",
    "dsp",
    "analysis",
    "types",
]

__version__ = "0.1.0"
