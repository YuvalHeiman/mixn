#!/usr/bin/env python3
"""Example: Run De-Esser then MultiBand processing on an input file.

Usage:
  python examples/process_audio.py --input path/to/input.wav --output path/to/out.wav \
      [--config configs/example_config.json]

If --input is omitted, the script will try the first file in input_audio/.
If --output is omitted, it writes to output_audio/<name>_processed.wav.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from audio_processing import io_utils
from audio_processing.dsp import DeEsser, MultiBandProcessor


def _default_output_for(input_path: Path) -> Path:
    name = input_path.stem + "_processed.wav"
    return Path("output_audio") / name


def _load_config(path: Path | None) -> Dict[str, Any]:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    p = argparse.ArgumentParser(description="De-Esser + MultiBand processing example")
    p.add_argument("--input", type=str, default=None, help="Path to input WAV/MP3")
    p.add_argument("--output", type=str, default=None, help="Path to output WAV")
    p.add_argument("--config", type=str, default=None, help="JSON config for processors")
    args = p.parse_args()

    # Resolve input
    in_path: Path
    if args.input is None:
        files = io_utils.list_audio_files()
        if not files:
            raise SystemExit("No input provided and no files found in input_audio/")
        in_path = files[0]
    else:
        in_path = Path(args.input)
        if not in_path.exists():
            raise SystemExit(f"Input not found: {in_path}")

    out_path = Path(args.output) if args.output else _default_output_for(in_path)

    # Load audio (channels, samples)
    audio, sr = io_utils.load_audio_stereo(in_path)

    # Load config
    cfg = _load_config(Path(args.config)) if args.config else {}

    # Build processors
    de_cfg = cfg.get("deesser", {}) if isinstance(cfg, dict) else {}
    mb_cfg = cfg.get("multiband", {}) if isinstance(cfg, dict) else {}

    deesser = DeEsser.from_config(de_cfg) if hasattr(DeEsser, "from_config") else DeEsser()
    mb = MultiBandProcessor.from_config(mb_cfg) if hasattr(MultiBandProcessor, "from_config") else MultiBandProcessor()

    # Process
    print(f"Processing: {in_path} @ {sr} Hz")
    y = deesser.process(audio, sr)
    y = mb.process(y, sr)

    # Normalize lightly to prevent surprise peak (optional)
    peak = float(np.max(np.abs(y))) + 1e-12
    if peak > 1.0:
        y = y / peak

    # Save
    io_utils.save_audio(out_path, y, sr)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
