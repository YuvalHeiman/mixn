# Terminal Music Mixing/Mastering App

A Python terminal-based app for analyzing an input track, suggesting per-stem adjustments, and applying basic mixing/mastering.

The app uses Rich for a clean CLI, filter-based heuristics for stem separation, and simple DSP blocks (EQ, compression, limiter, optional LUFS normalization).

## Features
- **Rich CLI**: Welcome menu, progress spinners/bars, tables/panels
- **Input validation**: Looks for `.wav` or `.mp3` in `input_audio/`
- **Analysis**: Approximate stems (vocals, drums, bass, other) and compute metrics
- **Suggestions**: Clear per-stem options (e.g., "Boost clarity (EQ)") plus master options
- **Interactive**: Pick one option per stem, confirm, then process
- **Processing**: Apply EQ/compression per stem; normalization/limiting on master
- **Output**: Saves to `output_audio/<name>_mastered.wav`

## Quick Start

1) Python 3.10+ is recommended.

2) Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

3) Place one or more `.wav` or `.mp3` files in `input_audio/`.

4) Run the app:

```bash
python main.py
```

5) Follow the prompts to analyze the file, select adjustments, confirm, and process.

## Notes on MP3 Support
- MP3 decoding is performed via the `ffmpeg` CLI directly (no pydub).
- Ensure both `ffmpeg` and `ffprobe` are in your PATH.
- If MP3 files fail to load, install FFmpeg (macOS example):

```bash
brew install ffmpeg
```

## Loudness Normalization (Optional)
- If `pyloudnorm` is installed (included in requirements), the master bus normalization to target LUFS will be enabled.
- If it is not available, the app will skip LUFS normalization and rely on limiting.

## Project Structure
```
.
├── main.py                          # Rich CLI orchestrator
├── audio_processing/
│   ├── __init__.py
│   ├── analysis.py                  # Stems, metrics, suggestions
│   ├── dsp.py                       # EQ, compression, limiter, normalization
│   ├── io_utils.py                  # Listing, loading, saving audio
│   └── types.py                     # Dataclasses and shared types
├── input_audio/                     # Put your input files here
├── output_audio/                    # Mastered output written here
├── requirements.txt
└── README.md
```

## How It Works (High-Level)
- `analysis.separate_stems_stereo()`: Uses simple filters (lowpass/bandpass/highpass) to approximate vocals/drums/bass/other.
- `analysis.compute_metrics()`: RMS, peak, crest factor (dB), spectral centroid.
- `analysis.generate_suggestions()`: Suggests per-stem EQ/compression and master normalization/limiting.
- `dsp.apply_option_to_stem()`: Applies a selected option to a stereo stem.
- `dsp.apply_master_options()`: Applies normalization (if available) and limiting.
- `io_utils`: Handles directory creation, file listing, loading with native SR, saving WAV output.

### New DSP Modules

- **DeEsser**: Detects sibilant energy (≈5–10 kHz) via STFT and applies frequency-selective gain reduction with attack/release and optional lookahead. Supports stereo-linked or per-channel processing and a "listen" mode to solo the affected band.
- **MultiBandProcessor**: Splits the signal into bands using Butterworth filters, applies per-band dynamics in two modes:
  - `mode="compressor"`: classic downward compression plus optional make-up gain.
  - `mode="dynamic_eq"`: adaptive gain per band, combining downward reduction above threshold and optional upward lift below threshold.

Import path:

```python
from audio_processing.dsp import DeEsser, MultiBandProcessor
```

### Example Script

Run the example pipeline (De-Esser ➜ Multiband):

```bash
python examples/process_audio.py --input input_audio/your.wav --config configs/example_config.json
```

If `--input` is omitted, the first file in `input_audio/` is used. Output is written to `output_audio/<name>_processed.wav`.

### JSON Configuration

Schema: `configs/dsp_config.schema.json`

Example: `configs/example_config.json`

```json
{
  "deesser": {"freq_range": [5000, 10000], "threshold_db": -20, "ratio": 4, "attack_ms": 5, "release_ms": 60, "lookahead_ms": 2},
  "multiband": {
    "mode": "compressor",
    "filter_order": 4,
    "bands": [
      {"low_hz": 20, "high_hz": 120,  "threshold_db": -30, "ratio": 2.0, "attack_ms": 15, "release_ms": 150, "makeup_db": 0.0},
      {"low_hz": 120, "high_hz": 500,  "threshold_db": -28, "ratio": 2.0, "attack_ms": 15, "release_ms": 150, "makeup_db": 0.0},
      {"low_hz": 500, "high_hz": 2000, "threshold_db": -24, "ratio": 2.0, "attack_ms": 10, "release_ms": 120, "makeup_db": 0.0},
      {"low_hz": 2000, "high_hz": 6000, "threshold_db": -24, "ratio": 2.5, "attack_ms": 8,  "release_ms": 100, "makeup_db": 0.0},
      {"low_hz": 6000, "high_hz": 20000,"threshold_db": -26, "ratio": 3.0, "attack_ms": 5,  "release_ms": 80,  "makeup_db": 0.0}
    ]
  }
}
```

### Testing

Unit tests cover:
- **De-Esser** reduces energy in 5–10 kHz on synthetic sibilant bursts.
- **Multiband Compressor** reduces dynamic variation in a target band.
- **Dynamic EQ** can apply upward lift when below threshold.

Run tests:

```bash
pytest -q
```

## Limitations & Expectations
- Stem separation is heuristic for demonstration, not studio-grade source separation.
- DSP is intentionally simple and educational.
- Real-time playback is not included; this is an offline process.

## Troubleshooting
- If you see "No audio files found": ensure your `.wav`/`.mp3` files are in `input_audio/`.
- If MP3 loading fails: install FFmpeg (see above), or use WAV files.
- If normalization errors occur: it may be due to `pyloudnorm`; you can remove it from `requirements.txt` and skip normalization.

## License
MIT
