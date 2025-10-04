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
