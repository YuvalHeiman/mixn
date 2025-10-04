"""Terminal Music Mixing/Mastering App (Rich CLI)

App Flow
--------
1) Initialization: show welcome menu.
2) Validate and select an input file from `input_audio/`.
3) Analyze the audio: stems, metrics, suggestions (with progress spinner).
4) Show analysis report in Rich tables/panels.
5) Interactive selection: per-stem choose adjustments via menus.
6) Confirmation screen of chosen adjustments.
7) Processing: apply chosen adjustments with progress.
8) Save to `output_audio/` with `_mastered` suffix.
9) Return to main menu.

This file orchestrates the UX; heavy lifting resides in `audio_processing/`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys
import time
# Dependency preflight: fail fast with clear guidance if a package is missing.
try:
    import numpy as np
except ImportError:
    print("Missing required dependency: numpy. Install dependencies with:")
    print("  pip install -r requirements.txt")
    sys.exit(1)
try:
    import scipy  # noqa: F401
except ImportError:
    print("Missing required dependency: scipy. Install dependencies with:")
    print("  pip install -r requirements.txt")
    sys.exit(1)
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.prompt import Prompt, Confirm
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
except ImportError:
    print("Missing required dependency: rich. Install dependencies with:")
    print("  pip install -r requirements.txt")
    sys.exit(1)
# Ensure audio deps exist before importing our audio_processing modules.
try:
    import soundfile as sf  # noqa: F401
except ImportError:
    print("Missing required dependency: soundfile. Install dependencies with:")
    print("  pip install -r requirements.txt")
    sys.exit(1)

# Optional: warn if ffmpeg/ffprobe are not available (affects MP3 decoding)
import shutil as _shutil
if _shutil.which("ffmpeg") is None or _shutil.which("ffprobe") is None:
    print("Warning: ffmpeg/ffprobe not found in PATH. MP3 decoding may fail.\n"
          "Install with: brew install ffmpeg")

from audio_processing.io_utils import (
    ensure_directories,
    list_audio_files,
    load_audio_stereo,
    save_audio,
    build_output_path,
)
from audio_processing.analysis import analyze_audio
from audio_processing.dsp import apply_option_to_stem, apply_master_options
from audio_processing.visualization import create_waveform_change_report
from audio_processing.types import AdjustOption, AnalysisResult

console = Console()


# ====================================
# UI helpers
# ====================================

def clear_screen() -> None:
    console.clear()


def welcome_screen() -> None:
    clear_screen()
    title = "Music Mixing/Mastering CLI"
    panel = Panel.fit(
        "Place .mp3 or .wav files into [bold]input_audio/[/bold]\n\n"
        "Choose a file to analyze, select adjustments per stem,\n"
        "and save the mastered result to [bold]output_audio/[/bold].",
        title=title,
        border_style="cyan",
    )
    console.print(panel)


def main_menu() -> str:
    console.print("\n[bold]Main Menu[/bold]")
    choices = {
        "1": "Analyze & Process a File",
        "2": "Exit",
    }
    for k, v in choices.items():
        console.print(f"  [cyan]{k}[/cyan]) {v}")
    selection = Prompt.ask("Select an option", choices=list(choices.keys()), default="1")
    return selection


# ====================================
# File selection and loading
# ====================================

def pick_input_file() -> Optional[Path]:
    files = list_audio_files()
    if not files:
        console.print(Panel("No audio files found in [bold]input_audio/[/bold].\n\n"
                            "- Supported: .wav, .mp3\n"
                            "- Please add at least one file and try again.",
                            title="No Files", border_style="red"))
        return None
    table = Table(title="Available Input Files", show_lines=True)
    table.add_column("#", justify="right", style="cyan", no_wrap=True)
    table.add_column("Filename", style="white")
    for idx, p in enumerate(files, 1):
        table.add_row(str(idx), p.name)
    console.print(table)

    valid_choices = [str(i) for i in range(1, len(files) + 1)]
    sel = Prompt.ask("Pick a file number", choices=valid_choices)
    return files[int(sel) - 1]


def load_audio_with_feedback(path: Path) -> Optional[Tuple[np.ndarray, int]]:
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task_id = progress.add_task("Loading audio...", total=None)
            y, sr = load_audio_stereo(path)
            time.sleep(0.2)
            progress.update(task_id, description="Loaded.")
        return y, sr
    except Exception as e:
        console.print(Panel(f"Failed to load file: {e}", title="Error", border_style="red"))
        return None


# ====================================
# Analysis and reporting
# ====================================

def run_analysis_with_progress(path: Path, y: np.ndarray, sr: int) -> AnalysisResult:
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task_id = progress.add_task("Analyzing (stems, EQ, compression, loudness)...", total=None)
        result = analyze_audio(str(path), y, sr)
        time.sleep(0.2)
        progress.update(task_id, description="Analysis complete.")
    return result


def show_findings(result: AnalysisResult) -> None:
    console.print(Panel(f"Analysis Report for [bold]{Path(result.file_path).name}[/bold]", border_style="green"))

    # Metrics per stem
    metrics_table = Table(title="Stem Metrics", show_lines=True)
    metrics_table.add_column("Stem", style="cyan")
    metrics_table.add_column("RMS", justify="right")
    metrics_table.add_column("Peak", justify="right")
    metrics_table.add_column("Crest dB", justify="right")
    metrics_table.add_column("Centroid Hz", justify="right")

    for stem_name, m in result.metrics.items():
        metrics_table.add_row(
            stem_name,
            f"{m.get('rms', 0.0):.4f}",
            f"{m.get('peak', 0.0):.4f}",
            f"{m.get('crest_db', 0.0):.2f}",
            f"{m.get('centroid_hz', 0.0):.1f}",
        )
    console.print(metrics_table)

    # Suggestions overview
    for group in ["vocals", "drums", "bass", "other", "master"]:
        opts = result.suggestions.get(group, [])
        if not opts:
            continue
        t = Table(title=f"Suggested Adjustments: {group}")
        t.add_column("#", justify="right", style="cyan")
        t.add_column("Option", style="white")
        t.add_column("Category", style="magenta")
        for idx, opt in enumerate(opts, 1):
            t.add_row(str(idx), opt.label, opt.category)
        console.print(t)


# ====================================
# Interactive selection
# ====================================

def select_single_option(group: str, options: List[AdjustOption]) -> Optional[AdjustOption] | List[AdjustOption]:
    if not options:
        return None

    console.print(Panel(f"Select an adjustment for [bold]{group}[/bold]", border_style="cyan"))
    choices_map: Dict[str, Optional[AdjustOption] | List[AdjustOption]] = {"0": None}
    console.print("  [cyan]0[/cyan]) Skip")
    # Offer apply-all option for this stem
    choices_map["A"] = options
    console.print("  [cyan]A[/cyan]) Apply all suggested")
    for i, opt in enumerate(options, 1):
        console.print(f"  [cyan]{i}[/cyan]) {opt.label} [dim]({opt.category})[/dim]")
        choices_map[str(i)] = opt

    sel = Prompt.ask("Your choice", choices=list(choices_map.keys()), default="0")
    return choices_map[sel]


def select_master_options(options: List[AdjustOption]) -> List[AdjustOption]:
    """Offer all master options together (e.g., multiband, normalization, limiter)."""
    choice = select_single_option("master", options)
    if isinstance(choice, list):
        return choice
    elif isinstance(choice, AdjustOption):
        return [choice]
    return []


def choose_mode() -> str:
    """Ask the user to apply all suggestions or manually select them."""
    console.print(Panel("Choose mode", border_style="cyan"))
    console.print("  [cyan]A[/cyan]) Apply all suggested")
    console.print("  [cyan]B[/cyan]) Select which adjustments to apply")
    sel = Prompt.ask("Mode", choices=["A", "B"], default="B")
    return sel


def interactive_selection(result: AnalysisResult) -> Dict[str, Optional[AdjustOption] | List[AdjustOption]]:
    selections: Dict[str, Optional[AdjustOption] | List[AdjustOption]] = {}
    mode = choose_mode()
    if mode == "A":
        # Apply all suggested for each stem and master
        for group in ["vocals", "drums", "bass", "other"]:
            selections[group] = list(result.suggestions.get(group, []))
        selections["master"] = list(result.suggestions.get("master", []))
        return selections

    # Manual selection per stem
    for group in ["vocals", "drums", "bass", "other"]:
        opt = select_single_option(group, result.suggestions.get(group, []))
        selections[group] = opt
    # Master options can include both normalization and limiting, chosen sequentially.
    selections["master"] = select_master_options(result.suggestions.get("master", []))
    return selections


# ====================================
# Confirmation
# ====================================

def summarize_choices(selections: Dict[str, Optional[AdjustOption] | List[AdjustOption]]) -> None:
    t = Table(title="Chosen Adjustments")
    t.add_column("Target", style="cyan")
    t.add_column("Selection", style="white")

    for group in ["vocals", "drums", "bass", "other"]:
        opt = selections.get(group)  # type: ignore[assignment]
        if isinstance(opt, AdjustOption):
            t.add_row(group, opt.label)
        elif isinstance(opt, list):
            if len(opt) == 0:
                t.add_row(group, "Skip")
            else:
                t.add_row(group, ", ".join(o.label for o in opt))
        else:
            t.add_row(group, "Skip")

    master_opts = selections.get("master", [])
    if isinstance(master_opts, list) and master_opts:
        t.add_row("master", ", ".join(o.label for o in master_opts))
    else:
        t.add_row("master", "Skip")

    console.print(t)


# ====================================
# Processing
# ====================================

def process_with_progress(result: AnalysisResult, selections: Dict[str, Optional[AdjustOption] | List[AdjustOption]]) -> np.ndarray:
    sr = result.sr
    processed_stems: Dict[str, np.ndarray] = {}

    # Process stems
    stem_order = ["vocals", "drums", "bass", "other"]
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Processing stems...", total=len(stem_order))
        for stem_name in stem_order:
            stem_audio = result.stems[stem_name]  # (C, N)
            opt = selections.get(stem_name)
            processed = stem_audio
            if isinstance(opt, AdjustOption):
                console.print(f"Applying: {opt.label} -> {stem_name} ...")
                processed = apply_option_to_stem(processed, sr, opt)
                console.print("Done.")
            elif isinstance(opt, list):
                if len(opt) > 0:
                    for o in opt:
                        console.print(f"Applying: {o.label} -> {stem_name} ...")
                        processed = apply_option_to_stem(processed, sr, o)
                    console.print("Done.")
            processed_stems[stem_name] = processed
            progress.advance(task)

    # Sum mix
    mix = sum(processed_stems[s] for s in stem_order)

    # Master options
    master_opts = selections.get("master", [])
    if isinstance(master_opts, list) and master_opts:
        console.print("Applying master bus processing...")
        for o in master_opts:
            console.print(f"  - {o.label}")
        mix = apply_master_options(mix, sr, master_opts)
        console.print("Master bus done.")

    # Safety clip
    mix = np.clip(mix, -1.0, 1.0)
    return mix


# ====================================
# Main loop
# ====================================

def run_once() -> None:
    ensure_directories()
    path = pick_input_file()
    if path is None:
        return

    loaded = load_audio_with_feedback(path)
    if loaded is None:
        return
    y, sr = loaded

    # Analyze
    result = run_analysis_with_progress(path, y, sr)

    # Report
    show_findings(result)

    # Interactive choices
    selections = interactive_selection(result)

    # Confirm
    console.print(Panel("Review your selections below.", title="Confirmation", border_style="yellow"))
    summarize_choices(selections)
    if not Confirm.ask("Proceed with processing?", default=True):
        console.print("Cancelled. Returning to main menu.")
        return

    # Processing
    mix = process_with_progress(result, selections)

    # Save
    out_path = build_output_path(Path(result.file_path))
    save_audio(out_path, mix, result.sr)

    # Waveform change report (before vs after)
    try:
        # Build applied context from the selections to enrich the PNG annotations
        applied_context: Dict[str, List[str]] = {}
        for grp in ["vocals", "drums", "bass", "other"]:
            opt = selections.get(grp)
            labels: List[str] = []
            if isinstance(opt, list):
                labels = [o.label for o in opt]
            elif isinstance(opt, AdjustOption):
                labels = [opt.label]
            if labels:
                applied_context[grp] = labels

        master_opts = selections.get("master", [])
        if isinstance(master_opts, list) and master_opts:
            applied_context["master"] = [o.label for o in master_opts]

        change_map, png_path, json_path = create_waveform_change_report(
            before=y,
            after=mix,
            sr=result.sr,
            out_audio_path=out_path,
            sibilant_thresh_db=0.6,
            band_thresh_db=0.4,
            applied_context=applied_context if applied_context else None,
        )

        # Summary panel
        summ = change_map.get("summary", {})
        png_line = f"Waveform report: [cyan]{png_path}[/cyan]" if png_path.exists() else (
            "Waveform report: not created (install matplotlib to enable plotting)."
        )
        msg = (
            f"Saved mastered file to\n[bold green]{out_path}[/bold green]\n\n"
            f"{png_line}\n"
            f"Change details: [cyan]{json_path}[/cyan]\n\n"
            f"Events — De-Esser: [bold]{summ.get('num_deesser_events', 0)}[/bold], "
            f"Per-band changes: [bold]{summ.get('num_band_events', 0)}[/bold]"
        )
        console.print(Panel(msg, title="Done", border_style="green"))

        # Show top few events inline
        events = change_map.get("events", [])
        if events:
            t = Table(title="Change Events (top 10)")
            t.add_column("Type", style="magenta")
            t.add_column("Band", style="cyan")
            t.add_column("Start (s)", justify="right")
            t.add_column("End (s)", justify="right")
            t.add_column("Delta dB", justify="right")
            # Sort by magnitude of change (most negative first)
            events_sorted = sorted(events, key=lambda e: e.get("avg_delta_db", 0.0))[:10]
            for ev in events_sorted:
                t.add_row(
                    str(ev.get("type", "")),
                    str(ev.get("band", "")),
                    f"{float(ev.get('start_sec', 0.0)):.3f}",
                    f"{float(ev.get('end_sec', 0.0)):.3f}",
                    f"{float(ev.get('avg_delta_db', 0.0)):.2f}",
                )
            console.print(t)
    except Exception as e:
        console.print(Panel(f"Saved mastered file to\n[bold green]{out_path}[/bold green]\n\n"
                            f"[yellow]Note[/yellow]: Failed to generate waveform report: {e}\n"
                            f"Install matplotlib to enable plotting: pip install matplotlib",
                            title="Done", border_style="green"))


def main() -> None:
    while True:
        welcome_screen()
        sel = main_menu()
        if sel == "1":
            run_once()
            if not Confirm.ask("Return to main menu?", default=True):
                break
        else:
            break
    console.print("Goodbye!")


if __name__ == "__main__":
    main()
