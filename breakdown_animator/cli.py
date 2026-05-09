"""breakdown_animator.cli
======================
High-performance, Rich-formatted headless Command Line Interface for
batch conversion of 3D render sequences to optimized GIFs and WebPs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from breakdown_animator.export_engine import (
    DitherMode,
    ExportFormat,
    ExportProfile,
    ExportResult,
    HarmonizeMode,
    QuantizeMethod,
    export_sequence,
)
from breakdown_animator.presets import (
    PRESET_CONFIGS,
    PresetName,
    get_preset_profile,
)
from breakdown_animator.sequence_engine import (
    DirectoryNotFoundError,
    EmptySequenceError,
    SequenceEngineError,
    SequenceGroup,
    clean_folder_path,
    discover_sequence_folders,
    group_sequences,
    scan_directory,
)


def build_parser() -> argparse.ArgumentParser:
    """Constructs and returns the CLI ArgumentParser."""
    parser = argparse.ArgumentParser(
        prog="animforge",
        description="animforge (breakdown-animator) — 3D Render Breakdown Animation Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "folder",
        nargs="?",
        default=None,
        help="Path to directory containing breakdown render image sequences (or first of multiple folders).",
    )
    parser.add_argument(
        "extra_folders",
        nargs="*",
        default=[],
        help="Additional folders to bulk process in batch mode.",
    )
    parser.add_argument(
        "--bulk",
        "-b",
        action="store_true",
        help="Automatically discover and batch convert all subfolders containing image sequences.",
    )
    parser.add_argument(
        "--step-duration",
        "--frame-duration",
        "-d",
        type=float,
        default=None,
        dest="frame_duration",
        help="Display duration per breakdown step in seconds (e.g. 1.0 or 1.5). Overrides --fps.",
    )
    parser.add_argument(
        "--preset",
        "-p",
        default="artstation",
        choices=[
            "artstation",
            "breakdown",
            "breakdown-slow",
            "breakdown-fast",
            "turntable",
            "portfolio-4k",
            "portfolio_4k",
            "discord",
            "slack",
            "custom",
        ],
        help="Optimization preset profile.",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["gif", "webp"],
        default=None,
        help="Explicit output animation format (overrides preset).",
    )
    parser.add_argument(
        "--camera",
        "-c",
        default="all",
        help="Target camera/pass name to export, or 'all' to batch convert all discovered passes.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Playback framerate in frames per second (FPS).",
    )
    parser.add_argument(
        "--hold-last",
        type=float,
        default=None,
        help="Freeze duration on final frame in seconds (e.g. 2.0).",
    )
    parser.add_argument(
        "--max-size-mb",
        type=float,
        default=None,
        help="Target file size budget in megabytes (triggers auto-tuning).",
    )
    parser.add_argument(
        "--scale",
        "-s",
        default=None,
        help="Scale preset (100%%, 4k, 1080p, 720p, 540p, 50%%, or WxH).",
    )
    parser.add_argument(
        "--colors",
        type=int,
        default=None,
        help="Palette color count for GIF (2-256) or quality for WebP (1-100).",
    )
    parser.add_argument(
        "--dither",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable Floyd-Steinberg dithering for GIF palette quantization.",
    )
    parser.add_argument(
        "--harmonize",
        choices=["letterbox", "fit", "crop", "stretch"],
        default="letterbox",
        help="Dimension harmonization mode for mixed resolution passes.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help="Output directory for generated animation files (defaults to <folder>/output).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect sequence metadata and planned conversions without encoding files.",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress informational console output.",
    )
    parser.add_argument(
        "--tui",
        action="store_true",
        help="Launch the interactive Textual TUI interface.",
    )

    return parser


def run_cli(args: Optional[Sequence[str]] = None) -> int:
    """
    Executes the headless CLI workflow with single-folder or batch multi-folder processing.
    
    Returns:
        Exit code: 0 on success, 1 on error.
    """
    if sys.platform == "win32":
        try:
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            if hasattr(sys.stderr, "reconfigure"):
                sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = build_parser()
    parsed_args = parser.parse_args(args)
    console = Console(quiet=parsed_args.quiet, safe_box=True)

    if not parsed_args.folder:
        console.print("[bold red]Error:[/bold red] No folder path specified.")
        parser.print_help()
        return 1

    # 1. Collect & clean all target directories
    raw_inputs = [parsed_args.folder] + list(parsed_args.extra_folders)
    target_dirs: List[Path] = []

    for raw in raw_inputs:
        cleaned = clean_folder_path(raw)
        p = Path(cleaned)
        if not p.exists() or not p.is_dir():
            console.print(f"[bold red]Error:[/bold red] Directory does not exist: [yellow]{p}[/yellow]")
            return 1

        if parsed_args.bulk:
            sub_dirs = discover_sequence_folders(p, max_depth=2)
            if sub_dirs:
                for sd in sub_dirs:
                    if sd not in target_dirs:
                        target_dirs.append(sd)
            else:
                target_dirs.append(p)
        else:
            try:
                direct_frames = scan_directory(p)
                if direct_frames:
                    if p not in target_dirs:
                        target_dirs.append(p)
                else:
                    # Check if subfolders contain sequences
                    sub_dirs = discover_sequence_folders(p, max_depth=2, include_root_if_has_images=False)
                    if sub_dirs:
                        console.print(f"[bold cyan]Notice:[/bold cyan] Discovered [bold]{len(sub_dirs)}[/bold] subfolder sequences in [yellow]{p.name}[/yellow].")
                        for sd in sub_dirs:
                            if sd not in target_dirs:
                                target_dirs.append(sd)
                    else:
                        if p not in target_dirs:
                            target_dirs.append(p)
            except Exception:
                if p not in target_dirs:
                    target_dirs.append(p)

    # 2. Discover and group sequences per directory
    folder_jobs: List[Tuple[Path, Dict[str, SequenceGroup]]] = []
    for fld in target_dirs:
        try:
            paths = scan_directory(fld)
            if not paths:
                console.print(f"[bold red]Error:[/bold red] No valid image sequence frames found in [yellow]{fld}[/yellow]")
                return 1
            groups = group_sequences(paths)
            if not groups:
