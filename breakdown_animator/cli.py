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
                console.print(f"[bold red]Error:[/bold red] No sequence passes could be parsed from [yellow]{fld}[/yellow]")
                return 1

            # Filter camera passes
            target_groups: Dict[str, SequenceGroup] = {}
            if parsed_args.camera.lower() == "all":
                target_groups = groups
            else:
                cam_target = parsed_args.camera.strip()
                matched = False
                for gid, grp in groups.items():
                    if gid.lower() == cam_target.lower() or (grp.camera_name and grp.camera_name.lower() == cam_target.lower()):
                        target_groups[gid] = grp
                        matched = True
                if not matched:
                    console.print(f"[bold red]Error:[/bold red] Camera/pass '[yellow]{cam_target}[/yellow]' not found in folder.")
                    console.print(f"Available passes: {', '.join(groups.keys())}")
                    return 1

            folder_jobs.append((fld, target_groups))
        except Exception as exc:
            console.print(f"[bold red]Error scanning sequence in '{fld}':[/bold red] {exc}")
            return 1

    # 3. Dry-run inspection
    if parsed_args.dry_run:
        console.print(Panel.fit(
            f"[bold cyan]AnimForge Sequence Dry-Run Inspection[/bold cyan]\n"
            f"Target Folders: [bold]{len(folder_jobs)}[/bold]",
            border_style="cyan",
        ))

        table = Table(title="Discovered Sequence Passes", border_style="blue")
        table.add_column("Folder", style="dim cyan")
        table.add_column("Pass / Camera ID", style="bold cyan")
        table.add_column("Frames", justify="right", style="green")
        table.add_column("Resolution", justify="center")
        table.add_column("Aspect", justify="center")
        table.add_column("Uncompressed RAM", justify="right")
        table.add_column("Status", justify="center")

        for fld, t_groups in folder_jobs:
            for gid, grp in t_groups.items():
                cam_label = grp.camera_name or gid
                status_tag = "[yellow]Mixed Res[/yellow]" if grp.has_mismatched_dimensions else "[green]Uniform[/green]"
                table.add_row(
                    fld.name,
                    cam_label,
                    str(grp.total_frames),
                    f"{grp.common_width}x{grp.common_height}",
                    grp.aspect_ratio_str,
                    f"~{grp.estimated_memory_mb:.1f} MB",
                    status_tag,
                )
        console.print(table)
        console.print("[dim]Dry run complete. No animation files written.[/dim]")
        return 0

    # 4. Resolve Export Profile
    base_profile = get_preset_profile(parsed_args.preset)

    if parsed_args.format:
        base_profile.format = ExportFormat.from_str(parsed_args.format)
    if parsed_args.frame_duration is not None:
        base_profile.frame_duration = parsed_args.frame_duration
        base_profile.fps = max(0.1, 1.0 / parsed_args.frame_duration)
    elif parsed_args.fps is not None:
        base_profile.fps = parsed_args.fps
        base_profile.frame_duration = None
    if parsed_args.hold_last is not None:
        base_profile.hold_last_seconds = parsed_args.hold_last
    if parsed_args.max_size_mb is not None:
        base_profile.max_size_mb = parsed_args.max_size_mb
        base_profile.auto_tune = True
    if parsed_args.scale:
        base_profile.scale = parsed_args.scale
    if parsed_args.colors is not None:
        if base_profile.format == ExportFormat.GIF:
            base_profile.color_count = parsed_args.colors
        else:
            base_profile.quality = parsed_args.colors
    if parsed_args.dither is not None:
        base_profile.dither = DitherMode.FLOYD_STEINBERG if parsed_args.dither else DitherMode.NONE
    if parsed_args.harmonize:
        base_profile.harmonize_mode = HarmonizeMode(parsed_args.harmonize)

    total_passes = sum(len(t_groups) for _, t_groups in folder_jobs)
    console.print(Panel.fit(
        f"[bold cyan]AnimForge Export Pipeline[/bold cyan]\n"
        f"Folders: [bold]{len(folder_jobs)}[/bold] | Total Sequence Passes: [bold]{total_passes}[/bold] | Preset: [bold]{parsed_args.preset}[/bold] | Format: [bold]{base_profile.format.value.upper()}[/bold]",
        border_style="cyan",
    ))

    # 5. Execute Export Batch
    results: List[ExportResult] = []
    is_multi_folder = len(folder_jobs) > 1

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        for fld_idx, (fld, t_groups) in enumerate(folder_jobs, start=1):
            out_dir = Path(parsed_args.output_dir) if parsed_args.output_dir else (fld / "output")
            out_dir.mkdir(parents=True, exist_ok=True)

            for gid, grp in t_groups.items():
                cam_name = grp.camera_name or gid
                ext = "gif" if base_profile.format == ExportFormat.GIF else "webp"
                if is_multi_folder:
                    out_file = out_dir / f"{fld.name}_{gid}_breakdown.{ext}"
                    display_name = f"[{fld_idx}/{len(folder_jobs)}] {fld.name}/{cam_name}"
                else:
                    out_file = out_dir / f"{gid}_breakdown.{ext}"
                    display_name = cam_name

                task = progress.add_task(f"Exporting {display_name}...", total=100)

                def make_progress_cb(t_id: TaskID, dname: str):
                    def cb(phase: str, cur: int, total: int, pct: float, msg: str):
                        progress.update(t_id, completed=int(pct * 100), description=f"[{phase}] {dname}")
                    return cb

                try:
                    res = export_sequence(
                        frames=grp.frame_paths,
                        output_path=out_file,
                        profile=base_profile,
                        progress_callback=make_progress_cb(task, display_name),
                    )
                    results.append(res)
                    progress.update(task, completed=100, description=f"[green][OK] Done:[/green] {display_name}")
                except Exception as exc:
                    progress.update(task, description=f"[red][FAIL] Failed:[/red] {display_name}")
                    console.print(f"[bold red]Failed exporting pass '{gid}' in '{fld.name}':[/bold red] {exc}")
                    return 1

    # 6. Summary Table
    summary_table = Table(title="Export Summary & Compatibility", border_style="green")
    summary_table.add_column("Output File", style="bold cyan")
    summary_table.add_column("Frames", justify="right")
    summary_table.add_column("Resolution", justify="center")
    summary_table.add_column("Duration", justify="right")
    summary_table.add_column("File Size", justify="right", style="bold")
    summary_table.add_column("ArtStation (<=10M)", justify="center")
    summary_table.add_column("Discord (<=8M)", justify="center")
    summary_table.add_column("Slack (<=5M)", justify="center")

    for r in results:
        art_pass = "[bold green]PASS[/bold green]" if r.file_size_mb <= 10.0 else "[bold red]FAIL[/bold red]"
        disc_pass = "[bold green]PASS[/bold green]" if r.file_size_mb <= 8.0 else "[bold red]FAIL[/bold red]"
        slack_pass = "[bold green]PASS[/bold green]" if r.file_size_mb <= 5.0 else "[bold red]FAIL[/bold red]"

        summary_table.add_row(
            r.output_path.name,
            str(r.total_frames),
            f"{r.dimensions[0]}x{r.dimensions[1]}",
            f"{r.duration_seconds:.2f}s",
            f"{r.file_size_mb:.2f} MB",
            art_pass,
            disc_pass,
            slack_pass,
        )

    console.print(summary_table)
    dest_msg = f"output directory: [bold]{Path(parsed_args.output_dir).resolve()}[/bold]" if parsed_args.output_dir else "their respective output folders"
    console.print(f"[bold green]✨ Successfully exported {len(results)} animation(s) to {dest_msg}.[/bold green]")
    return 0
