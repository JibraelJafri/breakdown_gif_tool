"""breakdown_animator.wizard
========================
Sleek, fast, interactive terminal wizard for 3D/Tech artists.
Features arrow-key navigation, instant preset cards, drag-and-drop path
cleaning, live progress feedback, and post-export quick actions.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from rich.align import Align
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

from breakdown_animator.export_engine import (
    DitherMode,
    ExportFormat,
    ExportProfile,
    ExportResult,
    HarmonizeMode,
    export_sequence,
)
from breakdown_animator.presets import (
    PRESET_CONFIGS,
    PresetName,
    get_preset_profile,
)
from breakdown_animator.sequence_engine import (
    SequenceGroup,
    clean_folder_path,
    discover_sequence_folders,
    group_sequences,
    scan_directory,
)


class Key:
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    ENTER = "ENTER"
    SPACE = "SPACE"
    ESCAPE = "ESCAPE"
    BACKSPACE = "BACKSPACE"
    CHAR = "CHAR"
    OTHER = "OTHER"


def get_key() -> Tuple[str, str]:
    if sys.platform == "win32":
        import msvcrt
        ch = msvcrt.getch()
        if ch in (b"\x00", b"\xe0"):
            ch2 = msvcrt.getch()
            if ch2 == b"H":
                return (Key.UP, "")
            elif ch2 == b"P":
                return (Key.DOWN, "")
            elif ch2 == b"K":
                return (Key.LEFT, "")
            elif ch2 == b"M":
                return (Key.RIGHT, "")
            return (Key.OTHER, "")
        elif ch == b"\r":
            return (Key.ENTER, "")
        elif ch == b" ":
            return (Key.SPACE, " ")
        elif ch == b"\x1b":
            return (Key.ESCAPE, "")
        elif ch == b"\x08":
            return (Key.BACKSPACE, "")
        else:
            try:
                decoded = ch.decode("utf-8", errors="ignore")
                return (Key.CHAR, decoded)
            except Exception:
                return (Key.OTHER, "")
    else:
        import termios
        import tty
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
            if ch == "\x1b":
                import select
                r, _, _ = select.select([sys.stdin], [], [], 0.05)
                if r:
                    ch2 = sys.stdin.read(1)
                    if ch2 == "[":
                        ch3 = sys.stdin.read(1)
                        if ch3 == "A":
                            return (Key.UP, "")
                        elif ch3 == "B":
                            return (Key.DOWN, "")
                        elif ch3 == "C":
                            return (Key.RIGHT, "")
                        elif ch3 == "D":
                            return (Key.LEFT, "")
                return (Key.ESCAPE, "")
            elif ch in ("\r", "\n"):
                return (Key.ENTER, "")
            elif ch == " ":
                return (Key.SPACE, " ")
            elif ch in ("\x7f", "\x08"):
                return (Key.BACKSPACE, "")
            else:
                return (Key.CHAR, ch)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def prompt_select(
    console: Console,
    title: str,
    options: List[Tuple[str, str, str]],
    default_index: int = 0,
) -> str:
    if not sys.stdin.isatty():
        return options[default_index][0]

    selected = default_index
    total = len(options)

    def render_menu() -> Table:
        table = Table.grid(padding=(0, 1))
        table.add_column(justify="left")
        
        table.add_row(Text.from_markup(f"[bold cyan]? {title}[/bold cyan] [dim](Use ↑/↓ arrows, Enter to confirm)[/dim]\n"))
        
        for idx, (opt_key, opt_title, opt_sub) in enumerate(options):
            is_active = (idx == selected)
            if is_active:
                prefix = "[bold cyan] ❯ [/bold cyan]"
                row_text = Text.from_markup(f"{prefix}[bold white on #1e3a8a] {opt_title:<30} [/bold white on #1e3a8a] [dim cyan]{opt_sub}[/dim cyan]")
            else:
                prefix = "   "
                row_text = Text.from_markup(f"{prefix}[bold]{opt_title:<30}[/bold] [dim]{opt_sub}[/dim]")
            table.add_row(row_text)
        
        table.add_row(Text.from_markup("\n"))
        return table

    with Live(render_menu(), console=console, refresh_per_second=30, transient=True) as live:
        while True:
            live.update(render_menu())
            ktype, kval = get_key()
            if ktype == Key.UP:
                selected = (selected - 1) % total
            elif ktype == Key.DOWN:
                selected = (selected + 1) % total
            elif ktype == Key.ENTER:
                break
            elif ktype == Key.CHAR and kval in ("q", "Q"):
                console.print("[yellow]Wizard cancelled.[/yellow]")
                sys.exit(0)

    chosen = options[selected]
    console.print(f"[bold cyan]✔ {title}[/bold cyan] [bold green]› {chosen[1]}[/bold green]")
    return chosen[0]


def prompt_folder_path(console: Console, initial_path: Optional[str] = None) -> Path:
    current_val = initial_path or ""
    
    while True:
        if current_val:
            raw_path = current_val
            current_val = ""
        else:
            console.print("\n[bold cyan]? 📁 Drag & drop or enter breakdown render folder (or parent batch directory):[/bold cyan] [dim](or 'q' to quit)[/dim]")
            raw_path = console.input("[bold blue]  ❯ [/bold blue]").strip()

        if raw_path.lower() in ("q", "quit", "exit"):
            console.print("[yellow]Cancelled.[/yellow]")
            sys.exit(0)

        cleaned = clean_folder_path(raw_path)
        p = Path(cleaned).expanduser().resolve()
        if not p.exists():
            console.print(f"  [bold red]✖ Directory does not exist:[/bold red] [dim]{p}[/dim]")
            continue
        if not p.is_dir():
            console.print(f"  [bold red]✖ Path is not a directory:[/bold red] [dim]{p}[/dim]")
            continue

        try:
            frames = scan_directory(p)
            if not frames:
                sub_dirs = discover_sequence_folders(p, max_depth=2, include_root_if_has_images=False)
                if not sub_dirs:
                    console.print(f"  [bold yellow]⚠ No valid image frames found in:[/bold yellow] [dim]{p}[/dim]")
                    console.print("  [dim]Supported: .png, .jpg, .jpeg, .tga, .bmp, .webp, .tif[/dim]")
                    continue
            return p
        except Exception as exc:
            console.print(f"  [bold red]✖ Error scanning directory:[/bold red] {exc}")
            continue


def open_folder_in_explorer(folder: Path) -> None:
    try:
        if sys.platform == "win32":
            os.startfile(str(folder))
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(folder)])
        else:
            subprocess.Popen(["xdg-open", str(folder)])
    except Exception:
        pass


def copy_to_clipboard(text: str) -> bool:
    try:
        if sys.platform == "win32":
            subprocess.run("clip", input=text.strip().encode("utf-16"), check=True)
            return True
        elif sys.platform == "darwin":
            subprocess.run("pbcopy", input=text.strip().encode("utf-8"), check=True)
            return True
        elif sys.platform.startswith("linux"):
            subprocess.run(["xclip", "-selection", "clipboard"], input=text.strip().encode("utf-8"), check=True)
            return True
    except Exception:
        pass
    return False


def run_wizard(initial_path: Optional[str] = None) -> int:
    if sys.platform == "win32":
        try:
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            if hasattr(sys.stderr, "reconfigure"):
                sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    console = Console(safe_box=True)

    console.print(Panel(
        Align.center(
            "[bold white]✨ AnimForge — 3D Breakdown Studio[/bold white]\n"
            "[dim cyan]Zero-Binary Animated GIF & WebP Converter for 3D & Tech Artists[/dim cyan]"
        ),
        border_style="cyan",
        padding=(1, 2),
    ))

    while True:
        folder = prompt_folder_path(console, initial_path=initial_path)
        initial_path = None

        # Check for subfolder sequences
        direct_frames = scan_directory(folder) if folder.is_dir() else []
        sub_dirs = discover_sequence_folders(folder, max_depth=2, include_root_if_has_images=False)

        target_folders: List[Path] = []
        if sub_dirs:
            if direct_frames:
                sub_options = [
                    ("batch_all", f"⚡ Batch process all {len(sub_dirs) + 1} folders (root + subfolders)", "Export all render passes across all folders"),
                    ("root_only", f"📁 Process root folder only ({len(direct_frames)} frames)", "Ignore subdirectories"),
                    ("subfolders_only", f"🗂️ Batch process {len(sub_dirs)} subfolders only", "Process subfolders only"),
                ]
                choice = prompt_select(console, "Multiple sequence folders detected:", sub_options, default_index=0)
                if choice == "batch_all":
                    target_folders = [folder] + sub_dirs
                elif choice == "root_only":
                    target_folders = [folder]
                else:
                    target_folders = sub_dirs
            else:
                sub_options = [
                    ("all", f"⚡ Batch process all {len(sub_dirs)} subfolders", "Export all detected sequences automatically"),
                ]
                for sd in sub_dirs:
                    sub_options.append((str(sd), f"📁 {sd.name}", "Process this subfolder only"))
                choice = prompt_select(console, f"Found {len(sub_dirs)} sequence folders:", sub_options, default_index=0)
                if choice == "all":
                    target_folders = sub_dirs
                else:
                    target_folders = [Path(choice)]
        else:
            target_folders = [folder]

        # Scan sequence passes for each target folder
        folder_groups: List[Tuple[Path, Dict[str, SequenceGroup]]] = []
        table = Table(title="🔍 Discovered Sequence Passes", border_style="blue", show_header=True, header_style="bold cyan")
        if len(target_folders) > 1:
            table.add_column("Folder", style="dim cyan")
        table.add_column("Pass / Camera", style="bold white")
        table.add_column("Frames", justify="right", style="green")
        table.add_column("Resolution", justify="center")
        table.add_column("Aspect", justify="center")
        table.add_column("RAM Footprint", justify="right")

        for fld in target_folders:
            f_frames = scan_directory(fld)
            f_groups = group_sequences(f_frames)
            if f_groups:
                folder_groups.append((fld, f_groups))
                for gid, grp in f_groups.items():
                    label = grp.camera_name or gid
                    if len(target_folders) > 1:
                        table.add_row(
                            fld.name,
                            label,
                            str(grp.total_frames),
                            f"{grp.common_width}x{grp.common_height}",
                            grp.aspect_ratio_str,
                            f"~{grp.estimated_memory_mb:.1f} MB",
                        )
                    else:
                        table.add_row(
                            label,
                            str(grp.total_frames),
                            f"{grp.common_width}x{grp.common_height}",
                            grp.aspect_ratio_str,
                            f"~{grp.estimated_memory_mb:.1f} MB",
                        )

        console.print(table)
        console.print("")

        preset_options = [
            ("artstation", "🎨 ArtStation Breakdown (1.0s/step)", "≤ 10MB budget, 1.0s step duration, 2.0s hold pause (Auto-tuned)"),
            ("breakdown-slow", "🔍 Detailed Study Breakdown (1.8s/step)", "≤ 10MB budget, 1.8s step duration, 3.0s hold pause"),
            ("breakdown-fast", "⚡ Brisk Breakdown (0.5s/step)", "≤ 10MB budget, 0.5s step duration, 1.5s hold pause"),
            ("portfolio-4k", "🌐 4K Portfolio WebP (1.0s/step)", "True 24-bit color, VP8 high quality, 2.0s hold"),
            ("turntable", "🔄 3D Turntable (12 FPS continuous)", "Smooth 12 FPS playback for 360 rotating models or dissolves"),
            ("discord", "💬 Discord Breakdown (0.8s/step)", "≤ 8MB budget, 0.8s step duration, 1.5s hold pause"),
            ("slack", "⚡ Slack Breakdown (0.8s/step)", "≤ 5MB budget, 0.8s step duration, 1.0s hold pause"),
            ("custom", "🛠️ Custom Parameters", "Customize Step Duration, Hold, Resolution & Colors"),
        ]

        preset_key = prompt_select(console, "Select Target Preset:", preset_options, default_index=0)
        profile = get_preset_profile(preset_key)

        if preset_key == "custom":
            console.print("\n[bold cyan]🛠️ Configure Custom Settings:[/bold cyan]")
            
            fmt_opt = prompt_select(console, "Format:", [
                ("gif", "Animated GIF", "Broad compatibility"),
                ("webp", "Animated WebP", "High color depth, 85% smaller"),
            ])
            profile.format = ExportFormat.from_str(fmt_opt)

            timing_mode = prompt_select(console, "Timing Mode:", [
                ("step", "⏱️ Seconds per step (Breakdown slideshow)", "Recommended: 0.5s to 2.0s per stage"),
                ("fps", "🎞️ Frames per second (Turntable animation)", "For continuous 3D rotation or video"),
            ])
            if timing_mode == "step":
                dur_str = console.input("  • Seconds per step [default: 1.0s]: ").strip()
                if dur_str:
                    try:
                        profile.frame_duration = float(dur_str)
                        profile.fps = max(0.1, 1.0 / profile.frame_duration)
                    except ValueError:
                        pass
            else:
                fps_str = console.input(f"  • Target FPS [default: {profile.fps}]: ").strip()
                if fps_str:
                    try:
                        profile.fps = float(fps_str)
                        profile.frame_duration = None
                    except ValueError:
                        pass
            
            hold_str = console.input(f"  • Hold final frame (seconds) [default: {profile.hold_last_seconds}s]: ").strip()
            if hold_str:
                try:
                    profile.hold_last_seconds = float(hold_str)
                except ValueError:
                    pass
            
            scale_opt = prompt_select(console, "Resolution Scale:", [
                ("100%", "Native 100%", "Original render dimensions"),
                ("4k", "4K UHD (3840x2160)", "Downscale or upscale to 4K"),
                ("1080p", "1080p FHD (1920x1080)", "Standard HD presentation"),
                ("720p", "720p HD (1280x720)", "Fast, compact resolution"),
            ])
            profile.scale = scale_opt

            if profile.format == ExportFormat.GIF:
                mb_str = console.input("  • Max file size budget in MB [e.g. 10.0 or enter to skip]: ").strip()
                if mb_str:
                    try:
                        profile.max_size_mb = float(mb_str)
                        profile.auto_tune = True
                    except ValueError:
                        pass

        # Camera selection (if single folder with multiple passes)
        export_jobs: List[Tuple[Path, Dict[str, SequenceGroup]]] = []
        if len(folder_groups) == 1:
            fld, grps = folder_groups[0]
            if len(grps) > 1:
                cam_options = [("all", f"🌟 All Passes ({len(grps)} sequences)", "Batch render all cameras")]
                for gid, grp in grps.items():
                    cam_label = grp.camera_name or gid
                    cam_options.append((gid, f"🎥 {cam_label}", f"{grp.total_frames} frames ({grp.common_width}x{grp.common_height})"))
                
                cam_choice = prompt_select(console, "Select Passes to Export:", cam_options, default_index=0)
                if cam_choice == "all":
                    export_jobs.append((fld, grps))
                else:
                    export_jobs.append((fld, {cam_choice: grps[cam_choice]}))
            else:
                export_jobs.append((fld, grps))
        else:
            export_jobs = folder_groups

        total_passes = sum(len(grps) for _, grps in export_jobs)
        console.print(f"\n[bold cyan]🚀 Rendering {total_passes} animation(s) across {len(export_jobs)} folder(s)...[/bold cyan]")
        start_time = time.time()
        results: List[ExportResult] = []
        is_multi_folder = len(export_jobs) > 1

        last_out_dir: Optional[Path] = None

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            for fld_idx, (fld, grps) in enumerate(export_jobs, start=1):
                out_dir = fld / "output"
                out_dir.mkdir(parents=True, exist_ok=True)
                last_out_dir = out_dir

                for gid, grp in grps.items():
                    cam_name = grp.camera_name or gid
                    ext = "gif" if profile.format == ExportFormat.GIF else "webp"
                    if is_multi_folder:
                        out_file = out_dir / f"{fld.name}_{gid}_breakdown.{ext}"
                        display_name = f"[{fld_idx}/{len(export_jobs)}] {fld.name}/{cam_name}"
                    else:
                        out_file = out_dir / f"{gid}_breakdown.{ext}"
                        display_name = cam_name

                    task = progress.add_task(f"Rendering {display_name}...", total=100)

                    def make_progress_cb(t_id: TaskID, cname: str):
                        def cb(phase: str, cur: int, total: int, pct: float, msg: str):
                            progress.update(t_id, completed=int(pct * 100), description=f"[{phase}] {cname}")
                        return cb

                    try:
                        res = export_sequence(
                            frames=grp.frame_paths,
                            output_path=out_file,
                            profile=profile,
                            progress_callback=make_progress_cb(task, display_name),
                        )
                        results.append(res)
                        progress.update(task, completed=100, description=f"[bold green][OK] Done:[/bold green] {display_name}")
                    except Exception as exc:
                        progress.update(task, description=f"[bold red][FAIL] Failed:[/bold red] {display_name}")
                        console.print(f"[bold red]Failed exporting pass '{gid}' in '{fld.name}':[/bold red] {exc}")

        elapsed = time.time() - start_time

        console.print("\n" + "━" * 70)
        console.print(f"[bold green]🎉 Export Complete![/bold green] [dim](Elapsed: {elapsed:.1f}s)[/dim]\n")

        summary_table = Table(border_style="green", show_header=True, header_style="bold cyan")
        summary_table.add_column("File Name", style="bold white")
        summary_table.add_column("Resolution", justify="center")
        summary_table.add_column("Duration", justify="right")
        summary_table.add_column("File Size", justify="right", style="bold")
        summary_table.add_column("ArtStation (≤10M)", justify="center")
        summary_table.add_column("Discord (≤8M)", justify="center")

        for r in results:
            art_badge = "[bold green]PASS[/bold green]" if r.file_size_mb <= 10.0 else "[bold red]FAIL[/bold red]"
            disc_badge = "[bold green]PASS[/bold green]" if r.file_size_mb <= 8.0 else "[bold red]FAIL[/bold red]"
            summary_table.add_row(
                r.output_path.name,
                f"{r.dimensions[0]}x{r.dimensions[1]}",
                f"{r.duration_seconds:.2f}s",
                f"{r.file_size_mb:.2f} MB",
                art_badge,
                disc_badge,
            )

        console.print(summary_table)
        if last_out_dir:
            console.print(f"📁 [bold]Saved to output directories (e.g.):[/bold] [cyan]{last_out_dir.resolve()}[/cyan]")
        console.print("━" * 70)

        action_options = [
            ("open", "📂 Open Output Folder in File Explorer", "Launch native folder browser"),
            ("copy", "📋 Copy Output Folder Path", "Copy path string to clipboard"),
            ("another", "🔄 Render Another Sequence", "Pick another render folder"),
            ("quit", "🚪 Exit AnimForge", "Quit to terminal"),
        ]

        action_choice = prompt_select(console, "Next Action:", action_options, default_index=0)

        if action_choice == "open" and last_out_dir:
            open_folder_in_explorer(last_out_dir)
            console.print("[green]✔ Opened folder in Explorer.[/green]")
            break
        elif action_choice == "copy" and last_out_dir:
            if copy_to_clipboard(str(last_out_dir.resolve())):
                console.print(f"[green]✔ Copied to clipboard:[/green] {last_out_dir.resolve()}")
            else:
                console.print(f"[cyan]Path:[/cyan] {last_out_dir.resolve()}")
            break
        elif action_choice == "another":
            console.print("\n")
            continue
        else:
            console.print("[dim]Goodbye![/dim]")
            break

    return 0
