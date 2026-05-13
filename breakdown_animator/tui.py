"""breakdown_animator.tui
======================
Interactive dark-mode Terminal User Interface (TUI) built with Textual
for inspecting 3D render breakdown sequences, configuring presets,
and executing live multi-phase animation exports.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Grid, Horizontal, ScrollableContainer, Vertical
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Checkbox,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    ProgressBar,
    Select,
    Static,
)

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
    SequenceEngineError,
    SequenceGroup,
    clean_folder_path,
    discover_sequence_folders,
    group_sequences,
    scan_directory,
)


# ============================================================================
# TUI Stylesheet (Dark Aesthetic)
# ============================================================================

TUI_CSS = """
Screen {
    background: #0f131a;
    color: #e2e8f0;
}

Header {
    background: #18202f;
    color: #38bdf8;
    text-style: bold;
    dock: top;
    height: 1;
}

Footer {
    background: #18202f;
    color: #94a3b8;
    dock: bottom;
    height: 1;
}

#top_bar {
    height: 3;
    padding: 0 1;
    background: #141b26;
    border-bottom: solid #2a374a;
}

#dir_input {
    width: 1fr;
    background: #1e293b;
    color: #f8fafc;
    border: tall #3b82f6;
}

#btn_scan {
    width: 16;
    margin-left: 1;
    background: #2563eb;
    color: #ffffff;
    text-style: bold;
}

#main_layout {
    layout: horizontal;
    height: 1fr;
    padding: 1;
}

#left_column {
    width: 1fr;
    margin-right: 1;
    background: #141b26;
    border: round #2a374a;
    padding: 1;
}

#right_column {
    width: 1fr;
    background: #141b26;
    border: round #2a374a;
    padding: 1;
}

.panel_title {
    color: #38bdf8;
    text-style: bold;
    margin-bottom: 1;
    border-bottom: solid #2a374a;
    padding-bottom: 0;
}

#sequence_table {
    height: 9;
    background: #0f131a;
    border: solid #2a374a;
    margin-bottom: 1;
}

#sequence_details {
    height: auto;
    background: #1e293b;
    border: solid #334155;
    padding: 1;
    color: #cbd5e1;
    margin-bottom: 1;
}

.control_row {
    height: 3;
    margin-bottom: 1;
    layout: horizontal;
}

.control_label {
    width: 16;
    color: #94a3b8;
    padding-top: 1;
}

.control_input {
    width: 1fr;
    background: #1e293b;
    color: #f8fafc;
}

#bottom_panel {
    height: 6;
    padding: 1;
    background: #141b26;
    border-top: solid #2a374a;
    dock: bottom;
}

#progress_bar {
    width: 1fr;
    margin-bottom: 1;
}

#status_message {
    color: #38bdf8;
    height: 1;
}

#action_bar {
    layout: horizontal;
    align: right middle;
    height: 3;
}

#btn_export {
    background: #10b981;
    color: #ffffff;
    text-style: bold;
    width: 22;
}

#btn_cancel {
    background: #ef4444;
    color: #ffffff;
    width: 14;
    margin-left: 1;
}

/* Modal Dialog */
SummaryModal {
    align: center middle;
}

#modal_dialog {
    width: 65;
    height: auto;
    background: #18202f;
    border: thick #38bdf8;
    padding: 2;
}

.badge_pass {
    color: #10b981;
    text-style: bold;
}

.badge_fail {
    color: #ef4444;
    text-style: bold;
}
"""


# ============================================================================
# Output Summary Modal
# ============================================================================

class SummaryModal(ModalScreen):
    """Modal dialog presenting export completion metrics and platform size limit badges."""

    def __init__(self, result: ExportResult, output_dir: Path):
        super().__init__()
        self.result = result
        self.output_dir = output_dir

    def compose(self) -> ComposeResult:
        res = self.result
        size_mb = res.file_size_mb

        artstation_badge = "[PASS] <= 10.0 MB" if size_mb <= 10.0 else "[FAIL] > 10.0 MB"
        discord_badge = "[PASS] <= 8.0 MB" if size_mb <= 8.0 else "[FAIL] > 8.0 MB"
        slack_badge = "[PASS] <= 5.0 MB" if size_mb <= 5.0 else "[FAIL] > 5.0 MB"

        art_class = "badge_pass" if size_mb <= 10.0 else "badge_fail"
        disc_class = "badge_pass" if size_mb <= 8.0 else "badge_fail"
        slack_class = "badge_pass" if size_mb <= 5.0 else "badge_fail"

        yield Vertical(
            Label("🎉 Animation Export Complete", classes="panel_title"),
            Static(f"• File: [bold cyan]{res.output_path.name}[/bold cyan]"),
            Static(f"• Format: [bold]{res.format.value.upper()}[/bold]"),
            Static(f"• Frames: [bold]{res.total_frames}[/bold] | Resolution: [bold]{res.dimensions[0]}x{res.dimensions[1]}[/bold]"),
            Static(f"• Duration: [bold]{res.duration_seconds:.2f}s[/bold] | Size: [bold]{size_mb:.2f} MB[/bold]"),
            Static(f"• Auto-Tuned: [bold]{'Yes (' + str(res.attempts) + ' passes)' if res.auto_tuned else 'No'}[/bold]"),
            Static(""),
            Label("Platform Size Compatibility:"),
            Static(f"  ArtStation (<=10 MB): [{art_class}]{artstation_badge}[/{art_class}]"),
            Static(f"  Discord (<=8 MB):    [{disc_class}]{discord_badge}[/{disc_class}]"),
            Static(f"  Slack (<=5 MB):      [{slack_class}]{slack_badge}[/{slack_class}]"),
            Static(""),
            Horizontal(
                Button("Open Output Folder", id="btn_open_folder", variant="primary"),
                Button("Close", id="btn_close_modal", variant="default"),
                classes="control_row",
            ),
            id="modal_dialog",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn_close_modal":
            self.dismiss()
        elif event.button.id == "btn_open_folder":
            self._open_folder(self.output_dir)

    def _open_folder(self, folder: Path) -> None:
        try:
            if sys.platform == "win32":
                os.startfile(str(folder))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(folder)])
            else:
                subprocess.Popen(["xdg-open", str(folder)])
        except Exception:
            pass


# ============================================================================
# Main Textual Application
# ============================================================================

class AnimForgeApp(App):
    """Breakdown Animator (animforge) Interactive Terminal User Interface."""

    TITLE = "AnimForge — 3D Breakdown Animator"
    SUB_TITLE = "Pure-Pillow GIF & WebP Sequence Engine"
    CSS = TUI_CSS

    BINDINGS = [
        Binding("q", "quit", "Quit", priority=True),
        Binding("r", "refresh_sequence", "Refresh / Scan"),
        Binding("e", "trigger_export", "Export Animation"),
        Binding("d", "toggle_dark", "Toggle Theme"),
    ]

    current_groups: reactive[Dict[str, SequenceGroup]] = reactive({})
    active_group_id: reactive[str] = reactive("")
    is_exporting: reactive[bool] = reactive(False)

    def __init__(self, initial_path: Optional[str] = None):
        super().__init__()
        self.initial_path = initial_path or ""
        self.cancel_token: Optional[threading.Event] = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        # Top Directory Input Bar
        yield Horizontal(
            Input(
                value=self.initial_path,
                placeholder="Enter or paste path to breakdown renders folder...",
                id="dir_input",
            ),
            Button("Scan Folder", id="btn_scan"),
            id="top_bar",
        )

        # Main 2-Column Working Layout
        yield Horizontal(
            # Left: Sequence Inspector & Group Browser
            Vertical(
                Label("📁 Sequence Inspector", classes="panel_title"),
                DataTable(id="sequence_table"),
                Static("No sequence loaded.", id="sequence_details"),
                id="left_column",
            ),
            # Right: Preset & Encoding Controls
            Vertical(
                Label("⚙️ Preset & Export Controls", classes="panel_title"),
                ScrollableContainer(
                    Horizontal(
                        Label("Preset:", classes="control_label"),
                        Select(
                            [
                                ("ArtStation Breakdown (1.0s/step)", "artstation"),
                                ("Detailed Study (1.8s/step)", "breakdown-slow"),
                                ("Brisk Breakdown (0.5s/step)", "breakdown-fast"),
                                ("3D Turntable (12 FPS)", "turntable"),
                                ("Portfolio 4K WebP", "portfolio-4k"),
                                ("Discord Breakdown (0.8s/step)", "discord"),
                                ("Slack Breakdown (0.8s/step)", "slack"),
                                ("Custom Settings", "custom"),
                            ],
                            value="artstation",
                            id="preset_select",
                            classes="control_input",
                        ),
                        classes="control_row",
                    ),
                    Horizontal(
                        Label("Format:", classes="control_label"),
                        Select(
                            [("GIF Animation", "gif"), ("WebP Animation", "webp")],
                            value="gif",
                            id="format_select",
                            classes="control_input",
                        ),
                        classes="control_row",
                    ),
                    Horizontal(
                        Label("Target FPS:", classes="control_label"),
                        Input(value="1.0", id="fps_input", classes="control_input"),
                        classes="control_row",
                    ),
                    Horizontal(
                        Label("Hold End (s):", classes="control_label"),
                        Input(value="2.0", id="hold_input", classes="control_input"),
                        classes="control_row",
                    ),
                    Horizontal(
                        Label("Scale Preset:", classes="control_label"),
                        Select(
                            [
                                ("Native 100%", "100%"),
                                ("4K UHD (3840x2160)", "4k"),
                                ("1080p FHD (1920x1080)", "1080p"),
                                ("720p HD (1280x720)", "720p"),
                                ("540p QHD (960x540)", "540p"),
                            ],
                            value="100%",
                            id="scale_select",
                            classes="control_input",
                        ),
                        classes="control_row",
                    ),
                    Horizontal(
                        Label("Max Size (MB):", classes="control_label"),
                        Input(value="10.0", id="budget_input", classes="control_input"),
                        classes="control_row",
                    ),
                    Horizontal(
                        Label("Colors / Qual:", classes="control_label"),
                        Input(value="256", id="colors_input", classes="control_input"),
                        classes="control_row",
                    ),
                    Horizontal(
                        Label("Dithering:", classes="control_label"),
                        Checkbox("Floyd-Steinberg Error Diffusion", value=True, id="dither_checkbox"),
                        classes="control_row",
                    ),
                    Horizontal(
                        Label("Harmonize:", classes="control_label"),
                        Select(
                            [
                                ("Letterbox (Padded Canvas)", "letterbox"),
                                ("Center Crop", "crop"),
                                ("Stretch", "stretch"),
                            ],
                            value="letterbox",
                            id="harmonize_select",
                            classes="control_input",
                        ),
                        classes="control_row",
                    ),
                ),
                id="right_column",
            ),
            id="main_layout",
        )

        # Bottom Progress & Actions Bar
        yield Vertical(
            ProgressBar(total=100, show_eta=False, id="progress_bar"),
            Horizontal(
                Static("Ready. Enter directory path or press Scan.", id="status_message"),
                Container(
                    Horizontal(
                        Button("Export Animation", id="btn_export", variant="success"),
                        Button("Cancel", id="btn_cancel", variant="error"),
                    ),
                    id="action_bar",
                ),
            ),
            id="bottom_panel",
        )

        yield Footer()

    def on_mount(self) -> None:
        """Initialize table columns and trigger scan if initial path provided."""
        table = self.query_one("#sequence_table", DataTable)
        table.cursor_type = "row"
        table.add_column("Camera / Pass", key="group_id")
        table.add_column("Frames", key="frames")
        table.add_column("Resolution", key="resolution")
        table.add_column("Aspect", key="aspect")

        if self.initial_path:
            self.scan_path(self.initial_path)

    # ------------------------------------------------------------------------
    # User Interactions & Event Handlers
    # ------------------------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "btn_scan":
            path_val = self.query_one("#dir_input", Input).value.strip()
            self.scan_path(path_val)
        elif button_id == "btn_export":
            self.start_export()
        elif button_id == "btn_cancel":
            self.cancel_export()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "dir_input":
            self.scan_path(event.value.strip())

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "preset_select":
            self._apply_preset(event.value)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        row_key = event.row_key.value
        if row_key in self.current_groups:
            self.active_group_id = row_key
            self._update_details_view(self.current_groups[row_key])

    def action_refresh_sequence(self) -> None:
        path_val = self.query_one("#dir_input", Input).value.strip()
        if path_val:
            self.scan_path(path_val)

    def action_trigger_export(self) -> None:
        self.start_export()

    # ------------------------------------------------------------------------
    # Preset Synchronization
    # ------------------------------------------------------------------------

    def _apply_preset(self, preset_name: str) -> None:
        try:
            profile = get_preset_profile(preset_name)
            self.query_one("#format_select", Select).value = profile.format.value
            self.query_one("#fps_input", Input).value = str(profile.fps)
            self.query_one("#hold_input", Input).value = str(profile.hold_last_seconds)
            self.query_one("#budget_input", Input).value = str(profile.max_size_mb) if profile.max_size_mb else ""
            if profile.format == ExportFormat.GIF:
                self.query_one("#colors_input", Input).value = str(profile.color_count)
            else:
                self.query_one("#colors_input", Input).value = str(profile.quality)
            self.query_one("#dither_checkbox", Checkbox).value = (profile.get_dither_mode() == DitherMode.FLOYD_STEINBERG)
        except Exception:
            pass

    # ------------------------------------------------------------------------
    # Sequence Scanning & Inspection
    # ------------------------------------------------------------------------

    def scan_path(self, path_str: str) -> None:
        status = self.query_one("#status_message", Static)
        details = self.query_one("#sequence_details", Static)
        table = self.query_one("#sequence_table", DataTable)

        if not path_str:
            status.update("[yellow]Please specify a sequence folder path.[/yellow]")
            return

        cleaned = clean_folder_path(path_str)
        p = Path(cleaned)
        if not p.exists() or not p.is_dir():
            status.update(f"[red]Directory does not exist:[/red] {path_str}")
            details.update("[red]Directory not found on disk.[/red]")
            table.clear()
            self.current_groups = {}
            self.active_group_id = ""
            return

        try:
            status.update(f"Scanning [cyan]{p.name}[/cyan]...")
            paths = scan_directory(p)
            if not paths:
                sub_dirs = discover_sequence_folders(p, max_depth=2, include_root_if_has_images=False)
                if sub_dirs:
                    p = sub_dirs[0]
                    paths = scan_directory(p)
                    status.update(f"[cyan]Found {len(sub_dirs)} subfolders. Loaded: {p.name}[/cyan]")
                else:
                    status.update("[yellow]No supported image frames found in folder.[/yellow]")
                    details.update("Folder contains zero valid images.")
                    table.clear()
                    self.current_groups = {}
                    self.active_group_id = ""
                    return

            groups = group_sequences(paths)
            self.current_groups = groups
            table.clear()

            first_group: Optional[SequenceGroup] = None
            for gid, group in groups.items():
                if first_group is None:
                    first_group = group
                label_name = group.camera_name or gid
                table.add_row(
                    label_name,
                    str(group.total_frames),
                    f"{group.common_width}x{group.common_height}",
                    group.aspect_ratio_str,
                    key=gid,
                )

            if first_group:
                self.active_group_id = first_group.group_id
                self._update_details_view(first_group)

            status.update(f"[green]Discovered {len(paths)} frames across {len(groups)} sequence pass(es).[/green]")

        except Exception as exc:
            status.update(f"[red]Scan failed:[/red] {exc}")
            details.update(f"[red]Error analyzing folder: {exc}[/red]")
            table.clear()
            self.current_groups = {}
            self.active_group_id = ""

    def _update_details_view(self, group: SequenceGroup) -> None:
        details = self.query_one("#sequence_details", Static)
        tag = group.camera_name or group.group_id
        mismatch_warn = " [yellow](Mixed Res - Will Harmonize)[/yellow]" if group.has_mismatched_dimensions else ""

        text = (
            f"[bold cyan]Sequence Pass:[/bold cyan] {tag}\n"
            f"[bold]• Frame Count:[/bold] {group.total_frames} frames\n"
            f"[bold]• Resolution:[/bold] {group.common_width}x{group.common_height}{mismatch_warn}\n"
            f"[bold]• Aspect Ratio:[/bold] {group.aspect_ratio_str} ({group.aspect_ratio:.2f})\n"
            f"[bold]• Uncompressed RAM:[/bold] ~{group.estimated_memory_mb:.1f} MB\n"
            f"[bold]• First Frame:[/bold] {group.frames[0].filename if group.frames else 'N/A'}"
        )
        details.update(text)

    # ------------------------------------------------------------------------
    # Export Execution Worker (Background Thread)
    # ------------------------------------------------------------------------

    def start_export(self) -> None:
        if self.is_exporting:
            return

        if not self.current_groups or not self.active_group_id:
            self.query_one("#status_message", Static).update("[yellow]Please load a valid sequence first.[/yellow]")
            return

        group = self.current_groups[self.active_group_id]
        input_dir = Path(clean_folder_path(self.query_one("#dir_input", Input).value.strip()))
        out_dir = input_dir / "output"

        # Build export profile from inputs
        fmt_str = self.query_one("#format_select", Select).value
        out_fmt = ExportFormat.from_str(fmt_str)
        ext = "gif" if out_fmt == ExportFormat.GIF else "webp"

        out_name = f"{group.group_id}_breakdown.{ext}"
        out_path = out_dir / out_name

        try:
            fps = float(self.query_one("#fps_input", Input).value.strip())
        except ValueError:
            fps = 1.0

        frame_duration = (1.0 / fps) if fps <= 3.0 else None

        try:
            hold = float(self.query_one("#hold_input", Input).value.strip())
        except ValueError:
            hold = 2.0

        budget_str = self.query_one("#budget_input", Input).value.strip()
        try:
            max_size_mb = float(budget_str) if budget_str else None
        except ValueError:
            max_size_mb = None

        colors_val = self.query_one("#colors_input", Input).value.strip()
        try:
            parsed_colors = int(colors_val)
        except ValueError:
            parsed_colors = 256

        scale_val = self.query_one("#scale_select", Select).value
        dither_val = self.query_one("#dither_checkbox", Checkbox).value
        harmonize_val = HarmonizeMode(self.query_one("#harmonize_select", Select).value)

        profile = ExportProfile(
            format=out_fmt,
            fps=fps,
            frame_duration=frame_duration,
            hold_last_seconds=hold,
            color_count=parsed_colors if out_fmt == ExportFormat.GIF else 256,
            quality=parsed_colors if out_fmt == ExportFormat.WEBP else 85,
            dither=DitherMode.FLOYD_STEINBERG if dither_val else DitherMode.NONE,
            scale=scale_val,
            max_size_mb=max_size_mb,
            auto_tune=bool(max_size_mb is not None),
            harmonize_mode=harmonize_val,
        )

        self.cancel_token = threading.Event()
        self.is_exporting = True
        self.run_export_worker(group.frame_paths, out_path, profile, out_dir)

    @work(exclusive=True, thread=True)
    def run_export_worker(
        self,
        frame_paths: List[Path],
        out_path: Path,
        profile: ExportProfile,
        out_dir: Path,
    ) -> None:
        pbar = self.query_one("#progress_bar", ProgressBar)
        status = self.query_one("#status_message", Static)

        def on_progress(phase: str, cur: int, total: int, pct: float, msg: str):
            self.call_from_thread(pbar.update, progress=pct * 100)
            self.call_from_thread(status.update, f"[{phase}] {msg}")

        try:
            result = export_sequence(
                frames=frame_paths,
                output_path=out_path,
                profile=profile,
                progress_callback=on_progress,
                cancel_token=self.cancel_token,
            )
            self.call_from_thread(self._on_export_success, result, out_dir)
        except InterruptedError:
            self.call_from_thread(status.update, "[yellow]Export cancelled by user.[/yellow]")
            self.call_from_thread(pbar.update, progress=0)
        except Exception as exc:
            self.call_from_thread(status.update, f"[red]Export failed:[/red] {exc}")
        finally:
            self.is_exporting = False

    def _on_export_success(self, result: ExportResult, out_dir: Path) -> None:
        status = self.query_one("#status_message", Static)
        status.update(f"[bold green]Export complete:[/bold green] {result.output_path.name} ({result.file_size_mb:.2f} MB)")
        self.push_screen(SummaryModal(result, out_dir))

    def cancel_export(self) -> None:
        if self.is_exporting and self.cancel_token:
            self.cancel_token.set()
            self.query_one("#status_message", Static).update("[yellow]Cancelling export...[/yellow]")
