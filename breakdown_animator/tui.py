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
