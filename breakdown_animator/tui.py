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

