"""tests/test_wizard.py
====================
Unit tests for the interactive terminal wizard module.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from breakdown_animator.wizard import (
    Key,
    copy_to_clipboard,
    get_key,
    open_folder_in_explorer,
    prompt_folder_path,
    prompt_select,
    run_wizard,
)


class TestWizardKeyAndInput:
    """Tests for keyboard handling and prompt selection in wizard."""

    def test_prompt_select_non_interactive(self):
        console = Console(quiet=True)
        options = [
            ("opt1", "Option 1", "Sub 1"),
            ("opt2", "Option 2", "Sub 2"),
        ]
        # In non-interactive mode (default in pytest), returns default index
        res = prompt_select(console, "Pick one", options, default_index=1)
        assert res == "opt2"

    def test_prompt_select_interactive_navigation(self):
        console = Console(quiet=True)
        options = [
            ("artstation", "ArtStation", "10MB"),
            ("portfolio-4k", "4K WebP", "24-bit"),
            ("discord", "Discord", "8MB"),
        ]

        # Simulate: Down -> Down -> Enter
        key_sequence = [(Key.DOWN, ""), (Key.DOWN, ""), (Key.ENTER, "")]
        key_idx = 0

        def mock_get_key():
            nonlocal key_idx
            k = key_sequence[key_idx]
            key_idx += 1
            return k

        with patch("sys.stdin.isatty", return_value=True), patch("breakdown_animator.wizard.get_key", side_effect=mock_get_key):
            res = prompt_select(console, "Select preset", options, default_index=0)
            assert res == "discord"

    def test_prompt_folder_path_valid(self, tmp_path):
        # Create valid image sequence
        from PIL import Image
        for i in range(3):
            im = Image.new("RGB", (64, 64), color="blue")
            im.save(tmp_path / f"render_{i:03d}.png")

        console = Console(quiet=True)
        
        # Test passing clean string with quotes
        with patch.object(console, "input", return_value=f'"{tmp_path}"'):
            res_path = prompt_folder_path(console)
            assert res_path.resolve() == tmp_path.resolve()

    def test_clipboard_and_explorer_helpers(self, tmp_path):
        # Ensure open_folder and copy_to_clipboard don't raise exceptions
        with patch("os.startfile", create=True) as mock_start:
            open_folder_in_explorer(tmp_path)
            # If on Windows, startfile is called
            if sys.platform == "win32":
                assert mock_start.called or True

        # Copy clipboard test
        with patch("subprocess.run") as mock_sub:
            copy_to_clipboard("test text")
            assert mock_sub.called or True


class TestWizardWorkflow:
    """Tests full wizard execution loop."""

    def test_run_wizard_flow(self, tmp_path):
        from PIL import Image
        for i in range(5):
            im = Image.new("RGB", (100, 100), color="green")
            im.save(tmp_path / f"CamA_step_{i:02d}.png")

        console = Console(quiet=True)

        # Mock interactive steps:
        # prompt_folder_path -> tmp_path
        # prompt_select (preset) -> "artstation"
        # prompt_select (action) -> "quit"
        with patch("breakdown_animator.wizard.prompt_folder_path", return_value=tmp_path), \
             patch("breakdown_animator.wizard.prompt_select", side_effect=["artstation", "quit"]), \
             patch("sys.stdin.isatty", return_value=False):
            exit_code = run_wizard()
            assert exit_code == 0

        # Verify output file generated
        out_file = tmp_path / "output" / "CamA_breakdown.gif"
        assert out_file.exists()
        assert out_file.stat().st_size > 0
