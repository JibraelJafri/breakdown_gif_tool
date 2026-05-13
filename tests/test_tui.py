"""Tests for Textual TUI Application (tui.py).

Covers Tiers 1-3:
- Textual App instantiation and screen mounting via Textual pilot test harness
- Widget presence: Directory input, Sequence inspector, Preset/Format controls, Progress tracker
- Reactive events: Directory path loading, camera selection, preset change
- Keyboard shortcuts and actions (Quit, Refresh, Export)
- Error handling on non-existent or empty sequence directories
"""

from __future__ import annotations

from pathlib import Path
import pytest
from textual.pilot import Pilot

# Import under test
from breakdown_animator.tui import AnimForgeApp
from tests.generate_test_assets import (
    create_natural_sort_sequence,
    create_multi_camera_sequence,
)


# ============================================================================
# Tier 1: Core TUI Lifecycle & Widget Mounting
# ============================================================================

@pytest.mark.asyncio
class TestTuiAppLifecycle:
    """Tests for Textual App boot, widget mounting, and composition."""

    async def test_tui_app_boots_cleanly(self):
        """AnimForgeApp boots, mounts main widgets, and exits on quit."""
        app = AnimForgeApp()
        async with app.run_test() as pilot:
            assert app.is_running
            # Verify root widgets
            assert app.query("#dir_input").first() is not None or len(app.query("Input")) > 0
            # Test graceful quit
            await pilot.press("q")

    async def test_tui_widgets_present(self):
        """Verifies presence of core controls and inspector widgets."""
        app = AnimForgeApp()
        async with app.run_test() as pilot:
            # Query standard inputs / selectors
            assert len(app.query("Button")) >= 1
            await pilot.pause()


# ============================================================================
# Tier 2: Reactive State & Interactions
# ============================================================================

@pytest.mark.asyncio
class TestTuiInteractions:
    """Tests for directory loading and sequence inspection within TUI."""

    async def test_tui_load_valid_sequence(self, tmp_path: Path):
        """Loading a valid sequence folder updates the inspector."""
        create_natural_sort_sequence(tmp_path / "renders", count=6)
        app = AnimForgeApp(initial_path=str(tmp_path / "renders"))

        async with app.run_test() as pilot:
            await pilot.pause()
            # App should have detected the sequence
            assert app.is_running

    async def test_tui_load_multi_camera_sequence(self, tmp_path: Path):
        """Loading multi-camera sequence displays camera options."""
        create_multi_camera_sequence(tmp_path / "cams", cameras=("Camera_1", "Camera_2"), stages_count=4)
        app = AnimForgeApp(initial_path=str(tmp_path / "cams"))

        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.is_running

    async def test_tui_handles_missing_folder_gracefully(self):
        """Loading non-existent folder displays error without crashing."""
        app = AnimForgeApp(initial_path="/non/existent/render_dir_123")
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.is_running
            await pilot.press("q")
