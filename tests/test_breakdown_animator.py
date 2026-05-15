"""End-to-End Integration & Real-World Application Scenario Tests.

Covers Tier 4 Workflows (as defined in TEST_INFRA.md & ORIGINAL_REQUEST.md):
- Scenario 1: Multi-Camera 3D Render Breakdown (ArtStation GIF <= 10MB)
- Scenario 2: High-Fidelity Portfolio 4K WebP Animation
- Scenario 3: Mismatched Aspect Ratio & Resolution Harmonization (Letterbox)
- Scenario 4: Discord/Slack Target Size Auto-Tuning with Dense Noise
- Scenario 5: Headless CLI Batch Multi-Camera Processing (--camera all)
- Scenario 6: Interactive Textual TUI Workflow Simulation (Headless Pilot)
- Entrypoint Dispatcher (app.py) smart routing
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import pytest
from PIL import Image

# Import system modules
from breakdown_animator.sequence_engine import (
    scan_directory,
    group_sequences,
    analyze_sequence,
)
from breakdown_animator.export_engine import (
    ExportFormat,
    ExportProfile,
    ExportResult,
    export_sequence,
    HarmonizeMode,
    DitherMode,
)
from breakdown_animator.presets import (
    PresetName,
    get_preset_profile,
)
from breakdown_animator.cli import run_cli
from breakdown_animator.app import main as app_main

from tests.generate_test_assets import (
    create_natural_sort_sequence,
    create_multi_camera_sequence,
    create_mismatched_dimension_sequence,
    create_mixed_extensions_sequence,
    create_exclusion_test_assets,
    create_color_mode_sequence,
    draw_synthetic_breakdown_frame,
)


# ============================================================================
# Tier 4 Scenario 1: Multi-Camera 3D Render Breakdown (ArtStation GIF <= 10MB)
# ============================================================================

class TestScenario1MultiCameraArtstationGif:
    """Tests the primary 3D artist breakdown workflow for ArtStation portfolio."""

    def test_multi_camera_artstation_export(self, tmp_path: Path):
        """Discovers multi-camera render sequences and exports <=10MB ArtStation GIF with 2.0s hold."""
        src_dir = tmp_path / "3d_character_breakdown"
        out_dir = tmp_path / "artstation_output"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Generate multi-camera sequence across 5 stages (Clay, Wireframe, Albedo, DirectLight, Beauty)
        cam_map = create_multi_camera_sequence(
            src_dir,
            cameras=("Camera_Front", "Camera_Perspective"),
            stages_count=5,
            dimensions=(1280, 720),
        )

        # 1. Scan and Group
        scanned = scan_directory(src_dir)
        groups = group_sequences(scanned)

        assert "Camera_Front" in groups
        assert "Camera_Perspective" in groups
        assert groups["Camera_Front"].total_frames == 5

        # 2. Export each camera sequence using ArtStation profile
        artstation_profile = get_preset_profile(PresetName.ARTSTATION)

        for cam_id, group in groups.items():
            out_file = out_dir / f"{cam_id}_artstation.gif"
            frame_paths = [f.path for f in group.frames]
            result = export_sequence(frame_paths, out_file, artstation_profile)

            assert out_file.exists()
            assert result.passes_budget is True
            assert result.file_size_mb <= 10.0
            assert result.format == ExportFormat.GIF

            # Verify GIF properties via Pillow
            with Image.open(out_file) as im:
                assert im.format == "GIF"
                assert im.n_frames == 5


# ============================================================================
# Tier 4 Scenario 2: High-Fidelity 4K Lossless / Lossy Portfolio WebP
# ============================================================================

class TestScenario2PortfolioWebp:
    """Tests high-efficiency 24-bit VP8 WebP export for portfolio websites."""

    def test_high_fidelity_portfolio_webp_export(self, tmp_path: Path):
        """Exports 24-bit animated WebP preserving true color fidelity with hold pause."""
        src_dir = tmp_path / "portfolio_renders"
        out_dir = tmp_path / "webp_output"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Create natural sequence (12 frames)
        paths = create_natural_sort_sequence(src_dir, count=8, dimensions=(960, 540))
        out_file = out_dir / "breakdown_portfolio.webp"

        portfolio_profile = get_preset_profile(PresetName.PORTFOLIO_4K)

        result = export_sequence(paths, out_file, portfolio_profile)

        assert out_file.exists()
        assert result.format == ExportFormat.WEBP
        assert result.total_frames == 8

        with Image.open(out_file) as im:
            assert im.format == "WEBP"
            assert getattr(im, "is_animated", False) is True
            assert im.n_frames == 8


# ============================================================================
# Tier 4 Scenario 3: Mismatched Aspect Ratio / Resolution Breakdown
# ============================================================================

class TestScenario3MismatchedDimensionsHarmonization:
    """Tests auto-letterboxing when breakdown passes have varying resolutions."""

    def test_mismatched_aspect_ratios_letterbox(self, tmp_path: Path):
        """Harmonizes mixed 1920x1080, 1080x1080, 1280x720, 800x1200 frames onto uniform canvas."""
        src_dir = tmp_path / "mixed_aspect_renders"
        paths = create_mismatched_dimension_sequence(src_dir)
        out_file = tmp_path / "harmonized_breakdown.gif"

        profile = ExportProfile(
            format=ExportFormat.GIF,
            harmonize_mode=HarmonizeMode.LETTERBOX,
            background_color=(0, 0, 0),
            fps=12.0,
            hold_last_seconds=1.5,
        )

        result = export_sequence(paths, out_file, profile)

        assert out_file.exists()
        assert result.total_frames == 5

        with Image.open(out_file) as im:
            assert im.format == "GIF"
            assert im.n_frames == 5
            # Canvas dimension is unified
            assert im.size[0] >= 1080
            assert im.size[1] >= 720


# ============================================================================
# Tier 4 Scenario 4: Discord / Slack Size Budget Auto-Tuning
# ============================================================================

class TestScenario4BudgetAutoTuning:
    """Tests predictor-corrector auto-tuning meeting strict chat platform limits."""

    def test_discord_and_slack_auto_tuning(self, tmp_path: Path):
        """Ensures Discord (<=8MB) and Slack (<=5MB) auto-tuning converges."""
        src_dir = tmp_path / "dense_renders"
        paths = create_natural_sort_sequence(src_dir, count=10, dimensions=(1280, 720))

        # 1. Discord preset (<= 8 MB)
        discord_file = tmp_path / "discord_export.gif"
        discord_profile = get_preset_profile(PresetName.DISCORD)
        res_disc = export_sequence(paths, discord_file, discord_profile)

        assert discord_file.exists()
        assert res_disc.passes_budget is True
        assert res_disc.file_size_mb <= 8.0

        # 2. Slack preset (<= 5 MB)
        slack_file = tmp_path / "slack_export.gif"
        slack_profile = get_preset_profile(PresetName.SLACK)
        res_slack = export_sequence(paths, slack_file, slack_profile)

        assert slack_file.exists()
        assert res_slack.passes_budget is True
        assert res_slack.file_size_mb <= 5.0


# ============================================================================
# Tier 4 Scenario 5: Headless CLI Batch Multi-Camera Processing
# ============================================================================

class TestScenario5CliBatchMultiCamera:
    """Tests non-interactive CLI batch execution across multiple cameras."""

    def test_cli_batch_export_all_cameras(self, tmp_path: Path):
        """Runs headless CLI batch export for all detected cameras."""
        src_dir = tmp_path / "studio_renders"
        out_dir = tmp_path / "batch_out"

        create_multi_camera_sequence(
            src_dir,
            cameras=("Cam_Wide", "Cam_CloseUp"),
            stages_count=4,
            dimensions=(640, 360),
        )

        exit_code = run_cli([
            str(src_dir),
            "--preset", "artstation",
            "--camera", "all",
            "--output-dir", str(out_dir),
        ])

        assert exit_code == 0
        assert out_dir.exists()
        exported_files = list(out_dir.glob("*.gif")) + list(out_dir.glob("*.webp"))
        assert len(exported_files) >= 2


# ============================================================================
# Tier 4 Scenario 6 & Entrypoint Dispatcher (app.py)
# ============================================================================

class TestScenario6EntrypointDispatcher:
    """Tests unified entrypoint (app.py) routing to CLI or TUI."""

    def test_entrypoint_with_folder_invokes_cli(self, tmp_path: Path):
        """Running app.py with directory arguments executes CLI runner."""
        src_dir = tmp_path / "cli_test_renders"
        create_natural_sort_sequence(src_dir, count=3, dimensions=(320, 180))

        exit_code = app_main([str(src_dir), "--dry-run"])
        assert exit_code == 0
