"""Tests for Headless CLI Interface (cli.py, app.py).

Covers Tiers 1-3:
- CLI argument parsing & validation
- Flag overrides (--preset, --camera, --fps, --hold-last, --max-size-mb, --scale, --colors, --dither)
- Dry-run inspection mode (--dry-run)
- Batch multi-camera processing (--camera all)
- Custom output directory creation (--output-dir)
- Exit codes: 0 on clean execution, non-zero (1) on missing folders or validation errors
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import pytest

# Import under test
from breakdown_animator.cli import build_parser, run_cli
from tests.generate_test_assets import (
    create_natural_sort_sequence,
    create_multi_camera_sequence,
)


# ============================================================================
# Tier 1: Core CLI Argument Parsing & Flag Tests
# ============================================================================

class TestCliArgumentParser:
    """Tests for ArgumentParser creation and flag mapping."""

    def test_parser_defaults(self):
        """Default arguments reflect standard configuration."""
        parser = build_parser()
        args = parser.parse_args(["my_renders"])

        assert args.folder == "my_renders"
        assert args.preset == "artstation"
        assert args.camera == "all"
        assert args.dry_run is False

    def test_parser_help_formatting(self):
        """format_help expands successfully without percent-formatting errors."""
        parser = build_parser()
        help_text = parser.format_help()
        assert "animforge" in help_text
        assert "--bulk" in help_text
        assert "--step-duration" in help_text

    def test_parser_custom_flags(self):
        """Explicit CLI flag overrides are properly parsed."""
        parser = build_parser()
        args = parser.parse_args([
            "path/to/folder",
            "--preset", "discord",
            "--format", "gif",
            "--camera", "Camera_1",
            "--fps", "15.0",
            "--hold-last", "2.5",
            "--max-size-mb", "8.0",
            "--scale", "720p",
            "--colors", "128",
            "--no-dither",
            "--output-dir", "custom/out",
        ])

        assert args.folder == "path/to/folder"
        assert args.preset == "discord"
        assert args.format == "gif"
        assert args.camera == "Camera_1"
        assert args.fps == 15.0
        assert args.hold_last == 2.5
        assert args.max_size_mb == 8.0
        assert args.scale == "720p"
        assert args.colors == 128
        assert args.dither is False
        assert args.output_dir == "custom/out"


class TestCliExecution:
    """Tests for run_cli execution with synthetic fixtures."""

    def test_cli_dry_run_inspection(self, tmp_path: Path, capsys):
        """--dry-run displays sequence details and exits 0 without creating files."""
        create_natural_sort_sequence(tmp_path / "renders", count=5)
        out_dir = tmp_path / "renders" / "output"

        exit_code = run_cli([str(tmp_path / "renders"), "--dry-run"])

        assert exit_code == 0
        # Dry run must not create output animation files
        if out_dir.exists():
            files = list(out_dir.glob("*.gif")) + list(out_dir.glob("*.webp"))
            assert len(files) == 0

    def test_cli_basic_export_workflow(self, tmp_path: Path):
        """Runs standard headless conversion and verifies generated output."""
        src_dir = tmp_path / "renders"
        out_dir = tmp_path / "output_dest"
        create_natural_sort_sequence(src_dir, count=4, dimensions=(320, 180))

        exit_code = run_cli([
            str(src_dir),
            "--preset", "artstation",
            "--output-dir", str(out_dir),
        ])

        assert exit_code == 0
        assert out_dir.exists()
        output_files = list(out_dir.glob("*.gif")) + list(out_dir.glob("*.webp"))
        assert len(output_files) >= 1

    def test_cli_multi_camera_batch_export(self, tmp_path: Path):
        """--camera all converts all detected camera passes."""
        src_dir = tmp_path / "multi_cam_renders"
        out_dir = tmp_path / "multi_cam_out"
        create_multi_camera_sequence(src_dir, cameras=("Camera_1", "Camera_2"), stages_count=3)

        exit_code = run_cli([
            str(src_dir),
            "--camera", "all",
            "--output-dir", str(out_dir),
        ])

        assert exit_code == 0
        assert out_dir.exists()
        output_files = list(out_dir.glob("*.*"))
        assert len(output_files) >= 2


# ============================================================================
# Tier 2: Boundary & Error Exit Codes
# ============================================================================

class TestCliErrorHandling:
    """Tests for non-zero exit codes on invalid paths and missing sequences."""

    def test_cli_non_existent_folder_exits_code_1(self):
        """Non-existent directory returns exit code 1."""
        exit_code = run_cli(["/non/existent/render_dir/xyz987"])
        assert exit_code == 1

    def test_cli_empty_folder_exits_code_1(self, tmp_path: Path):
        """Empty folder with 0 images returns exit code 1."""
        empty_dir = tmp_path / "empty_dir"
        empty_dir.mkdir()

        exit_code = run_cli([str(empty_dir)])
        assert exit_code == 1

    def test_cli_single_camera_filter_not_found(self, tmp_path: Path):
        """Requesting non-existent camera tag returns error exit code."""
        src_dir = tmp_path / "renders"
        create_natural_sort_sequence(src_dir, count=3)

        exit_code = run_cli([str(src_dir), "--camera", "NonExistentCam_99"])
        assert exit_code == 1
