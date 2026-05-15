"""tests/test_bulk_and_timing.py
==============================
Tests for multi-folder bulk processing, path cleaning, and breakdown frame timing presets.
"""

from __future__ import annotations

import os
from pathlib import Path
import pytest
from PIL import Image

from breakdown_animator.cli import build_parser, run_cli
from breakdown_animator.export_engine import (
    ExportFormat,
    ExportProfile,
    calculate_frame_durations,
    export_sequence,
)
from breakdown_animator.presets import (
    PresetName,
    get_preset_profile,
)
from breakdown_animator.sequence_engine import (
    clean_folder_path,
    discover_sequence_folders,
    split_multiple_paths,
)
from tests.generate_test_assets import create_natural_sort_sequence


class TestPathCleaningAndDiscovery:
    """Tests for path sanitization and multi-folder sequence discovery."""

    def test_clean_folder_path_powershell_and_quotes(self):
        """Strips PowerShell & prefix, quotes, and file:/// prefixes."""
        assert "my_folder" in clean_folder_path("& 'C:\\renders\\my_folder'")
        assert "my_folder" in clean_folder_path('"C:\\renders\\my_folder"')
        assert "my_folder" in clean_folder_path("file:///C:/renders/my_folder")
        assert "my_folder" in clean_folder_path("  'C:/renders/my_folder/'  ")

    def test_split_multiple_paths(self):
        """Splits multiple quoted paths."""
        raw = '"C:\\renders\\shot1" "C:\\renders\\shot2"'
        paths = split_multiple_paths(raw)
        assert len(paths) == 2
        assert any("shot1" in p for p in paths)
        assert any("shot2" in p for p in paths)

    def test_discover_sequence_folders(self, tmp_path: Path):
        """Discovers subfolders containing valid images, ignores empty or output dirs."""
        root = tmp_path / "project_renders"
        root.mkdir()

        # Shot 1 with images
        shot1 = root / "shot1"
        create_natural_sort_sequence(shot1, count=3)

        # Shot 2 with images
        shot2 = root / "shot2"
        create_natural_sort_sequence(shot2, count=4)

        # Empty folder
        empty_fld = root / "notes"
        empty_fld.mkdir()

        # Output folder with images (should be ignored)
        out_fld = root / "output"
        create_natural_sort_sequence(out_fld, count=2)

        discovered = discover_sequence_folders(root, max_depth=2, include_root_if_has_images=False)
        discovered_names = [p.name for p in discovered]

        assert "shot1" in discovered_names
        assert "shot2" in discovered_names
        assert "notes" not in discovered_names
        assert "output" not in discovered_names


class TestBreakdownTimingAndPresets:
    """Tests for breakdown timing math and sensible presentation speeds."""

    def test_calculate_frame_durations_step_duration(self):
        """frame_duration=1.0 sets base frame delay to 1000ms, not 83ms."""
        durations = calculate_frame_durations(
            total_frames=4,
            hold_last_seconds=2.5,
            frame_duration=1.0,
        )
        assert len(durations) == 4
        assert durations[0] == 1000
        assert durations[1] == 1000
        assert durations[2] == 1000
        # Final frame has hold delay: 1000 + 2500 = 3500ms
        assert durations[3] == 3500

    def test_calculate_frame_durations_slow_breakdown(self):
        """frame_duration=1.8 sets 1800ms per step."""
        durations = calculate_frame_durations(
            total_frames=3,
            hold_last_seconds=3.0,
            frame_duration=1.8,
        )
        assert durations[0] == 1800
        assert durations[1] == 1800
        assert durations[2] == 4800  # 1800 + 3000

    def test_calculate_frame_durations_minimum_clamp(self):
        """Frame duration is clamped to safe minimum and 120 FPS maps to ~8ms."""
        durations = calculate_frame_durations(fps=120.0, total_frames=2, hold_last_seconds=0.0)
        assert durations == [8, 8]
        assert all(d >= 1 for d in durations)

    def test_presets_sensible_breakdown_pacing(self):
        """ArtStation, slow, and fast breakdown presets have human-readable step pacing."""
        p_art = get_preset_profile(PresetName.ARTSTATION)
        assert p_art.frame_duration == 1.0
        assert p_art.hold_last_seconds == 2.0

        p_slow = get_preset_profile(PresetName.BREAKDOWN_SLOW)
        assert p_slow.frame_duration == 1.8
        assert p_slow.hold_last_seconds == 3.0

        p_fast = get_preset_profile(PresetName.BREAKDOWN_FAST)
        assert p_fast.frame_duration == 0.5
        assert p_fast.hold_last_seconds == 1.5

        p_turn = get_preset_profile(PresetName.TURNTABLE)
        assert p_turn.fps == 12.0
        assert p_turn.frame_duration is None

    def test_export_sequence_with_step_duration(self, tmp_path: Path):
        """Exporting with step duration creates GIF with intended frame delays."""
        seq_dir = tmp_path / "test_steps"
        paths = create_natural_sort_sequence(seq_dir, count=3, dimensions=(200, 200))
        out_gif = tmp_path / "step_timing.gif"

        profile = ExportProfile(
            format=ExportFormat.GIF,
            frame_duration=1.2,
            hold_last_seconds=2.0,
        )

        res = export_sequence(
            frames=paths,
            output_path=out_gif,
            profile=profile,
        )

        assert res.output_path.exists()
        # Verify GIF duration metadata
        with Image.open(out_gif) as im:
            assert getattr(im, "is_animated", False)
            assert im.n_frames == 3
            # Frame 0 duration ~ 1200ms
            im.seek(0)
            assert abs(im.info.get("duration", 0) - 1200) <= 20
            # Frame 2 duration ~ 3200ms (1200 + 2000)
            im.seek(2)
            assert abs(im.info.get("duration", 0) - 3200) <= 20


class TestCliMultiFolderAndBulk:
    """Tests for CLI multi-folder positional arguments and --bulk flag."""

    def test_cli_parser_extra_folders_and_duration_flag(self):
        """Parser accepts multiple folders and --step-duration."""
        parser = build_parser()
        args = parser.parse_args([
            "folderA", "folderB", "folderC",
            "-d", "1.5",
            "--bulk",
        ])
        assert args.folder == "folderA"
        assert args.extra_folders == ["folderB", "folderC"]
        assert args.frame_duration == 1.5
        assert args.bulk is True

    def test_cli_bulk_flag_processes_subfolders(self, tmp_path: Path):
        """--bulk automatically discovers and converts all subfolder sequences."""
        root = tmp_path / "multi_shots"
        root.mkdir()
        shot1 = root / "shot1"
        shot2 = root / "shot2"
        create_natural_sort_sequence(shot1, count=3, dimensions=(160, 90))
        create_natural_sort_sequence(shot2, count=3, dimensions=(160, 90))

        unified_out = tmp_path / "all_outputs"

        exit_code = run_cli([
            str(root),
            "--bulk",
            "--preset", "artstation",
            "--output-dir", str(unified_out),
        ])

        assert exit_code == 0
        assert unified_out.exists()
        generated = list(unified_out.glob("*.gif"))
        assert len(generated) == 2
        names = [g.name for g in generated]
        assert any("shot1" in n for n in names)
        assert any("shot2" in n for n in names)

    def test_cli_multiple_positional_folders(self, tmp_path: Path):
        """Passing multiple folder paths directly processes each one."""
        fld1 = tmp_path / "folder_alpha"
        fld2 = tmp_path / "folder_beta"
        create_natural_sort_sequence(fld1, count=3, dimensions=(160, 90))
        create_natural_sort_sequence(fld2, count=3, dimensions=(160, 90))

        unified_out = tmp_path / "positional_outputs"

        exit_code = run_cli([
            str(fld1), str(fld2),
            "-d", "1.0",
            "--output-dir", str(unified_out),
        ])

        assert exit_code == 0
        assert unified_out.exists()
        generated = list(unified_out.glob("*.gif"))
        assert len(generated) == 2
