"""Tests for Sequence Engine (sequence_engine.py).

Covers Tiers 1-3:
- Image file discovery across supported extensions (.png, .jpg, .webp, .bmp, .tga, .tif)
- Output animation and OS system file exclusion (.gif, animated .webp, .DS_Store, Thumbs.db)
- Natural alphanumeric sorting (frame_1 .. frame_12, multi-number segments)
- Camera & pass auto-tag grouping and isolation
- Sequence metadata extraction (frame counts, dimensions, aspect ratio, memory estimation)
- Mode normalization and color mode detection
- Dimension mismatch detection
- Empty folder, missing directory, single frame, and corrupted image edge cases
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import pytest
from PIL import Image

from breakdown_animator.sequence_engine import (
    FrameInfo,
    SequenceGroup,
    scan_directory,
    natural_sort_key,
    natural_sort_paths,
    group_sequences,
    analyze_sequence,
    inspect_frame,
    is_valid_image_file,
    is_animated_image,
    normalize_frame_mode,
    load_and_normalize_frame,
    extract_group_tag,
    SequenceEngineError,
    SequenceDiscoveryError,
    EmptySequenceError,
    DirectoryNotFoundError,
    CorruptImageError,
    DimensionMismatchError,
)
from tests.generate_test_assets import (
    create_natural_sort_sequence,
    create_multi_camera_sequence,
    create_mismatched_dimension_sequence,
    create_mixed_extensions_sequence,
    create_corrupt_asset_sequence,
    create_exclusion_test_assets,
    create_color_mode_sequence,
    draw_synthetic_breakdown_frame,
)


# ============================================================================
# Tier 1: Core Functional Tests
# ============================================================================

class TestSequenceDiscoveryAndFiltering:
    """Tests for scan_directory, file inclusion, and exclusion rules."""

    def test_scan_discovers_supported_image_extensions(self, tmp_path: Path):
        """Discovers .png, .jpg, .jpeg, .webp, .bmp, .tga, .tif, .tiff files."""
        create_mixed_extensions_sequence(tmp_path)
        # Add a tga and tif file
        im = Image.new("RGB", (100, 100), (255, 0, 0))
        im.save(tmp_path / "extra_01.tga")
        im.save(tmp_path / "extra_02.tif")

        scanned = scan_directory(tmp_path)
        filenames = [p.name for p in scanned]

        assert len(scanned) == 7
        assert "pass_01.png" in filenames
        assert "pass_02.jpg" in filenames
        assert "pass_03.webp" in filenames
        assert "pass_04.bmp" in filenames
        assert "pass_05.jpeg" in filenames
        assert "extra_01.tga" in filenames
        assert "extra_02.tif" in filenames

    def test_scan_ignores_gifs_and_system_files(self, tmp_path: Path):
        """Excludes .gif, .DS_Store, Thumbs.db, desktop.ini, and hidden dotfiles."""
        create_exclusion_test_assets(tmp_path)

        scanned = scan_directory(tmp_path)
        filenames = [p.name for p in scanned]

        # Valid files only
        assert len(scanned) == 3
        for name in filenames:
            assert name.startswith("valid_render_")
            assert not name.endswith(".gif")
            assert not name.startswith(".")
            assert name not in ("Thumbs.db", "desktop.ini")

    def test_scan_ignores_animated_webp_files(self, tmp_path: Path):
        """Distinguishes static single-frame WebPs from animated WebP outputs."""
        create_exclusion_test_assets(tmp_path)
        # Add a static webp
        static_webp = tmp_path / "static_frame_01.webp"
        im = Image.new("RGB", (100, 100), (50, 100, 150))
        im.save(static_webp, format="WEBP")

        scanned = scan_directory(tmp_path)
        filenames = [p.name for p in scanned]

        assert "static_frame_01.webp" in filenames
        assert "existing_breakdown.webp" not in filenames


class TestNaturalAlphanumericSorting:
    """Tests for natural numeric sorting behavior."""

    def test_natural_sort_unpadded_numbers(self, tmp_path: Path):
        """frame_1 .. frame_12 sorts frame_9 < frame_10 < frame_11 < frame_12."""
        paths = create_natural_sort_sequence(tmp_path, count=12)
        # Shuffle paths arbitrarily
        shuffled = list(reversed(paths))
        sorted_paths = natural_sort_paths(shuffled)

        expected_order = [f"frame_{i}.png" for i in range(1, 13)]
        actual_order = [p.name for p in sorted_paths]
        assert actual_order == expected_order

    def test_natural_sort_multi_segment_numbers(self):
        """Sorts multiple numerical segments hierarchically."""
        raw_names = [
            "Stage_02_Camera_1_002.png",
            "Stage_01_Camera_2_001.png",
            "Stage_01_Camera_1_002.png",
            "Stage_01_Camera_1_001.png",
            "Stage_02_Camera_1_001.png",
        ]
        paths = [Path(f"/mock/{name}") for name in raw_names]
        sorted_paths = natural_sort_paths(paths)

        expected = [
            "Stage_01_Camera_1_001.png",
            "Stage_01_Camera_1_002.png",
            "Stage_01_Camera_2_001.png",
            "Stage_02_Camera_1_001.png",
            "Stage_02_Camera_1_002.png",
        ]
        assert [p.name for p in sorted_paths] == expected

    def test_natural_sort_handles_strings_without_numbers(self):
        """Gracefully falls back to alphabetical sorting for purely alphabetic names."""
        names = ["render_zebra.png", "render_alpha.png", "render_beta.png"]
        paths = [Path(f"/mock/{n}") for n in names]
        sorted_paths = natural_sort_paths(paths)
        assert [p.name for p in sorted_paths] == [
            "render_alpha.png",
            "render_beta.png",
            "render_zebra.png",
        ]


class TestCameraGroupingAndClustering:
    """Tests for grouping sequences by camera/pass tags."""

    def test_group_multi_camera_sequences(self, tmp_path: Path):
        """Groups Camera_1, Camera_2, Camera_3 into distinct SequenceGroups."""
        cam_dict = create_multi_camera_sequence(
            tmp_path,
            cameras=("Camera_1", "Camera_2", "Camera_3"),
            stages_count=5,
        )
        scanned = scan_directory(tmp_path)
        groups = group_sequences(scanned)

        assert "Camera_1" in groups
        assert "Camera_2" in groups
        assert "Camera_3" in groups
        assert groups["Camera_1"].total_frames == 5
        assert groups["Camera_2"].total_frames == 5
        assert groups["Camera_3"].total_frames == 5

    def test_group_default_sequence_when_no_camera_tags(self, tmp_path: Path):
        """Assigns un-tagged sequences to 'default' group."""
        paths = create_natural_sort_sequence(tmp_path, count=6)
        scanned = scan_directory(tmp_path)
        groups = group_sequences(scanned)

        assert "default" in groups
        assert groups["default"].total_frames == 6
        assert groups["default"].camera_name is None or groups["default"].camera_name == "default"


class TestSequenceMetadataExtraction:
    """Tests for analyzing frame dimensions, memory calculation, and aspect ratios."""

    def test_analyze_uniform_sequence_metadata(self, tmp_path: Path):
        """Calculates correct width, height, frame count, aspect ratio, and memory."""
        paths = create_natural_sort_sequence(tmp_path, count=10, dimensions=(1920, 1080))
        group = analyze_sequence(paths, group_id="default")

        assert group.total_frames == 10
        assert group.common_width == 1920
        assert group.common_height == 1080
        assert group.has_mismatched_dimensions is False
        # Memory: 1920 * 1080 * 4 * 10 bytes = 82,944,000 bytes (~79.1 MB)
        assert group.estimated_memory_bytes == 1920 * 1080 * 4 * 10
        assert len(group.frames) == 10
        assert group.frames[0].width == 1920
        assert group.frames[0].height == 1080
        assert group.frames[0].mode == "RGB"

    def test_analyze_mismatched_dimensions(self, tmp_path: Path):
        """Flags has_mismatched_dimensions=True when frame resolutions vary."""
        paths = create_mismatched_dimension_sequence(tmp_path)
        group = analyze_sequence(paths, group_id="mismatched")

        assert group.has_mismatched_dimensions is True
        assert group.total_frames == 5
        # Common width/height is set to max canvas or reference first frame
        assert group.common_width >= 1920
        assert group.common_height >= 1080


# ============================================================================
# Tier 2: Boundary & Error Handling Tests
# ============================================================================

class TestSequenceEngineEdgeCases:
    """Boundary conditions, missing files, corrupted files, and special modes."""

    def test_empty_folder_raises_empty_sequence_error(self, tmp_path: Path):
        """Scanning an empty directory returns empty list, analyzing raises EmptySequenceError."""
        scanned = scan_directory(tmp_path)
        assert scanned == []

        with pytest.raises((EmptySequenceError, ValueError)):
            analyze_sequence([], group_id="empty")

    def test_missing_directory_raises_error(self):
        """Scanning non-existent directory raises DirectoryNotFoundError."""
        missing = Path("/non/existent/path/xyz_123")
        with pytest.raises((DirectoryNotFoundError, FileNotFoundError)):
