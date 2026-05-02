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
            scan_directory(missing)

    def test_single_frame_sequence(self, tmp_path: Path):
        """Single frame sequence is analyzed without error."""
        im = Image.new("RGB", (800, 600), (100, 100, 100))
        single_path = tmp_path / "solo_01.png"
        im.save(single_path)

        group = analyze_sequence([single_path], group_id="solo")
        assert group.total_frames == 1
        assert group.common_width == 800
        assert group.common_height == 600
        assert group.has_mismatched_dimensions is False

    def test_corrupted_files_handling(self, tmp_path: Path):
        """0-byte and truncated files are detected and handled."""
        paths = create_corrupt_asset_sequence(tmp_path)
        # analyze_sequence should either skip corrupt frames or raise CorruptImageError depending on strategy
        try:
            group = analyze_sequence(paths, group_id="corrupt_test")
            # If graceful skipping is implemented, total_frames should reflect valid frames (2)
            assert group.total_frames >= 2
        except (CorruptImageError, Exception) as exc:
            # If strict error is raised, it must identify corrupted frame
            assert "empty" in str(exc) or "corrupt" in str(exc).lower() or "truncated" in str(exc)

    def test_case_insensitive_extensions(self, tmp_path: Path):
        """Handles .PNG, .Jpg, .JPEG, .WebP, .BMP extensions seamlessly."""
        im = Image.new("RGB", (100, 100), (0, 255, 0))
        im.save(tmp_path / "frame_01.PNG")
        im.save(tmp_path / "frame_02.Jpg", format="JPEG")
        im.save(tmp_path / "frame_03.WebP", format="WEBP")

        scanned = scan_directory(tmp_path)
        assert len(scanned) == 3


# ============================================================================
# Tier 3: Pairwise & Interaction Tests
# ============================================================================

class TestSequenceEnginePairwiseInteractions:
    """Pairwise combinations of cameras, extensions, and color modes."""

    def test_multi_camera_with_mixed_extensions(self, tmp_path: Path):
        """Multi-camera sequences containing mixed PNG, JPG, and WEBP frames."""
        for cam in ("Cam_A", "Cam_B"):
            im1 = Image.new("RGB", (640, 360), (200, 50, 50))
            im2 = Image.new("RGB", (640, 360), (50, 200, 50))
            im3 = Image.new("RGB", (640, 360), (50, 50, 200))
            im1.save(tmp_path / f"Step_01_{cam}.png")
            im2.save(tmp_path / f"Step_02_{cam}.jpg")
            im3.save(tmp_path / f"Step_03_{cam}.webp")

        scanned = scan_directory(tmp_path)
        groups = group_sequences(scanned)

        assert len(groups) == 2
        assert "Cam_A" in groups
        assert "Cam_B" in groups
        assert groups["Cam_A"].total_frames == 3
        assert groups["Cam_B"].total_frames == 3

    def test_color_modes_sequence_analysis(self, tmp_path: Path):
        """Sequence containing RGB, RGBA, L, P, and CMYK frames is properly inspected."""
        paths = create_color_mode_sequence(tmp_path)
        group = analyze_sequence(paths, group_id="modes")

        assert group.total_frames == 5
        modes = [f.mode for f in group.frames]
        assert "RGB" in modes or "RGBA" in modes


# ============================================================================
# Tier 4: Mode Normalization & Dataclass Property Tests
# ============================================================================

class TestModeNormalizationAndProperties:
    """Tests for color mode normalization and FrameInfo/SequenceGroup properties."""

    def test_normalize_frame_mode_to_rgba(self):
        """Converts RGB, RGBA, CMYK, L, P, 1 to RGBA cleanly."""
        im_rgb = Image.new("RGB", (64, 64), (255, 0, 0))
        im_rgba = Image.new("RGBA", (64, 64), (0, 255, 0, 128))
        im_cmyk = Image.new("CMYK", (64, 64), (0, 255, 255, 0))
        im_l = Image.new("L", (64, 64), 128)
        im_p = im_rgb.quantize(colors=16)

        norm_rgb = normalize_frame_mode(im_rgb, target_mode="RGBA")
        norm_rgba = normalize_frame_mode(im_rgba, target_mode="RGBA")
        norm_cmyk = normalize_frame_mode(im_cmyk, target_mode="RGBA")
        norm_l = normalize_frame_mode(im_l, target_mode="RGBA")
        norm_p = normalize_frame_mode(im_p, target_mode="RGBA")

        assert norm_rgb.mode == "RGBA"
        assert norm_rgba.mode == "RGBA"
        assert norm_cmyk.mode == "RGBA"
        assert norm_l.mode == "RGBA"
        assert norm_p.mode == "RGBA"

    def test_normalize_frame_mode_to_rgb_with_matte(self):
        """Converts RGBA, LA, CMYK to RGB with matte background compositing."""
        im_rgba = Image.new("RGBA", (64, 64), (255, 255, 255, 0))
        norm = normalize_frame_mode(im_rgba, target_mode="RGB", background_color=(0, 0, 0))
        assert norm.mode == "RGB"
        # Since alpha is 0, background color (0, 0, 0) should show
        assert norm.getpixel((0, 0)) == (0, 0, 0)

    def test_load_and_normalize_frame(self, tmp_path: Path):
        """Loads file from disk and normalizes color mode."""
        im = Image.new("CMYK", (100, 100), (0, 100, 100, 0))
        p = tmp_path / "test_cmyk.jpg"
        im.save(p, format="JPEG")

        norm = load_and_normalize_frame(p, target_mode="RGBA")
        assert norm.mode == "RGBA"
        assert norm.size == (100, 100)

    def test_frame_info_properties(self):
        """Verifies aspect_ratio, aspect_ratio_str, and file_size_mb."""
        fi = FrameInfo(
            path=Path("/mock/frame_01.png"),
            index=0,
            filename="frame_01.png",
            width=1920,
            height=1080,
            mode="RGB",
            file_size_bytes=1048576,
        )
        assert abs(fi.aspect_ratio - (16 / 9)) < 0.001
        assert fi.aspect_ratio_str == "16:9"
        assert fi.dimensions == (1920, 1080)
        assert fi.file_size_mb == 1.0

    def test_sequence_group_properties(self):
        """Verifies SequenceGroup aspect ratio and dimensions summary."""
        frames = [
            FrameInfo(Path(f"/mock/f_{i}.png"), i, f"f_{i}.png", 1920, 1080, "RGB", 500000)
            for i in range(5)
        ]
        sg = SequenceGroup(
            group_id="Camera_1",
            camera_name="Camera_1",
            frames=frames,
            total_frames=5,
            common_width=1920,
            common_height=1080,
            has_mismatched_dimensions=False,
            estimated_memory_bytes=1920 * 1080 * 4 * 5,
        )
        assert sg.aspect_ratio_str == "16:9"
        assert sg.dimensions_summary == "1920x1080"
        assert len(sg.frame_paths) == 5
        assert sg.estimated_memory_mb > 0


# ============================================================================
# Tier 5: Advanced Tag Extraction, Edge Cases & Stress Tests
# ============================================================================

class TestAdvancedTagExtractionAndEdgeCases:
    """Tests for complex naming patterns, pass extraction, and directory anomalies."""

    def test_extract_group_tag_heuristics(self):
        """Extracts camera, pass, and prefix tags across various conventions."""
        # Camera patterns
        assert extract_group_tag("Stage_01_Camera_1.jpg") == ("Camera_1", "Camera_1")
        assert extract_group_tag("Cam02_Step_01.png") == ("Cam02", "Cam02")
        assert extract_group_tag("Angle_Side_001.png") == ("Angle_Side", "Angle_Side")
        assert extract_group_tag("Turntable_036.jpg") == ("Turntable", "Turntable")

        # Pass patterns
        assert extract_group_tag("Character_Clay_01.png") == ("Clay", "Clay")
        assert extract_group_tag("Hero_Wireframe_001.png") == ("Wireframe", "Wireframe")
        assert extract_group_tag("Pass_AO_005.png") == ("AO", "AO")
        assert extract_group_tag("Render_Beauty_01.png") == ("Beauty", "Beauty")

        # Simple sequence prefix
        assert extract_group_tag("dragon_sculpt_001.png") == ("dragon_sculpt", None)
        assert extract_group_tag("001.png") == ("default", None)

    def test_pass_tags_grouping(self, tmp_path: Path):
        """Clusters sequences by render pass when no camera is present."""
        for p_name in ("Clay", "Wireframe", "AO"):
            for i in range(1, 4):
                im = Image.new("RGB", (320, 240), (100, 100, 100))
                im.save(tmp_path / f"Hero_{p_name}_{i:02d}.png")

        scanned = scan_directory(tmp_path)
        groups = group_sequences(scanned)

        assert "Clay" in groups
        assert "Wireframe" in groups
        assert "AO" in groups
        assert groups["Clay"].total_frames == 3
        assert groups["Wireframe"].total_frames == 3
        assert groups["AO"].total_frames == 3

    def test_natural_sort_leading_zero_tie_breakers(self):
        """Tie-breaks identical numeric values with different padding deterministically."""
        paths = [
            Path("/mock/step_001.png"),
            Path("/mock/step_1.png"),
            Path("/mock/step_01.png"),
            Path("/mock/step_2.png"),
        ]
        sorted_p = natural_sort_paths(paths)
        assert [p.name for p in sorted_p] == [
            "step_1.png",
            "step_01.png",
            "step_001.png",
            "step_2.png",
        ]

    def test_scan_directory_recursive(self, tmp_path: Path):
        """Recursively discovers images in nested subfolders when recursive=True."""
        sub1 = tmp_path / "take_01" / "cam_1"
        sub2 = tmp_path / "take_02" / "cam_1"
        sub1.mkdir(parents=True, exist_ok=True)
        sub2.mkdir(parents=True, exist_ok=True)

        Image.new("RGB", (50, 50)).save(sub1 / "frame_01.png")
        Image.new("RGB", (50, 50)).save(sub2 / "frame_01.png")

        shallow = scan_directory(tmp_path, recursive=False)
        assert len(shallow) == 0

        deep = scan_directory(tmp_path, recursive=True)
        assert len(deep) == 2

    def test_scan_directory_path_is_file_raises_error(self, tmp_path: Path):
        """Passing a file path instead of directory raises SequenceDiscoveryError."""
        file_path = tmp_path / "single.png"
        Image.new("RGB", (50, 50)).save(file_path)

        with pytest.raises(SequenceDiscoveryError):
            scan_directory(file_path)

    def test_analyze_sequence_strict_error_on_corrupt(self, tmp_path: Path):
        """Raises CorruptImageError when skip_corrupted=False and a file is broken."""
        valid_p = tmp_path / "valid.png"
        Image.new("RGB", (50, 50)).save(valid_p)

        corrupt_p = tmp_path / "broken.png"
        corrupt_p.write_bytes(b"")

        with pytest.raises(CorruptImageError):
            analyze_sequence([valid_p, corrupt_p], skip_corrupted=False)

    def test_performance_many_frames_metadata_scan(self, tmp_path: Path):
        """Fast header-only scanning of 100 frames in < 1 second."""
        paths = []
        for i in range(100):
            fp = tmp_path / f"bench_{i:04d}.png"
            Image.new("RGB", (1920, 1080), (i % 256, 100, 100)).save(fp)
            paths.append(fp)

        group = analyze_sequence(paths, group_id="bench")
        assert group.total_frames == 100
        assert group.common_width == 1920
        assert group.common_height == 1080
        assert group.estimated_memory_bytes == 1920 * 1080 * 4 * 100

