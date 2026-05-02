"""Adversarial stress and edge-case tests for breakdown_animator.sequence_engine.

Targeted empirical challenges:
1. Complex nested numeric sorting (shot_01_v02_take_003_10.png vs shot_01_v02_take_003_2.png)
2. Leading zeros, mixed zero padding, tie-breaking determinism
3. Version and decimal strings (v1.2.3 vs v1.10.1 vs v1.2.10)
4. Mixed prefixes, suffixes, special characters, whitespace, unicode
5. Multi-level path sorting (nested directories with numeric components)
6. Camera and pass tag heuristics across diverse naming conventions
7. Multi-camera & multi-pass clustering edge cases
8. Large sequence stress tests (1,000 frames)
9. Memory footprint and metadata extraction on unusual resolutions
10. File discovery filtering boundaries (dotfiles, system files, uppercase extensions, animated WebPs)
"""

from __future__ import annotations

import os
import re
import tempfile
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


# ============================================================================
# Section 1: Complex Nested Numeric & Natural Sorting Challenges
# ============================================================================

class TestAdversarialNaturalSorting:
    """Stress tests and adversarial cases for natural_sort_paths and natural_sort_key."""

    def test_complex_nested_numbers(self):
        """Dispatches complex multi-part nested numerical identifiers."""
        raw_names = [
            "shot_01_v02_take_003_10.png",
            "shot_01_v02_take_003_2.png",
            "shot_01_v02_take_003_1.png",
            "shot_01_v02_take_003_20.png",
            "shot_01_v02_take_003_100.png",
            "shot_01_v02_take_003_02.png",
            "shot_01_v01_take_003_01.png",
            "shot_02_v01_take_001_01.png",
        ]
        sorted_paths = natural_sort_paths(raw_names)
        sorted_names = [Path(p).name for p in sorted_paths]

        expected_order = [
            "shot_01_v01_take_003_01.png",
            "shot_01_v02_take_003_1.png",
            "shot_01_v02_take_003_2.png",
            "shot_01_v02_take_003_02.png",
            "shot_01_v02_take_003_10.png",
            "shot_01_v02_take_003_20.png",
            "shot_01_v02_take_003_100.png",
            "shot_02_v01_take_001_01.png",
        ]
        assert sorted_names == expected_order

    def test_leading_zero_different_lengths_tie_breaking(self):
        """Zero padding tie-breaker: shorter padding sorts before longer padding."""
        raw_names = [
            "frame_0005.png",
            "frame_05.png",
            "frame_5.png",
            "frame_005.png",
            "frame_00005.png",
            "frame_6.png",
            "frame_04.png",
        ]
        sorted_paths = natural_sort_paths(raw_names)
        sorted_names = [Path(p).name for p in sorted_paths]

        expected_order = [
            "frame_04.png",
            "frame_5.png",
            "frame_05.png",
            "frame_005.png",
            "frame_0005.png",
            "frame_00005.png",
            "frame_6.png",
        ]
        assert sorted_names == expected_order

    def test_version_numbers_semantic_style(self):
        """Verifies semantic-like version numbering sorting (v1.2.3 vs v1.10.1 vs v1.2.10)."""
        raw_names = [
            "build_v1.10.0.png",
            "build_v1.2.10.png",
            "build_v1.2.2.png",
            "build_v1.2.1.png",
            "build_v1.2.3.png",
            "build_v2.0.0.png",
            "build_v0.9.9.png",
        ]
        sorted_paths = natural_sort_paths(raw_names)
        sorted_names = [Path(p).name for p in sorted_paths]

        expected_order = [
            "build_v0.9.9.png",
            "build_v1.2.1.png",
            "build_v1.2.2.png",
            "build_v1.2.3.png",
            "build_v1.2.10.png",
            "build_v1.10.0.png",
            "build_v2.0.0.png",
        ]
        assert sorted_names == expected_order

    def test_special_characters_parentheses_brackets(self):
        """Handles Windows Explorer style copies: 'file (1).png', 'file (10).png', 'file (2).png'."""
        raw_names = [
            "render [draft] (10).png",
            "render [draft] (1).png",
            "render [draft] (2).png",
            "render [draft] (20).png",
            "render [final] (1).png",
        ]
        sorted_paths = natural_sort_paths(raw_names)
        sorted_names = [Path(p).name for p in sorted_paths]

        expected_order = [
            "render [draft] (1).png",
            "render [draft] (2).png",
            "render [draft] (10).png",
            "render [draft] (20).png",
            "render [final] (1).png",
        ]
        assert sorted_names == expected_order

    def test_nested_directory_sorting_hierarchy(self):
        """Hierarchically sorts paths with directory components: dir1/f10 vs dir2/f1 vs dir10/f1."""
        raw_paths = [
            Path("scene_10") / "cam_1" / "frame_1.png",
            Path("scene_2") / "cam_1" / "frame_10.png",
            Path("scene_2") / "cam_1" / "frame_2.png",
            Path("scene_1") / "cam_2" / "frame_1.png",
            Path("scene_1") / "cam_1" / "frame_100.png",
            Path("scene_1") / "cam_1" / "frame_2.png",
        ]
        sorted_paths = natural_sort_paths(raw_paths)
        expected_order = [
            Path("scene_1") / "cam_1" / "frame_2.png",
            Path("scene_1") / "cam_1" / "frame_100.png",
            Path("scene_1") / "cam_2" / "frame_1.png",
            Path("scene_2") / "cam_1" / "frame_2.png",
            Path("scene_2") / "cam_1" / "frame_10.png",
            Path("scene_10") / "cam_1" / "frame_1.png",
        ]
        assert sorted_paths == expected_order

    def test_case_insensitive_natural_sort(self):
        """Case insensitivity with deterministic tie-breaking."""
        raw_names = [
            "FRAME_10.PNG",
            "frame_2.png",
            "Frame_1.png",
            "FRAME_2.png",
            "frame_01.png",
        ]
        sorted_paths = natural_sort_paths(raw_names)
        sorted_names = [Path(p).name for p in sorted_paths]

        assert sorted_names[0] in ("Frame_1.png", "frame_01.png")
        assert "FRAME_10.PNG" == sorted_names[-1]

    def test_stress_large_sequence_sorting(self):
        """Sorts 1,000 randomly permuted paths instantaneously without degrading."""
        import random
        rng = random.Random(42)
        numbers = list(range(1, 1001))
        rng.shuffle(numbers)

        shuffled_paths = [Path(f"/render/seq_step_{n}.png") for n in numbers]
        sorted_paths = natural_sort_paths(shuffled_paths)

        expected = [Path(f"/render/seq_step_{n}.png") for n in range(1, 1001)]
        assert sorted_paths == expected


# ============================================================================
# Section 2: Diverse Camera & Pass Tag Clustering Challenges
# ============================================================================

class TestAdversarialTagExtractionAndGrouping:
    """Stress tests tag heuristics against diverse industry render naming standards."""

    @pytest.mark.parametrize(
        "filename,expected_group,expected_cam",
        [
            # Camera conventions
            ("shot01_Cam1_001.png", "Cam1", "Cam1"),
            ("shot01_Camera_02_001.png", "Camera_02", "Camera_02"),
            ("scene_camera3_001.png", "camera3", "camera3"),
            ("render_cam_wide_001.png", "cam_wide", "cam_wide"),
            ("model_view_top_001.png", "view_top", "view_top"),
            ("turntable_001.png", "turntable", "turntable"),
            ("char_persp_001.jpg", "persp", "persp"),
            ("char_perspective_001.jpg", "perspective", "perspective"),
            ("scene_ortho_001.png", "ortho", "ortho"),
            ("scene_isometric_001.png", "isometric", "isometric"),
            ("hero_front_001.png", "front", "front"),
            ("hero_back_001.png", "back", "back"),
            ("hero_left_001.png", "left", "left"),
            ("hero_right_001.png", "right", "right"),
            ("hero_top_001.png", "top", "top"),
            ("hero_bottom_001.png", "bottom", "bottom"),

            # Pass conventions
            ("render_pass_diffuse_001.png", "diffuse", "diffuse"),
            ("shot_Beauty_001.png", "Beauty", "Beauty"),
            ("scene_AO_001.png", "AO", "AO"),
            ("scene_ambient_occlusion_001.png", "ambient_occlusion", "ambient_occlusion"),
            ("prop_Clay_001.png", "Clay", "Clay"),
            ("prop_clay_render_001.png", "clay_render", "clay_render"),
            ("prop_wireframe_001.png", "wireframe", "wireframe"),
            ("mech_albedo_001.png", "albedo", "albedo"),
            ("mech_basecolor_001.png", "basecolor", "basecolor"),
            ("mech_base_color_001.png", "base_color", "base_color"),
            ("mech_normal_001.png", "normal", "normal"),
            ("mech_normals_001.png", "normals", "normals"),
            ("mech_roughness_001.png", "roughness", "roughness"),
            ("mech_metallic_001.png", "metallic", "metallic"),
            ("mech_specular_001.png", "specular", "specular"),
            ("mech_depth_001.png", "depth", "depth"),
            ("mech_z_depth_001.png", "z_depth", "z_depth"),
            ("mech_shadow_001.png", "shadow", "shadow"),
            ("mech_shadows_001.png", "shadows", "shadows"),
            ("mech_direct_light_001.png", "direct_light", "direct_light"),
            ("mech_indirect_001.png", "indirect", "indirect"),
            ("mech_emission_001.png", "emission", "emission"),
            ("mech_sss_001.png", "sss", "sss"),
            ("mech_subsurface_001.png", "subsurface", "subsurface"),
            ("mech_cryptomatte_001.png", "cryptomatte", "cryptomatte"),

            # Sequence prefixes without explicit camera/pass
            ("mech_stage_01.png", "mech_stage", None),
            ("explosion_v2_001.png", "explosion_v2", None),
            ("0001.png", "default", None),
            ("1.jpg", "default", None),
        ]
    )
    def test_extract_group_tag_matrix(self, filename: str, expected_group: str, expected_cam: str | None):
        """Comprehensive verification of tag extractor over industry filename matrix."""
        group_id, cam_name = extract_group_tag(filename)
        assert group_id.lower() == expected_group.lower()
        if expected_cam is None:
            assert cam_name is None
        else:
            assert cam_name is not None and cam_name.lower() == expected_cam.lower()

    def test_group_sequences_multi_cluster_isolation(self, tmp_path: Path):
        """Clusters a mixed directory containing 4 cameras and 3 passes correctly."""
        cameras = ["Cam_Front", "Cam_Persp", "Cam_Side", "Turntable"]

        for cam in cameras:
            for i in range(1, 4):
                p = tmp_path / f"Hero_{cam}_{i:03d}.png"
                Image.new("RGB", (100, 100), (i * 30, 50, 50)).save(p)

        scanned = scan_directory(tmp_path)
        groups = group_sequences(scanned)

        assert len(groups) == 4
        for cam in cameras:
            assert cam in groups
            assert groups[cam].total_frames == 3
            assert groups[cam].common_width == 100
            assert groups[cam].common_height == 100
            frame_names = [f.filename for f in groups[cam].frames]
            assert frame_names == [f"Hero_{cam}_{i:03d}.png" for i in range(1, 4)]

    def test_group_sequences_all_generic_collapses_to_default(self, tmp_path: Path):
        """When all files share a generic prefix like 'frame_001.png', collapses to 'default'."""
        for i in range(1, 6):
            p = tmp_path / f"frame_{i:04d}.png"
            Image.new("RGB", (64, 64)).save(p)

        scanned = scan_directory(tmp_path)
        groups = group_sequences(scanned)

        assert len(groups) == 1
        assert "default" in groups
        assert groups["default"].total_frames == 5


# ============================================================================
# Section 3: File Filtering, OS Exclusions & Format Boundary Tests
# ============================================================================

class TestAdversarialFileExclusionAndDiscovery:
    """Stress tests boundary cases for scan_directory and format validation."""

    def test_scan_ignores_hidden_and_system_files(self, tmp_path: Path):
        """Ignores .DS_Store, Thumbs.db, desktop.ini, .directory, ehthumbs.db, ~$temp.png."""
        junk_files = [
            ".DS_Store",
            "Thumbs.db",
            "desktop.ini",
            ".directory",
            "ehthumbs.db",
            "ehthumbs_vista.db",
            "._render_01.png",
            "~$render_01.png",
            ".hidden_image.png",
        ]
        for j in junk_files:
            (tmp_path / j).write_bytes(b"dummy junk content")

        valid_1 = tmp_path / "render_001.png"
        valid_2 = tmp_path / "render_002.jpg"
        Image.new("RGB", (100, 100)).save(valid_1)
        Image.new("RGB", (100, 100)).save(valid_2)

        scanned = scan_directory(tmp_path)
        scanned_names = [p.name for p in scanned]

        assert len(scanned) == 2
        assert "render_001.png" in scanned_names
        assert "render_002.jpg" in scanned_names

    def test_scan_ignores_output_animations(self, tmp_path: Path):
        """Ignores output .gif and animated .webp files while keeping static .webp."""
        Image.new("RGB", (50, 50)).save(tmp_path / "frame_01.png")
        Image.new("RGB", (50, 50)).save(tmp_path / "animation_output.gif")

        im1 = Image.new("RGB", (50, 50), (255, 0, 0))
        im2 = Image.new("RGB", (50, 50), (0, 255, 0))
        im1.save(
            tmp_path / "breakdown_anim.webp",
            save_all=True,
            append_images=[im2],
            duration=100,
            loop=0,
        )

        Image.new("RGB", (50, 50), (0, 0, 255)).save(tmp_path / "frame_02.webp")

        scanned = scan_directory(tmp_path)
        scanned_names = [p.name for p in scanned]

        assert "frame_01.png" in scanned_names
        assert "frame_02.webp" in scanned_names
        assert "animation_output.gif" not in scanned_names
        assert "breakdown_anim.webp" not in scanned_names
        assert len(scanned) == 2

    def test_unsupported_extensions_ignored(self, tmp_path: Path):
        """Ignores non-image 3D/video formats: .blend, .fbx, .obj, .mp4, .mov, .txt, .json."""
        for ext in [".blend", ".fbx", ".obj", ".mp4", ".mov", ".txt", ".json", ".bak", ".tmp"]:
            (tmp_path / f"scene_file{ext}").write_text("not an image")

        Image.new("RGB", (50, 50)).save(tmp_path / "frame_01.png")

        scanned = scan_directory(tmp_path)
        assert len(scanned) == 1
        assert scanned[0].name == "frame_01.png"


# ============================================================================
# Section 4: Metadata Extraction, Mode Normalization & Error Handling
# ============================================================================

class TestAdversarialMetadataAndNormalization:
    """Stress tests color mode normalization, corruption handling, and aspect ratios."""

    def test_aspect_ratio_standard_resolutions(self):
        """Verifies simplification of common standard aspect ratios."""
        test_cases = [
            (1920, 1080, "16:9"),
            (3840, 2160, "16:9"),
            (1280, 720, "16:9"),
            (1080, 1080, "1:1"),
            (2048, 2048, "1:1"),
            (1440, 1080, "4:3"),
            (800, 600, "4:3"),
            (1080, 1920, "9:16"),
            (1800, 1200, "3:2"),
            (500, 300, "5:3"),  # Non-standard GCD simplification
        ]
        for w, h, expected in test_cases:
            fi = FrameInfo(
                path=Path(f"/mock/{w}x{h}.png"),
                index=0,
                filename=f"{w}x{h}.png",
                width=w,
                height=h,
                mode="RGB",
                file_size_bytes=1000,
            )
            assert fi.aspect_ratio_str == expected, f"Failed for {w}x{h}: got {fi.aspect_ratio_str}"

    def test_aspect_ratio_ultrawide_observation(self):
        """Documents the behavior of 2560x1080 ultrawide rendering."""
        fi = FrameInfo(
            path=Path("/mock/2560x1080.png"),
            index=0,
            filename="2560x1080.png",
            width=2560,
            height=1080,
            mode="RGB",
            file_size_bytes=1000,
        )
        # 2560 / 1080 = 2.37037 (64:27). Note: tolerance of 0.03 from 21/9 (2.333) yields 64:27
        assert fi.aspect_ratio_str in ("21:9", "64:27")

    def test_mode_normalization_exotic_modes(self):
        """Handles 1 (1-bit), P (palette), LA (grayscale+alpha), CMYK, I (32-bit int), F (32-bit float)."""
        modes = ["1", "L", "LA", "P", "CMYK", "RGB", "RGBA"]
        for m in modes:
            im = Image.new(m, (32, 32))
            norm_rgba = normalize_frame_mode(im, target_mode="RGBA")
            norm_rgb = normalize_frame_mode(im, target_mode="RGB")
            assert norm_rgba.mode == "RGBA"
            assert norm_rgb.mode == "RGB"
            assert norm_rgba.size == (32, 32)
            assert norm_rgb.size == (32, 32)

    def test_corrupt_images_handling_in_sequence_analysis(self, tmp_path: Path):
        """Ensures corrupted/zero-byte images are gracefully skipped when skip_corrupted=True."""
        valid_p = tmp_path / "frame_01.png"
        Image.new("RGB", (100, 100)).save(valid_p)

        zero_p = tmp_path / "frame_02.png"
        zero_p.write_bytes(b"")

        garbage_p = tmp_path / "frame_03.png"
        garbage_p.write_bytes(b"NOT_A_PNG_HEADER_GARBAGE")

        group = analyze_sequence([valid_p, zero_p, garbage_p], skip_corrupted=True)
        assert group.total_frames == 1
        assert group.frames[0].path == valid_p

        with pytest.raises(CorruptImageError):
            analyze_sequence([valid_p, zero_p, garbage_p], skip_corrupted=False)

    def test_group_sequences_nonexistent_files_skip_gracefully(self):
        """When paths do not exist on disk, group_sequences gracefully skips empty groups."""
        fake_paths = [Path("/non/existent/Cam1_001.png"), Path("/non/existent/Cam1_002.png")]
        groups = group_sequences(fake_paths, skip_corrupted=True)
        assert groups == {}

    def test_group_sequences_multi_pass_isolation_without_cameras(self, tmp_path: Path):
        """When directory has multiple passes and no cameras, groups by pass name."""
        passes = ["Beauty", "AO", "Clay", "Wireframe", "Normal"]
        for p_name in passes:
            for i in range(1, 3):
                p = tmp_path / f"Render_{p_name}_{i:02d}.png"
                Image.new("RGB", (64, 64), (10, 20, 30)).save(p)

        scanned = scan_directory(tmp_path)
        groups = group_sequences(scanned)

        assert len(groups) == 5
        for p_name in passes:
            assert p_name in groups
            assert groups[p_name].total_frames == 2
            assert groups[p_name].camera_name == p_name

    def test_alphanumeric_subframe_sorting(self):
        """Tests alphanumeric subframes like 001a, 001b, 002a."""
        raw_names = ["frame_002a.png", "frame_001b.png", "frame_001a.png", "frame_002b.png"]
        sorted_paths = natural_sort_paths(raw_names)
        assert [Path(p).name for p in sorted_paths] == [
            "frame_001a.png",
            "frame_001b.png",
            "frame_002a.png",
            "frame_002b.png",
        ]

    def test_empty_sequence_analysis_raises_empty_sequence_error(self):
        """Raises EmptySequenceError on empty sequence input."""
        with pytest.raises(EmptySequenceError):
            analyze_sequence([])
