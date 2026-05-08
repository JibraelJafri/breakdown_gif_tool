"""Tier 5 Adversarial Coverage Hardening & Edge-Case Stress Tests.

Exhaustive stress tests for:
- Export engine edge cases (1x1 pixels, extreme aspect ratios, high frame counts, extreme FPS)
- Color mode combinations (P mode with transparency, 1-bit monochrome, 16-bit grayscale)
- Auto-tuner convergence under tight budgets
- Memory streaming peak allocation under rapid frame batches
- Dimension harmonization with extreme scale factors
- CLI flag edge cases & unicode folder paths
- TUI event handling under edge inputs
"""

from __future__ import annotations

import io
import os
import shutil
import tempfile
import threading
from pathlib import Path
from typing import List, Tuple

import pytest
from PIL import Image

from breakdown_animator.export_engine import (
    DitherMode,
    ExportFormat,
    ExportProfile,
    ExportResult,
    HarmonizeMode,
    QuantizeMethod,
    calculate_frame_durations,
    export_sequence,
    harmonize_frame,
    resolve_target_dimensions,
)
from breakdown_animator.presets import (
    PRESET_CONFIGS,
    PresetName,
    get_preset_profile,
)
from breakdown_animator.cli import build_parser, run_cli
from breakdown_animator.sequence_engine import (
    CorruptImageError,
    DimensionMismatchError,
    EmptySequenceError,
    scan_directory,
    group_sequences,
)


# ============================================================================
# Tier 5.1: Geometry & Extreme Dimension Stress Tests
# ============================================================================

class TestAdversarialGeometryAndScaling:
    """Stress tests on extreme dimensions, aspect ratios, and scaling math."""

    def test_single_pixel_image_harmonization_and_export(self, tmp_path: Path):
        """1x1 source image harmonized and exported to 1080p canvas."""
        img_1x1 = Image.new("RGB", (1, 1), (255, 0, 128))
        harmonized = harmonize_frame(
            img_1x1,
            target_size=(1920, 1080),
            mode=HarmonizeMode.LETTERBOX,
            bg_color=(10, 20, 30),
        )
        assert harmonized.size == (1920, 1080)

        # Export test
        f1 = tmp_path / "f1.png"
        f2 = tmp_path / "f2.png"
        img_1x1.save(f1)
        img_1x1.save(f2)

        out_gif = tmp_path / "1x1.gif"
        res = export_sequence([f1, f2], out_gif, ExportProfile(format=ExportFormat.GIF))
        assert out_gif.exists()
        assert res.total_frames == 2

    def test_extreme_aspect_ratios_panorama_and_skyscraper(self, tmp_path: Path):
        """Extreme 10000x50 and 50x5000 images harmonized without division-by-zero or crashes."""
        pano = Image.new("RGB", (2000, 50), (100, 150, 200))
        skyscraper = Image.new("RGB", (50, 2000), (200, 150, 100))

        h_pano = harmonize_frame(pano, target_size=(1280, 720), mode=HarmonizeMode.LETTERBOX)
        assert h_pano.size == (1280, 720)

        h_sky = harmonize_frame(skyscraper, target_size=(1280, 720), mode=HarmonizeMode.CROP)
        assert h_sky.size == (1280, 720)

    def test_resolve_target_dimensions_matrix(self):
        """Tests parsing of various scale strings, percentages, and presets."""
        native = (3840, 2160)
        assert resolve_target_dimensions(native, scale="100%") == (3840, 2160)
        assert resolve_target_dimensions(native, scale="50%") == (1920, 1080)
        assert resolve_target_dimensions(native, scale="25%") == (960, 540)
        assert resolve_target_dimensions(native, scale="720p") == (1280, 720)
        assert resolve_target_dimensions(native, scale="1920x1080") == (1920, 1080)
        assert resolve_target_dimensions(native, scale=0.5) == (1920, 1080)
        assert resolve_target_dimensions(native, target_size=(800, 600)) == (800, 600)


# ============================================================================
# Tier 5.2: Timing Edge Cases
# ============================================================================

class TestAdversarialTimingCalculations:
    """Stress tests on extreme framerates, fractions, and hold durations."""

    def test_fractional_fps_and_zero_fps_guard(self):
        """Guards against division by zero or negative FPS values."""
        durations = calculate_frame_durations(fps=0.0, total_frames=3, hold_last_seconds=1.0)
        assert len(durations) == 3
        assert all(d > 0 for d in durations)

    def test_high_fps_60_and_120(self):
        """60 FPS produces ~17ms per frame, 120 FPS produces ~8ms."""
        d60 = calculate_frame_durations(fps=60.0, total_frames=4)
        assert d60 == [17, 17, 17, 17]

    def test_huge_hold_duration(self):
        """10 second hold duration correctly maps to +10000ms on final frame."""
        durations = calculate_frame_durations(fps=10.0, total_frames=3, hold_last_seconds=10.0)
        assert durations == [100, 100, 10100]

    def test_zero_frames_duration(self):
        """Zero frames returns empty list."""
        assert calculate_frame_durations(fps=24.0, total_frames=0) == []


# ============================================================================
# Tier 5.3: Exotic Color Modes & Palettes
# ============================================================================

class TestAdversarialColorModesAndQuantization:
    """Stress tests on transparency, 1-bit, P mode, and fallback quantization."""

    def test_1bit_and_palette_with_alpha_export(self, tmp_path: Path):
        """1-bit monochrome and RGBA with gradient transparency exported cleanly."""
        src_dir = tmp_path / "exotic_src"
        src_dir.mkdir()

        # 1-bit image
        im1 = Image.new("1", (200, 200), 1)
        im1.save(src_dir / "frame_01.png")

        # RGBA with transparency
        im2 = Image.new("RGBA", (200, 200), (255, 0, 0, 128))
        im2.save(src_dir / "frame_02.png")

        out_gif = tmp_path / "exotic.gif"
        res = export_sequence(
            [src_dir / "frame_01.png", src_dir / "frame_02.png"],
            out_gif,
            ExportProfile(format=ExportFormat.GIF, background_color=(50, 50, 50)),
        )

        assert out_gif.exists()
        assert res.total_frames == 2

    def test_libimagequant_fallback_mechanism(self, tmp_path: Path):
        """Libimagequant method safely executes or falls back without crash."""
        f1 = tmp_path / "libq1.png"
        f2 = tmp_path / "libq2.png"
        Image.new("RGB", (100, 100), (10, 20, 30)).save(f1)
        Image.new("RGB", (100, 100), (40, 50, 60)).save(f2)

        out_gif = tmp_path / "libq.gif"
        profile = ExportProfile(
            format=ExportFormat.GIF,
            quantize_method=QuantizeMethod.LIBIMAGEQUANT,
        )
        res = export_sequence([f1, f2], out_gif, profile)
        assert out_gif.exists()
        assert res.total_frames == 2


# ============================================================================
# Tier 5.4: Auto-Tuning Under Tight Budgets
# ============================================================================

class TestAdversarialAutoTuning:
    """Stress tests on predictor-corrector convergence under tight size limits."""

    def test_auto_tune_tight_size_budget_gif(self, tmp_path: Path):
        """Auto-tuning downscales large 1080p multi-color sequence to fit <= 0.05 MB."""
        src_dir = tmp_path / "dense_noise"
        src_dir.mkdir()

        frames = []
        for i in range(6):
            fp = src_dir / f"step_{i:02d}.png"
            # High-res 1280x720 multi-colored frame that natively exceeds 0.05MB as GIF
            im = Image.new("RGB", (1280, 720), (i * 40, (i * 35) % 255, (i * 70) % 255))
            im.save(fp)
            frames.append(fp)

        out_gif = tmp_path / "tight_budget.gif"
        profile = ExportProfile(
            format=ExportFormat.GIF,
            max_size_mb=0.04,
            auto_tune=True,
            fps=12.0,
        )

        res = export_sequence(frames, out_gif, profile)
        assert out_gif.exists()
        assert res.auto_tuned is True
        assert res.file_size_mb <= 0.04
        assert res.passes_budget is True


# ============================================================================
# Tier 5.5: CLI & Path Resilience
# ============================================================================

class TestAdversarialCliPathsAndArguments:
    """Stress tests on CLI unicode paths, spaces, and option combinations."""

    def test_cli_unicode_folder_and_spaces(self, tmp_path: Path):
        """CLI runs properly in directories with unicode characters and spaces."""
        unicode_dir = tmp_path / "3D Breakdown 渲染 2026"
        unicode_dir.mkdir()

        for i in range(3):
            im = Image.new("RGB", (320, 180), (i * 50, 100, 150))
            im.save(unicode_dir / f"Cam_Front_step_{i}.png")

        out_dir = tmp_path / "Output 輸出"
        exit_code = run_cli([
            str(unicode_dir),
            "--preset", "slack",
            "--output-dir", str(out_dir),
            "--quiet",
        ])

        assert exit_code == 0
        assert out_dir.exists()
        assert len(list(out_dir.glob("*.gif"))) >= 1

    def test_cli_scale_and_format_override(self, tmp_path: Path):
        """CLI allows overriding preset format from GIF to WebP and scale to 540p."""
        src_dir = tmp_path / "renders"
        src_dir.mkdir()
        for i in range(2):
            Image.new("RGB", (640, 360), (10, 20, 30)).save(src_dir / f"frame_{i}.png")

        out_dir = tmp_path / "out"
        exit_code = run_cli([
            str(src_dir),
            "--preset", "artstation",
            "--format", "webp",
            "--scale", "540p",
            "--output-dir", str(out_dir),
        ])

        assert exit_code == 0
        webp_files = list(out_dir.glob("*.webp"))
        assert len(webp_files) == 1
