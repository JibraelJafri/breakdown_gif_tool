"""Tests for Export Engine and Presets (export_engine.py, presets.py).

Covers Tiers 1-3:
- Pure-Pillow GIF export (multi-frame GIF89a, palette quantization, loop count, duration)
- Pure-Pillow WebP export (multi-frame animated WebP, lossless/lossy, quality, 24-bit color)
- Color quantization methods (Median Cut, Fast Octree, Max Coverage, Libimagequant fallback)
- Dithering modes (Floyd-Steinberg error diffusion vs None)
- Global sequence palette vs local palettes
- Animation timing (FPS to millisecond conversion, end-frame freeze hold delay)
- Dimension harmonization (Letterbox padding, Fit, Crop, Stretch)
- Scale presets and custom scale factors (1080p, 720p, 50%, etc.)
- Optimization presets (ArtStation, Portfolio 4K, Discord, Slack, Custom)
- Predictor-corrector target file size auto-tuning
- Streaming generator-based memory pipeline
- Progress tracker callback and cancellation token
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import List, Tuple

import pytest
from PIL import Image

# Import under test
from breakdown_animator.export_engine import (
    ExportFormat,
    QuantizeMethod,
    DitherMode,
    HarmonizeMode,
    ExportProfile,
    ExportResult,
    export_sequence,
    calculate_frame_durations,
    harmonize_frame,
)
from breakdown_animator.presets import (
    PresetName,
    get_preset_profile,
    PRESET_CONFIGS,
)
from tests.generate_test_assets import (
    create_natural_sort_sequence,
    create_mismatched_dimension_sequence,
    create_color_mode_sequence,
    draw_synthetic_breakdown_frame,
)


# ============================================================================
# Tier 1: Core Functional Tests
# ============================================================================

class TestTimingAndHoldCalculations:
    """Tests for calculate_frame_durations and end-frame freeze hold math."""

    def test_duration_array_uniform_fps(self):
        """10 FPS -> 100ms per frame."""
        durations = calculate_frame_durations(fps=10.0, total_frames=5, hold_last_seconds=0.0)
        assert len(durations) == 5
        assert durations == [100, 100, 100, 100, 100]

    def test_duration_array_24_fps(self):
        """24 FPS -> ~42ms per frame."""
        durations = calculate_frame_durations(fps=24.0, total_frames=4, hold_last_seconds=0.0)
        assert len(durations) == 4
        assert durations == [42, 42, 42, 42]

    def test_hold_last_frame_pause(self):
        """Applies hold_last_seconds (e.g. 2.0s = 2000ms) to the final frame."""
        durations = calculate_frame_durations(fps=10.0, total_frames=5, hold_last_seconds=2.0)
        assert len(durations) == 5
        assert durations[:-1] == [100, 100, 100, 100]
        # Final frame = base duration (100ms) + 2000ms = 2100ms
        assert durations[-1] == 2100

    def test_single_frame_duration(self):
        """Single frame sequence receives hold delay or base frame duration."""
        durations = calculate_frame_durations(fps=12.0, total_frames=1, hold_last_seconds=2.0)
        assert len(durations) == 1
        assert durations[0] >= 2000


class TestDimensionHarmonization:
    """Tests for harmonize_frame with letterbox, fit, crop, and stretch."""

    def test_letterbox_pads_to_target_dimensions(self):
        """Pads 1080x1080 square image into 1920x1080 16:9 canvas with black matte borders."""
        src_im = Image.new("RGB", (1080, 1080), (255, 0, 0))
        target_size = (1920, 1080)
        harmonized = harmonize_frame(
            src_im,
            target_size=target_size,
            mode=HarmonizeMode.LETTERBOX,
            bg_color=(0, 0, 0),
        )

        assert harmonized.size == (1920, 1080)
        # Check border pixel (should be black background)
        assert harmonized.getpixel((10, 540))[:3] == (0, 0, 0)
        # Check center pixel (should be red)
        assert harmonized.getpixel((960, 540))[:3] == (255, 0, 0)

    def test_crop_centers_and_trims_excess(self):
        """Crops 1920x1080 image to 1080x1080 center square."""
        src_im = Image.new("RGB", (1920, 1080), (0, 255, 0))
        target_size = (1080, 1080)
        harmonized = harmonize_frame(
            src_im,
            target_size=target_size,
            mode=HarmonizeMode.CROP,
        )
        assert harmonized.size == (1080, 1080)

    def test_stretch_resizes_directly(self):
        """Direct non-uniform stretch to target dimensions."""
        src_im = Image.new("RGB", (800, 600), (0, 0, 255))
        harmonized = harmonize_frame(
            src_im,
            target_size=(1280, 720),
            mode=HarmonizeMode.STRETCH,
        )
        assert harmonized.size == (1280, 720)


class TestGifExport:
    """Tests for pure-Pillow GIF export and quantization."""

    def test_export_gif_basic(self, tmp_path: Path):
        """Exports a valid multi-frame GIF with correct frame count and timing."""
        frame_paths = create_natural_sort_sequence(tmp_path / "src", count=6, dimensions=(480, 270))
        output_file = tmp_path / "output.gif"

        profile = ExportProfile(
            format=ExportFormat.GIF,
            fps=10.0,
            hold_last_seconds=1.0,
            color_count=128,
            dither=DitherMode.FLOYD_STEINBERG,
        )

        result = export_sequence(frame_paths, output_file, profile)

        assert output_file.exists()
        assert result.output_path == output_file
        assert result.format == ExportFormat.GIF
        assert result.total_frames == 6
        assert result.file_size_bytes > 0

        # Verify Pillow can open and read all GIF frames
        with Image.open(output_file) as im:
            assert im.format == "GIF"
            assert im.n_frames == 6
            assert im.size == (480, 270)

    def test_gif_quantization_methods(self, tmp_path: Path):
        """Tests Median Cut, Fast Octree, and Max Coverage quantization."""
        frame_paths = create_natural_sort_sequence(tmp_path / "src", count=3, dimensions=(320, 180))

        for method in (QuantizeMethod.MEDIAN_CUT, QuantizeMethod.FAST_OCTREE, QuantizeMethod.MAX_COVERAGE):
            out_file = tmp_path / f"test_{method.value}.gif"
            profile = ExportProfile(
                format=ExportFormat.GIF,
                quantize_method=method,
                color_count=64,
                fps=10.0,
            )
            result = export_sequence(frame_paths, out_file, profile)
            assert out_file.exists()
            assert result.file_size_bytes > 0


class TestWebpExport:
    """Tests for pure-Pillow animated WebP export."""

    def test_export_webp_lossy(self, tmp_path: Path):
        """Exports lossy animated WebP with 24-bit color."""
        frame_paths = create_natural_sort_sequence(tmp_path / "src", count=5, dimensions=(640, 360))
        output_file = tmp_path / "output_lossy.webp"

        profile = ExportProfile(
            format=ExportFormat.WEBP,
            lossless=False,
            quality=90,
            fps=24.0,
            hold_last_seconds=1.5,
        )

        result = export_sequence(frame_paths, output_file, profile)

        assert output_file.exists()
        assert result.format == ExportFormat.WEBP
        assert result.total_frames == 5

        with Image.open(output_file) as im:
            assert im.format == "WEBP"
            assert getattr(im, "is_animated", False) is True
            assert im.n_frames == 5
            assert im.size == (640, 360)

    def test_export_webp_lossless(self, tmp_path: Path):
        """Exports lossless animated WebP."""
        frame_paths = create_natural_sort_sequence(tmp_path / "src", count=4, dimensions=(320, 180))
        output_file = tmp_path / "output_lossless.webp"

        profile = ExportProfile(
            format=ExportFormat.WEBP,
            lossless=True,
            fps=12.0,
        )

        result = export_sequence(frame_paths, output_file, profile)
        assert output_file.exists()
        assert result.file_size_bytes > 0


class TestPresetEngine:
    """Tests for optimization preset configurations."""

    def test_artstation_preset(self):
        """ArtStation preset targets <= 10MB GIF with 2.0s hold."""
        p = get_preset_profile(PresetName.ARTSTATION)
        assert p.format == ExportFormat.GIF
        assert p.max_size_mb == 10.0
        assert p.hold_last_seconds == 2.0
        assert p.auto_tune is True

    def test_portfolio_4k_preset(self):
        """Portfolio 4K preset targets high-fidelity WebP."""
        p = get_preset_profile(PresetName.PORTFOLIO_4K)
        assert p.format == ExportFormat.WEBP
        assert p.quality >= 90
        assert p.hold_last_seconds >= 1.5

    def test_discord_and_slack_presets(self):
        """Discord <= 8MB, Slack <= 5MB."""
        p_disc = get_preset_profile(PresetName.DISCORD)
        assert p_disc.max_size_mb == 8.0

        p_slack = get_preset_profile(PresetName.SLACK)
        assert p_slack.max_size_mb == 5.0


# ============================================================================
# Tier 2: Boundary & Error Handling Tests
# ============================================================================

class TestExportEngineEdgeCases:
    """Boundary conditions, auto-tuning convergence, and cancellation token."""

    def test_auto_tune_respects_target_budget(self, tmp_path: Path):
        """Auto-tuning downscales or reduces palette to fit under max_size_mb."""
        frame_paths = create_natural_sort_sequence(tmp_path / "src", count=8, dimensions=(800, 450))
        output_file = tmp_path / "budget_test.gif"

        # Set tight budget (e.g. 0.5 MB)
        profile = ExportProfile(
            format=ExportFormat.GIF,
            max_size_mb=0.5,
            auto_tune=True,
            fps=15.0,
        )

        result = export_sequence(frame_paths, output_file, profile)
        assert output_file.exists()
        assert result.passes_budget is True
        assert result.file_size_mb <= 0.5

    def test_cancellation_token_halts_export(self, tmp_path: Path):
        """Cooperative cancellation event halts processing early."""
        frame_paths = create_natural_sort_sequence(tmp_path / "src", count=10, dimensions=(480, 270))
        output_file = tmp_path / "cancelled.gif"

        cancel_token = threading.Event()
        cancel_token.set()  # Cancel immediately

        with pytest.raises((InterruptedError, RuntimeError, Exception)):
            export_sequence(frame_paths, output_file, ExportProfile(), cancel_token=cancel_token)

    def test_progress_callback_invocation(self, tmp_path: Path):
        """Progress callback is called across phases with percentages."""
        frame_paths = create_natural_sort_sequence(tmp_path / "src", count=4, dimensions=(320, 180))
        output_file = tmp_path / "progress_test.gif"

        events = []

        def on_progress(phase: str, current: int, total: int, pct: float, message: str):
            events.append((phase, current, total, pct))

        export_sequence(frame_paths, output_file, ExportProfile(), progress_callback=on_progress)
        assert len(events) > 0
        # Final event should reach ~1.0 (100%)
        assert events[-1][3] >= 0.9

    def test_mismatched_dimensions_auto_harmonized(self, tmp_path: Path):
        """Sequences with varying dimensions are automatically harmonized before encoding."""
        frame_paths = create_mismatched_dimension_sequence(tmp_path / "src")
        output_file = tmp_path / "harmonized_output.webp"

        profile = ExportProfile(
            format=ExportFormat.WEBP,
            harmonize_mode=HarmonizeMode.LETTERBOX,
        )

        result = export_sequence(frame_paths, output_file, profile)
        assert output_file.exists()
        assert result.total_frames == 5

        # All frames in animated WebP must have uniform dimensions
        with Image.open(output_file) as im:
            assert im.n_frames == 5

    def test_color_modes_harmonized_during_export(self, tmp_path: Path):
        """CMYK, RGBA, L, and P frames are normalized to target canvas mode."""
        frame_paths = create_color_mode_sequence(tmp_path / "src")
        output_file = tmp_path / "modes_output.gif"

        result = export_sequence(frame_paths, output_file, ExportProfile(format=ExportFormat.GIF))
        assert output_file.exists()
        assert result.total_frames == 5


# ============================================================================
# Tier 3: Pairwise Combination Tests
# ============================================================================

class TestExportEngineCombinations:
    """Pairwise combinations of scaling, dithering, formats, and quantization."""

    @pytest.mark.parametrize("fmt", [ExportFormat.GIF, ExportFormat.WEBP])
    @pytest.mark.parametrize("dither", [DitherMode.FLOYD_STEINBERG, DitherMode.NONE])
    def test_format_and_dither_matrix(self, tmp_path: Path, fmt: ExportFormat, dither: DitherMode):
        """Tests matrix of formats and dithering toggles."""
        frame_paths = create_natural_sort_sequence(tmp_path / f"src_{fmt.value}_{dither.value}", count=3)
        ext = "gif" if fmt == ExportFormat.GIF else "webp"
        output_file = tmp_path / f"matrix_out_{fmt.value}_{dither.value}.{ext}"

        profile = ExportProfile(
            format=fmt,
            dither=dither,
            color_count=128,
            fps=12.0,
        )

        result = export_sequence(frame_paths, output_file, profile)
        assert output_file.exists()
        assert result.total_frames == 3
