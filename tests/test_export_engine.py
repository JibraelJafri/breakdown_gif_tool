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
