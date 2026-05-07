"""breakdown_animator.export_engine
================================
High-performance, pure-Pillow export engine for animated GIF and WebP generation
with zero external binary dependencies. Features global palette quantization,
aspect ratio harmonization, timing math, streaming generator processing,
and predictor-corrector file-size auto-tuning.
"""

from __future__ import annotations

import gc
import math
import os
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

from PIL import Image, features

from breakdown_animator.sequence_engine import (
    CorruptImageError,
    DimensionMismatchError,
    EmptySequenceError,
    FrameInfo,
    load_and_normalize_frame,
    normalize_frame_mode,
)


# ============================================================================
# Enums
# ============================================================================

class ExportFormat(str, Enum):
    """Target animation container format."""
    GIF = "gif"
    WEBP = "webp"

    @classmethod
    def from_str(cls, val: Union[str, ExportFormat]) -> ExportFormat:
        if isinstance(val, cls):
            return val
        s = str(val).strip().lower().lstrip(".")
        if s == "gif":
            return cls.GIF
        elif s in ("webp", "animated_webp"):
            return cls.WEBP
        raise ValueError(f"Unsupported export format '{val}'. Expected 'gif' or 'webp'.")


class QuantizeMethod(str, Enum):
    """Pillow color quantization algorithm for 8-bit palette images."""
    MEDIAN_CUT = "median_cut"
    FAST_OCTREE = "fast_octree"
    MAX_COVERAGE = "max_coverage"
    LIBIMAGEQUANT = "libimagequant"


class DitherMode(str, Enum):
    """Color dithering mode for palette quantization."""
    FLOYD_STEINBERG = "floyd_steinberg"
    NONE = "none"

    @classmethod
    def from_val(cls, val: Union[str, bool, DitherMode]) -> DitherMode:
        if isinstance(val, cls):
            return val
        if isinstance(val, bool):
            return cls.FLOYD_STEINBERG if val else cls.NONE
        s = str(val).strip().lower()
        if s in ("floyd_steinberg", "floyd", "fs", "dither", "true", "1"):
            return cls.FLOYD_STEINBERG
        elif s in ("none", "no", "nodither", "no_dither", "false", "0"):
            return cls.NONE
        return cls.FLOYD_STEINBERG


class HarmonizeMode(str, Enum):
    """Dimension harmonization method for mismatched frame aspect ratios."""
    LETTERBOX = "letterbox"
    FIT = "fit"
    CROP = "crop"
    STRETCH = "stretch"


# Standard scale resolution map
SCALE_PRESETS: Dict[str, Tuple[int, int]] = {
    "4k": (3840, 2160),
    "2160p": (3840, 2160),
    "1440p": (2560, 1440),
    "2k": (2560, 1440),
    "1080p": (1920, 1080),
    "fhd": (1920, 1080),
    "720p": (1280, 720),
    "hd": (1280, 720),
    "540p": (960, 540),
    "qhd": (960, 540),
    "480p": (854, 480),
    "sd": (854, 480),
    "360p": (640, 360),
}


# ============================================================================
# Dataclasses
# ============================================================================

@dataclass
class ExportProfile:
    """Configuration profile governing the animation export pipeline."""
    format: ExportFormat = ExportFormat.GIF
    fps: float = 12.0
    frame_duration: Optional[float] = None
    hold_last_seconds: float = 0.0
    color_count: int = 256
    quantize_method: QuantizeMethod = QuantizeMethod.MEDIAN_CUT
    dither: Union[DitherMode, bool] = DitherMode.FLOYD_STEINBERG
    lossless: bool = False
    quality: int = 85
    scale: Optional[Union[str, float, int, Tuple[int, int]]] = None
    target_size: Optional[Tuple[int, int]] = None
    harmonize_mode: HarmonizeMode = HarmonizeMode.LETTERBOX
    background_color: Tuple[int, int, int] = (0, 0, 0)
    max_size_mb: Optional[float] = None
    auto_tune: bool = False
    loop: int = 0
    sample_global_palette: bool = True
    num_palette_samples: int = 5
    method: int = 6
    minimize_size: bool = True

    def get_dither_mode(self) -> DitherMode:
        return DitherMode.from_val(self.dither)


@dataclass
class ExportResult:
    """Metadata and statistics returned upon completion of an export."""
    output_path: Path
    format: ExportFormat
    total_frames: int
    file_size_bytes: int
    file_size_mb: float
    passes_budget: bool
    duration_ms: int
    dimensions: Tuple[int, int]
    auto_tuned: bool = False
    attempts: int = 1
    duration_seconds: float = 0.0

    def __post_init__(self):
        if self.duration_seconds == 0.0 and self.duration_ms > 0:
            self.duration_seconds = self.duration_ms / 1000.0


# ============================================================================
# Timing Math
# ============================================================================

def calculate_frame_durations(
    fps: Optional[float] = 12.0,
    total_frames: int = 1,
    hold_last_seconds: float = 0.0,
    frame_duration: Optional[float] = None,
) -> List[int]:
    """
    Computes an array of integer millisecond display durations for each frame in the sequence,
    incorporating the final-frame freeze hold delay.
    
    Args:
        fps: Playback rate in frames per second (1.0 to 60.0). Used if frame_duration is None.
        total_frames: Number of frames in the sequence.
        hold_last_seconds: Duration to freeze/pause on the final frame in seconds.
        frame_duration: Direct duration per normal step/frame in seconds (e.g. 1.0 or 1.5).
                        Overrides fps if provided.
        
    Returns:
        List of integer milliseconds for each frame.
    """
    if total_frames <= 0:
        return []

    if frame_duration is not None and frame_duration > 0:
        base_ms = max(1, round(float(frame_duration) * 1000.0))
    elif fps is not None and fps > 0:
        safe_fps = max(0.1, min(120.0, float(fps)))
        base_ms = max(1, round(1000.0 / safe_fps))
    else:
        base_ms = 1000

    hold_ms = max(0, round(float(hold_last_seconds) * 1000.0))

    if total_frames == 1:
        return [base_ms + hold_ms]

    durations = [base_ms] * (total_frames - 1)
    durations.append(base_ms + hold_ms)
    return durations


# ============================================================================
# Dimension Harmonization & Scaling
# ============================================================================
