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

def resolve_target_dimensions(
    original_size: Tuple[int, int],
    scale: Optional[Union[str, float, int, Tuple[int, int]]] = None,
    target_size: Optional[Tuple[int, int]] = None,
) -> Tuple[int, int]:
    """
    Resolves the final canvas dimensions from native size and scale options.
    """
    if target_size is not None and target_size[0] > 0 and target_size[1] > 0:
        return (int(target_size[0]), int(target_size[1]))

    orig_w, orig_h = original_size
    if scale is None or scale == "" or scale == "100%" or scale == "native":
        return (orig_w, orig_h)

    if isinstance(scale, tuple) and len(scale) == 2:
        return (int(scale[0]), int(scale[1]))

    if isinstance(scale, (int, float)):
        f = float(scale)
        if f > 0:
            return (max(1, round(orig_w * f)), max(1, round(orig_h * f)))
        return (orig_w, orig_h)

    s_str = str(scale).strip().lower()
    if s_str in SCALE_PRESETS:
        preset_w, preset_h = SCALE_PRESETS[s_str]
        # Preserve aspect ratio fit within preset bounding box
        ratio = min(preset_w / orig_w, preset_h / orig_h)
        return (max(1, round(orig_w * ratio)), max(1, round(orig_h * ratio)))

    if s_str.endswith("%"):
        try:
            pct = float(s_str.rstrip("%")) / 100.0
            return (max(1, round(orig_w * pct)), max(1, round(orig_h * pct)))
        except ValueError:
            pass

    if "x" in s_str:
        parts = s_str.split("x")
        if len(parts) == 2:
            try:
                return (int(parts[0].strip()), int(parts[1].strip()))
            except ValueError:
                pass

    return (orig_w, orig_h)


def harmonize_frame(
    im: Image.Image,
    target_size: Tuple[int, int],
    mode: Union[HarmonizeMode, str] = HarmonizeMode.LETTERBOX,
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    background_color: Optional[Tuple[int, int, int]] = None,
) -> Image.Image:
    """
    Harmonizes an individual image frame to the target dimensions using
    letterboxing, cropping, stretching, or proportional fitting.
    
    Args:
        im: Source PIL Image.
        target_size: Desired (width, height) output dimensions.
        mode: HarmonizeMode (LETTERBOX, FIT, CROP, STRETCH).
        bg_color: Background matte color tuple (R, G, B).
        background_color: Optional alias for bg_color.
        
    Returns:
        Harmonized PIL Image matching target_size.
    """
    effective_bg = background_color if background_color is not None else bg_color
    h_mode = HarmonizeMode(mode) if not isinstance(mode, HarmonizeMode) else mode

    tw, th = target_size
    sw, sh = im.size

    if (sw, sh) == (tw, th):
        return im.copy()

    if h_mode in (HarmonizeMode.LETTERBOX, HarmonizeMode.FIT):
        scale = min(tw / sw, th / sh)
        nw = max(1, round(sw * scale))
        nh = max(1, round(sh * scale))
        resized = im.resize((nw, nh), resample=Image.Resampling.LANCZOS)

        canvas_mode = "RGBA" if im.mode == "RGBA" else "RGB"
        canvas_bg = (effective_bg[0], effective_bg[1], effective_bg[2], 255) if canvas_mode == "RGBA" else effective_bg
        canvas = Image.new(canvas_mode, (tw, th), canvas_bg)

        paste_x = (tw - nw) // 2
        paste_y = (th - nh) // 2

        if resized.mode == "RGBA" and canvas_mode == "RGBA":
            canvas.paste(resized, (paste_x, paste_y), mask=resized.split()[3])
        elif resized.mode == "RGBA" and canvas_mode == "RGB":
            bg_tile = Image.new("RGB", (nw, nh), effective_bg)
            bg_tile.paste(resized.convert("RGB"), mask=resized.split()[3])
            canvas.paste(bg_tile, (paste_x, paste_y))
        else:
            canvas.paste(resized.convert(canvas_mode), (paste_x, paste_y))

        return canvas

    elif h_mode == HarmonizeMode.CROP:
        scale = max(tw / sw, th / sh)
        nw = max(tw, round(sw * scale))
        nh = max(th, round(sh * scale))
        resized = im.resize((nw, nh), resample=Image.Resampling.LANCZOS)
        crop_x = (nw - tw) // 2
        crop_y = (nh - th) // 2
        return resized.crop((crop_x, crop_y, crop_x + tw, crop_y + th))

    elif h_mode == HarmonizeMode.STRETCH:
        return im.resize((tw, th), resample=Image.Resampling.LANCZOS)

    return im.copy()


# ============================================================================
