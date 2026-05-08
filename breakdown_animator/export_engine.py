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
# Quantization & Palette Engineering
# ============================================================================

def _get_pillow_quantize_constant(method: QuantizeMethod) -> int:
    """Maps QuantizeMethod enum to Pillow Image.Quantize integer constant with fallback."""
    if method == QuantizeMethod.FAST_OCTREE:
        return Image.Quantize.FASTOCTREE
    elif method == QuantizeMethod.MAX_COVERAGE:
        return Image.Quantize.MAXCOVERAGE
    elif method == QuantizeMethod.LIBIMAGEQUANT:
        if hasattr(Image.Quantize, "LIBIMAGEQUANT") and features.check_feature("libimagequant"):
            return Image.Quantize.LIBIMAGEQUANT
        return Image.Quantize.MEDIANCUT
    return Image.Quantize.MEDIANCUT


def generate_global_sequence_palette(
    frame_paths: Sequence[Union[Path, str]],
    color_count: int = 256,
    method: QuantizeMethod = QuantizeMethod.MEDIAN_CUT,
    num_samples: int = 5,
    target_size: Optional[Tuple[int, int]] = None,
    harmonize_mode: HarmonizeMode = HarmonizeMode.LETTERBOX,
    background_color: Tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """
    Generates a single global representative palette for the entire image sequence
    by compositing uniformly sampled sub-frames, preventing palette flickering across frames.
    """
    n = len(frame_paths)
    if n == 0:
        raise EmptySequenceError("Cannot generate palette from empty frame list.")

    sample_count = min(n, max(1, num_samples))
    indices = [int(round(i * (n - 1) / (sample_count - 1))) if sample_count > 1 else 0 for i in range(sample_count)]
    unique_indices = sorted(set(indices))

    sample_images: List[Image.Image] = []
    for idx in unique_indices:
        try:
            im = load_and_normalize_frame(frame_paths[idx], target_mode="RGB", background_color=background_color)
            if target_size:
                im = harmonize_frame(im, target_size=target_size, mode=harmonize_mode, bg_color=background_color)
            # Downsample for fast representative palette generation
            w, h = im.size
            if w > 320 or h > 240:
                scale = min(320 / w, 240 / h)
                im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), resample=Image.Resampling.BOX)
            sample_images.append(im)
        except Exception:
            continue

    quant_const = _get_pillow_quantize_constant(method)

    if not sample_images:
        fallback = Image.new("RGB", (16, 16), background_color)
        try:
            return fallback.quantize(colors=color_count, method=quant_const)
        except Exception:
            return fallback.quantize(colors=color_count, method=Image.Quantize.MEDIANCUT)

    total_w = sum(img.width for img in sample_images)
    max_h = max(img.height for img in sample_images)
    composite = Image.new("RGB", (total_w, max_h), background_color)

    cur_x = 0
    for img in sample_images:
        composite.paste(img, (cur_x, 0))
        cur_x += img.width
        img.close()

    try:
        palette_im = composite.quantize(colors=min(256, max(2, color_count)), method=quant_const)
    except Exception:
        palette_im = composite.quantize(colors=min(256, max(2, color_count)), method=Image.Quantize.MEDIANCUT)
    composite.close()
    return palette_im


# ============================================================================
# Core Sequence Export & Auto-Tuning Engine
# ============================================================================

def _export_single_pass(
    frame_paths: List[Path],
    output_path: Path,
    profile: ExportProfile,
    target_dimensions: Tuple[int, int],
    durations: List[int],
    progress_callback: Optional[Callable[[str, int, int, float, str], None]] = None,
    cancel_token: Optional[threading.Event] = None,
) -> Tuple[int, Tuple[int, int]]:
    """
    Executes a single pass of sequence transformation and file encoding.
    Returns (file_size_bytes, final_dimensions).
    """
    total_frames = len(frame_paths)
    out_fmt = profile.format

    def report_progress(phase: str, current: int, total: int, pct: float, msg: str):
        if progress_callback:
            progress_callback(phase, current, total, pct, msg)

    if cancel_token and cancel_token.is_set():
        raise InterruptedError("Export operation cancelled by user token.")

    report_progress("Validating", 0, total_frames, 0.05, "Validating sequence frames...")

    dither_mode = profile.get_dither_mode()
    dither_flag = Image.Dither.FLOYDSTEINBERG if dither_mode == DitherMode.FLOYD_STEINBERG else Image.Dither.NONE

    # 1. Prepare palette for GIF if applicable
    global_palette_im: Optional[Image.Image] = None
    if out_fmt == ExportFormat.GIF and profile.sample_global_palette:
        report_progress("Sampling Palette", 0, total_frames, 0.15, "Sampling global sequence color palette...")
        try:
            global_palette_im = generate_global_sequence_palette(
                frame_paths=frame_paths,
                color_count=profile.color_count,
                method=profile.quantize_method,
                num_samples=profile.num_palette_samples,
                target_size=target_dimensions,
                harmonize_mode=profile.harmonize_mode,
                background_color=profile.background_color,
            )
        except Exception:
            global_palette_im = None

    if cancel_token and cancel_token.is_set():
        raise InterruptedError("Export operation cancelled by user token.")

    # 2. Process, harmonize, and quantize frames
    processed_frames: List[Image.Image] = []
    quant_const = _get_pillow_quantize_constant(profile.quantize_method)

    for i, p in enumerate(frame_paths):
        if cancel_token and cancel_token.is_set():
            for f in processed_frames:
                f.close()
            if global_palette_im:
                global_palette_im.close()
            raise InterruptedError("Export operation cancelled by user token.")

        pct = 0.20 + (0.60 * (i + 1) / total_frames)
        report_progress(
            "Processing & Resizing" if out_fmt == ExportFormat.WEBP else "Quantizing",
            i + 1,
            total_frames,
            pct,
            f"Processing frame {i + 1}/{total_frames}...",
        )

        target_mode = "RGBA" if (out_fmt == ExportFormat.WEBP and profile.lossless) else "RGB"
        frame_im = load_and_normalize_frame(p, target_mode=target_mode, background_color=profile.background_color)
        harmonized = harmonize_frame(
            frame_im,
            target_size=target_dimensions,
            mode=profile.harmonize_mode,
            bg_color=profile.background_color,
        )
        frame_im.close()

        if out_fmt == ExportFormat.GIF:
            if global_palette_im is not None:
                quantized = harmonized.quantize(palette=global_palette_im, dither=dither_flag)
            else:
                try:
                    quantized = harmonized.quantize(
                        colors=profile.color_count,
                        method=quant_const,
                        dither=dither_flag,
                    )
                except Exception:
                    quantized = harmonized.quantize(
                        colors=profile.color_count,
                        method=Image.Quantize.MEDIANCUT,
                        dither=dither_flag,
                    )
            harmonized.close()
            processed_frames.append(quantized)
        else:
            processed_frames.append(harmonized)

    if not processed_frames:
        raise EmptySequenceError("No processed frames to export.")

    # 3. Save animation
    report_progress("Encoding", total_frames, total_frames, 0.90, f"Encoding {out_fmt.value.upper()} animation...")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    first_frame = processed_frames[0]
    rest_frames = processed_frames[1:]

    try:
        if out_fmt == ExportFormat.GIF:
            first_frame.save(
                output_path,
                save_all=True,
                append_images=rest_frames,
                duration=durations,
                loop=profile.loop,
                optimize=True,
                format="GIF",
            )
        else:  # WEBP
            first_frame.save(
                output_path,
                save_all=True,
                append_images=rest_frames,
                duration=durations,
                loop=profile.loop,
                lossless=profile.lossless,
                quality=profile.quality,
                method=profile.method,
                minimize_size=profile.minimize_size,
                format="WEBP",
            )
    finally:
        for f in processed_frames:
            try:
                f.close()
            except Exception:
                pass
        if global_palette_im:
            try:
                global_palette_im.close()
            except Exception:
                pass
        gc.collect()

    file_size = output_path.stat().st_size if output_path.exists() else 0
    report_progress("Finished", total_frames, total_frames, 1.0, f"Export complete ({file_size / (1024*1024):.2f} MB).")
    return file_size, target_dimensions


def export_sequence(
