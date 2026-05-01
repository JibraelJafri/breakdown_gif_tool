"""Synthetic Breakdown Render Asset Generator for animforge / breakdown-animator.

Generates deterministic, multi-camera, multi-stage 3D render breakdown image
sequences using Pillow for unit, integration, and E2E testing.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFilter, ImageFont


# ============================================================================
# Stage Definitions & Color Palettes
# ============================================================================

STAGES = [
    {
        "id": "01_Clay_AO",
        "name": "Clay / Ambient Occlusion",
        "bg_color": (210, 210, 215),
        "primary_color": (160, 160, 165),
        "secondary_color": (110, 110, 115),
        "wireframe": False,
        "lighting": "flat",
    },
    {
        "id": "02_Wireframe",
        "name": "Wireframe Overlay",
        "bg_color": (40, 44, 52),
        "primary_color": (70, 75, 85),
        "secondary_color": (0, 255, 200),
        "wireframe": True,
        "lighting": "flat",
    },
    {
        "id": "03_Albedo",
        "name": "Base Color / Albedo",
        "bg_color": (180, 190, 200),
        "primary_color": (220, 80, 60),
        "secondary_color": (60, 140, 220),
        "wireframe": False,
        "lighting": "flat",
    },
    {
        "id": "04_DirectLight",
        "name": "Direct & Specular Lighting",
        "bg_color": (20, 25, 35),
        "primary_color": (240, 120, 70),
        "secondary_color": (255, 230, 140),
        "wireframe": False,
        "lighting": "directional",
    },
    {
        "id": "05_Beauty_Comp",
        "name": "Final Composite / Beauty",
        "bg_color": (15, 18, 28),
        "primary_color": (255, 130, 80),
        "secondary_color": (100, 220, 255),
        "wireframe": False,
        "lighting": "beauty",
    },
]


# ============================================================================
# Synthetic Image Rendering Engine
# ============================================================================

def draw_synthetic_breakdown_frame(
    width: int,
    height: int,
    stage_idx: int,
    camera_name: str,
    frame_number: int,
    total_frames: int,
    format_mode: str = "RGB",
) -> Image.Image:
    """Renders a single synthetic 3D breakdown render frame with visual geometry."""
    stage = STAGES[stage_idx % len(STAGES)]
    bg_color = stage["bg_color"]
    primary_color = stage["primary_color"]
    secondary_color = stage["secondary_color"]
    stage_name = stage["name"]

    # Always draw in RGB mode first for consistent geometry and text rendering
    im = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(im)

    # Geometry center and bounds
    cx, cy = width // 2, height // 2
    radius = min(width, height) // 4

    # Camera perspective shift offset
    cam_offset_x = 0
    if "Camera_2" in camera_name or "CamB" in camera_name:
        cam_offset_x = int(radius * 0.4)
    elif "Camera_3" in camera_name or "CamC" in camera_name:
        cam_offset_x = -int(radius * 0.4)

    center_x = cx + cam_offset_x

    # Render simulated 3D geometry (Sphere / Cylinder / Box)
    if stage["lighting"] == "directional":
        # Multi-layer shading for direct lighting
        for r in range(radius, 0, -5):
            factor = r / radius
            shade = (
                int(primary_color[0] * (1.0 - factor * 0.5)),
                int(primary_color[1] * (1.0 - factor * 0.5)),
                int(primary_color[2] * (1.0 - factor * 0.5)),
            )
            draw.ellipse(
                [center_x - r, cy - r, center_x + r, cy + r],
                fill=shade,
            )
        # Specular highlight
        hl_x = center_x - radius // 3
        hl_y = cy - radius // 3
        hl_r = radius // 4
        draw.ellipse(
            [hl_x - hl_r, hl_y - hl_r, hl_x + hl_r, hl_y + hl_r],
            fill=(255, 255, 255),
        )
    elif stage["lighting"] == "beauty":
        # Rich layered gradient
        for r in range(radius, 0, -4):
            t = r / radius
            col = (
                int(primary_color[0] * (1.0 - t) + secondary_color[0] * t),
                int(primary_color[1] * (1.0 - t) + secondary_color[1] * t),
                int(primary_color[2] * (1.0 - t) + secondary_color[2] * t),
            )
            draw.ellipse(
                [center_x - r, cy - r, center_x + r, cy + r],
                fill=col,
            )
        # Rim light crescent
        draw.arc(
            [center_x - radius - 4, cy - radius - 4, center_x + radius + 4, cy + radius + 4],
            start=180,
            end=300,
            fill=(255, 255, 255),
            width=max(2, width // 200),
        )
    else:
        # Flat clay or albedo solid geometry
        draw.ellipse(
            [center_x - radius, cy - radius, center_x + radius, cy + radius],
            fill=primary_color,
            outline=secondary_color,
            width=2,
        )

    # Wireframe overlay
    if stage["wireframe"]:
        wire_color = secondary_color
        grid_step = max(16, min(width, height) // 16)
        # Latitude / Longitude lines
        for offset in range(-radius, radius, grid_step):
            if abs(offset) < radius:
                chord_half = int(math.sqrt(radius * radius - offset * offset))
                # Horizontal wire
                draw.line(
                    [(center_x - chord_half, cy + offset), (center_x + chord_half, cy + offset)],
                    fill=wire_color,
                    width=1,
                )
                # Vertical wire
                draw.line(
                    [(center_x + offset, cy - chord_half), (center_x + offset, cy + chord_half)],
                    fill=wire_color,
                    width=1,
                )
        # Outer wire circle
        draw.ellipse(
            [center_x - radius, cy - radius, center_x + radius, cy + radius],
            outline=wire_color,
            width=2,
        )

    # Top Header Banner
    banner_height = max(32, height // 14)
    draw.rectangle([0, 0, width, banner_height], fill=(20, 24, 30))

    # Text overlays
    title_text = f"{camera_name} | Stage {stage_idx + 1}/{total_frames}: {stage_name}"
    frame_text = f"Frame {frame_number:02d} | {width}x{height}"

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    draw.text((12, banner_height // 4), title_text, fill=(255, 255, 255), font=font)
    draw.text((12, height - banner_height // 2 - 12), frame_text, fill=(180, 190, 205), font=font)

    # Lower progress / step bar
    bar_width = width - 24
    bar_y = height - 10
    step_width = bar_width / max(1, total_frames)
    draw.rectangle([12, bar_y, width - 12, bar_y + 4], fill=(60, 65, 75))
    draw.rectangle(
        [12, bar_y, int(12 + (stage_idx + 1) * step_width), bar_y + 4],
        fill=(0, 200, 255),
    )

    if format_mode == "RGB":
        return im
    elif format_mode == "RGBA":
        return im.convert("RGBA")
    elif format_mode == "L":
        return im.convert("L")
    elif format_mode == "CMYK":
        return im.convert("CMYK")
    elif format_mode == "P":
        return im.quantize(colors=256)
    return im.convert(format_mode)


# ============================================================================
# Generator Functions for Test Suites & Fixtures
# ============================================================================

def create_natural_sort_sequence(
    output_dir: Path | str,
    count: int = 12,
    prefix: str = "frame",
    ext: str = "png",
    dimensions: Tuple[int, int] = (640, 360),
) -> List[Path]:
    """Generates sequence with unpadded numbers (frame_1 .. frame_12) to test natural sorting."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    created_paths = []

    for i in range(1, count + 1):
        filename = f"{prefix}_{i}.{ext}"
        filepath = out / filename
        stage_idx = (i - 1) % len(STAGES)
        im = draw_synthetic_breakdown_frame(
            width=dimensions[0],
            height=dimensions[1],
            stage_idx=stage_idx,
            camera_name="Default_Camera",
            frame_number=i,
            total_frames=count,
        )
        im.save(filepath)
        created_paths.append(filepath)

    return created_paths


def create_multi_camera_sequence(
    output_dir: Path | str,
    cameras: Tuple[str, ...] = ("Camera_1", "Camera_2"),
    stages_count: int = 5,
    dimensions: Tuple[int, int] = (640, 360),
) -> Dict[str, List[Path]]:
    """Generates multi-camera breakdown render sequences in the same folder."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results: Dict[str, List[Path]] = {cam: [] for cam in cameras}

    for cam in cameras:
        for idx in range(stages_count):
            stage = STAGES[idx % len(STAGES)]
            stage_slug = stage["id"]
            filename = f"Stage_{idx + 1:02d}_{stage_slug}_{cam}.png"
            filepath = out / filename
            im = draw_synthetic_breakdown_frame(
                width=dimensions[0],
                height=dimensions[1],
                stage_idx=idx,
                camera_name=cam,
                frame_number=idx + 1,
                total_frames=stages_count,
            )
            im.save(filepath)
            results[cam].append(filepath)

    return results


def create_mismatched_dimension_sequence(
    output_dir: Path | str,
) -> List[Path]:
    """Generates sequence with intentionally varying resolutions to test harmonization."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    configs = [
        ("Step_01_Clay_FHD.png", (1920, 1080), 0),
        ("Step_02_Wire_Square.png", (1080, 1080), 1),
        ("Step_03_Albedo_HD.png", (1280, 720), 2),
        ("Step_04_Light_Vertical.png", (800, 1200), 3),
        ("Step_05_Beauty_FHD.png", (1920, 1080), 4),
    ]

    paths = []
    for filename, (w, h), s_idx in configs:
        fp = out / filename
        im = draw_synthetic_breakdown_frame(
            width=w,
            height=h,
            stage_idx=s_idx,
            camera_name="Mismatched_Cam",
            frame_number=s_idx + 1,
            total_frames=len(configs),
        )
        im.save(fp)
        paths.append(fp)

    return paths


def create_mixed_extensions_sequence(
    output_dir: Path | str,
    dimensions: Tuple[int, int] = (640, 360),
) -> List[Path]:
    """Generates valid frames across diverse image extensions (.png, .jpg, .webp, .bmp)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    items = [
        ("pass_01.png", 0, "PNG"),
        ("pass_02.jpg", 1, "JPEG"),
        ("pass_03.webp", 2, "WEBP"),
        ("pass_04.bmp", 3, "BMP"),
        ("pass_05.jpeg", 4, "JPEG"),
    ]

    paths = []
    for filename, s_idx, fmt in items:
        fp = out / filename
        im = draw_synthetic_breakdown_frame(
            width=dimensions[0],
            height=dimensions[1],
            stage_idx=s_idx,
            camera_name="Mixed_Ext_Cam",
            frame_number=s_idx + 1,
            total_frames=len(items),
        )
        im.save(fp, format=fmt)
        paths.append(fp)

    return paths


def create_corrupt_asset_sequence(
    output_dir: Path | str,
    dimensions: Tuple[int, int] = (640, 360),
) -> List[Path]:
    """Generates valid frames mixed with 0-byte, truncated, and corrupt image files."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []

    # 1. Valid Frame 1
    fp1 = out / "frame_01_valid.png"
    im1 = draw_synthetic_breakdown_frame(dimensions[0], dimensions[1], 0, "Corrupt_Test", 1, 4)
    im1.save(fp1)
    paths.append(fp1)

    # 2. Corrupt 0-byte file
    fp_zero = out / "frame_02_empty.png"
    fp_zero.write_bytes(b"")
    paths.append(fp_zero)

    # 3. Truncated PNG header
    fp_trunc = out / "frame_03_truncated.png"
    fp_trunc.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00")
    paths.append(fp_trunc)

    # 4. Garbage text file with image extension
    fp_garbage = out / "frame_04_garbage.jpg"
    fp_garbage.write_text("This is not an image file. Corrupted header content.")
    paths.append(fp_garbage)

    # 5. Valid Frame 5
    fp5 = out / "frame_05_valid.png"
    im5 = draw_synthetic_breakdown_frame(dimensions[0], dimensions[1], 4, "Corrupt_Test", 5, 4)
    im5.save(fp5)
    paths.append(fp5)

    return paths


def create_exclusion_test_assets(
    output_dir: Path | str,
    dimensions: Tuple[int, int] = (480, 270),
) -> List[Path]:
    """Generates valid frames alongside OS junk, existing GIF, and animated WebP files."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []

    # Valid PNG frames
    for i in range(1, 4):
        fp = out / f"valid_render_{i:02d}.png"
        im = draw_synthetic_breakdown_frame(dimensions[0], dimensions[1], i - 1, "Exclusion_Cam", i, 3)
        im.save(fp)
        paths.append(fp)

    # OS System junk files
    ds_store = out / ".DS_Store"
    ds_store.write_bytes(b"\x00\x00\x00\x01Bud1\x00\x00\x10\x00")
    paths.append(ds_store)

    thumbs_db = out / "Thumbs.db"
    thumbs_db.write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1")
    paths.append(thumbs_db)

    desktop_ini = out / "desktop.ini"
    desktop_ini.write_text("[.ShellClassInfo]\nIconResource=icon.ico,0\n")
    paths.append(desktop_ini)

    hidden_dot = out / ".hidden_frame_01.png"
    im_dot = Image.new("RGB", (100, 100), (0, 0, 0))
    im_dot.save(hidden_dot)
    paths.append(hidden_dot)

    # Existing pre-rendered GIF animation
    gif_path = out / "existing_animation_output.gif"
    f1 = Image.new("RGB", (100, 100), (255, 0, 0))
    f2 = Image.new("RGB", (100, 100), (0, 255, 0))
    f1.save(gif_path, format="GIF", save_all=True, append_images=[f2], duration=100, loop=0)
    paths.append(gif_path)

    # Existing animated WebP
    webp_anim_path = out / "existing_breakdown.webp"
    w1 = Image.new("RGB", (100, 100), (0, 0, 255))
    w2 = Image.new("RGB", (100, 100), (255, 255, 0))
    w1.save(webp_anim_path, format="WEBP", save_all=True, append_images=[w2], duration=100, loop=0)
    paths.append(webp_anim_path)

    return paths


def create_color_mode_sequence(
    output_dir: Path | str,
    dimensions: Tuple[int, int] = (480, 270),
) -> List[Path]:
    """Generates frames in RGB, RGBA, L (Grayscale), P (Palette), and CMYK modes."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []

    # 1. RGB
    fp_rgb = out / "mode_01_rgb.png"
    draw_synthetic_breakdown_frame(dimensions[0], dimensions[1], 0, "Mode_Cam", 1, 5, "RGB").save(fp_rgb)
    paths.append(fp_rgb)

    # 2. RGBA
    fp_rgba = out / "mode_02_rgba.png"
    draw_synthetic_breakdown_frame(dimensions[0], dimensions[1], 1, "Mode_Cam", 2, 5, "RGBA").save(fp_rgba)
    paths.append(fp_rgba)

    # 3. L (Grayscale)
    fp_l = out / "mode_03_grayscale.png"
    draw_synthetic_breakdown_frame(dimensions[0], dimensions[1], 2, "Mode_Cam", 3, 5, "L").save(fp_l)
    paths.append(fp_l)

    # 4. P (Palette)
    fp_p = out / "mode_04_palette.png"
    base_p = draw_synthetic_breakdown_frame(dimensions[0], dimensions[1], 3, "Mode_Cam", 4, 5, "RGB")
    base_p.quantize(colors=64).save(fp_p)
    paths.append(fp_p)

    # 5. CMYK (saved as TIFF or JPEG)
    fp_cmyk = out / "mode_05_cmyk.jpg"
    base_cmyk = draw_synthetic_breakdown_frame(dimensions[0], dimensions[1], 4, "Mode_Cam", 5, 5, "RGB")
    base_cmyk.convert("CMYK").save(fp_cmyk, format="JPEG")
    paths.append(fp_cmyk)

    return paths


# ============================================================================
# Master Fixture Generator
# ============================================================================

def generate_all_fixtures(base_dir: Path | str, verbose: bool = True) -> Dict[str, Path]:
    """Generates all synthetic fixture datasets in the target base directory."""
    root = Path(base_dir)
    root.mkdir(parents=True, exist_ok=True)

    fixtures = {
        "natural_sort": root / "natural_sort",
        "multi_camera": root / "multi_camera",
        "mismatched_dimensions": root / "mismatched_dimensions",
        "mixed_extensions": root / "mixed_extensions",
        "corrupted_files": root / "corrupted_files",
        "exclusion_test": root / "exclusion_test",
        "color_modes": root / "color_modes",
        "single_frame": root / "single_frame",
        "empty_folder": root / "empty_folder",
    }

    if verbose:
        print(f"[*] Generating test assets in: {root.resolve()}")

    # 1. Natural Sort Sequence (frame_1 .. frame_12)
    create_natural_sort_sequence(fixtures["natural_sort"], count=12)

    # 2. Multi-Camera Sequence (Camera_1, Camera_2, Camera_3 across 5 stages)
    create_multi_camera_sequence(
        fixtures["multi_camera"],
        cameras=("Camera_1", "Camera_2", "Camera_3"),
        stages_count=5,
    )

    # 3. Mismatched Dimensions
    create_mismatched_dimension_sequence(fixtures["mismatched_dimensions"])

    # 4. Mixed Extensions (.png, .jpg, .webp, .bmp)
    create_mixed_extensions_sequence(fixtures["mixed_extensions"])

    # 5. Corrupted Files (empty, truncated, garbage)
    create_corrupt_asset_sequence(fixtures["corrupted_files"])

    # 6. Exclusion Test (.DS_Store, Thumbs.db, existing .gif, animated .webp)
    create_exclusion_test_assets(fixtures["exclusion_test"])

    # 7. Color Modes (RGB, RGBA, L, P, CMYK)
    create_color_mode_sequence(fixtures["color_modes"])

    # 8. Single Frame
    single_dir = fixtures["single_frame"]
    single_dir.mkdir(parents=True, exist_ok=True)
    draw_synthetic_breakdown_frame(640, 360, 0, "SingleCam", 1, 1).save(single_dir / "single_frame_01.png")

    # 9. Empty Folder
    fixtures["empty_folder"].mkdir(parents=True, exist_ok=True)

    if verbose:
        print("[+] All synthetic breakdown test fixtures generated successfully.")

    return fixtures


# ============================================================================
# CLI Runner
# ============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="AnimForge Synthetic Test Asset Generator for breakdown-animator."
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="tests/fixtures",
        help="Destination directory for synthetic fixtures (default: tests/fixtures)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing output directory before generating",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress console progress output",
    )

    args = parser.parse_args()
    target_dir = Path(args.output_dir)

    if args.clean and target_dir.exists():
        shutil.rmtree(target_dir)

    generate_all_fixtures(target_dir, verbose=not args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
