"""tests/test_adversarial_m1_challenger2.py
=========================================
Empirical Adversarial Verification Suite for Milestone M1.
Challenger: challenger_m1_2

Targeting:
1. Corrupt files (0-byte, truncated bytes, random binary data, renamed text files, header corruption, partial rasters).
2. Animated vs static WebP differentiation (VP8, VP8L, VP8X static with alpha/EXIF/ICC, VP8X animated, corrupted WebP).
3. Exclusion of OS junk (.DS_Store, Thumbs.db, desktop.ini, ._*, .directory, ~$lock, non-image extensions).
4. Memory benchmarking on 1,000 synthetic image files using tracemalloc (peak RAM < 5.0 MB).
5. Complex Unicode filenames, symbol-heavy tags, mixed color modes (CMYK, 1, L, LA, P, I, F), and dimension calculations.
"""

import gc
import os
import struct
import tempfile
import tracemalloc
from pathlib import Path
from typing import List

import pytest
from PIL import Image

from breakdown_animator.sequence_engine import (
    CorruptImageError,
    DirectoryNotFoundError,
    EmptySequenceError,
    FrameInfo,
    SequenceDiscoveryError,
    SequenceEngineError,
    SequenceGroup,
    analyze_sequence,
    extract_group_tag,
    group_sequences,
    inspect_frame,
    is_animated_image,
    is_valid_image_file,
    load_and_normalize_frame,
    natural_sort_paths,
    normalize_frame_mode,
    scan_directory,
)


class TestAdversarialCorruption:
    """Adversarial testing for all forms of corrupted, truncated, and malformed files."""

    def test_zero_byte_files(self, tmp_path: Path):
        """0-byte files in various extensions should be detected as corrupt."""
        corrupt_files = [
            tmp_path / "zero_001.png",
            tmp_path / "zero_002.jpg",
            tmp_path / "zero_003.webp",
            tmp_path / "zero_004.tga",
            tmp_path / "zero_005.bmp",
            tmp_path / "zero_006.tif",
        ]
        for f in corrupt_files:
            f.touch()  # Creates 0-byte file

        # inspect_frame must raise CorruptImageError
        for f in corrupt_files:
            with pytest.raises(CorruptImageError) as exc_info:
                inspect_frame(f)
            assert "empty" in str(exc_info.value).lower() or "corrupt" in str(exc_info.value).lower()

        # load_and_normalize_frame must raise CorruptImageError
        for f in corrupt_files:
            with pytest.raises(CorruptImageError):
                load_and_normalize_frame(f)

        # analyze_sequence with skip_corrupted=False must raise CorruptImageError
        with pytest.raises(CorruptImageError):
            analyze_sequence(corrupt_files, skip_corrupted=False)

        # analyze_sequence with skip_corrupted=True on only-corrupted files raises EmptySequenceError
        with pytest.raises(EmptySequenceError):
            analyze_sequence(corrupt_files, skip_corrupted=True)

        # group_sequences returns empty dict when all are corrupt
        groups = group_sequences(corrupt_files, skip_corrupted=True)
        assert groups == {}

    def test_truncated_headers(self, tmp_path: Path):
        """Files with truncated image magic bytes (e.g. 4 bytes of PNG or JPEG)."""
        trunc_png = tmp_path / "trunc_001.png"
        trunc_png.write_bytes(b"\x89PNG\r\n\x1a\n")  # Just header, no IHDR chunk

        trunc_jpg = tmp_path / "trunc_002.jpg"
        trunc_jpg.write_bytes(b"\xff\xd8\xff\xe0")  # Just SOI marker

        trunc_bmp = tmp_path / "trunc_003.bmp"
        trunc_bmp.write_bytes(b"BM\x00\x00")  # Truncated BMP header

        trunc_webp = tmp_path / "trunc_004.webp"
        trunc_webp.write_bytes(b"RIFF\x10\x00\x00\x00WEBPVP8 ")  # Truncated WebP header

        for f in [trunc_png, trunc_jpg, trunc_bmp, trunc_webp]:
            with pytest.raises(CorruptImageError):
                inspect_frame(f)
            with pytest.raises(CorruptImageError):
                load_and_normalize_frame(f)

    def test_random_binary_data(self, tmp_path: Path):
        """Arbitrary pseudorandom binary junk renamed as images."""
        random_files = []
        for i, ext in enumerate([".png", ".jpg", ".jpeg", ".webp", ".tga", ".bmp", ".tif", ".tiff"]):
            f = tmp_path / f"random_{i:03d}{ext}"
            f.write_bytes(os.urandom(1024))
            random_files.append(f)

        for f in random_files:
            with pytest.raises(CorruptImageError):
                inspect_frame(f)
            with pytest.raises(CorruptImageError):
                load_and_normalize_frame(f)

    def test_renamed_non_image_text_and_code(self, tmp_path: Path):
        """Text, JSON, Python, HTML, PDF content renamed with image extensions."""
        bad_files = {
            "plain_text.png": b"Hello world, this is just plain text content!",
            "fake_json.jpg": b'{"status": "error", "message": "not an image"}',
            "fake_html.webp": b"<!DOCTYPE html><html><body><h1>Fake</h1></body></html>",
            "fake_pdf.tif": b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj",
            "fake_exe.bmp": b"MZ\x90\x00\x03\x00\x00\x00\x04\x00\x00\x00\xff\xff",
        }
        for name, content in bad_files.items():
            f = tmp_path / name
            f.write_bytes(content)

            with pytest.raises(CorruptImageError):
                inspect_frame(f)
            with pytest.raises(CorruptImageError):
                load_and_normalize_frame(f)

    def test_partial_raster_corruption_in_stream(self, tmp_path: Path):
        """Image with valid header structure but corrupted/truncated payload."""
        valid_img = Image.new("RGB", (64, 64), color=(255, 0, 0))
        valid_path = tmp_path / "valid.png"
        valid_img.save(valid_path, format="PNG")

        raw_bytes = valid_path.read_bytes()
        # Truncate halfway through the image data
        corrupt_path = tmp_path / "half_truncated.png"
        corrupt_path.write_bytes(raw_bytes[: len(raw_bytes) // 2])

        try:
            frame_info = inspect_frame(corrupt_path)
            assert frame_info.width == 64
            with pytest.raises(CorruptImageError):
                load_and_normalize_frame(corrupt_path)
        except CorruptImageError:
            pass

    def test_mixed_valid_and_corrupt_sequence_filtering(self, tmp_path: Path):
        """Sequence with interleaved valid and corrupt frames."""
        valid_files = []
        for i in range(1, 6):
            p = tmp_path / f"Frame_{i:02d}.png"
            Image.new("RGB", (100, 100), color=(i * 20, 50, 100)).save(p, format="PNG")
            valid_files.append(p)

        # Inject corrupt frames
        corrupt_1 = tmp_path / "Frame_02_corrupt.png"
        corrupt_1.write_bytes(b"BAD DATA NOT AN IMAGE")

        corrupt_2 = tmp_path / "Frame_04_corrupt.png"
        corrupt_2.touch()  # 0-byte

        all_files = scan_directory(tmp_path)
        assert len(all_files) == 7

        group = analyze_sequence(all_files, group_id="mixed", skip_corrupted=True)
        assert group.total_frames == 5
        assert [f.filename for f in group.frames] == [
            "Frame_01.png",
            "Frame_02.png",
            "Frame_03.png",
            "Frame_04.png",
            "Frame_05.png",
        ]

        with pytest.raises(CorruptImageError):
            analyze_sequence(all_files, group_id="mixed", skip_corrupted=False)


class TestAdversarialWebPDifferentiation:
    """Adversarial tests for static vs animated WebP detection."""

    def test_static_webp_vp8_lossy(self, tmp_path: Path):
        """Static lossy VP8 WebP should NOT be flagged as animated."""
        img = Image.new("RGB", (128, 128), color=(100, 150, 200))
        p = tmp_path / "static_lossy.webp"
        img.save(p, format="WEBP", lossless=False, quality=80)

        assert is_animated_image(p) is False
        assert is_valid_image_file(p) is True

        frame = inspect_frame(p)
        assert frame.width == 128
        assert frame.height == 128

    def test_static_webp_vp8l_lossless(self, tmp_path: Path):
        """Static lossless VP8L WebP should NOT be flagged as animated."""
        img = Image.new("RGBA", (128, 128), color=(100, 150, 200, 255))
        p = tmp_path / "static_lossless.webp"
        img.save(p, format="WEBP", lossless=True)

        assert is_animated_image(p) is False
        assert is_valid_image_file(p) is True

    def test_static_webp_vp8x_extended_with_alpha_and_metadata(self, tmp_path: Path):
        """Static VP8X WebP (with alpha / ICC / EXIF but no animation) should NOT be flagged."""
        img = Image.new("RGBA", (64, 64), color=(50, 100, 150, 128))
        p = tmp_path / "static_vp8x.webp"
        img.save(p, format="WEBP", lossless=False)

        assert is_animated_image(p) is False
        assert is_valid_image_file(p) is True

    def test_animated_webp_multi_frame(self, tmp_path: Path):
        """Multi-frame animated WebP MUST be flagged as animated and excluded."""
        frame1 = Image.new("RGB", (64, 64), color=(255, 0, 0))
        frame2 = Image.new("RGB", (64, 64), color=(0, 255, 0))
        frame3 = Image.new("RGB", (64, 64), color=(0, 0, 255))

        anim_p = tmp_path / "animated_render.webp"
        frame1.save(
            anim_p,
            format="WEBP",
            save_all=True,
            append_images=[frame2, frame3],
            duration=100,
            loop=0,
        )

        assert is_animated_image(anim_p) is True
        assert is_valid_image_file(anim_p) is False

        files = scan_directory(tmp_path)
        assert anim_p not in files

    def test_manually_crafted_vp8x_animation_flag(self, tmp_path: Path):
        """Craft exact byte-level WebP VP8X header with animation flag (bit 1 = 0x02)."""
        crafted_anim = tmp_path / "crafted_anim.webp"
        flags = 0x02  # animation bit set
        canvas_w = 100 - 1
        canvas_h = 100 - 1
        vp8x_chunk = b"VP8X" + struct.pack("<I", 10) + bytes([flags, 0, 0, 0]) + canvas_w.to_bytes(3, "little") + canvas_h.to_bytes(3, "little")
        riff_payload = b"WEBP" + vp8x_chunk
        riff_header = b"RIFF" + struct.pack("<I", len(riff_payload)) + riff_payload
        crafted_anim.write_bytes(riff_header)

        assert is_animated_image(crafted_anim) is True
        assert is_valid_image_file(crafted_anim) is False

    def test_manually_crafted_vp8x_non_animation_flag(self, tmp_path: Path):
        """Craft exact byte-level WebP VP8X header with alpha+icc flags (0x10 | 0x20 = 0x30) but NO animation."""
        crafted_static = tmp_path / "crafted_static.webp"
        flags = 0x30  # Alpha + ICC, NO animation bit
        canvas_w = 100 - 1
        canvas_h = 100 - 1
        vp8x_chunk = b"VP8X" + struct.pack("<I", 10) + bytes([flags, 0, 0, 0]) + canvas_w.to_bytes(3, "little") + canvas_h.to_bytes(3, "little")
        riff_payload = b"WEBP" + vp8x_chunk
        riff_header = b"RIFF" + struct.pack("<I", len(riff_payload)) + riff_payload
        crafted_static.write_bytes(riff_header)

        assert is_animated_image(crafted_static) is False


class TestAdversarialOSJunkAndExclusions:
    """Adversarial tests for OS metadata, system lock files, and unsupported extensions."""

    def test_all_os_junk_filenames_and_variations(self, tmp_path: Path):
        """Check all variations of OS junk filenames and casing."""
        junk_names = [
            ".DS_Store",
            ".ds_store",
            ".DS_STORE",
            "Thumbs.db",
            "thumbs.db",
            "THUMBS.DB",
            "desktop.ini",
            "Desktop.ini",
            "DESKTOP.INI",
            ".directory",
            "ehthumbs.db",
            "ehthumbs_vista.db",
            "._DS_Store",
            "._Thumbs.db",
            "._render_001.png",
            "._camera_1.jpg",
            ".hidden_image.png",
            ".dotfile.webp",
            "~$lock_file.png",
            "~$presentation.jpg",
        ]

        for name in junk_names:
            p = tmp_path / name
            p.write_bytes(b"dummy metadata content")
            assert is_valid_image_file(p) is False, f"Failed to reject OS junk: {name}"

        assert scan_directory(tmp_path) == []

    def test_non_image_extensions_rejected(self, tmp_path: Path):
        """Unsupported and output animation extensions must be rejected."""
        unsupported = [
            "animation.gif",
            "output.GIF",
            "video.mp4",
            "render.mov",
            "clip.avi",
            "scene.blend",
            "model.obj",
            "mesh.fbx",
            "data.json",
            "notes.txt",
            "backup.bak",
            "temp.tmp",
            "script.py",
            "document.pdf",
            "archive.zip",
        ]

        for name in unsupported:
            p = tmp_path / name
            p.write_bytes(b"test non-image payload")
            assert is_valid_image_file(p) is False, f"Failed to reject unsupported file: {name}"

        assert scan_directory(tmp_path) == []

    def test_supported_extensions_case_insensitivity(self, tmp_path: Path):
        """Valid image extensions in all casing permutations must be accepted."""
        valid_names = [
            "frame_01.png",
            "frame_02.PNG",
            "frame_03.Png",
            "frame_04.jpg",
            "frame_05.JPG",
            "frame_06.jpeg",
            "frame_07.JPEG",
            "frame_08.tga",
            "frame_09.TGA",
            "frame_10.bmp",
            "frame_11.BMP",
            "frame_12.tif",
            "frame_13.TIF",
            "frame_14.tiff",
            "frame_15.TIFF",
        ]

        for name in valid_names:
            p = tmp_path / name
            img = Image.new("RGB", (10, 10), color=(255, 0, 0))
            if name.lower().endswith((".jpg", ".jpeg")):
                img.save(p, format="JPEG")
            elif name.lower().endswith((".tif", ".tiff")):
                img.save(p, format="TIFF")
            elif name.lower().endswith(".bmp"):
                img.save(p, format="BMP")
            elif name.lower().endswith(".tga"):
                img.save(p, format="TGA")
            else:
                img.save(p, format="PNG")

            assert is_valid_image_file(p) is True, f"Failed to accept valid image: {name}"

        scanned = scan_directory(tmp_path)
        assert len(scanned) == len(valid_names)


class TestAdversarialComplexFilenamesAndColorModes:
    """Stress tests on complex symbols, Unicode names, and all PIL color mode normalizations."""

    def test_unicode_and_special_char_filenames(self, tmp_path: Path):
        """Filenames with spaces, brackets, Unicode emojis, hashes, and dashes."""
        filenames = [
            "Render [Cam #1] (4K UHD) - Step 01.png",
            "Render [Cam #1] (4K UHD) - Step 02.png",
            "Render [Cam #2] (4K UHD) - Step 01.png",
            "Render [Cam #2] (4K UHD) - Step 02.png",
            "✨_Hero_Clay_Pass_001.png",
            "✨_Hero_Clay_Pass_002.png",
        ]

        for fn in filenames:
            p = tmp_path / fn
            Image.new("RGB", (32, 32), color=(10, 20, 30)).save(p, format="PNG")

        scanned = scan_directory(tmp_path)
        assert len(scanned) == 6

        groups = group_sequences(scanned, skip_corrupted=True)
        assert len(groups) >= 2
        for gid, grp in groups.items():
            assert grp.total_frames > 0
            assert grp.common_width == 32

    def test_exhaustive_color_mode_normalization(self):
        """Stress-test normalization from 1, L, LA, P, RGB, RGBA, CMYK, I, F to RGB and RGBA."""
        # Create test images in various modes
        modes_to_test = {
            "1": Image.new("1", (16, 16), color=1),
            "L": Image.new("L", (16, 16), color=128),
            "LA": Image.new("LA", (16, 16), color=(128, 200)),
            "P": Image.new("P", (16, 16)),
            "RGB": Image.new("RGB", (16, 16), color=(255, 100, 50)),
            "RGBA": Image.new("RGBA", (16, 16), color=(255, 100, 50, 180)),
            "CMYK": Image.new("CMYK", (16, 16), color=(100, 50, 0, 20)),
            "I": Image.new("I", (16, 16), color=65535),
            "F": Image.new("F", (16, 16), color=1.0),
        }

        # Normalize to RGBA
        for mode_name, src_img in modes_to_test.items():
            norm_rgba = normalize_frame_mode(src_img, target_mode="RGBA")
            assert norm_rgba.mode == "RGBA", f"Failed RGBA normalization for mode {mode_name}"
            assert norm_rgba.size == (16, 16)

        # Normalize to RGB with background matting
        for mode_name, src_img in modes_to_test.items():
            norm_rgb = normalize_frame_mode(src_img, target_mode="RGB", background_color=(50, 50, 50))
            assert norm_rgb.mode == "RGB", f"Failed RGB normalization for mode {mode_name}"
            assert norm_rgb.size == (16, 16)

        # Invalid target mode raises ValueError
        with pytest.raises(ValueError):
            normalize_frame_mode(modes_to_test["RGB"], target_mode="HSV")


class TestAdversarialMemoryAndPerformance:
    """Stress testing memory footprint with 1,000 synthetic image files."""

    def test_tracemalloc_1000_images_peak_ram_under_5mb(self, tmp_path: Path):
        """
        Verify that scanning and extracting metadata for 1,000 synthetic images
        strictly consumes < 5.0 MB peak RAM.
        """
        gc.collect()
        num_images = 1000

        # Pre-generate 1,000 minimal images across 2 camera sequences
        created_paths: List[Path] = []
        base_img = Image.new("RGB", (64, 64), color=(128, 64, 32))

        temp_sample = tmp_path / "temp_sample.png"
        base_img.save(temp_sample, format="PNG")
        sample_png_bytes = temp_sample.read_bytes()
        temp_sample.unlink()

        for i in range(1, (num_images // 2) + 1):
            p1 = tmp_path / f"Stage_01_Cam_Front_{i:04d}.png"
            p1.write_bytes(sample_png_bytes)
            created_paths.append(p1)

            p2 = tmp_path / f"Stage_01_Cam_Persp_{i:04d}.png"
            p2.write_bytes(sample_png_bytes)
            created_paths.append(p2)

        assert len(created_paths) == num_images

        # Now benchmark memory during discovery, sorting, clustering, and metadata inspection
        gc.collect()
        tracemalloc.start()
        tracemalloc.reset_peak()

        try:
            # 1. Directory scan
            scanned_paths = scan_directory(tmp_path)
            assert len(scanned_paths) == num_images

            # 2. Sequence grouping and metadata analysis
            groups = group_sequences(scanned_paths, skip_corrupted=True)
            assert len(groups) == 2
            assert "Cam_Front" in groups
            assert "Cam_Persp" in groups
            assert groups["Cam_Front"].total_frames == 500
            assert groups["Cam_Persp"].total_frames == 500

            # 3. Direct analyze_sequence
            seq_group = analyze_sequence(scanned_paths, group_id="full_1000", skip_corrupted=True)
            assert seq_group.total_frames == 1000
            assert seq_group.common_width == 64
            assert seq_group.common_height == 64

            current_bytes, peak_bytes = tracemalloc.get_traced_memory()
            peak_mb = peak_bytes / (1024.0 * 1024.0)
            current_mb = current_bytes / (1024.0 * 1024.0)

            print(f"\n[BENCHMARK] 1,000 synthetic frames metadata scan:")
            print(f"  Current Memory: {current_mb:.3f} MB ({current_bytes} bytes)")
            print(f"  Peak Memory:    {peak_mb:.3f} MB ({peak_bytes} bytes)")
            print(f"  Budget Limit:   5.000 MB")

            # Strict assertion: Peak RAM must stay < 5.0 MB
            assert peak_mb < 5.0, f"Peak memory {peak_mb:.3f} MB exceeded 5.0 MB limit!"
        finally:
            tracemalloc.stop()

    def test_natural_sort_ordering_adversarial_matrix(self):
        """Stress-test natural sort algorithm on complex edge-case filenames."""
        unsorted = [
            "file_100.png",
            "file_1.png",
            "file_01.png",
            "file_001.png",
            "file_2.png",
            "file_10.png",
            "file_20.png",
            "file_2a.png",
            "file_2b.png",
            "file_10a.png",
            "file_10b.png",
            "step_9_pass_2.png",
            "step_10_pass_1.png",
            "step_10_pass_2.png",
            "step_9_pass_10.png",
            "step_9_pass_1.png",
        ]

        sorted_paths = natural_sort_paths(unsorted)
        sorted_names = [p.name for p in sorted_paths]

        # Key assertions
        assert sorted_names.index("file_1.png") < sorted_names.index("file_2.png")
        assert sorted_names.index("file_2.png") < sorted_names.index("file_10.png")
        assert sorted_names.index("file_10.png") < sorted_names.index("file_20.png")
        assert sorted_names.index("file_20.png") < sorted_names.index("file_100.png")
        assert sorted_names.index("file_2a.png") < sorted_names.index("file_2b.png")
        assert sorted_names.index("file_2b.png") < sorted_names.index("file_10a.png")

        # Multi-segment checks
        assert sorted_names.index("step_9_pass_1.png") < sorted_names.index("step_9_pass_2.png")
        assert sorted_names.index("step_9_pass_2.png") < sorted_names.index("step_9_pass_10.png")
        assert sorted_names.index("step_9_pass_10.png") < sorted_names.index("step_10_pass_1.png")
        assert sorted_names.index("step_10_pass_1.png") < sorted_names.index("step_10_pass_2.png")
