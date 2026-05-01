"""breakdown_animator.sequence_engine
=================================
Core sequence engine for discovering, filtering, naturally sorting,
clustering, and extracting metadata from 3D render breakdown image sequences.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union
from PIL import Image, UnidentifiedImageError

# ============================================================================
# Supported Formats & Exclusions
# ============================================================================

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({
    ".png",
    ".jpg",
    ".jpeg",
    ".tga",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
})

IGNORED_FILENAMES: frozenset[str] = frozenset({
    ".ds_store",
    "thumbs.db",
    "desktop.ini",
    ".directory",
    "ehthumbs.db",
    "ehthumbs_vista.db",
})

IGNORED_EXTENSIONS: frozenset[str] = frozenset({
    ".gif",
    ".tmp",
    ".bak",
    ".mp4",
    ".mov",
    ".avi",
    ".txt",
    ".json",
    ".blend",
    ".obj",
    ".fbx",
})

IGNORED_DIRNAMES: frozenset[str] = frozenset({
    "output",
    "out",
    "exports",
    "export",
    ".git",
    ".idea",
    ".vscode",
    "__pycache__",
    ".pytest_cache",
    ".agents",
    ".gemini",
    "node_modules",
})


# ============================================================================
# Exception Hierarchy
# ============================================================================

class SequenceEngineError(Exception):
    """Base exception for all sequence engine operations."""
    pass


class SequenceDiscoveryError(SequenceEngineError):
    """Raised when a directory cannot be accessed, does not exist, or permissions are denied."""
    pass


class DirectoryNotFoundError(SequenceDiscoveryError, FileNotFoundError):
    """Raised when a specified directory does not exist on disk."""
    pass


class EmptySequenceError(SequenceEngineError, ValueError):
    """Raised when an operation is attempted on a sequence with zero valid frames."""
    pass


class CorruptImageError(SequenceEngineError):
    """Raised when an image file is corrupted, truncated, or unreadable by Pillow."""
    def __init__(self, path: Union[Path, str], reason: str = "Unreadable or corrupt image file"):
        self.path = Path(path)
        self.reason = reason
        super().__init__(f"Corrupt image at '{self.path}': {self.reason}")


class DimensionMismatchError(SequenceEngineError):
    """Raised when frames in a sequence have mismatched dimensions under strict validation."""
    pass


# ============================================================================
# Dataclasses
# ============================================================================

@dataclass(frozen=True)
class FrameInfo:
    """Represents metadata for an individual image frame in a breakdown sequence."""
    path: Path
    index: int
    filename: str
    width: int
    height: int
    mode: str
    file_size_bytes: int

    @property
    def aspect_ratio(self) -> float:
        """Returns numerical aspect ratio (width / height)."""
        return self.width / self.height if self.height > 0 else 0.0

    @property
    def dimensions(self) -> Tuple[int, int]:
        """Returns tuple of (width, height)."""
        return (self.width, self.height)

    @property
    def aspect_ratio_str(self) -> str:
        """Returns standard simplified aspect ratio string (e.g. '16:9', '1:1', '4:3')."""
        if self.height <= 0 or self.width <= 0:
            return "unknown"
        ratio = self.width / self.height
        if abs(ratio - 16 / 9) < 0.02:
            return "16:9"
        elif abs(ratio - 1.0) < 0.02:
            return "1:1"
        elif abs(ratio - 4 / 3) < 0.02:
            return "4:3"
        elif abs(ratio - 21 / 9) < 0.03:
            return "21:9"
        elif abs(ratio - 3 / 2) < 0.02:
            return "3:2"
        elif abs(ratio - 9 / 16) < 0.02:
            return "9:16"
        g = math.gcd(self.width, self.height)
        if g > 1:
            return f"{self.width // g}:{self.height // g}"
        return f"{self.width}:{self.height}"

    @property
    def file_size_mb(self) -> float:
        """Returns file size in Megabytes."""
        return self.file_size_bytes / (1024.0 * 1024.0)


@dataclass
class SequenceGroup:
    """Represents a clustered, ordered group of breakdown render frames."""
    group_id: str
    camera_name: Optional[str]
    frames: List[FrameInfo]
    total_frames: int
    common_width: int
    common_height: int
    has_mismatched_dimensions: bool
    estimated_memory_bytes: int

    @property
    def aspect_ratio(self) -> float:
        """Returns the canonical numerical aspect ratio."""
        return self.common_width / self.common_height if self.common_height > 0 else 0.0

    @property
    def aspect_ratio_str(self) -> str:
        """Returns aspect ratio string of canonical resolution."""
        if self.common_height <= 0 or self.common_width <= 0:
            return "unknown"
        ratio = self.common_width / self.common_height
        if abs(ratio - 16 / 9) < 0.02:
            return "16:9"
        elif abs(ratio - 1.0) < 0.02:
            return "1:1"
        elif abs(ratio - 4 / 3) < 0.02:
            return "4:3"
        elif abs(ratio - 21 / 9) < 0.03:
            return "21:9"
        elif abs(ratio - 3 / 2) < 0.02:
            return "3:2"
        elif abs(ratio - 9 / 16) < 0.02:
            return "9:16"
        g = math.gcd(self.common_width, self.common_height)
        if g > 1:
            return f"{self.common_width // g}:{self.common_height // g}"
        return f"{self.common_width}:{self.common_height}"

    @property
    def estimated_memory_mb(self) -> float:
        """Returns estimated uncompressed RGBA memory footprint in Megabytes."""
        return self.estimated_memory_bytes / (1024.0 * 1024.0)

    @property
    def frame_paths(self) -> List[Path]:
        """Returns an ordered list of file paths for all frames in sequence."""
        return [f.path for f in self.frames]

    @property
    def dimensions_summary(self) -> str:
        """Returns formatted string summarizing resolution and dimension status."""
        if self.has_mismatched_dimensions:
            return f"{self.common_width}x{self.common_height} (Mixed Resolutions)"
        return f"{self.common_width}x{self.common_height}"


# ============================================================================

# ============================================================================
# Image Format & System Junk Filtering
# ============================================================================

def is_animated_image(path: Union[str, Path]) -> bool:
    """
    Performs a fast, non-destructive header probe to check if an image file
    (e.g., animated WebP or GIF) contains multi-frame animation.
    """
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".gif":
        return True
    if ext == ".webp":
        try:
            with open(p, "rb") as f:
                header = f.read(32)
                if len(header) >= 21 and header[:4] == b"RIFF" and header[8:12] == b"WEBP":
                    chunk_type = header[12:16]
                    if chunk_type == b"VP8X":
                        flags = header[20]
                        # Bit 1 (0x02) at byte offset 20 indicates animation in VP8X
                        return bool(flags & 0x02)
                    elif chunk_type in (b"VP8 ", b"VP8L"):
                        return False
        except Exception:
            pass
        try:
            with Image.open(p) as im:
                return bool(getattr(im, "is_animated", False) and getattr(im, "n_frames", 1) > 1)
        except Exception:
            return False
    return False


def is_valid_image_file(path: Union[str, Path]) -> bool:
    """
    Determines whether a file is a supported static image format,
    filtering out OS metadata, hidden files, output GIFs, and animated WebPs.
    """
    p = Path(path)
    name_lower = p.name.lower()
    if name_lower in IGNORED_FILENAMES:
        return False
    if name_lower.startswith(".") or name_lower.startswith("._") or name_lower.startswith("~$"):
        return False
    ext = p.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS or ext in IGNORED_EXTENSIONS:
        return False
    if is_animated_image(p):
        return False
    return True


def scan_directory(directory_path: Union[str, Path], recursive: bool = False) -> List[Path]:
    """
    Scans a directory and returns a list of valid, filtered source image paths,
    naturally sorted.
    
    Raises:
        DirectoryNotFoundError: If the directory does not exist.
        SequenceDiscoveryError: If the target path is not a directory or cannot be accessed.
    """
    dir_path = Path(directory_path)
    if not dir_path.exists():
        raise DirectoryNotFoundError(f"Directory does not exist: {dir_path}")
    if not dir_path.is_dir():
        raise SequenceDiscoveryError(f"Target path is a file, expected directory: {dir_path}")

    found: List[Path] = []
    try:
        if recursive:
            for root, _, files in os.walk(dir_path):
                for f in files:
                    fp = Path(root) / f
                    if is_valid_image_file(fp):
                        found.append(fp)
        else:
            for entry in dir_path.iterdir():
                if entry.is_file() and is_valid_image_file(entry):
                    found.append(entry)
    except PermissionError as exc:
        raise SequenceDiscoveryError(f"Permission denied accessing directory: {dir_path}") from exc
    except Exception as exc:
        if isinstance(exc, (DirectoryNotFoundError, SequenceDiscoveryError)):
            raise
        raise SequenceDiscoveryError(f"Error scanning directory '{dir_path}': {exc}") from exc

    return natural_sort_paths(found)


def clean_folder_path(raw_path: Union[str, Path]) -> str:
    """
    Cleans raw user input or terminal drag-and-drop artifacts for folder paths.
    Handles PowerShell copy-paste artifacts (& '...', quotes, whitespace, file:/// URLs).
    """
    cleaned = str(raw_path).strip()
    if cleaned.startswith("& "):
        cleaned = cleaned[2:].strip()
    cleaned = cleaned.strip("'\"`")
    if cleaned.startswith("file:///"):
        cleaned = cleaned[8:]
    elif cleaned.startswith("file://"):
        cleaned = cleaned[7:]
    cleaned = cleaned.strip("'\"` ")
    # On Windows, preserve root like C:\, but strip trailing slashes for nested dirs
    p = Path(cleaned)
    return str(p)


def split_multiple_paths(raw_input: str) -> List[str]:
    """
    Splits a raw input string that may contain multiple paths separated by quotes,
    commas, newlines, or whitespace between quoted strings.
    """
    if not raw_input:
        return []
    s = raw_input.strip()
    # Match quoted strings or non-whitespace blocks
    matches = re.findall(r'"([^"]+)"|\'([^\']+)\'|`([^`]+)`|(\S+)', s)
    paths: List[str] = []
    for m in matches:
        val = next(item for item in m if item)
        val_clean = clean_folder_path(val)
        if val_clean and val_clean not in paths:
            paths.append(val_clean)
    return paths


def discover_sequence_folders(
    root_dir: Union[str, Path],
    max_depth: int = 2,
    include_root_if_has_images: bool = True,
) -> List[Path]:
    """
    Discovers directories that directly contain valid breakdown image sequence frames.
    
    If root_dir itself contains valid images, it is included (unless overridden).
    Also traverses subdirectories up to max_depth to find child folders containing images.
    Ignores output directories, system directories, and version control directories.
    
    Returns:
        List of naturally sorted Path objects representing sequence directories.
    """
    root = Path(clean_folder_path(root_dir)).resolve()
    if not root.exists() or not root.is_dir():
        return []

    discovered: Set[Path] = set()

    # Check root itself
    if include_root_if_has_images:
        try:
            root_frames = [
                entry for entry in root.iterdir()
                if entry.is_file() and is_valid_image_file(entry)
            ]
            if root_frames:
                discovered.add(root)
        except Exception:
            pass

    # Walk subdirectories up to max_depth
    root_parts_len = len(root.parts)
    for dirpath, dirnames, filenames in os.walk(root):
        current_p = Path(dirpath)
        depth = len(current_p.parts) - root_parts_len
        if depth > max_depth:
            dirnames.clear()
            continue

        # Filter out ignored directories in-place to prevent descent
        dirnames[:] = [
            d for d in dirnames
            if d.lower() not in IGNORED_DIRNAMES and not d.startswith(".")
        ]

        if current_p == root:
            continue

        if current_p.name.lower() in IGNORED_DIRNAMES or current_p.name.startswith("."):
            continue

        has_images = any(is_valid_image_file(current_p / f) for f in filenames)
        if has_images:
            discovered.add(current_p)

    return natural_sort_paths(list(discovered))


# ============================================================================
