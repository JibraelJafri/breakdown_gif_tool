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
