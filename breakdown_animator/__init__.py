"""breakdown_animator (animforge)
================================
Modern, high-performance 3D render breakdown animation engine for Python.
Converts progressive render sequences into optimized animated GIFs and WebPs
for ArtStation, Discord, Slack, and portfolio websites with zero external binaries.
"""

__version__ = "1.0.0"
__author__ = "AnimForge Team"

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
)
from breakdown_animator.presets import (
    PRESET_CONFIGS,
    PresetName,
    get_preset_profile,
)
from breakdown_animator.sequence_engine import (
    CorruptImageError,
    DimensionMismatchError,
    DirectoryNotFoundError,
    EmptySequenceError,
    FrameInfo,
    SequenceEngineError,
    SequenceGroup,
    analyze_sequence,
    group_sequences,
    inspect_frame,
    load_and_normalize_frame,
    natural_sort_key,
    natural_sort_paths,
    normalize_frame_mode,
    scan_directory,
)

__all__ = [
    "__version__",
    # Sequence Engine
    "scan_directory",
    "group_sequences",
    "analyze_sequence",
    "inspect_frame",
    "natural_sort_key",
    "natural_sort_paths",
    "normalize_frame_mode",
    "load_and_normalize_frame",
    "FrameInfo",
    "SequenceGroup",
    "SequenceEngineError",
    "DirectoryNotFoundError",
    "EmptySequenceError",
    "CorruptImageError",
    "DimensionMismatchError",
    # Export Engine
    "ExportFormat",
    "QuantizeMethod",
    "DitherMode",
    "HarmonizeMode",
    "ExportProfile",
    "ExportResult",
    "calculate_frame_durations",
    "harmonize_frame",
    "export_sequence",
    # Presets
    "PresetName",
    "PRESET_CONFIGS",
    "get_preset_profile",
]
