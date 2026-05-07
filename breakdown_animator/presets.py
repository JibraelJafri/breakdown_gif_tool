"""breakdown_animator.presets
==========================
Optimization presets for ArtStation, Portfolio 4K WebP, Discord, Slack, and Custom targets.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional, Union

from breakdown_animator.export_engine import (
    DitherMode,
    ExportFormat,
    ExportProfile,
    HarmonizeMode,
    QuantizeMethod,
)


class PresetName(str, Enum):
    """Supported export preset identifiers."""
    ARTSTATION = "artstation"
    BREAKDOWN = "breakdown"
    BREAKDOWN_SLOW = "breakdown-slow"
    BREAKDOWN_FAST = "breakdown-fast"
    TURNTABLE = "turntable"
    PORTFOLIO_4K = "portfolio-4k"
    PORTFOLIO_4K_ALT = "portfolio_4k"
    DISCORD = "discord"
    SLACK = "slack"
    CUSTOM = "custom"

    @classmethod
    def normalize(cls, value: Union[str, PresetName]) -> PresetName:
        if isinstance(value, cls):
            if value == cls.PORTFOLIO_4K_ALT:
                return cls.PORTFOLIO_4K
            return value
        v = str(value).strip().lower().replace("_", "-")
        if v in ("artstation", "art-station", "breakdown", "default"):
            return cls.ARTSTATION
        elif v in ("breakdown-slow", "slow", "study", "detail"):
            return cls.BREAKDOWN_SLOW
        elif v in ("breakdown-fast", "fast", "quick", "brisk"):
            return cls.BREAKDOWN_FAST
        elif v in ("turntable", "continuous", "spin", "360", "turn"):
            return cls.TURNTABLE
        elif v in ("portfolio-4k", "portfolio", "4k", "portfolio_4k"):
            return cls.PORTFOLIO_4K
        elif v == "discord":
            return cls.DISCORD
        elif v == "slack":
            return cls.SLACK
        elif v == "custom":
            return cls.CUSTOM
        raise ValueError(
            f"Unknown preset name: '{value}'. Expected one of: artstation, breakdown-slow, breakdown-fast, turntable, portfolio-4k, discord, slack, custom"
        )


PRESET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "artstation": {
        "name": "ArtStation Breakdown GIF",
        "description": "Standard 3D/VFX breakdown (1.0s per step, 2.0s final freeze, <= 10.0 MB auto-tuned).",
        "format": ExportFormat.GIF,
        "fps": 1.0,
        "frame_duration": 1.0,
        "hold_last_seconds": 2.0,
        "color_count": 256,
        "quantize_method": QuantizeMethod.MEDIAN_CUT,
        "dither": DitherMode.FLOYD_STEINBERG,
        "max_size_mb": 10.0,
        "auto_tune": True,
        "harmonize_mode": HarmonizeMode.LETTERBOX,
        "loop": 0,
    },
    "breakdown-slow": {
        "name": "Detailed Study Breakdown",
        "description": "Relaxed pacing for detailed texture & pass study (1.8s per step, 3.0s hold, <= 10.0 MB).",
        "format": ExportFormat.GIF,
        "fps": 0.55,
        "frame_duration": 1.8,
        "hold_last_seconds": 3.0,
        "color_count": 256,
        "quantize_method": QuantizeMethod.MEDIAN_CUT,
        "dither": DitherMode.FLOYD_STEINBERG,
        "max_size_mb": 10.0,
        "auto_tune": True,
        "harmonize_mode": HarmonizeMode.LETTERBOX,
        "loop": 0,
    },
    "breakdown-fast": {
        "name": "Brisk Breakdown",
        "description": "Snappy progression overview (0.5s per step, 1.5s hold, <= 10.0 MB).",
        "format": ExportFormat.GIF,
        "fps": 2.0,
        "frame_duration": 0.5,
        "hold_last_seconds": 1.5,
        "color_count": 256,
        "quantize_method": QuantizeMethod.MEDIAN_CUT,
        "dither": DitherMode.FLOYD_STEINBERG,
        "max_size_mb": 10.0,
        "auto_tune": True,
        "harmonize_mode": HarmonizeMode.LETTERBOX,
        "loop": 0,
    },
    "turntable": {
        "name": "3D Turntable / Continuous (12 FPS)",
        "description": "Smooth 12 FPS playback for 360 rotating models or dissolves (1.0s pause).",
        "format": ExportFormat.GIF,
        "fps": 12.0,
        "frame_duration": None,
        "hold_last_seconds": 1.0,
        "color_count": 256,
        "quantize_method": QuantizeMethod.MEDIAN_CUT,
        "dither": DitherMode.FLOYD_STEINBERG,
        "max_size_mb": 10.0,
        "auto_tune": True,
        "harmonize_mode": HarmonizeMode.LETTERBOX,
        "loop": 0,
    },
    "portfolio-4k": {
        "name": "Portfolio 4K WebP",
        "description": "Ultra-high fidelity 24-bit animated WebP for portfolio websites (1.0s per step, 2.0s hold).",
        "format": ExportFormat.WEBP,
        "fps": 1.0,
        "frame_duration": 1.0,
        "hold_last_seconds": 2.0,
        "quality": 92,
        "lossless": False,
        "method": 6,
        "minimize_size": True,
        "max_size_mb": None,
        "auto_tune": False,
        "harmonize_mode": HarmonizeMode.LETTERBOX,
        "loop": 0,
    },
    "discord": {
        "name": "Discord Breakdown",
        "description": "Optimized GIF under 8.0 MB for Discord embeds (0.8s per step, 1.5s hold).",
        "format": ExportFormat.GIF,
        "fps": 1.25,
        "frame_duration": 0.8,
        "hold_last_seconds": 1.5,
        "color_count": 128,
        "quantize_method": QuantizeMethod.MEDIAN_CUT,
        "dither": DitherMode.FLOYD_STEINBERG,
        "max_size_mb": 8.0,
        "auto_tune": True,
        "harmonize_mode": HarmonizeMode.LETTERBOX,
        "loop": 0,
    },
    "slack": {
        "name": "Slack Breakdown",
        "description": "Compact GIF under 5.0 MB for Slack attachments (0.8s per step, 1.0s hold).",
        "format": ExportFormat.GIF,
        "fps": 1.25,
        "frame_duration": 0.8,
        "hold_last_seconds": 1.0,
        "color_count": 128,
        "quantize_method": QuantizeMethod.MEDIAN_CUT,
        "dither": DitherMode.FLOYD_STEINBERG,
        "max_size_mb": 5.0,
        "auto_tune": True,
        "harmonize_mode": HarmonizeMode.LETTERBOX,
        "loop": 0,
    },
    "custom": {
        "name": "Custom",
        "description": "User-defined custom encoding settings.",
        "format": ExportFormat.GIF,
        "fps": 1.0,
        "frame_duration": 1.0,
        "hold_last_seconds": 2.0,
        "color_count": 256,
        "quantize_method": QuantizeMethod.MEDIAN_CUT,
        "dither": DitherMode.FLOYD_STEINBERG,
        "max_size_mb": None,
        "auto_tune": False,
        "harmonize_mode": HarmonizeMode.LETTERBOX,
        "loop": 0,
    },
}


def get_preset_profile(preset: Union[PresetName, str]) -> ExportProfile:
    """
    Returns an ExportProfile initialized with preset defaults.
    """
    normalized = PresetName.normalize(preset)
    key = "portfolio-4k" if normalized in (PresetName.PORTFOLIO_4K, PresetName.PORTFOLIO_4K_ALT) else normalized.value
    cfg = PRESET_CONFIGS[key]
    return ExportProfile(
        format=cfg["format"],
        fps=cfg.get("fps", 1.0),
        frame_duration=cfg.get("frame_duration"),
        hold_last_seconds=cfg.get("hold_last_seconds", 0.0),
        color_count=cfg.get("color_count", 256),
        quantize_method=cfg.get("quantize_method", QuantizeMethod.MEDIAN_CUT),
        dither=cfg.get("dither", DitherMode.FLOYD_STEINBERG),
        lossless=cfg.get("lossless", False),
        quality=cfg.get("quality", 85),
        max_size_mb=cfg.get("max_size_mb"),
        auto_tune=cfg.get("auto_tune", False),
        harmonize_mode=cfg.get("harmonize_mode", HarmonizeMode.LETTERBOX),
        loop=cfg.get("loop", 0),
        method=cfg.get("method", 6),
        minimize_size=cfg.get("minimize_size", True),
    )
