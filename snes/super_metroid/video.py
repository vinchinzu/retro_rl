"""Super Metroid continuous-video presets on top of ``retro_harness.video``.

The shared recorder lives in :mod:`retro_harness.video`. This module only
adds KPDR showcase gates (Zebes landing, opening-credits cutoff) and re-exports
the shared API so existing imports keep working.
"""

from __future__ import annotations

from typing import Any, Literal

from retro_harness.video import (
    FOOTER_HEIGHT,
    FrameVideoWriter,
    VideoCaptureConfig,
    VideoRecorder,
    concat_videos,
    format_snes_buttons,
    probe_video_evidence,
    render_button_footer,
    should_capture_frame,
    short_clock,
)

# Landing Site after Ceres escape — natural Zebes start for showcase trims.
ZEBES_LANDING_ROOM_ID = 0x91F8

# Default frame index after the Nintendo/title lead-in (before first Ceres
# control ~10.8k). Override via continuous_video_config(start_frame=…).
DEFAULT_OPENING_CREDITS_CUTOFF = 900

VideoStartMode = Literal["power_on", "zebes", "after_credits", "frame"]

__all__ = [
    "DEFAULT_OPENING_CREDITS_CUTOFF",
    "FOOTER_HEIGHT",
    "FrameVideoWriter",
    "VideoCaptureConfig",
    "VideoRecorder",
    "VideoStartMode",
    "ZEBES_LANDING_ROOM_ID",
    "concat_videos",
    "continuous_video_config",
    "format_snes_buttons",
    "opening_credits_cutoff",
    "probe_video_evidence",
    "render_button_footer",
    "should_capture_frame",
    "short_clock",
]


def opening_credits_cutoff(frame: int | None = None) -> int:
    """Inclusive frame after which capture may begin (title lead-in trim)."""
    if frame is None:
        return DEFAULT_OPENING_CREDITS_CUTOFF
    if frame < 0:
        raise ValueError("opening credits cutoff must be >= 0")
    return frame


def continuous_video_config(
    *,
    start: VideoStartMode = "power_on",
    start_frame: int | None = None,
    hq: bool = False,
    **overrides: Any,
) -> VideoCaptureConfig:
    """Build a :class:`VideoCaptureConfig` with Metroid showcase start gates.

    ``start``:
      - ``power_on`` — write every frame
      - ``zebes`` — latch when Landing Site (``0x91F8``) is entered
      - ``after_credits`` — skip until ``start_frame`` or
        :data:`DEFAULT_OPENING_CREDITS_CUTOFF`
      - ``frame`` — skip until explicit ``start_frame``

    ``hq=True`` applies :meth:`VideoCaptureConfig.high_quality` defaults
    (3× scale, CRF 15, slow preset) before overrides.
    """
    if start not in ("power_on", "zebes", "after_credits", "frame"):
        raise ValueError(f"unknown video start mode: {start!r}")

    gate: dict[str, Any] = {}
    if start == "zebes":
        gate["start_room_id"] = ZEBES_LANDING_ROOM_ID
        gate["start_frame"] = None
    elif start == "after_credits":
        gate["start_frame"] = opening_credits_cutoff(start_frame)
        gate["start_room_id"] = None
    elif start == "frame":
        if start_frame is None:
            raise ValueError("start='frame' requires start_frame")
        gate["start_frame"] = start_frame
        gate["start_room_id"] = None
    else:
        gate["start_frame"] = None
        gate["start_room_id"] = None

    if hq:
        # high_quality already sets scale/crf/preset; gates + overrides apply.
        return VideoCaptureConfig.high_quality(**{**gate, **overrides})
    return VideoCaptureConfig(**{**gate, **overrides})
