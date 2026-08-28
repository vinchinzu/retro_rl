"""Super Metroid continuous-video presets on top of ``retro_harness.video``.

The shared recorder lives in :mod:`retro_harness.video`. This module only
adds KPDR showcase start gates (opening-credits cutoff, Zebes landing) and
re-exports the shared API so existing imports keep working. Layout and
encode knobs come from :class:`VideoCaptureConfig` factories.
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


def _start_gates(
    start: VideoStartMode, start_frame: int | None
) -> dict[str, Any]:
    if start == "zebes":
        return {"start_room_id": ZEBES_LANDING_ROOM_ID, "start_frame": None}
    if start == "after_credits":
        return {
            "start_frame": opening_credits_cutoff(start_frame),
            "start_room_id": None,
        }
    if start == "frame":
        if start_frame is None:
            raise ValueError("start='frame' requires start_frame")
        return {"start_frame": start_frame, "start_room_id": None}
    return {"start_frame": None, "start_room_id": None}


def continuous_video_config(
    *,
    start: VideoStartMode = "after_credits",
    start_frame: int | None = None,
    hq: bool = False,
    **overrides: Any,
) -> VideoCaptureConfig:
    """Build a :class:`VideoCaptureConfig` with Metroid showcase start gates.

    ``start``:
      - ``power_on`` — write every frame (debug; includes Nintendo/title)
      - ``zebes`` — latch when Landing Site (``0x91F8``) is entered
      - ``after_credits`` — skip Nintendo/title through
        :data:`DEFAULT_OPENING_CREDITS_CUTOFF` (product default)
      - ``frame`` — skip until explicit ``start_frame``

    Product YouTube: ``after_credits`` + ``layout="youtube"`` (1080p60
    sidebars). Never prepend intro cards. ``hq=True`` on youtube only
    tightens CRF/preset — scale auto-fits 1920x1080. Native ``hq`` uses
    :meth:`VideoCaptureConfig.high_quality`.
    """
    if start not in ("power_on", "zebes", "after_credits", "frame"):
        raise ValueError(f"unknown video start mode: {start!r}")

    kwargs: dict[str, Any] = {**_start_gates(start, start_frame), **overrides}
    layout = str(kwargs.get("layout", "youtube"))
    if layout == "youtube":
        if hq:
            kwargs.setdefault("crf", 15)
            kwargs.setdefault("preset", "slow")
        return VideoCaptureConfig.youtube(**kwargs)
    if hq:
        return VideoCaptureConfig.high_quality(**kwargs)
    return VideoCaptureConfig(**kwargs)
