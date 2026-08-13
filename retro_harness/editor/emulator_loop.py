"""Editor-facing re-exports of shared emulator session helpers."""

from __future__ import annotations

from retro_harness.emulator_session import (
    DEFAULT_TURBO_FRAME_PREVIEW_INTERVAL,
    EmulatorSpeedController,
    FrameTimingTracker,
    after_step_wram_flags,
    build_script_recording_document,
    compact_snapshot,
    format_fps_status,
    format_speed_label,
    frame_budget_ms_for_speed,
    should_include_wram,
    should_preview_turbo_frame,
    step_delay_ms,
    step_repeat_for_speed,
)

__all__ = [
    "DEFAULT_TURBO_FRAME_PREVIEW_INTERVAL",
    "EmulatorSpeedController",
    "FrameTimingTracker",
    "after_step_wram_flags",
    "build_script_recording_document",
    "compact_snapshot",
    "format_fps_status",
    "format_speed_label",
    "frame_budget_ms_for_speed",
    "should_include_wram",
    "should_preview_turbo_frame",
    "step_delay_ms",
    "step_repeat_for_speed",
]
