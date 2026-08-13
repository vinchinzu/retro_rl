"""Shared non-UI emulator session helpers (speed, turbo preview, WRAM, recordings).

Qt EmbeddedEmulatorPanel and pygame PlaySession differ in I/O, but share timing
policy: speed ladders, turbo frame-skip intervals, and unthrottled delay rules.
Kept outside ``retro_harness.editor`` so PlaySession does not import Qt.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Sequence

# Match PlaySession turbo preview cadence (see play_session._TURBO_RENDER_INTERVAL).
DEFAULT_TURBO_FRAME_PREVIEW_INTERVAL = 8


def frame_budget_ms_for_speed(speed: float, *, base_frame_ms: int = 16) -> int:
    """Return wall-clock delay between emulator steps at ``speed``."""

    if speed <= 0:
        return base_frame_ms
    return max(1, int(round(base_frame_ms / speed)))


def format_speed_label(speed: float) -> str:
    """Human-readable speed multiplier for HUD / status lines."""

    if speed != int(speed):
        return f"{speed:g}x"
    return f"{int(speed)}x"


def step_repeat_for_speed(speed: float, *, speed_uses_frame_repeat: bool) -> int:
    """How many emulator frames to advance in one wall-clock tick."""

    if speed_uses_frame_repeat and speed >= 2.0:
        return max(1, int(round(speed)))
    return 1


def should_preview_turbo_frame(
    step_counter: int,
    *,
    turbo: bool,
    interval: int = DEFAULT_TURBO_FRAME_PREVIEW_INTERVAL,
) -> bool:
    """Return True when a turbo/fast-forward tick should refresh the visible frame.

    ``step_counter`` is 1-based for the panel (incremented before the check) and
    0-based frame counts work for PlaySession via ``frame_count % interval == 0``.
    """

    if not turbo:
        return True
    interval = max(1, int(interval))
    return step_counter % interval == 0


def step_delay_ms(
    *,
    speed: float,
    repeat: int,
    frame_ms: float,
    target_frame_ms: int,
    base_frame_ms: int,
    unthrottled_speed_threshold: float,
) -> int:
    """Wall-clock delay before the next step, or 0 when unthrottled."""

    if speed >= unthrottled_speed_threshold:
        return 0
    target_tick_ms = base_frame_ms if repeat > 1 else target_frame_ms
    return max(0, int(target_tick_ms) - int(frame_ms))


def should_include_wram(
    *,
    include_wram_when_stepping: bool,
    force_wram_next_step: bool,
    wram_sync_interval_frames: int,
    frame: int,
    last_wram_sync_frame: int,
    synced_tilemap: int | None,
    tilemap_id: int | None,
) -> bool:
    """Decide whether the next bridge step should request a WRAM payload."""

    if not include_wram_when_stepping:
        return False
    if force_wram_next_step:
        return True
    if wram_sync_interval_frames <= 0:
        return True
    if frame <= 0 or last_wram_sync_frame < 0:
        return True
    if (
        synced_tilemap is not None
        and tilemap_id is not None
        and tilemap_id != synced_tilemap
    ):
        return True
    return frame - last_wram_sync_frame >= wram_sync_interval_frames


def after_step_wram_flags(
    snapshot: dict[str, object],
    *,
    wram_sync_interval_frames: int,
    synced_tilemap: int | None,
    tilemap_id: int | None,
    frame: int,
) -> tuple[int | None, bool]:
    """Return ``(new_last_wram_sync_frame, force_wram_next_step)``.

    ``new_last_wram_sync_frame`` is None when the last-sync marker should be left alone.
    """

    if snapshot.get("wramBase64") or snapshot.get("wramRaw"):
        return frame, False
    if wram_sync_interval_frames <= 0:
        return None, False
    if (
        synced_tilemap is not None
        and tilemap_id is not None
        and tilemap_id != synced_tilemap
    ):
        return None, True
    return None, False


def compact_snapshot(
    snapshot: dict[str, object],
    keys: Sequence[str],
) -> dict[str, object]:
    if not keys:
        return dict(snapshot)
    return {key: snapshot[key] for key in keys if key in snapshot}


def build_script_recording_document(
    *,
    name: str,
    button_order: Sequence[str],
    state_file: str | None,
    selected_state_file: str | None,
    start_capture: dict[str, object] | None,
    room_label: str | None,
    total_frames: int,
    segments: list[dict[str, object]],
    markers: list[dict[str, object]],
    last_snapshot: dict[str, object] | None,
    recording_format: str,
    recording_version: int,
    recording_tool: str,
    bridge_module: str,
    recording_path: Path,
    recorded_at: str | None = None,
) -> dict[str, object]:
    """JSON-serializable editor script recording payload (no I/O)."""

    stamp = recorded_at or datetime.now().isoformat(timespec="seconds")
    return {
        "format": recording_format,
        "version": recording_version,
        "tool": recording_tool,
        "name": name,
        "recordedAt": stamp,
        "buttonOrder": list(button_order),
        "stateFile": state_file or selected_state_file,
        "selectedStateFile": selected_state_file,
        "startCapture": start_capture,
        "roomLabel": room_label,
        "totalFrames": total_frames,
        "segments": segments,
        "markers": markers,
        "lastSnapshot": last_snapshot,
        "captureEachSegment": False,
        "aiUse": {
            "description": (
                f"Replay with {bridge_module} run_script to reproduce labeled emulator states."
            ),
            "headlessCommand": (
                f"PYTHONPATH=. uv run --project .. python -m {bridge_module} "
                f"--script {recording_path}"
            ),
        },
    }


def format_fps_status(
    *,
    avg_fps: float,
    frame_ms: float,
    bridge_step_ms: float,
    speed: float,
    ram_recording: bool = False,
    script_recording: bool = False,
) -> str:
    bridge_text = f"  bridge {bridge_step_ms:.0f}ms" if bridge_step_ms > 0 else ""
    mode_bits: list[str] = [format_speed_label(speed)]
    if ram_recording:
        mode_bits.append("RAM rec")
    if script_recording:
        mode_bits.append("script rec")
    mode_text = f"  [{' | '.join(mode_bits)}]"
    return f"FPS {avg_fps:4.1f}  frame {frame_ms:4.0f}ms{bridge_text}{mode_text}"


@dataclass
class EmulatorSpeedController:
    """Mutable speed ladder + turbo frame-skip counter for a play surface."""

    levels: tuple[float, ...]
    default_index: int = 0
    base_frame_ms: int = 16
    speed_uses_frame_repeat: bool = False
    skip_frame_when_turbo: bool = True
    turbo_speed_threshold: float = 4.0
    turbo_frame_preview_interval: int = DEFAULT_TURBO_FRAME_PREVIEW_INTERVAL
    unthrottled_speed_threshold: float = 8.0
    index: int = field(init=False)
    turbo_step_counter: int = 0
    target_frame_ms: int = field(init=False)

    def __post_init__(self) -> None:
        if not self.levels:
            raise ValueError("speed levels must be non-empty")
        self.index = max(0, min(int(self.default_index), len(self.levels) - 1))
        self._refresh_target()

    @property
    def multiplier(self) -> float:
        return float(self.levels[self.index])

    def _refresh_target(self) -> None:
        self.target_frame_ms = frame_budget_ms_for_speed(
            self.multiplier,
            base_frame_ms=self.base_frame_ms,
        )

    def reset(self) -> None:
        self.index = max(0, min(int(self.default_index), len(self.levels) - 1))
        self.turbo_step_counter = 0
        self._refresh_target()

    def decrease(self) -> bool:
        if self.index <= 0:
            return False
        self.index -= 1
        self.turbo_step_counter = 0
        self._refresh_target()
        return True

    def increase(self) -> bool:
        if self.index >= len(self.levels) - 1:
            return False
        self.index += 1
        self.turbo_step_counter = 0
        self._refresh_target()
        return True

    def step_repeat(self) -> int:
        return step_repeat_for_speed(
            self.multiplier,
            speed_uses_frame_repeat=self.speed_uses_frame_repeat,
        )

    def should_include_frame(self) -> bool:
        speed = self.multiplier
        if speed < self.turbo_speed_threshold:
            return True
        if not self.skip_frame_when_turbo:
            return True
        self.turbo_step_counter += 1
        return should_preview_turbo_frame(
            self.turbo_step_counter,
            turbo=True,
            interval=self.turbo_frame_preview_interval,
        )

    def delay_ms(self, *, repeat: int, frame_ms: float) -> int:
        return step_delay_ms(
            speed=self.multiplier,
            repeat=repeat,
            frame_ms=frame_ms,
            target_frame_ms=self.target_frame_ms,
            base_frame_ms=self.base_frame_ms,
            unthrottled_speed_threshold=self.unthrottled_speed_threshold,
        )

    def label(self) -> str:
        return format_speed_label(self.multiplier)


@dataclass
class FrameTimingTracker:
    """Rolling FPS samples for status labels."""

    maxlen: int = 30
    samples: deque[float] = field(init=False)
    last_frame_ms: float = 0.0
    last_bridge_step_ms: float = 0.0

    def __post_init__(self) -> None:
        self.samples = deque(maxlen=self.maxlen)

    def record(
        self,
        frame_ms: float,
        *,
        bridge_step_ms: float | None = None,
    ) -> float:
        self.last_frame_ms = frame_ms
        if bridge_step_ms is not None:
            self.last_bridge_step_ms = bridge_step_ms
        if frame_ms > 0:
            self.samples.append(1000.0 / frame_ms)
        if not self.samples:
            return 0.0
        return sum(self.samples) / len(self.samples)

    def status_text(
        self,
        *,
        frame_ms: float,
        speed: float,
        ram_recording: bool = False,
        script_recording: bool = False,
    ) -> str:
        avg_fps = self.record(frame_ms)
        return format_fps_status(
            avg_fps=avg_fps,
            frame_ms=frame_ms,
            bridge_step_ms=self.last_bridge_step_ms,
            speed=speed,
            ram_recording=ram_recording,
            script_recording=script_recording,
        )


__all__ = [
    "DEFAULT_TURBO_FRAME_PREVIEW_INTERVAL",
    "frame_budget_ms_for_speed",
    "EmulatorSpeedController",
    "FrameTimingTracker",
    "after_step_wram_flags",
    "build_script_recording_document",
    "compact_snapshot",
    "format_fps_status",
    "format_speed_label",
    "should_include_wram",
    "should_preview_turbo_frame",
    "step_delay_ms",
    "step_repeat_for_speed",
]
