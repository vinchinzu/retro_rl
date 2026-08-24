"""Level 4 dark 0x21 map pickup (waypoints, no live BFS).

Leftover play 0x21 (16,141). RoomItemId 0x17, 5× Gel. Dark: do not grant
candle. Isolated MAP_21_SAMPLE_PATH is state-BFS — not this tape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.combat import should_swing_at
from zelda_i.level4_dungeon import (
    LEVEL4,
    LEVEL4_MAP_BIT,
    ROOM_21_SPEC,
    ROOM_L4_MAP_21,
)
from zelda_i.level4_occupancy import (
    ROOM_21_CLIP_BUDGET,
    ROOM_21_PICKUP_XY,
    ROOM_21_WAYPOINTS,
)
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot

__all__ = [
    "MapPickPhase",
    "Level4MapPickController",
    "level4_mappick_stages",
    "level4_mappick_success",
    "make_mappick_controller",
]

CLIP_BUDGET = ROOM_21_CLIP_BUDGET
MAP_PICK_HOLD = 240


class MapPickPhase(Enum):
    PATH = auto()
    HOLD = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class Level4MapPickController:
    """West leftover → corridor waypoints → ADDR_MAP bit 0x08."""

    max_frames: int = 12000
    phase: MapPickPhase = MapPickPhase.PATH
    frames: int = 0
    phase_frames: int = 0
    path_index: int = 0
    combat_frames: int = 0
    success: bool = False
    notes: list[str] = field(default_factory=list)
    samples: list[dict[str, Any]] = field(default_factory=list)
    _last_xy: tuple[int, int] | None = None
    _stall: int = 0

    def _set_phase(self, phase: MapPickPhase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            self._stall = 0
            if note:
                self.notes.append(note)

    def _fail(self, note: str) -> FrameAction:
        self._set_phase(MapPickPhase.FAILED, note)
        return FrameAction(nes_idle_action(), note)

    def _sample(self, snap: ZeldaSnapshot, reason: str) -> None:
        sample = {
            "frame": self.frames,
            "x": int(snap.link_x),
            "y": int(snap.link_y),
            "phase": self.phase.name,
            "path_index": self.path_index,
            "reason": reason,
            "stall": self._stall,
            "map": int(snap.map),
        }
        if (
            not self.samples
            or self.samples[-1]["reason"] != reason
            or self.frames - self.samples[-1]["frame"] >= 250
        ):
            self.samples.append(sample)

    def _got_map(self, snap: ZeldaSnapshot) -> bool:
        return bool(int(snap.map) & LEVEL4_MAP_BIT)

    def _live(self, snap: ZeldaSnapshot) -> tuple:
        return ROOM_21_SPEC.live_enemies(snap)

    def _step_dir(
        self,
        snap: ZeldaSnapshot,
        *buttons: str,
        reason: str,
    ) -> FrameAction:
        live = self._live(snap)
        direction = buttons[0] if buttons else "RIGHT"
        if live:
            nearest = min(
                live,
                key=lambda o: abs(o.x - snap.link_x) + abs(o.y - snap.link_y),
            )
            if should_swing_at(
                snap.link_x, snap.link_y, direction, (nearest,)
            ) or (abs(nearest.x - snap.link_x) + abs(nearest.y - snap.link_y) <= 24):
                self.combat_frames += 1
                if (self.combat_frames % 6) < 3:
                    return FrameAction(
                        nes_action(*buttons, "A"), f"{reason}_slash"
                    )
        return FrameAction(nes_action(*buttons), reason)

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.phase_frames += 1
        xy = (int(snap.link_x), int(snap.link_y))
        if self._last_xy == xy:
            self._stall += 1
        else:
            self._stall = 0
            self._last_xy = xy

        if self.phase is MapPickPhase.DONE:
            return FrameAction(nes_idle_action(), "done")
        if self.phase is MapPickPhase.FAILED:
            return FrameAction(nes_idle_action(), "failed")
        if snap.mode == 17:
            return self._fail("link_death")
        if self.frames >= self.max_frames:
            self._sample(snap, "timeout")
            return self._fail(f"timeout_{xy[0]}_{xy[1]}")
        if snap.level != LEVEL4:
            return FrameAction(nes_idle_action(), "wait_level4")
        if snap.transitioning or snap.mode in (4, 6, 7):
            return FrameAction(nes_action("RIGHT"), "scroll")
        if snap.mode == 8:
            return FrameAction(nes_idle_action(), "hurt_freeze")
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.screen != ROOM_L4_MAP_21:
            return self._fail(f"wrong_room_0x{snap.screen:02x}")
        if self._got_map(snap):
            self.success = True
            self._set_phase(MapPickPhase.DONE, "map_bit")
            return FrameAction(nes_idle_action(), "done")

        if self.phase is MapPickPhase.PATH:
            if self._stall >= CLIP_BUDGET:
                self._sample(snap, "map_solid")
                return self._fail(f"map_solid_{xy[0]}_{xy[1]}")
            wps = ROOM_21_WAYPOINTS
            i = self.path_index
            while i < len(wps):
                wx, wy = wps[i]
                y_tol = 2 if (wx, wy) == ROOM_21_PICKUP_XY else 4
                if abs(xy[0] - wx) <= 4 and abs(xy[1] - wy) <= y_tol:
                    i += 1
                    self.path_index = i
                    self._sample(snap, f"waypoint_{i}")
                    continue
                # Clip east of the west column: skip leftover x=32 waypoints.
                if wx <= 32 and xy[0] > 48:
                    i += 1
                    self.path_index = i
                    self._sample(snap, f"skip_west_{i}")
                    continue
                break
            if i >= len(wps):
                self._set_phase(MapPickPhase.HOLD, "at_pickup")
            else:
                gx, gy = wps[i]
                dx, dy = gx - xy[0], gy - xy[1]
                # v11 leftover (48,189): RIGHT still x=49 wall. Clip
                # RIGHT+DOWN off the SE corner of the west column.
                if xy[0] <= 48 and xy[1] >= 185 and gx > xy[0]:
                    return self._step_dir(
                        snap, "RIGHT", "DOWN", reason="join_map_clip"
                    )
                if abs(dx) > 4:
                    return self._step_dir(
                        snap,
                        "RIGHT" if dx > 0 else "LEFT",
                        reason="join_map_x",
                    )
                if dy != 0:
                    return self._step_dir(
                        snap,
                        "DOWN" if dy > 0 else "UP",
                        reason="join_map_y",
                    )
                return FrameAction(nes_idle_action(), "map_idle")

        if self.phase is MapPickPhase.HOLD:
            if self.phase_frames >= MAP_PICK_HOLD:
                self._sample(snap, "hold_timeout")
                return self._fail(f"hold_timeout_{xy[0]}_{xy[1]}")
            hx, hy = ROOM_21_PICKUP_XY
            if abs(xy[0] - hx) > 2:
                return self._step_dir(
                    snap,
                    "RIGHT" if xy[0] < hx else "LEFT",
                    reason="hold_align_x",
                )
            if abs(xy[1] - hy) > 2:
                return self._step_dir(
                    snap,
                    "DOWN" if xy[1] < hy else "UP",
                    reason="hold_align_y",
                )
            return FrameAction(nes_idle_action(), "map_hold")
        return FrameAction(nes_idle_action(), "idle")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "phase": self.phase.name,
            "frames": self.frames,
            "path_index": self.path_index,
            "notes": list(self.notes),
            "segment": "level4_map_pickup_0x21",
            "waypoints": [list(p) for p in ROOM_21_WAYPOINTS],
            "pickup_xy": list(ROOM_21_PICKUP_XY),
            "samples": list(self.samples),
        }


def make_mappick_controller() -> Level4MapPickController:
    return Level4MapPickController()


def level4_mappick_stages():
    path = make_mappick_controller()
    return (("level4_map_pickup_0x21", path, path.max_frames),)


def level4_mappick_success(snap: ZeldaSnapshot) -> bool:
    """Play-ready 0x21 with ADDR_MAP bit 0x08. Do not require gel clear."""
    return (
        snap.level == LEVEL4
        and snap.screen == ROOM_L4_MAP_21
        and snap.mode == PLAY_MODE
        and not snap.transitioning
        and bool(int(snap.map) & LEVEL4_MAP_BIT)
    )
