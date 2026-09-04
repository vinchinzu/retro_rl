"""Fixture-live Level 9 Magical-Key prefix dest hops.

0x76 leftover → north door UP. Dest is RAM (hyp 0x66 Old Man TF gate).
Natural-spine factories in ``natural_path`` stay fail-closed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.dungeon.hop_controller import (
    HopController,
    WAIT_SCROLL_B,
    dungeon_align_then_push,
)
from zelda_i.dungeon.ops import DOOR_TARGETS
from zelda_i.level9.dungeon import LEVEL9, ROOM_LEVEL9_ENTRY, ROOM_OLD_MAN_TF, ROOM_RED_RING_HYP
from zelda_i.ram import PASSAGE_MODE, PLAY_MODE, ZeldaSnapshot

NORTH_DOOR = DOOR_TARGETS["UP"]  # (120, 93)
WEST_DOOR = DOOR_TARGETS["LEFT"]  # (32, 141)
NORTH_ORIGIN = ROOM_LEVEL9_ENTRY  # 0x76
NORTH_DEST_HYP = ROOM_OLD_MAN_TF  # 0x66
NORTH_DEST_POSE = (120, 205)  # live P1/P2 south mouth
WEST_ORIGIN = ROOM_OLD_MAN_TF  # 0x66
WEST_DEST_HYP = 0x65
WEST_DEST_POSE = (224, 141)  # live W1 east mouth
RED_RING = ROOM_RED_RING_HYP  # 0x07
_DOOR_TOL = 4
_SAMPLE_PERIOD = 12
_MAX_FRAMES = 4000


def is_north_neighbor(origin: int, dest: int) -> bool:
    """Same column, one dungeon row north (``$EB - 0x10``)."""
    return dest == origin - 0x10


def is_west_neighbor(origin: int, dest: int) -> bool:
    """Same row, one dungeon column west (``$EB - 1``)."""
    return dest == origin - 1


def north_76_step(snap: ZeldaSnapshot) -> FrameAction:
    """x-align to 120, then UP. No occupancy, no sword, no LEFT/RIGHT push."""
    return dungeon_align_then_push(
        snap,
        push_dir="UP",
        target_x=NORTH_DOOR[0],
        x_tol=_DOOR_TOL,
        reason="north_76",
    )


def west_66_step(snap: ZeldaSnapshot) -> FrameAction:
    """y-align to 141, then LEFT. No occupancy, no sword, no UP."""
    return dungeon_align_then_push(
        snap,
        push_dir="LEFT",
        target_y=WEST_DOOR[1],
        door_plane=WEST_DOOR[0],
        y_tol=_DOOR_TOL,
        reason="west_66",
    )


def _leftover(snap: ZeldaSnapshot) -> dict[str, Any]:
    return {
        "x": int(snap.link_x),
        "y": int(snap.link_y),
        "mode": int(snap.mode),
        "screen": int(snap.screen),
        "tile": int(snap.colliding_tile),
        "keys": int(snap.keys),
        "bombs": int(snap.bombs),
        "triforce": int(snap.triforce),
    }


@dataclass(kw_only=True)
class Level9North76Controller(HopController):
    """0x76 leftover → north door UP. Dest is RAM; fail non-north / 0x07."""

    spec_id: str = "level9_north_76"
    max_frames: int = _MAX_FRAMES
    require_level: int = LEVEL9
    wait_modes: tuple[int, ...] = WAIT_SCROLL_B
    done_reason: str = "left_0x76_north"
    dest: int | None = None
    route_eligible: bool = False
    leftover: dict[str, Any] = field(default_factory=dict)
    writes: int = 0

    @property
    def stage_id(self) -> str:
        return self.spec_id

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        if snap.mode != PLAY_MODE or snap.transitioning:
            return False
        if snap.screen in (RED_RING, NORTH_ORIGIN):
            return False
        if self.dest is not None:
            return snap.screen == self.dest
        return is_north_neighbor(NORTH_ORIGIN, snap.screen)

    def on_arrive(self, snap: ZeldaSnapshot) -> str:
        return f"play_0x{snap.screen:02x}_{snap.link_x}_{snap.link_y}"

    def scroll_action(self, snap: ZeldaSnapshot) -> FrameAction:
        del snap
        return FrameAction(nes_action("UP"), "north_scroll")

    def emit(
        self, snap: ZeldaSnapshot, action: FrameAction, *, force: bool = False
    ) -> FrameAction:
        if force or not self.leftover or self.frames % _SAMPLE_PERIOD == 0:
            self.leftover = _leftover(snap)
        return action

    def guard(self, snap: ZeldaSnapshot) -> FrameAction | None:
        blocked = HopController.guard(self, snap)
        if blocked is not None:
            return blocked
        if snap.mode == PASSAGE_MODE:
            return self.mark_fail(f"cellar_0x{snap.screen:02x}")
        if snap.screen == RED_RING:
            return self.mark_fail("red_ring_0x07")
        if (
            snap.mode == PLAY_MODE
            and not snap.transitioning
            and snap.screen != NORTH_ORIGIN
        ):
            if self.dest is not None and snap.screen != self.dest:
                return self.mark_fail(f"unexpected_play_0x{snap.screen:02x}")
            if not is_north_neighbor(NORTH_ORIGIN, snap.screen):
                return self.mark_fail(f"not_north_neighbor_0x{snap.screen:02x}")
        return None

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.screen != NORTH_ORIGIN:
            return FrameAction(nes_action("UP"), "north_settle")
        return north_76_step(snap)

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "spec_id": self.spec_id,
            "dest_screen": self.dest,
            "dest_hyp": NORTH_DEST_HYP,
            "evidence": "fixture-live",
            "route_eligible": False,
            "natural_entry": False,
            "writes": int(self.writes),
            "door": "UP",
            "leftover": dict(self.leftover),
        }


@dataclass(kw_only=True)
class Level9West66Controller(HopController):
    """0x66 leftover → west shutter LEFT. Dest is RAM; fail non-west / 0x07."""

    spec_id: str = "level9_west_66"
    max_frames: int = _MAX_FRAMES
    require_level: int = LEVEL9
    wait_modes: tuple[int, ...] = WAIT_SCROLL_B
    done_reason: str = "left_0x66_west"
    dest: int | None = None
    route_eligible: bool = False
    leftover: dict[str, Any] = field(default_factory=dict)
    writes: int = 0

    @property
    def stage_id(self) -> str:
        return self.spec_id

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        if snap.mode != PLAY_MODE or snap.transitioning:
            return False
        if snap.screen in (RED_RING, WEST_ORIGIN):
            return False
        if self.dest is not None:
            return snap.screen == self.dest
        return is_west_neighbor(WEST_ORIGIN, snap.screen)

    def on_arrive(self, snap: ZeldaSnapshot) -> str:
        return f"play_0x{snap.screen:02x}_{snap.link_x}_{snap.link_y}"

    def scroll_action(self, snap: ZeldaSnapshot) -> FrameAction:
        del snap
        return FrameAction(nes_action("LEFT"), "west_scroll")

    def emit(
        self, snap: ZeldaSnapshot, action: FrameAction, *, force: bool = False
    ) -> FrameAction:
        if force or not self.leftover or self.frames % _SAMPLE_PERIOD == 0:
            self.leftover = _leftover(snap)
        return action

    def guard(self, snap: ZeldaSnapshot) -> FrameAction | None:
        blocked = HopController.guard(self, snap)
        if blocked is not None:
            return blocked
        if snap.mode == PASSAGE_MODE:
            return self.mark_fail(f"cellar_0x{snap.screen:02x}")
        if snap.screen == RED_RING:
            return self.mark_fail("red_ring_0x07")
        if (
            snap.mode == PLAY_MODE
            and not snap.transitioning
            and snap.screen != WEST_ORIGIN
        ):
            if self.dest is not None and snap.screen != self.dest:
                return self.mark_fail(f"unexpected_play_0x{snap.screen:02x}")
            if not is_west_neighbor(WEST_ORIGIN, snap.screen):
                return self.mark_fail(f"not_west_neighbor_0x{snap.screen:02x}")
        return None

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.screen != WEST_ORIGIN:
            return FrameAction(nes_action("LEFT"), "west_settle")
        return west_66_step(snap)

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "spec_id": self.spec_id,
            "dest_screen": self.dest,
            "dest_hyp": WEST_DEST_HYP,
            "evidence": "fixture-live",
            "route_eligible": False,
            "natural_entry": False,
            "writes": int(self.writes),
            "door": "LEFT",
            "leftover": dict(self.leftover),
        }


def make_north_76_controller(
    *, dest: int | None = None
) -> Level9North76Controller:
    return Level9North76Controller(dest=dest)


def make_west_66_controller(
    *, dest: int | None = None
) -> Level9West66Controller:
    return Level9West66Controller(dest=dest)


__all__ = [
    "NORTH_DEST_HYP",
    "NORTH_DEST_POSE",
    "NORTH_DOOR",
    "NORTH_ORIGIN",
    "RED_RING",
    "WEST_DEST_HYP",
    "WEST_DEST_POSE",
    "WEST_DOOR",
    "WEST_ORIGIN",
    "Level9North76Controller",
    "Level9West66Controller",
    "is_north_neighbor",
    "is_west_neighbor",
    "make_north_76_controller",
    "make_west_66_controller",
    "north_76_step",
    "west_66_step",
]
