"""Level 7 one-frame path policies.

``north_door_79_step`` / ``EntryNorthDoorController`` walk ``0x79`` south
mouth to live north dest ``0x69``.  Unobserved stages stay fail-closed
blockers.  A source hypothesis must never press a direction in the
cumulative spine, silently consume a timeout budget, or become
route-eligible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.anchors import SCREEN_LEVEL7_ENTRY_ROOM
from zelda_i.dungeon.hop_controller import HopController
from zelda_i.level7.graph import LEVEL7_ROOM_BY_ID, MOLDORMS, ledger_notes
from zelda_i.ram import ADDR_CANDLE, ADDR_FOOD, PLAY_MODE, ZeldaSnapshot, read_u8
from zelda_i.walk.physics import OccupancyWalker

LEVEL7 = 7
ENTRY_SCREEN = SCREEN_LEVEL7_ENTRY_ROOM  # 0x79
NORTH_DOOR_X = 120
NORTH_DOOR_Y = 93
SOUTH_MOUTH_Y = 205
NORTH_X_TOL = 4
DOOR_Y_TOL = 4
NORTH_DOOR = (NORTH_DOOR_X, NORTH_DOOR_Y)
ENTRY_NORTH_MAX_FRAMES = 4000


def north_of_entry_ram_id() -> int | None:
    """Live ``$EB`` of the room north of entry, or None until observed."""
    return LEVEL7_ROOM_BY_ID[MOLDORMS].ram_id


class Level7PathController(Protocol):
    """Minimal one-frame controller contract consumed by chapter stages."""

    max_frames: int
    frames: int
    success: bool
    failed: bool

    def step(self, snap: ZeldaSnapshot) -> FrameAction: ...

    def report(self) -> dict[str, object]: ...


@dataclass
class UnverifiedLevel7PathController:
    """Stop immediately when a chapter has no live one-frame policy."""

    stage_id: str
    missing_evidence: str
    max_frames: int = 1
    frames: int = 0
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)

    def report(self) -> dict[str, object]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "spec_id": self.stage_id,
            "evidence": "hypothesis",
            "route_eligible": False,
            "missing_evidence": self.missing_evidence,
            "notes": list(self.notes),
        }

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.failed = True
        note = (
            f"blocked_unverified:{self.stage_id}:"
            f"L{snap.level}:0x{snap.screen:02x}:m{snap.mode}:"
            f"xy={snap.link_x},{snap.link_y}"
        )
        if not self.notes:
            self.notes.append(note)
        return FrameAction(nes_idle_action(), "blocked_unverified")


def unverified_path_controller(
    stage_id: str, missing_evidence: str, *, notes: list[str] | None = None
) -> UnverifiedLevel7PathController:
    """Return a fresh blocker; controller instances are never shared."""
    controller = UnverifiedLevel7PathController(stage_id, missing_evidence)
    if notes:
        controller.notes.extend(notes)
    return controller


def north_door_79_step(
    snap: ZeldaSnapshot,
    *,
    walker: OccupancyWalker | None = None,
    dest: int | None = None,
) -> FrameAction:
    """One frame of 0x79 → north-door policy. Occupancy to (120, 93), then UP."""
    if snap.level != LEVEL7:
        return FrameAction(nes_idle_action(), "wait_level7")
    if snap.transitioning:
        return FrameAction(nes_action("UP"), "north_scroll")
    if snap.mode != PLAY_MODE:
        return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
    if dest is not None and snap.screen == dest:
        return FrameAction(nes_idle_action(), "north_arrived")
    if snap.screen != ENTRY_SCREEN:
        return FrameAction(nes_idle_action(), f"unexpected_room_0x{snap.screen:02x}")

    xy = (int(snap.link_x), int(snap.link_y))
    gx, gy = NORTH_DOOR
    if xy[1] <= gy + DOOR_Y_TOL:
        if walker is not None:
            walker.last_dir = None
        if abs(xy[0] - gx) > NORTH_X_TOL:
            btn = "LEFT" if xy[0] > gx else "RIGHT"
            return FrameAction(nes_action(btn), "north_align_x")
        return FrameAction(nes_action("UP"), "north_push")
    if walker is None:
        if abs(xy[0] - gx) > NORTH_X_TOL:
            btn = "LEFT" if xy[0] > gx else "RIGHT"
            return FrameAction(nes_action(btn), "north_align_x")
        return FrameAction(nes_action("UP"), "north_leave_mouth")
    walker.observe(xy)
    direction = walker.next_dir(xy, NORTH_DOOR)
    if direction is None:
        return FrameAction(nes_idle_action(), "occupancy_stand")
    return FrameAction(nes_action(direction), f"occ_{direction.lower()}")


@dataclass(kw_only=True)
class EntryNorthDoorController(HopController):
    """0x79 south mouth → live north dest. Occupancy miss → block → replan."""

    spec_id: str = "level7_entry_first_door"
    max_frames: int = ENTRY_NORTH_MAX_FRAMES
    require_level: int = LEVEL7
    done_reason: str = "left_0x79"
    walker: OccupancyWalker = field(
        default_factory=lambda: OccupancyWalker(goal=NORTH_DOOR)
    )
    dest: int | None = field(default_factory=north_of_entry_ram_id)

    @property
    def stage_id(self) -> str:
        return self.spec_id

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        if (
            snap.level != LEVEL7
            or snap.mode != PLAY_MODE
            or snap.transitioning
            or snap.screen == ENTRY_SCREEN
        ):
            return False
        if self.dest is not None:
            return snap.screen == self.dest
        return True

    def on_arrive(self, snap: ZeldaSnapshot) -> str:
        return f"arrived_0x{snap.screen:02x}_{snap.link_x}_{snap.link_y}"

    def timeout_note(self, snap: ZeldaSnapshot) -> str:
        return (
            f"timeout_0x{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
            f"_mode={snap.mode}_misses={self.walker.misses}"
        )

    def scroll_action(self, snap: ZeldaSnapshot) -> FrameAction:
        self.walker.last_dir = None
        return FrameAction(nes_action("UP"), "north_scroll")

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        action = north_door_79_step(snap, walker=self.walker, dest=self.dest)
        if action.reason.startswith("unexpected_room"):
            return self.mark_fail(action.reason)
        return action

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "misses": self.walker.misses,
            "spec_id": self.spec_id,
            "stage_id": self.spec_id,
            "dest_screen": self.dest,
            "evidence": "fixture-live",
            "route_eligible": False,
            "door": "UP",
        }


@dataclass
class HungryGoriyaGateController:
    """Food is a RAM gate; the room itself is still unobserved."""

    stage_id: str = "level7_entry_to_hungry_goriya"
    max_frames: int = 1
    frames: int = 0
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)
    _env: Any = field(default=None, init=False, repr=False)

    def bind_env(self, env: Any) -> None:
        self._env = env

    def _fail(self, reason: str) -> FrameAction:
        self.failed = True
        if not self.notes:
            self.notes.extend(ledger_notes())
            self.notes.append(reason)
        return FrameAction(nes_idle_action(), reason)

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        if self._env is None:
            return self._fail("hungry_goriya_env_not_bound")
        food = int(read_u8(self._env.get_ram(), ADDR_FOOD))
        if food < 1:
            return self._fail("hungry_goriya_requires_food")
        note = (
            f"blocked_unverified:{self.stage_id}:"
            f"L{snap.level}:0x{snap.screen:02x}:m{snap.mode}"
        )
        return self._fail(note)

    def report(self) -> dict[str, object]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "spec_id": self.stage_id,
            "evidence": "hypothesis",
            "route_eligible": False,
            "writes": 0,
            "notes": list(self.notes),
        }


@dataclass
class RedCandlePickupController:
    """ADDR_CANDLE 1→2 must happen naturally; room id is still unknown."""

    stage_id: str = "level7_red_candle_pickup"
    max_frames: int = 1
    frames: int = 0
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)
    _env: Any = field(default=None, init=False, repr=False)

    def bind_env(self, env: Any) -> None:
        self._env = env

    def _fail(self, reason: str) -> FrameAction:
        self.failed = True
        if not self.notes:
            self.notes.append(reason)
        return FrameAction(nes_idle_action(), reason)

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        if self._env is None:
            return self._fail("red_candle_env_not_bound")
        candle = int(read_u8(self._env.get_ram(), ADDR_CANDLE))
        if candle >= 2:
            return self._fail("red_candle_room_unobserved")
        return self._fail(
            f"red_candle_still_{candle}:L{snap.level}:0x{snap.screen:02x}"
        )

    def report(self) -> dict[str, object]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "spec_id": self.stage_id,
            "evidence": "hypothesis",
            "route_eligible": False,
            "writes": 0,
            "notes": list(self.notes),
        }


__all__ = [
    "ENTRY_SCREEN",
    "NORTH_DOOR",
    "NORTH_DOOR_X",
    "NORTH_DOOR_Y",
    "SOUTH_MOUTH_Y",
    "EntryNorthDoorController",
    "HungryGoriyaGateController",
    "Level7PathController",
    "RedCandlePickupController",
    "UnverifiedLevel7PathController",
    "north_door_79_step",
    "north_of_entry_ram_id",
    "unverified_path_controller",
]
