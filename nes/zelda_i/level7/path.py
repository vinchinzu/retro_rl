"""Level 7 one-frame path policies.

``north_door_79_step`` / ``EntryNorthDoorController`` walk ``0x79`` south
mouth to live north dest ``0x69``.  ``room69_east_step`` /
``Room69EastController`` clear the ``0x69`` goriyas then walk the east
doorway, which is an **OPEN** gate (black passage on the spawn frame) —
``cur_opened_doors`` never sets its RIGHT bit, exactly like the ``0x79``
north door.  The post-clear traverse is a deterministic waypoint micro,
not occupancy: the ``0x69`` centre row is walled at ``y=141`` and a
per-pixel grid boxes Link in after four graded misses on one cell.
Unobserved stages stay fail-closed blockers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.anchors import SCREEN_LEVEL7_ENTRY_ROOM
from zelda_i.combat import nearest_enemy, should_swing_at
from zelda_i.door_graph.core import DoorDir
from zelda_i.dungeon.behaviors import (
    GORIYA_BLUE_TYPE,
    GORIYA_TYPE,
    EnemyKind,
    engagement_hint,
    is_projectile,
    live_among,
)
from zelda_i.dungeon.engine import AliveRule
from zelda_i.dungeon.hop_controller import HopController, dungeon_align_then_push
from zelda_i.level7.graph import (
    KEESE,
    LEVEL7_ROOM_BY_ID,
    MOLDORMS,
    ledger_notes,
)
from zelda_i.ram import (
    ADDR_CANDLE,
    ADDR_FOOD,
    PLAY_MODE,
    ZeldaObject,
    ZeldaSnapshot,
    read_u8,
)
from zelda_i.walk.physics import OccupancyWalker

LEVEL7 = 7
ENTRY_SCREEN = SCREEN_LEVEL7_ENTRY_ROOM  # 0x79
ROOM_69 = 0x69
NORTH_DOOR_X = 120
NORTH_DOOR_Y = 93
SOUTH_MOUTH_Y = 205
NORTH_X_TOL = 4
DOOR_Y_TOL = 4
NORTH_DOOR = (NORTH_DOOR_X, NORTH_DOOR_Y)
EAST_DOOR_X = 208
EAST_DOOR_Y = 141
EAST_DOOR = (EAST_DOOR_X, EAST_DOOR_Y)
# 0x69 traverse band: the centre row carries impassable tiles either side of
# x=128, so cross on the live-clear y=109 band and drop on the east column.
EAST_BAND_Y = 109
EAST_APPROACH_X = 204
ENTRY_NORTH_MAX_FRAMES = 4000
ROOM69_EAST_MAX_FRAMES = 6000
_SWING_PERIOD = 8
_SWING_HOLD = 4
_GORIYA_TYPES = frozenset({GORIYA_BLUE_TYPE, GORIYA_TYPE})
_OPP = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}
# Same inland box as dungeon.engine avoid_walls. West door column is x=32.
_INLAND_X = (56, 200)
_INLAND_Y = (109, 173)


def north_of_entry_ram_id() -> int | None:
    """Live ``$EB`` of the room north of entry, or None until observed."""
    return LEVEL7_ROOM_BY_ID[MOLDORMS].ram_id


def east_of_room69_ram_id() -> int | None:
    """Live ``$EB`` of the room east of ``0x69``, or None until observed."""
    return LEVEL7_ROOM_BY_ID[KEESE].ram_id


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


def _combatants(snap: ZeldaSnapshot) -> tuple[ZeldaObject, ...]:
    return tuple(obj for obj in snap.objects if 1 <= obj.slot <= 12)


def live_goriyas(snap: ZeldaSnapshot) -> tuple[ZeldaObject, ...]:
    """HP-live Goriya slots (blue 0x05 / red 0x06). Boomerang 0x5C is excluded."""
    return tuple(
        obj
        for obj in live_among(_combatants(snap), AliveRule.TYPE_AND_HP)
        if (int(obj.type_id) & 0xFF) in _GORIYA_TYPES
    )


def _projectiles(snap: ZeldaSnapshot) -> tuple[ZeldaObject, ...]:
    return tuple(obj for obj in _combatants(snap) if is_projectile(obj))


def _leave_wall(snap: ZeldaSnapshot) -> FrameAction | None:
    """Step toward the playable interior. Do not chase the west wall."""
    x, y = int(snap.link_x), int(snap.link_y)
    if x < _INLAND_X[0]:
        direction = "RIGHT"
    elif x > _INLAND_X[1]:
        direction = "LEFT"
    elif y < _INLAND_Y[0]:
        direction = "DOWN"
    elif y > _INLAND_Y[1]:
        direction = "UP"
    else:
        return None
    return FrameAction(nes_action(direction), "leave_wall")


def _east_push(snap: ZeldaSnapshot) -> FrameAction:
    return dungeon_align_then_push(
        snap,
        push_dir="RIGHT",
        target_y=EAST_DOOR_Y,
        y_tol=DOOR_Y_TOL,
        door_plane=EAST_DOOR_X,
        reason="east",
    )


def _goriya_fight(
    snap: ZeldaSnapshot, target: ZeldaObject, *, frames: int
) -> FrameAction:
    hint = engagement_hint(
        EnemyKind.GORIYA, snap, target, projectiles=_projectiles(snap)
    )
    if should_swing_at(
        snap.link_x, snap.link_y, hint.face, (target,), hint=hint
    ):
        if frames % _SWING_PERIOD < _SWING_HOLD:
            return FrameAction(nes_action(hint.face, "A"), "goriya_slash")
        return FrameAction(nes_action(hint.face), "goriya_face")
    leave = _leave_wall(snap)
    if leave is not None:
        return leave
    if hint.retreat:
        return FrameAction(nes_action(_OPP[hint.face]), "goriya_retreat")
    return FrameAction(nes_action(hint.face), "goriya_chase")


def east_route_step(snap: ZeldaSnapshot) -> FrameAction:
    """Deterministic 0x69 traverse: y=109 band → east column → door row → push.

    Every waypoint is live (`room69_east_v3/v4` samples reached ``(200,109)``
    and ``(204,141)`` mid-fight).
    """
    x, y = int(snap.link_x), int(snap.link_y)
    if x < EAST_APPROACH_X - NORTH_X_TOL:
        if abs(y - EAST_BAND_Y) > DOOR_Y_TOL:
            btn = "UP" if y > EAST_BAND_Y else "DOWN"
            return FrameAction(nes_action(btn), "east_band_y")
        return FrameAction(nes_action("RIGHT"), "east_band_x")
    if abs(y - EAST_DOOR_Y) > DOOR_Y_TOL:
        btn = "UP" if y > EAST_DOOR_Y else "DOWN"
        return FrameAction(nes_action(btn), "east_door_y")
    return _east_push(snap)


def room69_east_step(
    snap: ZeldaSnapshot,
    *,
    dest: int | None = None,
    saw_goriya: bool = False,
    frames: int = 0,
) -> FrameAction:
    """One frame of 0x69 kill-clear → east door. Occupancy miss → block → replan.

    The east exit is an OPEN doorway, so nothing waits on a door bit; the
    centre-row obstacles are routed around by the occupancy grid.
    """
    if snap.level != LEVEL7:
        return FrameAction(nes_idle_action(), "wait_level7")
    if snap.transitioning:
        return FrameAction(nes_action("RIGHT"), "east_scroll")
    if snap.mode != PLAY_MODE:
        return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
    if dest is not None and snap.screen == dest:
        return FrameAction(nes_idle_action(), "east_arrived")
    if snap.screen != ROOM_69:
        return FrameAction(nes_idle_action(), f"unexpected_room_0x{snap.screen:02x}")

    live = live_goriyas(snap)
    if live:
        target = nearest_enemy(snap.link_x, snap.link_y, live)
        if target is None:
            return FrameAction(nes_idle_action(), "goriya_missing")
        return _goriya_fight(snap, target, frames=frames)

    if not saw_goriya:
        return FrameAction(nes_idle_action(), "spawn_wait")
    return east_route_step(snap)


@dataclass(kw_only=True)
class Room69EastController(HopController):
    """0x69 south mouth: kill-clear goriyas, then walk the earned east door."""

    spec_id: str = "level7_room69_east"
    max_frames: int = ROOM69_EAST_MAX_FRAMES
    require_level: int = LEVEL7
    done_reason: str = "left_0x69"
    dest: int | None = field(default_factory=east_of_room69_ram_id)
    saw_goriya: bool = False
    east_opened_frame: int | None = None
    obj_types: list[int] = field(default_factory=list)

    @property
    def stage_id(self) -> str:
        return self.spec_id

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        if (
            snap.level != LEVEL7
            or snap.mode != PLAY_MODE
            or snap.transitioning
            or snap.screen in {ENTRY_SCREEN, ROOM_69}
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
            f"_mode={snap.mode}_saw={int(self.saw_goriya)}"
            f"_east_f={self.east_opened_frame}"
        )

    def scroll_action(self, snap: ZeldaSnapshot) -> FrameAction:
        if snap.link_y >= 189:
            return self.mark_fail("south_backtrack")
        return FrameAction(nes_action("RIGHT"), "east_scroll")

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        live = live_goriyas(snap)
        if live:
            self.saw_goriya = True
        types = sorted(
            {
                int(obj.type_id) & 0xFF
                for obj in _combatants(snap)
                if (int(obj.type_id) & 0xFF) not in (0, 0xFF)
            }
        )
        if types and types != self.obj_types:
            self.obj_types = types
            self._note("obj_types:" + ",".join(f"0x{t:02x}" for t in types))
        # Telemetry only: the east exit is an OPEN doorway, so this bit is
        # expected to stay clear. Movement never waits on it.
        if snap.cur_opened_doors & DoorDir.RIGHT and self.east_opened_frame is None:
            self.east_opened_frame = self.frames
            self._note(
                f"east_opened_f{self.frames}_doors={snap.cur_opened_doors}"
                f"_dead={snap.room_all_dead}"
            )
        action = room69_east_step(
            snap,
            dest=self.dest,
            saw_goriya=self.saw_goriya,
            frames=self.frames,
        )
        if action.reason.startswith("unexpected_room"):
            if snap.screen == ENTRY_SCREEN:
                return self.mark_fail("south_backtrack")
            return self.mark_fail(action.reason)
        return action

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "spec_id": self.spec_id,
            "stage_id": self.spec_id,
            "dest_screen": self.dest,
            "saw_goriya": self.saw_goriya,
            "east_opened_frame": self.east_opened_frame,
            "obj_types": list(self.obj_types),
            "evidence": "fixture-live",
            "route_eligible": False,
            "door": "RIGHT",
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
    "EAST_APPROACH_X",
    "EAST_BAND_Y",
    "EAST_DOOR",
    "EAST_DOOR_X",
    "EAST_DOOR_Y",
    "ENTRY_SCREEN",
    "NORTH_DOOR",
    "NORTH_DOOR_X",
    "NORTH_DOOR_Y",
    "ROOM_69",
    "SOUTH_MOUTH_Y",
    "EntryNorthDoorController",
    "HungryGoriyaGateController",
    "Level7PathController",
    "RedCandlePickupController",
    "Room69EastController",
    "UnverifiedLevel7PathController",
    "east_of_room69_ram_id",
    "east_route_step",
    "live_goriyas",
    "north_door_79_step",
    "north_of_entry_ram_id",
    "room69_east_step",
    "unverified_path_controller",
]
