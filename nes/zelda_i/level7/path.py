"""Level 7 one-frame path policies.

``north_door_79_step`` / ``EntryNorthDoorController`` walk ``0x79`` south
mouth to live north dest ``0x69``.  ``room69_east_step`` /
``Room69EastController`` clear the ``0x69`` goriyas then walk the east
doorway, which is an **OPEN** gate (black passage on the spawn frame) —
``cur_opened_doors`` never sets its RIGHT bit, exactly like the ``0x79``
north door.  The post-clear traverse is a deterministic waypoint micro,
not occupancy: the ``0x69`` centre row is walled at ``y=141`` and a
per-pixel grid boxes Link in after four graded misses on one cell.

``room_6a_east_step`` / ``Room6AEastController`` walk the unlit ``0x6A``
KEESE room (entry pin carries Candle 0) from the west mouth to live east
dest ``0x6B``.  Its ``y=141`` centre band is walled at ``x=48``; the top
of the room is an open corridor, so the traverse rises the west column to
``y=93``, crosses, drops the east column to the door row and pushes the
OPEN east doorway.  Keese never block the doorway.

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
    KEESE_TYPE,
    WALLMASTER_TYPE,
    EnemyKind,
    engagement_hint,
    is_off_wall,
    is_projectile,
    live_among,
)
from zelda_i.dungeon.bomb_wall import BombWallController, BombWallPhase
from zelda_i.dungeon.engine import AliveRule
from zelda_i.dungeon.hop_controller import (
    HopController,
    WAIT_SCROLL_B,
    dungeon_align_then_push,
)
from zelda_i.dungeon.pause_select import B_SLOT_BOMBS, PauseSelectController
from zelda_i.level7.graph import (
    CANDLE_PUSH,
    DIGDOGGER_1,
    GORIYA_HINT,
    GORIYA_POST_RUPEE,
    GORIYA_PRE_DIG,
    HIDDEN_RUPEES,
    KEESE,
    LEVEL7_ROOM_BY_ID,
    MOLDORMS,
    OLD_MAN_NOSE,
    TIP_OF_NOSE,
)
from zelda_i.ram import (
    PLAY_MODE,
    ZeldaObject,
    ZeldaSnapshot,
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
# Kill-clear budget ahead of the west BOMB wall, plus the wall's own
# BOMB_N_MAX_FRAMES (16000) once goriyas are cleared.
ROOM69_WEST_BOMB_MAX_FRAMES = 22000
# 0x6A KEESE dark room: west mouth spawn is (16,141). The y=141 centre band
# is walled just past the door corridor (x=48, tile 0xB1), but the top of the
# room is an open horizontal corridor — a blind y-scan crossed y=93 from x=40
# to x=208. Candle 0 on the entry pin keeps the room unlit, so the traverse
# is a deterministic waypoint micro on that top band, mirroring the 0x69 east
# route: rise the west column to y=93, cross to the east column, drop to the
# door row, push RIGHT through the OPEN east doorway.
ROOM_6A = 0x6A
ROOM_6A_WEST_MOUTH = (16, 141)
ROOM_6A_DOOR_Y = 141
ROOM_6A_TOP_BAND_Y = 93
ROOM_6A_EAST_COLUMN_X = 200
ROOM_6A_EAST_PLANE = 224
ROOM_6A_MOUTH_X = 32
ROOM6A_EAST_MAX_FRAMES = 4000
# 0x6B GORIYA_HINT: entered at the west mouth (16,141) from 0x6A.  A central X
# of diamond blocks walls the y=141 centre band at x~96; the y=109 band is
# clear x=32..208.  The east dest is live $EB=0x6C (DIGDOGGER_1) — an OPEN
# doorway at y=141 on the east side.  The traverse (after the six goriya 0x05
# are cleared, upstream) rides the y=109 band east past the central X, drops
# the east column to the door row, and pushes RIGHT.  2/2 byte-identical
# (recordings/6b_right_v1.json / 6b_right_v2.json, 283f).
ROOM_6B = 0x6B
ROOM_6B_WEST_MOUTH = (16, 141)
ROOM_6B_DOOR_Y = 141
ROOM_6B_MID_BAND_Y = 109
ROOM_6B_EAST_COLUMN_X = 200
ROOM_6B_EAST_PLANE = 224
ROOM_6B_MOUTH_X = 32
ROOM6B_EAST_MAX_FRAMES = 4000
# 0x6B NORTH door: an OPEN notch at x~118 on the y=93 top band (NOT x=128 —
# solid), opens after the six 0x6B goriya 0x05 are cleared upstream.  Dest is
# live $EB=0x5B (OLD_MAN_NOSE — bubble 0x40 + statue 0x50), a dead-end spur.
# 2/2 byte-identical (recordings/6b_north_dest_v1/v2.json, arrived frame 189).
ROOM_6B_NORTH_NOTCH_X = 118
ROOM_6B_TOP_BAND_Y = 93
ROOM6B_NORTH_MAX_FRAMES = 3000
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


def east_of_room6a_ram_id() -> int | None:
    """Live ``$EB`` of the room east of ``0x6A``, or None until observed."""
    return LEVEL7_ROOM_BY_ID[GORIYA_HINT].ram_id


def east_of_room6b_ram_id() -> int | None:
    """Live ``$EB`` of the room east of ``0x6B``, or None until observed."""
    return LEVEL7_ROOM_BY_ID[DIGDOGGER_1].ram_id


def north_of_room6b_ram_id() -> int | None:
    """Live ``$EB`` of the room north of ``0x6B`` (OLD_MAN_NOSE spur)."""
    return LEVEL7_ROOM_BY_ID[OLD_MAN_NOSE].ram_id


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


def live_keese(snap: ZeldaSnapshot) -> tuple[ZeldaObject, ...]:
    """Type-live Keese. HP stays 0 while alive — never TYPE_AND_HP."""
    return tuple(
        obj
        for obj in live_among(_combatants(snap), AliveRule.TYPE)
        if (int(obj.type_id) & 0xFF) == KEESE_TYPE
    )



def _projectiles(snap: ZeldaSnapshot) -> tuple[ZeldaObject, ...]:
    return tuple(obj for obj in _combatants(snap) if is_projectile(obj))


def _leave_wall(
    snap: ZeldaSnapshot,
    *,
    inland_x: tuple[int, int] = _INLAND_X,
    inland_y: tuple[int, int] = _INLAND_Y,
) -> FrameAction | None:
    """Step toward the playable interior. Do not chase the west wall."""
    x, y = int(snap.link_x), int(snap.link_y)
    if x < inland_x[0]:
        direction = "RIGHT"
    elif x > inland_x[1]:
        direction = "LEFT"
    elif y < inland_y[0]:
        direction = "DOWN"
    elif y > inland_y[1]:
        direction = "UP"
    else:
        return None
    step = _leave_wall_lattice(x, y, direction, inland_x, inland_y)
    if step is not None:
        return FrameAction(nes_action(step), "leave_wall_lattice")
    return FrameAction(nes_action(direction), "leave_wall")


_STEP = {"UP": (0, -8), "DOWN": (0, 8), "LEFT": (-8, 0), "RIGHT": (8, 0)}


def _leave_wall_lattice(
    x: int, y: int, direction: str, inland_x: tuple[int, int], inland_y: tuple[int, int]
) -> str | None:
    """Lattice route inland, only when the cardinal step is into a block.

    0x38 (power-on gathered spine) pressed UP at (200,181) for 14000f: x=200
    is not one of the room's open columns. An open cardinal keeps the old
    step, so rooms that already leave the wall walk it unchanged.
    """
    from zelda_i.dungeon.tilemap import has_room_tile_map, ow_walkable_nodes
    from zelda_i.walk import live_env
    from zelda_i.walk.physics import lattice_route, lattice_starts, lattice_step

    env = live_env.current()
    if env is None or not has_room_tile_map(env.get_ram()):
        return None
    nodes = ow_walkable_nodes(env.get_ram(), overworld=False)
    starts = [n for n in lattice_starts(x, y) if n in nodes]
    if not starts:
        return None
    sx, sy = min(starts, key=lambda n: abs(n[0] - x) + abs(n[1] - y))
    dx, dy = _STEP[direction]
    if (sx + dx, sy + dy) in nodes:
        return None
    goals = {
        n for n in nodes
        if inland_x[0] <= n[0] <= inland_x[1] and inland_y[0] <= n[1] <= inland_y[1]
    }
    if not goals:
        return None
    near = min(abs(n[0] - x) + abs(n[1] - y) for n in goals)
    goals = {n for n in goals if abs(n[0] - x) + abs(n[1] - y) <= near + 16}
    route = lattice_route(nodes, (x, y), goals)
    if not route:
        return None
    return lattice_step(x, y, route[0])


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
    snap: ZeldaSnapshot,
    target: ZeldaObject,
    *,
    frames: int,
    inland_x: tuple[int, int] = _INLAND_X,
    inland_y: tuple[int, int] = _INLAND_Y,
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
    leave = _leave_wall(snap, inland_x=inland_x, inland_y=inland_y)
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


def room_6a_east_step(
    snap: ZeldaSnapshot,
    *,
    dest: int | None = None,
) -> FrameAction:
    """One frame of 0x6A (KEESE dark room) west mouth → east door.

    The room is unlit (entry pin carries Candle 0) and the east exit is an
    OPEN doorway, so nothing waits on light or a door bit. The spawn mouth
    and the east door share the ``y=141`` band, so the traverse is a blind
    straight push on that band; a Keese knock only bumps ``y`` off it and
    the align step steers back. Keese never block the doorway.
    """
    if snap.level != LEVEL7:
        return FrameAction(nes_idle_action(), "wait_level7")
    if snap.transitioning:
        return FrameAction(nes_action("RIGHT"), "east6a_scroll")
    if snap.mode != PLAY_MODE:
        return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
    if dest is not None and snap.screen == dest:
        return FrameAction(nes_idle_action(), "east6a_arrived")
    if snap.screen != ROOM_6A:
        return FrameAction(nes_idle_action(), f"unexpected_room_0x{snap.screen:02x}")

    x, y = int(snap.link_x), int(snap.link_y)
    if x < ROOM_6A_MOUTH_X:
        return FrameAction(nes_action("RIGHT"), "east6a_leave_mouth")
    # Rise the west column while still west of the east drop point.
    if x < ROOM_6A_EAST_COLUMN_X - NORTH_X_TOL:
        if y > ROOM_6A_TOP_BAND_Y + DOOR_Y_TOL:
            return FrameAction(nes_action("UP"), "east6a_rise")
        return FrameAction(nes_action("RIGHT"), "east6a_cross")
    # On the east column: drop to the door row, then push the OPEN doorway.
    if abs(y - ROOM_6A_DOOR_Y) > DOOR_Y_TOL:
        btn = "UP" if y > ROOM_6A_DOOR_Y else "DOWN"
        return FrameAction(nes_action(btn), "east6a_drop_y")
    return dungeon_align_then_push(
        snap,
        push_dir="RIGHT",
        target_y=ROOM_6A_DOOR_Y,
        y_tol=DOOR_Y_TOL,
        door_plane=ROOM_6A_EAST_PLANE,
        reason="east6a",
    )


@dataclass(kw_only=True)
class Room6AEastController(HopController):
    """0x6A west mouth: blind y=141 push through the OPEN east doorway."""

    spec_id: str = "level7_room6a_east"
    max_frames: int = ROOM6A_EAST_MAX_FRAMES
    require_level: int = LEVEL7
    done_reason: str = "left_0x6a"
    dest: int | None = field(default_factory=east_of_room6a_ram_id)

    @property
    def stage_id(self) -> str:
        return self.spec_id

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        if (
            snap.level != LEVEL7
            or snap.mode != PLAY_MODE
            or snap.transitioning
            or snap.screen in {ENTRY_SCREEN, ROOM_69, ROOM_6A}
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
            f"_mode={snap.mode}"
        )

    def scroll_action(self, snap: ZeldaSnapshot) -> FrameAction:
        if snap.screen == ROOM_69:
            return self.mark_fail("west_backtrack")
        return FrameAction(nes_action("RIGHT"), "east6a_scroll")

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        action = room_6a_east_step(snap, dest=self.dest)
        if action.reason.startswith("unexpected_room"):
            if snap.screen == ROOM_69:
                return self.mark_fail("west_backtrack")
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
            "evidence": "fixture-live",
            "route_eligible": False,
            "door": "RIGHT",
        }


def room_6b_east_step(
    snap: ZeldaSnapshot,
    *,
    dest: int | None = None,
) -> FrameAction:
    """One frame of 0x6B (GORIYA_HINT) west mouth → live east dest 0x6C.

    Assumes the six goriya ``0x05`` are already cleared upstream (the recon
    fixture / a prior kill-clear).  A central X of diamond blocks walls the
    ``y=141`` centre band at ``x~96``; the ``y=109`` band is clear
    ``x=32..208``.  So the traverse rides the ``y=109`` band east past the
    central X, drops the east column to the door row, then pushes the OPEN
    east doorway (plane ``x=224``).
    """
    if snap.level != LEVEL7:
        return FrameAction(nes_idle_action(), "wait_level7")
    if snap.transitioning:
        return FrameAction(nes_action("RIGHT"), "east6b_scroll")
    if snap.mode != PLAY_MODE:
        return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
    if dest is not None and snap.screen == dest:
        return FrameAction(nes_idle_action(), "east6b_arrived")
    if snap.screen != ROOM_6B:
        return FrameAction(nes_idle_action(), f"unexpected_room_0x{snap.screen:02x}")

    x, y = int(snap.link_x), int(snap.link_y)
    if x < ROOM_6B_MOUTH_X:
        return FrameAction(nes_action("RIGHT"), "east6b_leave_mouth")
    # Ride the y=109 mid band east until on the east drop column.
    if x < ROOM_6B_EAST_COLUMN_X - NORTH_X_TOL:
        if abs(y - ROOM_6B_MID_BAND_Y) > DOOR_Y_TOL:
            btn = "UP" if y > ROOM_6B_MID_BAND_Y else "DOWN"
            return FrameAction(nes_action(btn), "east6b_band_y")
        return FrameAction(nes_action("RIGHT"), "east6b_cross")
    # On the east column: drop to the door row, then push the OPEN doorway.
    if abs(y - ROOM_6B_DOOR_Y) > DOOR_Y_TOL:
        btn = "UP" if y > ROOM_6B_DOOR_Y else "DOWN"
        return FrameAction(nes_action(btn), "east6b_drop_y")
    return dungeon_align_then_push(
        snap,
        push_dir="RIGHT",
        target_y=ROOM_6B_DOOR_Y,
        y_tol=DOOR_Y_TOL,
        door_plane=ROOM_6B_EAST_PLANE,
        reason="east6b",
    )


@dataclass(kw_only=True)
class Room6BEastController(HopController):
    """0x6B west mouth → OPEN east doorway to live dest 0x6C (DIGDOGGER_1).

    Goriyas are assumed cleared upstream.  Waypoint micro: ride the ``y=109``
    band east past the central X of diamond blocks, drop the east column to
    ``y=141``, push RIGHT through the OPEN east doorway.
    """

    spec_id: str = "level7_room6b_east"
    max_frames: int = ROOM6B_EAST_MAX_FRAMES
    require_level: int = LEVEL7
    done_reason: str = "left_0x6b"
    dest: int | None = field(default_factory=east_of_room6b_ram_id)

    @property
    def stage_id(self) -> str:
        return self.spec_id

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        if (
            snap.level != LEVEL7
            or snap.mode != PLAY_MODE
            or snap.transitioning
            or snap.screen in {ENTRY_SCREEN, ROOM_69, ROOM_6A, ROOM_6B}
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
            f"_mode={snap.mode}"
        )

    def scroll_action(self, snap: ZeldaSnapshot) -> FrameAction:
        if snap.screen == ROOM_6A:
            return self.mark_fail("west_backtrack")
        return FrameAction(nes_action("RIGHT"), "east6b_scroll")

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        action = room_6b_east_step(snap, dest=self.dest)
        if action.reason.startswith("unexpected_room"):
            if snap.screen == ROOM_6A:
                return self.mark_fail("west_backtrack")
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
            "evidence": "fixture-live",
            "route_eligible": False,
            "door": "RIGHT",
        }


@dataclass(frozen=True)
class Level7BombWall:
    """Geometry for a Level 7 bomb-wall traverse (``dungeon.bomb_wall`` contract).

    Satisfies ``BombWallLike``: ``room`` / ``stand`` / ``face`` / ``opens_to``.
    """

    room: int
    stand: tuple[int, int]
    face: str
    opens_to: int


# 0x69 west BOMB wall -> $EB=0x68 (source GORIYA_BOMB_HUB LEFT -> KEESE_TRAPS).
# Stand ~(44,141) facing LEFT; cur_opened_doors LEFT bit sets. The candle-path
# branch after the Stalfos-key dead-end. 2/2 (recordings/69_branch_v2/v3.json).
L7_ROOM69_WEST_BOMB = Level7BombWall(
    room=ROOM_69, stand=(44, 141), face="LEFT", opens_to=0x68
)
# The 69_branch_v2/v3 fixture spawned Link near the stand already, so its
# naive TO_STAND walk never had to cross the room. Live arrival is the south
# mouth (120,205) (EntryNorthDoorController leftover); a naive goto from
# there drags the dominant-axis walk straight into the centre-row diamond
# blocks flanking x~96 at y~141 (same obstruction ``east_route_step`` routes
# around via the y=109 band, mirrored here). Rise the open x=120 column to
# the EAST_BAND_Y band, cross west on that open row, then drop the west
# column to the stand.
L7_ROOM69_WEST_APPROACH = (
    (NORTH_DOOR_X, EAST_BAND_Y),
    (L7_ROOM69_WEST_BOMB.stand[0], EAST_BAND_Y),
)


def _room69_west_bomb_wall() -> BombWallController:
    return BombWallController(
        wall=L7_ROOM69_WEST_BOMB,
        level=LEVEL7,
        select_item=B_SLOT_BOMBS,
        approach_waypoints=L7_ROOM69_WEST_APPROACH,
    )


@dataclass(kw_only=True)
class Room69WestBombController:
    """0x69 south mouth: kill-clear live goriyas, then bomb the west wall.

    Live census shows the five 0x69 goriyas HP=0 for their first few frames
    (still materializing), then HP=80 and throwing boomerangs. A naive
    approach-and-place reaches the stand fine but the goriyas interrupt the
    bomb placement (``bomb_not_consumed``: Link is knocked off the B-press
    before the blast). Clear them first with the same fight loop
    ``Room69EastController`` uses.

    ``bomb_not_consumed`` persisted even after the clear: the pause-close
    animation is still visually on-screen 24 frames (``CLOSE_SETTLE_FRAMES``)
    after the START press that closes it — the B-press-to-place lands while
    the game is still fading back from the item-select overlay and is
    dropped, and every WAIT-phase input after that lands on the same dead
    overlay (Link's xy never moves during the whole WAIT window). Placing
    ``PauseSelectController`` here, right after the clear and well before the
    walk to the stand, gives that overlay hundreds of frames of approach/
    ``TO_STAND``/``FACE`` walk to finish closing; by the time
    ``BombWallController`` reaches its own ``SELECT`` phase, ``$0656`` already
    reads bombs, so its internal pause-select short-circuits on the very
    first frame (no second pause, no second close-animation race) straight
    into ``PLACE``.
    """

    spec_id: str = "level7_room69_west_bomb"
    max_frames: int = ROOM69_WEST_BOMB_MAX_FRAMES
    frames: int = 0
    saw_goriya: bool = False
    preselect_done: bool = False
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)
    bomb: BombWallController = field(default_factory=_room69_west_bomb_wall)
    _preselect: PauseSelectController | None = field(
        default=None, init=False, repr=False
    )
    _env: Any = field(default=None, init=False, repr=False)

    @property
    def stage_id(self) -> str:
        return self.spec_id

    @property
    def select_item(self) -> int | None:
        return self.bomb.select_item

    @property
    def stand(self) -> tuple[int, int]:
        return self.bomb.stand

    @property
    def face(self) -> str:
        return self.bomb.face

    @property
    def from_room(self) -> int:
        return self.bomb.from_room

    @property
    def to_room(self) -> int:
        return self.bomb.to_room

    def bind_env(self, env: Any) -> None:
        self._env = env
        self.bomb.bind_env(env)

    def _note(self, note: str) -> None:
        if not self.notes or self.notes[-1] != note:
            self.notes.append(note)

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        if self.success or self.failed:
            return FrameAction(nes_idle_action(), "done" if self.success else "failed")
        if self.frames >= self.max_frames:
            self.failed = True
            self._note("timeout")
            return FrameAction(nes_idle_action(), "timeout")
        if snap.level != LEVEL7:
            return FrameAction(nes_idle_action(), "wait_level7")
        if snap.transitioning:
            return FrameAction(nes_idle_action(), "settle")
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")

        if snap.screen == ROOM_69:
            live = live_goriyas(snap)
            if live:
                self.saw_goriya = True
                target = nearest_enemy(snap.link_x, snap.link_y, live)
                if target is None:
                    return FrameAction(nes_idle_action(), "goriya_missing")
                return _goriya_fight(snap, target, frames=self.frames)
            if not self.saw_goriya:
                return FrameAction(nes_idle_action(), "spawn_wait")
            if "cleared" not in self.notes:
                self._note("cleared")
            if not self.preselect_done:
                if self._preselect is None:
                    self._preselect = PauseSelectController(
                        want=B_SLOT_BOMBS, name="bombs"
                    )
                    self._preselect.bind_env(self._env)
                driven = self._preselect.drive(snap)
                for note in self._preselect.notes:
                    if note not in self.notes:
                        self.notes.append(note)
                if self._preselect.failed:
                    self.failed = True
                    self._note(self._preselect.fail_reason or "preselect_failed")
                    return FrameAction(nes_idle_action(), "preselect_failed")
                if driven is not None:
                    return driven
                self.preselect_done = True
                self._note("preselected_bombs")

        action = self.bomb.step(snap)
        for note in self.bomb.notes:
            if note not in self.notes:
                self.notes.append(note)
        if self.bomb.phase is BombWallPhase.DONE:
            self.success = True
        elif self.bomb.phase is BombWallPhase.FAILED:
            self.failed = True
        return action

    def report(self) -> dict[str, Any]:
        rep = dict(self.bomb.report())
        rep["success"] = self.success
        rep["failed"] = self.failed
        rep["frames"] = self.frames
        rep["saw_goriya"] = self.saw_goriya
        rep["spec_id"] = self.spec_id
        rep["stage_id"] = self.spec_id
        rep["notes"] = list(self.notes)
        return rep


def room_6b_north_step(
    snap: ZeldaSnapshot,
    *,
    dest: int | None = None,
) -> FrameAction:
    """One frame of 0x6B west mouth → OPEN north notch → live dest 0x5B.

    Assumes the six 0x6B goriya are cleared upstream.  Route: ride the
    ``y=109`` mid band to ``x~118`` (past the central X of diamond blocks),
    rise to the ``y=93`` top band, push UP through the notch.
    """
    if snap.level != LEVEL7:
        return FrameAction(nes_idle_action(), "wait_level7")
    if snap.transitioning:
        return FrameAction(nes_action("UP"), "north6b_scroll")
    if snap.mode != PLAY_MODE:
        return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
    if dest is not None and snap.screen == dest:
        return FrameAction(nes_idle_action(), "north6b_arrived")
    if snap.screen != ROOM_6B:
        return FrameAction(nes_idle_action(), f"unexpected_room_0x{snap.screen:02x}")

    x, y = int(snap.link_x), int(snap.link_y)
    if abs(x - ROOM_6B_NORTH_NOTCH_X) > NORTH_X_TOL and y > ROOM_6B_TOP_BAND_Y + DOOR_Y_TOL:
        # still below the top band: align x on the mid band first
        if abs(y - ROOM_6B_MID_BAND_Y) > DOOR_Y_TOL:
            btn = "UP" if y > ROOM_6B_MID_BAND_Y else "DOWN"
            return FrameAction(nes_action(btn), "north6b_band_y")
        btn = "LEFT" if x > ROOM_6B_NORTH_NOTCH_X else "RIGHT"
        return FrameAction(nes_action(btn), "north6b_align_x")
    if abs(x - ROOM_6B_NORTH_NOTCH_X) > NORTH_X_TOL:
        btn = "LEFT" if x > ROOM_6B_NORTH_NOTCH_X else "RIGHT"
        return FrameAction(nes_action(btn), "north6b_notch_x")
    return FrameAction(nes_action("UP"), "north6b_push")


@dataclass(kw_only=True)
class Room6BNorthController(HopController):
    """0x6B west mouth → OPEN north notch (x~118) to live dest 0x5B.

    Dead-end spur (OLD_MAN_NOSE).  Goriyas assumed cleared upstream.
    """

    spec_id: str = "level7_room6b_north"
    max_frames: int = ROOM6B_NORTH_MAX_FRAMES
    require_level: int = LEVEL7
    done_reason: str = "left_0x6b_north"
    dest: int | None = field(default_factory=north_of_room6b_ram_id)

    @property
    def stage_id(self) -> str:
        return self.spec_id

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        if (
            snap.level != LEVEL7
            or snap.mode != PLAY_MODE
            or snap.transitioning
            or snap.screen in {ENTRY_SCREEN, ROOM_69, ROOM_6A, ROOM_6B}
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
            f"_mode={snap.mode}"
        )

    def scroll_action(self, snap: ZeldaSnapshot) -> FrameAction:
        return FrameAction(nes_action("UP"), "north6b_scroll")

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        action = room_6b_north_step(snap, dest=self.dest)
        if action.reason.startswith("unexpected_room"):
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
            "evidence": "fixture-live",
            "route_eligible": False,
            "door": "UP",
        }


ROOM_18 = 0x18
ROOM_08 = 0x08
ROOM_19 = 0x19
ROOM_1A = 0x1A


def north_of_room18_ram_id() -> int | None:
    """Live ``$EB`` of the room north of ``0x18`` (HIDDEN_RUPEES)."""
    return LEVEL7_ROOM_BY_ID[HIDDEN_RUPEES].ram_id


def east_of_room08_ram_id() -> int | None:
    """Live ``$EB`` of the room east of ``0x08`` (GORIYA_POST_RUPEE)."""
    return LEVEL7_ROOM_BY_ID[GORIYA_POST_RUPEE].ram_id


def east_of_room19_ram_id() -> int | None:
    """Live ``$EB`` of the room east of ``0x19`` (CANDLE_PUSH)."""
    return LEVEL7_ROOM_BY_ID[CANDLE_PUSH].ram_id


# 0x18 MAP north BOMB wall -> $EB=0x08 (HIDDEN_RUPEES). Stand (120,93)
# face UP. 2/2 (recordings/18_bn_v2/v3.json). Skip MAP_EAST_LOCK.
L7_ROOM18_NORTH_BOMB = Level7BombWall(
    room=ROOM_18, stand=(120, 93), face="UP", opens_to=0x08
)
# 0x08 diamond-cross east BOMB wall -> $EB=0x09. South-band then east
# column to (208,141) face RIGHT. 2/2 (08_be_v2/v3.json).
L7_ROOM08_EAST_BOMB = Level7BombWall(
    room=ROOM_08, stand=(208, 141), face="RIGHT", opens_to=0x09
)
L7_ROOM08_EAST_APPROACH = ((200, 189), (200, 141), (208, 141))
# 0x19 north mouth (from 0x09) east BOMB wall -> $EB=0x1A. North-east around
# (208,93) then down the east column to (208,141) face RIGHT.
L7_ROOM19_EAST_BOMB = Level7BombWall(
    room=ROOM_19, stand=(208, 141), face="RIGHT", opens_to=0x1A
)
L7_ROOM19_EAST_APPROACH = ((208, 93), (208, 141))


def east_of_room1a_ram_id() -> int | None:
    """Live ``$EB`` of the room east of ``0x1A`` (GORIYA_PRE_DIG)."""
    return LEVEL7_ROOM_BY_ID[GORIYA_PRE_DIG].ram_id


# 0x1A CANDLE_PUSH east BOMB wall -> $EB=0x1B. South-around from the
# cellar-return leftover (96,157): (96,189)->(208,189)->(208,141) face
# RIGHT. 2/2 (1a_be_v1/v2).
L7_ROOM1A_EAST_BOMB = Level7BombWall(
    room=ROOM_1A, stand=(208, 141), face="RIGHT", opens_to=0x1B
)
L7_ROOM1A_EAST_APPROACH = ((96, 189), (208, 189), (208, 141))


ROOM_0C = 0x0C
ROOM_0D = 0x0D


def east_of_room0c_ram_id() -> int | None:
    """Live ``$EB`` of the room east of ``0x0C`` (TIP_OF_NOSE)."""
    return LEVEL7_ROOM_BY_ID[TIP_OF_NOSE].ram_id


# 0x0C DODONGOS_BOSS_PATH east BOMB wall -> $EB=0x0D. East-around the
# y=141 tile-181 mass: (120,165)->(200,165)->(200,141)->(208,141) face
# RIGHT. 2/2 (0c_be_v2/v3). Dead: y=141 centre RIGHT.
L7_ROOM0C_EAST_BOMB = Level7BombWall(
    room=ROOM_0C, stand=(208, 141), face="RIGHT", opens_to=0x0D
)
L7_ROOM0C_EAST_APPROACH = ((120, 165), (200, 165), (200, 141), (208, 141))


def _reexport(module: str, *names: str) -> dict[str, str]:
    return {name: module for name in names}


_REEXPORT: dict[str, str] = {}
_REEXPORT.update(
    _reexport(
        "zelda_i.level7.west",
        "ROOM_09",
        "ROOM_49",
        "ROOM_58",
        "ROOM_58_NORTH_EAST_X",
        "ROOM_58_NORTH_MID_Y",
        "ROOM_58_NORTH_TOP_Y",
        "ROOM_58_NORTH_X",
        "ROOM_59",
        "ROOM_68",
        "ROOM_68_MID_Y",
        "ROOM_68_SAFE_X",
        "ROOM_68_SOUTH_X",
        "ROOM_68_TRAP_ROW_Y",
        "ROOM_6C",
        "Room6CEastController",
        "Room68NorthController",
        "Room68DownController",
        "Room49UpController",
        "Room09DownController",
        "Room58EastController",
        "Room58NorthController",
        "Room59UpController",
        "east_of_room6c_ram_id",
        "east_of_room58_ram_id",
        "north_of_room68_ram_id",
        "south_of_room68_ram_id",
        "north_of_room58_ram_id",
        "south_of_room09_ram_id",
        "north_of_room49_ram_id",
        "north_of_room59_ram_id",
        "room_49_up_step",
        "room_09_down_step",
        "room_6c_east_step",
        "room_68_north_step",
        "room_68_down_step",
        "room_58_north_step",
    )
)
_REEXPORT.update(
    _reexport(
        "zelda_i.level7.digdogger",
        "ROOM_39",
        "Room39LeftController",
        "Room4AReturnController",
        "Room1BKeyEastController",
        "play_of_room4a_ram_id",
        "room_4a_return_step",
        "east_of_room1b_ram_id",
        "room_1b_key_east_step",
        "west_of_room39_ram_id",
        "room_39_left_step",
    )
)
_REEXPORT.update(
    _reexport(
        "zelda_i.level7.hungry",
        "ROOM_38",
        "Room38UpController",
        "north_of_room38_ram_id",
        "room_38_up_step",
    )
)
_REEXPORT.update(
    _reexport(
        "zelda_i.level7.cellar",
        "Room1ACandleController",
        "cellar_of_room1a_ram_id",
    )
)
_REEXPORT.update(
    _reexport(
        "zelda_i.level7.stairs0d",
        "Room0DClearController",
        "room_0d_clear_step",
        "ROOM_0D_BLOCK",
        "ROOM_0D_BLOCK_STAND",
        "ROOM_0D_BLOCK_AFTER_RIGHT",
        "ROOM_0D_BLOCK_CELL_AFTER_RIGHT",
        "ROOM_0D_STAIR_CELL",
        "ROOM_0D_NORTH_ARM",
        "ROOM_0D_STAIR_WARP_HYP",
    )
)


def __getattr__(name: str):
    module_name = _REEXPORT.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_REEXPORT) | set(__all__))


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
    "ROOM_6A",
    "ROOM_6A_EAST_PLANE",
    "ROOM_6A_WEST_MOUTH",
    "ROOM_6B",
    "ROOM_6B_EAST_PLANE",
    "ROOM_6B_WEST_MOUTH",
    "SOUTH_MOUTH_Y",
    "L7_ROOM69_WEST_APPROACH",
    "L7_ROOM69_WEST_BOMB",
    "L7_ROOM18_NORTH_BOMB",
    "L7_ROOM08_EAST_BOMB",
    "L7_ROOM08_EAST_APPROACH",
    "L7_ROOM19_EAST_BOMB",
    "L7_ROOM19_EAST_APPROACH",
    "L7_ROOM1A_EAST_BOMB",
    "L7_ROOM1A_EAST_APPROACH",
    "L7_ROOM0C_EAST_BOMB",
    "L7_ROOM0C_EAST_APPROACH",
    "east_of_room0c_ram_id",
    "EntryNorthDoorController",
    "Level7BombWall",
    "Level7PathController",
    "Room69EastController",
    "Room69WestBombController",
    "Room6AEastController",
    "Room6BEastController",
    "Room6BNorthController",
    "UnverifiedLevel7PathController",
    "east_of_room69_ram_id",
    "east_of_room6a_ram_id",
    "east_of_room6b_ram_id",
    "north_of_room6b_ram_id",
    "north_of_room18_ram_id",
    "east_of_room08_ram_id",
    "east_of_room19_ram_id",
    "east_of_room1a_ram_id",
    "east_route_step",
    "room_6a_east_step",
    "room_6b_east_step",
    "room_6b_north_step",
    "live_goriyas",
    "north_door_79_step",
    "north_of_entry_ram_id",
    "room69_east_step",
    "unverified_path_controller",
]
__all__ += list(_REEXPORT)
