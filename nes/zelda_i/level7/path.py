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
    EnemyKind,
    engagement_hint,
    is_projectile,
    live_among,
)
from zelda_i.dungeon.engine import AliveRule
from zelda_i.dungeon.hop_controller import (
    HopController,
    WAIT_SCROLL_B,
    dungeon_align_then_push,
)
from zelda_i.level7.graph import (
    BOMB_UPGRADE,
    CANDLE_PUSH,
    DIGDOGGER_1,
    FORCED_DIGDOGGER,
    DIGDOGGER_2,
    DODONGOS_UPGRADE,
    GORIYA_BUBBLE,
    GORIYA_COMPASS,
    GORIYA_POST_RUPEE,
    GORIYA_PRE_DIG,
    GORIYA_PRE_HUNGRY,
    HIDDEN_RUPEES,
    HUNGRY_GORIYA,
    OLD_MAN_NOSE,
    RED_CANDLE_CELLAR,
    ROPES_KEY,
    STALFOS_KEY,
    WEST_LOCK_SKIP,
    GORIYA_HINT,
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
# 0x6C (DIGDOGGER_1): entry (16,141) west mouth, digdogger 0x38 + statue 0x55.
# East dest is live $EB=0x6D (STALFOS_KEY — stalfos 0x2a + small_key 0x19,
# a dead-end).  Traverse rides the y=141 band east; a digdogger bump nudges
# Link through the east door.  2/2 byte-identical
# (recordings/6c_right_v1/v2.json).
ROOM_6C = 0x6C
ROOM_6C_DOOR_Y = 141
ROOM_6C_EAST_PLANE = 224
ROOM6C_EAST_MAX_FRAMES = 4000
# 0x68 (KEESE_TRAPS): dark, 4 blade traps 0x49 (corners) + 4 keese 0x1b.
# Reached via the 0x69 west bomb wall (Link enters ~(208,93) NE).  The north
# door to live $EB=0x58 (DODONGOS_UPGRADE) is OPEN — align x=120 on the top,
# push UP.  2/2 byte-identical (recordings/68_up_v1/v2.json).
ROOM_68 = 0x68
ROOM_68_NORTH_X = 120
ROOM_68_TOP_BAND_Y = 93
ROOM68_NORTH_MAX_FRAMES = 4000
# 0x68 DOWN: OPEN south door to live $EB=0x78 (ROPES_KEY — ropes 0x28 +
# small_key 0x19 on the floor, dead-end).  Blade traps 0x49 in the four
# corners; OccupancyWalker poisons on knockback so this is a waypoint
# micro: peel west to x=160 (off the east trap column), drop to y=141
# (between trap rows y~93 / y~189), align x=120, push DOWN.  If knocked
# onto the y~189 trap row off-x, rise first.  2/2 byte-identical
# (recordings/68_down_v3/v4.json, arrived frame 297).
ROOM_68_SOUTH_X = 120
ROOM_68_SAFE_X = 160
ROOM_68_MID_Y = 141
ROOM_68_TRAP_ROW_Y = 189
ROOM68_DOWN_MAX_FRAMES = 4000
# 0x58 (DODONGOS_UPGRADE): dark, 3x invulnerable roamers 0x31 (hp 240) —
# dodge, do NOT try to kill.  Link spawns bottom (120,205).  The EAST door
# to live $EB=0x59 (GORIYA_COMPASS) is OPEN (keys unchanged).  A central
# structure walls the y=141 band west of x~129, so the route climbs the
# east-open column: (120,165) → (200,165) → (200,141) → push RIGHT.
# 2/2 byte-identical (recordings/58_east_v2/v3.json).
ROOM_58 = 0x58
ROOM_58_EAST_COLUMN_X = 200
ROOM_58_MID_Y = 165
ROOM_58_DOOR_Y = 141
ROOM58_EAST_MAX_FRAMES = 4000
# 0x58 NORTH: KEY door (keys 4→3) to live $EB=0x48 (BOMB_UPGRADE — bubble
# 0x40 + 0x4f, old-man "I BET YOU'D LIKE TO HAVE -100" bomb-capacity,
# dead-end).  A central 2-block mass walls the x=120 column around y=141,
# so east-around: climb y=165, RIGHT to x=160, UP to y=93, align x=120,
# push UP.  Dodge 3x invuln 0x31; do not fight.  Do NOT write max_bombs
# (upgrade is a 100-rupee purchase, unobserved).  2/2 byte-identical
# (recordings/58_north_v2/v3.json, arrived frame 337).
ROOM_58_NORTH_X = 120
ROOM_58_NORTH_EAST_X = 160
ROOM_58_NORTH_MID_Y = 165
ROOM_58_NORTH_TOP_Y = 93
ROOM58_NORTH_MAX_FRAMES = 4000
# 0x59 (GORIYA_COMPASS): lit; goriya 0x05 + 0x06 + boomerang 0x5c.  Entry
# (16,141) W mouth.  The kill-clear opens the UP door bit (cur_opened_doors
# bit 3 = UP) but the naive clear boxes Link at (48,125).  A central mass
# fills ~x100..190 / y118..165; the route is a perimeter waypoint micro:
# rise the west side to the y~100 open band, go west to x~44, rise to the
# y~64 top band, cross to x=120, push UP.  Dest is live $EB=0x49
# (GORIYA_BUBBLE — goriya 0x05 + keese 0x1b + bubble residual 0x2b, entry
# (120,205) S mouth).  2/2 byte-identical (recordings/59_up_v2/v3.json,
# arrived frame 2329).  RIGHT (KILL_CLEAR) -> COMPASS stays a hyp dead-end.
ROOM_59 = 0x59
ROOM_59_WEST_MOUTH = (16, 141)
ROOM_59_MID_BAND_Y = 100
ROOM_59_TOP_BAND_Y = 93
ROOM_59_WEST_COLUMN_X = 44
ROOM_59_NORTH_X = 120
ROOM_59_NORTH_PLANE_Y = 93
ROOM59_UP_MAX_FRAMES = 5000
# 0x49 (GORIYA_BUBBLE): entry (120,205) S mouth. Kill-clear goriya 0x05 then
# walk UP across the full-width water moat (~y120, tile 0xF4) at x=120.
# Stepladder required (recon fixture pokes ADDR_LADDER). Dest live $EB=0x39
# (DIGDOGGER_2: digdogger 0x38 + statue 0x55). 2/2 (49_up_v1/v2).
ROOM_49 = 0x49
ROOM_49_NORTH_X = 120
ROOM_49_NORTH_PLANE_Y = 93
ROOM_49_NORTH_BAND_Y = 109  # land north of the moat (walkable east-west)
ROOM_49_MOAT_SOUTH_Y = 133  # first solid row without the Stepladder
ROOM49_UP_MAX_FRAMES = 7000
# 0x39 (DIGDOGGER_2): entry (120,205) S mouth. LEFT door is OPEN on spawn —
# skip the 0x38 fight. Rise the centre column to y=141, hold LEFT. Dest live
# $EB=0x38 (GORIYA_PRE_HUNGRY). 2/2 (39_left_v2/v3). Do not hug SW statue.
ROOM_39 = 0x39
ROOM_39_DOOR_Y = 141
ROOM_39_WEST_PLANE = 16
ROOM39_LEFT_MAX_FRAMES = 4000
# 0x38 (GORIYA_PRE_HUNGRY): entry (208,141) E mouth, diamond floor. The y=149
# interior row blocks UP at x=120/104/88/200. Rise the east mouth pocket
# x=208 to y=93, cross to x=120, KEY-UP (keys 4->3) to live $EB=0x28.
# 2/2 (38_up_v6/v7). Compass room_item 0x0f stays uncollected.
ROOM_38 = 0x38
ROOM_38_EAST_POCKET_X = 208
ROOM_38_NORTH_X = 120
ROOM_38_TOP_BAND_Y = 93
ROOM38_UP_MAX_FRAMES = 8000
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


def east_of_room6c_ram_id() -> int | None:
    """Live ``$EB`` of the room east of ``0x6C`` (STALFOS_KEY), or None."""
    return LEVEL7_ROOM_BY_ID[STALFOS_KEY].ram_id


def north_of_room68_ram_id() -> int | None:
    """Live ``$EB`` of the room north of ``0x68`` (DODONGOS_UPGRADE), or None."""
    return LEVEL7_ROOM_BY_ID[DODONGOS_UPGRADE].ram_id


def east_of_room58_ram_id() -> int | None:
    """Live ``$EB`` of the room east of ``0x58`` (GORIYA_COMPASS), or None."""
    return LEVEL7_ROOM_BY_ID[GORIYA_COMPASS].ram_id


def north_of_room59_ram_id() -> int | None:
    """Live ``$EB`` of the room north of ``0x59`` (GORIYA_BUBBLE), or None."""
    return LEVEL7_ROOM_BY_ID[GORIYA_BUBBLE].ram_id


def north_of_room49_ram_id() -> int | None:
    """Live ``$EB`` of the room north of ``0x49`` (DIGDOGGER_2), or None."""
    return LEVEL7_ROOM_BY_ID[DIGDOGGER_2].ram_id


def west_of_room39_ram_id() -> int | None:
    """Live ``$EB`` of the room west of ``0x39`` (GORIYA_PRE_HUNGRY), or None."""
    return LEVEL7_ROOM_BY_ID[GORIYA_PRE_HUNGRY].ram_id


def north_of_room38_ram_id() -> int | None:
    """Live ``$EB`` of the room north of ``0x38`` (HUNGRY_GORIYA), or None."""
    return LEVEL7_ROOM_BY_ID[HUNGRY_GORIYA].ram_id



def south_of_room68_ram_id() -> int | None:
    """Live ``$EB`` of the room south of ``0x68`` (ROPES_KEY), or None."""
    return LEVEL7_ROOM_BY_ID[ROPES_KEY].ram_id


def north_of_room58_ram_id() -> int | None:
    """Live ``$EB`` of the room north of ``0x58`` (BOMB_UPGRADE), or None."""
    return LEVEL7_ROOM_BY_ID[BOMB_UPGRADE].ram_id


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


def room_6c_east_step(
    snap: ZeldaSnapshot,
    *,
    dest: int | None = None,
    stuck: int = 0,
    bump: int = 0,
) -> FrameAction:
    """One frame of 0x6C west mouth → live east dest 0x6D (STALFOS_KEY).

    Ride the ``y=141`` band east; the digdogger ``0x38`` may briefly block —
    a short UP bump (managed by the controller) nudges Link past it and the
    east door.
    """
    if snap.level != LEVEL7:
        return FrameAction(nes_idle_action(), "wait_level7")
    if snap.transitioning:
        return FrameAction(nes_action("RIGHT"), "east6c_scroll")
    if snap.mode != PLAY_MODE:
        return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
    if dest is not None and snap.screen == dest:
        return FrameAction(nes_idle_action(), "east6c_arrived")
    if snap.screen != ROOM_6C:
        return FrameAction(nes_idle_action(), f"unexpected_room_0x{snap.screen:02x}")

    y = int(snap.link_y)
    if bump > 0:
        return FrameAction(nes_action("UP"), "east6c_bump")
    if y > ROOM_6C_DOOR_Y + DOOR_Y_TOL:
        return FrameAction(nes_action("UP"), "east6c_drop_up")
    if y < ROOM_6C_DOOR_Y - DOOR_Y_TOL:
        return FrameAction(nes_action("DOWN"), "east6c_drop_down")
    return FrameAction(nes_action("RIGHT"), "east6c_push")


@dataclass(kw_only=True)
class Room6CEastController(HopController):
    """0x6C west mouth → east door to live dest 0x6D (STALFOS_KEY)."""

    spec_id: str = "level7_room6c_east"
    max_frames: int = ROOM6C_EAST_MAX_FRAMES
    require_level: int = LEVEL7
    done_reason: str = "left_0x6c"
    dest: int | None = field(default_factory=east_of_room6c_ram_id)
    _last_x: int | None = None
    _stuck: int = 0
    _bump: int = 0

    @property
    def stage_id(self) -> str:
        return self.spec_id

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        if (
            snap.level != LEVEL7
            or snap.mode != PLAY_MODE
            or snap.transitioning
            or snap.screen in {ENTRY_SCREEN, ROOM_69, ROOM_6A, ROOM_6B, ROOM_6C}
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
            f"_mode={snap.mode}_stuck={self._stuck}"
        )

    def scroll_action(self, snap: ZeldaSnapshot) -> FrameAction:
        if snap.screen == ROOM_6B:
            return self.mark_fail("west_backtrack")
        return FrameAction(nes_action("RIGHT"), "east6c_scroll")

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        x = int(snap.link_x)
        if self._bump > 0:
            self._bump -= 1
        elif self._last_x is not None and x == self._last_x:
            self._stuck += 1
            if self._stuck > 6:
                self._bump = 10
                self._stuck = 0
        else:
            self._stuck = 0
        self._last_x = x
        action = room_6c_east_step(snap, dest=self.dest, bump=self._bump)
        if action.reason.startswith("unexpected_room"):
            if snap.screen == ROOM_6B:
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


def room_68_north_step(
    snap: ZeldaSnapshot,
    *,
    dest: int | None = None,
) -> FrameAction:
    """One frame of 0x68 (KEESE_TRAPS) → OPEN north door → live dest 0x58.

    Link enters ~(208,93) from the 0x69 west bomb wall.  Route: ride the
    top band west to ``x=120``, push UP.  Blade traps 0x49 / keese only chip;
    assist soaks it.
    """
    if snap.level != LEVEL7:
        return FrameAction(nes_idle_action(), "wait_level7")
    if snap.transitioning:
        return FrameAction(nes_action("UP"), "north68_scroll")
    if snap.mode != PLAY_MODE:
        return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
    if dest is not None and snap.screen == dest:
        return FrameAction(nes_idle_action(), "north68_arrived")
    if snap.screen != ROOM_68:
        return FrameAction(nes_idle_action(), f"unexpected_room_0x{snap.screen:02x}")

    x, y = int(snap.link_x), int(snap.link_y)
    if y > ROOM_68_TOP_BAND_Y + DOOR_Y_TOL and abs(x - ROOM_68_NORTH_X) > NORTH_X_TOL:
        return FrameAction(nes_action("UP"), "north68_rise")
    if abs(x - ROOM_68_NORTH_X) > NORTH_X_TOL:
        btn = "LEFT" if x > ROOM_68_NORTH_X else "RIGHT"
        return FrameAction(nes_action(btn), "north68_align_x")
    return FrameAction(nes_action("UP"), "north68_push")


@dataclass(kw_only=True)
class Room68NorthController(HopController):
    """0x68 KEESE_TRAPS → OPEN north door to live dest 0x58 (DODONGOS_UPGRADE).

    Recon-wired only.  Reached via the 0x69 west bomb wall.
    """

    spec_id: str = "level7_room68_north"
    max_frames: int = ROOM68_NORTH_MAX_FRAMES
    require_level: int = LEVEL7
    done_reason: str = "left_0x68_north"
    dest: int | None = field(default_factory=north_of_room68_ram_id)

    @property
    def stage_id(self) -> str:
        return self.spec_id

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        if (
            snap.level != LEVEL7
            or snap.mode != PLAY_MODE
            or snap.transitioning
            or snap.screen in {ENTRY_SCREEN, ROOM_69, ROOM_6A, ROOM_6B, ROOM_68}
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
        return FrameAction(nes_action("UP"), "north68_scroll")

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        action = room_68_north_step(snap, dest=self.dest)
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


@dataclass(kw_only=True)
class Room58EastController(HopController):
    """0x58 (DODONGOS_UPGRADE) → OPEN east door to live dest 0x59.

    3x invulnerable 0x31 roamers are dodged (assist soaks chip damage).
    Waypoint micro up the east-open column: (120,165) → (200,165) →
    (200,141) → push RIGHT.  Recon-wired only.
    """

    spec_id: str = "level7_room58_east"
    max_frames: int = ROOM58_EAST_MAX_FRAMES
    require_level: int = LEVEL7
    done_reason: str = "left_0x58"
    dest: int | None = field(default_factory=east_of_room58_ram_id)
    _phase: str = "climb"

    @property
    def stage_id(self) -> str:
        return self.spec_id

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        if (
            snap.level != LEVEL7
            or snap.mode != PLAY_MODE
            or snap.transitioning
            or snap.screen in {ENTRY_SCREEN, ROOM_69, ROOM_68, ROOM_58}
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
            f"_mode={snap.mode}_phase={self._phase}"
        )

    def scroll_action(self, snap: ZeldaSnapshot) -> FrameAction:
        return FrameAction(nes_action("RIGHT"), "east58_scroll")

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        if snap.screen != ROOM_58:
            return self.mark_fail(f"unexpected_room_0x{snap.screen:02x}")
        x, y = int(snap.link_x), int(snap.link_y)
        if self._phase == "climb":
            if y > ROOM_58_MID_Y + DOOR_Y_TOL:
                return FrameAction(nes_action("UP"), "east58_climb")
            self._phase = "cross"
        if self._phase == "cross":
            if x < ROOM_58_EAST_COLUMN_X - NORTH_X_TOL:
                if abs(y - ROOM_58_MID_Y) > DOOR_Y_TOL:
                    return FrameAction(
                        nes_action("UP" if y > ROOM_58_MID_Y else "DOWN"),
                        "east58_cross_y",
                    )
                return FrameAction(nes_action("RIGHT"), "east58_cross_x")
            self._phase = "drop"
        if self._phase == "drop":
            if abs(y - ROOM_58_DOOR_Y) > DOOR_Y_TOL:
                return FrameAction(
                    nes_action("UP" if y > ROOM_58_DOOR_Y else "DOWN"), "east58_drop"
                )
            self._phase = "push"
        return dungeon_align_then_push(
            snap,
            push_dir="RIGHT",
            target_y=ROOM_58_DOOR_Y,
            y_tol=DOOR_Y_TOL,
            door_plane=224,
            reason="east58",
        )

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


@dataclass(kw_only=True)
class Room59UpController(HopController):
    """0x59 (GORIYA_COMPASS) west mouth: kill-clear the goriya 0x05/0x06,
    then the perimeter waypoint micro around the central mass to the UP door
    -> live dest 0x49 (GORIYA_BUBBLE).

    The kill-clear sets ``cur_opened_doors`` bit 3 (UP) but boxes Link at
    ``(48,125)``.  Phases: rise to the ``y~100`` open west band, go west to
    ``x~44``, rise to the ``y~64`` top band, cross to ``x=120``, push UP.
    2/2 byte-identical (recordings/59_up_v2/v3.json).  Recon-wired only.
    """

    spec_id: str = "level7_room59_up"
    max_frames: int = ROOM59_UP_MAX_FRAMES
    require_level: int = LEVEL7
    done_reason: str = "left_0x59_north"
    dest: int | None = field(default_factory=north_of_room59_ram_id)
    saw_goriya: bool = False
    _phase: str = "clear"

    @property
    def stage_id(self) -> str:
        return self.spec_id

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        if (
            snap.level != LEVEL7
            or snap.mode != PLAY_MODE
            or snap.transitioning
            or snap.screen in {ENTRY_SCREEN, ROOM_69, ROOM_68, ROOM_58, ROOM_59}
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
            f"_mode={snap.mode}_phase={self._phase}_saw={int(self.saw_goriya)}"
        )

    def scroll_action(self, snap: ZeldaSnapshot) -> FrameAction:
        return FrameAction(nes_action("UP"), "up59_scroll")

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        if snap.screen != ROOM_59:
            if snap.screen in {ROOM_58, ENTRY_SCREEN}:
                return self.mark_fail("west_backtrack")
            return self.mark_fail(f"unexpected_room_0x{snap.screen:02x}")

        live = live_goriyas(snap)
        if live:
            self.saw_goriya = True
            target = nearest_enemy(snap.link_x, snap.link_y, live)
            if target is None:
                return FrameAction(nes_idle_action(), "goriya_missing")
            return _goriya_fight(snap, target, frames=self.frames)
        if not self.saw_goriya:
            return FrameAction(nes_idle_action(), "spawn_wait")

        x, y = int(snap.link_x), int(snap.link_y)
        if self._phase == "clear":
            self._phase = "rise1"
        if self._phase == "rise1":
            if y > ROOM_59_MID_BAND_Y + DOOR_Y_TOL:
                return FrameAction(nes_action("UP"), "up59_rise1")
            self._phase = "west"
        if self._phase == "west":
            if x > ROOM_59_WEST_COLUMN_X + NORTH_X_TOL:
                return FrameAction(nes_action("LEFT"), "up59_west")
            self._phase = "rise2"
        if self._phase == "rise2":
            if y > ROOM_59_TOP_BAND_Y + DOOR_Y_TOL:
                return FrameAction(nes_action("UP"), "up59_rise2")
            self._phase = "cross"
        if self._phase == "cross":
            if abs(x - ROOM_59_NORTH_X) > NORTH_X_TOL:
                btn = "LEFT" if x > ROOM_59_NORTH_X else "RIGHT"
                return FrameAction(nes_action(btn), "up59_cross")
            self._phase = "push"
        return dungeon_align_then_push(
            snap,
            push_dir="UP",
            target_x=ROOM_59_NORTH_X,
            x_tol=NORTH_X_TOL,
            reason="up59",
        )

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


def room_68_down_step(
    snap: ZeldaSnapshot,
    *,
    dest: int | None = None,
) -> FrameAction:
    """One frame of 0x68 (KEESE_TRAPS) → OPEN south door → live dest 0x78.

    Peel off the east trap column, drop between trap rows, align x=120,
    push DOWN.  OccupancyWalker poisons on blade-trap knockback — waypoints.
    """
    if snap.level != LEVEL7:
        return FrameAction(nes_idle_action(), "wait_level7")
    if snap.transitioning:
        return FrameAction(nes_action("DOWN"), "south68_scroll")
    if snap.mode != PLAY_MODE:
        return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
    if dest is not None and snap.screen == dest:
        return FrameAction(nes_idle_action(), "south68_arrived")
    if snap.screen != ROOM_68:
        return FrameAction(nes_idle_action(), f"unexpected_room_0x{snap.screen:02x}")

    x, y = int(snap.link_x), int(snap.link_y)
    if abs(x - ROOM_68_SOUTH_X) > NORTH_X_TOL and abs(y - ROOM_68_TRAP_ROW_Y) <= 8:
        return FrameAction(nes_action("UP"), "south68_off_trap")
    if x > ROOM_68_SAFE_X + NORTH_X_TOL:
        return FrameAction(nes_action("LEFT"), "south68_peel")
    if y < ROOM_68_MID_Y - DOOR_Y_TOL:
        return FrameAction(nes_action("DOWN"), "south68_drop")
    if abs(x - ROOM_68_SOUTH_X) > NORTH_X_TOL:
        btn = "LEFT" if x > ROOM_68_SOUTH_X else "RIGHT"
        return FrameAction(nes_action(btn), "south68_align_x")
    return FrameAction(nes_action("DOWN"), "south68_push")


@dataclass(kw_only=True)
class Room68DownController(HopController):
    """0x68 KEESE_TRAPS → OPEN south door to live dest 0x78 (ROPES_KEY).

    Recon-wired only.  0x78 is a dead-end (ropes 0x28 + floor key 0x19).
    """

    spec_id: str = "level7_room68_down"
    max_frames: int = ROOM68_DOWN_MAX_FRAMES
    require_level: int = LEVEL7
    done_reason: str = "left_0x68_south"
    dest: int | None = field(default_factory=south_of_room68_ram_id)

    @property
    def stage_id(self) -> str:
        return self.spec_id

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        if (
            snap.level != LEVEL7
            or snap.mode != PLAY_MODE
            or snap.transitioning
            or snap.screen in {ENTRY_SCREEN, ROOM_69, ROOM_6A, ROOM_6B, ROOM_68}
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
        return FrameAction(nes_action("DOWN"), "south68_scroll")

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        action = room_68_down_step(snap, dest=self.dest)
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
            "door": "DOWN",
        }


def room_58_north_step(
    snap: ZeldaSnapshot,
    *,
    dest: int | None = None,
) -> FrameAction:
    """One frame of 0x58 → KEY north door → live dest 0x48 (BOMB_UPGRADE).

    East-around the central 2-block mass, then the x=120 channel.  Dodge
    0x31; do not fight.  Key spend is natural (door KEY).
    """
    if snap.level != LEVEL7:
        return FrameAction(nes_idle_action(), "wait_level7")
    if snap.transitioning:
        return FrameAction(nes_action("UP"), "north58_scroll")
    if snap.mode != PLAY_MODE:
        return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
    if dest is not None and snap.screen == dest:
        return FrameAction(nes_idle_action(), "north58_arrived")
    if snap.screen != ROOM_58:
        return FrameAction(nes_idle_action(), f"unexpected_room_0x{snap.screen:02x}")

    x, y = int(snap.link_x), int(snap.link_y)
    if y <= ROOM_58_NORTH_TOP_Y + DOOR_Y_TOL:
        if abs(x - ROOM_58_NORTH_X) > NORTH_X_TOL:
            btn = "LEFT" if x > ROOM_58_NORTH_X else "RIGHT"
            return FrameAction(nes_action(btn), "north58_align_x")
        return FrameAction(nes_action("UP"), "north58_push")
    if y > ROOM_58_NORTH_MID_Y + DOOR_Y_TOL:
        return FrameAction(nes_action("UP"), "north58_climb")
    if x < ROOM_58_NORTH_EAST_X - NORTH_X_TOL:
        return FrameAction(nes_action("RIGHT"), "north58_east")
    return FrameAction(nes_action("UP"), "north58_rise")


@dataclass(kw_only=True)
class Room58NorthController(HopController):
    """0x58 DODONGOS_UPGRADE → KEY north door to live dest 0x48.

    Recon-wired only.  0x48 is a dead-end old-man bomb-capacity room
    (100 rupees).  Do not write max_bombs.
    """

    spec_id: str = "level7_room58_north"
    max_frames: int = ROOM58_NORTH_MAX_FRAMES
    require_level: int = LEVEL7
    done_reason: str = "left_0x58_north"
    dest: int | None = field(default_factory=north_of_room58_ram_id)

    @property
    def stage_id(self) -> str:
        return self.spec_id

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        if (
            snap.level != LEVEL7
            or snap.mode != PLAY_MODE
            or snap.transitioning
            or snap.screen in {ENTRY_SCREEN, ROOM_69, ROOM_68, ROOM_58}
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
        return FrameAction(nes_action("UP"), "north58_scroll")

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        action = room_58_north_step(snap, dest=self.dest)
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


def room_49_up_step(
    snap: ZeldaSnapshot,
    *,
    dest: int | None = None,
    saw_goriya: bool = False,
    frames: int = 0,
) -> FrameAction:
    """One frame of 0x49 kill-clear → north door across the water moat.

    The Stepladder lets Link walk UP at ``x=120`` through tile ``0xF4``.
    The north exit is OPEN-like (``cur_opened_doors`` stays 0); goriya
    clear is still required so they do not box the south mouth.
    """
    if snap.level != LEVEL7:
        return FrameAction(nes_idle_action(), "wait_level7")
    if snap.transitioning:
        return FrameAction(nes_action("UP"), "up49_scroll")
    if snap.mode != PLAY_MODE:
        return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
    if dest is not None and snap.screen == dest:
        return FrameAction(nes_idle_action(), "up49_arrived")
    if snap.screen != ROOM_49:
        return FrameAction(nes_idle_action(), f"unexpected_room_0x{snap.screen:02x}")

    live = live_goriyas(snap)
    if live:
        target = nearest_enemy(snap.link_x, snap.link_y, live)
        if target is None:
            return FrameAction(nes_idle_action(), "goriya_missing")
        return _goriya_fight(snap, target, frames=frames)
    if not saw_goriya:
        return FrameAction(nes_idle_action(), "spawn_wait")

    x, y = int(snap.link_x), int(snap.link_y)
    keese = live_keese(snap)
    if keese:
        target = nearest_enemy(x, y, keese)
        if target is not None:
            hint = engagement_hint(
                EnemyKind.KEESE, snap, target, projectiles=_projectiles(snap)
            )
            if should_swing_at(x, y, hint.face, (target,), hint=hint):
                if frames % _SWING_PERIOD < _SWING_HOLD:
                    return FrameAction(nes_action(hint.face, "A"), "keese_slash")
                return FrameAction(nes_action(hint.face), "keese_face")
        # Never chase keese onto the water — 49_ctl_v3 pinned at (64,117).
        on_water = ROOM_49_NORTH_BAND_Y < y <= ROOM_49_MOAT_SOUTH_Y
        if on_water:
            return FrameAction(nes_action("UP"), "keese_off_water")

    # Stepladder crosses the moat on the facing axis only — do not strafe
    # on the water. Align x on south land, hold UP across, then align north.
    if y > ROOM_49_NORTH_BAND_Y + DOOR_Y_TOL:
        if y > ROOM_49_MOAT_SOUTH_Y and abs(x - ROOM_49_NORTH_X) > NORTH_X_TOL:
            btn = "LEFT" if x > ROOM_49_NORTH_X else "RIGHT"
            return FrameAction(nes_action(btn), "up49_south_align")
        btn = "UP"
        if keese and frames % _SWING_PERIOD < _SWING_HOLD:
            return FrameAction(nes_action(btn, "A"), "up49_cross_slash")
        return FrameAction(nes_action(btn), "up49_cross")
    if keese:
        # North land, keese still up: face them but stay off the water.
        target = nearest_enemy(x, y, keese)
        if target is not None and int(target.y) <= ROOM_49_NORTH_BAND_Y + DOOR_Y_TOL:
            hint = engagement_hint(
                EnemyKind.KEESE, snap, target, projectiles=_projectiles(snap)
            )
            return FrameAction(nes_action(hint.face), "keese_chase_land")
        if abs(x - ROOM_49_NORTH_X) > NORTH_X_TOL:
            btn = "LEFT" if x > ROOM_49_NORTH_X else "RIGHT"
            return FrameAction(nes_action(btn), "keese_wait_align")
        return FrameAction(nes_action("UP", "A"), "keese_door_slash")
    return dungeon_align_then_push(
        snap,
        push_dir="UP",
        target_x=ROOM_49_NORTH_X,
        x_tol=NORTH_X_TOL,
        reason="up49",
    )


@dataclass(kw_only=True)
class Room49UpController(HopController):
    """0x49 (GORIYA_BUBBLE) south mouth: kill-clear goriya 0x05, then UP
    across the water moat at x=120 (Stepladder) to live dest 0x39.

    2/2 byte-identical (recordings/49_up_v1/v2.json).  Recon-wired only.
    Requires ADDR_LADDER=1 on the recon fixture.
    """

    spec_id: str = "level7_room49_up"
    max_frames: int = ROOM49_UP_MAX_FRAMES
    require_level: int = LEVEL7
    done_reason: str = "left_0x49_north"
    dest: int | None = field(default_factory=north_of_room49_ram_id)
    saw_goriya: bool = False

    @property
    def stage_id(self) -> str:
        return self.spec_id

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        if (
            snap.level != LEVEL7
            or snap.mode != PLAY_MODE
            or snap.transitioning
            or snap.screen in {ENTRY_SCREEN, ROOM_69, ROOM_68, ROOM_58, ROOM_59, ROOM_49}
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
        )

    def scroll_action(self, snap: ZeldaSnapshot) -> FrameAction:
        return FrameAction(nes_action("UP"), "up49_scroll")

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        if snap.screen != ROOM_49:
            if snap.screen in {ROOM_59, ENTRY_SCREEN}:
                return self.mark_fail("south_backtrack")
            return self.mark_fail(f"unexpected_room_0x{snap.screen:02x}")

        if live_goriyas(snap):
            self.saw_goriya = True
        action = room_49_up_step(
            snap, dest=self.dest, saw_goriya=self.saw_goriya, frames=self.frames
        )
        if action.reason.startswith("unexpected_room"):
            if snap.screen in {ROOM_59, ENTRY_SCREEN}:
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
            "evidence": "fixture-live",
            "route_eligible": False,
            "door": "UP",
        }


def room_39_left_step(
    snap: ZeldaSnapshot,
    *,
    dest: int | None = None,
) -> FrameAction:
    """One frame of 0x39 south mouth → OPEN west door (skip Digdogger).

    Stay on the centre column until ``y=141`` — the SW statue boxes
    ``(48,189)``. Then hold LEFT on the door row.
    """
    if snap.level != LEVEL7:
        return FrameAction(nes_idle_action(), "wait_level7")
    if snap.transitioning:
        return FrameAction(nes_action("LEFT"), "left39_scroll")
    if snap.mode != PLAY_MODE:
        return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
    if dest is not None and snap.screen == dest:
        return FrameAction(nes_idle_action(), "left39_arrived")
    if snap.screen != ROOM_39:
        return FrameAction(nes_idle_action(), f"unexpected_room_0x{snap.screen:02x}")

    x, y = int(snap.link_x), int(snap.link_y)
    if y > ROOM_39_DOOR_Y + DOOR_Y_TOL:
        if abs(x - NORTH_DOOR_X) > NORTH_X_TOL:
            btn = "LEFT" if x > NORTH_DOOR_X else "RIGHT"
            return FrameAction(nes_action(btn), "left39_center")
        return FrameAction(nes_action("UP"), "left39_rise")
    if y < ROOM_39_DOOR_Y - DOOR_Y_TOL:
        return FrameAction(nes_action("DOWN"), "left39_drop")
    return dungeon_align_then_push(
        snap,
        push_dir="LEFT",
        target_y=ROOM_39_DOOR_Y,
        y_tol=DOOR_Y_TOL,
        door_plane=ROOM_39_WEST_PLANE,
        reason="left39",
    )


@dataclass(kw_only=True)
class Room39LeftController(HopController):
    """0x39 (DIGDOGGER_2) south mouth → OPEN west door to live dest 0x38.

    Skips the Digdogger fight.  2/2 (recordings/39_left_v2/v3.json).
    Recon-wired only.
    """

    spec_id: str = "level7_room39_left"
    max_frames: int = ROOM39_LEFT_MAX_FRAMES
    require_level: int = LEVEL7
    done_reason: str = "left_0x39_west"
    dest: int | None = field(default_factory=west_of_room39_ram_id)

    @property
    def stage_id(self) -> str:
        return self.spec_id

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        if (
            snap.level != LEVEL7
            or snap.mode != PLAY_MODE
            or snap.transitioning
            or snap.screen in {ENTRY_SCREEN, ROOM_69, ROOM_49, ROOM_39}
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
        return FrameAction(nes_action("LEFT"), "left39_scroll")

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        action = room_39_left_step(snap, dest=self.dest)
        if action.reason.startswith("unexpected_room"):
            if snap.screen == ROOM_49:
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
            "evidence": "fixture-live",
            "route_eligible": False,
            "door": "LEFT",
        }


def room_38_up_step(
    snap: ZeldaSnapshot,
    *,
    dest: int | None = None,
    saw_goriya: bool = False,
    frames: int = 0,
) -> FrameAction:
    """One frame of 0x38 kill-clear → east-pocket rise → KEY north door.

    Interior ``y=149`` is a diamond wall (UP blocked at x=120/104/88/200).
    Recollect the east mouth pocket ``x=208``, rise to ``y=93``, cross to
    ``x=120``, push UP. The key consume is natural.
    """
    if snap.level != LEVEL7:
        return FrameAction(nes_idle_action(), "wait_level7")
    if snap.transitioning:
        return FrameAction(nes_action("UP"), "up38_scroll")
    if snap.mode != PLAY_MODE:
        return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
    if dest is not None and snap.screen == dest:
        return FrameAction(nes_idle_action(), "up38_arrived")
    if snap.screen != ROOM_38:
        return FrameAction(nes_idle_action(), f"unexpected_room_0x{snap.screen:02x}")

    live = live_goriyas(snap)
    if live:
        target = nearest_enemy(snap.link_x, snap.link_y, live)
        if target is None:
            return FrameAction(nes_idle_action(), "goriya_missing")
        return _goriya_fight(snap, target, frames=frames)
    if not saw_goriya:
        return FrameAction(nes_idle_action(), "spawn_wait")

    x, y = int(snap.link_x), int(snap.link_y)
    if y > ROOM_38_TOP_BAND_Y + 12 and x < ROOM_38_EAST_POCKET_X - NORTH_X_TOL:
        return FrameAction(nes_action("RIGHT"), "up38_pocket")
    if y > ROOM_38_TOP_BAND_Y + DOOR_Y_TOL:
        return FrameAction(nes_action("UP"), "up38_rise")
    return dungeon_align_then_push(
        snap,
        push_dir="UP",
        target_x=ROOM_38_NORTH_X,
        x_tol=NORTH_X_TOL,
        reason="up38",
    )


@dataclass(kw_only=True)
class Room38UpController(HopController):
    """0x38 (GORIYA_PRE_HUNGRY) east mouth: kill-clear, east-pocket rise,
    KEY-UP to live dest 0x28 (HUNGRY_GORIYA).  2/2 (38_up_v6/v7).
    Recon-wired only.
    """

    spec_id: str = "level7_room38_up"
    max_frames: int = ROOM38_UP_MAX_FRAMES
    require_level: int = LEVEL7
    done_reason: str = "left_0x38_north"
    dest: int | None = field(default_factory=north_of_room38_ram_id)
    saw_goriya: bool = False

    @property
    def stage_id(self) -> str:
        return self.spec_id

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        if (
            snap.level != LEVEL7
            or snap.mode != PLAY_MODE
            or snap.transitioning
            or snap.screen in {ENTRY_SCREEN, ROOM_49, ROOM_39, ROOM_38}
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
        )

    def scroll_action(self, snap: ZeldaSnapshot) -> FrameAction:
        return FrameAction(nes_action("UP"), "up38_scroll")

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        if live_goriyas(snap):
            self.saw_goriya = True
        action = room_38_up_step(
            snap, dest=self.dest, saw_goriya=self.saw_goriya, frames=self.frames
        )
        if action.reason.startswith("unexpected_room"):
            if snap.screen == ROOM_39:
                return self.mark_fail("east_backtrack")
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
ROOM_09 = 0x09
ROOM_19 = 0x19
ROOM_1A = 0x1A
ROOM_09_SOUTH_Y = 189
ROOM_09_SOUTH_X = 120
ROOM09_DOWN_MAX_FRAMES = 8000


def north_of_room18_ram_id() -> int | None:
    """Live ``$EB`` of the room north of ``0x18`` (HIDDEN_RUPEES)."""
    return LEVEL7_ROOM_BY_ID[HIDDEN_RUPEES].ram_id


def east_of_room08_ram_id() -> int | None:
    """Live ``$EB`` of the room east of ``0x08`` (GORIYA_POST_RUPEE)."""
    return LEVEL7_ROOM_BY_ID[GORIYA_POST_RUPEE].ram_id


def south_of_room09_ram_id() -> int | None:
    """Live ``$EB`` of the room south of ``0x09`` (WEST_LOCK_SKIP)."""
    return LEVEL7_ROOM_BY_ID[WEST_LOCK_SKIP].ram_id


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
# 0x19 diamond floor east BOMB wall -> $EB=0x1A. South-around
# (96,141)->(96,189)->(208,189)->(208,141) face RIGHT. 2/2 (19_be_v5/v6).
L7_ROOM19_EAST_BOMB = Level7BombWall(
    room=ROOM_19, stand=(208, 141), face="RIGHT", opens_to=0x1A
)
L7_ROOM19_EAST_APPROACH = ((96, 141), (96, 189), (208, 189), (208, 141))


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


def room_09_down_step(
    snap: ZeldaSnapshot,
    *,
    dest: int | None = None,
    saw_goriya: bool = False,
    frames: int = 0,
) -> FrameAction:
    """One frame of 0x09 kill-clear → south shutter.

    Dead: south is OPEN on spawn. The shutter walks only after the goriya
    0x05/0x06 clear; ``cur_opened_doors`` stays LEFT. Drop to ``y=189``
    (west of the statue row), align ``x=120``, push DOWN.
    """
    if snap.level != LEVEL7:
        return FrameAction(nes_idle_action(), "wait_level7")
    if snap.transitioning:
        return FrameAction(nes_action("DOWN"), "down09_scroll")
    if snap.mode != PLAY_MODE:
        return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
    if dest is not None and snap.screen == dest:
        return FrameAction(nes_idle_action(), "down09_arrived")
    if snap.screen != ROOM_09:
        return FrameAction(nes_idle_action(), f"unexpected_room_0x{snap.screen:02x}")

    live = live_goriyas(snap)
    if live:
        target = nearest_enemy(snap.link_x, snap.link_y, live)
        if target is None:
            return FrameAction(nes_idle_action(), "goriya_missing")
        return _goriya_fight(snap, target, frames=frames)
    if not saw_goriya:
        return FrameAction(nes_idle_action(), "spawn_wait")

    x, y = int(snap.link_x), int(snap.link_y)
    if y < ROOM_09_SOUTH_Y - DOOR_Y_TOL:
        return FrameAction(nes_action("DOWN"), "down09_drop")
    return dungeon_align_then_push(
        snap,
        push_dir="DOWN",
        target_x=ROOM_09_SOUTH_X,
        x_tol=NORTH_X_TOL,
        reason="down09",
    )


@dataclass(kw_only=True)
class Room09DownController(HopController):
    """0x09 (GORIYA_POST_RUPEE) west mouth: kill-clear, south shutter
    to live dest 0x19 (WEST_LOCK_SKIP).  2/2 (09_down_v2/v3).
    Recon-wired only.
    """

    spec_id: str = "level7_room09_down"
    max_frames: int = ROOM09_DOWN_MAX_FRAMES
    require_level: int = LEVEL7
    done_reason: str = "left_0x09_south"
    dest: int | None = field(default_factory=south_of_room09_ram_id)
    saw_goriya: bool = False

    @property
    def stage_id(self) -> str:
        return self.spec_id

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        if (
            snap.level != LEVEL7
            or snap.mode != PLAY_MODE
            or snap.transitioning
            or snap.screen in {ENTRY_SCREEN, ROOM_18, ROOM_08, ROOM_09}
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
        )

    def scroll_action(self, snap: ZeldaSnapshot) -> FrameAction:
        return FrameAction(nes_action("DOWN"), "down09_scroll")

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        if live_goriyas(snap):
            self.saw_goriya = True
        action = room_09_down_step(
            snap, dest=self.dest, saw_goriya=self.saw_goriya, frames=self.frames
        )
        if action.reason.startswith("unexpected_room"):
            if snap.screen == ROOM_08:
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
            "door": "DOWN",
        }


ROOM_4A = 0x4A
ROOM1A_CANDLE_MAX_FRAMES = 16000
CELLAR_MODE = 9
PUSHABLE_BLOCK = 0x68


def cellar_of_room1a_ram_id() -> int | None:
    """Live ``$EB`` of the Red Candle cellar off ``0x1A``."""
    return LEVEL7_ROOM_BY_ID[RED_CANDLE_CELLAR].ram_id


def _pushable_block_y(snap: ZeldaSnapshot) -> int | None:
    for obj in snap.objects:
        if 1 <= int(obj.slot) <= 12 and int(obj.type_id) == PUSHABLE_BLOCK:
            return int(obj.y)
    return None


@dataclass(kw_only=True)
class Room1ACandleController(HopController):
    """0x1A: kill-clear (incl. NE goriya), push 0x68 UP, stairs to cellar
    ``0x4A``, walk onto Red Candle.  ADDR_CANDLE 0→2 NATURAL.

    Dead: L5 south-face UP while a goriya still lives NE of the plus
    (``room_all_dead`` stays 0, block does not slide).  2/2 (1a_push_v16/v18).
    Recon-wired only.  Pad leftover is cellar ``0x4A`` ``(135,141)`` mode 9.
    """

    spec_id: str = "level7_room1a_candle"
    max_frames: int = ROOM1A_CANDLE_MAX_FRAMES
    require_level: int = LEVEL7
    done_reason: str = "red_candle_natural"
    dest: int | None = field(default_factory=cellar_of_room1a_ram_id)
    saw_goriya: bool = False
    _phase: str = "clear"
    _hunt_i: int = 0

    @property
    def stage_id(self) -> str:
        return self.spec_id

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        return int(snap.candle) >= 2

    def on_arrive(self, snap: ZeldaSnapshot) -> str:
        return f"candle_{snap.candle}_0x{snap.screen:02x}_{snap.link_x}_{snap.link_y}"

    def timeout_note(self, snap: ZeldaSnapshot) -> str:
        return (
            f"timeout_0x{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
            f"_mode={snap.mode}_phase={self._phase}_c={snap.candle}"
        )

    def scroll_action(self, snap: ZeldaSnapshot) -> FrameAction:
        return FrameAction(nes_action("RIGHT"), "candle_scroll")

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        if int(snap.candle) >= 2:
            return self.mark_done(snap)
        if snap.mode == CELLAR_MODE or snap.screen == ROOM_4A:
            return self._cellar(snap)
        if snap.screen != ROOM_1A:
            return self.mark_fail(f"unexpected_room_0x{snap.screen:02x}")

        live = live_goriyas(snap)
        if live:
            self.saw_goriya = True
            if self._phase == "clear" and self.frames > 2800:
                self._phase = "hunt"
            if self._phase == "hunt":
                return self._hunt(snap, live)
            target = nearest_enemy(snap.link_x, snap.link_y, live)
            if target is None:
                return FrameAction(nes_idle_action(), "goriya_missing")
            return _goriya_fight(snap, target, frames=self.frames)
        if not self.saw_goriya:
            return FrameAction(nes_idle_action(), "spawn_wait")

        by = _pushable_block_y(snap)
        x, y = int(snap.link_x), int(snap.link_y)
        if by is None or by > 132:
            if y < 189 - DOOR_Y_TOL and x < 150:
                return FrameAction(nes_action("DOWN"), "candle_south")
            if abs(x - 96) > NORTH_X_TOL:
                return FrameAction(
                    nes_action("LEFT" if x > 96 else "RIGHT"), "candle_stand_x"
                )
            if y > 162:
                return FrameAction(nes_action("UP"), "candle_stand_y")
            return FrameAction(nes_action("UP"), "candle_push")
        if abs(x - 136) > 6 or abs(y - 141) > 6:
            if abs(y - 141) > DOOR_Y_TOL:
                return FrameAction(
                    nes_action("UP" if y > 141 else "DOWN"), "candle_stairs_y"
                )
            return FrameAction(
                nes_action("RIGHT" if x < 136 else "LEFT"), "candle_stairs_x"
            )
        return FrameAction(nes_action("RIGHT"), "candle_stairs_push")

    def _hunt(self, snap: ZeldaSnapshot, live: tuple) -> FrameAction:
        wps = ((32, 189), (192, 189), (192, 93), (160, 93))
        if self._hunt_i >= len(wps):
            target = nearest_enemy(snap.link_x, snap.link_y, live)
            if target is None:
                return FrameAction(nes_idle_action(), "goriya_missing")
            return _goriya_fight(snap, target, frames=self.frames)
        tx, ty = wps[self._hunt_i]
        x, y = int(snap.link_x), int(snap.link_y)
        if abs(x - tx) <= 4 and abs(y - ty) <= 4:
            self._hunt_i += 1
            return FrameAction(nes_idle_action(), "candle_hunt_next")
        if abs(y - ty) > 4:
            return FrameAction(
                nes_action("UP" if y > ty else "DOWN"), "candle_hunt_y"
            )
        return FrameAction(
            nes_action("LEFT" if x > tx else "RIGHT"), "candle_hunt_x"
        )

    def _cellar(self, snap: ZeldaSnapshot) -> FrameAction:
        x, y = int(snap.link_x), int(snap.link_y)
        if y < 180:
            return FrameAction(nes_action("DOWN"), "cellar_drop")
        if x < 172:
            return FrameAction(nes_action("RIGHT"), "cellar_east")
        if y > 145:
            return FrameAction(nes_action("UP"), "cellar_climb")
        if x > 124:
            return FrameAction(nes_action("LEFT"), "cellar_candle")
        return FrameAction(nes_idle_action(), "cellar_idle")

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
            "door": "STAIRS",
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


ROOM_4A_WEST_X = 48
ROOM_4A_EAST_COL = 176
ROOM_4A_FLOOR_Y = 189
ROOM_4A_ALIGN = 4
ROOM4A_RETURN_MAX_FRAMES = 4000


def play_of_room4a_ram_id() -> int | None:
    """Live ``$EB`` of CANDLE_PUSH after the 0x4A west-ladder stairs return."""
    return LEVEL7_ROOM_BY_ID[CANDLE_PUSH].ram_id


def room_4a_return_step(snap: ZeldaSnapshot) -> FrameAction:
    """One-frame 0x4A stairs return: east drop, floor west, west-ladder UP.

    Dead: walk off the candle pad at y=141 (tile 243) as the return.
    """
    x, y = int(snap.link_x), int(snap.link_y)
    if y >= ROOM_4A_FLOOR_Y - ROOM_4A_ALIGN:
        if abs(x - ROOM_4A_WEST_X) > ROOM_4A_ALIGN:
            btn = "LEFT" if x > ROOM_4A_WEST_X else "RIGHT"
            return FrameAction(nes_action(btn), "cellar_floor_west")
        return FrameAction(nes_action("UP"), "cellar_west_climb")
    if abs(x - ROOM_4A_WEST_X) <= 8:
        return FrameAction(nes_action("UP"), "cellar_west_up")
    if x < ROOM_4A_EAST_COL - ROOM_4A_ALIGN:
        return FrameAction(nes_action("RIGHT"), "cellar_to_east")
    return FrameAction(nes_action("LEFT", "DOWN"), "cellar_east_drop")


@dataclass(kw_only=True)
class Room4AReturnController(HopController):
    """0x4A cellar leftover → west-ladder stairs → live play ``0x1A``.

    2/2 (4a_ret_v7/v8).  Recon-wired only.  ``route_eligible=false``.
    """

    spec_id: str = "level7_room4a_return"
    max_frames: int = ROOM4A_RETURN_MAX_FRAMES
    require_level: int = LEVEL7
    wait_modes: tuple[int, ...] = WAIT_SCROLL_B
    done_reason: str = "left_0x4a_stairs"
    dest: int | None = field(default_factory=play_of_room4a_ram_id)

    @property
    def stage_id(self) -> str:
        return self.spec_id

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        return (
            self.dest is not None
            and snap.mode == PLAY_MODE
            and not snap.transitioning
            and snap.screen == self.dest
        )

    def on_arrive(self, snap: ZeldaSnapshot) -> str:
        return f"play_0x{snap.screen:02x}_{snap.link_x}_{snap.link_y}"

    def timeout_note(self, snap: ZeldaSnapshot) -> str:
        return (
            f"timeout_0x{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
            f"_mode={snap.mode}"
        )

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        if snap.screen != ROOM_4A and snap.mode != 9:
            if self.dest is not None and snap.screen == self.dest:
                return FrameAction(nes_idle_action(), "wait_dest")
            return self.mark_fail(f"unexpected_room_0x{snap.screen:02x}")
        return room_4a_return_step(snap)

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
            "door": "STAIRS",
        }


ROOM_1B = 0x1B
ROOM_1C = 0x1C
ROOM1B_KEY_EAST_MAX_FRAMES = 4000


def east_of_room1b_ram_id() -> int | None:
    """Live ``$EB`` of the room east of ``0x1B`` (FORCED_DIGDOGGER)."""
    return LEVEL7_ROOM_BY_ID[FORCED_DIGDOGGER].ram_id


def room_1b_key_east_step(snap: ZeldaSnapshot) -> FrameAction:
    """One-frame 0x1B y=141 KEY-RIGHT. Goriyas are tanked, not cleared."""
    x, y = int(snap.link_x), int(snap.link_y)
    if abs(y - EAST_DOOR_Y) > DOOR_Y_TOL:
        return FrameAction(
            nes_action("UP" if y > EAST_DOOR_Y else "DOWN"), "keyeast_align_y"
        )
    if x < EAST_DOOR_X - 2:
        return FrameAction(nes_action("RIGHT"), "keyeast_approach")
    return FrameAction(nes_action("RIGHT"), "keyeast_push")


@dataclass(kw_only=True)
class Room1BKeyEastController(HopController):
    """0x1B GORIYA_PRE_DIG KEY-east → live play ``0x1C`` (FORCED_DIGDOGGER).

    2/2 (1b_ke_v2/v3).  Recon-wired only.  Natural key spend 3→2.
    """

    spec_id: str = "level7_room1b_key_east"
    max_frames: int = ROOM1B_KEY_EAST_MAX_FRAMES
    require_level: int = LEVEL7
    wait_modes: tuple[int, ...] = WAIT_SCROLL_B
    done_reason: str = "left_0x1b_east"
    dest: int | None = field(default_factory=east_of_room1b_ram_id)

    @property
    def stage_id(self) -> str:
        return self.spec_id

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        return (
            self.dest is not None
            and snap.mode == PLAY_MODE
            and not snap.transitioning
            and snap.screen == self.dest
        )

    def on_arrive(self, snap: ZeldaSnapshot) -> str:
        return f"play_0x{snap.screen:02x}_{snap.link_x}_{snap.link_y}"

    def timeout_note(self, snap: ZeldaSnapshot) -> str:
        return (
            f"timeout_0x{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
            f"_mode={snap.mode}"
        )

    def scroll_action(self, snap: ZeldaSnapshot) -> FrameAction:
        return FrameAction(nes_action("RIGHT"), "keyeast_scroll")

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        if snap.screen != ROOM_1B:
            if self.dest is not None and snap.screen == self.dest:
                return FrameAction(nes_idle_action(), "wait_dest")
            return self.mark_fail(f"unexpected_room_0x{snap.screen:02x}")
        return room_1b_key_east_step(snap)

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
    "ROOM_49",
    "ROOM_39",
    "ROOM_38",
    "ROOM_58",
    "ROOM_59",
    "ROOM_68",
    "ROOM_6C",
    "SOUTH_MOUTH_Y",
    "L7_ROOM69_WEST_BOMB",
    "L7_ROOM18_NORTH_BOMB",
    "L7_ROOM08_EAST_BOMB",
    "L7_ROOM08_EAST_APPROACH",
    "L7_ROOM19_EAST_BOMB",
    "L7_ROOM19_EAST_APPROACH",
    "L7_ROOM1A_EAST_BOMB",
    "L7_ROOM1A_EAST_APPROACH",
    "EntryNorthDoorController",
    "HungryGoriyaGateController",
    "Level7BombWall",
    "Level7PathController",
    "RedCandlePickupController",
    "Room69EastController",
    "Room6AEastController",
    "Room6BEastController",
    "Room6BNorthController",
    "Room6CEastController",
    "Room68NorthController",
    "Room68DownController",
    "Room49UpController",
    "Room39LeftController",
    "Room38UpController",
    "Room09DownController",
    "Room1ACandleController",
    "Room4AReturnController",
    "Room1BKeyEastController",
    "Room58EastController",
    "Room58NorthController",
    "Room59UpController",
    "UnverifiedLevel7PathController",
    "east_of_room69_ram_id",
    "east_of_room6a_ram_id",
    "east_of_room6b_ram_id",
    "east_of_room6c_ram_id",
    "north_of_room6b_ram_id",
    "east_of_room58_ram_id",
    "north_of_room68_ram_id",
    "south_of_room68_ram_id",
    "north_of_room58_ram_id",
    "north_of_room38_ram_id",
    "north_of_room18_ram_id",
    "east_of_room08_ram_id",
    "south_of_room09_ram_id",
    "east_of_room19_ram_id",
    "east_of_room1a_ram_id",
    "cellar_of_room1a_ram_id",
    "play_of_room4a_ram_id",
    "room_4a_return_step",
    "east_of_room1b_ram_id",
    "room_1b_key_east_step",
    "west_of_room39_ram_id",
    "north_of_room49_ram_id",
    "north_of_room59_ram_id",
    "east_route_step",
    "room_49_up_step",
    "room_39_left_step",
    "room_38_up_step",
    "room_09_down_step",
    "room_6a_east_step",
    "room_6b_east_step",
    "room_6b_north_step",
    "room_6c_east_step",
    "room_68_north_step",
    "room_68_down_step",
    "room_58_north_step",
    "live_goriyas",
    "north_door_79_step",
    "north_of_entry_ram_id",
    "room69_east_step",
    "unverified_path_controller",
]
