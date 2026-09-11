"""Fixture-live Level 8 north-column one-frame policies (0x7E → 0x1E).

Promotes the settled interior chain into spine-composable controllers.
Not route eligible; not a power-on claim.  Local sword-clear specs are
never registered as ``DungeonRoomSpec`` rows.  Bomb walls use
``BombWallController``; north key/shutter doors use ``exit_door``
geometry (``DOOR_TARGETS['UP']``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.combat import in_sword_hitbox, manhattan
from zelda_i.dungeon.bomb_wall import BombWallController, BombWallPhase
from zelda_i.dungeon.engine import (
    AliveRule,
    CombatTuning,
    DoorRoute,
    DungeonPhase,
    DungeonRoomSpec,
    GenericDungeonRoomController,
    RewardKind,
    RewardSpec,
)
from zelda_i.dungeon.door_hop import door_band_goal
from zelda_i.dungeon.hop_controller import HopController, WAIT_SCROLL_B
from zelda_i.dungeon.ids import MANHANDLA_OBJECT_TYPE
from zelda_i.dungeon.ops import DOOR_TARGETS
from zelda_i.dungeon.pause_select import B_SLOT_BOMBS
from zelda_i.level8.dungeon import LEVEL8
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot
from zelda_i.walk.physics import OccupancyWalker, follow_path, predicted_xy

# Live recon rooms.  0x0C is unregistered in dungeon.ids (0x0B is "darknut");
# colour is a walkthrough correlation, not an observation.
TYPE_0C = 0x0C
STATUE_FIREBALL = 0x55
SMALL_KEY_ITEM = 0x19
# Heart-safe 0x5E (hc=3, no refill): occupancy around bodies, rear/flank sword.
CONTACT_MAN = 14
SWORD_MAN = 20
FLANK_STANDOFF = 16
ENEMY_BLOCK_R = 10
# Live leftover after the 0x6E bomb hole is (120, 189). DOWN from there
# scrolls back into 0x6E (ROM l8_npv4_5e, 232f). Occupancy ymax holds that
# plane; SOUTH_HOLD_Y refuses DOWN on the south band.
ROOM_5E_OCC_BOUNDS: tuple[int, int, int, int] = (40, 216, 77, 189)
ROOM_3E_OCC_BOUNDS: tuple[int, int, int, int] = (40, 216, 77, 189)
ROOM_3E_STATUE_BLOCKS: frozenset[tuple[int, int]] = frozenset(
    (x, y) for x in range(84, 111) for y in range(126, 151)
) | frozenset(
    (x, y) for x in range(132, 159) for y in range(126, 151)
)
SOUTH_HOLD_Y = 181
ENTRY_COLUMN_X = 120
# ROM l8_npv4_5e_inland died at x=116 (4px off). Peel past the shield column
# before inland UP. 16px matches FLANK_STANDOFF.
COLUMN_PEEL = 16
COLUMN_PEEL_3E = 48
# Statue-row waist. ROM l8_npv4_5e_west died (104,117) hunting north of this.
WAIST_Y = 141
ROOM_ENTRY = 0x7E
ROOM_MANHANDLA = 0x6E
ROOM_DARKNUT_KEY = 0x5E
ROOM_SHUTTER = 0x4E
ROOM_BLUE_DARKNUTS = 0x3E
ROOM_MAP_MANHANDLA = 0x2E
ROOM_GOHMA = 0x1E

NORTH_MANHANDLA_ROOMS = frozenset({ROOM_ENTRY, ROOM_MANHANDLA, ROOM_DARKNUT_KEY})
DARKNUT_KEY_ROOMS = frozenset(
    {
        ROOM_DARKNUT_KEY,
        ROOM_SHUTTER,
        ROOM_BLUE_DARKNUTS,
        ROOM_MAP_MANHANDLA,
        ROOM_GOHMA,
    }
)

NORTH_DOOR = DOOR_TARGETS["UP"]  # (120, 93)
BOMB_NORTH_STAND = (120, 105)
# A long shield-RNG clear can leave Link off-centre and south of the 0x3E
# statue row (~y=141); a bare y-first (120,109) then rams a statue.  Rally
# on the statue-free centre column first (south, then the north band).
BOMB_NORTH_APPROACH_3E: tuple[tuple[int, int], ...] = ((120, 157), (120, 109))
CENTER_KEY_STAND = (120, 141)
# Peel off the 0x2E centre column (map 0x17). Door push is separate so an
# overshoot north of y=93 is not walked back south.
MAP_SKIP_WAYPOINTS: tuple[tuple[int, int], ...] = ((88, 109), (120, 109))
MANHANDLA_SETTLE_FRAMES = 120
DARKNUT_SETTLE_FRAMES = 8
KEY_FREEZE_FRAMES = 40

_SWORD_PATROL: tuple[tuple[int, int], ...] = (
    (64, 109),
    (120, 109),
    (176, 109),
    (176, 141),
    (176, 173),
    (120, 173),
    (64, 173),
    (64, 141),
    (120, 141),
    (100, 125),
    (140, 157),
    (80, 157),
    (160, 125),
)


class BombWall6ENorth:
    """0x6E Manhandla north wall → 0x5E. Stand (120,105) face UP."""

    room = ROOM_MANHANDLA
    stand = BOMB_NORTH_STAND
    face = "UP"
    opens_to = ROOM_DARKNUT_KEY


class BombWall3ENorth:
    """0x3E north wall → 0x2E. Approach y=109 (statue row at y~141)."""

    room = ROOM_BLUE_DARKNUTS
    stand = BOMB_NORTH_STAND
    face = "UP"
    opens_to = ROOM_MAP_MANHANDLA


# Blue-darknut (0x0C) rooms hold 5-6 HP128 bodies that turn to block frontal
# sword hits, so the patrol clear is high-variance: fixture replays land
# ~2000f but a bad shield-RNG power-on run needs far more.  Give the
# multi-darknut rooms the budget the probe's `fight_clear` used; the
# single-Manhandla rooms keep the tight cap.  (rr-6o7.2)
_MANHANDLA_CLEAR_FRAMES = 5_000
_DARKNUT_CLEAR_FRAMES = 16_000


def _sword_clear_spec(
    room: int,
    types: tuple[int, ...],
    spec_id: str,
    *,
    max_frames: int = _MANHANDLA_CLEAR_FRAMES,
) -> DungeonRoomSpec:
    """Local fight_clear engine row. Not registered; not on L8_THROUGH."""
    return DungeonRoomSpec(
        spec_id=spec_id,
        source_room=room,
        room_id=room,
        entry=DoorRoute("DOWN", ((120, 205),)),
        enemy_types=types,
        expected_enemy_count=1,
        alive_rule=AliveRule.TYPE_AND_HP,
        combat=CombatTuning(
            patrol=_SWORD_PATROL,
            engage_distance=48,
            attack_phase=2,
            patrol_attack_period=6,
            patrol_attack_hold=3,
            engage_attack_period=5,
            engage_attack_hold=3,
        ),
        reward=RewardSpec(kind=RewardKind.CLEAR_ONLY, settle_all_dead=0),
        max_frames=max_frames,
        level=LEVEL8,
    )


CLEAR_6E_SPEC = _sword_clear_spec(
    ROOM_MANHANDLA, (MANHANDLA_OBJECT_TYPE,), "l8_clear_0x6e_manhandla"
)
CLEAR_3E_SPEC = _sword_clear_spec(
    ROOM_BLUE_DARKNUTS,
    (TYPE_0C,),
    "l8_clear_0x3e_0x0c",
    max_frames=_DARKNUT_CLEAR_FRAMES,
)
CLEAR_2E_SPEC = _sword_clear_spec(
    ROOM_MAP_MANHANDLA, (MANHANDLA_OBJECT_TYPE,), "l8_clear_0x2e_manhandla"
)


def _live_of(snap: ZeldaSnapshot, types: tuple[int, ...]) -> tuple:
    want = frozenset(types)
    return tuple(
        obj
        for obj in snap.objects
        if 1 <= obj.slot <= 12 and obj.type_id in want and obj.hp > 0
    )


def _in_front_of_shield(lx: int, ly: int, obj) -> bool:
    """True when Link is in the Darknut's facing cone (frontal shield)."""
    dx = lx - int(obj.x)
    dy = ly - int(obj.y)
    facing = int(obj.facing)
    if facing == 0x01:
        return dx > 0 and abs(dx) >= abs(dy)
    if facing == 0x02:
        return dx < 0 and abs(dx) >= abs(dy)
    if facing == 0x04:
        return dy > 0 and abs(dy) >= abs(dx)
    if facing == 0x08:
        return dy < 0 and abs(dy) >= abs(dx)
    return False


def _away_dir(lx: int, ly: int, obj) -> str:
    dx = int(obj.x) - lx
    dy = int(obj.y) - ly
    if abs(dx) >= abs(dy):
        return "LEFT" if dx > 0 else "RIGHT"
    return "UP" if dy > 0 else "DOWN"


def _toward_dir(lx: int, ly: int, obj) -> str:
    dx = int(obj.x) - lx
    dy = int(obj.y) - ly
    if abs(dx) >= abs(dy) and dx != 0:
        return "RIGHT" if dx > 0 else "LEFT"
    if dy != 0:
        return "DOWN" if dy > 0 else "UP"
    return "UP"


def _flank_candidates(obj) -> tuple[tuple[int, int], ...]:
    s = FLANK_STANDOFF
    ox, oy = int(obj.x), int(obj.y)
    facing = int(obj.facing)
    if facing == 0x08:
        rear, a, b = (ox, oy + s), (ox - s, oy), (ox + s, oy)
    elif facing == 0x04:
        rear, a, b = (ox, oy - s), (ox - s, oy), (ox + s, oy)
    elif facing == 0x01:
        rear, a, b = (ox - s, oy), (ox, oy - s), (ox, oy + s)
    elif facing == 0x02:
        rear, a, b = (ox + s, oy), (ox, oy - s), (ox, oy + s)
    else:
        rear, a, b = (ox, oy + s), (ox - s, oy), (ox + s, oy)
    return (rear, a, b)


def _enemy_disk(
    obj, xy: tuple[int, int], dest: tuple[int, int] | None
) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    ox, oy = int(obj.x), int(obj.y)
    r = ENEMY_BLOCK_R
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            if abs(dx) + abs(dy) > r:
                continue
            cell = (ox + dx, oy + dy)
            if cell == xy or cell == dest:
                continue
            cells.add(cell)
    return cells


def _shield_axis(xy: tuple[int, int], obj) -> set[tuple[int, int]]:
    """Block the facing-axis corridor so BFS cannot walk into the shield."""
    if int(getattr(obj, "type_id", TYPE_0C)) != TYPE_0C:
        return set()
    if not _in_front_of_shield(xy[0], xy[1], obj):
        return set()
    ox, oy = int(obj.x), int(obj.y)
    lx, ly = xy
    cells: set[tuple[int, int]] = set()
    facing = int(obj.facing)
    if facing in (0x04, 0x08):
        y0, y1 = (ly, oy) if ly < oy else (oy, ly)
        for y in range(y0, y1 + 1):
            cells.add((ox, y))
    else:
        x0, x1 = (lx, ox) if lx < ox else (ox, lx)
        for x in range(x0, x1 + 1):
            cells.add((x, oy))
    cells.discard(xy)
    return cells


def _goto(
    snap: ZeldaSnapshot, tx: int, ty: int, *, reason: str, tol: int = 4
) -> FrameAction | None:
    """One x-then-y step toward (tx, ty). None when inside *tol* (ops.goto)."""
    if abs(snap.link_x - tx) > tol:
        btn = "RIGHT" if snap.link_x < tx else "LEFT"
        return FrameAction(nes_action(btn), f"{reason}_x")
    if abs(snap.link_y - ty) > tol:
        btn = "DOWN" if snap.link_y < ty else "UP"
        return FrameAction(nes_action(btn), f"{reason}_y")
    return None


def _north_door(snap: ZeldaSnapshot, *, reason: str = "north_door") -> FrameAction:
    """Leftover-relative UP. Off-column leftover uses door x, not leftover x.

    Never walk south after overshooting the north band — that oscillates on
    the door tile (live 0x7E y=87/89). Knockback at x=208 must LEFT, not UP.
    """
    tx, ty = door_band_goal(
        "UP", (int(snap.link_x), int(snap.link_y)), NORTH_DOOR
    )
    if abs(snap.link_x - tx) > 4:
        btn = "RIGHT" if snap.link_x < tx else "LEFT"
        return FrameAction(nes_action(btn), f"{reason}_x")
    if snap.link_y > ty + 4:
        return FrameAction(nes_action("UP"), f"{reason}_y")
    return FrameAction(nes_action("UP"), f"{reason}_push")


@dataclass(kw_only=True)
class _NorthColumnBase(HopController):
    """Shared L8 north-column guards: fixture-live, no writes, known rooms."""

    wait_modes: tuple[int, ...] = WAIT_SCROLL_B
    require_level: int = LEVEL8
    writes: int = 0
    evidence: str = "fixture-live"
    route_eligible: bool = False
    _env: Any = field(default=None, init=False, repr=False)
    _room: int | None = field(default=None, init=False)
    _room_frames: int = field(default=0, init=False)
    _traveled: bool = field(default=False, init=False)
    _clear: GenericDungeonRoomController | None = field(default=None, init=False, repr=False)
    _wall: BombWallController | None = field(default=None, init=False, repr=False)
    _walker: OccupancyWalker = field(default_factory=lambda: OccupancyWalker(sticky=True), init=False, repr=False)
    _map_wp: int = field(default=0, init=False)
    _key_wait: int = field(default=0, init=False)
    _keys_in: int | None = field(default=None, init=False)
    _bomb_cd: int = field(default=0, init=False)
    _3e_peeled: bool = field(default=False, init=False)
    _3e_wall_bombed: bool = field(default=False, init=False)
    _bomb_retreat: int = field(default=0, init=False)
    _retreat_dir: str = field(default="DOWN", init=False)

    def bind_env(self, env: Any) -> None:
        self._env = env
        if self._wall is not None:
            self._wall.bind_env(env)

    def scroll_action(self, snap: ZeldaSnapshot) -> FrameAction:
        del snap
        return FrameAction(nes_action("UP"), "north_scroll")

    def _track_room(self, snap: ZeldaSnapshot) -> None:
        if self._room is None:
            self._room = int(snap.screen)
            return
        if int(snap.screen) != self._room:
            self._room = int(snap.screen)
            self._room_frames = 0
            self._traveled = True
            self._clear = None
            self._wall = None
            self._walker = OccupancyWalker(sticky=True)
            self._map_wp = 0
            self._key_wait = 0
            self._keys_in = None
            self._bomb_cd = 0
            self._3e_peeled = False
            self._3e_wall_bombed = False
            self._bomb_retreat = 0
            self._retreat_dir = "DOWN"
            return
        self._room_frames += 1

    def _spawn_wait(self, snap: ZeldaSnapshot, frames: int) -> FrameAction | None:
        if self._clear is not None and self._clear.max_live_enemies > 0:
            return None
        if self._traveled and self._room_frames < frames:
            return FrameAction(
                nes_idle_action(), f"settle_0x{snap.screen:02x}"
            )
        return None

    def _fight(self, snap: ZeldaSnapshot, spec: DungeonRoomSpec) -> FrameAction:
        if self._clear is None:
            self._clear = GenericDungeonRoomController(spec)
            self._clear.phase = DungeonPhase.FIGHT
        action = self._clear.step(snap)
        if self._clear.phase is DungeonPhase.FAILED:
            note = (
                self._clear.notes[-1]
                if self._clear.notes
                else f"clear_0x{snap.screen:02x}_failed"
            )
            if "left_target_room" in note or action.reason == "left_target_room":
                return self.mark_fail(
                    f"clear_0x{snap.screen:02x}_first_departure_guard"
                )
            return self.mark_fail(note)
        return action

    def _bomb(
        self,
        snap: ZeldaSnapshot,
        wall: Any,
        *,
        approach: tuple[tuple[int, int], ...] = (),
    ) -> FrameAction:
        if snap.bombs <= 0:
            return self.mark_fail(f"no_bombs_0x{snap.screen:02x}")
        if self._wall is None:
            self._wall = BombWallController(
                wall=wall,
                level=LEVEL8,
                approach_waypoints=approach,
                # 0x3E north wall: residual darknut/statue-projectile state
                # after a long shield-RNG clear can stall the approach to the
                # (120,105) stand well past the old 8k budget.
                max_frames=16_000,
                select_item=B_SLOT_BOMBS,
            )
            if self._env is not None:
                self._wall.bind_env(self._env)
        action = self._wall.step(snap)
        if self._wall.phase is BombWallPhase.FAILED:
            note = (
                self._wall.notes[-1]
                if self._wall.notes
                else f"bomb_north_0x{snap.screen:02x}_failed"
            )
            return self.mark_fail(note)
        return action

    def _north_key(self, snap: ZeldaSnapshot, *, reason: str) -> FrameAction:
        if snap.keys <= 0:
            return self.mark_fail(f"no_keys_0x{snap.screen:02x}")
        return _north_door(snap, reason=reason)

    def _path_around(
        self,
        xy: tuple[int, int],
        dest: tuple[int, int],
        bodies: tuple,
    ) -> str | None:
        """BFS with live bodies as occupancy. No path → None (caller stands)."""
        extra: set[tuple[int, int]] = set()
        for obj in bodies:
            extra |= _enemy_disk(obj, xy, dest)
            extra |= _shield_axis(xy, obj)
        return self._walker.next_dir(xy, dest, extra_blocked=extra, sticky=True)

    def _flank_dest(
        self, xy: tuple[int, int], target, bodies: tuple
    ) -> tuple[int, int] | None:
        grid = self._walker.grid
        for cand in _flank_candidates(target):
            dest = (
                min(max(int(cand[0]), grid.xmin), grid.xmax),
                min(max(int(cand[1]), WAIST_Y), grid.ymax),
            )
            if dest == xy or self._path_around(xy, dest, bodies) is not None:
                return dest
        return None

    def _bind_5e_grid(self) -> None:
        g = self._walker.grid
        xmin, xmax, ymin, ymax = ROOM_5E_OCC_BOUNDS
        if (g.xmin, g.xmax, g.ymin, g.ymax) != (xmin, xmax, ymin, ymax):
            g.xmin, g.xmax, g.ymin, g.ymax = xmin, xmax, ymin, ymax

    def _bind_3e_grid(self) -> None:
        g = self._walker.grid
        xmin, xmax, ymin, ymax = ROOM_3E_OCC_BOUNDS
        if (g.xmin, g.xmax, g.ymin, g.ymax) != (xmin, xmax, ymin, ymax):
            g.xmin, g.xmax, g.ymin, g.ymax = xmin, xmax, ymin, ymax
        g.blocked.update(ROOM_3E_STATUE_BLOCKS)

    def _south_hold(self, xy: tuple[int, int], direction: str) -> str | None:
        """No DOWN off the 0x5E south bomb hole. Peel or stand."""
        if direction != "DOWN" or xy[1] < SOUTH_HOLD_Y:
            return direction
        peel = "LEFT" if xy[0] >= 120 else "RIGHT"
        nxt = predicted_xy(xy[0], xy[1], peel)
        if not self._walker.grid.passable(*nxt):
            return None
        return peel

    def _up_axis_free(self, xy: tuple[int, int], bodies: tuple) -> bool:
        """UP off the south band is legal: not a shield column, not contact."""
        nxt = predicted_xy(xy[0], xy[1], "UP")
        if not self._walker.grid.passable(*nxt):
            return False
        # Still on x=120±15 with a south-facing 0x0C on the entry column:
        # 4px LEFT (ROM death x=116) is not off-axis. Peel first.
        if abs(xy[0] - ENTRY_COLUMN_X) < COLUMN_PEEL:
            for obj in bodies:
                if int(getattr(obj, "type_id", TYPE_0C)) != TYPE_0C:
                    continue
                if int(obj.facing) == 0x04 and abs(int(obj.x) - ENTRY_COLUMN_X) <= 8:
                    return False
        for obj in bodies:
            if manhattan(nxt[0], nxt[1], obj.x, obj.y) < CONTACT_MAN:
                return False
            if nxt in _shield_axis(xy, obj):
                return False
        return True

    def _column_peel_dir(
        self, xy: tuple[int, int], bodies: tuple, *, peel_dist: int = COLUMN_PEEL
    ) -> str | None:
        """LEFT (else RIGHT) off x=120 until |x-120|>=peel_dist. None if boxed.

        Holds through contact: a 1px backstep was the 459f (116,189) death.
        """
        del bodies
        if abs(xy[0] - ENTRY_COLUMN_X) >= peel_dist:
            return None
        for peel in ("LEFT", "RIGHT"):
            nxt = predicted_xy(xy[0], xy[1], peel)
            if self._walker.grid.passable(*nxt):
                return peel
        return None

    def _slash(self, btn: str) -> FrameAction:
        self._walker.last_dir = None
        if self._room_frames % 6 < 4:
            return FrameAction(nes_action(btn, "A"), "combat_slash")
        return FrameAction(nes_action(btn), "combat_slash")

    def _flank_rear_slash(self, xy: tuple[int, int], live: tuple) -> FrameAction | None:
        """Flank or rear attack opportunities on any live Darknut."""
        for d in live:
            ox, oy, ofc = int(d.x), int(d.y), int(d.facing)
            if ofc in (0x04, 0x08) and abs(oy - xy[1]) <= 8 and 10 <= abs(ox - xy[0]) <= 22:
                return self._slash("RIGHT" if xy[0] < ox else "LEFT")
            if ofc in (0x01, 0x02) and abs(ox - xy[0]) <= 8 and 10 <= abs(oy - xy[1]) <= 22:
                return self._slash("DOWN" if xy[1] < oy else "UP")
            if ofc == 0x04 and abs(ox - xy[0]) <= 6 and 10 <= (oy - xy[1]) <= 22:
                return self._slash("DOWN")
            if ofc == 0x08 and abs(ox - xy[0]) <= 6 and 10 <= (xy[1] - oy) <= 22:
                return self._slash("UP")
            if ofc == 0x01 and abs(oy - xy[1]) <= 6 and 10 <= (ox - xy[0]) <= 22:
                return self._slash("RIGHT")
            if ofc == 0x02 and abs(oy - xy[1]) <= 6 and 10 <= (xy[0] - ox) <= 22:
                return self._slash("LEFT")
        return None

    def _heart_safe_darknut(self, snap: ZeldaSnapshot) -> FrameAction:
        """0x5E type 0x0C: side-stepping, south entry peel, rear/flank attacks."""
        room = int(snap.screen)
        if room == ROOM_BLUE_DARKNUTS:
            return self._combat_3e(snap)

        xy = (int(snap.link_x), int(snap.link_y))
        self._bind_5e_grid()

        live = _live_of(snap, (TYPE_0C,))
        self._walker.observe(xy)

        # Stand if boxed by obstacles
        if all(
            not self._walker.grid.passable(*predicted_xy(xy[0], xy[1], d))
            for d in ("UP", "DOWN", "LEFT", "RIGHT")
        ):
            self._walker.last_dir = None
            return FrameAction(nes_idle_action(), "occupancy_stand")

        bodies = live + _live_of(snap, (STATUE_FIREBALL,))

        # Column peel before contact: hold LEFT until off entry column, then inland UP
        if xy[1] >= SOUTH_HOLD_Y:
            peel = self._column_peel_dir(xy, bodies, peel_dist=COLUMN_PEEL)
            if peel is not None:
                self._walker.last_dir = peel
                return FrameAction(nes_action(peel), "column_peel")
            self._walker.last_dir = "UP"
            return FrameAction(nes_action("UP"), "inland_leave")

        if not live:
            self._walker.last_dir = None
            return FrameAction(nes_idle_action(), "occupancy_stand")

        target = min(live, key=lambda o: manhattan(xy[0], xy[1], o.x, o.y))
        dist = manhattan(xy[0], xy[1], target.x, target.y)

        slash = self._flank_rear_slash(xy, live)
        if slash is not None:
            return slash


        # Immediate threat: side-step perpendicular to incoming Darknut
        for d in live:
            ox, oy, ofc = int(d.x), int(d.y), int(d.facing)
            d_dist = abs(xy[0] - ox) + abs(xy[1] - oy)
            if d_dist <= 28:
                if ofc == 0x04 and xy[1] >= oy and abs(xy[0] - ox) <= 12:
                    btn = "LEFT" if (xy[0] <= ox and xy[0] > 48) or xy[0] >= 200 else "RIGHT"
                    nxt = predicted_xy(xy[0], xy[1], btn)
                    if not self._walker.grid.passable(*nxt):
                        btn = "RIGHT" if btn == "LEFT" else "LEFT"
                        nxt = predicted_xy(xy[0], xy[1], btn)
                    if self._walker.grid.passable(*nxt):
                        self._walker.last_dir = None
                        return FrameAction(nes_action(btn), "combat_flank")
                if ofc == 0x08 and xy[1] <= oy and abs(xy[0] - ox) <= 12:
                    btn = "LEFT" if (xy[0] <= ox and xy[0] > 48) or xy[0] >= 200 else "RIGHT"
                    nxt = predicted_xy(xy[0], xy[1], btn)
                    if not self._walker.grid.passable(*nxt):
                        btn = "RIGHT" if btn == "LEFT" else "LEFT"
                        nxt = predicted_xy(xy[0], xy[1], btn)
                    if self._walker.grid.passable(*nxt):
                        self._walker.last_dir = None
                        return FrameAction(nes_action(btn), "combat_flank")
                if ofc == 0x02 and xy[0] <= ox and abs(xy[1] - oy) <= 12:
                    btn = "UP" if (xy[1] <= oy and xy[1] > 96) or xy[1] >= 173 else "DOWN"
                    nxt = predicted_xy(xy[0], xy[1], btn)
                    if not self._walker.grid.passable(*nxt):
                        btn = "DOWN" if btn == "UP" else "UP"
                        nxt = predicted_xy(xy[0], xy[1], btn)
                    if self._walker.grid.passable(*nxt):
                        self._walker.last_dir = None
                        return FrameAction(nes_action(btn), "combat_flank")
                if ofc == 0x01 and xy[0] >= ox and abs(xy[1] - oy) <= 12:
                    btn = "UP" if (xy[1] <= oy and xy[1] > 96) or xy[1] >= 173 else "DOWN"
                    nxt = predicted_xy(xy[0], xy[1], btn)
                    if not self._walker.grid.passable(*nxt):
                        btn = "DOWN" if btn == "UP" else "UP"
                        nxt = predicted_xy(xy[0], xy[1], btn)
                    if self._walker.grid.passable(*nxt):
                        self._walker.last_dir = None
                        return FrameAction(nes_action(btn), "combat_flank")

        # Tactical approach: offset onto the parallel flank lane
        tfc = int(target.facing)
        tx, ty = int(target.x), int(target.y)
        if tfc in (0x04, 0x08):
            cand_x = min(max(tx - 16 if tx >= 120 else tx + 16, 48), 200)
            if abs(xy[0] - cand_x) > 4:
                btn = "RIGHT" if xy[0] < cand_x else "LEFT"
                nxt = predicted_xy(xy[0], xy[1], btn)
                if self._walker.grid.passable(*nxt):
                    self._walker.last_dir = None
                    return FrameAction(nes_action(btn), "combat_approach")
            if abs(xy[1] - ty) > 6:
                btn = "DOWN" if xy[1] < ty else "UP"
                nxt = predicted_xy(xy[0], xy[1], btn)
                if self._walker.grid.passable(*nxt):
                    self._walker.last_dir = None
                    return FrameAction(nes_action(btn), "combat_approach")
            face = "RIGHT" if xy[0] < tx else "LEFT"
            self._walker.last_dir = None
            return FrameAction(nes_action(face), "combat_approach")
        else:
            cand_y = min(max(ty - 16 if ty >= 150 else ty + 16, 96), 173)
            if abs(xy[1] - cand_y) > 4:
                btn = "DOWN" if xy[1] < cand_y else "UP"
                nxt = predicted_xy(xy[0], xy[1], btn)
                if self._walker.grid.passable(*nxt):
                    self._walker.last_dir = None
                    return FrameAction(nes_action(btn), "combat_approach")
            if abs(xy[0] - tx) > 6:
                btn = "RIGHT" if xy[0] < tx else "LEFT"
                nxt = predicted_xy(xy[0], xy[1], btn)
                if self._walker.grid.passable(*nxt):
                    self._walker.last_dir = None
                    return FrameAction(nes_action(btn), "combat_approach")
            face = "DOWN" if xy[1] < ty else "UP"
            self._walker.last_dir = None
            return FrameAction(nes_action(face), "combat_approach")

    def _combat_3e(self, snap: ZeldaSnapshot) -> FrameAction:
        """Room 0x3E Blue Darknuts: west corridor kiting, tactical bombs, and north wall clear."""
        if self._bomb_cd > 0:
            self._bomb_cd -= 1

        xy = (int(snap.link_x), int(snap.link_y))
        live = _live_of(snap, (TYPE_0C,))
        fireballs = _live_of(snap, (STATUE_FIREBALL,))
        self._bind_3e_grid()

        # 1. South doorway entry
        if xy[1] > 189:
            self._walker.last_dir = "UP"
            return FrameAction(nes_action("UP"), "combat_door_enter")

        # 2. Initial peel to x=64, y=165 for corridor tactical placement
        if not self._3e_peeled:
            if xy[0] > 64 and xy[1] >= SOUTH_HOLD_Y:
                self._walker.last_dir = "LEFT"
                return FrameAction(nes_action("LEFT"), "column_peel")
            if xy[1] > 165:
                self._walker.last_dir = "UP"
                return FrameAction(nes_action("UP"), "peel_north")
            self._3e_peeled = True

        # 3. Post-bomb retreat to safety
        if self._bomb_retreat > 0:
            self._bomb_retreat -= 1
            if self._retreat_dir == "DOWN":
                btn = "DOWN" if xy[1] < 181 else None
            else:
                btn = "UP" if xy[1] > 93 else None
            if btn is not None:
                self._walker.last_dir = btn
                return FrameAction(nes_action(btn), f"bomb_retreat_{btn.lower()}")

        # 4. At north bomb stand (120, <=105): place bomb facing UP and push through
        if abs(xy[0] - 120) <= 6 and xy[1] <= 105:
            if snap.bombs > 0 and not self._3e_wall_bombed:
                self._3e_wall_bombed = True
                self._bomb_cd = 75
                self._walker.last_dir = "UP"
                return FrameAction(nes_action("UP", "B"), "bomb_north_wall")
            self._walker.last_dir = "UP"
            return FrameAction(nes_action("UP"), "push_north_hole")

        # 5. Fireball avoidance
        for fb in fireballs:
            fx, fy = int(fb.x), int(fb.y)
            if manhattan(xy[0], xy[1], fx, fy) <= 20:
                if abs(fy - xy[1]) <= 8:
                    btn = "DOWN" if xy[1] >= fy and xy[1] < 181 else "UP"
                    self._walker.last_dir = btn
                    return FrameAction(nes_action(btn), f"dodge_fb_{btn.lower()}")
                if abs(fx - xy[0]) <= 8:
                    if xy[1] <= 100 and xy[0] >= 96:
                        btn = "RIGHT"
                    else:
                        btn = "LEFT" if xy[0] <= fx and xy[0] > 48 else "RIGHT"
                    self._walker.last_dir = btn
                    return FrameAction(nes_action(btn), f"dodge_fb_{btn.lower()}")

        # 6. Immediate incoming threats (enemies heading toward Link)
        for d in live:
            ox, oy, ofc = int(d.x), int(d.y), int(d.facing)
            if ofc == 0x04 and oy < xy[1] and abs(ox - xy[0]) <= 12 and (xy[1] - oy) <= 36:
                if 16 <= (xy[1] - oy) <= 40 and snap.bombs >= 2 and self._bomb_cd <= 0:
                    self._bomb_cd = 75
                    self._bomb_retreat = 45
                    self._retreat_dir = "DOWN"
                    self._walker.last_dir = "UP"
                    return FrameAction(nes_action("UP", "B"), "threat_bomb_up")
                if abs(ox - xy[0]) <= 6:
                    btn = "RIGHT" if (xy[0] <= 64 and xy[0] < 72) else "LEFT"
                    self._walker.last_dir = btn
                    return FrameAction(nes_action(btn), "threat_sidestep_x")
                btn = "DOWN" if xy[1] < 181 else "UP"
                self._walker.last_dir = btn
                return FrameAction(nes_action(btn), "threat_retreat_down")

            if ofc == 0x08 and oy > xy[1] and abs(ox - xy[0]) <= 12 and (oy - xy[1]) <= 36:
                if 16 <= (oy - xy[1]) <= 40 and snap.bombs >= 2 and self._bomb_cd <= 0:
                    self._bomb_cd = 75
                    self._bomb_retreat = 45
                    self._retreat_dir = "UP"
                    self._walker.last_dir = "DOWN"
                    return FrameAction(nes_action("DOWN", "B"), "threat_bomb_down")
                if abs(ox - xy[0]) <= 6:
                    btn = "RIGHT" if (xy[0] <= 64 and xy[0] < 72) else "LEFT"
                    self._walker.last_dir = btn
                    return FrameAction(nes_action(btn), "threat_sidestep_x")
                btn = "UP" if xy[1] > 93 else "DOWN"
                self._walker.last_dir = btn
                return FrameAction(nes_action(btn), "threat_retreat_up")

            if ofc == 0x02 and ox > xy[0] and abs(oy - xy[1]) <= 14 and (ox - xy[0]) <= 40:
                btn = "UP" if xy[1] > 125 else "DOWN"
                self._walker.last_dir = btn
                return FrameAction(nes_action(btn), "threat_sidestep_y")

            if ofc == 0x01 and ox < xy[0] and abs(oy - xy[1]) <= 12 and (xy[0] - ox) <= 36:
                btn = "DOWN" if xy[1] < 181 else "UP"
                self._walker.last_dir = btn
                return FrameAction(nes_action(btn), "threat_sidestep_y")

        # 7. Flank / rear attacks on ANY Darknut
        slash = self._flank_rear_slash(xy, live)
        if slash is not None:
            return slash


        # 8. Advance north towards BOMB_NORTH_STAND if corridor & north passage are clear
        corridor_clear = not any(d.x <= 76 and d.y >= 93 for d in live)
        north_clear = not any(93 <= d.y <= 116 and d.x <= 130 for d in live)

        if corridor_clear and north_clear:
            if xy[1] > 109 and xy[0] <= 72:
                self._walker.last_dir = "UP"
                return FrameAction(nes_action("UP"), "advance_north")
            if xy[0] < 120 and xy[1] <= 112:
                self._walker.last_dir = "RIGHT"
                return FrameAction(nes_action("RIGHT"), "advance_east")
            if xy[1] > 105 and abs(xy[0] - 120) <= 4:
                self._walker.last_dir = "UP"
                return FrameAction(nes_action("UP"), "advance_stand")
            if abs(xy[0] - 120) <= 4 and xy[1] <= 105:
                if snap.bombs > 0 and not self._3e_wall_bombed:
                    self._3e_wall_bombed = True
                    self._bomb_cd = 75
                    self._walker.last_dir = "UP"
                    return FrameAction(nes_action("UP", "B"), "bomb_north_wall")
                self._walker.last_dir = "UP"
                return FrameAction(nes_action("UP"), "push_north_hole")

        # 9. Kite in west corridor
        if xy[0] <= 72:
            if xy[1] < 133:
                self._walker.last_dir = "DOWN"
                return FrameAction(nes_action("DOWN"), "kite_down")
            if xy[1] > 165:
                self._walker.last_dir = "UP"
                return FrameAction(nes_action("UP"), "kite_up")
            if live:
                target = min(live, key=lambda o: manhattan(xy[0], xy[1], o.x, o.y))
                btn = "UP" if target.y < xy[1] else "DOWN"
                self._walker.last_dir = btn
                return FrameAction(nes_action(btn), f"face_target_{btn.lower()}")

        self._walker.last_dir = None
        return FrameAction(nes_idle_action(), "combat_stand")

    def report(self) -> dict[str, Any]:
        out = super().report()
        out.update(
            {
                "evidence": self.evidence,
                "route_eligible": self.route_eligible,
                "natural_entry": False,
                "writes": self.writes,
                "notes": list(self.notes),
            }
        )
        return out


@dataclass(kw_only=True)
class Level8NorthManhandlaController(_NorthColumnBase):
    """0x7E UP → clear 0x6E sword-only → bomb-N (120,105) → 0x5E."""

    spec_id: str = "level8_north_manhandla_bomb"
    max_frames: int = 26_000
    done_reason: str = "arrived_0x5e"

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        return (
            snap.level == LEVEL8
            and snap.screen == ROOM_DARKNUT_KEY
            and snap.mode == PLAY_MODE
            and not snap.transitioning
        )

    def on_arrive(self, snap: ZeldaSnapshot) -> str:
        return f"arrived_0x{snap.screen:02x}_{snap.link_x}_{snap.link_y}"

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        self._track_room(snap)
        if snap.screen == ROOM_ENTRY:
            return _north_door(snap, reason="free_north_0x7e")
        if snap.screen == ROOM_MANHANDLA:
            if _live_of(snap, (MANHANDLA_OBJECT_TYPE,)):
                return self._fight(snap, CLEAR_6E_SPEC)
            wait = self._spawn_wait(snap, MANHANDLA_SETTLE_FRAMES)
            if wait is not None:
                return wait
            return self._bomb(snap, BombWall6ENorth())
        return self.mark_fail(f"l8_north_unknown_room_0x{snap.screen:02x}")


@dataclass(kw_only=True)
class Level8DarknutKeyController(_NorthColumnBase):
    """0x5E clear/key + shutter-N 0x4E + key-N 0x3E + bomb-N 0x2E + key-N 0x1E.

    0x4E mixed census is not walked.  0x2E map (room_item 0x17) is skipped
    on the y=109 band.  Stops at settled 0x1E; does not fight the 0x33 body.
    """

    spec_id: str = "level8_darknut_key_up"
    # 0x5E + 0x3E can each need a full _DARKNUT_CLEAR_FRAMES shield-RNG fight,
    # plus the 0x2E Manhandla and the inter-room walks.
    max_frames: int = 55_000
    done_reason: str = "arrived_0x1e"

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        return (
            snap.level == LEVEL8
            and snap.screen == ROOM_GOHMA
            and snap.mode == PLAY_MODE
            and not snap.transitioning
        )

    def on_arrive(self, snap: ZeldaSnapshot) -> str:
        return f"arrived_0x{snap.screen:02x}_{snap.link_x}_{snap.link_y}"

    def _map_skip_north(self, snap: ZeldaSnapshot) -> FrameAction:
        while self._map_wp < len(MAP_SKIP_WAYPOINTS):
            tx, ty = MAP_SKIP_WAYPOINTS[self._map_wp]
            action = _goto(snap, tx, ty, reason=f"map_skip_{self._map_wp}", tol=5)
            if action is not None:
                return action
            self._map_wp += 1
        return _north_door(snap, reason="north_key_0x2e")

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        self._track_room(snap)
        room = int(snap.screen)
        if room == ROOM_MANHANDLA:
            # Open 0x6E bomb hole: knockback/DOWN leftover must re-enter 0x5E.
            return _north_door(snap, reason="reenter_north_0x6e")
        if room == ROOM_DARKNUT_KEY:
            if self._keys_in is None:
                self._keys_in = int(snap.keys)
            live = _live_of(snap, (TYPE_0C,))
            any_0c = any(obj.type_id == TYPE_0C for obj in snap.objects if 1 <= obj.slot <= 12)
            if live or (any_0c and self._room_frames < 40 and snap.keys <= self._keys_in):
                return self._heart_safe_darknut(snap)
            wait = self._spawn_wait(snap, DARKNUT_SETTLE_FRAMES)
            if wait is not None:
                return wait
            # room_item 0x19 can linger after the natural pickup; latch on
            # the key-count rise (live 9→10) so we do not walk back south.
            if snap.keys <= self._keys_in and snap.room_item_id == SMALL_KEY_ITEM:
                action = _goto(
                    snap, CENTER_KEY_STAND[0], CENTER_KEY_STAND[1],
                    reason="center_key", tol=3,
                )
                if action is not None:
                    return action
                self._key_wait += 1
                if self._key_wait < KEY_FREEZE_FRAMES:
                    return FrameAction(nes_idle_action(), "center_key_freeze")
            return _north_door(snap, reason="shutter_north_0x5e")
        if room == ROOM_SHUTTER:
            # Mixed 0x4E census is not a clear target; north key is live.
            return self._north_key(snap, reason="key_north_0x4e")
        if room == ROOM_BLUE_DARKNUTS:
            live = _live_of(snap, (TYPE_0C,))
            any_0c = any(obj.type_id == TYPE_0C for obj in snap.objects if 1 <= obj.slot <= 12)
            if live or (any_0c and self._room_frames < 40):
                return self._heart_safe_darknut(snap)
            wait = self._spawn_wait(snap, DARKNUT_SETTLE_FRAMES)
            if wait is not None:
                return wait
            return self._bomb(
                snap, BombWall3ENorth(), approach=BOMB_NORTH_APPROACH_3E
            )
        if room == ROOM_MAP_MANHANDLA:
            if _live_of(snap, (MANHANDLA_OBJECT_TYPE,)):
                return self._fight(snap, CLEAR_2E_SPEC)
            wait = self._spawn_wait(snap, MANHANDLA_SETTLE_FRAMES)
            if wait is not None:
                return wait
            return self._map_skip_north(snap)
        return self.mark_fail(f"l8_north_unknown_room_0x{snap.screen:02x}")


def make_north_manhandla_controller() -> Level8NorthManhandlaController:
    return Level8NorthManhandlaController()


def make_darknut_key_controller() -> Level8DarknutKeyController:
    return Level8DarknutKeyController()


__all__ = [
    "BOMB_NORTH_APPROACH_3E",
    "BOMB_NORTH_STAND",
    "BombWall3ENorth",
    "BombWall6ENorth",
    "COLUMN_PEEL_3E",
    "DARKNUT_KEY_ROOMS",
    "Level8DarknutKeyController",
    "Level8NorthManhandlaController",
    "MAP_SKIP_WAYPOINTS",
    "NORTH_MANHANDLA_ROOMS",
    "ROOM_3E_OCC_BOUNDS",
    "ROOM_3E_STATUE_BLOCKS",
    "ROOM_BLUE_DARKNUTS",
    "ROOM_DARKNUT_KEY",
    "ROOM_ENTRY",
    "ROOM_GOHMA",
    "ROOM_MANHANDLA",
    "ROOM_MAP_MANHANDLA",
    "ROOM_SHUTTER",
    "TYPE_0C",
    "make_darknut_key_controller",
    "make_north_manhandla_controller",
]
