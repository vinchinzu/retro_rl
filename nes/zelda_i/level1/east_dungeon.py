"""Level 1 east wing specs: 0x44 Goriya, 0x45 Wallmaster.

Specs + the Survival 0x44 bounds-engage overlay. Clean M5 keeps
``ROOM_44_SPEC`` / ``ROOM_45_SPEC``. Stop predicates stay on the specs.
"""

from __future__ import annotations

from dataclasses import replace

from zelda_i.combat import should_swing_at
from zelda_i.dungeon.engine import (
    AliveRule,
    CombatTuning,
    DoorRoute,
    DungeonRoomSpec,
    GenericDungeonRoomController,
    GORIYA_OBJECT_TYPE,
    RewardKind,
    RewardSpec,
    WALLMASTER_OBJECT_TYPE,
    register_room_spec,
)
from zelda_i.level1.path import LEVEL_1
from zelda_i.ram import ZeldaObject, ZeldaSnapshot

# Open floor at door height. Live timeout sat at (87, 101) on the north
# statue band: patrol included (80, 93) and engage=64 never reached the
# Goriyas. Stay on y=141 (west door → east door). Occupancy chase blocks
# statue cells on a miss and BFS-replans; no path falls back to this line.
_ROOM_44_PATROL: tuple[tuple[int, int], ...] = (
    (48, 141),
    (80, 141),
    (120, 141),
    (160, 141),
    (192, 141),
)

# Stay inland. Dormant Wallmasters at x=0 still grab on the west door
# (x=32) after TYPE_AND_HP treats them as dead.
_WALLMASTER_PATROL: tuple[tuple[int, int], ...] = (
    (32, 117),
    (80, 117),
    (120, 117),
    (160, 117),
    (80, 117),
    (32, 141),
)

ROOM_44_SPEC = DungeonRoomSpec(
    spec_id="level1_room44",
    source_room=0x43,
    room_id=0x44,
    entry=DoorRoute(
        "RIGHT",
        ((120, 93), (208, 93), (208, 141)),
    ),
    enemy_types=(GORIYA_OBJECT_TYPE,),
    expected_enemy_count=3,
    alive_rule=AliveRule.TYPE_AND_HP,
    combat=CombatTuning(
        patrol=_ROOM_44_PATROL,
        engage_distance=64,
        patrol_attack_period=8,
        patrol_attack_hold=4,
        attack_phase=7,
        occupancy_patrol=True,
    ),
    reward=RewardSpec(kind=RewardKind.CLEAR_ONLY),
    room_item_id=0x1D,
    level=LEVEL_1,
)

# Survival bow-splice: occupancy boxed at (40,93) (v3). y=141-only left
# two Goriyas (v4 leftover (73,133)). 3-row loop left two (v5 (56,141)).
# last_live=2 is two type-0x06 HP>0; thrown boomerang is 0x5C slot 11.
# Occupancy chase on the open floor; xmin=16 so west leftovers are not
# boxed at default xmin=40. ymin=109 excludes the north door y=93.
# Not Clean.
_ROOM_44_SURVIVAL_PATROL: tuple[tuple[int, int], ...] = (
    (48, 117),
    (120, 117),
    (192, 117),
    (192, 141),
    (120, 141),
    (48, 141),
    (48, 165),
    (120, 165),
    (192, 165),
)
_ROOM_44_SURVIVAL_BOUNDS: tuple[int, int, int, int] = (16, 216, 109, 189)
ROOM_44_SURVIVAL_SPEC = replace(
    ROOM_44_SPEC,
    spec_id="level1_room44_survival",
    combat=replace(
        ROOM_44_SPEC.combat,
        patrol=_ROOM_44_SURVIVAL_PATROL,
        engage_distance=80,
        attack_phase=6,
        occupancy_patrol=True,
        occupancy_bounds=_ROOM_44_SURVIVAL_BOUNDS,
    ),
)

ROOM_45_SPEC = DungeonRoomSpec(
    spec_id="level1_room45",
    source_room=0x44,
    room_id=0x45,
    entry=DoorRoute(
        "RIGHT",
        (
            (80, 101),
            (80, 93),
            (160, 93),
            (160, 101),
            (208, 101),
            (208, 141),
        ),
    ),
    enemy_types=(WALLMASTER_OBJECT_TYPE,),
    expected_enemy_count=8,
    alive_rule=AliveRule.TYPE_AND_HP,
    combat=CombatTuning(
        patrol=_WALLMASTER_PATROL,
        # Dormant Wallmasters sit just outside the wall (x=0).  A wider
        # engage radius makes Link face and slash into the doorway instead of
        # walking a vertical patrol forever once only those slots remain.
        engage_distance=80,
        engage_dominant_axis=True,
        attack_phase=0,
        patrol_attack_period=8,
        patrol_attack_hold=4,
    ),
    reward=RewardSpec(
        kind=RewardKind.FIXED_INVENTORY,
        inventory_field="keys",
        # Single south-wall target. Hunt the live stand (152, 189) via the
        # east column. Do not linger on the south wall.
        waypoints=(
            (160, 141),
            (160, 173),
            (152, 189),
            (120, 189),
            (80, 141),
            (120, 141),
        ),
    ),
    room_item_id=0x19,
    max_frames=9000,
    level=LEVEL_1,
)

# Survival overlay only. Clean M5 keeps ROOM_45_SPEC (x=160 east-column hunt).
# Off-wall fight avoids the grab-to-entrance. Continuous combat ends in the
# y=149–157 band; south of that at x=80/120/160 is solid, so collect first
# walks the free east column at x=208 (same column the entry route uses).
ROOM_45_SURVIVAL_SPEC = replace(
    ROOM_45_SPEC,
    spec_id="level1_room45_survival",
    combat=replace(
        ROOM_45_SPEC.combat,
        engage_distance=56,
        contact_backstep=16,
        avoid_walls=True,
        inland_dash=48,
    ),
    reward=replace(
        ROOM_45_SPEC.reward,
        waypoints=(
            (208, 157),
            (208, 189),
            (152, 189),
            (208, 141),
            (160, 141),
            (32, 157),
            (32, 189),
        ),
    ),
)


class Room44SurvivalController(GenericDungeonRoomController):
    """East-first south aisle to x=192; occupancy only from the east column.

    West mouth cannot stand at x=32. Tunnel x<24 is RIGHT (v9 DOWN sat
    in the door). y=141 RIGHT hits the east statues (v11). Occupancy
    from x=80 boxed at (80,157) (v13). Forced y=165 to x=192, then UP
    if the remaining Goriya is north. Not Clean.
    """

    _WEST_MOUTH_X = 48
    _INLAND_X = 80
    _SOUTH_LANE_Y = 165
    _DOOR_LANE_Y = 141
    _LANE_TOL = 6
    _DOOR_ROW = (133, 149)
    _TUNNEL_X = 24
    _EAST_COL_X = 192
    _NORTH_LANE_Y = 117
    _NORTH_GORIYA_Y = 125

    def _west_inland_step(self, snap: ZeldaSnapshot):
        """Peel east on y=141/165. Do not occupancy-chase from x=32."""
        x, y = int(snap.link_x), int(snap.link_y)
        if x >= self._INLAND_X:
            return None
        self.combat_frames += 1
        tuning = self.spec.combat
        if x < self._TUNNEL_X:
            # v9 leftover (16,141): west-door tunnel. DOWN is solid; only RIGHT.
            direction = "RIGHT"
        elif x <= self._WEST_MOUTH_X:
            # West-mouth statue misses must not poison the inland chase.
            self.walker = self._make_walker()
            door_lo, door_hi = self._DOOR_ROW
            if door_lo <= y <= door_hi:
                direction = "DOWN"
            elif y < self._SOUTH_LANE_Y - self._LANE_TOL:
                direction = "DOWN"
            elif y > self._SOUTH_LANE_Y + self._LANE_TOL:
                direction = "UP"
            else:
                direction = "RIGHT"
        else:
            # v12 leftover (56,109): DOWN off the north band is solid.
            # Left to the west aisle, then the mouth policy walks south.
            if y <= self._NORTH_LANE_Y:
                self.walker = self._make_walker()
                direction = "LEFT"
            else:
                lane = min(
                    (self._DOOR_LANE_Y, self._SOUTH_LANE_Y),
                    key=lambda ly: abs(y - ly),
                )
                if abs(y - lane) > self._LANE_TOL:
                    direction = "DOWN" if y < lane else "UP"
                else:
                    direction = "RIGHT"
        return self._swing(
            direction,
            "west_inland",
            period=tuning.engage_attack_period,
            hold=tuning.engage_attack_hold,
        )

    def _east_column_step(self, snap: ZeldaSnapshot, nearest: ZeldaObject):
        """South aisle y=165 then x=192. Occupancy from the west boxes statues.

        v11 y=141 RIGHT hit the east statues. v13 leftover (80,157) occupancy
        chased gy>=125 and boxed 14 cells. Walk the south aisle to the east
        column for every remaining Goriya; occupancy only from x=192.
        """
        x, y = int(snap.link_x), int(snap.link_y)
        at_col = abs(x - self._EAST_COL_X) <= self._LANE_TOL
        gy = int(nearest.y)
        if at_col:
            if (
                gy < self._NORTH_GORIYA_Y
                and y > self._NORTH_LANE_Y + self._LANE_TOL
            ):
                self.combat_frames += 1
                self.walker = self._make_walker()
                return self._swing(
                    "UP",
                    "east_column",
                    period=self.spec.combat.engage_attack_period,
                    hold=self.spec.combat.engage_attack_hold,
                )
            return None
        self.combat_frames += 1
        self.walker = self._make_walker()
        if y <= self._NORTH_LANE_Y:
            direction = "LEFT"
        elif abs(y - self._SOUTH_LANE_Y) > self._LANE_TOL:
            direction = "DOWN" if y < self._SOUTH_LANE_Y else "UP"
        else:
            direction = "RIGHT" if x < self._EAST_COL_X else "LEFT"
        return self._swing(
            direction,
            "east_column",
            period=self.spec.combat.engage_attack_period,
            hold=self.spec.combat.engage_attack_hold,
        )

    def _combat(self, snap: ZeldaSnapshot, live: tuple[ZeldaObject, ...]):
        bounds = self.spec.combat.occupancy_bounds
        if not live or bounds is None:
            return super()._combat(snap, live)
        xmin, xmax, ymin, ymax = bounds
        inland = tuple(
            obj
            for obj in live
            if xmin <= int(obj.x) <= xmax and ymin <= int(obj.y) <= ymax
        )
        if inland:
            peel = self._west_inland_step(snap)
            if peel is not None:
                return peel
            nearest = min(
                inland,
                key=lambda obj: abs(int(obj.x) - snap.link_x)
                + abs(int(obj.y) - snap.link_y),
            )
            around = self._east_column_step(snap, nearest)
            if around is not None:
                return around
            return super()._combat(snap, inland)
        # v6/v7: occupancy-chasing the y=93 Goriya boxed at (66,109).
        self.combat_frames += 1
        nearest = min(
            live,
            key=lambda obj: abs(int(obj.x) - snap.link_x)
            + abs(int(obj.y) - snap.link_y),
        )
        return self._engage(snap, nearest, direction=None)

    def _engage(
        self,
        snap: ZeldaSnapshot,
        target: ZeldaObject,
        direction: str | None = None,
    ):
        bounds = self.spec.combat.occupancy_bounds
        if bounds is None:
            return super()._engage(snap, target, direction)
        xmin, xmax, ymin, ymax = bounds
        x, y = int(snap.link_x), int(snap.link_y)
        if not (xmin <= x <= xmax and ymin <= y <= ymax):
            if y < ymin:
                direction = "DOWN"
            elif y > ymax:
                direction = "UP"
            elif x < xmin:
                direction = "RIGHT"
            else:
                direction = "LEFT"
            return super()._engage(snap, target, direction)
        if direction is None:
            # No occupancy path. v6 greedy-walked LEFT into a statue at
            # (66,109) for 62 misses. Slash if the blade reaches; else patrol.
            dx = int(target.x) - x
            dy = int(target.y) - y
            if abs(dx) > 10:
                face = "RIGHT" if dx > 0 else "LEFT"
            elif abs(dy) > 10:
                face = "DOWN" if dy > 0 else "UP"
            elif abs(dx) >= abs(dy):
                face = "RIGHT" if dx >= 0 else "LEFT"
            else:
                face = "DOWN" if dy >= 0 else "UP"
            if should_swing_at(x, y, face, (target,)):
                tuning = self.spec.combat
                return self._swing(
                    face,
                    "combat_engage",
                    period=tuning.engage_attack_period,
                    hold=tuning.engage_attack_hold,
                )
            return self._patrol(snap)
        nx, ny = self._wall_step(x, y, direction)
        if not (xmin <= nx <= xmax and ymin <= ny <= ymax):
            tuning = self.spec.combat
            return self._swing(
                direction,
                "combat_engage",
                period=tuning.engage_attack_period,
                hold=tuning.engage_attack_hold,
            )
        return super()._engage(snap, target, direction)


for _spec in (ROOM_44_SPEC, ROOM_45_SPEC):
    register_room_spec(_spec)

__all__ = [
    "ROOM_44_SPEC",
    "ROOM_44_SURVIVAL_SPEC",
    "ROOM_45_SPEC",
    "ROOM_45_SURVIVAL_SPEC",
    "Room44SurvivalController",
]
