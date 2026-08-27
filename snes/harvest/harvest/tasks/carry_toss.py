"""Carry farm debris to a named water stand and toss it.

This module deliberately knows nothing about fence selection or corridor
opening. Its contract starts after a lift has succeeded and ends after carry
RAM clears at a real pond stand and the player leaves a boxed south lip.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Optional, Tuple

from retro_harness import ActionResult, TaskResult, TaskStatus, WorldState

from harvest.core.animal_status import read_held_item
from harvest.core.tile_catalog import ADDR_INPUT_LOCK
from harvest.maps.farm_pond import (
    COW_BARN_EAST_FACE_TILES,
    EAST_SPUR_FA_A8_BANK,
    EAST_SPUR_FA_APPROACH,
    EAST_SPUR_FA_EAST_LANE_X,
    EAST_SPUR_FA_FACE,
    EAST_SPUR_FA_SOUTH_OPEN_X,
    EAST_SPUR_FA_STAND,
    EAST_SPUR_FA_WATER,
    EAST_SPUR_FA_WEST_DROP_X,
    EAST_SPUR_FA_WEST_LIP,
    HORSE_BARN_WALL_TILES,
)
from harvest.tasks.farm_toss import in_place_toss_actions
from harvest.tasks.nav import Navigator, Pathfinder, VIEWPORT_HOP_TILES, make_action
from harvest.tasks.pond_policy import (
    ALT_SOUTH_LIP_STAND,
    PRIMARY_POND_FACE,
    PRIMARY_POND_STAND,
)


PondStand = Tuple[Tuple[int, int], str]

SOUTH_LIP_STANDS = frozenset({PRIMARY_POND_STAND, ALT_SOUTH_LIP_STAND})
POND_WEST_EGRESS_STAND = (29, 35)

# North-of-barn carry. y=14 is the 0x02 highway; y=26 is south of the barns.
# horse_barn_edges: dump NE stones at EAST_SPUR_FA_STAND, not the F0 south lip.
# x=31 y=18–21 is the cow-barn east wall, not dirt. F0 vias stay for south field.
_BARN_EAST_HIGHWAY_Y = 14
_BARN_EAST_HIGHWAY_X = 31
_BARN_SOUTH_JOIN_Y = 26
_POND_EAST_BYPASS_X = 35
_FA_APPROACH_X = EAST_SPUR_FA_APPROACH[0]
_CORRIDOR_BFS_STEPS = 48
_COW_BARN_WEST_FACE = frozenset(
    (x, y) for x in range(29, 31) for y in range(18, 22)
)
_BARN_PUSH_FACES = (
    HORSE_BARN_WALL_TILES | COW_BARN_EAST_FACE_TILES | _COW_BARN_WEST_FACE
)


def _needs_barn_east_corridor(tile: Tuple[int, int]) -> bool:
    """True until south of the barn, including a stand under the horse barn."""
    return tile[1] < _BARN_SOUTH_JOIN_Y


def _adjacent_run(
    start: Tuple[int, int], goal: Tuple[int, int]
) -> list[Tuple[int, int]]:
    """Goal-inclusive, start-exclusive 4-adjacent tiles. X then Y."""
    x, y = start
    gx, gy = goal
    tiles: list[Tuple[int, int]] = []
    while x != gx:
        x += 1 if gx > x else -1
        tiles.append((x, y))
    while y != gy:
        y += 1 if gy > y else -1
        tiles.append((x, y))
    return tiles


def _barn_east_pond_vias(player: Tuple[int, int]) -> list[Tuple[int, int]]:
    vias: list[Tuple[int, int]] = []
    if player[0] != _BARN_EAST_HIGHWAY_X:
        vias.append((player[0], _BARN_EAST_HIGHWAY_Y))
        vias.append((_BARN_EAST_HIGHWAY_X, _BARN_EAST_HIGHWAY_Y))
    vias.extend(
        (
            (_BARN_EAST_HIGHWAY_X, _BARN_SOUTH_JOIN_Y),
            (_POND_EAST_BYPASS_X, _BARN_SOUTH_JOIN_Y),
            (_POND_EAST_BYPASS_X, PRIMARY_POND_STAND[1]),
            PRIMARY_POND_STAND,
        )
    )
    return vias


def _east_spur_fa_vias(player: Tuple[int, int]) -> list[Tuple[int, int]]:
    """y=14 highway to the west dirt of the 0xFA toss stand.

    Last hop is (45,16) east onto (46,16). Do not route onto water
    (46,14)/(46,15) — that hugs (45,14) facing the spur. Horse-barn
    takeoff leaves south, then west around y=23. West onto (16,20) is a
    pocket: 0xD8 / sprite walls, no north exit.
    """
    vias: list[Tuple[int, int]] = []
    cur = player
    under_horse = player == (17, 20) or (
        16 <= player[0] <= 18 and 19 <= player[1] <= 21
    )
    if under_horse:
        vias.extend(((17, 23), (13, 23), (13, _BARN_EAST_HIGHWAY_Y)))
        cur = vias[-1]
    # East of the spur. y<=13 cannot south at x=46-50 (live leftover lip).
    # Cross at x=51, then y=16 west onto the stand. A8 is no-go from the
    # north; repair may skirt y=17. Do not walk (46,14)/(46,15) water.
    if cur[0] >= EAST_SPUR_FA_STAND[0]:
        if cur[1] <= 13:
            open_x = EAST_SPUR_FA_SOUTH_OPEN_X
            if cur[0] != open_x:
                vias.append((open_x, cur[1]))
                cur = vias[-1]
            if cur[1] != EAST_SPUR_FA_STAND[1]:
                vias.append((open_x, EAST_SPUR_FA_STAND[1]))
            vias.append(EAST_SPUR_FA_STAND)
            return vias
        lane_x = EAST_SPUR_FA_EAST_LANE_X
        if cur[0] != lane_x:
            vias.append((lane_x, cur[1]))
            cur = vias[-1]
        if cur[1] != EAST_SPUR_FA_APPROACH[1]:
            vias.append((lane_x, EAST_SPUR_FA_APPROACH[1]))
        vias.append(EAST_SPUR_FA_STAND)
        return vias
    drop_x = EAST_SPUR_FA_WEST_DROP_X
    if cur[1] != _BARN_EAST_HIGHWAY_Y and cur[0] <= drop_x:
        vias.append((cur[0], _BARN_EAST_HIGHWAY_Y))
        cur = vias[-1]
    if cur[0] != drop_x and cur[1] <= _BARN_EAST_HIGHWAY_Y:
        vias.append((drop_x, cur[1]))
        cur = vias[-1]
    if cur[1] != EAST_SPUR_FA_STAND[1]:
        vias.append((cur[0], EAST_SPUR_FA_STAND[1]))
        cur = vias[-1]
    if cur != EAST_SPUR_FA_APPROACH:
        vias.append(EAST_SPUR_FA_APPROACH)
    vias.append(EAST_SPUR_FA_STAND)
    return vias


def _corridor_vias(
    player: Tuple[int, int], dest: Tuple[int, int]
) -> list[Tuple[int, int]]:
    if dest == EAST_SPUR_FA_STAND:
        return _east_spur_fa_vias(player)
    return _barn_east_pond_vias(player)


def _geometric_barn_east_path(
    player: Tuple[int, int], dest: Tuple[int, int] = PRIMARY_POND_STAND
) -> list[Tuple[int, int]]:
    """Adjacent vias to dest. FA stays on y=14; F0 still south-joins at x=31."""
    path: list[Tuple[int, int]] = []
    cur = player
    for via in _corridor_vias(player, dest):
        path.extend(_adjacent_run(cur, via))
        cur = via
    if dest == EAST_SPUR_FA_STAND:
        path = [
            tile
            for tile in path
            if tile not in EAST_SPUR_FA_WATER and tile not in EAST_SPUR_FA_WEST_LIP
        ]
    return path


def _repair_walkable_path(
    pathfinder: Pathfinder,
    ram,
    start: Tuple[int, int],
    geometric: list[Tuple[int, int]],
    dest: Tuple[int, int],
) -> list[Tuple[int, int]]:
    """Keep the highway order, but BFS around live solids (2x2 / 0xA6 / barn)."""
    if not geometric:
        return geometric
    if all(
        pathfinder.is_walkable(ram, x, y, current_pos=start) for x, y in geometric
    ):
        return geometric

    saved = set(pathfinder.temp_blocked)
    pathfinder.temp_blocked.update(_BARN_PUSH_FACES)
    pathfinder.temp_blocked.update(EAST_SPUR_FA_WATER)
    pathfinder.temp_blocked.update(EAST_SPUR_FA_A8_BANK)
    pathfinder.temp_blocked.update(EAST_SPUR_FA_WEST_LIP)
    try:
        path: list[Tuple[int, int]] = []
        cur = start
        i = 0
        while i < len(geometric):
            nxt = geometric[i]
            if nxt == cur:
                i += 1
                continue
            if (
                abs(nxt[0] - cur[0]) + abs(nxt[1] - cur[1]) == 1
                and pathfinder.is_walkable(ram, nxt[0], nxt[1], current_pos=cur)
            ):
                path.append(nxt)
                cur = nxt
                i += 1
                continue
            goal = None
            goal_i = i
            for j in range(i, len(geometric)):
                cand = geometric[j]
                if cand != cur and pathfinder.is_walkable(
                    ram, cand[0], cand[1], current_pos=cur
                ):
                    goal = cand
                    goal_i = j
                    break
            if goal is None:
                goal = dest
                goal_i = len(geometric) - 1
            hop = pathfinder.find_path(
                ram, cur, goal, max_steps=_CORRIDOR_BFS_STEPS
            )
            if not hop:
                i = goal_i + 1
                continue
            path.extend(hop)
            cur = hop[-1]
            if cur == goal:
                i = goal_i + 1
            else:
                break
        return path
    finally:
        pathfinder.temp_blocked.clear()
        pathfinder.temp_blocked.update(saved)


def _barn_east_pond_path(
    player: Tuple[int, int],
    ram=None,
    pathfinder: Optional[Pathfinder] = None,
    dest: Tuple[int, int] = EAST_SPUR_FA_STAND,
) -> list[Tuple[int, int]]:
    geometric = _geometric_barn_east_path(player, dest)
    if ram is None or pathfinder is None:
        return geometric
    return _repair_walkable_path(pathfinder, ram, player, geometric, dest)


def farm_toss_stands(player: Tuple[int, int]) -> Tuple[PondStand, ...]:
    """Verified dump stand. North of the barns uses tape 0xFA; south uses F0.

    North F9 refills a can but its A1 bank push-blocks held debris. The
    horse_barn_edges slice tossed leftover stones from (46,16) face-up into
    east_spur_fa (46,15). Do not haul that cluster around the cow barn to F0.
    """
    if player[1] < _BARN_SOUTH_JOIN_Y:
        return ((EAST_SPUR_FA_STAND, EAST_SPUR_FA_FACE),)
    return ((PRIMARY_POND_STAND, PRIMARY_POND_FACE),)


@dataclass
class CarryToPondStand:
    """Navigate while holding debris, then toss from a verified dirt stand."""

    stasis_repath: int = 180
    debug: bool = False
    _pathfinder: Pathfinder = field(default_factory=Pathfinder, init=False)
    _navigator: Navigator = field(init=False)
    _stand: Optional[Tuple[int, int]] = field(default=None, init=False)
    _face: str = field(default="up", init=False)
    _actions: deque = field(default_factory=deque, init=False)
    _toss_started: bool = field(default=False, init=False)
    _egress_goal: Optional[Tuple[int, int]] = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._navigator = Navigator(self._pathfinder)
        self._pathfinder.no_go_tiles.update(EAST_SPUR_FA_WATER)
        self._pathfinder.no_go_tiles.update(EAST_SPUR_FA_A8_BANK)
        self._pathfinder.no_go_tiles.update(EAST_SPUR_FA_WEST_LIP)

    def reset(self, world: WorldState) -> None:
        self._stand = None
        self._face = "up"
        self._actions.clear()
        self._toss_started = False
        self._egress_goal = None
        self._pathfinder.temp_blocked.clear()
        self._navigator.path = []
        self._navigator.update(world.ram)

    @staticmethod
    def _carrying(world: WorldState) -> bool:
        return bool(int(world.ram[0xD2]) & 0x02) or read_held_item(world.ram) != 0

    def _choose_stand(self, world: WorldState, stands: Iterable[PondStand]) -> bool:
        player = self._navigator.current_tile
        ordered = tuple(stands)
        if _needs_barn_east_corridor(player) and ordered:
            path = _barn_east_pond_path(
                player,
                ram=world.ram,
                pathfinder=self._pathfinder,
                dest=ordered[0][0],
            )
            if path:
                self._stand, self._face = ordered[0]
                self._navigator.path = path
                return True
        for stand, face in ordered:
            path = self._pathfinder.find_path(
                world.ram, player, stand, max_steps=VIEWPORT_HOP_TILES
            )
            if path is not None:
                self._stand, self._face = stand, face
                self._navigator.path = path
                return True
        return False

    def step(self, world: WorldState) -> TaskResult:
        self._navigator.update(world.ram)
        if not self._carrying(world):
            if self._toss_started:
                input_lock = int(world.ram[ADDR_INPUT_LOCK])
                if input_lock != 1:
                    return TaskResult(
                        status=TaskStatus.RUNNING,
                        reason="wait for pond toss animation",
                    )
                player = self._navigator.current_tile
                if self._egress_goal is not None and player != self._egress_goal:
                    if (
                        not self._navigator.path
                        or self._navigator.path[-1] != self._egress_goal
                    ):
                        self._navigator.path = self._pathfinder.find_path(
                            world.ram, player, self._egress_goal
                        ) or []
                    action = self._navigator.follow_path(world.ram)
                    return TaskResult(
                        status=TaskStatus.RUNNING,
                        action=ActionResult(
                            action if action is not None else make_action()
                        ),
                        reason=(
                            "egress boxed pond stand toward "
                            f"{self._egress_goal}"
                        ),
                    )
                self._egress_goal = None
            return TaskResult(
                status=TaskStatus.SUCCESS,
                reason="pond toss complete" if self._toss_started else "hands empty",
            )

        if self._actions:
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(self._actions.popleft()),
                reason="toss at pond stand",
            )

        player = self._navigator.current_tile
        if self._stand is None and not self._choose_stand(world, farm_toss_stands(player)):
            return TaskResult(status=TaskStatus.RUNNING, reason="pond stand outside viewport")

        assert self._stand is not None
        if player == self._stand:
            # The south lip blocks pixel-centering toward its water edge (live
            # landing is commonly x=518,y=558).  Tile occupancy is the verified
            # stand contract; face and pulse from there.
            self._toss_started = True
            self._egress_goal = (
                POND_WEST_EGRESS_STAND
                if self._stand in SOUTH_LIP_STANDS
                else None
            )
            self._navigator.path = []
            self._actions.extend(in_place_toss_actions(face=self._face))
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(self._actions.popleft()),
                reason=f"toss from pond stand {self._stand}",
            )

        if self._navigator.stasis > self.stasis_repath or not self._navigator.path:
            if self._navigator.path:
                nxt = self._navigator.path[0]
                fa_east_lip = (
                    self._stand == EAST_SPUR_FA_STAND
                    and player[1] <= 13
                    and player[0] >= EAST_SPUR_FA_STAND[0]
                )
                # Do not temp-block the x=51 south-cross (or the east hops
                # onto it). Blocking (47,13) is how the old lip sat 400k.
                if not fa_east_lip:
                    self._pathfinder.block_push_facing(
                        world.ram, nxt, pixel_moved=False
                    )
            self._stand = None
            self._navigator.path = []
            if not self._choose_stand(world, farm_toss_stands(player)):
                return TaskResult(status=TaskStatus.RUNNING, reason="repath around farm solid")

        action = self._navigator.follow_path(world.ram)
        return TaskResult(
            status=TaskStatus.RUNNING,
            action=ActionResult(action if action is not None else make_action()),
            reason=f"carry to pond stand {self._stand}",
        )


__all__ = [
    "CarryToPondStand",
    "POND_WEST_EGRESS_STAND",
    "SOUTH_LIP_STANDS",
    "farm_toss_stands",
]
