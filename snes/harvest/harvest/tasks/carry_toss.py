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


def farm_toss_stands(player: Tuple[int, int]) -> Tuple[PondStand, ...]:
    """Return the verified debris dump stand.

    North F9 refills a watering can, but its A1 bank push-blocks held debris;
    a throw from farther west merely re-drops the object on dry ground.  F0 is
    reachable from the north paddock around the live 2x2 boulder, so keep one
    honest destination instead of cycling through false local minima.
    """
    del player
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
        for stand, face in stands:
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
                self._pathfinder.block_push_facing(
                    world.ram, self._navigator.path[0], pixel_moved=False
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
