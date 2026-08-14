"""Reactive hoe / plant / water / carry-select skills. No tape replay.

Split from the ``d2_farm_plant`` human path: stand on the cell, Y until the
tile ID changes, X until the wanted tool is selected. One planted+watered
cell is enough for CROP_ESTABLISH.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import FrozenSet, Optional, Tuple

from retro_harness import ActionResult, Task, TaskResult, TaskStatus, WorldState

from harvest.core.carry import backpack_tool, seed_item_id, selected_tool
from harvest.core.task_progress import ProgressSnapshot
from harvest.core.tile_catalog import (
    LARGE_ROCK_TILES,
    STONE,
    STUMP_TILES,
    Tool,
    WEED,
)
from harvest.tasks.crop_geometry import FRESH_TILLED
from harvest.tasks.nav import TILE_SIZE, get_pos_from_ram, get_tile_at, make_action

# Hoe Y on a bush/stone is a no-op (live D2: timeout tid=0x01 beside 0x03).
HOE_BLOCKED_TILES: FrozenSet[int] = (
    frozenset({WEED, STONE}) | STUMP_TILES | LARGE_ROCK_TILES
)

# Potato dry/wet pair from the tape (0x54 → 0x55).
PLANTED_DRY = 0x54
PLANTED_WET = 0x55
HOED_OR_PLANTED: FrozenSet[int] = frozenset({FRESH_TILLED, PLANTED_DRY, PLANTED_WET})
PLANTED_OR_WET: FrozenSet[int] = frozenset({PLANTED_DRY, PLANTED_WET}) | frozenset(
    range(0x1E, 0x70)
)
WET_CROP: FrozenSet[int] = frozenset({PLANTED_WET}) | frozenset(
    tid for tid in range(0x1E, 0x70) if tid % 2 == 1
)


@dataclass
class SelectCarrySkill(Task):
    """X-swap until ``wanted`` is selected. Instant success if already selected."""

    name: str = "select_carry"
    wanted: int = 0
    timeout: int = 90

    _steps: int = field(default=0, init=False)

    def reset(self, world: WorldState) -> None:
        self._steps = 0

    def can_start(self, world: WorldState) -> bool:
        return True

    def progress_snapshot(self) -> ProgressSnapshot:
        return ProgressSnapshot(
            task_name=self.name,
            phase_text="swap",
            step_count=self._steps,
            details=(("wanted", hex(self.wanted)),),
        )

    def step(self, world: WorldState) -> TaskResult:
        self._steps += 1
        sel = int(selected_tool(world.ram))
        if sel == int(self.wanted):
            return TaskResult(status=TaskStatus.SUCCESS, reason=f"selected 0x{self.wanted:02X}")
        back = int(backpack_tool(world.ram))
        if back != int(self.wanted):
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=f"0x{self.wanted:02X} not in carry pair",
            )
        if self._steps > self.timeout:
            return TaskResult(status=TaskStatus.FAILURE, reason="select carry timeout")
        tap = self._steps % 6 == 1
        return TaskResult(
            status=TaskStatus.RUNNING,
            action=ActionResult(make_action(x=True) if tap else make_action()),
            reason="x-swap carry",
        )


@dataclass
class UseToolUntilTileSkill(Task):
    """Use a tool until the watched metatile is in ``done_ids``.

    ``target_tile``/``face`` support tools such as the hoe, which act on the
    faced adjacent tile rather than the player's stand tile.  With no target,
    the skill retains the current-cell behavior used by seed bags and cans.
    """

    name: str = "use_tool_until_tile"
    tool_id: int = 0
    done_ids: FrozenSet[int] = field(default_factory=frozenset)
    blocked_ids: FrozenSet[int] = field(default_factory=frozenset)
    target_tile: Optional[Tuple[int, int]] = None
    face: Optional[str] = None
    timeout: int = 240

    _steps: int = field(default=0, init=False)

    def reset(self, world: WorldState) -> None:
        self._steps = 0

    def can_start(self, world: WorldState) -> bool:
        return True

    def progress_snapshot(self) -> ProgressSnapshot:
        return ProgressSnapshot(
            task_name=self.name,
            phase_text="use_tool",
            step_count=self._steps,
            details=(("tool", hex(self.tool_id)),),
        )

    def step(self, world: WorldState) -> TaskResult:
        self._steps += 1
        pos = get_pos_from_ram(world.ram)
        player_tile = (pos.x // TILE_SIZE, pos.y // TILE_SIZE)
        tile = self.target_tile or player_tile
        tid = int(get_tile_at(world.ram, tile[0], tile[1]))
        if tid in self.done_ids:
            return TaskResult(
                status=TaskStatus.SUCCESS,
                reason=f"{self.name} tile=0x{tid:02X} at {tile}",
            )
        if tid in self.blocked_ids:
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=f"{self.name} blocked tid=0x{tid:02X} at {tile}",
            )
        if self._steps > self.timeout:
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=f"{self.name} timeout tid=0x{tid:02X} at {tile}",
            )
        sel = int(selected_tool(world.ram))
        if sel != int(self.tool_id):
            if int(backpack_tool(world.ram)) == int(self.tool_id):
                return TaskResult(
                    status=TaskStatus.RUNNING,
                    action=ActionResult(make_action(x=True)),
                    reason="x-swap before tool use",
                )
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=f"tool 0x{self.tool_id:02X} not selected",
            )
        phase = self._steps % 48
        if self.face is not None and phase < 4:
            action = make_action(**{self.face: True})
        elif 8 <= phase < 28:
            action = make_action(y=True)
        else:
            action = make_action()
        return TaskResult(
            status=TaskStatus.RUNNING,
            action=ActionResult(action),
            reason=f"Y until done tid=0x{tid:02X}",
        )


def hoe_until_tilled_skill(
    *,
    target_tile: Optional[Tuple[int, int]] = None,
    face: Optional[str] = None,
    timeout: int = 240,
) -> UseToolUntilTileSkill:
    return UseToolUntilTileSkill(
        name="hoe_until_tilled",
        tool_id=int(Tool.HOE),
        done_ids=HOED_OR_PLANTED,
        blocked_ids=HOE_BLOCKED_TILES,
        target_tile=target_tile,
        face=face,
        timeout=timeout,
    )


def plant_until_crop_skill(
    *,
    seed_type: str = "potato",
    target_tile: Optional[Tuple[int, int]] = None,
    timeout: int = 240,
) -> UseToolUntilTileSkill:
    return UseToolUntilTileSkill(
        name="plant_until_crop",
        tool_id=seed_item_id(seed_type),
        done_ids=PLANTED_OR_WET,
        target_tile=target_tile,
        timeout=timeout,
    )


def water_until_wet_skill(*, timeout: int = 240) -> UseToolUntilTileSkill:
    return UseToolUntilTileSkill(
        name="water_until_wet",
        tool_id=int(Tool.WATERING_CAN),
        done_ids=WET_CROP,
        timeout=timeout,
    )
