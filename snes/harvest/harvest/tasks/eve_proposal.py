"""Propose to Eve with the blue feather. Married when this succeeds.

On a sunny Saturday before 17:00 Eve stands on the mountain (map ``0x10``) at
(504,216), the script at ``$B3B7DE``. Using the feather with her hearts at 250
or more answers ``0x017B``, sets wife bit 4 (``$7F1F66`` ``0x10``) and starts
the wedding; the day then rolls to 06:00 in the house. The check only passes
when the point in front of the farmer lies in the 25 pixel box right of and
below her anchor (``$83AE26``): stand at (503,229) facing up, not 2 pixels
lower. See docs/EVE_BAR.md.

The mountain is a long switchback; ``MOUNTAIN_TO_EVE`` is the route from the
entry at (328,728), found by probing hops with emulator save/restore.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from harvest.core.carry import ADDR_TOOL_BACKPACK, ADDR_TOOL_SELECTED
from harvest.core.npc_catalog import current_dialogue_registers, game_objects
from harvest.core.ram_catalog import read_ram_u16, read_ram_u8, read_ram_value
from harvest.core.tile_catalog import ADDR_TILEMAP, is_path_tilemap
from harvest.maps.map_routes import segment_waypoints
from harvest.maps.map_types import Waypoint
from harvest.planner.tasks.multi_nav import MultiMapNavTask
from harvest.tasks.nav import get_pos_from_ram, make_action
from retro_harness import ActionResult, Task, TaskResult, TaskStatus, WorldState

MOUNTAIN_TILEMAP = 0x10
PATH_TILEMAP = 0x0C
FARM_TILEMAPS = (0x00, 0x01, 0x02, 0x03)
FEATHER = 0x18
EVE_SPRITES = range(0x0244, 0x024A)
STAND_OFFSET = (-1, 13)  # from Eve's pixel; the interaction box is exact to a pixel
PROPOSAL_FLOOR = 250
WIFE_BIT = 0x0010  # $7F1F66
SATURDAY = 6
CLOSING_HOUR = 17  # Eve's scripts stop at 17:00
RAIN, SNOW, STORM = 0x0002, 0x0008, 0x0010
ADDR_WEATHER_FLAGS = 0x0196
ADDR_WIFE_FLAGS = 0x11F66

MOUNTAIN_TO_EVE: Tuple[Tuple[int, int], ...] = (
    (328, 696), (512, 696), (512, 680), (520, 680), (520, 568), (488, 568),
    (488, 472), (248, 472), (248, 424), (80, 424), (80, 392), (72, 392),
    (72, 360), (120, 360), (120, 328), (400, 328), (400, 312), (408, 312),
    (408, 248), (472, 248), (472, 240), (496, 240),
)

ALIGN_TIMEOUT = 600
USE_TIMEOUT = 400
WEDDING_TIMEOUT = 6000
NAV_TIMEOUT = 30000


def proposal_blockers(ram: np.ndarray) -> Optional[str]:
    """Why the proposal cannot work right now, or None."""
    if read_ram_u16(ram, ADDR_WIFE_FLAGS) & 0x1F:
        return "already married"
    hearts = read_ram_value(ram, "eve_hearts")
    if hearts < PROPOSAL_FLOOR:
        return f"eve hearts {hearts} < {PROPOSAL_FLOOR}"
    if read_ram_value(ram, "weekday") != SATURDAY:
        return "Eve is on the mountain on Saturdays only"
    if read_ram_value(ram, "hour") >= CLOSING_HOUR:
        return "Eve leaves the mountain at 17:00"
    if read_ram_u16(ram, ADDR_WEATHER_FLAGS, live_offset=False) & (RAIN | SNOW | STORM):
        return "no sunny Saturday"
    return None


@dataclass
class EveProposalTask(Task):
    """Walk to Eve on the mountain, use the feather, ride the wedding to the morning."""

    name: str = "propose_eve"
    timeout: int = 60000
    check_blockers: bool = True

    _phase: str = field(default="to_mountain", init=False)
    _frames: int = field(default=0, init=False)
    _steps: int = field(default=0, init=False)
    _nav: Optional[Task] = field(default=None, init=False)
    _start_day: int = field(default=0, init=False)
    _eve: Optional[Tuple[int, int]] = field(default=None, init=False)
    _tapped: int = field(default=0, init=False)

    @staticmethod
    def _tilemap(ram: np.ndarray) -> int:
        return int(ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(ram) else 0

    @staticmethod
    def _find_eve(ram: np.ndarray) -> Optional[Tuple[int, int]]:
        for obj in game_objects(ram):
            if not obj.is_player and obj.sprite_table_idx in EVE_SPRITES:
                return obj.pixel
        return None

    def can_start(self, world: WorldState) -> bool:
        return True

    @property
    def progress_text(self) -> str:
        return f"{self._phase} frames={self._frames}"

    def reset(self, world: WorldState) -> None:
        self._start_day = read_ram_value(world.ram, "day")
        self._steps = self._tapped = 0
        self._nav, self._eve = None, None
        self._begin("to_mountain")

    def _begin(self, phase: str) -> None:
        self._phase, self._frames = phase, 0

    @staticmethod
    def _act(reason: str, **buttons: bool) -> TaskResult:
        return TaskResult(
            status=TaskStatus.RUNNING, action=ActionResult(make_action(**buttons)), reason=reason
        )

    @staticmethod
    def _fail(reason: str) -> TaskResult:
        return TaskResult(status=TaskStatus.FAILURE, reason=reason)

    # ── step ─────────────────────────────────────────────────────────
    def step(self, world: WorldState) -> TaskResult:
        self._steps += 1
        self._frames += 1
        ram = world.ram
        if self._steps > self.timeout:
            return self._fail(f"eve proposal timeout in {self._phase}")
        if read_ram_u16(ram, ADDR_WIFE_FLAGS) & WIFE_BIT and self._phase != "wedding":
            self._begin("wedding")
        if self._phase == "to_mountain" and self._steps == 1 and self.check_blockers:
            blocked = proposal_blockers(ram)
            if blocked:
                return self._fail(f"eve proposal blocked: {blocked}")
        return getattr(self, f"_phase_{self._phase}")(world)

    @staticmethod
    def hops_to_eve(tilemap: int) -> Optional[List[Waypoint]]:
        route = [Waypoint(MOUNTAIN_TILEMAP, px, radius=6) for px in MOUNTAIN_TO_EVE]
        if tilemap == MOUNTAIN_TILEMAP:
            return route
        if is_path_tilemap(tilemap):
            return segment_waypoints("path_to_mountain") + route
        if tilemap in FARM_TILEMAPS:
            return segment_waypoints("farm_to_path", "path_to_mountain") + route
        return None

    def _phase_to_mountain(self, world: WorldState) -> TaskResult:
        ram = world.ram
        if self._nav is None:
            hops = self.hops_to_eve(self._tilemap(ram))
            if hops is None:
                return self._fail(f"eve proposal cannot start from tilemap=0x{self._tilemap(ram):02X}")
            self._nav = MultiMapNavTask(waypoints=hops, timeout=NAV_TIMEOUT, initial_settle_frames=30)
            self._nav.reset(world)
        result = self._nav.step(world)
        if result.status == TaskStatus.FAILURE:
            return self._fail(f"could not reach Eve on the mountain: {result.reason}")
        if result.status == TaskStatus.SUCCESS:
            self._nav = None
            self._begin("settle")
            return self._act("on_mountain")
        return TaskResult(status=TaskStatus.RUNNING, action=result.action, reason="walk_to_eve")

    def _phase_settle(self, world: WorldState) -> TaskResult:
        if self._frames < 30:
            return self._act("mountain_settle")
        self._eve = self._find_eve(world.ram)
        if self._eve is None:
            return self._fail("Eve is not on the mountain (sunny Saturday before 17:00?)")
        self._begin("equip")
        return self._act("eve_found")

    def _phase_equip(self, world: WorldState) -> TaskResult:
        ram = world.ram
        if read_ram_u8(ram, ADDR_TOOL_SELECTED) == FEATHER:
            self._begin("align")
            return self._act("feather_ready")
        if read_ram_u8(ram, ADDR_TOOL_BACKPACK) != FEATHER:
            return self._fail("no blue feather in the carry pair")
        if self._frames > 120:
            return self._fail("could not select the blue feather")
        return self._act("swap_tools", x=self._frames % 20 < 4)

    def _phase_align(self, world: WorldState) -> TaskResult:
        ram = world.ram
        eve = self._find_eve(ram) or self._eve
        sx, sy = eve[0] + STAND_OFFSET[0], eve[1] + STAND_OFFSET[1]
        pos = get_pos_from_ram(ram)
        dx, dy = sx - pos.x, sy - pos.y
        if self._frames > ALIGN_TIMEOUT:
            return self._fail(f"could not stand below Eve: at ({pos.x},{pos.y}) wanted ({sx},{sy})")
        if abs(dx) <= 1 and abs(dy) <= 1:
            self._begin("use")
            return self._act("below_eve", up=True)
        if abs(dx) >= abs(dy) and abs(dx) > 1:
            return self._act("align", right=dx > 0, left=dx < 0)
        return self._act("align", down=dy > 0, up=dy < 0)

    def _phase_use(self, world: WorldState) -> TaskResult:
        ram = world.ram
        if self._frames > USE_TIMEOUT:
            return self._fail(f"the feather did not propose (hearts {read_ram_value(ram, 'eve_hearts')})")
        if current_dialogue_registers(ram)["input_lock"] == 2:
            return self._act("proposal_text", a=self._frames % 12 < 3)
        # Y swings the tool; hold Up so the farmer keeps facing her.
        return self._act("use_feather", y=self._frames % 90 in (4, 5, 6), up=True)

    def _phase_wedding(self, world: WorldState) -> TaskResult:
        ram = world.ram
        house = read_ram_value(ram, "upgrade_flags")
        expected_map = 0x17 if house & 0x80 else 0x16 if house & 0x40 else 0x15
        if (read_ram_value(ram, "day") != self._start_day
                and self._tilemap(ram) == expected_map
                and read_ram_value(ram, "input_lock") == 1
                and not read_ram_value(ram, "game_state") & 0x8040):
            return TaskResult(
                status=TaskStatus.SUCCESS, reason="married Eve; the wedding rolled the day"
            )
        if self._frames > WEDDING_TIMEOUT:
            return self._fail("the wedding never ended the day")
        if current_dialogue_registers(ram)["input_lock"] == 2:
            return self._act("wedding_text", a=self._frames % 12 < 3)
        return self._act("wedding_wait")


__all__ = ["EveProposalTask", "MOUNTAIN_TO_EVE", "proposal_blockers"]
