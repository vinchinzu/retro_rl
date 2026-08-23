"""Reactive hoe / plant / water / carry-select skills. No tape replay.

Split from the ``d2_farm_plant`` human path: hoe the 8-tile ring around the
untilled notch, stand on the center, Y until the bag spends. One-cell
hoe of the notch leaves 1–2 dry 0x54 and does not spend the bag.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import FrozenSet, Optional, Tuple

from retro_harness import ActionResult, Task, TaskResult, TaskStatus, WorldState

from harvest.core.animal_status import read_held_item
from harvest.core.carry import SEED_ITEM, backpack_tool, seed_item_id, selected_tool
from harvest.core.ram_catalog import read_ram_value
from harvest.core.task_progress import ProgressSnapshot
from harvest.core.tile_catalog import (
    ADDR_INPUT_LOCK,
    LARGE_ROCK_TILES,
    STONE,
    STUMP_TILES,
    Tool,
    WEED,
)
from harvest.tasks.crop_geometry import FRESH_TILLED
from harvest.tasks.nav import TILE_SIZE, get_pos_from_ram, get_tile_at, make_action

_SEED_IDS = frozenset(SEED_ITEM.values())


def _input_unlocked(ram) -> bool:
    """True when the farmer can accept X/Y. ``input_lock==1`` is free-move."""
    if ADDR_INPUT_LOCK >= len(ram):
        return True
    return int(ram[ADDR_INPUT_LOCK]) == 1

# Hoe Y on a bush/stone is a no-op (live D2: timeout tid=0x01 beside 0x03).
HOE_BLOCKED_TILES: FrozenSet[int] = (
    frozenset({WEED, STONE}) | STUMP_TILES | LARGE_ROCK_TILES
)
# RAM player_direction: 0 down, 1 up, 2 right, 3 left.
_FACE_CODE = {"down": 0, "up": 1, "right": 2, "left": 3}

# Potato dry/wet pair from the tape (0x54 → 0x55).
PLANTED_DRY = 0x54
PLANTED_WET = 0x55
# 3x3 around (13,28): 8 tilled ring tiles, untilled notch for the plant stand.
PLOT_RING_SIZE = 8
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
        # Brief wait so a door/tool animation is not still eating X, then tap
        # anyway — lock can stick at 0 on the shed outdoor stand.
        if not _input_unlocked(world.ram) and self._steps <= 20:
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(make_action()),
                reason="wait input unlock",
            )
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
    _saw_tool: bool = field(default=False, init=False)
    _face_ok_step: int = field(default=0, init=False)

    def reset(self, world: WorldState) -> None:
        self._steps = 0
        self._saw_tool = False
        self._face_ok_step = 0

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
        sel = int(selected_tool(world.ram))
        back = int(backpack_tool(world.ram))
        wanted = int(self.tool_id)
        if sel == wanted or back == wanted:
            self._saw_tool = True
        if tid in self.done_ids:
            return TaskResult(
                status=TaskStatus.SUCCESS,
                reason=f"{self.name} tile=0x{tid:02X} at {tile}",
            )
        # Seed bags leave the carry pair when spent, often a few frames before
        # the metatile updates. Treat that as planted — do not fail-closed
        # with "tool not selected".
        if (
            self._saw_tool
            and wanted in _SEED_IDS
            and sel != wanted
            and back != wanted
        ):
            return TaskResult(
                status=TaskStatus.SUCCESS,
                reason=f"{self.name} bag spent tid=0x{tid:02X} at {tile}",
            )
        if tid in self.blocked_ids:
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=f"{self.name} blocked tid=0x{tid:02X} at {tile}",
            )
        held = int(read_held_item(world.ram))
        facing = int(read_ram_value(world.ram, "player_direction") or 0)
        if self._steps > self.timeout:
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=(
                    f"{self.name} timeout tid=0x{tid:02X} at {tile} "
                    f"pos={player_tile} held=0x{held:02X} face={facing}"
                ),
            )
        if held:
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(make_action()),
                reason=f"drop held=0x{held:02X} before tool use",
            )
        if not _input_unlocked(world.ram):
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(make_action()),
                reason="wait input unlock",
            )
        if self.target_tile is not None:
            dist = abs(player_tile[0] - tile[0]) + abs(player_tile[1] - tile[1])
            if dist != 1:
                return TaskResult(
                    status=TaskStatus.RUNNING,
                    action=ActionResult(make_action()),
                    reason=f"wait adjacent to {tile} from {player_tile}",
                )
        if sel != wanted:
            if back == wanted:
                return TaskResult(
                    status=TaskStatus.RUNNING,
                    action=ActionResult(make_action(x=True)),
                    reason="x-swap before tool use",
                )
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=f"tool 0x{wanted:02X} not selected",
            )
        wanted_face = _FACE_CODE.get(self.face) if self.face else None
        if (
            wanted_face is not None
            and self._face_ok_step == 0
            and facing != wanted_face
            and self._steps <= 24
        ):
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(make_action(**{self.face: True})),
                reason=f"face {self.face} (ram={facing})",
            )
        if self._face_ok_step == 0:
            self._face_ok_step = self._steps
        wait = self._steps - self._face_ok_step
        if wait < 6:
            action = make_action()
        elif wait < 26:
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
        blocked_ids=HOE_BLOCKED_TILES,
        target_tile=target_tile,
        timeout=timeout,
    )


def water_until_wet_skill(
    *,
    target_tile: Optional[Tuple[int, int]] = None,
    face: Optional[str] = None,
    timeout: int = 240,
) -> UseToolUntilTileSkill:
    return UseToolUntilTileSkill(
        name="water_until_wet",
        tool_id=int(Tool.WATERING_CAN),
        done_ids=WET_CROP,
        target_tile=target_tile,
        face=face,
        timeout=timeout,
    )


def count_ring_tiles(ram, center: Tuple[int, int], done_ids: FrozenSet[int]) -> int:
    """Count 3x3 ring tiles (not the notch) whose IDs are in ``done_ids``."""
    from harvest.tasks.crop_geometry import plot_tiles

    return sum(
        1
        for tx, ty in plot_tiles(center, include_center=False)
        if int(get_tile_at(ram, tx, ty)) in done_ids
    )


def count_ring_planted(ram, center: Tuple[int, int]) -> int:
    return count_ring_tiles(ram, center, PLANTED_OR_WET)


def count_ring_tilled(ram, center: Tuple[int, int]) -> int:
    return count_ring_tiles(ram, center, HOED_OR_PLANTED)


def count_ring_wet(ram, center: Tuple[int, int]) -> int:
    return count_ring_tiles(ram, center, WET_CROP)


def hoe_stand_px(stand: Tuple[int, int], face: str) -> Tuple[int, int]:
    """Pixel stand nudged *away* from the faced tile.

    Nav radius can land on the near edge of the stand tile; a face tap
    from that edge walks onto the hoe target. y=30 is the fence lip —
    extra south never arrives.
    """
    away = {"up": (0, 5), "down": (0, -5), "left": (5, 0), "right": (-5, 0)}
    dx, dy = away.get(face, (0, 0))
    if face == "up" and stand[1] >= 30:
        dy = 0
    return (stand[0] * TILE_SIZE + 8 + dx, stand[1] * TILE_SIZE + 8 + dy)


# Adjacent stand, face toward the target. South-of-target / face-up first so
# the well-body HOE_PLAN stand (15,27) for (14,27) remaps to (14,28).
_HOE_ALT_STANDS: Tuple[Tuple[Tuple[int, int], str], ...] = (
    ((0, 1), "up"),
    ((-1, 0), "right"),
    ((1, 0), "left"),
    ((0, -1), "down"),
)
# y=31 is the solid 0x05 wall; y=30 is the lip that never settles a face-up.
_FENCE_LIP_Y = 30
# Tight: radius 6 accepted (11,29) as the (12,29) hoe stand (live miss).
_RING_NAV_RADIUS = 3


def _pocket_hoe_stand_blocked(
    center: Tuple[int, int],
    target: Tuple[int, int],
    stand: Tuple[int, int],
    face: str,
) -> bool:
    """True when a pocket hoe stand cannot settle (no-go / fence / leftover stone)."""
    from harvest.maps.farm_pond import FARM_NO_GO_TILES

    if stand == target or stand in FARM_NO_GO_TILES:
        return True
    if stand[1] >= _FENCE_LIP_Y:
        return True
    # Face-up on y=29 nudges into y=30 leftover stones (live (12,30) sealed
    # nav_hoe_ring_6_up). Face-right on (12,29) is still the first hoe stand.
    if face == "up" and stand[1] >= 29:
        return True
    # East of the 3x3 on the fence-adjacent row. Live leftover stone at
    # (16,29) seals hoe_stand_px's rightward face-left nudge on (15,29).
    if stand == (center[0] + 2, center[1] + 1):
        return True
    return False


def remap_pocket_hoe_stand(
    center: Tuple[int, int],
    target: Tuple[int, int],
    stand: Tuple[int, int],
    face: str,
) -> Tuple[Tuple[int, int], str]:
    """Pocket-only hoe stand remaps. Does not rewrite HOE_PLAN.

    Fence-lip (cx, cy+2) face-up becomes west of the bottom ring, face-right.
    Well-body / fence-lip / leftover-stone stands pick an adjacent-to-target
    alternate (prefer south, face up; skip y>=30).
    """
    if target == (center[0], center[1] + 1) and face == "up":
        stand = (center[0] - 1, center[1] + 1)
        face = "right"
    if not _pocket_hoe_stand_blocked(center, target, stand, face):
        return stand, face
    for (dx, dy), alt_face in _HOE_ALT_STANDS:
        alt = (target[0] + dx, target[1] + dy)
        if _pocket_hoe_stand_blocked(center, target, alt, alt_face):
            continue
        return alt, alt_face
    return stand, face


def _ring_nav_tool_skills(
    plan,
    *,
    nav_prefix: str,
    make_tool,
    timeout: int,
    nav_timeout: int,
):
    from harvest.tasks.skills import NavSkill

    skills: list = []
    for index, (target, stand, face) in enumerate(plan):
        px, py = hoe_stand_px(stand, face)
        skills.append(
            NavSkill(
                name=f"{nav_prefix}_{index}_{face}",
                target_px=(px, py),
                radius=_RING_NAV_RADIUS,
                soft_radius=_RING_NAV_RADIUS,
                timeout=nav_timeout,
                require_tilemap=0x00,
            )
        )
        skills.append(make_tool(target, face, timeout))
    return skills


def pocket_hoe_ring_skills(
    center: Tuple[int, int],
    *,
    timeout: int = 240,
    nav_timeout: int = 4500,
):
    """Nav to each HOE_PLAN stand and till the faced ring tile.

    Does not hoe the notch. Starts west of (13,29) so the first swing is
    already facing right. Shed-door leave is owned by
    ``farm_nav_pocket_hoe_stand_skill``. Well-body, fence-lip, and the
    leftover-stone east stand (15,29) are remapped.
    """
    from harvest.tasks.crop_geometry import hoe_plan

    plan = []
    for target, stand, face in hoe_plan(center):
        stand, face = remap_pocket_hoe_stand(center, target, stand, face)
        plan.append((target, stand, face))
    # Start with that bottom-center cell — nav_pocket_hoe_stand lands there.
    ordered = plan[-1:] + plan[:-1]
    return _ring_nav_tool_skills(
        ordered,
        nav_prefix="nav_hoe_ring",
        make_tool=lambda tile, facing, to: hoe_until_tilled_skill(
            target_tile=tile, face=facing, timeout=to
        ),
        timeout=timeout,
        nav_timeout=nav_timeout,
    )


def pocket_water_ring_skills(
    center: Tuple[int, int],
    *,
    timeout: int = 240,
    nav_timeout: int = 4500,
):
    """Water the 8-ring from WATER_PLAN_CENTER stands. Skip the untilled notch.

    Cardinals from the center; corners from right-middle / left-middle.
    """
    from harvest.tasks.crop_geometry import WATER_PLAN_CENTER

    cx, cy = center
    plan = [
        ((cx + tdx, cy + tdy), (cx + sdx, cy + sdy), face)
        for tdx, tdy, sdx, sdy, face in WATER_PLAN_CENTER
        if (tdx, tdy) != (0, 0)
    ]
    return _ring_nav_tool_skills(
        plan,
        nav_prefix="nav_water_ring",
        make_tool=lambda tile, facing, to: water_until_wet_skill(
            target_tile=tile, face=facing, timeout=to
        ),
        timeout=timeout,
        nav_timeout=nav_timeout,
    )


@dataclass
class PlantPlotSkill(Task):
    """Y the seed bag on the untilled notch until the 8-ring is crop.

    Potato plants the 3x3 around the player. Center stays untilled. Bag
    spend with fewer than ``min_planted`` tiles is a miss (the 1-cell path).
    """

    name: str = "plant_until_plot"
    seed_type: str = "potato"
    center: Tuple[int, int] = (13, 28)
    min_planted: int = PLOT_RING_SIZE
    timeout: int = 480

    _steps: int = field(default=0, init=False)
    _saw_tool: bool = field(default=False, init=False)
    _tool_id: int = field(default=0, init=False)

    def reset(self, world: WorldState) -> None:
        self._steps = 0
        self._saw_tool = False
        self._tool_id = int(seed_item_id(self.seed_type))

    def can_start(self, world: WorldState) -> bool:
        return True

    def progress_snapshot(self) -> ProgressSnapshot:
        return ProgressSnapshot(
            task_name=self.name,
            phase_text="plant_plot",
            step_count=self._steps,
            details=(("min_planted", self.min_planted),),
        )

    def step(self, world: WorldState) -> TaskResult:
        self._steps += 1
        planted = count_ring_planted(world.ram, self.center)
        sel = int(selected_tool(world.ram))
        back = int(backpack_tool(world.ram))
        wanted = int(self._tool_id or seed_item_id(self.seed_type))
        if sel == wanted or back == wanted:
            self._saw_tool = True
        if planted >= int(self.min_planted):
            return TaskResult(
                status=TaskStatus.SUCCESS,
                reason=f"{self.name} planted={planted} at {self.center}",
            )
        bag_gone = self._saw_tool and sel != wanted and back != wanted
        if bag_gone:
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=(
                    f"{self.name} bag spent planted={planted} "
                    f"< {self.min_planted} at {self.center}"
                ),
            )
        if self._steps > self.timeout:
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=(
                    f"{self.name} timeout planted={planted} "
                    f"< {self.min_planted} at {self.center}"
                ),
            )
        if not _input_unlocked(world.ram):
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(make_action()),
                reason="wait input unlock",
            )
        if sel != wanted:
            if back == wanted:
                return TaskResult(
                    status=TaskStatus.RUNNING,
                    action=ActionResult(make_action(x=True)),
                    reason="x-swap before plant",
                )
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=f"tool 0x{wanted:02X} not selected",
            )
        phase = self._steps % 48
        action = make_action(y=True) if 8 <= phase < 28 else make_action()
        return TaskResult(
            status=TaskStatus.RUNNING,
            action=ActionResult(action),
            reason=f"Y until plot planted={planted}",
        )


def plant_until_plot_skill(
    *,
    seed_type: str = "potato",
    center: Optional[Tuple[int, int]] = None,
    min_planted: int = PLOT_RING_SIZE,
    timeout: int = 480,
) -> PlantPlotSkill:
    from harvest.maps.farm_pond import WEST_POCKET_PLANT_CENTER

    return PlantPlotSkill(
        seed_type=seed_type,
        center=center or WEST_POCKET_PLANT_CENTER,
        min_planted=min_planted,
        timeout=timeout,
    )


def _can_level(ram) -> int:
    try:
        return int(read_ram_value(ram, "watering_can") or 0)
    except Exception:
        return 0


@dataclass
class EnsureCanFilledTask(Task):
    """No-op when the can already holds ``min_level``. Else open the y=31
    pond gap (corridor_only) and Y at the F0 stand until the can fills.

    D2 shelf pickup leaves watering_can=0. Pocket water with an empty can
    is a 0x54 timeout (live Y1_After_Buy_Potato).
    """

    name: str = "ensure_can_filled"
    min_level: int = 8
    timeout: int = 12_000

    _steps: int = field(default=0, init=False)
    _phase: str = field(default="check", init=False)
    _sub: Optional[Task] = field(default=None, init=False)
    _face_ok_step: int = field(default=0, init=False)

    def reset(self, world: WorldState) -> None:
        self._steps = 0
        self._phase = "check"
        self._sub = None
        self._face_ok_step = 0

    def can_start(self, world: WorldState) -> bool:
        return True

    def progress_snapshot(self) -> ProgressSnapshot:
        return ProgressSnapshot(
            task_name=self.name,
            phase_text=self._phase,
            step_count=self._steps,
            details=(("min_level", self.min_level),),
        )

    def step(self, world: WorldState) -> TaskResult:
        self._steps += 1
        level = _can_level(world.ram)
        if level >= int(self.min_level):
            return TaskResult(
                status=TaskStatus.SUCCESS,
                reason=f"{self.name} can={level}",
            )
        if self._steps > self.timeout:
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=f"{self.name} timeout can={level} phase={self._phase}",
            )
        if self._phase == "check":
            from harvest.tasks.fence_flow import FenceClearLoopTask

            self._phase = "fence"
            self._sub = FenceClearLoopTask(
                max_fences=2,
                max_steps_per_fence=1600,
                corridor_only=True,
            )
            self._sub.reset(world)
        if self._phase == "fence" and self._sub is not None:
            result = self._sub.step(world)
            if result.status == TaskStatus.RUNNING:
                return result
            from harvest.tasks.skills import farm_nav_to_pond_refill_skill

            self._phase = "nav"
            self._sub = farm_nav_to_pond_refill_skill()
            self._sub.reset(world)
        if self._phase == "nav" and self._sub is not None:
            result = self._sub.step(world)
            if result.status == TaskStatus.RUNNING:
                return result
            self._phase = "fill"
            self._sub = None
            self._face_ok_step = 0
        if self._phase == "fill":
            if not _input_unlocked(world.ram):
                return TaskResult(
                    status=TaskStatus.RUNNING,
                    action=ActionResult(make_action()),
                    reason="wait input unlock",
                )
            sel = int(selected_tool(world.ram))
            if sel != int(Tool.WATERING_CAN):
                back = int(backpack_tool(world.ram))
                if back == int(Tool.WATERING_CAN):
                    return TaskResult(
                        status=TaskStatus.RUNNING,
                        action=ActionResult(make_action(x=True)),
                        reason="x-swap can before fill",
                    )
                return TaskResult(
                    status=TaskStatus.FAILURE,
                    reason="watering can not in carry pair",
                )
            from harvest.maps.map_config import farm_pond_refill_primary_stand
            from harvest.tasks.skills import farm_pond_refill_face

            stand, face = farm_pond_refill_primary_stand()
            face = farm_pond_refill_face() or face
            pos = get_pos_from_ram(world.ram)
            player_tile = (pos.x // TILE_SIZE, pos.y // TILE_SIZE)
            if player_tile != stand:
                dx = stand[0] - player_tile[0]
                dy = stand[1] - player_tile[1]
                step = (
                    {"up": True}
                    if dy < 0 and abs(dy) >= abs(dx)
                    else {"down": True}
                    if dy > 0 and abs(dy) >= abs(dx)
                    else {"left": True}
                    if dx < 0
                    else {"right": True}
                )
                self._face_ok_step = 0
                return TaskResult(
                    status=TaskStatus.RUNNING,
                    action=ActionResult(make_action(**step)),
                    reason=f"walk onto pond stand {stand} from {player_tile}",
                )
            facing = int(read_ram_value(world.ram, "player_direction") or 0)
            wanted_face = _FACE_CODE.get(face)
            if (
                wanted_face is not None
                and self._face_ok_step == 0
                and facing != wanted_face
            ):
                return TaskResult(
                    status=TaskStatus.RUNNING,
                    action=ActionResult(make_action(**{face: True})),
                    reason=f"face {face} (ram={facing})",
                )
            if self._face_ok_step == 0:
                self._face_ok_step = self._steps
            wait = self._steps - self._face_ok_step
            action = make_action(y=True) if 6 <= wait < 26 else make_action()
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(action),
                reason=f"Y fill can={level}",
            )
        return TaskResult(
            status=TaskStatus.FAILURE,
            reason=f"{self.name} unknown phase {self._phase}",
        )
