"""Registry mapping :class:`PhaseKind` to day-plan sub-task builders."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional

from retro_harness import ActionResult, Task, TaskResult, TaskStatus, WorldState

from harvest.maps.map_config import ROUTES
from harvest.planner.day_phase_types import DayPlannerPolicy, PhaseKind, PhaseSpec
from harvest.planner.day_plan_status import FARM_TILEMAP, TASKS_DIR

if TYPE_CHECKING:
    from harvest.core.world_context import WorldContext
from harvest.planner.tasks.chicken_sale import (
    ChickenSaleEventTask,
    ChickenSaleFollowupTask,
    ChickenSaleRequestTask,
    CoopPickupChickenTask,
    CowPurchaseTask,
    DropCarriedChickenTask,
)
from harvest.planner.tasks.home import GoToSleepTask, ReturnHomeTask, GatherHouseWoodTask, HouseUpgradeTask
from harvest.planner.tasks.inventory import (
    DeadlineCheckTask,
    EnsureAnimalToolsTask,
    EnsureCarryToolTask,
    EnsureCropSeedsTask,
    ExitToFarmTask,
    FarmExitTask,
    RecordingSliceSpec,
    WaitUntilTimeTask,
    load_recording_slice,
)
from harvest.planner.tasks.navigation import (
    CrossMapRecordedTask,
    MultiMapNavTask,
    NavTask,
    RecordedTransitionTask,
)
from harvest.planner.tasks.transitions import DirectionalTransitionTask, ExitBuildingTask
from harvest.tasks.coop_task import CoopChoresTask
from harvest.tasks.cow_task import CowChoresTask
from harvest.tasks.crop_planter import CropWaterTask
from harvest.tasks.eve_loop_task import EveTalkLoopTask
from harvest.tasks.eve_proposal import EveProposalTask
from harvest.tasks.flower_festival import FlowerFestivalTask
from harvest.tasks.farm_clear_task import FarmClearTask
from harvest.tasks.nav import Point
from harvest.core.tile_catalog import Tool
from harvest.tasks.harvest_task import HarvestTask, crop_nav_target_px, live_harvestable_crop_tiles
from harvest.tasks.buy_seeds import BuyShelfItemTask
from harvest.tasks.mountain_berry import MountainBerryTask
from harvest.tasks.mountain_grape_ship import MountainGrapeShipTask
from harvest.tasks.recorded_task import RecordedTask

PhaseTaskBuilder = Callable[["TaskBuildContext", PhaseSpec, WorldState], Optional[Task]]


@dataclass(frozen=True)
class TaskBuildContext:
    """Inputs shared by all phase task builders.

    Keep builders pure functions of ``(ctx, spec, world)``. Optional policy and
    world_context let skills share calendar/stamina facts and cached reads
    without re-probing RAM on every construction.
    """

    seed_type: str = "potato"
    tasks_dir: str = TASKS_DIR
    state_name: Optional[str] = None
    policy: Optional[DayPlannerPolicy] = None
    world_context: Optional["WorldContext"] = None


@dataclass
class _ImmediateSuccessTask(Task):
    """Trivial task that reports SUCCESS on the first step (nothing to do)."""

    name: str = "noop_success"
    reason: str = "nothing to do"

    def reset(self, world: WorldState) -> None:
        return None

    def can_start(self, world: WorldState) -> bool:
        return True

    def step(self, world: WorldState) -> TaskResult:
        return TaskResult(status=TaskStatus.SUCCESS, reason=self.reason)


def _build_exit(
    ctx: TaskBuildContext, spec: PhaseSpec, _world: WorldState
) -> Task:
    return ExitBuildingTask(
        target_tilemap=spec.params.get("target_tilemap", 0x00),
        dialog_frames=spec.params.get("dialog_frames", 120),
        timeout=spec.params.get("timeout", 600),
    )


def _build_farm_building_exit(
    ctx: TaskBuildContext, spec: PhaseSpec, _world: WorldState
) -> Task:
    return ExitToFarmTask(
        tasks_dir=ctx.tasks_dir,
        house_timeout=spec.params.get("timeout", 2200),
    )


def _build_farm_exit(
    ctx: TaskBuildContext, spec: PhaseSpec, _world: WorldState
) -> Task:
    px = spec.params.get("target_px", (40, 424))
    return FarmExitTask(
        target_px=Point(px[0], px[1]),
        radius=spec.params.get("radius", 12),
        timeout=spec.params.get("timeout", 3000),
    )


def _build_nav(ctx: TaskBuildContext, spec: PhaseSpec, world: WorldState) -> Task:
    px = spec.params.get("target_px", (0, 0))
    if spec.phase == "NAV_CROP":
        px = crop_nav_target_px(
            world.ram,
            ctx.state_name,
            fallback_px=px,
        )
    return NavTask(
        name=f"nav_{spec.phase}",
        target_px=Point(px[0], px[1]),
        radius=spec.params.get("radius", 16),
        soft_radius=spec.params.get("soft_radius"),
        timeout=spec.params.get("timeout", 3000),
    )


def _build_recorded(
    ctx: TaskBuildContext, spec: PhaseSpec, _world: WorldState
) -> Optional[Task]:
    task_name = spec.params.get("task_name", "")
    try:
        return RecordedTask.load(task_name, ctx.tasks_dir)
    except FileNotFoundError:
        print(f"[DAY_PLAN] Recording not found: {task_name}")
        return None


def _build_recorded_slice(
    ctx: TaskBuildContext, spec: PhaseSpec, _world: WorldState
) -> Task:
    return load_recording_slice(
        RecordingSliceSpec(
            task_name=spec.params.get("task_name", ""),
            start_frame=spec.params.get("start_frame", 0),
            end_frame=spec.params.get("end_frame"),
        ),
        ctx.tasks_dir,
    )


def _build_cow_purchase(
    ctx: TaskBuildContext, spec: PhaseSpec, _world: WorldState
) -> Task:
    return CowPurchaseTask(
        species=str(spec.params.get("species", "cow")),
        timeout=int(spec.params.get("timeout", 3600)),
    )


def _build_hen_delivery(
    ctx: TaskBuildContext, spec: PhaseSpec, _world: WorldState
) -> Task:
    from harvest.tasks.coop_task import HenDeliveryTask

    return HenDeliveryTask()


def _build_pickup_chicken(
    ctx: TaskBuildContext, spec: PhaseSpec, _world: WorldState
) -> Task:
    return CoopPickupChickenTask(
        name=f"pickup_chicken_{spec.phase.lower()}",
        timeout=spec.params.get("timeout", 1800),
    )


def _build_drop_chicken(
    ctx: TaskBuildContext, spec: PhaseSpec, _world: WorldState
) -> Task:
    target_px = spec.params.get("target_px", (60, 480))
    return DropCarriedChickenTask(
        name=f"drop_chicken_{spec.phase.lower()}",
        target_px=(int(target_px[0]), int(target_px[1])),
        radius=spec.params.get("radius", 2),
        timeout=spec.params.get("timeout", 3000),
    )


def _build_chicken_sale_followup(
    ctx: TaskBuildContext, spec: PhaseSpec, _world: WorldState
) -> Task:
    return ChickenSaleFollowupTask(
        name=f"chicken_sale_{spec.phase.lower()}",
        task_name=spec.params.get("task_name", "sell_chicken"),
        start_frame=spec.params.get("start_frame", 1295),
        end_frame=spec.params.get("end_frame"),
        tasks_dir=ctx.tasks_dir,
        require_start_px=spec.params.get("require_start_px", (60, 480)),
        start_tolerance=spec.params.get("start_tolerance", 12),
        success_settle_frames=spec.params.get("success_settle_frames", 30),
    )


def _build_chicken_sale_request(
    ctx: TaskBuildContext, spec: PhaseSpec, _world: WorldState
) -> Task:
    return ChickenSaleRequestTask(
        name=f"chicken_sale_request_{spec.phase.lower()}",
        task_name=spec.params.get("task_name", "sell_chicken"),
        start_frame=spec.params.get("start_frame", 2863),
        end_frame=spec.params.get("end_frame", 3297),
        tasks_dir=ctx.tasks_dir,
        require_start_px=spec.params.get("require_start_px", (201, 158)),
        start_tolerance=spec.params.get("start_tolerance", 6),
        timeout=spec.params.get("timeout", 900),
    )


def _build_chicken_sale_event(
    ctx: TaskBuildContext, spec: PhaseSpec, _world: WorldState
) -> Task:
    return ChickenSaleEventTask(
        name=f"chicken_sale_event_{spec.phase.lower()}",
        standby_px=spec.params.get("standby_px", (62, 448)),
        payout_px=spec.params.get("payout_px", (146, 457)),
        event_hour=spec.params.get("event_hour", 15),
        target_sales=spec.params.get("target_sales", 1),
        timeout=spec.params.get("timeout", 18000),
        success_settle_frames=spec.params.get("success_settle_frames", 30),
    )


def _build_recorded_transition(
    ctx: TaskBuildContext, spec: PhaseSpec, _world: WorldState
) -> Task:
    return RecordedTransitionTask(
        name=f"recorded_transition_{spec.phase}",
        task_name=spec.params.get("task_name", ""),
        target_tilemap=spec.params.get("target_tilemap", 0x00),
        origin_tilemap=spec.params.get("origin_tilemap"),
        tasks_dir=ctx.tasks_dir,
        timeout=spec.params.get("timeout", 2000),
        min_frames_before_success=spec.params.get("min_frames_before_success", 1),
    )


def _build_cross_map(
    ctx: TaskBuildContext, spec: PhaseSpec, _world: WorldState
) -> Task:
    recording_name = spec.params.get("recording_name", "")
    stock_field = spec.params.get("stock_field", "")
    if not stock_field and "potato" in str(recording_name):
        stock_field = "potato_seeds"
    return CrossMapRecordedTask(
        name=f"cross_map_{spec.phase}",
        exit_direction=spec.params.get("exit_direction", "left"),
        recording_name=recording_name,
        recording_start=spec.params.get("recording_start", 0),
        origin_tilemap=spec.params.get("origin_tilemap", 0x00),
        tasks_dir=ctx.tasks_dir,
        timeout=spec.params.get("timeout", 5000),
        continue_after_return=spec.params.get("continue_after_return", 0),
        stock_field=stock_field,
        require_purchase=bool(stock_field),
    )


def _shop_bag_count(spec: PhaseSpec, world: WorldState, stock_field: str) -> int:
    """Bags to buy this trip: one per ring still waiting on seed, less stock.

    The shop round trip is ~2480 f whether it carries one bag or six, so a
    harvest day that empties several rings should not pay for it per ring.
    Capped by ``max_bags`` (default: every ring site) and by the task's own
    wallet clamp (``BuySeedsTask._bags_target``). In Spring a ripe one-shot
    ring counts: the shop runs before the harvest empties it.
    """
    from harvest.core.ram_catalog import read_ram_value
    from harvest.maps.farm_pond import POCKET_PLANT_CENTERS, pocket_plant_targets

    if "target_stock" in spec.params:
        return max(1, int(spec.params["target_stock"]) - int(read_ram_value(world.ram, stock_field)))
    cap = int(spec.params.get("max_bags", len(POCKET_PLANT_CENTERS)))
    if cap <= 1:
        return 1
    try:
        wanted = len(pocket_plant_targets(world.ram, replant_ripe=True))
        in_stock = int(read_ram_value(world.ram, stock_field) or 0)
    except Exception:
        return 1
    return max(1, min(cap, wanted - in_stock))


def _build_shop_buy(
    ctx: TaskBuildContext, spec: PhaseSpec, _world: WorldState
) -> Task:
    stock_field = spec.params.get("stock_field", "potato_seeds")
    tool_ids = tuple(int(t) for t in spec.params.get("tool_ids", ()))
    return BuyShelfItemTask(
        name=f"shop_buy_{spec.phase.lower()}",
        timeout=spec.params.get("timeout", 14_000),
        nav_timeout=spec.params.get("nav_timeout", 6_000),
        stock_field=stock_field,
        bags=1 if tool_ids else _shop_bag_count(spec, _world, stock_field),
        tool_ids=tool_ids,
    )


def _build_grass(ctx: TaskBuildContext, spec: PhaseSpec, world: WorldState) -> Task:
    from harvest.maps.farm_pond import SHIP_BIN_LANE_TILES
    from harvest.tasks.grass_planter import GrassPlantTask, PASTURE_BOUNDS
    from harvest.tasks.primitives import TaskSequence
    from harvest.maps.map_config import Waypoint

    cutting = bool(spec.params.get("cut_only", False))
    bounds = tuple(spec.params.get("bounds", PASTURE_BOUNDS))
    work = GrassPlantTask(bounds=bounds, no_go_rects=[], cut_only=cutting,
                          target_tiles=0 if cutting else int(spec.params.get("target_tiles", 32)),
                          bag_limit=0 if cutting else int(spec.params.get("bag_limit", 2)))
    fetch = (EnsureCarryToolTask(tool_id=int(Tool.SICKLE)) if cutting
             else EnsureCropSeedsTask(seed_type="grass"))
    nav = MultiMapNavTask(waypoints=[Waypoint(0x00, px, radius=8) for px in
                         ((136, 424), (232, 392), (296, 392), (312, 488), (312, 520), (296, 536), (264, 536))],
                         timeout=4000, auto_slice=True, allow_tiles=SHIP_BIN_LANE_TILES)
    return TaskSequence(tasks=[fetch, nav, work], name=spec.phase.lower())


def _build_directional_transition(
    ctx: TaskBuildContext, spec: PhaseSpec, _world: WorldState
) -> Task:
    return DirectionalTransitionTask(
        name=f"directional_transition_{spec.phase.lower()}",
        direction=spec.params.get("direction", "down"),
        origin_tilemap=spec.params.get("origin_tilemap"),
        target_tilemap=spec.params.get("target_tilemap", FARM_TILEMAP),
        timeout=spec.params.get("timeout", 600),
        min_frames_before_success=spec.params.get("min_frames_before_success", 15),
        settle_frames=spec.params.get("settle_frames", 0),
        stand_tile=spec.params.get("stand_tile"),
        stand_tolerance=spec.params.get("stand_tolerance", 0),
        target_stand_tile=spec.params.get("target_stand_tile"),
        target_stand_tolerance=spec.params.get("target_stand_tolerance", 0),
        door_align_px=spec.params.get("door_align_px"),
        door_align_tolerance=spec.params.get("door_align_tolerance", 4),
        overshoot_limit_px=spec.params.get("overshoot_limit_px"),
        require_empty_hands=spec.params.get("require_empty_hands", False),
        clear_hands_limit=spec.params.get("clear_hands_limit", 4),
        walk_into_door=spec.params.get("walk_into_door", False),
    )


def _build_multi_nav(
    ctx: TaskBuildContext, spec: PhaseSpec, _world: WorldState
) -> Optional[Task]:
    route_name = spec.params.get("route", "")
    waypoints = ROUTES.get(route_name, [])
    if not waypoints:
        print(f"[DAY_PLAN] Unknown route: {route_name}")
        return None
    return MultiMapNavTask(
        name=f"multi_nav_{spec.phase}",
        waypoints=list(waypoints),
        timeout=spec.params.get("timeout", 8000),
        initial_settle_frames=spec.params.get("initial_settle_frames", 60),
        auto_slice=True,
    )


def _build_mountain_berry(
    ctx: TaskBuildContext, spec: PhaseSpec, _world: WorldState
) -> Task:
    if spec.params.get("ship", False):
        return MountainGrapeShipTask(
            name=f"mountain_grape_ship_{spec.phase.lower()}",
            timeout=spec.params.get("timeout", 20_000),
            pick_timeout=spec.params.get("pick_timeout", 12_000),
            nav_timeout=spec.params.get("nav_timeout", 12_000),
            pick_attempts=spec.params.get("pick_attempts", 3),
            target_count=spec.params.get("count", 1),
            shop_bail_hour=spec.params.get("shop_bail_hour", 10),
            forage=spec.params.get("forage", "grape"),
        )
    return MountainBerryTask(
        name=f"mountain_berry_{spec.phase.lower()}",
        timeout=spec.params.get("timeout", 12_000),
        nav_timeout=spec.params.get("nav_timeout", 6_000),
        pick_attempts=spec.params.get("pick_attempts", 0),
        approach_only=spec.params.get("approach_only", True),
    )


def _build_ensure_tool(
    ctx: TaskBuildContext, spec: PhaseSpec, _world: WorldState
) -> Task:
    return EnsureCarryToolTask(
        name=f"ensure_tool_{spec.phase.lower()}",
        tool_id=spec.params.get("tool_id", int(Tool.WATERING_CAN)),
        tasks_dir=ctx.tasks_dir,
    )


def _build_ensure_animal_tools(
    ctx: TaskBuildContext, spec: PhaseSpec, _world: WorldState
) -> Task:
    return EnsureAnimalToolsTask(
        name=f"ensure_animal_tools_{spec.phase.lower()}",
        tasks_dir=ctx.tasks_dir,
        first_tool_id=spec.params.get("first_tool_id", int(Tool.MILKER)),
        second_tool_id=spec.params.get("second_tool_id", int(Tool.BRUSH)),
    )


def _build_ensure_seed(
    ctx: TaskBuildContext, spec: PhaseSpec, _world: WorldState
) -> Task:
    return EnsureCropSeedsTask(
        name=f"ensure_seed_{spec.phase.lower()}",
        seed_type=spec.params.get("seed_type", ctx.seed_type),
        tasks_dir=ctx.tasks_dir,
    )


def _build_deadline(
    ctx: TaskBuildContext, spec: PhaseSpec, _world: WorldState
) -> Task:
    return DeadlineCheckTask(
        name=f"deadline_{spec.phase.lower()}",
        latest_hour=spec.params.get("latest_hour", 17),
        latest_minute=spec.params.get("latest_minute", 0),
    )


def _build_wait_until_time(
    ctx: TaskBuildContext, spec: PhaseSpec, _world: WorldState
) -> Task:
    return WaitUntilTimeTask(
        name=f"wait_until_{spec.phase.lower()}",
        target_hour=spec.params.get("target_hour", 12),
        target_minute=spec.params.get("target_minute", 0),
        timeout=spec.params.get("timeout", 4000),
    )


def _build_harvest(ctx: TaskBuildContext, spec: PhaseSpec, _world: WorldState) -> Task:
    return HarvestTask(
        name=f"harvest_{spec.phase.lower()}",
        state_name=ctx.state_name,
    )


def _build_clear_field(
    ctx: TaskBuildContext, spec: PhaseSpec, _world: WorldState
) -> Optional[Task]:
    if spec.phase == "D2_FARM_CLEAR":
        # Marker only. DayPlanTask splices the next live child.
        return None
    bounds = spec.params.get("farm_bounds")
    if bounds is not None:
        bounds = tuple(int(v) for v in bounds)
    raw_priority = spec.params.get("priority")
    priority = None
    if raw_priority:
        from harvest.core.tile_catalog import DebrisType

        priority = [DebrisType[str(name).upper()] for name in raw_priority]
    return FarmClearTask(
        name=f"clear_field_{spec.phase.lower()}",
        tasks_dir=ctx.tasks_dir,
        fetch_tools=spec.params.get("fetch_tools", True),
        timeout=spec.params.get("timeout", 120000),
        prefer_lift_for_weeds=spec.params.get("prefer_lift_for_weeds", True),
        prefer_lift_for_stones=spec.params.get("prefer_lift_for_stones", False),
        farm_bounds=bounds,
        priority=priority,
        handoff=str(spec.params.get("handoff") or ""),
        quota=spec.params.get("quota"),
    )


def _build_fence_clear(
    ctx: TaskBuildContext, spec: PhaseSpec, _world: WorldState
) -> Task:
    """Open y=31 fence gap so south-farm routes (berry, pond) can BFS."""
    from harvest.core.tile_catalog import DebrisType
    from harvest.tasks.fence_flow import FenceClearLoopTask

    raw_max = spec.params.get("max_fences", 2)
    max_fences = None if raw_max is None else int(raw_max)
    raw_types = spec.params.get("debris_types") or ("fence",)
    debris_types = tuple(DebrisType[str(name).upper()] for name in raw_types)
    raw_steps = spec.params.get("max_steps_per_fence")
    # Phase timeout is the whole-loop budget. One stuck post must not eat it.
    max_steps_per_fence = int(raw_steps) if raw_steps is not None else 2400
    bounds = spec.params.get("farm_bounds")
    if bounds is not None:
        bounds = tuple(int(v) for v in bounds)
    return FenceClearLoopTask(
        name=f"fence_clear_{spec.phase.lower()}",
        max_fences=max_fences,
        corridor_only=bool(spec.params.get("corridor_only", True)),
        max_steps_per_fence=max_steps_per_fence,
        max_failures=int(spec.params.get("max_failures", 3)),
        pond_dump=bool(spec.params.get("pond_dump", False)),
        debris_types=debris_types,
        farm_bounds=bounds,
        timeout=int(spec.params.get("timeout", 0) or 0),
    )


def _build_coop_chores(
    ctx: TaskBuildContext, spec: PhaseSpec, _world: WorldState
) -> Task:
    return CoopChoresTask(
        name=f"coop_{spec.phase.lower()}",
        egg_mode=spec.params.get("egg_mode", "auto"),
        max_feed_adults=spec.params.get("max_feed_adults"),
    )


def _build_cow_chores(
    ctx: TaskBuildContext, spec: PhaseSpec, _world: WorldState
) -> Task:
    return CowChoresTask(
        name=f"cow_{spec.phase.lower()}",
        talk=spec.params.get("talk", True),
        brush=spec.params.get("brush", True),
        milk=spec.params.get("milk", True),
        feed=spec.params.get("feed", True),
    )


def _build_eve_talk_loop(
    ctx: TaskBuildContext, spec: PhaseSpec, _world: WorldState
) -> Task:
    return EveTalkLoopTask(
        name=f"eve_{spec.phase.lower()}",
        target_hearts=spec.params.get("target_hearts", 10),
        target_points=spec.params.get("target_points"),
        max_entries=spec.params.get("max_entries", 80),
        timeout=spec.params.get("timeout", 360000),
    )


def _build_flower_festival(
    ctx: TaskBuildContext, spec: PhaseSpec, _world: WorldState
) -> Task:
    return FlowerFestivalTask(name=f"festival_{spec.phase.lower()}", timeout=spec.params.get("timeout", 60000))


def _build_propose_eve(
    ctx: TaskBuildContext, spec: PhaseSpec, _world: WorldState
) -> Task:
    return EveProposalTask(name=f"eve_{spec.phase.lower()}", timeout=spec.params.get("timeout", 60000))


def _build_crop(ctx: TaskBuildContext, spec: PhaseSpec, world: WorldState) -> Task:
    refill_bounds = spec.params.get("refill_bounds")
    work_mode = spec.params.get("work_mode", "full")
    # First plant: reactive 8-ring hoe+plant. D2 water is work_mode=pocket.
    if spec.phase == "CROP_ESTABLISH" or str(work_mode) == "establish":
        from harvest.maps.farm_pond import (
            next_unplanted_pocket_center,
            ring_establish_affordable,
        )
        from harvest.tasks.skills import farm_pocket_plant_skill

        center = next_unplanted_pocket_center(world.ram)
        if center is None:
            return _ImmediateSuccessTask(
                name=f"crop_{spec.phase.lower()}_noop",
                reason="no_work:no pocket ring needs planting",
            )
        if not ring_establish_affordable(world.ram, center):
            return _ImmediateSuccessTask(
                name=f"crop_{spec.phase.lower()}_noop",
                reason=f"no_work:stamina for hoe and standing-crop watering ({center})",
            )
        return farm_pocket_plant_skill(
            seed_type=ctx.seed_type,
            center=center,
            ram=world.ram,
            include_water=False,
        )
    if str(work_mode) == "pocket":
        from harvest.maps.farm_pond import pocket_water_center
        from harvest.tasks.skills import farm_pocket_water_skill

        return farm_pocket_water_skill(center=pocket_water_center(world.ram))
    skip_water_tiles = set(live_harvestable_crop_tiles(world.ram, ctx.state_name))
    return CropWaterTask(
        seed_type=ctx.seed_type,
        work_mode=str(work_mode),
        refill_bounds=refill_bounds,
        skip_water_tiles=skip_water_tiles,
    )


def _min_stamina_param(params: dict) -> int | None:
    raw = params.get("min_stamina", "full")
    if raw in (None, "full", "max", ""):
        return None
    return int(raw)


def _build_hot_spring(
    ctx: TaskBuildContext, spec: PhaseSpec, _world: WorldState
) -> Task:
    from harvest.tasks.hot_spring import HotSpringStaminaTask

    return HotSpringStaminaTask(
        name=f"hot_spring_{spec.phase.lower()}",
        min_stamina=_min_stamina_param(spec.params),
        return_to_farm=bool(spec.params.get("return_to_farm", True)),
        tasks_dir=ctx.tasks_dir,
        timeout=int(spec.params.get("timeout", 24000)),
    )


def _build_return_home(
    ctx: TaskBuildContext, spec: PhaseSpec, _world: WorldState
) -> Task:
    return ReturnHomeTask(tasks_dir=ctx.tasks_dir)


def _build_sleep(
    ctx: TaskBuildContext, spec: PhaseSpec, _world: WorldState
) -> Task:
    return GoToSleepTask(tasks_dir=ctx.tasks_dir)


def _build_ready_to_go_home(
    ctx: TaskBuildContext, spec: PhaseSpec, _world: WorldState
) -> Task:
    from harvest.planner.tasks.home import ReadyToGoHomeTask

    return ReadyToGoHomeTask()


_WIFE_MASK = 0x001F
_FEATHER = 0x18
_PROPOSE_HEARTS = 250

_FEATHER_SHED = 0x11F02
_FEATHER_BIT = 0x80


def _u16(ram, key: str) -> int:
    from harvest.core.ram_catalog import read_ram_value

    return int(read_ram_value(ram, key))


def _wife_bits(ram) -> int:
    return _u16(ram, "marriage_flags") & _WIFE_MASK


def _best_girl(ram) -> tuple[str, int] | None:
    # Only Ann has a measured proposal route. Other hearts never select it.
    hearts = _u16(ram, "ann_hearts")
    return ("ann", hearts) if hearts >= _PROPOSE_HEARTS else None


def proposal_phase_for_ram(ram) -> PhaseSpec | None:
    from harvest.core.carry import tool_in_carry_pair
    from harvest.core.ram_catalog import read_ram_u8
    from harvest.planner.world_probe import WorldProbe

    probe = WorldProbe(ram=ram)
    if (_wife_bits(ram) or _best_girl(ram) is None or probe.weekday() != 6
            or probe.clock().hour >= 11 or probe.is_storm() or probe.is_festival()):
        return None
    owned = tool_in_carry_pair(ram, _FEATHER) or bool(read_ram_u8(ram, _FEATHER_SHED) & _FEATHER_BIT)
    return PhaseSpec("PROPOSE", PhaseKind.PROPOSE, failure_policy="optional") if owned else None


def _ann_stand(weekday: int) -> tuple[int, int, int, int]:
    # get_engaged: Summer Saturday at (402,901), feather Y+Up.
    return (0x04, 402, 889, 0x0226)


def _to_ann(tilemap: int, weekday: int, ram=None) -> list | None:
    from harvest.maps.map_routes import ROUTES, segment_waypoints
    from harvest.maps.map_types import Waypoint
    from harvest.planner.day_plan_status import is_farm_tilemap

    if weekday != 6:
        return None
    # Clear the sealed east gate along its measured open face first.
    town = list(segment_waypoints("town_to_shop_door")[:2])
    town += list(ROUTES["d1_town_to_ann"])
    town.append(Waypoint(tilemap=0x04, target_px=(402, 901), radius=0))
    if tilemap == 0x04:
        girl = _live_girl(ram, _ann_stand(weekday)[3]) if ram is not None else None
        if girl is not None:
            return [Waypoint(tilemap=0x04, target_px=(girl.pixel[0] + 13, girl.pixel[1] + 9), radius=0)]
        return town
    if is_farm_tilemap(tilemap):
        return list(ROUTES["farm_to_town"]) + town
    if tilemap == 0x0C:
        return list(ROUTES["path_to_town"]) + town
    if tilemap == 0x1C:
        return segment_waypoints("shop_to_town") + town
    return None


def _to_farm_for_feather(tilemap: int, ram=None) -> list | None:
    from harvest.maps.map_routes import ROUTES, segment_waypoints

    if tilemap == 0x1C:
        return segment_waypoints("shop_to_town")
    tail = segment_waypoints("path_to_farm")
    if tilemap == 0x04:
        from harvest.tasks.nav import get_pos_from_ram
        from harvest.maps.map_routes import slice_route_from_position
        square = list(segment_waypoints("town_shop_to_path"))
        if ram is not None:
            pos = get_pos_from_ram(ram)
            square = slice_route_from_position(square, pos.x, pos.y, tilemap=tilemap)
        return square + tail
    if tilemap == 0x0C:
        return list(ROUTES["path_to_farm"])
    return None

def _nav(waypoints: list, timeout: int) -> Task:
    from harvest.planner.tasks.multi_nav import MultiMapNavTask

    return MultiMapNavTask(
        name="propose_nav",
        waypoints=list(waypoints),
        timeout=timeout,
        initial_settle_frames=8,
    )


def _live_girl(ram, sprite: int):
    from harvest.core.npc_catalog import game_objects
    best, best_delta = None, 99
    for obj in game_objects(ram):
        delta = abs(int(obj.sprite_table_idx) - int(sprite))
        if not obj.is_player and delta <= 8 and delta < best_delta:
            best, best_delta = obj, delta
    return best


@dataclass
class ProposalTask(Task):
    name: str = "propose"
    timeout: int = 12000

    _child: Optional[Task] = field(default=None, init=False)
    _legs: int = field(default=0, init=False)
    _steps: int = field(default=0, init=False)
    _press: int = field(default=0, init=False)
    _mode: str = field(default="plan", init=False)
    _logged: bool = field(default=False, init=False)
    _last_pixel: tuple | None = field(default=None, init=False)
    _stasis: int = field(default=0, init=False)
    _offer_face: str = field(default="up", init=False)

    def reset(self, world: WorldState) -> None:
        self._child = None
        self._legs = 0
        self._steps = 0
        self._press = 0
        self._mode = "plan"
        self._logged = False
        self._last_pixel = None
        self._stasis = 0
        self._offer_face = "up"

    def can_start(self, world: WorldState) -> bool:
        return True

    def _tilemap(self, ram) -> int:
        from harvest.core.tile_catalog import ADDR_TILEMAP

        return int(ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(ram) else 0

    def _begin(self, world: WorldState, task: Task, label: str) -> TaskResult:
        self._legs += 1
        if self._legs > 8:
            return TaskResult(status=TaskStatus.FAILURE, reason="proposal route budget")
        self._mode = label
        self._last_pixel = None
        self._stasis = 0
        self._child = task
        task.reset(world)
        print(f"[PROPOSE] {label} map=0x{self._tilemap(world.ram):02X}", flush=True)
        return task.step(world)

    def _plan(self, world: WorldState) -> TaskResult:
        from harvest.core.carry import selected_tool, tool_in_carry_pair
        from harvest.planner.day_plan_status import SHED_TILEMAP, is_farm_tilemap
        from harvest.planner.tasks.inventory import EnsureCarryToolTask, ExitToFarmTask
        from harvest.planner.tasks.shed_slots import SwapCarrySlotsTask

        ram = world.ram
        if not self._logged:
            self._logged = True
            print(
                f"[PROPOSE] start flags=0x{_u16(ram, 'marriage_flags'):04X} "
                f"hour=0x{_u16(ram, 'hour'):02X} tool=0x{selected_tool(ram):02X}", flush=True)
        girl = _best_girl(ram)
        if girl is None:
            return TaskResult(status=TaskStatus.FAILURE, reason="no girl at 250 hearts")
        if not tool_in_carry_pair(ram, _FEATHER):
            tilemap = self._tilemap(ram)
            waypoints = _to_farm_for_feather(tilemap, ram)
            if waypoints is not None:
                return self._begin(world, _nav(waypoints, 20000), "fetch_route")
            if tilemap == SHED_TILEMAP or is_farm_tilemap(tilemap):
                fetch = EnsureCarryToolTask(tool_id=_FEATHER, nav_timeout=14000, exit_when_done=True)
                return self._begin(world, fetch, "fetch_shed")
            return self._begin(world, ExitToFarmTask(), "fetch_exit")
        if selected_tool(ram) != _FEATHER:
            return self._begin(world, SwapCarrySlotsTask(), "swap_feather")
        hour = _u16(ram, "hour")
        if _u16(ram, "weekday") != 6 or hour >= 11:
            return TaskResult(status=TaskStatus.FAILURE, reason=f"proposal hour gate hour=0x{hour:02X}")
        weekday = _u16(ram, "weekday")
        tilemap = self._tilemap(ram)
        if _close_enough(ram, weekday):
            self._mode = "press"
            self._press = 0
            self._offer_face = _face_girl(ram)
            return self._offer(world)
        waypoints = _to_ann(tilemap, weekday, ram)
        if waypoints is None:
            return self._begin(world, ExitToFarmTask(), "ann_exit")
        return self._begin(world, _nav(waypoints, 28000), "ann_route")

    def _offer(self, world: WorldState) -> TaskResult:
        from harvest.core.ram_catalog import read_ram_u8
        from harvest.core.tile_catalog import ADDR_INPUT_LOCK
        from harvest.tasks.nav import get_pos_from_ram, make_action

        ram = world.ram
        if _wife_bits(ram):
            return TaskResult(status=TaskStatus.SUCCESS, reason=f"wife bit 0x{_wife_bits(ram):04X}")
        self._press += 1
        if self._press > 1500:
            return TaskResult(status=TaskStatus.FAILURE, reason="wife bit stayed clear")
        hour = _u16(ram, "hour")
        if hour >= 11:
            return TaskResult(status=TaskStatus.FAILURE, reason=f"proposal hour gate hour=0x{hour:02X}")
        if self._press == 1:
            sprite = _ann_stand(_u16(ram, "weekday"))[3]
            girl = _live_girl(ram, sprite)
            pos = get_pos_from_ram(ram)
            seen = None if girl is None else (hex(girl.sprite_table_idx), girl.pixel)
            print(f"[PROPOSE] offer pos=({pos.x},{pos.y}) girl={seen}", flush=True)
        locked = int(ram[ADDR_INPUT_LOCK]) if ADDR_INPUT_LOCK < len(ram) else 1
        # get_engaged 2768: Y+Up through the animation, then face/idle;
        # frame 2943 confirms with A. Early A pulses consume the offer.
        face = self._offer_face
        if self._press <= 6:
            action = make_action(**{face: True})
        elif self._press <= 38:
            action = make_action(**{face: True}, y=True)
        elif self._press <= 68:
            action = make_action(**{face: True})
        elif self._press <= 186:
            action = make_action()
        else:
            action = make_action(a=self._press % 2 == 0)
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))

    def step(self, world: WorldState) -> TaskResult:
        self._steps += 1
        if _wife_bits(world.ram):
            return TaskResult(status=TaskStatus.SUCCESS, reason=f"wife bit 0x{_wife_bits(world.ram):04X}")
        if self._steps > self.timeout:
            return TaskResult(status=TaskStatus.FAILURE, reason="proposal timeout")
        if self._mode == "press":
            return self._offer(world)
        if self._child is not None:
            if self._mode.endswith("route"):
                from harvest.tasks.nav import get_pos_from_ram
                pos = get_pos_from_ram(world.ram)
                pixel = (self._tilemap(world.ram), pos.x, pos.y)
                self._stasis = self._stasis + 1 if pixel == self._last_pixel else 0
                self._last_pixel = pixel
                if self._stasis >= 180:
                    return TaskResult(status=TaskStatus.FAILURE, reason="proposal navigation stalled 180 frames")
            result = self._child.step(world)
            if result.status == TaskStatus.RUNNING:
                return result
            self._child = None
            if result.status != TaskStatus.SUCCESS:
                reason = f"{self._mode} failed: {result.reason or result.status.value}"
                return TaskResult(status=TaskStatus.FAILURE, reason=reason)
            self._mode = "plan"
        return self._plan(world)


def _close_enough(ram, weekday: int) -> bool:
    from harvest.core.tile_catalog import ADDR_TILEMAP
    from harvest.tasks.nav import get_pos_from_ram

    stand_map, x, y, sprite = _ann_stand(weekday)
    tilemap = int(ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(ram) else 0
    if tilemap != stand_map:
        return False
    pos = get_pos_from_ram(ram)
    girl = _live_girl(ram, sprite)
    if girl is None:
        return False
    return abs(pos.x - girl.pixel[0]) <= 15 and 6 <= pos.y - girl.pixel[1] <= 10


def _face_girl(ram) -> str:
    from harvest.tasks.nav import get_pos_from_ram

    weekday = _u16(ram, "weekday")
    _map, x, y, sprite = _ann_stand(weekday)
    girl = _live_girl(ram, sprite)
    if girl is not None:
        x, y = int(girl.pixel[0]), int(girl.pixel[1])
    pos = get_pos_from_ram(ram)
    dx, dy = x - pos.x, y - pos.y
    if abs(dx) > abs(dy):
        return "right" if dx > 0 else "left"
    return "down" if dy > 0 else "up"


def _build_propose(
    ctx: TaskBuildContext, spec: PhaseSpec, _world: WorldState
) -> Task:
    return ProposalTask()


def _build_marriage_skill(ctx: TaskBuildContext, spec: PhaseSpec, _world: WorldState) -> Task:
    from harvest.tasks.proposal import AnnCourtTask, FeatherBuyTask
    return AnnCourtTask() if spec.kind == PhaseKind.COURT else FeatherBuyTask()


PHASE_TASK_BUILDERS: dict[PhaseKind, PhaseTaskBuilder] = {
    PhaseKind.HOUSE_WOOD: lambda ctx, spec, world: GatherHouseWoodTask(),
    PhaseKind.HOME_UPGRADE: lambda ctx, spec, world: HouseUpgradeTask(goal=spec.params.get("goal", "order")),
    PhaseKind.EXIT: _build_exit,
    PhaseKind.FARM_BUILDING_EXIT: _build_farm_building_exit,
    PhaseKind.FARM_EXIT: _build_farm_exit,
    PhaseKind.NAV: _build_nav,
    PhaseKind.RECORDED: _build_recorded,
    PhaseKind.RECORDED_SLICE: _build_recorded_slice,
    PhaseKind.COW_PURCHASE: _build_cow_purchase,
    PhaseKind.HEN_DELIVERY: _build_hen_delivery,
    PhaseKind.PICKUP_CHICKEN: _build_pickup_chicken,
    PhaseKind.DROP_CHICKEN: _build_drop_chicken,
    PhaseKind.CHICKEN_SALE_FOLLOWUP: _build_chicken_sale_followup,
    PhaseKind.CHICKEN_SALE_REQUEST: _build_chicken_sale_request,
    PhaseKind.CHICKEN_SALE_EVENT: _build_chicken_sale_event,
    PhaseKind.RECORDED_TRANSITION: _build_recorded_transition,
    PhaseKind.CROSS_MAP: _build_cross_map,
    PhaseKind.SHOP_BUY: _build_shop_buy,
    PhaseKind.DIRECTIONAL_TRANSITION: _build_directional_transition,
    PhaseKind.MULTI_NAV: _build_multi_nav,
    PhaseKind.MOUNTAIN_BERRY: _build_mountain_berry,
    PhaseKind.ENSURE_TOOL: _build_ensure_tool,
    PhaseKind.ENSURE_ANIMAL_TOOLS: _build_ensure_animal_tools,
    PhaseKind.ENSURE_SEED: _build_ensure_seed,
    PhaseKind.DEADLINE: _build_deadline,
    PhaseKind.WAIT_UNTIL_TIME: _build_wait_until_time,
    PhaseKind.HARVEST: _build_harvest,
    PhaseKind.CLEAR_FIELD: _build_clear_field,
    PhaseKind.FENCE_CLEAR: _build_fence_clear,
    PhaseKind.COOP_CHORES: _build_coop_chores,
    PhaseKind.COW_CHORES: _build_cow_chores,
    PhaseKind.EVE_TALK_LOOP: _build_eve_talk_loop,
    PhaseKind.FLOWER_FESTIVAL: _build_flower_festival,
    PhaseKind.PROPOSE_EVE: _build_propose_eve,
    PhaseKind.CROP: _build_crop,
    PhaseKind.GRASS: _build_grass,
    PhaseKind.HOT_SPRING: _build_hot_spring,
    PhaseKind.RETURN_HOME: _build_return_home,
    PhaseKind.SLEEP: _build_sleep,
    PhaseKind.READY_TO_GO_HOME: _build_ready_to_go_home,
    PhaseKind.PROPOSE: _build_propose,
    PhaseKind.COURT: _build_marriage_skill,
    PhaseKind.FEATHER_BUY: _build_marriage_skill,
}


def build_phase_task(
    ctx: TaskBuildContext,
    spec: PhaseSpec,
    world: WorldState,
) -> Optional[Task]:
    """Create the sub-task for a phase spec, or None if unsupported."""
    if not isinstance(spec.kind, PhaseKind):
        return None
    builder = PHASE_TASK_BUILDERS.get(spec.kind)
    if builder is None:
        return None
    return builder(ctx, spec, world)
__all__ = [
    "PHASE_TASK_BUILDERS",
    "PhaseTaskBuilder",
    "TaskBuildContext",
    "build_phase_task",
]
