"""Registry mapping :class:`PhaseKind` to day-plan sub-task builders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

from retro_harness import Task, WorldState

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
from harvest.planner.tasks.home import GoToSleepTask, ReturnHomeTask
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
from harvest.tasks.farm_clear_task import FarmClearTask
from harvest.tasks.nav import Point
from harvest.core.tile_catalog import Tool
from harvest.tasks.harvest_task import HarvestTask, crop_nav_target_px, live_harvestable_crop_tiles
from harvest.tasks.berry_ship import BerryShipTask
from harvest.tasks.buy_seeds import BuySeedsTask
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
        task_name=spec.params.get("task_name", "buy_cow"),
        start_frame=spec.params.get("start_frame", 1631),
        end_frame=spec.params.get("end_frame", 2328),
        tasks_dir=ctx.tasks_dir,
    )


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


def _build_shop_buy(
    ctx: TaskBuildContext, spec: PhaseSpec, _world: WorldState
) -> Task:
    return BuySeedsTask(
        name=f"shop_buy_{spec.phase.lower()}",
        timeout=spec.params.get("timeout", 14_000),
        nav_timeout=spec.params.get("nav_timeout", 6_000),
        stock_field=spec.params.get("stock_field", "potato_seeds"),
    )


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
        )
    return MountainBerryTask(
        name=f"mountain_berry_{spec.phase.lower()}",
        timeout=spec.params.get("timeout", 12_000),
        nav_timeout=spec.params.get("nav_timeout", 6_000),
        pick_attempts=spec.params.get("pick_attempts", 0),
        approach_only=spec.params.get("approach_only", True),
    )


def _build_berry_ship(
    ctx: TaskBuildContext, spec: PhaseSpec, _world: WorldState
) -> Optional[Task]:
    route_name = spec.params.get("route", "")
    waypoints = ROUTES.get(route_name, [])
    if not waypoints:
        print(f"[DAY_PLAN] Unknown berry route: {route_name}")
        return None
    return BerryShipTask(
        name=f"berry_ship_{spec.phase.lower()}",
        waypoints=list(waypoints),
        timeout=spec.params.get("timeout", 18000),
        initial_settle_frames=spec.params.get("initial_settle_frames", 20),
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
) -> Task:
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
    return FenceClearLoopTask(
        name=f"fence_clear_{spec.phase.lower()}",
        max_fences=max_fences,
        corridor_only=bool(spec.params.get("corridor_only", True)),
        max_steps_per_fence=max_steps_per_fence,
        max_failures=int(spec.params.get("max_failures", 3)),
        pond_dump=bool(spec.params.get("pond_dump", False)),
        debris_types=debris_types,
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
        task_name=spec.params.get("task_name", "talk_eve_loop"),
        tasks_dir=ctx.tasks_dir,
        target_hearts=spec.params.get("target_hearts", 10),
        max_loops=spec.params.get("max_loops", 300),
        timeout=spec.params.get("timeout", 360000),
    )


def _build_crop(ctx: TaskBuildContext, spec: PhaseSpec, world: WorldState) -> Task:
    refill_bounds = spec.params.get("refill_bounds")
    work_mode = spec.params.get("work_mode", "full")
    # First plant: reactive 8-ring hoe+plant. D2 water is work_mode=pocket.
    if spec.phase == "CROP_ESTABLISH" or str(work_mode) == "establish":
        from harvest.tasks.skills import farm_pocket_plant_skill

        return farm_pocket_plant_skill(seed_type=ctx.seed_type, include_water=False)
    if str(work_mode) == "pocket":
        from harvest.tasks.skills import farm_pocket_water_skill

        return farm_pocket_water_skill()
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


PHASE_TASK_BUILDERS: dict[PhaseKind, PhaseTaskBuilder] = {
    PhaseKind.EXIT: _build_exit,
    PhaseKind.FARM_BUILDING_EXIT: _build_farm_building_exit,
    PhaseKind.FARM_EXIT: _build_farm_exit,
    PhaseKind.NAV: _build_nav,
    PhaseKind.RECORDED: _build_recorded,
    PhaseKind.RECORDED_SLICE: _build_recorded_slice,
    PhaseKind.COW_PURCHASE: _build_cow_purchase,
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
    PhaseKind.BERRY_SHIP: _build_berry_ship,
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
    PhaseKind.CROP: _build_crop,
    PhaseKind.HOT_SPRING: _build_hot_spring,
    PhaseKind.RETURN_HOME: _build_return_home,
    PhaseKind.SLEEP: _build_sleep,
    PhaseKind.READY_TO_GO_HOME: _build_ready_to_go_home,
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
