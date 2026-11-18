"""Static phase specs and named phase sequences.

Domain builders live in sibling modules and are re-exported here so existing
imports (`from harvest.planner.day_phase_catalog import …`) keep working:

- ``day_phase_berry`` — berry forage / ship
- ``day_phase_chicken`` — coop chores + chicken sale
- ``day_phase_cow`` — barn cow primitives
"""

from __future__ import annotations

from typing import Dict, List

from harvest.core.tile_catalog import Tool
from harvest.planner.day_plan_status import FARM_TILEMAP
from harvest.planner.day_phase_types import DayPlannerPolicy, PhaseSpec

# Domain modules (re-exported below; no cycle — they only import PhaseSpec).
from harvest.planner.day_phase_berry import (  # noqa: F401
    BERRY_CUTOFF_HOUR,
    MOUNTAIN_BERRY_PHASE,
    MOUNTAIN_BERRY_PHASES,
    OPTIONAL_BERRY_PHASES,
    mountain_berry_count_for_day,
    mountain_berry_phase,
)
from harvest.planner.day_phase_chicken import (  # noqa: F401
    CHICKEN_AFTER_BARN_PHASES,
    CHICKEN_PHASES,
    CHICKEN_SALE_BATCH_DROP_POINTS,
    COOP_CHORES_PHASE,
    COOP_CURRENT_PHASES,
    COOP_GIFT_PHASE,
    DROP_CHICKEN_FOR_SALE_PHASE,
    ENTER_COOP_FOR_CHICKEN_SALE_PHASE,
    ENTER_COOP_PHASE,
    EXIT_ANIMAL_SHOP_AFTER_CHICKEN_SALE_PHASE,
    EXIT_COOP_FOR_CHICKEN_SALE_PHASE,
    EXIT_COOP_PHASE,
    NAV_BARN_TO_COOP_PHASE,
    NAV_TO_ANIMAL_SHOP_FOR_CHICKEN_SALE_PHASE,
    NAV_TO_COOP_FOR_CHICKEN_SALE_PHASE,
    NAV_TO_COOP_PHASE,
    OPTIONAL_CHICKEN_SALE_PHASES,
    OPTIONAL_WIFE_GIFT_PHASES,
    PICKUP_CHICKEN_FOR_SALE_PHASE,
    REQUEST_CHICKEN_SALE_PHASE,
    RETURN_FARM_AFTER_CHICKEN_SALE_PHASE,
    SELL_CHICKEN_PHASE,
    SELL_CHICKEN_TEST_PHASES,
    SELL_THREE_CHICKENS_BATCH_TEST_PHASES,
    SELL_THREE_CHICKENS_TEST_PHASES,
    chicken_sale_batch_phases,
    chicken_sale_cycle_phases,
    chicken_sale_stage_phases,
)
from harvest.planner.day_phase_cow import (  # noqa: F401
    BARN_CURRENT_COW_PHASES,
    COW_CHORES_PHASE,
    COW_ROUNDUP_PHASE,
    ENTER_BARN_PHASE,
    EXIT_BARN_PHASE,
    NAV_TO_BARN_PHASE,
)
from harvest.planner.day_phase_calendar import (  # noqa: F401
    ATTEND_FESTIVAL_PHASE,
    REPAIR_CROPS_PHASE,
    SUNDAY_CHURCH_PHASE,
    SUNDAY_MOUNTAIN_PHASE,
    attend_festival_phase,
)

EXIT_HOUSE_PHASE = PhaseSpec(
    "EXIT_HOUSE",
    "exit",
    {"target_tilemap": 0x00, "dialog_frames": 120, "timeout": 900},
)

EXIT_TO_FARM_PHASE = PhaseSpec(
    "EXIT_TO_FARM",
    "farm_building_exit",
    required_ram=("tilemap", "player_x", "player_y", "input_lock"),
    estimated_frames=1200,
    failure_modes=("unknown_scene", "dialogue_stuck", "invalid_coords", "house_size_mismatch"),
)

LEAVE_HOUSE_TO_FARM_PHASE = PhaseSpec(
    "LEAVE_HOUSE_TO_FARM",
    "recorded_transition",
    {
        "task_name": "leave_house_to_farm",
        "origin_tilemap": 0x15,
        "target_tilemap": 0x00,
        "timeout": 1000,
        "min_frames_before_success": 30,
    },
)

NAV_FARM_EXIT_PHASE = PhaseSpec(
    "NAV_FARM_EXIT",
    "farm_exit",
    # West gate approach for town/shop. Timeout must cover south-farm → west
    # (post berry ship at bin) with BFS hops; 3k was a pure B-run thrash budget.
    {"target_px": (40, 424), "radius": 16, "timeout": 10000},
    failure_policy="optional",
)

LEAVE_FARM_WEST_PHASE = PhaseSpec(
    "LEAVE_FARM_WEST",
    "recorded_transition",
    {
        "task_name": "leave_farm_west",
        "origin_tilemap": 0x00,
        "target_tilemap": 0x0C,
        "timeout": 1200,
        "min_frames_before_success": 60,
    },
    failure_policy="optional",
)

EXIT_FARM_WEST_PHASE = PhaseSpec(
    "EXIT_FARM_WEST",
    "directional_transition",
    {
        "direction": "left",
        "origin_tilemap": FARM_TILEMAP,
        "target_tilemap": 0x0C,
        "timeout": 3000,
        "stand_tile": (0, 26),
        "stand_tolerance": 0,
        "min_frames_before_success": 15,
        "settle_frames": 45,
    },
    failure_policy="optional",
)

BUY_SEEDS_PHASE = PhaseSpec(
    "BUY_SEEDS",
    "shop_buy",
    {
        "stock_field": "potato_seeds",
        "timeout": 14000,
        "nav_timeout": 6000,
        # Kept for summer CrossMap fallback / tape analysis — not replayed.
        "recording_name": "buy_potato_seeds_d2",
    },
    failure_policy="optional",
)


LIVE_SHOP_SEED_TYPES = frozenset({"corn", "tomato", "turnip"})


def pasture_day_phases(ram, *, afternoon: bool = False) -> List[PhaseSpec]:
    """This morning's grass errand (:func:`livestock_econ.pasture_errand`) as phases.

    After noon a short herd still cuts; Fall also banks Winter's reserve. In y23_final the Y2
    Summer mornings went to the mountain wood trips, the outdoor plan expanded
    after 12:00 every day, and the hay hit 0 on D13 with 101 tiles of grass.

    A score run adds the whole-field cover bags (``field_cover``) to the same shop trip. The hoeing itself is the
    evening ``COVER_FIELD`` phase (:func:`field_cover_phases`), after the crops.
    """
    from harvest.core.ram_catalog import read_ram_value
    from harvest.planner import run_strategy
    from harvest.planner.field_cover import BAG_BUY_FRAMES, STACK_CAP, cover_buy_allowed, cover_errand_for
    from harvest.planner.livestock_econ import hay_cut_target, hay_wanted, pasture_errand, pasture_target_tiles

    errand = pasture_errand(ram)
    if afternoon:
        goal = hay_cut_target(ram) if int(read_ram_value(ram, "season")) == 2 else hay_wanted(ram)
        short = int(read_ram_value(ram, "stored_grass")) < goal
        errand = type(errand)(cut=errand.cut and short)
    stock = int(read_ram_value(ram, "grass_seeds"))
    herd_stock = errand.buy_to_stock if errand.buy_to_stock else stock
    buy_to = errand.buy_to_stock
    cover_buy = False
    if run_strategy.field_cover() and int(read_ram_value(ram, "hour")) < 15:
        buying = cover_buy_allowed(ram)
        cover = cover_errand_for(ram, herd_stock, errand.plant, buying=buying)
        print(f"[COVER] morning shop={'open' if buying else 'shut'} buy={cover.buy} bags={cover.bags} "
              f"behind={cover.behind} money={int(read_ram_value(ram, 'money'))} reason={cover.reason}")
        if cover.buy > 0:
            buy_to = min(STACK_CAP, herd_stock + cover.buy)      # the clerk charges for bags past the u8 stack
            cover_buy = True
    phases: List[PhaseSpec] = []
    if errand.cut:
        params = {"cut_only": True, "hay_goal": hay_cut_target(ram)}
        if run_strategy.field_cover():
            params["bounds"] = (2, 3, 62, 60)
        phases.append(PhaseSpec("CUT_FODDER", "grass", params, failure_policy="optional"))
    if buy_to:
        # The clerk loop is ~650 frames a bag (21 bags hit the old 14,000f cap); the shop clock is frozen.
        timeout = max(14_000, 4_000 + BAG_BUY_FRAMES * (buy_to - stock))
        buy = PhaseSpec(
            "BUY_GRASS_SEEDS", "shop_buy",
            {"stock_field": "grass_seeds", "target_stock": buy_to, "timeout": timeout,
             **({"latest_start_hour": 15} if cover_buy else {})},
            failure_policy="optional",
        )
        # Feeding already claimed the morning. With hay left, shop before an optional top-up cut;
        # the cut can continue after 18:00. An empty loft keeps the cut first.
        phases.insert(0 if cover_buy and int(read_ram_value(ram, "stored_grass")) > 0 else len(phases), buy)
    if errand.plant:
        # The planter stops at its own target. Without this one it stops at 32,
        # so every bag past the cow gate sat in stock while the phase "succeeded".
        phases.append(PhaseSpec(
            "PLANT_PASTURE", "grass", {"target_tiles": pasture_target_tiles(ram)}, failure_policy="optional",
        ))
    return phases


def animal_purchase_phases(species: str) -> List[PhaseSpec]:
    """Counter trip. A hen is then carried into the coop (``HEN_DELIVERY``), fed, and the coop left."""
    if species in ("chicken", "hen_pickup"):
        trip = [PhaseSpec("NAV_TO_ANIMAL_SHOP", "multi_nav",
                          {"route": "farm_to_animal_shop_counter_sale", "timeout": 7000},
                          failure_policy="optional")]
        if species == "chicken":
            trip.append(PhaseSpec("BUY_COW_VENDOR", "cow_purchase", {"species": species},
                                  failure_policy="optional"))
        # Delivery ends inside the coop. Feed her tonight: an unfed night costs three laying days.
        return trip + [
            PhaseSpec("HEN_DELIVERY", "hen_delivery", failure_policy="optional"),
            PhaseSpec("COOP_CHORES", "coop_chores", {"egg_mode": "ship"}, failure_policy="optional",
                      required_maps=(0x28,)),
            PhaseSpec("EXIT_COOP", "directional_transition", dict(EXIT_COOP_PHASE.params),
                      failure_policy="optional"),
        ]
    return [
        PhaseSpec("NAV_TO_ANIMAL_SHOP", "multi_nav",
                  {"route": "farm_to_animal_shop_counter_sale", "timeout": 7000},
                  failure_policy="optional"),
        PhaseSpec("BUY_COW_VENDOR", "cow_purchase", {"species": species},
                  failure_policy="optional"),
        PhaseSpec("EXIT_ANIMAL_SHOP", "multi_nav",
                  {"route": "animal_shop_to_town", "timeout": 3000}, failure_policy="optional"),
        PhaseSpec("RETURN_FARM_AFTER_COW_PURCHASE", "multi_nav",
                  {"route": "animal_shop_town_to_farm", "timeout": 7000}, failure_policy="optional"),
    ]


def buy_seeds_phase(
    *,
    recording_name: str | None = None,
    recording_start: int | None = None,
    seed_type: str | None = None,
) -> PhaseSpec:
    """Build a BUY_SEEDS phase for the active seasonal shop recording."""
    params = dict(BUY_SEEDS_PHASE.params)
    if seed_type is not None:
        params["seed_type"] = seed_type
    if seed_type in LIVE_SHOP_SEED_TYPES:
        # Summer corn/tomato: the same RAM-closed shop buy as potato, at the
        # stand under that shelf bag. The recorded CrossMap tape only ever
        # ended in the shop and never bought.
        params["stock_field"] = f"{seed_type}_seeds"
        return PhaseSpec(
            BUY_SEEDS_PHASE.phase,
            BUY_SEEDS_PHASE.kind,
            params,
            failure_policy=BUY_SEEDS_PHASE.failure_policy,
        )
    if recording_name:
        params["recording_name"] = recording_name
        # Summer still uses the recorded CrossMap until a nav+RAM buy lands.
        if recording_start is None and recording_name == "buy_summer":
            params["recording_start"] = 0
            params["exit_direction"] = "left"
            params["origin_tilemap"] = 0x00
            params["timeout"] = 5000
            params["continue_after_return"] = 200
            return PhaseSpec(
                BUY_SEEDS_PHASE.phase,
                "cross_map",
                params,
                failure_policy=BUY_SEEDS_PHASE.failure_policy,
            )
        if recording_start is not None:
            params["recording_start"] = recording_start
    elif recording_start is not None:
        params["recording_start"] = recording_start
    return PhaseSpec(
        BUY_SEEDS_PHASE.phase,
        BUY_SEEDS_PHASE.kind,
        params,
        failure_policy=BUY_SEEDS_PHASE.failure_policy,
    )

NAV_CROP_PHASE = PhaseSpec(
    "NAV_CROP",
    "nav",
    # Soft radius keeps planting from failing when debris blocks the last
    # few pixels; CropWaterTask re-homes onto exact plot centers.
    # Optional: crop phase navigates itself if this pre-nav times out.
    # Virgin plant days: land at preferred field (tile 15,29), not ship area —
    # ship-area start caused plot plans at unreachable southern centers.
    {
        "target_px": (248, 472),
        "radius": 28,
        "soft_radius": 64,
        # A real NAV_CROP is 60-700 f (rec1: median 161); the 9,000 f it had let a wedge burn 150 s or a day.
        "timeout": 2400,
    },
    failure_policy="optional",
    required_maps=(0x00,),
    estimated_frames=4000,
    failure_modes=("no_path", "debris_block", "viewport_bfs_stale"),
)

HARVEST_ROUTE_PHASE = PhaseSpec(
    "HARVEST_ROUTE",
    "harvest",
    failure_policy="optional",
    required_maps=(0x00,),
    required_ram=("tilemap", "money"),
    estimated_frames=6000,
    failure_modes=("no_mature", "bin_path_fail", "ship_money_not_instant"),
)

CLEAR_FIELD_PHASE = PhaseSpec(
    "CLEAR_FIELD",
    "clear_field",
    # Day clear is for crop-path keep-alive only. Empty early-spring days defer
    # bush/weed lift thrash to the evening after the 5pm shipping window.
    {"timeout": 3500},
    failure_policy="optional",
    required_maps=(0x00,),
    estimated_frames=3500,
    failure_modes=("timeout_budget", "tool_missing", "stamina_low"),
)

CLEAR_BUSHES_PHASE = PhaseSpec(
    "CLEAR_BUSHES",
    "clear_field",
    {
        "timeout": 12000,
        "priority": ["weed"],
        "handoff": "type_clear",
        "fetch_tools": False,
        "prefer_lift_for_weeds": True,
        "prefer_lift_for_stones": False,
    },
    failure_policy="optional",
    required_maps=(0x00,),
    estimated_frames=8000,
    failure_modes=("timeout_budget",),
)

TILL_FIELD_PHASE = PhaseSpec(
    "TILL_FIELD",
    "grass",
    {
        "till_only": True,
        "till_limit": 32,
        "timeout": 12000,
    },
    failure_policy="optional",
    required_maps=(0x00,),
    estimated_frames=12000,
    failure_modes=("timeout_budget", "stamina_low"),
)

DYNAMIC_OUTDOOR_PLAN_PHASE = PhaseSpec(
    "DYNAMIC_OUTDOOR_PLAN",
    "dynamic_outdoor_plan",
)

ENSURE_WATERING_CAN_PHASE = PhaseSpec(
    "ENSURE_WATERING_CAN",
    "ensure_tool",
    {"tool_id": int(Tool.WATERING_CAN)},
    # Optional: shelf miss must not recovery-loop the day (rr-6byj pattern).
    failure_policy="optional",
    required_tools=("watering_can",),
    estimated_frames=2500,
    failure_modes=("shelf_miss", "carry_full", "wrong_house_size"),
)

ENSURE_CROP_SEEDS_PHASE = PhaseSpec(
    "ENSURE_CROP_SEEDS",
    "ensure_seed",
    # Optional so a shed miss soft-skips plant phases instead of required
    # recovery-retry hanging multi-day (rr-6byj multi_nav 1-wp loop).
    failure_policy="optional",
    required_tools=("seed",),
    estimated_frames=2500,
    failure_modes=("bag_on_shelf_not_carry", "stock_zero", "carry_swap_lost_seed"),
)

ENSURE_ANIMAL_TOOLS_PHASE = PhaseSpec(
    "ENSURE_ANIMAL_TOOLS",
    "ensure_animal_tools",
    {
        "first_tool_id": int(Tool.BRUSH),
        "second_tool_id": int(Tool.MILKER),
    },
)

ENSURE_BRUSH_PHASE = PhaseSpec(
    "ENSURE_BRUSH",
    "ensure_tool",
    {"tool_id": int(Tool.BRUSH)},
    failure_policy="optional",
)

ENSURE_MILKER_PHASE = PhaseSpec(
    "ENSURE_MILKER",
    "ensure_tool",
    {"tool_id": int(Tool.MILKER)},
)

# Plant pass: hoe + seed only (two-slot carry holds seeds+hoe).
CROP_ESTABLISH_PHASE = PhaseSpec(
    "CROP_ESTABLISH",
    "crop",
    # Refill unused in establish; keep broad bounds for any residual water.
    {"work_mode": "establish", "refill_bounds": (3, 14, 62, 60)},
    required_maps=(0x00,),
    required_tools=("hoe", "seed"),
    estimated_frames=8000,
    failure_modes=("unreachable_center", "seed_not_in_carry", "no_path_to_hoe"),
)

# Water pass: watering can only; waters already-established crops.
# Include north F9 spur (y~12–13 CheckToolSuccess fill) as well as mid pond
# and south FC. y_min=14 excluded F9 on dry fixture so empty-can always
# fell through to y=31 fence thrash.
CROP_WATER_PHASE = PhaseSpec(
    "CROP_WATER",
    "crop",
    {"work_mode": "water", "refill_bounds": (3, 10, 62, 60)},
    required_maps=(0x00,),
    required_tools=("watering_can",),
    estimated_frames=6000,
    failure_modes=("empty_can", "refill_fail", "no_plots", "precheck_tool_success"),
)

# Optional stamina refill at outdoor mountain hot spring (tilemap 0x10 pond).
# min_stamina "full" = soak until current == max, then return to farm.
HOT_SPRING_STAMINA_PHASE = PhaseSpec(
    "HOT_SPRING_STAMINA",
    "hot_spring",
    {"min_stamina": "full", "return_to_farm": True},
    failure_policy="optional",
    required_ram=("stamina", "tilemap"),
    estimated_frames=12000,
    failure_modes=("nav_fail", "bath_not_entered", "corridor_debris"),
)

RETURN_HOME_PHASE = PhaseSpec(
    "RETURN_HOME",
    "return_home",
    required_ram=("tilemap", "player_x", "player_y"),
    estimated_frames=4000,
    failure_modes=("path_fail", "door_held_item"),
)

GO_TO_SLEEP_PHASE = PhaseSpec(
    "GO_TO_SLEEP",
    "sleep",
    required_ram=("tilemap", "hour", "day"),
    estimated_frames=2500,
    failure_modes=("bed_miss", "scene_wake_stuck", "return_home_first"),
)

# Town loop whose success sets the planner "ready to go home" flag.
TOWN_EXPLORE_PHASE = PhaseSpec(
    "TOWN_EXPLORE",
    "multi_nav",
    {"route": "town_explore", "timeout": 10000, "initial_settle_frames": 30},
    failure_policy="optional",
)

# Flag phase: successful town work (or explicit mark) appends end-of-day.
READY_TO_GO_HOME_PHASE = PhaseSpec(
    "READY_TO_GO_HOME",
    "ready_to_go_home",
)

# Chained startup macros for early Spring mornings (tools from shed).
GET_HAMMER_MACRO_PHASE = PhaseSpec(
    "GET_HAMMER",
    "recorded",
    {"task_name": "get_hammer"},
    failure_policy="optional",
)
GET_AXE_MACRO_PHASE = PhaseSpec(
    "GET_AXE",
    "recorded",
    {"task_name": "get_axe"},
    failure_policy="optional",
)
GET_SICKLE_MACRO_PHASE = PhaseSpec(
    "GET_SICKLE",
    "recorded",
    {"task_name": "get_sickle"},
    failure_policy="optional",
)
LEAVE_HOUSE_MACRO_PHASE = PhaseSpec(
    "LEAVE_HOUSE_MACRO",
    "recorded_transition",
    {
        "task_name": "leave_house_to_farm",
        "origin_tilemap": 0x15,
        "target_tilemap": 0x00,
        "timeout": 1000,
        "min_frames_before_success": 30,
    },
    failure_policy="optional",
)

# ── Eve relationship loop phase ──

EVE_TALK_LOOP_PHASE = PhaseSpec(
    "EVE_TALK_LOOP",
    "eve_talk_loop",
    {"target_hearts": 10},
)

# Composed cow sequences (need EXIT_TO_FARM / ENSURE_ANIMAL_TOOLS from this module).
COW_PHASES: List[PhaseSpec] = [
    ENSURE_ANIMAL_TOOLS_PHASE,
    EXIT_TO_FARM_PHASE,
    NAV_TO_BARN_PHASE,
    ENTER_BARN_PHASE,
    COW_CHORES_PHASE,
    EXIT_BARN_PHASE,
]


def tool_purchase_phases(items=None) -> List[PhaseSpec]:
    """One general-store trip (farm -> 0x22 -> farm) buying each unowned affordable tool.

    ``items`` are Tool ids or names (``"brush"``, ``"milker"``); default is both
    cow tools, brush first. The task skips owned tools and ones the wallet
    cannot cover at its turn, so the caller only gates on ``cow_tool_trip_affordable``,
    the money it wants to keep, and ``tool_shop_open`` (both in ``livestock_econ``).
    """
    names = {"brush": int(Tool.BRUSH), "milker": int(Tool.MILKER), "sprinkler": int(Tool.SPRINKLER)}
    ids = [names[i.lower()] if isinstance(i, str) else int(i)
           for i in (items if items is not None else ("brush", "milker"))]
    return [PhaseSpec(
        "BUY_COW_TOOLS", "shop_buy",
        {"tool_ids": ids, "timeout": 16000, "nav_timeout": 6000},
        failure_policy="optional",
    )]


def cow_day_phases(ram) -> List[PhaseSpec]:
    """Feed and talk even before the brush and milker have been bought."""
    from harvest.core.animal_status import cows_outside
    from harvest.core.ram_catalog import read_ram_value
    from harvest.planner.livestock_econ import missing_cow_tools

    missing = missing_cow_tools(ram)
    roundup = [COW_ROUNDUP_PHASE] if cows_outside(ram) else []
    if not missing:
        tools, leave, *barn = COW_PHASES
        return [tools, leave, *roundup, *barn]
    # The brush alone is worth +3 happiness a day on a calf, so fetch it
    # without the milker. An empty loft cannot set a trough bit: feed only
    # when hay exists.
    brush = int(Tool.BRUSH) not in missing
    feed = int(read_ram_value(ram, "stored_grass")) > 0
    return [*([ENSURE_BRUSH_PHASE] if brush else []),
            EXIT_TO_FARM_PHASE, *roundup, NAV_TO_BARN_PHASE, ENTER_BARN_PHASE,
            PhaseSpec("COW_CHORES", "cow_chores",
                      {"talk": True, "brush": brush, "milk": False, "feed": feed},
                      required_maps=(0x27,)), EXIT_BARN_PHASE]

BUY_COW_PHASES: List[PhaseSpec] = [
    EXIT_TO_FARM_PHASE,
    *animal_purchase_phases("cow"),
    ENSURE_ANIMAL_TOOLS_PHASE,
    NAV_TO_BARN_PHASE,
    ENTER_BARN_PHASE,
    COW_CHORES_PHASE,
]

BUY_COW_FIRST_PHASES: List[PhaseSpec] = [
    *BUY_COW_PHASES,
    EXIT_BARN_PHASE,
]

# Day 1 sequence: house exit → light clear → buy seeds → hoe/plant → water →
# town explore (sets go-home flag) → return home → sleep into day 2.
# Plant pass uses hoe+seeds first (only 2 carry slots); can comes after.
# Keep in sync with crop_establish_phases() / crop_water_phases() in
# day_plan_phases.py (catalog cannot import that module without a cycle).
DAY1_PHASES: List[PhaseSpec] = [
    EXIT_TO_FARM_PHASE,
    CLEAR_FIELD_PHASE,
    NAV_FARM_EXIT_PHASE,
    BUY_SEEDS_PHASE,
    ENSURE_CROP_SEEDS_PHASE,
    NAV_CROP_PHASE,
    CROP_ESTABLISH_PHASE,
    ENSURE_WATERING_CAN_PHASE,
    NAV_CROP_PHASE,
    CROP_WATER_PHASE,
    TOWN_EXPLORE_PHASE,
    READY_TO_GO_HOME_PHASE,
    RETURN_HOME_PHASE,
    GO_TO_SLEEP_PHASE,
]

# Boot / morning → next calendar day. Prefers reusable recorded macros where
# they exist, then autonomous return-home + sleep (always finds house).
BOOT_TO_DAY2_PHASES: List[PhaseSpec] = [
    EXIT_TO_FARM_PHASE,
    GET_HAMMER_MACRO_PHASE,
    GET_AXE_MACRO_PHASE,
    GET_SICKLE_MACRO_PHASE,
    CLEAR_FIELD_PHASE,
    NAV_FARM_EXIT_PHASE,
    BUY_SEEDS_PHASE,
    ENSURE_CROP_SEEDS_PHASE,
    NAV_CROP_PHASE,
    CROP_ESTABLISH_PHASE,
    ENSURE_WATERING_CAN_PHASE,
    NAV_CROP_PHASE,
    CROP_WATER_PHASE,
    TOWN_EXPLORE_PHASE,
    READY_TO_GO_HOME_PHASE,
    RETURN_HOME_PHASE,
    GO_TO_SLEEP_PHASE,
]

# Focused field-clear run (tools + wipe debris until stamina/clean).
CLEAR_PHASES: List[PhaseSpec] = [
    EXIT_TO_FARM_PHASE,
    CLEAR_FIELD_PHASE,
]

# Spring Day 4 named sequence: mountain grape then plant + water.
SPRING4_PHASES: List[PhaseSpec] = [
    EXIT_TO_FARM_PHASE,
    mountain_berry_phase(count=2),
    NAV_FARM_EXIT_PHASE,
    buy_seeds_phase(),
    ENSURE_CROP_SEEDS_PHASE,
    NAV_CROP_PHASE,
    CROP_ESTABLISH_PHASE,
    ENSURE_WATERING_CAN_PHASE,
    NAV_CROP_PHASE,
    CROP_WATER_PHASE,
]

# Seed stock already exists: keep this legacy named sequence safe while the
# recorded berry route is disabled.
BERRIES_WATER_PHASES: List[PhaseSpec] = [
    EXIT_TO_FARM_PHASE,
    ENSURE_WATERING_CAN_PHASE,
    NAV_CROP_PHASE,
    CROP_WATER_PHASE,
]

# Sunday: shop is closed. Water only while the recorded berry route is disabled.
SUNDAY_PHASES: List[PhaseSpec] = [
    EXIT_TO_FARM_PHASE,
    ENSURE_WATERING_CAN_PHASE,
    NAV_CROP_PHASE,
    CROP_WATER_PHASE,
]

# Resume from an in-house save and go directly to the field for watering.
RESUME_WATER_PHASES: List[PhaseSpec] = [
    EXIT_TO_FARM_PHASE,
    ENSURE_WATERING_CAN_PHASE,
    NAV_CROP_PHASE,
    CROP_WATER_PHASE,
]

HARVEST_PHASES: List[PhaseSpec] = [
    EXIT_TO_FARM_PHASE,
    NAV_CROP_PHASE,
    HARVEST_ROUTE_PHASE,
]

EVE_TALK_LOOP_PHASES: List[PhaseSpec] = [
    EVE_TALK_LOOP_PHASE,
]

FLOWER_FESTIVAL_PHASES: List[PhaseSpec] = [attend_festival_phase("spring", 23)]

PROPOSE_EVE_PHASES: List[PhaseSpec] = [
    PhaseSpec(
        "PROPOSE_EVE",
        "propose_eve",
        {"ends_day": True},  # the wedding rolls the day
        failure_policy="optional",
    )
]

PHASE_SEQUENCES: Dict[str, List[PhaseSpec]] = {
    "day1": DAY1_PHASES,
    "boot_to_day2": BOOT_TO_DAY2_PHASES,
    "clear": CLEAR_PHASES,
    "spring4": SPRING4_PHASES,
    "berries_water": BERRIES_WATER_PHASES,
    "sunday": SUNDAY_PHASES,
    "resume_water": RESUME_WATER_PHASES,
    "harvest": HARVEST_PHASES,
    "buy_cow": BUY_COW_PHASES,
    "pasture": [
        EXIT_TO_FARM_PHASE,
        PhaseSpec("BUY_GRASS_SEEDS", "shop_buy",
                  {"stock_field": "grass_seeds", "target_stock": 4}),
        PhaseSpec("PLANT_PASTURE", "grass"),
    ],
    "mushroom_gather": [
        EXIT_TO_FARM_PHASE,
        PhaseSpec("MUSHROOM_GATHER", "mountain_berry", {"ship": True, "forage": "mushroom"}),
    ],
    "buy_cow_counter": [
        EXIT_TO_FARM_PHASE,
        PhaseSpec(
            "NAV_ANIMAL_COUNTER",
            "multi_nav",
            {"route": "farm_to_animal_shop_counter_sale", "timeout": 45000},
        ),
        PhaseSpec("BUY_COW_AT_COUNTER", "cow_purchase", {"species": "cow", "timeout": 3600}),
    ],
    "buy_chicken_counter": [
        EXIT_TO_FARM_PHASE,
        PhaseSpec(
            "NAV_ANIMAL_COUNTER",
            "multi_nav",
            {"route": "farm_to_animal_shop_counter_sale", "timeout": 45000},
        ),
        PhaseSpec(
            "BUY_CHICKEN_AT_COUNTER",
            "cow_purchase",
            {"species": "chicken", "timeout": 3600},
        ),
    ],
    "sell_chicken_test": SELL_CHICKEN_TEST_PHASES,
    "sell_three_chickens_test": SELL_THREE_CHICKENS_TEST_PHASES,
    "sell_three_chickens_batch_test": SELL_THREE_CHICKENS_BATCH_TEST_PHASES,
    "eve_loop": EVE_TALK_LOOP_PHASES,
    "flower_festival": FLOWER_FESTIVAL_PHASES,
    "propose_eve": PROPOSE_EVE_PHASES,
    "mountain_berry": MOUNTAIN_BERRY_PHASES,
}

# Backwards compat alias
PHASE_SEQUENCE = DAY1_PHASES


# ── Dynamic day plan constants ───────────────────────────────────
#
# Priority order (documented for orchestrator / day_plan_phases consumers):
#   1. Exit building (always)
#   2. Buy the first cow when affordable
#   3. Barn cow chores (if cows need feed/attention)
#   4. Chicken coop chores (if chickens > 0)
#   5. Harvest ripe crops (if any)
#   6. Ensure watering can/seeds + crop work
#   7. Berry/shop money route (if still early enough for round trip)

OPTIONAL_SHOP_PHASES = frozenset({
    "BUY_SEEDS_WINDOW",
    "NAV_FARM_EXIT",
    "BUY_SEEDS",
    "BUY_COW_TOOLS",
    "TOWN_EXPLORE",
    "GET_HAMMER",
    "GET_AXE",
    "GET_SICKLE",
    "LEAVE_HOUSE_MACRO",
})
# Buying a cow is discretionary; farming is not. run16 reached the cow
# threshold on D26 for the first time (~$5000) and the 12-phase purchase day
# replaced the whole income plan with *required* phases — NAV_TO_ANIMAL_SHOP
# failed, so D26 and D27 both earned nothing and D28 died in return_home.
# Failing this route must defer it and fall through to the farm work.
OPTIONAL_COW_PURCHASE_PHASES = frozenset({
    "NAV_TO_ANIMAL_SHOP",
    "BUY_COW_VENDOR",
    "EXIT_ANIMAL_SHOP",
    "RETURN_FARM_AFTER_COW_PURCHASE",
    # The barn tail belongs to the same group: skipping only the purchase
    # would leave these as required phases for a cow that was never bought,
    # and the day would die one phase later instead.
    "ENSURE_ANIMAL_TOOLS",
    "ENSURE_BRUSH",
    "NAV_TO_BARN",
    "ENTER_BARN",
    "COW_CHORES",
    "EXIT_BARN",
})
OPTIONAL_MONEY_PHASES = (
    OPTIONAL_BERRY_PHASES
    | OPTIONAL_SHOP_PHASES
    | OPTIONAL_CHICKEN_SALE_PHASES
    | OPTIONAL_COW_PURCHASE_PHASES
    | OPTIONAL_WIFE_GIFT_PHASES
)

# Phases whose success marks the day ready for return-home/sleep.
GO_HOME_TRIGGER_PHASES = frozenset({
    "TOWN_EXPLORE",
    "READY_TO_GO_HOME",
    "BUY_SEEDS",
})


def crop_establish_phases() -> List[PhaseSpec]:
    """Hoe + plant pass: ensure seeds/hoe, walk to field, establish plots."""
    return [
        ENSURE_CROP_SEEDS_PHASE,
        NAV_CROP_PHASE,
        CROP_ESTABLISH_PHASE,
    ]


def pocket_clear_phase() -> PhaseSpec:
    """Lift weeds/stones inside the y=31 fence so hoe stands have a BFS path."""
    from harvest.maps.map_config import WEST_PLANT_POCKET_BOUNDS

    return PhaseSpec(
        "CLEAR_PLOT",
        "clear_field",
        {
            "timeout": 7000,
            "fetch_tools": False,
            "prefer_lift_for_weeds": True,
            "prefer_lift_for_stones": True,
            "farm_bounds": WEST_PLANT_POCKET_BOUNDS,
            "priority": ["weed", "stone"],
        },
        failure_policy="optional",
        required_maps=(0x00,),
        estimated_frames=5000,
        failure_modes=("timeout_budget", "pocket_sealed"),
    )


def pocket_plant_phases() -> List[PhaseSpec]:
    """Shed hoe+seeds first, then pocket clear → hoe/plant → can/water."""
    from harvest.planner.d2_work import d2_post_shop_work_phases

    return d2_post_shop_work_phases()


def crop_water_phases(*, include_nav: bool = True) -> List[PhaseSpec]:
    """Water pass: ensure can, optional field nav, water established crops."""
    phases: List[PhaseSpec] = [ENSURE_WATERING_CAN_PHASE]
    if include_nav:
        phases.append(NAV_CROP_PHASE)
    phases.append(CROP_WATER_PHASE)
    return phases


def _crop_work_phases(
    *,
    has_harvest: bool,
    has_waterable: bool,
    has_seeds: bool,
    is_rainy: bool,
    late_day: bool,
    policy: DayPlannerPolicy,
) -> List[PhaseSpec]:
    plant_seeds = bool(has_seeds and policy.include_planting and not late_day)
    needs_manual_water = bool(not is_rainy and (has_waterable or plant_seeds))
    needs_crop_phase = plant_seeds or needs_manual_water
    if late_day:
        needs_crop_phase = bool(has_waterable and not is_rainy)
        plant_seeds = False
        needs_manual_water = bool(has_waterable and not is_rainy)
    if not policy.include_watering or not needs_crop_phase:
        return []

    phases: List[PhaseSpec] = []
    if plant_seeds:
        phases.extend(crop_establish_phases())
    if needs_manual_water:
        # Each watering pair owns its stand approach. The fixed field pre-nav
        # can be sealed by grown crops and waste 9,000f before the same work.
        phases.extend(crop_water_phases(include_nav=False))
    return phases


__all__ = [
    "EXIT_HOUSE_PHASE",
    "EXIT_TO_FARM_PHASE",
    "LEAVE_HOUSE_TO_FARM_PHASE",
    "NAV_FARM_EXIT_PHASE",
    "LEAVE_FARM_WEST_PHASE",
    "EXIT_FARM_WEST_PHASE",
    "BUY_SEEDS_PHASE",
    "buy_seeds_phase",
    "tool_purchase_phases",
    "MOUNTAIN_BERRY_PHASE",
    "MOUNTAIN_BERRY_PHASES",
    "mountain_berry_count_for_day",
    "mountain_berry_phase",
    "NAV_CROP_PHASE",
    "HARVEST_ROUTE_PHASE",
    "CLEAR_BUSHES_PHASE",
    "TILL_FIELD_PHASE",
    "DYNAMIC_OUTDOOR_PLAN_PHASE",
    "ENSURE_WATERING_CAN_PHASE",
    "ENSURE_CROP_SEEDS_PHASE",
    "ENSURE_ANIMAL_TOOLS_PHASE",
    "ENSURE_BRUSH_PHASE",
    "ENSURE_MILKER_PHASE",
    "CROP_ESTABLISH_PHASE",
    "CROP_WATER_PHASE",
    "HOT_SPRING_STAMINA_PHASE",
    "RETURN_HOME_PHASE",
    "GO_TO_SLEEP_PHASE",
    "EVE_TALK_LOOP_PHASE",
    "EVE_TALK_LOOP_PHASES",
    "FLOWER_FESTIVAL_PHASES",
    "PROPOSE_EVE_PHASES",
    "NAV_TO_COOP_PHASE",
    "NAV_BARN_TO_COOP_PHASE",
    "ENTER_COOP_PHASE",
    "COOP_CHORES_PHASE",
    "COOP_GIFT_PHASE",
    "EXIT_COOP_PHASE",
    "NAV_TO_COOP_FOR_CHICKEN_SALE_PHASE",
    "ENTER_COOP_FOR_CHICKEN_SALE_PHASE",
    "PICKUP_CHICKEN_FOR_SALE_PHASE",
    "EXIT_COOP_FOR_CHICKEN_SALE_PHASE",
    "DROP_CHICKEN_FOR_SALE_PHASE",
    "NAV_TO_ANIMAL_SHOP_FOR_CHICKEN_SALE_PHASE",
    "REQUEST_CHICKEN_SALE_PHASE",
    "EXIT_ANIMAL_SHOP_AFTER_CHICKEN_SALE_PHASE",
    "RETURN_FARM_AFTER_CHICKEN_SALE_PHASE",
    "SELL_CHICKEN_PHASE",
    "CHICKEN_SALE_BATCH_DROP_POINTS",
    "chicken_sale_stage_phases",
    "chicken_sale_cycle_phases",
    "chicken_sale_batch_phases",
    "CHICKEN_PHASES",
    "CHICKEN_AFTER_BARN_PHASES",
    "COOP_CURRENT_PHASES",
    "NAV_TO_BARN_PHASE",
    "ENTER_BARN_PHASE",
    "COW_CHORES_PHASE",
    "EXIT_BARN_PHASE",
    "COW_PHASES",
    "BARN_CURRENT_COW_PHASES",
    "BUY_COW_PHASES",
    "BUY_COW_FIRST_PHASES",
    "CLEAR_FIELD_PHASE",
    "CLEAR_PHASES",
    "DAY1_PHASES",
    "BOOT_TO_DAY2_PHASES",
    "TOWN_EXPLORE_PHASE",
    "READY_TO_GO_HOME_PHASE",
    "GET_HAMMER_MACRO_PHASE",
    "GET_AXE_MACRO_PHASE",
    "GET_SICKLE_MACRO_PHASE",
    "LEAVE_HOUSE_MACRO_PHASE",
    "SPRING4_PHASES",
    "BERRIES_WATER_PHASES",
    "SUNDAY_PHASES",
    "RESUME_WATER_PHASES",
    "HARVEST_PHASES",
    "SELL_CHICKEN_TEST_PHASES",
    "SELL_THREE_CHICKENS_TEST_PHASES",
    "SELL_THREE_CHICKENS_BATCH_TEST_PHASES",
    "PHASE_SEQUENCES",
    "PHASE_SEQUENCE",
    "BERRY_CUTOFF_HOUR",
    "OPTIONAL_BERRY_PHASES",
    "OPTIONAL_SHOP_PHASES",
    "OPTIONAL_CHICKEN_SALE_PHASES",
    "OPTIONAL_COW_PURCHASE_PHASES",
    "OPTIONAL_MONEY_PHASES",
    "GO_HOME_TRIGGER_PHASES",
    "crop_establish_phases",
    "crop_water_phases",
    "pocket_clear_phase",
    "pocket_plant_phases",
    "_crop_work_phases",
    "ATTEND_FESTIVAL_PHASE",
    "SUNDAY_CHURCH_PHASE",
    "SUNDAY_MOUNTAIN_PHASE",
    "REPAIR_CROPS_PHASE",
]
