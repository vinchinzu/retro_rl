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
    ENTER_BARN_PHASE,
    EXIT_BARN_PHASE,
    NAV_TO_BARN_PHASE,
)
from harvest.planner.day_phase_calendar import (  # noqa: F401
    ATTEND_FESTIVAL_PHASE,
    REPAIR_CROPS_PHASE,
    SUNDAY_CHURCH_PHASE,
    SUNDAY_MOUNTAIN_PHASE,
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


LIVE_SHOP_SEED_TYPES = frozenset({"corn", "tomato"})


def pasture_day_phases(ram) -> List[PhaseSpec]:
    """Establish at least 32 pasture tiles; cut feed before purchases."""
    from harvest.core.ram_catalog import read_ram_value
    from harvest.core.tile_catalog import MATURE_GRASS_TILES
    from harvest.tasks.grass_planter import PASTURE_BOUNDS
    from harvest.tasks.nav import get_tile_at
    from harvest.planner.day_plan_status import SHOP_CLOSED_WEEKDAYS

    season = int(read_ram_value(ram, "season"))
    day = int(read_ram_value(ram, "day"))
    if season not in (1, 2):
        return []
    planted = int(read_ram_value(ram, "planted_grass"))
    if season == 1 and 2 <= day <= 20 and planted < 32:
        stock = int(read_ram_value(ram, "grass_seeds"))
        bags = (32 - planted + 8) // 9
        phases = []
        if stock < bags:
            if (int(read_ram_value(ram, "weekday")) in SHOP_CLOSED_WEEKDAYS
                    or int(read_ram_value(ram, "money")) < (bags - stock) * 500 + 300):
                # Existing stock can still make progress on closed-shop days.
                return [PhaseSpec("PLANT_PASTURE", "grass", failure_policy="optional")] if stock else []
            phases.append(PhaseSpec("BUY_GRASS_SEEDS", "shop_buy",
                          {"stock_field": "grass_seeds", "target_stock": bags},
                          failure_policy="optional"))
        phases.append(PhaseSpec("PLANT_PASTURE", "grass", failure_policy="optional"))
        return phases
    x0, y0, x1, y1 = PASTURE_BOUNDS
    if int(read_ram_value(ram, "stored_grass")) < 48 and any(
        get_tile_at(ram, x, y) in MATURE_GRASS_TILES
        for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)
    ):
        return [PhaseSpec("CUT_FODDER", "grass", {"cut_only": True}, failure_policy="optional")]
    return []


def summer_animal_species(ram, *, cows: bool = True, chickens: bool = True) -> str | None:
    """One cow, then one chicken, with pasture, hay and spending reserve."""
    from harvest.core.ram_catalog import read_ram_value
    from harvest.planner.day_plan_status import SHOP_CLOSED_WEEKDAYS
    from harvest.planner.tasks.sale_menu import purchase_block_reason, species_spec

    value = lambda key: int(read_ram_value(ram, key))
    if (value("season") != 1 or value("day") < 14
            or value("weekday") in SHOP_CLOSED_WEEKDAYS
            or value("planted_grass") < 32 or value("stored_grass") < 16):
        return None
    species = "cow" if cows and value("num_cows") == 0 else (
        "chicken" if chickens and value("num_cows") > 0 and value("num_chickens") == 0 else None
    )
    if species is None or purchase_block_reason(ram, species) is not None:
        return None
    return species if value("money") >= int(species_spec(species)["price"]) + 300 else None


def animal_purchase_phases(species: str) -> List[PhaseSpec]:
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
        "timeout": 9000,
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
    {"task_name": "talk_eve_loop", "target_hearts": 10},
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


def cow_day_phases(ram) -> List[PhaseSpec]:
    """Feed and talk even before the brush and milker have been bought."""
    from harvest.core.ram_catalog import read_ram_u8
    from harvest.core.carry import carry_pair_items

    carried = carry_pair_items(ram)
    shelf = read_ram_u8(ram, 0x11F01)
    brush = int(Tool.BRUSH) in carried or bool(shelf & 0x40)
    milker = int(Tool.MILKER) in carried or bool(shelf & 0x20)
    if brush and milker:
        return list(COW_PHASES)
    return [EXIT_TO_FARM_PHASE, NAV_TO_BARN_PHASE, ENTER_BARN_PHASE,
            PhaseSpec("COW_CHORES", "cow_chores",
                      {"talk": True, "brush": False, "milk": False, "feed": True},
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
        prefer_harvest_nav = bool(
            policy.include_harvest and has_harvest and not late_day and not plant_seeds
        )
        phases.extend(crop_water_phases(include_nav=not prefer_harvest_nav))
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
    "MOUNTAIN_BERRY_PHASE",
    "MOUNTAIN_BERRY_PHASES",
    "mountain_berry_count_for_day",
    "mountain_berry_phase",
    "NAV_CROP_PHASE",
    "HARVEST_ROUTE_PHASE",
    "DYNAMIC_OUTDOOR_PLAN_PHASE",
    "ENSURE_WATERING_CAN_PHASE",
    "ENSURE_CROP_SEEDS_PHASE",
    "ENSURE_ANIMAL_TOOLS_PHASE",
    "ENSURE_MILKER_PHASE",
    "CROP_ESTABLISH_PHASE",
    "CROP_WATER_PHASE",
    "HOT_SPRING_STAMINA_PHASE",
    "RETURN_HOME_PHASE",
    "GO_TO_SLEEP_PHASE",
    "EVE_TALK_LOOP_PHASE",
    "EVE_TALK_LOOP_PHASES",
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
