"""Static phase specs and named phase sequences."""

from __future__ import annotations

from typing import Dict, List

from harvest.tasks.farm_clearer import Tool
from harvest.planner.day_plan_status import FARM_TILEMAP
from harvest.planner.day_phase_types import PhaseSpec

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
    {"target_px": (40, 424), "radius": 12, "timeout": 3000},
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
    "cross_map",
    {
        "exit_direction": "left",
        "recording_name": "buy_potato_seeds",
        "recording_start": 483,
        "origin_tilemap": 0x00,
        "timeout": 5000,
        "continue_after_return": 200,
    },
    failure_policy="optional",
)


def buy_seeds_phase(
    *,
    recording_name: str | None = None,
    recording_start: int | None = None,
) -> PhaseSpec:
    """Build a BUY_SEEDS phase for the active seasonal shop recording."""
    params = dict(BUY_SEEDS_PHASE.params)
    if recording_name:
        params["recording_name"] = recording_name
        # Summer buy_summer starts on the farm exit approach; spring slice
        # trims the house-exit preamble from buy_potato_seeds.
        if recording_start is None and recording_name == "buy_summer":
            params["recording_start"] = 0
        elif recording_start is not None:
            params["recording_start"] = recording_start
    elif recording_start is not None:
        params["recording_start"] = recording_start
    return PhaseSpec(
        BUY_SEEDS_PHASE.phase,
        BUY_SEEDS_PHASE.kind,
        params,
        failure_policy=BUY_SEEDS_PHASE.failure_policy,
    )

GET_BERRIES_AND_SHIP_PHASE = PhaseSpec(
    "GET_BERRIES_AND_SHIP",
    "recorded",
    # Shipping bin cutoff: berries must be in the bin before 17:00 (5 PM)
    # or the shipper has already collected for the day and no payment posts.
    {"task_name": "get_two_berries_and_ship_after_farm_exit"},
    failure_policy="optional",
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
    # Short morning slice only — seed shop closes early and must not be starved.
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
    required_tools=("watering_can",),
    estimated_frames=2500,
    failure_modes=("shelf_miss", "carry_full", "wrong_house_size"),
)

ENSURE_CROP_SEEDS_PHASE = PhaseSpec(
    "ENSURE_CROP_SEEDS",
    "ensure_seed",
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
HOT_SPRING_STAMINA_PHASE = PhaseSpec(
    "HOT_SPRING_STAMINA",
    "hot_spring",
    {"min_stamina": 40, "return_to_farm": True},
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

# ── Chicken coop phases ──

NAV_TO_COOP_PHASE = PhaseSpec(
    "NAV_TO_COOP",
    "multi_nav",
    {"route": "farm_to_coop", "timeout": 5000},
)

NAV_BARN_TO_COOP_PHASE = PhaseSpec(
    "NAV_TO_COOP",
    "multi_nav",
    {"route": "barn_to_coop", "timeout": 2500, "initial_settle_frames": 0},
)

ENTER_COOP_PHASE = PhaseSpec(
    "ENTER_COOP",
    "directional_transition",
    {
        "direction": "up",
        "origin_tilemap": 0x00,
        "target_tilemap": 0x28,
        "timeout": 900,
        "stand_tile": (28, 22),
        "stand_tolerance": 0,
        "target_stand_tile": (8, 12),
        "target_stand_tolerance": 1,
        "settle_frames": 60,
        "door_align_px": 28 * 16 + 8,
        "overshoot_limit_px": 330,
        "require_empty_hands": True,
    },
)

COOP_CHORES_PHASE = PhaseSpec(
    "COOP_CHORES",
    "coop_chores",
    {"egg_mode": "auto"},
    required_maps=(0x28,),
    estimated_frames=4000,
    failure_modes=("feed_timeout", "egg_stuck", "multi_adult", "dynamic_egg_tile"),
)

# Gift variant — exit coop holding the egg for delivery to an NPC.
COOP_GIFT_PHASE = PhaseSpec(
    "COOP_GIFT",
    "coop_chores",
    {"egg_mode": "gift"},
    required_maps=(0x28,),
    estimated_frames=4000,
    failure_modes=("feed_timeout", "egg_stuck", "gift_exit_fail"),
)

EXIT_COOP_PHASE = PhaseSpec(
    "EXIT_COOP",
    "directional_transition",
    {
        "direction": "down",
        "origin_tilemap": 0x28,
        "target_tilemap": 0x00,
        "timeout": 1500,
        "stand_tile": (8, 12),
        "stand_tolerance": 1,
        "door_align_px": 8 * 16 + 8,
        "settle_frames": 5,
    },
)

CHICKEN_PHASES: List[PhaseSpec] = [
    NAV_TO_COOP_PHASE,
    ENTER_COOP_PHASE,
    COOP_CHORES_PHASE,
    EXIT_COOP_PHASE,
]

CHICKEN_AFTER_BARN_PHASES: List[PhaseSpec] = [
    NAV_BARN_TO_COOP_PHASE,
    ENTER_COOP_PHASE,
    COOP_CHORES_PHASE,
    EXIT_COOP_PHASE,
]

COOP_CURRENT_PHASES: List[PhaseSpec] = [
    COOP_CHORES_PHASE,
    EXIT_COOP_PHASE,
]

NAV_TO_COOP_FOR_CHICKEN_SALE_PHASE = PhaseSpec(
    "NAV_TO_COOP_FOR_SALE",
    "multi_nav",
    {"route": "farm_to_coop_sale", "timeout": 5000, "initial_settle_frames": 0},
    failure_policy="optional",
)

ENTER_COOP_FOR_CHICKEN_SALE_PHASE = PhaseSpec(
    "ENTER_COOP_FOR_SALE",
    "directional_transition",
    dict(ENTER_COOP_PHASE.params),
    failure_policy="optional",
)

PICKUP_CHICKEN_FOR_SALE_PHASE = PhaseSpec(
    "PICKUP_CHICKEN_FOR_SALE",
    "pickup_chicken",
    {"timeout": 1800},
    failure_policy="optional",
)

EXIT_COOP_FOR_CHICKEN_SALE_PHASE = PhaseSpec(
    "EXIT_COOP_FOR_SALE",
    "directional_transition",
    dict(EXIT_COOP_PHASE.params),
    failure_policy="optional",
)

DROP_CHICKEN_FOR_SALE_PHASE = PhaseSpec(
    "DROP_CHICKEN_FOR_SALE",
    "drop_chicken",
    {"target_px": (60, 480), "radius": 2, "timeout": 3000},
    failure_policy="optional",
)

SELL_CHICKEN_PHASE = PhaseSpec(
    "SELL_CHICKEN",
    "chicken_sale_event",
    {"standby_px": (62, 448), "payout_px": (146, 457), "timeout": 18000},
    failure_policy="optional",
)

NAV_TO_ANIMAL_SHOP_FOR_CHICKEN_SALE_PHASE = PhaseSpec(
    "NAV_TO_ANIMAL_SHOP_FOR_SALE",
    "multi_nav",
    {"route": "farm_to_animal_shop_counter_sale", "timeout": 9000, "initial_settle_frames": 0},
    failure_policy="optional",
)

REQUEST_CHICKEN_SALE_PHASE = PhaseSpec(
    "REQUEST_CHICKEN_SALE",
    "chicken_sale_request",
    {"task_name": "sell_chicken", "start_frame": 2863, "end_frame": 3297},
    failure_policy="optional",
)

EXIT_ANIMAL_SHOP_AFTER_CHICKEN_SALE_PHASE = PhaseSpec(
    "EXIT_ANIMAL_SHOP_AFTER_CHICKEN_SALE",
    "multi_nav",
    {"route": "animal_shop_to_town", "timeout": 3000},
    failure_policy="optional",
)

WAIT_BEFORE_CHICKEN_PICKUP_RETURN_PHASE = PhaseSpec(
    "WAIT_BEFORE_CHICKEN_PICKUP_RETURN",
    "wait_until_time",
    {"target_hour": 12, "target_minute": 10, "timeout": 5000},
    failure_policy="optional",
)

RETURN_FARM_AFTER_CHICKEN_SALE_PHASE = PhaseSpec(
    "RETURN_FARM_AFTER_CHICKEN_SALE",
    "multi_nav",
    {"route": "town_to_farm_west_gate_sale", "timeout": 7000, "initial_settle_frames": 0},
    failure_policy="optional",
)

CHICKEN_SALE_BATCH_DROP_POINTS: List[tuple[int, int]] = [
    (60, 480),
    (60, 480),
    (60, 480),
]


def chicken_sale_stage_phases(
    *,
    suffix: str = "",
    failure_policy: str = "optional",
    drop_target_px: tuple[int, int] = (60, 480),
) -> List[PhaseSpec]:
    suffix_text = f"_{suffix}" if suffix else ""
    return [
        PhaseSpec(
            f"NAV_TO_COOP_FOR_SALE{suffix_text}",
            "multi_nav",
            dict(NAV_TO_COOP_FOR_CHICKEN_SALE_PHASE.params),
            failure_policy=failure_policy,
        ),
        PhaseSpec(
            f"ENTER_COOP_FOR_SALE{suffix_text}",
            "directional_transition",
            dict(ENTER_COOP_FOR_CHICKEN_SALE_PHASE.params),
            failure_policy=failure_policy,
        ),
        PhaseSpec(
            f"PICKUP_CHICKEN_FOR_SALE{suffix_text}",
            "pickup_chicken",
            dict(PICKUP_CHICKEN_FOR_SALE_PHASE.params),
            failure_policy=failure_policy,
        ),
        PhaseSpec(
            f"EXIT_COOP_FOR_SALE{suffix_text}",
            "directional_transition",
            dict(EXIT_COOP_FOR_CHICKEN_SALE_PHASE.params),
            failure_policy=failure_policy,
        ),
        PhaseSpec(
            f"DROP_CHICKEN_FOR_SALE{suffix_text}",
            "drop_chicken",
            {**DROP_CHICKEN_FOR_SALE_PHASE.params, "target_px": drop_target_px},
            failure_policy=failure_policy,
        ),
    ]


def chicken_sale_cycle_phases(
    *,
    suffix: str = "",
    failure_policy: str = "optional",
    wait_before_return: bool = True,
) -> List[PhaseSpec]:
    suffix_text = f"_{suffix}" if suffix else ""
    phases = [
        *chicken_sale_stage_phases(suffix=suffix, failure_policy=failure_policy),
        PhaseSpec(
            f"NAV_TO_ANIMAL_SHOP_FOR_SALE{suffix_text}",
            "multi_nav",
            dict(NAV_TO_ANIMAL_SHOP_FOR_CHICKEN_SALE_PHASE.params),
            failure_policy=failure_policy,
        ),
        PhaseSpec(
            f"REQUEST_CHICKEN_SALE{suffix_text}",
            "chicken_sale_request",
            dict(REQUEST_CHICKEN_SALE_PHASE.params),
            failure_policy=failure_policy,
        ),
        PhaseSpec(
            f"EXIT_ANIMAL_SHOP_AFTER_CHICKEN_SALE{suffix_text}",
            "multi_nav",
            dict(EXIT_ANIMAL_SHOP_AFTER_CHICKEN_SALE_PHASE.params),
            failure_policy=failure_policy,
        ),
    ]
    if wait_before_return:
        phases.append(
            PhaseSpec(
                f"WAIT_BEFORE_CHICKEN_PICKUP_RETURN{suffix_text}",
                "wait_until_time",
                dict(WAIT_BEFORE_CHICKEN_PICKUP_RETURN_PHASE.params),
                failure_policy=failure_policy,
            )
        )
    phases.extend(
        [
            PhaseSpec(
                f"RETURN_FARM_AFTER_CHICKEN_SALE{suffix_text}",
                "multi_nav",
                dict(RETURN_FARM_AFTER_CHICKEN_SALE_PHASE.params),
                failure_policy=failure_policy,
            ),
            PhaseSpec(
                f"SELL_CHICKEN{suffix_text}",
                "chicken_sale_event",
                dict(SELL_CHICKEN_PHASE.params),
                failure_policy=failure_policy,
            ),
        ]
    )
    return phases


def chicken_sale_batch_phases(
    *,
    count: int = 3,
    suffix: str = "",
    failure_policy: str = "optional",
) -> List[PhaseSpec]:
    suffix_text = f"_{suffix}" if suffix else ""
    phases: List[PhaseSpec] = []
    for index in range(count):
        drop_target = CHICKEN_SALE_BATCH_DROP_POINTS[min(index, len(CHICKEN_SALE_BATCH_DROP_POINTS) - 1)]
        stage_suffix = f"{suffix_text[1:]}_STAGE_{index + 1}" if suffix_text else f"STAGE_{index + 1}"
        phases.extend(
            chicken_sale_stage_phases(
                suffix=stage_suffix,
                failure_policy=failure_policy,
                drop_target_px=drop_target,
            )
        )
    phases.append(
        PhaseSpec(
            f"NAV_TO_ANIMAL_SHOP_FOR_SALE{suffix_text}",
            "multi_nav",
            dict(NAV_TO_ANIMAL_SHOP_FOR_CHICKEN_SALE_PHASE.params),
            failure_policy=failure_policy,
        )
    )
    for index in range(count):
        phases.append(
            PhaseSpec(
                f"REQUEST_CHICKEN_SALE{suffix_text}_{index + 1}",
                "chicken_sale_request",
                dict(REQUEST_CHICKEN_SALE_PHASE.params),
                failure_policy=failure_policy,
            )
        )
    phases.extend(
        [
            PhaseSpec(
                f"EXIT_ANIMAL_SHOP_AFTER_CHICKEN_SALE{suffix_text}",
                "multi_nav",
                dict(EXIT_ANIMAL_SHOP_AFTER_CHICKEN_SALE_PHASE.params),
                failure_policy=failure_policy,
            ),
            PhaseSpec(
                f"RETURN_FARM_AFTER_CHICKEN_SALE{suffix_text}",
                "multi_nav",
                dict(RETURN_FARM_AFTER_CHICKEN_SALE_PHASE.params),
                failure_policy=failure_policy,
            ),
            PhaseSpec(
                f"SELL_CHICKEN{suffix_text}",
                "chicken_sale_event",
                {
                    **SELL_CHICKEN_PHASE.params,
                    "event_hour": 0,
                    "target_sales": count,
                    "timeout": 24000,
                },
                failure_policy=failure_policy,
            ),
        ]
    )
    return phases


# ── Barn cow phases ──

NAV_TO_BARN_PHASE = PhaseSpec(
    "NAV_TO_BARN",
    "multi_nav",
    {"route": "farm_to_barn", "timeout": 10000},
)

ENTER_BARN_PHASE = PhaseSpec(
    "ENTER_BARN",
    "directional_transition",
    {
        "direction": "up",
        "origin_tilemap": 0x00,
        "target_tilemap": 0x27,
        "timeout": 900,
        "stand_tile": (20, 22),
        "stand_tolerance": 0,
        "target_stand_tile": (8, 22),
        "target_stand_tolerance": 1,
        "settle_frames": 45,
        "door_align_px": 20 * 16 + 8,
        "overshoot_limit_px": 330,
        "require_empty_hands": True,
    },
)

COW_CHORES_PHASE = PhaseSpec(
    "COW_CHORES",
    "cow_chores",
    {"talk": True, "brush": True, "milk": True, "feed": True},
)

EXIT_BARN_PHASE = PhaseSpec(
    "EXIT_BARN",
    "directional_transition",
    {
        "direction": "down",
        "origin_tilemap": 0x27,
        "target_tilemap": 0x00,
        "timeout": 1800,
        "stand_tile": (8, 22),
        "stand_tolerance": 1,
        "door_align_px": 8 * 16 + 8,
        "settle_frames": 5,
    },
)

COW_PHASES: List[PhaseSpec] = [
    ENSURE_ANIMAL_TOOLS_PHASE,
    EXIT_TO_FARM_PHASE,
    NAV_TO_BARN_PHASE,
    ENTER_BARN_PHASE,
    COW_CHORES_PHASE,
    EXIT_BARN_PHASE,
]

BARN_CURRENT_COW_PHASES: List[PhaseSpec] = [
    COW_CHORES_PHASE,
    EXIT_BARN_PHASE,
]

BUY_COW_PHASES: List[PhaseSpec] = [
    EXIT_TO_FARM_PHASE,
    PhaseSpec("NAV_TO_ANIMAL_SHOP", "multi_nav", {"route": "farm_to_animal_shop_staging", "timeout": 12000}),
    PhaseSpec(
        "BUY_COW_VENDOR",
        "cow_purchase",
        {"task_name": "buy_cow", "start_frame": 1631, "end_frame": 2328},
    ),
    PhaseSpec("EXIT_ANIMAL_SHOP", "multi_nav", {"route": "animal_shop_to_town", "timeout": 3000}),
    PhaseSpec("RETURN_FARM_AFTER_COW_PURCHASE", "multi_nav", {"route": "town_to_farm", "timeout": 7000}),
    PhaseSpec(
        "NAME_COW",
        "recorded_slice",
        {"task_name": "buy_cow", "start_frame": 3262, "end_frame": 5001},
    ),
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

# Spring Day 4: buy seeds, ship 2 berries, plant + water
SPRING4_PHASES: List[PhaseSpec] = [
    EXIT_TO_FARM_PHASE,
    NAV_FARM_EXIT_PHASE,
    PhaseSpec(
        "BUY_SEEDS",
        "cross_map",
        {
            "exit_direction": "left",
            "recording_name": "buy_potato_seeds",
            "recording_start": 483,
            "origin_tilemap": 0x00,
            "timeout": 5000,
            "continue_after_return": 200,
        },
        failure_policy="optional",
    ),
    PhaseSpec(
        "SHIP_BERRY_1",
        "multi_nav",
        {"route": "berry_ship", "timeout": 8000},
        failure_policy="optional",
    ),
    PhaseSpec(
        "SHIP_BERRY_2",
        "multi_nav",
        {"route": "berry_ship", "timeout": 8000},
        failure_policy="optional",
    ),
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

SELL_CHICKEN_TEST_PHASES: List[PhaseSpec] = chicken_sale_cycle_phases(
    suffix="TEST",
    failure_policy="required",
    wait_before_return=False,
)

SELL_THREE_CHICKENS_TEST_PHASES: List[PhaseSpec] = [
    phase
    for cycle in range(1, 4)
    for phase in chicken_sale_cycle_phases(
        suffix=f"TEST_{cycle}",
        failure_policy="required",
        wait_before_return=False,
    )
]

SELL_THREE_CHICKENS_BATCH_TEST_PHASES: List[PhaseSpec] = chicken_sale_batch_phases(
    count=3,
    suffix="BATCH_TEST",
    failure_policy="required",
)

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
    "sell_chicken_test": SELL_CHICKEN_TEST_PHASES,
    "sell_three_chickens_test": SELL_THREE_CHICKENS_TEST_PHASES,
    "sell_three_chickens_batch_test": SELL_THREE_CHICKENS_BATCH_TEST_PHASES,
    "eve_loop": EVE_TALK_LOOP_PHASES,
}

# Backwards compat alias
PHASE_SEQUENCE = DAY1_PHASES


# ── Dynamic day plan builder ─────────────────────────────────────
#
# Priority order:
#   1. Exit building (always)
#   2. Buy the first cow when affordable
#   3. Barn cow chores (if cows need feed/attention)
#   4. Chicken coop chores (if chickens > 0)
#   5. Harvest ripe crops (if any)
#   6. Ensure watering can/seeds + crop work
#   7. Berry/shop money route (if still early enough for round trip)

BERRY_CUTOFF_HOUR = 15  # latest hour to start a berry run
OPTIONAL_MONEY_PHASES = frozenset({
    "BUY_SEEDS_WINDOW",
    "NAV_FARM_EXIT",
    "BUY_SEEDS",
    "TOWN_EXPLORE",
    "GET_HAMMER",
    "GET_AXE",
    "GET_SICKLE",
    "LEAVE_HOUSE_MACRO",
    "SELL_CHICKEN_WINDOW",
    "NAV_SELL_CHICKEN_START",
    "NAV_TO_COOP_FOR_SALE",
    "ENTER_COOP_FOR_SALE",
    "PICKUP_CHICKEN_FOR_SALE",
    "EXIT_COOP_FOR_SALE",
    "DROP_CHICKEN_FOR_SALE",
    "NAV_TO_ANIMAL_SHOP_FOR_SALE",
    "REQUEST_CHICKEN_SALE",
    "EXIT_ANIMAL_SHOP_AFTER_CHICKEN_SALE",
    "RETURN_FARM_AFTER_CHICKEN_SALE",
    "SELL_CHICKEN",
    "BERRY_RUN_WINDOW",
    "LEAVE_FARM_WEST",
    "EXIT_FARM_WEST",
    "BERRY_RECORDING_WINDOW",
    "GET_BERRIES_AND_SHIP",
})

# Phases whose success marks the day ready for return-home/sleep.
GO_HOME_TRIGGER_PHASES = frozenset({
    "TOWN_EXPLORE",
    "READY_TO_GO_HOME",
    "BUY_SEEDS",
})


__all__ = [
    "EXIT_HOUSE_PHASE",
    "EXIT_TO_FARM_PHASE",
    "LEAVE_HOUSE_TO_FARM_PHASE",
    "NAV_FARM_EXIT_PHASE",
    "LEAVE_FARM_WEST_PHASE",
    "EXIT_FARM_WEST_PHASE",
    "BUY_SEEDS_PHASE",
    "buy_seeds_phase",
    "GET_BERRIES_AND_SHIP_PHASE",
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
    "OPTIONAL_MONEY_PHASES",
    "GO_HOME_TRIGGER_PHASES",
]
