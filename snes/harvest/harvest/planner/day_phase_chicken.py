"""Chicken coop and chicken-sale phase specs and builders."""

from __future__ import annotations

from typing import List

from harvest.planner.day_phase_types import PhaseSpec

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

OPTIONAL_CHICKEN_SALE_PHASES = frozenset({
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
})

__all__ = [
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
    "WAIT_BEFORE_CHICKEN_PICKUP_RETURN_PHASE",
    "RETURN_FARM_AFTER_CHICKEN_SALE_PHASE",
    "SELL_CHICKEN_PHASE",
    "CHICKEN_SALE_BATCH_DROP_POINTS",
    "chicken_sale_stage_phases",
    "chicken_sale_cycle_phases",
    "chicken_sale_batch_phases",
    "CHICKEN_PHASES",
    "CHICKEN_AFTER_BARN_PHASES",
    "COOP_CURRENT_PHASES",
    "SELL_CHICKEN_TEST_PHASES",
    "SELL_THREE_CHICKENS_TEST_PHASES",
    "SELL_THREE_CHICKENS_BATCH_TEST_PHASES",
    "OPTIONAL_CHICKEN_SALE_PHASES",
]
