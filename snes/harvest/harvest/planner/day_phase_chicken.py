"""Chicken coop and chicken-sale phase specs and builders."""

from __future__ import annotations

from typing import List

from harvest.planner.day_phase_types import DayPlannerPolicy, PhaseSpec

# ── Chicken coop phases ──

COOP_EXIT_INPUT_LOCK_FRAMES = 8  # counted from the end of the fade-in; the pad wakes 3 frames later

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
        # The farm fades in after the door and the farmer ignores every button
        # until it ends (``screen_faded_in``), ~70 frames at (456, 360). A route
        # started sooner reads that as a push block, cycles left, down, up, and
        # the up lands first: back into the coop, 27 rescues in Y3 Spring.
        "settle_frames": COOP_EXIT_INPUT_LOCK_FRAMES,
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

def splice_coop_reentry(schedule, index: int, chores: PhaseSpec, tilemap: int, done: set) -> bool:
    """After a recovery walked the farmer out of the coop, walk back in once.

    ``COOP_CHORES`` that times out walking to the feed bin is recovered by an
    exit to the farm; the retry then reads "not in coop" and blocks the day with
    the hens unfed (Y1 Fall D24, Y2 Spring D3 and D13). ``done`` is the day's
    one-shot ledger.
    """
    if chores.phase != "COOP_CHORES" or "COOP_REENTRY" in done or int(tilemap) != 0x00:
        return False
    done.add("COOP_REENTRY")
    schedule.splice_at(index, [NAV_TO_COOP_PHASE, ENTER_COOP_PHASE, chores])
    print("[DAY_PLAN] COOP_CHORES recovered outside the coop; walking back in once")
    return True


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
    # Default settle: it also reruns after the path-to-town relocalization.
    {"route": "farm_to_animal_shop_counter_sale", "timeout": 9000},
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
    # The cow buy's return, with its settle: the town door holds input for a
    # few frames, and a walk started at once read that as stuck and replanned
    # back into the shop (and its menu: 900G of feed in one pin run).
    {"route": "animal_shop_town_to_farm", "timeout": 7000},
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
    event_hour: int = 15,
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
                {**SELL_CHICKEN_PHASE.params, "event_hour": event_hour},
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

def _chicken_oversupplied(adult_chickens: int, policy: DayPlannerPolicy) -> bool:
    return policy.include_chickens and adult_chickens > policy.max_adult_chickens


def _coop_chores_phase(*, oversupplied: bool, policy: DayPlannerPolicy) -> PhaseSpec:
    """A hen waiting for sale is not fed; the incubator still follows the hatch gate."""
    if not oversupplied and not policy.hold_flock_growth:
        return COOP_CHORES_PHASE
    params = {"egg_mode": "ship" if policy.hold_flock_growth else "auto"}
    if oversupplied:
        params["max_feed_adults"] = policy.max_adult_chickens
    return PhaseSpec("COOP_CHORES", "coop_chores", params)


def _chicken_phases(
    *,
    exited_barn: bool,
    oversupplied: bool,
    policy: DayPlannerPolicy,
) -> List[PhaseSpec]:
    phases = list(CHICKEN_AFTER_BARN_PHASES if exited_barn else CHICKEN_PHASES)
    chores = _coop_chores_phase(oversupplied=oversupplied, policy=policy)
    return [chores if phase.phase == "COOP_CHORES" else phase for phase in phases]


def _coop_current_phases(*, oversupplied: bool, policy: DayPlannerPolicy) -> List[PhaseSpec]:
    return [_coop_chores_phase(oversupplied=oversupplied, policy=policy), EXIT_COOP_PHASE]


def _chicken_sale_phases(
    *,
    adult_chickens: int,
    hour: int,
    is_sunday: bool,
    policy: DayPlannerPolicy,
    is_festival: bool = False,
) -> List[PhaseSpec]:
    if (
        not policy.include_chicken_sales
        or not policy.include_shop_run
        or is_sunday
        or is_festival
        or not _chicken_oversupplied(adult_chickens, policy)
        or hour >= policy.chicken_sale_cutoff_hour
    ):
        return []
    # The ROM pays as soon as the farmer re-enters the farm with the request
    # flag set and the hen at the drop (``Event_UNKChickenRelated``); there is
    # no hour in the check, so the trip does not wait out the morning.
    phases = chicken_sale_cycle_phases(failure_policy="optional", wait_before_return=False, event_hour=0)
    return [
        PhaseSpec(
            "SELL_CHICKEN_WINDOW",
            "deadline",
            {"latest_hour": policy.chicken_sale_cutoff_hour, "latest_minute": 0},
            failure_policy="optional",
        ),
        *phases,
    ]


# Married Eve takes the morning egg at home (``harvest.tasks.eve_gift``) while
# her hearts sit below the score window. The chain runs before the day's work;
# a failed step skips the rest of it.
WIFE_GIFT_PHASES: List[PhaseSpec] = [
    PhaseSpec("NAV_TO_COOP_FOR_GIFT", "multi_nav", dict(NAV_TO_COOP_PHASE.params), failure_policy="optional"),
    PhaseSpec("ENTER_COOP_FOR_GIFT", "directional_transition", dict(ENTER_COOP_PHASE.params),
              failure_policy="optional"),
    PhaseSpec("COOP_GIFT", "coop_chores", {"egg_mode": "gift"}, failure_policy="optional"),
    PhaseSpec("EXIT_COOP_FOR_GIFT", "directional_transition", dict(EXIT_COOP_PHASE.params),
              failure_policy="optional"),
    PhaseSpec("NAV_HOUSE_FOR_GIFT", "nav", {"target_px": (136, 376), "radius": 8, "soft_radius": 14,
                                           "via_px": [(456, 392), (136, 392)],
                                           "timeout": 4000}, failure_policy="optional"),
    # The house door takes the farmer with an egg in hand.
    PhaseSpec("ENTER_HOUSE_FOR_GIFT", "directional_transition", {
        "direction": "up", "origin_tilemap": 0x00, "target_tilemap": 0x16, "timeout": 2500,
        "settle_frames": 20, "stand_tile": (8, 23), "door_align_px": 136, "overshoot_limit_px": 328,
    }, failure_policy="optional"),
    PhaseSpec("GIVE_WIFE", "give_wife", {}, failure_policy="optional"),
]
OPTIONAL_WIFE_GIFT_PHASES = frozenset(phase.phase for phase in WIFE_GIFT_PHASES)

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
    "WIFE_GIFT_PHASES",
    "OPTIONAL_WIFE_GIFT_PHASES",
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
    "_chicken_oversupplied",
    "_chicken_phases",
    "_chicken_sale_phases",
    "_coop_current_phases",
]
