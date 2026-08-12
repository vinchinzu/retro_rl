"""Berry forage / ship phase specs and builders."""

from __future__ import annotations

from harvest.planner.day_phase_types import PhaseSpec

GET_BERRIES_AND_SHIP_PHASE = PhaseSpec(
    "GET_BERRIES_AND_SHIP",
    "recorded",
    # Legacy blind recording — prefer SHIP_BERRY_* multi_nav on the farm.
    # Kept for named sequences / tests that still reference the phase name.
    {"task_name": "get_two_berries_and_ship_after_farm_exit"},
    failure_policy="optional",
)

# y=31 fence (x≈11–29) seals house north-pocket from south berry bush.
# Corridor-only lift opens ≥1 gap so berry_ship BFS can go south (not thrash
# into the wall at ~(336,486)).
OPEN_FENCE_GAP_PHASE = PhaseSpec(
    "OPEN_FENCE_GAP",
    "fence_clear",
    {
        # Berry route only needs the carry-south crossing; it never returns
        # through the gap. One wall post avoids a second north-side re-entry.
        "max_fences": 1,
        "corridor_only": True,
        "timeout": 8000,
    },
    failure_policy="optional",
    required_maps=(0x00,),
    estimated_frames=4000,
    failure_modes=("no_fence", "lift_fail", "timeout"),
)

# Farm forage bush ~(585,920) → shipping bin ~(1001,969). Wild berries are ON
# the farm south of the fence (not mountain / not west exit).
SHIP_BERRY_PHASE = PhaseSpec(
    "SHIP_BERRY",
    "berry_ship",
    {"route": "berry_ship", "timeout": 18000, "initial_settle_frames": 20},
    failure_policy="optional",
    required_maps=(0x00,),
    estimated_frames=10000,
    failure_modes=("bush_unreachable", "bin_path_fail", "fence_sealed", "no_berry_tile"),
)


def ship_berry_phases(*, count: int = 2, open_fence: bool = True) -> list[PhaseSpec]:
    """Fence-gap (if needed) then multi_nav berry_ship loops to the bin."""
    n = max(1, int(count))
    contract = SHIP_BERRY_PHASE.contract
    phases: list[PhaseSpec] = []
    if open_fence:
        phases.append(OPEN_FENCE_GAP_PHASE)
    for i in range(1, n + 1):
        params = dict(SHIP_BERRY_PHASE.params)
        if i > 1:
            params["route"] = "berry_ship_repeat"
        phases.append(
            PhaseSpec(
                f"SHIP_BERRY_{i}" if n > 1 else "SHIP_BERRY",
                SHIP_BERRY_PHASE.kind,
                params,
                failure_policy=SHIP_BERRY_PHASE.failure_policy,
                contract=contract,
            )
        )
    return phases


# Spring D2 house → path fork → first mountain grape/berry. Reactive, not tape.
MOUNTAIN_BERRY_PHASE = PhaseSpec(
    "MOUNTAIN_BERRY",
    "mountain_berry",
    {"timeout": 12000, "nav_timeout": 6000, "approach_only": False, "pick_attempts": 3},
    failure_policy="optional",
    required_maps=(0x15, 0x00, 0x0C, 0x10),
    estimated_frames=4500,
    failure_modes=("nav_fail", "no_forage", "hands_full"),
)

MOUNTAIN_BERRY_PHASES: list[PhaseSpec] = [
    PhaseSpec("EXIT_TO_FARM", "farm_building_exit"),
    MOUNTAIN_BERRY_PHASE,
]

BERRY_CUTOFF_HOUR = 15  # latest hour to start a berry run
# Berry forage is independent of seed-shop; a failed bush run must not
# cascade-skip NAV_FARM_EXIT / BUY_SEEDS (wallet still funds potato).
OPTIONAL_BERRY_PHASES = frozenset({
    "BERRY_RUN_WINDOW",
    "LEAVE_FARM_WEST",
    "EXIT_FARM_WEST",
    "BERRY_RECORDING_WINDOW",
    "GET_BERRIES_AND_SHIP",
    "OPEN_FENCE_GAP",
    "SHIP_BERRY",
    "SHIP_BERRY_1",
    "SHIP_BERRY_2",
    "MOUNTAIN_BERRY",
})

__all__ = [
    "GET_BERRIES_AND_SHIP_PHASE",
    "OPEN_FENCE_GAP_PHASE",
    "SHIP_BERRY_PHASE",
    "MOUNTAIN_BERRY_PHASE",
    "MOUNTAIN_BERRY_PHASES",
    "ship_berry_phases",
    "BERRY_CUTOFF_HOUR",
    "OPTIONAL_BERRY_PHASES",
]
