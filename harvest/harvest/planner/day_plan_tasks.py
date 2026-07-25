"""Compatibility imports for day-plan task implementations.

Task classes now live under :mod:`harvest.planner.tasks` so planner,
navigation, inventory, and animal-shop behavior can evolve independently.
This module preserves the historic import surface for tests and tools.
"""

from __future__ import annotations

from harvest.planner.tasks.transitions import (
    ExitBuildingTask,
    DirectionalTransitionTask,
)

from harvest.planner.tasks.inventory import (
    RecordingSliceSpec,
    ShedToolSpec,
    ShedSeedSpec,
    SHED_TOOL_SPECS,
    SHED_SEED_SPECS,
    tool_in_carry_pair,
    seed_in_carry_pair,
    shed_farm_route_name,
    load_recording_slice,
    DeadlineCheckTask,
    WaitUntilTimeTask,
    ExitToFarmTask,
    SwapCarrySlotsTask,
    ShedShelfToolTask,
    EnsureCarryToolTask,
    EnsureAnimalToolsTask,
    EnsureCropSeedsTask,
    FarmExitTask,
)

from harvest.planner.tasks.chicken_sale import (
    ITEM_CHICKEN,
    CoopPickupChickenTask,
    DropCarriedChickenTask,
    ChickenSaleFollowupTask,
    ChickenSaleRequestTask,
    ChickenSaleEventTask,
    CowPurchaseTask,
)

from harvest.planner.tasks.navigation import (
    MAX_HOP,
    STALE_TILE_IDS,
    find_frontier_path,
    find_loaded_direction,
    NavTask,
    CrossMapRecordedTask,
    RecordedTransitionTask,
    MultiMapNavTask,
)

from harvest.planner.tasks.home import (
    HOUSE_FRONT_PX,
    HOUSE_DOOR_FRONT_PX,
    HOUSE_BED_STAND_PX,
    HOUSE_SLEEP_TRANSITION_TILEMAP,
    HOUSE_BED_STAND_TOLERANCE,
    ReturnHomeTask,
    GoToSleepTask,
)

__all__ = [
    "ExitBuildingTask",
    "DirectionalTransitionTask",
    "RecordingSliceSpec",
    "ShedToolSpec",
    "ShedSeedSpec",
    "SHED_TOOL_SPECS",
    "SHED_SEED_SPECS",
    "tool_in_carry_pair",
    "seed_in_carry_pair",
    "shed_farm_route_name",
    "load_recording_slice",
    "DeadlineCheckTask",
    "WaitUntilTimeTask",
    "ExitToFarmTask",
    "SwapCarrySlotsTask",
    "ShedShelfToolTask",
    "EnsureCarryToolTask",
    "EnsureAnimalToolsTask",
    "EnsureCropSeedsTask",
    "FarmExitTask",
    "ITEM_CHICKEN",
    "CoopPickupChickenTask",
    "DropCarriedChickenTask",
    "ChickenSaleFollowupTask",
    "ChickenSaleRequestTask",
    "ChickenSaleEventTask",
    "CowPurchaseTask",
    "MAX_HOP",
    "STALE_TILE_IDS",
    "find_frontier_path",
    "find_loaded_direction",
    "NavTask",
    "CrossMapRecordedTask",
    "RecordedTransitionTask",
    "MultiMapNavTask",
    "HOUSE_FRONT_PX",
    "HOUSE_DOOR_FRONT_PX",
    "HOUSE_BED_STAND_PX",
    "HOUSE_SLEEP_TRANSITION_TILEMAP",
    "HOUSE_BED_STAND_TOLERANCE",
    "ReturnHomeTask",
    "GoToSleepTask",
]
