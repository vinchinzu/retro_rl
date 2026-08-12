"""Inventory, shed, and simple deadline tasks used by the day planner.

Compatibility barrel: implementations live in focused modules
(:mod:`inventory_common`, :mod:`inventory_time`, :mod:`inventory_exit`,
:mod:`inventory_shed`). Existing
``from harvest.planner.tasks.inventory import X`` imports keep working.
"""

from __future__ import annotations

from harvest.core.carry import (
    ADDR_TOOL_BACKPACK,
    ADDR_TOOL_SELECTED,
    SEED_ITEM,
    seed_in_carry_pair,
    tool_in_carry_pair,
)
from harvest.planner.tasks.inventory_common import (
    EVENT_1F68_DOG_OWNED,
    EVENT_1F68_MORNING_INTRO_DONE,
    EVENT_1F68_OUTDOOR_INTRO_MASK,
    farm_free_move_ready,
    farm_house_front_softlock,
    outdoor_intro_flags_ready,
)
from harvest.planner.tasks.inventory_time import (
    DeadlineCheckTask,
    FarmShippingWaitTask,
    WaitUntilTimeTask,
)
from harvest.planner.tasks.inventory_exit import (
    CompleteOutdoorMorningIntroTask,
    ExitToFarmTask,
    FarmExitTask,
)
from harvest.planner.tasks.inventory_shed import (
    SHED_SEED_SPECS,
    SHED_TOOL_SPECS,
    EnsureAnimalToolsTask,
    EnsureCarryToolTask,
    EnsureCropSeedsTask,
    RecordingSliceSpec,
    ShedFetchItemTask,
    ShedSeedSpec,
    ShedShelfSpec,
    ShedShelfToolTask,
    ShedToolSpec,
    SwapCarrySlotsTask,
    load_recording_slice,
    shed_enter_transition,
    shed_farm_route_name,
)

# Re-export carry helpers for existing importers.
__carry_exports__ = (
    "tool_in_carry_pair",
    "seed_in_carry_pair",
    "SEED_ITEM",
    "ADDR_TOOL_SELECTED",
    "ADDR_TOOL_BACKPACK",
)

__all__ = [
    "RecordingSliceSpec",
    "ShedShelfSpec",
    "ShedToolSpec",
    "ShedSeedSpec",
    "SHED_TOOL_SPECS",
    "SHED_SEED_SPECS",
    "tool_in_carry_pair",
    "seed_in_carry_pair",
    "shed_farm_route_name",
    "shed_enter_transition",
    "farm_free_move_ready",
    "outdoor_intro_flags_ready",
    "farm_house_front_softlock",
    "CompleteOutdoorMorningIntroTask",
    "EVENT_1F68_OUTDOOR_INTRO_MASK",
    "EVENT_1F68_DOG_OWNED",
    "EVENT_1F68_MORNING_INTRO_DONE",
    "load_recording_slice",
    "DeadlineCheckTask",
    "WaitUntilTimeTask",
    "FarmShippingWaitTask",
    "ExitToFarmTask",
    "SwapCarrySlotsTask",
    "ShedShelfToolTask",
    "ShedFetchItemTask",
    "EnsureCarryToolTask",
    "EnsureAnimalToolsTask",
    "EnsureCropSeedsTask",
    "FarmExitTask",
    # Carry re-exports (not always listed historically but used via __carry_exports__)
    "SEED_ITEM",
    "ADDR_TOOL_SELECTED",
    "ADDR_TOOL_BACKPACK",
]
