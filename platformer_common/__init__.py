"""Reusable side-scroller speedrun optimization framework.

Provides config-driven level definitions, progress tracking, genetic algorithm
optimization, and hill climbing for SNES platformer TAS/speedrun automation.
"""

from platformer_common.level_config import (
    PlatformerRAM,
    LevelConfig,
    LEVEL_REGISTRY,
    register_level,
    get_level_config,
    list_levels,
)
from platformer_common.progress import (
    ProgressTracker,
    MonotonicAxisTracker,
    CompositeAxisTracker,
    HighWaterWithBacktrack,
    WaypointTracker,
    make_progress_tracker,
)
from platformer_common.evaluator import Evaluator, EvalResult
from platformer_common.actions import (
    DEFAULT_PLATFORMER_ACTIONS,
    NUM_BUTTONS,
    action_index_to_buttons,
    buttons_to_action_index,
)
from platformer_common.bk2_extract import (
    extract_raw_actions_from_bk2,
    extract_action_indices_from_bk2,
    save_actions,
    load_actions,
)
