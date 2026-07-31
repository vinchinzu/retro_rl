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
from platformer_common.frame_tools import (
    analyze_seed_static,
    cleanup_auto_inputs,
    compress_hold_window,
    count_leading_idle,
    count_trailing_idle,
    find_button_hold_stalls,
    find_stalls,
    is_idle_frame,
    load_raw_frames,
    save_raw_seed,
    search_hold_compressions,
    trim_after_completion,
    trim_leading_idle,
)
from platformer_common.segment_hillclimb import (
    PrefixCheckpoint,
    segment_hillclimb_raw,
)
from platformer_common.rle_ops import (
    compress_rle,
    expand_rle,
    mutate_rle,
    rle_normalize,
    rle_total_frames,
)
from platformer_common.rle_optimize import (
    RleWindow,
    SMB_BOTTLENECK_WINDOWS,
    phase_shift_transitions,
    rle_ga_window,
    rle_hillclimb_window,
)
