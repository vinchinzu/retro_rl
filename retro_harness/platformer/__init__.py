"""Reusable side-scroller speedrun optimization framework.

Provides config-driven level definitions, progress tracking, genetic algorithm
optimization, and hill climbing for SNES platformer TAS/speedrun automation.
"""

from retro_harness.platformer.level_config import (
    PlatformerRAM,
    LevelConfig,
    LEVEL_REGISTRY,
    register_level,
    get_level_config,
    list_levels,
)
from retro_harness.platformer.progress import (
    ProgressTracker,
    MonotonicAxisTracker,
    CompositeAxisTracker,
    HighWaterWithBacktrack,
    WaypointTracker,
    make_progress_tracker,
)
from retro_harness.platformer.evaluator import Evaluator, EvalResult
from retro_harness.platformer.actions import (
    DEFAULT_PLATFORMER_ACTIONS,
    NUM_BUTTONS,
    action_index_to_buttons,
    buttons_to_action_index,
)
from retro_harness.platformer.bk2_extract import (
    extract_raw_actions_from_bk2,
    extract_action_indices_from_bk2,
    save_actions,
    load_actions,
)
from retro_harness.platformer.frame_tools import (
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
from retro_harness.platformer.segment_hillclimb import (
    PrefixCheckpoint,
    segment_hillclimb_raw,
)
from retro_harness.platformer.rle_ops import (
    compress_rle,
    expand_rle,
    mutate_rle,
    rle_normalize,
    rle_total_frames,
)
from retro_harness.platformer.rle_optimize import (
    RleWindow,
    get_smb_bottleneck_windows,
    phase_shift_transitions,
    rle_ga_window,
    rle_hillclimb_window,
)

# Back-compat alias (lazy via rle_optimize.__getattr__ when accessed as attribute)
def __getattr__(name: str):
    if name == "SMB_BOTTLENECK_WINDOWS":
        return get_smb_bottleneck_windows()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
