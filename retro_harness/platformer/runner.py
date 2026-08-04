"""CLI entry point for platformer speedrun optimizer (compatibility facade).

Implementation lives under ``retro_harness.platformer.cli`` and
``retro_harness.platformer.replay_hud``. This module re-exports the symbols
historically imported from ``runner`` so existing callers and tests keep working.

All commands take a --level flag to select the level to optimize.
Level configs are registered by importing retro_harness.platformer.levels.
"""

from __future__ import annotations

# Trigger level registration on import (legacy behavior)
import retro_harness.platformer.levels  # noqa: F401

from retro_harness.platformer.cli.helpers import (  # noqa: F401
    _get_action_table,
    _parse_room_id_arg,
    _resolve_config,
)
from retro_harness.platformer.cli.main import main  # noqa: F401
from retro_harness.platformer.cli.optimize import (  # noqa: F401
    _load_raw_seeds_from_dir,
    _load_seeds_from_dir,
    _parse_window,
    cmd_analyze_seed,
    cmd_hillclimb,
    cmd_hillclimb_raw,
    cmd_neuro,
    cmd_optimize,
    cmd_segment_hillclimb,
    cmd_trim_seed,
)
from retro_harness.platformer.cli.play import cmd_play  # noqa: F401
from retro_harness.platformer.cli.practice import (  # noqa: F401
    _best_practice_attempt,
    _load_practice_pb_frames,
    _practice_completion_token,
    cmd_practice,
)
from retro_harness.platformer.cli.selftest import cmd_selftest  # noqa: F401
from retro_harness.platformer.cli.watch import (  # noqa: F401
    _recording_start_state,
    cmd_auto_state,
    cmd_chain,
    cmd_chain_live,
    cmd_chain_optimize,
    cmd_chain_video,
    cmd_extract,
    cmd_extract_all,
    cmd_list_levels,
    cmd_list_routes,
    cmd_prepare_seeds,
    cmd_trace_map,
    cmd_verify,
    cmd_watch,
    cmd_watch_bk2,
)
from retro_harness.platformer.replay_hud import (  # noqa: F401
    _button_names,
    _replay_with_hud,
)

__all__ = [
    "main",
    "_resolve_config",
    "_get_action_table",
    "_parse_room_id_arg",
    "_practice_completion_token",
    "_load_practice_pb_frames",
    "_best_practice_attempt",
    "_recording_start_state",
    "_load_seeds_from_dir",
    "_load_raw_seeds_from_dir",
    "_parse_window",
    "_button_names",
    "_replay_with_hud",
    "cmd_list_levels",
    "cmd_extract",
    "cmd_verify",
    "cmd_optimize",
    "cmd_hillclimb_raw",
    "cmd_analyze_seed",
    "cmd_trim_seed",
    "cmd_segment_hillclimb",
    "cmd_hillclimb",
    "cmd_neuro",
    "cmd_watch",
    "cmd_watch_bk2",
    "cmd_extract_all",
    "cmd_prepare_seeds",
    "cmd_auto_state",
    "cmd_practice",
    "cmd_play",
    "cmd_selftest",
    "cmd_trace_map",
    "cmd_list_routes",
    "cmd_chain",
    "cmd_chain_live",
    "cmd_chain_optimize",
    "cmd_chain_video",
]


if __name__ == "__main__":
    main()
