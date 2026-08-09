"""Platformer training view of the natural SM-rando Landing entry corpus."""

from __future__ import annotations

from dataclasses import replace

from retro_harness.platformer.level_config import get_level_config, register_level
from sm_rando.paths import GAME
from sm_rando.observations import landing_entry_features
import super_metroid.platformer_levels  # noqa: F401 - registers vanilla source row

SM_RANDO_LANDING_ENTRY = replace(
    get_level_config("sm_landing_site"),
    level_id="sm_rando_landing_entry",
    display_name="SM Rando natural Landing entry → Parlor",
    game_name=GAME,
    game_dir_name="sm_rando",
    start_state="NONE",
    neuro_observation_fn=landing_entry_features,
)

register_level(
    SM_RANDO_LANDING_ENTRY,
    "sm_rando_landing",
)

__all__ = ["SM_RANDO_LANDING_ENTRY"]
