"""Zelda I level8 package."""

from zelda_i.level8.dungeon import (
    LEVEL8,
    LEVEL8_ROOM_SPECS,
    Level8ChapterSpec,
    Level8ClearEndpoint,
    Level8Topology,
    level8_clear_stop,
    level8_entry_stop,
    level8_magic_key_ledger,
    level8_magic_key_stop,
)
from zelda_i.level8.entry import (
    BushBurnTarget,
    PostLevel7Handoff,
    make_burn_level8_bush_controller,
    make_isolated_bush_recon_controller,
    make_post_l7_to_bush_controller,
    make_select_red_candle_controller,
)
from zelda_i.level8.hops import (
    l8_hops,
)
from zelda_i.level8.recon import (
    LEVEL8_INTERIOR_ROOM_RECON,
    Level8InteriorRoomRecon,
)
from zelda_i.level8.spine import (
    L8_STOPS,
    L8_THROUGH,
    continue_level8_spine,
)
from zelda_i.level8.suffix import (
    LEVEL8_SUFFIX_GATES,
    Level8SuffixGate,
    Level8SuffixLineage,
    suffix_stages,
)

__all__ = [
    "BushBurnTarget",
    "L8_STOPS",
    "L8_THROUGH",
    "LEVEL8",
    "LEVEL8_INTERIOR_ROOM_RECON",
    "LEVEL8_ROOM_SPECS",
    "LEVEL8_SUFFIX_GATES",
    "Level8ChapterSpec",
    "Level8ClearEndpoint",
    "Level8InteriorRoomRecon",
    "Level8SuffixGate",
    "Level8SuffixLineage",
    "Level8Topology",
    "PostLevel7Handoff",
    "continue_level8_spine",
    "l8_hops",
    "level8_clear_stop",
    "level8_entry_stop",
    "level8_magic_key_ledger",
    "level8_magic_key_stop",
    "make_burn_level8_bush_controller",
    "make_isolated_bush_recon_controller",
    "make_post_l7_to_bush_controller",
    "make_select_red_candle_controller",
    "suffix_stages",
]
