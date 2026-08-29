"""Survival-spine L3 hops through the natural Raft boundary.

The continuous runner attaches these rows, then the carried-bomb boss suffix.
"""

from __future__ import annotations

from zelda_i.chain import PredicateStopController
from zelda_i.door_graph import (
    L3_DARKNUTS,
    L3_ENTRY,
    L3_NORTH_ZOLS,
    L3_WEST_KEY,
    LEVEL_3_DOOR_GRAPH,
    InventoryCaps,
    RoomExit,
)
from zelda_i.level3_dungeon import (
    LEVEL3,
    ROOM_5B_SPEC,
    ROOM_L3_DARKNUTS,
    ROOM_L3_ENTRY,
    ROOM_L3_RAFT_PASSAGE,
)
from zelda_i.level3_overworld import (
    POST_L2_PATH_MAX_FRAMES,
    POST_L2_SETTLE_MAX_FRAMES,
    OverworldPostL2ToLevel3Controller,
    PostL2TriforceSettleController,
)
from zelda_i.level3_path import Level3NorthChainController, Level3WestKeyController
from zelda_i.level3_raft_path import (
    CLEAR_59_MAX_FRAMES,
    CLEAR_69_MAX_FRAMES,
    DOWN_69_MAX_FRAMES,
    LEFT_5B_MAX_FRAMES,
    SPAWN_SETTLE_FRAMES,
    STAIRS_69_MAX_FRAMES,
    PASSAGE_RAFT_MAX_FRAMES,
    Level3RaftPathController,
)
from zelda_i.ram import PASSAGE_MODE
from zelda_i.spine_hops import SpineHop, ready

WEST_KEY_SPINE_MAX_FRAMES = 8000
NORTH_CHAIN_SPINE_MAX_FRAMES = 32000  # 0x6b zols + occupancy north + 0x5b Darknuts
COMPASS_SPINE_MAX_FRAMES = LEFT_5B_MAX_FRAMES + 100
WEST_DARKNUTS_SPINE_MAX_FRAMES = 3000
SOUTH_DARKNUTS_SPINE_MAX_FRAMES = (
    CLEAR_59_MAX_FRAMES + DOWN_69_MAX_FRAMES + SPAWN_SETTLE_FRAMES + 500
)
RAFT_SPINE_MAX_FRAMES = (
    CLEAR_69_MAX_FRAMES
    + STAIRS_69_MAX_FRAMES
    + PASSAGE_RAFT_MAX_FRAMES
    + SPAWN_SETTLE_FRAMES
    + 500
)
_DEST_6B_ROOMS = (L3_ENTRY, L3_WEST_KEY, L3_NORTH_ZOLS, L3_DARKNUTS)

__all__ = [
    "dest_6b_room_plan",
    "l3_hops",
]


def dest_6b_room_plan() -> tuple[RoomExit, ...]:
    """Offline room sequence 0x7c → 0x5b (kill-clear on 0x6b north)."""
    path = LEVEL_3_DOOR_GRAPH.bfs_path(
        L3_ENTRY,
        L3_DARKNUTS,
        InventoryCaps(can_clear=True),
    )
    if path is None:
        raise RuntimeError("L3 door graph has no 0x7c → 0x5b path")
    rooms = (L3_ENTRY, *[edge.target_room for edge in path])
    if rooms != _DEST_6B_ROOMS:
        raise RuntimeError(
            f"L3 dest 0x5b rooms {[hex(r) for r in rooms]} "
            f"!= {[hex(r) for r in _DEST_6B_ROOMS]}"
        )
    return path


def _dest_6b_stages():
    dest_6b_room_plan()
    return (
        ("west_key", Level3WestKeyController(), WEST_KEY_SPINE_MAX_FRAMES),
        (
            "north_chain",
            Level3NorthChainController(),
            NORTH_CHAIN_SPINE_MAX_FRAMES,
        ),
    )


def _raft_hop(through: str, stop: str, pred, name: str, max_frames: int) -> SpineHop:
    def stages():
        return (
            (
                stop,
                PredicateStopController(Level3RaftPathController(), pred, name),
                max_frames,
            ),
        )

    return SpineHop(through, stop, stages, pred)


def l3_hops(*, after_entry=None) -> tuple[SpineHop, ...]:
    """Entry → dest 0x5b → compass → west/south Darknuts → Raft."""
    compass = ready(level=LEVEL3, screen=0x5A)
    west = ready(level=LEVEL3, screen=0x59)
    south = ready(level=LEVEL3, screen=0x69)
    raft = ready(
        level=LEVEL3, screen=ROOM_L3_RAFT_PASSAGE, mode=PASSAGE_MODE, item="raft"
    )
    return (
        SpineHop(
            "l3-entry",
            "enter_level3",
            (
                (
                    "settle_l2_tf",
                    PostL2TriforceSettleController(),
                    POST_L2_SETTLE_MAX_FRAMES,
                ),
                (
                    "enter_level3",
                    OverworldPostL2ToLevel3Controller(require_dungeon=True),
                    POST_L2_PATH_MAX_FRAMES,
                ),
            ),
            ready(level=LEVEL3, screen=ROOM_L3_ENTRY),
            after=after_entry,
        ),
        SpineHop(
            "l3-dest-6b",
            "north_chain",
            _dest_6b_stages,
            ready(level=LEVEL3, screen=ROOM_L3_DARKNUTS, spec=ROOM_5B_SPEC),
        ),
        _raft_hop(
            "l3-compass",
            "compass_0x5a",
            compass,
            "level3_compass_0x5a",
            COMPASS_SPINE_MAX_FRAMES,
        ),
        _raft_hop(
            "l3-west-darknuts",
            "west_darknuts_0x59",
            west,
            "level3_west_darknuts_0x59",
            WEST_DARKNUTS_SPINE_MAX_FRAMES,
        ),
        _raft_hop(
            "l3-south-darknuts",
            "south_darknuts_0x69",
            south,
            "level3_south_darknuts_0x69",
            SOUTH_DARKNUTS_SPINE_MAX_FRAMES,
        ),
        _raft_hop(
            "l3-raft",
            "raft_0x0f",
            raft,
            "level3_raft",
            RAFT_SPINE_MAX_FRAMES,
        ),
    )
