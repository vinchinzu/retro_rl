"""Survival-spine L3: post-L2 OW → Manji Compass room 0x5a.

``--through level3`` stops at Compass room 0x5a after the verified west-key and
occupancy-dest chunks (rr-4d53.3.3.1). Raft → Manhandla → TF 0x04 stays the
parent bead and still poke-16 on the isolated suffix.
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
    ROOM_7B_SPEC,
    ROOM_L3_DARKNUTS,
    ROOM_L3_ENTRY,
    ROOM_L3_WEST_KEY,
    level3_reached_5b,
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
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot

WEST_KEY_SPINE_MAX_FRAMES = 8000
NORTH_CHAIN_SPINE_MAX_FRAMES = 16000
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


def level3_entry_stages():
    """After L2 TF: idle fanfare, then walk the Manji door and enter."""
    return (
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
    )


def level3_entry_success(snap: ZeldaSnapshot) -> bool:
    """Play mode in Manji entry 0x7c (predecessor of west key)."""
    return (
        snap.level == LEVEL3
        and snap.screen == ROOM_L3_ENTRY
        and snap.mode == PLAY_MODE
        and not snap.transitioning
    )


def level3_west_key_stages():
    """0x7c LEFT+UP → 0x7b Zol clear + key (rr-4d53.3.1.1)."""
    return (
        ("west_key", Level3WestKeyController(), WEST_KEY_SPINE_MAX_FRAMES),
    )


def level3_west_key_success(snap: ZeldaSnapshot) -> bool:
    """Predecessor stop (rr-4d53.3.1.1 closed): 0x7b with keys≥1."""
    return (
        snap.level == LEVEL3
        and snap.screen == ROOM_L3_WEST_KEY
        and snap.mode == PLAY_MODE
        and not snap.transitioning
        and snap.keys >= 1
        and not ROOM_7B_SPEC.live_enemies(snap)
    )


def level3_dest_6b_stages():
    """0x7c west key → 0x6b occupancy north dest into 0x5b."""
    dest_6b_room_plan()
    return (
        *level3_west_key_stages(),
        (
            "north_chain",
            Level3NorthChainController(),
            NORTH_CHAIN_SPINE_MAX_FRAMES,
        ),
    )


def level3_dest_6b_success(snap: ZeldaSnapshot) -> bool:
    """Spine stop for ``--through level3`` this leaf: play mode in 0x5b."""
    return (
        snap.level == LEVEL3
        and snap.screen == ROOM_L3_DARKNUTS
        and snap.mode == PLAY_MODE
        and not snap.transitioning
    )


def level3_dest_6b_success_ram(ram) -> bool:
    return level3_reached_5b(ram)


def level3_compass_stages():
    """Live 0x5b predecessor → west door → Compass room 0x5a."""
    return (
        (
            "compass_0x5a",
            PredicateStopController(
                Level3RaftPathController(),
                level3_compass_success,
                "level3_compass_0x5a",
            ),
            COMPASS_SPINE_MAX_FRAMES,
        ),
    )


def level3_compass_success(snap: ZeldaSnapshot) -> bool:
    """Spine stop for rr-4d53.3.3.1: playable Compass room 0x5a."""
    return (
        snap.level == LEVEL3
        and snap.screen == 0x5A
        and snap.mode == PLAY_MODE
        and not snap.transitioning
    )


def level3_west_darknuts_stages():
    """Live 0x5a predecessor → long KEY-LEFT → playable 0x59."""
    return (
        (
            "west_darknuts_0x59",
            PredicateStopController(
                Level3RaftPathController(),
                level3_west_darknuts_success,
                "level3_west_darknuts_0x59",
            ),
            WEST_DARKNUTS_SPINE_MAX_FRAMES,
        ),
    )


def level3_west_darknuts_success(snap: ZeldaSnapshot) -> bool:
    return (
        snap.level == LEVEL3
        and snap.screen == 0x59
        and snap.mode == PLAY_MODE
        and not snap.transitioning
    )


def level3_south_darknuts_stages():
    """Live 0x59 predecessor → clear → DOWN → playable 0x69."""
    return (
        (
            "south_darknuts_0x69",
            PredicateStopController(
                Level3RaftPathController(),
                level3_south_darknuts_success,
                "level3_south_darknuts_0x69",
            ),
            SOUTH_DARKNUTS_SPINE_MAX_FRAMES,
        ),
    )


def level3_south_darknuts_success(snap: ZeldaSnapshot) -> bool:
    return (
        snap.level == LEVEL3
        and snap.screen == 0x69
        and snap.mode == PLAY_MODE
        and not snap.transitioning
    )


def level3_raft_stages():
    """Live 0x69 predecessor → clear → stairs passage → natural Raft."""
    return (
        (
            "raft_0x0f",
            PredicateStopController(
                Level3RaftPathController(),
                level3_raft_success,
                "level3_raft",
            ),
            RAFT_SPINE_MAX_FRAMES,
        ),
    )


def level3_raft_success(snap: ZeldaSnapshot) -> bool:
    return snap.level == LEVEL3 and snap.screen == 0x0F and bool(snap.raft)
