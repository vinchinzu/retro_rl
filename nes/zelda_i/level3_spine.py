"""Survival-spine L3: post-L2 OW → Manji dest 0x5b (west key then occupancy).

``--through level3`` this pass stops at dest 0x5b after west key 0x7b
(rr-4d53.3.1.2). Raft → Manhandla → TF 0x04 stays the parent bead and still
poke-16 on the isolated suffix.
"""

from __future__ import annotations

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
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot

WEST_KEY_SPINE_MAX_FRAMES = 8000
NORTH_CHAIN_SPINE_MAX_FRAMES = 16000
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
