"""Level 7 stop predicates and hypothesized door/stair graph (no emulator)."""

from __future__ import annotations

import numpy as np

from zelda_i.door_graph.core import DoorDir, GateKind, InventoryCaps
from zelda_i.anchors import SCREEN_LEVEL7_ENTRY_ROOM
from zelda_i.level7.dungeon import (
    LEVEL7_COMPLETE_STOP,
    LEVEL7_ENTRY_STOP,
    LEVEL7_RED_CANDLE_STOP,
    TF_BEFORE_LEVEL7,
    level7_complete_stop,
    level7_entry_stop,
    level7_red_candle_stop,
)
from zelda_i.level7.graph import (
    ENTRY,
    EVIDENCE,
    HUNGRY_GORIYA,
    KEESE,
    LEVEL7_HYPOTHESIS_GRAPH,
    LEVEL7_ROOMS,
    MAP,
    MAP_EAST_LOCK,
    MOLDORMS,
    RED_CANDLE_CELLAR,
    ROUTE_ELIGIBLE,
    TRIFORCE,
    path_requires_food,
    path_uses_fifth_lock,
    preferred_path,
    ram_ids_observed,
)
from zelda_i.level7.spine import L7_STOPS, L7_THROUGH
from zelda_i.ram import (
    ADDR_HEALTH,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_SCREEN,
    ADDR_TRIFORCE,
    PLAY_MODE,
    read_snapshot,
)


def _ram(**fields: int) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = fields.get("mode", PLAY_MODE)
    ram[ADDR_LEVEL] = fields.get("level", 7)
    ram[ADDR_SCREEN] = fields.get("screen", 0)
    ram[ADDR_LINK_X] = fields.get("x", 120)
    ram[ADDR_LINK_Y] = fields.get("y", 141)
    ram[ADDR_TRIFORCE] = fields.get("triforce", TF_BEFORE_LEVEL7)
    ram[ADDR_HEALTH] = fields.get("health", 0xBB)
    return ram


def test_public_through_targets_are_exactly_three_chapters() -> None:
    assert L7_THROUGH == ("level7-entry", "level7-red-candle", "level7")
    assert set(L7_STOPS) == set(L7_THROUGH)


def test_entry_room_is_live_but_stop_stays_fail_closed() -> None:
    snap = read_snapshot(_ram(screen=SCREEN_LEVEL7_ENTRY_ROOM, x=120, y=205))
    assert LEVEL7_ENTRY_STOP.screen == SCREEN_LEVEL7_ENTRY_ROOM
    assert LEVEL7_ENTRY_STOP.observed
    assert LEVEL7_ENTRY_STOP.evidence == "fixture-live"
    assert not LEVEL7_ENTRY_STOP.route_eligible
    assert LEVEL7_RED_CANDLE_STOP.screen is None
    assert LEVEL7_COMPLETE_STOP.level is None
    # Spine evidence set is {natural-segment, spine-green}; fixture-live is not in it.
    assert not level7_entry_stop(snap, whistle=1, food=1)
    assert not level7_red_candle_stop(snap, candle=2, whistle=1, food=0)
    assert not level7_complete_stop(
        snap, candle=2, whistle=1, incoming_heart_containers=12
    )


def test_hypothesis_graph_entry_and_north_dest_have_ram_id() -> None:
    assert ram_ids_observed()
    entry = next(room for room in LEVEL7_ROOMS if room.source_id == ENTRY)
    assert entry.ram_id == SCREEN_LEVEL7_ENTRY_ROOM
    assert entry.evidence == "fixture-live"
    assert not entry.route_eligible
    north = next(room for room in LEVEL7_ROOMS if room.source_id == MOLDORMS)
    assert north.ram_id == 0x69
    assert north.evidence == "fixture-live"
    assert not north.route_eligible
    others = [
        room
        for room in LEVEL7_ROOMS
        if room.source_id not in {ENTRY, MOLDORMS}
    ]
    assert all(room.ram_id is None for room in others)
    assert all(room.evidence == EVIDENCE for room in others)
    assert all(room.route_eligible is ROUTE_ELIGIBLE for room in LEVEL7_ROOMS)
    assert all(room.source_id > 0x7F for room in LEVEL7_ROOMS)
    assert LEVEL7_HYPOTHESIS_GRAPH.level == 7


def test_entry_exits_match_live_png() -> None:
    """Live 0x79: north open to 0x69, east present dest unobserved."""
    exits = LEVEL7_HYPOTHESIS_GRAPH.edges_from(ENTRY)
    by_dir = {exit_.direction: exit_ for exit_ in exits}
    assert set(by_dir) == {DoorDir.UP, DoorDir.RIGHT}

    north = by_dir[DoorDir.UP]
    assert north.gate is GateKind.OPEN
    assert north.target_room == MOLDORMS
    assert north.verification == "fixture-live"
    assert north.is_pathfinding
    dest = next(room for room in LEVEL7_ROOMS if room.source_id == MOLDORMS)
    assert dest.ram_id == 0x69
    assert dest.name == "entry_north_goriya"

    east = by_dir[DoorDir.RIGHT]
    assert east.gate is not GateKind.SEALED
    assert east.target_room is None
    assert not east.is_pathfinding
    assert east.verification == "probe_geometry"

    back = LEVEL7_HYPOTHESIS_GRAPH.exit_between(MOLDORMS, ENTRY)
    assert back is not None
    assert back.direction is DoorDir.DOWN

    dest_exits = {e.direction: e for e in LEVEL7_HYPOTHESIS_GRAPH.edges_from(MOLDORMS)}
    assert DoorDir.UP not in dest_exits  # live 0x69 north is a sealed wall
    dest_east = dest_exits[DoorDir.RIGHT]
    assert dest_east.target_room is None
    assert dest_east.gate is GateKind.KILL_CLEAR
    assert not dest_east.is_pathfinding


def test_preferred_path_uses_bomb_skip_and_food_gate() -> None:
    caps = InventoryCaps(keys=4, bombs=8, can_clear=True)
    # Live 0x69 east dest is unobserved, so ENTRY is disconnected from the
    # hyp interior. Interior bomb/food/fifth-lock routing still holds from KEESE.
    assert preferred_path(ENTRY, MAP, caps) is None
    to_map = preferred_path(KEESE, MAP, caps)
    assert to_map is not None
    assert path_requires_food(to_map)
    assert any(exit_.target_room == HUNGRY_GORIYA for exit_ in to_map)
    to_candle = preferred_path(KEESE, RED_CANDLE_CELLAR, caps)
    assert to_candle is not None
    assert not path_uses_fifth_lock(to_candle)
    assert all(exit_.target_room != MAP_EAST_LOCK for exit_ in to_candle)
    to_tf = preferred_path(KEESE, TRIFORCE, caps)
    assert to_tf is not None
    assert any(exit_.gate is GateKind.BOMB for exit_ in to_tf)


def test_fifth_lock_needed_only_without_bombs() -> None:
    no_bombs = InventoryCaps(keys=8, bombs=0, can_clear=True)
    assert preferred_path(KEESE, RED_CANDLE_CELLAR, no_bombs) is None
    no_keys = InventoryCaps(keys=0, bombs=8, can_clear=True)
    assert preferred_path(KEESE, HUNGRY_GORIYA, no_keys) is None
