"""Level 7 stop predicates and hypothesized door/stair graph (no emulator)."""

from __future__ import annotations

import numpy as np

from zelda_i.level7.dungeon import POST_L7_ARROW_RUPEES
from zelda_i.door_graph.core import DoorDir, GateKind, InventoryCaps
from zelda_i.anchors import SCREEN_LEVEL7_ENTRY_ROOM
from zelda_i.level7.dungeon import (
    LEVEL7_COMPLETE_STOP,
    LEVEL7_ENTRY_STOP,
    LEVEL7_RED_CANDLE_STOP,
    MEASURED_POST_L7_EXIT,
    RED_CANDLE,
    TF_AFTER_LEVEL7,
    TF_BEFORE_LEVEL7,
    level7_complete_stop,
    level7_entry_stop,
    level7_red_candle_stop,
)
from zelda_i.level7.graph import (
    AQUAMENTUS,
    BOMB_UPGRADE,
    CANDLE_PUSH,
    DIGDOGGER_1,
    DIGDOGGER_2,
    DODONGOS_BOSS_PATH,
    DODONGOS_UPGRADE,
    PRE_BOSS,
    TIP_OF_NOSE,
    TRIFORCE,
    ENTRY,
    GORIYA_BUBBLE,
    GORIYA_COMPASS,
    GORIYA_POST_RUPEE,
    GORIYA_PRE_DIG,
    FORCED_DIGDOGGER,
    GORIYA_PRE_HUNGRY,
    HIDDEN_RUPEES,
    KEESE_TRAPS,
    OLD_MAN_NOSE,
    ROPES_KEY,
    STALFOS_KEY,
    EVIDENCE,
    GORIYA_HINT,
    HUNGRY_GORIYA,
    KEESE,
    LEVEL7_HYPOTHESIS_GRAPH,
    LEVEL7_ROOMS,
    MAP,
    MAP_EAST_LOCK,
    MOLDORMS,
    NOSE_CELLAR,
    RED_CANDLE_CELLAR,
    WEST_LOCK_SKIP,
    ROUTE_ELIGIBLE,
    TRIFORCE,
    path_requires_food,
    path_uses_fifth_lock,
    preferred_path,
    ram_ids_observed,
)
from zelda_i.level7.spine import L7_STOPS, L7_THROUGH
from zelda_i.ram import PLAY_MODE, read_snapshot
from zelda_i.tests.ram_helpers import make_ram

_DEFAULTS = {
    "mode": PLAY_MODE,
    "level": 7,
    "screen": 0,
    "x": 120,
    "y": 141,
    "triforce": TF_BEFORE_LEVEL7,
    "health": 0xBB,
}


def _ram(**fields: int) -> np.ndarray:
    return make_ram(_DEFAULTS, **fields)


def test_public_through_targets_are_three_chapters_plus_shop() -> None:
    assert set(L7_STOPS) == set(L7_THROUGH)


def test_entry_stop_is_spine_green_and_later_stops_stay_fail_closed() -> None:
    """All three L7 stops are spine-green; predicates still gate on items/TF."""
    snap = read_snapshot(_ram(screen=SCREEN_LEVEL7_ENTRY_ROOM, x=120, y=205))
    assert LEVEL7_ENTRY_STOP.screen == SCREEN_LEVEL7_ENTRY_ROOM
    assert LEVEL7_ENTRY_STOP.observed
    assert LEVEL7_ENTRY_STOP.evidence == "spine-green"
    assert LEVEL7_ENTRY_STOP.route_eligible
    assert LEVEL7_RED_CANDLE_STOP.screen == 0x4A
    assert LEVEL7_RED_CANDLE_STOP.observed
    assert LEVEL7_RED_CANDLE_STOP.route_eligible
    assert LEVEL7_COMPLETE_STOP.screen == 0x42
    assert LEVEL7_COMPLETE_STOP.level == 0
    assert LEVEL7_COMPLETE_STOP.observed
    assert LEVEL7_COMPLETE_STOP.route_eligible
    assert level7_entry_stop(snap, whistle=1, food=1)
    # Still gated on the natural items and the exact incoming Triforce.
    assert not level7_entry_stop(snap, whistle=0, food=1)
    assert not level7_entry_stop(snap, whistle=1, food=0)
    off_tf = read_snapshot(
        _ram(screen=SCREEN_LEVEL7_ENTRY_ROOM, x=120, y=205, triforce=0x1F)
    )
    assert not level7_entry_stop(off_tf, whistle=1, food=1)
    assert not level7_red_candle_stop(snap, candle=2, whistle=1, food=0)
    assert not level7_complete_stop(
        snap, candle=2, whistle=1, incoming_heart_containers=12
    )


def test_complete_stop_promoted_and_fails_closed_on_wrong_screen() -> None:
    """TF 0x7F + Candle 2 + HC+1 + full hearts at OW 0x42 is verified leave."""
    assert TF_AFTER_LEVEL7 == 0x7F
    assert RED_CANDLE == 2
    assert LEVEL7_COMPLETE_STOP.screen == 0x42
    assert LEVEL7_COMPLETE_STOP.level == 0
    assert LEVEL7_COMPLETE_STOP.evidence == "spine-green"
    assert LEVEL7_COMPLETE_STOP.route_eligible
    assert MEASURED_POST_L7_EXIT.verified is True
    assert MEASURED_POST_L7_EXIT.screen == 0x42
    assert MEASURED_POST_L7_EXIT.link_x == 96
    assert MEASURED_POST_L7_EXIT.link_y == 93
    assert MEASURED_POST_L7_EXIT.triforce == 0x7F
    assert MEASURED_POST_L7_EXIT.keys == 0
    assert MEASURED_POST_L7_EXIT.bombs == 0
    assert MEASURED_POST_L7_EXIT.rupees == POST_L7_ARROW_RUPEES
    assert MEASURED_POST_L7_EXIT.heart_containers == 12
    assert MEASURED_POST_L7_EXIT.selected_item == 1
    assert MEASURED_POST_L7_EXIT.arrows == 1
    assert MEASURED_POST_L7_EXIT.candle == 2
    assert MEASURED_POST_L7_EXIT.complete() is True
    assert MEASURED_POST_L7_EXIT.route_eligible is True

    # Dummy screen 0x00: fails closed
    dummy_ram = _ram(
        level=0,
        screen=0x00,
        x=112,
        y=125,
        triforce=TF_AFTER_LEVEL7,
        health=0x88,
    )
    snap_dummy = read_snapshot(dummy_ram)
    assert not level7_complete_stop(
        snap_dummy, candle=RED_CANDLE, whistle=1, incoming_heart_containers=8
    )

    # Correct screen 0x42: passes
    live_ram = _ram(
        level=0,
        screen=0x42,
        x=96,
        y=93,
        triforce=TF_AFTER_LEVEL7,
        health=0x88,
    )
    snap_live = read_snapshot(live_ram)
    assert snap_live.heart_containers == 9
    assert snap_live.health_is_full
    assert level7_complete_stop(
        snap_live, candle=RED_CANDLE, whistle=1, incoming_heart_containers=8
    )
    assert not level7_complete_stop(
        snap_live, candle=RED_CANDLE, whistle=1, incoming_heart_containers=None
    )


def test_hypothesis_graph_live_prefix_has_ram_ids() -> None:
    assert ram_ids_observed()
    live = {
        ENTRY: SCREEN_LEVEL7_ENTRY_ROOM,
        MOLDORMS: 0x69,
        KEESE: 0x6A,
        GORIYA_HINT: 0x6B,
        DIGDOGGER_1: 0x6C,
        OLD_MAN_NOSE: 0x5B,
        STALFOS_KEY: 0x6D,
        KEESE_TRAPS: 0x68,
        ROPES_KEY: 0x78,
        DODONGOS_UPGRADE: 0x58,
        BOMB_UPGRADE: 0x48,
        GORIYA_COMPASS: 0x59,
        GORIYA_BUBBLE: 0x49,
        DIGDOGGER_2: 0x39,
        GORIYA_PRE_HUNGRY: 0x38,
        HUNGRY_GORIYA: 0x28,
        MAP: 0x18,
        HIDDEN_RUPEES: 0x08,
        GORIYA_POST_RUPEE: 0x09,
        WEST_LOCK_SKIP: 0x19,
        CANDLE_PUSH: 0x1A,
        RED_CANDLE_CELLAR: 0x4A,
        GORIYA_PRE_DIG: 0x1B,
        FORCED_DIGDOGGER: 0x1C,
        DODONGOS_BOSS_PATH: 0x0C,
        TIP_OF_NOSE: 0x0D,
        NOSE_CELLAR: 0x7B,
        PRE_BOSS: 0x29,
        AQUAMENTUS: 0x2A,
        TRIFORCE: 0x2B,
    }
    for source_id, ram_id in live.items():
        room = next(r for r in LEVEL7_ROOMS if r.source_id == source_id)
        assert room.ram_id == ram_id
        assert room.evidence == "fixture-live"
        assert not room.route_eligible
    others = [room for room in LEVEL7_ROOMS if room.source_id not in live]
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
    assert dest_east.target_room == KEESE
    assert dest_east.gate is GateKind.OPEN  # walked live; the doors bit never sets
    assert dest_east.verification == "fixture-live"
    assert dest_east.is_pathfinding
    keese = next(room for room in LEVEL7_ROOMS if room.source_id == KEESE)
    assert keese.ram_id == 0x6A


def test_keese_traps_side_exits_are_fixture_live() -> None:
    """0x68 DOWN -> 0x78 ROPES_KEY; 0x58 UP KEY -> 0x48 BOMB_UPGRADE."""
    down = {
        e.direction: e for e in LEVEL7_HYPOTHESIS_GRAPH.edges_from(KEESE_TRAPS)
    }[DoorDir.DOWN]
    assert down.target_room == ROPES_KEY
    assert down.gate is GateKind.OPEN
    assert down.verification == "fixture-live"
    ropes = next(r for r in LEVEL7_ROOMS if r.source_id == ROPES_KEY)
    assert ropes.ram_id == 0x78

    north = {
        e.direction: e for e in LEVEL7_HYPOTHESIS_GRAPH.edges_from(DODONGOS_UPGRADE)
    }[DoorDir.UP]
    assert north.target_room == BOMB_UPGRADE
    assert north.gate is GateKind.KEY
    assert north.verification == "fixture-live"
    bomb = next(r for r in LEVEL7_ROOMS if r.source_id == BOMB_UPGRADE)
    assert bomb.ram_id == 0x48
    assert not bomb.route_eligible

    keese_exits = {e.direction: e for e in LEVEL7_HYPOTHESIS_GRAPH.edges_from(KEESE)}
    keese_east = keese_exits[DoorDir.RIGHT]
    assert keese_east.target_room == GORIYA_HINT
    assert keese_east.gate is GateKind.OPEN  # walked live; unlit, doors bit 0
    assert keese_east.verification == "fixture-live"
    assert keese_east.is_pathfinding
    hint = next(room for room in LEVEL7_ROOMS if room.source_id == GORIYA_HINT)
    assert hint.ram_id == 0x6B


def test_preferred_path_uses_bomb_skip_and_food_gate() -> None:
    caps = InventoryCaps(keys=4, bombs=8, can_clear=True)
    # ENTRY reaches the hyp interior through the live 0x79 -> 0x69 -> 0x6A prefix.
    from_entry = preferred_path(ENTRY, MAP, caps)
    assert from_entry is not None
    assert [exit_.target_room for exit_ in from_entry][:2] == [MOLDORMS, KEESE]
    to_map = preferred_path(KEESE, MAP, caps)
    assert to_map is not None
    assert path_requires_food(to_map)
    assert any(exit_.target_room == HUNGRY_GORIYA for exit_ in to_map)
    to_candle = preferred_path(KEESE, RED_CANDLE_CELLAR, caps)
    assert to_candle is not None
    assert not path_uses_fifth_lock(to_candle)
    assert all(exit_.target_room != MAP_EAST_LOCK for exit_ in to_candle)


def test_map_bomb_north_chain_is_fixture_live() -> None:
    """0x18 bomb-N 0x08 bomb-E 0x09 kill-S 0x19 bomb-E 0x1A."""
    caps = InventoryCaps(keys=4, bombs=8, can_clear=True)
    north = {
        e.direction: e for e in LEVEL7_HYPOTHESIS_GRAPH.edges_from(MAP)
    }[DoorDir.UP]
    assert north.target_room == HIDDEN_RUPEES
    assert north.gate is GateKind.BOMB
    assert north.verification == "fixture-live"
    rupees = next(r for r in LEVEL7_ROOMS if r.source_id == HIDDEN_RUPEES)
    assert rupees.ram_id == 0x08

    east = {
        e.direction: e for e in LEVEL7_HYPOTHESIS_GRAPH.edges_from(HIDDEN_RUPEES)
    }[DoorDir.RIGHT]
    assert east.target_room == GORIYA_POST_RUPEE
    assert east.gate is GateKind.BOMB
    assert east.verification == "fixture-live"
    post = next(r for r in LEVEL7_ROOMS if r.source_id == GORIYA_POST_RUPEE)
    assert post.ram_id == 0x09

    down = {
        e.direction: e
        for e in LEVEL7_HYPOTHESIS_GRAPH.edges_from(GORIYA_POST_RUPEE)
    }[DoorDir.DOWN]
    assert down.target_room == WEST_LOCK_SKIP
    assert down.gate is GateKind.KILL_CLEAR
    assert down.verification == "fixture-live"
    skip = next(r for r in LEVEL7_ROOMS if r.source_id == WEST_LOCK_SKIP)
    assert skip.ram_id == 0x19
    assert not skip.route_eligible

    candle_e = {
        e.direction: e for e in LEVEL7_HYPOTHESIS_GRAPH.edges_from(WEST_LOCK_SKIP)
    }[DoorDir.RIGHT]
    assert candle_e.target_room == CANDLE_PUSH
    assert candle_e.gate is GateKind.BOMB
    assert candle_e.verification == "fixture-live"
    push = next(r for r in LEVEL7_ROOMS if r.source_id == CANDLE_PUSH)
    assert push.ram_id == 0x1A
    assert not push.route_eligible
    stairs = {
        e.direction: e for e in LEVEL7_HYPOTHESIS_GRAPH.edges_from(CANDLE_PUSH)
    }[DoorDir.DOWN]
    assert stairs.target_room == RED_CANDLE_CELLAR
    assert stairs.verification == "fixture-live"
    cellar = next(r for r in LEVEL7_ROOMS if r.source_id == RED_CANDLE_CELLAR)
    assert cellar.ram_id == 0x4A
    assert not cellar.route_eligible
    back = {
        e.direction: e
        for e in LEVEL7_HYPOTHESIS_GRAPH.edges_from(RED_CANDLE_CELLAR)
    }[DoorDir.UP]
    assert back.target_room == CANDLE_PUSH
    assert back.gate is GateKind.OPEN
    assert back.verification == "fixture-live"
    pre_dig = {
        e.direction: e for e in LEVEL7_HYPOTHESIS_GRAPH.edges_from(CANDLE_PUSH)
    }[DoorDir.RIGHT]
    assert pre_dig.target_room == GORIYA_PRE_DIG
    assert pre_dig.gate is GateKind.BOMB
    assert pre_dig.verification == "fixture-live"
    gpd = next(r for r in LEVEL7_ROOMS if r.source_id == GORIYA_PRE_DIG)
    assert gpd.ram_id == 0x1B
    assert not gpd.route_eligible
    key_e = {
        e.direction: e
        for e in LEVEL7_HYPOTHESIS_GRAPH.edges_from(GORIYA_PRE_DIG)
    }[DoorDir.RIGHT]
    assert key_e.target_room == FORCED_DIGDOGGER
    assert key_e.gate is GateKind.KEY
    assert key_e.verification == "fixture-live"
    forced = next(r for r in LEVEL7_ROOMS if r.source_id == FORCED_DIGDOGGER)
    assert forced.ram_id == 0x1C
    assert not forced.route_eligible
    north = {
        e.direction: e
        for e in LEVEL7_HYPOTHESIS_GRAPH.edges_from(FORCED_DIGDOGGER)
    }[DoorDir.UP]
    assert north.target_room == DODONGOS_BOSS_PATH
    assert north.gate is GateKind.KILL_CLEAR
    assert north.verification == "fixture-live"
    boss_path = next(r for r in LEVEL7_ROOMS if r.source_id == DODONGOS_BOSS_PATH)
    assert boss_path.ram_id == 0x0C
    assert not boss_path.route_eligible
    nose_e = {
        e.direction: e
        for e in LEVEL7_HYPOTHESIS_GRAPH.edges_from(DODONGOS_BOSS_PATH)
    }[DoorDir.RIGHT]
    assert nose_e.target_room == TIP_OF_NOSE
    assert nose_e.gate is GateKind.BOMB
    assert nose_e.verification == "fixture-live"
    nose = next(r for r in LEVEL7_ROOMS if r.source_id == TIP_OF_NOSE)
    assert nose.ram_id == 0x0D
    assert not nose.route_eligible
    to_tf = preferred_path(KEESE, TRIFORCE, caps)
    assert to_tf is not None
    assert any(exit_.gate is GateKind.BOMB for exit_ in to_tf)


def test_fifth_lock_needed_only_without_bombs() -> None:
    no_bombs = InventoryCaps(keys=8, bombs=0, can_clear=True)
    assert preferred_path(KEESE, RED_CANDLE_CELLAR, no_bombs) is None
    no_keys = InventoryCaps(keys=0, bombs=8, can_clear=True)
    assert preferred_path(KEESE, HUNGRY_GORIYA, no_keys) is None
