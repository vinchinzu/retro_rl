"""Unit tests for Level 3 pure helpers (no emulator)."""

from __future__ import annotations

import numpy as np

from zelda_i.level3_dungeon import (
    DARKNUT_OBJECT_TYPE,
    KEY_DOOR_Y,
    KEESE_OBJECT_TYPE,
    LEVEL3_TRIFORCE_BIT,
    MANHANDLA_OBJECT_TYPE,
    NORTH_DOOR_X,
    PASSAGE_EXIT_WAYPOINTS,
    RAFT_CHANNEL_X,
    RAFT_PASSAGE_MODE,
    RAFT_PATH_PHASES,
    RAFT_PICKUP_X,
    RAFT_PICKUP_Y,
    RAFT_SOUTH_Y,
    ROOM_4B_SPEC,
    ROOM_59_SPEC,
    ROOM_5A_SPEC,
    ROOM_5B_SPEC,
    ROOM_69_SPEC,
    ROOM_6B_SPEC,
    ROOM_7B_SPEC,
    ROOM_ITEM_COMPASS,
    ROOM_ITEM_RAFT,
    ROOM_L3_BOSS,
    ROOM_L3_BOSS_PREP,
    ROOM_L3_BOMB_SHORTCUT,
    ROOM_L3_COMPASS,
    ROOM_L3_DARKNUTS,
    ROOM_L3_ENTRY,
    ROOM_L3_NORTH_ZOLS,
    ROOM_L3_RAFT_PASSAGE,
    ROOM_L3_SOUTH_DARKNUTS,
    ROOM_L3_WEST_DARKNUTS,
    ROOM_L3_WEST_KEY,
    ROOM_L3_ZOL_KEY_4B,
    STAIRS_69_RIGHT_Y,
    WEST_DOOR_APPROACH_Y,
    ZOL_OBJECT_TYPE,
    Level3NorthChainController,
    Level3NorthDoor7bController,
    Level3RaftPathController,
    Level3WestDoorController,
    Level3WestKeyController,
    level3_boss_prep_killables,
    level3_has_raft,
    level3_manhandla_live,
    level3_reached_5b,
    level3_reached_boss,
    level3_reached_boss_prep,
    level3_room_4b_zols_cleared,
    level3_room_6b_zols_cleared,
    level3_room_7b_key_success,
    north_door_7b_step,
    raft_passage_step,
    west_door_step,
)
from zelda_i.ram import (
    ADDR_KEYS,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_OBJ_HP,
    ADDR_OBJ_TYPE,
    ADDR_SCREEN,
    PLAY_MODE,
    read_snapshot,
)


def _ram(
    *,
    level: int = 3,
    room: int = ROOM_L3_ENTRY,
    x: int = 120,
    y: int = 205,
    mode: int = PLAY_MODE,
    keys: int = 0,
    zols: int = 0,
    darknuts: int = 0,
    hp: int = 32,
) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = mode
    ram[ADDR_LEVEL] = level
    ram[ADDR_SCREEN] = room
    ram[ADDR_LINK_X] = x
    ram[ADDR_LINK_Y] = y
    ram[ADDR_KEYS] = keys
    slot = 1
    for _ in range(zols):
        ram[ADDR_OBJ_TYPE + slot] = ZOL_OBJECT_TYPE
        ram[ADDR_OBJ_HP + slot] = hp
        ram[ADDR_LINK_X + slot] = 64 + slot * 16
        ram[ADDR_LINK_Y + slot] = 141
        slot += 1
    for _ in range(darknuts):
        ram[ADDR_OBJ_TYPE + slot] = DARKNUT_OBJECT_TYPE
        ram[ADDR_OBJ_HP + slot] = 64
        ram[ADDR_LINK_X + slot] = 80 + slot * 16
        ram[ADDR_LINK_Y + slot] = 141
        slot += 1
    return ram


def test_room_ids_and_spec() -> None:
    assert ROOM_L3_ENTRY == 0x7C
    assert ROOM_L3_WEST_KEY == 0x7B
    assert ROOM_L3_NORTH_ZOLS == 0x6B
    assert ROOM_L3_DARKNUTS == 0x5B
    assert ROOM_L3_ZOL_KEY_4B == 0x4B
    assert ROOM_L3_COMPASS == 0x5A
    assert ROOM_L3_RAFT_PASSAGE == 0x0F
    assert RAFT_PASSAGE_MODE == 9
    assert RAFT_CHANNEL_X == 176
    assert ROOM_ITEM_RAFT == 0x0C
    assert ROOM_ITEM_COMPASS == 0x16
    assert ROOM_7B_SPEC.room_id == 0x7B
    assert ROOM_7B_SPEC.level == 3
    assert ROOM_7B_SPEC.enemy_types == (ZOL_OBJECT_TYPE,)
    assert ROOM_7B_SPEC.expected_enemy_count == 6
    assert ROOM_7B_SPEC.room_item_id == 0x19
    assert ROOM_6B_SPEC.room_id == 0x6B
    assert ROOM_6B_SPEC.expected_enemy_count == 5
    assert ROOM_6B_SPEC.reward.settle_all_dead == 0
    assert ROOM_5B_SPEC.enemy_types == (DARKNUT_OBJECT_TYPE,)
    assert ROOM_5B_SPEC.expected_enemy_count == 3
    assert ROOM_4B_SPEC.room_id == 0x4B
    assert ROOM_4B_SPEC.expected_enemy_count == 3
    assert ROOM_4B_SPEC.enemy_types == (ZOL_OBJECT_TYPE,)
    assert ROOM_5A_SPEC.room_id == 0x5A
    assert ROOM_5A_SPEC.enemy_types == (KEESE_OBJECT_TYPE,)
    assert ROOM_5A_SPEC.room_item_id == ROOM_ITEM_COMPASS


def test_live_zols_type_and_hp() -> None:
    snap = read_snapshot(_ram(room=ROOM_L3_WEST_KEY, zols=6, hp=32))
    assert len(ROOM_7B_SPEC.live_enemies(snap)) == 6
    snap_dead = read_snapshot(_ram(room=ROOM_L3_WEST_KEY, zols=6, hp=0))
    assert len(ROOM_7B_SPEC.live_enemies(snap_dead)) == 0


def test_key_success_predicate() -> None:
    assert level3_room_7b_key_success(
        _ram(room=ROOM_L3_WEST_KEY, keys=1, zols=0)
    )
    assert not level3_room_7b_key_success(
        _ram(room=ROOM_L3_WEST_KEY, keys=0, zols=0)
    )
    assert not level3_room_7b_key_success(
        _ram(room=ROOM_L3_WEST_KEY, keys=1, zols=2, hp=32)
    )
    assert not level3_room_7b_key_success(
        _ram(room=ROOM_L3_ENTRY, keys=1, zols=0)
    )


def test_6b_clear_and_5b_predicates() -> None:
    assert level3_room_6b_zols_cleared(
        _ram(room=ROOM_L3_NORTH_ZOLS, zols=0)
    )
    assert not level3_room_6b_zols_cleared(
        _ram(room=ROOM_L3_NORTH_ZOLS, zols=3, hp=32)
    )
    assert level3_reached_5b(_ram(room=ROOM_L3_DARKNUTS, darknuts=3))
    assert not level3_reached_5b(_ram(room=ROOM_L3_NORTH_ZOLS))
    assert level3_room_4b_zols_cleared(
        _ram(room=ROOM_L3_ZOL_KEY_4B, zols=0)
    )
    assert not level3_room_4b_zols_cleared(
        _ram(room=ROOM_L3_ZOL_KEY_4B, zols=2, hp=32)
    )


def test_has_raft_predicate() -> None:
    from zelda_i.ram import ADDR_RAFT

    ram = _ram(room=ROOM_L3_RAFT_PASSAGE)
    assert not level3_has_raft(ram)
    ram[ADDR_RAFT] = 1
    assert level3_has_raft(ram)


def test_west_door_step_mouth_and_align() -> None:
    mouth = west_door_step(read_snapshot(_ram(y=205, x=120)))
    assert mouth.reason == "west_leave_mouth"

    align = west_door_step(
        read_snapshot(_ram(y=WEST_DOOR_APPROACH_Y + 12, x=100))
    )
    assert align.reason == "west_align_y"

    approach = west_door_step(
        read_snapshot(_ram(y=WEST_DOOR_APPROACH_Y, x=100))
    )
    assert approach.reason == "west_approach"

    diagonal = west_door_step(
        read_snapshot(_ram(y=WEST_DOOR_APPROACH_Y, x=40))
    )
    assert diagonal.reason == "west_diagonal_push"


def test_north_door_7b_step_align_and_push() -> None:
    align = north_door_7b_step(
        read_snapshot(_ram(room=ROOM_L3_WEST_KEY, x=40, y=141))
    )
    assert align.reason == "north_align_x"

    push = north_door_7b_step(
        read_snapshot(
            _ram(room=ROOM_L3_WEST_KEY, x=NORTH_DOOR_X, y=141)
        )
    )
    assert push.reason == "north_push"

    arrived = north_door_7b_step(
        read_snapshot(_ram(room=ROOM_L3_NORTH_ZOLS, x=120, y=205))
    )
    assert arrived.reason == "north_arrived_6b"


def test_west_door_controller_arrives() -> None:
    ctrl = Level3WestDoorController()
    action = ctrl.step(
        read_snapshot(_ram(room=ROOM_L3_WEST_KEY, x=200, y=141))
    )
    assert ctrl.success
    assert action.reason == "west_arrived"


def test_north_door_7b_controller_arrives() -> None:
    ctrl = Level3NorthDoor7bController()
    action = ctrl.step(
        read_snapshot(_ram(room=ROOM_L3_NORTH_ZOLS, x=120, y=205))
    )
    assert ctrl.success
    assert action.reason == "north_arrived_6b"


def test_west_key_controller_hand_off_notes() -> None:
    ctrl = Level3WestKeyController()
    # Already in 0x7b with no enemies and keys — combat should mark reward.
    snap = read_snapshot(_ram(room=ROOM_L3_WEST_KEY, keys=0, zols=0, x=120, y=141))
    # Door phase first: success on 0x7b
    ctrl.step(snap)
    assert ctrl.phase in {"combat", "done", "failed"} or ctrl.door.success


def test_north_chain_already_in_5b() -> None:
    ctrl = Level3NorthChainController()
    ctrl.step(read_snapshot(_ram(room=ROOM_L3_DARKNUTS, x=120, y=205)))
    assert ctrl.success
    assert ctrl.phase == "done"


def test_raft_geometry_constants() -> None:
    assert KEY_DOOR_Y == 141
    assert STAIRS_69_RIGHT_Y == 141
    assert RAFT_CHANNEL_X == 176
    assert RAFT_PICKUP_X == 136
    assert RAFT_PICKUP_Y == 141
    assert RAFT_SOUTH_Y == 189
    assert ROOM_L3_WEST_DARKNUTS == 0x59
    assert ROOM_L3_SOUTH_DARKNUTS == 0x69
    assert ROOM_59_SPEC.room_id == 0x59
    assert ROOM_59_SPEC.expected_enemy_count == 5
    assert ROOM_59_SPEC.enemy_types == (DARKNUT_OBJECT_TYPE,)
    assert ROOM_69_SPEC.room_id == 0x69
    assert ROOM_69_SPEC.expected_enemy_count == 8
    assert "settle_5b" in RAFT_PATH_PHASES
    assert "passage_raft" in RAFT_PATH_PHASES
    assert "done" in RAFT_PATH_PHASES


def test_raft_passage_step_geometry() -> None:
    # Entry spawn north: go south.
    south = raft_passage_step(
        read_snapshot(
            _ram(room=ROOM_L3_RAFT_PASSAGE, x=48, y=77, mode=RAFT_PASSAGE_MODE)
        )
    )
    assert south.reason == "passage_to_south"

    # South band: go to channel x.
    channel = raft_passage_step(
        read_snapshot(
            _ram(
                room=ROOM_L3_RAFT_PASSAGE,
                x=48,
                y=RAFT_SOUTH_Y,
                mode=RAFT_PASSAGE_MODE,
            )
        )
    )
    assert channel.reason == "passage_to_channel"

    # At channel base: go up.
    up = raft_passage_step(
        read_snapshot(
            _ram(
                room=ROOM_L3_RAFT_PASSAGE,
                x=RAFT_CHANNEL_X,
                y=RAFT_SOUTH_Y,
                mode=RAFT_PASSAGE_MODE,
            )
        )
    )
    assert up.reason == "passage_channel_up"

    # Pickup band: go left to raft.
    left = raft_passage_step(
        read_snapshot(
            _ram(
                room=ROOM_L3_RAFT_PASSAGE,
                x=RAFT_CHANNEL_X,
                y=RAFT_PICKUP_Y,
                mode=RAFT_PASSAGE_MODE,
            )
        )
    )
    assert left.reason in {"passage_to_raft", "passage_raft_touch"}


def test_raft_path_controller_phases_and_raft_success() -> None:
    ctrl = Level3RaftPathController()
    assert ctrl.phase == "settle_5b"
    # Settle frames then leave west.
    for _ in range(45):
        ctrl.step(read_snapshot(_ram(room=ROOM_L3_DARKNUTS, x=120, y=205)))
    assert ctrl.phase == "left_to_5a"
    # Simulate arrival in compass room.
    ctrl.phase = "key_to_59"
    ctrl.step(
        read_snapshot(
            _ram(room=ROOM_L3_WEST_DARKNUTS, x=200, y=141, keys=0)
        )
    )
    assert ctrl.phase == "spawn_59"
    # Success when has_raft flag set.
    ctrl2 = Level3RaftPathController()
    action = ctrl2.step(
        read_snapshot(_ram(room=ROOM_L3_RAFT_PASSAGE, x=136, y=141)),
        has_raft=True,
    )
    assert ctrl2.success
    assert ctrl2.phase == "done"
    assert action.reason == "done"
    assert "raft_acquired" in ctrl2.notes


def test_boss_room_constants_and_predicates() -> None:
    assert ROOM_L3_BOSS_PREP == 0x5D
    assert ROOM_L3_BOSS == 0x4D
    assert ROOM_L3_BOMB_SHORTCUT == 0x5C
    assert MANHANDLA_OBJECT_TYPE == 0x3C
    assert LEVEL3_TRIFORCE_BIT == 0x04
    assert PASSAGE_EXIT_WAYPOINTS[0] == (176, 141)

    assert level3_reached_boss_prep(
        _ram(room=ROOM_L3_BOSS_PREP, x=32, y=141)
    )
    assert not level3_reached_boss_prep(
        _ram(room=ROOM_L3_BOMB_SHORTCUT, x=120, y=141)
    )
    assert level3_reached_boss(_ram(room=ROOM_L3_BOSS, x=120, y=141))
    assert not level3_reached_boss(
        _ram(room=ROOM_L3_BOSS_PREP, x=120, y=93)
    )


def test_boss_prep_killables_ignore_invuln_0x2b() -> None:
    """0x5d: clear Zol/Keese; invuln 0x2b must not count as killable."""
    ram = _ram(room=ROOM_L3_BOSS_PREP, x=120, y=141)
    # Slot 1: Zol live; slot 2: invuln 0x2b; slot 3: Keese type
    ram[ADDR_OBJ_TYPE + 1] = ZOL_OBJECT_TYPE
    ram[ADDR_OBJ_HP + 1] = 32
    ram[ADDR_OBJ_TYPE + 2] = 0x2B
    ram[ADDR_OBJ_HP + 2] = 240
    ram[ADDR_OBJ_TYPE + 3] = KEESE_OBJECT_TYPE
    ram[ADDR_OBJ_HP + 3] = 0  # keese often HP0 while alive
    snap = read_snapshot(ram)
    killable = level3_boss_prep_killables(snap)
    types = {o.type_id for o in killable}
    assert ZOL_OBJECT_TYPE in types
    assert KEESE_OBJECT_TYPE in types
    assert 0x2B not in types


def test_manhandla_live_heads() -> None:
    ram = _ram(room=ROOM_L3_BOSS, x=120, y=141)
    for slot, hp in ((1, 64), (2, 64), (3, 0)):
        ram[ADDR_OBJ_TYPE + slot] = MANHANDLA_OBJECT_TYPE
        ram[ADDR_OBJ_HP + slot] = hp
    heads = level3_manhandla_live(read_snapshot(ram))
    assert len(heads) == 2
    assert all(o.hp > 0 for o in heads)
