"""Unit tests for Level 3 boss path library (no emulator)."""

from __future__ import annotations

import numpy as np

from zelda_i.door_graph.core import DoorDir
from zelda_i.dungeon_ops import (
    DOOR_TARGETS,
    GEL_SPLIT_OBJECT_TYPE,
    NON_COMBAT_TYPES,
    live_killables,
    room_fields,
)
from zelda_i.level3_boss_path import (
    BOMB_NORTH_STANDS,
    BOSS_PATH_PHASES,
    Level3BossPathController,
    PREP_CLEAR_TYPES,
    UP_APPROACHES,
    prep_5d_still_killable,
)
from zelda_i.level3_dungeon import (
    BOMB_STAND_59_RIGHT,
    BOMB_STAND_5B_RIGHT,
    DOOR_5C_RIGHT_Y,
    INVULN_MOVER_0X2B,
    KEESE_OBJECT_TYPE,
    MANHANDLA_OBJECT_TYPE,
    PASSAGE_EXIT_WAYPOINTS,
    ROOM_L3_BOSS,
    ROOM_L3_BOSS_PREP,
    ZOL_OBJECT_TYPE,
    level3_manhandla_live,
)
from zelda_i.ram import (
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
    room: int = ROOM_L3_BOSS_PREP,
    x: int = 120,
    y: int = 141,
    mode: int = PLAY_MODE,
) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = mode
    ram[ADDR_LEVEL] = level
    ram[ADDR_SCREEN] = room
    ram[ADDR_LINK_X] = x
    ram[ADDR_LINK_Y] = y
    return ram


def test_controller_defaults_and_phases() -> None:
    ctl = Level3BossPathController()
    assert ctl.phase == "exit_passage"
    assert ctl.success is False
    assert ctl.failed is False
    assert ctl.poke_bombs is None  # durable default: no recon poke
    assert ctl.continuous_mode is False
    assert ctl.reached_5d is False
    assert ctl.reached_4d is False
    assert "exit_passage" in BOSS_PATH_PHASES
    assert "manhandla" in BOSS_PATH_PHASES
    assert "done" in BOSS_PATH_PHASES
    rep = ctl.report()
    assert rep["phase"] == "exit_passage"
    assert rep["poke_bombs"] is None
    assert rep["continuous_mode"] is False
    assert rep["intervention_class"] == "survival"
    assert list(BOMB_STAND_59_RIGHT) == rep["geometry"]["bomb_stand_59"]
    assert list(BOMB_STAND_5B_RIGHT) == rep["geometry"]["bomb_stand_5b"]
    assert rep["geometry"]["door_5c_right_y"] == DOOR_5C_RIGHT_Y


def test_constants_align_with_level3_dungeon() -> None:
    assert BOMB_STAND_59_RIGHT == (192, 141)
    assert BOMB_STAND_5B_RIGHT == (192, 141)
    assert DOOR_5C_RIGHT_Y == 141
    assert PASSAGE_EXIT_WAYPOINTS[0] == (176, 141)
    assert PASSAGE_EXIT_WAYPOINTS[-1] == (48, 77)
    assert len(PASSAGE_EXIT_WAYPOINTS) == 4
    assert UP_APPROACHES[0] == (120, 93)
    assert BOMB_NORTH_STANDS[0] == (120, 101)
    assert ZOL_OBJECT_TYPE in PREP_CLEAR_TYPES
    assert GEL_SPLIT_OBJECT_TYPE in PREP_CLEAR_TYPES
    assert KEESE_OBJECT_TYPE in PREP_CLEAR_TYPES
    assert INVULN_MOVER_0X2B not in PREP_CLEAR_TYPES
    assert INVULN_MOVER_0X2B in NON_COMBAT_TYPES
    assert DOOR_TARGETS["RIGHT"] == (208, 141)
    assert DoorDir.RIGHT == 0x01
    assert DoorDir.UP == 0x08


def test_prep_killables_ignore_0x2b_slots_1_12() -> None:
    ram = _ram(room=ROOM_L3_BOSS_PREP)
    ram[ADDR_OBJ_TYPE + 1] = ZOL_OBJECT_TYPE
    ram[ADDR_OBJ_HP + 1] = 32
    ram[ADDR_OBJ_TYPE + 2] = INVULN_MOVER_0X2B
    ram[ADDR_OBJ_HP + 2] = 240
    ram[ADDR_OBJ_TYPE + 3] = KEESE_OBJECT_TYPE
    ram[ADDR_OBJ_HP + 3] = 0
    # Gel residual in slot 11 (LIVE seal on UP shutter)
    ram[ADDR_OBJ_TYPE + 11] = GEL_SPLIT_OBJECT_TYPE
    ram[ADDR_OBJ_HP + 11] = 0
    snap = read_snapshot(ram)
    killable = prep_5d_still_killable(snap)
    types = {o.type_id for o in killable}
    slots = {o.slot for o in killable}
    assert ZOL_OBJECT_TYPE in types
    assert KEESE_OBJECT_TYPE in types
    assert GEL_SPLIT_OBJECT_TYPE in types
    assert INVULN_MOVER_0X2B not in types
    assert 11 in slots

    # live_killables with only darknuts must not pick 0x2b
    assert live_killables(snap, (0x0B,)) == []


def test_manhandla_live_heads_via_library() -> None:
    ram = _ram(room=ROOM_L3_BOSS)
    for slot, hp in ((1, 64), (2, 32), (3, 0)):
        ram[ADDR_OBJ_TYPE + slot] = MANHANDLA_OBJECT_TYPE
        ram[ADDR_OBJ_HP + slot] = hp
    heads = level3_manhandla_live(read_snapshot(ram))
    assert len(heads) == 2
    assert all(o.hp > 0 for o in heads)


def test_room_fields_door_bits_use_door_dir() -> None:
    ram = _ram(room=ROOM_L3_BOSS_PREP)
    # cur_opened_doors not in _ram helper — set via raw if needed;
    # room_fields still returns door dict keys.
    snap = read_snapshot(ram)
    fields = room_fields(snap, ram)
    assert fields["sc"] == "0x5d"
    assert set(fields["doors"]) == {"R", "L", "D", "U", "raw"}
    assert fields["level"] == 3


def test_controller_fail_sets_phase() -> None:
    ctl = Level3BossPathController(tag="unit")
    out = ctl._fail("unit_test_fail")
    assert out["ok"] is False
    assert ctl.failed is True
    assert ctl.phase == "failed"
    assert "unit_test_fail" in ctl.notes
