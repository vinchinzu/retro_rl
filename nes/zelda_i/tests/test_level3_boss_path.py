"""Unit tests for Level 3 boss path library (no emulator)."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from zelda_i.dungeon_ops import (
    GEL_SPLIT_OBJECT_TYPE,
    live_killables,
)
from zelda_i.level3_boss_path import (
    Level3BossPathController,
    prep_5d_still_killable,
)
from zelda_i.level3_dungeon import (
    INVULN_MOVER_0X2B,
    KEESE_OBJECT_TYPE,
    ROOM_L3_BOSS_PREP,
    ZOL_OBJECT_TYPE,
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


def test_continuous_controller_forbids_state_restore() -> None:
    ctl = Level3BossPathController(continuous_mode=True)
    em = SimpleNamespace(set_state=lambda state: (_ for _ in ()).throw(AssertionError))
    with pytest.raises(RuntimeError, match="forbids"):
        ctl._restore_state(SimpleNamespace(em=em), object())
    assert ctl.state_restores == 0


def test_path_to_5d_has_no_5b_return_fight_clear() -> None:
    src = inspect.getsource(Level3BossPathController.path_to_5d)
    assert "inspect_5b_return" not in src
    assert "clear_5b_return" not in src
    assert "failed_clear_5b_return" not in src
    assert "BOMB_STAND_5B_RIGHT" in src
    assert 'bomb_stand(env, assist, total, "RIGHT", bx, by)' in src
