"""World adapter + detect_world unit tests."""

from __future__ import annotations

import numpy as np

from smz3.ram import (
    SM_ADDR_DOOR_TRANSITION,
    SM_ADDR_GAME_STATE,
    SM_ADDR_HEALTH,
    SM_ADDR_ROOM_ID,
    Z3_ADDR_MODULE,
    Z3_ADDR_SUBMODULE,
    read_snapshot,
)
from smz3.world import (
    ActiveWorld,
    DualWorldSessionHooks,
    context_for,
    detect_world,
    detect_world_stub,
)


def _poke(ram: np.ndarray, addr: int, value: int, width: int = 2) -> None:
    if width == 1:
        ram[addr] = value & 0xFF
    else:
        ram[addr] = value & 0xFF
        ram[addr + 1] = (value >> 8) & 0xFF


def test_context_packages() -> None:
    sm = context_for(ActiveWorld.SUPER_METROID)
    assert sm.package == "super_metroid"
    z3 = context_for(ActiveWorld.ALTTP)
    assert z3.package == "alttp"
    assert context_for(ActiveWorld.UNKNOWN).package is None


def test_detect_none_is_unknown() -> None:
    assert detect_world(None) is ActiveWorld.UNKNOWN
    assert detect_world_stub(None) is ActiveWorld.UNKNOWN


def test_detect_sm_controllable() -> None:
    ram = np.zeros(0x2000, dtype=np.uint8)
    _poke(ram, SM_ADDR_GAME_STATE, 8)
    _poke(ram, SM_ADDR_ROOM_ID, 0x91F8)
    _poke(ram, SM_ADDR_DOOR_TRANSITION, 0)
    _poke(ram, SM_ADDR_HEALTH, 99)
    ram[Z3_ADDR_MODULE] = 151
    assert detect_world(ram) is ActiveWorld.SUPER_METROID
    assert detect_world(read_snapshot(ram)) is ActiveWorld.SUPER_METROID


def test_detect_sm_title_menu() -> None:
    ram = np.zeros(0x2000, dtype=np.uint8)
    _poke(ram, SM_ADDR_GAME_STATE, 1)  # opening / title
    assert detect_world(ram) is ActiveWorld.MENU


def test_detect_sm_file_select() -> None:
    ram = np.zeros(0x2000, dtype=np.uint8)
    _poke(ram, SM_ADDR_GAME_STATE, 4)
    assert detect_world(ram) is ActiveWorld.MENU


def test_detect_alttp_overworld() -> None:
    ram = np.zeros(0x2000, dtype=np.uint8)
    _poke(ram, SM_ADDR_GAME_STATE, 0xCCCC)  # not an SM engine state
    ram[Z3_ADDR_MODULE] = 0x09
    ram[Z3_ADDR_SUBMODULE] = 0x00
    assert detect_world(ram) is ActiveWorld.ALTTP


def test_detect_alttp_dungeon() -> None:
    ram = np.zeros(0x2000, dtype=np.uint8)
    _poke(ram, SM_ADDR_GAME_STATE, 0xFFFF)
    ram[Z3_ADDR_MODULE] = 0x07
    ram[Z3_ADDR_SUBMODULE] = 0x00
    assert detect_world(ram) is ActiveWorld.ALTTP


def test_dual_session_plan() -> None:
    plan = DualWorldSessionHooks(seed_name="test_seed", bots=2).plan()
    assert plan["bots"] == 2
    assert plan["record_video"] is True
    assert plan["room_timeout_multiplier"] == 3.0
    assert "super_metroid" in plan["world_packages"].values()
