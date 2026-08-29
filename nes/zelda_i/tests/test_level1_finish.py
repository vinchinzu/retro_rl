from __future__ import annotations

import numpy as np

from zelda_i.level1_finish import (
    ROOM_BOOMERANG_GORIYA,
    ROOM_TRIFORCE,
    AquamentusPhase,
    Backtrack44Phase,
    Level1BacktrackTo44Controller,
    Level1AquamentusController,
    Level1TriforceController,
)
from zelda_i.combat import FACING_EAST
from zelda_i.ram import (
    ADDR_HEALTH,
    ADDR_LEVEL,
    ADDR_LINK_FACING,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_SCREEN,
    ADDR_TRIFORCE,
    PLAY_MODE,
    read_snapshot,
)


def _ram(*, room: int, x: int = 112, y: int = 149):
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = PLAY_MODE
    ram[ADDR_LEVEL] = 1
    ram[ADDR_SCREEN] = room
    ram[ADDR_LINK_X] = x
    ram[ADDR_LINK_Y] = y
    ram[ADDR_HEALTH] = 0x20
    return ram


def test_backtrack_controller_detects_room44() -> None:
    controller = Level1BacktrackTo44Controller(
        phase=Backtrack44Phase.ENTER_44
    )
    action = controller.step(
        read_snapshot(_ram(room=ROOM_BOOMERANG_GORIYA, x=16, y=141))
    )
    assert controller.success is True
    assert action.reason == "done"


def test_triforce_controller_collects_shard_bit() -> None:
    controller = Level1TriforceController()
    ram = _ram(room=ROOM_TRIFORCE, x=128, y=141)
    ram[ADDR_TRIFORCE] = 1
    action = controller.step(read_snapshot(ram))
    assert controller.success is True
    assert action.reason == "done"


def _place_aquamentus(ram, *, x: int, y: int, hp: int = 0x60) -> None:
    ram[0x034F + 1] = 0x3D
    ram[0x0485 + 1] = hp
    ram[0x0070 + 1] = x
    ram[0x0084 + 1] = y


def _place_fireball(ram, *, x: int, y: int, slot: int = 9) -> None:
    ram[0x034F + slot] = 0x55
    ram[0x0070 + slot] = x
    ram[0x0084 + slot] = y


def test_aquamentus_tank_hits_ignores_fireball() -> None:
    controller = Level1AquamentusController(
        phase=AquamentusPhase.ATTACK,
        boss_seen=True,
        initial_health=0x2F,
        tank_hits=True,
    )
    ram = _ram(room=0x35, x=184, y=140)
    ram[ADDR_LINK_FACING] = FACING_EAST
    _place_aquamentus(ram, x=200, y=140)
    _place_fireball(ram, x=192, y=140)

    action = controller.step(read_snapshot(ram))

    assert action.reason == "attack_aquamentus"
    assert controller.phase is AquamentusPhase.ATTACK
