from __future__ import annotations

import numpy as np

from zelda_i.level1_finish import (
    ROOM_BOOMERANG_GORIYA,
    ROOM_GEL_SWITCH,
    ROOM_KEY_GORIYA,
    ROOM_MAP,
    ROOM_OLD_MAN,
    ROOM_TRIFORCE,
    AquamentusPhase,
    Backtrack44Phase,
    Level1BacktrackTo44Controller,
    Level1AquamentusController,
    Level1Room42ExitController,
    Level1TriforceController,
    Room42ExitPhase,
)
from zelda_i.combat import FACING_EAST
from zelda_i.ram import (
    ADDR_CUR_OPENED_DOORS,
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


def _ram(*, room: int, x: int = 112, y: int = 149, doors: int = 4):
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = PLAY_MODE
    ram[ADDR_LEVEL] = 1
    ram[ADDR_SCREEN] = room
    ram[ADDR_LINK_X] = x
    ram[ADDR_LINK_Y] = y
    ram[ADDR_HEALTH] = 0x20
    ram[ADDR_CUR_OPENED_DOORS] = doors
    return ram


def test_room42_controller_pushes_block_then_enters_hint() -> None:
    controller = Level1Room42ExitController()
    action = controller.step(
        read_snapshot(_ram(room=ROOM_GEL_SWITCH, x=104, y=162))
    )
    assert action.reason == "align_switch_block_y"

    action = controller.step(read_snapshot(_ram(room=ROOM_GEL_SWITCH)))
    assert action.reason == "push_center_block"

    action = controller.step(
        read_snapshot(_ram(room=ROOM_GEL_SWITCH, y=133, doors=6))
    )
    assert controller.phase is Room42ExitPhase.ENTER_HINT
    assert action.reason == "align_hint_door"


def test_room42_controller_waits_in_hint_room() -> None:
    controller = Level1Room42ExitController(
        phase=Room42ExitPhase.ENTER_HINT
    )
    action = controller.step(read_snapshot(_ram(room=ROOM_OLD_MAN)))
    assert controller.phase is Room42ExitPhase.WAIT_HINT
    assert action.reason == "settle_hint"


def test_room42_controller_detects_map_room() -> None:
    controller = Level1Room42ExitController(
        phase=Room42ExitPhase.ENTER_MAP
    )
    action = controller.step(read_snapshot(_ram(room=ROOM_MAP)))
    assert controller.success is True
    assert controller.phase is Room42ExitPhase.DONE
    assert action.reason == "done"


def test_backtrack_controller_starts_on_room23_route() -> None:
    controller = Level1BacktrackTo44Controller()
    action = controller.step(
        read_snapshot(_ram(room=ROOM_KEY_GORIYA, x=136, y=117))
    )
    assert action.reason == "route_room23_south"


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


def test_aquamentus_controller_reacts_to_nearby_fireball() -> None:
    controller = Level1AquamentusController(
        phase=AquamentusPhase.ATTACK,
        boss_seen=True,
        initial_health=0x20,
    )
    ram = _ram(room=0x35, x=128, y=125)
    _place_aquamentus(ram, x=160, y=128)
    _place_fireball(ram, x=144, y=125)

    action = controller.step(read_snapshot(ram))

    assert controller.phase is AquamentusPhase.DODGE
    assert action.reason == "dodge_fireball"


def test_aquamentus_controller_closes_on_east_parked_boss() -> None:
    controller = Level1AquamentusController(
        phase=AquamentusPhase.ATTACK,
        boss_seen=True,
        initial_health=0x2F,
        tank_hits=True,
    )
    ram = _ram(room=0x35, x=128, y=140)
    _place_aquamentus(ram, x=200, y=140)

    action = controller.step(read_snapshot(ram))

    assert action.reason == "align_boss_stance"
    assert controller.phase is AquamentusPhase.ALIGN
    assert controller.last_boss == (200, 140)


def test_aquamentus_controller_swings_in_wooden_sword_range() -> None:
    controller = Level1AquamentusController(
        phase=AquamentusPhase.ATTACK,
        boss_seen=True,
        initial_health=0x2F,
        tank_hits=True,
    )
    ram = _ram(room=0x35, x=184, y=140)
    ram[ADDR_LINK_FACING] = FACING_EAST
    _place_aquamentus(ram, x=200, y=140)

    action = controller.step(read_snapshot(ram))

    assert action.reason == "attack_aquamentus"
    assert controller.phase is AquamentusPhase.ATTACK
    assert controller.attack_frames == 1


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


def test_aquamentus_controller_accepts_heart_container() -> None:
    controller = Level1AquamentusController(
        phase=AquamentusPhase.COLLECT_HEART,
        boss_seen=True,
        initial_health=0x20,
    )
    action = controller.step(
        read_snapshot(_ram(room=0x35, x=192, y=141))
    )
    assert action.reason == "wait_heart_pickup"

    ram = _ram(room=0x35, x=192, y=141)
    ram[ADDR_HEALTH] = 0x31
    action = controller.step(read_snapshot(ram))
    assert controller.success is True
    assert action.reason == "done"


def test_aquamentus_heart_detects_container_nibble() -> None:
    controller = Level1AquamentusController(
        phase=AquamentusPhase.COLLECT_HEART,
        boss_seen=True,
        initial_health=0x2F,
        initial_containers=3,
    )
    ram = _ram(room=0x35, x=192, y=141)
    ram[ADDR_HEALTH] = 0x3F
    action = controller.step(read_snapshot(ram))
    assert controller.success is True
    assert action.reason == "done"
