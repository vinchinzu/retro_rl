"""Unit tests for the Level 6 0x3A south-band CheckWarp walk."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from zelda_i.level6.path import BLOCK_OBJECT_TYPE
from zelda_i.level6.stairs3a_warp import (
    EAST_COLUMN_X,
    SOUTH_BAND_Y,
    Stairs3AWarpPhase,
    level6_stairs3a_warp_success,
    make_stairs_3a_warp_controller,
)
from zelda_i.ram import (
    ADDR_ARROWS,
    ADDR_BOW,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_OBJ_TYPE,
    ADDR_ROD,
    PLAY_MODE,
    read_snapshot,
)
from zelda_i.tests.ram_helpers import make_ram

_DEFAULTS = {
    "mode": PLAY_MODE,
    "level": 6,
    "screen": 0x3A,
    "x": 144,
    "y": 141,
    "triforce": 0x1F,
    "keys": 4,
    "bombs": 8,
    "tile": 0,
    "rod": 1,
    "rupees": 0,
}


def _ram(**fields: int) -> np.ndarray:
    return make_ram(_DEFAULTS, **fields)


def _plant_block(ram: np.ndarray, slot: int, x: int, y: int) -> None:
    ram[ADDR_OBJ_TYPE + slot] = BLOCK_OBJECT_TYPE
    ram[ADDR_LINK_X + slot] = x
    ram[ADDR_LINK_Y + slot] = y


class _AssignMem:
    def __init__(self) -> None:
        self.calls: list[tuple[int, str, int]] = []

    def assign(self, addr: int, fmt: str, val: int) -> None:
        self.calls.append((addr, fmt, val))


def _env_with_mem(mem: object) -> SimpleNamespace:
    data = SimpleNamespace(memory=mem)
    return SimpleNamespace(unwrapped=SimpleNamespace(data=data))


def _arm_pushed(ctl, ram: np.ndarray) -> None:
    _plant_block(ram, 11, 112, 136)
    ctl.inner.block_slot = 11
    ctl.inner.block_x0 = 112
    ctl.inner.block_y0 = 144
    ctl.inner.phase = ctl.inner.phase.__class__.PUSH


def test_leftover_clips_then_peel_south_after_push() -> None:
    from retro_harness.nes import nes_action

    leftover = _ram(level=6, screen=0x3A, x=144, y=141, keys=4, tile=118)
    leftover[ADDR_BOW] = 0
    leftover[ADDR_ARROWS] = 0
    _plant_block(leftover, 11, 112, 144)
    ctl = make_stairs_3a_warp_controller()
    mem = _AssignMem()
    ctl.bind_env(_env_with_mem(mem))
    act = ctl.step(read_snapshot(leftover))
    assert act.reason in ("stand_path", "stand_clip")
    assert list(act.action) in (
        list(nes_action("LEFT")),
        list(nes_action("DOWN")),
        list(nes_action("LEFT", "DOWN")),
    )
    assert list(act.action) != list(nes_action("UP"))
    assert mem.calls == []

    pushed = _ram(level=6, screen=0x3A, x=112, y=160, keys=4, tile=116)
    _arm_pushed(ctl, pushed)
    act = ctl.step(read_snapshot(pushed))
    assert act.reason == "peel_south"
    assert list(act.action) == list(nes_action("DOWN"))
    assert list(act.action) != list(nes_action("UP"))
    assert mem.calls == []
    assert ctl.position_assist["position_writes"] == 0
    assert ctl.phase is Stairs3AWarpPhase.PEEL

    band = _ram(level=6, screen=0x3A, x=112, y=SOUTH_BAND_Y, keys=4)
    _plant_block(band, 11, 112, 136)
    act = ctl.step(read_snapshot(band))
    assert act.reason == "east_column"
    assert list(act.action) == list(nes_action("RIGHT"))
    assert mem.calls == []

    door_y = _ram(level=6, screen=0x3A, x=112, y=189, keys=4)
    _plant_block(door_y, 11, 112, 136)
    act = ctl.step(read_snapshot(door_y))
    assert act.reason == "east_column"
    assert list(act.action) == list(nes_action("RIGHT"))

    column = _ram(level=6, screen=0x3A, x=EAST_COLUMN_X, y=189, keys=4)
    _plant_block(column, 11, 112, 136)
    act = ctl.step(read_snapshot(column))
    assert act.reason == "column_up"
    assert list(act.action) == list(nes_action("UP"))
    assert mem.calls == []
    assert ctl.position_assist["position_writes"] == 0


def test_east_column_south_holds_up() -> None:
    from retro_harness.nes import nes_action

    ctl = make_stairs_3a_warp_controller()
    ctl.phase = Stairs3AWarpPhase.EAST
    ram = _ram(level=6, screen=0x3A, x=EAST_COLUMN_X, y=189)
    act = ctl.step(read_snapshot(ram))
    assert act.reason == "column_up"
    assert list(act.action) == list(nes_action("UP"))
    assert not ctl.failed


def test_screen_3b_fails_closed() -> None:
    ram = _ram(level=6, screen=0x3B, x=16, y=141, rupees=17)
    ctl = make_stairs_3a_warp_controller()
    act = ctl.step(read_snapshot(ram))
    assert ctl.failed
    assert not ctl.success
    assert "east_room_0x3b" in act.reason or "east_room_0x3b" in ctl.notes
    leftover = ctl.leftover
    assert leftover
    assert leftover["screen"] == 0x3B
    assert leftover["x"] == 16
    assert leftover["y"] == 141
    assert leftover["rupees"] == 17


def test_mode9_or_new_play_is_success_not_gohma_neighbors() -> None:
    cellar = _ram(level=6, screen=0x3A, x=208, y=93, mode=9, tile=0x71)
    cellar[ADDR_ROD] = 1
    assert level6_stairs3a_warp_success(read_snapshot(cellar))
    emerge = _ram(level=6, screen=0x0A, x=120, y=205)
    emerge[ADDR_ROD] = 1
    assert level6_stairs3a_warp_success(read_snapshot(emerge))
    still = _ram(level=6, screen=0x3A, x=144, y=141)
    still[ADDR_ROD] = 1
    assert not level6_stairs3a_warp_success(read_snapshot(still))
    north = _ram(level=6, screen=0x29, x=120, y=205)
    north[ADDR_ROD] = 1
    assert not level6_stairs3a_warp_success(read_snapshot(north))
    east = _ram(level=6, screen=0x3B, x=16, y=141)
    east[ADDR_ROD] = 1
    assert not level6_stairs3a_warp_success(read_snapshot(east))


def test_warp_and_gohma_do_not_call_poke_link_position() -> None:
    import inspect

    from zelda_i.level6 import gohma, stairs3a_warp

    for module in (stairs3a_warp, gohma):
        src = inspect.getsource(module)
        assert "poke_link_position" not in src
        assert "mem_write" not in src
        assert "memory.assign" not in src


def test_no_env_walks_without_writing() -> None:
    from retro_harness.nes import nes_action

    leftover = _ram(level=6, screen=0x3A, x=112, y=160, keys=4)
    leftover[ADDR_ROD] = 1
    ctl = make_stairs_3a_warp_controller()
    _arm_pushed(ctl, leftover)
    act = ctl.step(read_snapshot(leftover))
    assert not ctl.failed
    assert act.reason == "peel_south"
    assert list(act.action) == list(nes_action("DOWN"))
    assert ctl.position_assist["position_writes"] == 0
    assert ctl.leftover
    assert ctl.leftover["x"] == 112
    assert ctl.leftover["y"] == 160
    assert "rupees" in ctl.leftover


def test_already_pushed_transitions_to_peel() -> None:
    from retro_harness.nes import nes_action
    from zelda_i.level6.stairs3a_warp import is_center_block_pushed

    # Block already at NE stairs (208, 96)
    pushed = _ram(level=6, screen=0x3A, x=107, y=141, keys=4)
    _plant_block(pushed, 11, 208, 96)
    snap = read_snapshot(pushed)
    assert is_center_block_pushed(snap)

    ctl = make_stairs_3a_warp_controller()
    act = ctl.step(snap)
    assert not ctl.failed
    assert ctl.phase is Stairs3AWarpPhase.PEEL
    assert act.reason == "peel_south"
    assert list(act.action) == list(nes_action("DOWN"))


def test_east_door_region_does_not_abort_peel() -> None:
    from retro_harness.nes import nes_action

    # Link at (200, 141) with block pushed: should peel south, not abort
    ram = _ram(level=6, screen=0x3A, x=200, y=141, keys=4)
    _plant_block(ram, 11, 208, 96)
    ctl = make_stairs_3a_warp_controller()
    act = ctl.step(read_snapshot(ram))
    assert not ctl.failed
    assert ctl.phase is Stairs3AWarpPhase.PEEL
    assert act.reason == "peel_south"
    assert list(act.action) == list(nes_action("DOWN"))


