"""Unit tests for the Level 6 0x3A one-shot Link-position warp."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from zelda_i.level6.path import BLOCK_OBJECT_TYPE
from zelda_i.level6.stairs3a_warp import (
    WARP_XY,
    level6_stairs3a_warp_success,
    make_stairs_3a_warp_controller,
)
from zelda_i.ram import (
    ADDR_ARROWS,
    ADDR_BOMBS,
    ADDR_BOW,
    ADDR_COLLIDING_TILE,
    ADDR_KEYS,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_OBJ_TYPE,
    ADDR_ROD,
    ADDR_SCREEN,
    ADDR_TRIFORCE,
    PLAY_MODE,
    read_snapshot,
)


def _ram(**fields: int) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = fields.get("mode", PLAY_MODE)
    ram[ADDR_LEVEL] = fields.get("level", 6)
    ram[ADDR_SCREEN] = fields.get("screen", 0x3A)
    ram[ADDR_LINK_X] = fields.get("x", 144)
    ram[ADDR_LINK_Y] = fields.get("y", 141)
    ram[ADDR_TRIFORCE] = fields.get("triforce", 0x1F)
    ram[ADDR_KEYS] = fields.get("keys", 4)
    ram[ADDR_BOMBS] = fields.get("bombs", 8)
    ram[ADDR_COLLIDING_TILE] = fields.get("tile", 0)
    ram[ADDR_ROD] = fields.get("rod", 1)
    return ram


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


def test_leftover_still_clips_then_poke_after_push() -> None:
    from retro_harness.nes import nes_action, nes_idle_action

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
    pushed[ADDR_ROD] = 1
    _plant_block(pushed, 11, 112, 136)
    ctl.inner.block_slot = 11
    ctl.inner.block_x0 = 112
    ctl.inner.block_y0 = 144
    ctl.inner.phase = ctl.inner.phase.__class__.PUSH
    act = ctl.step(read_snapshot(pushed))
    assert act.reason == "position_write"
    assert list(act.action) == list(nes_idle_action())
    assert mem.calls == [
        (ADDR_LINK_X, "|u1", WARP_XY[0]),
        (ADDR_LINK_Y, "|u1", WARP_XY[1]),
    ]
    assert ctl.position_assist is not None
    assert ctl.position_assist["position_writes"] == 1
    assert ctl.position_assist["progression_writes"] == 0


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


def test_no_env_fails_closed_without_writing() -> None:
    leftover = _ram(level=6, screen=0x3A, x=112, y=160, keys=4)
    leftover[ADDR_ROD] = 1
    _plant_block(leftover, 11, 112, 136)
    ctl = make_stairs_3a_warp_controller()
    ctl.inner.block_slot = 11
    ctl.inner.block_x0 = 112
    ctl.inner.block_y0 = 144
    ctl.inner.phase = ctl.inner.phase.__class__.PUSH
    act = ctl.step(read_snapshot(leftover))
    assert ctl.failed
    assert act.reason == "no_env_for_position_write"
    assert ctl.position_assist is None
