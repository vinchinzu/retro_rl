"""B-item constants and ensure_bomb fallbacks (no emulator)."""

from __future__ import annotations

from types import SimpleNamespace

from zelda_i.dungeon_ops import (
    ADDR_SELECTED_ITEM,
    B_ITEM_ARROWS,
    B_ITEM_BOMB,
    B_ITEM_BOMBS,
    B_ITEM_CANDLE,
    ensure_bomb,
)
from zelda_i.level2_puzzles import B_ITEM_BOMB_PROBE
from zelda_i.level9_ganon import B_ITEM_ARROWS as L9_ARROWS
from zelda_i.level9_ganon import B_ITEM_BOMBS as L9_BOMBS
from zelda_i.ram import ADDR_SELECTED_ITEM as RAM_SELECTED


def test_b_item_slot_is_bombs_1_arrows_2() -> None:
    assert ADDR_SELECTED_ITEM == RAM_SELECTED == 0x0656
    assert B_ITEM_BOMB == B_ITEM_BOMBS == B_ITEM_BOMB_PROBE == 1
    assert B_ITEM_ARROWS == 2
    assert B_ITEM_CANDLE == 4
    assert L9_BOMBS is B_ITEM_BOMBS
    assert L9_ARROWS is B_ITEM_ARROWS


class _AssignMem:
    def __init__(self) -> None:
        self.calls: list[tuple[int, str, int]] = []

    def assign(self, addr: int, fmt: str, val: int) -> None:
        self.calls.append((addr, fmt, val))


class _SetByteMem:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def set_byte(self, addr: int, val: int) -> None:
        self.calls.append((addr, val))


def _env_with_mem(mem: object) -> SimpleNamespace:
    data = SimpleNamespace(memory=mem)
    return SimpleNamespace(unwrapped=SimpleNamespace(data=data))


def test_ensure_bomb_prefers_memory_assign() -> None:
    mem = _AssignMem()
    assert ensure_bomb(_env_with_mem(mem)) == "selected_item=bomb"
    assert mem.calls == [(ADDR_SELECTED_ITEM, "|u1", B_ITEM_BOMB)]


def test_ensure_bomb_falls_back_to_set_byte() -> None:
    mem = _SetByteMem()
    assert ensure_bomb(_env_with_mem(mem)) == "selected_item=bomb"
    assert mem.calls == [(ADDR_SELECTED_ITEM, B_ITEM_BOMB)]
