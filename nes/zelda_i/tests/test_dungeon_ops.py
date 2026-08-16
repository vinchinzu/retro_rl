"""B-item constants and ensure_bomb fallbacks (no emulator)."""

from __future__ import annotations

from types import SimpleNamespace

from zelda_i.dungeon_ops import (
    ADDR_SELECTED_ITEM,
    B_ITEM_ARROWS,
    B_ITEM_BOMB,
    B_ITEM_BOMBS,
    B_ITEM_CANDLE,
    OWNED_INVENTORY_FIELDS,
    ensure_bomb,
    poke_bombs,
    poke_keys,
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


def test_owned_inventory_fields_are_counts_and_b_slot() -> None:
    assert OWNED_INVENTORY_FIELDS == frozenset({"bombs", "keys", "selected_item"})
    assert "magical_boomerang" not in OWNED_INVENTORY_FIELDS
    assert "triforce" not in OWNED_INVENTORY_FIELDS
    assert "max_bombs" not in OWNED_INVENTORY_FIELDS


def test_poke_bombs_and_keys_use_data_set_value() -> None:
    values: dict[str, int] = {}

    class _Data:
        def set_value(self, key: str, value: int) -> None:
            values[key] = int(value)

    env = SimpleNamespace(unwrapped=SimpleNamespace(data=_Data()))
    assert poke_bombs(env, 16) == "bombs=16"
    assert poke_keys(env, 2) == "keys=2"
    assert values == {"bombs": 16, "keys": 2}
