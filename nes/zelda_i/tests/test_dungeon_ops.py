"""B-item constants and owned-inventory pokes (no emulator)."""

from __future__ import annotations

from types import SimpleNamespace

from zelda_i.dungeon.ops import (
    ADDR_SELECTED_ITEM,
    B_ITEM_ARROWS,
    B_ITEM_BOMB,
    B_ITEM_BOMBS,
    B_ITEM_CANDLE,
    OWNED_INVENTORY_FIELDS,
    apply_owned_inventory,
    poke_bombs,
    poke_food,
    poke_keys,
)
from zelda_i.ram import ADDR_FOOD
from zelda_i.ram import ADDR_SELECTED_ITEM as RAM_SELECTED


class _AssignMem:
    def __init__(self) -> None:
        self.calls: list[tuple[int, str, int]] = []

    def assign(self, addr: int, fmt: str, val: int) -> None:
        self.calls.append((int(addr), fmt, int(val)))


def _env_with_mem(mem: object) -> SimpleNamespace:
    return SimpleNamespace(unwrapped=SimpleNamespace(data=SimpleNamespace(memory=mem)))


def test_poke_food_writes_only_the_food_byte() -> None:
    mem = _AssignMem()
    report = poke_food(_env_with_mem(mem), from_food=0)
    assert mem.calls == [(ADDR_FOOD, "|u1", 1)]
    assert report["food_writes"] == 1
    assert report["poke_food"] == 1
    assert report["progression_writes"] == 0
    assert report["capacity_writes"] == 0
    assert report["inventory_writes"] == 0
    assert report["bow_writes"] == 0
    assert report["state_load"] is False
    assert [w["field"] for w in report["writes"]] == ["food"]


def test_poke_food_skips_when_already_owned() -> None:
    mem = _AssignMem()
    report = poke_food(_env_with_mem(mem), from_food=1)
    assert mem.calls == []
    assert report["food_writes"] == 0
    assert report["poke_food"] == 0
    assert report["writes"] == []


def test_b_item_slot_is_bombs_1_arrows_2() -> None:
    assert ADDR_SELECTED_ITEM == RAM_SELECTED == 0x0656
    assert B_ITEM_BOMB == B_ITEM_BOMBS == 1
    assert B_ITEM_ARROWS == 2
    assert B_ITEM_CANDLE == 4


def test_poke_bombs_and_keys_use_data_set_value() -> None:
    values: dict[str, int] = {}

    class _Data:
        def set_value(self, key: str, value: int) -> None:
            values[key] = int(value)

    env = SimpleNamespace(unwrapped=SimpleNamespace(data=_Data()))
    assert poke_bombs(env, 16) == "bombs=16"
    assert poke_keys(env, 2) == "keys=2"
    assert values == {"bombs": 16, "keys": 2}


def test_apply_owned_inventory_tops_up_counts_and_selects_b() -> None:
    import numpy as np

    from zelda_i.ram import ADDR_BOMBS, ADDR_KEYS, ADDR_SELECTED_ITEM as RAM_B

    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_BOMBS] = 0
    ram[ADDR_KEYS] = 1
    ram[RAM_B] = 0
    values: dict[str, int] = {}

    class _Data:
        memory = None

        def set_value(self, key: str, value: int) -> None:
            values[key] = int(value)

    env = SimpleNamespace(
        get_ram=lambda: ram,
        unwrapped=SimpleNamespace(data=_Data(), em=None),
    )
    report = apply_owned_inventory(env, bombs=16, keys=2, select_bomb=True)
    assert report["poke_bombs"] == 16
    assert report["poke_keys"] == 2
    assert report["progression_writes"] == 0
    assert report["capacity_writes"] == 0
    assert values["bombs"] == 16
    assert values["keys"] == 2
    assert values["selected_item"] == B_ITEM_BOMB
    fields = {w["field"] for w in report["writes"]}
    assert fields == {"bombs", "keys", "selected_item"}


