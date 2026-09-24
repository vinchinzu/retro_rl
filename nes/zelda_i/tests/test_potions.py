"""Potion drink guard and the refill hold: logic without an emulator.

The live proof is ``tests/rom/test_potions.py``. These pin the two contracts a
ROM run cannot isolate cheaply: no potion means the guard is invisible, and a
held refill writes nothing while an unheld one still refills.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action
from zelda_i.assist import LastHeartAssist
from zelda_i.dungeon.pause_select import (
    B_SLOT_ARROWS,
    B_SLOT_BOMBS,
    B_SLOT_BOOMERANG,
    B_SLOT_POTION,
    PotionDrinkGuard,
    b_slot_owned,
    potion_drink_window,
)
from zelda_i.ram import (
    ADDR_ARROWS,
    ADDR_BOMBS,
    ADDR_BOOMERANG,
    ADDR_BOW,
    ADDR_MAGIC_BOOMERANG,
    ADDR_POTION,
    ADDR_SELECTED_ITEM,
    CAVE_MODE,
    PLAY_MODE,
    ZeldaObject,
    ZeldaSnapshot,
)


def _snap(
    *, mode: int = PLAY_MODE, health: int = 0x50, potion: int = 2, link_state: int = 0,
    updating: int = 1,
) -> ZeldaSnapshot:
    link = ZeldaObject(slot=0, type_id=0, x=112, y=141, facing=8, hp=0, state=link_state)
    return ZeldaSnapshot(
        mode=mode, level=0, screen=0x64, next_screen=0x64, link_x=112, link_y=141,
        facing=8, sword=2, bombs=0, rupees=0, keys=0, health=health, triforce=0,
        compass=0, dialog_timer=0, colliding_tile=0x26, room_item_id=0,
        room_all_dead=0, room_obj_count=0, cur_opened_doors=0, open_doorway_mask=0,
        objects=(link,), potion=potion, is_updating_mode=updating,
    )


class _Env:
    def __init__(self) -> None:
        self.ram = np.zeros(0x800, dtype=np.uint8)

    def get_ram(self) -> np.ndarray:
        return self.ram


@dataclass
class _Inner:
    steps: int = 0
    success: bool = False
    notes: list[str] = field(default_factory=list)

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.steps += 1
        return FrameAction(nes_action("LEFT"), "inner")


class _Data:
    def __init__(self) -> None:
        self.values: dict[str, int] = {}

    def set_value(self, key: str, value: int) -> None:
        self.values[key] = int(value)


@pytest.mark.parametrize(
    ("potion", "health"),
    [(0, 0x50), (2, 0x51), (1, 0x55)],
)
def test_guard_is_invisible_without_a_potion_or_above_the_last_heart(potion, health) -> None:
    inner = _Inner()
    guard = PotionDrinkGuard(inner=inner)
    guard.bind_env(_Env())
    snap = _snap(potion=potion, health=health)
    actions = [guard.step(snap) for _ in range(5)]
    assert inner.steps == 5
    assert all(a.action == nes_action("LEFT") for a in actions)
    assert guard.holds_refill(snap) is False


def test_guard_takes_the_frame_at_the_last_heart_with_a_potion() -> None:
    env = _Env()
    env.ram[ADDR_SELECTED_ITEM] = B_SLOT_POTION
    inner = _Inner()
    guard = PotionDrinkGuard(inner=inner)
    guard.bind_env(env)
    snap = _snap(potion=2, health=0x50)
    action = guard.step(snap)
    assert inner.steps == 0
    assert action.action == nes_action("B")  # potion already on B: drink at once
    assert guard.holds_refill(snap) is True


@pytest.mark.parametrize(
    ("kwargs", "paused", "menu", "ok"),
    [
        ({}, 0, 0, True),
        ({"mode": 7}, 0, 0, False),  # scroll
        ({"mode": 16}, 0, 0, False),  # stairs / cave mouth
        ({"mode": CAVE_MODE}, 0, 0, False),  # cave text
        ({"link_state": 0x11}, 0, 0, False),  # mid-swing
        ({"link_state": 0x40}, 0, 0, False),  # halted by text
        ({"updating": 0}, 0, 0, False),  # mode init
        ({}, 2, 0, False),  # a refill already running
        ({}, 0, 7, False),  # inventory open
    ],
)
def test_drink_window_only_in_settled_play(kwargs, paused, menu, ok) -> None:
    assert potion_drink_window(_snap(**kwargs), paused=paused, menu=menu) is ok


def test_b_slot_owned_reads_the_slot_item_with_the_two_exceptions() -> None:
    ram = np.zeros(0x800, dtype=np.uint8)
    assert not b_slot_owned(ram, B_SLOT_BOMBS)
    ram[ADDR_BOMBS] = 3
    assert b_slot_owned(ram, B_SLOT_BOMBS)
    ram[ADDR_POTION] = 1
    assert b_slot_owned(ram, B_SLOT_POTION)
    ram[ADDR_ARROWS] = 1
    assert not b_slot_owned(ram, B_SLOT_ARROWS)  # arrows need the bow
    ram[ADDR_BOW] = 1
    assert b_slot_owned(ram, B_SLOT_ARROWS)
    assert not b_slot_owned(ram, B_SLOT_BOOMERANG)
    ram[ADDR_MAGIC_BOOMERANG] = 1
    assert b_slot_owned(ram, B_SLOT_BOOMERANG)
    ram[ADDR_MAGIC_BOOMERANG] = 0
    ram[ADDR_BOOMERANG] = 1
    assert b_slot_owned(ram, B_SLOT_BOOMERANG)


def test_held_refill_writes_nothing_and_an_unheld_one_still_refills() -> None:
    assist = LastHeartAssist()
    data = _Data()
    hold = {"on": True}
    assist.refill_hold = lambda snap: hold["on"]
    last = _snap(health=0x50)
    assist.apply_snapshot(data, _snap(health=0x55), frame=0)
    assist.apply_snapshot(data, last, frame=1)
    assert data.values == {}
    assert assist.report()["refill_holds"] == 1
    hold["on"] = False
    assist.apply_snapshot(data, last, frame=2)
    assert data.values.get("health") == 0x55
    assert assist.telemetry.health.writes == 1


def test_restock_buys_only_when_short_and_affordable() -> None:
    """Between dungeons the 0x64 stop is free when there is nothing to buy:
    the stage ends on its first frame; otherwise it walks to the shop."""
    from zelda_i.overworld.cave_shop import make_potion_restock_controller
    from zelda_i.overworld.graph import ScreenHop

    hops = (ScreenHop(0x64, "RIGHT"),)

    def first(potion: int, rupees: int):
        ctrl = make_potion_restock_controller(hops=hops)
        snap = ZeldaSnapshot(
            mode=PLAY_MODE, level=0, screen=0x74, next_screen=0x74, link_x=120, link_y=141,
            facing=8, sword=2, bombs=4, rupees=rupees, keys=0, health=0x77, triforce=7,
            compass=0, dialog_timer=0, colliding_tile=0, room_item_id=0, room_all_dead=0,
            room_obj_count=0, cur_opened_doors=0, open_doorway_mask=0, objects=(),
            letter=2, potion=potion,
        )
        ctrl.step(snap)
        return ctrl

    # Full red, too poor for blue, or a blue with no red money (a second
    # blue is not a second drink): nothing to buy.
    for potion, rupees in ((2, 200), (0, 39), (1, 67)):
        assert first(potion=potion, rupees=rupees).success, (potion, rupees)
    for potion, rupees in ((0, 40), (1, 68), (0, 255)):
        ctrl = first(potion=potion, rupees=rupees)
        assert not ctrl.success and not getattr(ctrl, "failed", False), (potion, rupees)
