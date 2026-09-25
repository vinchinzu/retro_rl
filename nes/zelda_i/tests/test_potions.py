"""Potion drink guard and the refill hold: logic without an emulator.

The live proof is ``tests/rom/test_potions.py``. These pin the two contracts a
ROM run cannot isolate cheaply: no potion means the guard is invisible, and a
held refill writes nothing while an unheld one still refills.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pytest

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action
from zelda_i.assist import LastHeartAssist
from zelda_i.dungeon.pause_select import (
    B_SLOT_ARROWS,
    B_SLOT_BOMBS,
    B_SLOT_BOOMERANG,
    B_SLOT_CANDLE,
    B_SLOT_POTION,
    PotionDrinkGuard,
    b_slot_owned,
    lethal_hit_hearts,
    potion_drink_window,
    potion_due,
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


def test_a_retry_keeps_the_b_item_from_before_the_first_try() -> None:
    # An ignored B press aborts with the potion still selected; the next try
    # must still put the candle back (CL63: 0x5B's burn press drank instead).
    env = _Env()
    env.ram[ADDR_SELECTED_ITEM] = B_SLOT_POTION
    guard = PotionDrinkGuard(inner=_Inner())
    guard.bind_env(env)
    guard._prior = B_SLOT_CANDLE
    guard.step(_snap(potion=2, health=0x50))
    assert guard._prior == B_SLOT_CANDLE


def _foe(type_id: int, slot: int = 1) -> ZeldaObject:
    return ZeldaObject(slot=slot, type_id=type_id, x=80, y=141, facing=0, hp=4, state=0)


@pytest.mark.parametrize(
    ("foes", "ring", "hearts"),
    [
        ((), 0, 0.0),
        ((_foe(0x07),), 0, 0.5),  # octorok
        ((_foe(0x07), _foe(0x30, slot=2)), 0, 2.0),  # a gibdo is the worst
        ((_foe(0x30),), 1, 1.0),  # the blue ring halves it
        ((_foe(0x30),), 2, 0.5),  # the red ring halves it twice
        ((_foe(0x60), _foe(0x64, slot=11)), 0, 0.0),  # a floor drop, a cave trigger
    ],
)
def test_lethal_hit_is_the_worst_rom_damage_after_the_ring(foes, ring, hearts) -> None:
    snap = replace(_snap(), objects=(_snap().objects[0], *foes), ring=ring)
    assert lethal_hit_hearts(snap) == hearts


@pytest.mark.parametrize(
    ("health", "partial", "foe", "due"),
    [
        (0x51, 0xFF, 0x07, False),  # two hearts, an octorok: not yet
        (0x51, 0xFF, 0x30, True),  # two hearts, a gibdo takes two
        (0x51, 0x40, 0x30, True),  # 1.25 hearts
        (0x52, 0x40, 0x30, False),  # 2.25 hearts outlast one gibdo hit
        (0x50, 0xFF, 0x07, True),  # the last heart is still the floor
    ],
)
def test_potion_is_due_before_the_hit_that_kills(health, partial, foe, due) -> None:
    snap = replace(
        _snap(health=health), heart_partial=partial, objects=(_snap().objects[0], _foe(foe))
    )
    assert potion_due(snap) is due
    assert potion_due(replace(snap, potion=0)) is False


def test_a_due_drink_holds_off_swings_until_the_window() -> None:
    @dataclass
    class _Swinger(_Inner):
        def step(self, snap: ZeldaSnapshot) -> FrameAction:
            self.steps += 1
            return FrameAction(nes_action("LEFT", "A"), "swing")

    inner = _Swinger()
    guard = PotionDrinkGuard(inner=inner)
    guard.bind_env(_Env())
    action = guard.step(_snap(potion=2, health=0x50, link_state=0x11))  # mid-swing
    assert inner.steps == 1
    assert action.action == nes_action("LEFT")


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


def test_restock_skips_while_a_charge_hides_the_unshown_letter() -> None:
    # The potion shares the letter's subscreen slot: with a take-any charge
    # and the letter never shown there is nothing the shop can be shown.
    from zelda_i.overworld.cave_shop import restock_item

    def snap(letter: int) -> ZeldaSnapshot:
        return ZeldaSnapshot(
            mode=PLAY_MODE, level=0, screen=0x39, next_screen=0x39, link_x=120, link_y=141,
            facing=8, sword=2, bombs=4, rupees=113, keys=0, health=0x77, triforce=3,
            compass=0, dialog_timer=0, colliding_tile=0, room_item_id=0, room_all_dead=0,
            room_obj_count=0, cur_opened_doors=0, open_doorway_mask=0, objects=(),
            letter=letter, potion=1,
        )

    assert restock_item(snap(letter=1)) is None
    assert restock_item(snap(letter=2)) == "red"


def test_restock_keeps_a_reserve_for_the_next_buy() -> None:
    """Before L7 the wallet also owes the 60R Bait: red only if 60R remain
    after it, else blue, else nothing (run 29 arrived at 0x64 with 101R)."""
    from zelda_i.overworld.cave_shop import restock_item

    def snap(potion: int, rupees: int) -> ZeldaSnapshot:
        return ZeldaSnapshot(
            mode=PLAY_MODE, level=0, screen=0x64, next_screen=0x64, link_x=112, link_y=93,
            facing=8, sword=2, bombs=4, rupees=rupees, keys=0, health=0x77, triforce=0x3F,
            compass=0, dialog_timer=0, colliding_tile=0, room_item_id=0, room_all_dead=0,
            room_obj_count=0, cur_opened_doors=0, open_doorway_mask=0, objects=(),
            letter=2, potion=potion,
        )

    assert restock_item(snap(0, 101), reserve=60) == "blue"
    assert restock_item(snap(0, 128), reserve=60) == "red"
    assert restock_item(snap(0, 99), reserve=60) is None
    assert restock_item(snap(1, 128), reserve=60) == "red"
    assert restock_item(snap(1, 127), reserve=60) is None
    assert restock_item(snap(0, 40)) == "blue"
    assert restock_item(snap(2, 255)) is None
