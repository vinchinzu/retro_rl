"""Dog pickup flag (game_state 0x0800) guards and recovery (rr-bu62.19)."""

import numpy as np

from harvest.planner.tasks import home_recover as hr
from harvest.tasks.nav import make_action

A = hr._A_INDEX
B = hr._B_INDEX


def _ram(gs=0x4401, player=(100, 100), dog=(110, 100), tilemap=0, held=0):
    from harvest.core.ram_catalog import field_spec, live_wram_base

    ram = np.zeros(0x24000, dtype=np.uint8)
    base = live_wram_base(ram)
    ram[0xD2], ram[0xD3] = gs & 0xFF, gs >> 8
    for name, val in (("tilemap", tilemap), ("player_x", player[0]), ("player_y", player[1])):
        spec = field_spec(name)
        idx = spec.address + (base if spec.live_offset else 0)
        ram[idx] = val & 0xFF
        if spec.kind == "u16":
            ram[idx + 1] = val >> 8
    idx = hr.DOG_POS_ADDR + base
    ram[idx], ram[idx + 1] = dog[0] & 0xFF, dog[0] >> 8
    ram[idx + 2], ram[idx + 3] = dog[1] & 0xFF, dog[1] >> 8
    return ram


def test_stuck_flag_detection():
    assert hr.dog_pickup_stuck(_ram(gs=0x4C01))
    assert not hr.dog_pickup_stuck(_ram(gs=0x4401))
    assert not hr.dog_pickup_stuck(_ram(gs=0x4C81))  # carrying: legitimate


def test_a_becomes_b_beside_dog():
    ram = _ram()
    out = hr.strip_dog_hug_press(make_action(a=True), ram)
    assert not out[A] and out[B]


def test_a_kept_when_dog_far_or_indoors():
    far = _ram(dog=(300, 300))
    assert hr.strip_dog_hug_press(make_action(a=True), far)[A]
    indoors = _ram(tilemap=0x15)
    assert hr.strip_dog_hug_press(make_action(a=True), indoors)[A]
