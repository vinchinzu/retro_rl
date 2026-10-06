"""Open a reachable dungeon chest. The boomerang stand is CastleB1SecondKey."""

from __future__ import annotations

import numpy as np
import pytest

from alttp.dungeon.nav import open_free_chests
from alttp.paths import INTEGRATION_DIR
from alttp.ram import read_inventory
from alttp.scripts.pin_probe import open_pin


@pytest.mark.rom
def test_room_71_boomerang_chest_opens_in_play() -> None:
    pin = INTEGRATION_DIR / "CastleB1SecondKey.state"
    if not pin.is_file():
        pytest.skip("CastleB1SecondKey.state missing")
    env = open_pin("CastleB1SecondKey")
    try:
        before = read_inventory(np.asarray(env.get_ram(), dtype=np.uint8))
        assert before.boomerang == 0
        result = open_free_chests(env)
        after = read_inventory(np.asarray(env.get_ram(), dtype=np.uint8))
    finally:
        env.close()
    assert result.ok
    assert result.reason.startswith("opened ")
    assert after.boomerang == 1


@pytest.mark.rom
def test_room_71_chest_rejoins_rescue_from_real_predecessor() -> None:
    from alttp.opening_route.main_hall_to_zelda import run_from_b1_west

    pin = INTEGRATION_DIR / "Full_room_81.state"
    if not pin.is_file():
        pytest.skip("Full_room_81.state missing")
    env = open_pin("Full_room_81")
    try:
        result = run_from_b1_west(env)
        inventory = read_inventory(np.asarray(env.get_ram(), dtype=np.uint8))
    finally:
        env.close()
    phases = {phase.phase: phase for phase in result.phases}
    for label in ("open_chests", "rescue_alcove_nw", "alcove_wrap"):
        assert label in phases, result.blocker
        assert phases[label].ok, phases[label].detail
    assert inventory.boomerang == 1
    assert phases["alcove_wrap"].snapshot.link_x <= 688
    assert result.ok, result.blocker
    assert result.snapshot.has_zelda_follower
