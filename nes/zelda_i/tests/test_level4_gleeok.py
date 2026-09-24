"""Unit tests for Level 4 Gleeok TF-exit: no restore on the Survival spine."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from zelda_i.dungeon.gleeok import FIREBALL_DODGE_DIST
from zelda_i.level4.boss_combat import (
    APPROACH_SOUTH_Y,
    _room_item_heart_xy,
    FIREBALL_DODGE_DIST_LOW_HP,
    LOW_HP_THRESHOLD,
    Level4GleeokFightController,
    approach_dodge_thr,
    approach_goal,
    make_gleeok_fight_controller,
)
from zelda_i.level4.gleeok13 import run_level4_tf_suffix
from zelda_i.level4.spine import run_level4_entrance_tf
from zelda_i.level4.occupancy import ROOM_13_SPAWN_XY


def test_continuous_gleeok_forbids_state_restore() -> None:
    ctl = Level4GleeokFightController(continuous_mode=True)
    em = SimpleNamespace(
        set_state=lambda state: (_ for _ in ()).throw(AssertionError)
    )
    with pytest.raises(RuntimeError, match="forbids"):
        ctl._restore_state(SimpleNamespace(em=em), object())
    assert ctl.state_restores == 0


def test_lab_gleeok_restore_counts_set_state() -> None:
    calls: list[object] = []
    em = SimpleNamespace(set_state=lambda state: calls.append(state))
    ctl = make_gleeok_fight_controller()
    marker = object()
    ctl._restore_state(SimpleNamespace(em=em), marker)
    assert calls == [marker]
    assert ctl.state_restores == 1


def test_entrance_tf_runner_skips_ow_and_bomb_topup() -> None:
    src = inspect.getsource(run_level4_entrance_tf)
    assert "level4-entry" in src
    assert "topup" not in src
    from zelda_i.level4.spine import _clear_31_stages

    names = [name for name, _, _ in _clear_31_stages()]
    assert names[-1] == "level4_leave_0x31"
    assert '"route_eligible": False' in src
    assert "allow_pokes = False" in src
    assert "LEVEL4_TRIFORCE_BIT" in src


def test_spine_l4_tf_suffix_uses_continuous_mode() -> None:
    src = inspect.getsource(run_level4_tf_suffix)
    assert "continuous_mode=True" in src
    assert "make_gleeok_fight_controller" in src


def test_approach_dodge_widens_only_before_stand_on_low_hp() -> None:
    assert approach_dodge_thr(start_health=98, approached=False) == (
        FIREBALL_DODGE_DIST_LOW_HP
    )
    assert approach_dodge_thr(start_health=98, approached=True) == (
        FIREBALL_DODGE_DIST
    )
    assert approach_dodge_thr(
        start_health=LOW_HP_THRESHOLD, approached=False
    ) == FIREBALL_DODGE_DIST
    assert FIREBALL_DODGE_DIST_LOW_HP > FIREBALL_DODGE_DIST


def test_approach_goal_drops_south_before_body_x() -> None:
    sx, sy = ROOM_13_SPAWN_XY
    assert approach_goal(sx, sy, 160) == (sx, APPROACH_SOUTH_Y)
    assert approach_goal(sx, APPROACH_SOUTH_Y, 160) == (160, APPROACH_SOUTH_Y)
    assert approach_goal(120, APPROACH_SOUTH_Y, None) == (124, APPROACH_SOUTH_Y)


def test_gleeok_run_does_not_call_set_state_directly() -> None:
    restore_src = inspect.getsource(Level4GleeokFightController._restore_state)
    assert "env.em.set_state(state)" in restore_src
    run_src = inspect.getsource(Level4GleeokFightController.run)
    assert "env.em.set_state" not in run_src
    assert "_restore_state" in run_src
    assert "if self.continuous_mode:" in run_src


def test_hc_hunt_reads_the_room_item_slot() -> None:
    import numpy as np

    from zelda_i.ram import ADDR_ROOM_ITEM_ID, ADDR_ROOM_ITEM_X, ADDR_ROOM_ITEM_Y

    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_ROOM_ITEM_X], ram[ADDR_ROOM_ITEM_Y] = 208, 192
    ram[ADDR_ROOM_ITEM_ID] = 0x1A  # heart container, bottom-right of 0x13
    assert _room_item_heart_xy(ram) == (208, 192)
    ram[ADDR_ROOM_ITEM_ID] = 0x1B  # triforce: not the container any more
    assert _room_item_heart_xy(ram) is None


def test_0x12_push_stand_walks_the_lattice_around_the_block_pair(monkeypatch) -> None:
    """Power-on leftover (160,134) beside the (136..152, 133..141) blocks:
    the x-first hand walk pressed LEFT into them for 8000 frames."""
    from retro_harness.nes import nes_action
    from zelda_i.level4.gleeok13 import make_gleeok13_controller
    from zelda_i.ram import (
        ADDR_LEVEL,
        ADDR_LINK_X,
        ADDR_LINK_Y,
        ADDR_MODE,
        ADDR_SCREEN,
        PLAY_MODE,
        read_snapshot,
    )
    from zelda_i.tests.ram_helpers import room_tile_env
    from zelda_i.walk import live_env

    env = room_tile_env("0x12", level=4)
    ram = env.get_ram()
    ram[ADDR_LEVEL], ram[ADDR_SCREEN], ram[ADDR_MODE] = 4, 0x12, PLAY_MODE
    ram[ADDR_LINK_X], ram[ADDR_LINK_Y] = 160, 134
    monkeypatch.setattr(live_env, "_ENV", env)
    act = make_gleeok13_controller().step(read_snapshot(ram))
    assert act.reason == "stand_lattice"
    assert list(act.action) != list(nes_action("LEFT"))
