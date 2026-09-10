"""Unit tests for Level 3 boss path library (no emulator)."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.dungeon.bomb_wall import BombWallPhase
from zelda_i.dungeon.door_hop import door_band_goal
from zelda_i.dungeon.ops import (
    GEL_SPLIT_OBJECT_TYPE,
    live_killables,
)
from zelda_i.level3.boss_path import (
    RIGHT_5C_SPEC,
    UP_5D_SPEC,
    L3DoorHopController,
    Level3BossPathController,
    Level3ManhandlaController,
    level3_boss_suffix_stages,
    make_l3_bomb_5b,
    prep_5d_still_killable,
)
from zelda_i.level3.dungeon import (
    INVULN_MOVER_0X2B,
    KEESE_OBJECT_TYPE,
    MANHANDLA_OBJECT_TYPE,
    ROOM_L3_BOMB_SHORTCUT,
    ROOM_L3_BOSS,
    ROOM_L3_BOSS_PREP,
    ROOM_L3_DARKNUTS,
    ROOM_L3_TF,
    ZOL_OBJECT_TYPE,
)
from zelda_i.ram import (
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_OBJ_HP,
    ADDR_OBJ_TYPE,
    ADDR_TRIFORCE,
    PLAY_MODE,
    read_snapshot,
)
from zelda_i.tests.ram_helpers import make_ram


_DEFAULTS = {
    "mode": PLAY_MODE,
    "level": 3,
    "x": 120,
    "y": 141,
    "triforce": 0x03,
    "bombs": 4,
    "keys": 1,
    "health": 0x22,
}


def _ram(
    *,
    level: int = 3,
    room: int = ROOM_L3_BOSS_PREP,
    x: int = 120,
    y: int = 141,
    mode: int = PLAY_MODE,
    **fields: int,
) -> np.ndarray:
    ram = make_ram(
        _DEFAULTS, mode=mode, level=level, screen=room, x=x, y=y, **fields
    )
    return ram


def test_prep_killables_ignore_0x2b_slots_1_12() -> None:
    ram = _ram(room=ROOM_L3_BOSS_PREP)
    ram[ADDR_OBJ_TYPE + 1] = ZOL_OBJECT_TYPE
    ram[ADDR_OBJ_HP + 1] = 32
    ram[ADDR_OBJ_TYPE + 2] = INVULN_MOVER_0X2B
    ram[ADDR_OBJ_HP + 2] = 240
    ram[ADDR_OBJ_TYPE + 3] = KEESE_OBJECT_TYPE
    ram[ADDR_OBJ_HP + 3] = 0
    # Gel residual in slot 11 (LIVE seal on UP shutter)
    ram[ADDR_OBJ_TYPE + 11] = GEL_SPLIT_OBJECT_TYPE
    ram[ADDR_OBJ_HP + 11] = 0
    snap = read_snapshot(ram)
    killable = prep_5d_still_killable(snap)
    types = {o.type_id for o in killable}
    slots = {o.slot for o in killable}
    assert ZOL_OBJECT_TYPE in types
    assert KEESE_OBJECT_TYPE in types
    assert GEL_SPLIT_OBJECT_TYPE in types
    assert INVULN_MOVER_0X2B not in types
    assert 11 in slots

    # live_killables with only darknuts must not pick 0x2b
    assert live_killables(snap, (0x0B,)) == []


def test_continuous_controller_forbids_state_restore() -> None:
    ctl = Level3BossPathController(continuous_mode=True)
    em = SimpleNamespace(set_state=lambda state: (_ for _ in ()).throw(AssertionError))
    with pytest.raises(RuntimeError, match="forbids"):
        ctl._restore_state(SimpleNamespace(em=em), object())
    assert ctl.state_restores == 0


def test_path_to_5d_has_no_5b_return_fight_clear() -> None:
    src = inspect.getsource(Level3BossPathController.path_to_5d)
    assert "inspect_5b_return" not in src
    assert "clear_5b_return" not in src
    assert "failed_clear_5b_return" not in src
    assert "make_l3_bomb_5b" in src
    assert "DoorHopController" in src
    assert "push_dir(" not in src
    assert "idle(env, assist, total, 60)" not in src
    assert "idle(env, assist, total, 40)" not in src
    assert "idle(env, assist, total, 110)" not in src


def _snap(*, screen: int, **fields: int):
    return read_snapshot(_ram(room=screen, **fields))


def test_right_5c_leftover_relative_goal() -> None:
    """Wave 0 door_band_goal: on-band leftover keeps y; south mouth uses door y."""
    goal = (208, 141)
    assert door_band_goal("RIGHT", (120, 143), goal) == (208, 143)
    assert door_band_goal("RIGHT", (120, 181), goal) == (208, 141)
    assert door_band_goal("RIGHT", (208, 141), goal) == (208, 141)
    on_band = _snap(screen=ROOM_L3_BOMB_SHORTCUT, x=120, y=143)
    ctl = L3DoorHopController(RIGHT_5C_SPEC)
    first = ctl.step(on_band)
    assert ctl.goal == (208, 143)
    assert list(first.action) != list(nes_idle_action())
    south = _snap(screen=ROOM_L3_BOMB_SHORTCUT, x=120, y=181)
    ctl2 = L3DoorHopController(RIGHT_5C_SPEC)
    ctl2.step(south)
    assert ctl2.goal == (208, 141)


def test_up_5d_off_column_binds_door_x() -> None:
    assert door_band_goal("UP", (208, 157), (120, 93)) == (120, 109)
    assert door_band_goal("UP", (118, 157), (120, 93)) == (118, 109)
    leftover = _snap(screen=ROOM_L3_BOSS_PREP, x=208, y=157)
    ctl = L3DoorHopController(UP_5D_SPEC)
    first = ctl.step(leftover)
    assert ctl.goal[0] == 120
    assert list(first.action) != list(nes_idle_action())


@pytest.mark.parametrize("idle_frames", [0, 30, 90, 223])
def test_right_5c_jitter_still_arrives_dest(idle_frames: int) -> None:
    """Entry-frame jitter is not the hop; dest is RAM 0x5d play."""
    ctl = L3DoorHopController(RIGHT_5C_SPEC)
    wait = _snap(screen=ROOM_L3_BOMB_SHORTCUT, x=120, y=141, mode=6)
    for _ in range(idle_frames):
        act = ctl.step(wait)
        assert not ctl.success
        assert not ctl.failed
        assert list(act.action) == list(nes_action("RIGHT"))
    origin = _snap(screen=ROOM_L3_BOMB_SHORTCUT, x=120, y=141)
    ctl.step(origin)
    dest = _snap(screen=ROOM_L3_BOSS_PREP, x=32, y=141)
    done = ctl.step(dest)
    assert ctl.success
    assert done.reason.startswith("arrived_")


def test_bomb_5b_zero_bombs_fails_without_poke() -> None:
    ctl = make_l3_bomb_5b()
    assert getattr(ctl, "select_item", None) == 1
    empty = _snap(screen=ROOM_L3_DARKNUTS, x=192, y=141, bombs=0)
    ctl.step(empty)
    assert ctl.phase is BombWallPhase.FAILED
    assert any("no_bombs" in n for n in ctl.notes)


def test_bomb_5b_at_stand_faces_right() -> None:
    ctl = make_l3_bomb_5b()
    stand = _snap(screen=ROOM_L3_DARKNUTS, x=192, y=141, bombs=4)
    act = ctl.step(stand)
    assert ctl.phase is not BombWallPhase.FAILED
    assert act.reason == "face_right"
    assert list(act.action) == list(nes_action("RIGHT"))


def test_bomb_5b_off_stand_walks_to_dest() -> None:
    ctl = make_l3_bomb_5b()
    inland = _snap(screen=ROOM_L3_DARKNUTS, x=120, y=141, bombs=4)
    act = ctl.step(inland)
    assert ctl.phase is not BombWallPhase.FAILED
    assert list(act.action) != list(nes_idle_action())


def test_bomb_5b_inland_176_125_drops_to_waist() -> None:
    """Live leftover (176,125) must y-first to stand y=141, not RIGHT into a tile."""
    ctl = make_l3_bomb_5b()
    leftover = _snap(screen=ROOM_L3_DARKNUTS, x=176, y=125, bombs=10)
    act = ctl.step(leftover)
    assert ctl.phase is not BombWallPhase.FAILED
    assert act.reason in {"approach_y", "south_band"}
    assert list(act.action) == list(nes_action("DOWN"))


def _plant_manhandla(ram: np.ndarray, *, x: int = 128, y: int = 112) -> None:
    ram[ADDR_OBJ_TYPE + 1] = MANHANDLA_OBJECT_TYPE
    ram[ADDR_OBJ_HP + 1] = 64
    ram[ADDR_LINK_X + 1] = x
    ram[ADDR_LINK_Y + 1] = y


def test_manhandla_south_mouth_climbs() -> None:
    ram = _ram(room=ROOM_L3_BOSS, x=120, y=205, bombs=4)
    _plant_manhandla(ram)
    ctl = Level3ManhandlaController()
    act = ctl.step(read_snapshot(ram))
    assert not ctl.failed
    assert act.reason in {"climb", "approach", "fight_up"}
    assert list(act.action) == list(nes_action("UP"))


def test_manhandla_near_head_places_bomb() -> None:
    ram = _ram(room=ROOM_L3_BOSS, x=128, y=125, bombs=4)
    _plant_manhandla(ram, x=128, y=112)
    ctl = Level3ManhandlaController()
    reasons = [ctl.step(read_snapshot(ram)).reason for _ in range(8)]
    assert not ctl.failed
    assert any(r in {"place_bomb", "approach", "fight_up"} for r in reasons)


def test_manhandla_tf_bit_is_dest() -> None:
    ram = _ram(room=ROOM_L3_TF, x=120, y=141, triforce=0x07)
    ram[ADDR_TRIFORCE] = 0x07
    ctl = Level3ManhandlaController()
    act = ctl.step(read_snapshot(ram))
    assert ctl.success
    assert act.reason == "tf04"


def test_clean_suffix_stages_do_not_poke_inventory() -> None:
    names = []
    for name, ctl, max_frames in level3_boss_suffix_stages():
        names.append(name)
        assert max_frames > 0
        assert getattr(ctl, "poke_bombs", None) in (None, False)
        assert getattr(ctl, "route_eligible", False) is False
        src = inspect.getsource(type(ctl))
        assert "poke_bombs(" not in src
        assert "poke_keys(" not in src
    assert names == [
        "bomb_5b",
        "clear_5c",
        "right_5d",
        "clear_5d",
        "up_4d",
        "manhandla_tf",
    ]
    src = inspect.getsource(level3_boss_suffix_stages)
    assert "poke_bombs" not in src
    assert "make_l3_bomb_5b" in src
