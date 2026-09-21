"""Unit tests for the L2 Survival-spine 0x7d → Magical Boomerang table."""

from __future__ import annotations

import numpy as np

from retro_harness.controls import pressed_nes_buttons
from zelda_i.dungeon.engine import BLUE_GORIYA_OBJECT_TYPE, FIREBALL_OBJECT_TYPE
from zelda_i.level2.dungeon import ROOM_4F_SPEC
from zelda_i.level2.spine import (
    Clear4fPhase,
    Level2BacktrackTo7dController,
    Level2Clear4fController,
    Level2Enter6fKeyController,
    Level2NavPhase,
)
from zelda_i.ram import (
    ADDR_BOMBS,
    ADDR_HEALTH,
    ADDR_KEYS,
    ADDR_LEVEL,
    ADDR_LINK_FACING,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MAGIC_BOOMERANG,
    ADDR_MODE,
    ADDR_OBJ_HP,
    ADDR_OBJ_TYPE,
    ADDR_SCREEN,
    PLAY_MODE,
    read_snapshot,
)


def _snap(
    *,
    room: int,
    x: int = 120,
    y: int = 141,
    keys: int = 2,
    bombs: int = 0,
    mode: int = PLAY_MODE,
    boom: int = 0,
    goriya: tuple[int, int, int] | None = None,
    fireball: tuple[int, int] | None = None,
):
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = mode
    ram[ADDR_LEVEL] = 2
    ram[ADDR_SCREEN] = room
    ram[ADDR_LINK_X] = x
    ram[ADDR_LINK_Y] = y
    ram[ADDR_KEYS] = keys
    ram[ADDR_BOMBS] = bombs
    ram[ADDR_HEALTH] = 0x2F
    ram[ADDR_MAGIC_BOOMERANG] = boom
    if goriya is not None:
        gx, gy, hp = goriya
        ram[ADDR_OBJ_TYPE + 1] = BLUE_GORIYA_OBJECT_TYPE
        ram[ADDR_LINK_X + 1] = gx
        ram[ADDR_LINK_Y + 1] = gy
        ram[ADDR_OBJ_HP + 1] = hp
    if fireball is not None:
        fx, fy = fireball
        ram[ADDR_OBJ_TYPE + 2] = FIREBALL_OBJECT_TYPE
        ram[ADDR_LINK_X + 2] = fx
        ram[ADDR_LINK_Y + 2] = fy
        ram[ADDR_LINK_FACING + 2] = 0x02
    return read_snapshot(ram)


def test_backtrack_7d_recenters_live_timeout_pose() -> None:
    """survival_spine_l2_boom timed out in 0x6c at (128, 133) with ALIGN_TOL=6."""
    ctl = Level2BacktrackTo7dController()
    act = ctl.step(_snap(room=0x6C, x=128, y=133))
    assert act.reason == "door_align_y"
    assert pressed_nes_buttons(list(act.action)) == ["DOWN"]
    act = ctl.step(_snap(room=0x6C, x=136, y=136))
    assert act.reason == "door_align_y"
    assert pressed_nes_buttons(list(act.action)) == ["DOWN"]
    act = ctl.step(_snap(room=0x6C, x=136, y=141))
    assert act.reason == "door_push"
    assert pressed_nes_buttons(list(act.action)) == ["RIGHT"]


def test_enter_6f_fails_without_keys() -> None:
    ctl = Level2Enter6fKeyController()
    act = ctl.step(_snap(room=0x6E, keys=0))
    assert ctl.phase is Level2NavPhase.FAILED
    assert act.reason == "no_keys"
    assert ctl.success is False
    pushing = Level2Enter6fKeyController()
    pushing.door_phase = "push"
    act = pushing.step(_snap(room=0x6E, x=208, y=141, keys=0))
    assert pushing.phase is Level2NavPhase.WALK
    assert act.reason == "push_r"


def test_enter_6f_south_occupancy_sidesteps_diamonds() -> None:
    """Live timeout sat at (72, 181) then (112, 181); greedy UP hits diamonds."""
    ctl = Level2Enter6fKeyController()
    snap = _snap(room=0x6E, x=112, y=181, keys=2)
    first = ctl.step(snap)
    assert first.reason == "band_occ"
    first_dir = pressed_nes_buttons(list(first.action))
    second = ctl.step(snap)
    assert ctl.walker.misses == 1
    assert second.reason == "band_occ"
    assert pressed_nes_buttons(list(second.action)) != first_dir
    east_south = Level2Enter6fKeyController()
    east_snap = _snap(room=0x6E, x=200, y=181, keys=2)
    act = east_south.step(east_snap)
    assert act.reason == "band_occ"
    assert "RIGHT" not in pressed_nes_buttons(list(act.action))
    east_south.step(east_snap)
    assert east_south.walker.misses == 1


def test_room_4f_spec_occupancy_and_backstep() -> None:
    assert ROOM_4F_SPEC.combat.occupancy_patrol is True
    assert ROOM_4F_SPEC.combat.contact_backstep == 16


def test_clear4f_miss_blocks_ahead_and_replans() -> None:
    ctl = Level2Clear4fController()
    snap = _snap(room=0x4F, x=120, y=141, goriya=(120, 100, 0x50))
    first = ctl.step(snap)
    assert first.reason == "combat_approach"
    first_dir = pressed_nes_buttons(list(first.action))
    assert "UP" in first_dir
    second = ctl.step(snap)
    assert ctl.walker.misses == 1
    assert (120, 140) in ctl.walker.grid.blocked
    assert second.reason in {"combat_approach", "combat_wait"}
    if second.reason == "combat_approach":
        assert pressed_nes_buttons(list(second.action)) != first_dir


def test_clear4f_stands_when_no_path() -> None:
    ctl = Level2Clear4fController()
    grid = ctl.walker.grid
    for x in range(100, 141):
        grid.blocked.add((x, 170))
        grid.blocked.add((x, 190))
    for y in range(170, 191):
        grid.blocked.add((100, y))
        grid.blocked.add((140, y))
    snap = _snap(room=0x4F, x=120, y=181, goriya=(120, 109, 0x50))
    act = ctl.step(snap)
    assert act.reason == "combat_wait"
    assert pressed_nes_buttons(list(act.action)) == []
    assert ctl.walker.last_dir is None


def test_clear4f_peels_fireball_and_contact() -> None:
    balls = Level2Clear4fController()
    peel = balls.step(_snap(room=0x4F, x=120, y=141, fireball=(130, 141)))
    assert peel.reason == "ball_peel"
    assert "LEFT" in pressed_nes_buttons(list(peel.action))
    contact = Level2Clear4fController()
    back = contact.step(_snap(room=0x4F, x=120, y=141, goriya=(125, 141, 0x50)))
    assert back.reason == "combat_backstep"
    assert "LEFT" in pressed_nes_buttons(list(back.action))


def test_clear4f_does_not_poke() -> None:
    ctl = Level2Clear4fController()
    assert ctl.report()["poke"] is False
    assert ctl.report()["route_eligible"] is False
    done = ctl.step(_snap(room=0x4F, boom=1, goriya=(120, 100, 0x50)))
    assert ctl.phase is Clear4fPhase.DONE
    assert done.reason == "done"
