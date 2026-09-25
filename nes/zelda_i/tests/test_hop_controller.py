"""HopController timeout / death / scroll guard. No emulator."""

from __future__ import annotations

import dataclasses

import numpy as np

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.dungeon.hop_controller import (
    DEATH_MODE,
    LADDER_STILL_FRAMES,
    HopController,
    LadderEscape,
    LatticeDoorWalker,
    ladder_release,
)
from zelda_i.dungeon.ids import STEPLADDER_OBJECT_TYPE
from zelda_i.ram import (
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_SCREEN,
    PLAY_MODE,
    ZeldaObject,
    ZeldaSnapshot,
    read_snapshot,
)
from zelda_i.tests.ram_helpers import room_tile_env


class _DestHop(HopController):
    require_level = 6

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        return snap.screen == 0x3B and snap.mode == PLAY_MODE

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        del snap
        return FrameAction(nes_action("RIGHT"), "go")


def _snap(*, mode: int = PLAY_MODE, level: int = 6, screen: int = 0x3A) -> ZeldaSnapshot:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = mode
    ram[ADDR_LEVEL] = level
    ram[ADDR_SCREEN] = screen
    ram[ADDR_LINK_X] = 120
    ram[ADDR_LINK_Y] = 141
    return read_snapshot(ram)


def test_policy_runs_on_play_and_arrives() -> None:
    hop = _DestHop(max_frames=20)
    act = hop.step(_snap())
    assert list(act.action) == list(nes_action("RIGHT"))
    assert hop.success is False
    done = hop.step(_snap(screen=0x3B))
    assert hop.success is True
    assert list(done.action) == list(nes_idle_action())


def test_scroll_waits_and_death_fails() -> None:
    hop = _DestHop(max_frames=20)
    wait = hop.step(_snap(mode=6))
    assert list(wait.action) == list(nes_idle_action())
    assert hop.failed is False
    dead = hop.step(_snap(mode=DEATH_MODE))
    assert hop.failed is True
    assert list(dead.action) == list(nes_idle_action())


def test_timeout_fails_closed() -> None:
    hop = _DestHop(max_frames=2)
    hop.step(_snap())
    timed = hop.step(_snap())
    assert hop.failed is True
    assert hop.success is False
    assert list(timed.action) == list(nes_idle_action())


def test_wait_not_play_idles_until_play() -> None:
    hop = _DestHop(max_frames=20)
    waited = hop.wait_not_play(_snap(mode=11))
    assert waited is not None
    assert list(waited.action) == list(nes_idle_action())
    assert waited.reason == "wait_mode_11"
    assert hop.wait_not_play(_snap()) is None


# --- LadderEscape: L6 0x19, Blue Ring power-on 3 --------------------------
# Captured map: a full-height water column at x 160..175. The patrol walked
# Link UP the column's west edge on a vertical ladder to (152,173), where only
# DOWN moves him, and the old one-frame back-off swapped 2 px with the
# patrol's UP for 11778 of 15000 frames.


def _ladder_snap(x: int, y: int, *, ladder: tuple[int, int, int] | None) -> ZeldaSnapshot:
    snap = _snap(screen=0x19)
    objects = ()
    if ladder is not None:
        lx, ly, heading = ladder
        objects = (ZeldaObject(11, STEPLADDER_OBJECT_TYPE, lx, ly, heading, 64, 2),)
    return dataclasses.replace(snap, link_x=x, link_y=y, objects=objects)


def _pressed(act: FrameAction) -> str:
    names = ("UP", "DOWN", "LEFT", "RIGHT")
    return next(n for n, i in zip(names, (4, 5, 6, 7)) if act.action[i])


def test_square_on_ladder_turns_a_sideways_press_back_the_way_link_came() -> None:
    snap = _ladder_snap(152, 173, ladder=(152, 176, 0x08))
    assert ladder_release(snap, "LEFT") == "DOWN"
    assert ladder_release(snap, "UP") == "UP"  # along the axis: unchanged


def test_escape_backs_off_a_still_ladder_and_holds_until_off_it() -> None:
    env = room_tile_env("0x19", level=6)
    esc = LadderEscape()
    stuck = _ladder_snap(152, 173, ladder=(152, 176, 0x08))
    up = FrameAction(nes_action("UP"), "combat_patrol")
    for _ in range(LADDER_STILL_FRAMES):
        assert esc.filter(stuck, up, env=env).reason == "combat_patrol"
    act = esc.filter(stuck, up, env=env)
    assert (act.reason, _pressed(act)) == ("ladder_back_off", "DOWN")
    # Moving again, still on the ladder: the latch holds against the patrol.
    act = esc.filter(_ladder_snap(152, 175, ladder=(152, 176, 0x08)), up, env=env)
    assert (act.reason, _pressed(act)) == ("ladder_back_off", "DOWN")
    assert esc.escapes == 1


def test_escape_lands_sideways_on_a_walkable_node_then_hands_back() -> None:
    env = room_tile_env("0x19", level=6)
    esc = LadderEscape()
    up = FrameAction(nes_action("UP"), "combat_patrol")
    stuck = _ladder_snap(152, 173, ladder=(152, 176, 0x08))
    for _ in range(LADDER_STILL_FRAMES + 1):
        esc.filter(stuck, up, env=env)
    # The ladder is gone at the bottom row, but x=152 still straddles water.
    act = esc.filter(_ladder_snap(152, 189, ladder=None), up, env=env)
    assert (act.reason, _pressed(act)) == ("ladder_land", "LEFT")
    act = esc.filter(_ladder_snap(146, 189, ladder=None), up, env=env)
    assert (act.reason, _pressed(act)) == ("ladder_land", "LEFT")
    assert esc.filter(_ladder_snap(144, 189, ladder=None), up, env=env) is up


def test_escape_leaves_a_crossing_and_a_standing_policy_alone() -> None:
    env = room_tile_env("0x19", level=6)
    esc = LadderEscape()
    right = FrameAction(nes_action("RIGHT"), "combat_engage")
    for x in range(147, 176, 2):  # a moving horizontal crossing
        act = esc.filter(_ladder_snap(x, 125, ladder=(160, 128, 0x01)), right, env=env)
        assert list(act.action) == list(nes_action("RIGHT"))
    idle = FrameAction(nes_idle_action(), "combat_wait")
    stuck = _ladder_snap(152, 173, ladder=(152, 176, 0x08))
    for _ in range(3 * LADDER_STILL_FRAMES):
        assert list(esc.filter(stuck, idle, env=env).action) == list(nes_idle_action())
    assert esc.escapes == 0


def test_lattice_door_walker_latches_ladder_release_step_until_off_ladder() -> None:
    env = room_tile_env("0x19", level=6)
    walker = LatticeDoorWalker()
    snap = _ladder_snap(152, 173, ladder=(152, 176, 0x08))
    act = walker.action(env, snap, "LEFT", "exit")
    assert _pressed(act) == "DOWN"
    assert walker.ladder_step == "DOWN"

    snap_mid = _ladder_snap(152, 175, ladder=(152, 176, 0x08))
    act_mid = walker.action(env, snap_mid, "LEFT", "exit")
    assert _pressed(act_mid) == "DOWN"

    snap_off = _ladder_snap(144, 189, ladder=None)
    act_off = walker.action(env, snap_off, "LEFT", "exit")
    assert walker.ladder_step is None

