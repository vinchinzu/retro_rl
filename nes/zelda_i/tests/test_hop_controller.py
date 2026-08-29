"""HopController timeout / death / scroll guard. No emulator."""

from __future__ import annotations

import numpy as np

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.dungeon.hop_controller import DEATH_MODE, HopController
from zelda_i.ram import (
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_SCREEN,
    PLAY_MODE,
    ZeldaSnapshot,
    read_snapshot,
)


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
