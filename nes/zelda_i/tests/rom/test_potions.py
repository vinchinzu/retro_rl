"""Potions on the live ROM: buy at 0x64 after the letter, drink at the last heart.

Start pin ``GatherChain_exit_ring``: the gather chain's pose after the Blue
Ring (0x34, out of the cave), the 0x0E letter taken (``$0666`` 1), the Blue
Candle on B (``$0656`` 4). That is where the potion stage splices in: the
ring return walks 0x44, 0x54, 0x64. The pin's wallet is 0 (it predates the
hidden-rupee chain, forecast ~100R after the ring), so every test makes ONE
disclosed what-if write first: ``$066D`` = 100. The buy then walks to 0x64,
shows the letter inside, buys and puts the candle back on B.

The drink tests buy the same way, walk out, and make one more what-if write:
``$066F``/``$0670`` to a low heart. Those are measurement poses, not route
results. Everything after them (pause, select, B, the refill, the re-select,
the octorok's hit) is the ROM's.

    uv run pytest nes/zelda_i/tests/rom/test_potions.py -m rom -q
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import pytest

from retro_harness.env import make_env, reset_obs
from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_idle_action
from zelda_i.assist import LastHeartAssist
from zelda_i.dungeon.pause_select import B_SLOT_CANDLE
from zelda_i.overworld.cave_shop import (
    BLUE_POTION_PRICE,
    POTION_SHOP_SCREEN,
    RED_POTION_PRICE,
    make_potion_buy_controller,
)
from zelda_i.overworld.gather_segments import CaveExitController
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import (
    ADDR_HEALTH,
    ADDR_HEART_PARTIAL,
    ADDR_MENU_STATE,
    ADDR_RUPEES,
    ADDR_SELECTED_ITEM,
    CAVE_MODE,
    PLAY_MODE,
    health_byte_for_containers,
    read_snapshot,
    read_u8,
)
from zelda_i.route.chain import run_controller_stage

from .conftest import ROM, skip_unless_pin

PIN = "GatherChain_exit_ring"
WALLET = 100  # what-if: the hidden-rupee chain's forecast after the ring

pytestmark = [ROM, skip_unless_pin(PIN)]


@dataclass
class _Linger:
    """Idle inner stage: done after ``frames`` of its own steps, or ``until``.

    The guard skips the inner while it drinks, so the stage ends a few frames
    after control is handed back.
    """

    frames: int = 8
    until: Callable[[Any], bool] | None = None
    steps: int = 0
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)

    def step(self, snap) -> FrameAction:
        self.steps += 1
        done = self.until(snap) if self.until is not None else self.steps >= self.frames
        self.success = bool(done)
        return FrameAction(nes_idle_action(), "linger")


def _boot(env):
    reset_obs(env)
    env.unwrapped.data.memory.assign(ADDR_RUPEES, "|u1", WALLET)
    return read_snapshot(env.get_ram()), read_u8(env.get_ram(), ADDR_SELECTED_ITEM)


def _buy(env, item: str):
    ctl = make_potion_buy_controller(item=item)
    _, stage = run_controller_stage(
        env, None, name="potion_buy", controller=ctl, max_frames=ctl.max_frames,
        assist=LastHeartAssist(),
    )
    return ctl, stage


def _walk_out(env) -> None:
    exit_ctl = CaveExitController()
    _, stage = run_controller_stage(
        env, None, name="exit_potion", controller=exit_ctl, max_frames=exit_ctl.max_frames
    )
    assert stage.success, stage.report()


def _set_hearts(env, whole: int, partial: int) -> None:
    """What-if write: Link's owned containers, ``whole`` hearts showing."""
    snap = read_snapshot(env.get_ram())
    mem = env.unwrapped.data.memory
    mem.assign(ADDR_HEALTH, "|u1", health_byte_for_containers(snap.heart_containers, filled=whole))
    mem.assign(ADDR_HEART_PARTIAL, "|u1", partial)


@pytest.mark.parametrize(
    ("item", "potion", "price"),
    [("auto", 2, RED_POTION_PRICE), ("blue", 1, BLUE_POTION_PRICE)],
)
def test_live_buy_shows_the_letter_and_debits_the_price(item, potion, price) -> None:
    env = make_env(GAME, PIN, GAME_DIR, render_mode="rgb_array")
    try:
        start, selected0 = _boot(env)
        ctl, stage = _buy(env, item)
        end = read_snapshot(env.get_ram())
        selected1 = read_u8(env.get_ram(), ADDR_SELECTED_ITEM)
    finally:
        env.close()
    assert start.letter == 1 and start.potion == 0 and selected0 == B_SLOT_CANDLE
    assert stage.success, stage.report()
    assert end.potion == potion
    assert end.letter == 2  # shown once; the shop sells from now on
    assert ctl.report()["rupees_at_buy"] - end.rupees == price
    assert selected1 == selected0  # the candle is back on B
    assert (end.level, end.screen, end.mode) == (0, POTION_SHOP_SCREEN, CAVE_MODE)


def test_live_drinks_red_then_blue_instead_of_the_last_heart_refill() -> None:
    env = make_env(GAME, PIN, GAME_DIR, render_mode="rgb_array")
    try:
        _boot(env)
        _, bought = _buy(env, "auto")
        assert bought.success, bought.report()
        _walk_out(env)
        ends = []
        for _ in range(2):
            _set_hearts(env, whole=1, partial=0x80)
            low = read_snapshot(env.get_ram())
            assist = LastHeartAssist()
            _, stage = run_controller_stage(
                env, None, name="linger", controller=_Linger(), max_frames=2000, assist=assist
            )
            ram = env.get_ram()
            ends.append(
                (
                    low,
                    read_snapshot(ram),
                    stage,
                    assist,
                    read_u8(ram, ADDR_SELECTED_ITEM),
                    read_u8(ram, ADDR_MENU_STATE),
                )
            )
    finally:
        env.close()
    potions = []
    for low, end, stage, assist, selected, menu in ends:
        assert low.whole_hearts == 1
        assert stage.success, stage.report()
        assert end.health_is_full and end.heart_partial == 0xFF
        assert end.mode == PLAY_MODE and menu == 0
        assert assist.telemetry.health.writes == 0  # the potion, not the assist
        assert assist.telemetry.refill_holds > 0
        assert stage.report()["potion"]["drinks"] == 1
        assert selected == B_SLOT_CANDLE
        potions.append((low.potion, end.potion))
    assert potions == [(2, 1), (1, 0)]


def test_live_potion_answers_a_real_hit_at_the_last_heart() -> None:
    """1.125 hearts on 0x64 among octoroks: the hit that crosses the last heart
    is answered by a drink, not by the last-heart refill."""
    env = make_env(GAME, PIN, GAME_DIR, render_mode="rgb_array")
    try:
        _boot(env)
        _, bought = _buy(env, "auto")
        assert bought.success, bought.report()
        _walk_out(env)
        _set_hearts(env, whole=2, partial=0x20)
        assist = LastHeartAssist()
        linger = _Linger(until=lambda s: s.potion < 2 and s.health_is_full)
        _, stage = run_controller_stage(
            env, None, name="stand", controller=linger, max_frames=4000, assist=assist
        )
        end = read_snapshot(env.get_ram())
    finally:
        env.close()
    report = stage.report()
    assert stage.success, report
    assert report["potion"]["drinks"] == 1, report
    assert assist.telemetry.damage_events >= 1
    assert assist.telemetry.health.writes == 0
    assert assist.telemetry.deaths == 0
    assert end.potion == 1
