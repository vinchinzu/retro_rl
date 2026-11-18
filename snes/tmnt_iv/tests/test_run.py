"""Run layer without a ROM: HP meter, boot schedule, tape, lookahead guard.

The guard tests drive ``GuardedPolicy`` against a tiny deterministic core
whose RAM uses the real TMNT IV layout, so the plan/splice/restore logic is
exercised exactly as it runs on stable-retro.
"""

from __future__ import annotations

import numpy as np
import pytest

from retro_harness.actions import buttons
from retro_harness.bot_runner import NodeStatus, TickResult
from retro_harness.controls import SNES_A, SNES_START, SNES_UP, SNES_DOWN, SNES_RIGHT
from retro_harness.input_script import FrameAction
from tmnt_iv.lookahead import MACROS, GuardedPolicy, LookaheadConfig, ram_crc
from tmnt_iv.menus import BOOT_LAST, TURTLE_CHAR, TURTLES, boot_schedule, hard_boot_action
from tmnt_iv.observe import HpDelta, living_hp, policy_input
from tmnt_iv.policy import Stage1Policy
from tmnt_iv.ram import ADDR_EVENT, OFF_HP, OFF_X, OFF_Y, PLAYER_BASE, parse_game_state, write_u16le
from tmnt_iv.tape import Tape, load_tape
from tmnt_iv.tests._state import playing, ram


def test_living_hp_rejects_zero_and_sentinels() -> None:
    assert living_hp(1) and living_hp(80) and living_hp(0x60)
    assert not living_hp(0) and not living_hp(0x61) and not living_hp(-1)


def test_hp_delta_counts_live_drops_not_ko_unless_asked() -> None:
    meter = HpDelta.start(80)
    assert meter.note(64) == 16
    assert meter.note(64) == 0
    assert meter.note(0) == 0
    assert meter.damage == 16
    assert meter.min_hp == 64

    with_ko = HpDelta.start(16, count_zero=True)
    assert with_ko.note(0) == 16
    assert with_ko.damage == 16


def test_policy_input_unwraps_idle_when_tree_succeeds() -> None:
    action, reason = policy_input(
        Stage1Policy(),
        playing(player_x=64, player_y=192, stage=0, health=80),
    )
    assert action is not None
    assert isinstance(reason, str) and reason


def test_boot_schedule_rights_pick_the_turtle() -> None:
    assert set(TURTLES) == set(TURTLE_CHAR) == {"leo", "mike", "don", "raph"}
    for turtle, rights in TURTLES.items():
        sched = boot_schedule(turtle)
        assert sum(1 for f, b in sched.items() if b == ("RIGHT",) and f > 1400) == 5 * rights
        assert max(sched) == BOOT_LAST
    assert hard_boot_action(1490, "leo") == buttons("START")
    assert not any(hard_boot_action(1441, "leo"))
    assert hard_boot_action(1441, "mike") == buttons("RIGHT")


def test_tape_round_trip(tmp_path) -> None:
    tape = Tape(entry="NONE", turtle="mike", actions=[tuple(buttons("RIGHT"))] * 3, crcs={0: 7})
    path = tape.save(tmp_path / "x.json".replace(".json", ""))
    back = load_tape(path)
    assert back.entry == "NONE" and back.turtle == "mike"
    assert back.actions == tape.actions and back.crcs == {0: 7}


def test_macros_never_press_a_or_start() -> None:
    assert MACROS
    for label, frames in MACROS:
        assert frames, label
        for vec in frames:
            assert len(vec) == 12
            assert not vec[SNES_A] and not vec[SNES_START], label


# ---------------------------------------------------------------- toy core
_HIT_FRAME = 40
_BAND = (150, 170)


class _ToyCore:
    """Deterministic stand-in for ``env.em``: x/y walk, one timed hazard.

    At frame 40 a hazard costs 16 HP if the turtle's Y is inside 150-170.
    Walking RIGHT forever (the base policy below) stays in the band.
    """

    def __init__(self) -> None:
        self.ram = ram()
        self.ram[ADDR_EVENT] = 0x0A
        self.frame = 0
        self._mask = np.zeros(12, dtype=np.uint8)
        self.loads = 0

    def get_state(self) -> bytes:
        return self.frame.to_bytes(4, "little") + self.ram.tobytes()

    def set_state(self, state: bytes) -> None:
        self.loads += 1
        self.frame = int.from_bytes(state[:4], "little")
        self.ram[:] = np.frombuffer(state[4:], dtype=np.uint8)

    def set_button_mask(self, mask, _player: int = 0) -> None:
        self._mask = np.asarray(mask, dtype=np.uint8).copy()

    def step(self) -> None:
        x = int(self.ram[PLAYER_BASE + OFF_X]) | (int(self.ram[PLAYER_BASE + OFF_X + 1]) << 8)
        y = int(self.ram[PLAYER_BASE + OFF_Y]) | (int(self.ram[PLAYER_BASE + OFF_Y + 1]) << 8)
        x += 1 if self._mask[SNES_RIGHT] else 0
        y += -2 if self._mask[SNES_UP] else (2 if self._mask[SNES_DOWN] else 0)
        write_u16le(self.ram, PLAYER_BASE + OFF_X, max(1, x))
        write_u16le(self.ram, PLAYER_BASE + OFF_Y, max(1, y))
        self.frame += 1
        if self.frame == _HIT_FRAME and _BAND[0] <= y <= _BAND[1]:
            self.ram[PLAYER_BASE + OFF_HP] -= 16

    def get_ram(self) -> np.ndarray:
        return self.ram


class _WalkRight:
    def reset(self) -> None:
        return None

    def tick(self, state) -> TickResult:
        fa = FrameAction(action=buttons("RIGHT"), reason="walk_right")
        return TickResult(status=NodeStatus.RUNNING, action=fa, reason="walk_right")


def _play(policy, core: _ToyCore, frames: int = 80) -> int:
    for _ in range(frames):
        state = parse_game_state(core.get_ram(), frame=core.frame)
        action, _reason = policy_input(policy, state)
        core.set_button_mask(np.asarray(action, dtype=np.uint8))
        core.step()
    return int(core.ram[PLAYER_BASE + OFF_HP])


def test_toy_core_base_policy_takes_the_hit() -> None:
    assert _play(_WalkRight(), _ToyCore()) == 64


def test_guard_dodges_the_hit_and_returns_live_ram_untouched() -> None:
    core = _ToyCore()
    guard = GuardedPolicy(_WalkRight(), LookaheadConfig(horizon=32, chunk=8))  # type: ignore[arg-type]
    guard.bind(core, core.get_ram)
    state = parse_game_state(core.get_ram(), frame=0)
    before = ram_crc(core.get_ram())
    first = guard.tick(state)
    assert ram_crc(core.get_ram()) == before  # lookahead restored the live frame
    core.set_button_mask(np.asarray(first.action.action, dtype=np.uint8))
    core.step()
    assert _play(guard, core, frames=79) == 80
    report = guard.report()
    assert report["threats"] >= 1 and report["rescued"] >= 1
    assert report["forced"] == 0 and report["desyncs"] == 0


def test_guard_counts_a_desync_when_live_diverges() -> None:
    core = _ToyCore()
    guard = GuardedPolicy(_WalkRight(), LookaheadConfig(horizon=16, chunk=8))  # type: ignore[arg-type]
    guard.bind(core, core.get_ram)
    guard.tick(parse_game_state(core.get_ram(), frame=0))
    core.set_button_mask(np.asarray(buttons("DOWN"), dtype=np.uint8))  # not the plan
    core.step()
    guard.tick(parse_game_state(core.get_ram(), frame=1))
    assert guard.report()["desyncs"] == 1


@pytest.mark.parametrize("turtle", sorted(TURTLES))
def test_every_turtle_boot_ends_on_start(turtle: str) -> None:
    assert boot_schedule(turtle)[BOOT_LAST] == ("START",)


def test_stage_clear_zero_is_not_a_ko() -> None:
    from tmnt_iv.ram import is_stage_clear_zero

    assert is_stage_clear_zero(playing(health=0, player_x=0, extras={"event": 0x0A}))
    assert is_stage_clear_zero(playing(health=0, player_x=120, extras={"event": 0x19}))
    assert is_stage_clear_zero(playing(health=0, player_x=120, extras={"event": 0x0B}))
    assert not is_stage_clear_zero(playing(health=0, player_x=120, extras={"event": 0x0A}))
    assert not is_stage_clear_zero(playing(health=40, player_x=0, extras={"event": 0x19}))


def test_stalled_guard_accepts_an_affordable_hit_only() -> None:
    from tmnt_iv.lookahead import Branch, _Node
    from tmnt_iv.tests._state import enemy

    guard = GuardedPolicy(_WalkRight(), LookaheadConfig(stall_frames=600, hp_floor=0.5))  # type: ignore[arg-type]
    fight = playing(health=80, frame=0, enemies=(enemy(120, 160, 40, kind=0x60),))
    guard._note_progress(fight)
    guard._nodes.append(_Node(0, 1000, b"", None))  # type: ignore[arg-type]
    still = playing(health=80, frame=1000, enemies=(enemy(120, 160, 40, kind=0x60),))
    guard._note_progress(still)
    hit = Branch("policy", [], [], None, hp_lost=10, start_hp=80)
    deep = Branch("policy", [], [], None, hp_lost=50, start_hp=80)
    dead = Branch("policy", [], [], None, hp_lost=10, start_hp=80, life_lost=True)
    assert guard._affordable(hit)
    assert not guard._affordable(deep) and not guard._affordable(dead)
    # Damage dealt (or an empty screen) resets the clock: strict again.
    guard._note_progress(playing(health=80, frame=1000, enemies=(enemy(120, 160, 30, kind=0x60),)))
    assert not guard._affordable(hit)
