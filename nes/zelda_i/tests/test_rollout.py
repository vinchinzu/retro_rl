"""Offline cover for ``zelda_i.rollout`` — the ROM-truth prediction kernel.

No emulator here. A ``_FakeEmulator`` plays the part: it owns an integer
"state", steps a scripted RAM tape, and records every save/restore so the
tests can assert the thing that actually matters about this module — that a
caller's emulator comes back exactly where it was, on the happy path and on
the raising one. The live-ROM half is ``tests/rom/test_rollout_truth.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from zelda_i.ram import (
    ADDR_LINK_IFRAMES,
    ADDR_LINK_X,
    ADDR_OBJ_HP,
    ADDR_OBJ_TYPE,
    ADDR_RUPEES,
    ADDR_SCREEN,
    PLAY_MODE,
)
from zelda_i.rollout import (
    Outcome,
    Rollout,
    hold,
    press,
    swing_after,
)
from zelda_i.tests.ram_helpers import make_ram

_DEFAULTS = {"mode": PLAY_MODE, "level": 0, "screen": 0x77, "x": 100, "y": 100,
             "health": 0x22, "sword": 1}
LEFT, RIGHT, UP, DOWN, A = (press(n) for n in ("LEFT", "RIGHT", "UP", "DOWN", "A"))
IDLE = press()


def _ram(**fields) -> np.ndarray:
    ram = make_ram(_DEFAULTS, **fields)
    ram[0x0670] = 0xFF  # heart partial full; heart_value reads it
    return ram


class _FakeEmulator:
    """Steps a per-action RAM tape and ledgers every state save/restore."""

    def __init__(self, tapes: dict, base: np.ndarray) -> None:
        self.tapes = {tuple(k): v for k, v in tapes.items()}
        self.ram = base
        self.base = base
        self.mask = IDLE
        self.cursor = 0
        self.saves = 0
        self.restores = 0
        self.steps = 0
        self.raise_on_step = False

    # stable-retro surface
    def get_state(self) -> bytes:
        self.saves += 1
        return b"pin"

    def set_state(self, state: bytes) -> None:
        assert state == b"pin"
        self.restores += 1
        self.ram = self.base
        self.cursor = 0

    def set_button_mask(self, mask, _player: int = 0) -> None:
        # The real core rejects a Python int above 0xFF and misreads a small
        # one as a single-button array. Hold the rollout to the array contract.
        assert isinstance(mask, np.ndarray) and mask.dtype == np.uint8, mask
        self.mask = tuple(int(v) for v in mask)

    def step(self) -> None:
        if self.raise_on_step:
            raise RuntimeError("core died mid-rollout")
        self.steps += 1
        tape = self.tapes.get(self.mask, ())
        if self.cursor < len(tape):
            self.ram = tape[self.cursor]
        self.cursor += 1


class _FakeEnv:
    def __init__(self, em: _FakeEmulator) -> None:
        self.em = em

    def get_ram(self) -> np.ndarray:
        return self.em.ram


def _env(tapes: dict, base: np.ndarray | None = None):
    base = _ram() if base is None else base
    em = _FakeEmulator(tapes, base)
    return _FakeEnv(em), em


# --- plan builders ---------------------------------------------------- #

def test_press_is_the_same_action_vector_the_controllers_emit() -> None:
    from retro_harness.nes import nes_action

    for names in (("A",), ("UP",), ("LEFT", "A"), ()):
        assert press(*names) == tuple(int(v) for v in nes_action(*names))


def test_press_of_nothing_is_idle_and_is_not_a_bitmask() -> None:
    assert press() == (0,) * len(press())
    assert len(press()) >= 9, "a plan frame is an action vector, not an int"
    assert press("LEFT", "A") != press("LEFT")


def test_press_rejects_a_typo() -> None:
    with pytest.raises(ValueError, match="unknown NES button"):
        press("LEFTT")


def test_hold_stands_on_none_and_on_stand() -> None:
    assert hold(None, 3) == (IDLE, IDLE, IDLE)
    assert hold("STAND", 3) == (IDLE, IDLE, IDLE)
    assert hold("LEFT", 3) == (LEFT, LEFT, LEFT)
    assert hold("LEFT", 0) == ()
    assert hold("LEFT", -5) == ()


def test_swing_after_turns_first_and_presses_a_exactly_once() -> None:
    """A is an edge and a turn needs its own frames — both, or the blade
    goes out along the old axis (``AGENTS.md`` turn/swing trap)."""
    blade = press("LEFT", "A")
    plan = swing_after("LEFT", turn_frames=4, frames=10)
    assert len(plan) == 10
    assert plan[:4] == (LEFT,) * 4
    assert plan[4] == blade
    assert plan[5:] == (LEFT,) * 5
    assert sum(1 for f in plan if f == blade) == 1


def test_swing_after_never_drops_the_press_when_the_plan_is_short() -> None:
    plan = swing_after("UP", turn_frames=4, frames=2)
    assert plan[-1] == press("UP", "A")


# --- the kernel ------------------------------------------------------- #

def test_fan_restores_the_live_state_once_per_plan_and_at_the_end() -> None:
    env, em = _env({})
    rollout = Rollout(env)
    rollout.fan({"a": hold("LEFT", 3), "b": hold("RIGHT", 3)})
    assert em.saves == 1
    # one restore before each plan, one final restore back to the caller
    assert em.restores == 3
    assert env.get_ram() is em.base


def test_fan_restores_even_when_the_core_raises() -> None:
    env, em = _env({})
    em.raise_on_step = True
    with pytest.raises(RuntimeError):
        Rollout(env).fan({"a": hold("LEFT", 3)})
    assert em.restores >= 1
    assert env.get_ram() is em.base


def test_fan_of_nothing_never_touches_the_emulator() -> None:
    env, em = _env({})
    assert Rollout(env).fan({}) == ()
    assert (em.saves, em.restores, em.steps) == (0, 0, 0)


def test_fan_preserves_plan_order() -> None:
    env, _ = _env({})
    labels = [o.label for o in Rollout(env).fan(
        {"UP": hold("UP", 1), "DOWN": hold("DOWN", 1), "STAND": hold(None, 1)}
    )]
    assert labels == ["UP", "DOWN", "STAND"]


def test_frames_pads_a_short_plan_by_holding_its_last_frame() -> None:
    env, em = _env({})
    Rollout(env).fan({"a": (LEFT,)}, frames=5)
    assert em.steps == 5
    assert em.mask == LEFT


def test_frames_truncates_a_long_plan() -> None:
    env, em = _env({})
    out = Rollout(env).fan({"a": hold("LEFT", 40)}, frames=6)[0]
    assert (em.steps, out.frames) == (6, 6)


def test_an_empty_plan_padded_is_idle_not_a_crash() -> None:
    env, em = _env({})
    Rollout(env).fan({"a": ()}, frames=3)
    assert (em.steps, em.mask) == (3, IDLE)


# --- what the outcome reads off the tape ------------------------------ #

def test_contact_frame_is_the_iframe_rising_edge() -> None:
    tape = [_ram(link_iframes=v) for v in (0, 0, 24, 23, 22)]
    env, _ = _env({LEFT: tape})
    out = Rollout(env).run("left", hold("LEFT", 5))
    assert out.contact_frame == 3
    assert not out.safe and out.hits == 1


def test_a_decrementing_timer_is_the_same_hit_not_a_second_one() -> None:
    """Counting a cooling timer would rank standing in one body worse than
    walking through three."""
    tape = [_ram(link_iframes=v) for v in (24, 23, 22, 21)]
    env, _ = _env({LEFT: tape}, base=_ram(link_iframes=24))
    assert Rollout(env).run("left", hold("LEFT", 4)).contact_frame is None


def test_a_rearm_after_the_timer_expires_is_a_new_contact() -> None:
    tape = [_ram(link_iframes=v) for v in (2, 1, 0, 0, 24)]
    env, _ = _env({LEFT: tape}, base=_ram(link_iframes=3))
    assert Rollout(env).run("left", hold("LEFT", 5)).contact_frame == 5


def test_no_contact_is_safe() -> None:
    env, _ = _env({LEFT: [_ram() for _ in range(4)]})
    out = Rollout(env).run("left", hold("LEFT", 4))
    assert out.safe and out.contact_frame is None and out.hits == 0


def test_hp_lost_counts_a_wooden_chip_not_just_whole_hearts() -> None:
    """``$066F`` never moves on a chip; ``heart_value`` is 1/256 of a heart."""
    chip = _ram()
    chip[0x0670] = 0x7F
    env, _ = _env({LEFT: [chip]})
    out = Rollout(env).run("left", hold("LEFT", 1))
    assert out.hp_lost == 0xFF - 0x7F
    assert out.report()["hearts_lost"] == pytest.approx(0.5, abs=0.01)


def test_healing_is_not_negative_damage() -> None:
    hurt = _ram()
    hurt[0x0670] = 0x7F
    healed = _ram()
    env, _ = _env({LEFT: [healed]}, base=hurt)
    assert Rollout(env).run("left", hold("LEFT", 1)).hp_lost == 0


def test_moved_is_manhattan_from_the_frame_the_fan_started_on() -> None:
    env, _ = _env({RIGHT: [_ram(x=104, y=108)]})
    assert Rollout(env).run("r", hold("RIGHT", 1)).moved == 4 + 8


def test_kills_are_the_live_census_delta() -> None:
    def with_bodies(n: int) -> np.ndarray:
        ram = _ram()
        for slot in range(1, n + 1):
            ram[ADDR_OBJ_TYPE + slot] = 0x07  # octorok
            ram[ADDR_OBJ_HP + slot] = 16
            ram[0x0070 + slot] = 60          # object x
            ram[0x0084 + slot] = 60          # object y
        return ram

    env, _ = _env({A: [with_bodies(1)]}, base=with_bodies(3))
    assert Rollout(env).run("swing", (A,)).kills == 2


def test_leaving_the_screen_banks_no_kills_and_no_rupees() -> None:
    """A scroll renumbers every slot and hands out a fresh census."""
    gone = _ram(screen=0x78, rupees=40)
    env, _ = _env({UP: [gone]}, base=_ram(screen=0x77, rupees=10))
    out = Rollout(env).run("up", hold("UP", 1))
    assert out.left_screen and out.kills == 0 and out.rupees == 0


def test_rupees_are_positive_deltas_only() -> None:
    """A purchase is not a negative pickup."""
    env, _ = _env({UP: [_ram(rupees=2)]}, base=_ram(rupees=22))
    assert Rollout(env).run("up", hold("UP", 1)).rupees == 0


def test_first_contact_is_the_rom_twin_of_contact_frames() -> None:
    tape = [_ram(link_iframes=v) for v in (0, 24)]
    env, _ = _env({DOWN: tape})
    assert Rollout(env).first_contact(hold("DOWN", 2)) == 2


# --- the choice ------------------------------------------------------- #

def _hit_at(frame: int, n: int) -> list[np.ndarray]:
    return [_ram(link_iframes=24 if i + 1 >= frame else 0) for i in range(n)]


def test_best_step_prefers_the_direction_that_is_never_hit() -> None:
    env, _ = _env({LEFT: _hit_at(3, 8), RIGHT: _hit_at(6, 8), UP: _hit_at(2, 8)})
    best = Rollout(env).best_step(frames=8, directions=("LEFT", "RIGHT", "UP", "DOWN"))
    assert best.label == "DOWN" and best.safe


def test_best_step_prefers_the_later_hit_when_every_option_is_hit() -> None:
    env, _ = _env({LEFT: _hit_at(3, 8), RIGHT: _hit_at(6, 8),
                   UP: _hit_at(2, 8), DOWN: _hit_at(1, 8)})
    best = Rollout(env).best_step(frames=8, directions=("LEFT", "RIGHT", "UP", "DOWN"))
    assert best.label == "RIGHT" and best.contact_frame == 6


def test_best_step_breaks_a_safe_tie_on_ground_covered() -> None:
    """The wall-blindness fix: a direction into a rock moves 0 px and loses
    to one that actually walks, without any occupancy rectangle."""
    env, _ = _env({LEFT: [_ram(x=100)] * 8,            # walked into a rock
                   RIGHT: [_ram(x=100 + i) for i in range(1, 9)]})
    best = Rollout(env).best_step(frames=8, directions=("LEFT", "RIGHT"))
    assert best.label == "RIGHT" and best.moved == 8


def test_best_step_can_choose_to_stand() -> None:
    env, _ = _env({LEFT: _hit_at(2, 6), RIGHT: _hit_at(2, 6),
                   UP: _hit_at(2, 6), DOWN: _hit_at(2, 6)})
    assert Rollout(env).best_step(frames=6).label == "STAND"


def test_report_ledgers_the_restores_a_run_report_must_not_hide() -> None:
    env, _ = _env({})
    rollout = Rollout(env)
    rollout.fan({"a": hold("LEFT", 4), "b": hold("RIGHT", 4)})
    assert rollout.report() == {"rollouts": 2, "frames_rolled": 8}


def test_outcome_report_is_json_shaped() -> None:
    import json

    env, _ = _env({LEFT: [_ram(x=90)]})
    out = Rollout(env).run("left", hold("LEFT", 1))
    assert isinstance(out, Outcome)
    assert json.loads(json.dumps(out.report()))["label"] == "left"


# --- more than one ply ------------------------------------------------ #

def test_branching_saves_once_and_restores_at_the_end() -> None:
    """A search is one borrow, not one per segment: the restore that matters
    is the caller's, and it happens when the whole search is done."""
    env, em = _env({})
    rollout = Rollout(env)
    with rollout.branching() as branch:
        token, _ = branch.play("a", hold("LEFT", 3))
        branch.play("b", hold("RIGHT", 3), token)
    assert em.saves >= 1
    assert env.get_ram() is em.base
    assert rollout.report() == {"rollouts": 2, "frames_rolled": 6}


def test_branching_restores_even_when_a_segment_raises() -> None:
    """The restore is a ``finally``. A scorer that dies mid-search must not
    leave the walk's emulator somewhere else."""
    env, em = _env({})
    em.raise_on_step = True
    with pytest.raises(RuntimeError):
        with Rollout(env).branching() as branch:
            branch.play("a", hold("LEFT", 3))
    assert em.restores >= 1
    assert env.get_ram() is em.base


def test_a_branch_hands_back_a_token_and_not_just_an_outcome() -> None:
    """The one thing ``fan`` cannot do: a sequence continues from where the
    previous segment *ended*."""
    env, _ = _env({})
    with Rollout(env).branching() as branch:
        token, outcome = branch.play("a", hold("LEFT", 2))
    assert token is not None
    assert outcome.frames == 2
