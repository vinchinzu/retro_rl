"""Unit tests for eval_match checkpoint ranking helpers (no ROM)."""

from __future__ import annotations

from pathlib import Path

from mortal_kombat.eval_match import (
    RAW_EVAL_MAX_STEPS,
    checkpoint_steps,
    list_v3_checkpoints,
    make_raw_eval_env,
    may_promote,
    play_buttons_match,
)
from mortal_kombat.ram import Screen, is_match_lost, is_match_won, make_test_ram, parse_ram


def test_checkpoint_steps_numeric() -> None:
    assert checkpoint_steps(Path("mk1_v3_Match7_ppo_250000_steps.zip")) == 250000


def test_checkpoint_steps_final_outranks_numeric() -> None:
    final = checkpoint_steps(Path("mk1_v3_Match7_ppo_final.zip"))
    assert final == 10**18
    assert final > checkpoint_steps(Path("mk1_v3_Match7_ppo_999999999_steps.zip"))


def test_checkpoint_steps_unrelated() -> None:
    assert checkpoint_steps(Path("unrelated.zip")) == -1


def test_may_promote_threshold() -> None:
    assert may_promote(19) is False
    assert may_promote(20) is True


def test_list_v3_checkpoints_glob_sorted(tmp_path: Path) -> None:
    later = tmp_path / "mk1_v3_Match7_ppo_250000_steps.zip"
    earlier = tmp_path / "mk1_v3_Match7_ppo_100000_steps.zip"
    final = tmp_path / "mk1_v3_Match7_ppo_final.zip"
    (tmp_path / "other.zip").write_bytes(b"x")
    for path in (later, earlier, final):
        path.write_bytes(b"z")
    assert list_v3_checkpoints(tmp_path, "Match7") == [earlier, later, final]


def test_list_v3_checkpoints_names_include_missing_sorted(tmp_path: Path) -> None:
    present = tmp_path / "mk1_v3_Match7_ppo_100000_steps.zip"
    present.write_bytes(b"z")
    missing_mid = tmp_path / "mk1_v3_Match7_ppo_200000_steps.zip"
    missing_final = tmp_path / "mk1_v3_Match7_ppo_final.zip"
    names = [missing_final.name, present.name, missing_mid.name]
    result = list_v3_checkpoints(tmp_path, "Match7", names)
    assert result == [present, missing_mid, missing_final]
    assert result[0].is_file()
    assert not result[1].is_file()
    assert not result[2].is_file()


def _win_ram():
    return make_test_ram(p1_rounds=2, p2_rounds=0, timer=0, p1_health=80, p2_health=0)


def _loss_ram():
    return make_test_ram(p1_rounds=0, p2_rounds=2, timer=0, p1_health=0, p2_health=80)


def _continue_ram():
    return make_test_ram(p1_health=0, p2_health=0, timer=0, continue_timer=9)


class FakeEnv:
    def __init__(self, rams: list):
        self.rams = list(rams)
        self.i = 0
        self.unwrapped = self
        self.buttons: list = []
        self.n_reset = 0
        self.closed = False

    def get_ram(self):
        return self.rams[min(self.i, len(self.rams) - 1)]

    def reset(self, *args, **kwargs):
        del args, kwargs
        self.n_reset += 1
        self.i = 0
        return None, {}

    def step(self, buttons):
        self.buttons.append(buttons)
        if self.i < len(self.rams) - 1:
            self.i += 1
        return None, 0.0, False, False, {}

    def close(self) -> None:
        self.closed = True


class DummyPolicy:
    def __init__(self):
        self.calls: list[tuple] = []
        self.n_reset = 0

    def reset(self) -> None:
        self.n_reset += 1

    def act(self, ram, rgb, *, deterministic: bool = False):
        self.calls.append((ram, rgb, deterministic))
        return [0] * 12


def test_raw_eval_max_steps_constant() -> None:
    assert RAW_EVAL_MAX_STEPS == 60_000
    assert callable(make_raw_eval_env)


def test_play_buttons_match_win_between_rounds() -> None:
    ram = _win_ram()
    snap = parse_ram(ram)
    assert snap.screen is Screen.BETWEEN_ROUNDS
    assert is_match_won(snap)
    env = FakeEnv([ram])
    policy = DummyPolicy()
    assert play_buttons_match(policy, env) is True
    assert env.n_reset == 1
    assert policy.n_reset == 1
    assert policy.calls == []
    assert env.buttons == []


def test_play_buttons_match_loss() -> None:
    ram = _loss_ram()
    snap = parse_ram(ram)
    assert snap.screen is Screen.BETWEEN_ROUNDS
    assert is_match_lost(snap)
    env = FakeEnv([ram])
    policy = DummyPolicy()
    assert play_buttons_match(policy, env) is False
    assert policy.calls == []
    assert env.buttons == []
    assert env.buttons == []


def test_play_buttons_match_continue_screen() -> None:
    ram = _continue_ram()
    assert parse_ram(ram).screen is Screen.CONTINUE
    env = FakeEnv([ram])
    policy = DummyPolicy()
    assert play_buttons_match(policy, env) is False
    assert policy.calls == []


def test_play_buttons_match_counts_health_ko_edges_before_hud_settles() -> None:
    live = make_test_ram(p1_health=100, p2_health=10, timer=80)
    p2_ko = make_test_ram(p1_health=100, p2_health=0, timer=80)
    refill = make_test_ram(p1_health=161, p2_health=161, timer=153)
    env = FakeEnv([live, p2_ko, refill, live, p2_ko])
    assert play_buttons_match(DummyPolicy(), env) is True
    assert len(env.buttons) == 4


def test_play_buttons_match_calls_act_and_step() -> None:
    fight = make_test_ram()
    env = FakeEnv([fight, _win_ram()])
    policy = DummyPolicy()
    assert play_buttons_match(policy, env) is True
    assert policy.n_reset == 1
    assert len(policy.calls) == 1
    ram, rgb, deterministic = policy.calls[0]
    assert ram is fight
    assert rgb is None
    assert deterministic is True
    assert env.buttons == [[0] * 12]


def test_play_buttons_match_ignores_fight_hud_flicker() -> None:
    flicker = make_test_ram(p2_rounds=2, timer=90, p1_health=161, p2_health=136)
    snap = parse_ram(flicker)
    assert snap.screen is Screen.FIGHT
    assert is_match_lost(snap)
    env = FakeEnv([flicker, flicker, _win_ram()])
    policy = DummyPolicy()
    assert play_buttons_match(policy, env) is True
    assert len(policy.calls) == 2


def test_play_buttons_match_timeout() -> None:
    env = FakeEnv([make_test_ram()])
    policy = DummyPolicy()
    assert play_buttons_match(policy, env, max_steps=2) is False
    assert len(policy.calls) == 2
    assert len(env.buttons) == 2
