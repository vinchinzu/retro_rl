"""Unit tests for RAM-gated MK2 eval (no ROM)."""

from __future__ import annotations

from mortal_kombat_ii.eval_match import (
    EVAL_MAX_STEPS,
    RAW_EVAL_MAX_STEPS,
    make_cnn_eval_env,
    make_raw_eval_env,
    play_buttons_match,
    play_match,
)
from mortal_kombat_ii.ram import make_test_ram, parse_ram


class FakeEnv:
    def __init__(self, rams: list):
        self.rams = list(rams)
        self.i = 0
        self.unwrapped = self
        self.buttons: list = []
        self.n_reset = 0

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


class DummyPolicy:
    def __init__(self):
        self.calls: list[tuple] = []
        self.n_reset = 0

    def reset(self) -> None:
        self.n_reset += 1

    def act(self, ram, rgb, *, deterministic: bool = False):
        self.calls.append((ram, rgb, deterministic))
        return [0] * 12


class DummyModel:
    def __init__(self, infos: list[dict]):
        self.infos = list(infos)
        self.i = 0

    def predict(self, obs, deterministic: bool = False):
        del obs, deterministic
        return 0, None


class FakeCnnEnv:
    def __init__(self, infos: list[dict], *, terminate_at: int | None = None):
        self.infos = list(infos)
        self.terminate_at = terminate_at
        self.i = 0
        self.n_reset = 0

    def reset(self, *args, **kwargs):
        del args, kwargs
        self.n_reset += 1
        self.i = 0
        return object(), {}

    def step(self, action):
        del action
        info = self.infos[min(self.i, len(self.infos) - 1)]
        self.i += 1
        done = self.terminate_at is not None and self.i >= self.terminate_at
        return object(), 0.0, done, False, info


def test_eval_step_caps() -> None:
    assert RAW_EVAL_MAX_STEPS == 60_000
    assert EVAL_MAX_STEPS == 15_000
    assert callable(make_raw_eval_env)
    assert callable(make_cnn_eval_env)


def test_play_buttons_match_win_on_two_p2_kos() -> None:
    live = make_test_ram(p1_health=100, p2_health=10)
    p2_ko = make_test_ram(p1_health=100, p2_health=0)
    refill = make_test_ram(p1_health=161, p2_health=161)
    env = FakeEnv([live, p2_ko, refill, live, p2_ko])
    policy = DummyPolicy()
    assert play_buttons_match(policy, env) is True
    assert env.n_reset == 1
    assert policy.n_reset == 1
    assert len(env.buttons) == 4


def test_play_buttons_match_loss_on_two_p1_kos() -> None:
    live = make_test_ram(p1_health=10, p2_health=100)
    p1_ko = make_test_ram(p1_health=0, p2_health=100)
    env = FakeEnv([live, p1_ko, live, p1_ko])
    policy = DummyPolicy()
    assert play_buttons_match(policy, env) is False
    assert len(policy.calls) == 3


def test_play_buttons_match_timeout_is_loss() -> None:
    env = FakeEnv([make_test_ram()])
    policy = DummyPolicy()
    assert play_buttons_match(policy, env, max_steps=2) is False
    assert len(policy.calls) == 2
    assert len(env.buttons) == 2


def test_play_buttons_match_ignores_decoy_health_bytes() -> None:
    live = make_test_ram(p1_health=100, p2_health=10, decoy_020a=0, decoy_020e=0)
    assert parse_ram(live).p2_health == 10
    p2_ko = make_test_ram(p1_health=100, p2_health=0, decoy_020a=161, decoy_020e=161)
    refill = make_test_ram()
    env = FakeEnv([live, p2_ko, refill, live, p2_ko])
    assert play_buttons_match(DummyPolicy(), env) is True


def test_play_match_uses_fighting_env_round_info() -> None:
    env = FakeCnnEnv(
        [{"rounds_won": 2, "rounds_lost": 0}],
        terminate_at=1,
    )
    assert play_match(DummyModel([]), env) is True
    env_loss = FakeCnnEnv(
        [{"rounds_won": 0, "rounds_lost": 2}],
        terminate_at=1,
    )
    assert play_match(DummyModel([]), env_loss) is False
    env_tie = FakeCnnEnv(
        [{"rounds_won": 2, "rounds_lost": 2}],
        terminate_at=1,
    )
    assert play_match(DummyModel([]), env_tie) is False
