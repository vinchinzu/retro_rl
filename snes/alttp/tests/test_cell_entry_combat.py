"""Cell-entry forecasts must stay outside the live ledger and input tape."""

from __future__ import annotations

import numpy as np
import pytest

from alttp.opening_route.main_hall_to_zelda import _cell_entry_combat
from alttp.primitives import CombatResult
from alttp.ram import INDOORS, LINK_HP, MODULE, ROOM_ID, wram_index
from alttp.rollout import restores
from alttp.spine.ledger import RunLedger
from alttp.startup import action_for, build_boot_env, snapshot_env


class _Core:
    def __init__(self):
        self.ram = np.zeros(0x20000, dtype=np.uint8)
        self.ram[MODULE] = 0x07
        self.ram[INDOORS] = 1
        self.ram[ROOM_ID] = 0x80
        self.ram[wram_index(LINK_HP)] = 24
        self.frame = 0
        self.action = None

    def get_state(self):
        return self.frame.to_bytes(4, "little") + self.ram.tobytes()

    def set_state(self, state):
        self.frame = int.from_bytes(state[:4], "little")
        self.ram[:] = np.frombuffer(state[4:], dtype=np.uint8)

    def set_button_mask(self, action, _player):
        self.action = action

    def step(self):
        self.frame += 1
        # Charging straight in costs a heart; standing first avoids it.
        if self.frame == 1 and np.array_equal(self.action, action_for("B")):
            self.ram[wram_index(LINK_HP)] -= 8


class _Env:
    def __init__(self):
        self.em = _Core()
        self.actions = []

    def get_ram(self):
        return self.em.ram

    def step(self, action):
        self.actions.append(tuple(action))
        self.em.set_button_mask(action, 0)
        self.em.step()


def _ledger(env):
    ledger = RunLedger(print_on_close=False)
    ledger.observe(env)
    ledger.attach(env)
    return ledger


def test_failed_candidate_does_not_charge_live_frames_or_damage(monkeypatch):
    env = _Env()
    ledger = _ledger(env)
    attempts = []
    restores_before = restores()

    def fight(candidate, **_kwargs):
        attempts.append(snapshot_env(candidate).hp)
        candidate.step(action_for("B"))
        snap = snapshot_env(candidate)
        return CombatResult(snap.hp == 24, "candidate", 1, snap)

    monkeypatch.setattr("alttp.primitives.fight_nearby", fight)
    stand, result = _cell_entry_combat(env)

    assert len(attempts) == 2
    assert stand.frames == 20 and result.frames == 1 and result.ok
    assert ledger.frame - 1 == len(env.actions) == stand.frames + result.frames == 21
    assert ledger.damage_report()["damage"] == 0
    assert snapshot_env(env).hp == result.snapshot.hp == 24
    assert restores() - restores_before == 3


def test_every_failed_candidate_commits_only_the_final_failure(monkeypatch):
    env = _Env()
    ledger = _ledger(env)

    def fight(candidate, **_kwargs):
        candidate.step(action_for("B"))
        return CombatResult(False, "failed", 1, snapshot_env(candidate))

    monkeypatch.setattr("alttp.primitives.fight_nearby", fight)
    stand, result = _cell_entry_combat(env)

    assert not result.ok and stand.frames == 280
    assert ledger.frame - 1 == len(env.actions) == stand.frames + result.frames == 281
    assert ledger.damage_report()["damage"] == 0
    assert snapshot_env(env).hp == 24


def test_forecast_exception_restores_the_root_without_committing(monkeypatch):
    env = _Env()
    ledger = _ledger(env)
    root = env.em.get_state()

    def fight(candidate, **_kwargs):
        candidate.step(action_for("B"))
        raise RuntimeError("forecast failed")

    monkeypatch.setattr("alttp.primitives.fight_nearby", fight)
    with pytest.raises(RuntimeError, match="forecast failed"):
        _cell_entry_combat(env)

    assert env.em.get_state() == root
    assert env.actions == [] and ledger.frame == 1
    assert ledger.damage_report()["damage"] == 0


@pytest.mark.rom
def test_cell_entry_tactic_counts_only_committed_frames_and_replays_exactly():
    from alttp.paths import INTEGRATION_DIR

    if not (INTEGRATION_DIR / "rom.sfc").is_file():
        pytest.skip("ALTTP ROM missing")
    if not (INTEGRATION_DIR / "CastleB1Guard.state").is_file():
        pytest.skip("CastleB1Guard.state missing")
    env = build_boot_env("CastleB1Guard")
    try:
        env.reset()
        root = bytes(env.em.get_state())
        base_step = env.step
        ledger = _ledger(env)
        observed_step = env.step
        actions = []

        def record_step(action):
            actions.append(np.asarray(action, dtype=np.uint8).copy())
            return observed_step(action)

        env.step = record_step
        stand, result = _cell_entry_combat(env)
        committed = result.frames + (stand.frames if stand else 0)
        assert result.ok, result
        assert ledger.frame - 1 == len(actions) == committed
        final_ram = np.asarray(env.get_ram(), dtype=np.uint8).copy()
        damage = ledger.damage_report()["damage"]
        assert damage == 24 - result.snapshot.hp

        env.step = base_step
        env.em.set_state(root)
        for action in actions:
            env.step(action)
        np.testing.assert_array_equal(env.get_ram(), final_ram)
    finally:
        env.close()
