"""Procedural tournament swap / round-boundary tests (no ROM)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mortal_kombat.ram import LIU_KANG_ID, MAX_HEALTH, make_test_ram
from mortal_kombat.roster import PIXEL_FALLBACK, v3_filename
from mortal_kombat.tournament import TournamentRunner


class DummyPolicy:
    kind = "pixel"

    def __init__(self, name: str, kind: str):
        self.name = name
        self.kind = kind
        self.resets = 0
        self.acts = 0

    def reset(self) -> None:
        self.resets += 1

    def act(self, ram, rgb, *, deterministic: bool = False):
        del ram, rgb, deterministic
        self.acts += 1
        return np.zeros(12, dtype=np.int8)


class FakeEnv:
    def __init__(self, rams: list[np.ndarray]):
        self.rams = list(rams)
        self.i = 0
        self.unwrapped = self
        self.buttons: list[np.ndarray] = []

    def get_ram(self) -> np.ndarray:
        return self.rams[min(self.i, len(self.rams) - 1)]

    def step(self, buttons):
        self.buttons.append(np.asarray(buttons).copy())
        if self.i < len(self.rams) - 1:
            self.i += 1
        return None, 0, False, False, {}

    def render(self):
        return np.zeros((224, 256, 3), dtype=np.uint8)

    def close(self) -> None:
        return None


def _loader(path: Path, kind: str) -> DummyPolicy:
    return DummyPolicy(path.name, kind)


def _models(tmp_path: Path, *prefixes: str) -> None:
    for prefix in prefixes:
        (tmp_path / PIXEL_FALLBACK[prefix]).write_bytes(b"zip")


def _fight(*, match: int = 0, p2: int = 0, p1_rounds: int = 0, p2_rounds: int = 0) -> np.ndarray:
    return make_test_ram(
        p1_character=LIU_KANG_ID,
        p2_character=p2,
        match_counter=match,
        p1_rounds=p1_rounds,
        p2_rounds=p2_rounds,
        p1_health=MAX_HEALTH,
        p2_health=MAX_HEALTH,
        timer=90,
    )


def _ko(*, match: int = 0, p2: int = 0, p1_rounds: int = 1, p2_rounds: int = 0) -> np.ndarray:
    return make_test_ram(
        p1_character=LIU_KANG_ID,
        p2_character=p2,
        match_counter=match,
        p1_rounds=p1_rounds,
        p2_rounds=p2_rounds,
        p1_health=80,
        p2_health=0,
        timer=0,
    )


def _vs(*, match: int = 0, p1_rounds: int = 2) -> np.ndarray:
    return make_test_ram(
        p1_character=LIU_KANG_ID,
        p2_character=0,
        match_counter=match,
        p1_rounds=p1_rounds,
        p2_rounds=0,
        p1_health=0,
        p2_health=0,
        timer=0,
    )


def test_swaps_fight_then_match2_after_win(tmp_path: Path) -> None:
    _models(tmp_path, "Fight", "Match2")
    rams = (
        [_fight(match=0, p2=0)] * 4
        + [_ko(match=0, p1_rounds=1)] * 3
        + [_fight(match=0, p2=0, p1_rounds=1)] * 4
        + [_ko(match=0, p1_rounds=2)] * 3
        + [_vs(match=0, p1_rounds=2)] * 6
        + [_fight(match=1, p2=1)] * 6
    )
    runner = TournamentRunner(
        tmp_path, policy_loader=_loader, menu_quiet_frames=2
    )
    result = runner.run_on(FakeEnv(rams), max_frames=len(rams))
    assert result.wins == 1
    assert result.losses == 0
    prefixes = [item.split(":")[1] for item in result.swaps if item.startswith("fight:")]
    assert prefixes[0] == "Fight"
    assert "Match2" in prefixes
    screens = [event.screen for event in result.events]
    assert "BETWEEN_ROUNDS" in screens
    assert "FIGHT" in screens


def test_round_loss_swaps_to_pixel_backup(tmp_path: Path) -> None:
    v3 = tmp_path / v3_filename("Fight")
    v3.write_bytes(b"zip")
    fallback = PIXEL_FALLBACK["Fight"]
    (tmp_path / fallback).write_bytes(b"old")
    rams = (
        [_fight(match=0, p2=0)] * 3
        + [
            make_test_ram(
                p1_character=LIU_KANG_ID,
                p2_character=0,
                match_counter=0,
                p1_rounds=0,
                p2_rounds=1,
                p1_health=0,
                p2_health=40,
                timer=0,
            )
        ]
        * 3
    )
    runner = TournamentRunner(tmp_path, policy_loader=_loader)
    result = runner.run_on(FakeEnv(rams), max_frames=len(rams))
    assert any(item.startswith("round_loss:") for item in result.swaps)
    assert any(fallback in item for item in result.swaps)


def test_fight_garbage_p2_rounds_is_not_a_loss(tmp_path: Path) -> None:
    _models(tmp_path, "Fight")
    rams = [
        _fight(match=0, p2=0),
        make_test_ram(
            p1_character=LIU_KANG_ID,
            p2_character=0,
            match_counter=0,
            p1_rounds=1,
            p2_rounds=5,
            p1_health=MAX_HEALTH,
            p2_health=1,
            timer=1,
        ),
        _fight(match=0, p2=0, p1_rounds=1),
    ]
    runner = TournamentRunner(tmp_path, policy_loader=_loader)
    result = runner.run_on(FakeEnv(rams), max_frames=len(rams))
    assert result.losses == 0


def test_goro_swap_by_opponent_id(tmp_path: Path) -> None:
    _models(tmp_path, "Fight", "Goro")
    rams = [_fight(match=10, p2=7)] * 5
    runner = TournamentRunner(tmp_path, policy_loader=_loader)
    result = runner.run_on(FakeEnv(rams), max_frames=len(rams))
    assert any("Goro" in item for item in result.swaps)


def test_quiet_start_after_match_win(tmp_path: Path) -> None:
    _models(tmp_path, "Fight")
    rams = (
        [_fight(match=0, p2=0, p1_rounds=1)] * 2
        + [_ko(match=0, p1_rounds=2)] * 2
        + [_vs(match=0, p1_rounds=2)] * 8
    )
    env = FakeEnv(rams)
    runner = TournamentRunner(
        tmp_path, policy_loader=_loader, menu_quiet_frames=5
    )
    result = runner.run_on(env, max_frames=len(rams))
    assert result.wins == 1
    # First MENU frames after the 2-0 KO should be no-op, not START.
    pressed = [int(b[3]) for b in env.buttons]  # START is button 3
    assert 0 in pressed
