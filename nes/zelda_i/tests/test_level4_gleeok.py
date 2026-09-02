"""Unit tests for Level 4 Gleeok TF-exit: no restore on the Survival spine."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from zelda_i.level4.boss_combat import (
    Level4GleeokFightController,
    make_gleeok_fight_controller,
)
from zelda_i.level4.gleeok13 import run_level4_tf_suffix


def test_lab_gleeok_controller_defaults_to_restore_search() -> None:
    ctl = make_gleeok_fight_controller()
    assert ctl.continuous_mode is False
    assert ctl.state_restores == 0


def test_continuous_gleeok_forbids_state_restore() -> None:
    ctl = Level4GleeokFightController(continuous_mode=True)
    em = SimpleNamespace(
        set_state=lambda state: (_ for _ in ()).throw(AssertionError)
    )
    with pytest.raises(RuntimeError, match="forbids"):
        ctl._restore_state(SimpleNamespace(em=em), object())
    assert ctl.state_restores == 0


def test_lab_gleeok_restore_counts_set_state() -> None:
    calls: list[object] = []
    em = SimpleNamespace(set_state=lambda state: calls.append(state))
    ctl = make_gleeok_fight_controller()
    marker = object()
    ctl._restore_state(SimpleNamespace(em=em), marker)
    assert calls == [marker]
    assert ctl.state_restores == 1


def test_spine_l4_tf_suffix_uses_continuous_mode() -> None:
    src = inspect.getsource(run_level4_tf_suffix)
    assert "continuous_mode=True" in src
    assert "make_gleeok_fight_controller" in src


def test_gleeok_run_does_not_call_set_state_directly() -> None:
    restore_src = inspect.getsource(Level4GleeokFightController._restore_state)
    assert "env.em.set_state(state)" in restore_src
    run_src = inspect.getsource(Level4GleeokFightController.run)
    assert "env.em.set_state" not in run_src
    assert "_restore_state" in run_src
    assert "if self.continuous_mode:" in run_src
