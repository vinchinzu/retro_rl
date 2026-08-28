"""ROM-free tests for TMNT IV assist contract helpers."""

from __future__ import annotations

from tmnt_iv.assist import (
    EMERGENCY_HP_RESTORE,
    EMERGENCY_HP_THRESHOLD,
    FORM2_IFRAME_VALUE,
    apply_emergency_hp,
    apply_form2_iframe_hold,
    assist_integrity,
    evaluate_clean_integrity,
)
from tmnt_iv.scripts.record_full_hard_run import (
    RunMetrics,
    _EMERGENCY_HP_RESTORE,
    _EMERGENCY_HP_THRESHOLD,
    assist_integrity as rec_assist_integrity,
    evaluate_clean_integrity as rec_evaluate_clean_integrity,
)


class _FakeEnv:
    def __init__(self) -> None:
        self.writes: list[tuple[str, int]] = []

    def set_value(self, name: str, value: int) -> None:
        self.writes.append((name, value))


def test_contract_thresholds() -> None:
    assert EMERGENCY_HP_THRESHOLD == 16
    assert EMERGENCY_HP_RESTORE == 80
    assert FORM2_IFRAME_VALUE == 1
    assert _EMERGENCY_HP_THRESHOLD is EMERGENCY_HP_THRESHOLD
    assert _EMERGENCY_HP_RESTORE is EMERGENCY_HP_RESTORE


def test_record_full_hard_run_reexports_integrity() -> None:
    assert rec_assist_integrity is assist_integrity
    assert rec_evaluate_clean_integrity is evaluate_clean_integrity


def test_apply_emergency_hp_writes_on_threshold_and_zero() -> None:
    for health in (0, 1, 16):
        env = _FakeEnv()
        assert apply_emergency_hp(env, health) is True
        assert env.writes == [("player_hp", EMERGENCY_HP_RESTORE)]


def test_apply_emergency_hp_skips_safe_and_sentinel_hp() -> None:
    for health in (17, 28, 48, 80, 0x60, 0x61, -1):
        env = _FakeEnv()
        assert apply_emergency_hp(env, health) is False
        assert env.writes == []


def test_apply_form2_iframe_hold_only_stage9_event_0a() -> None:
    env = _FakeEnv()
    assert apply_form2_iframe_hold(env, stage=9, event=0x0A) is True
    assert env.writes == [("player_iframes", FORM2_IFRAME_VALUE)]

    for stage, event in ((9, 0x09), (9, 0x0B), (8, 0x0A), (0, 0x0A)):
        env = _FakeEnv()
        assert apply_form2_iframe_hold(env, stage=stage, event=event) is False
        assert env.writes == []


def test_assist_integrity_flags_from_module() -> None:
    clean = RunMetrics()
    ok, flags = evaluate_clean_integrity(clean)
    assert ok is True
    assert flags["clean_assists_zero"] is True
    assert flags["emergency_hp_zero"] is True
    assert flags["iframe_guard_zero"] is True

    dirty = RunMetrics(health_guard_interventions=1)
    ok_dirty, flags_dirty = evaluate_clean_integrity(dirty)
    assert ok_dirty is False
    assert flags_dirty["emergency_hp_zero"] is False

    assisted = assist_integrity(
        RunMetrics(health_guard_interventions=3, final_boss_iframe_guard_frames=12),
        require_clean_assists=False,
    )
    assert "clean_assists_zero" not in assisted
    assert assisted["emergency_hp_zero"] is False
    assert assisted["iframe_guard_zero"] is False
    assert assisted["life_losses_zero"] is True
    assert assisted["state_loads_zero"] is True
