"""Assist contract: emergency HP, form-2 iframe, Clean stems, freeze abort."""

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
from tmnt_iv.paths import (
    ASSISTED_FULL_RUN_DRY_REPORT,
    ASSISTED_FULL_RUN_STEM,
    CLEAN_FULL_RUN_STEM,
    RECORDINGS_DIR,
    clean_artifact_stem,
    default_full_run_paths,
)
from tmnt_iv.scripts.record_full_hard_run import (
    RunMetrics,
    _FREEZE_ABORT_FRAMES,
    _build_parser,
    resolve_cli_paths,
)


class _FakeEnv:
    def __init__(self) -> None:
        self.writes: list[tuple[str, int]] = []

    def set_value(self, name: str, value: int) -> None:
        self.writes.append((name, value))


def test_contract_thresholds_and_writes() -> None:
    assert EMERGENCY_HP_THRESHOLD == 16
    assert EMERGENCY_HP_RESTORE == 80
    assert FORM2_IFRAME_VALUE == 1

    for health in (0, 1, 16):
        env = _FakeEnv()
        assert apply_emergency_hp(env, health) is True
        assert env.writes == [("player_hp", EMERGENCY_HP_RESTORE)]
    for health in (17, 80, 0x61, -1):
        env = _FakeEnv()
        assert apply_emergency_hp(env, health) is False
        assert env.writes == []

    env = _FakeEnv()
    assert apply_form2_iframe_hold(env, stage=9, event=0x0A) is True
    assert env.writes == [("player_iframes", FORM2_IFRAME_VALUE)]
    for stage, event in ((9, 0x0B), (8, 0x0A)):
        env = _FakeEnv()
        assert apply_form2_iframe_hold(env, stage=stage, event=event) is False


def test_clean_integrity_requires_zero_assists_and_zero_lives_lost() -> None:
    ok, flags = evaluate_clean_integrity(RunMetrics())
    assert ok is True
    assert flags["clean_assists_zero"]
    assert flags["life_losses_zero"]
    assert flags["state_loads_zero"] is not True
    assert flags["stage_writes_zero"] is not True
    assert flags["lives_writes_zero"] is not True

    ok_hp, hp_flags = evaluate_clean_integrity(RunMetrics(health_guard_interventions=1))
    assert ok_hp is False
    assert hp_flags["emergency_hp_zero"] is False

    ok_iframe, iframe_flags = evaluate_clean_integrity(
        RunMetrics(final_boss_iframe_guard_frames=10)
    )
    assert ok_iframe is False
    assert iframe_flags["iframe_guard_zero"] is False

    assisted = assist_integrity(
        RunMetrics(health_guard_interventions=65, life_losses=0),
        require_clean_assists=False,
    )
    assert "clean_assists_zero" not in assisted
    assert assisted["life_losses_zero"] is True
    proven = assist_integrity(
        RunMetrics(),
        state_loads=0,
        stage_writes=0,
        lives_writes=0,
    )
    assert proven["state_loads_zero"] is True
    assert proven["stage_writes_zero"] is True
    assert proven["lives_writes_zero"] is True


def test_clean_stems_never_overwrite_assisted() -> None:
    assert clean_artifact_stem(ASSISTED_FULL_RUN_STEM) == (
        f"{ASSISTED_FULL_RUN_STEM}_clean"
    )
    assert clean_artifact_stem(CLEAN_FULL_RUN_STEM) == CLEAN_FULL_RUN_STEM
    a_video, a_report = default_full_run_paths()
    c_video, c_report = default_full_run_paths(clean=True)
    assert a_video == RECORDINGS_DIR / f"{ASSISTED_FULL_RUN_STEM}.mp4"
    assert c_video.name == f"{CLEAN_FULL_RUN_STEM}.mp4"
    assert c_video != a_video
    assert c_report != a_report
    _, c_dry = default_full_run_paths(clean=True, dry_run=True)
    assert c_dry.name == f"{CLEAN_FULL_RUN_STEM}_dry_run.json"
    assert c_dry.name != ASSISTED_FULL_RUN_DRY_REPORT


def test_clean_cli_isolates_artifacts() -> None:
    parser = _build_parser()
    args = parser.parse_args(["--clean", "--dry-run"])
    emergency_hp = not (args.clean or args.no_emergency_hp)
    iframe_hold = not (args.clean or args.no_iframe_hold)
    video, report = resolve_cli_paths(
        output=None,
        report=None,
        dry_run=True,
        clean_artifacts=not emergency_hp or not iframe_hold,
    )
    assert emergency_hp is False
    assert iframe_hold is False
    assert video.name == f"{CLEAN_FULL_RUN_STEM}.mp4"
    assert report.name == f"{CLEAN_FULL_RUN_STEM}_dry_run.json"

    default = parser.parse_args([])
    assert default.clean is False
    d_video, d_report = resolve_cli_paths(
        output=None, report=None, dry_run=False, clean_artifacts=False
    )
    assert d_video.name == f"{ASSISTED_FULL_RUN_STEM}.mp4"
    assert d_report.name == f"{ASSISTED_FULL_RUN_STEM}.json"


def test_freeze_abort_is_above_pin_dumpster_and_rail() -> None:
    assert 8_000 <= _FREEZE_ABORT_FRAMES < 50_000
