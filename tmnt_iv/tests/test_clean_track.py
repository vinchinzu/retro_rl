"""Clean-track infra: artifact stems, CLI flags, assist integrity."""

from __future__ import annotations

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
    _build_parser,
    assist_integrity,
    evaluate_clean_integrity,
    resolve_cli_paths,
)


def test_clean_artifact_stem_appends_once() -> None:
    assert clean_artifact_stem("tmnt_iv_full_hard_credits") == (
        "tmnt_iv_full_hard_credits_clean"
    )
    assert clean_artifact_stem("tmnt_iv_full_hard_clean") == (
        "tmnt_iv_full_hard_clean"
    )


def test_default_full_run_paths_assisted_unchanged() -> None:
    video, report = default_full_run_paths()
    assert video == RECORDINGS_DIR / f"{ASSISTED_FULL_RUN_STEM}.mp4"
    assert report == RECORDINGS_DIR / f"{ASSISTED_FULL_RUN_STEM}.json"

    dry_v, dry_r = default_full_run_paths(dry_run=True)
    assert dry_v == RECORDINGS_DIR / f"{ASSISTED_FULL_RUN_STEM}.mp4"
    assert dry_r == RECORDINGS_DIR / ASSISTED_FULL_RUN_DRY_REPORT


def test_default_full_run_paths_clean_isolated() -> None:
    a_video, a_report = default_full_run_paths()
    c_video, c_report = default_full_run_paths(clean=True)
    assert c_video.name == f"{CLEAN_FULL_RUN_STEM}.mp4"
    assert c_report.name == f"{CLEAN_FULL_RUN_STEM}.json"
    assert c_video != a_video
    assert c_report != a_report
    assert c_video.name != f"{ASSISTED_FULL_RUN_STEM}.mp4"
    assert c_report.name != f"{ASSISTED_FULL_RUN_STEM}.json"

    _, c_dry = default_full_run_paths(clean=True, dry_run=True)
    assert c_dry.name == f"{CLEAN_FULL_RUN_STEM}_dry_run.json"
    assert c_dry.name != ASSISTED_FULL_RUN_DRY_REPORT


def test_resolve_cli_paths_explicit_wins() -> None:
    custom_out = RECORDINGS_DIR / "custom_out.mp4"
    custom_rep = RECORDINGS_DIR / "custom_rep.json"
    video, report = resolve_cli_paths(
        output=custom_out,
        report=custom_rep,
        dry_run=True,
        clean_artifacts=True,
    )
    assert video == custom_out
    assert report == custom_rep


def test_resolve_cli_paths_clean_defaults() -> None:
    video, report = resolve_cli_paths(
        output=None,
        report=None,
        dry_run=True,
        clean_artifacts=True,
    )
    assert video.name == f"{CLEAN_FULL_RUN_STEM}.mp4"
    assert report.name == f"{CLEAN_FULL_RUN_STEM}_dry_run.json"


def test_assist_integrity_clean_requires_zeros() -> None:
    clean_metrics = RunMetrics()
    ok, flags = evaluate_clean_integrity(clean_metrics)
    assert ok is True
    assert flags["clean_assists_zero"] is True
    assert flags["emergency_hp_zero"] is True
    assert flags["iframe_guard_zero"] is True

    dirty = RunMetrics(health_guard_interventions=1)
    ok_dirty, flags_dirty = evaluate_clean_integrity(dirty)
    assert ok_dirty is False
    assert flags_dirty["clean_assists_zero"] is False
    assert flags_dirty["emergency_hp_zero"] is False

    iframe_dirty = RunMetrics(final_boss_iframe_guard_frames=10)
    ok_i, flags_i = evaluate_clean_integrity(iframe_dirty)
    assert ok_i is False
    assert flags_i["iframe_guard_zero"] is False


def test_assisted_integrity_does_not_require_clean() -> None:
    dirty = RunMetrics(health_guard_interventions=65, final_boss_iframe_guard_frames=100)
    flags = assist_integrity(dirty, require_clean_assists=False)
    assert "clean_assists_zero" not in flags
    assert flags["emergency_hp_zero"] is False
    assert flags["iframe_guard_zero"] is False


def test_parser_clean_flags() -> None:
    parser = _build_parser()
    args = parser.parse_args(["--clean", "--dry-run"])
    assert args.clean is True
    assert args.dry_run is True
    assert args.no_emergency_hp is False
    assert args.no_iframe_hold is False
    assert args.output is None
    assert args.report is None

    long_form = parser.parse_args(["--no-emergency-hp", "--no-iframe-hold"])
    assert long_form.no_emergency_hp is True
    assert long_form.no_iframe_hold is True
    assert long_form.clean is False


def test_cli_flag_matrix_defaults_assisted() -> None:
    """Mirrors main() flag resolution without running the emulator."""
    parser = _build_parser()

    def resolve(argv: list[str]) -> dict[str, object]:
        args = parser.parse_args(argv)
        emergency_hp = not (args.clean or args.no_emergency_hp)
        iframe_hold = not (args.clean or args.no_iframe_hold)
        clean = not emergency_hp and not iframe_hold
        clean_artifacts = not emergency_hp or not iframe_hold
        output, report = resolve_cli_paths(
            output=args.output,
            report=args.report,
            dry_run=args.dry_run,
            clean_artifacts=clean_artifacts,
        )
        return {
            "emergency_hp": emergency_hp,
            "iframe_hold": iframe_hold,
            "clean": clean,
            "clean_artifacts": clean_artifacts,
            "video": output.name,
            "report": report.name,
        }

    default = resolve([])
    assert default["emergency_hp"] is True
    assert default["iframe_hold"] is True
    assert default["clean"] is False
    assert default["clean_artifacts"] is False
    assert default["video"] == f"{ASSISTED_FULL_RUN_STEM}.mp4"
    assert default["report"] == f"{ASSISTED_FULL_RUN_STEM}.json"

    dry = resolve(["--dry-run"])
    assert dry["report"] == ASSISTED_FULL_RUN_DRY_REPORT
    assert dry["video"] == f"{ASSISTED_FULL_RUN_STEM}.mp4"

    clean = resolve(["--clean", "--dry-run"])
    assert clean["emergency_hp"] is False
    assert clean["iframe_hold"] is False
    assert clean["clean"] is True
    assert clean["clean_artifacts"] is True
    assert clean["video"] == f"{CLEAN_FULL_RUN_STEM}.mp4"
    assert clean["report"] == f"{CLEAN_FULL_RUN_STEM}_dry_run.json"

    # Either long form alone still isolates artifacts.
    no_hp = resolve(["--no-emergency-hp"])
    assert no_hp["emergency_hp"] is False
    assert no_hp["iframe_hold"] is True
    assert no_hp["clean"] is False
    assert no_hp["clean_artifacts"] is True
    assert no_hp["video"] == f"{CLEAN_FULL_RUN_STEM}.mp4"

    no_iframe = resolve(["--no-iframe-hold"])
    assert no_iframe["emergency_hp"] is True
    assert no_iframe["iframe_hold"] is False
    assert no_iframe["clean"] is False
    assert no_iframe["clean_artifacts"] is True
