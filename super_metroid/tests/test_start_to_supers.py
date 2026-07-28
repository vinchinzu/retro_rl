"""Unit tests for start-to-Supers report structure (no emulator)."""

from __future__ import annotations

from super_metroid.start_to_supers import (
    CONTROLLER_PATH,
    SupersRunReport,
    default_artifact_paths,
)


def test_default_artifact_paths() -> None:
    video, report = default_artifact_paths()
    assert video.name == "start_to_supers.mp4"
    assert report.name == "start_to_supers.json"


def test_controller_module_exists() -> None:
    assert CONTROLLER_PATH.is_file()


def test_supers_report_includes_super_collect_field() -> None:
    # Smoke: dataclass accepts super_collect=None for failed early exits.
    report = SupersRunReport(
        schema_version=1,
        success=False,
        outcome="failed:test",
        error="test",
        total_frames=0,
        encoded_frames=0,
        final_state={},
        splits=[],
        progress_events=[],
        transitions=[],
        segments=[],
        boss=None,
        super_collect=None,
        action_reasons=__import__("collections").Counter(),
        assist={},
        integrity={},
        route_plan={},
        policy_sources={},
        state_loads=0,
        progression_writes=0,
        video=None,
        source_policy="test",
        rom_sha256="",
        start_state="power_on",
        generated_at="",
    )
    payload = report.to_dict()
    assert "super_collect" in payload
    assert payload["super_collect"] is None
