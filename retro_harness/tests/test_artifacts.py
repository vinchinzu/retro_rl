"""Shared Clean-track artifact helpers."""

from __future__ import annotations

from pathlib import Path

from retro_harness.artifacts import clean_artifact_stem, recording_artifacts


def test_clean_artifact_stem_appends_once() -> None:
    assert clean_artifact_stem("bombs") == "bombs_clean"
    assert clean_artifact_stem("bombs_clean") == "bombs_clean"
    assert clean_artifact_stem("tmnt_iv_full_hard_credits") == (
        "tmnt_iv_full_hard_credits_clean"
    )


def test_recording_artifacts_clean_and_dry_run(tmp_path: Path) -> None:
    video, report = recording_artifacts(tmp_path, "run")
    assert video == tmp_path / "run.mp4"
    assert report == tmp_path / "run.json"

    c_video, c_report = recording_artifacts(tmp_path, "run", clean=True)
    assert c_video == tmp_path / "run_clean.mp4"
    assert c_report == tmp_path / "run_clean.json"

    dry_video, dry = recording_artifacts(tmp_path, "run", dry_run=True)
    assert dry == tmp_path / "run_dry_run.json"
    assert dry_video == tmp_path / "run.mp4"

    c_dry_video, c_dry = recording_artifacts(
        tmp_path, "run", clean=True, dry_run=True
    )
    assert c_dry == tmp_path / "run_clean_dry_run.json"
    assert c_dry_video == tmp_path / "run_clean.mp4"


def test_recording_artifacts_writes_only_inside_provided_dir(tmp_path: Path) -> None:
    nested = tmp_path / "nested" / "recordings"
    video, report = recording_artifacts(nested, "run", clean=True)
    assert video == nested / "run_clean.mp4"
    assert report == nested / "run_clean.json"
    assert video.exists() is False
    assert report.exists() is False
    assert {p for p in nested.rglob("*") if p.is_dir()} == set()
    assert nested.is_dir()
