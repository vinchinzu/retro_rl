"""Unit tests for scripts/compare_run.py. Tiny synthetic reports; no ROM."""

from __future__ import annotations

import json
from pathlib import Path

from zelda_i.scripts import compare_run as cli


def _stage(
    name: str,
    frames: int,
    *,
    success: bool = True,
    end_frame: int | None = None,
    frame_base: int = 0,
    max_frames: int = 10_000,
) -> dict:
    return {
        "name": name,
        "frames": frames,
        "success": success,
        "end_frame": frames if end_frame is None else end_frame,
        "frame_base": frame_base,
        "max_frames": max_frames,
    }


def _report(
    *,
    ok: bool = True,
    end_frame: int = 1000,
    failed_stage: str | None = None,
    stages: list[dict] | None = None,
) -> dict:
    return {
        "ok": ok,
        "through": "level1",
        "end_frame": end_frame,
        "failed_stage": failed_stage,
        "stages": list(stages or []),
        "poke_bombs": False,
        "poke_keys": False,
        "set_state_count": 0,
        "assist": {},
        "inventory_assist": {},
    }


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _row(diff: cli.RunDiff, name: str) -> cli.StageRow:
    for row in diff.common:
        if row.name == name:
            return row
    raise AssertionError(f"missing common stage {name!r}")


def test_matching_names() -> None:
    a = _report(
        end_frame=1000,
        stages=[_stage("boot", 199), _stage("clear", 801)],
    )
    b = _report(
        end_frame=1100,
        stages=[_stage("boot", 199), _stage("clear", 901)],
    )
    diff = cli.compare_reports(a, b)
    assert diff.end_frame_delta == 100
    assert diff.ok_a is True and diff.ok_b is True
    assert [row.name for row in diff.common] == ["boot", "clear"]
    assert _row(diff, "boot").delta == 0
    assert _row(diff, "clear").frames_a == 801
    assert _row(diff, "clear").frames_b == 901
    assert _row(diff, "clear").delta == 100
    text = cli.format_diff(diff)
    assert "frames_a" in text and "success_b" in text
    assert "boot" in text and "clear" in text


def test_missing_stage() -> None:
    a = _report(stages=[_stage("boot", 199), _stage("only_a", 50)])
    b = _report(stages=[_stage("boot", 199), _stage("only_b", 80)])
    diff = cli.compare_reports(a, b)
    assert [row.name for row in diff.common] == ["boot"]
    assert diff.only_a == ("only_a",)
    assert diff.only_b == ("only_b",)
    text = cli.format_diff(diff)
    assert "only in A: only_a" in text
    assert "only in B: only_b" in text
    assert "only_a" not in {row.name for row in diff.common}


def test_success_flip() -> None:
    a = _report(
        ok=True,
        failed_stage=None,
        stages=[_stage("ganon", 4200, success=True)],
    )
    b = _report(
        ok=False,
        failed_stage="ganon",
        end_frame=4200,
        stages=[_stage("ganon", 4200, success=False)],
    )
    diff = cli.compare_reports(a, b)
    row = _row(diff, "ganon")
    assert row.success_a is True
    assert row.success_b is False
    assert row.delta == 0
    assert diff.ok_a is True
    assert diff.ok_b is False
    assert diff.failed_stage_a is None
    assert diff.failed_stage_b == "ganon"
    text = cli.format_diff(diff)
    assert "ok            A=True  B=False" in text
    assert "failed_stage  A=None  B=ganon" in text


def test_negative_delta_speedup() -> None:
    a = _report(end_frame=1000, stages=[_stage("clear", 800)])
    b = _report(end_frame=850, stages=[_stage("clear", 650)])
    diff = cli.compare_reports(a, b)
    assert diff.end_frame_delta == -150
    assert _row(diff, "clear").delta == -150
    text = cli.format_diff(diff)
    assert "delta=-150" in text
    assert "+" not in text.split("clear", 1)[1].splitlines()[0]


def test_limit_worst_regressions() -> None:
    a = _report(
        stages=[
            _stage("fast", 200),
            _stage("worse", 100),
            _stage("mild", 50),
        ],
    )
    b = _report(
        stages=[
            _stage("fast", 150),
            _stage("worse", 180),
            _stage("mild", 60),
        ],
    )
    text = cli.format_diff(cli.compare_reports(a, b), limit=1)
    assert "worse" in text
    assert "fast" not in text
    assert "mild" not in text


def test_cli_matching_names(tmp_path: Path, capsys) -> None:
    a = _write(
        tmp_path / "a.json",
        _report(end_frame=1000, stages=[_stage("boot", 199), _stage("clear", 801)]),
    )
    b = _write(
        tmp_path / "b.json",
        _report(end_frame=1100, stages=[_stage("boot", 199), _stage("clear", 901)]),
    )
    assert cli.main([str(a), str(b)]) == 0
    out = capsys.readouterr().out
    assert "end_frame     A=1000  B=1100  delta=+100" in out
    assert "only in A: (none)" in out
    assert "only in B: (none)" in out


def test_missing_file_exits_2(tmp_path: Path, capsys) -> None:
    b = _write(tmp_path / "b.json", _report())
    missing = tmp_path / "missing.json"
    assert cli.main([str(missing), str(b)]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error:")


def test_invalid_json_exits_2(tmp_path: Path, capsys) -> None:
    a = tmp_path / "a.json"
    a.write_text("{not json", encoding="utf-8")
    b = _write(tmp_path / "b.json", _report())
    assert cli.main([str(a), str(b)]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error:")
