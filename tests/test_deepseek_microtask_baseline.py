"""Focused tests for the DeepSeek microtask baseline ledger validator."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
LEDGER = ROOT / "docs" / "evals" / "deepseek_microtask_baseline_v1.json"
CORPUS = ROOT / "docs" / "evals" / "deepseek_microtask_corpus_v1.json"

sys.path.insert(0, str(SCRIPTS))
from validate_deepseek_microtask_baseline import (  # noqa: E402
    ALLOWED_STATUSES,
    load_and_validate,
    validate_baseline,
)
from validate_deepseek_microtask_corpus import EXPECTED_CASE_COUNT  # noqa: E402


def _minimal_corpus_case(case_id: str, cmd: str = "uv run pytest x -q") -> dict:
    return {
        "id": case_id,
        "scope": "scope",
        "owned_paths": [f"retro_harness/{case_id}.py"],
        "acceptance_command": cmd,
        "expected_observable": "ok",
        "no_emulator": True,
        "conflict_group": "g",
        "rubric": {"pass": "p", "fail": "f"},
    }


def _minimal_ledger_case(
    case_id: str,
    *,
    cmd: str = "uv run pytest x -q",
    status: str = "PASS",
    exit_code: int = 0,
) -> dict:
    return {
        "id": case_id,
        "acceptance_command": cmd,
        "exit_code": exit_code,
        "duration_ms": 10,
        "status": status,
        "unusable_reason": None if status == "PASS" else "red_baseline",
        "stdout_summary": "ok",
        "stderr_summary": "",
        "artifact_paths": {
            "stdout": f"docs/evals/artifacts/{case_id}.stdout.txt",
            "stderr": f"docs/evals/artifacts/{case_id}.stderr.txt",
        },
    }


def _corpus(cases: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "kind": "deepseek_microtask_corpus",
        "corpus_id": "deepseek_microtask_corpus_v1",
        "cases": cases,
    }


def _ledger(cases: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "kind": "deepseek_microtask_baseline",
        "baseline_id": "deepseek_microtask_baseline_v1",
        "corpus_id": "deepseek_microtask_corpus_v1",
        "corpus_path": "docs/evals/deepseek_microtask_corpus_v1.json",
        "base_commit": "abc123",
        "captured_at_utc": "2026-08-09T00:00:00+00:00",
        "cases": cases,
    }


def test_shipped_ledger_exists_and_matches_corpus_ids() -> None:
    assert LEDGER.is_file()
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    assert ledger.get("kind") == "deepseek_microtask_baseline"
    assert ledger.get("schema_version") == 1
    assert len(ledger["cases"]) == EXPECTED_CASE_COUNT
    corpus_ids = [c["id"] for c in corpus["cases"]]
    ledger_ids = [c["id"] for c in ledger["cases"]]
    assert ledger_ids == corpus_ids


def test_shipped_ledger_validates_clean_with_artifacts() -> None:
    errors = load_and_validate(LEDGER, CORPUS, require_artifacts_exist=True)
    assert errors == []


def test_shipped_ledger_fields_and_statuses() -> None:
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    assert ledger.get("base_commit")
    assert ledger.get("captured_at_utc")
    assert ledger.get("bead") == "rr-zmfh.3"
    for case in ledger["cases"]:
        assert case["status"] in ALLOWED_STATUSES
        assert isinstance(case["exit_code"], int)
        assert isinstance(case["duration_ms"], int) and case["duration_ms"] >= 0
        assert case["acceptance_command"].strip()
        assert isinstance(case["stdout_summary"], str)
        assert isinstance(case["stderr_summary"], str)
        assert case["artifact_paths"]["stdout"]
        assert case["artifact_paths"]["stderr"]
        if case["status"] == "PASS":
            assert case["exit_code"] == 0


def test_shipped_commands_match_corpus_exactly() -> None:
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    by_id = {c["id"]: c for c in corpus["cases"]}
    for case in ledger["cases"]:
        assert case["acceptance_command"] == by_id[case["id"]]["acceptance_command"]


def test_validator_rejects_missing_corpus_case() -> None:
    cases = [_minimal_corpus_case(f"mt-{i:02d}") for i in range(20)]
    ledger_cases = [_minimal_ledger_case(f"mt-{i:02d}") for i in range(19)]
    errors = validate_baseline(_ledger(ledger_cases), _corpus(cases))
    assert any("missing from" in e and "mt-19" in e for e in errors)


def test_validator_rejects_extra_ledger_case() -> None:
    cases = [_minimal_corpus_case(f"mt-{i:02d}") for i in range(20)]
    ledger_cases = [_minimal_ledger_case(f"mt-{i:02d}") for i in range(20)]
    ledger_cases.append(_minimal_ledger_case("mt-extra"))
    errors = validate_baseline(_ledger(ledger_cases), _corpus(cases))
    assert any("mt-extra" in e and "missing from" in e for e in errors)


def test_validator_rejects_command_rewrite() -> None:
    cases = [_minimal_corpus_case(f"mt-{i:02d}", cmd="uv run pytest a -q") for i in range(20)]
    ledger_cases = [
        _minimal_ledger_case(f"mt-{i:02d}", cmd="uv run pytest a -q") for i in range(20)
    ]
    ledger_cases[0]["acceptance_command"] = "uv run pytest b -q"
    errors = validate_baseline(_ledger(ledger_cases), _corpus(cases))
    assert any("diverges from corpus" in e for e in errors)


def test_validator_rejects_invalid_status() -> None:
    cases = [_minimal_corpus_case(f"mt-{i:02d}") for i in range(20)]
    ledger_cases = [_minimal_ledger_case(f"mt-{i:02d}") for i in range(20)]
    ledger_cases[0]["status"] = "MAYBE"
    errors = validate_baseline(_ledger(ledger_cases), _corpus(cases))
    assert any("status must be one of" in e for e in errors)


def test_validator_rejects_pass_with_nonzero_exit() -> None:
    cases = [_minimal_corpus_case(f"mt-{i:02d}") for i in range(20)]
    ledger_cases = [_minimal_ledger_case(f"mt-{i:02d}") for i in range(20)]
    ledger_cases[0]["status"] = "PASS"
    ledger_cases[0]["exit_code"] = 1
    errors = validate_baseline(_ledger(ledger_cases), _corpus(cases))
    assert any("status PASS requires exit_code 0" in e for e in errors)


def test_validator_allows_unusable_red_baseline() -> None:
    cases = [_minimal_corpus_case(f"mt-{i:02d}") for i in range(20)]
    ledger_cases = [_minimal_ledger_case(f"mt-{i:02d}") for i in range(20)]
    ledger_cases[0]["status"] = "UNUSABLE"
    ledger_cases[0]["exit_code"] = 1
    ledger_cases[0]["unusable_reason"] = "red_baseline"
    errors = validate_baseline(_ledger(ledger_cases), _corpus(cases))
    assert errors == []


def test_cli_ok_on_shipped_ledger() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "validate_deepseek_microtask_baseline.py"),
            "--require-artifacts",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK:" in proc.stdout
