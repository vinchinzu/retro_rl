"""Focused tests for the DeepSeek microtask corpus validator and ship manifest."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
MANIFEST = ROOT / "docs" / "evals" / "deepseek_microtask_corpus_v1.json"

sys.path.insert(0, str(SCRIPTS))
from validate_deepseek_microtask_corpus import (  # noqa: E402
    EXPECTED_CASE_COUNT,
    load_and_validate,
    validate_corpus,
)


def _minimal_case(
    case_id: str,
    *,
    owned: list[str] | None = None,
    group: str = "g1",
    cmd: str = "uv run pytest retro_harness/tests/test_actions.py -q",
    no_emulator: bool = True,
    scope: str = "safe pure harness surface",
) -> dict:
    return {
        "id": case_id,
        "title": case_id,
        "scope": scope,
        "owned_paths": owned
        or [f"retro_harness/tests/test_{case_id.replace('-', '_')}.py"],
        "acceptance_command": cmd,
        "expected_observable": "pytest exits 0",
        "no_emulator": no_emulator,
        "conflict_group": group,
        "rubric": {
            "pass": "command exits 0",
            "fail": "command non-zero",
        },
    }


def _corpus(cases: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "kind": "deepseek_microtask_corpus",
        "corpus_id": "test",
        "cases": cases,
    }


def test_shipped_manifest_exists_and_is_object() -> None:
    assert MANIFEST.is_file()
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert data.get("kind") == "deepseek_microtask_corpus"
    assert len(data["cases"]) == EXPECTED_CASE_COUNT


def test_shipped_manifest_validates_clean() -> None:
    errors = load_and_validate(MANIFEST)
    assert errors == []


def test_shipped_case_ids_stable_and_unique() -> None:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    ids = [c["id"] for c in data["cases"]]
    assert len(ids) == len(set(ids)) == EXPECTED_CASE_COUNT
    assert ids[0] == "mt-01-actions"
    assert ids[-1] == "mt-20-docs-route-research-policy"
    for case in data["cases"]:
        assert case["no_emulator"] is True
        assert case["owned_paths"]
        assert case["scope"].strip()
        assert isinstance(case["acceptance_command"], str)
        assert case["rubric"]["pass"].strip()
        assert case["rubric"]["fail"].strip()


def test_validator_requires_exactly_20_cases() -> None:
    cases = [
        _minimal_case(f"mt-{i:02d}", owned=[f"retro_harness/mod_{i}.py"])
        for i in range(19)
    ]
    errors = validate_corpus(_corpus(cases))
    assert any("exactly 20" in e for e in errors)


def test_validator_rejects_duplicate_ids() -> None:
    cases = [
        _minimal_case(f"mt-{i:02d}", owned=[f"retro_harness/mod_{i}.py"])
        for i in range(20)
    ]
    cases[5]["id"] = cases[0]["id"]
    errors = validate_corpus(_corpus(cases))
    assert any("duplicate case id" in e for e in errors)


def test_validator_rejects_empty_scope() -> None:
    cases = [
        _minimal_case(f"mt-{i:02d}", owned=[f"retro_harness/mod_{i}.py"])
        for i in range(20)
    ]
    cases[0]["scope"] = "   "
    errors = validate_corpus(_corpus(cases))
    assert any("scope must be a nonempty string" in e for e in errors)


def test_validator_rejects_multi_command() -> None:
    cases = [
        _minimal_case(f"mt-{i:02d}", owned=[f"retro_harness/mod_{i}.py"])
        for i in range(20)
    ]
    cases[0]["acceptance_command"] = (
        "uv run pytest a.py -q && uv run pytest b.py -q"
    )
    errors = validate_corpus(_corpus(cases))
    assert any("single command" in e for e in errors)


def test_validator_rejects_emulator_like_command() -> None:
    cases = [
        _minimal_case(f"mt-{i:02d}", owned=[f"retro_harness/mod_{i}.py"])
        for i in range(20)
    ]
    cases[0]["acceptance_command"] = "uv run python -m retro_harness.setup_all_roms"
    errors = validate_corpus(_corpus(cases))
    assert any("emulator" in e.lower() or "ROM" in e for e in errors)


def test_validator_rejects_no_emulator_false() -> None:
    cases = [
        _minimal_case(
            f"mt-{i:02d}",
            owned=[f"retro_harness/mod_{i}.py"],
            no_emulator=(i != 0),
        )
        for i in range(20)
    ]
    cases[0]["no_emulator"] = False
    errors = validate_corpus(_corpus(cases))
    assert any("no_emulator must be true" in e for e in errors)


def test_validator_rejects_duplicate_ownership_in_conflict_group() -> None:
    cases = [
        _minimal_case(f"mt-{i:02d}", owned=[f"retro_harness/mod_{i}.py"], group="g")
        for i in range(20)
    ]
    cases[1]["owned_paths"] = list(cases[0]["owned_paths"])
    errors = validate_corpus(_corpus(cases))
    assert any("duplicate writable ownership" in e for e in errors)


def test_validator_allows_same_path_in_different_conflict_groups() -> None:
    cases = [
        _minimal_case(
            f"mt-{i:02d}",
            owned=[f"retro_harness/shared.py"] if i < 2 else [f"retro_harness/m_{i}.py"],
            group=f"g{i}",
        )
        for i in range(20)
    ]
    errors = validate_corpus(_corpus(cases))
    assert errors == []


def test_validator_rejects_forbidden_adventure_and_benchmark_paths() -> None:
    cases = [
        _minimal_case(f"mt-{i:02d}", owned=[f"retro_harness/mod_{i}.py"])
        for i in range(20)
    ]
    cases[0]["owned_paths"] = ["retro_harness/adventure/graph.py"]
    cases[1]["owned_paths"] = ["retro_harness/benchmark.py"]
    cases[2]["owned_paths"] = ["roms/SuperMetroid.sfc"]
    errors = validate_corpus(_corpus(cases))
    joined = "\n".join(errors)
    assert "adventure" in joined
    assert "benchmark.py" in joined
    assert "roms/" in joined


def test_validator_rejects_missing_rubric_pass_fail() -> None:
    cases = [
        _minimal_case(f"mt-{i:02d}", owned=[f"retro_harness/mod_{i}.py"])
        for i in range(20)
    ]
    cases[0]["rubric"] = {"pass": "ok"}
    errors = validate_corpus(_corpus(cases))
    assert any("rubric.fail" in e for e in errors)


def test_cli_ok_on_shipped_manifest() -> None:
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / "validate_deepseek_microtask_corpus.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK:" in proc.stdout
    assert str(EXPECTED_CASE_COUNT) in proc.stdout


def test_cli_fails_on_broken_manifest(tmp_path: Path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text(
        json.dumps(_corpus([_minimal_case("only-one")])),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "validate_deepseek_microtask_corpus.py"),
            "--manifest",
            str(broken),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1
    assert "FAIL:" in proc.stderr
