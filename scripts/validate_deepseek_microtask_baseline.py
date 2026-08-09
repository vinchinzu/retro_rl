#!/usr/bin/env python3
"""Validate the DeepSeek microtask baseline ledger against its corpus.

Pure standard library. Checks schema fields, status enum, 1:1 case coverage
with the corpus, and command identity (no silent rewrites).

Usage:
  uv run python scripts/validate_deepseek_microtask_baseline.py
  uv run python scripts/validate_deepseek_microtask_baseline.py --ledger PATH
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ALLOWED_STATUSES = frozenset({"PASS", "FAIL", "UNUSABLE"})
REQUIRED_ROOT_FIELDS = (
    "schema_version",
    "kind",
    "baseline_id",
    "corpus_id",
    "corpus_path",
    "base_commit",
    "captured_at_utc",
    "cases",
)
REQUIRED_CASE_FIELDS = (
    "id",
    "acceptance_command",
    "exit_code",
    "duration_ms",
    "status",
    "stdout_summary",
    "stderr_summary",
    "artifact_paths",
)


def monorepo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_ledger_path() -> Path:
    return monorepo_root() / "docs" / "evals" / "deepseek_microtask_baseline_v1.json"


def default_corpus_path() -> Path:
    return monorepo_root() / "docs" / "evals" / "deepseek_microtask_corpus_v1.json"


def _is_nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_baseline(
    ledger: Any,
    corpus: Any,
    *,
    ledger_label: str = "ledger",
    corpus_label: str = "corpus",
    require_artifacts_exist: bool = False,
    root: Path | None = None,
) -> list[str]:
    """Return human-readable validation errors (empty means OK)."""
    errors: list[str] = []

    if not isinstance(ledger, dict):
        return [f"{ledger_label}: root must be a JSON object"]
    if not isinstance(corpus, dict):
        return [f"{corpus_label}: root must be a JSON object"]

    for field in REQUIRED_ROOT_FIELDS:
        if field not in ledger:
            errors.append(f"{ledger_label}: missing required field {field!r}")

    if ledger.get("kind") != "deepseek_microtask_baseline":
        errors.append(
            f"{ledger_label}: kind must be 'deepseek_microtask_baseline' "
            f"(got {ledger.get('kind')!r})"
        )

    if ledger.get("schema_version") != 1:
        errors.append(
            f"{ledger_label}: schema_version must be 1 "
            f"(got {ledger.get('schema_version')!r})"
        )

    if not _is_nonempty_str(ledger.get("base_commit")):
        errors.append(f"{ledger_label}: base_commit must be a nonempty string")

    if not _is_nonempty_str(ledger.get("captured_at_utc")):
        errors.append(f"{ledger_label}: captured_at_utc must be a nonempty string")

    corpus_cases = corpus.get("cases")
    if not isinstance(corpus_cases, list):
        return errors + [f"{corpus_label}: 'cases' must be a list"]

    ledger_cases = ledger.get("cases")
    if not isinstance(ledger_cases, list):
        return errors + [f"{ledger_label}: 'cases' must be a list"]

    corpus_by_id: dict[str, dict[str, Any]] = {}
    for index, case in enumerate(corpus_cases):
        if not isinstance(case, dict):
            errors.append(f"{corpus_label} cases[{index}]: must be an object")
            continue
        cid = case.get("id")
        if not _is_nonempty_str(cid):
            errors.append(f"{corpus_label} cases[{index}]: missing id")
            continue
        cid = str(cid).strip()
        if cid in corpus_by_id:
            errors.append(f"{corpus_label}: duplicate case id {cid!r}")
        corpus_by_id[cid] = case

    ledger_ids: list[str] = []
    seen_ledger: set[str] = set()

    for index, case in enumerate(ledger_cases):
        label = f"{ledger_label} cases[{index}]"
        if not isinstance(case, dict):
            errors.append(f"{label}: must be an object")
            continue

        for field in REQUIRED_CASE_FIELDS:
            if field not in case:
                errors.append(f"{label}: missing required field {field!r}")

        cid = case.get("id")
        if not _is_nonempty_str(cid):
            errors.append(f"{label}: id must be a nonempty string")
            continue
        cid = str(cid).strip()
        if cid in seen_ledger:
            errors.append(f"{ledger_label}: duplicate case id {cid!r}")
        seen_ledger.add(cid)
        ledger_ids.append(cid)

        status = case.get("status")
        if status not in ALLOWED_STATUSES:
            errors.append(
                f"case {cid!r}: status must be one of "
                f"{sorted(ALLOWED_STATUSES)} (got {status!r})"
            )

        if not isinstance(case.get("exit_code"), int):
            errors.append(f"case {cid!r}: exit_code must be an int")
        if not isinstance(case.get("duration_ms"), int) or case.get("duration_ms") < 0:
            errors.append(f"case {cid!r}: duration_ms must be a non-negative int")

        if not _is_nonempty_str(case.get("acceptance_command")):
            errors.append(f"case {cid!r}: acceptance_command must be a nonempty string")

        # Summary strings may be empty when the process produced no output.
        for summary_field in ("stdout_summary", "stderr_summary"):
            if not isinstance(case.get(summary_field), str):
                errors.append(f"case {cid!r}: {summary_field} must be a string")

        artifacts = case.get("artifact_paths")
        if not isinstance(artifacts, dict):
            errors.append(f"case {cid!r}: artifact_paths must be an object")
        else:
            for key in ("stdout", "stderr"):
                path_val = artifacts.get(key)
                if not _is_nonempty_str(path_val):
                    errors.append(
                        f"case {cid!r}: artifact_paths.{key} must be a nonempty string"
                    )
                elif require_artifacts_exist and root is not None:
                    ap = root / str(path_val)
                    if not ap.is_file():
                        errors.append(
                            f"case {cid!r}: missing artifact file {path_val!r}"
                        )

        # Status policy checks (soft consistency with exit code).
        exit_code = case.get("exit_code")
        if isinstance(exit_code, int):
            if status == "PASS" and exit_code != 0:
                errors.append(
                    f"case {cid!r}: status PASS requires exit_code 0 "
                    f"(got {exit_code})"
                )
            if status == "UNUSABLE" and exit_code == 0:
                # Allow UNUSABLE+0 only if explicitly justified (e.g. flaky note).
                reason = case.get("unusable_reason")
                if reason not in {"flaky", "manual_override"}:
                    errors.append(
                        f"case {cid!r}: status UNUSABLE with exit_code 0 requires "
                        f"unusable_reason 'flaky' or 'manual_override'"
                    )

        if cid not in corpus_by_id:
            errors.append(
                f"case {cid!r}: present in ledger but missing from {corpus_label}"
            )
            continue

        corpus_cmd = str(corpus_by_id[cid].get("acceptance_command", "")).strip()
        ledger_cmd = str(case.get("acceptance_command", "")).strip()
        if corpus_cmd and ledger_cmd and corpus_cmd != ledger_cmd:
            errors.append(
                f"case {cid!r}: acceptance_command diverges from corpus "
                f"(ledger rewrote the command)"
            )

    missing = sorted(set(corpus_by_id) - seen_ledger)
    for cid in missing:
        errors.append(
            f"case {cid!r}: present in {corpus_label} but missing from {ledger_label}"
        )

    # Prefer corpus order equality when counts match.
    corpus_ids = [str(c.get("id", "")).strip() for c in corpus_cases if isinstance(c, dict)]
    if corpus_ids and ledger_ids and corpus_ids != ledger_ids and not missing:
        if set(corpus_ids) == set(ledger_ids):
            errors.append(
                f"{ledger_label}: case order differs from {corpus_label} "
                f"(coverage ok, order should match)"
            )

    ledger_corpus_id = ledger.get("corpus_id")
    corpus_id = corpus.get("corpus_id")
    if (
        _is_nonempty_str(ledger_corpus_id)
        and _is_nonempty_str(corpus_id)
        and ledger_corpus_id != corpus_id
    ):
        errors.append(
            f"{ledger_label}: corpus_id {ledger_corpus_id!r} does not match "
            f"{corpus_label} corpus_id {corpus_id!r}"
        )

    return errors


def load_json(path: Path) -> tuple[Any | None, str | None]:
    if not path.is_file():
        return None, f"not found: {path}"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON in {path}: {exc}"
    except OSError as exc:
        return None, f"cannot read {path}: {exc}"


def load_and_validate(
    ledger_path: Path,
    corpus_path: Path,
    *,
    require_artifacts_exist: bool = False,
) -> list[str]:
    ledger, err = load_json(ledger_path)
    if err:
        return [err]
    corpus, err = load_json(corpus_path)
    if err:
        return [err]
    return validate_baseline(
        ledger,
        corpus,
        ledger_label=str(ledger_path),
        corpus_label=str(corpus_path),
        require_artifacts_exist=require_artifacts_exist,
        root=monorepo_root(),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate DeepSeek microtask baseline ledger vs corpus"
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=None,
        help="Baseline ledger JSON (default: docs/evals/deepseek_microtask_baseline_v1.json)",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="Corpus JSON (default: docs/evals/deepseek_microtask_corpus_v1.json)",
    )
    parser.add_argument(
        "--require-artifacts",
        action="store_true",
        help="Require per-case stdout/stderr artifact files to exist on disk",
    )
    args = parser.parse_args(argv)
    ledger_path = args.ledger or default_ledger_path()
    corpus_path = args.corpus or default_corpus_path()
    errors = load_and_validate(
        ledger_path,
        corpus_path,
        require_artifacts_exist=args.require_artifacts,
    )
    if errors:
        print(f"FAIL: {ledger_path} ({len(errors)} error(s))", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1
    case_count = "?"
    try:
        data = json.loads(ledger_path.read_text(encoding="utf-8"))
        case_count = len(data.get("cases") or [])
    except (OSError, json.JSONDecodeError):
        pass
    print(f"OK: {ledger_path} ({case_count} cases; coverage matches corpus)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
