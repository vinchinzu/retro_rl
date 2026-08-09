#!/usr/bin/env python3
"""Capture a one-shot clean-worktree baseline for the DeepSeek microtask corpus.

Runs each corpus acceptance_command exactly once, records exit code, duration,
and stdout/stderr artifacts. Does not modify product code or re-run flaky cases.

Usage:
  uv run python scripts/capture_deepseek_microtask_baseline.py
  uv run python scripts/capture_deepseek_microtask_baseline.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ALLOWED_STATUSES = frozenset({"PASS", "FAIL", "UNUSABLE"})

# Heuristics for classifying non-zero exits without re-running.
MISSING_PREREQ_RE = re.compile(
    r"""
    (?:
        ModuleNotFoundError
      | No\ module\ named
      | FileNotFoundError
      | No\ such\ file\ or\ directory
      | ERROR:\s*file\s+or\s+directory\s+not\s+found
      | collection\s+failed
      | ImportError
      | cannot\s+import\s+name
      | command\s+not\s+found
      | No\ such\ option
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

PYTEST_RAN_RE = re.compile(
    r"(?:=+\s*(?:short test summary|FAILURES|ERRORS)\s*=+)|(?:\d+\s+failed)",
    re.IGNORECASE,
)


def monorepo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_corpus_path() -> Path:
    return monorepo_root() / "docs" / "evals" / "deepseek_microtask_corpus_v1.json"


def default_ledger_path() -> Path:
    return monorepo_root() / "docs" / "evals" / "deepseek_microtask_baseline_v1.json"


def default_artifact_dir() -> Path:
    return (
        monorepo_root()
        / "docs"
        / "evals"
        / "deepseek_microtask_baseline_v1_artifacts"
    )


def git_base_commit(root: Path) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return "UNKNOWN"
    return proc.stdout.strip()


def worktree_is_clean(root: Path) -> tuple[bool, str]:
    """Tracked-tree cleanliness only (ignore untracked capture tooling/artifacts)."""
    proc = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return False, f"git status failed: {proc.stderr.strip()}"
    dirty = proc.stdout.strip()
    if dirty:
        return False, f"tracked worktree not clean:\n{dirty}"
    return True, ""


def classify_status(exit_code: int, stdout: str, stderr: str) -> tuple[str, str | None]:
    """Return (status, unusable_reason).

    Policy (rr-zmfh.3): green baselines are PASS. Red product failures, missing
    prerequisites, and collection/import errors are UNUSABLE for pilot
    scheduling. FAIL is reserved for explicit non-zero outcomes that still
    produced a complete, interpretable test run *if* a future policy wants to
    distinguish them; this capture maps red → UNUSABLE per the bead.
    """
    combined = f"{stdout}\n{stderr}"
    if exit_code == 0:
        return "PASS", None
    if MISSING_PREREQ_RE.search(combined):
        return "UNUSABLE", "missing_prerequisites_or_import_error"
    if PYTEST_RAN_RE.search(combined) or "failed" in combined.lower():
        return "UNUSABLE", "red_baseline"
    return "UNUSABLE", "non_zero_exit"


def summarize_output(text: str, *, max_chars: int = 400) -> str:
    text = text.strip()
    if not text:
        return ""
    # Prefer last non-empty lines (pytest summaries live at the end).
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""
    tail = "\n".join(lines[-8:])
    if len(tail) > max_chars:
        return "…" + tail[-(max_chars - 1) :]
    return tail


def run_one(
    case: dict[str, Any],
    *,
    root: Path,
    artifact_dir: Path,
    timeout_s: float,
) -> dict[str, Any]:
    case_id = str(case["id"])
    command = str(case["acceptance_command"]).strip()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = artifact_dir / f"{case_id}.stdout.txt"
    stderr_path = artifact_dir / f"{case_id}.stderr.txt"

    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            check=False,
        )
        exit_code = int(proc.returncode)
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        exit_code = 124
        stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
        stderr = (stderr + f"\nTIMEOUT after {timeout_s}s").strip()
        timed_out = True

    duration_ms = int(round((time.perf_counter() - started) * 1000))
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")

    if timed_out:
        status, reason = "UNUSABLE", "timeout"
    else:
        status, reason = classify_status(exit_code, stdout, stderr)

    rel_stdout = str(stdout_path.relative_to(root)).replace("\\", "/")
    rel_stderr = str(stderr_path.relative_to(root)).replace("\\", "/")

    return {
        "id": case_id,
        "acceptance_command": command,
        "exit_code": exit_code,
        "duration_ms": duration_ms,
        "status": status,
        "unusable_reason": reason,
        "stdout_summary": summarize_output(stdout),
        "stderr_summary": summarize_output(stderr),
        "artifact_paths": {
            "stdout": rel_stdout,
            "stderr": rel_stderr,
        },
        "started_at_utc": started_at,
        "timed_out": timed_out,
    }


def build_ledger(
    *,
    corpus: dict[str, Any],
    cases: list[dict[str, Any]],
    base_commit: str,
    root: Path,
    artifact_dir: Path,
    timeout_s: float,
) -> dict[str, Any]:
    captured_at = datetime.now(timezone.utc).isoformat()
    results: list[dict[str, Any]] = []
    for case in cases:
        print(f"RUN {case['id']}: {case['acceptance_command']}", flush=True)
        result = run_one(case, root=root, artifact_dir=artifact_dir, timeout_s=timeout_s)
        print(
            f"  -> status={result['status']} exit={result['exit_code']} "
            f"duration_ms={result['duration_ms']}"
            + (
                f" reason={result['unusable_reason']}"
                if result.get("unusable_reason")
                else ""
            ),
            flush=True,
        )
        results.append(result)

    counts = {"PASS": 0, "FAIL": 0, "UNUSABLE": 0}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    return {
        "schema_version": 1,
        "kind": "deepseek_microtask_baseline",
        "baseline_id": "deepseek_microtask_baseline_v1",
        "corpus_id": corpus.get("corpus_id", "deepseek_microtask_corpus_v1"),
        "corpus_path": "docs/evals/deepseek_microtask_corpus_v1.json",
        "title": "DeepSeek microtask corpus — clean-worktree baseline ledger",
        "description": (
            "One-shot pre-agent baseline for each corpus acceptance_command. "
            "Captured from a clean worktree with no product-code changes and "
            "no emulator runs. Red/flaky/missing-prereq cases are UNUSABLE."
        ),
        "bead": "rr-zmfh.3",
        "base_commit": base_commit,
        "captured_at_utc": captured_at,
        "capture_policy": {
            "runs_per_case": 1,
            "no_emulator": True,
            "no_product_code_changes": True,
            "no_command_rewrites": True,
            "red_baseline_status": "UNUSABLE",
            "allowed_statuses": sorted(ALLOWED_STATUSES),
        },
        "summary": {
            "case_count": len(results),
            "pass": counts.get("PASS", 0),
            "fail": counts.get("FAIL", 0),
            "unusable": counts.get("UNUSABLE", 0),
        },
        "cases": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Capture DeepSeek microtask corpus clean baseline ledger"
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="Corpus JSON path (default: docs/evals/deepseek_microtask_corpus_v1.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Ledger JSON path (default: docs/evals/deepseek_microtask_baseline_v1.json)",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Directory for per-case stdout/stderr artifacts",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=180.0,
        help="Per-case command timeout (default: 180)",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow capture from a dirty worktree (not recommended)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load corpus and print planned commands without running them",
    )
    args = parser.parse_args(argv)

    root = monorepo_root()
    corpus_path = args.corpus or default_corpus_path()
    output_path = args.output or default_ledger_path()
    artifact_dir = args.artifact_dir or default_artifact_dir()

    if not corpus_path.is_file():
        print(f"corpus not found: {corpus_path}", file=sys.stderr)
        return 2

    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    cases = corpus.get("cases")
    if not isinstance(cases, list) or not cases:
        print("corpus has no cases", file=sys.stderr)
        return 2

    if args.dry_run:
        for case in cases:
            print(f"{case.get('id')}: {case.get('acceptance_command')}")
        print(f"dry-run: {len(cases)} commands (not executed)")
        return 0

    if not args.allow_dirty:
        clean, msg = worktree_is_clean(root)
        if not clean:
            print(msg, file=sys.stderr)
            print(
                "Refusing to capture baseline from a dirty worktree "
                "(pass --allow-dirty to override).",
                file=sys.stderr,
            )
            return 3

    base_commit = git_base_commit(root)
    ledger = build_ledger(
        corpus=corpus,
        cases=cases,
        base_commit=base_commit,
        root=root,
        artifact_dir=artifact_dir,
        timeout_s=args.timeout_seconds,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(ledger, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {output_path} "
        f"(PASS={ledger['summary']['pass']} "
        f"FAIL={ledger['summary']['fail']} "
        f"UNUSABLE={ledger['summary']['unusable']})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
