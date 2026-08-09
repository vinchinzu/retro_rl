#!/usr/bin/env python3
"""Validate the DeepSeek microtask corpus manifest.

Enforces a fixed 20-case, no-emulator, non-overlapping-ownership contract for
the later OpenRouter DeepSeek pilot. Pure standard library only.

Usage:
  uv run python scripts/validate_deepseek_microtask_corpus.py
  uv run python scripts/validate_deepseek_microtask_corpus.py --manifest PATH
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REQUIRED_CASE_FIELDS = (
    "id",
    "scope",
    "owned_paths",
    "acceptance_command",
    "expected_observable",
    "no_emulator",
    "conflict_group",
    "rubric",
)

REQUIRED_RUBRIC_FIELDS = ("pass", "fail")

EXPECTED_CASE_COUNT = 20

# Owned surfaces that must never appear in this corpus.
DEFAULT_FORBIDDEN_PREFIXES = (
    "retro_harness/adventure/",
    "roms/",
    "nes/",
    "snes/",
)
DEFAULT_FORBIDDEN_PATHS = frozenset(
    {
        "retro_harness/benchmark.py",
    }
)

# Acceptance commands must not look like live emulator / ROM / service work.
EMULATOR_COMMAND_RE = re.compile(
    r"""
    (?:
        \bgym[-_]?retro\b
      | \bretro\.make\b
      | \bsetup_all_roms\b
      | \bemulator_pool\b
      | \blive_play\b
      | \bplay_session\b
      | \bmake_env\b
      | \bload_state\b
      | \b\.state\b
      | \broms/
      | \bopenrouter\b
      | \bopencode\b
      | \bcurl\b
      | \bwget\b
      | \bssh\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Single shell command only (no chaining / pipelines / subshells).
MULTI_COMMAND_RE = re.compile(r"(?:&&|\|\||;|\||`|\$\(|\n)")


def monorepo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_manifest_path() -> Path:
    return monorepo_root() / "docs" / "evals" / "deepseek_microtask_corpus_v1.json"


def _as_nonempty_str(value: Any, field: str, case_id: str) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return f"case {case_id!r}: {field} must be a nonempty string"
    return None


def _normalize_path(path: str) -> str:
    return path.strip().replace("\\", "/").lstrip("./")


def validate_corpus(data: Any, *, manifest_label: str = "manifest") -> list[str]:
    """Return a list of human-readable validation errors (empty means OK)."""
    errors: list[str] = []

    if not isinstance(data, dict):
        return [f"{manifest_label}: root must be a JSON object"]

    cases = data.get("cases")
    if not isinstance(cases, list):
        return [f"{manifest_label}: 'cases' must be a list"]

    if len(cases) != EXPECTED_CASE_COUNT:
        errors.append(
            f"{manifest_label}: expected exactly {EXPECTED_CASE_COUNT} cases, "
            f"found {len(cases)}"
        )

    constraints = data.get("constraints") or {}
    forbidden_prefixes = list(DEFAULT_FORBIDDEN_PREFIXES)
    forbidden_paths = set(DEFAULT_FORBIDDEN_PATHS)
    if isinstance(constraints, dict):
        extra_prefixes = constraints.get("forbidden_owned_path_prefixes")
        if isinstance(extra_prefixes, list):
            for p in extra_prefixes:
                if isinstance(p, str) and p:
                    forbidden_prefixes.append(_normalize_path(p))
        extra_paths = constraints.get("forbidden_owned_paths")
        if isinstance(extra_paths, list):
            for p in extra_paths:
                if isinstance(p, str) and p:
                    forbidden_paths.add(_normalize_path(p))

    seen_ids: set[str] = set()
    # conflict_group -> path -> case_id
    ownership: dict[str, dict[str, str]] = defaultdict(dict)

    for index, case in enumerate(cases):
        label = f"cases[{index}]"
        if not isinstance(case, dict):
            errors.append(f"{label}: must be an object")
            continue

        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id.strip():
            errors.append(f"{label}: missing stable nonempty 'id'")
            case_id = f"<missing-id-{index}>"
        else:
            case_id = case_id.strip()
            if case_id in seen_ids:
                errors.append(f"duplicate case id: {case_id!r}")
            seen_ids.add(case_id)

        for field in REQUIRED_CASE_FIELDS:
            if field not in case:
                errors.append(f"case {case_id!r}: missing required field {field!r}")

        err = _as_nonempty_str(case.get("scope"), "scope", case_id)
        if err:
            errors.append(err)

        err = _as_nonempty_str(
            case.get("expected_observable"), "expected_observable", case_id
        )
        if err:
            errors.append(err)

        err = _as_nonempty_str(case.get("conflict_group"), "conflict_group", case_id)
        if err:
            errors.append(err)

        if case.get("no_emulator") is not True:
            errors.append(
                f"case {case_id!r}: no_emulator must be true (got {case.get('no_emulator')!r})"
            )

        owned = case.get("owned_paths")
        if not isinstance(owned, list) or not owned:
            errors.append(
                f"case {case_id!r}: owned_paths must be a nonempty list of paths"
            )
            owned_norm: list[str] = []
        else:
            owned_norm = []
            for p in owned:
                if not isinstance(p, str) or not p.strip():
                    errors.append(
                        f"case {case_id!r}: owned_paths entries must be nonempty strings"
                    )
                    continue
                owned_norm.append(_normalize_path(p))
            if not owned_norm:
                errors.append(f"case {case_id!r}: owned_paths produced empty scope")

        for path in owned_norm:
            if path in forbidden_paths:
                errors.append(
                    f"case {case_id!r}: owned path {path!r} is forbidden"
                )
            for prefix in forbidden_prefixes:
                if path == prefix.rstrip("/") or path.startswith(prefix):
                    errors.append(
                        f"case {case_id!r}: owned path {path!r} under forbidden "
                        f"prefix {prefix!r}"
                    )
                    break

        cmd = case.get("acceptance_command")
        if not isinstance(cmd, str) or not cmd.strip():
            errors.append(
                f"case {case_id!r}: acceptance_command must be one nonempty string"
            )
        else:
            cmd = cmd.strip()
            if MULTI_COMMAND_RE.search(cmd):
                errors.append(
                    f"case {case_id!r}: acceptance_command must be a single command "
                    f"(no &&, ||, ;, |, subshells, or newlines)"
                )
            if EMULATOR_COMMAND_RE.search(cmd):
                errors.append(
                    f"case {case_id!r}: acceptance_command looks like emulator/"
                    f"ROM/external-service work: {cmd!r}"
                )

        rubric = case.get("rubric")
        if not isinstance(rubric, dict):
            errors.append(f"case {case_id!r}: rubric must be an object with pass/fail")
        else:
            for rf in REQUIRED_RUBRIC_FIELDS:
                val = rubric.get(rf)
                if not isinstance(val, str) or not val.strip():
                    errors.append(
                        f"case {case_id!r}: rubric.{rf} must be a nonempty string"
                    )

        conflict_group = case.get("conflict_group")
        if isinstance(conflict_group, str) and conflict_group.strip():
            group = conflict_group.strip()
            group_map = ownership[group]
            for path in owned_norm:
                prior = group_map.get(path)
                if prior is not None and prior != case_id:
                    errors.append(
                        f"conflict_group {group!r}: duplicate writable ownership of "
                        f"{path!r} by {prior!r} and {case_id!r}"
                    )
                else:
                    group_map[path] = case_id

    return errors


def load_and_validate(manifest_path: Path) -> list[str]:
    if not manifest_path.is_file():
        return [f"manifest not found: {manifest_path}"]
    try:
        text = manifest_path.read_text(encoding="utf-8")
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return [f"invalid JSON in {manifest_path}: {exc}"]
    except OSError as exc:
        return [f"cannot read {manifest_path}: {exc}"]
    return validate_corpus(data, manifest_label=str(manifest_path))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate DeepSeek microtask corpus manifest"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to corpus JSON (default: docs/evals/deepseek_microtask_corpus_v1.json)",
    )
    args = parser.parse_args(argv)
    manifest = args.manifest or default_manifest_path()
    errors = load_and_validate(manifest)
    if errors:
        print(f"FAIL: {manifest} ({len(errors)} error(s))", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1
    print(f"OK: {manifest} ({EXPECTED_CASE_COUNT} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
