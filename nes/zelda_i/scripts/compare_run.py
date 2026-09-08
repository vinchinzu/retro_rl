"""Compare two Survival spine JSON reports (frame deltas + success/failure).

    uv run python nes/zelda_i/scripts/compare_run.py A.json B.json
    uv run python nes/zelda_i/scripts/compare_run.py A.json B.json --limit 10

Typical A is ``nes/zelda_i/recordings/BASELINE_20260907.json``. Prints
end_frame delta (B-A), ok flags, failed_stage, a per-stage table aligned by
name, and stages only in A or only in B. ``--limit N`` keeps the N worst
regressions (largest B-A frames). Exit 0 unless a file is missing or JSON is
invalid (then 2). Stdlib only; no ROM.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EXIT_OK = 0
EXIT_BAD_INPUT = 2


@dataclass(frozen=True)
class StageRow:
    name: str
    frames_a: int | None
    frames_b: int | None
    success_a: bool | None
    success_b: bool | None

    @property
    def delta(self) -> int | None:
        if self.frames_a is None or self.frames_b is None:
            return None
        return self.frames_b - self.frames_a


@dataclass(frozen=True)
class RunDiff:
    end_frame_a: int | None
    end_frame_b: int | None
    ok_a: Any
    ok_b: Any
    failed_stage_a: Any
    failed_stage_b: Any
    common: tuple[StageRow, ...]
    only_a: tuple[str, ...]
    only_b: tuple[str, ...]

    @property
    def end_frame_delta(self) -> int | None:
        if self.end_frame_a is None or self.end_frame_b is None:
            return None
        return self.end_frame_b - self.end_frame_a


def load_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: report must be a JSON object")
    return payload


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def stages_by_name(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    stages = report.get("stages")
    if not isinstance(stages, list):
        return out
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        name = stage.get("name")
        if isinstance(name, str) and name:
            out[name] = stage
    return out


def compare_reports(report_a: dict[str, Any], report_b: dict[str, Any]) -> RunDiff:
    by_a = stages_by_name(report_a)
    by_b = stages_by_name(report_b)
    common = tuple(
        StageRow(
            name=name,
            frames_a=_as_int(by_a[name].get("frames")),
            frames_b=_as_int(by_b[name].get("frames")),
            success_a=_as_bool(by_a[name].get("success")),
            success_b=_as_bool(by_b[name].get("success")),
        )
        for name in by_a
        if name in by_b
    )
    return RunDiff(
        end_frame_a=_as_int(report_a.get("end_frame")),
        end_frame_b=_as_int(report_b.get("end_frame")),
        ok_a=report_a.get("ok"),
        ok_b=report_b.get("ok"),
        failed_stage_a=report_a.get("failed_stage"),
        failed_stage_b=report_b.get("failed_stage"),
        common=common,
        only_a=tuple(name for name in by_a if name not in by_b),
        only_b=tuple(name for name in by_b if name not in by_a),
    )


def _fmt_int(value: int | None) -> str:
    return "-" if value is None else str(value)


def _fmt_delta(value: int | None) -> str:
    if value is None:
        return "-"
    return f"+{value}" if value > 0 else str(value)


def _fmt_bool(value: bool | None) -> str:
    if value is None:
        return "-"
    return "True" if value else "False"


def _fmt_flag(value: Any) -> str:
    return "None" if value is None else str(value)


def _pad_table(rows: list[tuple[str, ...]], *, right: tuple[bool, ...]) -> list[str]:
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    lines: list[str] = []
    for row in rows:
        cells = []
        for i, cell in enumerate(row):
            cells.append(cell.rjust(widths[i]) if right[i] else cell.ljust(widths[i]))
        lines.append("  ".join(cells))
    return lines


def _ranked(common: tuple[StageRow, ...]) -> tuple[StageRow, ...]:
    return tuple(
        sorted(
            common,
            key=lambda row: (row.delta is None, -(row.delta if row.delta is not None else 0)),
        )
    )


def format_diff(diff: RunDiff, *, limit: int | None = None) -> str:
    rows = diff.common if limit is None else _ranked(diff.common)[: max(0, limit)]
    table: list[tuple[str, ...]] = [
        ("name", "frames_a", "frames_b", "delta", "success_a", "success_b"),
    ]
    for row in rows:
        table.append(
            (
                row.name,
                _fmt_int(row.frames_a),
                _fmt_int(row.frames_b),
                _fmt_delta(row.delta),
                _fmt_bool(row.success_a),
                _fmt_bool(row.success_b),
            )
        )
    only_a = ", ".join(diff.only_a) if diff.only_a else "(none)"
    only_b = ", ".join(diff.only_b) if diff.only_b else "(none)"
    lines = [
        (
            f"end_frame     A={_fmt_int(diff.end_frame_a)}  "
            f"B={_fmt_int(diff.end_frame_b)}  "
            f"delta={_fmt_delta(diff.end_frame_delta)}"
        ),
        f"ok            A={_fmt_flag(diff.ok_a)}  B={_fmt_flag(diff.ok_b)}",
        (
            f"failed_stage  A={_fmt_flag(diff.failed_stage_a)}  "
            f"B={_fmt_flag(diff.failed_stage_b)}"
        ),
        "",
        *_pad_table(table, right=(False, True, True, True, True, True)),
        "",
        f"only in A: {only_a}",
        f"only in B: {only_b}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("a", type=Path, help="baseline spine JSON (A)")
    parser.add_argument("b", type=Path, help="candidate spine JSON (B)")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="top-N worst regressions (largest B-A frames)",
    )
    args = parser.parse_args(argv)
    try:
        report_a = load_report(args.a)
        report_b = load_report(args.b)
    except (OSError, ValueError, UnicodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_BAD_INPUT
    print(format_diff(compare_reports(report_a, report_b), limit=args.limit))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
