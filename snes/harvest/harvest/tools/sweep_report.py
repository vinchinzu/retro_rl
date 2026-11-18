"""Tabulate Year 2-3 sweep reports: did each seeded run finish clean?

    .venv/bin/python -m harvest.tools.sweep_report recordings/y23_sweep/<tag>_s*.json

A run is CLEAN when it reached the ending (or its requested end date) with no
rescue, a finished report, intervention class Clean and zero RAM writes or mid-run
state loads. Bad days are listed: a day the floor ended, a
failed required phase or a rescue. Reports from a killed run have no ``success``
and show as UNFINISHED.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def summarize(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text())
    journal = report.get("day_journal") or []
    rescues = sum(int(row.get("rescues") or 0) for row in journal)
    failures = report.get("day_failures") or []
    rescue_rows = [f for f in failures if f.get("status") == "rescue"]
    end = report.get("end") or {}
    bad = [
        f"S{row.get('plan_season')}D{row.get('plan_day')}"
        for row in journal
        if row.get("bad_day")
    ]
    finished = bool(report.get("success")) and bool(report.get("terminal", True))
    if report.get("until_ending"):
        finished = finished and bool((report.get("ending") or {}).get("finished"))
    clean_run = report.get("clean_run") or {}
    intervened = (
        clean_run.get("intervention_class", "Clean") != "Clean"
        or int(clean_run.get("mid_run_state_loads") or 0)
        or int(clean_run.get("ram_writes") or 0)
    )
    if finished and not rescues and not rescue_rows and not intervened:
        verdict = "CLEAN"
    elif finished:
        verdict = "FINISHED+RESCUES" if (rescues or rescue_rows) else "FINISHED+INTERVENED"
    else:
        verdict = "UNFINISHED"
    return {
        "name": path.stem,
        "verdict": verdict,
        "reason": str(report.get("reason") or "")[:60],
        "end": report.get("end_date") or "",
        "days": report.get("days_completed"),
        "frames": report.get("frames"),
        "bad_days": len(bad),
        "rescues": max(rescues, len(rescue_rows)),
        "money": end.get("money"),
        "cows": end.get("num_cows"),
        "chickens": end.get("num_chickens"),
        "bad_list": bad,
    }


def render(rows: list[dict[str, Any]]) -> str:
    head = ("run", "verdict", "end", "days", "bad", "rescues", "money", "cows", "hens", "frames")
    lines = [" | ".join(head)]
    for r in rows:
        lines.append(
            " | ".join(
                str(x)
                for x in (
                    r["name"], r["verdict"], r["end"], r["days"], r["bad_days"], r["rescues"],
                    r["money"], r["cows"], r["chickens"], r["frames"],
                )
            )
        )
    clean = sum(1 for r in rows if r["verdict"] == "CLEAN")
    lines.append(f"clean {clean}/{len(rows)}")
    for r in rows:
        if r["bad_list"]:
            lines.append(f"  {r['name']}: bad days {', '.join(r['bad_list'])}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reports", nargs="+", type=Path)
    args = parser.parse_args(argv)
    rows = [summarize(path) for path in args.reports if path.exists()]
    print(render(rows))
    return 0 if rows and all(r["verdict"] == "CLEAN" for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
