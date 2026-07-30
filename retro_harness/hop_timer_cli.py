"""Shared helpers for hop-timer probe CLIs (offline fixture / report emit).

Game probe scripts keep ROM/boot wiring local; use these helpers for the
duplicated offline path (load JSON samples → run timer → write report).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


def add_offline_args(parser: argparse.ArgumentParser) -> None:
    """Attach common offline fixture / output flags to a probe subparser."""
    parser.add_argument(
        "--fixture",
        type=Path,
        help="JSON snapshot fixture (list or {samples|frames: [...]})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Write timing report JSON to this path",
    )
    parser.add_argument(
        "--source",
        default="offline",
        help="Report source label (default: offline)",
    )


def run_offline_probe(
    *,
    fixture: Path,
    run_offline: Callable[..., dict[str, Any]],
    snapshots_from_json: Callable[[Any], Sequence[Any]],
    source: str = "offline",
    out: Path | None = None,
) -> dict[str, Any]:
    """Load a fixture, run the game timer offline, optionally write report."""
    data = json.loads(fixture.read_text(encoding="utf-8"))
    samples = snapshots_from_json(data)
    report = run_offline(samples, source=source)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def print_visit_summary(
    report: Mapping[str, Any],
    *,
    visit_key: str = "visits",
    frames_key: str = "room_frames",
    limit: int = 10,
) -> None:
    """Print a short ranked hop summary to stdout."""
    visits = list(report.get(visit_key, []))
    visits.sort(key=lambda v: int(v.get(frames_key, 0)), reverse=True)
    print(f"visits={report.get('visit_count', len(visits))} "
          f"discontinuities={report.get('discontinuity_count', 0)}")
    for row in visits[:limit]:
        print(
            f"  seq={row.get('sequence_index')} "
            f"{frames_key}={row.get(frames_key)} "
            f"dwell={row.get('dwell_frames')} "
            f"transition={row.get('transition_frames')}"
        )


__all__ = [
    "add_offline_args",
    "print_visit_summary",
    "run_offline_probe",
]
