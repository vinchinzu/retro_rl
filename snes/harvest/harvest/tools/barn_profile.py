"""Where the frames of one barn morning go: a per-frame profile of the planner's own cow section on a real pin.

Runs ``EXIT_TO_FARM`` .. ``EXIT_BARN`` from a morning pin exactly as the day plan does (no RAM writes, no loads)
and labels every barn frame with what ``CowChoresTask`` was doing: the job (``fodder`` ``trough`` ``talk``
``brush`` ``milk`` ``ship`` ``plan``), its stage (``walk`` ``step`` the press ``queue`` ``verify``), and whether the
farmer moved. The totals answer the questions a speed-up starts with: pixels walked, frames locked by the ROM's
animations, frames spent waiting, frames spent in resets.

    .venv/bin/python -m harvest.tools.barn_profile run PR_e2erec1_20261114_Y3_Spring_D18 --out /tmp/barn.json
    .venv/bin/python -m harvest.tools.barn_profile report /tmp/barn.json [/tmp/other.json]   # one or the mean of several

A 12-cow day (Y3 morning pins, 2026-11-14): about 22,500 frames, 12,400 px walked at 1 px/frame (the ROM runs the barn at one
tick per two frames from 11 cows up), 4,800 frames locked in the verify stage, 1.3 resets. See ``tasks/cow_task.py``.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import dataclasses
import json
import time
from pathlib import Path
from typing import Iterable, Optional

BARN_TILEMAP = 0x27
BARN_PHASES = ("NAV_TO_BARN", "ENTER_BARN", "COW_CHORES", "EXIT_BARN")


def frame_label(task) -> str:
    """What ``CowChoresTask`` is about to do this frame, as ``job.stage`` (the press queue and the plan are their own)."""
    job = task._job
    if task._action_queue:
        return f"queue:{job.kind if job else '?'}"
    if job is None:
        return "plan"
    stage = f"{job.kind}.{job.stage}"
    return stage + ".nudge" if job.stage == "walk" and task._walker._queue else stage


@contextlib.contextmanager
def labelled_frames(sink: dict):
    """Patch ``CowChoresTask.step`` so each call leaves its label in ``sink['label']``; restored on exit."""
    from harvest.tasks.cow_task import CowChoresTask

    original = CowChoresTask.step

    def step(self, world):
        sink["label"] = frame_label(self)
        return original(self, world)

    CowChoresTask.step = step
    try:
        yield
    finally:
        CowChoresTask.step = original


def run_profile(pin: str, *, max_frames: int = 120_000) -> dict:
    """Play the cow section of one morning pin and return its per-frame rows plus the phase results."""
    from harvest.core.animal_status import existing_cow_slots, read_cow_daily_flags, read_cow_happiness
    from harvest.core.ram_catalog import read_animal_slot_field
    from harvest.core.tile_catalog import ADDR_TILEMAP
    from harvest.planner.cow_cure import cure_day_phases
    from harvest.planner.day_phase_catalog import EXIT_TO_FARM_PHASE, cow_day_phases
    from harvest.planner.day_phase_types import DayPlannerPolicy
    from harvest.planner.day_plan import DayPlanTask
    from harvest.runtime.retro_setup import make_harvest_env
    from harvest.tasks.nav import get_pos_from_ram, make_action
    from retro_harness import AuditCapabilities, AuditedEnv, TaskStatus, WorldState

    env = AuditedEnv(make_harvest_env(pin, lean_step=True), capabilities=AuditCapabilities.all("harvest.barn_profile"))
    started = time.time()
    rows: list = []
    sink: dict = {}
    try:
        env.reset()
        ram = env.get_ram()
        phases = [EXIT_TO_FARM_PHASE, *cure_day_phases(ram), *cow_day_phases(ram)]
        task = DayPlanTask(phase_sequence=phases, policy=dataclasses.replace(DayPlannerPolicy(), include_end_day=False))
        task.reset(WorldState(frame=0, ram=env.get_ram(), info={}, obs=None))
        slots = existing_cow_slots(ram)
        happy_before = {s: read_cow_happiness(ram, s) for s in slots}
        previous = None
        result = None
        frame = 0
        with labelled_frames(sink):
            while frame < max_frames:
                ram = env.get_ram()
                sink["label"] = None
                result = task.step(WorldState(frame=frame, ram=ram, info={}, obs=None))
                point = get_pos_from_ram(ram)
                here = (point.x, point.y)
                moved = previous is not None and here != previous
                previous = here
                tilemap = int(ram[ADDR_TILEMAP])
                phase = (task.phase_text or "?").split()[0]
                label = sink["label"] if tilemap == BARN_TILEMAP else None
                rows.append((frame, phase, tilemap, label, int(moved), here[0], here[1]))
                if result.status != TaskStatus.RUNNING:
                    break
                env.step(result.action.action if result.action else make_action())
                frame += 1
        ram = env.get_ram()
        return {
            "pin": pin,
            "frames": frame,
            "status": result.status.value if result else None,
            "reason": str(result.reason) if result else None,
            "rows": rows,
            "phase_results": [{k: row.get(k) for k in ("phase", "status", "reason", "frames")} for row in task.phase_results],
            "homes": {s: int(read_animal_slot_field(ram, "cow", s, "home_map_raw")) for s in slots},
            "happy_before": happy_before,
            "happy_after": {s: read_cow_happiness(ram, s) for s in slots},
            "flags_after": {s: read_cow_daily_flags(ram, s) for s in slots},
            "ram_writes": env.audit().ram_writes,
            "mid_run_loads": env.audit().mid_run_loads,
            "wall_s": round(time.time() - started, 1),
        }
    finally:
        env.close()


def breakdown(profile: dict) -> dict:
    """Frames by phase and by ``job.stage`` (a walk split into moving and waiting), px walked by job, resets."""
    rows = profile["rows"]
    phases = collections.Counter(row[1] for row in rows)
    stages: collections.Counter = collections.Counter()
    walked: collections.Counter = collections.Counter()
    count = len(rows)

    def moving(index: int) -> bool:
        here = (rows[index][5], rows[index][6])
        return any((rows[j][5], rows[j][6]) != here for j in range(max(0, index - 2), min(count, index + 3)) if rows[j][2] == rows[index][2])

    for index, (_frame, _phase, tilemap, label, _moved, x, y) in enumerate(rows):
        if not label:
            continue
        kind, _, stage = label.partition(".")
        if label.startswith("queue:"):
            stages[f"{label.split(':')[1]}.queue"] += 1
        elif label == "plan":
            stages["plan"] += 1
        else:
            stage = stage.split(".")[0]
            stages[f"{kind}.{stage}" + (".move" if stage == "walk" and moving(index) else ".wait" if stage == "walk" else "")] += 1
        if index + 1 < count and rows[index + 1][2] == tilemap:
            step = abs(rows[index + 1][5] - x) + abs(rows[index + 1][6] - y)
            if step < 40:
                walked[kind.replace("queue:", "")] += step
    results = profile["phase_results"]
    return {
        "frames": profile["frames"],
        "barn_frames": sum(phases[name] for name in BARN_PHASES),
        "phases": dict(phases),
        "stages": dict(stages),
        "px_walked": dict(walked),
        "resets": sum(1 for row in results if row["phase"] == "COW_CHORES" and row["status"] == "retry"),
        "cows_home": all(home == BARN_TILEMAP for home in profile["homes"].values()),
        "min_happiness": min(profile["happy_after"].values(), default=None),
    }


def render(reports: Iterable[dict]) -> str:
    """One report, or the per-day mean of several."""
    reports = list(reports)
    n = len(reports)
    stages: collections.Counter = collections.Counter()
    walked: collections.Counter = collections.Counter()
    phases: collections.Counter = collections.Counter()
    for report in reports:
        stages.update(report["stages"])
        walked.update(report["px_walked"])
        phases.update(report["phases"])
    chores = sum(stages.values()) or 1
    lines = [
        f"{n} day(s): barn frames {sum(r['barn_frames'] for r in reports) / n:,.0f}  resets {sum(r['resets'] for r in reports) / n:.2f}"
        f"  all cows home {all(r['cows_home'] for r in reports)}  min happiness {min(r['min_happiness'] or 0 for r in reports)}",
        "phase frames: " + ", ".join(f"{name} {phases[name] / n:,.0f}" for name in BARN_PHASES),
        f"COW_CHORES frames by job.stage (mean {chores / n:,.0f}):",
    ]
    for key, value in sorted(stages.items(), key=lambda item: -item[1]):
        lines.append(f"  {key:22s}{value / n:8,.0f}  {100 * value / chores:5.1f}%")
    lines.append("px walked: " + ", ".join(f"{kind} {px / n:,.0f}" for kind, px in sorted(walked.items())) + f"  total {sum(walked.values()) / n:,.0f}")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("pin")
    run.add_argument("--out", required=True)
    run.add_argument("--max-frames", type=int, default=120_000)
    report = sub.add_parser("report")
    report.add_argument("profiles", nargs="+")
    args = parser.parse_args(argv)
    if args.command == "run":
        profile = run_profile(args.pin, max_frames=args.max_frames)
        Path(args.out).write_text(json.dumps(profile))
        print(render([breakdown(profile)]))
        return 0 if profile["status"] == "success" else 1
    print(render([breakdown(json.loads(Path(path).read_text())) for path in args.profiles]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
