"""How wrong is the straight line? Linear extrapolation vs. ROM truth.

Every combat layer in this tree predicts by extending a measured velocity
(``dungeon.tracking._velocity`` -> ``TrackedObject.at`` ->
``dungeon.threat.contact_frames``). That model has never been measured
against the machine. This probe does it the only way that is not another
model: save the state, roll the ROM forward, read where the bodies actually
went, restore.

    QT_QPA_PLATFORM=offscreen uv run python nes/zelda_i/scratch/probe_rollout.py --state At4A
    QT_QPA_PLATFORM=offscreen uv run python nes/zelda_i/scratch/probe_rollout.py --state At78

Read the ``>=16px`` column against ``threat.MIN_DODGE_BODY`` (16) and
``threat.DEFAULT_HORIZON`` (32): a dodge is only finished at frame 16, so
that row is the share of dodge decisions taken on a body that is not where
the model says it is.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path

from retro_harness.env import make_env, reset_obs
from retro_harness.segment_runner import configure_headless
from zelda_i.dungeon.ids import OBJECT_NAMES
from zelda_i.dungeon.tracking import HazardClass, ObjectTracker
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot
from zelda_i.rollout import Rollout, hold

OUT_DIR = Path(__file__).resolve().parent
HORIZONS = (8, 16, 32, 48)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", default="At4A")
    parser.add_argument("--frames", type=int, default=1500)
    parser.add_argument("--every", type=int, default=10)
    parser.add_argument("--settle", type=int, default=60)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args(argv)
    configure_headless()

    env = make_env(game=GAME, state=args.state, game_dir=GAME_DIR, render_mode=None)
    reset_obs(env)
    tracker = ObjectTracker()
    rollout = Rollout(env)
    idle = hold(None, max(HORIZONS))
    err: dict[int, dict[str, list[int]]] = {
        h: collections.defaultdict(list) for h in HORIZONS
    }
    points = 0

    for frame in range(args.frames):
        # Link stands still: the question is what the *bodies* do, and a
        # walking Link changes which of them are even on screen.
        rollout.em.set_button_mask(_idle_array(), 0)
        rollout.em.step()
        tracked = tracker.observe(read_snapshot(env.get_ram()))
        if frame < args.settle or frame % args.every:
            continue
        bodies = [t for t in tracked if t.hazard is HazardClass.BODY and t.hp > 0]
        if not bodies:
            continue

        # One rollout, sampled at every horizon on the way past.
        truth = _truth_at(rollout, idle, HORIZONS)
        points += 1
        for horizon in HORIZONS:
            for body in bodies:
                seat = truth[horizon].get(body.slot)
                if seat is None or seat[2] <= 0:
                    continue  # died or left; not a prediction error
                px, py = body.at(float(horizon))
                gap = int(abs(px - seat[0]) + abs(py - seat[1]))
                name = OBJECT_NAMES.get(body.type_id, f"unk_{body.type_id:#04x}")
                err[horizon][name].append(gap)
                err[horizon]["ALL"].append(gap)

    rows = _table(args.state, points, err)
    if args.tag:
        (OUT_DIR / f"{args.tag}.json").write_text(json.dumps(rows, indent=2))
    env.close()
    return 0


def _idle_array():
    import numpy as np

    from zelda_i.rollout import press

    return np.asarray(press(), dtype=np.uint8)


def _truth_at(
    rollout: Rollout, plan, horizons: tuple[int, ...]
) -> dict[int, dict[int, tuple[int, int, int]]]:
    """Where every slot really is at each horizon, sampled in one replay."""
    import numpy as np

    out: dict[int, dict[int, tuple[int, int, int]]] = {}
    state = rollout.em.get_state()
    try:
        for index, frame in enumerate(plan, start=1):
            rollout.em.set_button_mask(np.asarray(frame, dtype=np.uint8), 0)
            rollout.em.step()
            if index in horizons:
                snap = read_snapshot(rollout.env.get_ram())
                out[index] = {
                    int(o.slot): (int(o.x), int(o.y), int(o.hp)) for o in snap.objects
                }
    finally:
        rollout.em.set_state(state)
    return out


def _table(state: str, points: int, err) -> dict:
    print(f"state={state}  sample points={points}")
    header = f"{'horizon':>8} {'n':>6} {'mean px':>9} {'median':>8} {'p90':>7} {'>=8px':>7} {'>=16px':>7}"
    print(header)
    rows = []
    for horizon in HORIZONS:
        values = err[horizon]["ALL"]
        if not values:
            continue
        ordered = sorted(values)
        row = {
            "horizon": horizon,
            "n": len(values),
            "mean": round(statistics.mean(values), 1),
            "median": statistics.median(values),
            "p90": ordered[int(len(ordered) * 0.9)],
            "ge8_pct": round(100 * sum(1 for v in values if v >= 8) / len(values)),
            "ge16_pct": round(100 * sum(1 for v in values if v >= 16) / len(values)),
        }
        rows.append(row)
        print(
            f"{row['horizon']:>8} {row['n']:>6} {row['mean']:>9.1f} "
            f"{row['median']:>8.1f} {row['p90']:>7.0f} {row['ge8_pct']:>6}% {row['ge16_pct']:>6}%"
        )
    per_type = {}
    print("\nper type @ horizon 16 (mean px error):")
    for name, values in sorted(
        err[16].items(), key=lambda kv: -statistics.mean(kv[1]) if kv[1] else 0
    ):
        if name == "ALL" or not values:
            continue
        per_type[name] = round(statistics.mean(values), 1)
        print(f"   {name:<18} n={len(values):<5} mean={per_type[name]:6.1f}")
    return {"state": state, "points": points, "rows": rows, "per_type": per_type}


if __name__ == "__main__":
    raise SystemExit(main())
