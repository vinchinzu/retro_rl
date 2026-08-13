#!/usr/bin/env python3
"""Kinematic-window GA for the Ceres elevator shaft.

Not frame hillclimb and not runup/hold recipes. Each gene is a takeoff
window: x band, min momentum, x_sub band, L/R pump, release_vy. Seed is
``CERES_ELEV_HOPS``. Fitness: ship leave, then best_y, then fewer frames.

```bash
uv run python snes/super_metroid/scripts/probe/ceres_elev_ga.py --hours 5
```
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
_SNES_IMPORT_ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT, _SNES_IMPORT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from super_metroid.paths import GAME_DIR  # noqa: E402
from super_metroid.room_timer import format_segment_time  # noqa: E402
from super_metroid.routes.kpdr.ceres.elev_escape import (  # noqa: E402
    CeresShaftClimb,
)
from super_metroid.routes.kpdr.ceres.geometry import CERES_ELEV_HOPS  # noqa: E402
from super_metroid.takeoff import PlatformHop  # noqa: E402
from super_metroid.scripts.probe.ceres_elev_escape import (  # noqa: E402
    DEFAULT_ENTRY,
    HOP_KEY,
    _open_env,
    _run_climb,
)

DEFAULT_CKPT = GAME_DIR / "scratch" / "ceres_elev_ga.json"
_MOMS = (0, 1, 2)
_RELEASE = (0, 1, 2)


def mutate(
    hops: tuple[PlatformHop, ...], rng: random.Random
) -> tuple[PlatformHop, ...]:
    rows = list(hops)
    idx = rng.randrange(len(rows))
    hop = rows[idx]
    field = rng.choice(
        ("window", "momentum", "x_sub", "pump", "release", "side")
    )
    if field == "window":
        shift = rng.choice((-12, -8, -4, 4, 8, 12))
        lo = max(hop.x_lo, hop.takeoff.x_range[0] + shift)
        hi = min(hop.x_hi, hop.takeoff.x_range[1] + shift)
        if lo >= hi:
            lo, hi = hop.takeoff.x_range
        hop = hop.with_takeoff(x_range=(lo, hi))
    elif field == "momentum":
        hop = hop.with_takeoff(min_momentum=rng.choice(_MOMS))
    elif field == "x_sub":
        mid = rng.choice((0, 16384, 32768, 49152))
        hop = hop.with_takeoff(x_sub_range=(mid, mid + 16383))
    elif field == "pump":
        hop = hop.with_takeoff(pump=not hop.takeoff.pump)
    elif field == "release":
        hop = hop.with_takeoff(release_vy=rng.choice(_RELEASE))
    else:
        side = "LEFT" if hop.side == "RIGHT" else "RIGHT"
        hop = hop.with_takeoff(side=side)
    rows[idx] = hop
    return tuple(rows)


def crossover(
    a: tuple[PlatformHop, ...],
    b: tuple[PlatformHop, ...],
    rng: random.Random,
) -> tuple[PlatformHop, ...]:
    cut = rng.randrange(1, len(a))
    return tuple(a[:cut] + b[cut:])


def score(body: dict) -> tuple:
    ship = 1 if body.get("success") else 0
    best_y = int(body.get("best_y") or 900)
    frames = int((body.get("timing") or {}).get("frames") or 9_999)
    return (ship, -best_y, -frames)


def _eval(env, hops: tuple[PlatformHop, ...], pin_bytes: bytes) -> dict:
    env.em.set_state(pin_bytes)
    orig = CeresShaftClimb.hops
    try:
        CeresShaftClimb.hops = hops  # type: ignore[misc]
        _session, body = _run_climb(env)
    finally:
        CeresShaftClimb.hops = orig  # type: ignore[misc]
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, default=0.05)
    parser.add_argument("--pop", type=int, default=12)
    parser.add_argument("--state", default="enter")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    state_path = Path(args.state) if args.state != "enter" else DEFAULT_ENTRY
    env, loaded = _open_env(state_path)
    pin_bytes = env.em.get_state()
    deadline = time.time() + max(0.1, args.hours) * 3600

    seed = CERES_ELEV_HOPS
    pop = [seed]
    while len(pop) < args.pop:
        pop.append(mutate(seed, rng))

    best_hops = seed
    best_score = None
    best_body = None
    gen = 0
    evals = 0
    try:
        while time.time() < deadline:
            gen += 1
            scored: list[tuple[tuple, tuple[PlatformHop, ...], dict]] = []
            for hops in pop:
                body = _eval(env, hops, pin_bytes)
                evals += 1
                scored.append((score(body), hops, body))
            scored.sort(key=lambda row: row[0], reverse=True)
            if best_score is None or scored[0][0] > best_score:
                best_score, best_hops, best_body = scored[0]
                ckpt = {
                    "hop_key": HOP_KEY,
                    "state": loaded,
                    "generation": gen,
                    "evals": evals,
                    "score": list(best_score),
                    "hops": [h.to_dict() for h in best_hops],
                    "timing": (best_body or {}).get("timing"),
                    "best_y": (best_body or {}).get("best_y"),
                    "success": bool((best_body or {}).get("success")),
                    "notes": "Kinematic-window GA (x/x_sub/momentum/L+R). Not hillclimb.",
                }
                args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
                args.checkpoint.write_text(
                    json.dumps(ckpt, indent=2) + "\n", encoding="utf-8"
                )
                print(json.dumps(ckpt, indent=2))
                if best_score[0] == 1:
                    break
            elite = [row[1] for row in scored[: max(2, args.pop // 4)]]
            nxt = list(elite)
            while len(nxt) < args.pop:
                if rng.random() < 0.3 and len(elite) >= 2:
                    nxt.append(crossover(elite[0], elite[1], rng))
                else:
                    nxt.append(mutate(rng.choice(elite), rng))
            pop = nxt
    finally:
        env.close()

    if best_body is None:
        return 1
    print(
        json.dumps(
            {
                "command": "ga",
                "evals": evals,
                "generation": gen,
                "success": bool(best_body.get("success")),
                "timing": best_body.get("timing") or format_segment_time(0),
                "best_y": best_body.get("best_y"),
                "checkpoint": str(args.checkpoint),
            },
            indent=2,
        )
    )
    return 0 if best_body.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
