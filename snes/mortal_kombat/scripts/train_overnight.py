#!/usr/bin/env python3
"""Retrain all 12 Liu Kang fights overnight (RAM+hitbox v3, one job per fight).

Old pixel CNNs are not loaded (wrong observation). They stay on disk as
tournament fallbacks until each v3 zip exists. 16c/32t box: default is 12
parallel jobs × 2 envs.

Usage:
    uv run python snes/mortal_kombat/scripts/train_overnight.py --dry-run
    uv run python snes/mortal_kombat/scripts/train_overnight.py --steps 4000000
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
for _p in (_ROOT, _ROOT / "snes"):
    _t = str(_p)
    if _t not in sys.path:
        sys.path.insert(0, _t)

from mortal_kombat.paths import GAME_DIR, MODEL_DIR  # noqa: E402
from mortal_kombat.roster import STAGES, record_stage, v3_filename  # noqa: E402


def _job_cmd(prefix: str, steps: int, n_envs: int, max_hours: float) -> list[str]:
    script = Path(__file__).resolve().parent / "train_stage.py"
    cmd = [
        "uv",
        "run",
        "--extra",
        "ml",
        "python",
        str(script),
        "--prefix",
        prefix,
        "--steps",
        str(steps),
        "--n-envs",
        str(n_envs),
    ]
    if max_hours and max_hours > 0:
        cmd.extend(["--max-hours", str(max_hours)])
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=4_000_000)
    parser.add_argument("--n-envs", type=int, default=2)
    parser.add_argument("--jobs", type=int, default=12, help="Parallel stage jobs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--max-hours",
        type=float,
        default=10,
        help=(
            "Per-job wall cutoff writes *_ppo_{timesteps}_steps.zip, exits "
            "non-zero, and is not recorded as the incumbent. _final.zip is "
            "only for a finished step budget. 0 = no wall cap"
        ),
    )
    parser.add_argument(
        "--stages",
        default="",
        help="Comma prefixes (default: all 12 LiuKang fights)",
    )
    args = parser.parse_args()
    prefixes = [s.strip() for s in args.stages.split(",") if s.strip()] or [
        prefix for prefix, _, _ in STAGES
    ]
    log_dir = GAME_DIR / "logs" / "overnight_v3"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"stages={prefixes}")
    print(
        f"jobs={args.jobs} n_envs={args.n_envs} steps={args.steps} "
        f"max_hours={args.max_hours}"
    )
    print(f"logs={log_dir}")
    if args.dry_run:
        for prefix in prefixes:
            print(" ", " ".join(_job_cmd(prefix, args.steps, args.n_envs, args.max_hours)))
        return 0

    env = os.environ.copy()
    env.setdefault("SDL_VIDEODRIVER", "dummy")
    env.setdefault("SDL_AUDIODRIVER", "dummy")
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["TORCH_NUM_THREADS"] = "1"
    queue = list(prefixes)
    running: dict[str, subprocess.Popen] = {}
    logs: dict[str, object] = {}
    rc_by_stage: dict[str, int] = {}

    def _launch(prefix: str) -> None:
        log_path = log_dir / f"{prefix}.log"
        handle = log_path.open("w")
        logs[prefix] = handle
        proc = subprocess.Popen(
            _job_cmd(prefix, args.steps, args.n_envs, args.max_hours),
            cwd=str(GAME_DIR.parents[1]),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        running[prefix] = proc
        print(f"start {prefix} pid={proc.pid} log={log_path}", flush=True)

    while queue or running:
        while queue and len(running) < args.jobs:
            _launch(queue.pop(0))
        time.sleep(5)
        for prefix, proc in list(running.items()):
            code = proc.poll()
            if code is None:
                continue
            logs[prefix].close()
            rc_by_stage[prefix] = int(code)
            del running[prefix]
            zip_name = v3_filename(prefix)
            exists = (MODEL_DIR / zip_name).exists()
            print(f"done {prefix} rc={code} model={zip_name} exists={exists}", flush=True)
            if code == 0 and exists:
                record_stage(
                    prefix,
                    model=zip_name,
                    kind="ram_v3",
                    win_rate=None,
                    attempts=0,
                )

    failed = [p for p, rc in rc_by_stage.items() if rc != 0]
    print(f"finished {len(rc_by_stage)} jobs, failed={failed or 'none'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
