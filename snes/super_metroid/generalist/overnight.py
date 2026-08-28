"""Population PPO until a local-morning deadline. Not a product tip.

Two workers share 16 emulator processes. Each cycle trains, evals per
session, keeps the checkpoint if it beats the heuristic, otherwise discards
the seed and respawns. Same-room Crateria first; mix after Join beats the
teacher by ``PROMOTE_MARGIN``, or after both already sit at 100% Join
(the margin cannot fire at the ceiling). A widened corpus starts a new scoring phase: ``best.zip`` keeps
the last validated checkpoint until a full-corpus candidate is evaluated and
kept, but score comparisons and status ``best`` are phase-local. The one-room
ship freeze is not a phase — occupancy and door-potential have to work on
every same-room hop, not just Landing.

```bash
uv run python -m super_metroid.generalist.overnight --until 08:00 --n-jobs 2 --n-envs 8
```
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from super_metroid.generalist.farm import spawn_worker, terminate_worker_trees
from super_metroid.paths import MODELS_DIR

DEFAULT_OUT = MODELS_DIR / "generalist" / "overnight"
DEFAULT_CYCLE = 400_000
PROMOTE_MARGIN = 0.05
DISCARD_MARGIN = 0.10
JOIN_CEILING = 1.0 - 1e-9


def parse_until(text: str, now: datetime | None = None) -> datetime:
    """``HH:MM`` is the next local occurrence (tomorrow if already past)."""

    now = now or datetime.now()
    raw = (text or "").strip()
    if "T" in raw or raw.count("-") >= 2:
        return datetime.fromisoformat(raw)
    hour, minute = (int(part) for part in raw.split(":", 1))
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return target


def _join_rate(report: dict[str, Any], key: str) -> float | None:
    block = report.get(key)
    if not isinstance(block, dict):
        return None
    value = block.get("join_rate")
    return float(value) if value is not None else None


def _stall_rate(report: dict[str, Any], key: str) -> float | None:
    block = report.get(key)
    if not isinstance(block, dict):
        return None
    value = block.get("stall_rate")
    return float(value) if value is not None else None


def should_promote_join(ppo_join: float, heuristic_join: float) -> bool:
    """Widen when PPO beats the teacher, or both already sit at 100% Join."""

    if ppo_join >= heuristic_join + PROMOTE_MARGIN:
        return True
    return heuristic_join >= JOIN_CEILING and ppo_join >= heuristic_join


def decide_keep(
    *,
    ppo_join: float,
    heuristic_join: float,
    ppo_stall: float | None,
    heuristic_stall: float | None,
) -> str:
    """keep / discard / promote. Promote is keep-and-widen the corpus."""

    if heuristic_join <= 0:
        return "keep"
    if ppo_join + DISCARD_MARGIN < heuristic_join:
        return "discard"
    if should_promote_join(ppo_join, heuristic_join):
        return "promote"
    if (
        ppo_stall is not None
        and heuristic_stall is not None
        and ppo_join >= heuristic_join
        and ppo_stall + PROMOTE_MARGIN < heuristic_stall
    ):
        return "promote"
    return "keep"


def train_command(
    *,
    python: str,
    out_dir: Path,
    seed: int,
    timesteps: int,
    n_envs: int,
    same_room: bool,
    eval_episodes: int,
    bc: bool,
    checkpoint: Path | None,
    skip_baselines: bool,
    tag: str,
) -> list[str]:
    cmd = [
        python,
        "-m",
        "super_metroid.generalist.train",
        "--subset",
        "crateria",
        "--timesteps",
        str(int(timesteps)),
        "--n-envs",
        str(int(n_envs)),
        "--eval-episodes",
        str(int(eval_episodes)),
        "--seed",
        str(int(seed)),
        "--out",
        str(out_dir),
        "--tag",
        tag,
    ]
    if same_room:
        cmd.append("--same-room")
    if bc:
        cmd.append("--bc")
    if skip_baselines:
        cmd.append("--skip-baselines")
    if checkpoint is not None:
        cmd.extend(["--checkpoint", str(checkpoint)])
    return cmd


def _load_report(out_dir: Path, tag: str, seed: int) -> dict[str, Any] | None:
    path = out_dir / f"train_{tag}_s{seed}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_live_status(path: Path) -> dict[str, Any] | None:
    """A mid-run ``status.json`` (not a finished report) for parent restart."""

    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("finished_unix"):
        return None
    if int(payload.get("cycle") or 0) < 1:
        return None
    return payload


def _checkpoint_path(value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    return path if path.is_file() else None


def _widen_to_mix(
    *,
    slots: list[dict[str, Any]],
    best: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep best.zip; worker 0 resumes it on the widened corpus, others respawn."""

    best_zip = Path(str(best["checkpoint"]))
    for index, slot in enumerate(slots):
        slot["checkpoint"] = best_zip if index == 0 and best_zip.is_file() else None
    return slots, best


def run_overnight(
    *,
    until: datetime,
    out_dir: Path = DEFAULT_OUT,
    n_jobs: int = 2,
    n_envs: int = 8,
    cycle_timesteps: int = DEFAULT_CYCLE,
    eval_episodes: int = 8,
    python: str | None = None,
) -> dict[str, Any]:
    python = python or sys.executable
    out_dir.mkdir(parents=True, exist_ok=True)
    status_path = out_dir / "status.json"
    same_room = True
    slots: list[dict[str, Any]] = [
        {"seed": i, "checkpoint": None} for i in range(n_jobs)
    ]
    best: dict[str, Any] | None = None
    heuristic_join = 0.0
    heuristic_stall = 1.0
    cycle = 0
    phase_cycle = 0
    previous_phase_best: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    deadline_reached = False
    live = load_live_status(status_path)
    if live is not None:
        cycle = int(live["cycle"])
        same_room = bool(live.get("same_room", True))
        heuristic_join = float(live.get("heuristic_join") or 0.0)
        heuristic_stall = float(live.get("heuristic_stall") or 1.0)
        best = live.get("best") if isinstance(live.get("best"), dict) else None
        previous_phase_best = (
            live.get("previous_phase_best")
            if isinstance(live.get("previous_phase_best"), dict)
            else None
        )
        history = list(live.get("history") or [])
        workers = live.get("workers") if isinstance(live.get("workers"), list) else []
        restored: list[dict[str, Any]] = []
        for index, row in enumerate(workers[:n_jobs]):
            if not isinstance(row, dict):
                continue
            restored.append(
                {
                    "seed": int(row.get("seed", index)),
                    "checkpoint": _checkpoint_path(row.get("checkpoint")),
                }
            )
        if restored:
            while len(restored) < n_jobs:
                restored.append({"seed": len(restored), "checkpoint": None})
            slots = restored
        phase_cycle = 2 if heuristic_join > 0 else 0
        if (
            same_room
            and best is not None
            and should_promote_join(float(best.get("join_rate", -1)), heuristic_join)
        ):
            slots, previous_phase_best = _widen_to_mix(slots=slots, best=best)
            same_room = False
            best = None
            heuristic_join = 0.0
            heuristic_stall = 1.0
            phase_cycle = 0
        print(
            json.dumps(
                {
                    "event": "resume",
                    "cycle": cycle,
                    "same_room": same_room,
                    "heuristic_join": heuristic_join,
                    "best": best,
                    "previous_phase_best": previous_phase_best,
                }
            ),
            flush=True,
        )

    while datetime.now() < until:
        cycle += 1
        phase_cycle += 1
        remaining = (until - datetime.now()).total_seconds()
        if remaining < 180:
            break
        tag = "same_room" if same_room else "crateria"
        skip_baselines = phase_cycle > 1 and heuristic_join > 0
        handles: list[Any] = []
        procs: list[subprocess.Popen[bytes]] = []
        for slot in slots:
            seed = int(slot["seed"])
            cmd = train_command(
                python=python,
                out_dir=out_dir,
                seed=seed,
                timesteps=cycle_timesteps,
                n_envs=n_envs,
                same_room=same_room,
                eval_episodes=eval_episodes,
                bc=cycle == 1,
                checkpoint=slot["checkpoint"],
                skip_baselines=skip_baselines,
                tag=tag,
            )
            log_path = out_dir / f"worker_{slot['seed']}.log"
            handle, proc = spawn_worker(cmd, log_path, cycle=cycle)
            handles.append(handle)
            procs.append(proc)
        codes: list[int] = []
        try:
            for proc in procs:
                wait_seconds = (until - datetime.now()).total_seconds()
                if wait_seconds <= 0:
                    raise subprocess.TimeoutExpired(
                        getattr(proc, "args", "training"), wait_seconds
                    )
                codes.append(proc.wait(timeout=wait_seconds))
        except subprocess.TimeoutExpired:
            deadline_reached = True
            terminate_worker_trees(procs)
        finally:
            for handle in handles:
                handle.close()

        if deadline_reached:
            break

        cycle_rows: list[dict[str, Any]] = []
        for slot, code in zip(slots, codes):
            seed = int(slot["seed"])
            report = _load_report(out_dir, tag, seed) if code == 0 else None
            ppo_join = _join_rate(report or {}, "ppo") if report else None
            ppo_stall = _stall_rate(report or {}, "ppo") if report else None
            if report and "heuristic" in report:
                heuristic_join = float(report["heuristic"]["join_rate"])
                heuristic_stall = float(report["heuristic"]["stall_rate"])
            decision = "discard"
            if report is not None and ppo_join is not None:
                decision = decide_keep(
                    ppo_join=ppo_join,
                    heuristic_join=heuristic_join,
                    ppo_stall=ppo_stall,
                    heuristic_stall=heuristic_stall,
                )
            zip_path = out_dir / f"ppo_{tag}_s{seed}.zip"
            if decision == "discard" or not zip_path.is_file():
                slot["checkpoint"] = None
                slot["seed"] = seed + 17 * cycle + n_jobs
            else:
                slot["checkpoint"] = zip_path
                if best is None or ppo_join >= float(best.get("join_rate", -1)):
                    best = {
                        "join_rate": ppo_join,
                        "stall_rate": ppo_stall,
                        "checkpoint": str(zip_path),
                        "seed": seed,
                        "cycle": cycle,
                        "tag": tag,
                    }
                    (out_dir / "best.zip").write_bytes(zip_path.read_bytes())
            cycle_rows.append(
                {
                    "seed": seed,
                    "code": code,
                    "join_rate": ppo_join,
                    "stall_rate": ppo_stall,
                    "decision": decision,
                    "checkpoint": str(zip_path) if zip_path.is_file() else None,
                }
            )

        if (
            same_room
            and best is not None
            and should_promote_join(float(best["join_rate"]), heuristic_join)
        ):
            slots, previous_phase_best = _widen_to_mix(slots=slots, best=best)
            same_room = False
            best = None
            heuristic_join = 0.0
            heuristic_stall = 1.0
            phase_cycle = 0

        history.append({"cycle": cycle, "workers": cycle_rows, "best": best})
        payload = {
            "until": until.isoformat(timespec="seconds"),
            "cycle": cycle,
            "same_room": same_room,
            "heuristic_join": heuristic_join,
            "heuristic_stall": heuristic_stall,
            "best": best,
            "previous_phase_best": previous_phase_best,
            "workers": cycle_rows,
            "history": history[-12:],
            "n_jobs": n_jobs,
            "n_envs": n_envs,
            "cycle_timesteps": cycle_timesteps,
            "updated_unix": time.time(),
        }
        write_status(status_path, payload)
        print(json.dumps({k: payload[k] for k in payload if k != "history"}, indent=2))

    final = {
        "until": until.isoformat(timespec="seconds"),
        "finished_unix": time.time(),
        "best": best,
        "previous_phase_best": previous_phase_best,
        "cycles": cycle,
        "deadline_reached": deadline_reached,
        "status": str(status_path),
    }
    write_status(status_path, {**final, "history": history})
    return final


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--until", default="08:00")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--cycle-timesteps", type=int, default=DEFAULT_CYCLE)
    parser.add_argument("--eval-episodes", type=int, default=8)
    args = parser.parse_args(argv)
    # One emulator thread per worker; leave the rest to the sibling job.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    until = parse_until(args.until)
    run_overnight(
        until=until,
        out_dir=args.out,
        n_jobs=args.n_jobs,
        n_envs=args.n_envs,
        cycle_timesteps=args.cycle_timesteps,
        eval_episodes=args.eval_episodes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
