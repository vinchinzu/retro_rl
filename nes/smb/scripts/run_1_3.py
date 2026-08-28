#!/usr/bin/env python3
"""Isolated 1-3 from Level1_3.state: death-driven jump insert, then replay.

Completion is 1-4 control (``$075C==3``). Does not register via AreaNumber
and does not touch warp any%.

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
  uv run python -m smb.scripts.run_1_3 --search
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
  uv run python -m smb.scripts.run_1_3 --trials 2
```
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from retro_harness.env import make_env
from smb.paths import GAME_DIR, GAME_V0, RECORDINGS_DIR
from smb.ram import player_on_ground, read_snapshot
from smb.reactive_13 import (
    DEFAULT_1_3_SEED,
    DEFAULT_MAX_FRAMES,
    BunnyHopPolicy,
    Level13ReplayPolicy,
    is_1_4_control,
    play_1_3,
)
from smb.tas.fm2 import frames_to_nes9_rle_payload
from smb.tas.replay import get_state, set_state, to_action9
from smb.tas.stages import is_1_3_control, snap_fingerprint

_RB = [1, 0, 0, 0, 0, 0, 0, 1, 0]
_RBA = [1, 0, 0, 0, 0, 0, 0, 1, 1]

OUT_DIR = RECORDINGS_DIR / "segments_all_exits"
REPORT_PATH = OUT_DIR / "run_1_3_report.json"


def log(*parts: object) -> None:
    print(*parts, flush=True)


def make_1_3_env() -> Any:
    env = make_env(GAME_V0, "Level1_3", GAME_DIR, render_mode="rgb_array")
    env.reset()
    return env


def search_bunny(env: Any, start: Any) -> dict[str, Any]:
    """Sweep A-hold while bunny-hopping; 1-3 is athletic platforms."""
    t0 = time.time()
    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for hold in (8, 10, 12, 14, 16, 18, 20, 22, 24, 28, 32):
        set_state(env, start)
        policy = BunnyHopPolicy(hold_a=hold)
        result = play_1_3(env, policy=policy)
        row = {
            "hold_a": hold,
            "outcome": result["outcome"],
            "frames": result["frames"],
            "max_x": result["max_x"],
        }
        history.append(row)
        log(f"  bunny hold={hold} → {result['outcome']} max_x={result['max_x']} f{result['frames']}")
        if best is None or result["max_x"] > best["max_x"]:
            best = {**row, "recorded": result["recorded"], "snap": result["snap"]}
        if result["ok"]:
            payload = frames_to_nes9_rle_payload(
                result["recorded"],
                route_id="smb_1_3_clear",
                source=f"bunny-hop hold_a={hold} from Level1_3 (32-exit)",
                extra={
                    "level_id": "smb_1_3",
                    "start_state": "Level1_3",
                    "target": "1-4_control",
                    "hold_a": hold,
                    "verified_completed": True,
                    "completion": "dash_level=3 (LevelNumber), not AreaNumber",
                },
            )
            return {
                "ok": True,
                "method": "bunny",
                "hold_a": hold,
                "frames": result["frames"],
                "max_x": result["max_x"],
                "final": result["snap"],
                "history": history,
                "payload": payload,
                "elapsed_s": round(time.time() - t0, 1),
            }
    return {
        "ok": False,
        "method": "bunny",
        "best": {k: best[k] for k in best if k != "recorded"} if best else None,
        "history": history,
        "elapsed_s": round(time.time() - t0, 1),
    }


def _chunk(env: Any, hold_a: int, *, max_frames: int = 120) -> dict[str, Any]:
    """One jump (or run-up) from the current pose to the next landing."""
    rec: list[list[int]] = []
    snap = read_snapshot(env.get_ram())
    lives0 = int(snap.lives)
    for _ in range(8):
        if player_on_ground(env.get_ram()):
            break
        env.step(to_action9(_RB))
        rec.append(list(_RB))
        snap = read_snapshot(env.get_ram())
        if snap.lives < lives0 or snap.dying:
            return {"dead": True, "ok": False, "x": int(snap.player_x), "rec": rec, "snap": snap}
    saw_air = False
    for _ in range(hold_a):
        env.step(to_action9(_RBA))
        rec.append(list(_RBA))
        snap = read_snapshot(env.get_ram())
        if not player_on_ground(env.get_ram()):
            saw_air = True
        if snap.lives < lives0 or snap.dying:
            return {"dead": True, "ok": False, "x": int(snap.player_x), "rec": rec, "snap": snap}
        if is_1_4_control(snap):
            return {
                "dead": False,
                "ok": True,
                "x": int(snap.player_x),
                "rec": rec,
                "snap": snap,
                "state": get_state(env),
            }
    run_cap = 40 if hold_a == 0 else max_frames
    for i in range(run_cap):
        env.step(to_action9(_RB))
        rec.append(list(_RB))
        snap = read_snapshot(env.get_ram())
        if snap.lives < lives0 or snap.dying:
            return {"dead": True, "ok": False, "x": int(snap.player_x), "rec": rec, "snap": snap}
        if is_1_4_control(snap):
            return {
                "dead": False,
                "ok": True,
                "x": int(snap.player_x),
                "rec": rec,
                "snap": snap,
                "state": get_state(env),
            }
        grounded = player_on_ground(env.get_ram())
        if not grounded:
            saw_air = True
        elif hold_a == 0 and grounded and i >= 39:
            break
        elif hold_a > 0 and saw_air and grounded and i >= 2:
            break
    if hold_a > 0 and not saw_air:
        return {"dead": True, "ok": False, "x": int(snap.player_x), "rec": rec, "snap": snap}
    return {
        "dead": False,
        "ok": is_1_4_control(snap),
        "x": int(snap.player_x),
        "rec": rec,
        "snap": snap,
        "state": get_state(env),
    }


def _payload(recorded: list[list[int]], **extra: Any) -> dict[str, Any]:
    body = {
        "level_id": "smb_1_3",
        "start_state": "Level1_3",
        "target": "1-4_control",
        "verified_completed": True,
        "completion": "dash_level=3 (LevelNumber), not AreaNumber",
    }
    body.update(extra)
    return frames_to_nes9_rle_payload(
        recorded,
        route_id="smb_1_3_clear",
        source="Level1_3 32-exit isolated clear",
        extra=body,
    )


def search_greedy(env: Any, start: Any, *, hold_a: int = 20, max_deaths: int = 20) -> dict[str, Any]:
    """Bunny-hop hold=20, and at each death backtrack to earlier landings."""
    holds = (0, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 28, 32)
    t0 = time.time()
    history: list[dict[str, Any]] = []

    def run_bunny(from_state: Any, prefix: list[list[int]]) -> dict[str, Any]:
        set_state(env, from_state)
        pol = BunnyHopPolicy(hold_a=hold_a)
        rec = list(prefix)
        lives0 = int(read_snapshot(env.get_ram()).lives)
        landings: list[tuple[Any, list[list[int]], int]] = [
            (from_state, list(prefix), int(read_snapshot(env.get_ram()).player_x))
        ]
        last_ground_x = -1
        for _ in range(DEFAULT_MAX_FRAMES):
            ram = env.get_ram()
            snap = read_snapshot(ram)
            if is_1_4_control(snap):
                return {"ok": True, "rec": rec, "snap": snap, "landings": landings}
            if snap.dying or int(snap.lives) < lives0:
                return {
                    "ok": False,
                    "rec": rec,
                    "snap": snap,
                    "landings": landings,
                    "death_x": int(snap.player_x),
                }
            if player_on_ground(ram):
                x = int(snap.player_x)
                if x != last_ground_x:
                    landings.append((get_state(env), list(rec), x))
                    last_ground_x = x
            tick = pol.step(snap, on_ground=player_on_ground(ram))
            env.step(tick.action)
            rec.append([int(b) for b in tick.action[:9]])
        snap = read_snapshot(env.get_ram())
        return {"ok": False, "rec": rec, "snap": snap, "landings": landings, "death_x": int(snap.player_x)}

    state = start
    prefix: list[list[int]] = []
    bank: list[tuple[Any, list[list[int]], int]] = [(start, [], 40)]
    tried: set[tuple[int, int, int]] = set()
    for death_i in range(1, max_deaths + 1):
        trial = run_bunny(state, prefix)
        if trial["ok"]:
            log(f"  rescue bunny clear deaths={death_i} f{len(trial['rec'])}")
            return {
                "ok": True,
                "method": "bunny_rescue",
                "frames": len(trial["rec"]),
                "max_x": int(trial["snap"].player_x),
                "final": snap_fingerprint(trial["snap"]),
                "history": history,
                "payload": _payload(trial["rec"], deaths=death_i, hold_a=hold_a),
                "elapsed_s": round(time.time() - t0, 1),
            }
        death_x = int(trial.get("death_x") or 0)
        log(f"  rescue death {death_i} x={death_x} landings={len(trial['landings'])}")
        history.append({"death": death_i, "x": death_x, "n_landings": len(trial["landings"])})
        bank.extend(trial["landings"])
        rescued = False
        for land_state, land_rec, land_x in sorted(bank, key=lambda row: row[2], reverse=True):
            if land_x > death_x - 16:
                continue
            for run_f in (0, 10, 20, 30, 40):
                for hold in holds:
                    key = (land_x // 4, run_f, hold)
                    if key in tried:
                        continue
                    tried.add(key)
                    set_state(env, land_state)
                    lead: list[list[int]] = []
                    died = False
                    for _ in range(run_f):
                        env.step(to_action9(_RB))
                        lead.append(list(_RB))
                        snap_lead = read_snapshot(env.get_ram())
                        if snap_lead.dying:
                            died = True
                            break
                    if died:
                        continue
                    chunk = _chunk(env, hold)
                    if chunk["dead"]:
                        continue
                    rec = land_rec + lead + chunk["rec"]
                    if chunk["ok"]:
                        log(f"  rescue CLEAR x={land_x} run={run_f} hold={hold}")
                        return {
                            "ok": True,
                            "method": "bunny_rescue",
                            "frames": len(rec),
                            "max_x": int(chunk["x"]),
                            "final": snap_fingerprint(chunk["snap"]),
                            "history": history,
                            "payload": _payload(rec, deaths=death_i, hold=hold, run_f=run_f),
                            "elapsed_s": round(time.time() - t0, 1),
                        }
                    if int(chunk["x"]) > death_x + 8:
                        state = chunk["state"]
                        prefix = rec
                        log(
                            f"  rescue x={land_x} run={run_f} hold={hold} "
                            f"→ {chunk['x']} (was {death_x})"
                        )
                        rescued = True
                        break
                if rescued:
                    break
            if rescued:
                break
        if not rescued:
            log(f"  rescue stuck at death_x={death_x}")
            break
    return {
        "ok": False,
        "method": "bunny_rescue",
        "history": history,
        "elapsed_s": round(time.time() - t0, 1),
    }
    """Beam of landings: each step try A-holds, keep the furthest survivors."""
    holds = (0, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 28, 32)
    t0 = time.time()
    beam: list[tuple[Any, list[list[int]], int]] = [(start, [], 40)]
    history: list[dict[str, Any]] = []
    best_x = 40
    for n in range(1, max_chunks + 1):
        nxt: list[tuple[Any, list[list[int]], int]] = []
        seen_x: set[int] = set()
        for state, rec, mx in beam:
            set_state(env, state)
            snap = read_snapshot(env.get_ram())
            if is_1_4_control(snap):
                payload = frames_to_nes9_rle_payload(
                    rec,
                    route_id="smb_1_3_clear",
                    source="beam jump chunks from Level1_3 (32-exit)",
                    extra={
                        "level_id": "smb_1_3",
                        "start_state": "Level1_3",
                        "target": "1-4_control",
                        "chunks": n - 1,
                        "verified_completed": True,
                        "completion": "dash_level=3 (LevelNumber), not AreaNumber",
                    },
                )
                return {
                    "ok": True,
                    "method": "beam",
                    "frames": len(rec),
                    "max_x": int(snap.player_x),
                    "final": snap_fingerprint(snap),
                    "history": history,
                    "payload": payload,
                    "elapsed_s": round(time.time() - t0, 1),
                }
            for hold in holds:
                set_state(env, state)
                chunk = _chunk(env, hold)
                if chunk["dead"]:
                    continue
                new_rec = rec + chunk["rec"]
                x = int(chunk["x"])
                if chunk["ok"]:
                    payload = frames_to_nes9_rle_payload(
                        new_rec,
                        route_id="smb_1_3_clear",
                        source="beam jump chunks from Level1_3 (32-exit)",
                        extra={
                            "level_id": "smb_1_3",
                            "start_state": "Level1_3",
                            "target": "1-4_control",
                            "chunks": n,
                            "hold": hold,
                            "verified_completed": True,
                            "completion": "dash_level=3 (LevelNumber), not AreaNumber",
                        },
                    )
                    log(f"  beam chunk {n} hold={hold} x={x} CLEAR")
                    return {
                        "ok": True,
                        "method": "beam",
                        "frames": len(new_rec),
                        "max_x": x,
                        "final": snap_fingerprint(chunk["snap"]),
                        "history": history,
                        "payload": payload,
                        "elapsed_s": round(time.time() - t0, 1),
                    }
                key = x // 8
                if key in seen_x and x <= mx:
                    continue
                seen_x.add(key)
                nxt.append((chunk["state"], new_rec, x))
                if x > best_x:
                    best_x = x
        if not nxt:
            log(f"  beam chunk {n} empty (best_x={best_x})")
            break
        nxt.sort(key=lambda row: row[2], reverse=True)
        beam = nxt[:beam_n]
        history.append({"chunk": n, "best_x": beam[0][2], "beam": [row[2] for row in beam]})
        log(f"  beam chunk {n} xs={[row[2] for row in beam]}")
    return {
        "ok": False,
        "method": "beam",
        "max_x": best_x,
        "history": history,
        "elapsed_s": round(time.time() - t0, 1),
    }


def verify_seed(env: Any, seed_path: Path, *, trials: int = 1) -> dict[str, Any]:
    start = get_state(env)
    rows: list[dict[str, Any]] = []
    for trial in range(1, trials + 1):
        set_state(env, start)
        policy = Level13ReplayPolicy(seed_path=seed_path)
        result = play_1_3(env, policy=policy)
        log(
            f"trial {trial}/{trials} ok={result['ok']} "
            f"{result['outcome']} f{result['frames']} {result['snap']}"
        )
        rows.append({k: result[k] for k in ("ok", "outcome", "frames", "max_x", "snap")})
        if not result["ok"]:
            break
    return {
        "ok": bool(rows) and all(row["ok"] for row in rows),
        "trials": rows,
        "seed": str(seed_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search", action="store_true")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--seed", type=Path, default=DEFAULT_1_3_SEED)
    args = parser.parse_args(argv)

    t0 = time.time()
    env = make_1_3_env()
    report: dict[str, Any] = {"ok": False}
    try:
        snap0 = read_snapshot(env.get_ram())
        report["entry"] = snap_fingerprint(snap0)
        log(f"Level1_3 {report['entry']}")
        if not is_1_3_control(snap0):
            raise RuntimeError(f"Level1_3 is not 1-3 control: {report['entry']}")
        start = get_state(env)
        if args.search or not Path(args.seed).is_file():
            found = search_greedy(env, start)
            report["search"] = {k: found[k] for k in found if k != "payload"}
            if not found.get("ok"):
                report["elapsed_s"] = round(time.time() - t0, 1)
                OUT_DIR.mkdir(parents=True, exist_ok=True)
                REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                log(f"search failed {REPORT_PATH}")
                return 1
            args.seed.parent.mkdir(parents=True, exist_ok=True)
            args.seed.write_text(
                json.dumps(found["payload"], indent=2) + "\n", encoding="utf-8"
            )
            log(f"wrote {args.seed} frames={found['payload']['num_frames']}")
            env.close()
            env = make_1_3_env()
        report["verify"] = verify_seed(env, args.seed, trials=args.trials)
        report["ok"] = bool(report["verify"].get("ok"))
    finally:
        env.close()
    report["elapsed_s"] = round(time.time() - t0, 1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    log(f"report {REPORT_PATH} ok={report['ok']}")
    print(json.dumps({k: report[k] for k in ("ok", "elapsed_s") if k in report}, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
