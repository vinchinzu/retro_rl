"""Replay an SMB TAS movie and mark 32-exit (world, dash_level) boundaries.

Power-on fceumm often desyncs FCEUX movies (same as any% warps). Isolated
``Level1_1`` search is the working extract path: find an FM2 start that
clears 1-1, then export the body. Subsequent stages use control-relative
slices like HappyLee warps.

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
  uv run python -m smb.scripts.annotate_fm2
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
  uv run python -m smb.scripts.annotate_fm2 --isolated-1-1 --export-1-1
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
  uv run python -m smb.scripts.annotate_fm2 --search-1-2-flag --export-1-2-flag
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
  uv run python -m smb.scripts.annotate_fm2 --search-1-3 --export-1-3
```
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from smb.paths import MODELS_DIR
from smb.ram import PLAYER_STATE_DYING, read_snapshot, reached_ending
from smb.tas.annotate import AnnotateState, dash_key, is_live_control, stage_label
from smb.tas.fm2 import frames_to_nes9_rle_payload, parse_movie
from smb.reactive_12 import is_surface_control
from smb.tas.replay import IDLE, idle_until, make_level1_env, to_action9
from smb.tas.stages import is_1_3_control, is_1_4_control
from smb.tas.warpless import (
    WARPLESS_EXITS_REPORT,
    WARPLESS_FM2,
    WARPLESS_REPORT_DIR,
    WL_1_1_FM2_START,
    WL_1_1_LEAVE_FRAMES,
    WL_1_1_SETTLE,
    WL_1_2_FM2_START,
    WL_1_2_LEAVE_FRAMES,
    WL_1_3_FM2_HINT,
    WL_1_3_FM2_START,
    WL_1_3_LEAVE_FRAMES,
    summary_dict,
)


def _snap_brief(snap: Any) -> dict[str, int]:
    return {
        "world": int(snap.world),
        "level": int(snap.level),
        "dash_level": int(snap.dash_level),
        "oper_mode": int(snap.oper_mode),
        "player_state": int(snap.player_state),
        "player_x": int(snap.player_x),
        "player_y": int(snap.player_y),
        "timer": int(getattr(snap, "timer", 0) or 0),
        "lives": int(getattr(snap, "lives", 0) or 0),
    }


def annotate_poweron(
    frames: list[list[int]],
    *,
    max_frames: int | None = None,
    pad_before: int = 0,
    skip_movie: int = 0,
) -> dict[str, Any]:
    """Power-on replay; record dash-level stage marks until death/ending/cap."""
    from retro_harness.env import make_env
    from smb.paths import GAME_DIR, GAME_V0

    env = make_env(GAME_V0, "NONE", GAME_DIR, render_mode="rgb_array")
    env.reset()
    for _ in range(pad_before):
        env.step(IDLE)
    body = frames[skip_movie:]
    limit = len(body) if max_frames is None else min(len(body), max_frames)
    state = AnnotateState()
    for i in range(limit):
        env.step(to_action9(body[i]))
        abs_f = i + 1 + pad_before
        snap = read_snapshot(env.get_ram(), frame=abs_f)
        state.observe(snap, abs_f)
        if state.start_lives is not None and reached_ending(
            env.get_ram(), start_lives=state.start_lives
        ):
            state.ending_frame = abs_f
            break
        if state.death_frame is not None:
            break
    end = read_snapshot(env.get_ram(), frame=state.ending_frame or limit + pad_before)
    env.close()
    report = state.to_dict()
    report.update(
        {
            "mode": "poweron",
            "movie_frames_played": limit,
            "pad_before": pad_before,
            "skip_movie": skip_movie,
            "end_snapshot": _snap_brief(end),
        }
    )
    return report


def search_isolated_1_1(
    frames: list[list[int]],
    *,
    start_min: int = 180,
    start_max: int = 240,
    settle: int = 2,
    max_play: int = 2500,
) -> dict[str, Any]:
    """Level1_1 + settle + FM2 start grid; rank by 1-2 dash / flag-zone x."""
    from smb.tas.replay import get_state, set_state

    env = make_level1_env()
    for _ in range(settle):
        env.step(IDLE)
    pin = get_state(env)
    trials: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for si in range(start_min, start_max + 1):
        set_state(env, pin)
        start_lives: int | None = None
        max_x = 0
        death: int | None = None
        leave: int | None = None
        leave_to: str | None = None
        body = frames[si:]
        for i in range(min(len(body), max_play)):
            env.step(to_action9(body[i]))
            snap = read_snapshot(env.get_ram(), frame=i + 1)
            if start_lives is None:
                start_lives = int(snap.lives)
            px = int(snap.player_x)
            if 0 < px < 20000:
                max_x = max(max_x, px)
            if int(snap.lives) < start_lives or int(snap.player_state) == PLAYER_STATE_DYING:
                death = i + 1
                break
            world, dash = dash_key(snap)
            if world == 0 and dash >= 1 and is_live_control(snap):
                leave = i + 1
                leave_to = stage_label(world, dash)
                break
            if world == 0 and dash >= 1 and int(snap.oper_mode) in (0, 1, 2, 3):
                # castle / load also counts as 1-1 leave
                if dash != 0:
                    leave = i + 1
                    leave_to = stage_label(world, dash)
                    break
        tr = {
            "start_idx": si,
            "max_x": max_x,
            "death": death,
            "leave_frame": leave,
            "leave_to": leave_to,
            "parity": si % 2,
        }
        trials.append(tr)
        score = int(max_x)
        if leave is not None:
            score += 100_000 - int(leave)
        if best is None or score > int(best.get("_score", -1)):
            best = {**tr, "_score": score}
    env.close()
    clears = [t for t in trials if t["leave_frame"] is not None]
    return {
        "mode": "isolated_1_1",
        "settle": settle,
        "start_min": start_min,
        "start_max": start_max,
        "best": best,
        "n_clear": len(clears),
        "clears": clears[:12],
        "n_trials": len(trials),
    }


def export_1_1_slice(
    frames: list[list[int]],
    *,
    start_idx: int,
    body_frames: int,
    fm2_path: Path,
    out_path: Path | None = None,
) -> dict[str, Any]:
    """Write a 1-1 warpless body seed (no L+R sanitize)."""
    dest = out_path or (MODELS_DIR / "smb_1_1_warpless_slice.json")
    body = [list(f) for f in frames[start_idx : start_idx + body_frames]]
    payload = frames_to_nes9_rle_payload(
        body,
        route_id="smb_1_1_warpless",
        source=f"HappyLee & Mars608 warpless #3728M FM2 @{start_idx}",
        extra={
            "level_id": "smb_1_1_warpless",
            "start_state": "Level1_1",
            "settle_frames": 2,
            "game_name": "SuperMarioBros-Nes-v0",
            "target": "1_2_load",
            "body_frames": len(body),
            "leave_frames": len(body),
            "fm2": str(fm2_path),
            "fm2_start_index": start_idx,
            "route_id": "smb_all_exits",
            "stage_id": "1-1",
            "note": "32-exit / warpless 1-1. Do not fold into happylee warps slices.",
        },
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    payload["_path"] = str(dest)
    return payload


def search_1_2_flag(
    frames: list[list[int]],
    *,
    start_1_1: int = WL_1_1_FM2_START,
    body_1_1: int = WL_1_1_LEAVE_FRAMES,
    settle: int = WL_1_1_SETTLE,
    start_min: int = 2080,
    start_max: int = 2140,
    step: int = 1,
    max_play: int = 3500,
) -> dict[str, Any]:
    """After isolated warpless 1-1, search FM2 starts for 1-3 control (flag exit)."""
    from smb.tas.replay import get_state, set_state

    env = make_level1_env()
    for _ in range(settle):
        env.step(IDLE)
    play_n = min(body_1_1, len(frames) - start_1_1)
    for fr in frames[start_1_1 : start_1_1 + play_n]:
        env.step(to_action9(fr))

    wait, snap = idle_until(env, is_surface_control, max_wait=400)
    ctrl = {
        "wait": wait,
        "at_1_2": bool(is_surface_control(snap)),
        "snap": _snap_brief(snap),
    }
    if not ctrl["at_1_2"]:
        env.close()
        return {"mode": "1_2_flag", "control": ctrl, "best": None, "n_clear": 0}

    pin = get_state(env)
    start_lives = int(snap.lives)
    trials: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for si in range(start_min, start_max + 1, max(1, step)):
        set_state(env, pin)
        max_x = 0
        death: int | None = None
        leave: int | None = None
        leave_to: str | None = None
        warped = False
        body = frames[si:]
        for i in range(min(len(body), max_play)):
            env.step(to_action9(body[i]))
            now = read_snapshot(env.get_ram(), frame=i + 1)
            px = int(now.player_x)
            if 0 < px < 20000:
                max_x = max(max_x, px)
            if int(now.lives) < start_lives or int(now.player_state) == PLAYER_STATE_DYING:
                death = i + 1
                break
            if int(now.world) == 3:
                warped = True
                break
            if is_1_3_control(now):
                leave = i + 1
                leave_to = "1-3"
                break
        tr = {
            "start_idx": si,
            "max_x": max_x,
            "death": death,
            "leave_frame": leave,
            "leave_to": leave_to,
            "warped_w4": warped,
        }
        trials.append(tr)
        score = int(max_x)
        if leave is not None:
            score += 100_000 - int(leave)
        if warped:
            score -= 50_000
        if best is None or score > int(best.get("_score", -1)):
            best = {**tr, "_score": score}
        if leave is not None or (max_x or 0) > 2000:
            print(
                f"  12 si={si} max_x={max_x} leave={leave} death={death} w4={warped}",
                flush=True,
            )
    env.close()
    clears = [t for t in trials if t["leave_frame"] is not None]
    return {
        "mode": "1_2_flag",
        "control": ctrl,
        "start_min": start_min,
        "start_max": start_max,
        "step": step,
        "best": best,
        "n_clear": len(clears),
        "clears": clears[:12],
        "n_trials": len(trials),
    }


def export_1_2_flag_slice(
    frames: list[list[int]],
    *,
    start_idx: int,
    body_frames: int,
    fm2_path: Path,
    out_path: Path | None = None,
) -> dict[str, Any]:
    dest = out_path or (MODELS_DIR / "smb_1_2_warpless_flag_slice.json")
    body = [list(f) for f in frames[start_idx : start_idx + body_frames]]
    payload = frames_to_nes9_rle_payload(
        body,
        route_id="smb_1_2_warpless_flag",
        source=f"HappyLee & Mars608 warpless #3728M FM2 @{start_idx}",
        extra={
            "level_id": "smb_1_2_warpless_flag",
            "start_state": "1-2_control_after_warpless_1_1",
            "settle_frames": 0,
            "game_name": "SuperMarioBros-Nes-v0",
            "target": "1_3_control",
            "body_frames": len(body),
            "leave_frames": len(body),
            "fm2": str(fm2_path),
            "fm2_start_index": start_idx,
            "route_id": "smb_all_exits",
            "stage_id": "1-2",
            "note": "32-exit 1-2 flag pipe. Not the W4 warp. Do not fold into happylee 1-2 slice.",
        },
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    payload["_path"] = str(dest)
    return payload


def reach_1_3_control(
    frames: list[list[int]],
    *,
    start_1_1: int = WL_1_1_FM2_START,
    body_1_1: int = WL_1_1_LEAVE_FRAMES,
    settle: int = WL_1_1_SETTLE,
    start_1_2: int = WL_1_2_FM2_START,
    body_1_2: int = WL_1_2_LEAVE_FRAMES,
    max_wait: int = 400,
) -> dict[str, Any]:
    """Level1_1 → warpless 1-1 → 1-2 surface → 1-2 flag body → 1-3 control.

    Returns an open env pinned at 1-3 control when ``ok``; caller must close it.
    """
    env = make_level1_env()
    for _ in range(settle):
        env.step(IDLE)
    play_11 = min(body_1_1, len(frames) - start_1_1)
    for fr in frames[start_1_1 : start_1_1 + play_11]:
        env.step(to_action9(fr))
    wait12, snap12 = idle_until(env, is_surface_control, max_wait=max_wait)
    if not is_surface_control(snap12):
        env.close()
        return {
            "ok": False,
            "stage": "1_2_control",
            "ctrl_wait_1_2": wait12,
            "snap": _snap_brief(snap12),
        }
    play_12 = min(body_1_2, len(frames) - start_1_2)
    for fr in frames[start_1_2 : start_1_2 + play_12]:
        env.step(to_action9(fr))
    wait13, snap13 = idle_until(env, is_1_3_control, max_wait=max_wait)
    ok = is_1_3_control(snap13)
    if not ok:
        env.close()
        return {
            "ok": False,
            "stage": "1_3_control",
            "ctrl_wait_1_2": wait12,
            "ctrl_wait_1_3": wait13,
            "snap": _snap_brief(snap13),
        }
    return {
        "ok": True,
        "env": env,
        "ctrl_wait_1_2": wait12,
        "ctrl_wait_1_3": wait13,
        "snap": _snap_brief(snap13),
        "lives": int(snap13.lives),
    }


def search_1_3(
    frames: list[list[int]],
    *,
    start_min: int | None = None,
    start_max: int | None = None,
    window: int = 80,
    step: int = 1,
    max_play: int = 3200,
) -> dict[str, Any]:
    """After warpless 1-1 + 1-2 flag, search FM2 starts for 1-4 control."""
    from smb.tas.replay import get_state, set_state

    reached = reach_1_3_control(frames)
    if not reached.get("ok"):
        return {
            "mode": "1_3",
            "control": reached,
            "best": None,
            "n_clear": 0,
        }
    env = reached.pop("env")
    wait13 = int(reached["ctrl_wait_1_3"])
    center = WL_1_3_FM2_HINT + wait13
    lo = start_min if start_min is not None else max(0, center - window)
    hi = start_max if start_max is not None else center + window
    pin = get_state(env)
    start_lives = int(reached["lives"])
    trials: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    print(
        f"  13 control wait={wait13} center={center} "
        f"search {lo}..{hi} step={step} {reached['snap']}",
        flush=True,
    )
    try:
        for si in range(lo, hi + 1, max(1, step)):
            set_state(env, pin)
            max_x = 0
            death: int | None = None
            leave: int | None = None
            body = frames[si:]
            for i in range(min(len(body), max_play)):
                env.step(to_action9(body[i]))
                now = read_snapshot(env.get_ram(), frame=i + 1)
                px = int(now.player_x)
                if 0 < px < 20000:
                    max_x = max(max_x, px)
                if int(now.lives) < start_lives or int(now.player_state) == PLAYER_STATE_DYING:
                    death = i + 1
                    break
                if is_1_4_control(now):
                    leave = i + 1
                    break
            tr = {
                "start_idx": si,
                "max_x": max_x,
                "death": death,
                "leave_frame": leave,
                "leave_to": "1-4" if leave is not None else None,
            }
            trials.append(tr)
            score = int(max_x)
            if leave is not None:
                score += 100_000 - int(leave)
            if best is None or score > int(best.get("_score", -1)):
                best = {**tr, "_score": score}
            if leave is not None or (max_x or 0) > 400 or si == lo or si == hi:
                print(
                    f"  13 si={si} max_x={max_x} leave={leave} death={death}",
                    flush=True,
                )
    finally:
        env.close()
    clears = [t for t in trials if t["leave_frame"] is not None]
    return {
        "mode": "1_3",
        "control": reached,
        "center": center,
        "start_min": lo,
        "start_max": hi,
        "step": step,
        "best": best,
        "n_clear": len(clears),
        "clears": clears[:12],
        "n_trials": len(trials),
    }


def export_1_3_slice(
    frames: list[list[int]],
    *,
    start_idx: int,
    body_frames: int,
    fm2_path: Path,
    out_path: Path | None = None,
) -> dict[str, Any]:
    dest = out_path or (MODELS_DIR / "smb_1_3_warpless_slice.json")
    body = [list(f) for f in frames[start_idx : start_idx + body_frames]]
    payload = frames_to_nes9_rle_payload(
        body,
        route_id="smb_1_3_warpless",
        source=f"HappyLee & Mars608 warpless #3728M FM2 @{start_idx}",
        extra={
            "level_id": "smb_1_3_warpless",
            "start_state": "1-3_control_after_warpless_1_2_flag",
            "settle_frames": 0,
            "game_name": "SuperMarioBros-Nes-v0",
            "target": "1_4_control",
            "body_frames": len(body),
            "leave_frames": len(body),
            "fm2": str(fm2_path),
            "fm2_start_index": start_idx,
            "route_id": "smb_all_exits",
            "stage_id": "1-3",
            "note": (
                "32-exit 1-3 athletic (mushroom route in #3728M). "
                "Do not fold into happylee warps slices."
            ),
        },
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    payload["_path"] = str(dest)
    return payload


def verify_1_3_slice(
    frames: list[list[int]],
    *,
    start_idx: int = WL_1_3_FM2_START,
    body_frames: int = WL_1_3_LEAVE_FRAMES,
) -> dict[str, Any]:
    """Replay the 1-3 body from TAS 1-2 flag leave; success = 1-4 control."""
    reached = reach_1_3_control(frames)
    if not reached.get("ok"):
        return {"ok": False, "stage": "1_3_control", "control": reached}
    env = reached.pop("env")
    start_lives = int(reached["lives"])
    leave: int | None = None
    death: int | None = None
    max_x = 0
    snap = None
    try:
        body = frames[start_idx : start_idx + body_frames]
        for i, fr in enumerate(body):
            env.step(to_action9(fr))
            snap = read_snapshot(env.get_ram(), frame=i + 1)
            px = int(snap.player_x)
            if 0 < px < 20000:
                max_x = max(max_x, px)
            if int(snap.lives) < start_lives or int(snap.player_state) == PLAYER_STATE_DYING:
                death = i + 1
                break
            if is_1_4_control(snap):
                leave = i + 1
                break
    finally:
        env.close()
    return {
        "ok": leave is not None and death is None,
        "mode": "verify_1_3",
        "control": reached,
        "start_idx": start_idx,
        "body_frames": body_frames,
        "leave_frame": leave,
        "death": death,
        "max_x": max_x,
        "end_snap": _snap_brief(snap) if snap is not None else None,
    }


def isolated_1_3_settle(
    frames: list[list[int]],
    *,
    start_idx: int = WL_1_3_FM2_START,
    body_frames: int = WL_1_3_LEAVE_FRAMES,
    settle_max: int = 12,
) -> dict[str, Any]:
    """Level1_3 + settle grid + TAS 1-3 body. Isolated pin may not match TAS phase."""
    from retro_harness.env import make_env
    from smb.paths import GAME_DIR, GAME_V0
    from smb.tas.replay import get_state, set_state

    env = make_env(GAME_V0, "Level1_3", GAME_DIR, render_mode="rgb_array")
    env.reset()
    snap0 = read_snapshot(env.get_ram(), frame=0)
    if not is_1_3_control(snap0):
        env.close()
        return {
            "ok": False,
            "mode": "isolated_1_3",
            "reason": "Level1_3 is not 1-3 control",
            "snap": _snap_brief(snap0),
        }
    pin = get_state(env)
    body = frames[start_idx : start_idx + body_frames]
    trials: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    try:
        for settle in range(settle_max + 1):
            set_state(env, pin)
            for _ in range(settle):
                env.step(IDLE)
            start_lives = int(read_snapshot(env.get_ram(), frame=0).lives)
            leave: int | None = None
            death: int | None = None
            max_x = 0
            snap = None
            for i, fr in enumerate(body):
                env.step(to_action9(fr))
                snap = read_snapshot(env.get_ram(), frame=i + 1)
                px = int(snap.player_x)
                if 0 < px < 20000:
                    max_x = max(max_x, px)
                if int(snap.lives) < start_lives or int(snap.player_state) == PLAYER_STATE_DYING:
                    death = i + 1
                    break
                if is_1_4_control(snap):
                    leave = i + 1
                    break
            tr = {
                "settle": settle,
                "max_x": max_x,
                "death": death,
                "leave_frame": leave,
                "end_snap": _snap_brief(snap) if snap is not None else None,
            }
            trials.append(tr)
            print(
                f"  13 isolated settle={settle} max_x={max_x} leave={leave} death={death}",
                flush=True,
            )
            score = int(max_x)
            if leave is not None:
                score += 100_000 - int(leave)
            if best is None or score > int(best.get("_score", -1)):
                best = {**tr, "_score": score}
            if leave is not None:
                break
    finally:
        env.close()
    return {
        "ok": best is not None and best.get("leave_frame") is not None,
        "mode": "isolated_1_3",
        "entry": _snap_brief(snap0),
        "start_idx": start_idx,
        "body_frames": body_frames,
        "best": best,
        "trials": trials,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("fm2", type=Path, nargs="?", default=WARPLESS_FM2)
    p.add_argument("--report", type=Path, default=None)
    p.add_argument("--max-frames", type=int, default=12_000)
    p.add_argument("--skip-movie", type=int, default=0)
    p.add_argument("--pad-before", type=int, default=0)
    p.add_argument(
        "--isolated-1-1",
        action="store_true",
        help="Level1_1 grid-search FM2 starts (working extract path)",
    )
    p.add_argument("--1-1-start-min", type=int, default=190, dest="start_min")
    p.add_argument("--1-1-start-max", type=int, default=230, dest="start_max")
    p.add_argument("--export-1-1", action="store_true")
    p.add_argument("--1-1-start", type=int, default=None, dest="export_start")
    p.add_argument("--1-1-body", type=int, default=None, dest="export_body")
    p.add_argument(
        "--search-1-2-flag",
        action="store_true",
        help="after warpless 1-1, grid-search FM2 starts for 1-3 control",
    )
    p.add_argument("--export-1-2-flag", action="store_true")
    p.add_argument("--1-2-start-min", type=int, default=2080, dest="flag_min")
    p.add_argument("--1-2-start-max", type=int, default=2140, dest="flag_max")
    p.add_argument("--1-2-step", type=int, default=1, dest="flag_step")
    p.add_argument(
        "--search-1-3",
        action="store_true",
        help="after warpless 1-2 flag, grid-search FM2 starts for 1-4 control",
    )
    p.add_argument("--export-1-3", action="store_true")
    p.add_argument("--1-3-start-min", type=int, default=None, dest="s13_min")
    p.add_argument("--1-3-start-max", type=int, default=None, dest="s13_max")
    p.add_argument("--1-3-window", type=int, default=80, dest="s13_window")
    p.add_argument("--1-3-step", type=int, default=1, dest="s13_step")
    p.add_argument(
        "--verify-1-3",
        action="store_true",
        help="replay warpless 1-3 body from TAS 1-2 flag leave",
    )
    p.add_argument(
        "--isolated-1-3",
        action="store_true",
        help="Level1_3 settle grid + TAS 1-3 body (pin phase may miss)",
    )
    args = p.parse_args(argv)

    if not args.fm2.exists():
        print(f"missing {args.fm2}; run: uv run python -m smb.tas.fetch_refs", file=sys.stderr)
        return 1

    movie = parse_movie(args.fm2)
    WARPLESS_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    meta = {
        "movie": movie.summary(),
        "warpless": summary_dict(),
    }

    if args.isolated_1_1 or args.export_1_1:
        search = search_isolated_1_1(
            movie.frames,
            start_min=args.start_min,
            start_max=args.start_max,
        )
        meta["isolated_1_1"] = search
        print(json.dumps({k: search[k] for k in search if k != "clears"}, indent=2))
        best = search.get("best") or {}
        if args.export_1_1 and best.get("leave_frame"):
            si = args.export_start if args.export_start is not None else int(best["start_idx"])
            n = args.export_body if args.export_body is not None else int(best["leave_frame"])
            payload = export_1_1_slice(
                movie.frames, start_idx=si, body_frames=n, fm2_path=args.fm2
            )
            meta["exported_1_1"] = {k: payload[k] for k in payload if k != "segments"}
            print(f"wrote {payload.get('_path')} frames={payload['num_frames']}", file=sys.stderr)
        rp = args.report or (WARPLESS_REPORT_DIR / "isolated_1_1.json")
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {rp}", file=sys.stderr)
        return 0 if (search.get("best") or {}).get("leave_frame") else 1

    if args.search_1_2_flag or args.export_1_2_flag:
        search = search_1_2_flag(
            movie.frames,
            start_min=args.flag_min,
            start_max=args.flag_max,
            step=max(1, args.flag_step),
        )
        meta["1_2_flag"] = search
        print(json.dumps({k: search[k] for k in search if k != "clears"}, indent=2))
        best = search.get("best") or {}
        if args.export_1_2_flag and best.get("leave_frame"):
            payload = export_1_2_flag_slice(
                movie.frames,
                start_idx=int(best["start_idx"]),
                body_frames=int(best["leave_frame"]),
                fm2_path=args.fm2,
            )
            meta["exported_1_2_flag"] = {
                k: payload[k] for k in payload if k != "segments"
            }
            print(f"wrote {payload.get('_path')} frames={payload['num_frames']}", file=sys.stderr)
        rp = args.report or (WARPLESS_REPORT_DIR / "1_2_flag.json")
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {rp}", file=sys.stderr)
        return 0 if best.get("leave_frame") else 1

    if args.verify_1_3:
        report = verify_1_3_slice(movie.frames)
        meta["verify_1_3"] = report
        print(json.dumps(report, indent=2))
        rp = args.report or (WARPLESS_REPORT_DIR / "1_3_verify.json")
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {rp}", file=sys.stderr)
        return 0 if report.get("ok") else 1

    if args.isolated_1_3:
        report = isolated_1_3_settle(movie.frames)
        meta["isolated_1_3"] = report
        print(json.dumps({k: report[k] for k in report if k != "trials"}, indent=2))
        rp = args.report or (WARPLESS_REPORT_DIR / "isolated_1_3.json")
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {rp}", file=sys.stderr)
        return 0 if report.get("ok") else 1

    if args.search_1_3 or args.export_1_3:
        search = search_1_3(
            movie.frames,
            start_min=args.s13_min,
            start_max=args.s13_max,
            window=max(0, args.s13_window),
            step=max(1, args.s13_step),
        )
        meta["1_3"] = search
        print(json.dumps({k: search[k] for k in search if k != "clears"}, indent=2))
        best = search.get("best") or {}
        if args.export_1_3 and best.get("leave_frame"):
            payload = export_1_3_slice(
                movie.frames,
                start_idx=int(best["start_idx"]),
                body_frames=int(best["leave_frame"]),
                fm2_path=args.fm2,
            )
            meta["exported_1_3"] = {
                k: payload[k] for k in payload if k != "segments"
            }
            print(f"wrote {payload.get('_path')} frames={payload['num_frames']}", file=sys.stderr)
        rp = args.report or (WARPLESS_REPORT_DIR / "1_3.json")
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {rp}", file=sys.stderr)
        return 0 if best.get("leave_frame") else 1

    report = annotate_poweron(
        movie.frames,
        max_frames=args.max_frames,
        pad_before=args.pad_before,
        skip_movie=args.skip_movie,
    )
    meta["poweron"] = report
    print(json.dumps(report, indent=2))
    rp = args.report or WARPLESS_EXITS_REPORT
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {rp}", file=sys.stderr)
    return 0 if report.get("n_stages", 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
