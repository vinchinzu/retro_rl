"""Generic #3728M search / export / verify (control-relative, one StageSpec row).

Do not clone per-stage search_1_3 / export_1_3 / verify_1_3. CLI lives in
``smb.scripts.annotate_fm2``. Play path is ``smb.tas.warpless.play_warpless_to``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from smb.ram import PLAYER_STATE_DYING, read_snapshot
from smb.tas.fm2 import frames_to_nes9_rle_payload
from smb.tas.replay import get_state, idle_until, make_level1_env, set_state, to_action9
from smb.tas.warpless import (
    WL_1_1_SETTLE,
    WARPLESS_PUBLICATION_ID,
    _snap_brief,
    fm2_hint,
    get_leg,
    play_warpless_to,
    predecessor_leg,
)


def _is_warped(snap: Any, *, world: int, leave_world: int | None) -> bool:
    w = int(snap.world)
    leave = world if leave_world is None else leave_world
    if w == 3 and world < 3 and leave < 3:
        return True
    if w == 7 and world < 7 and leave < 7:
        return True
    return False


def reach_warpless_control(
    stage_id: str,
    *,
    settle: int = WL_1_1_SETTLE,
    max_wait: int = 400,
) -> dict[str, Any]:
    """Level1_1 + exported predecessor bodies + idle to this stage's control.

    Returns an open env when ``ok``; caller must close it.
    """
    from smb.tas.replay import IDLE

    leg = get_leg(stage_id)
    pred = predecessor_leg(stage_id)
    env = make_level1_env()
    if pred is None:
        for _ in range(settle):
            env.step(IDLE)
        snap = read_snapshot(env.get_ram(), frame=0)
        return {
            "ok": True,
            "env": env,
            "ctrl_wait": 0,
            "snap": _snap_brief(snap),
            "lives": int(snap.lives),
            "stage": leg.id,
        }

    report = play_warpless_to(env, to=pred.id, settle=settle, max_wait=max_wait)
    if not report.get("ok"):
        env.close()
        return {
            "ok": False,
            "stage": pred.id,
            "control": report,
            "snap": report.get("end_snap"),
        }

    wait, snap = idle_until(env, leg.control, max_wait=max_wait)
    ok = leg.control(snap)
    if not ok:
        env.close()
        return {
            "ok": False,
            "stage": leg.id,
            "ctrl_wait": wait,
            "snap": _snap_brief(snap),
            "pred": {k: report[k] for k in report if k != "stages"},
        }
    return {
        "ok": True,
        "env": env,
        "ctrl_wait": wait,
        "snap": _snap_brief(snap),
        "lives": int(snap.lives),
        "stage": leg.id,
        "pred_outcome": report.get("outcome"),
        "pred_frame": report.get("frame"),
    }


def search_warpless(
    frames: list[list[int]],
    stage_id: str,
    *,
    start_min: int | None = None,
    start_max: int | None = None,
    window: int | None = None,
    step: int = 1,
    max_play: int | None = None,
) -> dict[str, Any]:
    """Grid-search FM2 starts from this stage's control until leave / death."""
    leg = get_leg(stage_id)
    reached = reach_warpless_control(stage_id)
    if not reached.get("ok"):
        return {
            "mode": stage_id,
            "control": reached,
            "best": None,
            "n_clear": 0,
        }
    env = reached.pop("env")
    wait = int(reached["ctrl_wait"])
    center = fm2_hint(stage_id) + wait
    span = leg.search_window if window is None else window
    lo = start_min if start_min is not None else max(0, center - span)
    hi = start_max if start_max is not None else center + span
    cap = leg.max_play if max_play is None else max_play
    pin = get_state(env)
    start_lives = int(reached["lives"])
    trials: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    tag = stage_id.replace("-", "")
    print(
        f"  {tag} control wait={wait} center={center} "
        f"search {lo}..{hi} step={step} {reached['snap']}",
        flush=True,
    )
    try:
        for si in range(lo, hi + 1, max(1, step)):
            set_state(env, pin)
            max_x = 0
            death: int | None = None
            leave: int | None = None
            warped = False
            body = frames[si:]
            for i in range(min(len(body), cap)):
                env.step(to_action9(body[i]))
                now = read_snapshot(env.get_ram(), frame=i + 1)
                px = int(now.player_x)
                if 0 < px < 20_000:
                    max_x = max(max_x, px)
                if int(now.lives) < start_lives or int(now.player_state) == PLAYER_STATE_DYING:
                    death = i + 1
                    break
                if _is_warped(now, world=leg.world, leave_world=leg.leave_world):
                    warped = True
                    break
                if leg.leave(now):
                    leave = i + 1
                    break
            tr = {
                "start_idx": si,
                "max_x": max_x,
                "death": death,
                "leave_frame": leave,
                "leave_to": leg.leave_id if leave is not None else None,
                "warped": warped,
            }
            trials.append(tr)
            if _better_trial(tr, best, center):
                best = {**tr, "_score": _trial_score(tr, center)}
            if leave is not None or warped or (max_x or 0) > 400 or si in (lo, hi):
                print(
                    f"  {tag} si={si} max_x={max_x} leave={leave} death={death}"
                    + (f" warped={warped}" if warped else ""),
                    flush=True,
                )
    finally:
        env.close()
    clears = [t for t in trials if t["leave_frame"] is not None]
    return {
        "mode": stage_id,
        "control": reached,
        "center": center,
        "ctrl_wait": wait,
        "start_min": lo,
        "start_max": hi,
        "step": step,
        "max_play": cap,
        "best": best,
        "n_clear": len(clears),
        "clears": sorted(clears, key=lambda t: int(t["leave_frame"]))[:12],
        "n_trials": len(trials),
    }


def _trial_score(tr: dict[str, Any], center: int) -> tuple[int, int, int, int]:
    """Rank: clear, shorter leave, closer to center, higher max_x."""
    leave = tr.get("leave_frame")
    clear = 1 if leave is not None and not tr.get("warped") else 0
    leave_rank = -int(leave or 10**9)
    dist = -abs(int(tr["start_idx"]) - center)
    return (clear, leave_rank, dist, int(tr.get("max_x") or 0))


def _better_trial(tr: dict[str, Any], best: dict[str, Any] | None, center: int) -> bool:
    if best is None:
        return True
    return _trial_score(tr, center) > tuple(best.get("_score") or (0, 0, 0, 0))


def export_warpless_slice(
    frames: list[list[int]],
    *,
    stage_id: str,
    start_idx: int,
    body_frames: int,
    fm2_path: Path,
    out_path: Path | None = None,
) -> dict[str, Any]:
    """Write a #3728M body seed (no L+R sanitize)."""
    leg = get_leg(stage_id)
    dest = out_path or leg.seed_path
    body = [list(f) for f in frames[start_idx : start_idx + body_frames]]
    payload = frames_to_nes9_rle_payload(
        body,
        route_id=leg.level_id,
        source=f"HappyLee & Mars608 warpless #{WARPLESS_PUBLICATION_ID} FM2 @{start_idx}",
        extra={
            "level_id": leg.level_id,
            "start_state": leg.start_state,
            "settle_frames": 2 if leg.id == "1-1" else 0,
            "game_name": "SuperMarioBros-Nes-v0",
            "target": leg.target_name,
            "body_frames": len(body),
            "leave_frames": len(body),
            "fm2": str(fm2_path),
            "fm2_start_index": start_idx,
            "route_id": "smb_all_exits",
            "stage_id": leg.id,
            "note": leg.note,
        },
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    payload["_path"] = str(dest)
    return payload


def verify_warpless_slice(
    frames: list[list[int]],
    stage_id: str,
    *,
    start_idx: int | None = None,
    body_frames: int | None = None,
) -> dict[str, Any]:
    """Replay the body from the chained predecessor leave. Success = leave gate."""
    from smb.policy import load_nes9_rle_seed

    leg = get_leg(stage_id)
    si = leg.fm2_start if start_idx is None else start_idx
    n = leg.body_frames if body_frames is None else body_frames
    if (si <= 0 or n <= 0) and leg.seed_path.is_file():
        data = load_nes9_rle_seed(leg.seed_path)
        if si <= 0:
            si = int(data.get("fm2_start_index") or 0)
        if n <= 0:
            n = int(data.get("num_frames") or 0)
    if si <= 0 or n <= 0:
        return {
            "ok": False,
            "mode": f"verify_{stage_id}",
            "reason": "missing fm2 window (pass start_idx/body_frames or extract first)",
        }
    reached = reach_warpless_control(stage_id)
    if not reached.get("ok"):
        return {"ok": False, "stage": stage_id, "control": reached}
    env = reached.pop("env")
    start_lives = int(reached["lives"])
    leave: int | None = None
    death: int | None = None
    max_x = 0
    snap = None
    try:
        body = frames[si : si + n]
        for i, fr in enumerate(body):
            env.step(to_action9(fr))
            snap = read_snapshot(env.get_ram(), frame=i + 1)
            px = int(snap.player_x)
            if 0 < px < 20_000:
                max_x = max(max_x, px)
            if int(snap.lives) < start_lives or int(snap.player_state) == PLAYER_STATE_DYING:
                death = i + 1
                break
            if leg.leave(snap):
                leave = i + 1
                break
    finally:
        env.close()
    return {
        "ok": leave is not None and death is None,
        "mode": f"verify_{stage_id}",
        "control": reached,
        "start_idx": si,
        "body_frames": n,
        "leave_frame": leave,
        "leave_to": leg.leave_id,
        "death": death,
        "max_x": max_x,
        "end_snap": _snap_brief(snap) if snap is not None else None,
    }
