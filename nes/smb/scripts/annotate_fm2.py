"""Replay an SMB TAS movie and mark 32-exit (world, dash_level) boundaries.

Power-on fceumm often desyncs FCEUX movies (same as any% warps). Isolated
``Level1_1`` search is the working extract path: find an FM2 start that
clears 1-1, then export the body. Later stages share one search/export/verify
path (``smb.tas.warpless_extract``) driven by ``WARPLESS_LEGS``.

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
  uv run python -m smb.scripts.annotate_fm2 --isolated-1-1 --export-1-1
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
  uv run python -m smb.scripts.annotate_fm2 --search-1-4 --export-1-4
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
  uv run python -m smb.scripts.annotate_fm2 --verify-1-4
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
  uv run python -m smb.scripts.annotate_fm2 --search 2-1 --export
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
from smb.tas.replay import IDLE, make_level1_env, to_action9
from smb.tas.stages import is_1_3_control, is_1_4_control
from smb.tas.warpless import (
    WARPLESS_EXITS_REPORT,
    WARPLESS_FM2,
    WARPLESS_REPORT_DIR,
    WL_1_3_FM2_START,
    WL_1_3_LEAVE_FRAMES,
    get_leg,
    summary_dict,
)
from smb.tas.warpless_extract import (
    export_warpless_slice,
    search_warpless,
    verify_warpless_slice,
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


def export_1_3_slice(
    frames: list[list[int]],
    *,
    start_idx: int,
    body_frames: int,
    fm2_path: Path,
    out_path: Path | None = None,
) -> dict[str, Any]:
    """Back-compat wrapper used by tests; prefer ``export_warpless_slice``."""
    return export_warpless_slice(
        frames,
        stage_id="1-3",
        start_idx=start_idx,
        body_frames=body_frames,
        fm2_path=fm2_path,
        out_path=out_path,
    )


def verify_1_3_slice(
    frames: list[list[int]],
    *,
    start_idx: int = WL_1_3_FM2_START,
    body_frames: int = WL_1_3_LEAVE_FRAMES,
) -> dict[str, Any]:
    """Replay the 1-3 body from TAS 1-2 flag leave; success = 1-4 control."""
    return verify_warpless_slice(
        frames, "1-3", start_idx=start_idx, body_frames=body_frames
    )


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


def _stage_key(stage_id: str) -> str:
    return stage_id.strip().lower().replace("_", "-")


def _resolve_stage_job(args: argparse.Namespace) -> tuple[
    str | None, bool, bool, bool, dict[str, int | None]
]:
    """Map CLI aliases onto one (stage, search, export, verify, grid) job."""
    stage: str | None = None
    do_search = False
    do_export = False
    do_verify = False
    grid_min = args.grid_min
    grid_max = args.grid_max
    window = args.window
    step = args.step
    lead_max = args.lead_max

    if args.search_1_2_flag or args.export_1_2_flag:
        stage = "1-2"
        do_search = True
        do_export = bool(args.export_1_2_flag)
        grid_min = args.flag_min
        grid_max = args.flag_max
        step = args.flag_step
    if args.search_1_3 or args.export_1_3 or args.verify_1_3:
        stage = "1-3"
        do_search = bool(args.search_1_3 or args.export_1_3)
        do_export = bool(args.export_1_3)
        do_verify = bool(args.verify_1_3)
        if args.s13_min is not None:
            grid_min = args.s13_min
        if args.s13_max is not None:
            grid_max = args.s13_max
        if args.s13_window is not None:
            window = args.s13_window
        if args.s13_step is not None:
            step = args.s13_step
    if args.search_1_4 or args.export_1_4 or args.verify_1_4:
        stage = "1-4"
        do_search = bool(args.search_1_4 or args.export_1_4)
        do_export = bool(args.export_1_4)
        do_verify = bool(args.verify_1_4)
    if args.search_stage:
        stage = _stage_key(args.search_stage)
        do_search = True
    if args.do_export_flag:
        do_export = True
        if stage is not None:
            do_search = True
    if args.verify_stage:
        stage = _stage_key(args.verify_stage)
        do_verify = True
        if not args.search_stage:
            do_search = False

    grid = {
        "start_min": grid_min,
        "start_max": grid_max,
        "window": window,
        "step": max(1, int(step)),
        "lead_max": max(0, int(lead_max)),
        "from_pred": int(bool(args.from_pred)),
    }
    return stage, do_search, do_export, do_verify, grid


def _run_stage_job(
    frames: list[list[int]],
    fm2_path: Path,
    meta: dict[str, Any],
    *,
    stage: str,
    do_search: bool,
    do_export: bool,
    do_verify: bool,
    grid: dict[str, int | None],
    report_path: Path | None,
) -> int:
    try:
        get_leg(stage)
    except KeyError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    search: dict[str, Any] | None = None
    best: dict[str, Any] = {}
    if do_search:
        search = search_warpless(
            frames,
            stage,
            start_min=grid["start_min"],
            start_max=grid["start_max"],
            window=int(grid["window"] or 80),
            step=int(grid["step"] or 1),
            lead_max=int(grid.get("lead_max") or 0),
            from_pred=bool(grid.get("from_pred")),
        )
        meta[stage.replace("-", "_")] = search
        print(json.dumps({k: search[k] for k in search if k != "clears"}, indent=2))
        best = search.get("best") or {}

    if do_export:
        if not best.get("leave_frame"):
            print(f"ERROR: {stage} search found no leave; not exporting", file=sys.stderr)
            rp = report_path or (WARPLESS_REPORT_DIR / f"{stage.replace('-', '_')}.json")
            rp.parent.mkdir(parents=True, exist_ok=True)
            rp.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
            print(f"wrote {rp}", file=sys.stderr)
            return 1
        payload = export_warpless_slice(
            frames,
            stage_id=stage,
            start_idx=int(best["start_idx"]),
            body_frames=int(best["leave_frame"]),
            fm2_path=fm2_path,
            lead_idle=int(best.get("lead_idle") or 0),
        )
        meta[f"exported_{stage.replace('-', '_')}"] = {
            k: payload[k] for k in payload if k != "segments"
        }
        print(f"wrote {payload.get('_path')} frames={payload['num_frames']}", file=sys.stderr)

    if do_verify:
        si = int(best["start_idx"]) if best.get("start_idx") is not None else None
        n = int(best["leave_frame"]) if best.get("leave_frame") is not None else None
        report = verify_warpless_slice(
            frames,
            stage,
            start_idx=si,
            body_frames=n,
            lead_idle=int(best["lead_idle"]) if best.get("lead_idle") is not None else None,
        )
        meta[f"verify_{stage.replace('-', '_')}"] = report
        print(json.dumps(report, indent=2))
        rp = report_path or (
            WARPLESS_REPORT_DIR / f"{stage.replace('-', '_')}_verify.json"
        )
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {rp}", file=sys.stderr)
        return 0 if report.get("ok") else 1

    rp = report_path or (WARPLESS_REPORT_DIR / f"{stage.replace('-', '_')}.json")
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {rp}", file=sys.stderr)
    if do_search:
        return 0 if best.get("leave_frame") else 1
    return 0


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
        "--search",
        metavar="STAGE",
        dest="search_stage",
        default=None,
        help="grid-search FM2 starts for this 32-exit stage (1-2…8-4)",
    )
    p.add_argument(
        "--export",
        action="store_true",
        dest="do_export_flag",
        help="export the searched stage body (pair with --search STAGE)",
    )
    p.add_argument(
        "--verify",
        metavar="STAGE",
        dest="verify_stage",
        default=None,
        help="replay exported body from the chained predecessor",
    )
    p.add_argument("--window", type=int, default=80, help="FM2 start grid ±window")
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--start-min", type=int, default=None, dest="grid_min")
    p.add_argument("--start-max", type=int, default=None, dest="grid_max")
    p.add_argument(
        "--lead-max",
        type=int,
        default=0,
        help="extra idle frames after control, searched 0..N",
    )
    p.add_argument(
        "--from-pred",
        action="store_true",
        help="pin at predecessor leave (drop-in) instead of idling to control",
    )
    p.add_argument(
        "--search-1-2-flag",
        action="store_true",
        help="alias: --search 1-2 (flag pipe, not W4)",
    )
    p.add_argument("--export-1-2-flag", action="store_true")
    p.add_argument("--1-2-start-min", type=int, default=2080, dest="flag_min")
    p.add_argument("--1-2-start-max", type=int, default=2140, dest="flag_max")
    p.add_argument("--1-2-step", type=int, default=1, dest="flag_step")
    p.add_argument("--search-1-3", action="store_true")
    p.add_argument("--export-1-3", action="store_true")
    p.add_argument("--verify-1-3", action="store_true")
    p.add_argument("--1-3-start-min", type=int, default=None, dest="s13_min")
    p.add_argument("--1-3-start-max", type=int, default=None, dest="s13_max")
    p.add_argument("--1-3-window", type=int, default=None, dest="s13_window")
    p.add_argument("--1-3-step", type=int, default=None, dest="s13_step")
    p.add_argument("--search-1-4", action="store_true")
    p.add_argument("--export-1-4", action="store_true")
    p.add_argument("--verify-1-4", action="store_true")
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

    if args.isolated_1_3:
        report = isolated_1_3_settle(movie.frames)
        meta["isolated_1_3"] = report
        print(json.dumps({k: report[k] for k in report if k != "trials"}, indent=2))
        rp = args.report or (WARPLESS_REPORT_DIR / "isolated_1_3.json")
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {rp}", file=sys.stderr)
        return 0 if report.get("ok") else 1

    stage, do_search, do_export, do_verify, grid = _resolve_stage_job(args)
    if stage is not None:
        return _run_stage_job(
            movie.frames,
            args.fm2,
            meta,
            stage=stage,
            do_search=do_search,
            do_export=do_export,
            do_verify=do_verify,
            grid=grid,
            report_path=args.report,
        )

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
