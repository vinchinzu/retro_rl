"""Stitchless 8-3 probe: multi-leave 8-2 → re-gate 8-3 + continuous tails.

No natural_82 bridge. Pure HL/FX FM2 bodies after HL W8 predecessor.

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
  uv run python -u -m smb.scripts.probe_stitchless_8_3
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

import numpy as np

from smb.paths import GAME_DIR, GAME_V0, MODELS_DIR, RECORDINGS_DIR
from smb.policy import compress_nes9_rle
from smb.ram import PLAYER_STATE_DYING, reached_ending, read_snapshot
from smb.tas.fm2 import parse_fm2
from smb.tas.slice import (
    DEFAULT_FM2,
    HL_8_1_FM2_START,
    HL_8_2_FM2_START,
    is_8_2_control,
    is_8_3_control,
    is_8_4_control,
    probe_8_1_from_control,
    probe_8_2_from_control,
    probe_8_3_from_control,
    reach_8_1_control_after_hl_w8,
)

FX_FM2 = Path("nes/smb/tas/ref/flamexx_warps_rta_4_54_099.fm2")
IDLE = np.zeros(9, dtype=np.int8)
OUT_DIR = RECORDINGS_DIR / "tas_import"


def log(*a: object) -> None:
    print(*a, flush=True)


def _act(fr) -> np.ndarray:
    a = np.zeros(9, dtype=np.int8)
    for i, v in enumerate(fr[:9]):
        a[i] = int(v)
    return a


def _em(env):
    return env.em if hasattr(env, "em") else env.unwrapped.em


def _fp(snap) -> dict[str, int]:
    return {
        "world": int(snap.world),
        "level": int(snap.level),
        "x": int(snap.player_x),
        "y": int(snap.player_y),
        "t": int(snap.timer),
        "ps": int(snap.player_state),
        "lives": int(snap.lives),
    }


def wait_control(env, predicate, max_wait: int = 600) -> tuple[int, Any]:
    wait = 0
    snap = read_snapshot(env.get_ram(), 0)
    while wait < max_wait:
        snap = read_snapshot(env.get_ram(), 0)
        if predicate(snap):
            return wait, snap
        env.step(IDLE)
        wait += 1
    return wait, snap


def play_until(
    env,
    body,
    lives: int,
    *,
    max_play: int | None = None,
    stop_on_8_3_control: bool = False,
) -> dict[str, Any]:
    n = len(body) if max_play is None else min(len(body), max_play)
    snap0 = read_snapshot(env.get_ram(), 0)
    last = (int(snap0.world), int(snap0.level))
    exits: list[dict[str, Any]] = []
    max_x = 0
    max_x_after_ctrl = 0
    death = ending = None
    enter83 = enter84 = ctrl83_at = None
    for i in range(n):
        env.step(_act(body[i]))
        ram = env.get_ram()
        snap = read_snapshot(ram, i + 1)
        px = int(snap.player_x)
        w, l = int(snap.world), int(snap.level)
        if 0 < px < 20_000:
            max_x = max(max_x, px)
            if ctrl83_at is not None and w == 7 and l == 2 and px < 4000:
                max_x_after_ctrl = max(max_x_after_ctrl, px)
        key = (w, l)
        if key != last:
            exits.append(
                {
                    "i": i + 1,
                    "from": list(last),
                    "to": list(key),
                    "x": px,
                    "t": int(snap.timer),
                }
            )
            last = key
            if key == (7, 2) and enter83 is None:
                enter83 = i + 1
            if key == (7, 3) and enter84 is None:
                enter84 = i + 1
        if is_8_3_control(snap) and ctrl83_at is None:
            ctrl83_at = i + 1
            if stop_on_8_3_control:
                return {
                    "ok_ctrl83": True,
                    "ctrl83_at": ctrl83_at,
                    "enter83": enter83,
                    "fp": _fp(snap),
                    "exits": exits,
                    "played": i + 1,
                }
        if is_8_4_control(snap) and enter84 is None:
            enter84 = i + 1
        if reached_ending(ram, start_lives=lives):
            ending = i + 1
            break
        if int(snap.lives) < lives or int(snap.player_state) == PLAYER_STATE_DYING:
            death = {
                "i": i + 1,
                "w": w + 1,
                "l": l + 1,
                "x": px,
                "t": int(snap.timer),
                "ps": int(snap.player_state),
            }
            break
    return {
        "ending": ending,
        "death": death,
        "exits": exits,
        "max_x": max_x,
        "max_x_after_ctrl": max_x_after_ctrl,
        "enter83": enter83,
        "enter84": enter84,
        "ctrl83_at": ctrl83_at,
        "played": ending or (death or {}).get("i") or n,
    }


def export_body(
    path: Path,
    frames: list[list[int]],
    *,
    route_id: str,
    start_state: str,
    target: str,
    meta: dict[str, Any],
) -> Path:
    payload = {
        "format": "nes9_rle",
        "route_id": route_id,
        "game_name": "SuperMarioBros-Nes-v0",
        "num_frames": len(frames),
        "body_frames": len(frames),
        "start_state": start_state,
        "target": target,
        "stitchless": True,
        "source": "HappyLee/flamexx FM2 (no natural_82 bridge)",
        **meta,
        "segments": compress_nes9_rle(frames),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--8-2-start-min", dest="s82_min", type=int, default=10840)
    p.add_argument("--8-2-start-max", dest="s82_max", type=int, default=10940)
    p.add_argument("--8-2-step", dest="s82_step", type=int, default=2)
    p.add_argument("--8-3-start-min", dest="s83_min", type=int, default=13000)
    p.add_argument("--8-3-start-max", dest="s83_max", type=int, default=13500)
    p.add_argument("--8-3-step", dest="s83_step", type=int, default=5)
    p.add_argument(
        "--leads",
        type=str,
        default="0,1,2,5,10,14,21",
        help="comma list of lead idles after 8-3 control",
    )
    p.add_argument("--top-leaves", type=int, default=5, help="how many 8-2 leave variants to fan into 8-3")
    p.add_argument("--export-on-hit", action="store_true", default=True)
    p.add_argument("--refine", action="store_true", help="dense refine around best progress")
    args = p.parse_args(argv)

    t0 = time.time()
    from retro_harness.env import make_env

    hl = parse_fm2(DEFAULT_FM2).frames
    fx = parse_fm2(FX_FM2).frames if FX_FM2.exists() else None
    env = make_env(GAME_V0, "Level1_1", GAME_DIR, render_mode="rgb_array")
    env.reset()
    log("build HL → 8-1 control…")
    pred = reach_8_1_control_after_hl_w8(env)
    if not pred.get("success"):
        log("FAIL 8-1 control", pred)
        env.close()
        return 1
    lives81 = int(pred["control_snap"].lives)
    st81 = _em(env).get_state()
    log(f"8-1 control ok wait81={pred.get('ctrl_wait_8_1')} t={time.time()-t0:.1f}s")

    # 8-1 leave + 8-2 control
    _em(env).set_state(st81)
    tr81 = probe_8_1_from_control(env, hl, HL_8_1_FM2_START, start_lives=lives81)
    wait82, snap82 = wait_control(env, is_8_2_control)
    lives82 = int(snap82.lives)
    st82 = _em(env).get_state()
    log(f"8-1 leave={tr81.w4} wait82={wait82} fp={_fp(snap82)}")

    report: dict[str, Any] = {
        "note": "stitchless multi-leave 8-2 → re-gate 8-3; no natural_82",
        "pred_wait81": pred.get("ctrl_wait_8_1"),
        "leave81": tr81.w4,
        "wait82": wait82,
        "leaves_82": [],
        "fanout_83": [],
        "hits": [],
        "best": None,
    }

    # --- Phase 1: find 8-2 leaves (distinct leave frames) ---
    leave_hits: list[dict[str, Any]] = []
    for si in range(args.s82_min, args.s82_max + 1, args.s82_step):
        _em(env).set_state(st82)
        tr = probe_8_2_from_control(env, hl, si, start_lives=lives82)
        if tr.w4 is None or tr.death is not None:
            continue
        wait83, snap83 = wait_control(env, is_8_3_control)
        if not is_8_3_control(snap83):
            continue
        row = {
            "si82": si,
            "leave82": tr.w4,
            "wait83": wait83,
            "ctrl83_fp": _fp(snap83),
        }
        leave_hits.append(row)
        log(f"LEAVE82 si={si} leave={tr.w4} wait83={wait83} t={row['ctrl83_fp']['t']} x={row['ctrl83_fp']['x']}")

    # unique by (leave82, timer at control)
    seen: set[tuple[int, int]] = set()
    unique: list[dict[str, Any]] = []
    for row in sorted(leave_hits, key=lambda r: (r["leave82"], r["si82"])):
        key = (row["leave82"], row["ctrl83_fp"]["t"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    report["leaves_82"] = leave_hits
    report["unique_leave_classes"] = unique
    log(f"leave hits={len(leave_hits)} unique classes={len(unique)}")

    # Prefer fastest leaves first, then a few slower for phase diversity
    unique_sorted = sorted(unique, key=lambda r: r["leave82"])
    fan = unique_sorted[: max(1, args.top_leaves)]
    for row in reversed(unique_sorted):
        if row not in fan and len(fan) < args.top_leaves + 2:
            fan.append(row)

    leads = [int(x) for x in args.leads.split(",") if x.strip() != ""]
    best: dict[str, Any] | None = None
    found_leave = False

    def consider(row: dict[str, Any]) -> None:
        nonlocal best, found_leave
        report["fanout_83"].append(row)
        score = (
            1 if row.get("leave83") else 0,
            1 if row.get("ending") else 0,
            row.get("max_x") or 0,
            -(row.get("leave83") or 10**9),
        )
        row["_score"] = score
        if best is None or score > best["_score"]:
            best = row
            log(
                f"BEST so far leave83={row.get('leave83')} end={row.get('ending')} "
                f"max_x={row.get('max_x')} si82={row.get('si82')} si83={row.get('si83')} lead={row.get('lead')}"
            )
        if row.get("leave83"):
            found_leave = True

    def search_83(
        *,
        st83,
        lives83: int,
        si82: int,
        leave82: int,
        wait83: int,
        fp83: dict[str, int],
        movie,
        src: str,
        smin: int,
        smax: int,
        step: int,
        lead_list: list[int],
        max_play: int = 2800,
    ) -> None:
        nonlocal found_leave
        for lead in lead_list:
            if found_leave:
                return
            for si83 in range(smin, smax + 1, step):
                if found_leave:
                    return
                _em(env).set_state(st83)
                for _ in range(lead):
                    env.step(IDLE)
                tr = probe_8_3_from_control(
                    env, movie, si83, start_lives=lives83, max_play=max_play
                )
                row = {
                    "src": src,
                    "si82": si82,
                    "leave82": leave82,
                    "wait83": wait83,
                    "ctrl83_fp": fp83,
                    "lead": lead,
                    "si83": si83,
                    "leave83": tr.w4,
                    "death": tr.death,
                    "max_x": tr.max_x,
                }
                if tr.w4 is not None and tr.death is None:
                    report["hits"].append(row)
                    consider(row)
                    if args.export_on_hit:
                        body = [list(f) for f in movie[si83 : si83 + int(tr.w4)]]
                        name = (
                            "smb_8_3_happylee_slice.json"
                            if src.startswith("HL")
                            else "smb_8_3_flamexx_slice.json"
                        )
                        export_body(
                            MODELS_DIR / name,
                            body,
                            route_id=name.replace(".json", ""),
                            start_state="8-3_control_after_hl_8_2_stitchless",
                            target="8_4_load",
                            meta={
                                "fm2_start_index": si83,
                                "leave_frames": tr.w4,
                                "si82": si82,
                                "leave82": leave82,
                                "lead_idle": lead,
                                "ctrl_fp": fp83,
                                "src": src,
                            },
                        )
                        log(f"EXPORTED {name} leave83={tr.w4}")
                elif (tr.max_x or 0) >= 700:
                    consider(row)

    # --- Phase 2: for each leave class, re-gate 8-3 + offset search ---
    for leave in fan:
        if found_leave:
            break
        si82 = leave["si82"]
        log(f"=== fanout si82={si82} leave82={leave['leave82']} wait83={leave['wait83']} ===")
        _em(env).set_state(st82)
        tr82 = probe_8_2_from_control(env, hl, si82, start_lives=lives82)
        if tr82.w4 is None:
            continue
        wait83, snap83 = wait_control(env, is_8_3_control)
        if not is_8_3_control(snap83):
            log("  no 8-3 control")
            continue
        lives83 = int(snap83.lives)
        st83 = _em(env).get_state()
        fp83 = _fp(snap83)
        log(f"  ctrl83 fp={fp83}")

        search_83(
            st83=st83,
            lives83=lives83,
            si82=si82,
            leave82=int(tr82.w4),
            wait83=wait83,
            fp83=fp83,
            movie=hl,
            src="HL",
            smin=args.s83_min,
            smax=args.s83_max,
            step=args.s83_step,
            lead_list=leads,
        )

        if not found_leave and fx is not None:
            search_83(
                st83=st83,
                lives83=lives83,
                si82=si82,
                leave82=int(tr82.w4),
                wait83=wait83,
                fp83=fp83,
                movie=fx,
                src="FX",
                smin=13000,
                smax=15200,
                step=10,
                lead_list=[0, 1, 2, 5, 10, 21],
                max_play=3000,
            )

        # Continuous stitchless from this 8-2 start (no re-gate mid body)
        _em(env).set_state(st82)
        cont = play_until(env, hl[si82:], lives82, max_play=7000)
        consider(
            {
                "src": "HL_cont",
                "si82": si82,
                "leave82": tr82.w4,
                "enter83": cont.get("enter83"),
                "enter84": cont.get("enter84"),
                "ending": cont.get("ending"),
                "death": cont.get("death"),
                "leave83": cont.get("enter84"),
                "max_x": cont.get("max_x_after_ctrl") or 0,
                "si83": None,
                "lead": None,
                "ctrl83_at": cont.get("ctrl83_at"),
            }
        )
        log(
            f"  CONT si82={si82} enter83={cont.get('enter83')} e84={cont.get('enter84')} "
            f"max_after_ctrl={cont.get('max_x_after_ctrl')} death={cont.get('death')} end={cont.get('ending')}"
        )

    # Optional dense refine around best progress
    if args.refine and best and not found_leave and best.get("si83") is not None:
        log("refine around", best.get("si82"), best.get("si83"), best.get("lead"), best.get("max_x"))
        si82 = int(best["si82"])
        _em(env).set_state(st82)
        tr82 = probe_8_2_from_control(env, hl, si82, start_lives=lives82)
        wait83, snap83 = wait_control(env, is_8_3_control)
        st83 = _em(env).get_state()
        lives83 = int(snap83.lives)
        bsi = int(best["si83"])
        blead = int(best.get("lead") or 0)
        search_83(
            st83=st83,
            lives83=lives83,
            si82=si82,
            leave82=int(tr82.w4 or 0),
            wait83=wait83,
            fp83=_fp(snap83),
            movie=hl if str(best.get("src", "HL")).startswith("HL") else fx,
            src=f"{best.get('src')}_refine",
            smin=bsi - 30,
            smax=bsi + 30,
            step=1,
            lead_list=list(range(max(0, blead - 5), blead + 12)),
            max_play=3000,
        )

    report["best"] = {k: v for k, v in (best or {}).items() if k != "_score"}
    report["elapsed_s"] = time.time() - t0
    report["n_hits"] = len(report["hits"])
    out = OUT_DIR / "happylee_8_3_stitchless_fanout.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    # strip internal scores
    for row in report["fanout_83"]:
        row.pop("_score", None)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    log("=== SUMMARY ===")
    log("hits", report["n_hits"])
    log("best", report["best"])
    log("wrote", out, f"in {report['elapsed_s']:.1f}s")
    env.close()
    return 0 if report["hits"] else 2


if __name__ == "__main__":
    sys.exit(main())
