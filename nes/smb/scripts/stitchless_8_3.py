"""Stitchless 8-3: control-relative HL body + short state-gated skills.

Primary path for ``rr-34v``. Does **not** use natural_82 as a mid-splice.
Absolute FM2 offsets are search hints only — gates use rich RAM fingerprints.

Subcommands:

- ``fanout``  multi-leave 8-2 + nearby FM2 starts + even/odd leads; score
  max_x / survival / leave / timer_mod21 (not raw length)
- ``skill-resume``  prefix from progress seed → land-pin checkpoints →
  flagpole / hop grid / HB absorber (short skills)
- ``heal``  localized hold-trim / A-edge / delete near landmarks on a body
- ``verify``  re-run a candidate twice from reconstructed HL 8-3 control
- ``annotate``  dump rich handoff + landmark frames for a body

Artifacts use distinct names (``*_skills_*`` / timestamped evidence). Never
overwrite ``models/smb_8_3_stitchless_progress.json`` or shared hybrid seeds.

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
  uv run python -u -m smb.scripts.stitchless_8_3 skill-resume --quick
```
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from smb.paths import GAME_DIR, GAME_V0, MODELS_DIR, RECORDINGS_DIR
from smb.policy import compress_nes9_rle, expand_nes9_rle, load_nes9_rle_seed
from smb.ram import (
    PLAYER_STATE_AUTO_WALK,
    PLAYER_STATE_DYING,
    PLAYER_STATE_FLAGPOLE,
    read_snapshot,
    rich_handoff_fingerprint,
)
from smb.tas.fm2 import parse_fm2
from smb.tas.replay import IDLE as IDLE_NP
from smb.tas.replay import get_state, idle_until, set_state, to_action9
from smb.tas.skills_8_3 import (
    FLAGPOLE_STYLES,
    IDLE,
    RUN,
    RUN_JUMP,
    X_STAIR_APPROACH,
    flagpole_macro,
    hop_pattern,
    open_skill_catalog,
    score_trial,
)
from smb.tas.chain import reach_8_1_control_after_hl_w8, reach_stage_control
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
)

OUT_DIR = RECORDINGS_DIR / "tas_import"
PROGRESS_SEED = MODELS_DIR / "smb_8_3_stitchless_progress.json"
# Distinct from shared progress / hybrid — never write over those.
CANDIDATE_SEED = MODELS_DIR / "smb_8_3_stitchless_skills_candidate.json"
LEAVE_SEED = MODELS_DIR / "smb_8_3_stitchless_skills_leave.json"


def log(*a: object) -> None:
    print(*a, flush=True)


def _clone(frames: Sequence[Sequence[int]]) -> list[list[int]]:
    return [[int(x) for x in f[:9]] for f in frames]


def reach_hl_8_3_control(env) -> dict[str, Any]:
    """HL W8 → 8-1 → 8-2 → 8-3 control with rich fingerprint (no natural_82)."""
    hl = parse_fm2(DEFAULT_FM2).frames
    pred = reach_stage_control(env, "8-3")
    if not pred.get("success") or pred.get("control_snap") is None:
        return {
            "success": False,
            "stage": pred.get("stage") or "8_3_control",
            "pred": pred,
        }
    snap83 = pred["control_snap"]
    ram = env.get_ram()
    return {
        "success": True,
        "hl": hl,
        "leave_8_1": pred.get("leave_8_1"),
        "wait82": pred.get("ctrl_wait_8_2"),
        "leave_8_2": pred.get("leave_8_2"),
        "wait83": pred.get("ctrl_wait_8_3"),
        "lives83": int(snap83.lives),
        "control_fp": rich_handoff_fingerprint(ram, snap=snap83),
        "state": get_state(env),
    }


def eval_body(
    env,
    state,
    body: Sequence[Sequence[int]],
    lives: int,
    *,
    pad_idle: int = 1000,
    lead: int = 0,
) -> dict[str, Any]:
    """Play body from savestate; score leave / max_x / flag / death.

    Flagpole auto-walk + castle tally often needs **~700–900 idle frames** after
    the skill body ends — pad is folded into exported leave seeds.
    """
    set_state(env, state)
    for _ in range(lead):
        env.step(IDLE_NP)
    max_x = 0
    flag_at = leave = death = None
    last_fp: dict[str, object] | None = None
    seq = list(body) + [list(IDLE) for _ in range(pad_idle)]
    for i, fr in enumerate(seq):
        env.step(to_action9(fr))
        ram = env.get_ram()
        snap = read_snapshot(ram, i + 1)
        px = int(snap.player_x)
        ps = int(snap.player_state)
        w, l = int(snap.world), int(snap.level)
        if w == 7 and l == 2 and 0 < px < 20_000:
            max_x = max(max_x, px)
        if flag_at is None and ps in (
            PLAYER_STATE_FLAGPOLE,
            PLAYER_STATE_AUTO_WALK,
        ) and px > 3200:
            flag_at = i + 1
        if is_8_4_control(snap):
            fp = rich_handoff_fingerprint(ram, snap=snap)
            return {
                "ok": True,
                "leave": i + 1,
                "max_x": max_x,
                "flag_at": flag_at,
                "timer": int(snap.timer),
                "timer_mod21": int(snap.timer) % 21,
                "fp": fp,
                "played_seq_len": len(seq),
            }
        if int(snap.lives) < lives or ps == PLAYER_STATE_DYING:
            last_fp = rich_handoff_fingerprint(ram, snap=snap)
            death = i + 1
            return {
                "ok": False,
                "death": death,
                "max_x": max_x,
                "x": px,
                "y": int(snap.player_y),
                "flag_at": flag_at,
                "fp": last_fp,
            }
    snap = read_snapshot(env.get_ram(), 0)
    return {
        "ok": False,
        "exhausted": True,
        "max_x": max_x,
        "flag_at": flag_at,
        "x": int(snap.player_x),
        "fp": rich_handoff_fingerprint(env.get_ram(), snap=snap),
    }


def fold_leave_body(
    body: Sequence[Sequence[int]], leave: int
) -> list[list[int]]:
    """Include trailing idle so export length == leave (pad-fold)."""
    frames = _clone(body)
    if leave > len(frames):
        frames.extend(list(IDLE) for _ in range(leave - len(frames)))
    return frames[:leave]


def export_seed(
    path: Path,
    frames: list[list[int]],
    *,
    route_id: str,
    meta: dict[str, Any],
) -> Path:
    payload = {
        "format": "nes9_rle",
        "route_id": route_id,
        "game_name": "SuperMarioBros-Nes-v0",
        "num_frames": len(frames),
        "body_frames": len(frames),
        "start_state": "8-3_control_after_hl_8_2_stitchless",
        "target": "8_4_control",
        "stitchless": True,
        "pure_hl_tas_prefix": True,
        "no_natural_82_splice": True,
        "preserve_lr": True,
        **meta,
        "segments": compress_nes9_rle(frames),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def cmd_annotate(args: argparse.Namespace) -> int:
    from retro_harness.env import make_env

    seed = Path(args.seed)
    body = _clone(expand_nes9_rle(load_nes9_rle_seed(seed)))
    env = make_env(GAME_V0, "Level1_1", GAME_DIR, render_mode="rgb_array")
    env.reset()
    ctx = reach_hl_8_3_control(env)
    if not ctx["success"]:
        log("FAIL reach", ctx)
        env.close()
        return 1
    landmarks: list[dict[str, Any]] = []
    set_state(env, ctx["state"])
    for i, fr in enumerate(body):
        env.step(to_action9(fr))
        ram = env.get_ram()
        snap = read_snapshot(ram, i + 1)
        px = int(snap.player_x)
        mark = None
        if px >= X_STAIR_APPROACH and snap.grounded and px < 3200:
            mark = "stair_land"
        if int(snap.player_state) == PLAYER_STATE_FLAGPOLE:
            mark = "flag"
        if mark or (i + 1) % 200 == 0 or px in (1000, 2000, 3000):
            fp = rich_handoff_fingerprint(ram, snap=snap)
            landmarks.append({"i": i + 1, "mark": mark, **{k: fp[k] for k in (
                "player_x", "player_y", "x_speed", "y_speed", "grounded",
                "player_state", "timer", "timer_mod21", "x_frac", "y_frac",
                "frame_counter", "enemy_types", "n_hammer_bro",
            )}})
        if int(snap.lives) < ctx["lives83"] or int(snap.player_state) == PLAYER_STATE_DYING:
            break
    out = OUT_DIR / "happylee_8_3_skills_annotate.json"
    out.write_text(
        json.dumps(
            {
                "seed": str(seed),
                "ctrl_fp": ctx["control_fp"],
                "n_landmarks": len(landmarks),
                "landmarks": landmarks,
                "skills": open_skill_catalog(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    log("wrote", out, "ctrl", ctx["control_fp"].get("timer"), "n", len(landmarks))
    env.close()
    return 0


def cmd_fanout(args: argparse.Namespace) -> int:
    from retro_harness.env import make_env

    t0 = time.time()
    env = make_env(GAME_V0, "Level1_1", GAME_DIR, render_mode="rgb_array")
    env.reset()
    hl = parse_fm2(DEFAULT_FM2).frames
    pred = reach_8_1_control_after_hl_w8(env)
    if not pred.get("success"):
        log("FAIL 8-1", pred)
        env.close()
        return 1
    lives81 = int(pred["control_snap"].lives)
    tr81 = probe_8_1_from_control(env, hl, HL_8_1_FM2_START, start_lives=lives81)
    wait82, snap82 = idle_until(env, is_8_2_control)
    lives82 = int(snap82.lives)
    st82 = get_state(env)
    log(f"8-1 leave={tr81.leave_frame} wait82={wait82}")

    leads = [int(x) for x in args.leads.split(",") if x.strip() != ""]
    s82_min, s82_max, s82_step = args.s82_min, args.s82_max, args.s82_step
    s83_min, s83_max, s83_step = args.s83_min, args.s83_max, args.s83_step

    leave_rows: list[dict[str, Any]] = []
    for si82 in range(s82_min, s82_max + 1, s82_step):
        set_state(env, st82)
        tr = probe_8_2_from_control(env, hl, si82, start_lives=lives82)
        if not tr.ok:
            continue
        wait83, snap83 = idle_until(env, is_8_3_control)
        if not is_8_3_control(snap83):
            continue
        leave_rows.append(
            {
                "si82": si82,
                "leave82": tr.leave_frame,
                "wait83": wait83,
                "ctrl_fp": rich_handoff_fingerprint(env.get_ram(), snap=snap83),
            }
        )
        log(
            f"LEAVE82 si={si82} leave={tr.leave_frame} wait83={wait83} "
            f"t={leave_rows[-1]['ctrl_fp']['timer']} "
            f"fc={leave_rows[-1]['ctrl_fp']['frame_counter']}"
        )

    # unique leave length classes (phase may still match)
    seen: set[tuple[int, int]] = set()
    unique: list[dict[str, Any]] = []
    for row in sorted(leave_rows, key=lambda r: r["leave82"]):
        key = (int(row["leave82"]), int(row["ctrl_fp"]["timer"]))
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    fan = unique[: max(1, args.top_leaves)]
    best: dict[str, Any] | None = None
    trials: list[dict[str, Any]] = []

    for leave in fan:
        si82 = int(leave["si82"])
        set_state(env, st82)
        tr82 = probe_8_2_from_control(env, hl, si82, start_lives=lives82)
        wait83, snap83 = idle_until(env, is_8_3_control)
        st83 = get_state(env)
        lives83 = int(snap83.lives)
        fp83 = rich_handoff_fingerprint(env.get_ram(), snap=snap83)
        log(f"=== fan si82={si82} leave82={tr82.leave_frame} fp.t={fp83['timer']} ===")
        for lead in leads:
            for si83 in range(s83_min, s83_max + 1, s83_step):
                set_state(env, st83)
                for _ in range(lead):
                    env.step(IDLE_NP)
                tr = probe_8_3_from_control(
                    env, hl, si83, start_lives=lives83, max_play=args.max_play
                )
                row = {
                    "si82": si82,
                    "leave82": tr82.leave_frame,
                    "lead": lead,
                    "si83": si83,
                    "leave83": tr.leave_frame,
                    "death": tr.death,
                    "max_x": tr.max_x,
                    "ctrl_fp": fp83,
                }
                row["_score"] = score_trial(
                    {
                        "leave": tr.leave_frame,
                        "max_x": tr.max_x or 0,
                        "death": tr.death,
                        "timer_mod21": int(fp83.get("timer_mod21") or 0),
                    }
                )
                trials.append(row)
                if best is None or row["_score"] > best["_score"]:
                    best = row
                    log(
                        f"BEST leave={tr.leave_frame} max_x={tr.max_x} "
                        f"si83={si83} lead={lead}"
                    )
                if tr.ok and args.export_on_hit:
                    body = [list(f) for f in hl[si83 : si83 + int(tr.leave_frame)]]
                    export_seed(
                        LEAVE_SEED,
                        body,
                        route_id="smb_8_3_stitchless_skills_leave",
                        meta={
                            "leave_frames": tr.leave_frame,
                            "fm2_start_hint": si83,
                            "note": "FM2 body hint only — re-gate from control",
                            "si82": si82,
                            "lead": lead,
                            "ctrl_fp": fp83,
                            "verified_leave_8_4": True,
                            "source": "stitchless fanout (no natural_82)",
                        },
                    )
                    log("EXPORTED leave", tr.leave_frame)

    report = {
        "lane": "stitchless_skills_fanout",
        "n_leave82": len(leave_rows),
        "unique_classes": len(unique),
        "n_trials": len(trials),
        "best": {k: v for k, v in (best or {}).items() if k != "_score"},
        "top": sorted(
            ({k: v for k, v in r.items() if k != "_score"} for r in trials),
            key=lambda r: score_trial(
                {
                    "leave": r.get("leave83"),
                    "max_x": r.get("max_x") or 0,
                    "death": r.get("death"),
                }
            ),
            reverse=True,
        )[:20],
        "elapsed_s": time.time() - t0,
        "skills": open_skill_catalog(),
    }
    out = OUT_DIR / "happylee_8_3_skills_fanout.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    log("SUMMARY best", report["best"], "wrote", out)
    env.close()
    return 0 if best and best.get("leave83") else 2


def cmd_skill_resume(args: argparse.Namespace) -> int:
    """Prefix progress body to land-pins; short hop/flagpole skills for leave."""
    from retro_harness.env import make_env

    t0 = time.time()
    seed_in = Path(args.seed)
    body0 = _clone(expand_nes9_rle(load_nes9_rle_seed(seed_in)))
    env = make_env(GAME_V0, "Level1_1", GAME_DIR, render_mode="rgb_array")
    env.reset()
    ctx = reach_hl_8_3_control(env)
    if not ctx["success"]:
        log("FAIL", ctx)
        env.close()
        return 1
    st83 = ctx["state"]
    lives83 = int(ctx["lives83"])
    log("ctrl_fp", {k: ctx["control_fp"][k] for k in (
        "player_x", "player_y", "timer", "timer_mod21", "frame_counter",
        "x_frac", "enemy_types", "grounded",
    )})

    base_r = eval_body(env, st83, body0, lives83)
    log("BASE", {k: base_r.get(k) for k in (
        "ok", "leave", "max_x", "death", "x", "y", "flag_at",
    )})

    # Build land-pin checkpoints from prefix (grounded seats in stair band)
    set_state(env, st83)
    pins: list[dict[str, Any]] = []
    for i, fr in enumerate(body0):
        env.step(to_action9(fr))
        ram = env.get_ram()
        snap = read_snapshot(ram, i + 1)
        px = int(snap.player_x)
        if int(snap.lives) < lives83 or int(snap.player_state) == PLAYER_STATE_DYING:
            break
        if (
            snap.grounded
            and X_STAIR_APPROACH - 80 <= px <= 3300
            and int(snap.x_speed) >= 20
        ):
            pins.append(
                {
                    "i": i + 1,
                    "state": get_state(env),
                    "fp": rich_handoff_fingerprint(ram, snap=snap),
                    "prefix": _clone(body0[: i + 1]),
                }
            )
        # also every N frames in approach window for non-grounded cut experiments
        if args.dense_cuts and 1480 <= i + 1 <= 1560 and (i + 1) % 3 == 0:
            pins.append(
                {
                    "i": i + 1,
                    "state": get_state(env),
                    "fp": rich_handoff_fingerprint(ram, snap=snap),
                    "prefix": _clone(body0[: i + 1]),
                }
            )

    # Dedup by frame index (last wins)
    by_i = {p["i"]: p for p in pins}
    pins = [by_i[k] for k in sorted(by_i)]
    if args.quick:
        # keep ~every other grounded + endpoints
        pins = pins[::2][:40] if len(pins) > 40 else pins
    log(f"land_pins={len(pins)} first={pins[0]['i'] if pins else None} "
        f"last={pins[-1]['i'] if pins else None}")

    best = dict(base_r)
    best_body = _clone(body0)
    best["_score"] = score_trial(base_r)
    moves: list[dict[str, Any]] = []
    tries = 0

    def consider(cand: list[list[int]], meta: dict[str, Any], *, st=None, lead: int = 0) -> bool:
        nonlocal best, best_body, tries
        tries += 1
        r = eval_body(env, st if st is not None else st83, cand, lives83, lead=lead)
        sc = score_trial(r)
        if sc > best["_score"]:
            best = {**r, "_score": sc}
            best_body = _clone(cand)
            if r.get("ok") and r.get("leave"):
                best_body = best_body[: int(r["leave"])]
            row = {**meta, **{k: r.get(k) for k in (
                "ok", "leave", "max_x", "death", "x", "y", "flag_at", "timer_mod21",
            )}}
            moves.append(row)
            log("+", row)
            return True
        return False

    styles = list(FLAGPOLE_STYLES) if not args.quick else ["mid", "stairs", "lip", "tall"]
    hop_grid = (
        [(20, 8, 2), (28, 12, 3), (36, 10, 3), (24, 14, 4), (40, 8, 2), (32, 16, 5)]
        if not args.quick
        else [(28, 12, 3), (36, 10, 3), (24, 14, 4), (40, 8, 2)]
    )

    log("=== skill: flagpole styles @ land-pins ===")
    for pin in pins:
        pref = pin["prefix"]
        st = pin["state"]
        for style in styles:
            tail = flagpole_macro(style=style)
            consider(pref + tail, {"op": "flag_style", "cut": pin["i"], "style": style}, st=st83)
            if best.get("ok"):
                break
        if best.get("ok"):
            break
        for jh, gap, hops in hop_grid:
            for run0 in (0, 4, 8):
                tail = hop_pattern(run0=run0, jhold=jh, gap=gap, hops=hops, run_tail=120)
                consider(
                    pref + tail,
                    {"op": "hop", "cut": pin["i"], "jh": jh, "gap": gap, "hops": hops, "run0": run0},
                    st=st83,
                )
                if best.get("ok"):
                    break
            if best.get("ok"):
                break
        if best.get("ok"):
            break
        if pin["i"] % 20 == 0:
            log(f"  pin={pin['i']} x={pin['fp']['player_x']} best_max={best.get('max_x')} "
                f"flag={best.get('flag_at')} tries={tries}")

    # Closed-loop reactive multi-hop from each pin (state-gated)
    if not best.get("ok"):
        log("=== skill: closed-loop multi-hop ===")
        jholds = (20, 28, 36, 48) if args.quick else (16, 20, 24, 28, 32, 36, 40, 48, 56)
        for pin in pins:
            for jhold in jholds:
                for maxj in (3, 4, 5, 6):
                    set_state(env, pin["state"])
                    rec: list[list[int]] = []
                    jump_left = 0
                    jumps = 0
                    for _t in range(900):
                        ram = env.get_ram()
                        s = read_snapshot(ram, 0)
                        ps = int(s.player_state)
                        g = bool(s.grounded)
                        px = int(s.player_x)
                        if ps in (PLAYER_STATE_FLAGPOLE, PLAYER_STATE_AUTO_WALK):
                            fr = list(IDLE)
                        elif jump_left > 0:
                            fr = list(RUN_JUMP)
                            jump_left -= 1
                        elif g and jumps < maxj and px >= X_STAIR_APPROACH - 100:
                            fr = list(RUN_JUMP)
                            jump_left = jhold - 1
                            jumps += 1
                        else:
                            fr = list(RUN)
                        env.step(to_action9(fr))
                        rec.append(list(fr))
                        s2 = read_snapshot(env.get_ram(), 0)
                        if is_8_4_control(s2):
                            break
                        if int(s2.lives) < lives83 or int(s2.player_state) == PLAYER_STATE_DYING:
                            break
                    consider(
                        pin["prefix"] + rec,
                        {"op": "cl", "cut": pin["i"], "jhold": jhold, "maxj": maxj},
                        st=st83,
                    )
                    if best.get("ok"):
                        break
                if best.get("ok"):
                    break
            if best.get("ok"):
                break

    # Localized heal near death on best (hold trim / A-edge / del)
    if not best.get("ok") and args.heal:
        log("=== local heal near stall/death ===")
        body = _clone(best_body)
        br = eval_body(env, st83, body, lives83)
        death = int(br.get("death") or len(body))
        lo = max(0, death - 200)
        hi = min(len(body), death)
        # A-edge: extend/shorten A runs in window
        for at in range(lo, hi, 2 if args.quick else 1):
            if at >= len(body):
                break
            fr = list(body[at])
            # toggle A
            for a_val in (0, 1):
                if fr[8] == a_val:
                    continue
                cand = _clone(body)
                cand[at][8] = a_val
                consider(cand, {"op": "a_toggle", "at": at, "a": a_val})
                if best.get("ok"):
                    break
            if best.get("ok"):
                break
            # delete single frame
            if at + 1 < len(body):
                cand = body[:at] + body[at + 1 :]
                consider(cand, {"op": "del", "at": at})
                if best.get("ok"):
                    break
            # insert short hop
            for jh in (16, 24, 32):
                hop = [list(RUN_JUMP) for _ in range(jh)]
                consider(body[:at] + hop + body[at:], {"op": "ins_hop", "at": at, "jh": jh})
                if best.get("ok"):
                    break
            if best.get("ok"):
                break
            if at % 25 == 0:
                log(f"  heal at={at} max={best.get('max_x')} tries={tries}")

    final = eval_body(env, st83, best_body, lives83, pad_idle=700)
    log("FINAL", {k: final.get(k) for k in (
        "ok", "leave", "max_x", "death", "x", "y", "flag_at", "timer_mod21",
    )})

    ok_export = False
    export_body = _clone(best_body)
    verify_trials: list[dict[str, Any]] = []
    if final.get("ok") and final.get("leave"):
        # Fold idle pad so verify pad_idle=0 still reaches 8-4 control.
        export_body = fold_leave_body(best_body, int(final["leave"]))
        for t in range(2):
            r = eval_body(env, st83, export_body, lives83, pad_idle=0)
            verify_trials.append(
                {k: r.get(k) for k in ("ok", "leave", "max_x", "flag_at", "timer_mod21")}
            )
            log(f"VERIFY t{t+1}", verify_trials[-1])
        if all(v.get("ok") for v in verify_trials):
            ok_export = True
            export_seed(
                LEAVE_SEED,
                export_body,
                route_id="smb_8_3_stitchless_skills_leave",
                meta={
                    "leave_frames": final["leave"],
                    "flag_at": final.get("flag_at"),
                    "verified_leave_8_4": True,
                    "timer_mod21": final.get("timer_mod21"),
                    "verify_trials": verify_trials,
                    "ctrl_fp": ctx["control_fp"],
                    "source": "skill-resume land-pin/hop/flagpole; no natural_82 splice",
                    "base_seed": str(seed_in),
                    "ops": moves[-40:],
                    "no_natural_82_splice": True,
                    "preserve_lr": True,
                },
            )
            log("EXPORTED", LEAVE_SEED, final["leave"])

    if not ok_export:
        export_seed(
            CANDIDATE_SEED,
            best_body,
            route_id="smb_8_3_stitchless_skills_candidate",
            meta={
                "best": {k: final.get(k) for k in (
                    "ok", "leave", "max_x", "death", "x", "y", "flag_at", "timer_mod21",
                )},
                "ctrl_fp": ctx["control_fp"],
                "source": "skill-resume progress; shared progress seed untouched",
                "base_seed": str(seed_in),
                "n_ops": len(moves),
                "ops": moves[-40:],
                "note": "candidate only — no leave yet",
            },
        )
        log("candidate", CANDIDATE_SEED, final.get("max_x"))

    report = {
        "lane": "stitchless_skills_resume",
        "paused_prior": "grounded/pit-jump long-hop grids superseded by skill-resume",
        "base": {k: base_r.get(k) for k in ("ok", "max_x", "death", "x", "flag_at")},
        "final": {k: final.get(k) for k in (
            "ok", "leave", "max_x", "death", "x", "y", "flag_at", "timer_mod21",
        )},
        "n_pins": len(pins),
        "n_moves": len(moves),
        "n_tries": tries,
        "moves": moves[-30:],
        "verify_trials": verify_trials,
        "ctrl_fp": ctx["control_fp"],
        "shared_progress_untouched": str(PROGRESS_SEED.resolve()),
        "artifact": str(LEAVE_SEED if ok_export else CANDIDATE_SEED),
        "skills": open_skill_catalog(),
        "elapsed_s": time.time() - t0,
    }
    rep_path = OUT_DIR / "happylee_8_3_skills_resume.json"
    rep_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    log("WROTE", rep_path, f"tries={tries} in {report['elapsed_s']:.1f}s")
    env.close()
    return 0 if ok_export else 2


def cmd_verify(args: argparse.Namespace) -> int:
    from retro_harness.env import make_env

    seed = Path(args.seed)
    body = _clone(expand_nes9_rle(load_nes9_rle_seed(seed)))
    env = make_env(GAME_V0, "Level1_1", GAME_DIR, render_mode="rgb_array")
    env.reset()
    ctx = reach_hl_8_3_control(env)
    if not ctx["success"]:
        log("FAIL", ctx)
        env.close()
        return 1
    trials = []
    for t in range(args.trials):
        r = eval_body(env, ctx["state"], body, int(ctx["lives83"]), pad_idle=0)
        trials.append(r)
        log(f"t{t+1}", {k: r.get(k) for k in ("ok", "leave", "max_x", "flag_at")})
    out = OUT_DIR / "happylee_8_3_skills_verify.json"
    out.write_text(
        json.dumps(
            {
                "seed": str(seed),
                "ctrl_fp": ctx["control_fp"],
                "trials": [
                    {k: r.get(k) for k in ("ok", "leave", "max_x", "flag_at", "timer_mod21")}
                    for r in trials
                ],
                "all_ok": all(r.get("ok") for r in trials),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    log("wrote", out)
    env.close()
    return 0 if all(r.get("ok") for r in trials) else 2


def cmd_heal(args: argparse.Namespace) -> int:
    """Localized landmark heal (del / A-edge / hop-ins) — distinct candidate out."""
    # Reuse skill-resume with heal forced and denser window
    args.heal = True
    args.quick = getattr(args, "quick", False)
    args.dense_cuts = True
    args.seed = args.seed
    return cmd_skill_resume(args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fanout", help="multi-leave + FM2 start/lead fanout")
    f.add_argument("--s82-min", type=int, default=10840)
    f.add_argument("--s82-max", type=int, default=10940)
    f.add_argument("--s82-step", type=int, default=2)
    f.add_argument("--s83-min", type=int, default=12950)
    f.add_argument("--s83-max", type=int, default=13200)
    f.add_argument("--s83-step", type=int, default=5)
    f.add_argument("--leads", type=str, default="0,1,2,3,5,10,14,21")
    f.add_argument("--top-leaves", type=int, default=5)
    f.add_argument("--max-play", type=int, default=2800)
    f.add_argument("--export-on-hit", action="store_true", default=True)
    f.set_defaults(func=cmd_fanout)

    s = sub.add_parser("skill-resume", help="land-pin → short flagpole/hop skills")
    s.add_argument("--seed", type=str, default=str(PROGRESS_SEED))
    s.add_argument("--quick", action="store_true")
    s.add_argument("--heal", action="store_true", default=True)
    s.add_argument("--no-heal", action="store_false", dest="heal")
    s.add_argument("--dense-cuts", action="store_true", default=True)
    s.set_defaults(func=cmd_skill_resume)

    h = sub.add_parser("heal", help="localized heal alias of skill-resume")
    h.add_argument("--seed", type=str, default=str(PROGRESS_SEED))
    h.add_argument("--quick", action="store_true")
    h.set_defaults(func=cmd_heal)

    v = sub.add_parser("verify", help="verify candidate 2× from HL 8-3 control")
    v.add_argument("--seed", type=str, required=True)
    v.add_argument("--trials", type=int, default=2)
    v.set_defaults(func=cmd_verify)

    a = sub.add_parser("annotate", help="rich landmarks for a body")
    a.add_argument("--seed", type=str, default=str(PROGRESS_SEED))
    a.set_defaults(func=cmd_annotate)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
