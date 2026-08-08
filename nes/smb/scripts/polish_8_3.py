"""Polish control-relative 8-3 after HappyLee 8-2 (not pure FM2).

The hybrid bridge ``smb_8_3_natural_for_hl_hybrid.json`` already clears
HL 8-3 control → 8-4 control in **2227f**. Small frame deletes often keep
the same total (21-frame rule + flag timer). This tool:

1. Rebuilds once to HL 8-3 control and savestates.
2. Baseline-evaluates a body (default: natural bridge).
3. Hold-trim / delete / A-edge search on the **pre-flag** region.
4. Optionally rebuilds hybrid prefix+body+flamexx 8-4 for full-chain check.

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
  uv run python -m smb.scripts.polish_8_3 --windows body --delete-stride 3

# Faster: only hold trims
uv run python -m smb.scripts.polish_8_3 --no-delete --no-edge
```
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from smb.paths import GAME_DIR, GAME_V0, MODELS_DIR, RECORDINGS_DIR
from smb.policy import compress_nes9_rle, expand_nes9_rle, load_nes9_rle_seed
from smb.ram import PLAYER_STATE_DYING, read_snapshot, reached_ending
from smb.tas.replay import get_state, set_state, to_action9
from smb.tas.chain import reach_stage_control
from smb.tas.slice import (
    DEFAULT_HL_1_1,
    DEFAULT_FM2,
    is_8_4_control,
)

DEFAULT_BODY = MODELS_DIR / "smb_8_3_natural_for_hl_hybrid.json"
DEFAULT_OUT = MODELS_DIR / "smb_8_3_control_best.json"
DEFAULT_FX = MODELS_DIR / "smb_8_4_flamexx_slice.json"

# Pre-flag optimizable region (trailing ~722f is flag/timer/load idle).
WINDOW_PRESETS: dict[str, tuple[int, int]] = {
    "lead": (0, 80),
    "early": (30, 500),
    "mid": (500, 1000),
    "late": (1000, 1505),
    "body": (30, 1505),
    "all": (0, 10_000),
}

IDLE = [0, 0, 0, 0, 0, 0, 0, 0, 0]


def _clone(frames: Sequence[Sequence[int]]) -> list[list[int]]:
    return [[int(x) for x in f[:9]] for f in frames]


def reach_hl_8_3_control(env) -> dict[str, Any]:
    """Play HL chain to first ``is_8_3_control``; return waits + fingerprint."""
    pred = reach_stage_control(
        env, "8-3", fm2_path=DEFAULT_FM2, hl_1_1_path=DEFAULT_HL_1_1
    )
    if not pred.get("success") or pred.get("control_snap") is None:
        return {
            "success": False,
            "stage": pred.get("stage") or "8_3_control",
            "pred": pred,
        }
    snap = pred["control_snap"]
    return {
        "success": True,
        "ctrl_wait_8_1": pred.get("ctrl_wait_8_1"),
        "leave_8_1": pred.get("leave_8_1"),
        "ctrl_wait_8_2": pred.get("ctrl_wait_8_2"),
        "leave_8_2": pred.get("leave_8_2"),
        "ctrl_wait_8_3": pred.get("ctrl_wait_8_3"),
        "control_fp": {
            "player_x": int(snap.player_x),
            "player_y": int(snap.player_y),
            "timer": int(snap.timer),
            "player_state": int(snap.player_state),
            "lives": int(snap.lives),
        },
    }


def eval_to_8_4(
    env,
    body: Sequence[Sequence[int]],
    *,
    pad_idle: int = 0,
    start_lives: int | None = None,
) -> dict[str, Any]:
    """Play body (+ optional idle pad) until 8-4 control, death, or exhaust."""
    if start_lives is None:
        start_lives = int(read_snapshot(env.get_ram(), 0).lives)
    max_x = 0
    flag_at: int | None = None
    seq = list(body) + [list(IDLE) for _ in range(pad_idle)]
    for i, fr in enumerate(seq):
        env.step(to_action9(fr))
        snap = read_snapshot(env.get_ram(), i + 1)
        px = int(snap.player_x)
        if 0 < px < 20_000:
            max_x = max(max_x, px)
        if flag_at is None and int(snap.player_state) == 5 and px > 3000:
            flag_at = i + 1
        if is_8_4_control(snap):
            return {
                "ok": True,
                "frames": i + 1,
                "flag_at": flag_at,
                "max_x": max_x,
                "timer": int(snap.timer),
            }
        if int(snap.lives) < start_lives or int(snap.player_state) == PLAYER_STATE_DYING:
            return {
                "ok": False,
                "death": i + 1,
                "flag_at": flag_at,
                "max_x": max_x,
                "x": px,
            }
    snap = read_snapshot(env.get_ram(), 0)
    return {
        "ok": False,
        "exhausted": True,
        "flag_at": flag_at,
        "max_x": max_x,
        "x": int(snap.player_x),
        "level": int(snap.level),
    }


def eval_ending(env, body: Sequence[Sequence[int]], *, start_lives: int | None = None) -> dict[str, Any]:
    if start_lives is None:
        start_lives = int(read_snapshot(env.get_ram(), 0).lives)
    for i, fr in enumerate(body):
        env.step(to_action9(fr))
        ram = env.get_ram()
        if reached_ending(ram, start_lives=start_lives):
            return {"ok": True, "ending": i + 1}
        snap = read_snapshot(ram, i + 1)
        if int(snap.lives) < start_lives or int(snap.player_state) == PLAYER_STATE_DYING:
            return {"ok": False, "death": i + 1, "x": int(snap.player_x)}
    return {"ok": False, "exhausted": True}


@dataclass
class PolishReport:
    baseline: int
    best: int
    frames_saved: int
    improvements: list[dict[str, Any]] = field(default_factory=list)
    out_path: str | None = None
    ctrl_meta: dict[str, Any] = field(default_factory=dict)
    elapsed_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline,
            "best": self.best,
            "frames_saved": self.frames_saved,
            "improvements": self.improvements,
            "out_path": self.out_path,
            "ctrl_meta": self.ctrl_meta,
            "elapsed_s": self.elapsed_s,
        }


def _hold_trim(
    env,
    state: bytes,
    frames: list[list[int]],
    *,
    window: tuple[int, int],
    baseline: int,
    min_hold: int = 20,
    trims: Sequence[int] = (21, 15, 10, 8, 5, 3, 1),
    pad_idle: int = 40,
    verbose: bool = True,
) -> tuple[list[list[int]], int, list[dict[str, Any]]]:
    best = _clone(frames)
    best_clear = baseline
    moves: list[dict[str, Any]] = []
    lo, hi = window
    changed = True
    while changed:
        changed = False
        i = max(0, lo)
        holds: list[tuple[int, int]] = []
        while i < min(hi, len(best)):
            j = i + 1
            while j < len(best) and best[j] == best[i]:
                j += 1
            if j - i >= min_hold and any(best[i]):
                holds.append((i, j - i))
            i = j
        for start, hold in holds:
            for trim in trims:
                if hold - trim < 12:
                    continue
                cand = best[: start + hold - trim] + best[start + hold :]
                set_state(env, state)
                r = eval_to_8_4(env, cand, pad_idle=pad_idle)
                if r.get("ok") and int(r["frames"]) < best_clear:
                    if verbose:
                        print(
                            f"[HOLD] @{start} hold={hold} trim={trim} "
                            f"→ {r['frames']} (flag={r.get('flag_at')})",
                            flush=True,
                        )
                    best = cand
                    best_clear = int(r["frames"])
                    moves.append(
                        {
                            "op": "hold_trim",
                            "at": start,
                            "hold": hold,
                            "trim": trim,
                            "clear": best_clear,
                        }
                    )
                    changed = True
                    break
            if changed:
                break
    return best, best_clear, moves


def _delete_sweep(
    env,
    state: bytes,
    frames: list[list[int]],
    *,
    window: tuple[int, int],
    baseline: int,
    stride: int = 2,
    pad_idle: int = 40,
    verbose: bool = True,
) -> tuple[list[list[int]], int, list[dict[str, Any]]]:
    best = _clone(frames)
    best_clear = baseline
    moves: list[dict[str, Any]] = []
    lo, hi = max(0, window[0]), min(len(best), window[1])
    i = lo
    tries = 0
    t0 = time.time()
    while i < min(hi, len(best)):
        cand = best[:i] + best[i + 1 :]
        set_state(env, state)
        r = eval_to_8_4(env, cand, pad_idle=pad_idle)
        tries += 1
        if r.get("ok") and int(r["frames"]) < best_clear:
            if verbose:
                print(f"[DEL] @{i} → {r['frames']}", flush=True)
            best = cand
            best_clear = int(r["frames"])
            moves.append({"op": "delete", "at": i, "clear": best_clear})
            hi = min(hi, len(best))
            continue
        i += max(1, stride)
    if verbose:
        print(
            f"[DEL] window=[{lo}:{hi}] tries={tries} imps={len(moves)} "
            f"clear={best_clear} in {time.time() - t0:.1f}s",
            flush=True,
        )
    return best, best_clear, moves


def save_body(
    path: Path,
    frames: Sequence[Sequence[int]],
    *,
    baseline: int,
    best: int,
    source: Path,
) -> Path:
    nes9 = [[int(x) for x in f[:9]] for f in frames]
    payload = {
        "format": "nes9_rle",
        "route_id": "smb_8_3_control_best",
        "game_name": "SuperMarioBros-Nes-v0",
        "num_frames": len(nes9),
        "body_frames": len(nes9),
        "start_state": "8-3_control_after_happylee_8_2",
        "target": "8_4_control",
        "source": str(source),
        "optimization": {
            "tool": "smb.scripts.polish_8_3",
            "baseline_to_8_4_control": baseline,
            "best_to_8_4_control": best,
            "frames_saved": max(0, baseline - best),
        },
        "note": (
            "Control-relative 8-3 after HL 8-2. Not pure HappyLee FM2. "
            "Small trims may keep the same framerule class (2227)."
        ),
        "segments": compress_nes9_rle(nes9),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def polish(
    *,
    body_path: Path = DEFAULT_BODY,
    windows: Sequence[str] = ("body",),
    delete_stride: int = 3,
    do_delete: bool = True,
    do_hold: bool = True,
    out_path: Path = DEFAULT_OUT,
    verbose: bool = True,
) -> PolishReport:
    from retro_harness.env import make_env

    t0 = time.time()
    env = make_env(GAME_V0, "Level1_1", GAME_DIR, render_mode="rgb_array")
    env.reset()
    meta = reach_hl_8_3_control(env)
    if not meta.get("success"):
        env.close()
        raise RuntimeError(f"failed to reach HL 8-3 control: {meta}")
    state = get_state(env)
    body = _clone(expand_nes9_rle(load_nes9_rle_seed(body_path)))
    set_state(env, state)
    base_r = eval_to_8_4(env, body, pad_idle=0)
    if not base_r.get("ok"):
        env.close()
        raise RuntimeError(f"baseline body failed: {base_r}")
    baseline = int(base_r["frames"])
    if verbose:
        print(
            f"baseline to 8-4 control: {baseline}f "
            f"(flag≈{base_r.get('flag_at')}) wait83={meta.get('ctrl_wait_8_3')}",
            flush=True,
        )

    best = _clone(body)
    best_clear = baseline
    improvements: list[dict[str, Any]] = []

    for wname in windows:
        if wname not in WINDOW_PRESETS:
            raise KeyError(f"unknown window {wname!r}; choose {sorted(WINDOW_PRESETS)}")
        win = WINDOW_PRESETS[wname]
        if verbose:
            print(f"=== window {wname} {win} ===", flush=True)
        if do_hold:
            best, best_clear, moves = _hold_trim(
                env, state, best, window=win, baseline=best_clear, verbose=verbose
            )
            improvements.extend(moves)
        if do_delete:
            best, best_clear, moves = _delete_sweep(
                env,
                state,
                best,
                window=win,
                baseline=best_clear,
                stride=delete_stride,
                verbose=verbose,
            )
            improvements.extend(moves)

    out: Path | None = None
    if best_clear < baseline or True:
        # Always write so re-runs have an artifact even if framerule-locked.
        out = save_body(out_path, best, baseline=baseline, best=best_clear, source=body_path)
        if verbose:
            print(f"wrote {out} clear={best_clear} (−{baseline - best_clear})", flush=True)

    env.close()
    return PolishReport(
        baseline=baseline,
        best=best_clear,
        frames_saved=max(0, baseline - best_clear),
        improvements=improvements,
        out_path=str(out) if out else None,
        ctrl_meta={k: v for k, v in meta.items() if k != "pred"},
        elapsed_s=time.time() - t0,
    )


def probe_fx_trim(
    *,
    body_83: Path = DEFAULT_BODY,
    fx_path: Path = DEFAULT_FX,
    max_deletes: int = 40,
    verbose: bool = True,
) -> dict[str, Any]:
    """After natural 8-3, try deleting late FX frames for sub-5 (−1f class)."""
    from retro_harness.env import make_env

    env = make_env(GAME_V0, "Level1_1", GAME_DIR, render_mode="rgb_array")
    env.reset()
    meta = reach_hl_8_3_control(env)
    if not meta.get("success"):
        env.close()
        return {"success": False, "meta": meta}
    state83 = get_state(env)
    body = expand_nes9_rle(load_nes9_rle_seed(body_83))
    set_state(env, state83)
    r83 = eval_to_8_4(env, body, pad_idle=0)
    if not r83.get("ok"):
        env.close()
        return {"success": False, "r83": r83}
    state84 = get_state(env)
    fx = _clone(expand_nes9_rle(load_nes9_rle_seed(fx_path)))
    set_state(env, state84)
    base = eval_ending(env, fx)
    if not base.get("ok"):
        env.close()
        return {"success": False, "fx_base": base}
    best_end = int(base["ending"])
    best_fx = _clone(fx)
    moves: list[dict[str, Any]] = []
    if verbose:
        print(f"FX baseline ending={best_end} len={len(fx)}", flush=True)

    # Prefer late body (axe approach) then mid; stride grows for speed.
    for lo, hi, stride in (
        (best_end - 80, best_end, 1),
        (2000, best_end - 80, 2),
        (1000, 2000, 4),
    ):
        i = max(0, lo)
        while i < min(hi, len(best_fx)) and len(moves) < max_deletes:
            cand = best_fx[:i] + best_fx[i + 1 :]
            set_state(env, state84)
            r = eval_ending(env, cand)
            if r.get("ok") and int(r["ending"]) < best_end:
                if verbose:
                    print(f"[FX DEL] @{i} → {r['ending']}", flush=True)
                best_fx = cand
                best_end = int(r["ending"])
                moves.append({"at": i, "ending": best_end})
                continue
            i += stride

    env.close()
    return {
        "success": True,
        "baseline_ending": int(base["ending"]),
        "best_ending": best_end,
        "frames_saved": int(base["ending"]) - best_end,
        "moves": moves,
        "to_8_4_control": r83["frames"],
        "projected_total": 15_370 - 2_227 + int(r83["frames"]) + best_end
        if False
        else None,
        # prefix_to_8_4_control is 15370 with 2227 bridge; recompute if body changes.
        "note": "full hybrid total = (prefix_without_83_bridge) + 83_frames + fx_ending",
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--body", type=Path, default=DEFAULT_BODY)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--windows",
        type=str,
        default="body",
        help="comma list: lead,early,mid,late,body,all",
    )
    p.add_argument("--delete-stride", type=int, default=3)
    p.add_argument("--no-delete", action="store_true")
    p.add_argument("--no-hold", action="store_true")
    p.add_argument(
        "--probe-fx",
        action="store_true",
        help="after 8-3, try flamexx single-frame deletes for sub-5",
    )
    p.add_argument("--report", type=Path, default=None)
    args = p.parse_args(argv)

    if args.probe_fx:
        rep = probe_fx_trim(body_83=args.body, verbose=True)
        out = args.report or (RECORDINGS_DIR / "tas_import" / "polish_8_3_fx_probe.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(rep, indent=2), flush=True)
        return 0 if rep.get("success") else 1

    report = polish(
        body_path=args.body,
        windows=[w.strip() for w in args.windows.split(",") if w.strip()],
        delete_stride=args.delete_stride,
        do_delete=not args.no_delete,
        do_hold=not args.no_hold,
        out_path=args.out,
        verbose=True,
    )
    out = args.report or (RECORDINGS_DIR / "tas_import" / "polish_8_3_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    print(
        f"done baseline={report.baseline} best={report.best} "
        f"saved={report.frames_saved} in {report.elapsed_s:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
