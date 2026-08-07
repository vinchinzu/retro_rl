"""Rewrite the late 1-2 UG suffix for a clean World-4 pipe top-land.

Replaces the messy reverse → face-slam → floor → recovery tail of
``underground_from_control`` with a structured approach:

1. Keep the control-relative UG prefix through the ceiling run.
2. Reverse on the ceiling (B coast + BL).
3. Short BLA jump off the right platform into the warp room.
4. Hold DOWN so the first W4 pipe-top contact enters World 4.

No floor bounce and no face-slam at x≈2830. Checkpoints at underground
control (reuses :class:`UndergroundCheckpoint`).

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
  uv run python -m smb.scripts.polish_1_2_warp_pipe
uv run python -m smb.scripts.run_1_2 --predecessor stairs --trials 2
```
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from smb.paths import MODELS_DIR, RECORDINGS_DIR
from smb.ram import read_snapshot, segment_1_2_warp_success
from smb.reactive_12 import DEFAULT_FRAGMENTS, underground_frames
from smb.scripts.polish_1_2_ug import (
    UndergroundCheckpoint,
    save_fragments,
    verify_isolated,
    verify_natural_stairs,
)

# NES-9: B=0, D=5, L=6, R=7, A=8 (matches polish_1_2_ug edge comments).
B = [1, 0, 0, 0, 0, 0, 0, 0, 0]
D = [0, 0, 0, 0, 0, 1, 0, 0, 0]
L = [0, 0, 0, 0, 0, 0, 1, 0, 0]
BL = [1, 0, 0, 0, 0, 0, 1, 0, 0]
BLA = [1, 0, 0, 0, 0, 0, 1, 0, 1]
IDLE = [0, 0, 0, 0, 0, 0, 0, 0, 0]

# Ceiling BR run ends; reverse starts here in the polished 1507f fragment.
DEFAULT_SUFFIX_START = 1344


def _mk(seq: Sequence[tuple[list[int], int]]) -> list[list[int]]:
    out: list[list[int]] = []
    for btn, n in seq:
        out.extend([list(btn) for _ in range(n)])
    return out


def _quality(
    cp: UndergroundCheckpoint,
    frames: Sequence[Sequence[int]],
    *,
    pad: int = 20,
) -> dict[str, Any]:
    """Replay full UG from control; track floor / face-slam / W4."""
    cp.env.em.set_state(cp._ug_state)
    lives0 = cp._start_lives
    floor = False
    slam = False
    pipe_lands: list[tuple[int, int, int, int]] = []
    for i in range(len(frames) + pad):
        if i < len(frames):
            fr = list(frames[i][:9]) + [0] * 9
            act = np.array(fr[: cp.action_size], dtype=np.int8)
            if len(act) < cp.action_size:
                act = np.concatenate(
                    [act, np.zeros(cp.action_size - len(act), dtype=np.int8)]
                )
        else:
            act = np.zeros(cp.action_size, dtype=np.int8)
        cp.env.step(act)
        s = read_snapshot(cp.env.get_ram(), frame=i + 1)
        if s.player_x > 2700:
            if s.grounded and s.player_y >= 160:
                floor = True
            if s.x_speed == 0 and 120 <= s.player_y <= 150 and not s.grounded:
                slam = True
            if s.grounded and 100 <= s.player_y <= 135:
                pipe_lands.append((i + 1, s.player_x, s.player_y, s.x_speed))
        if s.lives < lives0 or s.dying:
            return {
                "ok": False,
                "frames": i + 1,
                "floor": floor,
                "slam": slam,
                "died": True,
                "pipe_lands": pipe_lands,
            }
        if segment_1_2_warp_success(cp.env.get_ram(), start_lives=lives0):
            return {
                "ok": True,
                "frames": i + 1,
                "floor": floor,
                "slam": slam,
                "died": False,
                "x": s.player_x,
                "y": s.player_y,
                "xs": s.x_speed,
                "world": s.world,
                "level_id": s.level_id,
                "pipe_lands": pipe_lands,
            }
    return {
        "ok": False,
        "frames": len(frames) + pad,
        "floor": floor,
        "slam": slam,
        "died": False,
        "pipe_lands": pipe_lands,
    }


def build_suffix(
    *,
    coast: int = 5,
    bl_n: int = 52,
    jump_n: int = 20,
    idle_n: int = 5,
    down_n: int = 25,
    margin_d: int = 2,
) -> list[list[int]]:
    """Hand-tuned reverse → short jump → DOWN enter suffix."""
    seq: list[tuple[list[int], int]] = []
    if coast:
        seq.append((B, coast))
    seq.append((BL, bl_n))
    if jump_n:
        seq.append((BLA, jump_n))
    if idle_n:
        seq.append((IDLE, idle_n))
    seq.append((D, down_n + margin_d))
    return _mk(seq)


def search_suffix(
    cp: UndergroundCheckpoint,
    prefix: list[list[int]],
    *,
    verbose: bool = True,
) -> tuple[list[list[int]], dict[str, Any]]:
    """Small grid around the known-good reverse/jump/DOWN recipe."""
    best_frames: list[list[int]] | None = None
    best_r: dict[str, Any] | None = None
    tried = 0
    for jump_n in (16, 18, 20, 22, 24):
        for idle_n in (0, 2, 3, 5, 8):
            for down_n in (18, 22, 25, 30):
                tried += 1
                suf = build_suffix(
                    coast=5, bl_n=52, jump_n=jump_n, idle_n=idle_n, down_n=down_n, margin_d=0
                )
                frames = prefix + suf
                r = _quality(cp, frames)
                if not (r["ok"] and not r["floor"] and not r["slam"]):
                    continue
                if best_r is None or r["frames"] < best_r["frames"]:
                    # Trim to clear + small DOWN margin for phase safety.
                    clear_i = r["frames"]  # 1-based clear index
                    trimmed = [list(f[:9]) for f in frames[:clear_i]] + [
                        list(D),
                        list(D),
                    ]
                    r2 = _quality(cp, trimmed)
                    if r2["ok"] and not r2["floor"] and not r2["slam"]:
                        best_frames = trimmed
                        best_r = {
                            **r2,
                            "recipe": {
                                "coast": 5,
                                "bl_n": 52,
                                "jump_n": jump_n,
                                "idle_n": idle_n,
                                "down_n": down_n,
                            },
                            "tried": tried,
                        }
                        if verbose:
                            print(
                                f"[warp-pipe] candidate clear={r2['frames']} "
                                f"j={jump_n} idle={idle_n} d={down_n} "
                                f"enter=({r2.get('x')},{r2.get('y')}) "
                                f"xs={r2.get('xs')}",
                                flush=True,
                            )
    if best_frames is None or best_r is None:
        raise RuntimeError("no clean W4 top-land suffix found in search grid")
    if verbose:
        print(
            f"[warp-pipe] best clear={best_r['frames']} "
            f"recipe={best_r['recipe']} tried={tried}",
            flush=True,
        )
    return best_frames, best_r


def optimize(
    *,
    fragments_in: Path = DEFAULT_FRAGMENTS,
    fragments_out: Path | None = None,
    suffix_start: int = DEFAULT_SUFFIX_START,
    verify_natural: bool = True,
    write: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    t0 = time.time()
    ug = underground_frames(fragments_in)
    if suffix_start >= len(ug) - 20:
        raise ValueError(f"suffix_start={suffix_start} too late for len={len(ug)}")
    prefix = [list(f[:9]) for f in ug[:suffix_start]]
    if verbose:
        print(
            f"[warp-pipe] load {fragments_in.name} ug={len(ug)} "
            f"suffix_start={suffix_start}",
            flush=True,
        )

    cp = UndergroundCheckpoint()
    try:
        base = _quality(cp, ug)
        if verbose:
            print(
                f"[warp-pipe] baseline clear={base.get('frames')} "
                f"floor={base.get('floor')} slam={base.get('slam')} ok={base.get('ok')}",
                flush=True,
            )
        best_frames, best_r = search_suffix(cp, prefix, verbose=verbose)
    finally:
        cp.close()

    out = fragments_out or (MODELS_DIR / "smb_1_2_reactive_fragments.json")
    report: dict[str, Any] = {
        "baseline": base,
        "best": {k: v for k, v in best_r.items() if k != "pipe_lands"},
        "best_pipe_lands": best_r.get("pipe_lands"),
        "ug_len": len(best_frames),
        "suffix_start": suffix_start,
        "elapsed_s": 0.0,
    }

    if write:
        base_clear = int(base["frames"]) if base.get("ok") else len(ug)
        save_fragments(
            out,
            best_frames,
            baseline_ug=base_clear,
            best_ug=int(best_r["frames"]),
            source=fragments_in,
        )
        data = json.loads(out.read_text(encoding="utf-8"))
        data["underground_from_control"]["note"] = (
            f"warp pipe top-land: ceiling reverse, short BLA jump, DOWN on W4 lip; "
            f"{base_clear}→{best_r['frames']} clean (no floor/face-slam)"
        )
        data["optimization"] = {
            "tool": "smb.scripts.polish_1_2_warp_pipe",
            "baseline_ug_clear": base_clear,
            "best_ug_clear": best_r["frames"],
            "frames_saved": max(0, base_clear - int(best_r["frames"])),
            "geometry": "ceiling reverse → right platform → W4 pipe-top DOWN",
            "no_floor_touch": True,
            "no_face_slam": True,
            "suffix_from_ug": suffix_start,
            "suffix_recipe": best_r.get("recipe"),
        }
        out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        report["out_path"] = str(out)
        if verbose:
            print(f"[warp-pipe] wrote {out} ug_len={len(best_frames)}", flush=True)

        iso = verify_isolated(out)
        report["isolated_verify"] = {
            "success": iso.get("success"),
            "frames": iso.get("frames"),
            "final": iso.get("final"),
            "log": iso.get("log"),
        }
        if verbose:
            print(
                f"[warp-pipe] isolated: success={iso.get('success')} "
                f"frames={iso.get('frames')}",
                flush=True,
            )
        if verify_natural:
            nat = verify_natural_stairs(out)
            report["natural_verify"] = nat
            if verbose:
                print(
                    f"[warp-pipe] natural: success={nat.get('success')} "
                    f"1-1={nat.get('frames_1_1')} 1-2={nat.get('frames_1_2')} "
                    f"total={nat.get('total')}",
                    flush=True,
                )

    report["elapsed_s"] = time.time() - t0
    rep_dir = RECORDINGS_DIR / "segment_1_2"
    rep_dir.mkdir(parents=True, exist_ok=True)
    (rep_dir / "polish_1_2_warp_pipe_report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    if verbose:
        print(f"[warp-pipe] DONE in {report['elapsed_s']:.1f}s", flush=True)
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fragments", type=Path, default=DEFAULT_FRAGMENTS)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument(
        "--suffix-start",
        type=int,
        default=DEFAULT_SUFFIX_START,
        help="UG fragment index where the reverse/jump suffix begins",
    )
    p.add_argument("--no-natural", action="store_true")
    p.add_argument("--no-write", action="store_true")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)
    optimize(
        fragments_in=args.fragments,
        fragments_out=args.out,
        suffix_start=args.suffix_start,
        verify_natural=not args.no_natural,
        write=not args.no_write,
        verbose=not args.quiet,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
