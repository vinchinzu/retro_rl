"""Polish the control-relative 1-2 underground fragment (W4 warp).

Starts at published ``Level1_2`` (surface control), plays reactive surface +
pipe load once, caches emulator state at underground control, then searches
over the underground RLE only. Success = World 4 without a lives drop.

Writes an updated ``underground_from_control`` into the reactive fragments
file when improved. Always re-verify with full reactive policy (isolated
Level1_2 and optional natural stairs predecessor).

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
  uv run python -m smb.scripts.polish_1_2_ug --delete-stride 1 --windows lead,mid,slam
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

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from retro_harness.env import make_env
from retro_harness.platformer.frame_tools import clone_frames
from smb.paths import GAME_DIR, GAME_V0, MODELS_DIR, RECORDINGS_DIR
from smb.policy import compress_nes9_rle, expand_nes9_rle
from smb.ram import read_snapshot, segment_1_2_warp_success
from smb.reactive_12 import (
    DEFAULT_FRAGMENTS,
    Reactive12Policy,
    is_underground_control,
    load_reactive_fragments,
    play_reactive_12,
    underground_frames,
)

# Named windows relative to underground control (fragment index).
# Derived from Level1_2 reactive trace: UG starts at absolute f=310.
WINDOW_PRESETS: dict[str, tuple[int, int]] = {
    # Leading idle + first accel (baseline: 34 idle then B+RIGHT)
    "lead": (0, 120),
    # Stall ~x=909 (absolute ~728–759 → ug ~418–449)
    "mid": (380, 480),
    # Wall-slam ~x=1635 (absolute ~1073 → ug ~763)
    "slam": (720, 850),
    # Late approach into warp vine/pipe (before long settle)
    "warp_approach": (1200, 1420),
    # Entire controllable body (skip last ~120f warp settle)
    "body": (0, 1420),
}


@dataclass
class UgEval:
    completed: bool
    frames: int
    max_x: int
    died: bool
    world: int = 0
    level_id: int = 0


@dataclass
class PolishReport:
    baseline_frames: int
    best_frames: int
    frames_saved: int
    improvements: list[dict[str, Any]] = field(default_factory=list)
    isolated_verify: dict[str, Any] | None = None
    natural_verify: dict[str, Any] | None = None
    out_path: str | None = None
    elapsed_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_frames": self.baseline_frames,
            "best_frames": self.best_frames,
            "frames_saved": self.frames_saved,
            "improvements": self.improvements,
            "isolated_verify": self.isolated_verify,
            "natural_verify": self.natural_verify,
            "out_path": self.out_path,
            "elapsed_s": self.elapsed_s,
        }


class UndergroundCheckpoint:
    """One Level1_2 env + state at underground control for fast UG evals."""

    def __init__(self) -> None:
        self.env = make_env(GAME_V0, "Level1_2", GAME_DIR, render_mode="rgb_array")
        self.env.reset()
        self.action_size = int(self.env.action_space.shape[0])
        self._ug_state: Any | None = None
        self._start_lives = 2
        self.surface_frames = 0
        self._build()

    def _idle(self) -> np.ndarray:
        return np.zeros(self.action_size, dtype=np.int8)

    def _nes9(self, buttons: Sequence[int]) -> np.ndarray:
        b = list(buttons[: self.action_size])
        if len(b) < self.action_size:
            b.extend([0] * (self.action_size - len(b)))
        return np.array(b, dtype=np.int8)

    def _build(self) -> None:
        """Play reactive surface + wait until underground control; cache state."""
        pol = Reactive12Policy(action_size=self.action_size)
        pol.reset()
        # Force through surface only — stop at first underground control tick.
        for step in range(800):
            snap = read_snapshot(self.env.get_ram(), frame=step)
            if is_underground_control(snap) and pol.phase.name in (
                "WAIT_UNDERGROUND",
                "UNDERGROUND",
            ):
                # First controllable UG frame is about to be consumed by policy;
                # cache *before* that action so UG fragment index 0 matches
                # Reactive12Policy's first underground_from_control frame.
                if pol.phase.name == "WAIT_UNDERGROUND":
                    # Let waiter observe so phase becomes UNDERGROUND, but we
                    # want state before the first UG macro frame.
                    pass
                # Re-read: if we're at control and still WAIT_UNDERGROUND, the
                # next pol.step would emit the first UG action. Cache now.
                if is_underground_control(snap):
                    # Advance policy once without env step to sync phase if needed
                    # Actually: on first control frame, pol.step emits first UG
                    # action. Cache pre-step state, then return.
                    self._ug_state = self.env.em.get_state()
                    self._start_lives = snap.lives
                    self.surface_frames = step
                    return
            tick = pol.step(snap)
            # Stop recording once we would enter underground playback
            if pol.phase.name == "UNDERGROUND" and pol.ug_index == 1:
                # We already stepped the first UG frame — rebuild carefully
                break
            self.env.step(tick.action)
        raise RuntimeError("failed to reach underground control from Level1_2")

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass

    def evaluate(self, ug_frames: Sequence[Sequence[int]], *, pad: int = 40) -> UgEval:
        """Replay *ug_frames* from underground control; success = World 4."""
        assert self._ug_state is not None
        self.env.em.set_state(self._ug_state)
        lives0 = self._start_lives
        max_x = 0
        n = len(ug_frames)
        # Extra idle pad so a slightly shorter clear still registers W4.
        total = n + pad
        for i in range(total):
            if i < n:
                act = self._nes9(ug_frames[i])
            else:
                act = self._idle()
            self.env.step(act)
            ram = self.env.get_ram()
            snap = read_snapshot(ram, frame=i + 1)
            max_x = max(max_x, snap.player_x)
            if snap.lives < lives0 or snap.dying:
                return UgEval(
                    completed=False,
                    frames=i + 1,
                    max_x=max_x,
                    died=True,
                    world=snap.world,
                    level_id=snap.level_id,
                )
            if segment_1_2_warp_success(ram, start_lives=lives0):
                return UgEval(
                    completed=True,
                    frames=i + 1,
                    max_x=max_x,
                    died=False,
                    world=snap.world,
                    level_id=snap.level_id,
                )
        snap = read_snapshot(self.env.get_ram())
        return UgEval(
            completed=False,
            frames=total,
            max_x=max_x,
            died=False,
            world=snap.world,
            level_id=snap.level_id,
        )


def _delete_sweep(
    frames: list[list[int]],
    cp: UndergroundCheckpoint,
    *,
    window: tuple[int, int],
    stride: int,
    baseline_clear: int,
    verbose: bool = True,
) -> tuple[list[list[int]], int, list[dict[str, Any]]]:
    best = clone_frames(frames)
    best_clear = baseline_clear
    moves: list[dict[str, Any]] = []
    lo, hi = window
    lo = max(0, lo)
    hi = min(len(best), hi)
    i = lo
    tries = 0
    t0 = time.time()
    while i < min(hi, len(best)):
        cand = best[:i] + best[i + 1 :]
        r = cp.evaluate(cand)
        tries += 1
        if r.completed and r.frames < best_clear:
            if verbose:
                print(
                    f"[DEL] @{i} → ug_clear {r.frames} "
                    f"(−{best_clear - r.frames}) len={len(cand)}",
                    flush=True,
                )
            best = cand
            best_clear = r.frames
            moves.append({"op": "delete", "at": i, "clear": r.frames})
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


def _edge_shift(
    frames: list[list[int]],
    cp: UndergroundCheckpoint,
    *,
    window: tuple[int, int],
    buttons: Sequence[int] = (8, 0),  # A, B
    shifts: Sequence[int] = (-3, -2, -1, 1, 2, 3),
    baseline_clear: int,
    verbose: bool = True,
) -> tuple[list[list[int]], int, list[dict[str, Any]]]:
    best = clone_frames(frames)
    best_clear = baseline_clear
    moves: list[dict[str, Any]] = []
    lo = max(1, window[0])
    hi = min(len(best), window[1])
    t0 = time.time()
    for btn in buttons:
        edges = [
            i
            for i in range(lo, hi)
            if i < len(best) and best[i][btn] != best[i - 1][btn]
        ]
        if verbose:
            print(f"[EDGE] btn={btn} edges={len(edges)}", flush=True)
        for edge in edges:
            for shift in shifts:
                cand = clone_frames(best)
                new_e = edge + shift
                if new_e <= 0 or new_e >= len(cand):
                    continue
                val = cand[edge][btn]
                prev = cand[edge - 1][btn]
                if val == prev:
                    continue
                if shift > 0:
                    for j in range(edge, min(new_e, len(cand))):
                        cand[j][btn] = prev
                else:
                    for j in range(new_e, edge):
                        cand[j][btn] = val
                r = cp.evaluate(cand)
                if r.completed and r.frames < best_clear:
                    if verbose:
                        print(
                            f"[EDGE] btn={btn} edge={edge} shift={shift} "
                            f"→ {r.frames}",
                            flush=True,
                        )
                    best = cand
                    best_clear = r.frames
                    moves.append(
                        {
                            "op": "edge",
                            "button": btn,
                            "edge": edge,
                            "shift": shift,
                            "clear": r.frames,
                        }
                    )
                    break
    if verbose:
        print(
            f"[EDGE] imps={len(moves)} clear={best_clear} "
            f"in {time.time() - t0:.1f}s",
            flush=True,
        )
    return best, best_clear, moves


def _hold_trim(
    frames: list[list[int]],
    cp: UndergroundCheckpoint,
    *,
    min_hold: int = 12,
    max_trim: int = 8,
    baseline_clear: int,
    verbose: bool = True,
) -> tuple[list[list[int]], int, list[dict[str, Any]]]:
    """Try shortening long identical button holds by 1..max_trim frames."""
    best = clone_frames(frames)
    best_clear = baseline_clear
    moves: list[dict[str, Any]] = []
    t0 = time.time()
    i = 0
    while i < len(best):
        j = i + 1
        while j < len(best) and best[j] == best[i]:
            j += 1
        hold = j - i
        if hold >= min_hold:
            for trim in range(1, min(max_trim, hold - 1) + 1):
                cand = best[: i + (hold - trim)] + best[j:]
                r = cp.evaluate(cand)
                if r.completed and r.frames < best_clear:
                    if verbose:
                        print(
                            f"[HOLD] @{i} hold={hold} trim={trim} → {r.frames}",
                            flush=True,
                        )
                    best = cand
                    best_clear = r.frames
                    moves.append(
                        {
                            "op": "hold_trim",
                            "at": i,
                            "hold": hold,
                            "trim": trim,
                            "clear": r.frames,
                        }
                    )
                    # Restart scan after mutation
                    i = 0
                    break
            else:
                i = j
                continue
            continue
        i = j
    if verbose:
        print(
            f"[HOLD] imps={len(moves)} clear={best_clear} "
            f"in {time.time() - t0:.1f}s",
            flush=True,
        )
    return best, best_clear, moves


def save_fragments(
    path: Path,
    ug_frames: Sequence[Sequence[int]],
    *,
    baseline_ug: int,
    best_ug: int,
    source: Path,
) -> Path:
    data = load_reactive_fragments(source)
    nes9 = [list(f[:9]) + [0] * max(0, 9 - len(f)) for f in ug_frames]
    nes9 = [f[:9] for f in nes9]
    data["underground_from_control"] = {
        "num_frames": len(nes9),
        "segments": compress_nes9_rle(nes9),
        "note": (
            f"control-relative UG polish: {baseline_ug}→{best_ug} "
            f"(−{baseline_ug - best_ug}f vs prior fragment eval)"
        ),
    }
    data["optimization"] = {
        "tool": "smb.scripts.polish_1_2_ug",
        "baseline_ug_clear": baseline_ug,
        "best_ug_clear": best_ug,
        "frames_saved": max(0, baseline_ug - best_ug),
        "source_fragments": str(source),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


def verify_isolated(fragments_path: Path) -> dict[str, Any]:
    env = make_env(GAME_V0, "Level1_2", GAME_DIR, render_mode="rgb_array")
    try:
        env.reset()
        pol = Reactive12Policy(
            fragments_path=fragments_path,
            action_size=int(env.action_space.shape[0]),
        )
        r = play_reactive_12(env, policy=pol, max_frames=4000)
        return {
            "success": r["success"],
            "frames": r["frames"],
            "log": r["policy"]["log"],
            "final": r["final"],
        }
    finally:
        env.close()


def verify_natural_stairs(fragments_path: Path) -> dict[str, Any]:
    """Level1_1 stairs seed → reactive 1-2 with given fragments."""
    from smb.full_run import read_state_bytes
    from smb.paths import INTEGRATION_V0_DIR
    from smb.policy import CONTINUOUS_SETTLE_FRAMES, frames_to_actions, load_nes9_rle_seed
    from smb.scripts.run_1_2 import STAIRS_1_1, _play_1_1_until_clear

    level1 = INTEGRATION_V0_DIR / "Level1_1.state"
    env = make_env(GAME_V0, "NONE", GAME_DIR, render_mode="rgb_array")
    try:
        env.reset()
        env.em.set_state(read_state_bytes(level1))
        idle = np.zeros(int(env.action_space.shape[0]), dtype=np.int8)
        for _ in range(CONTINUOUS_SETTLE_FRAMES):
            env.step(idle)
        seed_11 = expand_nes9_rle(load_nes9_rle_seed(STAIRS_1_1))
        s11 = _play_1_1_until_clear(env, seed_11)
        if not s11["success"]:
            return {"success": False, "stage": "1-1", "detail": s11}
        pol = Reactive12Policy(
            fragments_path=fragments_path,
            action_size=int(env.action_space.shape[0]),
        )
        s12 = play_reactive_12(env, policy=pol, max_frames=4000)
        return {
            "success": bool(s12["success"]),
            "frames_1_1": s11["frames"],
            "frames_1_2": s12["frames"],
            "total": s11["frames"] + s12["frames"],
            "log": s12["policy"]["log"],
            "final": s12.get("final"),
        }
    finally:
        env.close()


def optimize(
    *,
    fragments_in: Path = DEFAULT_FRAGMENTS,
    fragments_out: Path | None = None,
    window_names: Sequence[str] = ("lead", "mid", "slam", "body"),
    delete_stride: int = 1,
    do_edges: bool = True,
    do_holds: bool = True,
    verify_natural: bool = True,
    verbose: bool = True,
) -> tuple[list[list[int]], PolishReport]:
    t0 = time.time()
    ug = underground_frames(fragments_in)
    if verbose:
        print(f"[1-2-UG] load {fragments_in.name} ug_frames={len(ug)}", flush=True)

    cp = UndergroundCheckpoint()
    if verbose:
        print(
            f"[1-2-UG] checkpoint at surface_frames={cp.surface_frames} "
            f"(pre-UG control)",
            flush=True,
        )

    base = cp.evaluate(ug)
    if not base.completed:
        cp.close()
        raise RuntimeError(
            f"baseline UG fragment does not reach W4 "
            f"(frames={base.frames} max_x={base.max_x} died={base.died} "
            f"world={base.world} level_id={base.level_id})"
        )
    if verbose:
        print(
            f"[1-2-UG] baseline ug_clear={base.frames} max_x={base.max_x}",
            flush=True,
        )

    best = clone_frames(ug)
    best_clear = base.frames
    all_moves: list[dict[str, Any]] = []

    for name in window_names:
        if name not in WINDOW_PRESETS:
            raise SystemExit(f"unknown window {name!r}; known {list(WINDOW_PRESETS)}")
        w = WINDOW_PRESETS[name]
        # Clamp to current length (shrinks as we delete)
        w = (w[0], min(w[1], max(0, len(best) - 20)))
        if w[1] - w[0] < 8:
            continue
        if verbose:
            print(f"[1-2-UG] delete window {name} {w} stride={delete_stride}", flush=True)
        best, best_clear, moves = _delete_sweep(
            best,
            cp,
            window=w,
            stride=delete_stride,
            baseline_clear=best_clear,
            verbose=verbose,
        )
        for m in moves:
            m["window"] = name
        all_moves.extend(moves)

    if do_holds:
        if verbose:
            print("[1-2-UG] hold trim…", flush=True)
        best, best_clear, moves = _hold_trim(
            best, cp, baseline_clear=best_clear, verbose=verbose
        )
        all_moves.extend(moves)

    if do_edges:
        if verbose:
            print("[1-2-UG] edge shifts on body…", flush=True)
        w = (0, max(0, len(best) - 20))
        best, best_clear, moves = _edge_shift(
            best, cp, window=w, baseline_clear=best_clear, verbose=verbose
        )
        all_moves.extend(moves)

    # Final dense delete if we already found something (stride 1 full body)
    if delete_stride > 1 and all_moves:
        if verbose:
            print("[1-2-UG] dense body pass stride=1…", flush=True)
        best, best_clear, moves = _delete_sweep(
            best,
            cp,
            window=(0, max(0, len(best) - 20)),
            stride=1,
            baseline_clear=best_clear,
            verbose=verbose,
        )
        all_moves.extend(moves)

    cp.close()

    out = fragments_out or (MODELS_DIR / "smb_1_2_reactive_fragments.json")
    # Always write if improved; else write candidate under a different name
    if best_clear < base.frames:
        save_fragments(
            out,
            best,
            baseline_ug=base.frames,
            best_ug=best_clear,
            source=fragments_in,
        )
        verify_path = out
    else:
        # Keep baseline; still verify
        verify_path = fragments_in
        if verbose:
            print("[1-2-UG] no improvement; fragments unchanged", flush=True)

    iso = verify_isolated(verify_path)
    if verbose:
        print(
            f"[1-2-UG] isolated Level1_2: success={iso['success']} "
            f"frames={iso.get('frames')}",
            flush=True,
        )

    nat = None
    if verify_natural:
        # verify_isolated closes env; natural opens a new one
        nat = verify_natural_stairs(verify_path)
        if verbose:
            print(
                f"[1-2-UG] natural stairs: success={nat.get('success')} "
                f"1-1={nat.get('frames_1_1')} 1-2={nat.get('frames_1_2')} "
                f"total={nat.get('total')}",
                flush=True,
            )

    report = PolishReport(
        baseline_frames=base.frames,
        best_frames=best_clear,
        frames_saved=max(0, base.frames - best_clear),
        improvements=all_moves,
        isolated_verify=iso,
        natural_verify=nat,
        out_path=str(out) if best_clear < base.frames else None,
        elapsed_s=time.time() - t0,
    )
    rep_dir = RECORDINGS_DIR / "segment_1_2"
    rep_dir.mkdir(parents=True, exist_ok=True)
    (rep_dir / "polish_1_2_ug_report.json").write_text(
        json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    if verbose:
        print(
            f"[1-2-UG] DONE ug {base.frames}→{best_clear} "
            f"(−{report.frames_saved}f) imps={len(all_moves)} "
            f"in {report.elapsed_s:.1f}s",
            flush=True,
        )
    return best, report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--fragments",
        type=Path,
        default=DEFAULT_FRAGMENTS,
        help="input reactive fragments JSON",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output fragments path (default: overwrite models reactive)",
    )
    p.add_argument(
        "--windows",
        type=str,
        default="lead,mid,slam,body",
        help="comma-separated window names",
    )
    p.add_argument("--delete-stride", type=int, default=1)
    p.add_argument("--no-edges", action="store_true")
    p.add_argument("--no-holds", action="store_true")
    p.add_argument("--no-natural", action="store_true")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    names = [w.strip() for w in args.windows.split(",") if w.strip()]
    try:
        _best, report = optimize(
            fragments_in=args.fragments,
            fragments_out=args.out,
            window_names=names,
            delete_stride=args.delete_stride,
            do_edges=not args.no_edges,
            do_holds=not args.no_holds,
            verify_natural=not args.no_natural,
            verbose=not args.quiet,
        )
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(json.dumps(report.to_dict(), indent=2))
    return 0 if (report.isolated_verify or {}).get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
