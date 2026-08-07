"""Hill-climb Super Mario Bros. 8-1 from Level8_1 (natural_82 body).

Extracts the control-relative 8-1 body from ``smb_1_1_to_ending_natural_82``
(after the post-4-2 lead idle), evaluates with the ``smb_8_1`` LevelConfig /
``Level8_1`` practice state, then runs structured local search:

- single-frame deletion sweep
- A/B edge shifts
- long-hold trims

Baseline isolated clear is ~3444f (flag ~2951). Trailing flagpole/tally is
mostly auto — optimizable region is the pre-flag body.

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
  uv run python -m smb.scripts.polish_8_1 --windows early,mid,late --delete-stride 2

# Full body delete (slow) + hold/edge
uv run python -m smb.scripts.polish_8_1 --windows body --delete-stride 1
```

Does **not** retime 8-2+ or re-fold the continuous seed by default. Writes
``models/smb_8_1_control_best.json`` when improved.
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

import smb.platformer_levels  # noqa: F401 — register LevelConfigs

from retro_harness.platformer.evaluator import Evaluator
from retro_harness.platformer.frame_tools import (
    clone_frames,
    count_leading_idle,
    count_trailing_idle,
)
from retro_harness.platformer.level_config import get_level_config
from smb.paths import MODELS_DIR, RECORDINGS_DIR
from smb.policy import compress_nes9_rle, expand_nes9_rle, load_nes9_rle_seed

DEFAULT_SOURCE = MODELS_DIR / "smb_1_1_to_ending_natural_82.json"
# Natural path milestones (STATUS / natural_82_retime_report).
NAT_81_START = 8_962  # 4-2 exit → 8-1 segment
NAT_81_END = 12_628  # 8-1 exit (level change / successor entry)

# Body-relative windows (after lead-idle strip). Pre-flag body ≈ 0..2951.
WINDOW_PRESETS: dict[str, tuple[int, int]] = {
    "early": (0, 800),
    "mid": (800, 1600),
    "mid2": (1600, 2400),
    "late": (2400, 3000),  # flag approach
    "body": (0, 3000),  # full pre-flag-ish
    "all": (0, 10_000),  # entire seed incl. tally (rarely useful)
}


@dataclass
class PolishReport:
    baseline_clear: int
    best_clear: int
    frames_saved: int
    baseline_len: int
    best_len: int
    lead_idle_stripped: int
    improvements: list[dict[str, Any]] = field(default_factory=list)
    out_path: str | None = None
    elapsed_s: float = 0.0
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_clear": self.baseline_clear,
            "best_clear": self.best_clear,
            "frames_saved": self.frames_saved,
            "baseline_len": self.baseline_len,
            "best_len": self.best_len,
            "lead_idle_stripped": self.lead_idle_stripped,
            "improvements": self.improvements,
            "out_path": self.out_path,
            "elapsed_s": self.elapsed_s,
            "source": self.source,
        }


def _pad12(frames: Sequence[Sequence[int]]) -> list[list[int]]:
    out: list[list[int]] = []
    for f in frames:
        b = [int(x) for x in f[:12]]
        if len(b) < 12:
            b.extend([0] * (12 - len(b)))
        out.append(b)
    return out


def extract_8_1_body(
    source: Path = DEFAULT_SOURCE,
    *,
    start: int = NAT_81_START,
    end: int = NAT_81_END,
) -> tuple[list[list[int]], int]:
    """Return (body_frames, lead_idle_count) from continuous natural seed."""
    frames = expand_nes9_rle(load_nes9_rle_seed(source))
    if not 0 <= start < end <= len(frames):
        raise ValueError(f"8-1 slice [{start}, {end}) outside {len(frames)} frames")
    seg = frames[start:end]
    lead = count_leading_idle(seg)
    body = _pad12(seg[lead:])
    return body, lead


def _clear(result) -> int | None:
    if not result.completed:
        return None
    return int(result.total_frames)


def _delete_sweep(
    frames: list[list[int]],
    evaluator: Evaluator,
    *,
    window: tuple[int, int],
    stride: int,
    baseline_clear: int,
    max_tries: int | None = None,
    verbose: bool = True,
) -> tuple[list[list[int]], int, list[dict[str, Any]]]:
    best = clone_frames(frames)
    best_clear = baseline_clear
    moves: list[dict[str, Any]] = []
    lo, hi = max(0, window[0]), min(len(best), window[1])
    i = lo
    tries = 0
    t0 = time.time()
    while i < min(hi, len(best)):
        if max_tries is not None and tries >= max_tries:
            break
        cand = best[:i] + best[i + 1 :]
        r = evaluator.evaluate(cand, early_terminate=False)
        tries += 1
        c = _clear(r)
        if c is not None and c < best_clear:
            if verbose:
                print(
                    f"[DEL] @{i} → clear {c} (−{best_clear - c}) len={len(cand)}",
                    flush=True,
                )
            best = cand
            best_clear = c
            moves.append({"op": "delete", "at": i, "clear": c})
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
    evaluator: Evaluator,
    *,
    window: tuple[int, int],
    buttons: Sequence[int] = (8, 0),  # A, B in nes12/nes9 layout used by seeds
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
                r = evaluator.evaluate(cand, early_terminate=False)
                c = _clear(r)
                if c is not None and c < best_clear:
                    if verbose:
                        print(
                            f"[EDGE] btn={btn} edge={edge} shift={shift} → {c}",
                            flush=True,
                        )
                    best = cand
                    best_clear = c
                    moves.append(
                        {
                            "op": "edge",
                            "button": btn,
                            "edge": edge,
                            "shift": shift,
                            "clear": c,
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
    evaluator: Evaluator,
    *,
    window: tuple[int, int],
    min_hold: int = 12,
    max_trim: int = 8,
    baseline_clear: int,
    verbose: bool = True,
) -> tuple[list[list[int]], int, list[dict[str, Any]]]:
    best = clone_frames(frames)
    best_clear = baseline_clear
    moves: list[dict[str, Any]] = []
    lo, hi = max(0, window[0]), min(len(best), window[1])
    t0 = time.time()
    i = lo
    while i < min(hi, len(best)):
        j = i + 1
        while j < len(best) and best[j] == best[i]:
            j += 1
        hold = j - i
        if hold >= min_hold and i < hi:
            improved = False
            for trim in range(1, min(max_trim, hold - 1) + 1):
                cand = best[: i + (hold - trim)] + best[j:]
                r = evaluator.evaluate(cand, early_terminate=False)
                c = _clear(r)
                if c is not None and c < best_clear:
                    if verbose:
                        print(
                            f"[HOLD] @{i} hold={hold} trim={trim} → {c}",
                            flush=True,
                        )
                    best = cand
                    best_clear = c
                    moves.append(
                        {
                            "op": "hold_trim",
                            "at": i,
                            "hold": hold,
                            "trim": trim,
                            "clear": c,
                        }
                    )
                    hi = min(hi, len(best))
                    i = lo
                    improved = True
                    break
            if improved:
                continue
        i = j if hold >= min_hold else i + 1
    if verbose:
        print(
            f"[HOLD] imps={len(moves)} clear={best_clear} "
            f"in {time.time() - t0:.1f}s",
            flush=True,
        )
    return best, best_clear, moves


def save_body_seed(
    path: Path,
    frames: Sequence[Sequence[int]],
    *,
    baseline_clear: int,
    best_clear: int,
    source: Path,
    lead_idle_stripped: int,
) -> Path:
    nes9 = [[int(x) for x in f[:9]] for f in frames]
    payload = {
        "format": "nes9_rle",
        "route_id": "smb_8_1_control",
        "level_id": "smb_8_1",
        "start_state": "Level8_1",
        "settle_frames": 0,
        "game_name": "SuperMarioBros-Nes-v0",
        "num_frames": len(nes9),
        "verified_completed": True,
        "target": "8-1_clear",
        "source": (
            f"natural_82 8-1 body after {lead_idle_stripped}f lead idle strip; "
            f"polish {baseline_clear}→{best_clear}"
        ),
        "source_seed": str(source),
        "natural_slice": {
            "start": NAT_81_START,
            "end": NAT_81_END,
            "lead_idle_stripped": lead_idle_stripped,
        },
        "optimization": {
            "tool": "smb.scripts.polish_8_1",
            "baseline_clear": baseline_clear,
            "best_clear": best_clear,
            "frames_saved": max(0, baseline_clear - best_clear),
        },
        "segments": compress_nes9_rle(nes9),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def polish(
    *,
    source: Path = DEFAULT_SOURCE,
    windows: Sequence[str] = ("early", "mid", "mid2", "late"),
    delete_stride: int = 2,
    do_delete: bool = True,
    do_edge: bool = True,
    do_hold: bool = True,
    max_delete_tries: int | None = None,
    out: Path | None = None,
    verbose: bool = True,
) -> tuple[list[list[int]], PolishReport]:
    body, lead = extract_8_1_body(source)
    cfg = get_level_config("smb_8_1")
    evaluator = Evaluator(cfg)

    base = evaluator.evaluate(body, early_terminate=False)
    base_clear = _clear(base)
    if base_clear is None:
        raise RuntimeError(
            f"baseline 8-1 body does not complete (max_x={base.max_x}, "
            f"frames={base.total_frames})"
        )
    if verbose:
        print(
            f"[8-1] baseline clear={base_clear} len={len(body)} "
            f"lead_stripped={lead} trailing_idle≈{count_trailing_idle(body)}",
            flush=True,
        )

    best = clone_frames(body)
    best_clear = base_clear
    all_moves: list[dict[str, Any]] = []
    t0 = time.time()

    resolved: list[tuple[str, tuple[int, int]]] = []
    for name in windows:
        if name in WINDOW_PRESETS:
            resolved.append((name, WINDOW_PRESETS[name]))
        elif ":" in name:
            a, b = name.split(":", 1)
            resolved.append((name, (int(a), int(b))))
        else:
            raise SystemExit(
                f"unknown window {name!r}; known: {', '.join(WINDOW_PRESETS)} "
                f"or start:end"
            )

    for name, win in resolved:
        if verbose:
            print(f"\n=== window {name} {win} ===", flush=True)
        if do_delete:
            best, best_clear, moves = _delete_sweep(
                best,
                evaluator,
                window=win,
                stride=delete_stride,
                baseline_clear=best_clear,
                max_tries=max_delete_tries,
                verbose=verbose,
            )
            all_moves.extend({"window": name, **m} for m in moves)
        if do_hold:
            best, best_clear, moves = _hold_trim(
                best,
                evaluator,
                window=win,
                baseline_clear=best_clear,
                verbose=verbose,
            )
            all_moves.extend({"window": name, **m} for m in moves)
        if do_edge:
            best, best_clear, moves = _edge_shift(
                best,
                evaluator,
                window=win,
                baseline_clear=best_clear,
                verbose=verbose,
            )
            all_moves.extend({"window": name, **m} for m in moves)

    # Final verify
    final = evaluator.evaluate(best, early_terminate=False)
    final_clear = _clear(final)
    if final_clear is None:
        raise RuntimeError("post-polish seed failed to complete")
    best_clear = final_clear

    out_path = out or (MODELS_DIR / "smb_8_1_control_best.json")
    if best_clear < base_clear or not out_path.exists():
        save_body_seed(
            out_path,
            best,
            baseline_clear=base_clear,
            best_clear=best_clear,
            source=source,
            lead_idle_stripped=lead,
        )
        if verbose:
            print(f"[8-1] wrote {out_path}", flush=True)
    elif verbose:
        print(f"[8-1] no improvement; left {out_path} unchanged", flush=True)

    report = PolishReport(
        baseline_clear=base_clear,
        best_clear=best_clear,
        frames_saved=max(0, base_clear - best_clear),
        baseline_len=len(body),
        best_len=len(best),
        lead_idle_stripped=lead,
        improvements=all_moves,
        out_path=str(out_path) if best_clear <= base_clear else None,
        elapsed_s=time.time() - t0,
        source=str(source),
    )
    rep_dir = RECORDINGS_DIR / "segment_8_1"
    rep_dir.mkdir(parents=True, exist_ok=True)
    rep_path = rep_dir / "polish_8_1_report.json"
    rep_path.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    if verbose:
        print(
            f"[8-1] done clear {base_clear}→{best_clear} "
            f"(−{report.frames_saved}) imps={len(all_moves)} "
            f"in {report.elapsed_s:.1f}s; report={rep_path}",
            flush=True,
        )
    return best, report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    p.add_argument(
        "--windows",
        default="early,mid,mid2,late",
        help=f"comma list of {','.join(WINDOW_PRESETS)} or start:end",
    )
    p.add_argument("--delete-stride", type=int, default=2)
    p.add_argument("--max-delete-tries", type=int, default=None)
    p.add_argument("--no-delete", action="store_true")
    p.add_argument("--no-edge", action="store_true")
    p.add_argument("--no-hold", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--list-windows", action="store_true")
    p.add_argument("--baseline-only", action="store_true")
    args = p.parse_args(argv)

    if args.list_windows:
        for name, (a, b) in WINDOW_PRESETS.items():
            print(f"{name:8s}  [{a}, {b})")
        return 0

    if args.baseline_only:
        body, lead = extract_8_1_body(args.source)
        cfg = get_level_config("smb_8_1")
        r = Evaluator(cfg).evaluate(body, early_terminate=False)
        print(
            json.dumps(
                {
                    "lead_idle_stripped": lead,
                    "body_len": len(body),
                    "completed": r.completed,
                    "clear": _clear(r),
                    "max_x": r.max_x,
                },
                indent=2,
            )
        )
        return 0 if r.completed else 1

    windows = [w.strip() for w in args.windows.split(",") if w.strip()]
    _best, report = polish(
        source=args.source,
        windows=windows,
        delete_stride=args.delete_stride,
        do_delete=not args.no_delete,
        do_edge=not args.no_edge,
        do_hold=not args.no_hold,
        max_delete_tries=args.max_delete_tries,
        out=args.out,
        verbose=True,
    )
    print(json.dumps(report.to_dict(), indent=2)[-2000:])
    return 0 if report.best_clear <= report.baseline_clear else 1


if __name__ == "__main__":
    sys.exit(main())
