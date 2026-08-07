"""Systematic local search operators for already-strong 1-1 seeds.

Random hill-climb often stalls on polished seeds. These operators try
structured, exhaustive moves:

- single-frame deletion sweep (stride-controllable)
- A/B button edge shifts (±N frames)
- multi-frame hold shrink at a given index
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from retro_harness.platformer.evaluator import EvalResult, Evaluator
from retro_harness.platformer.frame_tools import clone_frames


@dataclass
class SearchReport:
    improvements: int = 0
    baseline_clear: int | None = None
    best_clear: int | None = None
    moves: list[dict[str, Any]] = field(default_factory=list)
    elapsed_s: float = 0.0


def _clear(result: EvalResult) -> int | None:
    if not result.completed:
        return None
    return int(result.total_frames)


def systematic_delete_sweep(
    frames: list[list[int]],
    evaluator: Evaluator,
    *,
    window: tuple[int, int] | None = None,
    stride: int = 1,
    max_tries: int | None = None,
    verbose: bool = True,
) -> tuple[list[list[int]], EvalResult, SearchReport]:
    """Try deleting one frame at a time in *window*; keep every improvement."""
    best = clone_frames(frames)
    base = evaluator.evaluate(best, early_terminate=False)
    report = SearchReport(baseline_clear=_clear(base), best_clear=_clear(base))
    if not base.completed:
        return best, base, report

    lo, hi = (0, len(best)) if window is None else window
    lo = max(0, lo)
    hi = min(len(best), hi)
    t0 = time.time()
    i = lo
    tries = 0
    while i < min(hi, len(best)):
        if max_tries is not None and tries >= max_tries:
            break
        cand = best[:i] + best[i + 1 :]
        r = evaluator.evaluate(cand, early_terminate=False)
        tries += 1
        c = _clear(r)
        if c is not None and c < (report.best_clear or c + 1):
            if verbose:
                print(f"[DEL] @{i} → clear {c} (−{(report.best_clear or c) - c})")
            best = cand
            base = r
            report.best_clear = c
            report.improvements += 1
            report.moves.append({"op": "delete", "at": i, "clear": c})
            # New frame sits at i; shrink hi; don't advance.
            hi = min(hi, len(best))
            continue
        i += max(1, stride)
    report.elapsed_s = time.time() - t0
    if verbose:
        print(
            f"[DEL] done imps={report.improvements} "
            f"clear {report.baseline_clear}→{report.best_clear} "
            f"tries={tries} in {report.elapsed_s:.1f}s"
        )
    return best, base, report


def _shift_button_edge(
    frames: list[list[int]],
    edge: int,
    button: int,
    shift: int,
) -> list[list[int]] | None:
    cand = clone_frames(frames)
    new_e = edge + shift
    if new_e <= 0 or new_e >= len(cand):
        return None
    val = cand[edge][button]
    prev = cand[edge - 1][button]
    if val == prev:
        return None
    if shift > 0:
        for j in range(edge, min(new_e, len(cand))):
            cand[j][button] = prev
    else:
        for j in range(new_e, edge):
            cand[j][button] = val
    return cand


def edge_shift_search(
    frames: list[list[int]],
    evaluator: Evaluator,
    *,
    buttons: Sequence[int] = (8, 0),  # A, B
    window: tuple[int, int] | None = None,
    shifts: Sequence[int] = (-3, -2, -1, 1, 2, 3),
    verbose: bool = True,
) -> tuple[list[list[int]], EvalResult, SearchReport]:
    """Shift each rising/falling edge of *buttons* by a few frames."""
    best = clone_frames(frames)
    base = evaluator.evaluate(best, early_terminate=False)
    report = SearchReport(baseline_clear=_clear(base), best_clear=_clear(base))
    if not base.completed:
        return best, base, report

    lo, hi = (1, len(best)) if window is None else window
    lo = max(1, lo)
    hi = min(len(best), hi)
    t0 = time.time()

    for btn in buttons:
        edges = [
            i
            for i in range(lo, hi)
            if i < len(best) and best[i][btn] != best[i - 1][btn]
        ]
        if verbose:
            print(f"[EDGE] button={btn} edges={len(edges)}")
        for edge in edges:
            for shift in shifts:
                cand = _shift_button_edge(best, edge, btn, shift)
                if cand is None:
                    continue
                r = evaluator.evaluate(cand, early_terminate=False)
                c = _clear(r)
                if c is not None and c < (report.best_clear or c + 1):
                    if verbose:
                        print(
                            f"[EDGE] btn={btn} edge={edge} shift={shift} → clear {c}"
                        )
                    best = cand
                    base = r
                    report.best_clear = c
                    report.improvements += 1
                    report.moves.append(
                        {
                            "op": "edge",
                            "button": btn,
                            "edge": edge,
                            "shift": shift,
                            "clear": c,
                        }
                    )
                    break  # next edge on improved seed

    report.elapsed_s = time.time() - t0
    if verbose:
        print(
            f"[EDGE] done imps={report.improvements} "
            f"clear {report.baseline_clear}→{report.best_clear} "
            f"in {report.elapsed_s:.1f}s"
        )
    return best, base, report


def polish_systematic(
    frames: list[list[int]],
    evaluator: Evaluator,
    *,
    flag_frame: int | None = None,
    delete_stride: int = 1,
    verbose: bool = True,
) -> tuple[list[list[int]], EvalResult, SearchReport]:
    """Delete sweep + edge shifts over the pre-flag body."""
    hard = (flag_frame - 2) if flag_frame else len(frames)
    hard = max(1, min(hard, len(frames)))
    best, result, rep1 = systematic_delete_sweep(
        frames,
        evaluator,
        window=(0, hard),
        stride=delete_stride,
        verbose=verbose,
    )
    best, result, rep2 = edge_shift_search(
        best,
        evaluator,
        window=(1, min(hard, len(best))),
        verbose=verbose,
    )
    combined = SearchReport(
        improvements=rep1.improvements + rep2.improvements,
        baseline_clear=rep1.baseline_clear,
        best_clear=rep2.best_clear or rep1.best_clear,
        moves=rep1.moves + rep2.moves,
        elapsed_s=rep1.elapsed_s + rep2.elapsed_s,
    )
    return best, result, combined
