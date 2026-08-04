"""Hierarchical RLE-aware local search for platformer seeds.

Optimizes a *window* of an RLE (or frame-expanded) seed while freezing the
prefix/suffix.  Prefer this over flat frame mutations when polishing long
TAS-like sequences (e.g. SMB 21k continuous seed).

Typical use for known bottlenecks::

    from retro_harness.platformer.rle_optimize import RleWindow, rle_hillclimb_window
    from retro_harness.platformer.frame_tools import load_raw_frames

    frames = load_raw_frames("smb/models/smb_1_1_to_ending.json")
    # 1-1 end-stairs region (approx frame window from STATUS notes)
    window = RleWindow(start=1700, end=1970, label="1-1-stairs")
    best, result = rle_hillclimb_window(frames, window, evaluator, max_iters=500)

Segment-level GA::

    best_runs = rle_ga_window(runs, window, evaluator, population=40, generations=50)
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from retro_harness.platformer.evaluator import EvalResult, Evaluator
from retro_harness.platformer.rle_ops import (
    RleMutateConfig,
    RleRun,
    RleSeq,
    SMB_ACTION_ATOMIC_PATTERNS,
    SMB_ATOMIC_PATTERNS,
    compress_rle,
    crossover_rle_frame_aligned,
    expand_rle,
    mutate_rle,
    rle_normalize,
    rle_replace_window,
    rle_slice_frames,
    rle_total_frames,
)


@dataclass(frozen=True)
class RleWindow:
    """Absolute frame window [start, end) inside a full seed."""

    start: int
    end: int
    label: str = ""
    # Optional soft progress anchors (for logging / future progress-aligned CX)
    min_progress: float | None = None
    max_progress: float | None = None

    def clamp(self, total: int) -> RleWindow:
        s = max(0, min(self.start, total))
        e = max(s, min(self.end, total))
        return RleWindow(s, e, self.label, self.min_progress, self.max_progress)

    @property
    def length(self) -> int:
        return max(0, self.end - self.start)


# Known SMB continuous-seed bottlenecks (frame indices into continuous seed).
# Measured 2026-07-30 on smb_1_1_to_ending.json with settle=14:
#   stairs wall-slam xs→0 at f≈1164 (x=2962) and f≈1210 (x=2994);
#   flag grab (player_state=4) at f≈1311. Old 1700–1974 window was castle
#   score-tally idle (player_state=5) — no optimizable control.
SMB_BOTTLENECK_WINDOWS: tuple[RleWindow, ...] = (
    RleWindow(1050, 1311, "1-1-stairs", min_progress=2700, max_progress=3200),
    RleWindow(350, 520, "1-1-first-pipe", min_progress=850, max_progress=950),
    # 4-2 occupies roughly frames 6366..9130 in the continuous seed (STATUS)
    RleWindow(6366, 9130, "4-2-full", min_progress=0, max_progress=None),
    # Tighter natural-entry polish region at the start of 4-2
    RleWindow(6366, 7000, "4-2-entry", min_progress=0, max_progress=None),
)


@dataclass
class RleIndividual:
    runs: RleSeq[Any]
    fitness: float = float("-inf")
    result: Optional[EvalResult] = None


def _frames_as_buttons(frames: Sequence[Sequence[int]], size: int = 12) -> list[list[int]]:
    out: list[list[int]] = []
    for f in frames:
        b = [int(x) for x in f[:size]]
        if len(b) < size:
            b.extend([0] * (size - len(b)))
        out.append(b)
    return out


def frames_to_rle(frames: Sequence[Any]) -> RleSeq[Any]:
    """Compress frames (ints or button lists) to RLE."""
    if not frames:
        return []
    if isinstance(frames[0], (list, tuple)):
        keys = [tuple(int(x) for x in f) for f in frames]
        return compress_rle(keys)
    return compress_rle([int(a) for a in frames])


def rle_to_frames(runs: Sequence[RleRun[Any]], *, button_mode: bool) -> list[Any]:
    """Expand RLE; button payloads become lists."""
    return expand_rle(runs, as_list=button_mode)


def stitch_window(
    full_runs: Sequence[RleRun[Any]],
    window: RleWindow,
    window_runs: Sequence[RleRun[Any]],
) -> RleSeq[Any]:
    """Replace *window* in *full_runs* with *window_runs*."""
    w = window.clamp(rle_total_frames(full_runs))
    return rle_replace_window(full_runs, w.start, w.end, window_runs)


def evaluate_rle_frames(
    runs: Sequence[RleRun[Any]],
    evaluator: Evaluator,
    *,
    button_mode: bool,
    early_terminate: bool = True,
) -> EvalResult:
    frames = rle_to_frames(runs, button_mode=button_mode)
    if button_mode:
        frames = _frames_as_buttons(frames)
    return evaluator.evaluate(frames, early_terminate=early_terminate)


def rle_hillclimb_window(
    seed_frames: Sequence[Any],
    window: RleWindow,
    evaluator: Evaluator,
    *,
    max_iters: int = 500,
    n_ops: int = 1,
    button_mode: bool | None = None,
    require_completion: bool = True,
    config: RleMutateConfig | None = None,
    verbose: bool = True,
    output_dir: Path | None = None,
) -> tuple[list[Any], EvalResult]:
    """Hillclimb only the RLE runs inside *window*; return full best frames.

    If the seed already completes and *require_completion* is True, candidates
    that die or fail to complete are rejected even if raw fitness is higher.
    """
    if button_mode is None:
        button_mode = not (seed_frames and isinstance(seed_frames[0], int))

    full_runs = frames_to_rle(seed_frames)
    total = rle_total_frames(full_runs)
    w = window.clamp(total)
    if w.length < 2:
        result = evaluate_rle_frames(full_runs, evaluator, button_mode=button_mode, early_terminate=False)
        return rle_to_frames(full_runs, button_mode=button_mode), result

    if config is None:
        if button_mode:
            config = RleMutateConfig(
                atomic_patterns=SMB_ATOMIC_PATTERNS,  # type: ignore[arg-type]
                button_size=12,
                button_flip_indices=(0, 5, 6, 7, 8),
            )
        else:
            config = RleMutateConfig(
                atomic_patterns=SMB_ACTION_ATOMIC_PATTERNS,  # type: ignore[arg-type]
                num_actions=len(evaluator.config.action_table or []),
            )

    best_runs = full_runs
    best_result = evaluate_rle_frames(
        best_runs, evaluator, button_mode=button_mode, early_terminate=False
    )
    best_fit = best_result.fitness

    if verbose:
        label = w.label or f"{w.start}:{w.end}"
        print(
            f"[RLE-HILL] window={label} frames={w.start}-{w.end} "
            f"({w.length}f) start_fit={best_fit:.1f} "
            f"completed={best_result.completed} total={best_result.total_frames}"
        )

    improvements = 0
    t0 = time.time()

    for it in range(max_iters):
        mid = rle_slice_frames(best_runs, w.start, w.end)
        mut = mutate_rle(mid, config=config, n_ops=n_ops)
        # Light global phase: occasional ±1f edge shift of the window content
        if random.random() < 0.1 and mut:
            edge = random.choice((0, len(mut) - 1))
            p, d = mut[edge]
            if random.random() < 0.5 and d > 1:
                mut[edge] = (p, d - 1)
            else:
                mut[edge] = (p, d + 1)
            mut = rle_normalize(mut)

        cand_runs = stitch_window(best_runs, w, mut)
        # Re-clamp window end if mutation changed length
        new_total = rle_total_frames(cand_runs)
        # Keep window anchored at start; end slides with length delta
        length_delta = new_total - total
        # For evaluation we always use full sequence
        result = evaluate_rle_frames(
            cand_runs, evaluator, button_mode=button_mode, early_terminate=True
        )

        accept = result.fitness > best_fit
        if require_completion and best_result.completed and not result.completed:
            accept = False
        if require_completion and best_result.completed and result.died:
            accept = False

        if accept:
            best_runs = cand_runs
            best_result = result
            best_fit = result.fitness
            total = new_total
            # Slide window end if the middle grew/shrunk so we keep polishing
            # the same relative region.
            w = RleWindow(
                w.start,
                min(new_total, w.end + length_delta),
                w.label,
                w.min_progress,
                w.max_progress,
            )
            improvements += 1
            if verbose:
                print(
                    f"[RLE-HILL] it={it} improved fit={best_fit:.1f} "
                    f"frames={result.total_frames} completed={result.completed}"
                )

        if verbose and it > 0 and it % 100 == 0:
            print(
                f"[RLE-HILL] it={it}/{max_iters} best={best_fit:.1f} "
                f"improvements={improvements} elapsed={time.time() - t0:.1f}s"
            )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        frames = rle_to_frames(best_runs, button_mode=button_mode)
        if button_mode:
            frames = _frames_as_buttons(frames)
        payload = {
            "format": "raw_buttons" if button_mode else "action_indices",
            "raw_buttons" if button_mode else "actions": frames,
            "num_frames": len(frames),
            "window": {"start": window.start, "end": window.end, "label": window.label},
            "fitness": best_fit,
            "completed": best_result.completed,
            "total_frames": best_result.total_frames,
            "improvements": improvements,
        }
        (output_dir / "rle_hill_best.json").write_text(json.dumps(payload, indent=2) + "\n")

    if verbose:
        print(
            f"[RLE-HILL] done improvements={improvements} "
            f"fit={best_fit:.1f} frames={best_result.total_frames} "
            f"elapsed={time.time() - t0:.1f}s"
        )

    out_frames = rle_to_frames(best_runs, button_mode=button_mode)
    if button_mode:
        out_frames = _frames_as_buttons(out_frames)
    return out_frames, best_result


def rle_ga_window(
    seed_frames: Sequence[Any],
    window: RleWindow,
    evaluator: Evaluator,
    *,
    population_size: int = 40,
    num_generations: int = 50,
    elite_count: int = 4,
    button_mode: bool | None = None,
    require_completion: bool = True,
    config: RleMutateConfig | None = None,
    verbose: bool = True,
    output_dir: Path | None = None,
) -> tuple[list[Any], EvalResult]:
    """Small GA that only mutates/crossovers the RLE window interior."""
    if button_mode is None:
        button_mode = not (seed_frames and isinstance(seed_frames[0], int))

    full_runs = frames_to_rle(seed_frames)
    total = rle_total_frames(full_runs)
    w = window.clamp(total)

    if config is None:
        if button_mode:
            config = RleMutateConfig(atomic_patterns=SMB_ATOMIC_PATTERNS)  # type: ignore[arg-type]
        else:
            n_act = len(evaluator.config.action_table or [])
            config = RleMutateConfig(
                atomic_patterns=SMB_ACTION_ATOMIC_PATTERNS,  # type: ignore[arg-type]
                num_actions=n_act,
            )

    base_mid = rle_slice_frames(full_runs, w.start, w.end)

    def stitch(mid: RleSeq[Any]) -> RleSeq[Any]:
        return stitch_window(full_runs, w, mid)

    def eval_mid(mid: RleSeq[Any]) -> EvalResult:
        return evaluate_rle_frames(stitch(mid), evaluator, button_mode=button_mode)

    population: list[RleIndividual] = [RleIndividual(runs=list(base_mid))]
    while len(population) < population_size:
        n_ops = 1 + int(len(population) > population_size // 2)
        population.append(RleIndividual(runs=mutate_rle(base_mid, config=config, n_ops=n_ops)))

    for ind in population:
        ind.result = eval_mid(ind.runs)
        ind.fitness = ind.result.fitness

    best = max(population, key=lambda x: x.fitness)
    best = RleIndividual(runs=list(best.runs), fitness=best.fitness, result=best.result)
    seed_completed = bool(best.result and best.result.completed)

    if verbose:
        print(
            f"[RLE-GA] pop={population_size} gens={num_generations} "
            f"window={w.label or w.start}:{w.end} start_fit={best.fitness:.1f}"
        )

    t0 = time.time()
    for gen in range(num_generations):
        population.sort(key=lambda x: x.fitness, reverse=True)
        if population[0].fitness > best.fitness:
            cand = population[0]
            if require_completion and seed_completed and cand.result and not cand.result.completed:
                pass
            else:
                best = RleIndividual(
                    runs=list(cand.runs), fitness=cand.fitness, result=cand.result
                )

        if verbose and (gen % 5 == 0 or gen == num_generations - 1):
            fr = best.result.total_frames if best.result else 0
            print(
                f"[RLE-GA] gen={gen:3d} best={best.fitness:.1f} frames={fr} "
                f"elapsed={time.time() - t0:.1f}s"
            )

        next_gen: list[RleIndividual] = []
        for i in range(min(elite_count, len(population))):
            next_gen.append(RleIndividual(runs=list(population[i].runs), fitness=population[i].fitness, result=population[i].result))

        while len(next_gen) < population_size:
            p1 = max(random.sample(population, min(3, len(population))), key=lambda x: x.fitness)
            p2 = max(random.sample(population, min(3, len(population))), key=lambda x: x.fitness)
            if random.random() < 0.6:
                child_runs = crossover_rle_frame_aligned(p1.runs, p2.runs)
            else:
                child_runs = list(p1.runs)
            child_runs = mutate_rle(child_runs, config=config, n_ops=1)
            child = RleIndividual(runs=child_runs)
            child.result = eval_mid(child.runs)
            child.fitness = child.result.fitness
            if require_completion and seed_completed and child.result and not child.result.completed:
                child.fitness = min(child.fitness, best.fitness - 1.0)
            next_gen.append(child)

        population = next_gen

    final_runs = stitch(best.runs)
    final_result = evaluate_rle_frames(
        final_runs, evaluator, button_mode=button_mode, early_terminate=False
    )
    out = rle_to_frames(final_runs, button_mode=button_mode)
    if button_mode:
        out = _frames_as_buttons(out)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        key = "raw_buttons" if button_mode else "actions"
        (output_dir / "rle_ga_best.json").write_text(
            json.dumps(
                {
                    key: out,
                    "num_frames": len(out),
                    "fitness": final_result.fitness,
                    "completed": final_result.completed,
                    "total_frames": final_result.total_frames,
                    "window": {"start": w.start, "end": w.end, "label": w.label},
                },
                indent=2,
            )
            + "\n"
        )

    if verbose:
        print(
            f"[RLE-GA] done fit={final_result.fitness:.1f} "
            f"frames={final_result.total_frames} completed={final_result.completed}"
        )
    return out, final_result


def phase_shift_transitions(
    seed_frames: Sequence[Any],
    transition_frames: Sequence[int],
    evaluator: Evaluator,
    *,
    max_shift: int = 3,
    button_mode: bool | None = None,
    verbose: bool = True,
) -> tuple[list[Any], EvalResult]:
    """Light global polish: ±shift idle padding at segment transition points.

    For each transition frame index, try inserting or deleting 1..max_shift
    idle frames. Keeps completion when the seed already completes.
    """
    if button_mode is None:
        button_mode = not (seed_frames and isinstance(seed_frames[0], int))

    runs = frames_to_rle(seed_frames)
    best_runs = runs
    best = evaluate_rle_frames(best_runs, evaluator, button_mode=button_mode, early_terminate=False)
    idle_payload: Any
    if button_mode:
        idle_payload = tuple(0 for _ in range(12))
    else:
        idle_payload = 0

    if verbose:
        print(
            f"[RLE-PHASE] transitions={list(transition_frames)} "
            f"max_shift={max_shift} start_fit={best.fitness:.1f}"
        )

    for t in transition_frames:
        for delta in range(-max_shift, max_shift + 1):
            if delta == 0:
                continue
            cand = list(best_runs)
            total = rle_total_frames(cand)
            pos = max(0, min(t, total))
            if delta > 0:
                cand = rle_replace_window(cand, pos, pos, [(idle_payload, delta)])
            else:
                # delete |delta| frames starting at pos
                end = min(total, pos + (-delta))
                cand = rle_replace_window(cand, pos, end, [])
            result = evaluate_rle_frames(cand, evaluator, button_mode=button_mode)
            if result.fitness > best.fitness:
                if best.completed and not result.completed:
                    continue
                best_runs = cand
                best = result
                if verbose:
                    print(f"[RLE-PHASE] t={t} delta={delta:+d} fit={best.fitness:.1f}")

    out = rle_to_frames(best_runs, button_mode=button_mode)
    if button_mode:
        out = _frames_as_buttons(out)
    return out, best
