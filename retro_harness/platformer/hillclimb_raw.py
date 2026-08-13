"""Raw-button hill climbing for games where action-index mapping is lossy.

Unlike the standard hill climber (which mutates action indices), this works
directly with 12-element SNES button arrays. Critical for Super Metroid where
complex button combos (aim+shoot, dash+jump) don't map cleanly to a discrete
action table.

Mutation strategies:
1. Toggle a single button in a single frame (30%)
2. Delete frames to trim (20%) — the main time saver
3. Shift a button press edge by a few frames (20%)
4. Replace a run of frames with a nearby pattern (15%)
5. Insert duplicate frames from a neighbor (15%)
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np

from retro_harness.platformer.evaluator import Evaluator, EvalResult
from retro_harness.platformer.frame_save import (
    RAW_MUTATION_WEIGHTS,
    accept_candidate,
    resolve_frame_save_mode,
)


def hillclimb_raw(
    raw_buttons: list[list[int]],
    evaluator: Evaluator,
    max_iterations: int = 1000,
    output_dir: Path | None = None,
    verbose: bool = True,
    *,
    window: tuple[int, int] | None = None,
    prefer_trim: bool | None = None,
    require_completion: bool | None = None,
    use_segment_engine: bool | None = None,
) -> tuple[list[list[int]], EvalResult]:
    """Hill climb on raw button arrays (NES 9 / SNES 12).

    This is the correct path for human controller recordings: mutations stay
    in button space and do not pass through a lossy action table.

    Returns (best_raw_buttons, best_result).

    When *prefer_trim* / *require_completion* are left as ``None``, a finishing
    seed auto-enables both (frame-save mode). Optional *window* or either flag
    routes through
    :func:`retro_harness.platformer.segment_hillclimb.segment_hillclimb_raw` for
    checkpoint-accelerated search. Pass ``use_segment_engine=True`` to force
    that path even without a window.
    """
    # Cheap baseline so auto defaults match seed quality.
    baseline = evaluator.evaluate(raw_buttons, early_terminate=False)
    mode = resolve_frame_save_mode(
        baseline.completed, prefer_trim, require_completion
    )
    prefer_trim = mode.prefer_trim
    require_completion = mode.require_completion

    if use_segment_engine is None:
        use_segment_engine = bool(
            window is not None or prefer_trim or require_completion
        )
    if use_segment_engine:
        from retro_harness.platformer.segment_hillclimb import segment_hillclimb_raw

        if verbose:
            print(
                f"[RAW-HILL] segment engine "
                f"(prefer_trim={prefer_trim}, require_completion={require_completion}, "
                f"window={window})"
            )
        # segment_hillclimb re-evaluates the seed; that is fine (deterministic).
        return segment_hillclimb_raw(
            raw_buttons,
            evaluator,
            window=window,
            max_iterations=max_iterations,
            prefer_trim=prefer_trim or require_completion,
            require_completion=require_completion,
            output_dir=output_dir,
            verbose=verbose,
        )

    if output_dir is None:
        output_dir = evaluator.config.runs_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    best = [list(f) for f in raw_buttons]
    best_result = baseline
    best_fitness = best_result.fitness
    button_count = max((len(f) for f in best), default=12)

    if verbose:
        status = "COMPLETE" if best_result.completed else "incomplete"
        print(f"[RAW-HILL] Start: fitness={best_fitness:.1f} frames={best_result.total_frames} "
              f"progress={best_result.max_progress:.1f} {status}")

    if best_fitness <= 0:
        print("[RAW-HILL] ERROR: seed has zero/negative fitness. Cannot optimize.")
        return best, best_result

    if require_completion and not best_result.completed:
        if verbose:
            print("[RAW-HILL] Seed does not complete; refusing require_completion run.")
        return best, best_result

    improvements = 0
    start_time = time.time()

    for iteration in range(max_iterations):
        candidate = [list(f) for f in best]
        n = len(candidate)
        if n < 10:
            break

        strategies, weights = RAW_MUTATION_WEIGHTS[prefer_trim]
        strategy = random.choices(strategies, weights=weights, k=1)[0]

        if strategy == "toggle":
            # Toggle 1-3 random buttons in a single frame
            pos = random.randint(0, n - 1)
            num_toggles = random.randint(1, 3)
            for _ in range(num_toggles):
                btn = random.randint(0, button_count - 1)
                if btn < len(candidate[pos]):
                    candidate[pos][btn] ^= 1

        elif strategy == "delete":
            # Delete 1-5 consecutive frames (trim)
            del_len = random.randint(1, min(5, n // 4))
            pos = random.randint(0, n - del_len)
            del candidate[pos:pos + del_len]

        elif strategy == "shift_edge":
            # Find a button transition and shift it by 1-3 frames
            btn = random.randint(0, button_count - 1)
            # Find frames where this button changes
            edges = []
            for i in range(1, n):
                if btn < len(candidate[i]) and candidate[i][btn] != candidate[i - 1][btn]:
                    edges.append(i)
            if edges:
                edge = random.choice(edges)
                shift = random.choice([-3, -2, -1, 1, 2, 3])
                new_edge = max(1, min(n - 1, edge + shift))
                val = candidate[edge][btn]
                prev_val = candidate[edge - 1][btn]
                if shift > 0:
                    # Extend the previous value forward
                    for i in range(edge, min(new_edge, n)):
                        candidate[i][btn] = prev_val
                else:
                    # Pull the new value backward
                    for i in range(max(0, new_edge), edge):
                        candidate[i][btn] = val

        elif strategy == "copy_run":
            # Copy a short run of frames from elsewhere in the sequence
            run_len = random.randint(2, min(8, n // 4))
            src = random.randint(0, n - run_len)
            dst = random.randint(0, n - run_len)
            if src != dst:
                pattern = [list(f) for f in candidate[src:src + run_len]]
                candidate[dst:dst + run_len] = pattern

        elif strategy == "insert":
            # Insert 1-3 copies of a frame (extend a hold)
            ins_len = random.randint(1, 3)
            pos = random.randint(0, n - 1)
            frame = list(candidate[pos])
            for _ in range(ins_len):
                candidate.insert(pos, list(frame))

        result = evaluator.evaluate(candidate, early_terminate=False)

        if accept_candidate(
            best_result, result, require_completion=require_completion
        ):
            best = candidate
            best_result = result
            best_fitness = result.fitness
            improvements += 1

            if verbose:
                elapsed = time.time() - start_time
                status = "COMPLETE" if result.completed else "incomplete"
                frames_delta = len(best) - len(raw_buttons)
                print(
                    f"[RAW-HILL] iter={iteration:5d} fitness={best_fitness:10.1f} "
                    f"frames={len(best):5d} ({frames_delta:+d}) "
                    f"progress={result.max_progress:7.1f} "
                    f"{status} improvements={improvements} "
                    f"strategy={strategy} elapsed={elapsed:.1f}s"
                )

        if iteration > 0 and iteration % 500 == 0:
            if verbose:
                elapsed = time.time() - start_time
                print(
                    f"[RAW-HILL] checkpoint iter={iteration} "
                    f"improvements={improvements} frames={len(best)} elapsed={elapsed:.1f}s"
                )
            _save_raw(best, best_result, iteration, output_dir)

    _save_raw(best, best_result, max_iterations, output_dir)

    if verbose:
        elapsed = time.time() - start_time
        frames_delta = len(best) - len(raw_buttons)
        print(f"\n[RAW-HILL] Done! {max_iterations} iters, {improvements} improvements in {elapsed:.1f}s")
        print(f"[RAW-HILL] Final: fitness={best_fitness:.1f} frames={len(best)} ({frames_delta:+d})")
        print(f"[RAW-HILL] Completed: {best_result.completed}")

    return best, best_result


def _save_raw(
    raw_buttons: list[list[int]],
    result: EvalResult,
    iteration: int,
    output_dir: Path,
) -> None:
    path = output_dir / f"hillclimb_raw_best.json"
    data = {
        "raw_buttons": raw_buttons,
        "num_frames": len(raw_buttons),
        "fitness": result.fitness,
        "iteration": iteration,
        "completed": result.completed,
        "total_frames": result.total_frames,
        "max_progress": result.max_progress,
        "bonus_frames": result.bonus_frames,
    }
    path.write_text(json.dumps(data, indent=2))
