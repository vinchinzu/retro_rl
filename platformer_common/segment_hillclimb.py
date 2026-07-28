"""Windowed raw-button hillclimb with prefix state caching.

Speeds up local search when only a sub-range of the seed is mutated: the
emulator is advanced once through the immutable prefix, the state is cached,
and every candidate starts from that checkpoint instead of frame 0.

Also biased toward **frame deletion** when the seed already completes, which
is the dominant way to save frames on TAS-like seeds.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from platformer_common.evaluator import EvalResult, Evaluator
from platformer_common.frame_tools import clone_frames, find_button_hold_stalls


class PrefixCheckpoint:
    """Cache emulator state after replaying a fixed prefix of buttons."""

    def __init__(self, evaluator: Evaluator) -> None:
        self.evaluator = evaluator
        self.prefix_len = 0
        self._state: Any | None = None
        self._action_size: int | None = None

    def build(self, frames: list[list[int]], prefix_len: int) -> None:
        """Replay frames[:prefix_len] from the level start and cache state."""
        ev = self.evaluator
        ev._ensure_env()
        assert ev._env is not None and ev._cached_state is not None
        ev._env.em.set_state(ev._cached_state)
        action_size = int(ev._env.action_space.shape[0])
        self._action_size = action_size
        prefix_len = max(0, min(prefix_len, len(frames)))
        for frame in frames[:prefix_len]:
            buttons = list(frame[:action_size])
            if len(buttons) < action_size:
                buttons.extend([0] * (action_size - len(buttons)))
            ev._env.step(np.array(buttons, dtype=np.int8))
        self._state = ev._env.em.get_state()
        self.prefix_len = prefix_len

    def evaluate_suffix(
        self,
        frames: list[list[int]],
        *,
        early_terminate: bool = False,
    ) -> EvalResult:
        """Evaluate full *frames* by restoring the prefix checkpoint.

        Only frames[prefix_len:] are stepped. Fitness still uses the standard
        Evaluator path by temporarily swapping the start cache.
        """
        if self._state is None:
            return self.evaluator.evaluate(frames, early_terminate=early_terminate)

        ev = self.evaluator
        ev._ensure_env()
        assert ev._env is not None
        # Point the evaluator's "start" at our checkpoint, evaluate only suffix,
        # then restore.
        saved = ev._cached_state
        saved_initial = ev._initial_values
        saved_cam = ev._initial_camera_x
        try:
            ev._cached_state = self._state
            # Re-read RAM at checkpoint for initial values / death baseline.
            ev._env.em.set_state(self._state)
            ram = ev._env.get_ram()
            values = ev._read_ram(ram)
            ev._initial_values = values
            ev._initial_camera_x = float(values.get("camera_x", 0))
            suffix = frames[self.prefix_len :]
            result = ev.evaluate(suffix, early_terminate=early_terminate)
            # Offset reported frames so total_frames is absolute from seed start.
            if result.total_frames > 0 or result.completed:
                result.total_frames = self.prefix_len + result.total_frames
            if result.completed:
                # fitness = completion_bonus - absolute_clear_frame
                # evaluate used suffix-relative frame_idx; correct it.
                clear_abs = result.total_frames
                result.fitness = ev.config.completion_bonus - (clear_abs - 1)
            return result
        finally:
            ev._cached_state = saved
            ev._initial_values = saved_initial
            ev._initial_camera_x = saved_cam


def _mutate_window(
    frames: list[list[int]],
    lo: int,
    hi: int,
    *,
    prefer_trim: bool,
    hold_stalls: list[tuple[int, int]],
    button_count: int = 12,
) -> tuple[list[list[int]], str]:
    """Mutate only frames[lo:hi]; return (candidate, strategy_name)."""
    candidate = clone_frames(frames)
    n = hi - lo
    if n < 2:
        return candidate, "noop"

    # Prefer delete when optimizing a completed TAS for frame saves.
    if prefer_trim:
        weights = {
            "delete": 40,
            "trim_hold": 25,
            "shift_edge": 15,
            "toggle": 10,
            "copy_run": 5,
            "insert": 5,
        }
    else:
        weights = {
            "toggle": 30,
            "delete": 20,
            "shift_edge": 20,
            "copy_run": 15,
            "insert": 15,
            "trim_hold": 0,
        }

    strategies = [k for k, w in weights.items() if w > 0]
    probs = [weights[k] for k in strategies]
    strategy = random.choices(strategies, weights=probs, k=1)[0]

    if strategy == "delete":
        del_len = random.randint(1, min(8, max(1, n // 4)))
        pos = random.randint(lo, max(lo, hi - del_len))
        del candidate[pos : pos + del_len]

    elif strategy == "trim_hold" and hold_stalls:
        start, length = random.choice(hold_stalls)
        # Map hold into current window if possible
        if start >= lo and start < hi:
            cut = random.randint(1, min(10, max(1, length // 2)))
            end = min(start + length, hi)
            keep_end = max(start + 1, end - cut)
            del candidate[keep_end:end]
        else:
            strategy = "delete"
            del_len = random.randint(1, min(5, max(1, n // 4)))
            pos = random.randint(lo, max(lo, hi - del_len))
            del candidate[pos : pos + del_len]

    elif strategy == "toggle":
        pos = random.randint(lo, hi - 1)
        if pos < len(candidate):
            for _ in range(random.randint(1, 3)):
                btn = random.randint(0, button_count - 1)
                if btn < len(candidate[pos]):
                    candidate[pos][btn] ^= 1

    elif strategy == "shift_edge":
        btn = random.randint(0, button_count - 1)
        edges = []
        for i in range(max(lo + 1, 1), min(hi, len(candidate))):
            if i < len(candidate) and candidate[i][btn] != candidate[i - 1][btn]:
                edges.append(i)
        if edges:
            edge = random.choice(edges)
            shift = random.choice([-3, -2, -1, 1, 2, 3])
            new_edge = max(lo + 1, min(hi - 1, edge + shift))
            val = candidate[edge][btn]
            prev_val = candidate[edge - 1][btn]
            if shift > 0:
                for i in range(edge, min(new_edge, hi, len(candidate))):
                    candidate[i][btn] = prev_val
            else:
                for i in range(max(lo, new_edge), edge):
                    if i < len(candidate):
                        candidate[i][btn] = val
        else:
            strategy = "toggle"
            pos = random.randint(lo, min(hi, len(candidate)) - 1)
            btn = random.randint(0, button_count - 1)
            candidate[pos][btn] ^= 1

    elif strategy == "copy_run":
        run_len = random.randint(2, min(8, max(2, n // 4)))
        if hi - lo >= run_len and len(candidate) >= hi:
            src = random.randint(lo, hi - run_len)
            dst = random.randint(lo, hi - run_len)
            if src != dst:
                pattern = [list(f) for f in candidate[src : src + run_len]]
                candidate[dst : dst + run_len] = pattern

    elif strategy == "insert":
        ins_len = random.randint(1, 3)
        pos = random.randint(lo, min(hi, len(candidate)) - 1)
        frame = list(candidate[pos])
        for _ in range(ins_len):
            candidate.insert(pos, list(frame))

    return candidate, strategy


def segment_hillclimb_raw(
    raw_buttons: list[list[int]],
    evaluator: Evaluator,
    *,
    window: tuple[int, int] | None = None,
    max_iterations: int = 1000,
    prefer_trim: bool = True,
    require_completion: bool = True,
    rebuild_checkpoint_every: int = 50,
    output_dir: Path | None = None,
    verbose: bool = True,
) -> tuple[list[list[int]], EvalResult]:
    """Hill-climb raw buttons, mutating only *window* (inclusive-exclusive).

    When *window* is None, mutates the full sequence but still uses trim bias
    and completion gating.
    """
    if output_dir is None:
        output_dir = evaluator.config.runs_dir
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best = clone_frames(raw_buttons)
    best_result = evaluator.evaluate(best, early_terminate=False)
    best_fitness = best_result.fitness

    if verbose:
        status = "COMPLETE" if best_result.completed else "incomplete"
        print(
            f"[SEG-HILL] Start: fitness={best_fitness:.1f} "
            f"frames={best_result.total_frames} progress={best_result.max_progress:.1f} "
            f"{status}"
        )

    if require_completion and not best_result.completed:
        print("[SEG-HILL] Seed does not complete; refusing require_completion run.")
        return best, best_result

    lo, hi = (0, len(best)) if window is None else window
    lo = max(0, lo)
    hi = min(len(best), hi)
    if lo >= hi:
        lo, hi = 0, len(best)

    if verbose:
        print(f"[SEG-HILL] Window [{lo}:{hi}] ({hi - lo} frames mutable)")

    checkpoint = PrefixCheckpoint(evaluator)
    checkpoint.build(best, lo)
    improvements = 0
    start_time = time.time()
    button_count = max(len(best[0]) if best else 12, 9)

    for iteration in range(max_iterations):
        # Hold stalls inside the current window for trim_hold mutations.
        holds = find_button_hold_stalls(best[lo:hi], min_length=16)
        hold_spans = [(lo + h.start, h.length) for h in holds]

        candidate, strategy = _mutate_window(
            best,
            lo,
            min(hi, len(best)),
            prefer_trim=prefer_trim and best_result.completed,
            hold_stalls=hold_spans,
            button_count=button_count,
        )

        # Prefix must stay identical for checkpoint reuse.
        if candidate[:lo] != best[:lo]:
            # Should not happen with windowed mutator; full eval fallback.
            result = evaluator.evaluate(candidate, early_terminate=False)
        else:
            # If length before lo changed (shouldn't), rebuild.
            result = checkpoint.evaluate_suffix(candidate, early_terminate=False)

        accept = False
        if require_completion:
            if result.completed and (
                not best_result.completed
                or result.total_frames < best_result.total_frames
                or (
                    result.total_frames == best_result.total_frames
                    and result.fitness > best_fitness
                )
            ):
                accept = True
        else:
            if result.fitness > best_fitness:
                accept = True

        if accept:
            best = candidate
            best_result = result
            best_fitness = result.fitness
            improvements += 1
            # Window end may shrink after deletes; keep lo fixed, clamp hi.
            hi = min(max(hi, lo + 1), len(best))
            if verbose:
                elapsed = time.time() - start_time
                print(
                    f"[SEG-HILL] iter={iteration:5d} fitness={best_fitness:10.1f} "
                    f"frames={best_result.total_frames:5d} "
                    f"seed_len={len(best):5d} "
                    f"strategy={strategy} improvements={improvements} "
                    f"elapsed={elapsed:.1f}s"
                )
            # Rebuild checkpoint after improvements (prefix unchanged, but
            # restores a clean core state).
            checkpoint.build(best, lo)

        elif iteration > 0 and iteration % rebuild_checkpoint_every == 0:
            checkpoint.build(best, lo)

        if iteration > 0 and iteration % 200 == 0:
            _save(best, best_result, iteration, output_dir, lo, hi)
            if verbose:
                print(
                    f"[SEG-HILL] checkpoint iter={iteration} "
                    f"improvements={improvements} frames={best_result.total_frames}"
                )

    _save(best, best_result, max_iterations, output_dir, lo, hi)
    if verbose:
        elapsed = time.time() - start_time
        print(
            f"\n[SEG-HILL] Done! {max_iterations} iters, {improvements} improvements "
            f"in {elapsed:.1f}s"
        )
        print(
            f"[SEG-HILL] Final: fitness={best_fitness:.1f} "
            f"frames={best_result.total_frames} seed_len={len(best)} "
            f"completed={best_result.completed}"
        )
    return best, best_result


def _save(
    raw_buttons: list[list[int]],
    result: EvalResult,
    iteration: int,
    output_dir: Path,
    lo: int,
    hi: int,
) -> None:
    path = output_dir / "segment_hillclimb_best.json"
    data = {
        "raw_buttons": raw_buttons,
        "num_frames": len(raw_buttons),
        "fitness": result.fitness,
        "iteration": iteration,
        "completed": result.completed,
        "total_frames": result.total_frames,
        "max_progress": result.max_progress,
        "bonus_frames": result.bonus_frames,
        "window": [lo, hi],
    }
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
