"""Hill climbing local search refinement for platformer optimization."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np

from retro_harness.platformer.actions import (
    DEFAULT_PLATFORMER_ACTIONS,
    action_index_to_buttons,
)
from retro_harness.platformer.evaluator import Evaluator, EvalResult
from retro_harness.platformer.frame_save import (
    INDEX_MUTATION_WEIGHTS,
    accept_candidate,
    resolve_frame_save_mode,
)


def hillclimb(
    actions: list[int],
    evaluator: Evaluator,
    max_iterations: int = 5000,
    output_dir: Path | None = None,
    verbose: bool = True,
    render_interval: int = 0,
    render_scale: int = 3,
    *,
    require_completion: bool | None = None,
    prefer_trim: bool | None = None,
) -> tuple[list[int], EvalResult]:
    """Local search refinement on an action-index sequence.

    Prefers **raw-button** hillclimb when the seed has faithful controller
    input (see ``cmd_hillclimb`` / ``hillclimb_raw``). This path mutates
    discrete action-table indices and is lossy for human pad recordings.

    Tries small perturbations and keeps improvements. Strategies:
    1. Single frame change (40%, or 20% when prefer_trim)
    2. Delete 1-3 frames (trim) — only when prefer_trim or seed completes
    3. Swap two adjacent frames
    4. Shift a segment by 1 frame (insert duplicate / drop one)
    5. Change a short run of frames

    Candidates are always evaluated with ``early_terminate=False`` so fitness
    is comparable to the seed baseline (stall kill would otherwise reject
    slower-but-completing routes unfairly).

    Args:
        actions: Starting action sequence.
        evaluator: Headless evaluator.
        max_iterations: Max improvement attempts.
        output_dir: Directory for saving results.
        verbose: Print progress.
        require_completion: If True, never accept a non-completing candidate.
            ``None`` (default) auto-enables when the seed itself completes.
        prefer_trim: Bias mutations toward frame deletion (for finished seeds).
            ``None`` (default) auto-enables when the seed itself completes.

    Returns:
        (best_actions, best_result)
    """
    config = evaluator.config
    num_actions = len(config.action_table or DEFAULT_PLATFORMER_ACTIONS)

    if output_dir is None:
        output_dir = config.runs_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    best_actions = list(actions)
    best_result = evaluator.evaluate(best_actions, early_terminate=False)
    best_fitness = best_result.fitness

    mode = resolve_frame_save_mode(
        best_result.completed, prefer_trim, require_completion
    )
    prefer_trim = mode.prefer_trim
    require_completion = mode.require_completion

    if verbose:
        print(f"[HILL] Starting fitness: {best_fitness:.1f}")
        print(
            f"[HILL] Completed: {best_result.completed}, "
            f"frames: {best_result.total_frames}, "
            f"require_completion={require_completion}, prefer_trim={prefer_trim}"
        )

    if require_completion and not best_result.completed:
        if verbose:
            print("[HILL] Seed does not complete; refusing require_completion run.")
        return best_actions, best_result

    improvements = 0
    start_time = time.time()

    for iteration in range(max_iterations):
        strategies, weights = INDEX_MUTATION_WEIGHTS[prefer_trim]
        strategy = random.choices(strategies, weights=weights, k=1)[0]

        candidate = list(best_actions)
        n = len(candidate)
        if n < 5:
            continue

        if strategy == "single":
            pos = random.randint(0, n - 1)
            candidate[pos] = random.randint(0, num_actions - 1)

        elif strategy == "delete":
            del_len = random.randint(1, min(3, n // 4))
            pos = random.randint(0, n - del_len)
            del candidate[pos : pos + del_len]

        elif strategy == "swap":
            pos = random.randint(0, n - 2)
            candidate[pos], candidate[pos + 1] = candidate[pos + 1], candidate[pos]

        elif strategy == "shift":
            if random.random() < 0.5 and n > 10:
                pos = random.randint(0, n - 1)
                del candidate[pos]
            else:
                pos = random.randint(0, n - 1)
                candidate.insert(pos, candidate[pos])

        elif strategy == "run_change":
            run_len = random.randint(2, min(5, n // 2))
            pos = random.randint(0, n - run_len)
            new_action = random.randint(0, num_actions - 1)
            for i in range(run_len):
                candidate[pos + i] = new_action

        # Match seed baseline: never stall-kill mid-episode.
        result = evaluator.evaluate(candidate, early_terminate=False)

        if accept_candidate(
            best_result, result, require_completion=require_completion
        ):
            best_actions = candidate
            best_result = result
            best_fitness = result.fitness
            improvements += 1

            if verbose:
                elapsed = time.time() - start_time
                status = "COMPLETE" if result.completed else "incomplete"
                print(
                    f"[HILL] iter={iteration:5d} fitness={best_fitness:10.1f} "
                    f"frames={result.total_frames:5d} progress={result.max_progress:7.1f} "
                    f"status={status} improvements={improvements} "
                    f"strategy={strategy} elapsed={elapsed:.1f}s"
                )

        if iteration > 0 and iteration % 1000 == 0:
            if verbose:
                elapsed = time.time() - start_time
                print(
                    f"[HILL] checkpoint iter={iteration} "
                    f"improvements={improvements} elapsed={elapsed:.1f}s"
                )
            # Checkpoint JSON only — skip screenshot (full re-eval cost).
            _save_hillclimb(
                best_actions, best_result, iteration, output_dir, evaluator=None
            )

        if render_interval > 0 and iteration > 0 and iteration % render_interval == 0:
            _render_best(config, best_actions, iteration, best_fitness, render_scale)

    _save_hillclimb(best_actions, best_result, max_iterations, output_dir, evaluator)

    if verbose:
        elapsed = time.time() - start_time
        print(f"\n[HILL] Done! {max_iterations} iterations, {improvements} improvements in {elapsed:.1f}s")
        print(f"[HILL] Final fitness: {best_fitness:.1f}")
        if best_result:
            print(f"[HILL] Completed: {best_result.completed}")
            print(f"[HILL] Frames: {best_result.total_frames}")
            print(f"[HILL] Max progress: {best_result.max_progress:.1f}")
            print(f"[HILL] Bonus frames: {best_result.bonus_frames}")

    return best_actions, best_result


def _render_best(config, actions: list[int], iteration: int, fitness: float, scale: int = 3) -> None:
    """Render the current best sequence visually."""
    from retro_harness.platformer.runner import _replay_with_hud
    title = f"Hill Climb iter={iteration} fitness={fitness:.0f}"
    print(f"[HILL] Rendering best (iter={iteration})...")
    _replay_with_hud(config, actions, scale=scale, title=title)


def _capture_screenshot(evaluator: Evaluator, actions: list[int], output_path: Path) -> None:
    """Replay actions and save the last visible frame as a PNG."""
    evaluator._ensure_env()
    env = evaluator._env
    assert env is not None and evaluator._cached_state is not None

    env.em.set_state(evaluator._cached_state)
    action_table = evaluator.config.action_table or DEFAULT_PLATFORMER_ACTIONS

    obs = None
    for action_idx in actions:
        buttons = action_index_to_buttons(action_idx, action_table)
        action_size = env.action_space.shape[0]
        if len(buttons) < action_size:
            buttons = buttons + [0] * (action_size - len(buttons))
        obs, _, _, _, _ = env.step(np.array(buttons, dtype=np.int8))

    if obs is not None:
        try:
            from PIL import Image
            img = Image.fromarray(obs)
            img.save(str(output_path))
        except ImportError:
            # Fallback: save raw numpy array
            np.save(str(output_path).replace(".png", ".npy"), obs)


def _save_hillclimb(
    actions: list[int],
    result: EvalResult,
    iteration: int,
    output_dir: Path,
    evaluator: Evaluator | None = None,
) -> None:
    path = output_dir / f"hillclimb_iter{iteration:06d}_best.json"
    data = {
        "actions": actions,
        "num_frames": len(actions),
        "fitness": result.fitness,
        "iteration": iteration,
        "completed": result.completed,
        "total_frames": result.total_frames,
        "max_x": result.max_x,
        "max_progress": result.max_progress,
        "bonus_frames": result.bonus_frames,
    }
    path.write_text(json.dumps(data, indent=2))

    # Capture screenshot of last frame
    if evaluator is not None:
        screenshot_path = output_dir / f"hillclimb_iter{iteration:06d}_screenshot.png"
        try:
            _capture_screenshot(evaluator, actions, screenshot_path)
        except Exception as e:
            print(f"[HILL] Screenshot failed: {e}")
