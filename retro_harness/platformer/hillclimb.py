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


def hillclimb(
    actions: list[int],
    evaluator: Evaluator,
    max_iterations: int = 5000,
    output_dir: Path | None = None,
    verbose: bool = True,
    render_interval: int = 0,
    render_scale: int = 3,
) -> tuple[list[int], EvalResult]:
    """Local search refinement on an action sequence.

    Tries small perturbations and keeps improvements. Strategies:
    1. Single frame change (50%)
    2. Swap two adjacent frames (15%)
    3. Shift a segment by 1 frame (15%)
    4. Change a short run of frames (20%)

    Args:
        actions: Starting action sequence.
        evaluator: Headless evaluator.
        max_iterations: Max improvement attempts.
        output_dir: Directory for saving results.
        verbose: Print progress.

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

    if verbose:
        print(f"[HILL] Starting fitness: {best_fitness:.1f}")
        print(f"[HILL] Completed: {best_result.completed}, frames: {best_result.total_frames}")

    improvements = 0
    start_time = time.time()

    for iteration in range(max_iterations):
        strategy = random.choices(
            ["single", "swap", "shift", "run_change"],
            weights=[50, 15, 15, 20],
            k=1,
        )[0]

        candidate = list(best_actions)
        n = len(candidate)
        if n < 5:
            continue

        if strategy == "single":
            pos = random.randint(0, n - 1)
            candidate[pos] = random.randint(0, num_actions - 1)

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

        result = evaluator.evaluate(candidate)

        if result.fitness > best_fitness:
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
            _save_hillclimb(best_actions, best_result, iteration, output_dir, evaluator)

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
