"""Genetic algorithm optimizer for platformer action sequences.

Enhancements over the DKC-specific version:
- Multiprocessing: evaluate_batch() uses a process pool for ~4x speedup
- Multi-seed init: accept multiple seed sequences to bootstrap population
- Checkpoint resume: continue from a saved checkpoint JSON
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Optional

from platformer_common.evaluator import Evaluator, EvalResult
from platformer_common.level_config import LevelConfig

# GA defaults (can be overridden per level or per run)
DEFAULT_MUTATION_RATE = 0.02
DEFAULT_CROSSOVER_RATE = 0.7
DEFAULT_TOURNAMENT_SIZE = 5
DEFAULT_ELITE_COUNT = 5
DEFAULT_DIVERSITY_INJECTION_INTERVAL = 50


@dataclass
class Individual:
    """A candidate solution: sequence of action indices."""

    actions: list[int]
    fitness: float = float("-inf")
    result: Optional[EvalResult] = None


# -- Mutation operators ------------------------------------------------------


def mutate(actions: list[int], rate: float = DEFAULT_MUTATION_RATE, num_actions: int = 14) -> list[int]:
    """Apply mutation operators to an action sequence."""
    actions = list(actions)
    n = len(actions)
    if n == 0:
        return actions

    for i in range(n):
        if random.random() < rate:
            actions[i] = random.randint(0, num_actions - 1)

    # Insert frames (5% chance)
    if random.random() < 0.05 and n > 10:
        pos = random.randint(0, n - 1)
        count = random.randint(1, 3)
        insert_action = actions[pos]
        for _ in range(count):
            actions.insert(pos, insert_action)

    # Delete frames (5% chance)
    if random.random() < 0.05 and n > 20:
        pos = random.randint(0, n - 4)
        count = min(random.randint(1, 3), n - pos - 1)
        del actions[pos : pos + count]

    # Extend/shorten hold (5% chance)
    if random.random() < 0.05 and n > 10:
        pos = random.randint(0, n - 2)
        run_end = pos + 1
        while run_end < n and actions[run_end] == actions[pos]:
            run_end += 1
        run_len = run_end - pos
        if random.random() < 0.5 and run_len > 1:
            del actions[pos]
        else:
            actions.insert(pos, actions[pos])

    # Swap segment (3% chance)
    if random.random() < 0.03 and n > 40:
        seg_len = random.randint(5, min(20, n // 4))
        pos1 = random.randint(0, n - seg_len - 1)
        pos2 = random.randint(0, n - seg_len - 1)
        if abs(pos1 - pos2) >= seg_len:
            seg1 = actions[pos1 : pos1 + seg_len]
            seg2 = actions[pos2 : pos2 + seg_len]
            actions[pos1 : pos1 + seg_len] = seg2
            actions[pos2 : pos2 + seg_len] = seg1

    return actions


def crossover(parent1: list[int], parent2: list[int]) -> tuple[list[int], list[int]]:
    """Single-point crossover between two action sequences."""
    min_len = min(len(parent1), len(parent2))
    if min_len < 4:
        return list(parent1), list(parent2)
    point = random.randint(1, min_len - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2


def crossover_segment(parent1: list[int], parent2: list[int]) -> list[int]:
    """Segment-based crossover: splice a segment from parent2 into parent1."""
    if len(parent1) < 10 or len(parent2) < 10:
        return list(parent1)
    min_len = min(len(parent1), len(parent2))
    seg_len = random.randint(5, min(50, min_len // 3))
    start = random.randint(0, min_len - seg_len)
    child = list(parent1)
    child[start : start + seg_len] = parent2[start : start + seg_len]
    return child


def tournament_select(population: list[Individual], k: int = DEFAULT_TOURNAMENT_SIZE) -> Individual:
    """Tournament selection."""
    candidates = random.sample(population, min(k, len(population)))
    return max(candidates, key=lambda ind: ind.fitness)


# -- Worker for multiprocessing ----------------------------------------------

# Module-level evaluator for worker processes (initialized once per worker)
_worker_evaluator: Evaluator | None = None


def _worker_init(config: LevelConfig) -> None:
    """Initialize evaluator in worker process."""
    global _worker_evaluator
    _worker_evaluator = Evaluator(config)


def _worker_evaluate(actions: list[int]) -> EvalResult:
    """Evaluate a single individual in a worker process."""
    assert _worker_evaluator is not None
    return _worker_evaluator.evaluate(actions)


# -- Main GA -----------------------------------------------------------------


def evaluate_batch(
    individuals: list[Individual],
    evaluator: Evaluator,
    pool: Pool | None = None,
) -> None:
    """Evaluate a batch of individuals, optionally using multiprocessing."""
    if pool is not None:
        results = pool.map(_worker_evaluate, [ind.actions for ind in individuals])
        for ind, result in zip(individuals, results):
            ind.result = result
            ind.fitness = result.fitness
    else:
        for ind in individuals:
            ind.result = evaluator.evaluate(ind.actions)
            ind.fitness = ind.result.fitness


def run_ga(
    seed_actions: list[int] | list[list[int]],
    evaluator: Evaluator,
    population_size: int | None = None,
    num_generations: int | None = None,
    elite_count: int = DEFAULT_ELITE_COUNT,
    output_dir: Path | None = None,
    verbose: bool = True,
    num_workers: int = 1,
    resume_from: Path | None = None,
    render_interval: int = 0,
) -> Individual:
    """Run genetic algorithm to optimize action sequence.

    Args:
        seed_actions: Initial action sequence(s). Pass a single list[int]
            or multiple seeds as list[list[int]].
        evaluator: Headless evaluator instance.
        population_size: Number of individuals (defaults to config value).
        num_generations: Max generations (defaults to config value).
        elite_count: Number of elites preserved each generation.
        output_dir: Directory to save checkpoints.
        verbose: Print progress.
        num_workers: Number of parallel workers (1 = serial).
        resume_from: Path to checkpoint JSON to resume from.
        render_interval: If >0, visually replay the best individual every N gens.

    Returns:
        Best individual found.
    """
    config = evaluator.config
    if population_size is None:
        population_size = config.population_size
    if num_generations is None:
        num_generations = config.num_generations
    if output_dir is None:
        output_dir = config.runs_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    action_table = config.action_table
    from platformer_common.actions import DEFAULT_PLATFORMER_ACTIONS

    num_actions = len(action_table or DEFAULT_PLATFORMER_ACTIONS)

    # Normalize seeds: support single seed or multiple seeds
    seeds: list[list[int]]
    if isinstance(seed_actions[0], int):
        seeds = [seed_actions]  # type: ignore[list-item]
    else:
        seeds = seed_actions  # type: ignore[assignment]

    # Handle checkpoint resume
    start_gen = 0
    if resume_from is not None and resume_from.exists():
        checkpoint = json.loads(resume_from.read_text())
        seeds = [checkpoint["actions"]]
        start_gen = checkpoint.get("generation", 0)
        if verbose:
            print(f"[GA] Resuming from {resume_from} (gen {start_gen}, fitness {checkpoint.get('fitness', '?')})")

    # Initialize population from seeds + mutations
    population: list[Individual] = []
    for seed in seeds:
        population.append(Individual(actions=list(seed)))

    # Fill remaining slots by cycling mutations across all seeds
    seed_idx = 0
    while len(population) < population_size:
        seed = seeds[seed_idx % len(seeds)]
        rate = DEFAULT_MUTATION_RATE * (1 + len(population) / population_size * 3)
        mutated = mutate(seed, rate=rate, num_actions=num_actions)
        population.append(Individual(actions=mutated))
        seed_idx += 1

    # Set up multiprocessing pool
    pool: Pool | None = None
    if num_workers > 1:
        pool = Pool(processes=num_workers, initializer=_worker_init, initargs=(config,))

    try:
        # Evaluate initial population
        if verbose:
            print(f"[GA] Evaluating initial population ({population_size} individuals, {num_workers} workers)...")

        evaluate_batch(population, evaluator, pool)

        best_ever = max(population, key=lambda ind: ind.fitness)
        best_ever = Individual(
            actions=list(best_ever.actions),
            fitness=best_ever.fitness,
            result=best_ever.result,
        )
        stall_gens = 0
        start_time = time.time()

        for gen in range(start_gen, start_gen + num_generations):
            population.sort(key=lambda ind: ind.fitness, reverse=True)

            gen_best = population[0]
            if gen_best.fitness > best_ever.fitness:
                best_ever = Individual(
                    actions=list(gen_best.actions),
                    fitness=gen_best.fitness,
                    result=gen_best.result,
                )
                stall_gens = 0
            else:
                stall_gens += 1

            if verbose and (gen % 10 == 0 or gen == start_gen + num_generations - 1):
                elapsed = time.time() - start_time
                status = "COMPLETE" if best_ever.result and best_ever.result.completed else "incomplete"
                frames = best_ever.result.total_frames if best_ever.result else 0
                progress = best_ever.result.max_progress if best_ever.result else 0
                print(
                    f"[GA] gen={gen:4d} best_fitness={best_ever.fitness:10.1f} "
                    f"frames={frames:5d} progress={progress:7.1f} "
                    f"status={status} stall={stall_gens} elapsed={elapsed:.1f}s"
                )

            # Render best individual if requested
            if render_interval > 0 and (gen % render_interval == 0 or gen == start_gen):
                if not _render_best(best_ever, evaluator, gen):
                    if verbose:
                        print("[GA] Render window closed, continuing headless...")
                    render_interval = 0  # disable further rendering

            # Checkpoint every 10 gens
            if gen % 10 == 0 and gen > start_gen:
                _save_checkpoint(best_ever, gen, output_dir, evaluator)

            # Diversity injection if stuck
            if stall_gens > 0 and stall_gens % DEFAULT_DIVERSITY_INJECTION_INTERVAL == 0:
                if verbose:
                    print(f"[GA] Injecting diversity (stall={stall_gens})")
                replace_count = population_size // 5
                new_individuals = []
                for i in range(replace_count):
                    rate = DEFAULT_MUTATION_RATE * random.uniform(2, 8)
                    ind = Individual(actions=mutate(best_ever.actions, rate=rate, num_actions=num_actions))
                    new_individuals.append(ind)
                evaluate_batch(new_individuals, evaluator, pool)
                for i, ind in enumerate(new_individuals):
                    population[population_size - 1 - i] = ind

            # Build next generation
            next_gen: list[Individual] = []

            # Elitism
            for i in range(min(elite_count, len(population))):
                next_gen.append(Individual(
                    actions=list(population[i].actions),
                    fitness=population[i].fitness,
                    result=population[i].result,
                ))

            # Fill with crossover + mutation
            children: list[Individual] = []
            while len(next_gen) + len(children) < population_size:
                parent1 = tournament_select(population)
                parent2 = tournament_select(population)

                if random.random() < DEFAULT_CROSSOVER_RATE:
                    if random.random() < 0.5:
                        c1, c2 = crossover(parent1.actions, parent2.actions)
                    else:
                        c1 = crossover_segment(parent1.actions, parent2.actions)
                        c2 = crossover_segment(parent2.actions, parent1.actions)
                else:
                    c1 = list(parent1.actions)
                    c2 = list(parent2.actions)

                c1 = mutate(c1, num_actions=num_actions)
                c2 = mutate(c2, num_actions=num_actions)

                children.append(Individual(actions=c1))
                if len(next_gen) + len(children) < population_size:
                    children.append(Individual(actions=c2))

            evaluate_batch(children, evaluator, pool)
            next_gen.extend(children)
            population = next_gen

        # Final save
        final_gen = start_gen + num_generations
        _save_checkpoint(best_ever, final_gen, output_dir, evaluator)

        if verbose:
            elapsed = time.time() - start_time
            print(f"\n[GA] Done! {num_generations} generations in {elapsed:.1f}s")
            print(f"[GA] Best fitness: {best_ever.fitness:.1f}")
            if best_ever.result:
                print(f"[GA] Completed: {best_ever.result.completed}")
                print(f"[GA] Frames: {best_ever.result.total_frames}")
                print(f"[GA] Progress: {best_ever.result.max_progress:.1f}")

        return best_ever

    finally:
        if pool is not None:
            pool.terminate()
            pool.join()


def _render_best(best: Individual, evaluator: Evaluator, gen: int) -> bool:
    """Replay the best individual visually. Returns False if user closed window."""
    import numpy as np

    try:
        import pygame
    except ImportError:
        print("[GA] pygame not installed, skipping render")
        return True

    config = evaluator.config
    from platformer_common.actions import action_index_to_buttons, DEFAULT_PLATFORMER_ACTIONS

    action_table = config.action_table or DEFAULT_PLATFORMER_ACTIONS
    from retro_harness.env import make_env

    env = make_env(
        game=config.game_name,
        state=config.start_state,
        game_dir=config.game_dir,
        render_mode="rgb_array",
    )
    obs, _ = env.reset()

    if not pygame.get_init():
        pygame.init()

    scale = 2
    width, height = obs.shape[1], obs.shape[0]
    screen = pygame.display.set_mode((width * scale, height * scale), pygame.SWSURFACE)
    status = "COMPLETE" if best.result and best.result.completed else "incomplete"
    pygame.display.set_caption(f"Gen {gen} | fitness={best.fitness:.0f} | {status}")
    clock = pygame.time.Clock()

    max_frames = best.result.total_frames if best.result else len(best.actions)
    running = True

    for frame_idx in range(min(max_frames, len(best.actions))):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
        if not running:
            break

        buttons = action_index_to_buttons(best.actions[frame_idx], action_table)
        action_size = env.action_space.shape[0]
        if len(buttons) < action_size:
            buttons = buttons + [0] * (action_size - len(buttons))
        obs, *_ = env.step(np.array(buttons, dtype=np.int8))

        surf = pygame.surfarray.make_surface(obs.swapaxes(0, 1))
        surf = pygame.transform.scale(surf, (width * scale, height * scale))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        clock.tick(60)

    env.close()
    # Don't quit pygame - keep the window for next render
    return running


def _save_checkpoint(
    best: Individual, gen: int, output_dir: Path,
    evaluator: Evaluator | None = None,
) -> None:
    """Save best individual as checkpoint + first/last frame screenshots."""
    path = output_dir / f"ga_gen{gen:04d}_best.json"
    data = {
        "actions": best.actions,
        "num_frames": len(best.actions),
        "fitness": best.fitness,
        "generation": gen,
        "completed": best.result.completed if best.result else False,
        "total_frames": best.result.total_frames if best.result else 0,
        "max_progress": best.result.max_progress if best.result else 0,
    }
    path.write_text(json.dumps(data, indent=2))

    # Save first + last frame screenshots for quick visual verification
    if evaluator is not None:
        try:
            _save_screenshots(best, gen, output_dir, evaluator)
        except Exception as e:
            print(f"[GA] Screenshot failed: {e}")


def _save_screenshots(
    best: Individual, gen: int, output_dir: Path, evaluator: Evaluator,
) -> None:
    """Replay best individual using evaluator's env and save first + last frame as PNG."""
    import numpy as np
    from PIL import Image
    from platformer_common.actions import action_index_to_buttons, DEFAULT_PLATFORMER_ACTIONS

    config = evaluator.config
    action_table = config.action_table or DEFAULT_PLATFORMER_ACTIONS

    # Reuse evaluator's env (stable-retro only allows one per process)
    evaluator._ensure_env()
    env = evaluator._env
    assert env is not None and evaluator._cached_state is not None
    env.em.set_state(evaluator._cached_state)

    obs = env.render()
    first_frame = obs.copy()

    max_frames = best.result.total_frames if best.result else len(best.actions)
    last_frame = first_frame

    for frame_idx in range(min(max_frames, len(best.actions))):
        buttons = action_index_to_buttons(best.actions[frame_idx], action_table)
        action_size = env.action_space.shape[0]
        if len(buttons) < action_size:
            buttons = buttons + [0] * (action_size - len(buttons))
        obs, *_ = env.step(np.array(buttons, dtype=np.int8))
        last_frame = obs

    prefix = output_dir / f"ga_gen{gen:04d}"
    Image.fromarray(first_frame).save(f"{prefix}_first.png")
    Image.fromarray(last_frame).save(f"{prefix}_last.png")
