#!/usr/bin/env python3
"""Multi-seed tournament: hill climb multiple seeds for the same segment.

Tries multiple seeds for the same segment, hill climbs each, keeps the best.
Seeds can come from recordings, synthetic generation, or previous hill climbs.

Usage:
    uv run python -m super_metroid_rl.scripts.tournament \
        --segment parlor_descent \
        --seeds recording:segments/seg01_parlor.json synthetic \
        --iterations 500 \
        --final-iterations 1000

Seed source formats:
    recording:path.json        Load raw_buttons from a segment JSON file
    synthetic                  Generate via seed_synth.py
    hillclimb:path.json        Load from a hill climb result file
    file:path.json             Load raw_buttons from any JSON file

The tournament hill climbs each seed for --iterations steps, then
optionally runs --final-iterations on the overall winner.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_seed(source: str, segment_id: str) -> dict:
    """Load a seed from a source descriptor.

    Args:
        source: Seed source string (e.g. "synthetic", "recording:foo.json")
        segment_id: Level config ID for the segment being optimized

    Returns:
        Dict with 'name', 'raw_buttons', 'source' keys.
    """
    if source == "synthetic":
        from super_metroid_rl.navigation.seed_synth import synthesize_seed
        try:
            from super_metroid_rl.navigation.map_data import load_world
            world = load_world()
        except Exception:
            world = None
        buttons = synthesize_seed(segment_id, world)
        return {
            "name": "synthetic",
            "raw_buttons": buttons,
            "source": "seed_synth",
        }

    # Parse prefix:path format
    if ":" in source:
        prefix, path_str = source.split(":", 1)
    else:
        prefix, path_str = "file", source

    path = Path(path_str)
    if not path.is_absolute():
        # Search relative to cwd, then project root
        candidates = [
            Path.cwd() / path,
            PROJECT_ROOT / "super_metroid_rl" / path,
            PROJECT_ROOT / path,
        ]
        for c in candidates:
            if c.exists():
                path = c
                break

    if not path.exists():
        raise FileNotFoundError(f"Seed file not found: {path_str}")

    data = json.loads(path.read_text())
    raw_buttons = data.get("raw_buttons", [])
    if not raw_buttons:
        raise ValueError(f"No raw_buttons in {path}")

    return {
        "name": f"{prefix}:{path.stem}",
        "raw_buttons": raw_buttons,
        "source": str(path),
    }


def tournament(
    config_id: str,
    seeds: list[dict],
    iterations_per_seed: int = 500,
    final_iterations: int = 0,
    output_dir: Path | None = None,
    start_state: str | None = None,
    verbose: bool = True,
) -> dict:
    """Hill climb multiple seeds in sequence, return best overall result.

    Args:
        config_id: Level config ID (e.g. "sm_parlor_descent")
        seeds: List of seed dicts with 'name', 'raw_buttons', 'source'
        iterations_per_seed: Hill climb iterations per seed
        final_iterations: Extra iterations on the overall winner (0 = skip)
        output_dir: Where to save results
        start_state: Override start state (default: from level config)
        verbose: Print progress

    Returns:
        Dict with best result info: name, fitness, frames, completed, path
    """
    import platformer_common.levels.super_metroid  # noqa: F401
    from platformer_common.evaluator import Evaluator
    from platformer_common.hillclimb_raw import hillclimb_raw
    from platformer_common.level_config import get_level_config

    config = get_level_config(config_id)
    if output_dir is None:
        output_dir = config.runs_dir / "tournament"
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Tournament: {config_id}")
        print(f"Seeds: {len(seeds)}")
        print(f"Iterations per seed: {iterations_per_seed}")
        if final_iterations:
            print(f"Final iterations: {final_iterations}")
        print(f"Output: {output_dir}")
        print()

    best_overall = None
    best_fitness = float("-inf")
    best_name = ""
    best_buttons = []
    start_time = time.time()

    for i, seed in enumerate(seeds):
        name = seed["name"]
        raw = seed["raw_buttons"]

        if verbose:
            print(f"--- Seed {i + 1}/{len(seeds)}: {name} ({len(raw)} frames) ---")

        # Create evaluator (fresh for each seed to avoid state leaks)
        ev = Evaluator(config, start_state=start_state)

        # Initial evaluation
        initial = ev.evaluate(raw, early_terminate=False)
        if verbose:
            status = "COMPLETE" if initial.completed else "incomplete"
            print(f"  Initial: fitness={initial.fitness:.1f} frames={initial.total_frames} "
                  f"progress={initial.max_progress:.1f} {status}")

        if initial.fitness <= 0:
            if verbose:
                print(f"  SKIP: zero/negative fitness, seed unusable")
            ev.close()
            continue

        # Hill climb this seed
        seed_dir = output_dir / f"seed_{i:02d}_{name.replace(':', '_')}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        optimized, result = hillclimb_raw(
            raw, ev, max_iterations=iterations_per_seed,
            output_dir=seed_dir, verbose=verbose,
        )

        if verbose:
            delta = len(optimized) - len(raw)
            print(f"  Result: fitness={result.fitness:.1f} frames={len(optimized)} ({delta:+d})")
            print()

        if result.fitness > best_fitness:
            best_fitness = result.fitness
            best_overall = result
            best_name = name
            best_buttons = optimized

        ev.close()

    if best_overall is None:
        if verbose:
            print("ERROR: No seed produced a valid result")
        return {"error": "no valid seeds"}

    if verbose:
        elapsed = time.time() - start_time
        print(f"=== Best seed: {best_name} ===")
        status = "COMPLETE" if best_overall.completed else "incomplete"
        print(f"  fitness={best_fitness:.1f} frames={len(best_buttons)} {status}")
        print(f"  Time: {elapsed:.1f}s")

    # Final polish on the winner
    if final_iterations > 0 and best_buttons:
        if verbose:
            print(f"\n--- Final polish: {final_iterations} iterations on {best_name} ---")
        ev = Evaluator(config, start_state=start_state)
        final_dir = output_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)

        best_buttons, best_overall = hillclimb_raw(
            best_buttons, ev, max_iterations=final_iterations,
            output_dir=final_dir, verbose=verbose,
        )
        best_fitness = best_overall.fitness
        ev.close()

    # Save tournament winner
    winner_path = output_dir / "tournament_best.json"
    winner_data = {
        "raw_buttons": best_buttons,
        "num_frames": len(best_buttons),
        "fitness": best_fitness,
        "completed": best_overall.completed,
        "total_frames": best_overall.total_frames,
        "max_progress": best_overall.max_progress,
        "winning_seed": best_name,
        "config_id": config_id,
        "iterations_per_seed": iterations_per_seed,
        "final_iterations": final_iterations,
        "num_seeds": len(seeds),
    }
    winner_path.write_text(json.dumps(winner_data, indent=2))

    if verbose:
        print(f"\nTournament winner saved: {winner_path}")
        print(f"Winner: {best_name} | fitness={best_fitness:.1f} | "
              f"frames={len(best_buttons)} | "
              f"{'COMPLETE' if best_overall.completed else 'incomplete'}")

    return {
        "name": best_name,
        "fitness": best_fitness,
        "frames": len(best_buttons),
        "completed": best_overall.completed,
        "path": str(winner_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Multi-seed hill climb tournament",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Seed source formats:
  synthetic                  Generate from collision data
  recording:path.json        Load from a segment JSON file
  hillclimb:path.json        Load from a hill climb result
  file:path.json             Load raw_buttons from any JSON
        """,
    )
    parser.add_argument("--segment", required=True,
                        help="Segment ID (e.g. parlor_descent or sm_parlor_descent)")
    parser.add_argument("--seeds", nargs="+", required=True,
                        help="Seed sources (e.g. synthetic recording:seg01.json)")
    parser.add_argument("--iterations", type=int, default=500,
                        help="Hill climb iterations per seed (default: 500)")
    parser.add_argument("--final-iterations", type=int, default=0,
                        help="Extra iterations on the winner (default: 0)")
    parser.add_argument("--output-dir", help="Output directory")
    parser.add_argument("--start-state", help="Override start state name")
    args = parser.parse_args()

    segment = args.segment
    if not segment.startswith("sm_"):
        segment = "sm_" + segment

    # Load all seeds
    print(f"Loading {len(args.seeds)} seeds for {segment}...")
    seeds = []
    for source in args.seeds:
        try:
            seed = load_seed(source, segment)
            seeds.append(seed)
            print(f"  {seed['name']}: {len(seed['raw_buttons'])} frames")
        except Exception as e:
            print(f"  ERROR loading '{source}': {e}")

    if not seeds:
        print("No valid seeds loaded!")
        sys.exit(1)

    out_dir = Path(args.output_dir) if args.output_dir else None

    result = tournament(
        config_id=segment,
        seeds=seeds,
        iterations_per_seed=args.iterations,
        final_iterations=args.final_iterations,
        output_dir=out_dir,
        start_state=args.start_state,
    )

    sys.exit(0 if result.get("completed") else 1)


if __name__ == "__main__":
    main()
