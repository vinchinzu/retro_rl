"""Optimization commands: GA, hillclimb, neuro, seed analysis/trim."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from retro_harness.platformer.actions import buttons_to_action_index
from retro_harness.platformer.bk2_extract import load_actions, save_actions
from retro_harness.platformer.cli.helpers import _get_action_table, _resolve_config
from retro_harness.platformer.evaluator import Evaluator


def _load_seeds_from_dir(
    seeds_dir: Path,
    min_frames: int = 60,
) -> list[list[int]]:
    """Load all attempt_*.json and recording_*.json from a directory as action-index lists."""
    seed_files = sorted(
        list(seeds_dir.glob("attempt_*.json"))
        + list(seeds_dir.glob("recording_*.json"))
    )
    # Exclude raw companion files
    seed_files = [f for f in seed_files if "_raw" not in f.stem]

    seeds: list[list[int]] = []
    for f in seed_files:
        try:
            actions = load_actions(f)
            if len(actions) >= min_frames:
                seeds.append(actions)
                print(f"  Loaded {f.name}: {len(actions)} frames")
            else:
                print(f"  Skipped {f.name}: {len(actions)} frames (< {min_frames} min)")
        except Exception as e:
            print(f"  Error loading {f.name}: {e}")
    return seeds


def _load_raw_seeds_from_dir(
    seeds_dir: Path,
    min_frames: int = 60,
) -> list[list[list[int]]]:
    """Load raw button arrays from attempt/recording files in a directory.

    Prefers companion _raw.json files, falls back to embedded raw_buttons.
    Skips files with no raw data available.
    """
    from retro_harness.platformer.bk2_extract import load_raw_buttons

    seed_files = sorted(
        list(seeds_dir.glob("attempt_*.json"))
        + list(seeds_dir.glob("recording_*.json"))
    )
    seed_files = [f for f in seed_files if "_raw" not in f.stem]

    seeds: list[list[list[int]]] = []
    for f in seed_files:
        try:
            raw = load_raw_buttons(f)
            if raw is None:
                continue
            if len(raw) >= min_frames:
                seeds.append(raw)
                print(f"  Loaded {f.name}: {len(raw)} raw frames")
            else:
                print(f"  Skipped {f.name}: {len(raw)} raw frames (< {min_frames} min)")
        except Exception as e:
            print(f"  Error loading {f.name}: {e}")
    return seeds


def cmd_optimize(args: argparse.Namespace) -> None:
    """Run GA optimization on seed action sequence(s)."""
    config = _resolve_config(args)
    start_state = getattr(args, "state", None)
    use_raw = getattr(args, "raw", False)

    seeds_dir = Path(args.seeds_dir) if args.seeds_dir else None
    seed_path = Path(args.seed) if args.seed else None

    if seeds_dir and seed_path:
        print("Error: --seed and --seeds-dir are mutually exclusive.")
        return
    if not seeds_dir and not seed_path:
        print("Error: provide --seed or --seeds-dir.")
        return

    print(f"Level: {config.display_name}")
    min_frames = args.min_frames

    output_dir = Path(args.output_dir) if args.output_dir else config.runs_dir
    evaluator = Evaluator(config, start_state=start_state)

    if use_raw:
        # Raw-button GA (no lossy action-index conversion)
        from retro_harness.platformer.genetic import run_ga_raw
        from retro_harness.platformer.bk2_extract import load_raw_buttons

        if seeds_dir:
            if not seeds_dir.exists():
                print(f"Error: seeds directory not found: {seeds_dir}")
                return
            print(f"Loading RAW seeds from {seeds_dir} (min {min_frames} frames)...")
            raw_seeds = _load_raw_seeds_from_dir(seeds_dir, min_frames=min_frames)
            if not raw_seeds:
                print("Error: no valid raw seeds found. Ensure _raw.json companion files exist.")
                return
            print(f"Loaded {len(raw_seeds)} raw seeds")
        else:
            raw = load_raw_buttons(seed_path)
            if raw is None:
                print(f"Error: no raw_buttons in {seed_path}")
                return
            raw_seeds = [raw]
            print(f"Seed: {len(raw)} raw frames from {seed_path}")

        best = run_ga_raw(
            seeds=raw_seeds,
            evaluator=evaluator,
            population_size=args.population,
            num_generations=args.generations,
            output_dir=output_dir,
        )

        final_path = output_dir / "ga_raw_best_final.json"
        data = {
            "raw_buttons": best.actions,
            "num_frames": len(best.actions),
            "fitness": best.fitness,
            "completed": best.result.completed if best.result else False,
            "total_frames": best.result.total_frames if best.result else 0,
            "max_progress": best.result.max_progress if best.result else 0,
            "level": config.level_id,
        }
        final_path.write_text(json.dumps(data, indent=2))
        print(f"\nSaved best to {final_path}")
    else:
        # Standard action-index GA
        from retro_harness.platformer.genetic import run_ga

        if seeds_dir:
            if not seeds_dir.exists():
                print(f"Error: seeds directory not found: {seeds_dir}")
                return
            print(f"Loading seeds from {seeds_dir} (min {min_frames} frames)...")
            all_seeds = _load_seeds_from_dir(seeds_dir, min_frames=min_frames)
            if not all_seeds:
                print("Error: no valid seeds found.")
                return
            print(f"Loaded {len(all_seeds)} seeds")
            seed_actions: list[int] | list[list[int]] = all_seeds
        else:
            if not seed_path.exists():
                print(f"Error: seed file not found: {seed_path}")
                return
            seed_actions = load_actions(seed_path)
            print(f"Seed: {len(seed_actions)} frames from {seed_path}")

        resume_from = Path(args.resume) if args.resume else None

        best = run_ga(
            seed_actions=seed_actions,
            evaluator=evaluator,
            population_size=args.population,
            num_generations=args.generations,
            output_dir=output_dir,
            num_workers=args.workers,
            resume_from=resume_from,
            render_interval=args.render,
        )

        final_path = output_dir / "ga_best_final.json"
        data = {
            "actions": best.actions,
            "num_frames": len(best.actions),
            "fitness": best.fitness,
            "completed": best.result.completed if best.result else False,
            "total_frames": best.result.total_frames if best.result else 0,
            "max_progress": best.result.max_progress if best.result else 0,
            "level": config.level_id,
        }
        final_path.write_text(json.dumps(data, indent=2))
        print(f"\nSaved best to {final_path}")

    evaluator.close()


def _parse_window(spec: str | None) -> tuple[int, int] | None:
    """Parse ``START:END`` window spec into an inclusive-exclusive range."""
    if not spec:
        return None
    text = spec.strip()
    if ":" not in text:
        raise SystemExit(f"window must be START:END, got {spec!r}")
    left, right = text.split(":", 1)
    return int(left), int(right)


def cmd_hillclimb_raw(args: argparse.Namespace) -> None:
    """Hill climb with raw button mutation (no lossy action-index conversion)."""
    from retro_harness.platformer.hillclimb_raw import hillclimb_raw
    from retro_harness.platformer.bk2_extract import load_raw_buttons

    config = _resolve_config(args)
    start_state = getattr(args, "state", None)
    seed_path = Path(args.seed)
    if not seed_path.exists():
        print(f"Error: seed file not found: {seed_path}")
        return

    raw = load_raw_buttons(seed_path)
    if raw is None:
        print(f"Error: no raw_buttons in {seed_path}")
        return

    window = _parse_window(getattr(args, "window", None))
    prefer_trim = bool(getattr(args, "prefer_trim", False))
    require_completion = bool(getattr(args, "require_completion", False))

    print(f"Seed: {len(raw)} raw frames from {seed_path}")
    print(f"Level: {config.display_name}")
    if start_state:
        print(f"State override: {start_state}")
    if args.entry_corpus:
        print(f"Entry corpus (train split): {args.entry_corpus}")
    if window:
        print(f"Window: [{window[0]}:{window[1]}]")
    if prefer_trim or require_completion:
        print(
            f"Frame-save mode: prefer_trim={prefer_trim} "
            f"require_completion={require_completion}"
        )

    output_dir = Path(args.output_dir) if args.output_dir else config.runs_dir
    evaluator = Evaluator(config, start_state=start_state)

    best_raw, best_result = hillclimb_raw(
        raw_buttons=raw,
        evaluator=evaluator,
        max_iterations=args.iterations,
        output_dir=output_dir,
        window=window,
        prefer_trim=prefer_trim,
        require_completion=require_completion,
    )

    out_name = (
        "segment_hillclimb_best.json"
        if (window or prefer_trim or require_completion)
        else "hillclimb_raw_best.json"
    )
    # segment engine always writes segment_hillclimb_best.json; classic writes hillclimb_raw_best
    saved = output_dir / out_name
    if not saved.exists():
        saved = output_dir / "hillclimb_raw_best.json"
    if not saved.exists():
        saved = output_dir / "segment_hillclimb_best.json"
    print(f"\nSaved best to {saved} ({len(best_raw)} frames, completed={best_result.completed})")
    evaluator.close()


def cmd_analyze_seed(args: argparse.Namespace) -> None:
    """Report idle prefixes, hold stalls, and (optionally) live clear frames."""
    from retro_harness.platformer.frame_tools import (
        analyze_seed_static,
        load_raw_frames,
    )

    seed_path = Path(args.seed)
    frames = load_raw_frames(seed_path)
    report = analyze_seed_static(frames)
    report["seed"] = str(seed_path)

    if not getattr(args, "static_only", False):
        config = _resolve_config(args)
        start_state = getattr(args, "state", None)
        evaluator = Evaluator(config, start_state=start_state)
        result = evaluator.evaluate(frames, early_terminate=False)
        report["completed"] = result.completed
        report["clear_frames"] = result.total_frames if result.completed else None
        report["fitness"] = result.fitness
        report["max_progress"] = result.max_progress
        report["died"] = result.died
        evaluator.close()

    print(json.dumps(report, indent=2))
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {out}")


def cmd_trim_seed(args: argparse.Namespace) -> None:
    """Trim leading idle / trailing post-clear pad / compress long holds."""
    from retro_harness.platformer.frame_tools import (
        load_raw_frames,
        save_raw_seed,
        search_hold_compressions,
        trim_after_completion,
        trim_leading_idle,
    )

    config = _resolve_config(args)
    start_state = getattr(args, "state", None)
    seed_path = Path(args.seed)
    frames = load_raw_frames(seed_path)
    evaluator = Evaluator(config, start_state=start_state)

    def evaluate(candidate: list[list[int]]):
        return evaluator.evaluate(candidate, early_terminate=False)

    current = frames
    notes: list[str] = []
    print(f"Seed: {len(current)} frames from {seed_path}")
    print(f"Level: {config.display_name}")

    if not args.no_leading:
        result = trim_leading_idle(
            current,
            evaluate,
            max_trim=args.max_leading,
            step=args.step,
            parity=args.parity,
            require_completion=not args.allow_incomplete,
            verbose=True,
        )
        print(
            f"Leading trim={result.trim} clear={result.clear_frames} "
            f"completed={result.completed} seed_len={len(result.frames)}"
        )
        current = result.frames
        notes.append(result.notes)

    if args.holds:
        result = search_hold_compressions(
            current,
            evaluate,
            min_hold=args.min_hold,
            require_completion=not args.allow_incomplete,
            verbose=True,
        )
        print(
            f"Hold compress: trim={result.trim} clear={result.clear_frames} "
            f"completed={result.completed} seed_len={len(result.frames)}"
        )
        current = result.frames
        notes.append(result.notes)

    if not args.no_trailing:
        result = trim_after_completion(
            current,
            evaluate,
            pad=args.pad,
            verbose=True,
        )
        print(
            f"Trailing trim={result.trim} clear={result.clear_frames} "
            f"completed={result.completed} seed_len={len(result.frames)}"
        )
        current = result.frames
        notes.append(result.notes)

    final = evaluate(current)
    output = Path(args.output) if args.output else (
        Path(args.output_dir) if args.output_dir else config.runs_dir
    ) / f"{seed_path.stem}_trimmed.json"
    if output.suffix != ".json":
        output = output / f"{seed_path.stem}_trimmed.json"

    save_raw_seed(
        output,
        current,
        metadata={
            "source": str(seed_path),
            "completed": final.completed,
            "total_frames": final.total_frames,
            "fitness": final.fitness,
            "max_progress": final.max_progress,
            "trim_notes": notes,
            "level": config.level_id,
        },
    )
    print(
        f"Wrote {output}: {len(frames)} -> {len(current)} frames "
        f"(clear={final.total_frames if final.completed else None}, "
        f"completed={final.completed})"
    )
    evaluator.close()


def cmd_segment_hillclimb(args: argparse.Namespace) -> None:
    """Checkpoint-accelerated hillclimb inside a frame window."""
    from retro_harness.platformer.segment_hillclimb import segment_hillclimb_raw
    from retro_harness.platformer.frame_tools import load_raw_frames, save_raw_seed

    config = _resolve_config(args)
    start_state = getattr(args, "state", None)
    seed_path = Path(args.seed)
    frames = load_raw_frames(seed_path)
    window = _parse_window(args.window)
    if window is None:
        raise SystemExit("--window START:END is required for segment-hillclimb")

    output_dir = Path(args.output_dir) if args.output_dir else config.runs_dir / "segment_hc"
    evaluator = Evaluator(config, start_state=start_state)
    print(f"Seed: {len(frames)} frames from {seed_path}")
    print(f"Level: {config.display_name}")
    print(f"Window: [{window[0]}:{window[1]}]")

    best, result = segment_hillclimb_raw(
        frames,
        evaluator,
        window=window,
        max_iterations=args.iterations,
        prefer_trim=not args.no_prefer_trim,
        require_completion=not args.allow_incomplete,
        output_dir=output_dir,
        verbose=True,
    )
    out = output_dir / "segment_hillclimb_best.json"
    # Also write a copy with metadata if the engine file exists
    if out.exists():
        print(f"Saved {out}")
    else:
        save_raw_seed(
            out,
            best,
            metadata={
                "completed": result.completed,
                "total_frames": result.total_frames,
                "fitness": result.fitness,
                "window": list(window),
                "level": config.level_id,
            },
        )
        print(f"Saved {out}")
    evaluator.close()


def cmd_hillclimb(args: argparse.Namespace) -> None:
    """Run hill climbing refinement on an action sequence."""
    from retro_harness.platformer.hillclimb import hillclimb
    from retro_harness.platformer.bk2_extract import load_raw_buttons

    config = _resolve_config(args)
    start_state = getattr(args, "state", None)
    seed_path = Path(args.seed)
    if not seed_path.exists():
        print(f"Error: seed file not found: {seed_path}")
        return

    # Prefer raw buttons if available (faithful replay)
    raw = load_raw_buttons(seed_path)
    if raw is not None:
        # Convert raw buttons to action indices for hill climbing
        # (hill climber mutates action indices, not raw buttons)
        seed_actions = [
            buttons_to_action_index(frame, action_table=_get_action_table(config))
            for frame in raw
        ]
        print(f"Seed: {len(seed_actions)} frames (from raw buttons) from {seed_path}")
    else:
        seed_actions = load_actions(seed_path)
        print(f"Seed: {len(seed_actions)} frames from {seed_path}")
    print(f"Level: {config.display_name}")
    if start_state:
        print(f"State override: {start_state}")

    output_dir = Path(args.output_dir) if args.output_dir else config.runs_dir
    evaluator = Evaluator(config, start_state=start_state)

    best_actions, best_result = hillclimb(
        actions=seed_actions,
        evaluator=evaluator,
        max_iterations=args.iterations,
        output_dir=output_dir,
        render_interval=args.render,
        render_scale=args.scale,
    )

    final_path = output_dir / "hillclimb_best_final.json"
    data = {
        "actions": best_actions,
        "num_frames": len(best_actions),
        "fitness": best_result.fitness,
        "completed": best_result.completed,
        "total_frames": best_result.total_frames,
        "max_x": best_result.max_x,
        "max_progress": best_result.max_progress,
        "bonus_frames": best_result.bonus_frames,
        "level": config.level_id,
    }
    final_path.write_text(json.dumps(data, indent=2))
    print(f"\nSaved best to {final_path}")

    evaluator.close()


def cmd_neuro(args: argparse.Namespace) -> None:
    """Run neuroevolution optimizer (evolve neural networks to play the level)."""
    from retro_harness.platformer.neuro import run_neuro_ga

    config = _resolve_config(args)
    start_state = getattr(args, "state", None)

    output_dir = Path(args.output_dir) if args.output_dir else config.runs_dir / "neuro"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Level: {config.display_name}")
    print(f"Population: {args.population}, Generations: {args.generations}")
    print(f"Hidden neurons: {args.hidden}, Max frames: {args.max_frames}")
    if start_state:
        print(f"State override: {start_state}")
    print(f"Output: {output_dir}")

    evaluator = Evaluator(config, start_state=start_state)

    best = run_neuro_ga(
        evaluator=evaluator,
        population_size=args.population,
        num_generations=args.generations,
        n_hidden=args.hidden,
        max_frames=args.max_frames,
        output_dir=output_dir,
        render=getattr(args, "render", False),
        render_scale=getattr(args, "scale", 3),
        entry_corpus_path=(Path(args.entry_corpus) if args.entry_corpus else None),
        obs_fn=config.neuro_observation_fn,
    )

    print(f"\nBest network: fitness={best.fitness:.1f}")
    if best.result:
        print(f"  progress={best.result.max_progress:.1f}, frames={best.result.total_frames}")
        print(f"  completed={best.result.completed}")
    print(f"Checkpoints saved in: {output_dir}")
    print(f"  neuro_best.json        (network weights)")
    print(f"  neuro_best_buttons.json (button replay for watch/verify)")

    evaluator.close()
