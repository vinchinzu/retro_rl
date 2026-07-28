"""CLI entry point for platformer speedrun optimizer.

All commands take a --level flag to select the level to optimize.
Level configs are registered by importing platformer_common.levels.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Trigger level registration on import
import platformer_common.levels  # noqa: F401

from platformer_common.level_config import LevelConfig, get_level_config, list_levels
from platformer_common.actions import (
    DEFAULT_PLATFORMER_ACTIONS,
    action_index_to_buttons,
    buttons_to_action_index,
)
from platformer_common.bk2_extract import (
    extract_action_indices_from_bk2,
    extract_raw_actions_from_bk2,
    save_actions,
    load_actions,
)
from platformer_common.evaluator import Evaluator


def _resolve_config(args: argparse.Namespace) -> LevelConfig:
    """Get level config from --level arg."""
    return get_level_config(args.level)


def _get_action_table(config: LevelConfig) -> list[list[int]]:
    return config.action_table or DEFAULT_PLATFORMER_ACTIONS


def _parse_room_id_arg(value: int | str) -> int:
    """Parse a decimal or hexadecimal room ID from CLI or wrapper arguments."""
    if isinstance(value, int):
        return value
    text = value.strip()
    try:
        return int(text, 0)
    except ValueError:
        return int(text, 16)


def _practice_completion_token(
    config: LevelConfig,
    values: dict[str, int],
    progress: float,
) -> tuple[str, int] | None:
    """Return a stable token while the configured completion signal is active."""
    if progress < config.completion_min_progress:
        return None
    if config.completion_signal == "ram_flag":
        key = config.completion_ram_key
        value = values.get(key) if key else None
        if value == config.completion_ram_value:
            return ("ram_flag", int(value))
        return None

    level_id = int(values.get("level_id", 0) or 0)
    main_level_ids = {config.target_level_id, *config.level_id_aliases}
    if level_id == 0 or level_id in main_level_ids:
        return None
    if config.completion_level_ids and level_id not in config.completion_level_ids:
        return None
    if level_id in config.completion_exclude_ids:
        return None
    return ("level_id", level_id)


def _load_practice_pb_frames(practice_dir: Path) -> int | None:
    """Load the fastest completed attempt from an existing practice directory."""
    best: int | None = None
    for path in practice_dir.glob("attempt_*.json"):
        if path.stem.endswith("_raw"):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            metadata = data.get("metadata", {})
            if not metadata.get("completed"):
                continue
            frames = int(metadata.get("total_frames", data.get("num_frames", 0)))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        if frames > 0 and (best is None or frames < best):
            best = frames
    return best


def _best_practice_attempt(attempts: list[dict]) -> int:
    """Choose fastest completion, falling back to furthest partial attempt."""
    completed = [attempt for attempt in attempts if attempt["completed"]]
    if completed:
        return min(completed, key=lambda attempt: attempt["frames"])["attempt"]
    if attempts:
        return max(attempts, key=lambda attempt: attempt["max_progress"])["attempt"]
    return -1


# -- Commands ----------------------------------------------------------------


def cmd_list_levels(args: argparse.Namespace) -> None:
    """List all registered levels."""
    from platformer_common.level_config import LEVEL_REGISTRY

    levels = list_levels()
    if not levels:
        print("No levels registered.")
        return

    print(f"{'ID':<30s} {'Display Name':<30s} {'Game':<30s} {'State'}")
    print("-" * 120)
    for cfg in levels:
        print(f"{cfg.level_id:<30s} {cfg.display_name:<30s} {cfg.game_name:<30s} {cfg.start_state}")

    # Show aliases
    print(f"\nAliases:")
    for alias, cfg in sorted(LEVEL_REGISTRY.items()):
        if alias != cfg.level_id:
            print(f"  {alias} -> {cfg.level_id}")


def cmd_extract(args: argparse.Namespace) -> None:
    """Extract action sequence from a bk2 recording."""
    config = _resolve_config(args)
    bk2_path = Path(args.bk2)
    if not bk2_path.exists():
        print(f"Error: bk2 file not found: {bk2_path}")
        return

    action_table = _get_action_table(config)
    print(f"Extracting from: {bk2_path}")

    raw = extract_raw_actions_from_bk2(bk2_path, bk2_to_env=config.bk2_to_env)
    print(f"Total raw frames: {len(raw)}")

    if args.raw_preview:
        print("\nFirst 10 raw frames (env button order: B Y Sel Sta U D L R A X L R):")
        for i, frame in enumerate(raw[:10]):
            print(f"  {i:4d}: {frame}")

    actions = extract_action_indices_from_bk2(
        bk2_path, action_table=action_table, bk2_to_env=config.bk2_to_env
    )
    print(f"Action indices: {len(actions)} frames")

    # Action distribution
    from collections import Counter

    dist = Counter(actions)
    num_actions = len(action_table)
    print(f"\nAction distribution ({num_actions} actions):")
    for idx in sorted(dist.keys()):
        print(f"  {idx:2d}: {dist[idx]:5d} frames ({dist[idx]/len(actions)*100:.1f}%)")

    output = Path(args.output) if args.output else config.runs_dir / f"{bk2_path.parent.name}_extracted.json"
    metadata = {"source_bk2": str(bk2_path), "raw_frames": len(raw), "level": config.level_id}
    save_actions(actions, output, metadata=metadata)


def _recording_start_state(path: Path) -> str | None:
    """Return an unambiguous start state embedded in a recording JSON."""
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    metadata = payload.get("metadata")
    candidates = [
        payload.get("state"),
        payload.get("start_state"),
        metadata.get("state") if isinstance(metadata, dict) else None,
    ]
    states = {state for state in candidates if isinstance(state, str) and state}
    return next(iter(states)) if len(states) == 1 else None


def cmd_verify(args: argparse.Namespace) -> None:
    """Verify an action sequence by replaying it headlessly."""
    from platformer_common.bk2_extract import load_raw_buttons

    config = _resolve_config(args)
    explicit_state = getattr(args, "state", None)
    actions_path = Path(args.actions)
    if not actions_path.exists():
        print(f"Error: actions file not found: {actions_path}")
        return
    metadata_state = _recording_start_state(actions_path)
    start_state = explicit_state or metadata_state

    raw = load_raw_buttons(actions_path)
    if raw is not None:
        actions: list[int] | list[list[int]] = raw
        print(f"Loaded {len(actions)} frames (raw buttons) from {actions_path}")
    else:
        actions = load_actions(actions_path)
        print(f"Loaded {len(actions)} frames (action indices) from {actions_path}")
    print(f"Level: {config.display_name}")
    if explicit_state:
        print(f"State override: {start_state}")
    elif metadata_state:
        print(f"State from recording metadata: {start_state}")

    evaluator = Evaluator(config, start_state=start_state)

    if getattr(args, "trace", False):
        print("Tracing level_id changes (no early termination)...")
        start = time.time()
        result = evaluator.evaluate_trace(actions)
        elapsed = time.time() - start
    else:
        print("Evaluating (no early termination)...")
        start = time.time()
        result = evaluator.evaluate(actions, early_terminate=False)
        elapsed = time.time() - start

    gameplay_frames = result.total_frames - result.gameplay_start_frame
    print(f"\nResult:")
    print(f"  Completed:      {result.completed}")
    print(f"  Died:           {result.died}")
    print(f"  Total frames:   {result.total_frames}")
    print(f"  Gameplay start: frame {result.gameplay_start_frame}")
    print(f"  Gameplay frames:{gameplay_frames}")
    print(f"  Gameplay secs:  {gameplay_frames / 60:.2f}s")
    print(f"  Timer frames:   {result.timer_frames}")
    print(f"  Timer secs:     {result.timer_frames / 60:.2f}s")
    print(f"  Max X:          {result.max_x:.1f}")
    print(f"  Max progress:   {result.max_progress:.1f}")
    print(f"  Final pos:      ({result.final_x:.1f}, {result.final_y:.1f})")
    print(f"  Level ID end:   0x{result.level_id_at_end:02X} ({result.level_id_at_end})")
    print(f"  Bonus frames:   {result.bonus_frames}")
    print(f"  Fitness:        {result.fitness:.1f}")
    print(f"  Eval time:      {elapsed:.2f}s")

    evaluator.close()


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
    from platformer_common.bk2_extract import load_raw_buttons

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
        from platformer_common.genetic import run_ga_raw
        from platformer_common.bk2_extract import load_raw_buttons

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
        from platformer_common.genetic import run_ga

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
    from platformer_common.hillclimb_raw import hillclimb_raw
    from platformer_common.bk2_extract import load_raw_buttons

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
    from platformer_common.frame_tools import (
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
    from platformer_common.frame_tools import (
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
    from platformer_common.segment_hillclimb import segment_hillclimb_raw
    from platformer_common.frame_tools import load_raw_frames, save_raw_seed

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
    from platformer_common.hillclimb import hillclimb
    from platformer_common.bk2_extract import load_raw_buttons

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
    from platformer_common.neuro import run_neuro_ga

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
    )

    print(f"\nBest network: fitness={best.fitness:.1f}")
    if best.result:
        print(f"  progress={best.result.max_progress:.1f}, frames={best.result.total_frames}")
        print(f"  completed={best.result.completed}")
    print(f"Checkpoints saved in: {output_dir}")
    print(f"  neuro_best.json        (network weights)")
    print(f"  neuro_best_buttons.json (button replay for watch/verify)")

    evaluator.close()


def _button_names(buttons: list[int]) -> str:
    """Format a 12-element button array as compact pressed-button string."""
    names = ["B", "Y", "Sel", "Sta", "U", "D", "L", "R", "A", "X", "L1", "R1"]
    pressed = [names[i] for i in range(min(len(buttons), len(names))) if buttons[i]]
    return "+".join(pressed) if pressed else "-"


def _replay_with_hud(
    config,
    actions: list[int] | list[list[int]],
    scale: int = 3,
    title: str | None = None,
    start_state: str | None = None,
    actions_path: Path | None = None,
) -> None:
    """Shared replay logic with HUD overlay for watch/hillclimb render.

    Controls:
      SPACE       Pause/resume
      RIGHT/LEFT  Step forward/backward one frame (while paused)
      N           Add note at current frame (while paused)
      1-5         Toggle issue tags on current frame (while paused)
      [ / ]       Decrease/increase playback speed
      ESC         Quit (saves annotations and trace if any)
    """
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "x11")

    import numpy as np
    import pygame

    action_table = _get_action_table(config)

    from retro_harness.env import make_env

    env = make_env(
        game=config.game_name,
        state=start_state or config.start_state,
        game_dir=config.game_dir,
        render_mode="rgb_array",
    )
    obs, _ = env.reset()
    initial_obs = obs.copy()

    from platformer_common.progress import make_progress_tracker

    # Save emulator state for rewind
    initial_emu_state = env.em.get_state()

    schema = config.ram_schema
    tracker = make_progress_tracker(config)
    ram = env.get_ram()
    initial_values = schema.read(ram)
    config.apply_computed(initial_values)
    initial_lives = initial_values.get("lives")
    initial_cam_x = float(initial_values.get("camera_x", 0))
    # Seed tracker with initial values (same as evaluator)
    tracker.reset()
    tracker.update(initial_values)
    # For camera-based games, gate death detection on camera scroll;
    # for player-position games (SM), start immediately.
    gameplay_started = initial_cam_x == 0

    # -- Annotation constants --
    TAG_KEYS = {
        pygame.K_1: "ledge_hit",
        pygame.K_2: "bad_path",
        pygame.K_3: "slow",
        pygame.K_4: "good",
        pygame.K_5: "other",
    }
    TAG_COLORS = {
        "ledge_hit": (255, 80, 80),
        "bad_path": (255, 165, 0),
        "slow": (255, 255, 0),
        "good": (80, 255, 80),
        "other": (160, 160, 160),
    }
    SPEEDS = [0.25, 0.5, 1.0, 2.0, 4.0]
    speed_idx = 2  # 1.0x

    # -- Load existing annotations --
    annotations: dict[int, dict] = {}
    annotations_changed = False
    annotations_file = None
    if actions_path is not None:
        annotations_file = actions_path.parent / f"{actions_path.stem}_annotations.json"
        if annotations_file.exists():
            try:
                data = json.loads(annotations_file.read_text())
                for entry in data.get("annotations", []):
                    annotations[entry["frame"]] = {
                        "tags": list(entry.get("tags", [])),
                        "note": entry.get("note", ""),
                    }
                print(f"Loaded {len(annotations)} annotations from {annotations_file}")
            except Exception as e:
                print(f"Warning: could not load annotations: {e}")

    # -- Trace collection --
    trace: list[dict] = []
    trace_rooms: dict[int, dict] = {}  # room_id -> {enter_frame, last_frame}

    pygame.init()
    width, height = obs.shape[1], obs.shape[0]
    timeline_h = 8
    screen = pygame.display.set_mode(
        (width * scale, height * scale + timeline_h), pygame.SWSURFACE
    )
    pygame.display.set_caption(title or f"Replay: {config.display_name}")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 16)

    running = True
    paused = False
    text_input_mode = False
    text_input_buf = ""
    max_progress = 0.0
    raw_mode = len(actions) > 0 and isinstance(actions[0], list)
    total_frames = len(actions)
    current_frame = -1  # most recently executed frame (-1 = initial state)
    current_values = dict(initial_values)
    current_buttons: list[int] | None = None
    current_in_sub = False

    def _get_buttons(frame_i: int) -> list[int]:
        """Convert actions[frame_i] to padded button array."""
        act = actions[frame_i]
        if raw_mode:
            btns = list(act)  # type: ignore[arg-type]
        else:
            btns = action_index_to_buttons(act, action_table)  # type: ignore[arg-type]
        action_size = env.action_space.shape[0]
        if len(btns) < action_size:
            btns = btns + [0] * (action_size - len(btns))
        elif len(btns) > action_size:
            btns = btns[:action_size]
        return btns

    def _simulate_to(target: int) -> None:
        """Re-simulate from initial state through actions[0..target].

        Updates nonlocal: obs, tracker, max_progress, gameplay_started,
        current_values, current_buttons, current_in_sub, current_frame.
        """
        nonlocal obs, tracker, max_progress, gameplay_started
        nonlocal current_values, current_buttons, current_in_sub, current_frame

        env.em.set_state(initial_emu_state)
        tracker = make_progress_tracker(config)
        _ram = env.get_ram()
        _vals = schema.read(_ram)
        config.apply_computed(_vals)
        _init_cam = float(_vals.get("camera_x", 0))
        _gs = _init_cam == 0
        tracker.reset()
        tracker.update(_vals)
        _mp = 0.0

        if target < 0:
            obs = initial_obs.copy()
            current_values = dict(initial_values)
            current_buttons = None
            current_in_sub = False
            current_frame = -1
            max_progress = 0.0
            gameplay_started = _gs
            return

        _in_sub = False
        for i in range(target + 1):
            btns = _get_buttons(i)
            obs, _, _, _, _ = env.step(np.array(btns, dtype=np.int8))
            _ram = env.get_ram()
            _vals = schema.read(_ram)
            config.apply_computed(_vals)
            _cam = float(_vals.get("camera_x", 0))
            _lid = _vals.get("level_id", config.target_level_id)
            _mids = {config.target_level_id} | set(config.level_id_aliases)
            _in_sub = _lid != 0 and _lid not in _mids
            if not _in_sub:
                _p = tracker.update(_vals)
                if _p > _mp:
                    _mp = _p
            if not _gs and _cam > _init_cam:
                _gs = True

        current_frame = target
        current_values = _vals
        current_buttons = _get_buttons(target)
        current_in_sub = _in_sub
        max_progress = _mp
        gameplay_started = _gs

    def _step_one() -> bool:
        """Execute one frame forward. Returns True if a frame was played."""
        nonlocal obs, max_progress, gameplay_started, running
        nonlocal current_values, current_buttons, current_in_sub, current_frame

        next_f = current_frame + 1
        if next_f >= total_frames:
            return False

        btns = _get_buttons(next_f)
        obs, reward, terminated, truncated, info = env.step(
            np.array(btns, dtype=np.int8)
        )

        ram = env.get_ram()
        values = schema.read(ram)
        config.apply_computed(values)
        cam_x = float(values.get("camera_x", 0))
        level_id = values.get("level_id", config.target_level_id)

        _main_ids = {config.target_level_id} | set(config.level_id_aliases)
        in_sub = level_id != 0 and level_id not in _main_ids

        if not in_sub:
            progress = tracker.update(values)
            if progress > max_progress:
                max_progress = progress

        if not gameplay_started and cam_x > initial_cam_x:
            gameplay_started = True

        current_frame = next_f
        current_values = values
        current_buttons = btns
        current_in_sub = in_sub

        # Collect trace point
        btn_str = _button_names(btns) if btns else "-"
        px = int(values.get("player_x", 0))
        py = int(values.get("player_y", 0))
        trace_pt: dict = {
            "frame": current_frame,
            "room_id": level_id,
            "x": px,
            "y": py,
            "buttons": btn_str,
        }
        health_val = values.get("health")
        if health_val is not None:
            trace_pt["health"] = int(health_val)
        # Speed: x delta from previous frame (pixels/frame)
        if trace:
            prev_x = trace[-1].get("x", px)
            prev_room = trace[-1].get("room_id", level_id)
            # Only compute speed within same room (avoids door transition spikes)
            if prev_room == level_id:
                trace_pt["speed_x"] = px - prev_x
            else:
                trace_pt["speed_x"] = 0
        else:
            trace_pt["speed_x"] = 0
        trace.append(trace_pt)

        # Track room transitions
        if level_id not in trace_rooms:
            trace_rooms[level_id] = {"enter_frame": current_frame, "last_frame": current_frame}
        else:
            trace_rooms[level_id]["last_frame"] = current_frame

        # Check completion
        if config.completion_signal == "level_id_change":
            if level_id not in _main_ids and level_id != 0:
                is_real = (
                    max_progress >= config.completion_min_progress
                    and (not config.completion_level_ids
                         or level_id in config.completion_level_ids)
                    and level_id not in config.completion_exclude_ids
                )
                if is_real:
                    print(f"  COMPLETED at frame {current_frame}: level_id=0x{level_id:04X}, progress={max_progress:.0f}")
                    running = False
                    return True
        elif config.completion_signal == "ram_flag":
            flag_val = values.get(config.completion_ram_key, None)
            if (flag_val is not None
                    and flag_val == config.completion_ram_value
                    and max_progress >= config.completion_min_progress):
                print(f"  COMPLETED at frame {current_frame}: {config.completion_ram_key}={flag_val}, progress={max_progress:.0f}")
                running = False
                return True

        # Check death
        if gameplay_started:
            for signal in config.death_signals:
                if signal == "lives_drop":
                    lives = values.get("lives")
                    if initial_lives is not None and lives is not None and lives < initial_lives:
                        print(f"  DIED at frame {current_frame}: lives {initial_lives}->{lives}, progress={max_progress:.0f}")
                        running = False
                        return True
                elif signal == "health_zero":
                    health = values.get("health", 1)
                    if health <= 0:
                        print(f"  DIED at frame {current_frame}: health=0, progress={max_progress:.0f}")
                        running = False
                        return True
                elif signal == "camera_reset":
                    if in_sub:
                        continue
                    if initial_cam_x > config.camera_reset_threshold and cam_x < initial_cam_x - config.camera_reset_threshold:
                        print(f"  DIED at frame {current_frame}: camera reset ({cam_x:.0f} << {initial_cam_x:.0f})")
                        running = False
                        return True

        if terminated or truncated:
            print(f"Episode ended at frame {current_frame}")
            running = False
            return True

        return True

    def _draw() -> None:
        """Render current frame + HUD + timeline."""
        surf = pygame.surfarray.make_surface(obs.swapaxes(0, 1))
        screen.blit(
            pygame.transform.scale(surf, (width * scale, height * scale)),
            (0, 0),
        )

        # HUD text
        btn_str = _button_names(current_buttons) if current_buttons else "-"
        lives_val = current_values.get("lives", "?")
        health_val = current_values.get("health")
        bonus_tag = " | BONUS" if current_in_sub else ""
        speed_str = f" | {SPEEDS[speed_idx]}x" if speed_idx != 2 else ""
        frame_display = max(current_frame, 0)

        lines: list[str] = []
        if paused:
            lines.append("PAUSED  [N]note [1-5]tag [LEFT/RIGHT]step [[ ]]speed")
        lines.append(
            f"F{frame_display}/{total_frames} | {btn_str}{bonus_tag}{speed_str}"
        )
        lines.append(
            f"progress={max_progress:.0f} | "
            + (f"hp={health_val}" if health_val is not None else f"lives={lives_val}")
            + f" | cam={float(current_values.get('camera_x', 0)):.0f}"
        )

        # Show annotations for current frame
        if current_frame in annotations:
            ann = annotations[current_frame]
            tags = ann.get("tags", [])
            note = ann.get("note", "")
            if tags:
                lines.append(f"TAGS: {', '.join(tags)}")
            if note:
                lines.append(f"NOTE: {note}")

        for i, line in enumerate(lines):
            text = font.render(line, True, (255, 255, 0))
            screen.blit(text, (4, 4 + i * 18))

        # Text input bar
        if text_input_mode:
            bar_y = height * scale - 28
            pygame.draw.rect(screen, (0, 0, 100), (0, bar_y, width * scale, 28))
            prompt = font.render(f"Note: {text_input_buf}_", True, (255, 255, 255))
            screen.blit(prompt, (4, bar_y + 5))

        # Timeline bar
        bar_y = height * scale
        bar_w = width * scale
        pygame.draw.rect(screen, (30, 30, 30), (0, bar_y, bar_w, timeline_h))
        if total_frames > 0:
            for ann_f, ann_data in annotations.items():
                if 0 <= ann_f < total_frames:
                    x = int(ann_f / total_frames * bar_w)
                    tags = ann_data.get("tags", [])
                    color = TAG_COLORS.get(tags[0], (160, 160, 160)) if tags else (160, 160, 160)
                    pygame.draw.rect(screen, color, (x - 1, bar_y, 3, timeline_h))
            px = int(max(current_frame, 0) / max(total_frames, 1) * bar_w)
            pygame.draw.rect(screen, (255, 255, 255), (px - 1, bar_y, 3, timeline_h))

        pygame.display.flip()

    # -- Main loop --
    while running:
        if text_input_mode:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    text_input_mode = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        text_input_mode = False
                        text_input_buf = ""
                    elif event.key == pygame.K_RETURN:
                        if current_frame not in annotations:
                            annotations[current_frame] = {"tags": [], "note": ""}
                        annotations[current_frame]["note"] = text_input_buf
                        annotations_changed = True
                        text_input_mode = False
                        text_input_buf = ""
                    elif event.key == pygame.K_BACKSPACE:
                        text_input_buf = text_input_buf[:-1]
                    elif event.unicode and event.unicode.isprintable():
                        text_input_buf += event.unicode
            _draw()
            clock.tick(30)

        elif paused:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = False
                    elif event.key == pygame.K_RIGHT:
                        _step_one()
                    elif event.key == pygame.K_LEFT:
                        if current_frame > 0:
                            _simulate_to(current_frame - 1)
                    elif event.key == pygame.K_n:
                        existing = annotations.get(current_frame, {}).get("note", "")
                        text_input_buf = existing
                        text_input_mode = True
                    elif event.key in TAG_KEYS:
                        tag = TAG_KEYS[event.key]
                        if current_frame not in annotations:
                            annotations[current_frame] = {"tags": [], "note": ""}
                        tags = annotations[current_frame]["tags"]
                        if tag in tags:
                            tags.remove(tag)
                        else:
                            tags.append(tag)
                        if not tags and not annotations[current_frame]["note"]:
                            del annotations[current_frame]
                        annotations_changed = True
                    elif event.key == pygame.K_LEFTBRACKET:
                        speed_idx = max(0, speed_idx - 1)
                    elif event.key == pygame.K_RIGHTBRACKET:
                        speed_idx = min(len(SPEEDS) - 1, speed_idx + 1)
            _draw()
            clock.tick(30)

        else:
            # Normal playback
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = True
                    elif event.key == pygame.K_LEFTBRACKET:
                        speed_idx = max(0, speed_idx - 1)
                    elif event.key == pygame.K_RIGHTBRACKET:
                        speed_idx = min(len(SPEEDS) - 1, speed_idx + 1)

            if not running or paused:
                if paused:
                    _draw()
                continue

            if current_frame + 1 >= total_frames:
                break

            _step_one()
            if not running:
                break

            _draw()
            clock.tick(int(60 * SPEEDS[speed_idx]))

    # Post-playback idle (2 seconds at last frame)
    if running and current_frame + 1 >= total_frames:
        for _ in range(120):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    break
            env.step(np.zeros(env.action_space.shape[0], dtype=np.int8))
            pygame.display.flip()
            clock.tick(60)

    # Save annotations on exit
    if annotations and annotations_file is not None:
        ann_list = []
        for frame_num in sorted(annotations):
            entry: dict = {"frame": frame_num}
            ann = annotations[frame_num]
            if ann.get("tags"):
                entry["tags"] = ann["tags"]
            if ann.get("note"):
                entry["note"] = ann["note"]
            ann_list.append(entry)
        out = {
            "actions_file": actions_path.name if actions_path else "",
            "annotations": ann_list,
        }
        annotations_file.write_text(json.dumps(out, indent=2))
        print(f"Saved {len(annotations)} annotations to {annotations_file}")

    # Save trace JSON on exit
    if trace and actions_path is not None:
        trace_file = actions_path.parent / f"{actions_path.stem}_trace.json"

        # Build rooms_visited summary
        rooms_visited = []
        for rid, info in sorted(trace_rooms.items(), key=lambda kv: kv[1]["enter_frame"]):
            rooms_visited.append({
                "room_id": rid,
                "enter_frame": info["enter_frame"],
                "exit_frame": info["last_frame"],
                "frames": info["last_frame"] - info["enter_frame"] + 1,
            })

        # Compute center of gravity (most-visited room, mean x/y)
        cog: dict = {}
        if rooms_visited:
            most_visited = max(rooms_visited, key=lambda r: r["frames"])
            rid = most_visited["room_id"]
            room_pts = [(pt["x"], pt["y"]) for pt in trace if pt["room_id"] == rid]
            if room_pts:
                mean_x = sum(p[0] for p in room_pts) / len(room_pts)
                mean_y = sum(p[1] for p in room_pts) / len(room_pts)
                cog = {"x": round(mean_x, 1), "y": round(mean_y, 1), "room_id": rid}

        # Build annotation list for trace
        ann_list_for_trace = []
        for frame_num in sorted(annotations):
            entry_t: dict = {"frame": frame_num}
            ann_t = annotations[frame_num]
            if ann_t.get("tags"):
                entry_t["tags"] = ann_t["tags"]
            if ann_t.get("note"):
                entry_t["note"] = ann_t["note"]
            ann_list_for_trace.append(entry_t)

        trace_out = {
            "level": config.level_id,
            "total_frames": max(current_frame, 0) + 1,
            "trace": trace,
            "rooms_visited": rooms_visited,
            "center_of_gravity": cog,
            "annotations": ann_list_for_trace,
        }
        trace_file.write_text(json.dumps(trace_out))
        print(f"Saved trace ({len(trace)} points) to {trace_file}")

    pygame.quit()
    env.close()


def cmd_watch(args: argparse.Namespace) -> None:
    """Watch an action sequence play out visually using pygame."""
    from platformer_common.bk2_extract import load_raw_buttons

    config = _resolve_config(args)
    start_state = getattr(args, "state", None)
    actions_path = Path(args.actions)
    if not actions_path.exists():
        print(f"Error: actions file not found: {actions_path}")
        return

    raw = load_raw_buttons(actions_path)
    if raw is not None:
        actions: list[int] | list[list[int]] = raw
        print(f"Loaded {len(actions)} frames (raw buttons) from {actions_path}")
    else:
        actions = load_actions(actions_path)
        print(f"Loaded {len(actions)} frames (action indices) from {actions_path}")
    print(f"Level: {config.display_name}")
    if start_state:
        print(f"State override: {start_state}")
    print("Controls: SPACE=pause  [/]=speed  N=note  1-5=tag  LEFT/RIGHT=step  ESC=quit")

    _replay_with_hud(config, actions, scale=args.scale, start_state=start_state, actions_path=actions_path)
    print("Done.")


def cmd_watch_bk2(args: argparse.Namespace) -> None:
    """Replay a bk2 recording visually using its embedded state."""
    import numpy as np
    import pygame
    import stable_retro as retro
    from retro_harness.env import add_custom_integrations

    config = _resolve_config(args)
    bk2_path = Path(args.bk2)
    if not bk2_path.exists():
        print(f"Error: bk2 file not found: {bk2_path}")
        return

    raw_actions = extract_raw_actions_from_bk2(bk2_path, bk2_to_env=config.bk2_to_env)
    print(f"Extracted {len(raw_actions)} frames from {bk2_path}")

    add_custom_integrations(config.game_dir)
    movie = retro.Movie(str(bk2_path))
    game = movie.get_game()
    env = retro.make(
        game=game,
        state=retro.State.NONE,
        render_mode="rgb_array",
        inttype=retro.data.Integrations.CUSTOM_ONLY,
    )
    env.initial_state = movie.get_state()
    obs, _ = env.reset()

    pygame.init()
    scale = args.scale
    width, height = obs.shape[1], obs.shape[0]
    screen = pygame.display.set_mode(
        (width * scale, height * scale), pygame.SWSURFACE
    )
    pygame.display.set_caption(f"BK2 Replay: {bk2_path.parent.name}")
    clock = pygame.time.Clock()

    print("Playing... (close window or press ESC to stop)")
    running = True
    for frame_idx, buttons in enumerate(raw_actions):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
        if not running:
            break

        action_size = env.action_space.shape[0]
        if len(buttons) < action_size:
            buttons = buttons + [0] * (action_size - len(buttons))

        obs, reward, terminated, truncated, info = env.step(
            np.array(buttons, dtype=np.int8)
        )

        surf = pygame.surfarray.make_surface(obs.swapaxes(0, 1))
        screen.blit(pygame.transform.scale(surf, screen.get_size()), (0, 0))
        pygame.display.flip()
        clock.tick(60)

        if terminated or truncated:
            print(f"Episode ended at frame {frame_idx}")
            break

    if running:
        for _ in range(120):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    break
            env.step(np.zeros(env.action_space.shape[0], dtype=np.int8))
            pygame.display.flip()
            clock.tick(60)

    pygame.quit()
    env.close()
    print("Done.")


def cmd_extract_all(args: argparse.Namespace) -> None:
    """Extract and evaluate all bk2 recordings in the recordings directory."""
    config = _resolve_config(args)
    recordings_dir = Path(args.recordings_dir) if args.recordings_dir else config.game_dir / "recordings"
    if not recordings_dir.exists():
        print(f"Error: recordings directory not found: {recordings_dir}")
        return

    bk2_files = sorted(recordings_dir.rglob("*.bk2"))
    if not bk2_files:
        print("No bk2 files found.")
        return

    action_table = _get_action_table(config)
    print(f"Found {len(bk2_files)} bk2 files")
    print(f"Level: {config.display_name}")

    evaluator = Evaluator(config)
    results = []

    for bk2_path in bk2_files:
        folder = bk2_path.parent.name
        print(f"\n--- {folder}/{bk2_path.name} ---")

        actions = extract_action_indices_from_bk2(
            bk2_path, action_table=action_table, bk2_to_env=config.bk2_to_env
        )
        print(f"  Frames: {len(actions)}")

        result = evaluator.evaluate(actions, early_terminate=False)
        print(f"  Completed: {result.completed}")
        print(f"  Fitness: {result.fitness:.1f}")
        print(f"  Max X: {result.max_x:.1f}")
        if result.completed:
            print(f"  Total frames: {result.total_frames}")
            print(f"  Timer: {result.timer_frames / 60:.2f}s")

        results.append({
            "bk2": str(bk2_path),
            "folder": folder,
            "num_frames": len(actions),
            "completed": result.completed,
            "fitness": result.fitness,
            "total_frames": result.total_frames,
            "max_x": result.max_x,
            "timer_seconds": result.timer_frames / 60 if result.completed else None,
        })

        output = config.runs_dir / f"{folder}_extracted.json"
        metadata = {"source_bk2": str(bk2_path), "level": config.level_id}
        save_actions(actions, output, metadata=metadata)

    evaluator.close()

    print("\n\n=== SUMMARY (sorted by fitness) ===")
    results.sort(key=lambda r: r["fitness"], reverse=True)
    for r in results:
        timer_str = f"{r['timer_seconds']:.2f}s" if r["timer_seconds"] else "N/A"
        status = "DONE" if r["completed"] else "FAIL"
        print(
            f"  {r['folder']:15s} {status:4s} "
            f"fitness={r['fitness']:10.1f} "
            f"frames={r['total_frames']:5d} "
            f"timer={timer_str:8s} "
            f"max_x={r['max_x']:7.1f}"
        )

    best = results[0] if results else None
    if best and best["completed"]:
        print(f"\nBest completed run: {best['folder']} ({best['timer_seconds']:.2f}s)")
        print(f"  Seed file: {config.runs_dir / (best['folder'] + '_extracted.json')}")


def cmd_prepare_seeds(args: argparse.Namespace) -> None:
    """Batch-process recordings: extract all BK2s, evaluate, save top N as seeds."""
    config = _resolve_config(args)
    recordings_dir = Path(args.recordings_dir) if args.recordings_dir else config.game_dir / "recordings"
    if not recordings_dir.exists():
        print(f"Error: recordings directory not found: {recordings_dir}")
        return

    bk2_files = sorted(recordings_dir.rglob("*.bk2"))
    if not bk2_files:
        print("No bk2 files found.")
        return

    action_table = _get_action_table(config)
    top_n = args.top
    print(f"Found {len(bk2_files)} bk2 files, selecting top {top_n}")
    print(f"Level: {config.display_name}")

    evaluator = Evaluator(config)
    candidates: list[tuple[float, str, list[int]]] = []

    for bk2_path in bk2_files:
        actions = extract_action_indices_from_bk2(
            bk2_path, action_table=action_table, bk2_to_env=config.bk2_to_env
        )
        result = evaluator.evaluate(actions, early_terminate=False)
        candidates.append((result.fitness, str(bk2_path), actions))
        status = "COMPLETE" if result.completed else "incomplete"
        print(f"  {bk2_path.name}: fitness={result.fitness:.1f} {status}")

    evaluator.close()

    # Sort by fitness descending, take top N
    candidates.sort(key=lambda c: c[0], reverse=True)
    seeds_dir = config.runs_dir / "seeds"
    seeds_dir.mkdir(parents=True, exist_ok=True)

    for i, (fitness, source, actions) in enumerate(candidates[:top_n]):
        output_path = seeds_dir / f"seed_{i:02d}.json"
        metadata = {"source_bk2": source, "fitness": fitness, "rank": i, "level": config.level_id}
        save_actions(actions, output_path, metadata=metadata)

    print(f"\nSaved {min(top_n, len(candidates))} seeds to {seeds_dir}")
    if candidates:
        print(f"Best: fitness={candidates[0][0]:.1f} from {Path(candidates[0][1]).name}")


def cmd_auto_state(args: argparse.Namespace) -> None:
    """Create a save state by navigating from an existing state."""
    from platformer_common.auto_state import parse_nav_string, navigate_and_save

    config = _resolve_config(args)
    steps = parse_nav_string(args.nav)

    result = navigate_and_save(
        game_name=config.game_name,
        game_dir=config.game_dir,
        from_state=args.from_state,
        save_name=config.start_state,
        steps=steps,
        ram=config.ram,
        expected_level_id=config.target_level_id if config.target_level_id != 0 else None,
        settle_frames=args.settle,
        save_screenshot=args.screenshot,
    )

    if not result.success:
        sys.exit(1)


def cmd_practice(args: argparse.Namespace) -> None:
    """Practice a level with auto-reset on death, saving all attempts."""
    config = _resolve_config(args)
    action_table = _get_action_table(config)
    start_state = getattr(args, "state", None) or config.start_state
    keep_playing = bool(getattr(args, "keep_playing", False))
    until_room_arg = getattr(args, "until_room", None)
    until_room = (
        _parse_room_id_arg(until_room_arg) if until_room_arg is not None else None
    )
    until_playable = bool(getattr(args, "until_playable", False))
    until_label = getattr(args, "until_label", None)
    keep_playing = keep_playing or until_room is not None
    room_debounce = max(1, int(getattr(args, "room_debounce", 3)))

    from retro_harness.env import make_env
    from retro_harness.play_session import PlaySession
    from platformer_common.progress import make_progress_tracker

    env = make_env(
        game=config.game_name,
        state=start_state,
        game_dir=config.game_dir,
        render_mode="rgb_array",
    )

    schema = config.ram_schema

    output_dir = getattr(args, "output_dir", None)
    practice_dir = Path(output_dir).expanduser() if output_dir else config.runs_dir / "practice"
    practice_dir.mkdir(parents=True, exist_ok=True)
    session_label = getattr(args, "session_label", None) or config.level_id

    # Auto-number from highest existing attempt
    existing = sorted(practice_dir.glob("attempt_*.json"))
    next_attempt = 0
    for p in existing:
        try:
            stem = p.stem
            if "_raw" in stem:
                continue
            n = int(stem.split("_")[1])
            next_attempt = max(next_attempt, n + 1)
        except (IndexError, ValueError):
            pass

    # Per-attempt state
    tracker = make_progress_tracker(config)
    tracker.reset()
    recorded_actions: list[int] = []
    recorded_raw: list[list[int]] = []
    best_progress = 0.0
    tracker_seeded = False

    # Session-wide stats
    attempt_num = next_attempt
    all_attempts: list[dict] = []
    session_best_progress = 0.0
    completion_pb_frames = _load_practice_pb_frames(practice_dir)
    result_flash = 0
    result_message = ""
    completion_candidate: tuple[str, int] | None = None
    completion_candidate_frames = 0
    discard_current = False
    split_message = ""
    split_flash = 0

    # Cache emulator state for instant reset
    env.reset()
    cached_emu_state = env.em.get_state()
    ram = env.get_ram()
    initial_values = schema.read(ram)
    config.apply_computed(initial_values)
    initial_lives = initial_values.get("lives")
    current_values = dict(initial_values)
    initial_room_id = int(initial_values.get("level_id", 0) or 0)

    # Continuous recording state. Split frame values are action counts, so they
    # are directly usable as exclusive slice boundaries.
    room_splits: list[dict] = []
    stable_room_id = initial_room_id
    segment_start_frame = 0
    room_candidate_id: int | None = None
    room_candidate_frame = 0
    room_candidate_count = 0
    room_candidate_values: dict[str, int] = {}
    attempt_has_input = False
    recording_checkpoints: dict[int, dict] = {}

    # Select toggle workaround
    _select_state = [0, False]

    def _save_attempt(
        completed: bool,
        terminal_reason: str,
        *,
        discard_trivial: bool = False,
    ) -> bool:
        nonlocal attempt_num, completion_pb_frames
        if not recorded_actions:
            return False
        if (
            discard_trivial
            and not attempt_has_input
            and best_progress <= 0
            and not room_splits
        ):
            print(f"  [DISCARD] empty tail attempt {attempt_num}")
            return False

        frame_count = len(recorded_actions)
        previous_pb = completion_pb_frames
        is_pb = completed and (previous_pb is None or frame_count < previous_pb)
        metadata = {
            "level": config.level_id,
            "source": "practice",
            "session_label": session_label,
            "attempt": attempt_num,
            "best_progress": best_progress,
            "total_frames": frame_count,
            "completed": completed,
            "state": start_state,
            "terminal_reason": terminal_reason,
            "start_room_id": initial_room_id,
            "end_room_id": int(current_values.get("level_id", 0) or 0),
            "until_room_id": until_room,
            "until_playable": until_playable,
            "until_label": until_label,
            "room_splits": [dict(split) for split in room_splits],
        }
        # Save action indices
        out_path = practice_dir / f"attempt_{attempt_num:03d}.json"
        save_actions(recorded_actions, out_path, metadata=metadata)

        # Save raw buttons
        raw_path = practice_dir / f"attempt_{attempt_num:03d}_raw.json"
        import json as _json
        with open(raw_path, "w") as _f:
            _json.dump({"raw_buttons": recorded_raw, "metadata": metadata}, _f)

        all_attempts.append({
            "attempt": attempt_num,
            "frames": frame_count,
            "max_progress": best_progress,
            "completed": completed,
            "is_pb": is_pb,
            "terminal_reason": terminal_reason,
            "room_splits": len(room_splits),
        })
        if is_pb:
            completion_pb_frames = frame_count
        attempt_num += 1
        return True

    def _observe_room(values: dict[str, int]) -> int | None:
        nonlocal stable_room_id, segment_start_frame, room_candidate_id
        nonlocal room_candidate_frame, room_candidate_count, room_candidate_values

        room_id = int(values.get("level_id", 0) or 0)
        if room_id == 0:
            return None
        if room_id == stable_room_id:
            room_candidate_id = None
            room_candidate_count = 0
            room_candidate_values = {}
            return None
        if room_id != room_candidate_id:
            room_candidate_id = room_id
            room_candidate_frame = len(recorded_raw)
            room_candidate_count = 1
            room_candidate_values = dict(values)
        else:
            room_candidate_count += 1
        if room_candidate_count < room_debounce:
            return None

        split = {
            "from_room_id": stable_room_id,
            "room_id": room_id,
            "frame": room_candidate_frame,
            "segment_start_frame": segment_start_frame,
            "segment_frames": room_candidate_frame - segment_start_frame,
        }
        for source, dest in (
            ("player_x", "x"),
            ("player_y", "y"),
            ("health", "health"),
            ("max_health", "max_health"),
            ("missiles", "missiles"),
            ("max_missiles", "max_missiles"),
            ("super_missiles", "super_missiles"),
            ("max_super_missiles", "max_super_missiles"),
            ("game_state", "game_state"),
            ("door_transition", "door_transition"),
        ):
            if source in room_candidate_values:
                split[dest] = int(room_candidate_values[source])
        split["configured_completion"] = (
            _practice_completion_token(config, room_candidate_values, best_progress)
            is not None
        )
        room_splits.append(split)
        stable_room_id = room_id
        segment_start_frame = room_candidate_frame
        room_candidate_id = None
        room_candidate_count = 0
        room_candidate_values = {}
        return room_id

    def _reset_attempt() -> None:
        nonlocal best_progress, tracker_seeded, tracker, result_flash
        nonlocal result_message, completion_candidate, completion_candidate_frames
        nonlocal discard_current, current_values
        nonlocal stable_room_id, segment_start_frame, room_candidate_id
        nonlocal room_candidate_frame, room_candidate_count, room_candidate_values
        nonlocal attempt_has_input, split_message, split_flash
        env.em.set_state(cached_emu_state)
        tracker = make_progress_tracker(config)
        tracker.reset()
        tracker_seeded = False
        recorded_actions.clear()
        recorded_raw.clear()
        best_progress = 0.0
        result_flash = 0
        result_message = ""
        completion_candidate = None
        completion_candidate_frames = 0
        discard_current = False
        current_values = dict(initial_values)
        _select_state[:] = [0, False]
        room_splits.clear()
        stable_room_id = initial_room_id
        segment_start_frame = 0
        room_candidate_id = None
        room_candidate_frame = 0
        room_candidate_count = 0
        room_candidate_values = {}
        attempt_has_input = False
        split_message = ""
        split_flash = 0
        recording_checkpoints.clear()

    def on_step(obs, reward, done, info):
        nonlocal best_progress, tracker_seeded, result_flash, result_message
        nonlocal session_best_progress, completion_candidate
        nonlocal completion_candidate_frames, current_values
        nonlocal attempt_has_input, split_message, split_flash

        if result_flash > 0:
            result_flash -= 1
            if result_flash == 0:
                _reset_attempt()
            return
        if split_flash > 0:
            split_flash -= 1

        # Record
        raw = session.last_action_post_sanitize
        idx = buttons_to_action_index(raw, action_table=action_table)
        recorded_actions.append(idx)
        recorded_raw.append(raw)
        attempt_has_input = attempt_has_input or any(raw)

        # Select workaround
        select_pressed = bool(raw[2])
        if select_pressed and not _select_state[1]:
            try:
                _select_state[0] ^= 1
                env.unwrapped.data.set_value("selected_item", _select_state[0])
            except Exception:
                pass
        _select_state[1] = select_pressed

        # Track progress
        ram = env.get_ram()
        values = schema.read(ram)
        config.apply_computed(values)
        current_values = values
        if not tracker_seeded:
            tracker.update(values)
            tracker_seeded = True
        progress = tracker.update(values)
        if progress > best_progress:
            best_progress = progress
        if best_progress > session_best_progress:
            session_best_progress = best_progress

        confirmed_room = _observe_room(values)
        if keep_playing:
            if confirmed_room is not None:
                latest_split = room_splits[-1]
                split_message = (
                    f"SPLIT 0x{latest_split['from_room_id']:04X} -> "
                    f"0x{confirmed_room:04X} @ {latest_split['frame']}f"
                )
                split_flash = 120
                print(f"  {split_message}")
            target_split_confirmed = bool(
                until_room is not None
                and room_splits
                and room_splits[-1]["room_id"] == until_room
            )
            game_state = values.get("game_state", info.get("game_state"))
            door_transition = values.get(
                "door_transition",
                info.get("door_transition", 0),
            )
            target_playable = (
                not until_playable
                or (game_state == 8 and not bool(door_transition))
            )
            if target_split_confirmed and target_playable:
                frames = len(recorded_actions)
                previous_pb = completion_pb_frames
                _save_attempt(completed=True, terminal_reason="until_room")
                if previous_pb is None or frames < previous_pb:
                    result_message = f">>> TARGET {frames}f | NEW PB <<<"
                else:
                    result_message = (
                        f">>> TARGET {frames}f | PB +{frames - previous_pb}f <<<"
                    )
                print(f"  {result_message.strip('> <')} attempt {attempt_num - 1}")
                result_flash = 60
                return
        else:
            completion_token = _practice_completion_token(config, values, best_progress)
            if completion_token is None:
                completion_candidate = None
                completion_candidate_frames = 0
            elif completion_token == completion_candidate:
                completion_candidate_frames += 1
            else:
                completion_candidate = completion_token
                completion_candidate_frames = 1

            required_completion_frames = max(1, config.completion_debounce_frames + 1)
            if completion_candidate_frames >= required_completion_frames:
                frames = len(recorded_actions)
                previous_pb = completion_pb_frames
                _save_attempt(completed=True, terminal_reason="configured_completion")
                if previous_pb is None or frames < previous_pb:
                    result_message = f">>> SUCCESS {frames}f | NEW PB <<<"
                else:
                    result_message = (
                        f">>> SUCCESS {frames}f | PB +{frames - previous_pb}f <<<"
                    )
                print(f"  {result_message.strip('> <')} attempt {attempt_num - 1}")
                result_flash = 60
                return

        # Death detection
        lives = values.get("lives")
        gameplay_started = len(recorded_actions) > 30  # small grace period
        if gameplay_started:
            is_dead = False
            for signal in config.death_signals:
                if signal == "lives_drop":
                    if initial_lives is not None and lives is not None and lives < initial_lives:
                        is_dead = True
                elif signal == "health_zero":
                    health = values.get("health", 1)
                    if health <= 0:
                        is_dead = True

            if is_dead:
                print(f"  DIED attempt {attempt_num}: {len(recorded_actions)}f, progress={best_progress:.0f}")
                _save_attempt(completed=False, terminal_reason="death")
                result_message = ">>> DIED <<<  (auto-resetting...)"
                result_flash = 60
                return

        if done:
            _save_attempt(completed=False, terminal_reason="env_done")
            result_message = ">>> EPISODE ENDED <<<  (auto-resetting...)"
            result_flash = 60

    def on_hud(info):
        lines = [
            f"{session_label} | attempt #{attempt_num} | {len(recorded_actions)}f",
            (
                f"start progress={best_progress:.0f} | best={session_best_progress:.0f}"
                if keep_playing
                else f"progress={best_progress:.0f} | best={session_best_progress:.0f}"
            ),
            f"saved: {len(all_attempts)} attempts | splits: {len(room_splits)}",
        ]
        telemetry = []
        room_id = current_values.get("level_id")
        if room_id is not None:
            telemetry.append(f"room=0x{room_id:04X}")
        if "player_x" in current_values and "player_y" in current_values:
            telemetry.append(f"pos=({current_values['player_x']},{current_values['player_y']})")
        if "health" in current_values:
            telemetry.append(f"hp={current_values['health']}")
        if "missiles" in current_values:
            maximum = current_values.get("max_missiles")
            value = current_values["missiles"]
            telemetry.append(
                f"missiles={value}/{maximum}"
                if maximum is not None
                else f"missiles={value}"
            )
        if "super_missiles" in current_values:
            maximum = current_values.get("max_super_missiles")
            value = current_values["super_missiles"]
            telemetry.append(
                f"supers={value}/{maximum}"
                if maximum is not None
                else f"supers={value}"
            )
        if telemetry:
            lines.append(" | ".join(telemetry))
        if completion_pb_frames is not None:
            delta = len(recorded_actions) - completion_pb_frames
            lines.append(f"completed PB={completion_pb_frames}f | delta={delta:+d}f")
        if until_room is not None:
            suffix = " playable" if until_playable else ""
            label = f" ({until_label})" if until_label else ""
            lines.append(f"target room=0x{until_room:04X}{label}{suffix}")
        if split_flash > 0:
            lines.insert(0, split_message)
        if saved_state_path[0]:
            lines.append(f"F5 state: {save_name} (progress={best_progress:.0f})")
        if result_flash > 0:
            lines.insert(0, result_message)
        return lines

    # Name for F5 save state (user can override via --save-name)
    save_name = getattr(args, "save_name", None) or f"Chained_{config.level_id}_practice"
    saved_state_path = [None]  # mutable ref for HUD

    def on_key_down(key):
        nonlocal discard_current
        import pygame as pg
        if key == pg.K_F5:
            # Save persistent .state at current position
            from retro_harness.env import save_state as _save_state
            path = _save_state(env, str(config.game_dir), config.game_name, save_name)
            saved_state_path[0] = path
            print(f"  [STATE SAVED] {save_name} at progress={best_progress:.0f} ({len(recorded_actions)}f)")
            print(f"  -> {path}")
            print(f"  Practice from here: uv run python -m platformer_common -l {config.level_id} practice --state {save_name}")
            return True
        if key == pg.K_r:
            # Discard current attempt and restart
            print(f"  [DISCARD] attempt {attempt_num}")
            _reset_attempt()
            discard_current = True
            return False  # let PlaySession handle env.reset()
        return False

    def trigger_save(slot: int) -> None:
        recording_checkpoints[slot] = {
            "frame": len(recorded_actions),
            "room_splits": [dict(split) for split in room_splits],
            "stable_room_id": stable_room_id,
            "segment_start_frame": segment_start_frame,
            "best_progress": best_progress,
            "attempt_has_input": attempt_has_input,
            "current_values": dict(current_values),
            "select_state": list(_select_state),
        }
        session.save_checkpoint(slot)

    def trigger_load(slot: int) -> None:
        nonlocal tracker, tracker_seeded, best_progress, stable_room_id
        nonlocal segment_start_frame, room_candidate_id, room_candidate_count
        nonlocal room_candidate_values, attempt_has_input, current_values
        nonlocal completion_candidate, completion_candidate_frames
        nonlocal split_message, split_flash, result_message, result_flash
        checkpoint = recording_checkpoints.get(slot)
        if checkpoint is None:
            print(f"[CHECKPOINT {slot}] no recording checkpoint")
            return
        if session.load_checkpoint(slot) is None:
            return
        frame = checkpoint["frame"]
        del recorded_actions[frame:]
        del recorded_raw[frame:]
        room_splits[:] = [dict(split) for split in checkpoint["room_splits"]]
        stable_room_id = checkpoint["stable_room_id"]
        segment_start_frame = checkpoint["segment_start_frame"]
        best_progress = checkpoint["best_progress"]
        attempt_has_input = checkpoint["attempt_has_input"]
        current_values = dict(checkpoint["current_values"])
        _select_state[:] = checkpoint["select_state"]
        room_candidate_id = None
        room_candidate_count = 0
        room_candidate_values = {}
        completion_candidate = None
        completion_candidate_frames = 0
        split_message = ""
        split_flash = 0
        result_message = ""
        result_flash = 0
        tracker = make_progress_tracker(config)
        tracker.reset()
        tracker_seeded = False
        print(f"  Recording truncated to {frame} frames and {len(room_splits)} splits")

    # Override PlaySession reset so R key uses our cached state
    def on_reset():
        if discard_current:
            env.em.set_state(cached_emu_state)

    session = PlaySession(
        env,
        game_dir=str(config.game_dir),
        game=config.game_name,
        scale=args.scale,
        title=f"PRACTICE: {session_label} | {config.display_name}",
    )
    session.on_step = on_step
    session.on_hud = on_hud
    session.on_key_down = on_key_down
    session.on_reset = on_reset
    session.on_trigger_save = trigger_save
    session.on_trigger_load = trigger_load

    print(f"Practice mode: {config.display_name}")
    print(f"Session: {session_label}")
    print(f"State: {start_state}")
    print(f"Output: {practice_dir}")
    if keep_playing:
        target = f"0x{until_room:04X}" if until_room is not None else "manual stop"
        print(f"Continuous rooms: on (target: {target}, debounce: {room_debounce})")
    print("\nControls:")
    print("  Arrow keys = D-pad    Z = B    X = A    A = Y    S = X")
    print("  F5 = save .state at current position (for later practice)")
    print("  TAB = turbo    R = discard & restart    ESC = save & quit")
    print("  On death or target: auto-saves attempt, resets after 1s\n")

    session.run()

    # Save final attempt if there are unsaved frames
    if recorded_actions and result_flash == 0:
        _save_attempt(
            completed=False,
            terminal_reason="user_exit",
            discard_trivial=True,
        )

    # Write summary
    summary = {
        "level": config.level_id,
        "session_label": session_label,
        "state": start_state,
        "total_attempts": len(all_attempts),
        "best_progress": session_best_progress,
        "best_completion_frames": completion_pb_frames,
        "until_room_id": until_room,
        "until_playable": until_playable,
        "until_label": until_label,
        "keep_playing": keep_playing,
        "attempts": all_attempts,
    }
    summary["best_attempt"] = _best_practice_attempt(all_attempts)
    summary_path = practice_dir / "practice_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n=== Practice Summary ===")
    print(f"Attempts: {len(all_attempts)}")
    print(f"Best progress: {session_best_progress:.0f}")
    if summary["best_attempt"] >= 0:
        best = next(
            attempt
            for attempt in all_attempts
            if attempt["attempt"] == summary["best_attempt"]
        )
        print(f"Best attempt: #{best['attempt']} ({best['frames']}f)")
    print(f"Saved to: {practice_dir}")


def cmd_play(args: argparse.Namespace) -> None:
    """Play a level manually while recording inputs as action indices."""
    import numpy as np

    config = _resolve_config(args)
    action_table = _get_action_table(config)
    start_state = getattr(args, "state", None) or config.start_state

    from retro_harness.env import make_env

    env = make_env(
        game=config.game_name,
        state=start_state,
        game_dir=config.game_dir,
        render_mode="rgb_array",
    )

    from retro_harness.play_session import PlaySession
    from platformer_common.progress import make_progress_tracker

    schema = config.ram_schema
    tracker = make_progress_tracker(config)
    tracker.reset()
    recorded_actions: list[int] = []
    recorded_raw: list[list[int]] = []
    recorded_raw_pre_sanitize: list[list[int]] = []
    best_progress = 0.0
    tracker_seeded = False

    # Workaround: stable-retro ignores SNES Select for SM weapon toggle.
    # Track toggle state and force via RAM on rising edge of Select.
    _select_state = [0, False]  # [current_item, was_pressed_last_frame]

    def on_step(obs, reward, done, info):
        nonlocal best_progress, tracker_seeded
        # Record the action index and raw buttons for this frame
        raw = list(last_raw_action)
        raw_pre = list(last_raw_action_pre_sanitize)
        idx = buttons_to_action_index(raw, action_table=action_table)
        recorded_actions.append(idx)
        recorded_raw.append(raw)
        recorded_raw_pre_sanitize.append(raw_pre)

        # Workaround: force weapon toggle via RAM on Select press edge
        select_pressed = bool(raw[2])  # SNES_SELECT
        if select_pressed and not _select_state[1]:
            try:
                _select_state[0] ^= 1
                env.unwrapped.data.set_value(
                    "selected_item", _select_state[0]
                )
            except Exception:
                pass
        _select_state[1] = select_pressed

        ram = env.get_ram()
        values = schema.read(ram)
        config.apply_computed(values)
        if not tracker_seeded:
            tracker.update(values)
            tracker_seeded = True
        progress = tracker.update(values)
        if progress > best_progress:
            best_progress = progress

    _ROOM_NAMES = {
        0x91F8: 'Landing Site', 0x92FD: 'Parlor', 0x9879: 'Flyway',
        0x9804: 'Bomb Torizo', 0x96BA: 'Climb', 0x975C: 'Pit Room',
        0x9AD9: 'BB Elevator', 0x9E9F: 'Morph Ball Room',
        0x9F11: 'Construction Zone', 0x9E52: 'First Missile Room',
        0xA011: 'BB E-Tank', 0x962A: 'Terminator Room',
        0x99BD: 'Green Pirates Shaft', 0x9BC8: 'Lower Mushrooms',
        0x9938: 'Green Brinstar Elev', 0x9F64: 'GB Main Shaft',
        0x9FBA: 'GB Fireflea', 0x9FC5: 'GB Missile Refill',
    }

    def on_hud(info):
        ram = env.get_ram()
        values = schema.read(ram)
        config.apply_computed(values)
        level_id = values.get("level_id", 0)
        health = values.get("health")
        lives = values.get("lives", "?")
        stat = f"hp={health}" if health is not None else f"lives={lives}"
        lines = [
            f"REC {len(recorded_actions)} frames | progress={best_progress:.0f}",
            f"{stat} | {_ROOM_NAMES.get(level_id, f'0x{level_id:04X}')}",
        ]
        # Show weapon status if SM (has selected_item in data.json)
        try:
            sel = env.unwrapped.data.lookup_value("selected_item")
            missiles = env.unwrapped.data.lookup_value("missiles")
            weapon = "MISSILES" if sel else "BEAM"
            lines.append(f"weapon={weapon} | missiles={missiles}")
        except Exception:
            pass
        return lines

    # Intercept raw actions before they go to the env
    last_raw_action = [0] * 12
    last_raw_action_pre_sanitize = [0] * 12
    _orig_gather = PlaySession._gather_action

    def patched_gather(self, pg, keyboard_action, controller_action, sanitize_action):
        nonlocal last_raw_action, last_raw_action_pre_sanitize
        action = _orig_gather(self, pg, keyboard_action, controller_action, sanitize_action)
        post = getattr(self, "_last_action_post_sanitize", action)
        pre = getattr(self, "_last_action_pre_sanitize", post)
        last_raw_action = list(post) if hasattr(post, "__iter__") else [0] * 12
        last_raw_action_pre_sanitize = list(pre) if hasattr(pre, "__iter__") else [0] * 12
        return action

    PlaySession._gather_action = patched_gather

    # Checkpoint recording state: map slot -> frame count at save time
    _checkpoint_frames: dict[int, int] = {}

    def on_key_down(key):
        import pygame as pg
        nonlocal best_progress, tracker_seeded

        _SLOT_KEYS = {pg.K_F1: 1, pg.K_F2: 2, pg.K_F3: 3, pg.K_F4: 4}

        if key in _SLOT_KEYS:
            slot = _SLOT_KEYS[key]
            mods = pg.key.get_mods()
            if mods & pg.KMOD_SHIFT:
                # Load checkpoint → truncate recording to that frame
                frame = session.load_checkpoint(slot)
                if frame is not None and slot in _checkpoint_frames:
                    rec_frame = _checkpoint_frames[slot]
                    del recorded_actions[rec_frame:]
                    del recorded_raw[rec_frame:]
                    del recorded_raw_pre_sanitize[rec_frame:]
                    # Reset progress tracker from truncated recording
                    tracker.reset()
                    tracker_seeded = False
                    best_progress = 0.0
                    print(f"  Recording truncated to {rec_frame} frames")
            else:
                # Save checkpoint → record frame position
                _checkpoint_frames[slot] = len(recorded_actions)
                session.save_checkpoint(slot)
            return True

        if key == pg.K_r:
            # Restart: reset env and clear recording
            recorded_actions.clear()
            recorded_raw.clear()
            recorded_raw_pre_sanitize.clear()
            tracker.reset()
            tracker_seeded = False
            best_progress = 0.0
            print("[RESTART] recording cleared")
            return False  # let PlaySession handle the env.reset()

        return False

    session = PlaySession(
        env,
        game_dir=str(config.game_dir),
        game=config.game_name,
        scale=args.scale,
        title=f"RECORD: {config.display_name}",
    )
    session.on_step = on_step
    session.on_hud = on_hud
    session.on_key_down = on_key_down

    # Recording-aware trigger hooks (L2=load, R2=save checkpoint 1)
    def trigger_save(slot: int) -> None:
        _checkpoint_frames[slot] = len(recorded_actions)
        session.save_checkpoint(slot)

    def trigger_load(slot: int) -> None:
        nonlocal best_progress, tracker_seeded
        frame = session.load_checkpoint(slot)
        if frame is not None and slot in _checkpoint_frames:
            rec_frame = _checkpoint_frames[slot]
            del recorded_actions[rec_frame:]
            del recorded_raw[rec_frame:]
            del recorded_raw_pre_sanitize[rec_frame:]
            tracker.reset()
            tracker_seeded = False
            best_progress = 0.0
            print(f"  Recording truncated to {rec_frame} frames")

    session.on_trigger_save = trigger_save
    session.on_trigger_load = trigger_load

    print(f"Recording: {config.display_name}")
    print(f"State: {start_state}")
    print(f"Action table: {len(action_table)} actions")
    print(f"\nControls:")
    print(f"  Arrow keys = D-pad")
    print(f"  Z = B    X = A    A = Y    S = X")
    print(f"  F1-F4 = save checkpoint    Shift+F1-F4 = load checkpoint")
    print(f"  F5 = export state to disk  R = restart & clear recording")
    print(f"  TAB = turbo    ESC = stop & save")
    print(f"  Controller: L2 = load checkpoint 1    R2 = save checkpoint 1\n")

    try:
        session.run()
    finally:
        # Restore original method
        PlaySession._gather_action = _orig_gather

    if not recorded_actions:
        print("No frames recorded.")
        return

    # Save recording
    output_dir = config.runs_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-increment filename
    existing = sorted(output_dir.glob("recording_*.json"))
    next_idx = 0
    for p in existing:
        try:
            n = int(p.stem.split("_")[1])
            next_idx = max(next_idx, n + 1)
        except (IndexError, ValueError):
            pass

    output_path = output_dir / f"recording_{next_idx:03d}.json"
    metadata = {
        "level": config.level_id,
        "source": "manual_play",
        "best_progress": best_progress,
        "total_frames": len(recorded_actions),
        "button_order": ["B", "Y", "Select", "Start", "Up", "Down", "Left", "Right", "A", "X", "L", "R"],
        "raw_buttons_note": "raw_buttons are post-sanitize env inputs (used for replay).",
        "raw_buttons_pre_sanitize_note": "raw_buttons_pre_sanitize are captured before directional conflict sanitization.",
    }
    save_actions(recorded_actions, output_path, metadata=metadata)

    # Also save raw 12-button arrays for faithful replay
    raw_path = output_path.with_name(output_path.stem + "_raw.json")
    import json as _json
    with open(raw_path, "w") as _f:
        _json.dump({
            "raw_buttons": recorded_raw,
            "raw_buttons_pre_sanitize": recorded_raw_pre_sanitize,
            "actions": recorded_actions,
            "metadata": metadata,
        }, _f)
    print(f"Raw buttons (post + pre-sanitize): {raw_path}")

    print(f"\nRecorded {len(recorded_actions)} frames")
    print(f"Best progress: {best_progress:.0f}")
    print(f"Saved to: {output_path}")
    print(f"\nNext steps:")
    print(f"  Verify:    uv run python -m platformer_common -l {config.level_id} verify --actions {output_path}")
    print(f"  Hillclimb: uv run python -m platformer_common -l {config.level_id} hillclimb --seed {output_path}")


def cmd_selftest(args: argparse.Namespace) -> None:
    """Self-test: verify death detection and level-change guards work correctly."""
    import numpy as np

    config = _resolve_config(args)
    print(f"=== Platformer Optimizer Self-Test: {config.display_name} ===\n")
    failures = 0

    evaluator = Evaluator(config)
    evaluator._ensure_env()

    initial_cam = evaluator._initial_camera_x
    initial_values = evaluator._initial_values
    print(f"State: {config.start_state}")
    print(f"  initial_camera_x = {initial_cam:.0f}")
    print(f"  initial_lives    = {initial_values.get('lives', 'N/A')}")

    # Check level_id is correct
    level_id = initial_values.get("level_id", -1)
    if level_id != config.target_level_id:
        print(f"  FAIL: level_id=0x{level_id:04X}, expected 0x{config.target_level_id:04X}")
        failures += 1
    else:
        print(f"  OK: level_id=0x{level_id:04X}")

    # Test 1: sequence that dies must be flagged as died, NOT completed
    print(f"\n[Test 1] Deterministic death probe")
    death_seq = config.selftest_death_actions or (([2] * 40 + [3] * 15 + [2] * 5 + [5] * 10) * 28)
    if not config.selftest_expect_death:
        print("  SKIP: no published deterministic death probe for this start state")
    else:
        result = evaluator.evaluate(death_seq[:2000], early_terminate=False)
        if not result.died:
            print(f"  FAIL: died={result.died}, expected True")
            failures += 1
        elif result.completed:
            print(f"  FAIL: completed={result.completed}, should be False when died")
            failures += 1
        else:
            print(f"  OK: died=True, completed=False, frame={result.total_frames}, progress={result.max_progress:.0f}")

    # Test 2: fitness for dead < alive at same progress
    print(f"\n[Test 2] Death fitness < alive fitness at same progress")
    dead_fitness_at_100 = 100 * config.progress_weight - config.death_penalty
    alive_fitness_at_100 = 100 * config.progress_weight
    if dead_fitness_at_100 >= alive_fitness_at_100:
        print(f"  FAIL: dead_fitness ({dead_fitness_at_100}) >= alive_fitness ({alive_fitness_at_100})")
        failures += 1
    else:
        print(f"  OK: dead@100={dead_fitness_at_100} < alive@100={alive_fitness_at_100}")

    # Test 3: short alive sequence stays in level
    print(f"\n[Test 3] Short alive sequence stays in level")
    short_result = evaluator.evaluate([0] * 60, early_terminate=False)
    if short_result.completed:
        print(f"  FAIL: 60 frames of nothing showed completed=True!")
        failures += 1
    elif short_result.died:
        print(f"  FAIL: 60 frames of nothing showed died=True!")
        failures += 1
    else:
        print(f"  OK: alive, not completed, level_id=0x{short_result.level_id_at_end:04X}")

    # Test 4: determinism
    print(f"\n[Test 4] Determinism check")
    r1 = evaluator.evaluate(death_seq[:500], early_terminate=False)
    r2 = evaluator.evaluate(death_seq[:500], early_terminate=False)
    if r1.fitness != r2.fitness or r1.total_frames != r2.total_frames:
        print(f"  FAIL: run1 fitness={r1.fitness:.0f}/frames={r1.total_frames} != run2")
        failures += 1
    else:
        print(f"  OK: both runs -> fitness={r1.fitness:.0f}, frames={r1.total_frames}")

    # Test 5: first-frame button stability
    print(f"\n[Test 5] First-frame button stability")
    action_table = _get_action_table(config)
    for idx in range(len(action_table)):
        r = evaluator.evaluate([idx], early_terminate=False)
        if r.completed:
            print(f"  FAIL: action {idx} on first frame triggered completion!")
            failures += 1
            break
    else:
        print(f"  OK: all {len(action_table)} actions stable on first frame")

    evaluator.close()

    print(f"\n{'=' * 40}")
    if failures == 0:
        print("ALL TESTS PASSED")
    else:
        print(f"{failures} TEST(S) FAILED")
    return failures


def cmd_trace_map(args: argparse.Namespace) -> None:
    """Render a position trace overlaid on an area map PNG."""
    config = _resolve_config(args)

    # Resolve trace path
    trace_path = None
    if getattr(args, "trace", None):
        trace_path = Path(args.trace)
    elif getattr(args, "actions", None):
        trace_path = Path(args.actions).parent / f"{Path(args.actions).stem}_trace.json"
    else:
        # Look in runs dir for most recent trace
        traces = sorted(config.runs_dir.glob("*_trace.json"))
        if traces:
            trace_path = traces[-1]

    if trace_path is None or not trace_path.exists():
        print(f"Error: trace file not found: {trace_path}")
        print("Run 'watch' first to generate a trace, or specify --trace path.")
        return

    output = Path(args.output) if getattr(args, "output", None) else trace_path.with_suffix(".png")
    map_dir = Path(args.map_dir) if getattr(args, "map_dir", None) else None

    # Dispatch to game-specific renderer
    if config.level_id.startswith("smb_"):
        from super_mario_bros.trace_renderer import render_smb_trace

        render_smb_trace(
            trace_path=trace_path,
            level_id=config.level_id,
            output_path=output,
            map_dir=map_dir,
        )
    else:
        from super_metroid_rl.navigation.trace_renderer import (
            render_trace_on_map,
            detect_area,
            _load_nodes,
            DEFAULT_EXPORT_DIR,
        )

        area = getattr(args, "area", None)
        if not area:
            trace_data = json.loads(trace_path.read_text())
            export_dir = Path(getattr(args, "map_dir", None) or DEFAULT_EXPORT_DIR)
            nodes = _load_nodes(export_dir)
            area = detect_area(trace_data, nodes)
            if not area:
                print("Error: could not auto-detect area. Specify --area.")
                return
            print(f"Auto-detected area: {area}")

        render_trace_on_map(
            trace_path=trace_path,
            area_name=area,
            output_path=output,
            map_dir=map_dir,
        )


# -- Route commands ----------------------------------------------------------


def cmd_list_routes(args: argparse.Namespace) -> None:
    """List all registered routes."""
    from platformer_common.route import list_routes

    routes = list_routes()
    if not routes:
        print("No routes registered.")
        return

    print(f"{'ID':<25s} {'Display Name':<40s} {'Segments':>8s}")
    print("-" * 75)
    for r in routes:
        print(f"{r.route_id:<25s} {r.display_name:<40s} {len(r.segments):>8d}")


def cmd_chain(args: argparse.Namespace) -> None:
    """Evaluate a full speedrun route (all segments independently)."""
    from platformer_common.route import get_route, evaluate_route

    route = get_route(args.route)
    result = evaluate_route(route, verbose=True)

    if result.all_completed:
        print(f"\nAll segments completed! Total: {result.total_frames}f "
              f"({result.total_frames / 60:.1f}s)")
    else:
        sys.exit(1)


def cmd_chain_live(args: argparse.Namespace) -> None:
    """Run a true end-to-end chain on a single emulator (no state reloads)."""
    from platformer_common.route import get_route, chain_live

    route = get_route(args.route)
    result = chain_live(
        route,
        save_states=args.save_states,
        verbose=True,
        video_path=args.video,
        video_scale=args.scale,
    )

    if result.all_completed:
        print(f"\nFull chain completed! {result.total_frames}f ({result.total_frames / 60:.1f}s)")
    else:
        sys.exit(1)


def cmd_chain_optimize(args: argparse.Namespace) -> None:
    """Iteratively hill-climb each segment from chained states."""
    from platformer_common.route import get_route, chain_optimize

    route = get_route(args.route)
    result = chain_optimize(
        route,
        iterations=args.iterations,
        verbose=True,
    )

    if result.all_completed:
        print(f"\nFull chain optimized! {result.total_frames}f ({result.total_frames / 60:.1f}s)")
    else:
        sys.exit(1)


def cmd_chain_video(args: argparse.Namespace) -> None:
    """Render a full speedrun route to a single MP4."""
    from platformer_common.route import get_route, record_route_video

    route = get_route(args.route)
    output = args.output or f"{route.route_id}.mp4"
    record_route_video(route, output, scale=args.scale)


# -- Main CLI ----------------------------------------------------------------


def main(default_level: str | None = None) -> None:
    """Build and run the CLI parser.

    Args:
        default_level: If set, use this level when --level is omitted.
            Used by game-specific wrappers (e.g., DKC optimizer).
    """
    parser = argparse.ArgumentParser(
        description="Platformer Speedrun Optimizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Global --level flag
    level_required = default_level is None
    parser.add_argument(
        "--level", "-l",
        default=default_level,
        required=False,
        help=f"Level ID or alias (default: {default_level or 'required'})",
    )

    sub = parser.add_subparsers(dest="command")

    # list-levels
    sub.add_parser("list-levels", help="List all registered levels")

    # extract
    p_extract = sub.add_parser("extract", help="Extract actions from a bk2 recording")
    p_extract.add_argument("--bk2", required=True, help="Path to bk2 file")
    p_extract.add_argument("--output", "-o", help="Output JSON path")
    p_extract.add_argument("--raw-preview", action="store_true")

    # extract-all
    p_extract_all = sub.add_parser("extract-all", help="Extract and evaluate all bk2 recordings")
    p_extract_all.add_argument("--recordings-dir", help="Recordings directory")

    # verify
    p_verify = sub.add_parser("verify", help="Verify action sequence via headless replay")
    p_verify.add_argument("--actions", required=True, help="Path to actions JSON")
    p_verify.add_argument("--trace", action="store_true", help="Log all level_id changes")
    p_verify.add_argument("--state", help="Override start state")

    # optimize
    p_optimize = sub.add_parser("optimize", help="Run GA optimization")
    p_optimize.add_argument("--seed", help="Path to seed actions JSON")
    p_optimize.add_argument("--seeds-dir", help="Directory of recordings to use as multi-seed (mutually exclusive with --seed)")
    p_optimize.add_argument("--min-frames", type=int, default=60, help="Skip seeds shorter than N frames (default: 60)")
    p_optimize.add_argument("--raw", action="store_true", help="Use raw-button GA (no lossy action-index conversion)")
    p_optimize.add_argument("--generations", type=int, default=None)
    p_optimize.add_argument("--population", type=int, default=None)
    p_optimize.add_argument("--output-dir", help="Output directory")
    p_optimize.add_argument("--workers", type=int, default=1, help="Parallel workers")
    p_optimize.add_argument("--resume", help="Resume from checkpoint JSON")
    p_optimize.add_argument("--render", type=int, nargs="?", const=1, default=0,
                            metavar="N", help="Render best every N gens (default: every gen)")
    p_optimize.add_argument("--state", help="Override start state")

    # hillclimb
    p_hill = sub.add_parser("hillclimb", help="Run hill climbing refinement")
    p_hill.add_argument("--seed", required=True, help="Path to seed actions JSON")
    p_hill.add_argument("--iterations", type=int, default=5000)
    p_hill.add_argument("--output-dir", help="Output directory")
    p_hill.add_argument("--render", type=int, nargs="?", const=100, default=0,
                        metavar="N", help="Render best every N iterations (default: every 100)")
    p_hill.add_argument("--scale", type=int, default=3, help="Render scale")
    p_hill.add_argument("--state", help="Override start state")

    # hillclimb-raw (raw button mutation, no lossy action-index conversion)
    p_hraw = sub.add_parser("hillclimb-raw", help="Hill climb with raw button mutation")
    p_hraw.add_argument("--seed", required=True, help="Path to seed JSON with raw_buttons")
    p_hraw.add_argument("--iterations", type=int, default=1000)
    p_hraw.add_argument("--output-dir", help="Output directory")
    p_hraw.add_argument("--state", help="Override start state")
    p_hraw.add_argument(
        "--window",
        help="Only mutate START:END (enables checkpoint-accelerated segment engine)",
    )
    p_hraw.add_argument(
        "--prefer-trim",
        action="store_true",
        help="Bias mutations toward frame deletion / hold shortening",
    )
    p_hraw.add_argument(
        "--require-completion",
        action="store_true",
        help="Never accept a candidate that fails to complete",
    )

    # analyze-seed (static + optional live eval)
    p_an = sub.add_parser(
        "analyze-seed",
        help="Report leading idle, hold stalls, optional clear-frame eval",
    )
    p_an.add_argument("--seed", required=True, help="Path to raw_buttons / nes9_rle seed")
    p_an.add_argument("--state", help="Override start state")
    p_an.add_argument(
        "--static-only",
        action="store_true",
        help="Skip emulator eval (idle/hold stats only)",
    )
    p_an.add_argument("--output", "-o", help="Write JSON report path")

    # trim-seed (deterministic frame-saving transforms)
    p_trim = sub.add_parser(
        "trim-seed",
        help="Trim leading idle, compress holds, drop post-clear pad",
    )
    p_trim.add_argument("--seed", required=True, help="Path to raw_buttons / nes9_rle seed")
    p_trim.add_argument("--state", help="Override start state")
    p_trim.add_argument("--output", "-o", help="Output JSON path")
    p_trim.add_argument("--output-dir", help="Output directory (if --output omitted)")
    p_trim.add_argument(
        "--parity",
        choices=("any", "even", "odd"),
        default="any",
        help="Leading-idle trim parity (SMB 1-1 needs even)",
    )
    p_trim.add_argument("--step", type=int, default=1, help="Leading-trim search step")
    p_trim.add_argument("--max-leading", type=int, default=None, help="Cap leading idle trim")
    p_trim.add_argument("--pad", type=int, default=30, help="Idle frames kept after clear")
    p_trim.add_argument("--no-leading", action="store_true", help="Skip leading-idle search")
    p_trim.add_argument("--no-trailing", action="store_true", help="Skip post-clear trim")
    p_trim.add_argument(
        "--holds",
        action="store_true",
        help="Also binary-search compress long identical-button holds",
    )
    p_trim.add_argument("--min-hold", type=int, default=30, help="Min hold length to probe")
    p_trim.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Accept non-completing candidates (default: require completion)",
    )

    # segment-hillclimb (windowed + checkpoint)
    p_seg = sub.add_parser(
        "segment-hillclimb",
        help="Checkpoint-accelerated hillclimb inside a frame window",
    )
    p_seg.add_argument("--seed", required=True, help="Path to raw_buttons / nes9_rle seed")
    p_seg.add_argument(
        "--window",
        required=True,
        help="Mutable frame range START:END (prefix is checkpointed)",
    )
    p_seg.add_argument("--iterations", type=int, default=1000)
    p_seg.add_argument("--output-dir", help="Output directory")
    p_seg.add_argument("--state", help="Override start state")
    p_seg.add_argument(
        "--no-prefer-trim",
        action="store_true",
        help="Disable delete/hold-trim mutation bias",
    )
    p_seg.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Allow non-completing fitness improvements",
    )

    # watch
    p_watch = sub.add_parser("watch", help="Watch action sequence visually")
    p_watch.add_argument("--actions", required=True, help="Path to actions JSON")
    p_watch.add_argument("--scale", type=int, default=3)
    p_watch.add_argument("--state", help="Override start state")

    # watch-bk2
    p_watch_bk2 = sub.add_parser("watch-bk2", help="Replay a bk2 recording visually")
    p_watch_bk2.add_argument("--bk2", required=True, help="Path to bk2 file")
    p_watch_bk2.add_argument("--scale", type=int, default=3)

    # prepare-seeds
    p_seeds = sub.add_parser("prepare-seeds", help="Extract and rank recordings, save top N as seeds")
    p_seeds.add_argument("--recordings-dir", help="Recordings directory")
    p_seeds.add_argument("--top", type=int, default=5, help="Number of top seeds to save")

    # auto-state
    p_auto = sub.add_parser("auto-state", help="Create save state via scripted navigation")
    p_auto.add_argument("--from-state", required=True, help="Starting state name")
    p_auto.add_argument("--nav", required=True, help="Navigation steps: 'BUTTON:hold:wait ...'")
    p_auto.add_argument("--settle", type=int, default=30, help="Extra NOOP frames after nav (default: 30)")
    p_auto.add_argument("--screenshot", action="store_true", help="Save screenshot for verification")

    # practice (auto-reset on death, saves all attempts)
    p_practice = sub.add_parser("practice", help="Practice with auto-reset on death, saving all attempts")
    p_practice.add_argument("--scale", type=int, default=3)
    p_practice.add_argument("--state", help="Override start state")
    p_practice.add_argument("--save-name", help="Name for F5 state save (default: Chained_{level}_practice)")
    p_practice.add_argument("--output-dir", help="Directory for attempt JSON and summary files")
    p_practice.add_argument("--session-label", help="Session label stored in metadata and shown in the HUD")
    p_practice.add_argument(
        "--continue",
        "--keep-playing",
        dest="keep_playing",
        action="store_true",
        help="Continue recording across configured segment-completion rooms",
    )
    p_practice.add_argument(
        "--until-room",
        type=_parse_room_id_arg,
        help="End the continuous attempt at this decimal or hexadecimal room ID",
    )
    p_practice.add_argument(
        "--room-debounce",
        type=int,
        default=3,
        help="Stable frames required to confirm a room split (default: 3)",
    )
    p_practice.add_argument(
        "--until-playable",
        action="store_true",
        help="At --until-room, wait for game_state 8 and no door transition",
    )
    p_practice.add_argument(
        "--until-label",
        help="Optional display label stored with the target-room recording",
    )

    # play (record)
    p_play = sub.add_parser("play", help="Play a level manually and record inputs")
    p_play.add_argument("--scale", type=int, default=3)
    p_play.add_argument("--state", help="Override start state (e.g. ResumeRun)")

    # selftest
    sub.add_parser("selftest", help="Run self-tests")

    # list-routes
    sub.add_parser("list-routes", help="List all registered speedrun routes")

    # chain (evaluate a full route)
    p_chain = sub.add_parser("chain", help="Evaluate a full speedrun route")
    p_chain.add_argument("--route", "-r", required=True, help="Route ID or alias")

    # chain-live (true end-to-end on single emulator)
    p_clive = sub.add_parser("chain-live", help="True end-to-end chain on single emulator (no state reloads)")
    p_clive.add_argument("--route", "-r", required=True, help="Route ID or alias")
    p_clive.add_argument("--save-states", action="store_true", help="Save chained states at each segment boundary")
    p_clive.add_argument("--video", help="Output MP4 path (optional)")
    p_clive.add_argument("--scale", type=int, default=3, help="Video pixel scale (default 3)")

    # chain-optimize (iterative hill climb from chained states)
    p_copt = sub.add_parser("chain-optimize", help="Iteratively hill-climb segments from chained states")
    p_copt.add_argument("--route", "-r", required=True, help="Route ID or alias")
    p_copt.add_argument("--iterations", type=int, default=2000, help="Hill climb iterations per segment (default 2000)")

    # chain-video (render full route to MP4)
    p_cvid = sub.add_parser("chain-video", help="Render a full speedrun route to MP4")
    p_cvid.add_argument("--route", "-r", required=True, help="Route ID or alias")
    p_cvid.add_argument("--output", "-o", help="Output MP4 path")
    p_cvid.add_argument("--scale", type=int, default=3, help="Pixel scale (default 3)")

    # trace-map
    p_trace = sub.add_parser("trace-map", help="Render position trace on area map")
    p_trace.add_argument("--trace", help="Path to trace JSON (auto-detected if omitted)")
    p_trace.add_argument("--actions", help="Actions file (to find trace alongside it)")
    p_trace.add_argument("--area", help="Area name: crateria, brinstar, norfair, etc.")
    p_trace.add_argument("-o", "--output", help="Output PNG path")
    p_trace.add_argument("--map-dir", help="Override map PNG directory")

    # neuro (neuroevolution)
    p_neuro = sub.add_parser("neuro", help="Neuroevolution optimizer (evolve neural networks)")
    p_neuro.add_argument("--state", help="Override start state")
    p_neuro.add_argument("--population", type=int, default=100, help="Population size (default 100)")
    p_neuro.add_argument("--generations", type=int, default=300, help="Number of generations (default 300)")
    p_neuro.add_argument("--hidden", type=int, default=20, help="Hidden layer size (default 20)")
    p_neuro.add_argument("--max-frames", type=int, default=6000, help="Max frames per evaluation (default 6000)")
    p_neuro.add_argument("--output-dir", help="Output directory (default: runs_dir/neuro)")
    p_neuro.add_argument("--render", action="store_true", help="Render best network live each generation")
    p_neuro.add_argument("--scale", type=int, default=3, help="Render pixel scale (default 3)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Validate --level is provided for commands that need it
    needs_level = args.command not in (
        "list-levels",
        "list-routes",
        "chain",
        "chain-live",
        "chain-optimize",
        "chain-video",
    )
    # analyze-seed can run static-only without a level
    if args.command == "analyze-seed" and getattr(args, "static_only", False):
        needs_level = False
    if needs_level and not args.level:
        parser.error(f"--level is required for '{args.command}'. Use 'list-levels' to see available levels.")

    commands = {
        "list-levels": cmd_list_levels,
        "extract": cmd_extract,
        "extract-all": cmd_extract_all,
        "verify": cmd_verify,
        "optimize": cmd_optimize,
        "hillclimb": cmd_hillclimb,
        "hillclimb-raw": cmd_hillclimb_raw,
        "analyze-seed": cmd_analyze_seed,
        "trim-seed": cmd_trim_seed,
        "segment-hillclimb": cmd_segment_hillclimb,
        "watch": cmd_watch,
        "watch-bk2": cmd_watch_bk2,
        "prepare-seeds": cmd_prepare_seeds,
        "auto-state": cmd_auto_state,
        "practice": cmd_practice,
        "play": cmd_play,
        "selftest": cmd_selftest,
        "trace-map": cmd_trace_map,
        "list-routes": cmd_list_routes,
        "chain": cmd_chain,
        "chain-live": cmd_chain_live,
        "chain-optimize": cmd_chain_optimize,
        "chain-video": cmd_chain_video,
        "neuro": cmd_neuro,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
