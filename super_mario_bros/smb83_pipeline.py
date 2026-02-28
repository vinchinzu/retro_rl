#!/usr/bin/env python3
"""Reproducible raw-button training/eval pipeline for SMB level 8-3.

Why this exists:
- SMB recordings can include button combos not representable by discrete action indices.
- The generic optimizer CLI requires a seed even when resuming.
- For 8-3 we want a raw-first pipeline with resume-only support and repeatable eval.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

# Ensure repo root is importable when script is invoked directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import platformer_common.levels.smb  # noqa: F401 - register SMB levels
from platformer_common.bk2_extract import load_actions, load_raw_buttons
from platformer_common.evaluator import Evaluator, EvalResult
from platformer_common.genetic import run_ga_raw
from platformer_common.level_config import get_level_config

DEFAULT_LEVEL = "smb_8_3"
DEFAULT_SEED = ROOT / "super_mario_bros" / "optimizer" / "runs" / "smb_8_3" / "recording_000.json"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SMB 8-3 raw training/evaluation pipeline")
    sub = p.add_subparsers(dest="command", required=True)

    p_audit = sub.add_parser("audit", help="Compare raw vs action-index playback for a seed file")
    p_audit.add_argument("--level", default=DEFAULT_LEVEL)
    p_audit.add_argument("--seed", type=Path, default=DEFAULT_SEED)
    p_audit.add_argument("--state", help="Optional state override")

    p_train = sub.add_parser("train", help="Run raw GA training with optional resume")
    p_train.add_argument("--level", default=DEFAULT_LEVEL)
    p_train.add_argument("--seed", type=Path, help="Seed JSON with raw companion (_raw.json)")
    p_train.add_argument("--seeds-dir", type=Path, help="Directory containing attempt_*.json/recording_*.json")
    p_train.add_argument("--resume", type=Path, help="Resume checkpoint with raw_buttons")
    p_train.add_argument("--population", type=int, default=12)
    p_train.add_argument("--generations", type=int, default=6)
    p_train.add_argument("--output-dir", type=Path)
    p_train.add_argument("--state", help="Optional state override")
    p_train.add_argument("--validation-runs", type=int, default=5)

    p_eval = sub.add_parser("eval", help="Evaluate a raw action file repeatedly")
    p_eval.add_argument("--level", default=DEFAULT_LEVEL)
    p_eval.add_argument("--actions", type=Path, help="Action JSON (raw_buttons or companion _raw)")
    p_eval.add_argument("--output-dir", type=Path, help="If --actions omitted, read <output-dir>/ga_raw_best_final.json")
    p_eval.add_argument("--runs", type=int, default=20)
    p_eval.add_argument("--state", help="Optional state override")

    return p.parse_args()


def _default_output_dir(level: str) -> Path:
    cfg = get_level_config(level)
    return cfg.runs_dir / "raw_ga_8_3"


def _load_raw(path: Path) -> list[list[int]]:
    raw = load_raw_buttons(path)
    if raw is None:
        raise ValueError(f"No raw_buttons found for {path}")
    return raw


def _load_raw_from_dir(seeds_dir: Path, min_frames: int = 60) -> list[list[list[int]]]:
    files = sorted(list(seeds_dir.glob("attempt_*.json")) + list(seeds_dir.glob("recording_*.json")))
    files = [f for f in files if "_raw" not in f.stem]

    seeds: list[list[list[int]]] = []
    for f in files:
        try:
            raw = load_raw_buttons(f)
            if raw is None or len(raw) < min_frames:
                continue
            seeds.append(raw)
        except Exception:
            continue
    return seeds


def _read_resume_checkpoint(path: Path) -> tuple[list[list[int]], int]:
    data = json.loads(path.read_text())
    if "raw_buttons" in data:
        raw = data["raw_buttons"]
    else:
        raw = _load_raw(path)
    gen = int(data.get("generation", 0))
    return raw, gen


def _eval_many(level: str, actions: list[list[int]], runs: int, state: str | None = None) -> list[EvalResult]:
    cfg = get_level_config(level)
    evaluator = Evaluator(cfg, start_state=state)
    try:
        out: list[EvalResult] = []
        for _ in range(runs):
            out.append(evaluator.evaluate(actions, early_terminate=False))
        return out
    finally:
        evaluator.close()


def _summarize(results: list[EvalResult]) -> dict[str, Any]:
    frames = [r.total_frames for r in results]
    progress = [r.max_progress for r in results]
    fitness = [r.fitness for r in results]
    completed = sum(1 for r in results if r.completed)
    return {
        "runs": len(results),
        "completed": completed,
        "completion_rate": completed / max(len(results), 1),
        "frames_min": min(frames),
        "frames_max": max(frames),
        "frames_mean": statistics.mean(frames),
        "progress_min": min(progress),
        "progress_max": max(progress),
        "fitness_mean": statistics.mean(fitness),
    }


def _print_summary(summary: dict[str, Any]) -> None:
    print(f"Runs:            {summary['runs']}")
    print(f"Completed:       {summary['completed']}/{summary['runs']} ({summary['completion_rate'] * 100:.1f}%)")
    print(
        f"Frames (min/max/avg): {summary['frames_min']} / {summary['frames_max']} / {summary['frames_mean']:.1f}"
    )
    print(
        f"Progress (min/max):   {summary['progress_min']:.1f} / {summary['progress_max']:.1f}"
    )
    print(f"Fitness avg:      {summary['fitness_mean']:.1f}")


def _save_raw_final(path: Path, raw_buttons: list[list[int]], result: EvalResult, generation: int, level: str) -> None:
    data = {
        "raw_buttons": raw_buttons,
        "num_frames": len(raw_buttons),
        "fitness": result.fitness,
        "completed": result.completed,
        "total_frames": result.total_frames,
        "max_progress": result.max_progress,
        "generation": generation,
        "level": level,
    }
    path.write_text(json.dumps(data, indent=2))


def cmd_audit(args: argparse.Namespace) -> int:
    seed = args.seed.resolve()
    if not seed.exists():
        print(f"Error: seed file not found: {seed}")
        return 1

    cfg = get_level_config(args.level)
    evaluator = Evaluator(cfg, start_state=args.state)

    try:
        raw = _load_raw(seed)
        idx = load_actions(seed)

        raw_result = evaluator.evaluate(raw, early_terminate=False)
        idx_result = evaluator.evaluate(idx, early_terminate=False)

        print(f"Level: {args.level}")
        print(f"Seed:  {seed}")
        print("\nRaw replay:")
        print(
            f"  completed={raw_result.completed} died={raw_result.died} "
            f"frames={raw_result.total_frames} progress={raw_result.max_progress:.1f} "
            f"fitness={raw_result.fitness:.1f}"
        )
        print("Action-index replay:")
        print(
            f"  completed={idx_result.completed} died={idx_result.died} "
            f"frames={idx_result.total_frames} progress={idx_result.max_progress:.1f} "
            f"fitness={idx_result.fitness:.1f}"
        )

        if raw_result.completed and not idx_result.completed:
            print("\nResult: index mapping is lossy for this seed; use raw-button optimization.")
        return 0
    finally:
        evaluator.close()


def cmd_train(args: argparse.Namespace) -> int:
    level = args.level
    cfg = get_level_config(level)
    output_dir = args.output_dir.resolve() if args.output_dir else _default_output_dir(level)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.seed and args.seeds_dir:
        print("Error: --seed and --seeds-dir are mutually exclusive.")
        return 1

    seeds: list[list[list[int]]] = []
    resume_generation = 0

    if args.resume:
        resume_path = args.resume.resolve()
        if not resume_path.exists():
            print(f"Error: resume checkpoint not found: {resume_path}")
            return 1
        resume_seed, resume_generation = _read_resume_checkpoint(resume_path)
        seeds.append(resume_seed)
        print(f"Resume seed: {len(resume_seed)} raw frames from {resume_path} (generation={resume_generation})")

    if args.seeds_dir:
        seeds_dir = args.seeds_dir.resolve()
        if not seeds_dir.exists():
            print(f"Error: seeds directory not found: {seeds_dir}")
            return 1
        dir_seeds = _load_raw_from_dir(seeds_dir)
        if not dir_seeds:
            print(f"Error: no usable raw seeds in {seeds_dir}")
            return 1
        seeds.extend(dir_seeds)
        print(f"Loaded {len(dir_seeds)} raw seeds from {seeds_dir}")
    elif args.seed:
        seed_path = args.seed.resolve()
        if not seed_path.exists():
            print(f"Error: seed file not found: {seed_path}")
            return 1
        raw = _load_raw(seed_path)
        seeds.append(raw)
        print(f"Seed: {len(raw)} raw frames from {seed_path}")
    elif not args.resume:
        default_seed = DEFAULT_SEED
        if not default_seed.exists():
            print("Error: no --seed/--seeds-dir/--resume given and default seed missing.")
            return 1
        raw = _load_raw(default_seed)
        seeds.append(raw)
        print(f"Seed: {len(raw)} raw frames from {default_seed}")

    # Deduplicate by length + first/last frame signatures to keep population diverse.
    dedup: dict[tuple[int, tuple[int, ...], tuple[int, ...]], list[list[int]]] = {}
    for s in seeds:
        if not s:
            continue
        key = (len(s), tuple(s[0]), tuple(s[-1]))
        dedup.setdefault(key, s)
    seeds = list(dedup.values())

    if not seeds:
        print("Error: no seeds available.")
        return 1

    print(f"Level:        {cfg.display_name}")
    print(f"Population:   {args.population}")
    print(f"Generations:  {args.generations}")
    print(f"Output dir:   {output_dir}")
    if args.state:
        print(f"State override: {args.state}")

    evaluator = Evaluator(cfg, start_state=args.state)
    try:
        best = run_ga_raw(
            seeds=seeds,
            evaluator=evaluator,
            population_size=args.population,
            num_generations=args.generations,
            output_dir=output_dir,
            verbose=True,
        )

        if best.result is None:
            best_result = evaluator.evaluate(best.actions, early_terminate=False)
        else:
            best_result = best.result

        final_generation = resume_generation + args.generations

        # Normalize final artifacts with cumulative generation number for resume-only workflows.
        final_path = output_dir / "ga_raw_best_final.json"
        _save_raw_final(final_path, best.actions, best_result, final_generation, level)

        ckpt_path = output_dir / "ga_raw_best.json"
        if ckpt_path.exists():
            ckpt = json.loads(ckpt_path.read_text())
            ckpt["generation"] = final_generation
            ckpt_path.write_text(json.dumps(ckpt, indent=2))

        print("\nBest result:")
        print(
            f"  completed={best_result.completed} frames={best_result.total_frames} "
            f"progress={best_result.max_progress:.1f} fitness={best_result.fitness:.1f}"
        )
        print(f"Saved: {final_path}")

        val_runs = max(args.validation_runs, 1)
        print(f"\nValidation ({val_runs} repeated eval runs):")
        val_results: list[EvalResult] = []
        for _ in range(val_runs):
            val_results.append(evaluator.evaluate(best.actions, early_terminate=False))
        summary = _summarize(val_results)
        _print_summary(summary)

        return 0
    finally:
        evaluator.close()


def cmd_eval(args: argparse.Namespace) -> int:
    level = args.level
    output_dir = args.output_dir.resolve() if args.output_dir else _default_output_dir(level)

    if args.actions:
        actions_path = args.actions.resolve()
    else:
        actions_path = output_dir / "ga_raw_best_final.json"

    if not actions_path.exists():
        print(f"Error: actions file not found: {actions_path}")
        return 1

    actions = _load_raw(actions_path)
    print(f"Level:  {level}")
    print(f"Input:  {actions_path}")
    print(f"Frames: {len(actions)}")

    summary = _summarize(_eval_many(level, actions, max(args.runs, 1), state=args.state))
    _print_summary(summary)
    return 0


def main() -> int:
    args = _parse_args()
    if args.command == "audit":
        return cmd_audit(args)
    if args.command == "train":
        return cmd_train(args)
    if args.command == "eval":
        return cmd_eval(args)
    print(f"Unknown command: {args.command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
