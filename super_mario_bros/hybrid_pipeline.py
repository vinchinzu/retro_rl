#!/usr/bin/env python3
"""Hybrid SMB route optimizer with immutable model registry.

Goals:
- Never overwrite existing best artifacts by default.
- Mine useful segments from saved runs and build splice seeds.
- Compose multiple sub-agents per weak segment (replay/GA/hillclimb/NEAT/PPO input).
- Produce a reproducible best-times report for the full route.

Usage examples:
    # Analyze current artifacts only
    uv run python -m super_mario_bros.hybrid_pipeline analyze

    # Run lightweight hybrid improvement on weakest 3 segments
    uv run python -m super_mario_bros.hybrid_pipeline run \
      --weak-top-k 3 \
      --ga-generations 6 --ga-population 20 \
      --hill-iterations 600 --hill-raw-iterations 600
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DEFAULT_REGISTRY = ROOT / "optimizer" / "model_registry.json"
DEFAULT_REPORT = ROOT / "optimizer" / "hybrid_report.json"

CANDIDATE_PRIORITY = [
    "hybrid_best_final.json",
    "hillclimb_best_final.json",
    "hillclimb_raw_best_final.json",
    "hillclimb_raw_best.json",
    "ga_best_final.json",
    "ga_raw_best_final.json",
    "ga_raw_best.json",
    "combined_82_complete.json",
    "recording_chained_complete.json",
    "recording_complete.json",
    "recording_000.json",
    "recording_001.json",
    "recording_002.json",
    "chained/hillclimb_best_final.json",
    "neuro/neuro_best_buttons.json",
]

CANDIDATE_GLOBS = [
    "recording_*.json",
    "attempt_*.json",
    "hillclimb*.json",
    "ga*.json",
    "combined*.json",
    "seed_*.json",
    "chained/*.json",
    "neuro/neuro_best_buttons.json",
    "candidates/**/*.json",
]

CHAINED_STATE_CANDIDATES: dict[str, list[str]] = {
    "smb_1_1": ["Chained_1-1", "Chained_smb_1_1"],
    "smb_1_2": ["Chained_1-2_toW4"],
    "smb_4_1": ["Chained_4-1"],
    "smb_4_2": ["Chained_4-2_toW8"],
    "smb_8_1": ["Chained_8-1"],
    "smb_8_2": ["Chained_8-2", "Chained_8-2_mid", "Chained_8-2_late"],
    "smb_8_3": ["Chained_8-3", "Chained_8-3_late"],
    "smb_8_4": ["Chained_8-4", "Chained_8-4_late"],
    "smb_8_4_1": ["Chained_8-4_seg1"],
    "smb_8_4_2": ["Chained_8-4_seg2"],
    "smb_8_4_3": ["Chained_8-4_seg3"],
    "smb_8_4_4": ["Chained_8-4_seg4"],
    "smb_8_4_5": ["Chained_8-4_seg5"],
}


def _register_levels() -> None:
    # Keep heavy imports lazy so unit tests for pure helpers can run without emulator deps.
    import platformer_common.levels.smb  # noqa: F401


def _get_level_config(config_id: str):
    _register_levels()
    from platformer_common.level_config import get_level_config

    return get_level_config(config_id)


def _get_route(route_id: str):
    _register_levels()
    from platformer_common.route import get_route

    return get_route(route_id)


def _load_recording_data(path: Path) -> tuple[list[int] | list[list[int]], bool]:
    from platformer_common.route import load_recording_data

    return load_recording_data(path)


def _action_index_to_buttons(idx: int, action_table: list[list[int]]) -> list[int]:
    from platformer_common.actions import action_index_to_buttons

    return action_index_to_buttons(idx, action_table=action_table)


def _buttons_to_action_index(buttons: list[int], action_table: list[list[int]]) -> int:
    from platformer_common.actions import buttons_to_action_index

    return buttons_to_action_index(buttons, action_table=action_table)


@dataclass
class CandidateMetric:
    path: str
    algorithm: str
    source: str
    is_raw: bool
    completed: bool
    completion_rate: float
    died_rate: float
    frames_mean: float
    total_frames: int
    fitness_mean: float
    max_progress: float
    num_actions: int
    notes: list[str] = field(default_factory=list)


@dataclass
class SegmentSummary:
    label: str
    config_id: str
    run_dir: str
    selected: CandidateMetric | None = None
    state_override: str = ""
    candidates: list[CandidateMetric] = field(default_factory=list)
    weak_score: float = 0.0
    sub_agents: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class SeedBundle:
    raw: list[list[list[int]]] = field(default_factory=list)
    indices: list[list[int]] = field(default_factory=list)


@dataclass
class HybridSettings:
    route: str = "smb_any_percent"
    selection_context: str = "standalone"  # standalone|chained
    max_candidates: int = 32
    eval_runs: int = 1
    force_eval: bool = False
    weak_top_k: int = 3
    ga_generations: int = 0
    ga_population: int = 20
    hill_iterations: int = 600
    hill_raw_iterations: int = 600
    neuro_generations: int = 0
    neuro_population: int = 24
    neuro_hidden: int = 20
    neuro_max_frames: int = 4500
    ppo_candidates_dir: str = ""
    ppo_command: str = ""
    ppo_output_name: str = "ppo_candidate.json"
    segments: str = ""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def infer_algorithm(path: Path) -> str:
    name = path.name.lower()
    parts = {p.lower() for p in path.parts}
    if name.startswith("seed_"):
        return "mined_seed"
    if "neuro" in parts or name.startswith("neuro"):
        return "neat"
    if "ppo" in name:
        return "ppo"
    if name.startswith("ga_raw"):
        return "ga_raw"
    if name.startswith("ga"):
        return "ga"
    if name.startswith("hillclimb_raw") or "raw_best" in name:
        return "hillclimb_raw"
    if name.startswith("hillclimb"):
        return "hillclimb"
    if name.startswith("recording") or name.startswith("attempt") or name.startswith("combined"):
        return "replay"
    return "unknown"


def candidate_rank_key(row: CandidateMetric) -> tuple[float, float, float, float]:
    """Lower is better."""
    if row.completion_rate > 0.0:
        return (0.0, -row.completion_rate, row.frames_mean, -row.fitness_mean)
    return (1.0, -row.max_progress, -row.fitness_mean, row.frames_mean)


def is_better(a: CandidateMetric, b: CandidateMetric) -> bool:
    return candidate_rank_key(a) < candidate_rank_key(b)


def weakness_score(row: CandidateMetric) -> float:
    """Higher means weaker segment."""
    if row.completion_rate < 1.0:
        # Prioritize unstable/incomplete segments first.
        return 1_000_000.0 + (1.0 - row.completion_rate) * 100_000.0 + max(0.0, 7000.0 - row.max_progress)
    return row.frames_mean


def build_sub_agent_plan(
    baseline: CandidateMetric,
    *,
    has_raw: bool,
    use_ga: bool,
    use_neuro: bool,
    use_ppo: bool,
) -> list[str]:
    plan = ["replay"]
    if baseline.completion_rate < 1.0 and use_ga and has_raw:
        plan.append("ga_raw")
    if has_raw:
        plan.append("hillclimb_raw")
    else:
        plan.append("hillclimb")
    if use_neuro:
        plan.append("neuro")
    if use_ppo:
        plan.append("ppo")
    return plan


def splice_sequences(
    seq_a: list[Any],
    seq_b: list[Any],
    *,
    fractions: tuple[float, ...] = (0.33, 0.5, 0.67),
) -> list[list[Any]]:
    out: list[list[Any]] = []
    if len(seq_a) < 4 or len(seq_b) < 4:
        return out
    for frac in fractions:
        cut_a = max(1, min(len(seq_a) - 1, int(len(seq_a) * frac)))
        cut_b = max(1, min(len(seq_b) - 1, int(len(seq_b) * frac)))
        out.append(list(seq_a[:cut_a]) + list(seq_b[cut_b:]))
    return out


def mine_splice_seeds(
    seeds: list[list[Any]],
    *,
    max_generated: int = 6,
) -> list[list[Any]]:
    if len(seeds) < 2:
        return []
    # Prefer stronger seeds (longer sequences usually capture more route structure).
    ranked = sorted(seeds, key=len, reverse=True)[:4]
    generated: list[list[Any]] = []
    dedupe: set[str] = set()

    for i in range(len(ranked)):
        for j in range(i + 1, len(ranked)):
            for child in splice_sequences(ranked[i], ranked[j]):
                key = hashlib.sha1(json.dumps(child, separators=(",", ":")).encode("utf-8")).hexdigest()
                if key in dedupe:
                    continue
                dedupe.add(key)
                generated.append(child)
                if len(generated) >= max_generated:
                    return generated
            for child in splice_sequences(ranked[j], ranked[i]):
                key = hashlib.sha1(json.dumps(child, separators=(",", ":")).encode("utf-8")).hexdigest()
                if key in dedupe:
                    continue
                dedupe.add(key)
                generated.append(child)
                if len(generated) >= max_generated:
                    return generated
    return generated


def _candidate_priority_key(path: Path, runs_dir: Path) -> tuple[int, str]:
    rel = str(path.relative_to(runs_dir)) if path.is_relative_to(runs_dir) else str(path)
    for idx, name in enumerate(CANDIDATE_PRIORITY):
        if rel.endswith(name):
            return idx, rel
    if path.name.startswith("ga_gen") and path.name.endswith("_best.json"):
        return len(CANDIDATE_PRIORITY), rel
    return len(CANDIDATE_PRIORITY) + 1, rel


def _skip_candidate(path: Path) -> bool:
    if not path.is_file() or path.suffix != ".json":
        return True
    stem = path.stem
    name = path.name
    if stem.endswith("_trace") or stem.endswith("_annotations"):
        return True
    if "mined" in path.parts and name.startswith("seed_"):
        return True
    if name.startswith("recording_") and stem.endswith("_raw"):
        return True
    if name.startswith("attempt_") and stem.endswith("_raw"):
        return True
    return False


def discover_candidate_files(runs_dir: Path, *, max_candidates: int) -> list[Path]:
    paths: list[Path] = []
    for pattern in CANDIDATE_GLOBS:
        paths.extend(runs_dir.glob(pattern))

    unique: dict[Path, None] = {}
    for p in paths:
        if _skip_candidate(p):
            continue
        unique[p.resolve()] = None

    resolved = list(unique.keys())
    resolved.sort(key=lambda p: _candidate_priority_key(p, runs_dir))
    return resolved[:max(1, max_candidates)]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _static_metrics(path: Path) -> dict[str, Any] | None:
    try:
        data = _load_json(path)
    except Exception:
        return None

    has_any = any(k in data for k in ("completed", "total_frames", "fitness", "max_progress"))
    if not has_any:
        return None

    return {
        "completed": bool(data.get("completed", False)),
        "completion_rate": 1.0 if data.get("completed", False) else 0.0,
        "died_rate": 0.0,
        "frames_mean": float(data.get("total_frames", data.get("num_frames", 0))),
        "total_frames": int(data.get("total_frames", data.get("num_frames", 0))),
        "fitness_mean": float(data.get("fitness", 0.0)),
        "max_progress": float(data.get("max_progress", 0.0)),
        "num_actions": int(data.get("num_frames", data.get("total_frames", 0))),
        "is_raw": bool("raw_buttons" in data),
    }


def _evaluate_actions(
    config_id: str,
    actions: list[int] | list[list[int]],
    *,
    eval_runs: int,
    start_state: str | None = None,
) -> dict[str, Any]:
    from platformer_common.evaluator import Evaluator

    cfg = _get_level_config(config_id)
    ev = Evaluator(cfg, start_state=start_state)
    try:
        results = [ev.evaluate(actions, early_terminate=False) for _ in range(max(1, eval_runs))]
    finally:
        ev.close()

    completion_rate = sum(1.0 for r in results if r.completed) / len(results)
    died_rate = sum(1.0 for r in results if r.died) / len(results)
    frames = [float(r.total_frames) for r in results]
    fitness = [float(r.fitness) for r in results]
    max_progress = max(float(r.max_progress) for r in results)

    return {
        "completed": completion_rate >= 0.999,
        "completion_rate": completion_rate,
        "died_rate": died_rate,
        "frames_mean": statistics.mean(frames),
        "total_frames": int(round(statistics.mean(frames))),
        "fitness_mean": statistics.mean(fitness),
        "max_progress": max_progress,
    }


def evaluate_candidate(
    config_id: str,
    path: Path,
    *,
    eval_runs: int,
    force_eval: bool,
    source: str,
    start_state: str | None = None,
) -> CandidateMetric:
    algo = infer_algorithm(path)
    static = _static_metrics(path)

    if static is not None and not force_eval:
        companion_raw = path.with_name(f"{path.stem}_raw.json").exists()
        return CandidateMetric(
            path=str(path),
            algorithm=algo,
            source=source,
            is_raw=bool(static.get("is_raw", False) or companion_raw),
            completed=bool(static["completed"]),
            completion_rate=float(static["completion_rate"]),
            died_rate=float(static["died_rate"]),
            frames_mean=float(static["frames_mean"]),
            total_frames=int(static["total_frames"]),
            fitness_mean=float(static["fitness_mean"]),
            max_progress=float(static["max_progress"]),
            num_actions=int(static["num_actions"]),
        )

    actions, is_raw = _load_recording_data(path)
    live = _evaluate_actions(config_id, actions, eval_runs=eval_runs, start_state=start_state)

    return CandidateMetric(
        path=str(path),
        algorithm=algo,
        source=source,
        is_raw=is_raw,
        completed=bool(live["completed"]),
        completion_rate=float(live["completion_rate"]),
        died_rate=float(live["died_rate"]),
        frames_mean=float(live["frames_mean"]),
        total_frames=int(live["total_frames"]),
        fitness_mean=float(live["fitness_mean"]),
        max_progress=float(live["max_progress"]),
        num_actions=len(actions),
    )


def _segment_filter(settings: HybridSettings) -> set[str]:
    if not settings.segments.strip():
        return set()
    return {chunk.strip() for chunk in settings.segments.split(",") if chunk.strip()}


def _state_exists(config_id: str, state_name: str) -> bool:
    cfg = _get_level_config(config_id)
    state_path = cfg.game_dir / "custom_integrations" / cfg.game_name / f"{state_name}.state"
    return state_path.exists()


def _resolve_state_for_context(config_id: str, context: str) -> str | None:
    if context != "chained":
        return None
    for state_name in CHAINED_STATE_CANDIDATES.get(config_id, []):
        if _state_exists(config_id, state_name):
            return state_name
    return None


def analyze_route(settings: HybridSettings) -> list[SegmentSummary]:
    route = _get_route(settings.route)
    selected_segments = _segment_filter(settings)
    context = settings.selection_context

    out: list[SegmentSummary] = []

    for seg in route.segments:
        if selected_segments and seg.config_id not in selected_segments:
            continue

        cfg = _get_level_config(seg.config_id)
        run_dir = cfg.runs_dir
        notes: list[str] = []
        rows: list[CandidateMetric] = []
        state_override = _resolve_state_for_context(seg.config_id, context)
        if context == "chained" and state_override is None:
            notes.append("no_chained_state")

        if not run_dir.exists():
            out.append(
                SegmentSummary(
                    label=seg.label or seg.config_id,
                    config_id=seg.config_id,
                    run_dir=str(run_dir),
                    state_override=state_override or "",
                    selected=None,
                    candidates=[],
                    notes=["run_dir_missing"],
                )
            )
            continue

        candidates = discover_candidate_files(run_dir, max_candidates=settings.max_candidates)
        if not candidates:
            out.append(
                SegmentSummary(
                    label=seg.label or seg.config_id,
                    config_id=seg.config_id,
                    run_dir=str(run_dir),
                    state_override=state_override or "",
                    selected=None,
                    candidates=[],
                    notes=["no_candidates"],
                )
            )
            continue

        force_eval = settings.force_eval or context == "chained"

        for path in candidates:
            try:
                rows.append(
                    evaluate_candidate(
                        seg.config_id,
                        path,
                        eval_runs=settings.eval_runs,
                        force_eval=force_eval,
                        source="existing",
                        start_state=state_override,
                    )
                )
            except Exception as exc:  # pragma: no cover - runtime dependent
                notes.append(f"eval_error:{path.name}:{exc}")

        rows.sort(key=candidate_rank_key)
        best = rows[0] if rows else None
        weak = weakness_score(best) if best is not None else 1_000_000_000.0

        out.append(
            SegmentSummary(
                label=seg.label or seg.config_id,
                config_id=seg.config_id,
                run_dir=str(run_dir),
                state_override=state_override or "",
                selected=best,
                candidates=rows,
                weak_score=weak,
                notes=notes,
            )
        )

    return out


def _candidate_to_raw(actions: list[int], action_table: list[list[int]]) -> list[list[int]]:
    return [_action_index_to_buttons(a, action_table=action_table) for a in actions]


def _candidate_to_indices(raw_buttons: list[list[int]], action_table: list[list[int]]) -> list[int]:
    return [_buttons_to_action_index(frame, action_table=action_table) for frame in raw_buttons]


def extract_seed_bundle(
    config_id: str,
    candidates: list[CandidateMetric],
    *,
    max_seed_sources: int = 6,
) -> SeedBundle:
    cfg = _get_level_config(config_id)
    from platformer_common.actions import DEFAULT_PLATFORMER_ACTIONS

    action_table = cfg.action_table or DEFAULT_PLATFORMER_ACTIONS

    out = SeedBundle()
    used = 0

    for row in candidates:
        if used >= max_seed_sources:
            break
        path = Path(row.path)
        try:
            actions, is_raw = _load_recording_data(path)
        except Exception:
            continue

        if is_raw:
            raw = [list(frame) for frame in actions]  # type: ignore[arg-type]
            out.raw.append(raw)
            out.indices.append(_candidate_to_indices(raw, action_table))
        else:
            idx = [int(a) for a in actions]  # type: ignore[arg-type]
            out.indices.append(idx)
            out.raw.append(_candidate_to_raw(idx, action_table))

        used += 1

    return out


def _write_mined_seeds(stage_dir: Path, bundle: SeedBundle) -> None:
    mined_dir = stage_dir / "mined"
    mined_dir.mkdir(parents=True, exist_ok=True)

    for i, seed in enumerate(bundle.indices):
        payload = {"actions": seed, "num_frames": len(seed), "source": "hybrid_mined"}
        (mined_dir / f"seed_indices_{i:02d}.json").write_text(json.dumps(payload, indent=2))

    for i, seed in enumerate(bundle.raw):
        payload = {"raw_buttons": seed, "num_frames": len(seed), "source": "hybrid_mined"}
        (mined_dir / f"seed_raw_{i:02d}.json").write_text(json.dumps(payload, indent=2))


def _run_ga_raw_stage(
    config_id: str,
    raw_seeds: list[list[list[int]]],
    *,
    generations: int,
    population: int,
    stage_dir: Path,
    start_state: str | None,
) -> Path | None:
    if generations <= 0 or population <= 1 or len(raw_seeds) < 2:
        return None

    from platformer_common.evaluator import Evaluator
    from platformer_common.genetic import run_ga_raw

    cfg = _get_level_config(config_id)
    stage_dir.mkdir(parents=True, exist_ok=True)
    evaluator = Evaluator(cfg, start_state=start_state)
    try:
        run_ga_raw(
            seeds=raw_seeds,
            evaluator=evaluator,
            population_size=population,
            num_generations=generations,
            output_dir=stage_dir,
            verbose=True,
        )
    finally:
        evaluator.close()

    out = stage_dir / "ga_raw_best.json"
    return out if out.exists() else None


def _run_hill_stage(
    config_id: str,
    *,
    seed_indices: list[int] | None,
    seed_raw: list[list[int]] | None,
    hill_iterations: int,
    hill_raw_iterations: int,
    stage_dir: Path,
    start_state: str | None,
) -> Path | None:
    from platformer_common.evaluator import Evaluator
    from platformer_common.hillclimb import hillclimb
    from platformer_common.hillclimb_raw import hillclimb_raw

    cfg = _get_level_config(config_id)
    stage_dir.mkdir(parents=True, exist_ok=True)

    if seed_raw is not None and hill_raw_iterations > 0:
        evaluator = Evaluator(cfg, start_state=start_state)
        try:
            best_raw, best_result = hillclimb_raw(
                raw_buttons=seed_raw,
                evaluator=evaluator,
                max_iterations=hill_raw_iterations,
                output_dir=stage_dir,
                verbose=True,
            )
        finally:
            evaluator.close()

        out = stage_dir / "hillclimb_raw_best.json"
        if not out.exists():
            payload = {
                "raw_buttons": best_raw,
                "num_frames": len(best_raw),
                "completed": best_result.completed,
                "total_frames": best_result.total_frames,
                "fitness": best_result.fitness,
                "max_progress": best_result.max_progress,
            }
            out.write_text(json.dumps(payload, indent=2))
        return out

    if seed_indices is not None and hill_iterations > 0:
        evaluator = Evaluator(cfg, start_state=start_state)
        try:
            best_actions, best_result = hillclimb(
                actions=seed_indices,
                evaluator=evaluator,
                max_iterations=hill_iterations,
                output_dir=stage_dir,
                verbose=True,
            )
        finally:
            evaluator.close()

        out = stage_dir / "hillclimb_best_final.json"
        payload = {
            "actions": best_actions,
            "num_frames": len(best_actions),
            "completed": best_result.completed,
            "total_frames": best_result.total_frames,
            "fitness": best_result.fitness,
            "max_progress": best_result.max_progress,
            "source": "hybrid_hillclimb",
        }
        out.write_text(json.dumps(payload, indent=2))
        return out

    return None


def _run_neuro_stage(
    config_id: str,
    *,
    generations: int,
    population: int,
    hidden: int,
    max_frames: int,
    stage_dir: Path,
    start_state: str | None,
) -> Path | None:
    if generations <= 0 or population <= 1:
        return None

    from platformer_common.evaluator import Evaluator
    from platformer_common.neuro import run_neuro_ga

    cfg = _get_level_config(config_id)
    out_dir = stage_dir / "neuro"
    out_dir.mkdir(parents=True, exist_ok=True)

    evaluator = Evaluator(cfg, start_state=start_state)
    try:
        run_neuro_ga(
            evaluator=evaluator,
            population_size=population,
            num_generations=generations,
            n_hidden=hidden,
            max_frames=max_frames,
            output_dir=out_dir,
            verbose=True,
            render=False,
        )
    finally:
        evaluator.close()

    out = out_dir / "neuro_best_buttons.json"
    return out if out.exists() else None


def _resolve_ppo_candidate(
    config_id: str,
    settings: HybridSettings,
    stage_dir: Path,
    *,
    start_state: str | None,
) -> Path | None:
    if settings.ppo_command:
        cfg = _get_level_config(config_id)
        out_dir = stage_dir / "ppo"
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = settings.ppo_command.format(
            level=config_id,
            state=start_state or cfg.start_state,
            run_dir=str(cfg.runs_dir),
            output_dir=str(out_dir),
        )
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"[PPO] command failed for {config_id}: {proc.stderr.strip()}")
            return None
        out = out_dir / settings.ppo_output_name
        return out if out.exists() else None

    if not settings.ppo_candidates_dir:
        return None

    ppo_dir = Path(settings.ppo_candidates_dir)
    candidates = [
        ppo_dir / f"{config_id}.json",
        ppo_dir / f"{config_id}_candidate.json",
        ppo_dir / f"{config_id}_raw.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def improve_segment(seg: SegmentSummary, settings: HybridSettings, run_id: str) -> SegmentSummary:
    if seg.selected is None:
        return seg

    cfg = _get_level_config(seg.config_id)
    start_state = seg.state_override or None
    stage_root = cfg.runs_dir / "candidates" / run_id
    stage_root.mkdir(parents=True, exist_ok=True)

    has_raw = any(c.is_raw for c in seg.candidates)
    seg.sub_agents = build_sub_agent_plan(
        seg.selected,
        has_raw=has_raw,
        use_ga=settings.ga_generations > 0,
        use_neuro=settings.neuro_generations > 0,
        use_ppo=bool(settings.ppo_candidates_dir or settings.ppo_command),
    )

    bundle = extract_seed_bundle(seg.config_id, seg.candidates)
    bundle.raw.extend(mine_splice_seeds(bundle.raw, max_generated=6))
    bundle.indices.extend(mine_splice_seeds(bundle.indices, max_generated=6))
    _write_mined_seeds(stage_root, bundle)

    generated: list[CandidateMetric] = []

    if "ga_raw" in seg.sub_agents:
        ga_path = _run_ga_raw_stage(
            seg.config_id,
            bundle.raw,
            generations=settings.ga_generations,
            population=settings.ga_population,
            stage_dir=stage_root / "ga_raw",
            start_state=start_state,
        )
        if ga_path is None:
            seg.notes.append("ga_raw_skipped")
        else:
            generated.append(
                evaluate_candidate(
                    seg.config_id,
                    ga_path,
                    eval_runs=settings.eval_runs,
                    force_eval=True,
                    source="generated",
                    start_state=start_state,
                )
            )

    # Hill stage uses best available seed (generated first, then baseline).
    seed_source = generated[0] if generated else seg.selected
    seed_idx: list[int] | None = None
    seed_raw: list[list[int]] | None = None

    try:
        seed_actions, seed_is_raw = _load_recording_data(Path(seed_source.path))
        from platformer_common.actions import DEFAULT_PLATFORMER_ACTIONS

        action_table = cfg.action_table or DEFAULT_PLATFORMER_ACTIONS
        if seed_is_raw:
            seed_raw = [list(f) for f in seed_actions]  # type: ignore[arg-type]
            seed_idx = _candidate_to_indices(seed_raw, action_table)
        else:
            seed_idx = [int(a) for a in seed_actions]  # type: ignore[arg-type]
            seed_raw = _candidate_to_raw(seed_idx, action_table)
    except Exception as exc:
        seg.notes.append(f"seed_load_error:{exc}")

    if "hillclimb_raw" in seg.sub_agents or "hillclimb" in seg.sub_agents:
        hill_path = _run_hill_stage(
            seg.config_id,
            seed_indices=seed_idx,
            seed_raw=seed_raw if "hillclimb_raw" in seg.sub_agents else None,
            hill_iterations=settings.hill_iterations,
            hill_raw_iterations=settings.hill_raw_iterations,
            stage_dir=stage_root / "hill",
            start_state=start_state,
        )
        if hill_path is None:
            seg.notes.append("hill_skipped")
        else:
            generated.append(
                evaluate_candidate(
                    seg.config_id,
                    hill_path,
                    eval_runs=settings.eval_runs,
                    force_eval=True,
                    source="generated",
                    start_state=start_state,
                )
            )

    if "neuro" in seg.sub_agents:
        neuro_path = _run_neuro_stage(
            seg.config_id,
            generations=settings.neuro_generations,
            population=settings.neuro_population,
            hidden=settings.neuro_hidden,
            max_frames=settings.neuro_max_frames,
            stage_dir=stage_root,
            start_state=start_state,
        )
        if neuro_path is None:
            seg.notes.append("neuro_skipped")
        else:
            generated.append(
                evaluate_candidate(
                    seg.config_id,
                    neuro_path,
                    eval_runs=settings.eval_runs,
                    force_eval=True,
                    source="generated",
                    start_state=start_state,
                )
            )

    if "ppo" in seg.sub_agents:
        ppo_path = _resolve_ppo_candidate(
            seg.config_id,
            settings,
            stage_root,
            start_state=start_state,
        )
        if ppo_path is None:
            seg.notes.append("ppo_skipped")
        else:
            try:
                generated.append(
                    evaluate_candidate(
                        seg.config_id,
                        ppo_path,
                        eval_runs=settings.eval_runs,
                        force_eval=True,
                        source="external_ppo",
                        start_state=start_state,
                    )
                )
            except Exception as exc:
                seg.notes.append(f"ppo_eval_error:{exc}")

    if generated:
        seg.candidates.extend(generated)
        seg.candidates.sort(key=candidate_rank_key)
        seg.selected = seg.candidates[0]
        seg.weak_score = weakness_score(seg.selected)

        # Write non-destructive per-segment best for the current hybrid run.
        best_path = cfg.runs_dir / "hybrid_best_final.json"
        best_payload = {
            "selected_path": seg.selected.path,
            "algorithm": seg.selected.algorithm,
            "completed": seg.selected.completed,
            "completion_rate": seg.selected.completion_rate,
            "total_frames": seg.selected.total_frames,
            "frames_mean": seg.selected.frames_mean,
            "fitness_mean": seg.selected.fitness_mean,
            "max_progress": seg.selected.max_progress,
            "source": seg.selected.source,
            "run_id": run_id,
            "updated_at": _utc_now(),
        }
        best_path.write_text(json.dumps(best_payload, indent=2))

    return seg


def summarize_route(segments: list[SegmentSummary]) -> dict[str, Any]:
    total_frames = 0
    completed_count = 0
    seg_rows = []
    for seg in segments:
        row = {
            "label": seg.label,
            "config_id": seg.config_id,
            "state_override": seg.state_override,
            "weak_score": seg.weak_score,
            "sub_agents": seg.sub_agents,
            "notes": seg.notes,
            "selected": asdict(seg.selected) if seg.selected else None,
        }
        if seg.selected and seg.selected.completion_rate > 0.0:
            total_frames += seg.selected.total_frames
            if seg.selected.completed:
                completed_count += 1
        seg_rows.append(row)

    return {
        "segments": seg_rows,
        "total_frames": total_frames,
        "total_seconds": round(total_frames / 60.0, 3),
        "completed_count": completed_count,
        "total_count": len(segments),
        "all_completed": completed_count == len(segments),
    }


def validate_chain_from_segments(route_id: str, segments: list[SegmentSummary]) -> dict[str, Any]:
    from platformer_common.route import RouteConfig, RouteSegment, chain_live

    base_route = _get_route(route_id)
    selected_map = {s.config_id: s.selected for s in segments if s.selected is not None}

    chain_segments: list[RouteSegment] = []
    for seg in base_route.segments:
        selected = selected_map.get(seg.config_id)
        if selected is None:
            continue
        chain_segments.append(
            RouteSegment(
                config_id=seg.config_id,
                label=seg.label,
                recording=selected.path,
                neuro_checkpoint=seg.neuro_checkpoint,
            )
        )

    route = RouteConfig(
        route_id=f"{route_id}_hybrid_selected",
        display_name=f"{base_route.display_name} (hybrid selected)",
        segments=chain_segments,
    )
    res = chain_live(route, save_states=False, verbose=False)

    return {
        "completed_count": res.completed_count,
        "total_count": len(route.segments),
        "all_completed": res.all_completed,
        "total_frames": res.total_frames,
        "total_seconds": round(res.total_frames / 60.0, 3),
        "segments": [
            {
                "label": s.segment.label,
                "config_id": s.segment.config_id,
                "status": s.status,
                "frames": s.frames,
                "recording": str(s.recording_path) if s.recording_path else "",
                "error": s.error,
            }
            for s in res.segments
        ],
    }


def _copy_to_registry(src: Path, registry_root: Path, config_id: str, run_id: str) -> str:
    registry_root.mkdir(parents=True, exist_ok=True)
    artifact_dir = registry_root / "artifacts" / config_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    digest = _sha256(src)[:12]
    dst = artifact_dir / f"{run_id}_{digest}_{src.name}"
    if not dst.exists():
        shutil.copy2(src, dst)
    return str(dst)


def _load_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "version": 1,
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "runs": [],
        }
    return json.loads(path.read_text())


def _append_registry_run(
    path: Path,
    *,
    route_id: str,
    run_id: str,
    settings: HybridSettings,
    baseline: dict[str, Any],
    improved: dict[str, Any],
) -> None:
    data = _load_registry(path)
    data["updated_at"] = _utc_now()

    registry_root = path.parent / "model_registry"
    selected_artifacts: list[dict[str, Any]] = []

    for seg in improved.get("segments", []):
        sel = seg.get("selected")
        if not sel:
            continue
        src = Path(sel.get("path", ""))
        if not src.exists():
            continue
        snap = _copy_to_registry(src, registry_root, seg["config_id"], run_id)
        selected_artifacts.append(
            {
                "config_id": seg["config_id"],
                "label": seg["label"],
                "snapshot": snap,
                "selected": sel,
            }
        )

    run_entry = {
        "run_id": run_id,
        "route_id": route_id,
        "created_at": _utc_now(),
        "settings": asdict(settings),
        "baseline": baseline,
        "improved": improved,
        "selected_artifacts": selected_artifacts,
    }

    data.setdefault("runs", []).append(run_entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def _pick_weak_segments(segments: list[SegmentSummary], top_k: int) -> list[SegmentSummary]:
    ranked = sorted(
        [s for s in segments if s.selected is not None],
        key=lambda s: s.weak_score,
        reverse=True,
    )
    return ranked[:max(0, top_k)]


def _print_summary(title: str, summary: dict[str, Any]) -> None:
    print(f"\n{title}")
    print(f"  Completed: {summary['completed_count']}/{summary['total_count']}")
    print(f"  Total: {summary['total_frames']}f ({summary['total_seconds']:.2f}s)")


def run_pipeline(settings: HybridSettings, *, improve: bool) -> dict[str, Any]:
    run_id = _run_id()
    print(f"Selection context: {settings.selection_context}")

    segments = analyze_route(settings)
    baseline = summarize_route(segments)

    _print_summary("Baseline", baseline)

    weak_rows = _pick_weak_segments(segments, settings.weak_top_k)
    weak_labels = [f"{s.label} ({s.config_id})" for s in weak_rows]
    if weak_labels:
        print("\nWeak segments:")
        for lbl in weak_labels:
            print(f"  - {lbl}")

    if improve:
        for seg in weak_rows:
            print(f"\n=== Improving {seg.label} [{seg.config_id}] ===")
            improve_segment(seg, settings, run_id)

    improved = summarize_route(segments)
    _print_summary("Improved", improved)

    chain_validation = None
    if not settings.segments.strip() and len(segments) > 1:
        chain_validation = validate_chain_from_segments(settings.route, segments)
        print("\nChain Validation")
        print(
            f"  Completed: {chain_validation['completed_count']}/{chain_validation['total_count']} "
            f"segments"
        )
        print(
            f"  Total: {chain_validation['total_frames']}f "
            f"({chain_validation['total_seconds']:.2f}s)"
        )

    report = {
        "generated_at": _utc_now(),
        "run_id": run_id,
        "route_id": settings.route,
        "improve_mode": improve,
        "settings": asdict(settings),
        "baseline": baseline,
        "improved": improved,
        "chain_validation": chain_validation,
        "weak_segments": [
            {
                "label": s.label,
                "config_id": s.config_id,
                "weak_score": s.weak_score,
                "baseline_selected": asdict(s.selected) if s.selected else None,
            }
            for s in weak_rows
        ],
    }

    return report


def _save_report(report: dict[str, Any], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))


def _build_settings(args: argparse.Namespace) -> HybridSettings:
    return HybridSettings(
        route=args.route,
        selection_context=args.selection_context,
        max_candidates=args.max_candidates,
        eval_runs=args.eval_runs,
        force_eval=args.force_eval,
        weak_top_k=args.weak_top_k,
        ga_generations=getattr(args, "ga_generations", 0),
        ga_population=getattr(args, "ga_population", 20),
        hill_iterations=getattr(args, "hill_iterations", 600),
        hill_raw_iterations=getattr(args, "hill_raw_iterations", 600),
        neuro_generations=getattr(args, "neuro_generations", 0),
        neuro_population=getattr(args, "neuro_population", 24),
        neuro_hidden=getattr(args, "neuro_hidden", 20),
        neuro_max_frames=getattr(args, "neuro_max_frames", 4500),
        ppo_candidates_dir=getattr(args, "ppo_candidates_dir", "") or "",
        ppo_command=getattr(args, "ppo_command", "") or "",
        ppo_output_name=getattr(args, "ppo_output_name", "ppo_candidate.json"),
        segments=args.segments or "",
    )


def cmd_analyze(args: argparse.Namespace) -> int:
    settings = _build_settings(args)
    report = run_pipeline(settings, improve=False)

    report_path = Path(args.report)
    registry_path = Path(args.registry)
    _save_report(report, report_path)
    _append_registry_run(
        registry_path,
        route_id=settings.route,
        run_id=report["run_id"],
        settings=settings,
        baseline=report["baseline"],
        improved=report["improved"],
    )

    print(f"\nSaved report: {report_path}")
    print(f"Updated registry: {registry_path}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    settings = _build_settings(args)
    report = run_pipeline(settings, improve=True)

    report_path = Path(args.report)
    registry_path = Path(args.registry)
    _save_report(report, report_path)
    _append_registry_run(
        registry_path,
        route_id=settings.route,
        run_id=report["run_id"],
        settings=settings,
        baseline=report["baseline"],
        improved=report["improved"],
    )

    print(f"\nSaved report: {report_path}")
    print(f"Updated registry: {registry_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SMB hybrid optimizer (registry + sub-agents)")
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--route", "-r", default="smb_any_percent", help="Route ID or alias")
        sp.add_argument("--segments", default="", help="Optional comma list of config IDs to include")
        sp.add_argument(
            "--selection-context",
            choices=("standalone", "chained"),
            default="standalone",
            help="Candidate scoring/start-state context (default standalone)",
        )
        sp.add_argument("--max-candidates", type=int, default=32, help="Max candidates per segment (default 32)")
        sp.add_argument("--eval-runs", type=int, default=1, help="Live eval repeats when evaluating recordings")
        sp.add_argument("--force-eval", action="store_true", help="Force live eval even if static metrics exist")
        sp.add_argument("--weak-top-k", type=int, default=3, help="Weak segments to target in run mode")
        sp.add_argument("--registry", default=str(DEFAULT_REGISTRY), help="Model registry JSON path")
        sp.add_argument("--report", default=str(DEFAULT_REPORT), help="Report JSON path")

    pa = sub.add_parser("analyze", help="Analyze existing artifacts, update registry, no training")
    add_common(pa)
    pa.set_defaults(func=cmd_analyze)

    pr = sub.add_parser("run", help="Analyze + improve weakest segments with hybrid sub-agents")
    add_common(pr)
    pr.add_argument("--ga-generations", type=int, default=0, help="GA-raw generations (default 0=off)")
    pr.add_argument("--ga-population", type=int, default=20, help="GA-raw population (default 20)")
    pr.add_argument("--hill-iterations", type=int, default=600, help="Hillclimb iterations for index actions")
    pr.add_argument("--hill-raw-iterations", type=int, default=600, help="Hillclimb iterations for raw buttons")
    pr.add_argument("--neuro-generations", type=int, default=0, help="NEAT generations (default 0=off)")
    pr.add_argument("--neuro-population", type=int, default=24, help="NEAT population")
    pr.add_argument("--neuro-hidden", type=int, default=20, help="NEAT hidden layer size")
    pr.add_argument("--neuro-max-frames", type=int, default=4500, help="Max frames per NEAT eval")
    pr.add_argument("--ppo-candidates-dir", default="", help="Directory containing external PPO candidate JSONs")
    pr.add_argument(
        "--ppo-command",
        default="",
        help=(
            "Optional command template to generate PPO candidate JSON. "
            "Placeholders: {level} {state} {run_dir} {output_dir}"
        ),
    )
    pr.add_argument("--ppo-output-name", default="ppo_candidate.json", help="Expected PPO output JSON name")
    pr.set_defaults(func=cmd_run)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
