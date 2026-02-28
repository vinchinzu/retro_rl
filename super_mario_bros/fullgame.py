#!/usr/bin/env python3
"""Unified full-game pipeline for Super Mario Bros any% route.

This module stitches level progression (including 8-3 and 8-4 segments)
into a coherent full-game workflow with two execution contexts:

1) standalone: evaluate each segment from its own start state
2) chained: replay the route end-to-end on one emulator session

It also performs context-aware recording selection to reduce failures and
runtime overhead caused by using the wrong artifact in the wrong context.

Usage examples:
    PYTHONPATH=. python -m super_mario_bros.fullgame train
    PYTHONPATH=. python -m super_mario_bros.fullgame eval --mode both
    PYTHONPATH=. python -m super_mario_bros.fullgame run
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import platformer_common.levels.smb  # noqa: F401 - register SMB levels + routes
from platformer_common.evaluator import Evaluator
from platformer_common.level_config import get_level_config
from platformer_common.route import (
    RouteConfig,
    RouteSegment,
    chain_live,
    evaluate_route,
    get_route,
    load_recording_data,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "optimizer" / "fullgame_manifest.json"
DEFAULT_REPORT = ROOT / "optimizer" / "fullgame_eval_report.json"

# Segment start states produced by chain-live --save-states. These are used
# for context-aware "live" scoring to match real chained execution conditions.
CHAINED_STATE_CANDIDATES = {
    "smb_1_1": ["Chained_1-1", "Chained_smb_1_1"],
    "smb_1_2": ["Chained_1-2_toW4"],
    "smb_4_1": ["Chained_4-1"],
    "smb_4_2": ["Chained_4-2_toW8"],
    "smb_8_1": ["Chained_8-1"],
    "smb_8_2": ["Chained_8-2", "Chained_8-2_mid", "Chained_8-2_late"],
    "smb_8_3": ["Chained_8-3"],
    "smb_8_4_1": ["Chained_8-4_seg1"],
    "smb_8_4_2": ["Chained_8-4_seg2"],
    "smb_8_4_3": ["Chained_8-4_seg3"],
    "smb_8_4_4": ["Chained_8-4_seg4"],
    "smb_8_4_5": ["Chained_8-4_seg5"],
}

PRIORITY_NAMES = [
    "recording_000.json",
    "recording_001.json",
    "recording_002.json",
    "hillclimb_best_final.json",
    "ga_best_final.json",
    "ga_raw_best_final.json",
    "ga_raw_best.json",
    "hillclimb_raw_best.json",
    "combined_82_complete.json",
    "recording_chained_complete.json",
    "chained/hillclimb_best_final.json",
]


@dataclass
class CandidateScore:
    recording: str
    completed: bool
    died: bool
    frames: int
    fitness: float
    max_progress: float
    state: str | None


def _state_exists(config_id: str, state_name: str) -> bool:
    cfg = get_level_config(config_id)
    state_path = cfg.game_dir / "custom_integrations" / cfg.game_name / f"{state_name}.state"
    return state_path.exists()


def _resolve_state(config_id: str, context: str) -> str | None:
    if context != "chained":
        return None
    candidates = CHAINED_STATE_CANDIDATES.get(config_id, [])
    for state_name in candidates:
        if _state_exists(config_id, state_name):
            return state_name
    return None


def _candidate_priority_key(path: Path) -> tuple[int, str]:
    rel = str(path)
    for idx, name in enumerate(PRIORITY_NAMES):
        if rel.endswith(name):
            return idx, rel
    if path.name.startswith("ga_gen") and path.name.endswith("_best.json"):
        return len(PRIORITY_NAMES), rel
    return len(PRIORITY_NAMES) + 1, rel


def discover_candidates(config_id: str, max_candidates: int = 20) -> list[Path]:
    cfg = get_level_config(config_id)
    run_dir = cfg.runs_dir
    candidates: list[Path] = []

    for name in PRIORITY_NAMES:
        p = run_dir / name
        if p.exists():
            candidates.append(p)

    candidates.extend(sorted(run_dir.glob("ga_gen*_best.json")))
    candidates.extend(sorted(run_dir.glob("recording_*.json")))

    # Keep JSON action files only; drop companion raw files.
    filtered: list[Path] = []
    seen: set[Path] = set()
    for p in candidates:
        if "_raw" in p.stem:
            continue
        if p.suffix != ".json":
            continue
        if p in seen:
            continue
        seen.add(p)
        filtered.append(p)

    filtered.sort(key=_candidate_priority_key)
    return filtered[:max_candidates]


def _rank_key(score: CandidateScore) -> tuple[int, int, float]:
    # Completed runs are ranked by speed (fewer frames).
    # Incomplete runs are ranked by progress quality (higher fitness/progress).
    if score.completed:
        return (0, score.frames, -score.fitness)
    return (1, int(-score.fitness), -score.max_progress)


def _evaluate_segment_candidates(
    config_id: str,
    context: str,
    max_candidates: int = 20,
) -> dict[str, Any]:
    cfg = get_level_config(config_id)
    state_override = _resolve_state(config_id, context)
    candidates = discover_candidates(config_id, max_candidates=max_candidates)

    result: dict[str, Any] = {
        "config_id": config_id,
        "context": context,
        "state": state_override,
        "scores": [],
        "best": None,
        "notes": [],
    }

    if not candidates:
        result["notes"].append("no_candidates")
        return result

    evaluator = Evaluator(cfg, start_state=state_override)
    scores: list[CandidateScore] = []

    try:
        for cand in candidates:
            rel = str(cand.relative_to(cfg.runs_dir))
            try:
                actions, _is_raw = load_recording_data(cand)
                er = evaluator.evaluate(actions, early_terminate=False)
                scores.append(
                    CandidateScore(
                        recording=rel,
                        completed=bool(er.completed),
                        died=bool(er.died),
                        frames=int(er.total_frames),
                        fitness=float(er.fitness),
                        max_progress=float(er.max_progress),
                        state=state_override,
                    )
                )
            except Exception as exc:  # pragma: no cover - runtime dependent
                result["notes"].append(f"eval_error:{rel}:{exc}")
    finally:
        evaluator.close()

    if not scores:
        result["notes"].append("no_valid_scores")
        return result

    scores.sort(key=_rank_key)
    best = scores[0]
    result["scores"] = [asdict(s) for s in scores]
    result["best"] = asdict(best)
    return result


def _evaluate_segment_candidates_worker(payload: dict[str, Any]) -> dict[str, Any]:
    # Re-import for worker processes.
    import platformer_common.levels.smb  # noqa: F401

    return _evaluate_segment_candidates(
        config_id=payload["config_id"],
        context=payload["context"],
        max_candidates=int(payload.get("max_candidates", 20)),
    )


def score_route_context(
    route: RouteConfig,
    context: str,
    *,
    workers: int = 1,
    max_candidates: int = 20,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    payloads = [
        {"config_id": seg.config_id, "context": context, "max_candidates": max_candidates}
        for seg in route.segments
    ]

    scored: list[dict[str, Any]] = []
    parallel_fallback = ""
    if workers > 1:
        try:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futs = [pool.submit(_evaluate_segment_candidates_worker, p) for p in payloads]
                for fut in as_completed(futs):
                    scored.append(fut.result())
            # Preserve route ordering.
            by_id = {s["config_id"]: s for s in scored}
            scored = [by_id[seg.config_id] for seg in route.segments if seg.config_id in by_id]
        except (PermissionError, OSError) as exc:
            parallel_fallback = f"parallel_disabled:{exc}"
            scored = []
            for p in payloads:
                scored.append(_evaluate_segment_candidates_worker(p))
    else:
        for p in payloads:
            scored.append(_evaluate_segment_candidates_worker(p))

    elapsed = time.perf_counter() - t0
    selected: dict[str, Any] = {}
    for item in scored:
        best = item.get("best")
        if best:
            selected[item["config_id"]] = best

    return {
        "context": context,
        "elapsed_sec": elapsed,
        "parallel_fallback": parallel_fallback,
        "segments": scored,
        "selected": selected,
    }


def _route_with_selected_recordings(route: RouteConfig, selected: dict[str, Any]) -> RouteConfig:
    new_segments: list[RouteSegment] = []
    for seg in route.segments:
        choice = selected.get(seg.config_id, {})
        recording = str(choice.get("recording", "")) if choice else ""
        new_segments.append(
            RouteSegment(
                config_id=seg.config_id,
                label=seg.label,
                recording=recording,
                neuro_checkpoint=seg.neuro_checkpoint,
            )
        )
    return RouteConfig(
        route_id=f"{route.route_id}_selected",
        display_name=f"{route.display_name} (selected)",
        segments=new_segments,
    )


def _summarize_route_result(rr: Any) -> dict[str, Any]:
    segments: list[dict[str, Any]] = []
    for s in rr.segments:
        er = s.eval_result
        segments.append(
            {
                "label": s.segment.label,
                "config_id": s.segment.config_id,
                "recording": str(s.recording_path) if s.recording_path else "",
                "completed": bool(er.completed) if er else False,
                "died": bool(er.died) if er else False,
                "frames": int(er.total_frames) if er else 0,
                "fitness": float(er.fitness) if er else 0.0,
                "error": s.error or "",
            }
        )
    return {
        "completed_count": int(rr.completed_count),
        "total_count": int(rr.total_count),
        "all_completed": bool(rr.all_completed),
        "total_frames": int(rr.total_frames),
        "segments": segments,
    }


def _summarize_chain_live_result(cr: Any, route_len: int) -> dict[str, Any]:
    segments: list[dict[str, Any]] = []
    seg_frame_sum = 0
    for s in cr.segments:
        seg_frame_sum += int(s.frames)
        segments.append(
            {
                "label": s.segment.label,
                "config_id": s.segment.config_id,
                "recording": str(s.recording_path) if s.recording_path else "",
                "status": s.status,
                "frames": int(s.frames),
                "error": s.error or "",
            }
        )
    return {
        "completed_count": int(cr.completed_count),
        "total_count": int(route_len),
        "all_completed": bool(cr.all_completed),
        "total_frames": int(cr.total_frames),
        "segment_frames_sum": int(seg_frame_sum),
        "transition_overhead_frames": int(cr.total_frames - seg_frame_sum),
        "segments": segments,
    }


def _top_policy_losses(selected: dict[str, Any], top_k: int = 5) -> list[dict[str, Any]]:
    rows = []
    for cfg, rec in selected.items():
        if rec.get("completed"):
            rows.append(
                {
                    "config_id": cfg,
                    "recording": rec.get("recording", ""),
                    "frames": int(rec.get("frames", 0)),
                    "fitness": float(rec.get("fitness", 0.0)),
                }
            )
    rows.sort(key=lambda x: x["frames"], reverse=True)
    return rows[:top_k]


def _implementation_mismatches(
    base_summary: dict[str, Any],
    selected_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    base_by_id = {s["config_id"]: s for s in base_summary.get("segments", [])}
    sel_by_id = {s["config_id"]: s for s in selected_summary.get("segments", [])}
    rows = []
    for cfg, base_seg in base_by_id.items():
        sel_seg = sel_by_id.get(cfg)
        if not sel_seg:
            continue
        if (not base_seg.get("completed")) and sel_seg.get("completed"):
            rows.append(
                {
                    "config_id": cfg,
                    "base_recording": base_seg.get("recording", ""),
                    "selected_recording": sel_seg.get("recording", ""),
                    "selected_frames": sel_seg.get("frames", 0),
                }
            )
    return rows


def run_training_profile(
    *,
    route_id: str,
    workers: int,
    max_candidates: int,
    refresh_chained_states: bool,
) -> dict[str, Any]:
    route = get_route(route_id)
    out: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "route_id": route.route_id,
        "route_display_name": route.display_name,
    }

    # Baselines with framework default artifact discovery.
    base_eval = evaluate_route(route, verbose=False)
    out["baseline_standalone"] = _summarize_route_result(base_eval)

    # Preserve chained states so chained-context scoring uses real in-route starts.
    # save_states=True is intentional here.
    if refresh_chained_states:
        chain_live(route, save_states=True, verbose=False)

    base_live = chain_live(route, save_states=True, verbose=False)
    out["baseline_chained"] = _summarize_chain_live_result(base_live, len(route.segments))

    # Context-aware scoring and selection.
    standalone_scores = score_route_context(
        route,
        "standalone",
        workers=workers,
        max_candidates=max_candidates,
    )
    chained_scores = score_route_context(
        route,
        "chained",
        workers=workers,
        max_candidates=max_candidates,
    )

    out["scores"] = {
        "standalone": standalone_scores,
        "chained": chained_scores,
    }

    # Build selected routes and evaluate.
    sel_route_standalone = _route_with_selected_recordings(route, standalone_scores["selected"])
    sel_eval = evaluate_route(sel_route_standalone, verbose=False)
    out["selected_standalone"] = _summarize_route_result(sel_eval)

    sel_route_chained = _route_with_selected_recordings(route, chained_scores["selected"])
    sel_live = chain_live(sel_route_chained, save_states=False, verbose=False)
    out["selected_chained"] = _summarize_chain_live_result(sel_live, len(route.segments))

    out["top_policy_losses_standalone"] = _top_policy_losses(standalone_scores["selected"])
    out["top_policy_losses_chained"] = _top_policy_losses(chained_scores["selected"])
    out["implementation_overhead_standalone"] = _implementation_mismatches(
        out["baseline_standalone"],
        out["selected_standalone"],
    )
    out["implementation_overhead_chained"] = _implementation_mismatches(
        out["baseline_chained"],
        out["selected_chained"],
    )
    return out


def _print_training_summary(training: dict[str, Any]) -> None:
    b_std = training["baseline_standalone"]
    s_std = training["selected_standalone"]
    b_live = training["baseline_chained"]
    s_live = training["selected_chained"]

    print("Full-game training profile complete\n")
    print("Standalone route:")
    print(f"  baseline: {b_std['completed_count']}/{b_std['total_count']} completed, {b_std['total_frames']}f")
    print(f"  selected: {s_std['completed_count']}/{s_std['total_count']} completed, {s_std['total_frames']}f")
    print("")
    print("Chained route:")
    print(f"  baseline: {b_live['completed_count']}/{b_live['total_count']} completed, {b_live['total_frames']}f")
    print(f"  selected: {s_live['completed_count']}/{s_live['total_count']} completed, {s_live['total_frames']}f")
    print(f"  selected transition overhead: {s_live['transition_overhead_frames']}f")
    print("")

    impl_std = training.get("implementation_overhead_standalone", [])
    impl_live = training.get("implementation_overhead_chained", [])
    if impl_std:
        print("Standalone implementation mismatches fixed by selection:")
        for r in impl_std:
            print(f"  - {r['config_id']}: {Path(r['base_recording']).name} -> {r['selected_recording']}")
        print("")
    if impl_live:
        print("Chained implementation mismatches fixed by selection:")
        for r in impl_live:
            print(f"  - {r['config_id']}: {Path(r['base_recording']).name} -> {r['selected_recording']}")
        print("")

    print("Top chained policy time-loss segments (selected artifacts):")
    for r in training.get("top_policy_losses_chained", [])[:5]:
        print(f"  - {r['config_id']}: {r['frames']}f via {r['recording']}")


def _build_route_from_manifest(route: RouteConfig, manifest: dict[str, Any], context: str) -> RouteConfig:
    selected = manifest.get("scores", {}).get(context, {}).get("selected", {})
    return _route_with_selected_recordings(route, selected)


def run_eval_from_manifest(
    *,
    route_id: str,
    manifest: dict[str, Any],
    mode: str,
    save_states: bool,
) -> dict[str, Any]:
    route = get_route(route_id)
    out: dict[str, Any] = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "route_id": route.route_id,
        "mode": mode,
    }

    if mode in ("standalone", "both"):
        selected_std = _build_route_from_manifest(route, manifest, "standalone")
        rr = evaluate_route(selected_std, verbose=True)
        out["standalone"] = _summarize_route_result(rr)

    if mode in ("chained", "both"):
        selected_live = _build_route_from_manifest(route, manifest, "chained")
        cr = chain_live(selected_live, save_states=save_states, verbose=True)
        out["chained"] = _summarize_chain_live_result(cr, len(route.segments))

    return out


def cmd_train(args: argparse.Namespace) -> int:
    training = run_training_profile(
        route_id=args.route,
        workers=args.workers,
        max_candidates=args.max_candidates,
        refresh_chained_states=args.refresh_chained_states,
    )

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(training, indent=2))
    print(f"Saved manifest: {manifest_path}")
    _print_training_summary(training)
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        print("Run 'train' first.")
        return 1

    manifest = json.loads(manifest_path.read_text())
    report = run_eval_from_manifest(
        route_id=args.route,
        manifest=manifest,
        mode=args.mode,
        save_states=args.save_states,
    )

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nSaved eval report: {report_path}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    t_args = argparse.Namespace(
        route=args.route,
        workers=args.workers,
        max_candidates=args.max_candidates,
        refresh_chained_states=args.refresh_chained_states,
        manifest=args.manifest,
    )
    rc = cmd_train(t_args)
    if rc != 0:
        return rc

    e_args = argparse.Namespace(
        route=args.route,
        manifest=args.manifest,
        report=args.report,
        mode="both",
        save_states=args.save_states,
    )
    return cmd_eval(e_args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SMB full-game pipeline (train/eval/run)")
    sub = p.add_subparsers(dest="command", required=True)

    for name in ("train", "profile"):
        pt = sub.add_parser(name, help="Score candidates per segment and write unified manifest")
        pt.add_argument("--route", "-r", default="smb_any_percent", help="Route ID or alias")
        pt.add_argument("--workers", type=int, default=1, help="Parallel segment scorers (default: 1)")
        pt.add_argument("--max-candidates", type=int, default=20, help="Max candidate recordings per segment")
        pt.add_argument(
            "--refresh-chained-states",
            action="store_true",
            help="Run a baseline chain-live with --save-states before scoring chained context",
        )
        pt.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Output manifest JSON path")
        pt.set_defaults(func=cmd_train)

    pe = sub.add_parser("eval", help="Evaluate selected manifest route (standalone/chained/both)")
    pe.add_argument("--route", "-r", default="smb_any_percent", help="Route ID or alias")
    pe.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Manifest JSON path")
    pe.add_argument("--report", default=str(DEFAULT_REPORT), help="Eval report JSON path")
    pe.add_argument(
        "--mode",
        choices=("standalone", "chained", "both"),
        default="both",
        help="Evaluation mode",
    )
    pe.add_argument("--save-states", action="store_true", help="Save chained states during chain-live eval")
    pe.set_defaults(func=cmd_eval)

    pr = sub.add_parser("run", help="Train (selection) then eval both contexts")
    pr.add_argument("--route", "-r", default="smb_any_percent", help="Route ID or alias")
    pr.add_argument("--workers", type=int, default=1, help="Parallel segment scorers (default: 1)")
    pr.add_argument("--max-candidates", type=int, default=20, help="Max candidate recordings per segment")
    pr.add_argument(
        "--refresh-chained-states",
        action="store_true",
        help="Run a baseline chain-live with --save-states before scoring chained context",
    )
    pr.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Output manifest JSON path")
    pr.add_argument("--report", default=str(DEFAULT_REPORT), help="Eval report JSON path")
    pr.add_argument("--save-states", action="store_true", help="Save chained states during chain-live eval")
    pr.set_defaults(func=cmd_run)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
