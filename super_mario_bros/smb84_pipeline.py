#!/usr/bin/env python3
"""Unified training/eval pipeline for SMB 8-4 segmented optimization.

Why this exists:
- Generic chain-live transition logic is tuned for level loads, not 8-4 area
  transitions (0x65/0xE5/0x02), which can desync segment starts.
- 8-4 artifacts are split across multiple scripts; this file stitches state
  validation, candidate scoring, training, and chain eval into one workflow.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# Ensure repo root is importable when invoked as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import platformer_common.levels.smb  # noqa: F401 - register SMB levels/routes
from platformer_common.actions import (
    DEFAULT_PLATFORMER_ACTIONS,
    action_index_to_buttons,
    buttons_to_action_index,
)
from platformer_common.evaluator import EvalResult, Evaluator
from platformer_common.hillclimb import hillclimb
from platformer_common.hillclimb_raw import hillclimb_raw
from platformer_common.level_config import get_level_config
from platformer_common.route import load_recording_data
from retro_harness.env import make_env

ROOT = Path(__file__).resolve().parent
GAME = "SuperMarioBros-Nes-v0"
STATES_DIR = ROOT / "custom_integrations" / GAME
DEFAULT_MANIFEST = ROOT / "optimizer" / "smb84_manifest.json"
DEFAULT_DIAG_REPORT = ROOT / "optimizer" / "smb84_diagnose.json"
DEFAULT_EVAL_REPORT = ROOT / "optimizer" / "smb84_eval_report.json"

# NES SMB RAM addresses used for start/completion diagnostics.
ADDR_AREA = 0x0750
ADDR_LIVES = 0x075A
ADDR_WORLD = 0x075F
ADDR_LEVEL = 0x0760
ADDR_GAME_MODE = 0x0770
ADDR_X_PAGE = 0x006D
ADDR_X_OFF = 0x0086


@dataclass(frozen=True)
class SegmentSpec:
    seg: int
    config_id: str
    state_name: str
    start_area: int
    completion_area: int | None = None
    completion_game_mode: int | None = None
    raw_preferred: bool = False


SEGMENTS: list[SegmentSpec] = [
    SegmentSpec(1, "smb_8_4_1", "Level8_4", start_area=0x65, completion_area=0xE5),
    SegmentSpec(2, "smb_8_4_2", "Level8_4_seg2", start_area=0xE5, completion_area=0x65),
    # seg3 is the most timing-sensitive; prefer raw-button optimization by default.
    SegmentSpec(3, "smb_8_4_3", "Level8_4_seg3", start_area=0x65, completion_area=0x02, raw_preferred=True),
    SegmentSpec(4, "smb_8_4_4", "Level8_4_seg4", start_area=0x02, completion_area=0x65),
    SegmentSpec(5, "smb_8_4_5", "Level8_4_seg5", start_area=0x65, completion_game_mode=2),
]
SEGMENT_BY_NUM = {s.seg: s for s in SEGMENTS}

CANDIDATE_PRIORITY = [
    "hillclimb_raw_best_final.json",
    "hillclimb_raw_best.json",
    "ga_raw_best_final.json",
    "hillclimb_best_final.json",
    "ga_best_final.json",
    "recording_000.json",
    "recording_001.json",
]


@dataclass
class CandidateStats:
    recording: str
    is_raw: bool
    runs: int
    completed_runs: int
    died_runs: int
    completion_rate: float
    frames_min: int
    frames_max: int
    frames_mean: float
    fitness_mean: float
    progress_max: float


@dataclass
class ChainSegmentResult:
    segment: int
    config_id: str
    recording: str
    status: str
    frames: int
    start_area: int
    end_area: int
    lives_start: int
    lives_end: int
    end_game_mode: int
    note: str = ""


def _parse_segments(raw: str | None, *, default_all: bool = True) -> list[SegmentSpec]:
    if raw is None:
        return list(SEGMENTS) if default_all else []
    txt = raw.strip().lower()
    if txt in {"all", "*"}:
        return list(SEGMENTS)

    picked: list[SegmentSpec] = []
    seen: set[int] = set()
    for chunk in txt.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            left, right = chunk.split("-", 1)
            a = int(left)
            b = int(right)
            lo, hi = sorted((a, b))
            for seg_num in range(lo, hi + 1):
                if seg_num in SEGMENT_BY_NUM and seg_num not in seen:
                    picked.append(SEGMENT_BY_NUM[seg_num])
                    seen.add(seg_num)
        else:
            seg_num = int(chunk)
            if seg_num in SEGMENT_BY_NUM and seg_num not in seen:
                picked.append(SEGMENT_BY_NUM[seg_num])
                seen.add(seg_num)
    if not picked:
        raise ValueError(f"No valid segments in '{raw}'. Valid: 1-5, comma-separated, or 'all'.")
    return picked


def _ram_snapshot(env) -> dict[str, int]:
    ram = env.get_ram()
    x = int(ram[ADDR_X_PAGE]) * 256 + int(ram[ADDR_X_OFF])
    return {
        "area": int(ram[ADDR_AREA]),
        "lives": int(ram[ADDR_LIVES]),
        "world": int(ram[ADDR_WORLD]),
        "level": int(ram[ADDR_LEVEL]),
        "game_mode": int(ram[ADDR_GAME_MODE]),
        "player_x": x,
    }


def _probe_state(state_name: str) -> dict[str, int]:
    env = make_env(GAME, state_name, ROOT, render_mode="rgb_array")
    try:
        env.reset()
        return _ram_snapshot(env)
    finally:
        env.close()


def _candidate_priority_key(path: Path) -> tuple[int, str]:
    rel = str(path)
    for idx, name in enumerate(CANDIDATE_PRIORITY):
        if rel.endswith(name):
            return idx, rel
    if path.name.startswith("ga_gen") and path.name.endswith("_best.json"):
        return len(CANDIDATE_PRIORITY), rel
    if path.name.startswith("recording_") and path.suffix == ".json":
        return len(CANDIDATE_PRIORITY) + 1, rel
    return len(CANDIDATE_PRIORITY) + 2, rel


def _discover_candidates(spec: SegmentSpec, *, max_candidates: int) -> list[Path]:
    cfg = get_level_config(spec.config_id)
    run_dir = cfg.runs_dir
    candidates: list[Path] = []

    for name in CANDIDATE_PRIORITY:
        p = run_dir / name
        if p.exists():
            candidates.append(p)

    candidates.extend(sorted(run_dir.glob("ga_gen*_best.json")))
    candidates.extend(sorted(run_dir.glob("recording_*.json")))

    filtered: list[Path] = []
    seen: set[Path] = set()
    for p in candidates:
        if p.suffix != ".json":
            continue
        # Skip companion raw files (recording_000_raw.json), but keep
        # first-class artifacts like hillclimb_raw_best_final.json.
        if p.stem.endswith("_raw"):
            continue
        if p in seen:
            continue
        seen.add(p)
        filtered.append(p)

    filtered.sort(key=_candidate_priority_key)
    return filtered[:max(max_candidates, 1)]


def _summarize_eval(results: list[EvalResult], *, is_raw: bool, recording: str) -> CandidateStats:
    completed = sum(1 for r in results if r.completed)
    died = sum(1 for r in results if r.died)
    frames = [int(r.total_frames) for r in results]
    fitness = [float(r.fitness) for r in results]
    progress = [float(r.max_progress) for r in results]
    return CandidateStats(
        recording=recording,
        is_raw=is_raw,
        runs=len(results),
        completed_runs=completed,
        died_runs=died,
        completion_rate=completed / max(len(results), 1),
        frames_min=min(frames),
        frames_max=max(frames),
        frames_mean=statistics.mean(frames),
        fitness_mean=statistics.mean(fitness),
        progress_max=max(progress),
    )


def _rank_candidate(c: CandidateStats) -> tuple[int, float, float, float]:
    if c.completed_runs > 0:
        # Completed candidates: prefer higher completion rate then faster.
        return (0, -c.completion_rate, c.frames_mean, -c.fitness_mean)
    # Incomplete: prefer fitness/progress as recovery signal.
    return (1, -c.fitness_mean, -c.progress_max, c.frames_mean)


def _evaluate_candidate(
    spec: SegmentSpec,
    path: Path,
    *,
    runs: int,
    state_override: str | None = None,
) -> CandidateStats:
    cfg = get_level_config(spec.config_id)
    actions, is_raw = load_recording_data(path)
    evaluator = Evaluator(cfg, start_state=state_override)
    try:
        out: list[EvalResult] = []
        for _ in range(max(runs, 1)):
            out.append(evaluator.evaluate(actions, early_terminate=False))
        return _summarize_eval(out, is_raw=is_raw, recording=path.name)
    finally:
        evaluator.close()


def _score_segment_candidates(
    spec: SegmentSpec,
    *,
    runs: int,
    max_candidates: int,
    state_override: str | None = None,
) -> dict[str, Any]:
    cfg = get_level_config(spec.config_id)
    cands = _discover_candidates(spec, max_candidates=max_candidates)
    rows: list[CandidateStats] = []
    notes: list[str] = []

    for p in cands:
        try:
            rows.append(_evaluate_candidate(spec, p, runs=runs, state_override=state_override))
        except Exception as exc:  # pragma: no cover - runtime dependent
            notes.append(f"eval_error:{p.name}:{exc}")

    best = sorted(rows, key=_rank_candidate)[0] if rows else None
    return {
        "segment": spec.seg,
        "config_id": spec.config_id,
        "state": state_override or "",
        "run_dir": str(cfg.runs_dir),
        "best": asdict(best) if best else None,
        "scores": [asdict(r) for r in sorted(rows, key=_rank_candidate)],
        "notes": notes,
    }


def _build_manifest(
    specs: list[SegmentSpec],
    *,
    eval_runs: int,
    max_candidates: int,
) -> dict[str, Any]:
    rows = [
        _score_segment_candidates(s, runs=eval_runs, max_candidates=max_candidates)
        for s in specs
    ]

    selected: dict[str, dict[str, Any]] = {}
    for row in rows:
        best = row.get("best")
        if best:
            selected[row["config_id"]] = {
                "recording": best["recording"],
                "is_raw": best["is_raw"],
                "completion_rate": best["completion_rate"],
                "frames_mean": best["frames_mean"],
                "fitness_mean": best["fitness_mean"],
                "run_dir": row["run_dir"],
            }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "level": "smb_8_4",
        "segments": rows,
        "selected": selected,
    }


def _load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _resolve_selected_path(spec: SegmentSpec, manifest: dict[str, Any]) -> Path:
    selected = manifest.get("selected", {}).get(spec.config_id)
    if not selected:
        raise FileNotFoundError(f"No selected recording for {spec.config_id} in manifest.")
    run_dir = Path(selected.get("run_dir", get_level_config(spec.config_id).runs_dir))
    rec_name = str(selected.get("recording", ""))
    if not rec_name:
        raise FileNotFoundError(f"Manifest selected entry for {spec.config_id} has empty recording.")
    rec_path = Path(rec_name)
    if not rec_path.is_absolute():
        rec_path = run_dir / rec_name
    if not rec_path.exists():
        raise FileNotFoundError(f"Selected recording missing: {rec_path}")
    return rec_path


def _summarize_results(results: list[EvalResult]) -> dict[str, Any]:
    frames = [r.total_frames for r in results]
    progress = [r.max_progress for r in results]
    fitness = [r.fitness for r in results]
    completed = sum(1 for r in results if r.completed)
    died = sum(1 for r in results if r.died)
    return {
        "runs": len(results),
        "completed": completed,
        "died": died,
        "completion_rate": completed / max(len(results), 1),
        "frames_min": min(frames),
        "frames_max": max(frames),
        "frames_mean": statistics.mean(frames),
        "progress_max": max(progress),
        "fitness_mean": statistics.mean(fitness),
    }


def _save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def _indices_to_raw(actions: list[int], action_table: list[list[int]]) -> list[list[int]]:
    return [action_index_to_buttons(a, action_table) for a in actions]


def _raw_to_indices(raw_buttons: list[list[int]], action_table: list[list[int]]) -> list[int]:
    return [buttons_to_action_index(frame, action_table=action_table) for frame in raw_buttons]


def _state_integrity(specs: list[SegmentSpec]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        state_path = STATES_DIR / f"{spec.state_name}.state"
        row: dict[str, Any] = {
            "segment": spec.seg,
            "config_id": spec.config_id,
            "state": spec.state_name,
            "state_exists": state_path.exists(),
        }
        if state_path.exists():
            try:
                probe = _probe_state(spec.state_name)
                row.update(
                    {
                        "area": probe["area"],
                        "lives": probe["lives"],
                        "world": probe["world"],
                        "level": probe["level"],
                        "player_x": probe["player_x"],
                        "game_mode": probe["game_mode"],
                        "area_ok": probe["area"] == spec.start_area,
                    }
                )
            except Exception as exc:  # pragma: no cover - runtime dependent
                row["probe_error"] = str(exc)
                row["area_ok"] = False
        else:
            row["area_ok"] = False
        rows.append(row)
    return rows


def _validate_state_integrity_or_raise(specs: list[SegmentSpec]) -> None:
    rows = _state_integrity(specs)
    bad = [
        r for r in rows
        if (not r.get("state_exists")) or (not r.get("area_ok"))
    ]
    if not bad:
        return
    parts = []
    for r in bad:
        parts.append(
            f"seg{r['segment']} state={r['state']} exists={r.get('state_exists')} "
            f"area=0x{int(r.get('area', -1)) & 0xFF:02X} expected=0x{SEGMENT_BY_NUM[r['segment']].start_area:02X}"
        )
    raise RuntimeError("State integrity check failed: " + "; ".join(parts))


def _run_chain_eval(
    specs: list[SegmentSpec],
    manifest: dict[str, Any],
    *,
    pad_frames: int,
    start_wait_frames: int,
    completion_debounce: int,
    start_x_tolerance: int,
) -> dict[str, Any]:
    expected_starts: dict[int, dict[str, int]] = {}
    for spec in specs:
        expected_starts[spec.seg] = _probe_state(spec.state_name)

    env = make_env(GAME, specs[0].state_name, ROOT, render_mode="rgb_array")
    env.reset()

    action_size = env.action_space.shape[0]
    no_input = np.zeros(action_size, dtype=np.int8)

    out_rows: list[ChainSegmentResult] = []
    total_frames = 0

    def _wait_for_start_area(spec: SegmentSpec) -> tuple[bool, int, dict[str, int]]:
        expected_area = spec.start_area
        expected_x = int(expected_starts[spec.seg]["player_x"])
        x_min = max(1, expected_x - max(start_x_tolerance, 0))
        waited = 0
        snap = _ram_snapshot(env)
        while waited <= start_wait_frames:
            ready = (
                snap["area"] == expected_area
                and snap["game_mode"] == 1
                and snap["player_x"] >= x_min
            )
            if ready:
                return True, waited, snap
            env.step(no_input)
            waited += 1
            snap = _ram_snapshot(env)
        return False, waited, snap

    try:
        for spec in specs:
            rec_path = _resolve_selected_path(spec, manifest)
            actions, is_raw = load_recording_data(rec_path)

            if is_raw:
                btn_len = len(actions[0]) if actions else action_size
                actions = list(actions) + [[0] * btn_len] * max(pad_frames, 0)
            else:
                actions = list(actions) + [0] * max(pad_frames, 0)

            start_ok, waited, start_snap = _wait_for_start_area(spec)
            total_frames += waited
            if not start_ok:
                exp_x = int(expected_starts[spec.seg]["player_x"])
                out_rows.append(
                    ChainSegmentResult(
                        segment=spec.seg,
                        config_id=spec.config_id,
                        recording=str(rec_path),
                        status="START_MISMATCH",
                        frames=0,
                        start_area=int(start_snap["area"]),
                        end_area=int(start_snap["area"]),
                        lives_start=int(start_snap["lives"]),
                        lives_end=int(start_snap["lives"]),
                        end_game_mode=int(start_snap["game_mode"]),
                        note=(
                            f"expected_start=0x{spec.start_area:02X} "
                            f"expected_x>={max(1, exp_x - max(start_x_tolerance, 0))}"
                        ),
                    )
                )
                break

            action_table = get_level_config(spec.config_id).action_table or DEFAULT_PLATFORMER_ACTIONS
            lives_start = int(start_snap["lives"])
            seg_frames = 0
            completed = False
            died = False
            note = ""
            saw_start = False
            end_snap = dict(start_snap)

            for action in actions:
                if is_raw:
                    buttons = list(action)
                else:
                    buttons = action_index_to_buttons(action, action_table)
                if len(buttons) < action_size:
                    buttons = buttons + [0] * (action_size - len(buttons))
                elif len(buttons) > action_size:
                    buttons = buttons[:action_size]

                env.step(np.array(buttons, dtype=np.int8))
                seg_frames += 1
                total_frames += 1

                snap = _ram_snapshot(env)
                end_snap = dict(snap)

                if snap["area"] == spec.start_area:
                    saw_start = True

                if spec.completion_game_mode is not None:
                    if snap["game_mode"] == spec.completion_game_mode:
                        completed = True
                        note = f"game_mode={spec.completion_game_mode}"
                        break
                elif spec.completion_area is not None:
                    if saw_start and snap["area"] == spec.completion_area:
                        stable = True
                        for _ in range(max(completion_debounce, 0)):
                            env.step(no_input)
                            seg_frames += 1
                            total_frames += 1
                            snap = _ram_snapshot(env)
                            end_snap = dict(snap)
                            if snap["area"] != spec.completion_area:
                                stable = False
                                break
                        if stable:
                            completed = True
                            note = f"area=0x{spec.completion_area:02X}"
                            break

                if snap["lives"] < lives_start:
                    died = True
                    note = "lives_drop"
                    break

            status = "COMPLETED" if completed else ("DIED" if died else "INCOMPLETE")
            out_rows.append(
                ChainSegmentResult(
                    segment=spec.seg,
                    config_id=spec.config_id,
                    recording=str(rec_path),
                    status=status,
                    frames=seg_frames,
                    start_area=spec.start_area,
                    end_area=int(end_snap["area"]),
                    lives_start=lives_start,
                    lives_end=int(end_snap["lives"]),
                    end_game_mode=int(end_snap["game_mode"]),
                    note=note,
                )
            )

            if status != "COMPLETED":
                break
    finally:
        env.close()

    completed_count = sum(1 for r in out_rows if r.status == "COMPLETED")
    return {
        "segments": [asdict(r) for r in out_rows],
        "completed_count": completed_count,
        "total_count": len(specs),
        "all_completed": completed_count == len(specs),
        "total_frames": total_frames,
    }


def _run_state_handoff_chain(
    specs: list[SegmentSpec],
    manifest: dict[str, Any],
    *,
    pad_frames: int,
    completion_debounce: int,
) -> dict[str, Any]:
    """Evaluate 8-4 as a stitched sequence with canonical per-segment resets.

    This is intentionally robust for regression/debug workflows: each segment
    starts from its validated .state file, avoiding cross-segment emulator drift.
    """
    out_rows: list[ChainSegmentResult] = []
    total_frames = 0

    for spec in specs:
        rec_path = _resolve_selected_path(spec, manifest)
        actions, is_raw = load_recording_data(rec_path)
        env = make_env(GAME, spec.state_name, ROOT, render_mode="rgb_array")
        env.reset()
        action_size = env.action_space.shape[0]
        no_input = np.zeros(action_size, dtype=np.int8)
        action_table = get_level_config(spec.config_id).action_table or DEFAULT_PLATFORMER_ACTIONS

        if is_raw:
            btn_len = len(actions[0]) if actions else action_size
            actions = list(actions) + [[0] * btn_len] * max(pad_frames, 0)
        else:
            actions = list(actions) + [0] * max(pad_frames, 0)

        try:
            start_snap = _ram_snapshot(env)
            if start_snap["area"] != spec.start_area:
                out_rows.append(
                    ChainSegmentResult(
                        segment=spec.seg,
                        config_id=spec.config_id,
                        recording=str(rec_path),
                        status="START_MISMATCH",
                        frames=0,
                        start_area=spec.start_area,
                        end_area=int(start_snap["area"]),
                        lives_start=int(start_snap["lives"]),
                        lives_end=int(start_snap["lives"]),
                        end_game_mode=int(start_snap["game_mode"]),
                        note=f"expected_start=0x{spec.start_area:02X}",
                    )
                )
                break

            lives_start = int(start_snap["lives"])
            end_snap = dict(start_snap)
            seg_frames = 0
            completed = False
            died = False
            note = ""
            saw_start = False

            for action in actions:
                if is_raw:
                    buttons = list(action)
                else:
                    buttons = action_index_to_buttons(action, action_table)
                if len(buttons) < action_size:
                    buttons = buttons + [0] * (action_size - len(buttons))
                elif len(buttons) > action_size:
                    buttons = buttons[:action_size]

                env.step(np.array(buttons, dtype=np.int8))
                seg_frames += 1
                total_frames += 1
                snap = _ram_snapshot(env)
                end_snap = dict(snap)

                if snap["area"] == spec.start_area:
                    saw_start = True

                if spec.completion_game_mode is not None:
                    if snap["game_mode"] == spec.completion_game_mode:
                        completed = True
                        note = f"game_mode={spec.completion_game_mode}"
                        break
                elif spec.completion_area is not None:
                    if saw_start and snap["area"] == spec.completion_area:
                        stable = True
                        for _ in range(max(completion_debounce, 0)):
                            env.step(no_input)
                            seg_frames += 1
                            total_frames += 1
                            snap = _ram_snapshot(env)
                            end_snap = dict(snap)
                            if snap["area"] != spec.completion_area:
                                stable = False
                                break
                        if stable:
                            completed = True
                            note = f"area=0x{spec.completion_area:02X}"
                            break

                if snap["lives"] < lives_start:
                    died = True
                    note = "lives_drop"
                    break

            status = "COMPLETED" if completed else ("DIED" if died else "INCOMPLETE")
            out_rows.append(
                ChainSegmentResult(
                    segment=spec.seg,
                    config_id=spec.config_id,
                    recording=str(rec_path),
                    status=status,
                    frames=seg_frames,
                    start_area=spec.start_area,
                    end_area=int(end_snap["area"]),
                    lives_start=lives_start,
                    lives_end=int(end_snap["lives"]),
                    end_game_mode=int(end_snap["game_mode"]),
                    note=note,
                )
            )
            if status != "COMPLETED":
                break
        finally:
            env.close()

    completed_count = sum(1 for r in out_rows if r.status == "COMPLETED")
    return {
        "segments": [asdict(r) for r in out_rows],
        "completed_count": completed_count,
        "total_count": len(specs),
        "all_completed": completed_count == len(specs),
        "total_frames": total_frames,
    }


def cmd_diagnose(args: argparse.Namespace) -> int:
    specs = _parse_segments(args.segments)

    state_rows = _state_integrity(specs)
    manifest = _build_manifest(
        specs,
        eval_runs=max(args.candidate_runs, 1),
        max_candidates=max(args.max_candidates, 1),
    )

    # Fast action-timing audit: compare raw-vs-index replay for selected seeds where raw is available.
    timing_audit: list[dict[str, Any]] = []
    for spec in specs:
        try:
            selected = manifest["selected"].get(spec.config_id, {})
            rec = selected.get("recording", "")
            if not rec:
                continue
            rec_path = Path(selected.get("run_dir", "")) / rec
            if not rec_path.exists():
                continue
            raw_actions, is_raw = load_recording_data(rec_path)
            if not is_raw:
                continue
            cfg = get_level_config(spec.config_id)
            action_table = cfg.action_table or DEFAULT_PLATFORMER_ACTIONS
            idx_actions = _raw_to_indices(raw_actions, action_table)
            ev = Evaluator(cfg)
            try:
                raw_res = ev.evaluate(raw_actions, early_terminate=False)
                idx_res = ev.evaluate(idx_actions, early_terminate=False)
            finally:
                ev.close()
            timing_audit.append(
                {
                    "segment": spec.seg,
                    "recording": rec_path.name,
                    "raw_completed": bool(raw_res.completed),
                    "idx_completed": bool(idx_res.completed),
                    "raw_frames": int(raw_res.total_frames),
                    "idx_frames": int(idx_res.total_frames),
                    "lossy_mapping": bool(raw_res.completed and not idx_res.completed),
                }
            )
        except Exception as exc:  # pragma: no cover - runtime dependent
            timing_audit.append({"segment": spec.seg, "error": str(exc)})

    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "segments": [s.seg for s in specs],
        "state_integrity": state_rows,
        "manifest_preview": manifest,
        "action_timing_audit": timing_audit,
    }
    out_path = Path(args.output)
    _save_json(out_path, report)

    print("State integrity:")
    for row in state_rows:
        ok = row.get("state_exists") and row.get("area_ok")
        status = "OK" if ok else "FAIL"
        area = row.get("area")
        area_txt = f"0x{area:02X}" if isinstance(area, int) else "n/a"
        print(
            f"  seg{row['segment']}: {status} state={row['state']} "
            f"area={area_txt} expected=0x{SEGMENT_BY_NUM[row['segment']].start_area:02X}"
        )

    print("\nBest candidates:")
    for row in report["manifest_preview"]["segments"]:
        best = row.get("best")
        if not best:
            print(f"  seg{row['segment']}: no candidate")
            continue
        print(
            f"  seg{row['segment']}: {best['recording']} "
            f"complete={best['completed_runs']}/{best['runs']} "
            f"frames_mean={best['frames_mean']:.1f}"
        )

    lossy = [r for r in timing_audit if r.get("lossy_mapping")]
    if lossy:
        print("\nAction-timing warnings:")
        for row in lossy:
            print(
                f"  seg{row['segment']}: raw completes but index fails "
                f"({row['recording']})"
            )

    print(f"\nSaved diagnose report: {out_path}")
    return 0


def cmd_manifest(args: argparse.Namespace) -> int:
    specs = _parse_segments(args.segments)
    _validate_state_integrity_or_raise(specs)

    manifest = _build_manifest(
        specs,
        eval_runs=max(args.candidate_runs, 1),
        max_candidates=max(args.max_candidates, 1),
    )
    out_path = Path(args.output)
    _save_json(out_path, manifest)

    print(f"Saved manifest: {out_path}")
    for row in manifest["segments"]:
        best = row.get("best")
        if not best:
            print(f"  seg{row['segment']}: no candidate")
            continue
        print(
            f"  seg{row['segment']}: {best['recording']} "
            f"completion={best['completion_rate']*100:.1f}% "
            f"frames={best['frames_mean']:.1f}"
        )
    return 0


def _resolve_training_seed(spec: SegmentSpec, explicit_seed: Path | None, max_candidates: int) -> Path:
    if explicit_seed is not None:
        if not explicit_seed.exists():
            raise FileNotFoundError(f"Seed not found: {explicit_seed}")
        return explicit_seed

    scored = _score_segment_candidates(spec, runs=1, max_candidates=max_candidates)
    best = scored.get("best")
    if best:
        cfg = get_level_config(spec.config_id)
        p = cfg.runs_dir / str(best["recording"])
        if p.exists():
            return p

    cfg = get_level_config(spec.config_id)
    fallback = cfg.runs_dir / "recording_000.json"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"No usable seed for {spec.config_id} in {cfg.runs_dir}")


def cmd_train(args: argparse.Namespace) -> int:
    specs = _parse_segments(args.segments)
    _validate_state_integrity_or_raise(specs)

    for spec in specs:
        cfg = get_level_config(spec.config_id)
        action_table = cfg.action_table or DEFAULT_PLATFORMER_ACTIONS

        explicit_seed = Path(args.seed).resolve() if args.seed and len(specs) == 1 else None
        seed_path = _resolve_training_seed(spec, explicit_seed, max_candidates=max(args.max_candidates, 1))
        seed_actions, seed_is_raw = load_recording_data(seed_path)

        evaluator = Evaluator(cfg, start_state=args.state)
        try:
            # Decide mode with a guardrail audit in auto mode.
            use_raw = args.mode == "raw"
            if args.mode == "auto":
                use_raw = spec.raw_preferred or seed_is_raw
                if seed_is_raw:
                    idx_seed = _raw_to_indices(seed_actions, action_table)
                    raw_res = evaluator.evaluate(seed_actions, early_terminate=False)
                    idx_res = evaluator.evaluate(idx_seed, early_terminate=False)
                    if raw_res.completed and not idx_res.completed:
                        use_raw = True
                        print(
                            f"[seg{spec.seg}] switching to raw mode: seed is lossy under index mapping "
                            f"({seed_path.name})"
                        )
            elif args.mode == "index":
                use_raw = False

            print(f"\n[seg{spec.seg}] {cfg.display_name}")
            print(f"  seed: {seed_path}")
            print(f"  mode: {'raw' if use_raw else 'index'}")
            if args.state:
                print(f"  state override: {args.state}")

            if use_raw:
                raw_seed = seed_actions if seed_is_raw else _indices_to_raw(seed_actions, action_table)
                base_res = evaluator.evaluate(raw_seed, early_terminate=False)
                best_raw, best_result = hillclimb_raw(
                    raw_buttons=raw_seed,
                    evaluator=evaluator,
                    max_iterations=max(args.iterations, 1),
                    output_dir=cfg.runs_dir,
                    verbose=True,
                )
                final_path = cfg.runs_dir / "hillclimb_raw_best_final.json"
                final_data = {
                    "raw_buttons": best_raw,
                    "num_frames": len(best_raw),
                    "fitness": best_result.fitness,
                    "completed": best_result.completed,
                    "total_frames": best_result.total_frames,
                    "max_progress": best_result.max_progress,
                    "segment": spec.seg,
                    "level": cfg.level_id,
                    "seed": str(seed_path),
                }
                final_path.write_text(json.dumps(final_data, indent=2))
                print(
                    f"  baseline: complete={base_res.completed} frames={base_res.total_frames} "
                    f"fitness={base_res.fitness:.1f}"
                )
                print(
                    f"  best:     complete={best_result.completed} frames={best_result.total_frames} "
                    f"fitness={best_result.fitness:.1f}"
                )
                print(f"  saved:    {final_path}")
                chosen_actions = best_raw
            else:
                idx_seed = seed_actions if not seed_is_raw else _raw_to_indices(seed_actions, action_table)
                base_res = evaluator.evaluate(idx_seed, early_terminate=False)
                best_idx, best_result = hillclimb(
                    actions=idx_seed,
                    evaluator=evaluator,
                    max_iterations=max(args.iterations, 1),
                    output_dir=cfg.runs_dir,
                    verbose=True,
                    render_interval=0,
                )
                final_path = cfg.runs_dir / "hillclimb_best_final.json"
                final_data = {
                    "actions": best_idx,
                    "num_frames": len(best_idx),
                    "fitness": best_result.fitness,
                    "completed": best_result.completed,
                    "total_frames": best_result.total_frames,
                    "max_progress": best_result.max_progress,
                    "segment": spec.seg,
                    "level": cfg.level_id,
                    "seed": str(seed_path),
                }
                final_path.write_text(json.dumps(final_data, indent=2))
                print(
                    f"  baseline: complete={base_res.completed} frames={base_res.total_frames} "
                    f"fitness={base_res.fitness:.1f}"
                )
                print(
                    f"  best:     complete={best_result.completed} frames={best_result.total_frames} "
                    f"fitness={best_result.fitness:.1f}"
                )
                print(f"  saved:    {final_path}")
                chosen_actions = best_idx

            val_runs = max(args.validation_runs, 1)
            val_results = [evaluator.evaluate(chosen_actions, early_terminate=False) for _ in range(val_runs)]
            summary = _summarize_results(val_results)
            print(
                f"  validation: complete={summary['completed']}/{summary['runs']} "
                f"frames_mean={summary['frames_mean']:.1f} fitness_mean={summary['fitness_mean']:.1f}"
            )
        finally:
            evaluator.close()

    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    specs = _parse_segments(args.segments)
    manifest_path = Path(args.manifest)

    if not manifest_path.exists():
        manifest = _build_manifest(
            specs,
            eval_runs=max(args.candidate_runs, 1),
            max_candidates=max(args.max_candidates, 1),
        )
        _save_json(manifest_path, manifest)
        print(f"Manifest missing; auto-generated: {manifest_path}")
    else:
        manifest = _load_manifest(manifest_path)

    standalone_rows: list[dict[str, Any]] = []
    for spec in specs:
        rec_path = _resolve_selected_path(spec, manifest)
        actions, _is_raw = load_recording_data(rec_path)
        cfg = get_level_config(spec.config_id)
        evaluator = Evaluator(cfg, start_state=args.state)
        try:
            runs = [evaluator.evaluate(actions, early_terminate=False) for _ in range(max(args.runs, 1))]
        finally:
            evaluator.close()
        summary = _summarize_results(runs)
        standalone_rows.append(
            {
                "segment": spec.seg,
                "config_id": spec.config_id,
                "recording": str(rec_path),
                "summary": summary,
            }
        )
        print(
            f"seg{spec.seg}: complete={summary['completed']}/{summary['runs']} "
            f"frames_mean={summary['frames_mean']:.1f} ({rec_path.name})"
        )

    chain_summary = None
    if args.chain:
        if args.chain_mode == "state":
            chain_summary = _run_state_handoff_chain(
                specs,
                manifest,
                pad_frames=max(args.pad_frames, 0),
                completion_debounce=max(args.completion_debounce, 0),
            )
        else:
            chain_summary = _run_chain_eval(
                specs,
                manifest,
                pad_frames=max(args.pad_frames, 0),
                start_wait_frames=max(args.start_wait_frames, 0),
                completion_debounce=max(args.completion_debounce, 0),
                start_x_tolerance=max(args.start_x_tolerance, 0),
            )
        print(
            f"\nChain ({args.chain_mode}): {chain_summary['completed_count']}/{chain_summary['total_count']} "
            f"segments completed, total_frames={chain_summary['total_frames']}"
        )
        for row in chain_summary["segments"]:
            print(
                f"  seg{row['segment']}: {row['status']} "
                f"frames={row['frames']} end_area=0x{row['end_area']:02X} note={row['note']}"
            )

    report = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path),
        "segments": [s.seg for s in specs],
        "standalone": standalone_rows,
        "chain_mode": args.chain_mode if args.chain else "",
        "chain": chain_summary,
    }
    report_path = Path(args.report)
    _save_json(report_path, report)
    print(f"\nSaved eval report: {report_path}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    # End-to-end pass: manifest selection + eval (+ optional chain smoke)
    m_args = argparse.Namespace(
        segments=args.segments,
        candidate_runs=args.candidate_runs,
        max_candidates=args.max_candidates,
        output=args.manifest,
    )
    rc = cmd_manifest(m_args)
    if rc != 0:
        return rc

    e_args = argparse.Namespace(
        segments=args.segments,
        manifest=args.manifest,
        runs=args.runs,
        chain=args.chain,
        chain_mode=args.chain_mode,
        pad_frames=args.pad_frames,
        start_wait_frames=args.start_wait_frames,
        completion_debounce=args.completion_debounce,
        start_x_tolerance=args.start_x_tolerance,
        report=args.report,
        state=args.state,
        candidate_runs=args.candidate_runs,
        max_candidates=args.max_candidates,
    )
    return cmd_eval(e_args)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SMB 8-4 segmented train/eval pipeline")
    sub = p.add_subparsers(dest="command", required=True)

    p_diag = sub.add_parser("diagnose", help="Validate states and score candidate artifacts")
    p_diag.add_argument("--segments", default="all", help="Segments list, e.g. 1,3-5 or all")
    p_diag.add_argument("--candidate-runs", type=int, default=2, help="Eval runs per candidate")
    p_diag.add_argument("--max-candidates", type=int, default=8, help="Max candidates per segment")
    p_diag.add_argument("--output", default=str(DEFAULT_DIAG_REPORT), help="Output diagnose JSON path")
    p_diag.set_defaults(func=cmd_diagnose)

    p_manifest = sub.add_parser("manifest", help="Build 8-4 candidate-selection manifest")
    p_manifest.add_argument("--segments", default="all", help="Segments list, e.g. 1,3-5 or all")
    p_manifest.add_argument("--candidate-runs", type=int, default=2, help="Eval runs per candidate")
    p_manifest.add_argument("--max-candidates", type=int, default=8, help="Max candidates per segment")
    p_manifest.add_argument("--output", default=str(DEFAULT_MANIFEST), help="Output manifest JSON path")
    p_manifest.set_defaults(func=cmd_manifest)

    p_train = sub.add_parser("train", help="Run hillclimb training for one or more 8-4 segments")
    p_train.add_argument("--segments", default="all", help="Segments list, e.g. 3 or 1,2,5")
    p_train.add_argument("--seed", help="Explicit seed file path (single-segment runs only)")
    p_train.add_argument("--mode", choices=("auto", "index", "raw"), default="auto")
    p_train.add_argument("--iterations", type=int, default=1000)
    p_train.add_argument("--validation-runs", type=int, default=3)
    p_train.add_argument("--max-candidates", type=int, default=8, help="When auto-picking seed")
    p_train.add_argument("--state", help="Optional state override")
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("eval", help="Evaluate selected 8-4 artifacts (standalone and optional chain)")
    p_eval.add_argument("--segments", default="all", help="Segments list, e.g. 1-5 or all")
    p_eval.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Manifest JSON path")
    p_eval.add_argument("--runs", type=int, default=5, help="Standalone repeated eval runs per segment")
    p_eval.add_argument("--chain", action="store_true", help="Run area-aware chain evaluation")
    p_eval.add_argument(
        "--chain-mode",
        choices=("state", "live"),
        default="state",
        help="state=canonical per-segment resets, live=single-session transitions",
    )
    p_eval.add_argument("--pad-frames", type=int, default=60, help="No-input pad frames per segment")
    p_eval.add_argument("--start-wait-frames", type=int, default=240, help="Max frames to wait for expected start area")
    p_eval.add_argument("--completion-debounce", type=int, default=6, help="Frames to confirm completion area persists")
    p_eval.add_argument("--start-x-tolerance", type=int, default=160, help="Allowed X delta from canonical state when segment starts")
    p_eval.add_argument("--report", default=str(DEFAULT_EVAL_REPORT), help="Eval report JSON path")
    p_eval.add_argument("--state", help="Optional standalone state override")
    p_eval.add_argument("--candidate-runs", type=int, default=2, help="If manifest auto-generated")
    p_eval.add_argument("--max-candidates", type=int, default=8, help="If manifest auto-generated")
    p_eval.set_defaults(func=cmd_eval)

    p_run = sub.add_parser("run", help="Build manifest and evaluate in one command")
    p_run.add_argument("--segments", default="all", help="Segments list, e.g. 1-5 or all")
    p_run.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Manifest JSON path")
    p_run.add_argument("--report", default=str(DEFAULT_EVAL_REPORT), help="Eval report JSON path")
    p_run.add_argument("--runs", type=int, default=3, help="Standalone eval runs per segment")
    p_run.add_argument("--chain", action="store_true", help="Run area-aware chain evaluation")
    p_run.add_argument(
        "--chain-mode",
        choices=("state", "live"),
        default="state",
        help="state=canonical per-segment resets, live=single-session transitions",
    )
    p_run.add_argument("--pad-frames", type=int, default=60)
    p_run.add_argument("--start-wait-frames", type=int, default=240)
    p_run.add_argument("--completion-debounce", type=int, default=6)
    p_run.add_argument("--start-x-tolerance", type=int, default=160)
    p_run.add_argument("--candidate-runs", type=int, default=2)
    p_run.add_argument("--max-candidates", type=int, default=8)
    p_run.add_argument("--state", help="Optional standalone state override")
    p_run.set_defaults(func=cmd_run)

    return p


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
