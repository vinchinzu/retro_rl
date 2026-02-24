#!/usr/bin/env python3
"""Evaluate the full speedrun route from Landing Site to Bomb Torizo.

Chains segments from multiple recordings into a unified 16-segment evaluation.
Each segment is evaluated independently using its saved state and raw buttons.

Usage:
    uv run python -m super_metroid_rl.scripts.eval_full_route
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import platformer_common.levels  # noqa: F401
from platformer_common.evaluator import Evaluator
from platformer_common.level_config import get_level_config

# Full speedrun route: 16 segments
# (config_id, state_name, segment_file_path_relative_to_runs_dir)
FULL_ROUTE = [
    # === Descent phase (recording_003) ===
    ("sm_landing_site",          "ZebesStart",           "segments/seg00_landing_site.json"),
    ("sm_parlor_descent",        "seg01_0x92FD",         "segments/seg01_parlor.json"),
    ("sm_climb_descent",         "seg02_0x96BA",         "segments/seg02_climb.json"),
    ("sm_pit_room_descent",      "seg03_0x975C",         "segments/seg03_pit_room.json"),
    ("sm_elevator_descent",      "seg04_0x97B5_fixed",   "segments/seg04_bb_elev_hallway.json"),
    ("sm_morph_ball_collect",    "seg05_0x9E9F_auto",    "segments/seg05_morph_ball_room.json"),
    ("sm_morph_to_construction", "seg06_0x9F11",         "segments/seg06_construction_zone.json"),
    ("sm_construction_to_missile","seg07_0xA107",        "segments/seg07_first_missile_room.json"),
    ("sm_missile_to_construction","seg08_0x9F11",        "segments/seg08_construction_zone.json"),
    ("sm_construction_to_morph", "seg09_0x9E9F",         "segments/seg09_morph_ball_room.json"),
    ("sm_elevator_return",       "seg10_0x97B5",         "segments/seg10_bb_elev_hallway.json"),
    # === Return phase (return recording from seg11_0x975C) ===
    ("sm_pit_room_return",       "seg11_0x975C",         "segments_return/seg00_pit_room.json"),
    ("sm_climb_return",          "ret00_0x96BA",         "segments_return/seg01_climb.json"),
    ("sm_parlor_to_flyway",      "ret01_0x92FD",         "segments_return/seg02_parlor.json"),
    ("sm_flyway_to_torizo",      "ret02_0x9879",         "segments_return/seg03_flyway.json"),
    # === Endpoint (no config - just report) ===
    # Bomb Torizo Room is the destination, not a traversal segment
]


def main():
    runs_dir = PROJECT_ROOT / "super_metroid_rl" / "optimizer" / "runs" / "sm_landing_site"

    print(f"Full route evaluation: {len(FULL_ROUTE)} segments\n")
    print(f"{'#':>2s}  {'Room':<22s}  {'Config':<30s}  {'State':<25s}  "
          f"{'Frames':>6s}  {'Progress':>8s}  {'Fitness':>10s}  {'Status'}")
    print("-" * 140)

    results = []
    cumulative_frames = 0

    for i, (config_id, state_name, seg_file_rel) in enumerate(FULL_ROUTE):
        seg_path = runs_dir / seg_file_rel
        if not seg_path.exists():
            print(f"  {i:2d}  {'?':<22s}  {config_id:<30s}  {state_name:<25s}  MISSING: {seg_file_rel}")
            results.append(None)
            continue

        seg_data = json.loads(seg_path.read_text())
        raw = seg_data["raw_buttons"]
        room_name = seg_data.get("metadata", {}).get("room_name", "?")

        # Extend raw buttons with 100 frames of no-input to catch delayed transitions
        raw_extended = raw + [[0] * 12] * 100

        try:
            config = get_level_config(config_id)
            ev = Evaluator(config, start_state=state_name)
            result = ev.evaluate(raw_extended, early_terminate=False)

            status = "COMPLETED" if result.completed else ("DIED" if result.died else "progress")
            print(f"  {i:2d}  {room_name:<22s}  {config_id:<30s}  {state_name:<25s}  "
                  f"{result.total_frames:>6d}f  {result.max_progress:>8.1f}  "
                  f"{result.fitness:>10.0f}  {status}")
            results.append(result)
            if result.completed:
                cumulative_frames += result.total_frames
            ev.close()
        except Exception as e:
            print(f"  {i:2d}  {room_name:<22s}  {config_id:<30s}  {state_name:<25s}  ERROR: {e}")
            import traceback; traceback.print_exc()
            results.append(None)

    # Summary
    completed = sum(1 for r in results if r and r.completed)
    total = sum(1 for r in results if r is not None)
    print(f"\n{'='*80}")
    print(f"Results: {completed}/{total} segments completed")
    print(f"Total completion frames: {cumulative_frames} ({cumulative_frames / 60:.1f}s)")

    if completed < total:
        failed = [i for i, r in enumerate(results) if r and not r.completed]
        missing = [i for i, r in enumerate(results) if r is None]
        if failed:
            print(f"Failed/incomplete: {failed}")
        if missing:
            print(f"Missing: {missing}")

    return 0 if completed == total else 1


if __name__ == "__main__":
    sys.exit(main())
