#!/usr/bin/env python3
"""Evaluate chopped segments using the platformer_common evaluator.

Maps recording segments to registered level configs and evaluates each with
the correct start state from the segment chop.

Usage:
    uv run python -m super_metroid_rl.scripts.eval_segments \
        --segments-dir optimizer/runs/sm_landing_site/segments
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import platformer_common.levels  # noqa: F401  trigger registration
from platformer_common.evaluator import Evaluator
from platformer_common.level_config import get_level_config


# Map room_id -> list of level_config_ids that START in that room
# (A room can appear in both descent and return phases)
ROOM_TO_CONFIGS = {
    0x91F8: ["sm_landing_site"],
    0x92FD: ["sm_parlor_descent", "sm_parlor_to_flyway"],
    0x96BA: ["sm_climb_descent", "sm_climb_return"],
    0x975C: ["sm_pit_room_descent", "sm_pit_room_return"],
    0x97B5: ["sm_elevator_descent", "sm_elevator_return"],
    0x9E9F: ["sm_morph_ball_collect", "sm_morph_to_construction", "sm_morph_ball_return"],
    0x9F11: ["sm_construction_to_missile", "sm_construction_to_morph"],
    0xA107: ["sm_missile_to_construction"],
    0x9879: ["sm_flyway_to_torizo"],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments-dir", required=True)
    args = parser.parse_args()

    seg_dir = Path(args.segments_dir)
    if not seg_dir.is_absolute():
        if (Path.cwd() / seg_dir / "manifest.json").exists():
            seg_dir = Path.cwd() / seg_dir
        else:
            seg_dir = PROJECT_ROOT / "super_metroid_rl" / seg_dir

    manifest = json.loads((seg_dir / "manifest.json").read_text())
    segments = manifest["segments"]
    start_state = manifest["start_state"]

    # Load full recording for extended replay
    rec_path = Path(manifest["source"])
    if not rec_path.exists():
        rec_path = seg_dir.parent / rec_path.name.replace(".json", "_raw.json")
    rec_data = json.loads(rec_path.read_text())
    full_raw = rec_data["raw_buttons"]

    print(f"Evaluating {len(segments)} segments\n")
    print(f"{'Seg':>3s}  {'Room':<22s}  {'Config':<30s}  {'Frames':>6s}  "
          f"{'Progress':>8s}  {'Fitness':>10s}  {'Status'}")
    print("-" * 110)

    results = []
    for seg in segments:
        i = seg["index"]
        room_id = seg["room_id"]
        room_name = seg["room_name"]
        start_f = seg["start_frame"]
        end_f = seg["end_frame"]

        # Determine state name
        if i == 0:
            state_name = start_state
        else:
            prev = segments[i - 1]
            state_name = prev.get("next_state_file")
            if not state_name:
                print(f"  {i:2d}  {room_name:<22s}  {'(no state)':30s}")
                continue

        # Find matching config based on room_id and next room (direction)
        configs = ROOM_TO_CONFIGS.get(room_id, [])
        if not configs:
            print(f"  {i:2d}  {room_name:<22s}  {'(no config for room)':30s}  "
                  f"{end_f - start_f:>6d}f")
            results.append(None)
            continue

        # Pick config: use (room_id, next_room_id) pair to disambiguate
        next_room = segments[i + 1]["room_id"] if i + 1 < len(segments) else None
        prev_room = segments[i - 1]["room_id"] if i > 0 else None
        TRANSITION_TO_CONFIG = {
            (0x9E9F, 0x9F11): "sm_morph_to_construction",   # morph -> construction (descent)
            (0x9F11, 0xA107): "sm_construction_to_missile",  # construction -> missile
            (0xA107, 0x9F11): "sm_missile_to_construction",  # missile -> construction (return)
            (0x9F11, 0x9E9F): "sm_construction_to_morph",   # construction -> morph (return)
            (0x9E9F, 0x97B5): "sm_morph_ball_return",       # morph -> elevator (return)
            (0x97B5, 0x975C): "sm_elevator_return",          # elevator -> pit (return)
            (0x975C, 0x96BA): "sm_pit_room_return",          # pit -> climb (return)
            (0x96BA, 0x92FD): "sm_climb_return",             # climb -> parlor (return)
            (0x92FD, 0x9879): "sm_parlor_to_flyway",         # parlor -> flyway
        }
        transition_key = (room_id, next_room)
        if transition_key in TRANSITION_TO_CONFIG:
            config_id = TRANSITION_TO_CONFIG[transition_key]
        elif len(configs) == 1:
            config_id = configs[0]
        else:
            config_id = configs[0]  # default to first

        config = get_level_config(config_id)

        # Extend raw buttons past segment boundary to catch room transition
        extended_end = min(end_f + 100, len(full_raw))
        raw = full_raw[start_f:extended_end]

        try:
            ev = Evaluator(config, start_state=state_name)
            result = ev.evaluate(raw, early_terminate=False)

            status = "COMPLETED" if result.completed else ("DIED" if result.died else "progress")
            print(f"  {i:2d}  {room_name:<22s}  {config_id:<30s}  {result.total_frames:>6d}f  "
                  f"{result.max_progress:>8.3f}  {result.fitness:>10.0f}  {status}")
            results.append(result)
            ev.close()
        except Exception as e:
            print(f"  {i:2d}  {room_name:<22s}  {config_id:<30s}  ERROR: {e}")
            results.append(None)

    # Summary
    completed = sum(1 for r in results if r and r.completed)
    total = sum(1 for r in results if r is not None)
    print(f"\n{completed}/{total} segments completed")

    # Total frames for completed segments
    total_frames = sum(r.total_frames for r in results if r and r.completed)
    print(f"Total completion frames: {total_frames} ({total_frames / 60:.1f}s)")


if __name__ == "__main__":
    main()
