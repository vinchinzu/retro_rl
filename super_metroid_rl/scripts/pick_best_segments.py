#!/usr/bin/env python3
"""Pick the best segment for each room across multiple recordings.

Given multiple chopped recordings (segment directories from chop_recording.py),
evaluates each recording's segments and picks the one with the highest fitness
for each room. Since segments are independent (each starts from a saved state),
we can freely mix segments from different recordings.

Usage:
    uv run python -m super_metroid_rl.scripts.pick_best_segments \
        --segments-dirs segments_rec003 segments_rec004 \
        --output-dir best_mixed/

    uv run python -m super_metroid_rl.scripts.pick_best_segments \
        --segments-dirs segments_rec003 segments_rec004 \
        --configs sm_parlor_descent sm_climb_descent \
        --output-dir best_mixed/
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# Map (current_room_id, next_room_id) → config_id for disambiguation
# when a room appears in both descent and return phases.
TRANSITION_TO_CONFIG: dict[tuple[int, int], str] = {
    (0x91F8, 0x92FD): "sm_landing_site",
    (0x92FD, 0x96BA): "sm_parlor_descent",
    (0x96BA, 0x975C): "sm_climb_descent",
    (0x975C, 0x97B5): "sm_pit_room_descent",
    (0x97B5, 0x9E9F): "sm_elevator_descent",
    (0x9E9F, 0x9F11): "sm_morph_to_construction",
    (0x9F11, 0xA107): "sm_construction_to_missile",
    (0xA107, 0x9F11): "sm_missile_to_construction",
    (0x9F11, 0x9E9F): "sm_construction_to_morph",
    (0x9E9F, 0x97B5): "sm_morph_ball_return",
    (0x97B5, 0x975C): "sm_elevator_return",
    (0x975C, 0x96BA): "sm_pit_room_return",
    (0x96BA, 0x92FD): "sm_climb_return",
    (0x92FD, 0x9879): "sm_parlor_to_flyway",
    (0x9879, 0x9804): "sm_flyway_to_torizo",
}

# Fallback: room_id → first config in descent order
ROOM_TO_CONFIG_DEFAULT: dict[int, str] = {
    0x91F8: "sm_landing_site",
    0x92FD: "sm_parlor_descent",
    0x96BA: "sm_climb_descent",
    0x975C: "sm_pit_room_descent",
    0x97B5: "sm_elevator_descent",
    0x9E9F: "sm_morph_ball_collect",
    0x9F11: "sm_construction_to_missile",
    0xA107: "sm_missile_to_construction",
    0x9879: "sm_flyway_to_torizo",
    0x9804: "sm_flyway_to_torizo",  # destination, not a traversal
}


def resolve_config_id(
    room_id: int,
    next_room_id: int | None,
    prev_room_id: int | None,
) -> str | None:
    """Determine which level config to use for a segment.

    Uses (room_id, next_room_id) transition pair for disambiguation.
    """
    if next_room_id is not None:
        key = (room_id, next_room_id)
        if key in TRANSITION_TO_CONFIG:
            return TRANSITION_TO_CONFIG[key]

    return ROOM_TO_CONFIG_DEFAULT.get(room_id)


def load_segments_from_dir(
    seg_dir: Path,
) -> list[dict]:
    """Load segment metadata and raw buttons from a chop_recording output dir.

    Returns list of dicts with keys:
        source_dir, index, room_id, room_name, raw_buttons,
        state_name, start_frame, end_frame, config_id
    """
    manifest_path = seg_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"  WARNING: No manifest.json in {seg_dir}, skipping")
        return []

    manifest = json.loads(manifest_path.read_text())
    segments = manifest["segments"]
    start_state = manifest.get("start_state", "ZebesStart")

    # Load the full raw buttons for extending past boundaries
    rec_source = manifest.get("source", "")
    full_raw = None
    rec_path = Path(rec_source)
    if not rec_path.exists():
        # Try relative to segment dir
        for candidate in [
            seg_dir.parent / rec_path.name,
            seg_dir.parent / rec_path.name.replace(".json", "_raw.json"),
        ]:
            if candidate.exists():
                rec_path = candidate
                break
    if rec_path.exists():
        try:
            full_raw = json.loads(rec_path.read_text()).get("raw_buttons")
        except Exception:
            pass

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
            prev_seg = segments[i - 1]
            state_name = prev_seg.get("next_state_file")
            if not state_name:
                continue

        # Load raw buttons from segment file
        name = room_name.replace(" ", "_").lower()
        seg_file = seg_dir / f"seg{i:02d}_{name}.json"
        if not seg_file.exists():
            # Try to find by glob
            matches = list(seg_dir.glob(f"seg{i:02d}_*.json"))
            if matches:
                seg_file = matches[0]
            else:
                continue

        seg_data = json.loads(seg_file.read_text())
        raw = seg_data.get("raw_buttons", [])

        # Extend past segment boundary to catch room transitions
        if full_raw and end_f < len(full_raw):
            extended_end = min(end_f + 120, len(full_raw))
            raw = full_raw[start_f:extended_end]

        # Determine config_id
        next_room = segments[i + 1]["room_id"] if i + 1 < len(segments) else None
        prev_room = segments[i - 1]["room_id"] if i > 0 else None
        config_id = resolve_config_id(room_id, next_room, prev_room)

        results.append({
            "source_dir": str(seg_dir),
            "source_name": seg_dir.name,
            "index": i,
            "room_id": room_id,
            "room_name": room_name,
            "raw_buttons": raw,
            "state_name": state_name,
            "start_frame": start_f,
            "end_frame": end_f,
            "config_id": config_id,
            "seg_file": str(seg_file),
        })

    return results


def pick_best_segments(
    segment_dirs: list[Path],
    config_ids: list[str] | None = None,
) -> dict[str, dict]:
    """Evaluate segments from multiple sources, pick the best per config.

    Args:
        segment_dirs: Directories containing chopped segments (manifest.json)
        config_ids: If set, only evaluate these config IDs

    Returns:
        Dict mapping config_id to best segment info:
        {config_id: {source, fitness, frames, completed, raw_buttons, state_name}}
    """
    import platformer_common.levels.super_metroid  # noqa: F401
    from platformer_common.evaluator import Evaluator
    from platformer_common.level_config import get_level_config

    # Collect all segments from all sources
    all_segments: list[dict] = []
    for seg_dir in segment_dirs:
        if not seg_dir.is_absolute():
            candidates = [
                Path.cwd() / seg_dir,
                PROJECT_ROOT / "super_metroid_rl" / seg_dir,
                PROJECT_ROOT / seg_dir,
            ]
            for c in candidates:
                if c.exists():
                    seg_dir = c
                    break

        print(f"Loading segments from {seg_dir.name}...")
        segs = load_segments_from_dir(seg_dir)
        all_segments.extend(segs)
        print(f"  Found {len(segs)} segments")

    if not all_segments:
        print("No segments found!")
        return {}

    # Group by config_id
    by_config: dict[str, list[dict]] = {}
    for seg in all_segments:
        cid = seg["config_id"]
        if cid is None:
            continue
        if config_ids and cid not in config_ids:
            continue
        by_config.setdefault(cid, []).append(seg)

    print(f"\nEvaluating {len(by_config)} segment configs "
          f"across {len(segment_dirs)} sources\n")
    print(f"{'Config':<30s}  {'Source':<20s}  {'Frames':>6s}  "
          f"{'Progress':>8s}  {'Fitness':>10s}  {'Status':>10s}  {'Best'}")
    print("-" * 110)

    best_per_config: dict[str, dict] = {}

    for config_id in sorted(by_config):
        candidates = by_config[config_id]
        best_fitness = float("-inf")
        best_entry = None

        for seg in candidates:
            raw = seg["raw_buttons"]
            state = seg["state_name"]
            source = seg["source_name"]

            try:
                config = get_level_config(config_id)
                ev = Evaluator(config, start_state=state)
                result = ev.evaluate(raw, early_terminate=False)
                ev.close()

                is_best = result.fitness > best_fitness
                if is_best:
                    best_fitness = result.fitness
                    best_entry = {
                        "source": source,
                        "source_dir": seg["source_dir"],
                        "fitness": result.fitness,
                        "frames": result.total_frames,
                        "completed": result.completed,
                        "max_progress": result.max_progress,
                        "raw_buttons": raw,
                        "state_name": state,
                        "seg_file": seg["seg_file"],
                    }

                status = "COMPLETED" if result.completed else (
                    "DIED" if result.died else "progress"
                )
                marker = " <-- BEST" if is_best else ""
                print(f"  {config_id:<30s}  {source:<20s}  {result.total_frames:>6d}f  "
                      f"{result.max_progress:>8.1f}  {result.fitness:>10.0f}  "
                      f"{status:>10s}{marker}")

            except Exception as e:
                print(f"  {config_id:<30s}  {source:<20s}  ERROR: {e}")

        if best_entry:
            best_per_config[config_id] = best_entry

    return best_per_config


def write_best_segments(
    best: dict[str, dict],
    output_dir: Path,
) -> None:
    """Write the best segments to an output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries = []
    for config_id in sorted(best):
        entry = best[config_id]
        raw = entry["raw_buttons"]
        state = entry["state_name"]

        # Write segment file
        seg_data = {
            "raw_buttons": raw,
            "num_frames": len(raw),
            "metadata": {
                "config_id": config_id,
                "source": entry["source"],
                "fitness": entry["fitness"],
                "completed": entry["completed"],
                "state_name": state,
            },
        }
        seg_path = output_dir / f"{config_id}.json"
        seg_path.write_text(json.dumps(seg_data, indent=2))

        manifest_entries.append({
            "config_id": config_id,
            "source": entry["source"],
            "fitness": entry["fitness"],
            "completed": entry["completed"],
            "frames": entry["frames"],
            "max_progress": entry["max_progress"],
            "state_name": state,
            "file": seg_path.name,
        })

    # Write manifest
    manifest = {
        "description": "Best segments picked from multiple sources",
        "num_segments": len(manifest_entries),
        "segments": manifest_entries,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Pick best segment per room from multiple recordings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--segments-dirs", nargs="+", required=True,
                        help="Segment directories (output of chop_recording.py)")
    parser.add_argument("--configs", nargs="*", default=None,
                        help="Only evaluate these config IDs (default: all)")
    parser.add_argument("--output-dir", default="best_mixed",
                        help="Output directory for best segments (default: best_mixed)")
    args = parser.parse_args()

    seg_dirs = [Path(d) for d in args.segments_dirs]
    best = pick_best_segments(seg_dirs, config_ids=args.configs)

    if not best:
        print("\nNo valid segments found!")
        sys.exit(1)

    # Summary
    completed = sum(1 for v in best.values() if v["completed"])
    total = len(best)
    total_frames = sum(v["frames"] for v in best.values() if v["completed"])

    print(f"\n{'=' * 80}")
    print(f"Best segments: {completed}/{total} completed")
    print(f"Total completion frames: {total_frames} ({total_frames / 60:.1f}s)")

    print(f"\nBest per config:")
    for cid in sorted(best):
        entry = best[cid]
        status = "OK" if entry["completed"] else "INCOMPLETE"
        print(f"  {cid:<30s}  {entry['source']:<20s}  "
              f"{entry['frames']:>5d}f  fitness={entry['fitness']:.0f}  {status}")

    # Write output
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    write_best_segments(best, output_dir)
    print(f"\nBest segments written to {output_dir}")


if __name__ == "__main__":
    main()
