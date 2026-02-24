#!/usr/bin/env python3
"""Resume recording from a specific segment's saved state.

Loads the emulator state saved at a segment boundary and starts a play
session. When done, the new recording is chopped and merged with the
existing segments.

Usage:
    # Re-record from segment 4 (BB Elev Hallway)
    uv run python -m super_metroid_rl.scripts.resume_record \
        --segments-dir optimizer/runs/sm_landing_site/segments \
        --from-seg 4

    # Re-record from a specific saved state
    uv run python -m super_metroid_rl.scripts.resume_record \
        --state seg04_0x97B5 --level sm_elevator_descent
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(
        description="Resume recording from a segment boundary",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Resume from segment 4 (uses saved state from chop)
  %(prog)s --segments-dir optimizer/runs/sm_landing_site/segments --from-seg 4

  # Resume from a named state with explicit level config
  %(prog)s --state seg04_0x97B5 --level sm_elevator_descent

  # Resume from segment 11 (Pit Room return) to finish the route
  %(prog)s --segments-dir optimizer/runs/sm_landing_site/segments --from-seg 11
""",
    )
    parser.add_argument("--segments-dir", help="Path to segments directory (from chop_recording)")
    parser.add_argument("--from-seg", type=int, help="Segment index to resume from")
    parser.add_argument("--state", help="Override: use this state name directly")
    parser.add_argument("--level", "-l", help="Level config ID (auto-detected from segment if omitted)")
    parser.add_argument("--scale", type=int, default=3, help="Display scale")
    args = parser.parse_args()

    if not args.segments_dir and not args.state:
        parser.error("Provide --segments-dir + --from-seg, or --state + --level")

    # Resolve state and level config
    if args.segments_dir:
        seg_dir = Path(args.segments_dir)
        if not seg_dir.is_absolute():
            # Try cwd first, then relative to super_metroid_rl/
            if (Path.cwd() / seg_dir / "manifest.json").exists():
                seg_dir = Path.cwd() / seg_dir
            else:
                game_dir = PROJECT_ROOT / "super_metroid_rl"
                seg_dir = game_dir / seg_dir
        manifest = json.loads((seg_dir / "manifest.json").read_text())
        segments = manifest["segments"]

        if args.from_seg is None:
            print("Available segments:")
            for seg in segments:
                i = seg["index"]
                state = seg.get("next_state_file", "(no state)")
                if i == 0:
                    state = manifest["start_state"]
                else:
                    state = segments[i - 1].get("next_state_file", "(no state)")
                print(f"  {i:2d}  {seg['room_name']:<22s}  {seg['num_frames']:>5d}f  state={state}")
            parser.error("Provide --from-seg N to select a segment")

        seg_idx = args.from_seg
        if seg_idx < 0 or seg_idx >= len(segments):
            parser.error(f"Segment {seg_idx} out of range (0-{len(segments)-1})")

        seg = segments[seg_idx]
        room_id = seg["room_id"]

        # Get state name
        if seg_idx == 0:
            state_name = manifest["start_state"]
        else:
            prev = segments[seg_idx - 1]
            state_name = prev.get("next_state_file")
            if not state_name:
                parser.error(f"Segment {seg_idx} has no saved state from previous segment")

        # Auto-detect level config using (prev_room, this_room) for direction
        prev_room = segments[seg_idx - 1]["room_id"] if seg_idx > 0 else None
        TRANSITION_TO_CONFIG = {
            # Descent phase
            (None, 0x91F8): "sm_landing_site",
            (0x91F8, 0x92FD): "sm_parlor_descent",
            (0x92FD, 0x96BA): "sm_climb_descent",
            (0x96BA, 0x975C): "sm_pit_room_descent",
            (0x975C, 0x97B5): "sm_elevator_descent",
            (0x97B5, 0x9E9F): "sm_morph_ball_collect",
            # Missile detour
            (0x9E9F, 0x9F11): "sm_construction_to_missile",
            (0x9F11, 0xA107): "sm_missile_to_construction",
            (0xA107, 0x9F11): "sm_construction_to_morph",
            # Return phase
            (0x9F11, 0x9E9F): "sm_morph_ball_return",
            (0x9E9F, 0x97B5): "sm_elevator_return",
            (0x97B5, 0x975C): "sm_pit_room_return",
            (0x975C, 0x96BA): "sm_climb_return",
            (0x96BA, 0x92FD): "sm_parlor_to_flyway",
            (0x92FD, 0x9879): "sm_flyway_to_torizo",
        }
        level_id = args.level or TRANSITION_TO_CONFIG.get((prev_room, room_id))
        if not level_id:
            # Fallback: simple room-based guess
            ROOM_FALLBACK = {
                0x91F8: "sm_landing_site", 0x92FD: "sm_parlor_descent",
                0x96BA: "sm_climb_descent", 0x975C: "sm_pit_room_descent",
                0x97B5: "sm_elevator_descent", 0x9E9F: "sm_morph_ball_collect",
                0x9879: "sm_flyway_to_torizo",
            }
            level_id = ROOM_FALLBACK.get(room_id)
        if not level_id:
            parser.error(f"Cannot auto-detect level config for room 0x{room_id:04X}. Use --level.")

        print(f"Resuming from segment {seg_idx}: {seg['room_name']}")
        print(f"  State: {state_name}")
        print(f"  Level config: {level_id}")
    else:
        state_name = args.state
        level_id = args.level
        if not level_id:
            parser.error("--level is required when using --state directly")

    # Import after arg parsing to avoid slow import on --help
    import platformer_common.levels  # noqa: F401
    from platformer_common.runner import cmd_play

    # Build a fake args namespace for cmd_play
    class FakeArgs:
        pass

    play_args = FakeArgs()
    play_args.level = level_id
    play_args.state = state_name
    play_args.scale = args.scale

    print(f"\nStarting play session...")
    print(f"  Arrow keys = D-pad, A = Y/run, Z = B/jump")
    print(f"  TAB = turbo, ESC = stop & save")
    print(f"  Controller supported\n")

    cmd_play(play_args)

    # After recording, remind user to chop
    print(f"\nTo chop this recording into segments:")
    print(f"  uv run python -m super_metroid_rl.scripts.chop_recording \\")
    print(f"    --recording <new_recording_raw.json> --start-state {state_name}")


if __name__ == "__main__":
    main()
