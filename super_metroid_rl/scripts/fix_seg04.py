#!/usr/bin/env python3
"""Fix seg04 (BB Elev Hallway descent) using the fixed state.

The original saved state at the room transition boundary doesn't preserve
elevator initialization flags. seg04_0x97B5_fixed.state was created by
replaying from seg03 through the door transition, preserving proper init.

This script finds the correct raw button slice and saves an updated segment.
"""
import json
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from retro_harness.env import make_env

LEVEL_ID_ADDR = 0x079B
PLAYER_X_ADDR = 0x0AF6
PLAYER_Y_ADDR = 0x0AFA

def read_u16(ram, addr):
    return int(ram[addr]) | (int(ram[addr + 1]) << 8)

def main():
    game_dir = PROJECT_ROOT / "super_metroid_rl"
    state_dir = game_dir / "custom_integrations" / "SuperMetroid-Snes"
    seg_dir = game_dir / "optimizer" / "runs" / "sm_landing_site" / "segments"

    # Load original recording
    rec_path = game_dir / "optimizer" / "runs" / "sm_landing_site" / "recording_003_raw.json"
    rec = json.loads(rec_path.read_text())
    full_raw = rec["raw_buttons"]

    # Original seg04 boundaries: start_frame=2953, end_frame=3256
    original_start = 2953
    original_end = 3256

    # The fixed state was saved ~16 frames into the room
    # Try different offsets to find the one that works
    env = make_env(
        game="SuperMetroid-Snes",
        state="seg04_0x97B5_fixed",
        game_dir=str(game_dir),
        render_mode="rgb_array",
    )

    best_offset = None

    for offset in range(original_start, original_start + 30):
        env.reset()

        ram = env.get_ram()
        start_room = read_u16(ram, LEVEL_ID_ADDR)
        start_x = read_u16(ram, PLAYER_X_ADDR)
        start_y = read_u16(ram, PLAYER_Y_ADDR)

        if offset == original_start:
            print(f"Fixed state: room=0x{start_room:04X}, pos=({start_x},{start_y})")

        # Replay from offset to original_end + some buffer
        test_end = min(original_end + 50, len(full_raw))
        buttons_slice = full_raw[offset:test_end]

        exit_room = None
        exit_frame = None

        for f_idx, buttons in enumerate(buttons_slice):
            action_arr = np.array(buttons, dtype=np.int8)
            action_size = env.action_space.shape[0]
            if len(buttons) < action_size:
                padded = np.zeros(action_size, dtype=np.int8)
                padded[:len(buttons)] = buttons
                action_arr = padded

            env.step(action_arr)

            ram = env.get_ram()
            cur_room = read_u16(ram, LEVEL_ID_ADDR)
            cur_y = read_u16(ram, PLAYER_Y_ADDR)

            if cur_room != start_room and exit_room is None:
                exit_room = cur_room
                exit_frame = f_idx
                break

        if exit_room == 0x9E9F:  # Morph Ball Room
            print(f"  offset={offset}: transitions to Morph Ball at frame {exit_frame} (absolute frame {offset + exit_frame})")
            if best_offset is None:
                best_offset = offset
        elif offset < original_start + 5:
            ram = env.get_ram()
            final_y = read_u16(ram, PLAYER_Y_ADDR)
            print(f"  offset={offset}: exit_room={'0x{:04X}'.format(exit_room) if exit_room else 'NONE'}, final_y={final_y}")

    if best_offset is None:
        print("ERROR: Could not find working offset!")
        env.close()
        return 1

    print(f"\nUsing offset {best_offset}")

    # Now extract the exact segment
    env.reset()
    ram = env.get_ram()
    start_room = read_u16(ram, LEVEL_ID_ADDR)

    seg_raw = full_raw[best_offset:original_end + 50]  # include some buffer
    actual_end_frame = None

    for f_idx, buttons in enumerate(seg_raw):
        action_arr = np.array(buttons, dtype=np.int8)
        action_size = env.action_space.shape[0]
        if len(buttons) < action_size:
            padded = np.zeros(action_size, dtype=np.int8)
            padded[:len(buttons)] = buttons
            action_arr = padded

        env.step(action_arr)

        ram = env.get_ram()
        cur_room = read_u16(ram, LEVEL_ID_ADDR)

        if cur_room != start_room:
            actual_end_frame = f_idx
            break

    if actual_end_frame is None:
        print("ERROR: No room transition found!")
        env.close()
        return 1

    # Trim to just the segment (before room transition)
    seg_buttons = full_raw[best_offset:best_offset + actual_end_frame]
    print(f"Segment: {len(seg_buttons)} frames (offset {best_offset} to {best_offset + actual_end_frame})")

    # Also compute action indices
    from platformer_common.levels.super_metroid import SM_ACTIONS
    from platformer_common.actions import buttons_to_action_index
    actions = [buttons_to_action_index(f, action_table=SM_ACTIONS) for f in seg_buttons]

    # Write updated segment file
    seg_data = {
        "raw_buttons": seg_buttons,
        "actions": actions,
        "num_frames": len(seg_buttons),
        "metadata": {
            "room_id": "0x97B5",
            "room_name": "BB Elev Hallway",
            "start_frame": best_offset,
            "end_frame": best_offset + actual_end_frame,
            "num_frames": len(seg_buttons),
            "source": "recording_003_raw.json",
            "segment_index": 4,
            "note": "Uses seg04_0x97B5_fixed.state (proper elevator init from door transition replay)"
        },
    }

    out_path = seg_dir / "seg04_bb_elev_hallway.json"
    out_path.write_text(json.dumps(seg_data, indent=2))
    print(f"Wrote {out_path}")

    env.close()

    # Now update the manifest to reference the fixed state
    manifest_path = seg_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())

    # Update seg03's next_state_file to point to the fixed state
    manifest["segments"][3]["next_state_file"] = "seg04_0x97B5_fixed"

    # Update seg04's frame boundaries
    manifest["segments"][4]["start_frame"] = best_offset
    manifest["segments"][4]["end_frame"] = best_offset + actual_end_frame
    manifest["segments"][4]["num_frames"] = actual_end_frame

    # Also update seg04's next_state_file to use the auto-saved one
    # (seg05_0x9E9F_auto was saved when replaying through the elevator)
    manifest["segments"][4]["next_state_file"] = "seg05_0x9E9F_auto"

    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Updated manifest: seg03.next_state_file -> seg04_0x97B5_fixed, seg04 -> seg05_0x9E9F_auto")

    print("\nDone! Now verify with eval_segments.py")
    return 0

if __name__ == "__main__":
    sys.exit(main())
