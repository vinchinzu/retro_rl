#!/usr/bin/env python3
"""Chop a full-run raw-button recording into per-room segments.

Single-pass replay: detects room transitions, saves emulator states at
each transition boundary, and writes per-segment raw button slices.

Usage:
    uv run python -m super_metroid_rl.scripts.chop_recording \
        --recording optimizer/runs/sm_landing_site/recording_003_raw.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from retro_harness.env import make_env

# SM RAM addresses
LEVEL_ID_ADDR = 0x079B  # u16 room ID
PLAYER_X_ADDR = 0x0AF6  # u16
PLAYER_Y_ADDR = 0x0AFA  # u16
HEALTH_ADDR = 0x09C2    # u16

# Room names for nice output
ROOM_NAMES = {
    0x91F8: "Landing Site",
    0x92FD: "Parlor",
    0x96BA: "Climb",
    0x975C: "Pit Room",
    0x97B5: "BB Elev Hallway",
    0x9E9F: "Morph Ball Room",
    0x9F11: "Construction Zone",
    0xA107: "First Missile Room",
    0x9804: "Bomb Torizo Room",
    0x9879: "Flyway",
}


def read_u16(ram: np.ndarray, addr: int) -> int:
    return int(ram[addr]) | (int(ram[addr + 1]) << 8)


def main():
    parser = argparse.ArgumentParser(description="Chop raw recording into per-room segments")
    parser.add_argument("--recording", required=True, help="Path to *_raw.json file")
    parser.add_argument("--start-state", default="ZebesStart", help="Initial emulator state")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: same as recording)")
    parser.add_argument("--save-states", action="store_true", default=True,
                        help="Save emulator states at room transitions")
    parser.add_argument("--no-save-states", dest="save_states", action="store_false")
    args = parser.parse_args()

    recording_path = Path(args.recording)
    if not recording_path.is_absolute():
        if (Path.cwd() / recording_path).exists():
            recording_path = Path.cwd() / recording_path
        else:
            recording_path = PROJECT_ROOT / "super_metroid_rl" / recording_path

    data = json.loads(recording_path.read_text())
    raw_buttons = data["raw_buttons"]
    total_frames = len(raw_buttons)
    print(f"Loaded {total_frames} raw frames from {recording_path.name}")

    output_dir = Path(args.output_dir) if args.output_dir else recording_path.parent / "segments"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Game setup
    game_dir = PROJECT_ROOT / "super_metroid_rl"
    state_dir = game_dir / "custom_integrations" / "SuperMetroid-Snes"

    env = make_env(
        game="SuperMetroid-Snes",
        state=args.start_state,
        game_dir=str(game_dir),
        render_mode="rgb_array",
    )
    env.reset()

    # Read initial room
    ram = env.get_ram()
    prev_room = read_u16(ram, LEVEL_ID_ADDR)
    room_name = ROOM_NAMES.get(prev_room, f"0x{prev_room:04X}")
    print(f"Starting in room 0x{prev_room:04X} ({room_name})")

    # Track segments: list of (start_frame, end_frame, room_id, state_bytes)
    segments: list[dict] = []
    current_segment_start = 0
    current_room = prev_room

    # Save initial state
    initial_state = env.em.get_state()

    # Workaround: stable-retro ignores SNES Select for SM weapon toggle.
    # Apply rising-edge RAM write so missile doors open during replay.
    _select_prev = False
    _select_val = 0
    _has_selected_item = None

    for frame_idx in range(total_frames):
        buttons = raw_buttons[frame_idx]
        action_arr = np.array(buttons, dtype=np.int8)
        # Pad if needed
        action_size = env.action_space.shape[0]
        if len(buttons) < action_size:
            action_arr = np.zeros(action_size, dtype=np.int8)
            action_arr[:len(buttons)] = buttons

        # Select toggle workaround
        if len(buttons) > 2 and buttons[2]:
            if not _select_prev:
                if _has_selected_item is None:
                    try:
                        env.unwrapped.data.lookup_value("selected_item")
                        _has_selected_item = True
                    except Exception:
                        _has_selected_item = False
                if _has_selected_item:
                    _select_val ^= 1
                    try:
                        env.unwrapped.data.set_value("selected_item", _select_val)
                    except Exception:
                        pass
            _select_prev = True
        else:
            _select_prev = False

        env.step(action_arr)

        ram = env.get_ram()
        room = read_u16(ram, LEVEL_ID_ADDR)

        if room != current_room:
            # Room transition detected!
            x = read_u16(ram, PLAYER_X_ADDR)
            y = read_u16(ram, PLAYER_Y_ADDR)
            hp = read_u16(ram, HEALTH_ADDR)
            old_name = ROOM_NAMES.get(current_room, f"0x{current_room:04X}")
            new_name = ROOM_NAMES.get(room, f"0x{room:04X}")

            seg_frames = frame_idx - current_segment_start
            print(f"  F{current_segment_start:>6}-{frame_idx:<6} ({seg_frames:>5}f) "
                  f"0x{current_room:04X} {old_name}")

            # Save segment info
            segments.append({
                "index": len(segments),
                "room_id": current_room,
                "room_name": old_name,
                "start_frame": current_segment_start,
                "end_frame": frame_idx,
                "num_frames": seg_frames,
            })

            # Save emulator state at transition point (start of new room)
            if args.save_states:
                transition_state = env.em.get_state()
                state_name = f"seg{len(segments):02d}_0x{room:04X}"
                state_path = state_dir / f"{state_name}.state"
                with gzip.open(state_path, "wb") as f:
                    f.write(transition_state)
                segments[-1]["next_state_file"] = state_name

            current_segment_start = frame_idx
            current_room = room

    # Final segment (may be incomplete - died or ran out of frames)
    seg_frames = total_frames - current_segment_start
    room_name = ROOM_NAMES.get(current_room, f"0x{current_room:04X}")
    print(f"  F{current_segment_start:>6}-{total_frames:<6} ({seg_frames:>5}f) "
          f"0x{current_room:04X} {room_name}")
    segments.append({
        "index": len(segments),
        "room_id": current_room,
        "room_name": room_name,
        "start_frame": current_segment_start,
        "end_frame": total_frames,
        "num_frames": seg_frames,
    })

    env.close()

    # Write per-segment raw button slices
    print(f"\nWriting {len(segments)} segment files to {output_dir}/")
    for seg in segments:
        i = seg["index"]
        start = seg["start_frame"]
        end = seg["end_frame"]
        room_id = seg["room_id"]
        name = ROOM_NAMES.get(room_id, f"0x{room_id:04X}").replace(" ", "_").lower()

        seg_data = {
            "raw_buttons": raw_buttons[start:end],
            "metadata": {
                "room_id": f"0x{room_id:04X}",
                "room_name": seg["room_name"],
                "start_frame": start,
                "end_frame": end,
                "num_frames": end - start,
                "source": recording_path.name,
                "segment_index": i,
            },
        }

        # Also save action indices for compat
        from platformer_common.levels.super_metroid import SM_ACTIONS
        from platformer_common.actions import buttons_to_action_index
        seg_data["actions"] = [
            buttons_to_action_index(frame, action_table=SM_ACTIONS)
            for frame in raw_buttons[start:end]
        ]
        seg_data["num_frames"] = end - start

        out_path = output_dir / f"seg{i:02d}_{name}.json"
        out_path.write_text(json.dumps(seg_data, indent=2))

        state_info = f" state={seg['next_state_file']}" if "next_state_file" in seg else ""
        print(f"  seg{i:02d}_{name}.json  ({end - start} frames){state_info}")

    # Write manifest
    manifest = {
        "source": str(recording_path),
        "start_state": args.start_state,
        "total_frames": total_frames,
        "num_segments": len(segments),
        "segments": segments,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest: {manifest_path}")
    print("Done!")


if __name__ == "__main__":
    main()
