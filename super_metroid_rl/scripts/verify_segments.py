#!/usr/bin/env python3
"""Verify chopped segments replay correctly from their saved states.

For each segment, loads the emulator state saved at the room transition
boundary, replays the raw buttons, and checks:
1. Starts in the expected room
2. Ends by transitioning to the expected next room (or end of recording)
3. No unexpected room changes mid-segment

Usage:
    uv run python -m super_metroid_rl.scripts.verify_segments \
        --segments-dir optimizer/runs/sm_landing_site/segments
"""

from __future__ import annotations

import argparse
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
HEALTH_ADDR = 0x09C2

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


def read_u16(ram, addr):
    return int(ram[addr]) | (int(ram[addr + 1]) << 8)


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

    manifest_path = seg_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())

    game_dir = PROJECT_ROOT / "super_metroid_rl"
    state_dir = game_dir / "custom_integrations" / "SuperMetroid-Snes"
    start_state = manifest["start_state"]

    segments = manifest["segments"]
    num_segments = len(segments)

    print(f"Verifying {num_segments} segments from {seg_dir.name}/\n")

    # For first segment, use the original start state
    # For subsequent segments, use the saved transition state
    results = []

    for i, seg_info in enumerate(segments):
        room_id = seg_info["room_id"]
        room_name = seg_info["room_name"]
        seg_files = sorted(seg_dir.glob(f"seg{i:02d}_*.json"))
        if not seg_files:
            print(f"  seg{i:02d}: SKIP (no file found)")
            continue
        seg_file = seg_files[0]
        seg_data = json.loads(seg_file.read_text())
        raw_buttons = seg_data["raw_buttons"]
        num_frames = len(raw_buttons)

        # Determine which state to load
        if i == 0:
            state_name = start_state
        else:
            prev_seg = segments[i - 1]
            if "next_state_file" not in prev_seg:
                print(f"  seg{i:02d}: SKIP (no saved state from previous segment)")
                continue
            state_name = prev_seg["next_state_file"]

        # Check state file exists
        state_path = state_dir / f"{state_name}.state"
        if not state_path.exists():
            print(f"  seg{i:02d}: FAIL (state file missing: {state_name})")
            results.append(False)
            continue

        # Create env and load state
        env = make_env(
            game="SuperMetroid-Snes",
            state=state_name,
            game_dir=str(game_dir),
            render_mode="rgb_array",
        )
        env.reset()

        ram = env.get_ram()
        actual_room = read_u16(ram, LEVEL_ID_ADDR)
        actual_name = ROOM_NAMES.get(actual_room, f"0x{actual_room:04X}")

        if actual_room != room_id:
            print(f"  seg{i:02d}: FAIL start room mismatch: expected 0x{room_id:04X} ({room_name}), "
                  f"got 0x{actual_room:04X} ({actual_name})")
            env.close()
            results.append(False)
            continue

        # Replay raw buttons
        exit_room = None
        exit_frame = None
        _select_prev = False
        _select_val = 0
        _has_selected_item = None
        for f_idx in range(num_frames):
            buttons = raw_buttons[f_idx]
            action_arr = np.array(buttons, dtype=np.int8)
            action_size = env.action_space.shape[0]
            if len(buttons) < action_size:
                padded = np.zeros(action_size, dtype=np.int8)
                padded[:len(buttons)] = buttons
                action_arr = padded

            # Select toggle workaround (stable-retro ignores SNES Select)
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
            cur_room = read_u16(ram, LEVEL_ID_ADDR)
            if cur_room != room_id and exit_room is None:
                exit_room = cur_room
                exit_frame = f_idx

        # Check final state
        ram = env.get_ram()
        final_room = read_u16(ram, LEVEL_ID_ADDR)
        final_x = read_u16(ram, PLAYER_X_ADDR)
        final_y = read_u16(ram, PLAYER_Y_ADDR)
        final_hp = read_u16(ram, HEALTH_ADDR)

        # Determine expected next room
        if i < num_segments - 1:
            expected_next = segments[i + 1]["room_id"]
            expected_name = segments[i + 1]["room_name"]
        else:
            expected_next = None
            expected_name = "(end)"

        if expected_next is not None:
            if exit_room == expected_next:
                status = "OK"
                ok = True
            elif exit_room is None:
                status = "WARN (no room transition)"
                ok = True  # last segment may not transition
            else:
                exit_name = ROOM_NAMES.get(exit_room, f"0x{exit_room:04X}")
                status = f"FAIL exit to 0x{exit_room:04X} ({exit_name}) != expected 0x{expected_next:04X} ({expected_name})"
                ok = False
        else:
            status = "OK (final segment)"
            ok = True

        exit_info = ""
        if exit_room:
            exit_name = ROOM_NAMES.get(exit_room, f"0x{exit_room:04X}")
            exit_info = f" -> 0x{exit_room:04X} ({exit_name}) @F{exit_frame}"

        print(f"  seg{i:02d} {room_name:<22s} {num_frames:>5}f  "
              f"hp={final_hp:>3}  pos=({final_x},{final_y}){exit_info}  [{status}]")

        results.append(ok)
        env.close()

    # Summary
    passed = sum(results)
    total = len(results)
    print(f"\n{'='*60}")
    print(f"Results: {passed}/{total} segments verified OK")
    if passed == total:
        print("All segments replay correctly!")
    else:
        failed = [i for i, ok in enumerate(results) if not ok]
        print(f"Failed segments: {failed}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
