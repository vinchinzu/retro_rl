"""Extract save states at each area transition in SMB 8-4.

Replays recording_001 and saves states at key transition frames.
These become the starting states for individual 8-4 segments.

Usage:
    uv run python super_mario_bros/extract_84_states.py
"""

import json
import gzip
from pathlib import Path

import retro
from platformer_common.level_config import get_level_config

ROOT = Path(__file__).resolve().parent
GAME = "SuperMarioBros-Nes-v0"
GAME_DIR = str(ROOT / "custom_integrations")
STATES_DIR = ROOT / "custom_integrations" / GAME
RECORDING = ROOT / "optimizer" / "runs" / "smb_8_4" / "recording_001.json"

# Area transitions discovered via --trace on the 8-4 recording.
# Frames where we want save states (just BEFORE the transition starts):
# These are the stable area boundaries, chosen a few frames before transition.
SAVE_POINTS = {
    # (frame, state_name, description)
    # Seg 1 start = Level8_4 (already exists, frame 0, area 0x65)
    # Save AFTER each transition so the segment starts in the new area.
    (790, "Level8_4_seg2", "In pipe section (0xE5), after transition at ~777"),
    (1125, "Level8_4_seg3", "Castle maze (0x65), after exit pipe at ~1115"),
    (1895, "Level8_4_seg4", "Underwater (0x02), after entering at ~1883"),
    (3030, "Level8_4_seg5", "Final castle (0x65), after exiting water at ~3020"),
}


def main():
    retro.data.Integrations.add_custom_path(GAME_DIR)
    env = retro.make(GAME, inttype=retro.data.Integrations.CUSTOM_ONLY)
    env.reset()

    # Load the Level8_4 start state
    state_path = STATES_DIR / "Level8_4.state"
    with gzip.open(state_path, "rb") as f:
        start_state = f.read()
    env.em.set_state(start_state)

    # Load recording actions. Prefer raw_buttons (replay-safe env-applied inputs);
    # fall back to action indices if raw companion data is unavailable.
    raw_path = RECORDING.with_name(RECORDING.stem + "_raw.json")
    with open(raw_path) as f:
        raw_data = json.load(f)
    if "raw_buttons" in raw_data:
        raw_actions = raw_data["raw_buttons"]
    elif "actions" in raw_data:
        action_table = get_level_config("smb_8_4").action_table
        if action_table is None:
            raise ValueError("smb_8_4 action table not configured")
        raw_actions = [
            action_table[int(a)] if int(a) < len(action_table) else action_table[0]
            for a in raw_data["actions"]
        ]
    else:
        raise ValueError(f"No raw_buttons/actions found in {raw_path}")

    # Replay and save states at transition points
    save_frames = {frame: (name, desc) for frame, name, desc in SAVE_POINTS}
    max_frame = max(save_frames.keys()) + 10

    for frame_idx in range(min(max_frame, len(raw_actions))):
        action = raw_actions[frame_idx]
        # Truncate to NES 9 buttons
        env_size = env.action_space.shape[0]
        step_action = action[:env_size] if len(action) > env_size else action
        env.step(step_action)

        if frame_idx in save_frames:
            name, desc = save_frames[frame_idx]
            state_data = env.em.get_state()
            out_path = STATES_DIR / f"{name}.state"
            with gzip.open(out_path, "wb") as f:
                f.write(state_data)
            # Read RAM to verify
            ram = env.get_ram()
            area = ram[0x0750]
            lives = ram[0x075A]
            x_page = ram[0x006D]
            x_off = ram[0x0086]
            player_x = int(x_page) * 256 + int(x_off)
            print(f"  [{name}] frame={frame_idx} area=0x{area:02X} x={player_x} lives={lives} -- {desc}")
            print(f"    Saved: {out_path}")

    env.close()
    print(f"\nDone! Created {len(SAVE_POINTS)} segment states.")
    print("\nNext: run 'uv run python -m super_mario_bros.optimizer list-levels' to verify segments.")


if __name__ == "__main__":
    main()
