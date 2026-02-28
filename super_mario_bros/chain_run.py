"""Chain SMB any% recordings into one continuous end-to-end run.

Replays each level's recording, then steps through the transition
(flagpole, score tally, new level load) before starting the next.
Reports what works and what breaks.

Usage:
    uv run python super_mario_bros/chain_run.py [--render] [--save-states]
"""

from __future__ import annotations

import argparse
import gzip
import time
from pathlib import Path

import numpy as np
import stable_retro as retro

ROOT = Path(__file__).resolve().parent
GAME = "SuperMarioBros-Nes-v0"
GAME_DIR = str(ROOT / "custom_integrations")
STATES_DIR = ROOT / "custom_integrations" / GAME
RUNS_DIR = ROOT / "optimizer" / "runs"

# Any% route: level_id, recording_dir, description
# level_id = (world-1)*4 + (level-1)
ROUTE = [
    ("smb_1_1", "smb_1_1", "1-1"),
    ("smb_1_2", "smb_1_2", "1-2 (warp to W4)"),
    ("smb_4_1", "smb_4_1", "4-1"),
    ("smb_4_2", "smb_4_2", "4-2 (warp to W8)"),
    ("smb_8_1", "smb_8_1", "8-1"),
    ("smb_8_2", "smb_8_2", "8-2"),
    ("smb_8_3", "smb_8_3", "8-3"),
    # 8-4 is one continuous level (use the full recording, not segments)
    ("smb_8_4", "smb_8_4", "8-4"),
]

# RAM addresses
ADDR = {
    "world": 0x075F,
    "level": 0x0760,
    "lives": 0x075A,
    "player_status": 0x000E,  # 0x0B = dying
    "game_mode": 0x0770,      # 0=demo, 1=playing, 2=end world, 3=game over
    "area_pointer": 0x0750,
    "x_page": 0x006D,
    "x_offset": 0x0086,
    "player_state": 0x000E,
    "timer_hundreds": 0x07F8,
}


def read_ram(env):
    ram = env.get_ram()
    return {
        "world": int(ram[ADDR["world"]]),
        "level": int(ram[ADDR["level"]]),
        "level_id": int(ram[ADDR["world"]]) * 4 + int(ram[ADDR["level"]]),
        "lives": int(ram[ADDR["lives"]]),
        "game_mode": int(ram[ADDR["game_mode"]]),
        "area_pointer": int(ram[ADDR["area_pointer"]]),
        "player_x": int(ram[ADDR["x_page"]]) * 256 + int(ram[ADDR["x_offset"]]),
        "timer": int(ram[ADDR["timer_hundreds"]]),
    }


def load_recording(run_dir: str) -> tuple[list[int] | list[list[int]], bool]:
    """Load recording from a run directory.

    Returns (actions, is_raw), preferring companion raw-button captures when
    available for highest-fidelity replay.
    """
    from platformer_common.route import load_recording_data

    act_path = RUNS_DIR / run_dir / "recording_000.json"
    actions, is_raw = load_recording_data(act_path)
    return actions, is_raw


# SMB action table (must match platformer_common/levels/smb.py)
# NES: [B, NULL, SELECT, START, UP, DOWN, LEFT, RIGHT, A]
SMB_ACTIONS = [
    [0,0,0,0,0,0,0,0,0],  # 0: NOTHING
    [0,0,0,0,0,0,0,1,0],  # 1: RIGHT
    [1,0,0,0,0,0,0,1,0],  # 2: RIGHT + B (run)
    [1,0,0,0,0,0,0,1,1],  # 3: RIGHT + B + A (run+jump)
    [0,0,0,0,0,0,0,1,1],  # 4: RIGHT + A (walk+jump)
    [0,0,0,0,0,0,0,0,1],  # 5: JUMP
    [0,0,0,0,0,0,1,0,0],  # 6: LEFT
    [1,0,0,0,0,0,1,0,0],  # 7: LEFT + B (run left)
    [1,0,0,0,0,0,1,0,1],  # 8: LEFT + B + A (run left+jump)
    [0,0,0,0,0,0,1,0,1],  # 9: LEFT + A (walk left+jump)
    [0,0,0,0,0,1,0,0,0],  # 10: DOWN
]


def step_transition(env, max_frames=600, target_world=None, target_level=None, label=""):
    """Step through level transition until the next level starts.

    Waits for game_mode to return to gameplay and the level to change.
    Returns (success, frames_taken, ram_state).
    """
    env_size = env.action_space.shape[0]
    no_input = np.zeros(env_size, dtype=np.int8)

    initial = read_ram(env)
    initial_lid = initial["level_id"]

    # Phase 1: Wait for transition to start (game_mode changes or level_id changes)
    for i in range(max_frames):
        env.step(no_input)
        vals = read_ram(env)

        # Check if we've arrived at the target level
        if target_world is not None and target_level is not None:
            if vals["world"] == target_world and vals["level"] == target_level:
                # Wait a few more frames for the level to fully load
                for _ in range(30):
                    env.step(no_input)
                vals = read_ram(env)
                print(f"    [{label}] Transition: arrived at W{vals['world']+1}-{vals['level']+1} "
                      f"after {i+31} frames (lives={vals['lives']}, x={vals['player_x']})")
                return True, i + 31, vals

        # Generic: level_id changed and game is in play mode
        if vals["level_id"] != initial_lid and vals["game_mode"] == 1:
            for _ in range(30):
                env.step(no_input)
            vals = read_ram(env)
            print(f"    [{label}] Transition: level_id {initial_lid} -> {vals['level_id']} "
                  f"after {i+31} frames (W{vals['world']+1}-{vals['level']+1}, lives={vals['lives']})")
            return True, i + 31, vals

    vals = read_ram(env)
    print(f"    [{label}] Transition TIMEOUT after {max_frames} frames "
          f"(game_mode={vals['game_mode']}, level_id={vals['level_id']}, lives={vals['lives']})")
    return False, max_frames, vals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true", help="Show pygame window")
    parser.add_argument("--save-states", action="store_true", help="Save chained states to disk")
    args = parser.parse_args()

    from retro_harness.env import make_env
    env = make_env(GAME, "Level1_1", GAME_DIR)
    # Cache initial state
    initial_state = env.em.get_state()

    env_size = env.action_space.shape[0]
    total_frames = 0
    results = []

    print("=" * 60)
    print("SMB Any% Chain Run")
    print("=" * 60)

    for i, (level_id, run_dir, desc) in enumerate(ROUTE):
        vals = read_ram(env)
        print(f"\n--- {desc} (W{vals['world']+1}-{vals['level']+1}) ---")
        print(f"  Start: lives={vals['lives']} x={vals['player_x']} game_mode={vals['game_mode']} "
              f"area=0x{vals['area_pointer']:02X}")

        # Save chained state if requested
        if args.save_states:
            state_data = env.em.get_state()
            state_name = f"Chained_{run_dir}"
            out = STATES_DIR / f"{state_name}.state"
            with gzip.open(out, "wb") as f:
                f.write(state_data)
            print(f"  Saved state: {out}")

        # Load and replay recording
        try:
            actions, is_raw = load_recording(run_dir)
        except FileNotFoundError:
            print(f"  NO RECORDING for {run_dir}")
            results.append((desc, "NO_RECORDING", 0))
            break

        died = False
        completed = False
        initial_lives = vals["lives"]
        seg_frames = 0

        for fi, action in enumerate(actions):
            if is_raw:
                a = np.array(action[:env_size], dtype=np.int8)
            else:
                buttons = SMB_ACTIONS[action] if action < len(SMB_ACTIONS) else SMB_ACTIONS[0]
                a = np.array(buttons[:env_size], dtype=np.int8)

            env.step(a)
            seg_frames += 1

            rv = read_ram(env)

            # Death check
            if rv["lives"] < initial_lives:
                print(f"  DIED at frame {fi} (x={rv['player_x']}, lives={rv['lives']})")
                died = True
                break

            # For 8-4: check game_mode for completion
            if level_id == "smb_8_4" and rv["game_mode"] == 2:
                print(f"  8-4 COMPLETED at frame {fi} (game_mode=2, Bowser defeated!)")
                completed = True
                break

        if not died and not completed:
            rv = read_ram(env)
            print(f"  Finished replay: {seg_frames} frames, x={rv['player_x']}, "
                  f"level_id={rv['level_id']}, lives={rv['lives']}")

        total_frames += seg_frames

        if died:
            results.append((desc, "DIED", seg_frames))
            print(f"  >>> CHAIN BROKEN at {desc}")
            break

        if level_id == "smb_8_4":
            # Final level - no transition needed
            results.append((desc, "COMPLETED" if completed else "ENDED", seg_frames))
            break

        # Step through transition to next level
        next_entry = ROUTE[i + 1] if i + 1 < len(ROUTE) else None
        if next_entry:
            # Figure out expected next world/level from the route
            # Parse from the level_id string (smb_W_L)
            parts = next_entry[0].split("_")
            if len(parts) >= 3:
                target_w = int(parts[1]) - 1  # 0-indexed
                target_l = int(parts[2]) - 1
            else:
                target_w, target_l = None, None

            success, trans_frames, trans_vals = step_transition(
                env, max_frames=600,
                target_world=target_w, target_level=target_l,
                label=desc,
            )
            total_frames += trans_frames

            if not success:
                results.append((desc, "TRANSITION_FAILED", seg_frames))
                print(f"  >>> TRANSITION FAILED after {desc}")
                break
            results.append((desc, "OK", seg_frames))
        else:
            results.append((desc, "OK", seg_frames))

    # Summary
    print("\n" + "=" * 60)
    print("CHAIN RUN SUMMARY")
    print("=" * 60)
    print(f"Total frames: {total_frames} ({total_frames/60:.1f}s)")
    print()
    for desc, status, frames in results:
        icon = {"OK": "✓", "COMPLETED": "✓", "ENDED": "~", "DIED": "✗",
                "TRANSITION_FAILED": "⚠", "NO_RECORDING": "?"}
        print(f"  {icon.get(status, '?')} {desc}: {status} ({frames} frames)")

    env.close()


if __name__ == "__main__":
    main()
