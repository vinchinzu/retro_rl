#!/usr/bin/env python3
"""
Cheat Extractor - Extract save states through the entire MK1 tournament.

Uses RAM manipulation (env.data.set_value) to instantly kill opponents,
then saves states at the start of each new match/endurance/boss fight.

Usage:
    python cheat_extractor.py --char LiuKang                # Full tournament
    python cheat_extractor.py --char LiuKang --start-from Match6  # Start from M6
    python cheat_extractor.py --char LiuKang --scan          # Also scan RAM for counters
    python cheat_extractor.py --all-chars                    # All 7 characters
"""

import os
import sys
import gzip
import time
from pathlib import Path

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import stable_retro as retro
from fighters_common.game_configs import get_game_config

CHARACTERS = ["LiuKang", "Sonya", "JohnnyCage", "Kano", "Raiden", "SubZero", "Scorpion"]
MAX_HEALTH = 161

# MK1 SNES Tournament progression (prefix, display_name, is_endurance)
# NOTE: SNES MK1 has 2 endurance rounds (not 3). Goro is the 2nd opponent
# of Endurance 2. Shang Tsung is the final fight after Goro.
TOURNAMENT = [
    ("Fight",       "Match 1",      False),
    ("Match2",      "Match 2",      False),
    ("Match3",      "Match 3",      False),
    ("Match4",      "Match 4",      False),
    ("Match5",      "Match 5",      False),
    ("Match6",      "Match 6",      False),
    ("Match7",      "Mirror Match", False),
    ("Endurance1",  "Endurance 1",  True),
    ("Endurance2",  "Endurance 2",  True),   # 2nd opponent = Goro
    ("ShangTsung",  "Shang Tsung",  False),
]


def create_env(config, game_dir, state_name):
    """Create a raw (unwrapped) retro environment."""
    retro.data.Integrations.add_custom_path(str(game_dir / "custom_integrations"))
    env = retro.make(
        game=config.game_id,
        state=state_name,
        render_mode="rgb_array",
        inttype=retro.data.Integrations.CUSTOM_ONLY,
        use_restricted_actions=retro.Actions.ALL,
    )
    return env


def step_noop(env, n=1):
    """Step with no input."""
    noop = np.zeros(12, dtype=np.int8)
    for _ in range(n):
        env.step(noop)


def step_start(env, n=4):
    """Press START button."""
    buttons = np.zeros(12, dtype=np.int8)
    buttons[3] = 1  # START
    for _ in range(n):
        env.step(buttons)


def read_state(env):
    """Read health and timer from RAM."""
    return {
        "health": env.data.lookup_value("health"),
        "enemy_health": env.data.lookup_value("enemy_health"),
        "timer": env.data.lookup_value("timer"),
    }


def kill_enemy(env):
    """Win the current round by setting enemy HP low and timer low.

    Setting enemy_health to 0 doesn't trigger real KO logic.
    Setting enemy_health to 1 requires landing a hit (unreliable).

    Instead: set enemy HP to 1 and timer to near-zero, so the round ends
    via timeout with health advantage. This triggers the game's natural
    round-end logic and proper match progression.
    """
    # Let a few frames pass so the fight is active
    step_noop(env, 30)
    # Keep our health maxed, set enemy low, and drain timer
    env.data.set_value("health", MAX_HEALTH)
    env.data.set_value("enemy_health", 1)
    env.data.set_value("timer", 1)
    # Wait for the round to end naturally via timeout
    # The game needs frames to process the timer expiry
    for i in range(600):
        step_noop(env, 1)
        # Keep reinforcing our health advantage in case of glitches
        if i % 30 == 0:
            env.data.set_value("health", MAX_HEALTH)
            env.data.set_value("enemy_health", 1)
        ehp = env.data.lookup_value("enemy_health")
        timer = env.data.lookup_value("timer")
        # Round ended when timer hits 0 and health values change
        if timer == 0 and ehp <= 1:
            step_noop(env, 120)  # Let round-end animation play
            return


def wait_for_health_reset(env, max_frames=6000, require_enemy_full=True,
                          no_press_frames=0):
    """Wait for health values to reset (new round or new match).

    Returns True if reset detected, False if timeout.

    Args:
        no_press_frames: Number of frames to wait before pressing START at all.
            Use this after match wins to let the game go through its natural
            post-match sequence (KO anim → victory pose → VS screen) without
            accidentally skipping out of tournament mode.
    """
    noop = np.zeros(12, dtype=np.int8)

    for i in range(max_frames):
        # Don't press anything during the initial grace period
        if i < no_press_frames:
            env.step(noop)
        # Then press START occasionally to advance through screens
        elif i % 120 == 90:
            step_start(env, 4)
        else:
            env.step(noop)

        s = read_state(env)

        # Check for health reset
        enemy_ok = s["enemy_health"] == MAX_HEALTH if require_enemy_full else True
        if s["health"] >= MAX_HEALTH and enemy_ok and s["timer"] > 50:
            # Stability check - must be stable for 60 consecutive frames
            stable = 0
            for _ in range(120):
                env.step(noop)
                s2 = read_state(env)
                enemy_ok2 = s2["enemy_health"] == MAX_HEALTH if require_enemy_full else True
                if s2["health"] >= MAX_HEALTH and enemy_ok2 and s2["timer"] > 50:
                    stable += 1
                    if stable >= 60:
                        return True
                else:
                    break
    return False


def win_match(env, verbose=True):
    """Win a standard match (best of 3 rounds) via health cheat."""
    # Round 1
    if verbose:
        print("    Round 1: ", end="", flush=True)
    kill_enemy(env)
    if verbose:
        print("KO!", flush=True)

    # Wait for round 2 (no grace period - rounds transition quickly)
    if not wait_for_health_reset(env, max_frames=4000, no_press_frames=0):
        if verbose:
            print("    WARNING: Could not detect round 2 start")
        return False

    # Round 2
    if verbose:
        print("    Round 2: ", end="", flush=True)
    kill_enemy(env)
    if verbose:
        print("KO! MATCH WIN!", flush=True)

    return True


def win_endurance_match(env, config, game_dir, char, prefix, verbose=True):
    """Win an endurance match (2 opponents, each best-of-3).

    For Endurance2, the second opponent is Goro - saves both the B state
    and a Goro alias.
    """
    # Beat first opponent
    if verbose:
        print("  Opponent 1:", flush=True)
    if not win_match(env, verbose):
        return False

    # Wait for second opponent to appear
    if verbose:
        print("  Waiting for opponent 2...", flush=True)

    # Grace period for endurance opponent transition
    if wait_for_health_reset(env, max_frames=8000, no_press_frames=600):
        # Save state for endurance half B
        b_name = f"{prefix}B_{char}"
        save_state(env, config, game_dir, b_name)
        if verbose:
            s = read_state(env)
            print(f"  Saved: {b_name}.state (HP:{s['health']} EHP:{s['enemy_health']} T:{s['timer']})")
        # Endurance2B is actually Goro - save alias
        if prefix == "Endurance2":
            goro_name = f"Goro_{char}"
            save_state(env, config, game_dir, goro_name)
            if verbose:
                print(f"  Saved alias: {goro_name}.state (Goro = Endurance 2 opp 2)")
    else:
        if verbose:
            print("  WARNING: Could not detect opponent 2 start")
            print("  Trying to continue anyway...")

    # Beat second opponent
    if verbose:
        print("  Opponent 2:", flush=True)
    if not win_match(env, verbose):
        return False

    return True


def save_state(env, config, game_dir, state_name):
    """Save current emulator state to disk."""
    state_data = env.em.get_state()
    save_path = game_dir / "custom_integrations" / config.game_id / f"{state_name}.state"
    with gzip.open(save_path, "wb") as f:
        f.write(state_data)
    return save_path


def find_start_index(start_from):
    """Find tournament index for a given prefix."""
    if not start_from:
        return 0
    for i, (prefix, _, _) in enumerate(TOURNAMENT):
        if prefix == start_from:
            return i
    print(f"ERROR: Unknown stage '{start_from}'. Valid: {[p for p,_,_ in TOURNAMENT]}")
    sys.exit(1)


def extract_tournament(char, config, game_dir, start_from=None, scan_ram=False):
    """Extract states for one character through the full tournament."""
    start_idx = find_start_index(start_from)

    # Determine starting state
    start_prefix = TOURNAMENT[start_idx][0]
    start_state = f"{start_prefix}_{char}"

    state_path = game_dir / "custom_integrations" / config.game_id / f"{start_state}.state"
    if not state_path.exists():
        print(f"  ERROR: Starting state {start_state}.state not found!")
        return False

    print(f"\n{'='*60}")
    print(f"  Character: {char}")
    print(f"  Starting:  {start_state} ({TOURNAMENT[start_idx][1]})")
    print(f"  Stages:    {len(TOURNAMENT) - start_idx} remaining")
    print(f"{'='*60}")

    env = create_env(config, game_dir, start_state)
    env.reset()

    # Initial RAM snapshot for scanning
    prev_ram = env.get_ram().copy() if scan_ram else None

    extracted = []

    for match_idx in range(start_idx, len(TOURNAMENT)):
        prefix, match_name, is_endurance = TOURNAMENT[match_idx]

        print(f"\n--- {match_name} (Stage {match_idx + 1}/12) ---")
        s = read_state(env)
        print(f"  State: HP={s['health']} EHP={s['enemy_health']} Timer={s['timer']}")

        # Win the match
        if is_endurance:
            success = win_endurance_match(env, config, game_dir, char, prefix)
        else:
            success = win_match(env)

        if not success:
            print(f"  FAILED at {match_name}!")
            break

        # Check if this is the last match (Shang Tsung)
        if match_idx >= len(TOURNAMENT) - 1:
            print(f"\n  TOURNAMENT COMPLETE!")
            break

        # Wait for next match to start
        # Long grace period (900 frames = ~15s) to let post-match sequence play:
        # KO anim → "FINISH HIM" → victory pose → VS screen → loading
        next_prefix, next_name, _ = TOURNAMENT[match_idx + 1]
        print(f"  Waiting for {next_name}...", end=" ", flush=True)

        if wait_for_health_reset(env, max_frames=12000, no_press_frames=900):
            next_state_name = f"{next_prefix}_{char}"
            save_state(env, config, game_dir, next_state_name)
            s = read_state(env)
            print(f"SAVED: {next_state_name}.state (HP:{s['health']} EHP:{s['enemy_health']} T:{s['timer']})")
            extracted.append(next_state_name)

            # RAM scan
            if scan_ram:
                curr_ram = env.get_ram().copy()
                diffs = np.where(prev_ram != curr_ram)[0]
                counters = [(int(a), int(prev_ram[a]), int(curr_ram[a]))
                            for a in diffs
                            if curr_ram[a] == prev_ram[a] + 1 and prev_ram[a] < 20]
                if counters:
                    print(f"  RAM counter candidates: {counters[:5]}")
                prev_ram = curr_ram
        else:
            print("FAILED - could not detect next match!")
            print("  The game may have ended or gotten stuck on a screen.")
            # Try harder - maybe we need more START presses
            print("  Trying extended wait with aggressive START pressing...")
            for retry in range(3000):
                if retry % 30 == 0:
                    step_start(env, 8)
                else:
                    step_noop(env)
                s = read_state(env)
                if s["health"] >= MAX_HEALTH and s["enemy_health"] >= MAX_HEALTH and s["timer"] > 50:
                    stable = 0
                    for _ in range(60):
                        step_noop(env)
                        s2 = read_state(env)
                        if s2["health"] >= MAX_HEALTH and s2["enemy_health"] >= MAX_HEALTH:
                            stable += 1
                    if stable >= 40:
                        next_state_name = f"{next_prefix}_{char}"
                        save_state(env, config, game_dir, next_state_name)
                        print(f"  RECOVERED! Saved: {next_state_name}.state")
                        extracted.append(next_state_name)
                        break
            else:
                print("  Could not recover. Stopping extraction.")
                break

    env.close()
    return extracted


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract MK1 tournament states via RAM cheats")
    parser.add_argument("--char", default="LiuKang", help="Character to extract for")
    parser.add_argument("--start-from", default=None,
                       help="Start from this stage (e.g., Match6, Endurance1)")
    parser.add_argument("--scan", action="store_true", help="Scan RAM for match counter")
    parser.add_argument("--all-chars", action="store_true", help="Extract for all 7 characters")
    args = parser.parse_args()

    config = get_game_config("mk1")
    game_dir = ROOT_DIR / config.game_dir_name

    print("=" * 60)
    print("MK1 CHEAT EXTRACTOR")
    print("=" * 60)

    start_time = time.time()

    if args.all_chars:
        chars = CHARACTERS
    else:
        chars = [args.char]

    all_extracted = {}
    for char in chars:
        extracted = extract_tournament(char, config, game_dir, args.start_from, args.scan)
        all_extracted[char] = extracted

    elapsed = time.time() - start_time

    # Summary
    print("\n" + "=" * 60)
    print("EXTRACTION SUMMARY")
    print(f"Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print("=" * 60)
    for char, states in all_extracted.items():
        if states:
            print(f"  {char}: {len(states)} new states - {', '.join(states)}")
        else:
            print(f"  {char}: FAILED or no new states")


if __name__ == "__main__":
    main()
