#!/usr/bin/env python3
"""
MK2 Cheat Extractor - Extract save states through the entire MK2 tournament.

Uses Pro Action Replay cheat codes (via env.em.add_cheat) to instantly win matches,
then saves states at the start of each new match.

Usage:
    python cheat_extractor.py --char LiuKang                # Full tournament
    python cheat_extractor.py --char LiuKang --start-from Match6  # Start from M6
    python cheat_extractor.py --all-chars                    # All 12 characters
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
from retro_harness.fighters.game_configs import get_game_config

# All 12 playable MK2 characters
CHARACTERS = [
    "LiuKang", "KungLao", "JohnnyCage", "Reptile", "SubZero", "ShangTsung",
    "Kitana", "Jax", "Mileena", "Baraka", "Scorpion", "Raiden"
]

MAX_HEALTH = 161

# MK2 SNES Tournament progression
# Opponent order varies by character, so we use generic numbering
# Each character fights 10-12 matches depending on path
TOURNAMENT = [
    ("Fight",       "Match 1",      False),
    ("Match2",      "Match 2",      False),
    ("Match3",      "Match 3",      False),
    ("Match4",      "Match 4",      False),
    ("Match5",      "Match 5",      False),
    ("Match6",      "Match 6",      False),
    ("Match7",      "Match 7",      False),
    ("Match8",      "Match 8",      False),
    ("ShangTsung",  "Shang Tsung",  False),  # Sub-boss
    ("Kintaro",     "Kintaro",      False),  # Boss 1 (4-armed)
    ("ShaoKahn",    "Shao Kahn",    False),  # Final boss
]

# RAM addresses for MK2 (high WRAM)
# P1 health: WRAM 0x2EFC -> SNES bus 0x7E2EFC (get_ram index 0x4EFD)
# P2 health: WRAM 0x30AA -> SNES bus 0x7E30AA (get_ram index 0x50AB)
# In get_ram(), addresses are offset by 0x2001 from WRAM
P1_HEALTH_GETRAM_ADDR = 0x4EFD  # get_ram() index
P2_HEALTH_GETRAM_ADDR = 0x50AB  # get_ram() index
# Pro Action Replay format uses SNES bus addresses (0x7E0000 + WRAM offset)
P1_HEALTH_PAR_ADDR = 0x7E2EFC  # SNES bus address for WRAM 0x2EFC
P2_HEALTH_PAR_ADDR = 0x7E30AA  # SNES bus address for WRAM 0x30AA


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


def read_health(env):
    """Read health from RAM directly."""
    ram = env.unwrapped.get_ram()
    return {
        "health": int(ram[P1_HEALTH_GETRAM_ADDR]),
        "enemy_health": int(ram[P2_HEALTH_GETRAM_ADDR]),
    }


def set_health_cheats(env, p1_hp, p2_hp):
    """Set health values via Pro Action Replay cheat codes."""
    # Clear existing cheats first
    env.em.clear_cheats()
    # Add new cheats in PAR format: AAAAAA:VV
    p1_cheat = f"{P1_HEALTH_PAR_ADDR:06X}:{p1_hp:02X}"
    p2_cheat = f"{P2_HEALTH_PAR_ADDR:06X}:{p2_hp:02X}"
    env.em.add_cheat(p1_cheat)
    env.em.add_cheat(p2_cheat)


def kill_enemy(env):
    """Win the current round by setting enemy HP low."""
    # Let a few frames pass so the fight is active
    step_noop(env, 30)
    
    # Set cheats: max P1 health, 1 HP for P2
    set_health_cheats(env, MAX_HEALTH, 1)
    
    # Wait for enemy to be KO'd (cheats will keep values locked)
    for i in range(300):
        step_noop(env, 1)
        
        # Check if enemy is KO'd
        health = read_health(env)
        if health["enemy_health"] == 0:
            # Clear cheats and let round-end animation play
            env.em.clear_cheats()
            step_noop(env, 120)
            return
    
    # Clear cheats if timeout
    env.em.clear_cheats()


def wait_for_health_reset(env, max_frames=6000, require_enemy_full=True,
                          no_press_frames=0):
    """Wait for health values to reset (new round or new match)."""
    noop = np.zeros(12, dtype=np.int8)

    for i in range(max_frames):
        if i < no_press_frames:
            env.step(noop)
        elif i % 120 == 90:
            step_start(env, 4)
        else:
            env.step(noop)

        health = read_health(env)
        
        enemy_ok = health["enemy_health"] == MAX_HEALTH if require_enemy_full else True
        if health["health"] >= MAX_HEALTH and enemy_ok:
            # Stability check
            stable = 0
            for _ in range(120):
                env.step(noop)
                health2 = read_health(env)
                enemy_ok2 = health2["enemy_health"] == MAX_HEALTH if require_enemy_full else True
                if health2["health"] >= MAX_HEALTH and enemy_ok2:
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

    # Wait for round 2
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


def extract_tournament(char, config, game_dir, start_from=None):
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

    extracted = []

    for match_idx in range(start_idx, len(TOURNAMENT)):
        prefix, match_name, is_endurance = TOURNAMENT[match_idx]

        print(f"\n--- {match_name} (Stage {match_idx + 1}/{len(TOURNAMENT)}) ---")
        health = read_health(env)
        print(f"  State: HP={health['health']} EHP={health['enemy_health']}")

        # Win the match
        success = win_match(env, verbose=True)

        if not success:
            print(f"  FAILED at {match_name}!")
            break

        # Check if this is the last match
        if match_idx >= len(TOURNAMENT) - 1:
            print(f"\n  TOURNAMENT COMPLETE!")
            break

        # Wait for next match to start
        next_prefix, next_name, _ = TOURNAMENT[match_idx + 1]
        print(f"  Waiting for {next_name}...", end=" ", flush=True)

        if wait_for_health_reset(env, max_frames=12000, no_press_frames=900):
            next_state_name = f"{next_prefix}_{char}"
            save_state(env, config, game_dir, next_state_name)
            health = read_health(env)
            print(f"SAVED: {next_state_name}.state (HP:{health['health']} EHP:{health['enemy_health']})")
            extracted.append(next_state_name)
        else:
            print("FAILED - could not detect next match!")
            print("  The game may have ended or gotten stuck on a screen.")
            break

    env.close()
    return extracted


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract MK2 tournament states via RAM cheats")
    parser.add_argument("--char", default="LiuKang", help="Character to extract for")
    parser.add_argument("--start-from", default=None,
                       help="Start from this stage (e.g., Match6, Kintaro)")
    parser.add_argument("--all-chars", action="store_true", help="Extract for all 12 characters")
    args = parser.parse_args()

    config = get_game_config("mk2")
    game_dir = ROOT_DIR / config.game_dir_name

    print("=" * 60)
    print("MK2 CHEAT EXTRACTOR")
    print("=" * 60)

    start_time = time.time()

    if args.all_chars:
        chars = CHARACTERS
    else:
        chars = [args.char]

    all_extracted = {}
    for char in chars:
        extracted = extract_tournament(char, config, game_dir, args.start_from)
        all_extracted[char] = extracted

    elapsed = time.time() - start_time

    # Summary
    print("\n" + "=" * 60)
    print("EXTRACTION SUMMARY")
    print(f"Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print("=" * 60)
    total_states = 0
    for char, states in all_extracted.items():
        if states:
            print(f"  {char}: {len(states)} new states - {', '.join(states)}")
            total_states += len(states)
        else:
            print(f"  {char}: FAILED or no new states")
    
    print(f"\nTotal: {total_states} states extracted")


if __name__ == "__main__":
    main()
