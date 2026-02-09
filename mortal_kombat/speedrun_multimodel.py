#!/usr/bin/env python3
"""
Multi-Model Tournament Runner - Uses the best model per stage.

Instead of one model for all 12 stages, loads specialist models where available
and a general model for everything else. Measures per-stage win rates and
estimates the probability of clearing the entire tournament.

Usage:
    python speedrun_multimodel.py                          # Default: test each stage 10x
    python speedrun_multimodel.py --attempts 20            # More attempts per stage
    python speedrun_multimodel.py --tournament 100         # Simulate 100 full tournaments
    python speedrun_multimodel.py --general models/X.zip   # Override general model
"""

import os
import sys
import time
from pathlib import Path

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
from stable_baselines3 import PPO
from fighters_common.fighting_env import (
    DirectRAMReader, DiscreteAction, FightingEnv, FightingGameConfig,
    FrameSkip, FrameStack, GrayscaleResize
)
from fighters_common.game_configs import get_game_config
import stable_retro as retro

# Full tournament stages
STAGES = [
    ("Fight",       "Match 1"),
    ("Match2",      "Match 2"),
    ("Match3",      "Match 3"),
    ("Match4",      "Match 4"),
    ("Match5",      "Match 5"),
    ("Match6",      "Match 6"),
    ("Match7",      "Mirror Match"),
    ("Endurance1",  "Endurance 1 (opp 1)"),
    ("Endurance1B", "Endurance 1 (opp 2)"),
    ("Endurance2",  "Endurance 2 (opp 1)"),
    ("Goro",        "Goro (=E2 opp 2)"),
    ("ShangTsung",  "Shang Tsung"),
]

# Per-stage model overrides. Maps stage prefix -> model filename.
# Stages not listed here use the general model.
STAGE_MODELS = {
    "ShangTsung":  "mk1_shangtsung_ppo_final.zip",
    "Goro":        "mk1_goro_ppo_final.zip",
}


def build_env(config, game_dir, state):
    retro.data.Integrations.add_custom_path(str(game_dir / "custom_integrations"))
    base_env = retro.make(
        game=config.game_id,
        state=state,
        render_mode="rgb_array",
        inttype=retro.data.Integrations.CUSTOM_ONLY,
        use_restricted_actions=retro.Actions.ALL,
    )
    fight_config = FightingGameConfig(
        max_health=config.max_health,
        health_key=config.health_key,
        enemy_health_key=config.enemy_health_key,
        ram_overrides=config.ram_overrides,
        actions=config.actions,
    )
    env = base_env
    if config.ram_overrides:
        env = DirectRAMReader(env, config.ram_overrides)
    env = FrameSkip(env, n_skip=4)
    env = GrayscaleResize(env, width=84, height=84)
    env = FightingEnv(env, fight_config)
    env = DiscreteAction(env, config.actions)
    env = FrameStack(env, n_frames=4)
    return env


def play_match(model, config, game_dir, state_name):
    """Play a single match. Returns (won: bool, frames: int)."""
    env = build_env(config, game_dir, state_name)
    obs, info = env.reset()
    frames = 0
    won = False

    for _ in range(15000):
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(action)
        frames += 1

        if terminated or truncated:
            rw = info.get("rounds_won", 0)
            rl = info.get("rounds_lost", 0)
            won = rw >= 2 and rw > rl
            break

    env.close()
    return won, frames


def resolve_models(model_dir, general_path):
    """Resolve per-stage model paths. Returns dict of stage_prefix -> model_path."""
    stage_model_paths = {}
    for prefix, _ in STAGES:
        if prefix in STAGE_MODELS:
            candidate = model_dir / STAGE_MODELS[prefix]
            if candidate.exists():
                stage_model_paths[prefix] = candidate
            else:
                print(f"  WARNING: Specialist {STAGE_MODELS[prefix]} not found, using general")
                stage_model_paths[prefix] = general_path
        else:
            stage_model_paths[prefix] = general_path
    return stage_model_paths


def load_models(stage_model_paths, device):
    """Load models, caching by path so shared models are only loaded once."""
    cache = {}
    stage_models = {}
    for prefix, path in stage_model_paths.items():
        path_str = str(path)
        if path_str not in cache:
            print(f"  Loading {path.name}...")
            cache[path_str] = PPO.load(path_str, device=device)
        stage_models[prefix] = cache[path_str]
    return stage_models


def test_per_stage(stage_models, stage_model_paths, config, game_dir, char, attempts):
    """Test each stage independently. Returns list of (prefix, display_name, win_rate, wins, losses, avg_frames)."""
    state_dir = game_dir / "custom_integrations" / config.game_id
    results = []

    print(f"\n{'Stage':<25} {'Model':<35} {'Win%':>6} {'W':>4} {'L':>4} {'Avg Fr':>7}")
    print("-" * 86)

    for prefix, display_name in STAGES:
        state_name = f"{prefix}_{char}"
        if not (state_dir / f"{state_name}.state").exists():
            print(f"{display_name:<25} {'':35} {'SKIP':>6}  (no state)")
            results.append((prefix, display_name, 0, 0, attempts, 0))
            continue

        model = stage_models[prefix]

        wins = 0
        losses = 0
        frame_counts = []

        for _ in range(attempts):
            won, frames = play_match(model, config, game_dir, state_name)
            if won:
                wins += 1
            else:
                losses += 1
            frame_counts.append(frames)

        win_rate = wins / max(1, wins + losses)
        avg_frames = np.mean(frame_counts) if frame_counts else 0
        results.append((prefix, display_name, win_rate, wins, losses, avg_frames))

        marker = " ***" if win_rate == 0 else ""
        model_label = stage_model_paths[prefix].name
        print(f"{display_name:<25} {model_label:<35} {win_rate:>5.0%} {wins:>4} {losses:>4} {avg_frames:>7.0f}{marker}")

    return results


def simulate_tournaments(stage_models, config, game_dir, char, n_tournaments):
    """Simulate N full tournament runs, chaining through all stages."""
    state_dir = game_dir / "custom_integrations" / config.game_id
    clears = 0
    furthest_counts = {prefix: 0 for prefix, _ in STAGES}
    stage_wins_total = {prefix: 0 for prefix, _ in STAGES}
    stage_attempts_total = {prefix: 0 for prefix, _ in STAGES}

    print(f"\nSimulating {n_tournaments} tournament runs...")

    for t in range(n_tournaments):
        cleared_all = True
        for i, (prefix, display_name) in enumerate(STAGES):
            state_name = f"{prefix}_{char}"
            if not (state_dir / f"{state_name}.state").exists():
                cleared_all = False
                break

            model = stage_models[prefix]
            won, frames = play_match(model, config, game_dir, state_name)
            stage_attempts_total[prefix] += 1

            if won:
                stage_wins_total[prefix] += 1
                furthest_counts[prefix] += 1
            else:
                cleared_all = False
                break

        if cleared_all:
            clears += 1

        # Progress update every 10 runs
        if (t + 1) % 10 == 0:
            print(f"  Run {t+1}/{n_tournaments}: {clears} clears so far", flush=True)

    # Results
    print(f"\n{'='*70}")
    print(f"TOURNAMENT SIMULATION: {n_tournaments} runs")
    print(f"{'='*70}")
    print(f"\n{'Stage':<25} {'Win%':>6}  {'W':>5}/{' Att':<5}  {'Reached':>8}")
    print("-" * 60)

    for prefix, display_name in STAGES:
        att = stage_attempts_total[prefix]
        w = stage_wins_total[prefix]
        reached = furthest_counts[prefix]
        wr = w / max(1, att)
        print(f"{display_name:<25} {wr:>5.0%}  {w:>5}/{att:<5}  {reached:>8}")

    print("-" * 60)
    print(f"\nFull clears: {clears}/{n_tournaments} ({clears/max(1,n_tournaments):.1%})")

    if clears > 0:
        print(f"Average attempts per clear: {n_tournaments/clears:.0f}")

    return clears


def main():
    import argparse
    parser = argparse.ArgumentParser(description="MK1 Multi-Model Tournament Runner")
    parser.add_argument("--char", default="LiuKang", help="Character to test")
    parser.add_argument("--attempts", type=int, default=10, help="Attempts per stage (for --per-stage mode)")
    parser.add_argument("--tournament", type=int, default=0, help="Number of tournament simulations (0=per-stage only)")
    parser.add_argument("--general", default=None, help="Override general model path")
    args = parser.parse_args()

    config = get_game_config("mk1")
    game_dir = ROOT_DIR / config.game_dir_name
    model_dir = game_dir / "models"

    # Find general model
    if args.general:
        general_path = Path(args.general)
        if not general_path.is_absolute():
            general_path = model_dir / args.general
    else:
        for candidate in [
            "mk1_fresh_ppo_final.zip",
            "mk1_match7_ppo_16000000_steps.zip",
            "mk1_match4_ppo_final.zip",
            "mk1_multichar_ppo_2000000_steps.zip",
        ]:
            general_path = model_dir / candidate
            if general_path.exists():
                break

    if not general_path.exists():
        print(f"ERROR: General model not found: {general_path}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve and load models
    print(f"Character: {args.char}")
    print(f"General model: {general_path.name}")
    print(f"Specialists: {dict(STAGE_MODELS)}")
    print(f"\nLoading models...")

    stage_model_paths = resolve_models(model_dir, general_path)
    stage_models = load_models(stage_model_paths, device)

    # Per-stage testing
    results = test_per_stage(stage_models, stage_model_paths, config, game_dir, args.char, args.attempts)

    # Summary
    print("-" * 86)
    total_wins = sum(r[3] for r in results)
    total_losses = sum(r[4] for r in results)
    overall_wr = total_wins / max(1, total_wins + total_losses)
    print(f"{'OVERALL':<25} {'':35} {overall_wr:>5.0%} {total_wins:>4} {total_losses:>4}")

    # Chain probability from measured win rates
    active_results = [r for r in results if r[3] + r[4] > 0]
    if all(r[2] > 0 for r in active_results):
        chain_prob = 1.0
        for r in active_results:
            chain_prob *= r[2]
        print(f"\nEstimated full-clear probability: {chain_prob:.2%} (~1 in {1/chain_prob:.0f})")
    else:
        zero_stages = [r[1] for r in active_results if r[2] == 0]
        print(f"\nFull clear impossible - 0% stages: {', '.join(zero_stages)}")

    # Tournament simulation
    if args.tournament > 0:
        simulate_tournaments(stage_models, config, game_dir, args.char, args.tournament)


if __name__ == "__main__":
    main()
