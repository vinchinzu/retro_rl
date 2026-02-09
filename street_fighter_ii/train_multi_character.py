#!/usr/bin/env python3
"""
Train a generic fighter using all 12 World Warriors.

This creates a model that learns universal fighting strategies by
training on different characters, making it more robust and generalizable.

Phase 1: Character rotation (this script)
  - Randomly select from 12 character starting states
  - Learn fundamental fighting skills
  - Target: 2-3M steps

Phase 2: Add opponent variety (use train_multi_opponent.py after this)
  - Mix character states + opponent progression states
  - Learn to handle different matchups
  - Target: 1M more steps

Phase 3: Fine-tuning (optional, manual)
  - Focus on difficult specific matchups
  - Target: 500k steps
"""

import sys
import random
from pathlib import Path

# Add parent directory to path
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

from fighters_common import train_ppo
from fighters_common.fighting_env import make_fighting_env
from stable_baselines3.common.callbacks import CheckpointCallback

# All 12 World Warriors character starting states
CHARACTER_STATES = [
    "Fight_Ryu",
    "Fight_EHonda",
    "Fight_Blanka",
    "Fight_Guile",
    "Fight_Ken",
    "Fight_ChunLi",
    "Fight_Zangief",
    "Fight_Dhalsim",
    "Fight_Balrog",
    "Fight_Vega",
    "Fight_Sagat",
    "Fight_MBison",
]

print("="*60)
print("SF2 TURBO MULTI-CHARACTER TRAINING - PHASE 1")
print("="*60)
print(f"\nTraining with {len(CHARACTER_STATES)} World Warriors:")
for i, state in enumerate(CHARACTER_STATES, 1):
    char_name = state.replace("Fight_", "")
    print(f"  {i:2d}. {char_name}")
print("\nEach episode will randomly select a character.")
print("This teaches universal fighting skills!")
print("="*60 + "\n")

# Monkey-patch the env creation to use random character states
_original_make_env = make_fighting_env

def random_character_make_env(*args, **kwargs):
    """Wrapper that randomly selects character state each episode."""
    # Pick random character for this environment instance
    state = random.choice(CHARACTER_STATES)
    kwargs['state'] = state

    # Log which character was selected (useful for debugging)
    if random.random() < 0.01:  # Log ~1% of selections
        char_name = state.replace("Fight_", "")
        print(f"  [Env] Selected character: {char_name}")

    return _original_make_env(*args, **kwargs)

# Replace the function
train_ppo.make_fighting_env = random_character_make_env

# Patch the training function to use custom model names
_original_train = train_ppo.train

def patched_train(args):
    """Patched train that uses multichar prefix for model names."""
    # Call original training
    _original_train(args)

    # The original saves as "{game}_ppo_final.zip"
    # We want to rename it to "{game}_multichar_ppo_final.zip"
    from fighters_common.game_configs import get_game_config
    game_config = get_game_config(args.game)
    game_dir = ROOT_DIR / game_config.game_dir_name
    model_dir = game_dir / "models"

    old_path = model_dir / f"{args.game}_ppo_final.zip"
    new_path = model_dir / f"{args.game}_multichar_ppo_final.zip"

    if old_path.exists():
        import shutil
        shutil.move(str(old_path), str(new_path))
        print(f"\nRenamed model: {new_path.name}")

        # Also rename the latest checkpoint
        import glob
        checkpoints = sorted(glob.glob(str(model_dir / f"{args.game}_ppo_*_steps.zip")))
        if checkpoints:
            latest = checkpoints[-1]
            new_checkpoint = latest.replace(f"{args.game}_ppo_", f"{args.game}_multichar_ppo_")
            shutil.copy(latest, new_checkpoint)
            print(f"Copied checkpoint: {Path(new_checkpoint).name}")

train_ppo.train = patched_train

if __name__ == '__main__':
    # Override default args to start fresh
    import sys

    # Add recommended training args if not provided
    if '--steps' not in sys.argv:
        sys.argv.extend(['--steps', '2000000'])  # 2M steps for Phase 1

    if '--game' not in sys.argv:
        sys.argv.extend(['--game', 'sf2'])

    # Provide a default state (will be randomized by monkey-patch per env)
    if '--state' not in sys.argv:
        sys.argv.extend(['--state', 'Fight_Ryu'])

    # Ensure we're NOT loading an existing model (start fresh)
    if '--load' in sys.argv:
        print("\n⚠️  WARNING: --load flag detected!")
        print("This training run should START FRESH for best results.")
        print("Remove --load to train from scratch.")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)

    print("\n🎮 Starting fresh multi-character training...")
    print("Model will be saved to: models/sf2_multichar_ppo_*.zip\n")

    train_ppo.main()
