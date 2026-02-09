#!/usr/bin/env python3
"""
Create fight-ready save states for all fighting games.

Boots each game from power-on, navigates menus, and saves a .state file
at the start of a fight. These states are used for PPO training.

Usage:
    python create_all_states.py          # Create states for all games
    python create_all_states.py --game sf2   # Single game only
"""

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.resolve()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fighters_common.game_configs import GAME_REGISTRY, get_game_config
from fighters_common.menu_nav import create_fight_state


def create_states(game_filter=None):
    """Create fight states for all (or one) game."""
    # Deduplicate (aliases point to same config)
    seen = set()
    configs = []
    for key, config in GAME_REGISTRY.items():
        if config.game_id not in seen:
            seen.add(config.game_id)
            configs.append(config)

    if game_filter:
        try:
            config = get_game_config(game_filter)
            configs = [config]
        except KeyError as e:
            print(e)
            return

    for config in configs:
        game_dir = ROOT_DIR / config.game_dir_name
        state_name = f"Fight_{config.game_id.split('-')[0]}"

        print(f"\n{'='*60}")
        print(f"Creating state for: {config.display_name}")
        print(f"  Game ID: {config.game_id}")
        print(f"  State name: {state_name}")
        print(f"{'='*60}")

        try:
            path = create_fight_state(
                game=config.game_id,
                game_dir=game_dir,
                state_name=state_name,
                menu_sequence=config.menu_sequence,
                settle_frames=config.menu_settle_frames,
            )
            print(f"  -> Saved: {path}")
        except Exception as e:
            print(f"  -> FAILED: {e}")
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Create fight-ready save states")
    parser.add_argument("--game", default=None, help="Game alias (sf2, ssf2, mk1, mk2) or full ID")
    args = parser.parse_args()
    create_states(args.game)


if __name__ == "__main__":
    main()
