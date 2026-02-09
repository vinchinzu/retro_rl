#!/usr/bin/env python3
"""Unified fighting game play script.

Supports all configured fighting games via --game flag or auto-detection
from the symlink directory name. Uses PlaySession for the main loop.

Usage:
    python run_bot.py play --game mk1 --state Fight_LiuKang
    python run_bot.py play --game sf2 --state Start
    python run_bot.py create-state --game sf2 --state-name Fight_Ryu_vs_CPU
"""

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Game directory mapping for auto-detection from symlink location
_DIR_TO_ALIAS = {
    "mortal_kombat": "mk1",
    "mortal_kombat_ii": "mk2",
    "street_fighter_ii": "sf2",
    "super_street_fighter_ii": "ssf2",
}


def _detect_game_dir() -> tuple[str | None, Path]:
    """Auto-detect game alias and game_dir from script location.

    If run_bot.py is symlinked from a per-game directory, use that directory
    to determine the game. Returns (alias, game_dir).
    """
    real = Path(__file__).resolve()
    caller = Path(sys.argv[0]).resolve().parent

    # If the caller dir is different from the real file's dir,
    # we're being run from a symlink in a game directory
    if caller != real.parent:
        alias = _DIR_TO_ALIAS.get(caller.name)
        return alias, caller

    return None, SCRIPT_DIR


def _resolve_game(args_game: str | None) -> tuple[str, Path]:
    """Resolve game ID and game directory."""
    from fighters_common.game_configs import get_game_config

    auto_alias, auto_dir = _detect_game_dir()
    game_alias = args_game or auto_alias

    if not game_alias:
        print("Error: --game required (or run from a per-game directory symlink)")
        print("Available: mk1, mk2, sf2, ssf2")
        sys.exit(1)

    config = get_game_config(game_alias)
    game_id = config.game_id

    # If auto-detected from symlink, use that dir; otherwise find the game dir
    if auto_alias and not args_game:
        game_dir = auto_dir
    else:
        # Map alias to expected directory name
        _ALIAS_TO_DIR = {"mk1": "mortal_kombat", "mk2": "mortal_kombat_ii",
                         "sf2": "street_fighter_ii", "ssf2": "super_street_fighter_ii"}
        dirname = _ALIAS_TO_DIR.get(game_alias.lower(), game_alias)
        game_dir = ROOT_DIR / dirname
        if not game_dir.exists():
            game_dir = SCRIPT_DIR

    return game_id, game_dir


def play(args):
    """Interactive play mode."""
    from retro_harness import add_custom_integrations, make_env, get_available_states

    game_id, game_dir = _resolve_game(args.game)
    add_custom_integrations(game_dir)

    states = get_available_states(game_id, game_dir)
    state = args.state
    if state not in states and states:
        print(f"State '{state}' not found. Available: {states[:10]}...")
        state = states[0]
        print(f"Using: {state}")

    env = make_env(game=game_id, state=state, game_dir=game_dir, render_mode="rgb_array")

    from retro_harness.play_session import PlaySession

    def hud_hook(info):
        health = info.get("health", "?")
        enemy = info.get("enemy_health", "?")
        return [f"P1: {health}  P2: {enemy}"]

    session = PlaySession(
        env,
        game_dir=str(game_dir),
        game=game_id,
        title=f"{game_id} - {state}",
    )
    session.on_hud = hud_hook
    session.run()


def create_state(args):
    """Navigate menus and create a fight-ready save state."""
    from fighters_common import create_fight_state
    from fighters_common.game_configs import get_game_config

    game_id, game_dir = _resolve_game(args.game)
    config = get_game_config(game_id)
    create_fight_state(
        game=game_id,
        game_dir=game_dir,
        state_name=args.state_name,
        menu_sequence=config.menu_sequence,
        settle_frames=config.menu_settle_frames,
    )


def main():
    parser = argparse.ArgumentParser(description="Fighting game interactive play")
    parser.add_argument("--game", "-g", help="Game alias (mk1, mk2, sf2, ssf2)")
    sub = parser.add_subparsers(dest="command")

    play_p = sub.add_parser("play", help="Interactive play")
    play_p.add_argument("--state", default="Start", help="State to load")

    state_p = sub.add_parser("create-state", help="Create fight-ready save state")
    state_p.add_argument("--state-name", default="Fight_vs_CPU", help="Name for saved state")

    args = parser.parse_args()
    if args.command == "play":
        play(args)
    elif args.command == "create-state":
        create_state(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
