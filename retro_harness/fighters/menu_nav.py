"""
Menu navigation utilities for fighting games.

Automates getting from title screen / power-on to an active fight,
then saves the resulting state for training.
"""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Optional

import numpy as np
import stable_retro as retro


class MenuNavigator:
    """Execute a sequence of button presses to navigate game menus."""

    def __init__(self, env: retro.RetroEnv, sequence: list[tuple[int, int]]):
        """
        Args:
            env: Active RetroEnv (booted from NONE or title state)
            sequence: List of (button_index, hold_frames).
                      button_index=-1 means wait with no input.
        """
        self.env = env
        self.sequence = sequence

    def run(self) -> tuple[np.ndarray, dict]:
        """Execute the full menu navigation sequence.

        Returns:
            (observation, info) after navigation is complete
        """
        obs, info = None, {}
        for button_idx, hold_frames in self.sequence:
            action = np.zeros(12, dtype=np.int8)
            if button_idx >= 0:
                action[button_idx] = 1
            for _ in range(hold_frames):
                obs, _, terminated, truncated, info = self.env.step(action)
                if terminated or truncated:
                    obs, info = self.env.reset()
        return obs, info


def navigate_to_fight(
    game: str,
    game_dir: str | Path,
    menu_sequence: list[tuple[int, int]],
    settle_frames: int = 120,
) -> retro.RetroEnv:
    """
    Boot a game from power-on and navigate menus to reach a fight.

    Args:
        game: stable-retro game ID
        game_dir: Path to game directory with custom_integrations/
        menu_sequence: Button sequence to navigate menus
        settle_frames: Extra frames to wait after menu nav

    Returns:
        Active RetroEnv positioned at the start of a fight
    """
    game_dir = Path(game_dir).resolve()
    integrations_path = game_dir / "custom_integrations"
    if integrations_path.exists():
        retro.data.Integrations.add_custom_path(str(integrations_path))

    env = retro.make(
        game=game,
        state=retro.State.NONE,
        render_mode="rgb_array",
        inttype=retro.data.Integrations.CUSTOM_ONLY,
        use_restricted_actions=retro.Actions.ALL,
    )
    obs, info = env.reset()

    # Navigate menus
    nav = MenuNavigator(env, menu_sequence)
    obs, info = nav.run()

    # Let the fight settle
    noop = np.zeros(12, dtype=np.int8)
    for _ in range(settle_frames):
        obs, _, _, _, info = env.step(noop)

    return env


def create_fight_state(
    game: str,
    game_dir: str | Path,
    state_name: str,
    menu_sequence: list[tuple[int, int]],
    settle_frames: int = 120,
) -> Path:
    """
    Navigate menus and save a fight-ready state file.

    Args:
        game: stable-retro game ID
        game_dir: Path to game directory
        state_name: Name for the saved state (without .state extension)
        menu_sequence: Button sequence
        settle_frames: Frames to wait after menu nav

    Returns:
        Path to saved .state file in custom_integrations/
    """
    env = navigate_to_fight(game, game_dir, menu_sequence, settle_frames)

    state_data = env.em.get_state()
    game_dir = Path(game_dir).resolve()
    save_path = game_dir / "custom_integrations" / game / f"{state_name}.state"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(save_path, "wb") as f:
        f.write(state_data)

    env.close()
    print(f"Saved fight state: {save_path}")
    return save_path
