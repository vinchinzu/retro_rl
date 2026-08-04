"""
Wrapper to randomize opponent matchups during training.
Each episode resets to a random opponent state.
"""

import random
from pathlib import Path
import gymnasium as gym
import stable_retro as retro


class RandomOpponentWrapper(gym.Wrapper):
    """
    Wrapper that randomizes which opponent state is loaded on each reset.

    Args:
        env: Base retro environment (must be at episode start)
        state_names: List of state names to randomly choose from
        game_dir: Path to game directory with custom_integrations/
    """

    def __init__(self, env, state_names: list[str], game_dir: str | Path):
        super().__init__(env)
        self.state_names = state_names
        self.game_dir = Path(game_dir)
        self.current_state = None

        # Verify all states exist
        game_id = env.gamename
        states_dir = self.game_dir / "custom_integrations" / game_id
        for state_name in state_names:
            state_path = states_dir / f"{state_name}.state"
            if not state_path.exists():
                raise FileNotFoundError(f"State not found: {state_path}")

        print(f"RandomOpponentWrapper initialized with {len(state_names)} states:")
        for state_name in state_names:
            print(f"  - {state_name}")

    def reset(self, **kwargs):
        """Reset to a random opponent state."""
        # Pick a random state
        self.current_state = random.choice(self.state_names)

        # Load the state
        self.unwrapped.load_state(self.current_state)

        # Call parent reset
        obs, info = self.env.reset(**kwargs)

        # Add state info
        info["opponent_state"] = self.current_state

        return obs, info


def make_random_opponent_env(base_env_fn, state_names: list[str], game_dir: str | Path):
    """
    Factory function to create an environment with random opponent selection.

    Args:
        base_env_fn: Function that creates the base environment
        state_names: List of state names to randomly choose from
        game_dir: Path to game directory

    Returns:
        Environment wrapped with RandomOpponentWrapper
    """
    env = base_env_fn()
    return RandomOpponentWrapper(env, state_names, game_dir)
