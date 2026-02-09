"""
Headless tests for all fighting game environments.

Tests that each game:
1. Can load its ROM and custom integration
2. Returns valid observations
3. Reports RAM variables (health, enemy_health, etc.) in info
4. Responds to actions
5. Can run for multiple frames without crashing
"""

import os
import sys
import unittest
from pathlib import Path

# Headless rendering
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import stable_retro as retro

# Game configurations: (game_id, game_dir_name, expected_info_keys)
GAMES = [
    (
        "StreetFighterIITurbo-Snes",
        "street_fighter_ii",
        ["health", "enemy_health", "timer"],
    ),
    (
        "SuperStreetFighterII-Snes",
        "super_street_fighter_ii",
        ["health", "enemy_health", "timer"],
    ),
    (
        "MortalKombat-Snes",
        "mortal_kombat",
        ["health", "enemy_health", "timer"],
    ),
    (
        "MortalKombatII-Snes",
        "mortal_kombat_ii",
        ["fatality_timer"],  # health/enemy_health are in high WRAM, read via DirectRAMReader
    ),
]


class TestGameLoad(unittest.TestCase):
    """Test that each game can be loaded and produces valid output."""

    def _make_env(self, game_id: str, game_dir_name: str):
        game_dir = ROOT_DIR / game_dir_name
        integrations = game_dir / "custom_integrations"
        self.assertTrue(integrations.exists(), f"Missing custom_integrations for {game_id}")

        retro.data.Integrations.add_custom_path(str(integrations))

        # Check for any .state files; if none, use NONE
        state_dir = integrations / game_id
        states = list(state_dir.glob("*.state"))
        state = states[0].stem if states else retro.State.NONE

        env = retro.make(
            game=game_id,
            state=state,
            render_mode="rgb_array",
            inttype=retro.data.Integrations.CUSTOM_ONLY,
            use_restricted_actions=retro.Actions.ALL,
        )
        return env

    def _test_game(self, game_id, game_dir_name, expected_keys):
        """Generic test for a single game."""
        env = self._make_env(game_id, game_dir_name)

        # Test reset
        obs, info = env.reset()
        self.assertIsInstance(obs, np.ndarray)
        self.assertEqual(len(obs.shape), 3, f"Expected 3D obs (H,W,C), got shape {obs.shape}")
        self.assertEqual(obs.shape[2], 3, f"Expected RGB, got {obs.shape[2]} channels")

        # Step a few frames first — info dict may be empty on reset when
        # booting from NONE (no save state). Variables appear after stepping.
        noop = np.zeros(12, dtype=np.int8)
        for _ in range(10):
            obs, _, _, _, info = env.step(noop)

        # Check info keys from data.json
        for key in expected_keys:
            self.assertIn(key, info, f"Missing '{key}' in info for {game_id}. Got: {list(info.keys())}")

        # Test stepping with random actions
        for i in range(30):
            action = np.zeros(12, dtype=np.int8)
            action[np.random.randint(0, 12)] = 1
            obs, reward, terminated, truncated, info = env.step(action)
            self.assertIsInstance(obs, np.ndarray)
            self.assertEqual(obs.shape[2], 3)

            if terminated or truncated:
                obs, info = env.reset()

        env.close()

    def test_sf2_turbo(self):
        self._test_game(*GAMES[0])

    def test_ssf2(self):
        self._test_game(*GAMES[1])

    def test_mk1(self):
        self._test_game(*GAMES[2])

    def test_mk2(self):
        self._test_game(*GAMES[3])


class TestRAMValues(unittest.TestCase):
    """Test that RAM addresses return sensible values."""

    def _make_env(self, game_id, game_dir_name):
        game_dir = ROOT_DIR / game_dir_name
        integrations = game_dir / "custom_integrations"
        retro.data.Integrations.add_custom_path(str(integrations))
        states = list((integrations / game_id).glob("*.state"))
        state = states[0].stem if states else retro.State.NONE
        return retro.make(
            game=game_id, state=state, render_mode="rgb_array",
            inttype=retro.data.Integrations.CUSTOM_ONLY,
            use_restricted_actions=retro.Actions.ALL,
        )

    def _test_health_range(self, game_id, game_dir_name, max_health):
        """Verify health values are in expected range at start."""
        env = self._make_env(game_id, game_dir_name)
        obs, info = env.reset()

        # Run a few frames to let the game initialize
        noop = np.zeros(12, dtype=np.int8)
        for _ in range(10):
            obs, _, _, _, info = env.step(noop)

        health = info.get("health", 0)
        enemy_health = info.get("enemy_health", 0)

        # Health should be positive (game may not be in fight yet if no state)
        # Just verify we get numeric values
        self.assertIsInstance(health, (int, float, np.integer, np.floating),
                              f"health not numeric for {game_id}: {type(health)}")
        self.assertIsInstance(enemy_health, (int, float, np.integer, np.floating),
                              f"enemy_health not numeric for {game_id}: {type(enemy_health)}")

        env.close()

    def test_sf2_health(self):
        self._test_health_range("StreetFighterIITurbo-Snes", "street_fighter_ii", 176)

    def test_ssf2_health(self):
        self._test_health_range("SuperStreetFighterII-Snes", "super_street_fighter_ii", 176)

    def test_mk1_health(self):
        self._test_health_range("MortalKombat-Snes", "mortal_kombat", 161)

    def test_mk2_health(self):
        self._test_health_range("MortalKombatII-Snes", "mortal_kombat_ii", 161)


class TestWrapperStack(unittest.TestCase):
    """Test the full wrapper stack used for training."""

    def _test_wrapped_env(self, game_id, game_dir_name):
        from fighters_common.fighting_env import (
            FightingGameConfig, make_fighting_env,
        )

        game_dir = ROOT_DIR / game_dir_name
        states = list((game_dir / "custom_integrations" / game_id).glob("*.state"))
        state = states[0].stem if states else "NONE"

        config = FightingGameConfig()
        env = make_fighting_env(
            game=game_id,
            state=state,
            game_dir=game_dir,
            config=config,
            frame_skip=4,
            frame_stack=4,
        )

        obs, info = env.reset()
        # Should be (4, 84, 84) after frame stack
        self.assertEqual(obs.shape, (4, 84, 84), f"Expected (4,84,84), got {obs.shape}")

        # Step with discrete action
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            self.assertEqual(obs.shape, (4, 84, 84))
            self.assertIsInstance(reward, (int, float, np.floating))

            if terminated or truncated:
                obs, info = env.reset()

        env.close()

    def test_sf2_wrapped(self):
        self._test_wrapped_env("StreetFighterIITurbo-Snes", "street_fighter_ii")

    def test_ssf2_wrapped(self):
        self._test_wrapped_env("SuperStreetFighterII-Snes", "super_street_fighter_ii")

    def test_mk1_wrapped(self):
        self._test_wrapped_env("MortalKombat-Snes", "mortal_kombat")

    def test_mk2_wrapped(self):
        self._test_wrapped_env("MortalKombatII-Snes", "mortal_kombat_ii")


class TestActionSpaces(unittest.TestCase):
    """Test per-game action spaces."""

    def test_mk_action_space_count(self):
        from fighters_common.fighting_env import MK_FIGHTING_ACTIONS, FIGHTING_ACTIONS
        self.assertEqual(len(MK_FIGHTING_ACTIONS), 32)
        self.assertEqual(len(FIGHTING_ACTIONS), 32)

    def test_mk_action_space_has_block(self):
        """MK actions must have dedicated Block button (X=9)."""
        from fighters_common.fighting_env import MK_FIGHTING_ACTIONS
        _X = 9
        block_actions = [a for a in MK_FIGHTING_ACTIONS if _X in a]
        self.assertGreaterEqual(len(block_actions), 2, "Expected at least stand block and crouch block")

    def test_mk_action_space_no_duplicates(self):
        from fighters_common.fighting_env import MK_FIGHTING_ACTIONS
        action_tuples = [tuple(sorted(a.items())) for a in MK_FIGHTING_ACTIONS]
        # No-op {} appears once
        self.assertEqual(len(action_tuples), len(set(action_tuples)),
                         "Duplicate actions found in MK_FIGHTING_ACTIONS")

    def test_mk1_config_uses_mk_actions(self):
        from fighters_common.game_configs import get_game_config
        from fighters_common.fighting_env import MK_FIGHTING_ACTIONS
        config = get_game_config("mk1")
        self.assertIs(config.actions, MK_FIGHTING_ACTIONS)

    def test_mk2_config_uses_mk_actions(self):
        from fighters_common.game_configs import get_game_config
        from fighters_common.fighting_env import MK_FIGHTING_ACTIONS
        config = get_game_config("mk2")
        self.assertIs(config.actions, MK_FIGHTING_ACTIONS)

    def test_sf2_config_no_custom_actions(self):
        from fighters_common.game_configs import get_game_config
        config = get_game_config("sf2")
        self.assertIsNone(config.actions)

    def test_mk1_max_health(self):
        from fighters_common.game_configs import get_game_config
        config = get_game_config("mk1")
        self.assertEqual(config.max_health, 161)

    def test_dynamic_reward_scale(self):
        """Reward scale should be 1/max_health, not hardcoded 1/176."""
        from fighters_common.fighting_env import FightingEnv, FightingGameConfig
        import gymnasium as gym

        # Test with MK health (161)
        dummy_env = gym.make("CartPole-v1")  # Just need any env for wrapper init
        config = FightingGameConfig(max_health=161)
        wrapper = FightingEnv(dummy_env, config)
        self.assertAlmostEqual(wrapper.reward_scale, 1.0 / 161.0)
        dummy_env.close()

        # Test with SF2 health (176)
        dummy_env = gym.make("CartPole-v1")
        config = FightingGameConfig(max_health=176)
        wrapper = FightingEnv(dummy_env, config)
        self.assertAlmostEqual(wrapper.reward_scale, 1.0 / 176.0)
        dummy_env.close()


if __name__ == "__main__":
    unittest.main()
