"""
Tests for menu navigation and save state creation.

These tests boot each game from NONE (power-on), execute the menu
navigation sequence, and verify the game reaches a playable state.
"""

import os
import sys
import unittest
from pathlib import Path

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import stable_retro as retro

from fighters_common.game_configs import GAME_REGISTRY, get_game_config
from fighters_common.menu_nav import MenuNavigator


class TestMenuNavigation(unittest.TestCase):
    """Test that menu sequences produce valid game states."""

    def _test_nav(self, game_alias: str, min_frames: int = 60):
        config = get_game_config(game_alias)
        game_dir = ROOT_DIR / config.game_dir_name
        integrations = game_dir / "custom_integrations"

        if not integrations.exists():
            self.skipTest(f"No custom_integrations for {config.game_id}")

        retro.data.Integrations.add_custom_path(str(integrations))

        env = retro.make(
            game=config.game_id,
            state=retro.State.NONE,
            render_mode="rgb_array",
            inttype=retro.data.Integrations.CUSTOM_ONLY,
            use_restricted_actions=retro.Actions.ALL,
        )
        obs, info = env.reset()

        # Execute menu sequence
        nav = MenuNavigator(env, config.menu_sequence)
        obs, info = nav.run()

        # Verify we got valid output
        self.assertIsInstance(obs, np.ndarray)
        self.assertEqual(len(obs.shape), 3)

        # Run some noop frames to settle
        noop = np.zeros(12, dtype=np.int8)
        for _ in range(min_frames):
            obs, _, terminated, truncated, info = env.step(noop)
            if terminated or truncated:
                break

        # The game should still be running (not crashed)
        self.assertIsInstance(obs, np.ndarray)

        env.close()

    def test_sf2_menu(self):
        self._test_nav("sf2")

    def test_ssf2_menu(self):
        self._test_nav("ssf2")

    def test_mk1_menu(self):
        self._test_nav("mk1")

    def test_mk2_menu(self):
        self._test_nav("mk2")


class TestCreateFightState(unittest.TestCase):
    """Test save state creation (creates actual .state files)."""

    def _test_state_creation(self, game_alias: str):
        import tempfile
        from fighters_common.menu_nav import create_fight_state

        config = get_game_config(game_alias)
        game_dir = ROOT_DIR / config.game_dir_name

        with tempfile.TemporaryDirectory() as tmp:
            # We can't easily test with tempdir since ROM needs to be in custom_integrations
            # Instead, test that the function runs without error using the real game dir
            # and a test state name
            state_name = f"_test_{game_alias}"
            try:
                path = create_fight_state(
                    game=config.game_id,
                    game_dir=game_dir,
                    state_name=state_name,
                    menu_sequence=config.menu_sequence,
                    settle_frames=30,  # Short settle for test speed
                )
                self.assertTrue(path.exists())
                self.assertTrue(path.stat().st_size > 0)
            finally:
                # Clean up test state
                test_state = game_dir / "custom_integrations" / config.game_id / f"{state_name}.state"
                if test_state.exists():
                    test_state.unlink()

    def test_sf2_state_creation(self):
        self._test_state_creation("sf2")

    def test_mk1_state_creation(self):
        self._test_state_creation("mk1")


if __name__ == "__main__":
    unittest.main()
