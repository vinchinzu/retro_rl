"""
Smoke tests for the PPO training pipeline.

Verifies that a model can be created, trained for a few steps,
saved, and loaded without errors. Does not test convergence.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

from retro_harness.repo import monorepo_root, resolve_game_dir

ROOT_DIR = monorepo_root()

import numpy as np
import pytest

pytestmark = [pytest.mark.ml, pytest.mark.rom]


class TestPPOSmoke(unittest.TestCase):
    """Quick smoke test: create env, create PPO, train 128 steps, save/load."""

    def _test_game(self, game_id: str, game_dir_name: str):
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv

        from retro_harness.fighters.fighting_env import FightingGameConfig, make_fighting_env
        from retro_harness.fighters.train_ppo import FighterCNN

        game_dir = resolve_game_dir(game_dir_name)
        states = list((game_dir / "custom_integrations" / game_id).glob("*.state"))
        state = states[0].stem if states else "NONE"

        config = FightingGameConfig()

        def make_env():
            return make_fighting_env(
                game=game_id, state=state, game_dir=game_dir,
                config=config, frame_skip=4, frame_stack=4,
            )

        env = DummyVecEnv([make_env])

        policy_kwargs = dict(
            features_extractor_class=FighterCNN,
            features_extractor_kwargs=dict(features_dim=64),  # Small for test
            net_arch=dict(pi=[32], vf=[32]),  # Small for test
        )

        model = PPO(
            "CnnPolicy", env,
            policy_kwargs=policy_kwargs,
            n_steps=64,
            batch_size=32,
            n_epochs=1,
            verbose=0,
            device="cpu",
        )

        # Train briefly
        model.learn(total_timesteps=128)

        # Save and reload
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test_model.zip")
            model.save(path)
            loaded = PPO.load(path, env=env, device="cpu")

            # Predict
            obs = env.reset()
            action, _ = loaded.predict(obs, deterministic=True)
            self.assertEqual(action.shape[0], 1)  # Single env

        env.close()

    def test_sf2_ppo_smoke(self):
        self._test_game("StreetFighterIITurbo-Snes", "street_fighter_ii")

    def test_mk1_ppo_smoke(self):
        self._test_game("MortalKombat-Snes", "mortal_kombat")


if __name__ == "__main__":
    unittest.main()
