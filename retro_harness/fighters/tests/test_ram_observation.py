"""Tests for RAM observation vector building and wrapper shape."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

from retro_harness.repo import monorepo_root, resolve_game_dir

ROOT_DIR = monorepo_root()

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from retro_harness.fighters.ram_observation import (
    MK1_RAM_FEATURES,
    RamObservation,
    build_ram_features,
)


class TestBuildRamFeatures(unittest.TestCase):
    def test_distance_feature(self):
        info = {"p1_x": 120, "p2_x": 211}
        vector, _ = build_ram_features(MK1_RAM_FEATURES, info, {})
        dist_idx = len(MK1_RAM_FEATURES) - 1
        self.assertAlmostEqual(vector[dist_idx], 91 / 255.0, places=4)

    def test_normalizes_health_and_timer(self):
        info = {
            "health": 161,
            "enemy_health": 80,
            "timer": 77,
            "p2_character": 3,
            "p1_rounds": 1,
            "p2_rounds": 0,
            "match_counter": 5,
            "p1_x": 120,
            "p2_x": 211,
            "p1_y": 130,
        }
        vector, _ = build_ram_features(MK1_RAM_FEATURES, info, {})
        self.assertEqual(vector.shape, (len(MK1_RAM_FEATURES),))
        self.assertAlmostEqual(vector[0], 1.0)
        self.assertAlmostEqual(vector[1], 80 / 161.0)
        self.assertAlmostEqual(vector[4], 77 / 154.0)
        self.assertAlmostEqual(vector[8], 5 / 11.0)

    def test_health_delta_uses_previous_values(self):
        info = {"health": 140, "enemy_health": 120}
        _, prev = build_ram_features(MK1_RAM_FEATURES, info, {})
        info2 = {"health": 120, "enemy_health": 100}
        vector, _ = build_ram_features(MK1_RAM_FEATURES, info2, prev)
        self.assertAlmostEqual(vector[2], -20 / 161.0, places=4)
        self.assertAlmostEqual(vector[3], -20 / 161.0, places=4)


class TestRamObservationWrapper(unittest.TestCase):
    def _make_base_env(self):
        base = MagicMock(spec=gym.Env)
        base.observation_space = spaces.Box(
            low=0, high=255, shape=(240, 320, 3), dtype=np.uint8
        )
        base.action_space = spaces.MultiBinary(12)
        base.reset.return_value = (
            np.zeros((240, 320, 3), dtype=np.uint8),
            {
                "health": 161,
                "enemy_health": 161,
                "timer": 100,
                "p2_character": 1,
                "p1_rounds": 0,
                "p2_rounds": 0,
                "match_counter": 0,
                "p1_x": 120,
                "p2_x": 211,
                "p1_y": 130,
            },
        )
        base.step.return_value = (
            np.zeros((240, 320, 3), dtype=np.uint8),
            0.0,
            False,
            False,
            {
                "health": 150,
                "enemy_health": 140,
                "timer": 99,
                "p2_character": 1,
                "p1_rounds": 0,
                "p2_rounds": 0,
                "match_counter": 0,
                "p1_x": 180,
                "p2_x": 211,
                "p1_y": 130,
            },
        )
        return base

    def test_observation_space_is_float_vector(self):
        env = RamObservation(self._make_base_env())
        self.assertEqual(env.observation_space.shape, (len(MK1_RAM_FEATURES),))
        self.assertEqual(env.observation_space.dtype, np.float32)

    def test_reset_and_step_return_ram_vector(self):
        env = RamObservation(self._make_base_env())
        obs, info = env.reset()
        self.assertEqual(obs.shape, (len(MK1_RAM_FEATURES),))
        self.assertIn("health", info)

        obs2, reward, terminated, truncated, info2 = env.step(np.zeros(12, dtype=np.int8))
        self.assertEqual(obs2.shape, (len(MK1_RAM_FEATURES),))
        self.assertLess(obs2[0], obs[0])


if __name__ == "__main__":
    unittest.main()
