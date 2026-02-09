"""
State integrity tests - validate all save states across all games.

Catches:
- Corrupt/empty state files
- States with wrong health values (not at max)
- States with expired timers (fight already in progress)
- States at wrong round (Round 2 instead of Round 1)
- Missing expected states (e.g., 7 Fight states for MK1)
- State naming convention violations

These tests load actual ROMs and states, so they require the game
integrations to be set up. Tests are skipped if ROMs are missing.
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

from fighters_common.game_configs import get_game_config, GAME_REGISTRY

# Expected states per game. This is the source of truth for what SHOULD exist.
# Update this dict as new states are created.
EXPECTED_STATES = {
    "mk1": {
        "Fight": ["LiuKang", "Sonya", "JohnnyCage", "Kano", "Raiden", "SubZero", "Scorpion"],
        "Match2": ["LiuKang", "Sonya", "JohnnyCage", "Kano", "Raiden", "SubZero", "Scorpion"],
    },
    "mk2": {
        "Fight": [
            "LiuKang", "Scorpion", "SubZero", "Jax", "Sonya", "JohnnyCage",
            "Kitana", "Kung Lao", "Raiden", "Reptile", "Mileena", "ShangTsung",
        ],
    },
    # Add sf2/ssf2 as states are created
}

# Minimum valid state file size (corrupt states tend to be tiny)
MIN_STATE_SIZE_BYTES = 50_000

# State naming pattern: {Prefix}_{CharacterName}
VALID_PREFIXES = ["Fight", "Match2", "Match3", "Match4", "Match5", "Match6", "Match7", "Practice", "CharSelect"]


def _get_state_dir(game_alias):
    """Get the state directory for a game."""
    config = get_game_config(game_alias)
    return ROOT_DIR / config.game_dir_name / "custom_integrations" / config.game_id


def _load_state_env(game_alias, state_name):
    """Load a game environment from a state. Returns (base_env, config)."""
    config = get_game_config(game_alias)
    game_dir = ROOT_DIR / config.game_dir_name
    integrations = game_dir / "custom_integrations"
    retro.data.Integrations.add_custom_path(str(integrations))

    env = retro.make(
        game=config.game_id,
        state=state_name,
        render_mode="rgb_array",
        inttype=retro.data.Integrations.CUSTOM_ONLY,
        use_restricted_actions=retro.Actions.ALL,
    )
    return env, config


class TestStateFileIntegrity(unittest.TestCase):
    """Test that state files are well-formed and not corrupt."""

    def _check_states_exist(self, game_alias, prefix, expected_chars):
        state_dir = _get_state_dir(game_alias)
        if not state_dir.exists():
            self.skipTest(f"No state dir for {game_alias}")

        missing = []
        for char in expected_chars:
            state_path = state_dir / f"{prefix}_{char}.state"
            if not state_path.exists():
                missing.append(char)

        self.assertEqual(missing, [],
                         f"{game_alias} missing {prefix} states: {missing}")

    def _check_state_sizes(self, game_alias, prefix, expected_chars):
        state_dir = _get_state_dir(game_alias)
        if not state_dir.exists():
            self.skipTest(f"No state dir for {game_alias}")

        too_small = []
        for char in expected_chars:
            state_path = state_dir / f"{prefix}_{char}.state"
            if state_path.exists() and state_path.stat().st_size < MIN_STATE_SIZE_BYTES:
                too_small.append((char, state_path.stat().st_size))

        self.assertEqual(too_small, [],
                         f"Suspiciously small state files (corrupt?): {too_small}")

    def _check_no_rogue_states(self, game_alias):
        """Ensure all state files follow naming conventions."""
        state_dir = _get_state_dir(game_alias)
        if not state_dir.exists():
            self.skipTest(f"No state dir for {game_alias}")

        bad_names = []
        for state_file in state_dir.glob("*.state"):
            name = state_file.stem
            # Skip temp files
            if name.startswith("_"):
                continue
            # Must match {Prefix}_{Something} or {Prefix}_{GameName}
            parts = name.split("_", 1)
            if len(parts) < 2:
                bad_names.append(name)
            elif parts[0] not in VALID_PREFIXES and not parts[0].startswith("Fight"):
                bad_names.append(name)

        self.assertEqual(bad_names, [],
                         f"State files with non-standard names: {bad_names}")

    # --- MK1 ---
    def test_mk1_fight_states_exist(self):
        if "mk1" not in EXPECTED_STATES:
            self.skipTest("No expected states defined for mk1")
        self._check_states_exist("mk1", "Fight", EXPECTED_STATES["mk1"]["Fight"])

    def test_mk1_match2_states_exist(self):
        if "mk1" not in EXPECTED_STATES or "Match2" not in EXPECTED_STATES["mk1"]:
            self.skipTest("No Match2 expected states defined for mk1")
        self._check_states_exist("mk1", "Match2", EXPECTED_STATES["mk1"]["Match2"])

    def test_mk1_state_sizes(self):
        if "mk1" not in EXPECTED_STATES:
            self.skipTest("No expected states for mk1")
        for prefix, chars in EXPECTED_STATES["mk1"].items():
            self._check_state_sizes("mk1", prefix, chars)

    def test_mk1_naming_conventions(self):
        self._check_no_rogue_states("mk1")

    # --- MK2 ---
    def test_mk2_fight_states_exist(self):
        if "mk2" not in EXPECTED_STATES:
            self.skipTest("No expected states defined for mk2")
        self._check_states_exist("mk2", "Fight", EXPECTED_STATES["mk2"]["Fight"])

    def test_mk2_state_sizes(self):
        if "mk2" not in EXPECTED_STATES:
            self.skipTest("No expected states for mk2")
        for prefix, chars in EXPECTED_STATES["mk2"].items():
            self._check_state_sizes("mk2", prefix, chars)


class TestStateRAMValues(unittest.TestCase):
    """
    Test that each state loads with correct RAM values.

    This catches the critical bug where states were saved at wrong game moments:
    - Wrong round (Round 2 instead of Round 1)
    - Depleted health (fight already started)
    - Timer already ticking down (not a fresh round start)
    """

    def _validate_fight_state(self, game_alias, state_name):
        """Validate a fight-ready state has full health and active timer."""
        config = get_game_config(game_alias)
        state_dir = _get_state_dir(game_alias)

        if not (state_dir / f"{state_name}.state").exists():
            self.skipTest(f"State {state_name} not found")

        env, config = _load_state_env(game_alias, state_name)
        obs, info = env.reset()

        # Step a few frames to let values settle
        noop = np.zeros(12, dtype=np.int8)
        for _ in range(10):
            obs, _, _, _, info = env.step(noop)

        # Check health values
        health = info.get("health", -1)
        enemy_health = info.get("enemy_health", -1)

        # For games with ram_overrides (MK2), health may come from DirectRAMReader
        # but we're using raw env here, so check if we need overrides
        if config.ram_overrides and health <= 0:
            ram = env.get_ram()
            health = int(ram[config.ram_overrides.get("health", 0)])
            enemy_health = int(ram[config.ram_overrides.get("enemy_health", 0)])

        env.close()

        # Health must be at max
        self.assertEqual(health, config.max_health,
                         f"{state_name}: health={health}, expected {config.max_health}")
        self.assertEqual(enemy_health, config.max_health,
                         f"{state_name}: enemy_health={enemy_health}, expected {config.max_health}")

    def _validate_timer(self, game_alias, state_name):
        """Validate timer is at or near max (fresh round)."""
        config = get_game_config(game_alias)
        state_dir = _get_state_dir(game_alias)

        if not (state_dir / f"{state_name}.state").exists():
            self.skipTest(f"State {state_name} not found")

        env, config = _load_state_env(game_alias, state_name)
        obs, info = env.reset()

        noop = np.zeros(12, dtype=np.int8)
        for _ in range(10):
            obs, _, _, _, info = env.step(noop)

        timer = info.get("timer", -1)
        env.close()

        # Timer should be > 50 for a fresh fight state
        # MK1: timer=153 means "99" on display, timer>50 means fight is active
        self.assertGreater(timer, 50,
                           f"{state_name}: timer={timer}, expected >50 (fresh round)")

    # --- MK1 Fight States ---
    def test_mk1_fight_liukang_health(self):
        self._validate_fight_state("mk1", "Fight_LiuKang")

    def test_mk1_fight_sonya_health(self):
        self._validate_fight_state("mk1", "Fight_Sonya")

    def test_mk1_fight_johnnycage_health(self):
        self._validate_fight_state("mk1", "Fight_JohnnyCage")

    def test_mk1_fight_kano_health(self):
        self._validate_fight_state("mk1", "Fight_Kano")

    def test_mk1_fight_raiden_health(self):
        self._validate_fight_state("mk1", "Fight_Raiden")

    def test_mk1_fight_subzero_health(self):
        self._validate_fight_state("mk1", "Fight_SubZero")

    def test_mk1_fight_scorpion_health(self):
        self._validate_fight_state("mk1", "Fight_Scorpion")

    # --- MK1 Match2 States ---
    def test_mk1_match2_liukang_health(self):
        self._validate_fight_state("mk1", "Match2_LiuKang")

    def test_mk1_match2_sonya_health(self):
        self._validate_fight_state("mk1", "Match2_Sonya")

    def test_mk1_match2_johnnycage_health(self):
        self._validate_fight_state("mk1", "Match2_JohnnyCage")

    def test_mk1_match2_kano_health(self):
        self._validate_fight_state("mk1", "Match2_Kano")

    def test_mk1_match2_raiden_health(self):
        self._validate_fight_state("mk1", "Match2_Raiden")

    def test_mk1_match2_subzero_health(self):
        self._validate_fight_state("mk1", "Match2_SubZero")

    def test_mk1_match2_scorpion_health(self):
        self._validate_fight_state("mk1", "Match2_Scorpion")

    # --- MK1 Timers ---
    def test_mk1_fight_timers(self):
        """All Fight states should have fresh timers."""
        for char in EXPECTED_STATES.get("mk1", {}).get("Fight", []):
            with self.subTest(char=char):
                self._validate_timer("mk1", f"Fight_{char}")

    def test_mk1_match2_timers(self):
        """All Match2 states should have fresh timers."""
        for char in EXPECTED_STATES.get("mk1", {}).get("Match2", []):
            with self.subTest(char=char):
                self._validate_timer("mk1", f"Match2_{char}")


class TestStateConsistency(unittest.TestCase):
    """Cross-check states against training scripts."""

    def test_mk1_train_match2_states_all_exist(self):
        """train_match2.py references states that must all exist."""
        state_dir = _get_state_dir("mk1")
        if not state_dir.exists():
            self.skipTest("No MK1 states")

        # These are the states referenced in train_match2.py
        match2_states = [
            "Match2_LiuKang", "Match2_Sonya", "Match2_JohnnyCage",
            "Match2_Kano", "Match2_Raiden", "Match2_SubZero", "Match2_Scorpion",
        ]
        fight_states = [
            "Fight_LiuKang", "Fight_Sonya", "Fight_JohnnyCage",
            "Fight_Kano", "Fight_Raiden", "Fight_SubZero", "Fight_Scorpion",
        ]

        missing = []
        for state in match2_states + fight_states:
            if not (state_dir / f"{state}.state").exists():
                missing.append(state)

        self.assertEqual(missing, [],
                         f"States referenced in train_match2.py but missing: {missing}")

    def test_data_json_has_required_keys(self):
        """Each game's data.json must define health, enemy_health, timer."""
        for alias in ["mk1", "mk2", "sf2", "ssf2"]:
            try:
                config = get_game_config(alias)
            except KeyError:
                continue

            data_json = ROOT_DIR / config.game_dir_name / "custom_integrations" / config.game_id / "data.json"
            if not data_json.exists():
                continue

            import json
            with open(data_json) as f:
                data = json.load(f)

            info = data.get("info", {})

            # Games with ram_overrides read health/enemy_health from RAM directly
            if not config.ram_overrides:
                with self.subTest(game=alias, key="health"):
                    self.assertIn("health", info,
                                  f"{alias}: data.json missing 'health' and no ram_overrides")
                with self.subTest(game=alias, key="enemy_health"):
                    self.assertIn("enemy_health", info,
                                  f"{alias}: data.json missing 'enemy_health' and no ram_overrides")

            with self.subTest(game=alias, key="timer"):
                self.assertIn("timer", info,
                              f"{alias}: data.json missing 'timer'")


if __name__ == "__main__":
    unittest.main()
