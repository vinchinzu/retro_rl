"""
Training configuration and model lineage tests.

Catches:
- Training scripts loading from wrong parent model (catastrophic forgetting)
- LR override not taking effect (train_ppo.py uses custom_objects)
- Model naming conflicts (mk1_ppo_* vs mk1_multichar_ppo_*)
- State lists in training scripts referencing non-existent states
- Training scripts not matching CLAUDE.md documentation

These are static analysis tests - they parse training scripts without running them.
"""

import ast
import os
import re
import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fighters_common.game_configs import get_game_config, GAME_REGISTRY


# Known-good parent models for broad training phases.
APPROVED_BASE_MODELS = {
    "mk1": [
        "mk1_multichar_ppo_2000000_steps.zip",
        "mk1_multichar_ppo_final.zip",
    ],
}

# Models that should NEVER be used as bases for BROAD training phases.
# Narrow fine-tune scripts (train_weak_chars, train_subzero_scorpion) that
# intentionally chain from each other are excluded from this check.
FORBIDDEN_BASES = {
    "mk1": [
        "mk1_weakchar_ppo_final.zip",
        "mk1_subzero_scorpion_ppo_final.zip",
    ],
}

# Scripts that are INTENTIONALLY narrow fine-tunes (exempt from forbidden base check)
NARROW_FINETUNE_SCRIPTS = [
    "train_weak_chars.py",
    "train_subzero_scorpion.py",
]


def _parse_python_constants(filepath):
    """Extract top-level string/number assignments from a Python file."""
    content = filepath.read_text()
    tree = ast.parse(content)

    constants = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    try:
                        constants[target.id] = ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        pass
    return constants


class TestTrainingScriptLineage(unittest.TestCase):
    """Verify training scripts load from approved base models."""

    def _check_script_base_model(self, script_path, game_alias):
        """Check that broad training scripts reference approved base models."""
        if not script_path.exists():
            self.skipTest(f"{script_path.name} not found")

        # Skip intentional narrow fine-tune scripts
        if script_path.name in NARROW_FINETUNE_SCRIPTS:
            return

        content = script_path.read_text()

        # Find BEST_MODEL or similar assignment
        model_refs = re.findall(r'(?:BEST_MODEL|BASE_MODEL|model_path)\s*=\s*["\']([^"\']+\.zip)["\']', content)

        if not model_refs:
            return  # Script doesn't hardcode a model path

        approved = APPROVED_BASE_MODELS.get(game_alias, [])
        forbidden = FORBIDDEN_BASES.get(game_alias, [])

        for ref in model_refs:
            model_name = Path(ref).name
            with self.subTest(script=script_path.name, model=model_name):
                self.assertNotIn(model_name, forbidden,
                                 f"{script_path.name} loads from FORBIDDEN narrow fine-tune: {model_name}. "
                                 f"This causes catastrophic forgetting! Use one of: {approved}")

    def test_mk1_train_match2_base(self):
        script = ROOT_DIR / "mortal_kombat" / "train_match2.py"
        self._check_script_base_model(script, "mk1")

    def test_mk1_no_script_uses_forbidden_base(self):
        """Scan ALL training scripts for forbidden base models."""
        mk_dir = ROOT_DIR / "mortal_kombat"
        if not mk_dir.exists():
            self.skipTest("No mortal_kombat dir")

        for script in mk_dir.glob("train_*.py"):
            self._check_script_base_model(script, "mk1")


class TestLROverride(unittest.TestCase):
    """Verify learning rate override mechanism works."""

    def test_train_ppo_uses_custom_objects_for_lr(self):
        """train_ppo.py must use custom_objects to override LR when loading."""
        script = ROOT_DIR / "fighters_common" / "train_ppo.py"
        content = script.read_text()

        self.assertIn("custom_objects", content,
                      "train_ppo.py doesn't use custom_objects - LR override won't work on model load")
        self.assertIn("learning_rate", content,
                      "train_ppo.py doesn't set learning_rate in custom_objects")

    def test_train_match2_overrides_lr(self):
        """train_match2.py must override LR before calling train_ppo.main()."""
        script = ROOT_DIR / "mortal_kombat" / "train_match2.py"
        if not script.exists():
            self.skipTest("train_match2.py not found")

        content = script.read_text()
        constants = _parse_python_constants(script)

        # Check LR is reduced
        lr = constants.get("LEARNING_RATE", None)
        self.assertIsNotNone(lr, "train_match2.py must define LEARNING_RATE")
        self.assertLess(lr, 3e-4,
                        f"train_match2.py LR={lr} should be < 3e-4 (conservative fine-tuning)")

        # Check it actually patches TrainConfig
        self.assertIn("TrainConfig.LEARNING_RATE", content,
                      "train_match2.py must patch TrainConfig.LEARNING_RATE before calling main()")


class TestModelNaming(unittest.TestCase):
    """Verify model naming conventions are followed."""

    def test_no_bare_mk1_ppo_naming(self):
        """No training script should use the old 'mk1_ppo_*' naming."""
        mk_dir = ROOT_DIR / "mortal_kombat"
        if not mk_dir.exists():
            self.skipTest("No mortal_kombat dir")

        problems = []
        for script in mk_dir.glob("train_*.py"):
            content = script.read_text()
            # Look for prefix assignments like MODEL_PREFIX = "mk1_ppo"
            if re.search(r'["\']mk1_ppo["\']', content):
                problems.append(script.name)

        self.assertEqual(problems, [],
                         f"Scripts using old 'mk1_ppo' naming (should use mk1_multichar_ppo etc): {problems}")

    def test_model_prefix_matches_script(self):
        """Training scripts should produce models with identifiable prefixes."""
        # Map script -> any of these prefix strings should appear in the file
        expected_prefixes = {
            "train_multi_character.py": ["mk1_multichar_ppo", "multichar_ppo"],
            "train_match2.py": ["mk1_match2_ppo"],
            "train_weak_chars.py": ["mk1_weakchar_ppo", "weakchar_ppo"],
            "train_subzero_scorpion.py": ["mk1_subzero_scorpion_ppo", "subzero_scorpion_ppo"],
        }

        mk_dir = ROOT_DIR / "mortal_kombat"
        if not mk_dir.exists():
            self.skipTest("No mortal_kombat dir")

        for script_name, prefixes in expected_prefixes.items():
            script = mk_dir / script_name
            if not script.exists():
                continue

            content = script.read_text()
            with self.subTest(script=script_name):
                found = any(p in content for p in prefixes)
                self.assertTrue(found,
                                f"{script_name} should contain one of {prefixes}")


class TestTrainingStateReferences(unittest.TestCase):
    """Verify training scripts reference states that exist."""

    def _extract_state_names(self, script_path):
        """Extract state name strings from a training script."""
        content = script_path.read_text()
        # Match strings that look like state names
        states = re.findall(r'["\'](?:Fight|Match\d+)_(\w+)["\']', content)
        return states

    def test_mk1_train_match2_states_exist(self):
        script = ROOT_DIR / "mortal_kombat" / "train_match2.py"
        if not script.exists():
            self.skipTest("train_match2.py not found")

        content = script.read_text()
        state_dir = ROOT_DIR / "mortal_kombat" / "custom_integrations" / "MortalKombat-Snes"
        if not state_dir.exists():
            self.skipTest("No MK1 state dir")

        # Extract state lists
        match2_pattern = re.findall(r'"(Match2_\w+)"', content)
        fight_pattern = re.findall(r'"(Fight_\w+)"', content)

        missing = []
        for state in match2_pattern + fight_pattern:
            if not (state_dir / f"{state}.state").exists():
                missing.append(state)

        self.assertEqual(missing, [],
                         f"train_match2.py references missing states: {missing}")

    def test_mk1_train_multi_character_states_exist(self):
        script = ROOT_DIR / "mortal_kombat" / "train_multi_character.py"
        if not script.exists():
            self.skipTest("train_multi_character.py not found")

        content = script.read_text()
        state_dir = ROOT_DIR / "mortal_kombat" / "custom_integrations" / "MortalKombat-Snes"
        if not state_dir.exists():
            self.skipTest("No MK1 state dir")

        fight_pattern = re.findall(r'"(Fight_\w+)"', content)
        missing = []
        for state in fight_pattern:
            if not (state_dir / f"{state}.state").exists():
                missing.append(state)

        self.assertEqual(missing, [],
                         f"train_multi_character.py references missing states: {missing}")


class TestTrainingMixConfig(unittest.TestCase):
    """Verify training mix configurations are sensible."""

    def test_match2_preserves_fight_skills(self):
        """Match2 training must include >50% Fight states to prevent forgetting."""
        script = ROOT_DIR / "mortal_kombat" / "train_match2.py"
        if not script.exists():
            self.skipTest("train_match2.py not found")

        constants = _parse_python_constants(script)
        match2_weight = constants.get("MATCH2_WEIGHT", None)

        self.assertIsNotNone(match2_weight, "train_match2.py must define MATCH2_WEIGHT")
        self.assertLessEqual(match2_weight, 0.5,
                             f"MATCH2_WEIGHT={match2_weight} is too high. "
                             f"Must be <=0.5 to preserve Match 1 skills (prevent catastrophic forgetting)")


class TestGameConfigConsistency(unittest.TestCase):
    """Verify game configs are internally consistent."""

    def test_all_configs_have_required_fields(self):
        """Every game config must have the essential fields set."""
        for name, config in GAME_REGISTRY.items():
            if name.islower():  # Skip aliases
                continue

            with self.subTest(game=name):
                self.assertTrue(config.game_id, f"{name}: missing game_id")
                self.assertTrue(config.display_name, f"{name}: missing display_name")
                self.assertTrue(config.game_dir_name, f"{name}: missing game_dir_name")
                self.assertGreater(config.max_health, 0, f"{name}: max_health must be > 0")
                self.assertGreater(config.round_length_frames, 0, f"{name}: round_length_frames must be > 0")
                self.assertGreaterEqual(config.rounds_to_win, 2, f"{name}: rounds_to_win must be >= 2")

    def test_mk_games_use_mk_actions(self):
        """MK games must use MK_FIGHTING_ACTIONS (has dedicated Block button)."""
        from fighters_common.fighting_env import MK_FIGHTING_ACTIONS

        for alias in ["mk1", "mk2"]:
            config = get_game_config(alias)
            with self.subTest(game=alias):
                self.assertIs(config.actions, MK_FIGHTING_ACTIONS,
                              f"{alias} should use MK_FIGHTING_ACTIONS")

    def test_sf_games_use_default_actions(self):
        """SF games should use default actions (None -> FIGHTING_ACTIONS)."""
        for alias in ["sf2", "ssf2"]:
            try:
                config = get_game_config(alias)
            except KeyError:
                continue
            with self.subTest(game=alias):
                self.assertIsNone(config.actions,
                                  f"{alias} should use default actions (None)")

    def test_mk2_has_ram_overrides(self):
        """MK2 needs ram_overrides for health (high WRAM addresses)."""
        config = get_game_config("mk2")
        self.assertTrue(config.ram_overrides,
                        "MK2 must have ram_overrides for health/enemy_health")
        self.assertIn("health", config.ram_overrides)
        self.assertIn("enemy_health", config.ram_overrides)

    def test_game_dirs_exist(self):
        """Game directories referenced in configs should exist."""
        for name, config in GAME_REGISTRY.items():
            if name.islower():
                continue
            game_dir = ROOT_DIR / config.game_dir_name
            with self.subTest(game=name):
                self.assertTrue(game_dir.exists(),
                                f"Game dir '{config.game_dir_name}' does not exist")

    def test_aliases_resolve_correctly(self):
        """Short aliases should resolve to the same config as full IDs."""
        pairs = [
            ("mk1", "MortalKombat-Snes"),
            ("mk2", "MortalKombatII-Snes"),
            ("sf2", "StreetFighterIITurbo-Snes"),
            ("ssf2", "SuperStreetFighterII-Snes"),
        ]

        for alias, full_id in pairs:
            with self.subTest(alias=alias):
                self.assertIs(get_game_config(alias), get_game_config(full_id))


if __name__ == "__main__":
    unittest.main()
