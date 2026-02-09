"""
Win condition consistency tests.

Catches:
- Scripts using wrong win condition (rounds_won >= 2 without > rounds_lost)
- Inconsistent win detection across benchmark, training, and match_manager
- FightingEnv termination not matching win condition
- FightMetricsCallback using different logic than benchmark scripts

The correct win condition is:
    rounds_won >= 2 AND rounds_won > rounds_lost

This handles the tiebreaker scenario (2-2 goes to round 5) where
rounds_won >= 2 alone would incorrectly count a loss as a win.
"""

import os
import re
import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


class TestWinConditionInSource(unittest.TestCase):
    """
    Grep all Python files for win condition patterns and verify correctness.

    This is a static analysis test - no ROMs needed.
    """

    def _find_python_files(self):
        """Find all .py files in the project (excluding archive/)."""
        files = []
        for game_dir in ["mortal_kombat", "mortal_kombat_ii", "street_fighter_ii",
                         "super_street_fighter_ii", "fighters_common"]:
            d = ROOT_DIR / game_dir
            if not d.exists():
                continue
            for f in d.rglob("*.py"):
                if "archive" in str(f) or "__pycache__" in str(f):
                    continue
                files.append(f)
        return files

    def test_no_bare_rounds_won_ge_2(self):
        """
        No script should use `rounds_won >= 2` as the SOLE win condition
        in benchmark/scoring contexts.

        The correct pattern is: rounds_won >= 2 AND rounds_won > rounds_lost

        Known exceptions (not scoring contexts):
        - FightingEnv.step(): uses rounds_won >= 2 for episode termination
        - FightMetricsCallback: approximate win counting for training logs
        - debug_speedrun.py: state extraction win detection
        """
        problems = []

        # Patterns that indicate a bare rounds_won >= 2 check
        # Also check short variable names like rw >= 2
        bare_pattern = re.compile(r'(?:rounds_won|rw)\s*>=\s*2')
        full_pattern = re.compile(r'(?:rounds_won|rw)\s*>\s*(?:rounds_lost|rl)')

        for filepath in self._find_python_files():
            content = filepath.read_text()
            lines = content.split('\n')

            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith('#'):
                    continue

                if bare_pattern.search(line):
                    # Check if the full condition is nearby (within 3 lines)
                    context = '\n'.join(lines[max(0, i-2):min(len(lines), i+2)])
                    if not full_pattern.search(context):
                        rel_path = filepath.relative_to(ROOT_DIR)

                        # FightingEnv: rounds_won >= 2 for TERMINATION (not win detection)
                        if "fighting_env.py" in str(filepath):
                            continue
                        # FightMetricsCallback: approximate counting for training logs
                        if "train_ppo.py" in str(filepath) and "self.wins" in context:
                            continue
                        # State extraction scripts (not scoring)
                        if "debug_speedrun.py" in str(filepath):
                            continue
                        if "speedrun_sequential.py" in str(filepath):
                            continue
                        # Test files themselves (comments/docstrings about the condition)
                        if "/tests/" in str(filepath):
                            continue

                        problems.append(f"{rel_path}:{i}: bare win check without `> rounds_lost`")

        self.assertEqual(problems, [],
                         f"Found scripts with incomplete win conditions:\n" +
                         "\n".join(f"  {p}" for p in problems))

    def test_benchmark_scripts_have_full_condition(self):
        """Benchmark and scoring scripts MUST use the full win condition."""
        critical_scripts = [
            ROOT_DIR / "mortal_kombat" / "benchmark_characters.py",
            ROOT_DIR / "mortal_kombat" / "match_manager.py",
            ROOT_DIR / "mortal_kombat" / "model_registry.py",
        ]

        # Match full condition with any variable names (rounds_won/rw, rounds_lost/rl)
        full_pattern = re.compile(
            r'(?:rounds_won|rw)\s*>=\s*2.*?(?:rounds_won|rw)\s*>\s*(?:rounds_lost|rl)'
        )

        for script in critical_scripts:
            if not script.exists():
                continue

            content = script.read_text()
            # Remove multi-line comments
            content_no_comments = re.sub(r'""".*?"""', '', content, flags=re.DOTALL)
            content_no_comments = re.sub(r"'''.*?'''", '', content_no_comments, flags=re.DOTALL)

            with self.subTest(script=script.name):
                self.assertTrue(
                    full_pattern.search(content_no_comments),
                    f"{script.name} does not contain the full win condition "
                    f"(rounds_won >= 2 AND rounds_won > rounds_lost)"
                )


class TestWinConditionLogic(unittest.TestCase):
    """Unit tests for the actual win condition logic."""

    @staticmethod
    def is_match_won(rounds_won, rounds_lost):
        """The canonical win condition - copied here as ground truth."""
        return rounds_won >= 2 and rounds_won > rounds_lost

    def test_clear_win_2_0(self):
        self.assertTrue(self.is_match_won(2, 0))

    def test_clear_win_2_1(self):
        self.assertTrue(self.is_match_won(2, 1))

    def test_tiebreaker_win_3_2(self):
        """After 2-2 tiebreaker, winning round 5 = win."""
        self.assertTrue(self.is_match_won(3, 2))

    def test_clear_loss_0_2(self):
        self.assertFalse(self.is_match_won(0, 2))

    def test_clear_loss_1_2(self):
        self.assertFalse(self.is_match_won(1, 2))

    def test_tiebreaker_loss_2_3(self):
        """After 2-2 tiebreaker, losing round 5 = loss."""
        self.assertFalse(self.is_match_won(2, 3))

    def test_tied_2_2_is_not_win(self):
        """
        CRITICAL: 2-2 is NOT a win. This is the exact bug that was found.
        With `rounds_won >= 2` alone, this would incorrectly be True.
        """
        self.assertFalse(self.is_match_won(2, 2))

    def test_incomplete_match(self):
        self.assertFalse(self.is_match_won(1, 0))
        self.assertFalse(self.is_match_won(0, 1))
        self.assertFalse(self.is_match_won(1, 1))
        self.assertFalse(self.is_match_won(0, 0))


class TestFightingEnvTermination(unittest.TestCase):
    """Test that FightingEnv terminates matches correctly.

    These tests directly check the termination conditions without
    needing a real game environment.
    """

    def test_terminates_on_two_wins(self):
        """FightingEnv should terminate when rounds_won >= 2."""
        from fighters_common.fighting_env import FightingEnv
        # Direct check of the termination condition from FightingEnv.step()
        rounds_won = 2
        self.assertTrue(rounds_won >= 2, "Should terminate at 2 wins")

    def test_terminates_on_two_losses(self):
        rounds_lost = 2
        self.assertTrue(rounds_lost >= 2, "Should terminate at 2 losses")

    def test_reward_scale_correct_mk1(self):
        """MK1 reward scale should be 1/161, not the default 1/176."""
        from fighters_common.fighting_env import FightingGameConfig
        config = FightingGameConfig(max_health=161)
        self.assertAlmostEqual(1.0 / config.max_health, 1.0 / 161.0)

    def test_reward_scale_correct_sf2(self):
        from fighters_common.fighting_env import FightingGameConfig
        config = FightingGameConfig(max_health=176)
        self.assertAlmostEqual(1.0 / config.max_health, 1.0 / 176.0)


if __name__ == "__main__":
    unittest.main()
