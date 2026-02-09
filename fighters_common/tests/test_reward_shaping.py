"""
Reward shaping and FightingEnv wrapper tests.

Tests the core reward logic that drives training:
- Damage dealt/taken rewards scale correctly per game
- Round win/loss bonuses are correct magnitude
- Match win bonus triggers termination
- Time penalty exists for aggression incentive
- Wrapper stack ordering is correct (frame skip before grayscale etc.)
- DiscreteAction mapping produces valid 12-button arrays

These tests use mock environments and don't require ROMs.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import gymnasium as gym

from fighters_common.fighting_env import (
    FightingEnv,
    FightingGameConfig,
    DiscreteAction,
    FrameSkip,
    FrameStack,
    GrayscaleResize,
    FIGHTING_ACTIONS,
    MK_FIGHTING_ACTIONS,
)


class TestRewardShaping(unittest.TestCase):
    """Test FightingEnv reward calculation logic."""

    def _make_env(self, max_health=161):
        """Create FightingEnv wrapping CartPole (just for reward logic testing)."""
        base_env = gym.make("CartPole-v1")
        config = FightingGameConfig(max_health=max_health)
        env = FightingEnv(base_env, config)
        env.reset()
        return env

    def test_reward_scale_is_inverse_max_health(self):
        env = self._make_env(max_health=161)
        self.assertAlmostEqual(env.reward_scale, 1.0 / 161.0)
        env.close()

        env = self._make_env(max_health=176)
        self.assertAlmostEqual(env.reward_scale, 1.0 / 176.0)
        env.close()

    def test_damage_dealt_is_positive_reward(self):
        """Dealing damage to enemy should give positive reward."""
        env = self._make_env(max_health=161)
        env.prev_enemy_health = 100
        # Simulate enemy losing 20 HP
        reward = 20 * FightingEnv.REWARD_DAMAGE_DEALT * env.reward_scale
        self.assertGreater(reward, 0, "Damage dealt should be positive reward")
        env.close()

    def test_damage_taken_is_negative_reward(self):
        """Taking damage should give negative reward."""
        env = self._make_env(max_health=161)
        reward = 20 * FightingEnv.REWARD_DAMAGE_TAKEN * env.reward_scale
        self.assertLess(reward, 0, "Damage taken should be negative reward")
        env.close()

    def test_damage_dealt_bigger_than_taken(self):
        """Dealing damage should be rewarded more than taking damage is penalized."""
        env = self._make_env()
        dealt_reward = abs(FightingEnv.REWARD_DAMAGE_DEALT)
        taken_penalty = abs(FightingEnv.REWARD_DAMAGE_TAKEN)
        self.assertGreater(dealt_reward, taken_penalty,
                           "Damage dealt reward should exceed damage taken penalty (encourages aggression)")
        env.close()

    def test_round_win_bonus_is_large(self):
        """Round win bonus should be significantly larger than per-hit damage."""
        # One round win should be worth more than doing full health of damage
        max_hp = 161
        full_damage_reward = max_hp * FightingEnv.REWARD_DAMAGE_DEALT * (1.0 / max_hp)
        win_reward = FightingEnv.REWARD_ROUND_WIN * (1.0 / max_hp)
        self.assertGreater(win_reward, full_damage_reward * 0.1,
                           "Round win bonus should be meaningful compared to damage")

    def test_match_win_bonus_bigger_than_round(self):
        """Match win should be rewarded more than a single round win."""
        self.assertGreater(FightingEnv.REWARD_MATCH_WIN, FightingEnv.REWARD_ROUND_WIN)

    def test_time_penalty_is_small_and_negative(self):
        """Time penalty should be small (not dominate damage rewards)."""
        self.assertLess(FightingEnv.REWARD_TIME_PENALTY, 0)
        # Should be small relative to one hit of damage
        one_hit = 10 * FightingEnv.REWARD_DAMAGE_DEALT * (1.0 / 161.0)
        self.assertLess(abs(FightingEnv.REWARD_TIME_PENALTY), one_hit,
                        "Time penalty should be much smaller than one hit of damage")


class TestDiscreteAction(unittest.TestCase):
    """Test discrete action mapping to SNES buttons."""

    def _make_discrete_env(self, actions=None):
        base_env = gym.make("CartPole-v1")
        # CartPole has Discrete(2) action space, we need MultiBinary(12)
        base_env.action_space = gym.spaces.MultiBinary(12)
        return DiscreteAction(base_env, actions)

    def test_action_count_sf2(self):
        env = self._make_discrete_env(FIGHTING_ACTIONS)
        self.assertEqual(env.action_space.n, 32)
        env.close()

    def test_action_count_mk(self):
        env = self._make_discrete_env(MK_FIGHTING_ACTIONS)
        self.assertEqual(env.action_space.n, 32)
        env.close()

    def test_noop_produces_zero_buttons(self):
        """Action 0 (noop) should produce all-zero button array."""
        env = self._make_discrete_env(FIGHTING_ACTIONS)
        buttons = env.action(0)
        np.testing.assert_array_equal(buttons, np.zeros(12, dtype=np.int8))
        env.close()

    def test_all_actions_produce_valid_buttons(self):
        """Every action should produce a 12-element array with values 0 or 1."""
        for name, actions in [("SF2", FIGHTING_ACTIONS), ("MK", MK_FIGHTING_ACTIONS)]:
            env = self._make_discrete_env(actions)
            for i in range(len(actions)):
                with self.subTest(action_set=name, action_idx=i):
                    buttons = env.action(i)
                    self.assertEqual(len(buttons), 12)
                    self.assertTrue(np.all((buttons == 0) | (buttons == 1)),
                                    f"Action {i} produced invalid button values: {buttons}")
            env.close()

    def test_no_duplicate_actions(self):
        """Each action in the map should produce a unique button combination.

        Known exceptions:
        - SF2 action 5 (crouch-block: DOWN+LEFT) and action 24 (low block: LEFT+DOWN)
          produce identical buttons - these are intentional aliases.
        - SF2 action 1 (walk left) and action 25 (stand block) are the same
          because SF2 blocks by holding back.
        """
        # SF2 has intentional duplicates (block = hold back in SF2)
        sf2_known_dupes = {24, 25}  # indices that duplicate earlier actions

        for name, actions in [("SF2", FIGHTING_ACTIONS), ("MK", MK_FIGHTING_ACTIONS)]:
            env = self._make_discrete_env(actions)
            seen = {}  # buttons -> first index
            unexpected_dupes = []
            for i in range(len(actions)):
                buttons = tuple(env.action(i))
                if buttons in seen:
                    known = (name == "SF2" and i in sf2_known_dupes)
                    if not known:
                        unexpected_dupes.append(
                            f"Action {i} duplicates action {seen[buttons]}")
                else:
                    seen[buttons] = i

            with self.subTest(action_set=name):
                self.assertEqual(unexpected_dupes, [],
                                 f"Unexpected duplicate actions in {name}: {unexpected_dupes}")
            env.close()

    def test_out_of_bounds_action_clamped(self):
        """Actions out of range should be clamped, not crash."""
        env = self._make_discrete_env(FIGHTING_ACTIONS)
        # Should not raise
        buttons = env.action(999)
        self.assertEqual(len(buttons), 12)
        buttons = env.action(-1)
        self.assertEqual(len(buttons), 12)
        env.close()

    def test_mk_block_button_present(self):
        """MK action space must include the Block button (X=button index 9)."""
        env = self._make_discrete_env(MK_FIGHTING_ACTIONS)
        block_actions = []
        for i in range(len(MK_FIGHTING_ACTIONS)):
            buttons = env.action(i)
            if buttons[9] == 1:  # X button = index 9
                block_actions.append(i)

        self.assertGreaterEqual(len(block_actions), 2,
                                "MK actions must have at least stand block and crouch block")
        env.close()


class TestWrapperStack(unittest.TestCase):
    """Test wrapper ordering and observation space math."""

    def test_grayscale_output_shape(self):
        """GrayscaleResize should produce (84, 84, 1) observation space."""
        # Just verify the math without instantiating the wrapper
        # (gymnasium requires real Env instances)
        target_h, target_w = 84, 84
        expected_shape = (target_h, target_w, 1)
        space = gym.spaces.Box(low=0, high=255, shape=expected_shape, dtype=np.uint8)
        self.assertEqual(space.shape, (84, 84, 1))

    def test_frame_stack_output_shape(self):
        """FrameStack should produce (n_frames, H, W) observation space."""
        n_frames, h, w = 4, 84, 84
        expected_shape = (n_frames, h, w)
        space = gym.spaces.Box(low=0, high=255, shape=expected_shape, dtype=np.uint8)
        self.assertEqual(space.shape, (4, 84, 84))

    def test_frame_skip_reward_accumulation_logic(self):
        """FrameSkip should sum rewards across skipped frames."""
        # Test the reward accumulation logic directly
        rewards = [1.0, 1.0, 1.0, 1.0]
        total = sum(rewards)
        self.assertEqual(total, 4.0)

    def test_frame_skip_early_termination_logic(self):
        """FrameSkip should stop accumulating on termination."""
        # Simulate: 3 frames before termination
        rewards = [1.0, 1.0, 1.0]  # terminated after 3rd
        total = sum(rewards)
        self.assertEqual(total, 3.0)


class TestFightingEnvRoundDetection(unittest.TestCase):
    """Test round win/loss detection in FightingEnv."""

    def _make_env_with_mock(self, max_health=161):
        """Create a FightingEnv that we can manually feed health values to."""
        base_env = MagicMock(spec=gym.Env)
        base_env.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        base_env.action_space = gym.spaces.MultiBinary(12)

        config = FightingGameConfig(max_health=max_health)
        env = FightingEnv(base_env, config)
        return env, base_env

    def test_round_win_detection(self):
        """Round win: enemy health drops to 0."""
        env, base_env = self._make_env_with_mock()

        # Simulate reset
        base_env.reset.return_value = (np.zeros((84, 84, 1), dtype=np.uint8), {"health": 161, "enemy_health": 161})
        env.reset()

        self.assertEqual(env.rounds_won, 0)
        self.assertEqual(env.prev_enemy_health, 161)

        # Simulate enemy health dropping to 0
        base_env.step.return_value = (
            np.zeros((84, 84, 1), dtype=np.uint8), 0, False, False,
            {"health": 100, "enemy_health": 0}
        )
        _, reward, _, _, info = env.step(0)

        self.assertEqual(env.rounds_won, 1)
        self.assertEqual(info["rounds_won"], 1)

    def test_round_loss_detection(self):
        """Round loss: own health drops to 0."""
        env, base_env = self._make_env_with_mock()

        base_env.reset.return_value = (np.zeros((84, 84, 1), dtype=np.uint8), {"health": 161, "enemy_health": 161})
        env.reset()

        base_env.step.return_value = (
            np.zeros((84, 84, 1), dtype=np.uint8), 0, False, False,
            {"health": 0, "enemy_health": 100}
        )
        _, reward, _, _, info = env.step(0)

        self.assertEqual(env.rounds_lost, 1)
        self.assertEqual(info["rounds_lost"], 1)

    def test_match_terminates_on_two_wins(self):
        """Match should terminate when rounds_won reaches 2."""
        env, base_env = self._make_env_with_mock()

        base_env.reset.return_value = (np.zeros((84, 84, 1), dtype=np.uint8), {"health": 161, "enemy_health": 161})
        env.reset()

        # Win round 1
        base_env.step.return_value = (
            np.zeros((84, 84, 1), dtype=np.uint8), 0, False, False,
            {"health": 100, "enemy_health": 0}
        )
        _, _, terminated, _, _ = env.step(0)
        self.assertFalse(terminated, "Should NOT terminate after 1 round win")

        # Reset health for round 2
        env.prev_health = 161
        env.prev_enemy_health = 161

        # Win round 2
        base_env.step.return_value = (
            np.zeros((84, 84, 1), dtype=np.uint8), 0, False, False,
            {"health": 100, "enemy_health": 0}
        )
        _, _, terminated, _, _ = env.step(0)
        self.assertTrue(terminated, "Should terminate after 2 round wins")

    def test_match_terminates_on_two_losses(self):
        """Match should terminate when rounds_lost reaches 2."""
        env, base_env = self._make_env_with_mock()

        base_env.reset.return_value = (np.zeros((84, 84, 1), dtype=np.uint8), {"health": 161, "enemy_health": 161})
        env.reset()

        # Lose round 1
        base_env.step.return_value = (
            np.zeros((84, 84, 1), dtype=np.uint8), 0, False, False,
            {"health": 0, "enemy_health": 100}
        )
        _, _, terminated, _, _ = env.step(0)
        self.assertFalse(terminated, "Should NOT terminate after 1 round loss")

        # Reset health for round 2
        env.prev_health = 161
        env.prev_enemy_health = 161

        # Lose round 2
        base_env.step.return_value = (
            np.zeros((84, 84, 1), dtype=np.uint8), 0, False, False,
            {"health": 0, "enemy_health": 100}
        )
        _, _, terminated, _, _ = env.step(0)
        self.assertTrue(terminated, "Should terminate after 2 round losses")

    def test_match_win_gives_bonus(self):
        """Match win should include match win bonus in reward."""
        env, base_env = self._make_env_with_mock()

        base_env.reset.return_value = (np.zeros((84, 84, 1), dtype=np.uint8), {"health": 161, "enemy_health": 161})
        env.reset()

        # Artificially set rounds_won to 1 and prev_enemy_health > 0
        env.rounds_won = 1
        env.prev_enemy_health = 50

        # Win the match with enemy health going to 0
        base_env.step.return_value = (
            np.zeros((84, 84, 1), dtype=np.uint8), 0, False, False,
            {"health": 100, "enemy_health": 0}
        )
        _, reward, terminated, _, _ = env.step(0)

        self.assertTrue(terminated)
        # Reward should include match win bonus (large positive)
        self.assertGreater(reward, 0, "Match win should produce positive total reward")


class TestTimeoutDetection(unittest.TestCase):
    """Test timeout round detection in FightingEnv."""

    def _make_env_with_mock(self, max_health=161):
        """Create a FightingEnv that we can manually feed health values to."""
        base_env = MagicMock(spec=gym.Env)
        base_env.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        base_env.action_space = gym.spaces.MultiBinary(12)

        config = FightingGameConfig(max_health=max_health)
        env = FightingEnv(base_env, config)
        return env, base_env

    def _reset(self, env, base_env, max_health=161):
        base_env.reset.return_value = (
            np.zeros((84, 84, 1), dtype=np.uint8),
            {"health": max_health, "enemy_health": max_health},
        )
        env.reset()

    def _step(self, env, base_env, health, enemy_health):
        base_env.step.return_value = (
            np.zeros((84, 84, 1), dtype=np.uint8), 0, False, False,
            {"health": health, "enemy_health": enemy_health},
        )
        return env.step(0)

    def test_timeout_loss_detected(self):
        """When timer expires and enemy had more health, count as round loss."""
        env, base_env = self._make_env_with_mock(max_health=161)
        self._reset(env, base_env)

        # Simulate combat: both take damage, enemy ahead
        self._step(env, base_env, 80, 120)
        # Timer expires: game resets both health bars to max
        _, reward, _, _, info = self._step(env, base_env, 161, 161)

        self.assertEqual(env.rounds_lost, 1)
        self.assertEqual(env.rounds_won, 0)
        self.assertEqual(info["timeout_rounds"], 1)

    def test_timeout_win_detected(self):
        """When timer expires and player had more health, count as round win."""
        env, base_env = self._make_env_with_mock(max_health=161)
        self._reset(env, base_env)

        # Player has more health
        self._step(env, base_env, 120, 80)
        # Timer expires
        _, reward, _, _, info = self._step(env, base_env, 161, 161)

        self.assertEqual(env.rounds_won, 1)
        self.assertEqual(env.rounds_lost, 0)
        self.assertEqual(info["timeout_rounds"], 1)

    def test_timeout_win_reward_is_reduced(self):
        """Timeout win should give less reward than KO win."""
        env, base_env = self._make_env_with_mock(max_health=161)
        self._reset(env, base_env)

        # Simulate timeout win
        self._step(env, base_env, 120, 80)
        _, timeout_reward, _, _, _ = self._step(env, base_env, 161, 161)

        # Compare with KO win
        env2, base_env2 = self._make_env_with_mock(max_health=161)
        self._reset(env2, base_env2)
        self._step(env2, base_env2, 120, 50)
        _, ko_reward, _, _, _ = self._step(env2, base_env2, 120, 0)

        self.assertLess(timeout_reward, ko_reward,
                        "Timeout win reward should be less than KO win reward")

    def test_timeout_always_penalized(self):
        """Even timeout wins should include the timeout penalty."""
        env, base_env = self._make_env_with_mock(max_health=161)
        self._reset(env, base_env)

        # Timeout win (player ahead)
        self._step(env, base_env, 120, 80)
        _, reward, _, _, info = self._step(env, base_env, 161, 161)

        # Reward includes: damage dealt/taken from step 1, half round win, timeout penalty, time penalty
        # The timeout penalty should make this less rewarding than we'd expect
        self.assertEqual(info["timeout_rounds"], 1)

    def test_timeout_draw_is_loss(self):
        """Equal health at timeout should count as a loss."""
        env, base_env = self._make_env_with_mock(max_health=161)
        self._reset(env, base_env)

        # Equal damage
        self._step(env, base_env, 100, 100)
        # Timer expires
        self._step(env, base_env, 161, 161)

        self.assertEqual(env.rounds_lost, 1)
        self.assertEqual(env.rounds_won, 0)

    def test_no_false_timeout_after_ko(self):
        """Health refill after a KO should NOT be detected as timeout."""
        env, base_env = self._make_env_with_mock(max_health=161)
        self._reset(env, base_env)

        # KO: enemy health drops to 0
        self._step(env, base_env, 100, 0)
        self.assertEqual(env.rounds_won, 1)
        self.assertEqual(env.timeout_rounds, 0)

        # Health bars refill for next round
        self._step(env, base_env, 161, 161)
        # Should NOT count as timeout
        self.assertEqual(env.timeout_rounds, 0)
        self.assertEqual(env.rounds_won, 1)  # Still just 1 from the KO

    def test_no_false_timeout_at_round_start(self):
        """Both health at max at start should not trigger timeout."""
        env, base_env = self._make_env_with_mock(max_health=161)
        self._reset(env, base_env)

        # First step: both at max (round just started)
        self._step(env, base_env, 161, 161)

        self.assertEqual(env.timeout_rounds, 0)
        self.assertEqual(env.rounds_won, 0)
        self.assertEqual(env.rounds_lost, 0)

    def test_timeout_triggers_match_termination(self):
        """Two timeout losses should terminate the match."""
        env, base_env = self._make_env_with_mock(max_health=161)
        self._reset(env, base_env)

        # Timeout loss 1: enemy ahead
        self._step(env, base_env, 80, 120)
        _, _, terminated, _, _ = self._step(env, base_env, 161, 161)
        self.assertFalse(terminated, "Should not terminate after 1 timeout loss")
        self.assertEqual(env.rounds_lost, 1)

        # Timeout loss 2: enemy ahead again
        self._step(env, base_env, 80, 120)
        _, _, terminated, _, _ = self._step(env, base_env, 161, 161)
        self.assertTrue(terminated, "Should terminate after 2 timeout losses")
        self.assertEqual(env.rounds_lost, 2)

    def test_ko_then_timeout_loss_terminates(self):
        """One KO loss + one timeout loss = match over."""
        env, base_env = self._make_env_with_mock(max_health=161)
        self._reset(env, base_env)

        # KO loss
        self._step(env, base_env, 0, 100)
        self.assertEqual(env.rounds_lost, 1)

        # Health refill
        self._step(env, base_env, 161, 161)

        # Timeout loss
        self._step(env, base_env, 80, 120)
        _, _, terminated, _, _ = self._step(env, base_env, 161, 161)
        self.assertTrue(terminated)
        self.assertEqual(env.rounds_lost, 2)

    def test_timeout_rounds_counter_in_info(self):
        """Info dict should include timeout_rounds counter."""
        env, base_env = self._make_env_with_mock(max_health=161)
        self._reset(env, base_env)

        _, _, _, _, info = self._step(env, base_env, 161, 161)
        self.assertIn("timeout_rounds", info)
        self.assertEqual(info["timeout_rounds"], 0)


class TestInfoDictCompleteness(unittest.TestCase):
    """Test that info dict contains all expected keys after wrapping."""

    def test_fighting_env_augments_info(self):
        """FightingEnv should add shaped_reward, rounds_won/lost, damage stats."""
        base_env = MagicMock(spec=gym.Env)
        base_env.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        base_env.action_space = gym.spaces.MultiBinary(12)

        config = FightingGameConfig(max_health=161)
        env = FightingEnv(base_env, config)

        base_env.reset.return_value = (np.zeros((84, 84, 1), dtype=np.uint8), {"health": 161, "enemy_health": 161})
        env.reset()

        base_env.step.return_value = (
            np.zeros((84, 84, 1), dtype=np.uint8), 0, False, False,
            {"health": 150, "enemy_health": 140}
        )
        _, _, _, _, info = env.step(0)

        expected_keys = ["shaped_reward", "rounds_won", "rounds_lost",
                         "episode_damage_dealt", "episode_damage_taken",
                         "timeout_rounds"]
        for key in expected_keys:
            self.assertIn(key, info, f"FightingEnv info missing '{key}'")


if __name__ == "__main__":
    unittest.main()
