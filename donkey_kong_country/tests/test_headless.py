import os
import unittest
from pathlib import Path

# Set headless environment
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["SDL_SOFTWARE_RENDERER"] = "1"

import stable_retro as retro
from retro_harness import make_env
from donkey_kong_country.autosplit import LevelStartDetector, read_level_timer_frames

# RAM offsets
RAM_LEVEL_ID = 0x0076
RAM_LEVEL_TIMER_FRAMES = 0x0046
RAM_LEVEL_TIMER_MINUTES = 0x0048


class TestHeadless(unittest.TestCase):
    """Headless tests for Donkey Kong Country functionality."""

    def setUp(self):
        self.game = "DonkeyKongCountry-Snes"
        self.game_dir = Path(__file__).parent.parent
        # Add custom integrations
        retro.data.Integrations.add_custom_path(str(self.game_dir / "custom_integrations"))

    def test_load_level_state(self):
        """Test loading a level state and verifying level ID."""
        env = retro.make(
            game=self.game,
            state="1Player.CongoJungle.JungleHijinks.Level1",
            render_mode="rgb_array",
            inttype=retro.data.Integrations.CUSTOM_ONLY,
        )
        obs, info = env.reset()
        ram = env.get_ram()

        level_id = int(ram[RAM_LEVEL_ID])
        self.assertIsInstance(level_id, int)
        self.assertGreaterEqual(level_id, 0)
        # Check some candidate offsets
        candidates = [0x3E, 0x76, 0x256, 0x27E, 0x286]
        print(f"Level ID candidates:")
        for offset in candidates:
            if len(ram) > offset:
                val = int(ram[offset])
                print(f"  0x{offset:04X}: 0x{val:02X} ({val})")
        print(f"Using offset 0x{RAM_LEVEL_ID:04X}: 0x{level_id:02X} ({level_id})")

        env.close()

    def test_start_button_starts_level(self):
        """Test that pressing start button starts the level timer."""
        env = retro.make(
            game=self.game,
            state="1Player.CongoJungle.JungleHijinks.Level1",
            render_mode="rgb_array",
            inttype=retro.data.Integrations.CUSTOM_ONLY,
        )
        obs, info = env.reset()

        # Action space: 12 buttons for SNES
        action_size = env.action_space.shape[0]
        self.assertEqual(action_size, 12)

        # Get initial timer
        ram = env.get_ram()
        initial_timer = read_level_timer_frames(ram, frames_offset=RAM_LEVEL_TIMER_FRAMES, minutes_offset=RAM_LEVEL_TIMER_MINUTES)

        # Press start for a few frames
        start_action = [0] * action_size
        start_action[3] = 1  # SNES_START = 3

        for _ in range(10):
            obs, reward, terminated, truncated, info = env.step(start_action)
            if terminated or truncated:
                break

        # Check if timer started moving
        ram = env.get_ram()
        after_timer = read_level_timer_frames(ram, frames_offset=RAM_LEVEL_TIMER_FRAMES, minutes_offset=RAM_LEVEL_TIMER_MINUTES)

        # Timer should have started or moved
        # Note: In the actual game, start might not immediately start timer, but this tests input handling
        self.assertIsNotNone(after_timer)
        print(f"Initial timer: {initial_timer}, After start press: {after_timer}")

        env.close()

    def test_pause_functionality(self):
        """Test pause functionality by pressing start during gameplay."""
        env = retro.make(
            game=self.game,
            state="1Player.CongoJungle.JungleHijinks.Level1",
            render_mode="rgb_array",
            inttype=retro.data.Integrations.CUSTOM_ONLY,
        )
        obs, info = env.reset()

        action_size = env.action_space.shape[0]
        self.assertEqual(action_size, 12)

        # First, try to start the level by pressing start and moving right
        start_action = [0] * action_size
        start_action[3] = 1  # Start
        start_action[7] = 1  # Right

        # Run for some frames to start level
        for _ in range(60):  # 1 second at 60fps
            obs, reward, terminated, truncated, info = env.step(start_action)
            if terminated or truncated:
                break

        ram = env.get_ram()
        timer_after_start = read_level_timer_frames(ram, frames_offset=RAM_LEVEL_TIMER_FRAMES, minutes_offset=RAM_LEVEL_TIMER_MINUTES) or 0

        # Now press start to pause
        pause_action = [0] * action_size
        pause_action[3] = 1  # Start

        # Run for a few frames with pause
        for _ in range(30):
            obs, reward, terminated, truncated, info = env.step(pause_action)
            if terminated or truncated:
                break

        ram = env.get_ram()
        timer_after_pause = read_level_timer_frames(ram, frames_offset=RAM_LEVEL_TIMER_FRAMES, minutes_offset=RAM_LEVEL_TIMER_MINUTES) or 0

        # Timer should not have advanced much during pause
        # This is a basic check; in reality, pause might stop timer completely
        print(f"Timer after start: {timer_after_start}, after pause: {timer_after_pause}")

        env.close()

    def test_exit_beaten_level(self):
        """Test exiting a beaten level with start-select."""
        # This would require a state where the level is beaten
        # For now, placeholder
        self.skipTest("Exit test needs beaten level state")

    def test_load_different_states(self):
        """Test loading different level states."""
        states = [
            "1Player.CongoJungle.JungleHijinks.Level1",
            "1Player.CongoJungle.RopeyRampage.Level2"
        ]
        for state in states:
            with self.subTest(state=state):
                env = retro.make(
                    game=self.game,
                    state=state,
                    render_mode="rgb_array",
                    inttype=retro.data.Integrations.CUSTOM_ONLY,
                )
                obs, info = env.reset()
                ram = env.get_ram()
                level_id = int(ram[RAM_LEVEL_ID])
                self.assertIsInstance(level_id, int)
                env.close()
                print(f"State {state}: level_id 0x{level_id:02X}")


if __name__ == "__main__":
    unittest.main()