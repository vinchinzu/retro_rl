from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from harvest.runtime.editor_bridge import (
    HarvestEditorBridgeRuntime,
    build_snapshot,
    read_env_wram,
)
from harvest.core.tile_catalog import ADDR_INPUT_LOCK
from retro_harness.controls import SNES_BUTTON_NAMES


class _FakeEmulator:
    def set_state(self, raw: bytes) -> None:
        self.raw_state = raw

    def save_state(self, path: str) -> None:
        Path(path).write_bytes(b"state")


class _FakeEnv:
    def __init__(self) -> None:
        self.ram = np.zeros(0x24000, dtype=np.uint8)
        self.ram[ADDR_INPUT_LOCK] = 1
        self.img = np.zeros((224, 256, 3), dtype=np.uint8)
        self.em = _FakeEmulator()
        self.actions: list[list[int]] = []
        self.closed = False

    def reset(self):
        return self.render(), {}

    def step(self, action):
        self.actions.append([int(value) for value in action])
        return self.render(), 0.0, False, False, {}

    def render(self):
        return self.img

    def get_ram(self) -> np.ndarray:
        return self.ram

    def close(self) -> None:
        self.closed = True


class _FakeBot:
    day_plan_enabled = True
    day_plan_started = False
    day_plan_task = SimpleNamespace(phases=(SimpleNamespace(kind="cow_chores"),))
    disable_reason = None

    def __init__(self) -> None:
        self.enabled = False
        self.env = None
        self.prepare_calls = 0
        self.bot_action_calls = 0

    def set_env(self, env: object) -> None:
        self.env = env

    def prepare_for_enable(self) -> None:
        self.prepare_calls += 1

    def get_goal_text(self) -> str:
        return "animal chores"

    def get_action(self, game_state: object, obs: object) -> np.ndarray:
        del game_state, obs
        self.bot_action_calls += 1
        action = np.zeros(len(SNES_BUTTON_NAMES), dtype=np.int32)
        action[8] = 1
        return action


class HarvestEditorBridgeRuntimeTests(unittest.TestCase):
    def _runtime(self, tmpdir: str) -> tuple[HarvestEditorBridgeRuntime, _FakeEnv]:
        env = _FakeEnv()

        def make_env(state_name: str | None) -> _FakeEnv:
            env.state_name = state_name
            return env

        runtime = HarvestEditorBridgeRuntime(
            project_root=Path(tmpdir),
            states_dir=Path(tmpdir) / "states",
            capture_dir=Path(tmpdir) / "captures",
            hot_save_path=Path(tmpdir) / "hot.state",
            button_order=SNES_BUTTON_NAMES,
            make_env=make_env,
            read_wram=read_env_wram,
            build_snapshot=build_snapshot,
        )
        return runtime, env

    def test_autoplay_bridge_warmup_then_manual_after_disable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime, env = self._runtime(tmpdir)
            bots: list[_FakeBot] = []
            bot_state_names: list[str | None] = []

            def build_bot() -> _FakeBot:
                bot_state_names.append(runtime._autoplay_state_name)
                bot = _FakeBot()
                bots.append(bot)
                return bot

            runtime._build_autoplay_bot = build_bot  # type: ignore[method-assign]
            runtime.start_session(state_file="latest")

            status = runtime.set_autoplay(
                enabled=True,
                state_name="Y1_Livestock_Compact_chicken_cow",
            )
            self.assertTrue(status["autoplayEnabled"])
            self.assertEqual(status["autoplayGoal"], "animal chores")
            self.assertEqual(bot_state_names, ["Y1_Livestock_Compact_chicken_cow"])
            self.assertEqual(bots[0].prepare_calls, 1)

            snapshot, _frame_rgb, _elapsed = runtime.step(include_frame=False)
            self.assertTrue(snapshot["autoplayEnabled"])
            self.assertEqual(snapshot["logicalAction"][0], 1)
            self.assertEqual(env.actions[-1][0], 1)
            self.assertEqual(bots[0].bot_action_calls, 0)

            runtime.set_autoplay(enabled=False)
            manual_right = [0] * len(SNES_BUTTON_NAMES)
            manual_right[7] = 1
            snapshot, _frame_rgb, _elapsed = runtime.step(
                action=manual_right,
                include_frame=False,
            )

            self.assertFalse(snapshot["autoplayEnabled"])
            self.assertEqual(snapshot["logicalAction"], manual_right)
            self.assertEqual(env.actions[-1], manual_right)


if __name__ == "__main__":
    unittest.main()
