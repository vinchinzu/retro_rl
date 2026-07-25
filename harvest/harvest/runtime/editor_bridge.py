"""Stdio emulator bridge for the Harvest Moon editor."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from harvest.paths import GAME, PROJECT_DIR, STATES_DIR
from harvest.runtime.editor_snapshot import (
    ADDR_TILEMAP,
    ADDR_TOOL,
    ADDR_X,
    ADDR_Y,
    CAPTURE_DIR,
    DEFAULT_STATE_PATH,
    HOT_SAVE_PATH,
    map_name,
)
from harvest.runtime.retro_setup import make_harvest_env as _make_harvest_env
from retro_harness.controls import SNES_BUTTON_NAMES, init_controller
from retro_harness.editor.bridge_runtime import EditorBridgeRuntime
from retro_harness.editor.bridge_server import run_stdio_bridge

GAME_ID = GAME


def _env_wram_block(env: object) -> object:
    if hasattr(env, "data") and hasattr(env.data, "memory"):
        blocks = getattr(env.data.memory, "blocks", {})
        if 0x7E0000 in blocks:
            return blocks[0x7E0000]
    if hasattr(env, "get_ram"):
        return np.asarray(env.get_ram(), dtype=np.uint8)
    raise RuntimeError("Could not read emulator WRAM")


def read_env_wram(env: object) -> bytes:
    block = _env_wram_block(env)
    if isinstance(block, np.ndarray):
        return block.astype(np.uint8, copy=False).tobytes()
    return bytes(block)


def player_position(ram: bytes) -> tuple[int, int]:
    if ADDR_X + 1 >= len(ram) or ADDR_Y + 1 >= len(ram):
        return 0, 0
    x = ram[ADDR_X] | (ram[ADDR_X + 1] << 8)
    y = ram[ADDR_Y] | (ram[ADDR_Y + 1] << 8)
    return int(x), int(y)


def tilemap_id_from_ram(ram: bytes) -> int:
    if ADDR_TILEMAP < len(ram):
        return int(ram[ADDR_TILEMAP])
    return 0


def build_snapshot(
    env: object,
    obs: object,
    info: dict[str, object],
    frame_counter: int,
    logical_action: list[int],
) -> dict[str, object]:
    del info
    wram = _env_wram_block(env)
    tilemap_id = tilemap_id_from_ram(wram)
    px, py = player_position(wram)
    tool_id = int(wram[ADDR_TOOL]) if ADDR_TOOL < len(wram) else 0
    tool_name = {
        0x00: "None",
        0x01: "Sickle",
        0x02: "Hoe",
        0x03: "Hammer",
        0x04: "Axe",
        0x0F: "Brush",
        0x10: "Watering Can",
    }.get(tool_id, f"0x{tool_id:02X}")
    return {
        "game": GAME_ID,
        "frameCounter": frame_counter,
        "tilemapId": tilemap_id,
        "mapName": map_name(tilemap_id),
        "playerX": px,
        "playerY": py,
        "playerTileX": px // 16,
        "playerTileY": py // 16,
        "toolId": tool_id,
        "toolName": tool_name,
        "logicalAction": logical_action,
    }


def make_harvest_env(state_name: str | None) -> object:
    return _make_harvest_env(state_name)


@dataclass
class HarvestEditorBridgeRuntime(EditorBridgeRuntime):
    """Harvest-specific bridge runtime with optional in-process bot control."""

    _autoplay_enabled: bool = field(default=False, init=False)
    _autoplay_bot: object | None = field(default=None, init=False)
    _autoplay_state_name: str | None = field(default=None, init=False)
    _autoplay_cancel_until_clear: bool = field(default=False, init=False)
    _autoplay_cancel_frames: int = field(default=0, init=False)

    def _state_name_from_file(self, state_file: object | None) -> str | None:
        if not state_file:
            return None
        value = str(state_file)
        if value in {"NONE", "Reset"}:
            return None
        return os.path.splitext(os.path.basename(value))[0] or None

    def _disable_autoplay(self, *, drop_bot: bool = False) -> None:
        bot = self._autoplay_bot
        if bot is not None:
            setattr(bot, "enabled", False)
            if drop_bot:
                setattr(bot, "env", None)
        if drop_bot:
            self._autoplay_bot = None
        self._autoplay_enabled = False
        self._autoplay_cancel_until_clear = False
        self._autoplay_cancel_frames = 0

    def close(self) -> None:
        self._disable_autoplay(drop_bot=True)
        self._autoplay_state_name = None
        super().close()

    def start_session(
        self,
        *,
        state_file: str | None = None,
        rom_path: str | None = None,
    ) -> tuple[dict[str, object], bytes | None]:
        snapshot, frame_rgb = super().start_session(state_file=state_file, rom_path=rom_path)
        self._autoplay_state_name = self._state_name_from_file(state_file)
        self._autoplay_bot = None
        snapshot.update(self._autoplay_status())
        return snapshot, frame_rgb

    def _build_autoplay_bot(self):
        from harvest.runtime.harvest_bot import AutoClearBot

        return AutoClearBot(
            day_plan_enabled=True,
            auto_day_plan_state_name=self._autoplay_state_name,
        )

    def _autoplay_status(self) -> dict[str, object]:
        bot = self._autoplay_bot
        status: dict[str, object] = {
            "autoplayEnabled": bool(self._autoplay_enabled),
            "autoplayMode": "bot" if self._autoplay_enabled else "human",
        }
        if bot is not None:
            status["autoplayBotEnabled"] = bool(bot.enabled)
            status["autoplayGoal"] = bot.get_goal_text()
            if bot.disable_reason:
                status["autoplayDisableReason"] = bot.disable_reason
        return status

    def _skip_autoplay_cancel_warmup(self) -> bool:
        bot = self._autoplay_bot
        if (
            bot is None
            or not getattr(bot, "day_plan_enabled", False)
            or getattr(bot, "day_plan_started", False)
        ):
            return False
        phases = bot.day_plan_task.phases
        if not phases:
            return False
        return phases[0].kind == "recorded_transition"

    def _start_autoplay_cancel(self) -> None:
        self._autoplay_cancel_until_clear = True
        self._autoplay_cancel_frames = 0 if self._skip_autoplay_cancel_warmup() else 90

    def set_autoplay(
        self,
        *,
        enabled: bool,
        state_name: object | None = None,
    ) -> dict[str, object]:
        if self._env is None:
            raise RuntimeError("No active session")
        if state_name:
            requested_state = self._state_name_from_file(state_name)
            if requested_state != self._autoplay_state_name:
                self._autoplay_state_name = requested_state
                self._autoplay_bot = None
        if enabled:
            if self._autoplay_bot is None:
                self._autoplay_bot = self._build_autoplay_bot()
            self._autoplay_bot.set_env(self._env)
            self._autoplay_bot.enabled = True
            self._autoplay_bot.prepare_for_enable()
            self._autoplay_enabled = True
            self._start_autoplay_cancel()
        else:
            self._disable_autoplay()
        return self._autoplay_status()

    def _autoplay_warmup_action(self, ram: np.ndarray) -> list[int] | None:
        from harvest.runtime.harvest_bot import ADDR_INPUT_LOCK
        from harvest.tasks.farm_clearer import make_action

        input_lock = int(ram[ADDR_INPUT_LOCK]) if ADDR_INPUT_LOCK < len(ram) else 1
        if self._autoplay_cancel_until_clear:
            if input_lock != 1:
                action = make_action(
                    b=self._frame_counter % 2 == 0,
                    a=self._frame_counter % 2 == 1,
                )
                return [int(value) for value in action]
            self._autoplay_cancel_until_clear = False

        if self._autoplay_cancel_frames > 0:
            action = make_action(b=self._frame_counter % 2 == 0)
            self._autoplay_cancel_frames -= 1
            if self._autoplay_cancel_frames == 0:
                self._autoplay_cancel_until_clear = False
            return [int(value) for value in action]
        return None

    def _autoplay_action(self) -> list[int]:
        if self._env is None or not self._autoplay_enabled:
            return list(self._last_action)
        if self._autoplay_bot is None:
            self._autoplay_bot = self._build_autoplay_bot()
        bot = self._autoplay_bot
        bot.set_env(self._env)
        if not bot.enabled:
            self._disable_autoplay()
            return [0] * len(self.button_order)

        from harvest.runtime.harvest_bot import GameState

        ram = np.asarray(self._env.get_ram(), dtype=np.uint8)
        warmup_action = self._autoplay_warmup_action(ram)
        if warmup_action is not None:
            return warmup_action

        obs = getattr(self._env, "img", None)
        if obs is None:
            obs = np.zeros((1, 1, 3), dtype=np.uint8)
        game_state = GameState({}, ram)
        action = bot.get_action(game_state, obs)
        return [int(value) for value in action]

    def step(
        self,
        *,
        action: list[int] | None = None,
        repeat: int = 1,
        include_frame: bool = True,
        include_wram: bool = True,
    ) -> tuple[dict[str, object], bytes | None, float]:
        if self._autoplay_enabled:
            action = self._autoplay_action()
        snapshot, frame_rgb, elapsed_ms = super().step(
            action=action,
            repeat=repeat,
            include_frame=include_frame,
            include_wram=include_wram,
        )
        if self._autoplay_bot is not None and not self._autoplay_bot.enabled:
            self._disable_autoplay()
        snapshot.update(self._autoplay_status())
        return snapshot, frame_rgb, elapsed_ms

    def _snapshot(self, *args, **kwargs) -> tuple[dict[str, object], bytes | None]:
        snapshot, frame_rgb = super()._snapshot(*args, **kwargs)
        snapshot.update(self._autoplay_status())
        return snapshot, frame_rgb


def build_runtime() -> EditorBridgeRuntime:
    pygame = __import__("pygame")
    controller = init_controller(pygame)
    runtime = HarvestEditorBridgeRuntime(
        project_root=PROJECT_DIR,
        states_dir=STATES_DIR,
        capture_dir=CAPTURE_DIR,
        hot_save_path=HOT_SAVE_PATH,
        button_order=SNES_BUTTON_NAMES,
        make_env=make_harvest_env,
        read_wram=read_env_wram,
        build_snapshot=build_snapshot,
    )
    runtime.set_controller(controller)
    if controller is not None:
        runtime.set_controller_name(controller.get_name())
    return runtime


def main() -> None:
    parser = argparse.ArgumentParser(description="Harvest Moon editor bridge")
    parser.add_argument("--stdio", action="store_true", help="Run JSON-line bridge on stdio")
    args = parser.parse_args()
    if not args.stdio:
        parser.error("Only --stdio mode is supported")
    default_state = DEFAULT_STATE_PATH if DEFAULT_STATE_PATH.is_file() else None
    run_stdio_bridge(build_runtime(), default_state=default_state)


if __name__ == "__main__":
    main()
