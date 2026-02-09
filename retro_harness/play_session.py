"""Generic pygame play session for any stable-retro game.

Handles: pygame init, frame rendering, speed control ([/] and TAB turbo),
state save/load (F5/F7/F8), keyboard+controller input via retro_harness.controls,
HUD overlay, bot/human hot-swap (~ key), headless mode (HEADLESS=1 env var).

Customize via hooks (on_hud, on_step, on_key_down, on_reset) or subclass.

Usage::

    env = make_env("DonkeyKongCountry-Snes", "Level1", game_dir)
    session = PlaySession(env, game_dir=game_dir, game="DonkeyKongCountry-Snes")
    session.on_hud = lambda info: ["Health: 100"]
    session.run()
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np

if TYPE_CHECKING:
    from numpy import ndarray

# Must be set before pygame import for Hyprland/Arch compatibility
os.environ.setdefault("SDL_VIDEODRIVER", "x11")

SPEED_LEVELS = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
_DEFAULT_SPEED_IDX = 2  # 1.0x
_TURBO_RENDER_INTERVAL = 8
_HUD_LINE_HEIGHT = 18
_HUD_FONT_SIZE = 16
_HUD_COLOR = (255, 255, 0)
_HUD_MARGIN = 4


class PlaySession:
    """Generic pygame play session for any stable-retro game.

    Provides a complete main loop with rendering, input, speed control,
    state save/load, and bot/human hot-swap. Games customize behavior
    through hook callables or by subclassing.
    """

    def __init__(
        self,
        env,
        *,
        game_dir: Optional[str] = None,
        game: Optional[str] = None,
        scale: int = 3,
        title: Optional[str] = None,
        bot: Optional[Callable] = None,
        headless: Optional[bool] = None,
        action_size: int = 12,
        base_fps: int = 60,
    ):
        self.env = env
        self.game_dir = game_dir
        self.game = game
        self.scale = scale
        self.title = title or game or "retro_harness"
        self.action_size = action_size
        self.base_fps = base_fps

        # Headless auto-detect from env var
        if headless is None:
            self._headless = os.environ.get("HEADLESS", "").lower() in ("1", "true", "yes")
        else:
            self._headless = headless

        # Bot
        self._bot_fn: Optional[Callable] = bot
        self._bot_active: bool = bot is not None

        # Speed / turbo
        self._speed_idx: int = _DEFAULT_SPEED_IDX
        self._speed: float = SPEED_LEVELS[_DEFAULT_SPEED_IDX]
        self._turbo: bool = False

        # Frame state
        self._frame_count: int = 0
        self._last_obs: Optional[ndarray] = None
        self._last_info: dict = {}
        self.running: bool = False

        # State save/load
        self._working_state: Optional[bytes] = None
        self._last_save_name: Optional[str] = None

        # Pygame objects (initialized in run())
        self._screen = None
        self._font = None
        self._clock = None
        self._joystick = None

        # Hooks -- all optional, defaults do nothing
        self.on_hud: Callable[[dict], list[str]] = lambda info: []
        self.on_step: Callable[[ndarray, float, bool, dict], None] = lambda *a: None
        self.on_key_down: Callable[[int], bool] = lambda key: False
        self.on_key_up: Callable[[int], bool] = lambda key: False
        self.on_reset: Callable[[], None] = lambda: None
        self.on_close: Callable[[], None] = lambda: None

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def bot_active(self) -> bool:
        return self._bot_active

    @property
    def speed(self) -> float:
        return self._speed

    @property
    def turbo(self) -> bool:
        return self._turbo

    def save_state(self, name: str = "QuickSave") -> None:
        """Save current emulator state to memory and optionally to disk."""
        self._working_state = self.env.em.get_state()
        self._last_save_name = name
        if self.game_dir and self.game:
            from retro_harness.env import save_state
            path = save_state(self.env, self.game_dir, self.game, name)
            print(f"[SAVED] {name} -> {path}")
        else:
            print(f"[SAVED] {name} (memory only)")

    def load_state(self, name: Optional[str] = None) -> None:
        """Load state from memory (fast) or from disk."""
        if self._working_state is not None:
            self.env.em.set_state(self._working_state)
            label = self._last_save_name or "working state"
            print(f"[LOADED] {label}")
        elif name and self.game_dir and self.game:
            from pathlib import Path
            state_path = (
                Path(self.game_dir) / "custom_integrations" / self.game / f"{name}.state"
            )
            if state_path.exists():
                import gzip
                with gzip.open(state_path, "rb") as f:
                    data = f.read()
                self.env.em.set_state(data)
                self._working_state = data
                self._last_save_name = name
                print(f"[LOADED] {name} from disk")
            else:
                print(f"[LOAD] state file not found: {state_path}")
        else:
            print("[LOAD] no saved state available")

    def set_bot(self, bot_fn: Optional[Callable]) -> None:
        """Set or change the bot function."""
        self._bot_fn = bot_fn
        if bot_fn is None:
            self._bot_active = False

    def run(self) -> None:
        """Main loop. Blocks until quit."""
        import pygame

        self.running = True

        # Get initial observation
        obs, info = self.env.reset()
        self._last_obs = obs
        self._last_info = info
        h, w = obs.shape[:2]

        if not self._headless:
            pygame.init()
            self._screen = pygame.display.set_mode((w * self.scale, h * self.scale))
            pygame.display.set_caption(self.title)
            self._font = pygame.font.SysFont("monospace", _HUD_FONT_SIZE)
            self._clock = pygame.time.Clock()
            from retro_harness.controls import init_controller
            self._joystick = init_controller(pygame)
        else:
            # Minimal init for headless -- no display
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            pygame.init()
            self._clock = pygame.time.Clock()

        try:
            self._main_loop(pygame, obs, info)
        finally:
            self.on_close()
            self.env.close()
            pygame.quit()

    def _main_loop(self, pg, obs: ndarray, info: dict) -> None:
        from retro_harness.controls import (
            keyboard_action,
            controller_action,
            sanitize_action,
        )

        while self.running:
            # --- Events ---
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.running = False
                elif event.type == pg.KEYDOWN:
                    self._handle_keydown(pg, event.key)
                elif event.type == pg.KEYUP:
                    self.on_key_up(event.key)

            if not self.running:
                break

            # --- Input ---
            action = self._gather_action(pg, keyboard_action, controller_action, sanitize_action)

            # --- Step ---
            step_result = self.env.step(action)
            # Handle both 4-tuple (old gym) and 5-tuple (new gym) APIs
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, reward, done, info = step_result

            self._frame_count += 1
            self._last_obs = obs
            self._last_info = info
            self.on_step(obs, reward, done, info)

            # --- Render ---
            if not self._headless:
                if not self._turbo or self._frame_count % _TURBO_RENDER_INTERVAL == 0:
                    self._render_frame(pg, obs)
                    self._draw_hud(pg, info)
                    pg.display.flip()

            # --- Tick ---
            if self._turbo:
                self._clock.tick(0)
            else:
                self._clock.tick(int(self.base_fps * self._speed))

            # --- Done ---
            if done:
                self.on_reset()
                reset_result = self.env.reset()
                if isinstance(reset_result, tuple):
                    obs, info = reset_result[0], reset_result[1] if len(reset_result) > 1 else {}
                else:
                    obs = reset_result
                    info = {}
                self._last_obs = obs
                self._last_info = info

    def _render_frame(self, pg, obs: ndarray) -> None:
        if self._screen is None:
            return
        surf = pg.surfarray.make_surface(obs.swapaxes(0, 1))
        scaled = pg.transform.scale(surf, self._screen.get_size())
        self._screen.blit(scaled, (0, 0))

    def _draw_hud(self, pg, info: dict) -> None:
        if self._screen is None or self._font is None:
            return

        lines: list[str] = []

        # Built-in status line
        speed_str = "TURBO" if self._turbo else f"{self._speed:.2g}x"
        mode_str = "BOT" if self._bot_active else "HUMAN"
        lines.append(f"F{self._frame_count} | {speed_str} | {mode_str}")

        # Game-specific lines from hook
        game_lines = self.on_hud(info)
        if game_lines:
            lines.extend(game_lines)

        for i, line in enumerate(lines):
            text_surf = self._font.render(line, True, _HUD_COLOR)
            self._screen.blit(text_surf, (_HUD_MARGIN, _HUD_MARGIN + i * _HUD_LINE_HEIGHT))

    def _gather_action(self, pg, keyboard_action, controller_action, sanitize_action):
        action = [0] * self.action_size

        if self._bot_active and self._bot_fn is not None:
            bot_action = self._bot_fn(self._last_obs, self._last_info)
            if bot_action is not None:
                return np.asarray(bot_action) if not isinstance(bot_action, np.ndarray) else bot_action

        # Human input
        if not self._headless:
            keys = pg.key.get_pressed()
            keyboard_action(keys, action, pg)
            if self._joystick:
                controller_action(self._joystick, action)
            sanitize_action(action)

        return action

    def _check_controller_hotswap(self) -> bool:
        """Check L+R+SELECT chord on controller for bot toggle."""
        if self._joystick is None:
            return False
        try:
            nb = self._joystick.get_numbuttons()
            l_btn = self._joystick.get_button(4) if nb > 4 else False
            r_btn = self._joystick.get_button(5) if nb > 5 else False
            sel_btn = self._joystick.get_button(6) if nb > 6 else False
            return bool(l_btn and r_btn and sel_btn)
        except Exception:
            return False

    def _handle_keydown(self, pg, key: int) -> None:
        # Let game-specific handler run first
        if self.on_key_down(key):
            return

        if key == pg.K_ESCAPE:
            self.running = False
        elif key == pg.K_F5:
            self.save_state()
        elif key in (pg.K_F7, pg.K_F8):
            self.load_state()
        elif key == pg.K_LEFTBRACKET:
            self._speed_idx = max(0, self._speed_idx - 1)
            self._speed = SPEED_LEVELS[self._speed_idx]
            print(f"[SPEED] {self._speed:.2g}x")
        elif key == pg.K_RIGHTBRACKET:
            self._speed_idx = min(len(SPEED_LEVELS) - 1, self._speed_idx + 1)
            self._speed = SPEED_LEVELS[self._speed_idx]
            print(f"[SPEED] {self._speed:.2g}x")
        elif key == pg.K_TAB:
            self._turbo = not self._turbo
            print(f"[{'TURBO ON' if self._turbo else 'TURBO OFF'}]")
        elif key == pg.K_BACKQUOTE:
            self._bot_active = not self._bot_active
            print(f"[{'BOT' if self._bot_active else 'HUMAN'}] mode")
        elif key == pg.K_r:
            self.on_reset()
            reset_result = self.env.reset()
            if isinstance(reset_result, tuple):
                self._last_obs = reset_result[0]
                self._last_info = reset_result[1] if len(reset_result) > 1 else {}
            else:
                self._last_obs = reset_result
                self._last_info = {}
            print("[RESET]")
