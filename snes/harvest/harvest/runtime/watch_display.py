"""Headed pygame watch + bot [ ] speed (2x/4x via frame-repeat).

Harvest's pygame window vsyncs on Wayland/X11. Stepping one emu frame then
``Clock.tick(60 * speed)`` cannot beat the display refresh, so 2x/4x looked
like 1x. The editor already uses ``speed_uses_frame_repeat``; bot play and
probe ``--watch`` do the same: N emu steps per 60 Hz present.
"""

from __future__ import annotations

import os
import time

import numpy as np

SPEED_LEVELS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)
DEFAULT_BOT_SPEED = 4.0
DISPLAY_FPS = 60
UNTHROTTLED_FROM = 8.0
TURBO_PREVIEW_INTERVAL = 8


def configure_headed() -> None:
    """Drop dummy/HEADLESS drivers and pick a real SDL video backend."""
    os.environ.pop("HEADLESS", None)
    driver = os.environ.get("SDL_VIDEODRIVER", "").lower()
    if driver == "dummy":
        os.environ.pop("SDL_VIDEODRIVER", None)
        driver = ""
    if "SDL_VIDEODRIVER" not in os.environ:
        if os.environ.get("WAYLAND_DISPLAY"):
            os.environ["SDL_VIDEODRIVER"] = "wayland"
        elif os.environ.get("DISPLAY"):
            os.environ["SDL_VIDEODRIVER"] = "x11"
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("SDL_HINT_RENDER_VSYNC", "0")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")


def configure_headless() -> None:
    os.environ.setdefault("HEADLESS", "1")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.pop("INFINITE_STAMINA", None)


def default_speed_index(speed: float = DEFAULT_BOT_SPEED) -> int:
    levels = SPEED_LEVELS
    return min(range(len(levels)), key=lambda idx: abs(levels[idx] - speed))


def pace_present(tick_fps: int, holder) -> None:
    """Sleep only the leftover time in the present slot.

    ``Clock.tick(60)`` after a vsync ``flip()`` double-waits and pins 2x/4x to
    1x. Subtract time already spent in emu steps + blit.
    """
    now = time.perf_counter()
    if tick_fps <= 0:
        holder._next_present = now
        return
    target_dt = 1.0 / float(tick_fps)
    next_t = float(getattr(holder, "_next_present", 0.0) or 0.0)
    target = next_t + target_dt
    if target < now - target_dt:
        target = now
    while target > now:
        time.sleep(min(0.002, target - now))
        now = time.perf_counter()
    holder._next_present = target


def bot_speed_timing(
    speed: float,
    *,
    turbo: bool = False,
    bot: bool = True,
) -> tuple[int, int, bool]:
    """Return ``(emu_repeat, clock_tick_fps, skip_most_presents)``.

    ``clock_tick_fps`` 0 means unthrottled (``Clock.tick(0)``).
    Bot 2x/4x repeats that many emu frames per 60 Hz present so [ ] is real
    even when the compositor vsyncs. Human play stays 1 emu step / loop.
    """
    speed = float(speed)
    if turbo or speed >= UNTHROTTLED_FROM:
        return 1, 0, True
    if bot and speed >= 2.0:
        return max(1, int(round(speed))), DISPLAY_FPS, False
    tick = max(1, int(round(DISPLAY_FPS * speed)))
    return 1, tick, False


def fast_env_step(env, action, *, update_obs: bool):
    """Step stable-retro; skip the per-frame info dict and optional blit obs."""
    for player, player_action in enumerate(env.action_to_array(action)):
        if env.movie:
            for button_idx in range(env.num_buttons):
                env.movie.set_key(button_idx, player_action[button_idx], player)
        env.em.set_button_mask(player_action, player)
    if env.movie:
        env.movie.step()
    env.em.step()
    env.data.update_ram()
    if update_obs:
        return env._update_obs()
    return None


def display_set_mode(pygame, size: tuple[int, int], *, caption: str | None = None):
    """Open a window with vsync off; fall back from Wayland to X11."""

    def _open():
        try:
            return pygame.display.set_mode(size, vsync=0)
        except TypeError:
            return pygame.display.set_mode(size)

    try:
        screen = _open()
    except pygame.error:
        if os.environ.get("SDL_VIDEODRIVER") == "wayland" and os.environ.get("DISPLAY"):
            pygame.display.quit()
            os.environ["SDL_VIDEODRIVER"] = "x11"
            pygame.display.init()
            screen = _open()
        else:
            raise
    if caption:
        pygame.display.set_caption(caption)
    return screen


class WatchDisplay:
    """Probe-side pygame window: [ ] speed, TAB turbo, ESC/close."""

    def __init__(
        self,
        *,
        scale: int = 3,
        title: str = "Harvest",
        speed: float = DEFAULT_BOT_SPEED,
    ) -> None:
        self.scale = max(1, int(scale))
        self.title = title
        self.speed_idx = default_speed_index(speed)
        self.speed = float(SPEED_LEVELS[self.speed_idx])
        self.closed = False
        self._pg = None
        self._screen = None
        self._obs = None
        self._next_present = 0.0

    def start(self, obs) -> bool:
        configure_headed()
        import pygame

        pygame.init()
        self._pg = pygame
        arr = np.asarray(obs)
        if arr.ndim < 2:
            raise ValueError(f"expected image obs, got shape={arr.shape}")
        h, w = int(arr.shape[0]), int(arr.shape[1])
        caption = f"{self.title}  {self.speed:g}x  [ ] speed  TAB turbo  ESC quit"
        try:
            self._screen = display_set_mode(
                pygame, (w * self.scale, h * self.scale), caption=caption
            )
        except pygame.error as exc:
            print(f"[WATCH] display failed: {exc}", flush=True)
            self.closed = True
            return False
        print(
            f"[WATCH] {self.title} {self.speed:g}x  "
            "[ ] = speed down/up | TAB = turbo | ESC = quit",
            flush=True,
        )
        return self.present(obs, emu_frame=0)

    def pump(self) -> bool:
        if self.closed or self._pg is None:
            return False
        pygame = self._pg
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.closed = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.closed = True
                elif event.key in (pygame.K_LEFTBRACKET, pygame.K_COMMA, pygame.K_MINUS):
                    self._nudge_speed(-1)
                elif event.key in (
                    pygame.K_RIGHTBRACKET,
                    pygame.K_PERIOD,
                    pygame.K_EQUALS,
                    pygame.K_PLUS,
                ):
                    self._nudge_speed(1)
        return not self.closed

    def tab_held(self) -> bool:
        if self._pg is None or self.closed:
            return False
        return bool(self._pg.key.get_pressed()[self._pg.K_TAB])

    def emu_repeat(self) -> int:
        repeat, _tick, _skip = bot_speed_timing(
            self.speed, turbo=self.tab_held(), bot=True
        )
        return repeat

    def present(self, obs, *, emu_frame: int) -> bool:
        if not self.pump():
            return False
        self._obs = obs
        pygame = self._pg
        _repeat, tick_fps, skip = bot_speed_timing(
            self.speed, turbo=self.tab_held(), bot=True
        )
        should_blit = (not skip) or (int(emu_frame) % TURBO_PREVIEW_INTERVAL == 0)
        if should_blit and obs is not None:
            self._blit(obs)
        elif self._screen is not None:
            pygame.event.pump()
        pace_present(tick_fps, self)
        return not self.closed

    def close(self) -> None:
        self.closed = True
        if self._pg is not None:
            try:
                self._pg.quit()
            except Exception:
                pass
            self._pg = None
            self._screen = None

    def _nudge_speed(self, delta: int) -> None:
        self.speed_idx = max(0, min(len(SPEED_LEVELS) - 1, self.speed_idx + delta))
        self.speed = float(SPEED_LEVELS[self.speed_idx])
        print(f"[SPEED] {self.speed:g}x", flush=True)
        if self._pg is not None:
            self._pg.display.set_caption(
                f"{self.title}  {self.speed:g}x  [ ] speed  TAB turbo  ESC quit"
            )

    def _blit(self, obs) -> None:
        pygame = self._pg
        screen = self._screen
        if pygame is None or screen is None:
            return
        arr = np.asarray(obs)
        if arr.ndim != 3 or arr.shape[-1] < 3:
            return
        frame = arr[..., :3]
        h, w = frame.shape[0], frame.shape[1]
        surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        size = (w * self.scale, h * self.scale)
        if screen.get_size() != size:
            screen = display_set_mode(pygame, size)
            self._screen = screen
        scaled = pygame.transform.scale(surf, size)
        screen.blit(scaled, (0, 0))
        pygame.display.flip()
