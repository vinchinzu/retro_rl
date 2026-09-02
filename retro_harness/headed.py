"""Repo-wide headed watch for any stable-retro env.

Headless probes and duals have no window. ``--headed`` is the one flag that
opens one and plays the bot. Any CLI:

    from retro_harness.headed import add_headed_flag, attach_headed, idle_headed

    add_headed_flag(parser)
    ...
    if args.headed:
        pygame_mod = attach_headed(env, title="BOT", hud=hud_fn)
    try:
        play(env)
    finally:
        if args.headed:
            idle_headed(env, pygame_mod)

``PlaySession`` is the interactive human+bot loop; this module only attaches a
watch to an existing ``env.step``. Do not copy a per-game pygame loop.
"""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Callable
from typing import Any

HEADED_FLAG_HELP = (
    "Open a pygame window and play (bot on). Arch/Hyprland/Wayland. "
    "[ ] speed, TAB turbo. One repo-wide flag — not a per-game probe switch."
)
HEADED_ATTR = "_retro_headed"
SPEED_LEVELS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)
DEFAULT_BOT_SPEED = 1.0
DISPLAY_FPS = 60
UNTHROTTLED_FROM = 8.0
TURBO_PREVIEW_INTERVAL = 8


def configure_headless() -> None:
    """Dummy SDL for probe CLIs. Games may pop extra env keys after this."""
    os.environ.setdefault("HEADLESS", "1")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


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


def add_headed_flag(
    parser: argparse.ArgumentParser,
    *,
    help: str | None = None,
) -> argparse.Action:
    """Add ``--headed`` to any game CLI. Same flag name everywhere."""
    return parser.add_argument(
        "--headed",
        action="store_true",
        help=help or HEADED_FLAG_HELP,
    )


def default_speed_index(speed: float = DEFAULT_BOT_SPEED) -> int:
    return min(range(len(SPEED_LEVELS)), key=lambda idx: abs(SPEED_LEVELS[idx] - speed))


def bot_speed_timing(
    speed: float,
    *,
    turbo: bool = False,
    bot: bool = True,
) -> tuple[int, int, bool]:
    """Return ``(emu_repeat, clock_tick_fps, skip_most_presents)``.

    ``clock_tick_fps`` 0 means unthrottled. Bot 2x/4x repeats that many emu
    frames per 60 Hz present so [ ] is real even when the compositor vsyncs.
    Callers must loop ``headed_emu_repeat(env)``; wrapping ``env.step`` stays
    one emu frame so SM hops stay 1:1 at default 1x.
    """
    speed = float(speed)
    if turbo or speed >= UNTHROTTLED_FROM:
        return 1, 0, True
    if bot and speed >= 2.0:
        return max(1, int(round(speed))), DISPLAY_FPS, False
    tick = max(1, int(round(DISPLAY_FPS * speed)))
    return 1, tick, False


def pace_present(tick_fps: int, holder: Any) -> None:
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


def headed_emu_repeat(env: Any) -> int:
    """How many ``env.step`` calls the probe should make per present."""
    state = getattr(env, HEADED_ATTR, None)
    if state is None:
        return 1
    return int(state.emu_repeat())


class _HeadedState:
    def __init__(
        self,
        pygame_mod: Any,
        screen: Any,
        *,
        title: str,
        scale: int,
        hud: Callable[[Any], str] | None,
        speed: float,
        w: int,
        h: int,
    ) -> None:
        self._pg = pygame_mod
        self._screen = screen
        self.title = title
        self.scale = scale
        self.hud = hud
        self.w = w
        self.h = h
        self.speed_idx = default_speed_index(speed)
        self.speed = float(SPEED_LEVELS[self.speed_idx])
        self.frame = 0
        self._since_present = 0
        self._next_present = 0.0
        self._font = pygame_mod.font.SysFont("monospace", 16)
        self._set_caption()

    def emu_repeat(self) -> int:
        repeat, _tick, _skip = bot_speed_timing(
            self.speed, turbo=self.tab_held(), bot=True
        )
        return repeat

    def tab_held(self) -> bool:
        return bool(self._pg.key.get_pressed()[self._pg.K_TAB])

    def after_emu(self, env: Any) -> None:
        self.pump()
        self.frame += 1
        self._since_present += 1
        turbo = self.tab_held()
        repeat, tick_fps, skip = bot_speed_timing(
            self.speed, turbo=turbo, bot=True
        )
        if skip:
            should_blit = self.frame % TURBO_PREVIEW_INTERVAL == 0
            should_pace = True
            tick_fps = 0
        elif self._since_present < repeat:
            should_blit = False
            should_pace = False
        else:
            should_blit = True
            should_pace = True
        if should_blit:
            self._blit(env)
        elif should_pace:
            self._pg.event.pump()
        if should_pace:
            pace_present(tick_fps, self)
            self._since_present = 0

    def pump(self) -> None:
        pygame = self._pg
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt("headed window closed")
            if event.type != pygame.KEYDOWN:
                continue
            if event.key == pygame.K_ESCAPE:
                raise KeyboardInterrupt("headed window closed")
            if event.key in (pygame.K_LEFTBRACKET, pygame.K_COMMA, pygame.K_MINUS):
                self._nudge(-1)
            elif event.key in (
                pygame.K_RIGHTBRACKET,
                pygame.K_PERIOD,
                pygame.K_EQUALS,
                pygame.K_PLUS,
            ):
                self._nudge(1)

    def _nudge(self, delta: int) -> None:
        self.speed_idx = max(0, min(len(SPEED_LEVELS) - 1, self.speed_idx + delta))
        self.speed = float(SPEED_LEVELS[self.speed_idx])
        print(f"[SPEED] {self.speed:g}x", flush=True)
        self._set_caption()

    def _set_caption(self) -> None:
        self._pg.display.set_caption(
            f"{self.title}  {self.speed:g}x  [ ] speed  TAB turbo  ESC quit"
        )

    def _blit(self, env: Any) -> None:
        import numpy as np

        pygame = self._pg
        frame = env.render()
        surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        size = (self.w * self.scale, self.h * self.scale)
        if self.scale != 1:
            surf = pygame.transform.scale(surf, size)
        self._screen.blit(surf, (0, 0))
        if self.hud is not None:
            self._screen.blit(
                self._font.render(self.hud(env), True, (255, 255, 0)), (8, 8)
            )
        pygame.display.flip()


def attach_headed(
    env: Any,
    *,
    title: str = "BOT",
    scale: int = 3,
    fps: int = 60,
    hud: Callable[[Any], str] | None = None,
    speed: float = DEFAULT_BOT_SPEED,
) -> Any:
    """Blit ``env.step`` to a pygame window. Returns the pygame module.

    ``fps`` is kept for callers; 2x/4x uses frame-repeat + 60 Hz present, not
    ``Clock.tick(60 * speed)`` (Wayland vsync would pin that to 1x).
    """
    del fps
    configure_headed()
    import pygame

    pygame.init()
    shape = getattr(getattr(env, "observation_space", None), "shape", None)
    if shape is not None and len(shape) >= 2:
        h, w = int(shape[0]), int(shape[1])
    else:
        h, w = 224, 256
    try:
        screen = pygame.display.set_mode((w * scale, h * scale))
    except pygame.error:
        os.environ["SDL_VIDEODRIVER"] = "x11"
        pygame.display.quit()
        pygame.init()
        screen = pygame.display.set_mode((w * scale, h * scale))
    state = _HeadedState(
        pygame,
        screen,
        title=title,
        scale=scale,
        hud=hud,
        speed=speed,
        w=w,
        h=h,
    )
    setattr(env, HEADED_ATTR, state)
    orig = env.step

    def step(action):
        out = orig(action)
        state.after_emu(env)
        return out

    env.step = step  # type: ignore[method-assign]
    print(
        f"[HEADED] window up SDL_VIDEODRIVER={os.environ.get('SDL_VIDEODRIVER')} "
        f"scale={scale} {state.speed:g}x — {title}",
        flush=True,
    )
    return pygame


def idle_headed(
    env: Any,
    pygame_mod: Any,
    *,
    frames: int = 3600,
) -> None:
    """Keep the window open after the bot so the stuck pose is visible."""
    from retro_harness.actions import idle_action

    try:
        for _ in range(frames):
            env.step(idle_action())
    except KeyboardInterrupt:
        pass
    try:
        pygame_mod.quit()
    except Exception:  # noqa: BLE001
        pass


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


def gym_env_step(env, action, obs, *, update_obs: bool):
    """Gym-shaped ``(obs, reward, terminated, truncated, info)`` around :func:`fast_env_step`."""
    if env.img is None and env.ram is None:
        raise RuntimeError("Please call env.reset() before stepping")
    new_obs = fast_env_step(env, action, update_obs=update_obs)
    if update_obs and new_obs is not None:
        obs = new_obs
    try:
        terminated = bool(env.data.is_done())
    except Exception:
        terminated = False
    return obs, 0.0, terminated, False, {}


def preview_interval(speed: float) -> int:
    """How often to blit when high-speed autoplay skips most presents."""
    if speed <= 4.0:
        return 1
    if speed <= 8.0:
        return 30
    if speed <= 32.0:
        return 45
    return 60


class WatchDisplay:
    """Probe-side pygame window: [ ] speed, TAB turbo, ESC/close.

    Unlike :func:`attach_headed`, this does not wrap ``env.step``. Probes
    that step the emulator themselves call :meth:`present` after each
    batch. Default speed is 4x (frame-repeat); hop probes that need 1:1
    should use :func:`attach_headed`.
    """

    def __init__(
        self,
        *,
        scale: int = 3,
        title: str = "BOT",
        speed: float = 4.0,
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
        import numpy as np

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
        import numpy as np

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
