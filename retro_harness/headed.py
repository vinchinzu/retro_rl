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
