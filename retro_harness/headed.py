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
from collections.abc import Callable
from typing import Any

HEADED_FLAG_HELP = (
    "Open a pygame window and play (bot on). Arch/Hyprland/Wayland. "
    "One repo-wide flag — not a per-game probe switch."
)


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


def attach_headed(
    env: Any,
    *,
    title: str = "BOT",
    scale: int = 3,
    fps: int = 60,
    hud: Callable[[Any], str] | None = None,
) -> Any:
    """Blit every ``env.step`` to a pygame window. Returns the pygame module."""
    configure_headed()
    import numpy as np
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
    pygame.display.set_caption(title)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 16)
    orig = env.step

    def step(action):
        out = orig(action)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt("headed window closed")
        frame = env.render()
        surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        if scale != 1:
            surf = pygame.transform.scale(surf, (w * scale, h * scale))
        screen.blit(surf, (0, 0))
        if hud is not None:
            screen.blit(font.render(hud(env), True, (255, 255, 0)), (8, 8))
        pygame.display.flip()
        clock.tick(fps)
        return out

    env.step = step  # type: ignore[method-assign]
    print(
        f"[HEADED] window up SDL_VIDEODRIVER={os.environ.get('SDL_VIDEODRIVER')} "
        f"scale={scale} — {title}",
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
