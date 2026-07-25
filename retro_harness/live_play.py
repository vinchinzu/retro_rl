"""Shared live-play launcher built on top of PlaySession."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from retro_harness.env import make_env
from retro_harness.play_session import PlaySession


def play_game(
    *,
    game: str,
    state: str,
    game_dir: str | Path,
    title: str | None = None,
    scale: int = 3,
    action_size: int = 12,
    base_fps: int = 60,
    initial_speed: float = 1.0,
    bot: Callable | None = None,
    on_hud: Callable[[dict], list[str]] | None = None,
    env_factory: Callable[[], Any] | None = None,
    session_factory: Callable[..., Any] = PlaySession,
    render_mode: str = "rgb_array",
    headless: bool | None = None,
    players: int | None = None,
    **env_kwargs,
) -> None:
    """Create an env, attach a PlaySession, and run it.

    This is the thin shared entrypoint game-specific CLIs should use instead of
    open-coding `make_env(...)` + `PlaySession(...)` every time.
    """

    game_dir = Path(game_dir).resolve()
    env = env_factory() if env_factory is not None else make_env(
        game=game,
        state=state,
        game_dir=game_dir,
        render_mode=render_mode,
        players=players,
        **env_kwargs,
    )
    session = session_factory(
        env,
        game_dir=str(game_dir),
        game=game,
        scale=scale,
        title=title or f"{game}: {state}",
        bot=bot,
        headless=headless,
        action_size=action_size,
        base_fps=base_fps,
        initial_speed=initial_speed,
    )
    if on_hud is not None:
        session.on_hud = on_hud
    session.run()
