"""Play / record spine entrypoints for ALTTP randomizer work.

Default human path is JP 1.0 ``FirstPlay`` (no title/name menus).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

from retro_harness.play_spine import RunManifest, run_play_spine
from alttp_rando.paths import FIRST_PLAY_STATE, GAME, GAME_DIR, RECORDINGS_DIR


def play(
    *,
    state: str = FIRST_PLAY_STATE,
    seed: str | None = None,
    seed_settings: Mapping[str, Any] | None = None,
    title: str | None = None,
    scale: int = 3,
    headless: bool | None = None,
    bot: Callable | None = None,
    env_factory: Callable[[], Any] | None = None,
    milestone: str | None = "first_play",
    **env_kwargs: Any,
) -> RunManifest:
    """Interactive play with automatic run manifest under ``recordings/``.

    For MP4 recording + auto-boot, prefer ``python -m alttp_rando.scripts.play``
    or ``./play``.
    """
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    return run_play_spine(
        game=GAME,
        game_dir=GAME_DIR,
        package="alttp_rando",
        state=state,
        title=title or "ALTTP Rando · FirstPlay",
        seed=seed,
        seed_settings=seed_settings,
        recordings_dir=RECORDINGS_DIR,
        scale=scale,
        headless=headless,
        bot=bot,
        env_factory=env_factory,
        milestone=milestone,
        **env_kwargs,
    )


def vanilla_skill_play(
    *,
    state: str = "LinksHouseWake",
    scale: int = 3,
    **kwargs: Any,
) -> RunManifest:
    """Play against vanilla ALTTP (USA) integration for opening-route practice."""
    from retro_harness.env import make_env

    vanilla_dir = Path(__file__).resolve().parent.parent / "alttp"
    integration = "Zelda3-Snes"

    def _factory() -> Any:
        return make_env(
            game=integration,
            state=state,
            game_dir=vanilla_dir,
            render_mode="rgb_array",
        )

    return play(
        state=state,
        seed=kwargs.pop("seed", "vanilla"),
        title="ALTTP Rando spine · vanilla skills",
        scale=scale,
        env_factory=_factory,
        milestone=kwargs.pop("milestone", "vanilla_opening_practice"),
        **kwargs,
    )
