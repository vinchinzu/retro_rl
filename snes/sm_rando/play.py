"""Play / record spine entrypoints for SM randomizer work.

Interactive controls (windowed PlaySession):
  arrows = D-pad   Z=B  X=A  A=Y  S=X   TAB=turbo   [/]=speed
  F5 = quicksave to this package's SMRando-Snes integration
  F7/F8 = load / cycle working state
  R = reset to start state   ESC = quit

Default play loads ``FirstPlay`` (Ceres elevator on vanilla ROM). Recording
is on by default via ``scripts/play`` → ``play_YYYYMMDD_HHMMSS.mp4``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

from retro_harness.play_spine import RunManifest, run_play_spine
from sm_rando.paths import FIRST_PLAY_STATE, GAME, GAME_DIR, RECORDINGS_DIR


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

    Uses SMRando-Snes (vanilla SM ROM) unless ``env_factory`` is provided.
    """
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    return run_play_spine(
        game=GAME,
        game_dir=GAME_DIR,
        package="sm_rando",
        state=state,
        title=title or f"SM Rando · {state}",
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
    state: str = "FirstAction",
    scale: int = 3,
    **kwargs: Any,
) -> RunManifest:
    """Play against vanilla Super Metroid integration (skill practice).

    Uses the mature ``super_metroid`` custom integration so room practice is
    fun immediately while seed ROM tooling is still scaffolding.
    """
    from retro_harness.env import make_env

    vanilla_dir = Path(__file__).resolve().parent.parent / "super_metroid"
    integration = "SuperMetroid-Snes"

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
        title="SM Rando spine · vanilla skills",
        scale=scale,
        env_factory=_factory,
        milestone=kwargs.pop("milestone", "vanilla_skill_practice"),
        **kwargs,
    )
