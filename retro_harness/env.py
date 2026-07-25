"""
Environment setup utilities for stable-retro games.
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import stable_retro as retro


def integration_dir(game_dir: str | Path, game: str | None = None) -> Path:
    """Return a game's custom integration root or one integration directory."""

    root = Path(game_dir).resolve() / "custom_integrations"
    return root / game if game else root


def state_path(game_dir: str | Path, game: str, name: str) -> Path:
    """Return the canonical path for a named development save state."""

    filename = name if name.endswith(".state") else f"{name}.state"
    return integration_dir(game_dir, game) / filename


def read_state_bytes(path: str | Path) -> bytes:
    """Read a raw or gzip-compressed stable-retro state."""

    raw = Path(path).read_bytes()
    return gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw


def write_state_bytes(path: str | Path, state_data: bytes) -> Path:
    """Write emulator state bytes in the repository's gzip state format."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output, "wb") as handle:
        handle.write(state_data)
    return output


@dataclass(frozen=True)
class GameSpec:
    """The small reusable identity/config object every game can start with."""

    game: str
    game_dir: Path
    action_size: int = 12
    players: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "game_dir", Path(self.game_dir).resolve())

    @property
    def integrations(self) -> Path:
        return integration_dir(self.game_dir)

    @property
    def states_dir(self) -> Path:
        return integration_dir(self.game_dir, self.game)

    def state_path(self, name: str) -> Path:
        return state_path(self.game_dir, self.game, name)

    def available_states(self) -> list[str]:
        return get_available_states(self.game, self.game_dir)

    def make_env(
        self,
        state: str | None = None,
        *,
        render_mode: str | None = "rgb_array",
        **kwargs: Any,
    ) -> retro.RetroEnv:
        return make_env(
            game=self.game,
            state=state,
            game_dir=self.game_dir,
            render_mode=render_mode,
            players=self.players,
            **kwargs,
        )

    def save_state(self, env: retro.RetroEnv, name: str) -> Path:
        return save_state(env, self.game_dir, self.game, name)


def add_custom_integrations(game_dir: str | Path) -> Path:
    """Add custom integrations path for a game directory.

    Args:
        game_dir: Path to the game directory containing custom_integrations/

    Returns:
        Path to the custom_integrations directory
    """
    integrations_path = integration_dir(game_dir)
    if integrations_path.exists():
        retro.data.Integrations.add_custom_path(str(integrations_path))
    return integrations_path


def make_env(
    game: str,
    state: str | None,
    game_dir: str | Path,
    render_mode: str | None = "rgb_array",
    players: Optional[int] = None,
    **kwargs,
) -> retro.RetroEnv:
    """Create a stable-retro environment with custom integrations.

    This automatically:
    - Adds the custom_integrations path for the game
    - Uses CUSTOM inttype to find custom states while allowing stable/imported ROM fallback

    Args:
        game: Game identifier (e.g., "DonkeyKongCountry-Snes")
        state: State name (e.g., "1Player.CongoJungle.JungleHijinks.Level1")
        game_dir: Path to the game directory containing custom_integrations/
        render_mode: Render mode ("rgb_array" or "human")
        **kwargs: Additional arguments passed to retro.make()

    Returns:
        Configured RetroEnv instance
    """
    add_custom_integrations(game_dir)

    # Handle special state values
    if state is None or state == "NONE":
        state = retro.State.NONE

    # Custom integrations often provide only states/scenario metadata and rely
    # on the stable/imported ROM entry for the actual rom.nes. Include STABLE
    # in the lookup set so custom states can fall back to stable ROM files.
    kwargs.setdefault("inttype", retro.data.Integrations.CUSTOM)

    # Default to Actions.ALL so SELECT/START reach the emulator. The
    # stable-retro default (Actions.FILTERED) strips any button not named
    # in the core's action combo list; snes9x.json omits SELECT and START,
    # which breaks in-game menus (e.g. UWNH save menu, Super Metroid item
    # select, SNES pause). Callers can still override via kwargs.
    kwargs.setdefault("use_restricted_actions", retro.Actions.ALL)

    make_kwargs = dict(
        game=game,
        state=state,
        render_mode=render_mode,
        **kwargs,
    )
    if players is not None:
        make_kwargs["players"] = players

    try:
        return retro.make(**make_kwargs)
    except TypeError:
        # Fallback for retro versions without players arg
        make_kwargs.pop("players", None)
        return retro.make(**make_kwargs)


def get_available_states(game: str, game_dir: str | Path) -> list[str]:
    """List available save states for a game.

    Args:
        game: Game identifier
        game_dir: Path to the game directory

    Returns:
        List of state names (without .state extension)
    """
    integrations_path = integration_dir(game_dir, game)

    if not integrations_path.exists():
        return []

    states = []
    for state_file in integrations_path.glob("*.state"):
        states.append(state_file.stem)
    return sorted(states)


def save_state(env: retro.RetroEnv, game_dir: str | Path, game: str, name: str) -> Path:
    """Save current emulator state to the game's custom integrations directory.

    Args:
        env: active RetroEnv
        game_dir: Path to game directory
        game: Game identifier (e.g., "DonkeyKongCountry-Snes")
        name: State base name (without .state)

    Returns:
        Path to the saved state in custom_integrations
    """
    state_data = env.em.get_state()
    return write_state_bytes(state_path(game_dir, game, name), state_data)


__all__ = [
    "GameSpec",
    "add_custom_integrations",
    "get_available_states",
    "integration_dir",
    "make_env",
    "read_state_bytes",
    "save_state",
    "state_path",
    "write_state_bytes",
]
