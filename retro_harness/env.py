"""
Environment setup utilities for stable-retro games.
"""

from __future__ import annotations

import zipfile
import shutil
import hashlib
import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from retro_harness.contracts import ContractBundle

import stable_retro as retro


def reset_obs(env: Any) -> tuple[Any, dict[str, Any]]:
    """Normalize gymnasium vs classic retro ``env.reset()`` to ``(obs, info)``.

    Gymnasium returns ``(obs, info)``. Older stable-retro returns ``obs`` only.
    Callers used to recopy this unwrap in every boot/probe loop.
    """
    result = env.reset()
    if isinstance(result, tuple) and len(result) == 2:
        info = result[1] if isinstance(result[1], dict) else {}
        return result[0], info
    return result, {}


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


def read_custom_state_bytes(
    game_dir: str | Path,
    game: str,
    state_name: str | None,
) -> bytes | None:
    """Load custom-integration ``*.state`` bytes, or None if not applicable.

    No-ops for missing / ``NONE`` / ``none`` names and absent files. Corrupt or
    unreadable files return None (``OSError`` / ``EOFError`` only); other errors
    propagate.
    """
    if not state_name or state_name in ("NONE", "none"):
        return None
    path = state_path(game_dir, game, state_name)
    if not path.is_file():
        return None
    try:
        return read_state_bytes(path)
    except (OSError, EOFError):
        return None


def resync_custom_state(
    env: Any,
    game_dir: str | Path,
    game: str,
    state_name: str | None,
) -> bool:
    """Re-apply a custom start state after reset to drop the free frame.

    stable-retro advances one blank frame on load/reset. Built-in package
    states are fine (play + verify both see it), but mid-session extracts used
    as custom start states need a re-apply so exact replays match.

    Returns True when state bytes were loaded and applied via ``env.em.set_state``.
    """
    data = read_custom_state_bytes(game_dir, game, state_name)
    if data is None:
        return False
    env.em.set_state(data)
    return True


def write_state_bytes(path: str | Path, state_data: bytes) -> Path:
    """Write emulator state bytes in the repository's gzip state format."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output, "wb") as handle:
        handle.write(state_data)
    return output


@dataclass(frozen=True)
class GameSpec:
    """Reusable game identity plus an optional full compatibility contract."""

    game: str
    game_dir: Path
    action_size: int = 12
    players: int | None = None
    contract: ContractBundle | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "game_dir", Path(self.game_dir).resolve())
        if self.contract is not None and not isinstance(self.contract, ContractBundle):
            raise TypeError("GameSpec contract must be a ContractBundle or None")

    @property
    def contract_digest(self) -> str | None:
        return self.contract.identity_digest if self.contract is not None else None

    def require_compatible_contract(self, expected: ContractBundle) -> None:
        """Fail closed when this environment lacks or disagrees with a model contract."""
        if self.contract is None:
            raise ValueError("GameSpec has no compatibility contract")
        expected.assert_compatible(self.contract)

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




# -- Shared ROM zip wiring (from retro_harness.rom_setup) ------------------------

ROM_EXTENSIONS = {".sfc", ".smc", ".fig", ".swc", ".nes"}
SNES_EXTENSIONS = {".sfc", ".smc", ".fig", ".swc"}
NES_EXTENSIONS = {".nes"}


def sha1_file(path: Path) -> str:
    """Return hex SHA1 of a file."""
    h = hashlib.sha1()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def extract_rom_from_zip(
    zip_path: Path,
    dest_dir: Path,
    *,
    extensions: set[str] | None = None,
) -> Path:
    """Extract the first matching ROM member from a zip into dest_dir.

    Returns:
        Path to the extracted ROM file.
    """
    allowed = extensions if extensions is not None else ROM_EXTENSIONS
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = [
            name
            for name in zf.namelist()
            if Path(name).suffix.lower() in allowed
            and not name.endswith("/")
        ]
        if not members:
            raise FileNotFoundError(f"No ROM with {sorted(allowed)} in zip: {zip_path}")
        member = members[0]
        out_name = Path(member).name
        out_path = dest_dir / out_name
        with zf.open(member) as src, out_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        return out_path


def integration_link_name(rom_path: Path) -> str:
    """Return the stable-retro ``rom.<ext>`` name for an extracted ROM."""
    suffix = rom_path.suffix.lower()
    if suffix in SNES_EXTENSIONS:
        return "rom.sfc"
    if suffix in NES_EXTENSIONS:
        return "rom.nes"
    raise ValueError(f"unsupported ROM extension for integration link: {rom_path}")


def link_rom_into_integration(
    rom_path: Path,
    integration_dir: Path,
    *,
    link_name: str | None = None,
) -> tuple[Path, Path]:
    """Symlink ROM into an integration dir and write rom.sha.

    Returns:
        (link_path, sha_path)
    """
    integration_dir.mkdir(parents=True, exist_ok=True)
    resolved_name = link_name or integration_link_name(rom_path)
    link_path = integration_dir / resolved_name
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    # Drop stale alternate-platform links so only one rom.* remains.
    for stale in integration_dir.glob("rom.*"):
        if stale.name == "rom.sha":
            continue
        if stale != link_path and (stale.is_symlink() or stale.is_file()):
            stale.unlink()
    link_path.symlink_to(rom_path.resolve())
    sha_path = integration_dir / "rom.sha"
    sha_path.write_text(sha1_file(rom_path) + "\n", encoding="utf-8")
    return link_path, sha_path


def setup_game_rom(
    *,
    shared_zip: Path,
    game_dir: Path,
    integration_name: str,
    extensions: set[str] | None = None,
) -> Path:
    """Extract shared zip ROM into game_dir/roms and wire integration.

    Returns:
        Path to the extracted ROM.
    """
    roms_dir = game_dir / "roms"
    integration_path = game_dir / "custom_integrations" / integration_name
    rom_path = extract_rom_from_zip(shared_zip, roms_dir, extensions=extensions)
    link_rom_into_integration(rom_path, integration_path)
    return rom_path


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
    "ROM_EXTENSIONS",
    "SNES_EXTENSIONS",
    "NES_EXTENSIONS",
    "sha1_file",
    "extract_rom_from_zip",
    "integration_link_name",
    "link_rom_into_integration",
    "setup_game_rom",
    "reset_obs",
]
