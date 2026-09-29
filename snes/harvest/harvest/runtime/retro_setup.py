"""Stable-retro setup helpers for the Harvest Moon integration.

The ROM itself is intentionally ignored, but Codex and local scripts should not
need to rediscover where it lives.  This module repairs the ignored
``custom_integrations/HarvestMoon-Snes/rom.sfc`` symlink from known local ROM
locations and registers the custom integration with an absolute path.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable

from harvest.paths import CUSTOM_INTEGRATIONS_DIR, GAME, GAME_DIR, PROJECT_DIR, SHARED_ROMS_DIR
from retro_harness.env import GameSpec

SCRIPT_DIR = PROJECT_DIR
INTEGRATION_PATH = CUSTOM_INTEGRATIONS_DIR
STATES_DIR = GAME_DIR
ROM_LINK = GAME_DIR / "rom.sfc"
ROM_SHA_PATH = GAME_DIR / "rom.sha"
HARVEST_GAME = GameSpec(GAME, PROJECT_DIR)

_ROM_ENV_VARS = ("HARVEST_MOON_ROM", "HM_ROM_PATH")
MUTABLE_STATE_NAMES = {"latest", "current"}
_ROM_FILENAMES = (
    "Harvest Moon.sfc",
    "Harvest Moon.smc",
    "rom.sfc",
    "rom.smc",
)


def _expected_rom_sha() -> str | None:
    try:
        text = ROM_SHA_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return text.split()[0].lower() if text else None


def _sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dedupe(paths: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
    for path in paths:
        key = os.fspath(path)
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return result


def harvest_rom_candidates() -> list[Path]:
    """Return local ROM paths worth trying, in priority order."""
    env_paths = [Path(value).expanduser() for name in _ROM_ENV_VARS if (value := os.getenv(name))]
    rom_dirs = [
        SCRIPT_DIR / "roms",
        SCRIPT_DIR.parent / "roms",
        SHARED_ROMS_DIR,
    ]
    file_paths = [directory / filename for directory in rom_dirs for filename in _ROM_FILENAMES]
    return _dedupe([*env_paths, *file_paths])


def find_harvest_rom() -> Path | None:
    """Find a local ROM matching ``rom.sha`` when possible."""
    expected = _expected_rom_sha()
    existing = [path for path in harvest_rom_candidates() if path.is_file()]
    if not existing:
        return None
    if expected is None:
        return existing[0]
    for path in existing:
        try:
            if _sha1(path).lower() == expected:
                return path
        except OSError:
            continue
    return None


def ensure_harvest_rom(*, required: bool = True, quiet: bool = False) -> Path | None:
    """Ensure stable-retro can resolve ``HarvestMoon-Snes/rom.sfc``.

    Returns the integration ROM path when ready.  If the checked-in symlink is
    broken, this repairs it to a matching ignored ROM found in known local ROM
    directories or in ``HARVEST_MOON_ROM``.
    """
    expected = _expected_rom_sha()

    if ROM_LINK.exists():
        if expected is None or _sha1(ROM_LINK).lower() == expected:
            return ROM_LINK

    if ROM_LINK.is_symlink():
        ROM_LINK.unlink()

    candidate = find_harvest_rom()
    if candidate is not None:
        GAME_DIR.mkdir(parents=True, exist_ok=True)
        if ROM_LINK.exists():
            raise FileExistsError(
                f"{ROM_LINK} exists but does not match {ROM_SHA_PATH}; "
                "move it aside or set HARVEST_MOON_ROM to the expected ROM."
            )
        relative_target = os.path.relpath(candidate.resolve(), ROM_LINK.parent)
        ROM_LINK.symlink_to(relative_target)
        if not quiet:
            print(f"[RETRO] Linked {ROM_LINK} -> {relative_target}")
        return ROM_LINK

    message = (
        "Harvest Moon ROM not found. Expected SHA1 "
        f"{expected or '<unknown>'}. Put the ROM at harvest/roms/Harvest Moon.sfc, "
        "retro_rl/roms/Harvest Moon.smc, or set HARVEST_MOON_ROM."
    )
    if required:
        raise FileNotFoundError(message)
    if not quiet:
        print(f"[RETRO] {message}")
    return None


def register_harvest_integration(retro_module, *, require_rom: bool = True) -> Path | None:
    """Register the custom integration by absolute path and optionally require ROM."""
    rom_path = ensure_harvest_rom(required=require_rom, quiet=not require_rom)
    integration = str(INTEGRATION_PATH.resolve())
    retro_module.data.Integrations._init()
    if integration not in retro_module.data.Integrations.CUSTOM_PATHS:
        retro_module.data.Integrations.add_custom_path(integration)
    return rom_path


# Seasonal map families (Maps_Graphics_Table order): each outdoor map exists
# once per season and every route, segment and landmark in this package is
# keyed on the SPRING id. Planners read a canonical id; the emulator is never
# written. 0x0F is left alone: it is also the sleep-transition map.
SEASONAL_TILEMAP_CANON = {
    0x01: 0x00, 0x02: 0x00, 0x03: 0x00,  # farm
    0x05: 0x04, 0x06: 0x04, 0x07: 0x04,  # town
    0x0D: 0x0C, 0x0E: 0x0C,              # path / fork
    0x11: 0x10, 0x12: 0x10, 0x13: 0x10,  # mountain
}


def canonical_seasonal_tilemap(tilemap: int) -> int:
    """Spring id for a summer/fall/winter outdoor map; others unchanged."""
    return SEASONAL_TILEMAP_CANON.get(int(tilemap), int(tilemap))


def canonicalize_ram_tilemap(ram):
    """Return ``ram`` with a seasonal tilemap id rewritten to its spring id.

    Read-side view for the bot only. Returns the same array when no rewrite
    is needed, so spring costs nothing.
    """
    from harvest.core.ram_catalog import LIVE_RAM_WRAM_OFFSET, field_spec

    base = int(field_spec("tilemap").address)
    for idx in (base, base + LIVE_RAM_WRAM_OFFSET):
        if idx >= len(ram):
            continue
        canon = SEASONAL_TILEMAP_CANON.get(int(ram[idx]))
        if canon is None:
            continue
        view = ram.copy()
        view[idx] = canon
        return view
    return ram


class SeasonalMapEnv:
    """Env proxy whose ``get_ram`` reports canonical (spring) map ids."""

    def __init__(self, env):
        self._env = env

    def get_ram(self):
        return canonicalize_ram_tilemap(self._env.get_ram())

    def __getattr__(self, name):
        return getattr(self._env, name)


def make_harvest_env(
    state: str | None = None,
    *,
    require_rom: bool = True,
    render_mode: str | None = "rgb_array",
    lean_step: bool = False,
    **kwargs,
):
    """Create the Harvest environment through the shared game specification.

    ``lean_step`` trims ``env.step`` to what the bot reads: it skips the
    framebuffer copy and the scenario ``compute_step`` (per-frame info dict,
    ~30% of ``env.step``; Harvest's scenario has no reward and ``done: never``).
    Steps then return ``obs=None, reward=0.0, done=False, info={}`` and
    ``env.render()`` fetches a fresh frame on demand.  Only for callers that
    never read those step results (headless soaks, no video); RAM is unchanged.
    """

    import stable_retro as retro

    register_harvest_integration(retro, require_rom=require_rom)
    kwargs.setdefault("inttype", retro.data.Integrations.ALL)
    kwargs.setdefault("use_restricted_actions", retro.Actions.ALL)
    env = HARVEST_GAME.make_env(state, render_mode=render_mode, **kwargs)
    if lean_step:
        _install_lean_step(env)
    return SeasonalMapEnv(env)


def _install_lean_step(env) -> None:
    """Make ``env.step`` skip the screen copy and scenario info (see ``make_harvest_env``)."""
    base = env.unwrapped

    def _update_obs():
        # img=None makes render() pull a fresh frame; ram non-None keeps
        # step()'s "call reset first" guard satisfied.
        base.img = None
        if base.ram is None:
            base.ram = base.get_ram()
        return None

    base._update_obs = _update_obs
    base.compute_step = lambda: (0.0, False, {})


def backup_mutable_start_state(state_name: str | None, record_name: str) -> str | None:
    """Copy mutable start states before recording and return the stable name."""
    if state_name not in MUTABLE_STATE_NAMES:
        return state_name

    source = STATES_DIR / f"{state_name}.state"
    if not source.exists():
        return state_name

    safe_record = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in record_name).strip("_")
    if not safe_record:
        safe_record = "recording"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{state_name}_backup_{safe_record}_{timestamp}"
    target = STATES_DIR / f"{backup_name}.state"
    shutil.copy2(source, target)
    return backup_name
