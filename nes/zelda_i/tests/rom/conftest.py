"""Shared harness for the ``zelda_i`` real-ROM test tier.

Every test under this directory is ``@pytest.mark.rom``: it boots the actual
NES ROM (or a named ``.state`` pin of it) via stable-retro and asserts on the
live RAM the ROM wrote (screen id, triforce byte, inventory counters, enemy
census). ``pyproject.toml`` deselects ``rom`` by default
(``-m "not ml and not rom"``), so this tier is opt-in:

    uv run pytest nes/zelda_i/tests/rom -m rom -q

See ``docs/TEST_TIERS.md`` for the policy this tier exists to enforce, and
``nes/zelda_i/tests/test_level7_hungry.py::test_live_feed_from_interior28_recon_fixture``
for the template every test here copies: ``configure_headless()`` ->
``make_env`` -> ``reset_obs`` -> step -> assert on ``read_snapshot(env.get_ram())``.

Traps (see ``nes/zelda_i/AGENTS.md``):
- One emulator per process. Each test opens and closes its own ``env``.
- ``$066F`` low nibble is whole hearts MINUS ONE; never assert an exact heart
  count from a pin without checking ``health_byte_is_coherent`` first
  (``nes/zelda_i/scripts/audit_pins.py``).
- Map rooms from ``$6530``, never ``$049E``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from retro_harness.env import state_path
from retro_harness.segment_runner import configure_headless
from zelda_i.paths import GAME, GAME_DIR, SHARED_ROM_ZIP

ROM = pytest.mark.rom


def pin_available(name: str) -> bool:
    """True when the shared ROM zip and the named ``.state`` pin both exist.

    Mirrors ``test_level7_hungry.py::_live_ready`` so a missing ROM or a
    missing/renamed fixture skips cleanly instead of erroring.
    """
    if not SHARED_ROM_ZIP.is_file():
        return False
    try:
        return bool(state_path(GAME_DIR, GAME, name).is_file())
    except Exception:
        return False


def rom_ready() -> bool:
    """True when the shared ROM zip alone is present (power-on spine tests)."""
    return SHARED_ROM_ZIP.is_file()


def skip_unless_pin(name: str):
    return pytest.mark.skipif(
        not pin_available(name), reason=f"Zelda I ROM or {name!r} fixture missing"
    )


def skip_unless_rom():
    return pytest.mark.skipif(
        not rom_ready(), reason="Zelda I ROM zip missing"
    )


@pytest.fixture(autouse=True, scope="session")
def _headless_once() -> None:
    configure_headless()
