"""Matrix checks for the boss catalog and available strategy implementations.

Parametrized over every ``BOSS_CATALOG`` id:
- each entry declares a positive ``room_id`` and a matching ``boss_id``
- if ``combat/<boss_id>.py`` exists, the package imports cleanly
- if ``wrap_<boss_id>_as_boss_strategy`` exists (SM-BOSS-WRAP-01+), it returns
  a strategy whose catalog room matches the registry (soft: skip when absent)
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from super_metroid.combat.features import BOSS_CATALOG
from super_metroid.combat import protocol


CATALOG_IDS = tuple(sorted(BOSS_CATALOG))
COMBAT_PACKAGE_DIR = Path(__file__).parents[1] / "combat"


@pytest.mark.parametrize("boss_id", CATALOG_IDS)
def test_catalog_entry_has_room_id_and_existing_strategy_imports_cleanly(
    boss_id: str,
) -> None:
    entry = BOSS_CATALOG[boss_id]
    assert entry.boss_id == boss_id
    assert isinstance(entry.room_id, int)
    assert entry.room_id > 0, f"{boss_id} must declare a valid room_id"

    module_path = COMBAT_PACKAGE_DIR / f"{boss_id}.py"
    if module_path.is_file():
        importlib.import_module(f"super_metroid.combat.{boss_id}")


@pytest.mark.parametrize("boss_id", CATALOG_IDS)
def test_available_boss_wrapper_matches_catalog(boss_id: str) -> None:
    """Soft-check: wrap_* presence is optional (skip when not registered)."""
    wrapper = getattr(protocol, f"wrap_{boss_id}_as_boss_strategy", None)
    if wrapper is None:
        pytest.skip(f"no wrapper registered for {boss_id}")

    strategy = wrapper()
    assert strategy.boss_id == boss_id
    assert strategy.catalog.room_id == BOSS_CATALOG[boss_id].room_id
    assert strategy.entry.room_id == strategy.catalog.room_id
