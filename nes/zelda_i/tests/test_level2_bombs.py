"""Unit tests for L2 natural bomb budget / farm (no emulator)."""

from __future__ import annotations

from zelda_i.level2_bombs import (
    L1_COMPLETE_BOMBS_MEASURED,
    L2_BOMB_BUDGET,
    L2_BOMB_BUDGET_BOOM,
    L2_BOMB_CARRY,
    L2_BOMB_FARM_ROOMS,
    L2_BOMB_FARM_SCREEN,
    L2_BOMB_WALLS,
    L2_DODONGO_MOUTHS,
    L2_ENTRY_BOMBS_MEASURED,
    bomb_budget,
    natural_bomb_plan,
    poke_kwarg_default,
)
from zelda_i.level2_boss_combat import fight_dodongo
from zelda_i.level2_boss_path import run_boss_path


def test_budget_is_six_successful_placements() -> None:
    assert L2_BOMB_WALLS == (
        (0x6F, 0x5F),
        (0x5F, 0x4F),
        (0x4F, 0x3F),
        (0x1E, 0x0E),
    )
    assert L2_DODONGO_MOUTHS == 2
    assert L2_BOMB_BUDGET == 6
    assert L2_BOMB_BUDGET_BOOM == 3
    assert L2_BOMB_CARRY == 8
    assert bomb_budget(through="boom") == 3
    assert bomb_budget(through="tf") == 6
    assert L2_BOMB_FARM_ROOMS == (0x4F, 0x3E, 0x1E)
    assert L2_BOMB_FARM_SCREEN == 0x1E


def test_measured_l2_entry_is_zero() -> None:
    assert L2_ENTRY_BOMBS_MEASURED == 0
    assert L1_COMPLETE_BOMBS_MEASURED == 4


def test_natural_plan_farm_when_short_or_unknown() -> None:
    unknown = natural_bomb_plan(None)
    assert unknown.action == "farm"
    assert unknown.farm_required
    assert unknown.poke_bombs is False
    assert "unknown_bombs_in" in unknown.notes
    assert "missing_field_is_not_zero" in unknown.notes

    entry = natural_bomb_plan(L2_ENTRY_BOMBS_MEASURED)
    assert entry.action == "farm"
    assert entry.farm_required
    assert any("pre_farm_short" in n for n in entry.notes)
    assert any("farm_after_0x4f_0x3e" in n for n in entry.notes)

    l1 = natural_bomb_plan(L1_COMPLETE_BOMBS_MEASURED)
    assert l1.action == "farm"
    assert l1.bombs_in == 4
    assert not any("pre_farm_short" in n for n in l1.notes)


def test_library_poke_defaults_off_cli_can_still_opt_in() -> None:
    assert poke_kwarg_default(fight_dodongo) is False
    assert poke_kwarg_default(run_boss_path) is False
    # Recon still has an explicit poke kwarg to pass --poke-bombs through.
    assert "poke" in fight_dodongo.__code__.co_varnames
    assert "poke" in run_boss_path.__code__.co_varnames
