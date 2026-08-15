"""Unit tests for L2 natural bomb budget / farm (no emulator)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

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
    BombFarmPhase,
    Level2BombFarmController,
    bomb_budget,
    bombs_from_snapshot,
    detect_poke_bombs,
    enough_bombs,
    natural_bomb_plan,
    poke_bombs_used,
    poke_kwarg_default,
    spine_bomb_flags,
    spine_bomb_report,
)
from zelda_i.level2_boss_combat import fight_dodongo
from zelda_i.level2_boss_path import run_boss_path
from zelda_i.ram import (
    ADDR_BOMBS,
    ADDR_HEALTH,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_OBJ_HP,
    ADDR_OBJ_TYPE,
    ADDR_SCREEN,
    ADDR_SWORD,
    ADDR_TRIFORCE,
    PLAY_MODE,
    read_snapshot,
)


def _ram(**fields: int) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = fields.get("mode", PLAY_MODE)
    ram[ADDR_LEVEL] = fields.get("level", 2)
    ram[ADDR_SCREEN] = fields.get("screen", 0x1E)
    ram[ADDR_LINK_X] = fields.get("x", 120)
    ram[ADDR_LINK_Y] = fields.get("y", 141)
    ram[ADDR_BOMBS] = fields.get("bombs", 0)
    ram[ADDR_SWORD] = fields.get("sword", 1)
    ram[ADDR_TRIFORCE] = fields.get("triforce", 0x01)
    health = fields.get("health")
    if health is None:
        filled = fields.get("filled", 3)
        containers_minus_1 = fields.get("containers", 4) - 1
        health = (containers_minus_1 << 4) | (filled & 0x0F)
    ram[ADDR_HEALTH] = health
    if "enemy_type" in fields:
        ram[ADDR_OBJ_TYPE + 1] = fields["enemy_type"]
        ram[ADDR_LINK_X + 1] = fields.get("enemy_x", 176)
        ram[ADDR_LINK_Y + 1] = fields.get("enemy_y", 141)
        ram[ADDR_OBJ_HP + 1] = fields.get("enemy_hp", 0x30)
    return ram


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


def test_bombs_from_snapshot_missing_is_not_zero() -> None:
    # survival_spine.json final: keys=0, room 0x7d, no bombs field
    tape = {"keys": 0, "room": 0x7D, "mode": 5, "level": 2, "x": 120, "y": 205}
    assert bombs_from_snapshot(tape) is None
    assert bombs_from_snapshot({"bombs": None}) is None
    assert bombs_from_snapshot(None) is None
    assert bombs_from_snapshot(SimpleNamespace(keys=0)) is None
    assert bombs_from_snapshot({"bombs": 4}) == 4
    assert bombs_from_snapshot(read_snapshot(_ram(bombs=0))) == 0
    assert bombs_from_snapshot(read_snapshot(_ram(bombs=8))) == 8


def test_enough_bombs_unknown_is_false() -> None:
    assert enough_bombs(None, through="tf") is False
    assert enough_bombs(2, through="boom") is False
    assert enough_bombs(3, through="boom") is True
    assert enough_bombs(5, through="tf") is False
    assert enough_bombs(6, through="tf") is True


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


def test_natural_plan_carry_at_eight() -> None:
    plan = natural_bomb_plan(8)
    assert plan.action == "carry"
    assert plan.farm_required is False
    assert plan.poke_bombs is False
    assert plan.report()["poke_bombs"] is False


def test_spine_flags_never_poke() -> None:
    flags = spine_bomb_flags()
    assert flags == {"poke_bombs": False}
    try:
        spine_bomb_flags(poke=True)
    except ValueError as exc:
        assert "cannot poke" in str(exc)
    else:
        raise AssertionError("spine_bomb_flags(poke=True) must raise")

    report = spine_bomb_report(0, through="tf", bombs_out=0)
    assert report["poke_bombs"] is False
    assert report["farm_required"] is True
    assert report["bombs_in"] == 0
    assert report["measured"]["level2_entrance"] == 0
    assert poke_bombs_used(report) is False


def test_detect_poke_bombs_from_reports() -> None:
    assert poke_bombs_used(None) is False
    assert poke_bombs_used({"poke": False, "poke_notes": ["poke=false"]}) is False
    assert poke_bombs_used({"poke_bombs": False}) is False
    assert poke_bombs_used({"poke": True}) is True
    assert poke_bombs_used({"poke_bombs": 16}) is True
    assert poke_bombs_used({"poke_notes": ["bombs=16"]}) is True
    assert poke_bombs_used({"fight": {"poke": True, "poke_notes": ["bombs=16"]}}) is True
    assert poke_bombs_used({"bomb_count_poke": True}) is True
    assert detect_poke_bombs is poke_bombs_used


def test_library_poke_defaults_off_cli_can_still_opt_in() -> None:
    assert poke_kwarg_default(fight_dodongo) is False
    assert poke_kwarg_default(run_boss_path) is False
    # Recon still has an explicit poke kwarg to pass --poke-bombs through.
    assert "poke" in fight_dodongo.__code__.co_varnames
    assert "poke" in run_boss_path.__code__.co_varnames


def test_farm_skips_when_min_zero() -> None:
    ctrl = Level2BombFarmController(min_bombs=0)
    act = ctrl.step(read_snapshot(_ram(bombs=0)))
    assert ctrl.success
    assert ctrl.phase is BombFarmPhase.DONE
    assert "farm_skipped" in ctrl.notes
    assert act.reason == "farm_done"


def test_farm_already_satisfied() -> None:
    ctrl = Level2BombFarmController(min_bombs=8)
    ctrl.step(read_snapshot(_ram(bombs=8)))
    assert ctrl.success
    assert ctrl.phase is BombFarmPhase.DONE
    assert any(n.startswith("farm_ok_") for n in ctrl.notes)
    assert ctrl.already_satisfied(read_snapshot(_ram(bombs=8)))
    assert not ctrl.already_satisfied(read_snapshot(_ram(bombs=4)))


def test_farm_patrols_and_chases_goriya() -> None:
    ctrl = Level2BombFarmController(min_bombs=8, max_frames=100)
    snap = read_snapshot(_ram(bombs=0, x=10, y=141))
    act = ctrl.step(snap)
    assert not ctrl.success
    assert ctrl.phase is BombFarmPhase.FARM
    assert "farm" in act.reason

    chase = read_snapshot(_ram(bombs=1, x=120, y=141, enemy_type=0x06, enemy_x=176))
    act = ctrl.step(chase)
    assert "farm_chase" in act.reason

    pickup = read_snapshot(_ram(bombs=2, x=120, y=141, enemy_type=0x62, enemy_x=160))
    act = ctrl.step(pickup)
    assert "farm_pickup" in act.reason

    ctrl.step(read_snapshot(_ram(bombs=8)))
    assert ctrl.success
    rep = ctrl.report()
    assert rep["start_bombs"] == 0
    assert rep["peak_bombs"] == 8
    assert rep["poke_bombs"] is False


def test_farm_death_and_wrong_screen_fail() -> None:
    death = Level2BombFarmController(min_bombs=8)
    death.step(read_snapshot(_ram(mode=17, bombs=0)))
    assert not death.success
    assert death.phase is BombFarmPhase.FAILED
    assert "link_death" in death.notes

    left = Level2BombFarmController(min_bombs=8)
    left.step(read_snapshot(_ram(screen=0x7D, bombs=0)))
    assert not left.success
    assert "left_farm_screen" in left.notes

    ow = Level2BombFarmController(min_bombs=8)
    ow.step(read_snapshot(_ram(level=0, screen=0x4A, bombs=0)))
    assert not ow.success
    assert "left_dungeon" in ow.notes


def test_farm_timeout_under_min() -> None:
    ctrl = Level2BombFarmController(min_bombs=8, max_frames=5)
    for _ in range(5):
        ctrl.step(read_snapshot(_ram(bombs=1)))
    assert not ctrl.success
    assert ctrl.phase is BombFarmPhase.FAILED
    assert any("farm_timeout" in n for n in ctrl.notes)
