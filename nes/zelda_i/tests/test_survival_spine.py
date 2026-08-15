"""Continuous Survival spine — no seamed viewing compose."""

from __future__ import annotations

import pytest

from zelda_i.level1_dungeon import ROOM_45_SPEC, ROOM_45_SURVIVAL_SPEC
from zelda_i.level1_finish import level1_triforce_stages
from zelda_i.level2_overworld import OverworldToLevel2Controller, PostTriforceSettleController
from zelda_i.survival_spine import (
    BOOT_POLICY,
    SPINE_THROUGH,
    level2_entry_stages,
    validate_l5_endpoint,
)


def test_spine_through_is_continuous_only() -> None:
    assert SPINE_THROUGH == ("level1", "level2")


def test_level2_entry_stages_settle_then_moon_door() -> None:
    names = [name for name, _, _ in level2_entry_stages()]
    assert names == ["settle_l1_tf", "enter_level2"]
    stages = {name: ctl for name, ctl, _ in level2_entry_stages()}
    assert isinstance(stages["settle_l1_tf"], PostTriforceSettleController)
    enter = stages["enter_level2"]
    assert isinstance(enter, OverworldToLevel2Controller)
    assert enter.door_path is True
    assert enter.require_dungeon is True


def test_level1_stages_survival_uses_off_wall_overlay() -> None:
    clean = {name: ctl for name, ctl, _ in level1_triforce_stages(natural_entry=True)}
    survival = {
        name: ctl for name, ctl, _ in level1_triforce_stages(natural_entry=True, survival=True)
    }
    assert "clear45_key" in clean
    assert clean["clear45_key"].spec is ROOM_45_SPEC
    assert survival["clear45_key"].spec is ROOM_45_SURVIVAL_SPEC
    assert ROOM_45_SPEC.combat.avoid_walls is False
    assert ROOM_45_SURVIVAL_SPEC.combat.avoid_walls is True
    assert ROOM_45_SPEC.reward.waypoints[0] == (160, 141)
    assert (152, 189) in ROOM_45_SPEC.reward.waypoints
    assert ROOM_45_SURVIVAL_SPEC.reward.waypoints[0] == (208, 157)
    assert (152, 189) in ROOM_45_SURVIVAL_SPEC.reward.waypoints
    assert (208, 189) in ROOM_45_SURVIVAL_SPEC.reward.waypoints


def test_survival_aquamentus_tanks_fireballs() -> None:
    clean = {name: ctl for name, ctl, _ in level1_triforce_stages(natural_entry=True)}
    survival = {
        name: ctl for name, ctl, _ in level1_triforce_stages(natural_entry=True, survival=True)
    }
    assert clean["aquamentus_heart"].tank_hits is False
    assert survival["aquamentus_heart"].tank_hits is True


def test_spine_boot_policy_is_first_slot_first_quest() -> None:
    assert BOOT_POLICY == {
        "file_slot": 1,
        "quest": 1,
        "playthrough": "first",
        "file_menu_select": False,
    }


def test_validate_l5_endpoint_requires_continuous_session() -> None:
    with pytest.raises(ValueError, match="continuous"):
        validate_l5_endpoint(
            {
                "ok": True,
                "final": {"level": 5, "screen": 0x14, "triforce": 0x1C},
                "assist": {"progression_writes": 0, "capacity_writes": 0},
            }
        )
    with pytest.raises(ValueError, match="seamed"):
        validate_l5_endpoint(
            {
                "ok": True,
                "continuous_emulator_session": True,
                "seamed": True,
                "final": {"level": 5, "screen": 0x14, "triforce": 0x1C},
                "assist": {"progression_writes": 0, "capacity_writes": 0},
            }
        )
    validate_l5_endpoint(
        {
            "ok": True,
            "continuous_emulator_session": True,
            "seamed": False,
            "final": {"level": 5, "screen": 0x14, "triforce": 0x1C},
            "assist": {"progression_writes": 0, "capacity_writes": 0},
        }
    )


def test_validate_l5_endpoint_fails_closed_on_progression_write() -> None:
    with pytest.raises(ValueError, match="progression writes"):
        validate_l5_endpoint(
            {
                "ok": True,
                "continuous_emulator_session": True,
                "final": {"level": 5, "screen": 0x14, "triforce": 0x1C},
                "assist": {"progression_writes": 1, "capacity_writes": 0},
            }
        )


def test_seamed_compose_module_is_gone() -> None:
    with pytest.raises(ModuleNotFoundError):
        __import__("zelda_i.scripts.compose_honest_route_recording")
