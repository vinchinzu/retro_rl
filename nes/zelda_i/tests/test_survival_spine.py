"""Continuous Survival spine — no seamed viewing compose."""

from __future__ import annotations

import pytest

import numpy as np

from zelda_i.level1_dungeon import ROOM_45_SPEC, ROOM_45_SURVIVAL_SPEC
from zelda_i.level1_finish import level1_triforce_stages
from zelda_i.level2_overworld import OverworldToLevel2Controller, PostTriforceSettleController
from zelda_i.level2_spine import level2_through_success
from zelda_i.ram import (
    ADDR_BOMBS,
    ADDR_HEALTH,
    ADDR_KEYS,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MAGIC_BOOMERANG,
    ADDR_MODE,
    ADDR_SCREEN,
    ADDR_TRIFORCE,
    PLAY_MODE,
    read_snapshot,
)
from zelda_i.survival_spine import (
    BOOT_POLICY,
    SPINE_THROUGH,
    SpineRun,
    level2_entry_stages,
    spine_final_fields,
    validate_l5_endpoint,
)


def test_spine_through_is_continuous_only() -> None:
    assert SPINE_THROUGH == ("level1", "level2")


def _l2_snap(*, room: int = 0x7D, boom: int = 0, bombs: int = 0, keys: int = 0):
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = PLAY_MODE
    ram[ADDR_LEVEL] = 2
    ram[ADDR_SCREEN] = room
    ram[ADDR_LINK_X] = 120
    ram[ADDR_LINK_Y] = 205
    ram[ADDR_KEYS] = keys
    ram[ADDR_BOMBS] = bombs
    ram[ADDR_MAGIC_BOOMERANG] = boom
    ram[ADDR_TRIFORCE] = 0x01
    ram[ADDR_HEALTH] = 0x2F
    return read_snapshot(ram)


def test_through_level2_requires_magic_boomerang() -> None:
    """through=level2 is boom owned, not merely Moon entry 0x7d."""
    assert not level2_through_success(_l2_snap(room=0x7D, boom=0))
    assert level2_through_success(_l2_snap(room=0x4F, boom=1))
    assert level2_through_success(_l2_snap(room=0x7D, boom=1))


def test_spine_report_includes_l2_entry_bomb_plan() -> None:
    run = SpineRun(through="level2", success=True, boot_frames=199)
    run.l2_entry = spine_final_fields(_l2_snap(bombs=4, keys=0))
    from zelda_i.level2_bombs import spine_bomb_report

    run.bombs = spine_bomb_report(4, through="boom", bombs_out=1)
    report = run.report()
    assert report["poke_bombs"] is False
    assert report["l2_entry"]["bombs"] == 4
    assert report["bombs"]["bombs_in"] == 4
    assert report["bombs"]["bombs_out"] == 1
    assert report["bombs"]["poke_bombs"] is False
    assert report["bombs"]["action"] == "farm"


def test_spine_final_fields_record_bombs_and_keys() -> None:
    fields = spine_final_fields(_l2_snap(room=0x4F, bombs=4, keys=2))
    assert fields["bombs"] == 4
    assert fields["keys"] == 2
    assert fields["room"] == 0x4F
    assert fields["level"] == 2
    assert fields["triforce"] == 0x01


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
