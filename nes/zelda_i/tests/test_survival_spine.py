"""Continuous Survival spine — no seamed viewing compose."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import numpy as np

from zelda_i.level1_dungeon import ROOM_45_SPEC, ROOM_45_SURVIVAL_SPEC
from zelda_i.level1_finish import level1_triforce_stages
from zelda_i.level2_overworld import OverworldToLevel2Controller, PostTriforceSettleController
from zelda_i.level2_spine import (
    level2_boom_success,
    level2_through_success,
    level2_to_boom_stages,
)
from zelda_i.level2_tf_spine import level2_tf_stages
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
from zelda_i.level3_spine import (
    level3_dest_6b_stages,
    level3_entry_stages,
    level3_west_key_stages,
)
from zelda_i.survival_spine import (
    BOOT_POLICY,
    SPINE_BOMB_RETOPUP,
    SPINE_THROUGH,
    SpineRun,
    level2_entry_stages,
    merge_inventory_assist,
    spine_final_fields,
    topup_owned_inventory,
    validate_l5_endpoint,
)


def test_spine_through_is_continuous_only() -> None:
    assert SPINE_THROUGH == ("level1", "level2", "level3")


def test_through_level3_stops_at_dest_0x5b() -> None:
    """Power-on → L3 this pass is dest 0x5b (rr-4d53.3.1.2), after west key."""
    names = [name for name, _, _ in level3_entry_stages()]
    assert names == ["settle_l2_tf", "enter_level3"]
    west = [name for name, _, _ in level3_west_key_stages()]
    assert west == ["west_key"]
    dest_names = [name for name, _, _ in level3_dest_6b_stages()]
    assert dest_names == ["west_key", "north_chain"]
    assert "north_chain" not in names
    run = SpineRun(through="level3", success=True, boot_frames=199)
    assert run.report()["stop"] == "level3_dest_0x5b"
    assert "l3_entry" in run.report()


def _l2_snap(
    *,
    room: int = 0x7D,
    boom: int = 0,
    bombs: int = 0,
    keys: int = 0,
    triforce: int = 0x01,
):
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = PLAY_MODE
    ram[ADDR_LEVEL] = 2
    ram[ADDR_SCREEN] = room
    ram[ADDR_LINK_X] = 120
    ram[ADDR_LINK_Y] = 205
    ram[ADDR_KEYS] = keys
    ram[ADDR_BOMBS] = bombs
    ram[ADDR_MAGIC_BOOMERANG] = boom
    ram[ADDR_TRIFORCE] = triforce
    ram[ADDR_HEALTH] = 0x2F
    return read_snapshot(ram)


def test_through_level2_requires_triforce_bit() -> None:
    """through=level2 is TF 0x02, not merely boom or Moon entry."""
    assert not level2_through_success(_l2_snap(room=0x4F, boom=1, triforce=0x01))
    assert not level2_boom_success(_l2_snap(room=0x4F, boom=0))
    assert level2_boom_success(_l2_snap(room=0x4F, boom=1))
    assert level2_through_success(_l2_snap(room=0x0D, boom=1, triforce=0x03))


def test_level2_tf_stages_follow_isolated_boss_path() -> None:
    names = [name for name, _, _ in level2_tf_stages()]
    assert names == [
        "bomb_north_4f",
        "clear3f",
        "enter_3e",
        "clear3e",
        "enter_2e",
        "clear2e",
        "enter_1e",
        "clear1e",
        "bomb_north_1e",
        "fight_dodongo",
        "collect_tf",
    ]


def test_spine_retopup_covers_first_l2_bomb_wall() -> None:
    """Power-on L2 entry is bombs=0; 0x6f north must get the Survival top-up."""
    names = [name for name, _, _ in level2_to_boom_stages()]
    assert "bomb_north_6f" in names
    assert "bomb_north_6f" in SPINE_BOMB_RETOPUP
    assert "bomb_north_5f" in SPINE_BOMB_RETOPUP
    tf_names = [name for name, _, _ in level2_tf_stages()]
    assert "bomb_north_4f" in tf_names
    assert "bomb_north_4f" in SPINE_BOMB_RETOPUP
    assert "bomb_north_1e" in SPINE_BOMB_RETOPUP
    assert "fight_dodongo" in SPINE_BOMB_RETOPUP


def test_merge_inventory_assist_appends_writes() -> None:
    first = {
        "writes": [{"field": "bombs", "from": 0, "to": 16}],
        "notes": ["bombs=16"],
        "poke_bombs": 16,
        "poke_keys": None,
    }
    extra = {
        "writes": [{"field": "keys", "from": 1, "to": 2}],
        "notes": ["keys=2"],
        "poke_bombs": 16,
        "poke_keys": 2,
    }
    merged = merge_inventory_assist(first, extra)
    assert len(merged["writes"]) == 2
    assert merged["notes"] == ["bombs=16", "keys=2"]
    assert merged["poke_bombs"] == 16
    assert merged["poke_keys"] == 2
    assert merge_inventory_assist(None, extra) is extra


def test_topup_owned_inventory_records_poke_on_run() -> None:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_BOMBS] = 0
    ram[ADDR_KEYS] = 1
    values: dict[str, int] = {}

    class _Data:
        memory = None

        def set_value(self, key: str, value: int) -> None:
            values[key] = int(value)

    env = SimpleNamespace(
        get_ram=lambda: ram,
        unwrapped=SimpleNamespace(data=_Data(), em=None),
    )
    run = SpineRun(through="level3", success=True, boot_frames=199)
    topup_owned_inventory(env, run)
    assert run.inventory_assist is not None
    assert run.inventory_assist["poke_bombs"] == 16
    assert run.inventory_assist["poke_keys"] == 2
    assert values["bombs"] == 16
    assert values["keys"] == 2
    report = run.report()
    assert report["poke_bombs"] == 16
    assert report["poke_keys"] == 2


def test_spine_report_includes_l2_entry_bomb_plan() -> None:
    run = SpineRun(through="level2", success=True, boot_frames=199)
    run.l2_entry = spine_final_fields(_l2_snap(bombs=4, keys=0))
    from zelda_i.level2_bombs import spine_bomb_report

    run.bombs = spine_bomb_report(4, through="tf", bombs_out=1)
    run.inventory_assist = {
        "poke_bombs": 16,
        "poke_keys": 2,
        "writes": [{"field": "bombs", "from": 2, "to": 16}],
        "progression_writes": 0,
        "capacity_writes": 0,
    }
    report = run.report()
    assert report["poke_bombs"] == 16
    assert report["poke_keys"] == 2
    assert report["inventory_assist"]["poke_bombs"] == 16
    assert report["l2_entry"]["bombs"] == 4
    assert report["bombs"]["bombs_in"] == 4
    assert report["bombs"]["bombs_out"] == 1
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
