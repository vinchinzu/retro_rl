"""Continuous Survival spine — assist, boot policy, and unique survival contracts."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from zelda_i.level1.finish import level1_triforce_stages
from zelda_i.level2.spine import level2_to_boom_stages
from zelda_i.level2.tf_spine import level2_tf_stages
from zelda_i.ram import (
    ADDR_BOMBS,
    ADDR_KEYS,
)
from zelda_i.spine.survival import (
    BOOT_POLICY,
    SPINE_BOMB_RETOPUP,
    SPINE_L1_KEY_RETOPUP,
    SpineRun,
    merge_inventory_assist,
    topup_owned_bombs,
    topup_owned_inventory,
    validate_l5_endpoint,
)


def test_l1_bow_splice_restores_key_before_backtrack44() -> None:
    from zelda_i.level1.bow_pickup import level1_survival_tf_stages

    names = [name for name, _, _ in level1_survival_tf_stages()]
    assert names.index("level1_bow_rejoin") < names.index("backtrack44")
    assert "backtrack44" in SPINE_L1_KEY_RETOPUP


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


def test_l3_boss_topup_preserves_carried_keys() -> None:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_BOMBS] = 8
    ram[ADDR_KEYS] = 4
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
    topup_owned_bombs(env, run)
    assert values["bombs"] == 16
    assert "keys" not in values
    assert run.inventory_assist["poke_bombs"] == 16
    assert run.inventory_assist["poke_keys"] is None


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
    with pytest.raises(ValueError, match="progression writes"):
        validate_l5_endpoint(
            {
                "ok": True,
                "continuous_emulator_session": True,
                "final": {"level": 5, "screen": 0x14, "triforce": 0x1C},
                "assist": {"progression_writes": 1, "capacity_writes": 0},
            }
        )
    validate_l5_endpoint(
        {
            "ok": True,
            "continuous_emulator_session": True,
            "seamed": False,
            "final": {"level": 5, "room": 0x14, "triforce": 0x1C},
            "assist": {"progression_writes": 0, "capacity_writes": 0},
        }
    )
