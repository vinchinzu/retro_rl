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
    SPINE_THROUGH,
    SpineRun,
    merge_inventory_assist,
    spine_final_fields,
    topup_owned_bombs,
    topup_owned_inventory,
    validate_l5_endpoint,
)


def test_spine_final_fields_records_rupees() -> None:
    """L6 leftover already records rupees; spine reports must too. No inventory poke."""
    snap = SimpleNamespace(
        mode=5,
        level=6,
        screen=0x0C,
        link_x=120,
        link_y=149,
        keys=2,
        bombs=8,
        rupees=42,
        health=0x77,
        triforce=0x3F,
        map=0x0A,
        rod=1,
        bow=1,
        arrows=1,
    )
    fields = spine_final_fields(snap)
    assert fields["rupees"] == 42
    assert fields["bombs"] == 8
    assert fields["arrows"] == 1


def test_level1_arrows_is_dedicated_not_on_default_tf() -> None:
    from zelda_i.level1.arrow_shop import level1_arrows_stages
    from zelda_i.level1.bow_pickup import level1_survival_tf_stages

    assert "level1-arrows" in SPINE_THROUGH
    tf_names = [name for name, _, _ in level1_survival_tf_stages()]
    assert "level1_arrows" not in tf_names
    arrow_names = [name for name, _, _ in level1_arrows_stages()]
    assert "level1_bow_pickup" in arrow_names
    assert "backtrack44" in arrow_names
    assert "backtrack44" in SPINE_L1_KEY_RETOPUP
    assert arrow_names[-1] == "level1_arrows"
    run = SpineRun(through="level1-arrows", success=True, boot_frames=1)
    assert run.report()["stop"] == "level1_arrows"


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


def test_spine_run_unmeasured_set_state_is_unknown() -> None:
    run = SpineRun(through="level4", success=True, boot_frames=1)
    report = run.report()
    assert report["set_state_count"] is None
    assert report["mid_run_state_load"] is None
    assert report["ok"] is True


def test_spine_run_measured_zero_set_state_is_not_a_load() -> None:
    run = SpineRun(through="level4", success=True, boot_frames=1)
    run.apply_state_audit(0)
    report = run.report()
    assert report["set_state_count"] == 0
    assert report["mid_run_state_load"] is False
    assert report["ok"] is True


def test_survival_spine_cli_wraps_audited_env() -> None:
    import inspect

    from zelda_i.scripts import run_survival_spine as cli

    src = inspect.getsource(cli.main)
    assert "AuditedEnv" in src
    assert "apply_state_audit" in src
    assert "zelda_i.survival_spine" in src


def test_level7_seam_is_wired_into_the_spine() -> None:
    import inspect

    from zelda_i.level7.spine import L7_THROUGH
    from zelda_i.spine import survival

    for target in L7_THROUGH:
        assert target in SPINE_THROUGH
        run = SpineRun(through=target, success=True, boot_frames=1)
        assert run.report()["stop"] is not None
    src = inspect.getsource(survival.run_survival_spine)
    assert "continue_level7_spine" in src


def test_through_for_predecessor_remaps_every_downstream_target() -> None:
    """rr-mzxn regression guard.

    Every L7/L8/L9 target must remap to the predecessor's own final stop
    when driving an earlier level's ``continue_*_spine`` -- not just the
    immediate next level. Missing L9 in the L6/L7 remap left Link stranded
    inside the L6 dungeon (L6) or raised ``ValueError`` (L7, whose own
    ``continue_level7_spine`` rejects any ``through`` outside L7_THROUGH).
    """
    from zelda_i.level6.spine import L6_THROUGH
    from zelda_i.level7.spine import L7_THROUGH
    from zelda_i.level8.spine import L8_THROUGH
    from zelda_i.level9.spine import L9_THROUGH
    from zelda_i.spine.survival import _through_for_predecessor

    downstream_of_l6 = L7_THROUGH + L8_THROUGH + L9_THROUGH
    for target in downstream_of_l6:
        assert (
            _through_for_predecessor(target, L6_THROUGH, "level6-exit")
            == "level6-exit"
        )
    for target in L6_THROUGH:
        assert _through_for_predecessor(target, L6_THROUGH, "level6-exit") == target

    downstream_of_l7 = L8_THROUGH + L9_THROUGH
    for target in downstream_of_l7:
        assert _through_for_predecessor(target, L7_THROUGH, "level7") == "level7"
    for target in L7_THROUGH:
        assert _through_for_predecessor(target, L7_THROUGH, "level7") == target

    for target in L9_THROUGH:
        assert _through_for_predecessor(target, L8_THROUGH, "level8") == "level8"
    for target in L8_THROUGH:
        assert _through_for_predecessor(target, L8_THROUGH, "level8") == target


def test_spine_run_measured_set_state_fails_the_run() -> None:
    run = SpineRun(through="level4", success=True, boot_frames=1)
    run.apply_state_audit(2)
    report = run.report()
    assert report["set_state_count"] == 2
    assert report["mid_run_state_load"] is True
    assert report["ok"] is False
    assert report["failed_stage"] == "mid_run_state_load"
