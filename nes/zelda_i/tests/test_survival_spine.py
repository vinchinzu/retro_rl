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
    from zelda_i.level1.finish import level1_triforce_stages

    assert "level1-arrows" in SPINE_THROUGH
    tf_names = [name for name, _, _ in level1_survival_tf_stages()]
    assert "level1_arrows" not in tf_names
    clean_names = [
        name for name, _, _ in level1_triforce_stages(natural_entry=True)
    ]
    assert "clear72_key" not in clean_names
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
    assert "clear72_key" not in names  # 0x63 skirt red; poke stays
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
    assert "tap.close()" in src
    assert "tap.abort()" in src


def test_video_tap_close_and_abort_without_writer() -> None:
    from zelda_i.runner import VideoTap

    tap = VideoTap(None, None, tag="t")
    info = tap.close()
    assert info["path"] is None
    assert info["encoded_frames"] == 0
    tap.abort()
    again = tap.close()
    assert again["path"] is None


def test_level7_seam_is_wired_into_the_spine() -> None:
    from zelda_i.level7.spine import L7_THROUGH, continue_level7_spine
    from zelda_i.spine.survival import SPINE_LEVELS

    for target in L7_THROUGH:
        assert target in SPINE_THROUGH
        run = SpineRun(through=target, success=True, boot_frames=1)
        assert run.report()["stop"] is not None
    (row,) = [row for row in SPINE_LEVELS if row.level == 7]
    assert row.run is continue_level7_spine
    assert row.through == L7_THROUGH


def test_spine_level_target_remaps_every_downstream_target() -> None:
    """rr-mzxn regression guard, now on ``SpineLevel.target``.

    Every target past level N must remap to row N's own handoff when driving
    that row -- not just the immediate next level. Missing L9 in the L6/L7
    remap left Link stranded inside the L6 dungeon (L6) or raised
    ``ValueError`` (L7, whose ``continue_level7_spine`` rejects any ``through``
    outside L7_THROUGH). A row's own ids must pass through unchanged so its
    ``attach_hops`` loop still stops there.
    """
    from zelda_i.spine.survival import SPINE_LEVELS

    rows = {row.level: row for row in SPINE_LEVELS}
    assert rows[6].handoff == "level6-exit"
    assert rows[7].handoff == "level7"
    assert rows[8].handoff == "level8"

    for index, row in enumerate(SPINE_LEVELS):
        downstream = [
            target
            for later in SPINE_LEVELS[index + 1 :]
            for target in later.through
        ]
        for target in downstream:
            assert row.target(target) == row.handoff
        for target in row.through:
            assert row.target(target) == target
        assert row.handoff in row.through


def test_spine_levels_cover_the_through_catalog_exactly() -> None:
    """One row table owns every ``--through`` id; no id is orphaned or dupe."""
    from zelda_i.spine.survival import SPINE_LEVELS, SPINE_STOPS

    assert tuple(t for row in SPINE_LEVELS for t in row.through) == SPINE_THROUGH
    assert len(SPINE_THROUGH) == len(set(SPINE_THROUGH))
    assert [row.level for row in SPINE_LEVELS] == list(range(1, 10))
    for row in SPINE_LEVELS:
        assert set(row.stops) == set(row.through), row.level
        assert callable(row.run)
    assert set(SPINE_STOPS) == set(SPINE_THROUGH)
    for target in SPINE_THROUGH:
        assert SpineRun(through=target, success=True, boot_frames=1).report()["stop"]


def test_spine_through_catalog_is_pinned() -> None:
    """Full ``--through`` list. A row edit that drops a stop fails here."""
    assert SPINE_THROUGH == PINNED_SPINE_THROUGH


def test_spine_run_measured_set_state_fails_the_run() -> None:
    run = SpineRun(through="level4", success=True, boot_frames=1)
    run.apply_state_audit(2)
    report = run.report()
    assert report["set_state_count"] == 2
    assert report["mid_run_state_load"] is True
    assert report["ok"] is False
    assert report["failed_stage"] == "mid_run_state_load"


# The full ``--through`` catalog, pinned. Regenerate only alongside a
# deliberate route change: SPINE_LEVELS is the source, this is the guard.
PINNED_SPINE_THROUGH: tuple[str, ...] = (
    "level1",
    "level1-bow",
    "level1-bow-cellar",
    "level1-bow-pickup",
    "level1-arrows",
    "level2-entry",
    "level2",
    "level3",
    "level4-entry",
    "level4-key",
    "level4-clear50",
    "level4-room40-key",
    "level4-room30",
    "level4-room31",
    "level4-clear31",
    "level4-room32",
    "level4-clear32",
    "level4-stepladder",
    "level4-exit60",
    "level4-west31",
    "level4-keyup20",
    "level4-room21",
    "level4-map",
    "level4-bomb11",
    "level4-key01",
    "level4-clear12",
    "level4-gleeok13",
    "level4",
    "level5-entry",
    "level5-clear66",
    "level5-east77",
    "level5-whistle",
    "level5-exit04",
    "level5",
    "level6-entry",
    "level6-east-key",
    "level6-west",
    "level6-compass",
    "level6-clear68",
    "level6-keese",
    "level6-clear58",
    "level6-room48",
    "level6-room38",
    "level6-clear38",
    "level6-room28",
    "level6-clear28",
    "level6-room18",
    "level6-settle18",
    "level6-gleeok18",
    "level6-postgleeok18",
    "level6-stairs18",
    "level6-room19",
    "level6-clear19",
    "level6-map19",
    "level6-room09",
    "level6-clear09",
    "level6-stairs09",
    "level6-rod",
    "level6-exit75",
    "level6-south09",
    "level6-south19",
    "level6-clear29",
    "level6-east29",
    "level6-south29",
    "level6-settle39",
    "level6-clear39",
    "level6-east39",
    "level6-settle3a",
    "level6-clear3a",
    "level6-stairs3a-warp",
    "level6-cellar08",
    "level6-south1d",
    "level6-west2d",
    "level6-north2c",
    "level6-gohma",
    "level6-heart",
    "level6-north0c",
    "level6",
    "level6-exit",
    "level6-north39",
    "level6-inland29",
    "level6-west19",
    "level6-south18",
    "level7-bait-shop",
    "level7-entry",
    "level7-red-candle",
    "level7",
    "level8-entry",
    "level8-magic-key",
    "level8",
    "level9-entry",
    "level9-silver-arrows",
    "level9-patra",
    "level9-credits",
)
