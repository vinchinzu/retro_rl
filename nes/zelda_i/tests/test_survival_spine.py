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
    ADDR_RUPEES,
)
from zelda_i.spine.survival import (
    BOOT_POLICY,
    SPINE_THROUGH,
    SpineRun,
    merge_inventory_assist,
    spine_final_fields,
    topup_owned_inventory,
)
from zelda_i.level5.spine import validate_l5_endpoint


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
        ring=1,
    )
    fields = spine_final_fields(snap)
    assert fields["rupees"] == 42
    assert fields["bombs"] == 8
    assert fields["arrows"] == 1
    assert fields["ring"] == 1


def test_main_spine_requires_blue_ring_at_l1_mouth() -> None:
    from zelda_i.spine.survival import gather_success

    snap = SimpleNamespace(level=0, screen=0x37, sword=2, ring=0, food=0)
    assert not gather_success(snap)
    snap.ring = 1
    assert not gather_success(snap)
    snap.food = 1
    assert gather_success(snap)


def test_ringless_downstream_save_point_is_obsolete(tmp_path, monkeypatch) -> None:
    from zelda_i.spine import survival

    path = tmp_path / "Old_level5_clear_0x77.state"
    path.write_bytes(b"old state")
    monkeypatch.setattr(survival, "state_path", lambda *_: path)
    monkeypatch.setattr(survival, "read_state_bytes", lambda _: b"old state")
    monkeypatch.setattr(survival, "read_snapshot", lambda _: SimpleNamespace(ring=0))
    env = SimpleNamespace(em=SimpleNamespace(set_state=lambda _: None), get_ram=lambda: None)
    run = SpineRun(through="level5", success=True, boot_frames=1)
    run.gather = {"engage_hearts": 1}
    run.save_points = "Old"
    with pytest.raises(ValueError, match="obsolete ringless main-spine save point"):
        survival.load_save_point(env, run, "level5_clear_0x77")


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
    assert arrow_names[-1] == "level1_arrows"
    run = SpineRun(through="level1-arrows", success=True, boot_frames=1)
    assert run.report()["stop"] == "level1_arrows"


def test_pre_l1_is_dedicated_gathering_not_l1_tf() -> None:
    """Gathering is before L1. Dedicated hop; not spliced onto TF stages."""
    import inspect

    from zelda_i.level1.bow_pickup import level1_survival_tf_stages
    from zelda_i.overworld.gathering import (
        pre_l1_bomb_shop_success,
        pre_l1_stages,
    )
    from zelda_i.spine.survival import (
        _L1_DEDICATED_HOPS,
        _boot_only_prefix,
        _continue_level1_spine,
        run_survival_spine,
    )

    assert "pre-l1" in SPINE_THROUGH
    hop = next(h for h in _L1_DEDICATED_HOPS if h.through == "pre-l1")
    assert hop.dedicated is True
    assert hop.stages is pre_l1_stages
    assert hop.success is pre_l1_bomb_shop_success
    assert hop.stop == "pre_l1_shop_p7"
    gather_names = [name for name, _, _ in pre_l1_stages()]
    tf_names = [name for name, _, _ in level1_survival_tf_stages()]
    assert gather_names[0] == "sword_cave"
    assert gather_names[1:] == ["bomb_walk", "bomb_topup", "bomb_buy"]
    assert not set(gather_names) & set(tf_names)
    continue_src = inspect.getsource(_continue_level1_spine)
    assert "_L1_DEDICATED_HOPS" in continue_src
    assert continue_src.index("attach_hops") < continue_src.index(
        "level1_survival_tf_stages"
    )
    prefix_src = inspect.getsource(run_survival_spine)
    assert 'through == "pre-l1"' in prefix_src
    assert "_boot_only_prefix" in prefix_src
    assert prefix_src.index('through == "pre-l1"') < prefix_src.index(
        'milestone="first_key"'
    )
    boot_src = inspect.getsource(_boot_only_prefix)
    assert "boot_to_ready" in boot_src
    assert "first_playthrough=True" in boot_src
    assert "clear53" not in boot_src
    run = SpineRun(through="pre-l1", success=True, boot_frames=1)
    assert run.report()["stop"] == "pre_l1_shop_p7"


def test_gather_is_the_default_prefix_and_stops_on_the_l1_mouth() -> None:
    """Default spine gathers first; the chain's refill is the only lever."""
    import inspect

    from zelda_i.assist import LastHeartAssist
    from zelda_i.overworld.gather_segments import chain_stages
    from zelda_i.spine.survival import (
        GATHER_ENGAGE_HEARTS,
        gather_assist,
        gather_stages,
        gathered_level1_stages,
        run_survival_spine,
    )

    params = inspect.signature(run_survival_spine).parameters
    assert params["gather"].default is True
    assert params["gather_engage_hearts"].default == GATHER_ENGAGE_HEARTS
    assert "gather" in SPINE_THROUGH
    assert SpineRun(through="gather", success=True, boot_frames=1).report()[
        "stop"
    ] == "gather_l1_mouth_0x37"
    names = [name for name, _, _ in gather_stages()]
    assert names == [name for name, _ in chain_stages()]
    assert names[0] == "exit_6f" and names[-1] == "walk_37"
    assert names.index("ring") < names.index("walk_37")
    assert all(limit > 0 for _, _, limit in gather_stages())
    l1 = [name for name, _, _ in gathered_level1_stages()]
    assert l1 == [
        "enter_level1",
        "first_key",
        "enter72",
        "clear72_key",
        "return73",
        "north",
        "clear63",
        "clear53",
    ]
    assert gather_assist(0) is None
    assert isinstance(gather_assist(1), LastHeartAssist)
    assert gather_assist(2).engage_at_whole_hearts == 2


def test_pre_l1_forces_assist_off() -> None:
    """Survival refill hides the $0670 chip that zeros the 10-kill 5-rupee."""
    import inspect

    from zelda_i.scripts import run_survival_spine as cli
    from zelda_i.spine.survival import run_survival_spine

    lib = inspect.getsource(run_survival_spine)
    pre_l1 = lib.split('if through == "pre-l1":', 1)[1].split("else:", 1)[0]
    assert "assist = None" in pre_l1
    assert "allow_pokes = False" in pre_l1
    cli_src = inspect.getsource(cli.main)
    assert 'args.through == "pre-l1"' in cli_src
    assert "infinite_life = False" in cli_src
    assert "allow_pokes = False" in cli_src


def test_l1_west_key_pays_for_the_bow_key_before_backtrack44() -> None:
    """0x72's key, taken straight after 0x74's, replaces the backtrack44 poke."""
    import inspect

    from zelda_i.level1.bow_pickup import level1_survival_tf_stages
    from zelda_i.spine import survival
    from zelda_i.spine.survival import gathered_level1_stages

    names = [name for name, _, _ in level1_survival_tf_stages()]
    assert names.index("level1_bow_rejoin") < names.index("backtrack44")
    prefix = [name for name, _, _ in gathered_level1_stages()]
    assert prefix.index("first_key") < prefix.index("clear72_key") < prefix.index("north")
    assert not hasattr(survival, "topup_owned_keys")
    assert "key_retopup" not in inspect.signature(survival._run_stages).parameters


def test_spine_level2_has_no_bomb_topup() -> None:
    """Natural 0x4A bomb buy and room drops retire L2 bomb top-ups (rr-doua)."""
    import inspect

    from zelda_i.spine import survival

    assert not hasattr(survival, "SPINE_BOMB_RETOPUP")
    assert "retopup" not in inspect.getsource(survival._continue_level2_spine)


def test_spine_level3_has_no_bomb_topup() -> None:
    """Natural Darknut drops retire L3 bomb top-ups (rr-doua)."""
    import inspect
    from zelda_i.spine.survival import _continue_level3_spine

    src = inspect.getsource(_continue_level3_spine)
    assert "topup_owned_bombs" not in src
    assert "topup_owned_inventory" not in src


def test_spine_level4_buys_bombs_instead_of_topup() -> None:
    """The 0x44 pack retires the two L4 bomb top-ups (rr-doua)."""
    import inspect

    from zelda_i.level4.spine import continue_level4_spine, l4_hops
    from zelda_i.spine import survival

    assert not hasattr(survival, "topup_owned_bombs")
    assert "topup" not in inspect.signature(l4_hops).parameters
    assert "topup_bombs" not in inspect.signature(continue_level4_spine).parameters
    entry = l4_hops(spine_fields=lambda snap: {})[0]
    names = [name for name, _, _ in entry.stages]
    assert names.index("potion_restock_l3") < names.index("bomb_restock_l3")
    assert names.index("exit_bomb_restock_l3") < names.index("enter_level4")
    assert all(hop.before is None for hop in l4_hops(spine_fields=lambda snap: {}))


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
    validate_l5_endpoint(
        {
            "ok": True,
            "continuous_emulator_session": True,
            "seamed": False,
            "final": {"level": 5, "room": 0x14, "triforce": 0x1C},
            "assist": None,
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


def test_run_survival_spine_allows_assist_none() -> None:
    """``--no-infinite-life`` passes assist=None; heart refill simply skips."""
    from zelda_i.spine.survival import run_survival_spine

    try:
        run_survival_spine(None, None, assist=None, through="level1")
    except ValueError as exc:
        raise AssertionError(f"assist=None must be legal, got {exc}") from exc
    except Exception:
        pass


def test_spine_script_keeps_clean_flag_through_imports() -> None:
    """``--clean`` must reach the parser. A level3.spine import used to drop it
    from ``sys.argv``, so every ``--clean`` spine run until 2026-09-24 was Survival."""
    import subprocess
    import sys
    from pathlib import Path

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    probe = (
        "import sys; sys.path.insert(0, sys.argv.pop(1)); "
        "import run_survival_spine; "
        "assert sys.argv[1:] == ['--clean'], sys.argv"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe, str(scripts), "--clean"],
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    assert out.returncode == 0, out.stderr[-2000:]


def test_topups_and_run_stages_noop_when_pokes_disallowed() -> None:
    """``--no-pokes`` / ``--clean`` skip every owned-inventory write."""
    import inspect

    from zelda_i.level7.spine import continue_level7_spine
    from zelda_i.level8.spine import continue_level8_spine
    from zelda_i.spine.survival import _run_stages

    src = inspect.getsource(_run_stages)
    assert "allow_pokes" in src
    assert "allow_pokes" in inspect.getsource(continue_level7_spine)
    assert "retopup" not in inspect.getsource(continue_level8_spine)

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
    run = SpineRun(through="level2", success=True, boot_frames=1, allow_pokes=False)

    class _Ctl:
        poke_arrows = True

    import zelda_i.spine.survival as surv

    orig = surv.run_controller_stage

    def fake_stage(env, obs, **kw):
        del env, obs
        return None, SimpleNamespace(
            success=True, end_frame=1, name=kw["name"], report=lambda: {}
        )

    try:
        surv.run_controller_stage = fake_stage
        ctl = _Ctl()
        assert _run_stages(
            env,
            run,
            (("gohma", ctl, 10),),
            assist=None,
            retopup=frozenset({"gohma"}),
        )
    finally:
        surv.run_controller_stage = orig
    assert ctl.poke_arrows is False

    topup_owned_inventory(env, run)
    assert values == {}
    assert run.inventory_assist is None

    run_on = SpineRun(through="level2", success=True, boot_frames=1)
    assert run_on.allow_pokes is True
    topup_owned_inventory(env, run_on)
    assert values["bombs"] == 16
    assert run_on.inventory_assist is not None


def test_spine_stages_never_write_the_wallet() -> None:
    """Rupees are earned (hidden caves, drops), never written: a short
    wallet goes into every top-up gate short, pokes allowed or not."""
    from zelda_i.spine.survival import _run_stages

    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_RUPEES] = 10
    values: dict[str, int] = {}

    class _Data:
        memory = None

        def set_value(self, key: str, value: int) -> None:
            values[key] = int(value)

    env = SimpleNamespace(
        get_ram=lambda: ram,
        unwrapped=SimpleNamespace(data=_Data(), em=None),
    )
    import zelda_i.spine.survival as surv

    orig = surv.run_controller_stage

    def fake_stage(env, obs, **kw):
        del env, obs
        return None, SimpleNamespace(
            success=True, end_frame=1, name=kw["name"], report=lambda: {}
        )

    try:
        surv.run_controller_stage = fake_stage
        for allow in (True, False):
            run = SpineRun(through="level7", success=True, boot_frames=1, allow_pokes=allow)
            gates = frozenset({"bomb_topup", "ring", "level7_bait_purchase"})
            stages = tuple((name, SimpleNamespace(), 10) for name in sorted(gates))
            assert _run_stages(env, run, stages, assist=None, retopup=gates)
    finally:
        surv.run_controller_stage = orig
    assert "rupees" not in values


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
    "pre-l1",
    "level1-bow",
    "level1-bow-cellar",
    "level1-bow-pickup",
    "level1-arrows",
    "level1-bombs",
    "gather",
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
    "level6-bomb28",
    "level6-room19",
    "level6-clear19",
    "level6-room09",
    "level6-clear09",
    "level6-stairs09",
    "level6-rod",
    "level6-exit75",
    "level6-south09",
    "level6-south19",
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
