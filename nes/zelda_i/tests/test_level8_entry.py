"""Level 8 fail-closed entry, graph, and chapter factories (no emulator)."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from retro_harness.nes import nes_action, nes_idle_action

from zelda_i.level8.bush import (
    MOUTH_STANDS,
    REFUTED_BUSH_AIM,
    REFUTED_PUSH,
    VERIFIED_BUSH_AIM,
    VERIFIED_BUSH_X,
    VERIFIED_BUSH_Y,
    VERIFIED_FACING,
    VERIFIED_PUSH,
    IsolatedBushReconController,
    make_isolated_bush_recon_controller,
)
from zelda_i.level8.dungeon import (
    GLEEOK_FOUR_HEAD_OBJECT_TYPE,
    GLEEOK_ROUTE,
    LEVEL8_HYPOTHESIS_ROOMS,
    LEVEL8_INTERIOR_0X0F_RECON,
    LEVEL8_INTERIOR_0X1E_RECON,
    LEVEL8_INTERIOR_0X1F_RECON,
    LEVEL8_INTERIOR_0X2E_RECON,
    LEVEL8_INTERIOR_0X3E_RECON,
    LEVEL8_INTERIOR_ROOM_RECON,
    LEVEL8_ROOM_SPECS,
    MAGIC_KEY_ROUTE,
    OMITTED_OPTIONAL_ROOMS,
    UNOBSERVED_LEVEL8_CLEAR,
    UNOBSERVED_LEVEL8_TOPOLOGY,
    Level8ClearEndpoint,
    Level8Topology,
    hypothesis_room_ids_unobserved,
    level8_clear_stop,
    level8_entry_stop,
    level8_magic_key_ledger,
    level8_magic_key_stop,
)
from zelda_i.level8.entry import (
    ADDR_CANDLE_USED,
    B_ITEM_CANDLE,
    CANDLE_RED,
    UNMEASURED_POST_L7_HANDOFF,
    UNVERIFIED_BUSH_BURN_TARGET,
    BushBurnTarget,
    PostLevel7Handoff,
    make_burn_level8_bush_controller,
    make_post_l7_to_bush_controller,
    make_select_red_candle_controller,
)
from zelda_i.level8.hops import l8_hops
from zelda_i.level8.path import (
    make_blue_gohma_controller,
    make_four_head_gleeok_controller,
)
from zelda_i.level8.spine import L8_STOPS, L8_THROUGH, continue_level8_spine
from zelda_i.ram import (
    ADDR_ARROWS,
    ADDR_BOMBS,
    ADDR_BOW,
    ADDR_CANDLE,
    ADDR_HEALTH,
    ADDR_KEYS,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MAGIC_KEY,
    ADDR_MODE,
    ADDR_SCREEN,
    ADDR_SELECTED_ITEM,
    ADDR_SWORD,
    ADDR_TRIFORCE,
    PLAY_MODE,
    read_snapshot,
)

_LEVEL8_DIR = Path(__file__).resolve().parents[1] / "level8"
_WRITE_MODULES = (
    "bush.py",
    "dungeon.py",
    "entry.py",
    "hops.py",
    "north_column.py",
    "path.py",
    "spine.py",
)


def _ram(**fields: int) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = fields.get("mode", PLAY_MODE)
    ram[ADDR_LEVEL] = fields.get("level", 0)
    ram[ADDR_SCREEN] = fields.get("screen", 0x6D)
    ram[ADDR_LINK_X] = fields.get("x", 48)
    ram[ADDR_LINK_Y] = fields.get("y", 93)
    ram[ADDR_TRIFORCE] = fields.get("triforce", 0x7F)
    ram[ADDR_SWORD] = fields.get("sword", 1)
    ram[ADDR_HEALTH] = fields.get("health", 0xBB)
    ram[ADDR_KEYS] = fields.get("keys", 4)
    ram[ADDR_BOMBS] = fields.get("bombs", 8)
    ram[ADDR_CANDLE] = fields.get("candle", CANDLE_RED)
    ram[ADDR_SELECTED_ITEM] = fields.get("selected", B_ITEM_CANDLE)
    ram[ADDR_CANDLE_USED] = fields.get("candle_used", 0)
    ram[ADDR_BOW] = fields.get("bow", 1)
    ram[ADDR_ARROWS] = fields.get("arrows", 1)
    ram[ADDR_MAGIC_KEY] = fields.get("magic_key", 0)
    return ram


def _env(ram: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(get_ram=lambda: ram)


def _verified_target() -> BushBurnTarget:
    """The swept-verified recipe, as a unit-test target (never route eligible).

    Backing evidence: nes/zelda_i/logs/level8_bush_burn_sweep.json and
    custom_integrations/LegendOfZelda-Nes/Level8EntranceReconFixture.provenance.json.
    """
    return BushBurnTarget(
        link_x=VERIFIED_BUSH_X,
        link_y=VERIFIED_BUSH_Y,
        facing=VERIFIED_FACING,
        push_direction=VERIFIED_PUSH,
        verified=True,
        route_eligible=False,
        evidence="unit-test",
    )


def test_public_through_names_unchanged() -> None:
    assert L8_THROUGH == ("level8-entry", "level8-magic-key", "level8")
    assert L8_STOPS == {
        "level8-entry": "level8_entry_live",
        "level8-magic-key": "level8_magic_key_natural",
        "level8": "level8_triforce_0x80",
    }
    hops = l8_hops(_env(_ram()))
    assert tuple(hop.through for hop in hops) == L8_THROUGH
    assert tuple(hop.stop for hop in hops) == tuple(L8_STOPS[t] for t in L8_THROUGH)


def test_continue_level8_spine_rejects_unknown_through() -> None:
    try:
        continue_level8_spine(
            None, None, through="level8-book", run_stages=lambda *_a, **_k: True
        )
    except ValueError as exc:
        assert "level8-book" in str(exc)
    else:
        raise AssertionError("unknown through must raise")


def test_incomplete_handoff_refuses_to_move() -> None:
    ram = _ram()
    ctl = make_post_l7_to_bush_controller()
    ctl.bind_env(_env(ram))
    act = ctl.step(read_snapshot(ram))
    assert not ctl.handoff.complete()
    assert not UNMEASURED_POST_L7_HANDOFF.verified
    assert ctl.phase.name == "FAILED"
    assert "post_l7_handoff_unmeasured" in ctl.notes
    assert list(act.action) == list(nes_idle_action())


def test_verified_false_handoff_still_refuses() -> None:
    handoff = PostLevel7Handoff(
        screen=0x42,
        link_x=120,
        link_y=141,
        keys=4,
        bombs=8,
        rupees=10,
        heart_containers=12,
        selected_item=4,
        whistle=1,
        food=0,
        rod=1,
        bow=1,
        arrows=1,
        candle=2,
        verified=False,
        route_eligible=False,
    )
    ram = _ram(screen=0x42, x=120, y=141)
    ctl = make_post_l7_to_bush_controller(handoff=handoff)
    ctl.bind_env(_env(ram))
    ctl.step(read_snapshot(ram))
    assert not handoff.complete()
    assert ctl.phase.name == "FAILED"


def test_unverified_burn_target_does_not_move() -> None:
    ram = _ram()
    ctl = make_burn_level8_bush_controller()
    ctl.bind_env(_env(ram))
    act = ctl.step(read_snapshot(ram))
    assert not UNVERIFIED_BUSH_BURN_TARGET.complete()
    assert ctl.failed
    assert not ctl.success
    assert "bush_burn_target_unverified" in ctl.notes
    assert list(act.action) == list(nes_idle_action())


def test_burn_budget_exhaust_on_0x6d_is_failure() -> None:
    ram = _ram(x=VERIFIED_BUSH_X, y=VERIFIED_BUSH_Y, candle=2, selected=4)
    target = _verified_target()
    ctl = make_burn_level8_bush_controller(target=target)
    ctl.burn_budget = 6
    ctl.bind_env(_env(ram))
    snap = read_snapshot(ram)
    for _ in range(20):
        ctl.step(snap)
        if ctl.failed:
            break
    assert ctl.failed
    assert not ctl.success
    assert snap.screen == 0x6D
    assert "burn_budget_exhausted_without_level8_entry" in ctl.notes


def test_isolated_recon_budget_exhaust_is_failure() -> None:
    ram = _ram(x=VERIFIED_BUSH_X, y=VERIFIED_BUSH_Y, candle=2, selected=4)
    ctl = make_isolated_bush_recon_controller()
    ctl.burn_budget = 6
    ctl.bind_env(_env(ram))
    snap = read_snapshot(ram)
    for _ in range(20):
        ctl.step(snap)
        if ctl.failed:
            break
    assert isinstance(ctl, IsolatedBushReconController)
    assert ctl.failed
    assert not ctl.success
    assert not ctl.route_eligible
    assert ctl.evidence == "fixture-live"
    assert "burn_budget_exhausted_without_level8_entry" in ctl.notes


def test_recon_default_is_the_swept_verified_recipe() -> None:
    # nes/zelda_i/logs/level8_bush_burn_sweep.json (5856 trials) +
    # Level8EntranceReconFixture.provenance.json: (136, 93) face RIGHT push
    # RIGHT. (144, 93)/UP is the refuted aim, and is not a mouth stand.
    ctl = make_isolated_bush_recon_controller()
    assert (ctl.link_x, ctl.link_y) == VERIFIED_BUSH_AIM == (136, 93)
    assert ctl.facing == VERIFIED_FACING == "RIGHT"
    assert ctl.push_direction == VERIFIED_PUSH == "RIGHT"
    assert REFUTED_BUSH_AIM == (144, 93)
    assert REFUTED_PUSH == "UP"
    assert REFUTED_BUSH_AIM not in {(x, y) for x, y, _f, _p in MOUTH_STANDS}
    assert VERIFIED_BUSH_AIM in {(x, y) for x, y, _f, _p in MOUTH_STANDS}
    # Every swept mouth stand fired and pushed the same direction.
    assert all(facing == push for _x, _y, facing, push in MOUTH_STANDS)
    assert ctl.report()["refuted_aim"] == [144, 93, "RIGHT", "UP"]
    assert not ctl.route_eligible


def _drive_to_mouth(ctl: Any, ram: np.ndarray) -> Any:
    """Validate on 0x6D, observe candle use, then raise the mode-16 mouth."""
    ctl.bind_env(_env(ram))
    ctl.step(read_snapshot(ram))
    assert not ctl.failed
    ram[ADDR_CANDLE_USED] = 1
    ctl.step(read_snapshot(ram))
    assert ctl.candle_use_observed
    ram[ADDR_MODE] = 16
    return ctl.step(read_snapshot(ram))


def test_right_push_target_drives_the_mouth_into_level8() -> None:
    # rr-i6hq: mode 16 must be answered with the recorded push_direction. The
    # sweep saw entry_room=null on all seven mouth stands (UP never completes);
    # Level8EntranceReconFixture reached live L8 0x7E by continuing RIGHT.
    ram = _ram(x=VERIFIED_BUSH_X, y=VERIFIED_BUSH_Y, candle=2, selected=4)
    ctl = make_burn_level8_bush_controller(target=_verified_target())
    act = _drive_to_mouth(ctl, ram)
    assert ctl.phase.name == "ENTER"
    assert "mouth_transition_observed" in ctl.notes
    assert list(act.action) == list(nes_action("RIGHT"))
    assert list(act.action) != list(nes_action("UP"))
    # The transition frames keep pushing RIGHT, never UP.
    ram[ADDR_MODE] = 6
    assert list(ctl.step(read_snapshot(ram)).action) == list(nes_action("RIGHT"))
    # Live L8 landing, matching the fixture provenance (screen 0x7E).
    ram[ADDR_MODE] = PLAY_MODE
    ram[ADDR_LEVEL] = 8
    ram[ADDR_SCREEN] = 0x7E
    ctl.step(read_snapshot(ram))
    assert ctl.success
    assert not ctl.failed
    assert ctl.observed_entry_room == 0x7E
    assert ctl.report()["route_eligible"] is False


def test_isolated_recon_right_push_drives_the_mouth_into_level8() -> None:
    ram = _ram(x=VERIFIED_BUSH_X, y=VERIFIED_BUSH_Y, candle=2, selected=4)
    ctl = make_isolated_bush_recon_controller()
    act = _drive_to_mouth(ctl, ram)
    assert ctl.phase.name == "ENTER"
    assert list(act.action) == list(nes_action("RIGHT"))
    ram[ADDR_MODE] = PLAY_MODE
    ram[ADDR_LEVEL] = 8
    ram[ADDR_SCREEN] = 0x7E
    ctl.step(read_snapshot(ram))
    assert ctl.success
    assert ctl.observed_entry_room == 0x7E
    assert ctl.report()["route_eligible"] is False
    assert ctl.evidence == "fixture-live"


def test_mouth_without_observed_candle_use_fails_closed() -> None:
    ram = _ram(x=VERIFIED_BUSH_X, y=VERIFIED_BUSH_Y, candle=2, selected=4)
    ctl = make_burn_level8_bush_controller(target=_verified_target())
    ctl.bind_env(_env(ram))
    ctl.step(read_snapshot(ram))
    ram[ADDR_MODE] = 16
    act = ctl.step(read_snapshot(ram))
    assert ctl.failed
    assert not ctl.success
    assert not ctl.candle_use_observed
    assert "mouth_transition_without_candle_use" in ctl.notes
    assert list(act.action) == list(nes_idle_action())


def test_level8_without_observed_candle_use_fails_closed() -> None:
    ram = _ram(x=VERIFIED_BUSH_X, y=VERIFIED_BUSH_Y, candle=2, selected=4)
    ctl = make_burn_level8_bush_controller(target=_verified_target())
    ctl.bind_env(_env(ram))
    ctl.step(read_snapshot(ram))
    ram[ADDR_LEVEL] = 8
    ram[ADDR_SCREEN] = 0x7E
    ctl.step(read_snapshot(ram))
    assert ctl.failed
    assert not ctl.success
    assert ctl.observed_entry_room is None
    assert "level8_entered_without_observed_candle_use" in ctl.notes


def test_isolated_recon_without_candle_fails_closed() -> None:
    ram = _ram(candle=0, selected=0)
    ctl = make_isolated_bush_recon_controller()
    ctl.bind_env(_env(ram))
    ctl.step(read_snapshot(ram))
    assert ctl.failed
    assert not ctl.success
    assert "bush_recon_candle_unowned" in ctl.notes


def test_select_candle_never_writes_selected_item() -> None:
    ram = _ram(selected=1)
    before = int(ram[ADDR_SELECTED_ITEM])
    ctl = make_select_red_candle_controller()
    ctl.bind_env(_env(ram))
    ctl.step(read_snapshot(ram))
    assert int(ram[ADDR_SELECTED_ITEM]) == before
    assert ctl.report()["writes"] == 0
    assert ctl.report()["normal_pause_input"] is True


def test_selected_item_is_never_assigned_in_l8_lane() -> None:
    for name in _WRITE_MODULES:
        tree = ast.parse((_LEVEL8_DIR / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr in {
                    "set_value",
                    "assign",
                    "set_bytes",
                }:
                    args = [ast.unparse(arg) for arg in node.args]
                    joined = " ".join(args)
                    assert "ADDR_SELECTED_ITEM" not in joined
                    assert "selected_item" not in joined
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    text = ast.unparse(target)
                    assert "ADDR_SELECTED_ITEM" not in text


def test_hypothesis_graph_has_no_ram_room_ids() -> None:
    assert hypothesis_room_ids_unobserved()
    assert LEVEL8_ROOM_SPECS == ()
    assert "book_stairs" in OMITTED_OPTIONAL_ROOMS
    assert "compass" in OMITTED_OPTIONAL_ROOMS
    assert MAGIC_KEY_ROUTE[0] == "entry"
    assert MAGIC_KEY_ROUTE[-1] == "magic_key_stairs"
    assert "book_stairs" not in MAGIC_KEY_ROUTE
    assert GLEEOK_ROUTE[-2] == "gleeok"
    assert GLEEOK_ROUTE[-1] == "triforce"
    # rr-6o7.1: exactly one disclosed live-recon room id (the entry, via
    # fixture-only OW 0x6D burn) -- every other hypothesis room stays
    # unobserved, and the disclosed one never claims route_eligible.
    disclosed = [room for room in LEVEL8_HYPOTHESIS_ROOMS if room.room_id is not None]
    assert [room.name for room in disclosed] == ["entry"]
    assert disclosed[0].room_id == 0x7E
    assert disclosed[0].evidence == "live_recon_fixture"
    assert all(
        room.room_id is None
        for room in LEVEL8_HYPOTHESIS_ROOMS
        if room.name != "entry"
    )
    assert all(not room.route_eligible for room in (UNOBSERVED_LEVEL8_TOPOLOGY,))


def test_level8_interior_0x3e_recon_is_fixture_only_not_route_eligible() -> None:
    # rr-6o7.2 groundwork: ONE guarded replay past 0x4E via the north key door.
    # 2/2 byte-identical (probe l8_4e_north B1/B2). Recon record only -- never
    # route eligible, never a DungeonRoomSpec, never on L8_THROUGH.
    from zelda_i.level8.spine import L8_THROUGH

    assert LEVEL8_INTERIOR_ROOM_RECON == (
        LEVEL8_INTERIOR_0X3E_RECON,
        LEVEL8_INTERIOR_0X2E_RECON,
        LEVEL8_INTERIOR_0X1E_RECON,
        LEVEL8_INTERIOR_0X1F_RECON,
        LEVEL8_INTERIOR_0X0F_RECON,
    )
    r = LEVEL8_INTERIOR_0X3E_RECON
    assert r.room_id == 0x3E
    assert r.entered_from == 0x4E and r.entry_gate == "north_key_door"
    assert (r.keys_in, r.keys_out) == (10, 9)  # one natural key spent
    assert (r.bombs_in, r.bombs_out) == (7, 7)  # no bombs used
    assert r.census == ((0x0C, 128, 6),)  # 6 x type 0x0C HP128 (unregistered)
    assert r.room_item_id == 0x03
    assert r.evidence == "live_recon_fixture"
    assert r.route_eligible is False
    assert r.fixture == "Level8Interior3EReconFixture"
    assert LEVEL8_ROOM_SPECS == ()  # still no canonical rows
    assert "level8-interior-0x3e" not in L8_THROUGH
    assert hypothesis_room_ids_unobserved()  # hypothesis graph untouched


def test_public_gate_predicates_fail_closed() -> None:
    snap = read_snapshot(_ram(level=8, screen=0x7D, candle=2, magic_key=1))
    assert not level8_entry_stop(snap, candle=2)
    assert not level8_magic_key_stop(snap, magic_key=1)
    assert not level8_clear_stop(snap, magic_key=1)
    live = Level8Topology(
        entry_room=0x7D,
        magic_key_room=0x1E,
        boss_room=0x13,
        triforce_room=0x03,
        evidence="unit-test",
        route_eligible=True,
    )
    assert level8_entry_stop(snap, candle=2, topology=live)
    assert not level8_entry_stop(snap, candle=1, topology=live)
    mk = read_snapshot(_ram(level=8, screen=0x1E, magic_key=1))
    assert level8_magic_key_stop(mk, magic_key=1, topology=live, magic_key_before=0)
    assert not level8_magic_key_stop(mk, magic_key=1, topology=live, magic_key_before=1)
    endpoint = Level8ClearEndpoint(
        level=0,
        screen=0x6D,
        mode=PLAY_MODE,
        incoming_heart_containers=12,
        outgoing_heart_containers=13,
        evidence="unit-test",
        route_eligible=True,
    )
    leave = read_snapshot(_ram(level=0, screen=0x6D, triforce=0xFF, magic_key=1, health=0xCC))
    assert level8_clear_stop(leave, magic_key=1, endpoint=endpoint)
    assert not level8_clear_stop(leave, magic_key=1, endpoint=UNOBSERVED_LEVEL8_CLEAR)


def test_magic_key_ledger_records_key_and_bomb_counts() -> None:
    snap = read_snapshot(_ram(keys=5, bombs=7, magic_key=1, triforce=0x7F))
    ledger = level8_magic_key_ledger(
        snap,
        magic_key_before=0,
        magic_key_after=1,
        keys_before=4,
        bombs_before=8,
    )
    assert ledger["keys_before"] == 4
    assert ledger["keys_after"] == 5
    assert ledger["bombs_before"] == 8
    assert ledger["bombs_after"] == 7
    assert ledger["magic_key_before"] == 0
    assert ledger["magic_key_after"] == 1
    assert ledger["triforce"] == 0x7F


def test_blue_gohma_factory_requires_natural_bow_and_does_not_poke() -> None:
    ram = _ram(level=8, bow=0, arrows=0)
    ctl = make_blue_gohma_controller()
    act = ctl.step(read_snapshot(ram))
    assert ctl.failed
    assert not ctl.success
    assert ctl.poked_arrows is False
    assert ctl.l6_room_check is False
    assert ctl.writes == 0
    assert int(ram[ADDR_ARROWS]) == 0
    assert int(ram[ADDR_BOW]) == 0
    assert "l8_gohma_requires_natural_bow_arrows" in ctl.notes
    assert list(act.action) == list(nes_idle_action())


def test_four_head_gleeok_factory_does_not_assume_0x45() -> None:
    # Live N7/N8 census: RAM type 0x45 HP160, not a ROM assumption.
    assert GLEEOK_FOUR_HEAD_OBJECT_TYPE == 0x45
    ram = _ram(level=8)
    ctl = make_four_head_gleeok_controller()
    act = ctl.step(read_snapshot(ram))
    assert ctl.failed
    assert not ctl.success
    assert ctl.observed_body_type == 0x45
    assert ctl.report()["assumed_0x45"] is False
    assert "left_0x3c_to_0x6d" in ctl.notes
    assert list(act.action) == list(nes_idle_action())


def test_magic_key_and_gleeok_stages_are_named_and_blocked() -> None:
    hops = l8_hops(_env(_ram()))
    mk_names = [name for name, _, _ in hops[1].stages()]
    clear_names = [name for name, _, _ in hops[2].stages()]
    assert mk_names == [
        "level8_north_manhandla_bomb",
        "level8_darknut_key_up",
        "level8_blue_gohma",
        "level8_magic_key_stairs",
    ]
    assert clear_names == [
        "level8_return_passage",
        "level8_four_head_gleeok",
        "level8_heart_shard_leave",
    ]
    for name, ctl, _ in hops[1].stages() + hops[2].stages():
        ctl.step(read_snapshot(_ram(level=8)))
        assert ctl.failed, name
        assert not ctl.success, name
