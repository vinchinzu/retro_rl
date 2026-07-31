"""Unit tests for full-knowledge Kraid Super-spray policy (no emulator)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from super_metroid.combat.features import kraid_catalog
from super_metroid.combat.kraid import (
    ROOM_KRAID,
    ROOM_VARIA,
    VARIA_MASK,
    KraidEvidence,
    KraidStrategy,
    KraidVariaEvidence,
    VariaEvidence,
    body_hp,
    fight_kraid_action,
)
from super_metroid.combat.protocol import wrap_kraid_as_boss_strategy
from super_metroid.ram import GameplayPhase, parse_state


def _state(**overrides):
    ram = np.zeros(0x2000, dtype=np.uint8)
    base = parse_state(ram, frame=0)
    return replace(
        base,
        phase=GameplayPhase.ORDINARY_GAMEPLAY,
        room_id=ROOM_KRAID,
        **overrides,
    )


def test_kraid_catalog_facts() -> None:
    catalog = kraid_catalog()
    assert catalog.room_id == 0xA59F
    assert catalog.max_hp == 1000
    assert catalog.primary_weapon == "supers"
    assert catalog.name == "Kraid"


def test_body_hp_reads_enemy0() -> None:
    state = _state(enemy0_hp=1000)
    assert body_hp(state) == 1000


def test_body_hp_uses_only_enemy0_value() -> None:
    state = _state(enemy0_hp=237, enemy0_x=475, enemy0_y=240)
    assert body_hp(state) == 237


def test_entry_lane_walks_right_when_near_door() -> None:
    """Doorway entry (~x=39) should walk right into the arena."""
    state = _state(samus_x=39, samus_y=395, enemy0_hp=1000)
    action = fight_kraid_action(state, frame_index=0)
    assert "RIGHT" in action


def test_too_far_right_backs_off() -> None:
    state = _state(samus_x=320, samus_y=395, enemy0_hp=800)
    action = fight_kraid_action(state, frame_index=0)
    assert "LEFT" in action


def test_mid_lane_faces_and_fires() -> None:
    strategy = KraidStrategy(fire_period=12, fire_hold_frames=6, jump_period=50, jump_hold_frames=10)
    state = _state(samus_x=120, samus_y=395, enemy0_hp=800)
    # Frame 0: jump + fire window.
    action = fight_kraid_action(state, frame_index=0, strategy=strategy)
    assert "RIGHT" in action
    assert "A" in action
    assert "X" in action
    # Frame 12: fire, no jump, dash allowed.
    action_fire = fight_kraid_action(state, frame_index=12, strategy=strategy)
    assert "RIGHT" in action_fire
    assert "X" in action_fire
    assert "A" not in action_fire
    assert "B" in action_fire


def test_low_hp_body_still_uses_spray_action() -> None:
    state = _state(samus_x=120, samus_y=395, enemy0_hp=1)
    action = fight_kraid_action(state, frame_index=0)
    assert "X" in action
    assert "RIGHT" in action


def test_zero_hp_body_dead_action_is_exit_oriented_not_fire() -> None:
    state = _state(samus_x=150, samus_y=395, enemy0_hp=0, pose=1)
    action = fight_kraid_action(state, frame_index=100, body_dead=True)
    assert "RIGHT" in action
    assert "X" not in action


def test_mid_arena_y_band_keeps_fire_or_horizontal_control() -> None:
    for y in (250, 320, 395, 460):
        state = _state(samus_x=120, samus_y=y, enemy0_hp=500)
        action = fight_kraid_action(state, frame_index=12)
        assert "X" in action or "RIGHT" in action


def test_kraid_strategy_defaults() -> None:
    strategy = KraidStrategy()
    assert strategy.min_x == 50
    assert strategy.max_x == 260
    assert strategy.jump_hold_frames == 10
    assert strategy.jump_period == 50
    assert strategy.fire_hold_frames == 6
    assert strategy.fire_period == 12
    assert strategy.max_fight_frames == 15_000
    assert strategy.boss_bit_grace_frames == 1_200


def test_body_dead_moves_right() -> None:
    state = _state(samus_x=150, samus_y=395, enemy0_hp=0, pose=1)
    action = fight_kraid_action(state, frame_index=100, body_dead=True)
    assert "RIGHT" in action


def test_offmap_idles() -> None:
    state = _state(samus_x=65000, samus_y=395, enemy0_hp=1000)
    assert fight_kraid_action(state, 0) == ()


def test_varia_constants() -> None:
    assert ROOM_VARIA == 0xA6E2
    assert VARIA_MASK == 0x0001


def test_varia_evidence_dict() -> None:
    evidence = VariaEvidence(
        start_frame=1520,
        varia_room_frame=1635,
        collect_frame=1975,
        end_frame=2475,
        final_items=0x1105,
        final_room_id=ROOM_VARIA,
        samus_x=118,
        samus_y=127,
        outcome="varia_collected",
    )
    d = evidence.to_dict()
    assert d["outcome"] == "varia_collected"
    assert d["final_items_hex"] == "0x1105"
    assert d["final_room_id_hex"] == "0xA6E2"


def test_varia_evidence_dict_keys_are_stable() -> None:
    evidence = VariaEvidence(
        start_frame=0,
        varia_room_frame=None,
        collect_frame=None,
        end_frame=10,
        final_items=0,
        final_room_id=ROOM_KRAID,
        samus_x=50,
        samus_y=395,
        outcome="no_varia_room",
    )
    assert set(evidence.to_dict()) == {
        "start_frame",
        "varia_room_frame",
        "collect_frame",
        "end_frame",
        "final_items",
        "final_items_hex",
        "final_room_id",
        "final_room_id_hex",
        "samus_x",
        "samus_y",
        "outcome",
    }


def test_kraid_varia_evidence_success_flag() -> None:
    fight = KraidEvidence(
        start_frame=0,
        body_zero_frame=1321,
        boss_bit_frame=1520,
        end_frame=1520,
        peak_body_hp=1000,
        min_body_hp=0,
        action_frames=1520,
        final_body_hp=0,
        boss_bit_set=True,
        outcome="kraid_defeated",
    )
    varia = VariaEvidence(
        start_frame=1520,
        varia_room_frame=1635,
        collect_frame=1975,
        end_frame=2475,
        final_items=0x1105,
        final_room_id=ROOM_VARIA,
        samus_x=118,
        samus_y=127,
        outcome="varia_collected",
    )
    assert KraidVariaEvidence(fight=fight, varia=varia).to_dict()["success"] is True


def test_kraid_varia_evidence_dict_keys_are_stable() -> None:
    fight = KraidEvidence(
        start_frame=0,
        body_zero_frame=None,
        boss_bit_frame=None,
        end_frame=10,
        peak_body_hp=1000,
        min_body_hp=1000,
        action_frames=10,
        final_body_hp=1000,
        boss_bit_set=False,
        outcome="timeout",
    )
    varia = VariaEvidence(
        start_frame=10,
        varia_room_frame=None,
        collect_frame=None,
        end_frame=10,
        final_items=0,
        final_room_id=ROOM_KRAID,
        samus_x=100,
        samus_y=395,
        outcome="skipped_fight_failed",
    )
    assert set(KraidVariaEvidence(fight=fight, varia=varia).to_dict()) == {
        "fight",
        "varia",
        "success",
    }


def test_kraid_boss_strategy_protocol_smoke() -> None:
    strategy = wrap_kraid_as_boss_strategy()
    assert strategy.boss_id == "kraid"
    assert strategy.catalog.primary_weapon == "supers"
    assert strategy.catalog.max_hp == 1000
    assert strategy.entry.matches(_state())
