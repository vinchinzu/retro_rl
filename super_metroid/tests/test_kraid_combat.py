"""Unit tests for full-knowledge Kraid Super-spray policy (no emulator)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from super_metroid.combat.features import kraid_catalog
from super_metroid.combat.kraid import (
    ROOM_KRAID,
    ROOM_VARIA,
    VARIA_MASK,
    KraidStrategy,
    KraidVariaEvidence,
    VariaEvidence,
    body_hp,
    fight_kraid_action,
)
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


def test_kraid_varia_evidence_success_flag() -> None:
    from super_metroid.combat.kraid import KraidEvidence

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
