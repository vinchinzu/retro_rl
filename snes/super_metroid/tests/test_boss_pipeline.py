"""Unit tests for boss pipeline foundations (catalog, protocol, primitives)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from super_metroid.combat.features import (
    BOSS_CATALOG,
    BOSS_SPINE_ORDER,
    FEATURE_DIM,
    boss_defeated_in_state,
    feature_vector,
    features_from_state,
    get_boss_catalog,
    kraid_catalog,
    list_boss_catalog,
    phantoon_catalog,
    validate_live_enemy,
)
from super_metroid.combat.audit import structured_combat_attempt_audit
from super_metroid.assist import AssistTelemetry
from super_metroid.combat.natural_entry import is_capture_frame
from super_metroid.combat.primitives import (
    PhaseMachine,
    face_toward_action,
    lane_hold_action,
    range_kite_action,
    spray_action,
)
from super_metroid.combat.protocol import (
    BossEvidence,
    BossSegment,
    CallableBossStrategy,
    strategy_summary,
    wrap_crocomire_as_boss_strategy,
    wrap_golden_torizo_as_boss_strategy,
    wrap_kraid_as_boss_strategy,
    wrap_mother_brain_as_boss_strategy,
    wrap_ridley_as_boss_strategy,
)
from super_metroid.policy import StateRequirement
from super_metroid.ram import GameplayPhase, parse_state


def _state(**overrides):
    ram = np.zeros(0x2000, dtype=np.uint8)
    base = parse_state(ram, frame=0)
    return replace(base, phase=GameplayPhase.ORDINARY_GAMEPLAY, **overrides)


def test_full_catalog_registry_covers_spine_bosses() -> None:
    required = {
        "bomb_torizo",
        "spore_spawn",
        "kraid",
        "phantoon",
        "botwoon",
        "draygon",
        "crocomire",
        "ridley",
        "golden_torizo",
        "mother_brain",
    }
    assert required.issubset(BOSS_CATALOG.keys())
    assert set(BOSS_SPINE_ORDER).issubset(BOSS_CATALOG.keys())
    # Spine order matches KPDR priority intent.
    assert BOSS_SPINE_ORDER[0] == "kraid"
    assert BOSS_SPINE_ORDER[-1] == "mother_brain"


def test_structured_combat_dry_run_emits_complete_audit_trail() -> None:
    telemetry = AssistTelemetry()
    telemetry.energy.writes = 2
    telemetry.ammo["missiles"].writes = 1
    audit = structured_combat_attempt_audit(telemetry)
    assert audit.has_complete_instrumentation
    assert audit.ram_writes == 3
    assert audit.mid_run_loads == 0
    assert audit.assists == {"unlimited_resources": 3}


def test_get_boss_catalog_and_list() -> None:
    kraid = get_boss_catalog("kraid")
    assert kraid.room_id == 0xA59F
    assert kraid.boss_bit_mask == 0x01
    assert kraid.continuous_status == "wired"
    ph = phantoon_catalog()
    assert ph.room_id == 0xCD13
    assert ph.max_hp == 2500
    deferred = list_boss_catalog(continuous_status="deferred")
    assert any(e.boss_id == "phantoon" for e in deferred)
    assert all(e.continuous_status == "deferred" for e in deferred)


def test_mother_brain_has_phases() -> None:
    mb = get_boss_catalog("mother_brain")
    assert len(mb.phases) == 3
    assert mb.defeat_event_id == 0x0E
    assert mb.phases[0].phase_id == "mb1"


def test_boss_defeated_in_state_uses_area_bits() -> None:
    catalog = kraid_catalog()
    # Brinstar is boss_bits[1]; Kraid bit 0x01.
    bits = (0, 0x01, 0, 0, 0, 0, 0, 0)
    state = _state(room_id=0xA59F, boss_bits=bits)
    assert boss_defeated_in_state(state, catalog)
    state_clear = _state(room_id=0xA59F, boss_bits=(0,) * 8)
    assert not boss_defeated_in_state(state_clear, catalog)


def test_validate_live_enemy_rejects_wrong_room() -> None:
    catalog = kraid_catalog()
    state = _state(room_id=0x0000, enemy0_hp=1000, enemy0_spritemap=0x1000, num_enemies=1)
    fails = validate_live_enemy(state, catalog)
    assert any("room" in f for f in fails)


def test_validate_live_enemy_accepts_kraid_body() -> None:
    catalog = kraid_catalog()
    state = _state(
        room_id=0xA59F,
        enemy0_hp=1000,
        enemy0_spritemap=0xABCD,
        num_enemies=3,
        samus_x=100,
        enemy0_x=300,
    )
    assert validate_live_enemy(state, catalog) == []
    assert validate_live_enemy(state, catalog, require_full_hp=True) == []


def test_features_use_catalog_inactive_spritemaps() -> None:
    catalog = get_boss_catalog("bomb_torizo")
    state = _state(
        room_id=0x9804,
        enemy0_hp=800,
        enemy0_spritemap=0x87D0,
        num_enemies=1,
        samus_x=81,
        enemy0_x=219,
    )
    feat = features_from_state(state, catalog)
    assert not feat.enemy_active
    vec = feature_vector(feat)
    assert vec.shape == (FEATURE_DIM,)


def test_is_capture_frame_modes() -> None:
    catalog = get_boss_catalog("bomb_torizo")
    active = _state(
        room_id=0x9804,
        enemy0_hp=800,
        enemy0_spritemap=0xAA12,
        num_enemies=1,
        samus_x=100,
        enemy0_x=200,
    )
    assert is_capture_frame(active, catalog, mode="active")
    statue = _state(
        room_id=0x9804,
        enemy0_hp=800,
        enemy0_spritemap=0x87D0,
        num_enemies=1,
        samus_x=81,
        enemy0_x=219,
    )
    assert is_capture_frame(statue, catalog, mode="statue")
    assert not is_capture_frame(statue, catalog, mode="active")


def test_primitives_lane_and_spray() -> None:
    assert "RIGHT" in lane_hold_action(30, min_x=50, max_x=260)
    assert "LEFT" in lane_hold_action(300, min_x=50, max_x=260)
    mid = lane_hold_action(120, min_x=50, max_x=260, face="RIGHT")
    assert mid == ("RIGHT",)
    spray = spray_action(0, fire_period=12, fire_hold_frames=6, jump_period=50, jump_hold_frames=10)
    assert "RIGHT" in spray and "A" in spray and "X" in spray
    face = face_toward_action(100, 200, fire=True)
    assert face == ("RIGHT", "X")
    kite = range_kite_action(100, 200, min_range=70, max_range=120, frame_index=0)
    assert "RIGHT" in kite


def test_phase_machine() -> None:
    m = PhaseMachine(["activate", "fight", "exit"])
    assert m.current == "activate"
    m.advance()
    assert m.current == "fight"
    m.advance()
    m.advance()
    assert m.done and m.ok
    m2 = PhaseMachine(["a"])
    m2.fail("timeout")
    assert not m2.ok
    assert m2.failed == "timeout"


def test_boss_evidence_to_dict() -> None:
    ev = BossEvidence.from_parts(
        boss_id="kraid",
        start_frame=0,
        end_frame=100,
        outcome="varia_collected",
        success=True,
        final_room_id=0xA6E2,
        boss_defeated=True,
        body_zero_frame=80,
    )
    d = ev.to_dict()
    assert d["success"] is True
    assert d["finalRoomIdHex"] == "0xA6E2"
    assert d["detail"]["body_zero_frame"] == 80


def test_callable_boss_strategy_and_segment() -> None:
    catalog = kraid_catalog()

    def play(session):  # pragma: no cover - not invoked here
        raise AssertionError("should not play")

    strategy = CallableBossStrategy(
        boss_id="kraid",
        play_fn=play,
        entry_requirement=StateRequirement(room_id=0xA59F),
        catalog_entry=catalog,
    )
    assert strategy.catalog.name == "Kraid"
    summary = strategy_summary(strategy)
    assert summary["roomIdHex"] == "0xA59F"
    segment = BossSegment(strategy=strategy, segment_id="kraid_entry_to_varia")
    assert segment.id == "kraid_entry_to_varia"
    assert segment.entry.room_id == 0xA59F


def test_wrap_kraid_strategy_entry() -> None:
    strategy = wrap_kraid_as_boss_strategy()
    assert strategy.boss_id == "kraid"
    assert strategy.entry.room_id == 0xA59F
    assert strategy.catalog.max_hp == 1000


def test_deferred_boss_wrappers_match_catalog_rooms() -> None:
    wrapped = (
        (wrap_ridley_as_boss_strategy(), 0xB32E),
        (wrap_mother_brain_as_boss_strategy(), 0xDD58),
        (wrap_crocomire_as_boss_strategy(), 0xA98D),
        (wrap_golden_torizo_as_boss_strategy(), 0xB283),
    )
    for strategy, room_id in wrapped:
        assert strategy.catalog.room_id == room_id
        assert strategy.entry.room_id == strategy.catalog.room_id
