"""Unit tests for the no-assist Phantoon left-corner policy (no emulator)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from super_metroid.combat.features import phantoon_catalog
from super_metroid.combat.phantoon import (
    PHANTOON_INVISIBLE,
    PHANTOON_VULNERABLE,
    ROOM_PHANTOON,
    SEAT_X,
    VULNERABLE_SPRITEMAPS,
    WEAPON_BEAM,
    WEAPON_MISSILES,
    PhantoonEvidence,
    PhantoonStrategy,
    eye_ilist_open,
    eye_open,
    fight_phantoon_action,
    floor_release_ok,
    func_vulnerable,
    in_release_band,
    charge_window_ok,
    rain_corner_morph,
    rain_phase,
    rain_vulnerable,
    right_park,
    phantoon_phase,
    play_phantoon_fight,
    seated,
)
from super_metroid.combat.protocol import wrap_phantoon_as_boss_strategy
from super_metroid.ram import GameplayPhase, parse_state


def _state(**overrides):
    ram = np.zeros(0x2000, dtype=np.uint8)
    base = parse_state(ram, frame=0)
    values = {
        "phase": GameplayPhase.ORDINARY_GAMEPLAY,
        "room_id": ROOM_PHANTOON,
        "samus_x": SEAT_X,
        "samus_y": 187,
        "pose": 1,
        "facing": 8,
        "enemy0_x": 120,
        "enemy0_y": 108,
        "enemy0_hp": 2500,
        "enemy0_spritemap": 0xDEDD,
        "missiles": 20,
        "max_missiles": 20,
        "selected_item": WEAPON_BEAM,
        "num_enemies": 4,
        "health": 299,
        "max_health": 299,
    }
    values.update(overrides)
    return replace(base, **values)


class _Session:
    """Small deterministic session double for the bounded fight loop."""

    def __init__(self, state, *, hp_after_step=None, set_boss_bit=False):
        self.state = state
        self.frame = state.frame
        self.actions = []
        self.hp_after_step = hp_after_step
        self.set_boss_bit = set_boss_bit
        self.env = None

    def step(self, action, reason):
        self.actions.append((action, reason))
        self.frame += 1
        updates = {"frame": self.frame}
        if self.hp_after_step is not None and len(self.actions) >= 8:
            updates["enemy0_hp"] = self.hp_after_step
        hp_now = updates.get("enemy0_hp", self.state.enemy0_hp)
        if self.set_boss_bit and hp_now == 0:
            bits = list(self.state.boss_bits)
            bits[3] |= 0x01
            updates["boss_bits"] = tuple(bits)
        self.state = replace(self.state, **updates)
        return self.state


def test_seated_is_standing_left_corner() -> None:
    assert seated(_state())
    assert not seated(_state(pose=29))
    assert not seated(_state(pose=81))
    assert not seated(_state(samus_x=90))
    assert not seated(_state(samus_y=120, pose=1))


def test_eye_open_uses_measured_body_spritemaps() -> None:
    assert 0xDEF1 in VULNERABLE_SPRITEMAPS
    assert eye_open(_state(enemy0_spritemap=0xDEF1))
    assert not eye_open(_state(enemy0_spritemap=0xDEDD))
    assert eye_ilist_open(0xCC53)
    assert eye_ilist_open(0xCCA1)
    assert not eye_ilist_open(0xCC7F)
    assert func_vulnerable(0xD60D)
    assert not func_vulnerable(0xD5E7)


def test_seated_closed_eye_charges() -> None:
    assert fight_phantoon_action(
        _state(), 0, PhantoonStrategy(weapon=WEAPON_BEAM)
    ) == ("X",)
    assert fight_phantoon_action(_state(), 0) == ()


def test_seated_open_eye_zero_ammo_does_not_fire() -> None:
    state = _state(
        enemy0_spritemap=0xDEF1,
        missiles=0,
        selected_item=WEAPON_MISSILES,
    )
    action = fight_phantoon_action(
        state, 0, PhantoonStrategy(weapon=WEAPON_MISSILES)
    )
    assert "X" not in action


def test_seated_open_eye_missiles_fires() -> None:
    state = _state(enemy0_spritemap=0xDEF1, selected_item=WEAPON_MISSILES)
    action = fight_phantoon_action(
        state, 0, PhantoonStrategy(weapon=WEAPON_MISSILES)
    )
    assert "X" in action


def test_rain_vulnerable_is_d767_d788_not_figure8() -> None:
    assert rain_vulnerable(0xD788)
    assert rain_vulnerable(0xD767)
    assert not rain_vulnerable(0xD60D)
    assert not rain_vulnerable(0xD5E7)


def test_charge_window_ok_skips_rain_and_right_fig8() -> None:
    assert charge_window_ok(0xD60D, 120)
    assert charge_window_ok(0xD4A8, 120)
    assert not charge_window_ok(0xD60D, 203)
    assert not charge_window_ok(0xD788, 128)
    assert not charge_window_ok(0xD767, 128)
    assert not charge_window_ok(0xD82A, 203)


def test_rain_phase_is_cycle_not_fig8() -> None:
    assert rain_phase(0xD82A)
    assert rain_phase(0xD73F)
    assert rain_phase(0xD767)
    assert rain_phase(0xD788)
    assert rain_phase(0xD7D5)
    assert rain_phase(0xD7F7)
    assert not rain_phase(0xD60D)
    assert not rain_phase(0xD5E7)
    assert not rain_phase(0xD4A8)


def test_right_park_and_rain_corner_morph() -> None:
    assert right_park(203)
    assert not right_park(120)
    assert rain_corner_morph(_state(pose=29, samus_x=40, samus_y=187))
    assert rain_corner_morph(_state(pose=29, samus_x=200, samus_y=187))
    assert not rain_corner_morph(_state(pose=1, samus_x=40, samus_y=187))
    assert not rain_corner_morph(_state(pose=29, samus_x=100, samus_y=187))


def test_floor_release_ok_is_stand_crouch_not_jump() -> None:
    assert floor_release_ok(_state(pose=1, samus_y=187))
    assert floor_release_ok(_state(pose=3, samus_y=187))
    assert floor_release_ok(_state(pose=11, samus_y=187))
    assert not floor_release_ok(_state(pose=21, samus_y=174))
    assert not floor_release_ok(_state(pose=25, samus_y=187))
    assert not floor_release_ok(_state(pose=81, samus_y=174))


def test_release_band_is_window1_height_not_floor_hop() -> None:
    """W1 charge chip dy=41; W2 miss y=148 vs 83 is dy=65, outside 28–56."""
    assert in_release_band(_state(samus_y=149, enemy0_y=108))
    assert in_release_band(_state(samus_y=149, enemy0_y=96))
    assert not in_release_band(_state(samus_y=174, enemy0_y=96))
    assert not in_release_band(_state(samus_y=187, enemy0_y=108))
    assert not in_release_band(_state(samus_y=100, enemy0_y=108))
    # Right fig-8 (203, 83): W1 dy band is y=111–139, not y≈149.
    assert in_release_band(_state(samus_y=124, enemy0_y=83, enemy0_x=203))
    assert in_release_band(_state(samus_y=139, enemy0_y=83, enemy0_x=203))
    assert not in_release_band(_state(samus_y=148, enemy0_y=83, enemy0_x=203))
    assert not in_release_band(_state(samus_y=187, enemy0_y=83, enemy0_x=203))


def test_seated_right_side_open_does_not_chase() -> None:
    state = _state(
        enemy0_spritemap=0xDEF1,
        enemy0_x=200,
        selected_item=WEAPON_MISSILES,
    )
    action = fight_phantoon_action(
        state, 0, PhantoonStrategy(weapon=WEAPON_MISSILES)
    )
    assert action == ()
    assert "X" not in fight_phantoon_action(
        state, 0, PhantoonStrategy(weapon=WEAPON_BEAM)
    )


def test_not_seated_walks_left_to_corner() -> None:
    state = _state(samus_x=120, pose=1)
    assert "LEFT" in fight_phantoon_action(state, 0)


def test_zero_hp_returns_idle() -> None:
    assert fight_phantoon_action(_state(enemy0_hp=0), 0) == ()


def test_phantoon_phase_open_spritemap_is_vulnerable() -> None:
    assert phantoon_phase(_state(enemy0_spritemap=0xDEF1)) == PHANTOON_VULNERABLE
    assert phantoon_phase(_state(enemy0_spritemap=0xDEDD)) == PHANTOON_INVISIBLE
    assert phantoon_phase(_state(enemy0_hp=0)) == "defeated"


def test_fight_labels_body_zero_without_boss_bit() -> None:
    session = _Session(_state(enemy0_hp=2500), hp_after_step=0)
    evidence = play_phantoon_fight(
        session,
        strategy=PhantoonStrategy(max_fight_frames=12, weapon=WEAPON_BEAM),
        require_boss_bit=False,
    )
    assert evidence.outcome == "phantoon_body_zero_no_boss_bit"
    assert evidence.body_zero_frame is not None
    assert evidence.boss_bit_set is False


def test_fight_waits_death_anim_for_boss_bit() -> None:
    session = _Session(
        _state(enemy0_hp=2500),
        hp_after_step=0,
        set_boss_bit=True,
    )
    evidence = play_phantoon_fight(
        session,
        strategy=PhantoonStrategy(
            max_fight_frames=40,
            weapon=WEAPON_BEAM,
            boss_bit_grace_frames=1_200,
        ),
        require_boss_bit=True,
    )
    assert evidence.outcome == "phantoon_defeated"
    assert evidence.body_zero_frame is not None
    assert evidence.boss_bit_frame is not None
    assert evidence.boss_bit_set is True


def test_phantoon_evidence_dict_preserves_phase_metrics() -> None:
    evidence = PhantoonEvidence(
        start_frame=10,
        body_zero_frame=80,
        boss_bit_frame=90,
        end_frame=90,
        peak_body_hp=2500,
        min_body_hp=0,
        action_frames=80,
        final_body_hp=0,
        boss_bit_set=True,
        outcome="phantoon_defeated",
        phase_transitions=(("vulnerable", 10), ("defeated", 80)),
        shots_fired=12,
        windows=4,
    )
    payload = evidence.to_dict()
    assert payload["body_zero_frame"] == 80
    assert payload["shots_fired"] == 12
    assert payload["windows"] == 4
    assert payload["outcome"] == "phantoon_defeated"


def test_wrapper_entry_room_and_catalog() -> None:
    strategy = wrap_phantoon_as_boss_strategy()
    assert strategy.entry.room_id == ROOM_PHANTOON
    assert strategy.catalog.name == "Phantoon"
    assert phantoon_catalog().max_hp == 2500
