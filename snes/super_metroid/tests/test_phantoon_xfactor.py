"""Unit tests for Ice-on X-Factor / popTOON helpers (no emulator)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from super_metroid.combat.phantoon import ROOM_PHANTOON, SEAT_X, WEAPON_MISSILES
from super_metroid.combat.phantoon_xfactor import (
    BEAM_CHARGE,
    BEAM_ICE,
    BEAM_SPAZER,
    BEAM_WAVE,
    PIN_BEAMS,
    PIN_ITEMS,
    PROJ_ICE_SBA,
    PROJ_WAVE_SBA,
    PoptoonProgress,
    PoptoonStep,
    WEAPON_POWER_BOMBS,
    classify_combo,
    decode_beams,
    ice_equipped,
    next_poptoon_step,
    super_ok,
    true_wave_shield,
    xfactor_fire_action,
    xfactor_ready,
)
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
        "missiles": 20,
        "max_missiles": 20,
        "super_missiles": 5,
        "max_super_missiles": 5,
        "power_bombs": 5,
        "max_power_bombs": 5,
        "selected_item": WEAPON_POWER_BOMBS,
        "equipped_beams": PIN_BEAMS,
        "equipped_items": PIN_ITEMS,
        "health": 299,
        "max_health": 299,
    }
    values.update(overrides)
    return replace(base, **values)


def test_ice_bit_is_0x0002_and_pin_is_not_wave_shield() -> None:
    """Pin beams 0x1007 include Ice (0x0002). True X-Factor is Charge+Wave+PB.

    Ice equipped changes the SBA (Ice Shield / Ice+Wave particles). There is
    no pause-menu beam-toggle helper — Ice-on Wave Shield stays red until
    measured otherwise.
    """
    assert BEAM_ICE == 0x0002
    assert BEAM_WAVE == 0x0001
    assert BEAM_SPAZER == 0x0004
    assert BEAM_CHARGE == 0x1000
    assert PIN_BEAMS == 0x1007
    assert PIN_BEAMS & BEAM_ICE
    assert PIN_BEAMS & BEAM_WAVE
    assert PIN_BEAMS & BEAM_SPAZER
    assert PIN_BEAMS & BEAM_CHARGE
    assert ice_equipped(PIN_BEAMS)
    assert not true_wave_shield(PIN_BEAMS)
    bits = decode_beams(PIN_BEAMS)
    assert bits == {"charge": True, "ice": True, "wave": True, "spazer": True}
    assert true_wave_shield(BEAM_CHARGE | BEAM_WAVE)
    assert not true_wave_shield(BEAM_CHARGE | BEAM_WAVE | BEAM_ICE)


def test_xfactor_ready_requires_pb_select_ammo_and_not_morph() -> None:
    assert xfactor_ready(_state())
    assert xfactor_fire_action(_state()) == ("X",)


def test_zero_pb_does_not_fire() -> None:
    dead_ammo = _state(power_bombs=0)
    assert not xfactor_ready(dead_ammo)
    assert xfactor_fire_action(dead_ammo) == ()


def test_wrong_select_or_morph_does_not_fire() -> None:
    assert not xfactor_ready(_state(selected_item=WEAPON_MISSILES))
    assert xfactor_fire_action(_state(selected_item=WEAPON_MISSILES)) == ()
    assert not xfactor_ready(_state(pose=29))
    assert xfactor_fire_action(_state(pose=29)) == ()


def test_super_ok_is_kill_only() -> None:
    assert super_ok(600)
    assert super_ok(1)
    assert not super_ok(601)
    assert not super_ok(2500)
    assert not super_ok(0)


def test_poptoon_next_action_round1_2_2_xf() -> None:
    p = PoptoonProgress()
    assert (
        next_poptoon_step(p, hp=2500, power_bombs=5, ice_on=True)
        is PoptoonStep.FIRE_MISSILE
    )
    p.missiles_this_round = 2
    assert (
        next_poptoon_step(p, hp=2300, power_bombs=5) is PoptoonStep.FIRE_MISSILE
    )
    p.missiles_this_round = 4
    assert (
        next_poptoon_step(p, hp=2100, power_bombs=5) is PoptoonStep.CHARGE_XFACTOR
    )


def test_poptoon_zero_pb_blocks_xfactor() -> None:
    p = PoptoonProgress(missiles_this_round=4)
    assert (
        next_poptoon_step(p, hp=2100, power_bombs=0) is PoptoonStep.BLOCKED_NO_PB
    )


def test_poptoon_ice_on_no_chip_blocks() -> None:
    p = PoptoonProgress(missiles_this_round=4)
    assert (
        next_poptoon_step(
            p, hp=2100, power_bombs=5, ice_on=True, combo_chips=False
        )
        is PoptoonStep.BLOCKED_ICE
    )


def test_poptoon_round2_super_only_if_kill() -> None:
    p = PoptoonProgress(round_index=2, missiles_this_round=4)
    assert next_poptoon_step(p, hp=600, power_bombs=5) is PoptoonStep.FIRE_SUPER
    assert (
        next_poptoon_step(p, hp=601, power_bombs=5) is PoptoonStep.CHARGE_XFACTOR
    )
    p.super_fired = True
    assert next_poptoon_step(p, hp=600, power_bombs=5) is PoptoonStep.DONE


def test_classify_combo_ice_vs_wave() -> None:
    assert classify_combo([]) == "none"
    assert classify_combo([PROJ_WAVE_SBA]) == "wave_shield"
    assert classify_combo([PROJ_ICE_SBA]) == "ice_shield"
    assert classify_combo([PROJ_ICE_SBA, PROJ_WAVE_SBA]) == "ice_wave_shield"
