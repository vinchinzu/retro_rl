from __future__ import annotations

import numpy as np

from punch_out.policy import (
    ATTACK_ACTS,
    BoutMode,
    COUNTER_FRAMES,
    DODGE_HOLD,
    DODGE_WAIT,
    GlassJoePolicy,
)
from punch_out.ram import (
    ADDR_FIGHT_FLAG,
    ADDR_HEALTH,
    ADDR_OPP_ACTION,
    ADDR_OPP_HEALTH,
    ADDR_OPP_PATTERN_SET,
    FIGHT_IN_RING,
)


def _ram(
    *,
    mac: int = 96,
    opp: int = 96,
    p3b: int = 115,
    act: int = 3,
) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_HEALTH] = mac
    ram[ADDR_OPP_HEALTH] = opp
    ram[ADDR_FIGHT_FLAG] = FIGHT_IN_RING
    ram[ADDR_OPP_PATTERN_SET] = p3b
    ram[ADDR_OPP_ACTION] = act
    ram[0x0323] = 2
    ram[0x0324] = 0
    return ram


def test_policy_waits_then_punches_taunt() -> None:
    pol = GlassJoePolicy()
    a = pol.tick(_ram(p3b=115))
    assert a.reason == "stand_idle"
    assert pol.mode == BoutMode.STAND

    a = pol.tick(_ram(p3b=150))
    assert pol.mode == BoutMode.PUNCH_TAUNT
    assert a.reason in ("taunt_a", "taunt_rec")


def test_policy_counts_opp_knockdown() -> None:
    pol = GlassJoePolicy()
    pol.tick(_ram(opp=96))
    pol.tick(_ram(opp=0))
    assert pol.opp_kd == 1
    assert pol.mode == BoutMode.WATCH_KD


def test_getup_uses_double_frame_presses() -> None:
    pol = GlassJoePolicy()
    pol.tick(_ram(mac=96))
    pol.tick(_ram(mac=0))
    assert pol.mode == BoutMode.GETUP
    reasons = []
    for _ in range(6):
        reasons.append(pol.tick(_ram(mac=0)).reason)
    assert "getup_a" in reasons
    assert "getup_b" in reasons
    assert "getup_idle" in reasons


def test_attack_act_arms_timed_dodge_via_reasons() -> None:
    pol = GlassJoePolicy()
    # Non-attack act first
    a = pol.tick(_ram(p3b=120, act=3))
    assert a.reason == "stand_idle"
    # Enter attack act → first frame of dodge wait
    a = pol.tick(_ram(p3b=120, act=7))
    assert a.reason == "dodge_wait"
    assert 7 in ATTACK_ACTS


def test_dodge_wait_then_hold_then_counter() -> None:
    pol = GlassJoePolicy()
    pol.tick(_ram(p3b=120, act=3))
    pol.tick(_ram(p3b=120, act=7))
    reasons = []
    for _ in range(DODGE_WAIT + DODGE_HOLD + COUNTER_FRAMES + 3):
        reasons.append(pol.tick(_ram(p3b=120, act=7)).reason)
    assert "dodge_wait" in reasons
    assert any(r.startswith("dodge_") and r != "dodge_wait" for r in reasons)
    assert "counter_a" in reasons or "counter_rec" in reasons
    # After timers drain, back to standing idle (no continuous L/R spam).
    assert reasons[-1] == "stand_idle"
    assert not any(r in ("survive_left", "survive_right") for r in reasons)


def test_taunt_beats_armed_dodge() -> None:
    pol = GlassJoePolicy()
    pol.tick(_ram(p3b=120, act=3))
    pol.tick(_ram(p3b=120, act=7))  # arm dodge
    a = pol.tick(_ram(p3b=150, act=7))
    assert pol.mode == BoutMode.PUNCH_TAUNT
    assert a.reason in ("taunt_a", "taunt_rec")


def test_getup_clears_dodge_timers() -> None:
    pol = GlassJoePolicy()
    pol.tick(_ram(p3b=120, act=3))
    pol.tick(_ram(p3b=120, act=7))  # arm dodge
    pol.tick(_ram(mac=0, p3b=120, act=7))
    assert pol.mode == BoutMode.GETUP
    # After get-up recovery, no leftover dodge reasons from pre-fall arm.
    pol.tick(_ram(mac=96, p3b=120, act=7))  # rise; same act should not re-arm
    a = pol.tick(_ram(mac=96, p3b=120, act=7))
    assert a.reason == "stand_idle"
