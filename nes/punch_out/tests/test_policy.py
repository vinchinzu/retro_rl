from __future__ import annotations

import numpy as np

from punch_out.policy import BoutMode, GlassJoePolicy
from punch_out.ram import (
    ADDR_FIGHT_FLAG,
    ADDR_HEALTH,
    ADDR_OPP_HEALTH,
    ADDR_OPP_PATTERN_SET,
    FIGHT_IN_RING,
)


def _ram(*, mac: int = 96, opp: int = 96, p3b: int = 115) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_HEALTH] = mac
    ram[ADDR_OPP_HEALTH] = opp
    ram[ADDR_FIGHT_FLAG] = FIGHT_IN_RING
    ram[ADDR_OPP_PATTERN_SET] = p3b
    ram[0x0323] = 2
    ram[0x0324] = 0
    return ram


def test_policy_waits_then_punches_taunt() -> None:
    pol = GlassJoePolicy()
    a = pol.tick(_ram(p3b=115))
    assert a.reason == "wait_taunt"
    assert pol.mode == BoutMode.WAIT_TAUNT

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
