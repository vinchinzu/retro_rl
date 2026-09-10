"""Unit tests for shared overworld leftover packets and inland-then-descend."""

from __future__ import annotations

import numpy as np

from zelda_i.overworld.stitch import (
    CANDLE_RED,
    CUMULATIVE_TF,
    HYP_SCREEN_53_INLAND_DESCEND,
    MOUTH_STITCHES,
    TF_BIT_BY_LEVEL,
    TF_BITS_ALL,
    UNMEASURED_HANDOFF,
    OverworldHandoff,
    enter_gate_ok,
    handoff_from_ram,
    inland_then_descend,
    y_band_travel_hop,
)
from zelda_i.ram import (
    ADDR_ARROWS,
    ADDR_BOMBS,
    ADDR_BOW,
    ADDR_CANDLE,
    ADDR_FOOD,
    ADDR_HEALTH,
    ADDR_KEYS,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MAGIC_KEY,
    ADDR_MODE,
    ADDR_ROD,
    ADDR_RUPEES,
    ADDR_SCREEN,
    ADDR_SELECTED_ITEM,
    ADDR_TRIFORCE,
    ADDR_WHISTLE,
    PLAY_MODE,
    read_snapshot,
)


def _ram(**fields: int) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = fields.get("mode", PLAY_MODE)
    ram[ADDR_LEVEL] = fields.get("level", 0)
    ram[ADDR_SCREEN] = fields.get("screen", 0x37)
    ram[ADDR_LINK_X] = fields.get("x", 112)
    ram[ADDR_LINK_Y] = fields.get("y", 125)
    ram[ADDR_HEALTH] = fields.get("health", 0x44)
    ram[ADDR_KEYS] = fields.get("keys", 3)
    ram[ADDR_BOMBS] = fields.get("bombs", 8)
    ram[ADDR_RUPEES] = fields.get("rupees", 0)
    ram[ADDR_TRIFORCE] = fields.get("triforce", 0)
    ram[ADDR_WHISTLE] = fields.get("whistle", 0)
    ram[ADDR_FOOD] = fields.get("food", 0)
    ram[ADDR_ROD] = fields.get("rod", 0)
    ram[ADDR_BOW] = fields.get("bow", 0)
    ram[ADDR_ARROWS] = fields.get("arrows", 0)
    ram[ADDR_CANDLE] = fields.get("candle", 0)
    ram[ADDR_SELECTED_ITEM] = fields.get("selected_item", 0)
    ram[ADDR_MAGIC_KEY] = fields.get("magic_key", 0)
    return ram


def _full_packet(**overrides: object) -> OverworldHandoff:
    base: dict[str, object] = {
        "screen": 0x42,
        "link_x": 112,
        "link_y": 125,
        "mode": PLAY_MODE,
        "triforce": 0x3F,
        "keys": 3,
        "bombs": 8,
        "rupees": 20,
        "heart_containers": 5,
        "selected_item": 0,
        "whistle": 1,
        "food": 1,
        "rod": 1,
        "bow": 1,
        "arrows": 1,
        "candle": CANDLE_RED,
        "verified": True,
        "route_eligible": False,
        "evidence": "fixture-live",
    }
    base.update(overrides)
    return OverworldHandoff(**base)  # type: ignore[arg-type]


def test_default_packet_is_not_complete() -> None:
    packet = OverworldHandoff()
    assert packet is not None
    assert packet.verified is False
    assert packet.route_eligible is False
    assert packet.candle is None
    assert not packet.complete()
    assert not UNMEASURED_HANDOFF.complete()
    assert UNMEASURED_HANDOFF.mismatch(read_snapshot(_ram()), _ram()) == (
        "handoff_unmeasured"
    )


def test_incomplete_verified_packet_is_not_complete() -> None:
    partial = OverworldHandoff(screen=0x22, link_x=112, link_y=125, verified=True)
    assert not partial.complete()
    filled_unverified = _full_packet(verified=False)
    assert not filled_unverified.complete()


def test_complete_packet_keeps_route_ineligible() -> None:
    packet = _full_packet()
    assert packet.complete()
    assert packet.route_eligible is False


def test_consumables_mismatch_allows_excess_but_forbids_deficit() -> None:
    # Baseline: keys=3, bombs=8, rupees=20, heart_containers=5 (health 0x44)
    packet = _full_packet(keys=3, bombs=8, rupees=20, heart_containers=5)
    base_ram = _ram(
        screen=0x42,
        x=112,
        y=125,
        health=0x44,
        keys=3,
        bombs=8,
        rupees=20,
        whistle=1,
        food=1,
        rod=1,
        bow=1,
        arrows=1,
        candle=CANDLE_RED,
        triforce=0x3F,
    )
    assert packet.mismatch(read_snapshot(base_ram), base_ram) is None

    # Having extra consumables (e.g. random drop) passes
    excess_ram = _ram(
        screen=0x42,
        x=112,
        y=125,
        health=0x44,
        keys=4,
        bombs=8,
        rupees=25,
        whistle=1,
        food=1,
        rod=1,
        bow=1,
        arrows=1,
        candle=CANDLE_RED,
        triforce=0x3F,
    )
    assert packet.mismatch(read_snapshot(excess_ram), excess_ram) is None

    # Having fewer consumables fails
    fewer_keys_ram = _ram(
        screen=0x42,
        x=112,
        y=125,
        health=0x44,
        keys=2,
        bombs=8,
        rupees=20,
        whistle=1,
        food=1,
        rod=1,
        bow=1,
        arrows=1,
        candle=CANDLE_RED,
        triforce=0x3F,
    )
    assert packet.mismatch(read_snapshot(fewer_keys_ram), fewer_keys_ram) == "handoff_keys_mismatch"


def test_tf_bits_are_one_through_ff() -> None:
    expected = {1: 0x01, 2: 0x02, 3: 0x04, 4: 0x08, 5: 0x10, 6: 0x20, 7: 0x40, 8: 0x80}
    assert TF_BIT_BY_LEVEL == expected
    assert TF_BITS_ALL == 0xFF
    assert CUMULATIVE_TF[6] == 0x3F
    assert CUMULATIVE_TF[7] == 0x7F
    assert CUMULATIVE_TF[8] == 0xFF
    bits = 0
    for level in range(1, 9):
        bits |= TF_BIT_BY_LEVEL[level]
        assert CUMULATIVE_TF[level] == bits
    assert bits == 0xFF


def test_l7_requires_whistle() -> None:
    assert not enter_gate_ok(7, OverworldHandoff())
    assert not enter_gate_ok(7, OverworldHandoff(whistle=0))
    assert enter_gate_ok(7, OverworldHandoff(whistle=1))


def test_l8_requires_candle_2() -> None:
    assert not enter_gate_ok(8, OverworldHandoff())
    assert not enter_gate_ok(8, OverworldHandoff(candle=1))
    assert enter_gate_ok(8, OverworldHandoff(candle=CANDLE_RED))


def test_l9_requires_full_triforce() -> None:
    assert not enter_gate_ok(9, OverworldHandoff())
    assert not enter_gate_ok(9, OverworldHandoff(triforce=0x7F))
    assert enter_gate_ok(9, OverworldHandoff(triforce=0xFF))


def test_mouth_table_l1_leave_through_l9_enter() -> None:
    assert tuple((row.from_level, row.to_level) for row in MOUTH_STITCHES) == tuple(
        (n, n + 1) for n in range(1, 9)
    )
    assert all(row.leave.verified is False for row in MOUTH_STITCHES)
    assert all(row.leave.route_eligible is False for row in MOUTH_STITCHES)
    by_to = {row.to_level: row for row in MOUTH_STITCHES}
    assert by_to[6].mouth_screen == 0x22
    assert by_to[6].status == "live"
    assert by_to[7].mouth_screen == 0x42
    # L6->L7 leave is measured + verified (Phase 1); mouth/pond still hypothesis.
    assert by_to[7].status == "measured leave / mouth+pond hypothesis"
    assert "MEASURED" in by_to[7].notes
    assert "0x22" in by_to[7].notes
    assert by_to[7].leave.verified is False  # doc row, not the live packet
    assert by_to[7].leave.evidence == "measured"
    assert by_to[7].enter_items == ("whistle",)
    assert by_to[8].mouth_screen == 0x6D
    assert "candle2" in by_to[8].enter_items
    assert by_to[9].mouth_screen == 0x05
    assert "tf_0xff" in by_to[9].enter_items


def test_inland_then_descend_left_before_down() -> None:
    spec = HYP_SCREEN_53_INLAND_DESCEND
    leftover = read_snapshot(_ram(screen=0x53, x=224, y=173))
    assert inland_then_descend(leftover, spec) == "LEFT"
    inland = read_snapshot(_ram(screen=0x53, x=160, y=173))
    assert inland_then_descend(inland, spec) == "DOWN"
    band = read_snapshot(_ram(screen=0x53, x=160, y=189))
    assert inland_then_descend(band, spec) == "LEFT"
    hop = y_band_travel_hop(0x52, "LEFT", spec)
    assert hop.align_y is None
    assert hop.y_band == (spec.y_lo, spec.y_hi)


def test_l4_and_l5_leave_xy_packed_ineligible() -> None:
    """Fixture-live L4/L5 leftover xy from Level4Complete / Level5Complete."""
    by_from = {row.from_level: row for row in MOUTH_STITCHES}
    l4 = by_from[4].leave
    assert (l4.screen, l4.link_x, l4.link_y) == (0x45, 128, 125)
    assert l4.triforce == 0x0F
    assert l4.xy_tolerance == 4
    assert l4.verified is False
    assert l4.route_eligible is False
    l5 = by_from[5].leave
    assert (l5.screen, l5.link_x, l5.link_y) == (0x0B, 112, 125)
    assert l5.triforce == 0x1F
    assert l5.whistle == 1
    assert l5.xy_tolerance == 4
    assert l5.verified is False
    assert l5.route_eligible is False


def test_handoff_from_ram_copies_leave_and_stays_ineligible() -> None:
    ram = _ram(
        screen=0x0B,
        x=112,
        y=125,
        triforce=0x1F,
        whistle=1,
        bow=1,
        candle=0,
    )
    packet = handoff_from_ram(ram, evidence="live", verified=False)
    assert packet.screen == 0x0B
    assert packet.triforce == 0x1F
    assert packet.whistle == 1
    assert packet.bow == 1
    assert packet.verified is False
    assert packet.route_eligible is False
    assert not packet.complete()
