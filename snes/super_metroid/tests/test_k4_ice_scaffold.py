"""Unit scaffold for K4 Ice pure package (no emulator)."""

from __future__ import annotations

from types import SimpleNamespace

from super_metroid.routes.kpdr import get_segment
from super_metroid.routes.kpdr.ice import (
    ICE_SUPER_Y_MAX,
    ICE_SUPER_Y_MIN,
    on_ice_super_lip,
    play_business_to_ice_gate,
)
from super_metroid.routes.kpdr.ice.geometry import in_ice_super_band
from super_metroid.routes.kpdr.rooms import ROOM_BUSINESS, ROOM_ICE_GATE


def test_business_to_ice_gate_export_and_registry() -> None:
    assert get_segment("business_to_ice_gate") is play_business_to_ice_gate


def test_on_ice_super_lip_predicate() -> None:
    lip = SimpleNamespace(
        room_id=ROOM_BUSINESS,
        samus_x=61,
        samus_y=922,
        velocity_y=0,
        pose=2,
    )
    assert on_ice_super_lip(lip) is True  # type: ignore[arg-type]

    too_high = SimpleNamespace(
        room_id=ROOM_BUSINESS,
        samus_x=61,
        samus_y=ICE_SUPER_Y_MIN - 20,
        velocity_y=0,
        pose=2,
    )
    assert on_ice_super_lip(too_high) is False  # type: ignore[arg-type]

    wrong_room = SimpleNamespace(
        room_id=ROOM_ICE_GATE,
        samus_x=61,
        samus_y=922,
        velocity_y=0,
        pose=2,
    )
    assert on_ice_super_lip(wrong_room) is False  # type: ignore[arg-type]


def test_in_ice_super_band_mid_x() -> None:
    mid = SimpleNamespace(
        room_id=ROOM_BUSINESS,
        samus_x=150,
        samus_y=(ICE_SUPER_Y_MIN + ICE_SUPER_Y_MAX) // 2,
        velocity_y=1,
        pose=25,
    )
    assert in_ice_super_band(mid) is True  # type: ignore[arg-type]
    assert on_ice_super_lip(mid) is False  # type: ignore[arg-type]
