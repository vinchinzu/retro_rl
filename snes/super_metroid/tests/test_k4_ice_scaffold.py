"""Unit scaffold for K4 Ice pure package (no emulator)."""

from __future__ import annotations

from types import SimpleNamespace

from super_metroid.routes.kpdr import get_segment
from super_metroid.routes.kpdr.ice import (
    ICE_BEAM_MASK,
    ICE_SUPER_Y_MAX,
    ICE_SUPER_Y_MIN,
    on_ice_super_lip,
    play_business_to_ice_gate,
    play_ice_acid_to_snake,
    play_ice_gate_to_acid,
    play_ice_snake_to_ice,
    play_ice_snake_to_tutorial,
    play_ice_to_snake,
)
from super_metroid.routes.kpdr.ice.geometry import (
    ACID_TO_SNAKE_RLE,
    ICE_LEAVE_DOOR_X,
    ICE_LEAVE_FRAMES,
    SNAKE_L4_Y,
    SNAKE_TOP_Y,
    SNAKE_TUTORIAL_DOOR_X,
    has_ice,
    in_ice_super_band,
    on_acid_floor,
    on_snake_floor,
    on_snake_top,
)
from super_metroid.routes.kpdr.rooms import (
    ROOM_BUSINESS,
    ROOM_ICE_ACID,
    ROOM_ICE_GATE,
    ROOM_ICE_SNAKE,
)


def test_business_to_ice_gate_export_and_registry() -> None:
    assert get_segment("business_to_ice_gate") is play_business_to_ice_gate
    assert get_segment("ice_gate_to_acid") is play_ice_gate_to_acid
    assert get_segment("ice_acid_to_snake") is play_ice_acid_to_snake
    assert get_segment("ice_snake_to_ice") is play_ice_snake_to_ice
    assert get_segment("ice_to_snake") is play_ice_to_snake
    assert get_segment("ice_snake_to_tutorial") is play_ice_snake_to_tutorial
    assert ICE_LEAVE_DOOR_X == 40
    assert ICE_LEAVE_FRAMES >= 200
    assert SNAKE_TUTORIAL_DOOR_X >= 200


def test_acid_to_snake_rle_loaded() -> None:
    assert len(ACID_TO_SNAKE_RLE) >= 10
    total = sum(n for n, _ in ACID_TO_SNAKE_RLE)
    assert total >= 500
    # Door open + enter tail present.
    assert any("X" in btns for _, btns in ACID_TO_SNAKE_RLE)


def test_on_acid_floor_predicate() -> None:
    floor = SimpleNamespace(
        room_id=ROOM_ICE_ACID,
        samus_x=470,
        samus_y=139,
        velocity_y=0,
        pose=2,
    )
    assert on_acid_floor(floor) is True  # type: ignore[arg-type]

    low = SimpleNamespace(
        room_id=ROOM_ICE_ACID,
        samus_x=470,
        samus_y=200,
        velocity_y=0,
        pose=2,
    )
    assert on_acid_floor(low) is False  # type: ignore[arg-type]


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


def test_snake_climb_bands_and_ice_mask() -> None:
    assert ICE_BEAM_MASK == 0x0002
    # y increases downward: mid door (L4) is below top shelf.
    assert SNAKE_L4_Y[0] > SNAKE_TOP_Y[1]
    floor = SimpleNamespace(
        room_id=ROOM_ICE_SNAKE,
        samus_x=216,
        samus_y=651,
        velocity_y=0,
        pose=2,
    )
    assert on_snake_floor(floor) is True  # type: ignore[arg-type]
    top = SimpleNamespace(
        room_id=ROOM_ICE_SNAKE,
        samus_x=120,
        samus_y=139,
        velocity_y=0,
        pose=9,
    )
    assert on_snake_top(top) is True  # type: ignore[arg-type]
    no_ice = SimpleNamespace(collected_beams=0x1005)
    assert has_ice(no_ice) is False  # type: ignore[arg-type]
    yes_ice = SimpleNamespace(collected_beams=0x1007)
    assert has_ice(yes_ice) is True  # type: ignore[arg-type]


def test_snake_tunnel_excludes_false_ledge() -> None:
    """Tunnel floor y~377 is not the morph trap ledge y~409."""
    from super_metroid.routes.kpdr.ice.geometry import (
        SNAKE_TUNNEL_Y,
        on_snake_false_ledge,
        on_snake_tunnel_band,
    )

    assert SNAKE_TUNNEL_Y[1] < 400
    tunnel = SimpleNamespace(
        room_id=ROOM_ICE_SNAKE,
        samus_x=204,
        samus_y=377,
        velocity_y=0,
        pose=30,
    )
    assert on_snake_tunnel_band(tunnel) is True  # type: ignore[arg-type]
    false = SimpleNamespace(
        room_id=ROOM_ICE_SNAKE,
        samus_x=219,
        samus_y=409,
        velocity_y=0,
        pose=30,
    )
    assert on_snake_tunnel_band(false) is False  # type: ignore[arg-type]
    assert on_snake_false_ledge(false) is True  # type: ignore[arg-type]
