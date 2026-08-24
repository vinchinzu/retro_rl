"""Unit locks for post-Ice Alpha PB / Moat continuous spine wiring."""

import super_metroid.routes.continuous as _continuous  # noqa: F401

from super_metroid.routes.kpdr.hops import (
    ALPHA_PB_ONLY_HOPS,
    MOAT_ONLY_HOPS,
    WS_ONLY_HOPS,
)
from super_metroid.routes.kpdr.post_ice_spine import POST_ICE_SPINE
from super_metroid.routes.kpdr.rooms import (
    ROOM_ALPHA_PB,
    ROOM_ICE,
    ROOM_MOAT,
    ROOM_WEST_OCEAN,
    ROOM_WS_ENTRANCE,
)
from super_metroid.routes.kpdr.spine import hops_for_tip, validate_spine


def test_post_ice_spine_covers_alpha_pb_and_moat() -> None:
    validate_spine()
    assert POST_ICE_SPINE[0].from_room == ROOM_ICE
    assert POST_ICE_SPINE[-1].to_room == ROOM_WS_ENTRANCE
    assert hops_for_tip("alpha_pb") == ALPHA_PB_ONLY_HOPS
    assert hops_for_tip("moat") == MOAT_ONLY_HOPS
    assert hops_for_tip("ws") == WS_ONLY_HOPS
    assert ALPHA_PB_ONLY_HOPS[-1].to_room == ROOM_ALPHA_PB
    assert MOAT_ONLY_HOPS[-1].from_room == ROOM_MOAT
    assert MOAT_ONLY_HOPS[-1].to_room == ROOM_WEST_OCEAN
    assert WS_ONLY_HOPS[-1].from_room == ROOM_WEST_OCEAN
    assert WS_ONLY_HOPS[-1].to_room == ROOM_WS_ENTRANCE
    # Replay hop must not emit a second Business→Warehouse DoorEdge.
    replay = next(h for h in ALPHA_PB_ONLY_HOPS if h.hop_id == "ice_business_to_warehouse")
    assert replay.emits_door_edge is False
    hop_ids = [h.hop_id for h in POST_ICE_SPINE]
    assert len(hop_ids) == len(set(hop_ids))
    assert MOAT_ONLY_HOPS[0].from_room == ROOM_ALPHA_PB
    assert [h.hop_id for h in MOAT_ONLY_HOPS] == [
        "alpha_pb_to_caterpillar",
        "caterpillar_to_elevator",
        "elevator_to_kihunter",
        "kihunter_to_moat",
        "moat_cross",
    ]
    assert [h.hop_id for h in WS_ONLY_HOPS] == ["west_ocean_to_ws"]
