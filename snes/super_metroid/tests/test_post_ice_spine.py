"""Unit locks for post-Ice Alpha PB / Moat continuous spine wiring."""

import super_metroid.routes.continuous as _continuous  # noqa: F401

from super_metroid.hop_glance import PHANTOON_LEAVE, WS_ENTRANCE_TO_MAIN
from super_metroid.routes.kpdr.hops import (
    ALPHA_PB_ONLY_HOPS,
    MOAT_ONLY_HOPS,
    PHANTOON_ONLY_HOPS,
    WS_ONLY_HOPS,
)
from super_metroid.routes.kpdr.post_ice_spine import POST_ICE_SPINE
from super_metroid.routes.kpdr.rooms import (
    ROOM_ALPHA_PB,
    ROOM_ICE,
    ROOM_MOAT,
    ROOM_PHANTOON,
    ROOM_WEST_OCEAN,
    ROOM_WS_BASEMENT,
    ROOM_WS_ENTRANCE,
)
from super_metroid.routes.kpdr.spine import hops_for_tip, validate_spine


def test_post_ice_spine_covers_alpha_pb_and_moat() -> None:
    validate_spine()
    assert POST_ICE_SPINE[0].from_room == ROOM_ICE
    assert POST_ICE_SPINE[-1].to_room == ROOM_WS_BASEMENT
    assert hops_for_tip("alpha_pb") == ALPHA_PB_ONLY_HOPS
    assert hops_for_tip("moat") == MOAT_ONLY_HOPS
    assert hops_for_tip("ws") == WS_ONLY_HOPS
    assert hops_for_tip("phantoon") == PHANTOON_ONLY_HOPS
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
    assert WS_ONLY_HOPS[-1].to_room == ROOM_WS_ENTRANCE
    assert [h.hop_id for h in PHANTOON_ONLY_HOPS] == [
        "ws_entrance_to_main",
        "ws_main_to_basement",
        "ws_basement_to_phantoon",
        "phantoon_fight",
        "phantoon_loot_exit",
    ]
    fight = next(h for h in PHANTOON_ONLY_HOPS if h.hop_id == "phantoon_fight")
    assert fight.from_room == ROOM_PHANTOON
    assert fight.to_room == ROOM_PHANTOON
    assert fight.emits_door_edge is False
    assert fight.use_transition_split is False
    leave = PHANTOON_ONLY_HOPS[-1]
    assert leave.hop_id == "phantoon_loot_exit"
    assert leave.from_room == ROOM_PHANTOON
    assert leave.to_room == ROOM_WS_BASEMENT
    assert leave.emits_door_edge is True
    assert leave.exit_direction == "left"
    assert leave.entry_direction == "right"
    assert leave.leave is PHANTOON_LEAVE
    entrance = next(h for h in PHANTOON_ONLY_HOPS if h.hop_id == "ws_entrance_to_main")
    assert entrance.leave is WS_ENTRANCE_TO_MAIN
    assert POST_ICE_SPINE[0].leave is None
    assert all(h.hop_id != "ws_basement_to_main" for h in POST_ICE_SPINE)
