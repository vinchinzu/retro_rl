"""Post-Ice compose traps: fight is in-room; attic is not on the Phantoon tip."""

import super_metroid.routes.continuous as _continuous  # noqa: F401

from super_metroid.routes.kpdr.hops import ALPHA_PB_ONLY_HOPS, PHANTOON_ONLY_HOPS
from super_metroid.routes.kpdr.post_ice_spine import POST_ICE_SPINE
from super_metroid.routes.kpdr.rooms import ROOM_PHANTOON, ROOM_WS_BASEMENT
from super_metroid.routes.kpdr.spine import validate_spine


def test_post_ice_spine_fight_and_attic_are_not_door_hops() -> None:
    validate_spine()
    hop_ids = [h.hop_id for h in POST_ICE_SPINE]
    assert len(hop_ids) == len(set(hop_ids))

    replay = next(h for h in ALPHA_PB_ONLY_HOPS if h.hop_id == "ice_business_to_warehouse")
    assert replay.emits_door_edge is False

    fight = next(h for h in PHANTOON_ONLY_HOPS if h.hop_id == "phantoon_fight")
    assert fight.from_room == ROOM_PHANTOON
    assert fight.to_room == ROOM_PHANTOON
    assert fight.emits_door_edge is False

    leave = PHANTOON_ONLY_HOPS[-1]
    assert leave.hop_id == "phantoon_loot_exit"
    assert leave.to_room == ROOM_WS_BASEMENT
    assert leave.emits_door_edge is True
    assert leave.exit_direction == "left"

    assert "ws_main_to_attic" not in hop_ids
    assert "ws_main_to_attic" not in {h.hop_id for h in PHANTOON_ONLY_HOPS}
