"""Unit locks for the K6 Alpha PB escape controller."""

from super_metroid.routes.kpdr.k6 import (
    play_alpha_pb_to_caterpillar,
    play_caterpillar_to_elevator,
    play_elevator_to_kihunter,
    play_kihunter_to_moat,
)
from super_metroid.routes.kpdr.k6.alpha_pb_escape import ROOM_ALPHA_PB, ROOM_CATERPILLAR
from super_metroid.routes.kpdr.k6.caterpillar_climb import ROOM_ELEVATOR
from super_metroid.routes.kpdr.k6.elevator_to_kihunter import ROOM_KIHUNTER
from super_metroid.routes.kpdr.k6.kihunter_to_moat import ROOM_MOAT


def test_alpha_pb_escape_exports() -> None:
    assert ROOM_ALPHA_PB == 0xA3AE
    assert ROOM_CATERPILLAR == 0xA322
    assert callable(play_alpha_pb_to_caterpillar)


def test_caterpillar_climb_exports() -> None:
    assert ROOM_ELEVATOR == 0x962A
    assert callable(play_caterpillar_to_elevator)


def test_elevator_to_kihunter_exports() -> None:
    assert ROOM_KIHUNTER == 0x948C
    assert callable(play_elevator_to_kihunter)


def test_kihunter_to_moat_exports() -> None:
    assert ROOM_MOAT == 0x95FF
    assert callable(play_kihunter_to_moat)


def test_moat_compose_chain_starts_at_alpha_pb() -> None:
    from super_metroid.routes.kpdr.hops import MOAT_ONLY_HOPS
    from super_metroid.routes.kpdr.rooms import ROOM_ALPHA_PB, ROOM_WEST_OCEAN

    assert MOAT_ONLY_HOPS[0].from_room == ROOM_ALPHA_PB
    assert MOAT_ONLY_HOPS[-1].to_room == ROOM_WEST_OCEAN


def test_k6_controllers_are_registered() -> None:
    from super_metroid.routes.kpdr.registry import KPDR_SEGMENTS

    assert KPDR_SEGMENTS["alpha_pb_to_caterpillar"] is play_alpha_pb_to_caterpillar
    assert KPDR_SEGMENTS["caterpillar_to_elevator"] is play_caterpillar_to_elevator
    assert KPDR_SEGMENTS["elevator_to_kihunter"] is play_elevator_to_kihunter
    assert KPDR_SEGMENTS["kihunter_to_moat"] is play_kihunter_to_moat
    from super_metroid.routes.kpdr.moat import play_moat_cross
    from super_metroid.routes.kpdr.west_ocean import play_west_ocean_over_ocean_spark

    assert KPDR_SEGMENTS["moat_cross"] is play_moat_cross
    assert KPDR_SEGMENTS["west_ocean_to_ws"] is play_west_ocean_over_ocean_spark
    from super_metroid.routes.kpdr.k6 import (
        play_phantoon_loot_exit,
        play_phantoon_room_fight,
        play_ws_basement_to_phantoon,
        play_ws_entrance_to_main,
        play_ws_main_to_basement,
    )

    assert KPDR_SEGMENTS["ws_entrance_to_main"] is play_ws_entrance_to_main
    assert KPDR_SEGMENTS["ws_main_to_basement"] is play_ws_main_to_basement
    assert KPDR_SEGMENTS["ws_basement_to_phantoon"] is play_ws_basement_to_phantoon
    assert KPDR_SEGMENTS["phantoon_fight"] is play_phantoon_room_fight
    assert KPDR_SEGMENTS["phantoon_loot_exit"] is play_phantoon_loot_exit
