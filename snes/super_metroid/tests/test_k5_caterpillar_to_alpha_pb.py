"""Unit locks for K5 Caterpillar → Alpha PB wiring."""

from super_metroid.routes.kpdr.k5 import play_caterpillar_to_alpha_pb
from super_metroid.routes.kpdr.k5.caterpillar_to_alpha_pb import (
    ROOM_ALPHA_PB,
    ROOM_CATERPILLAR,
    _entry_shelf_dir,
)


def test_caterpillar_to_alpha_pb_exports() -> None:
    assert ROOM_CATERPILLAR == 0xA322
    assert ROOM_ALPHA_PB == 0xA3AE
    assert callable(play_caterpillar_to_alpha_pb)


def test_caterpillar_to_alpha_pb_is_registered() -> None:
    from super_metroid.routes.kpdr.registry import KPDR_SEGMENTS

    assert KPDR_SEGMENTS["caterpillar_to_alpha_pb"] is play_caterpillar_to_alpha_pb


def test_entry_shelf_recenters_off_the_right_ledge() -> None:
    """Compose Cacatac knockback lands ~(155,1389); walk back, do not mash A."""
    assert _entry_shelf_dir(39) == "RIGHT"
    assert _entry_shelf_dir(77) == "RIGHT"
    assert _entry_shelf_dir(90) is None
    assert _entry_shelf_dir(100) is None
    assert _entry_shelf_dir(101) == "LEFT"
    assert _entry_shelf_dir(110) == "LEFT"
    assert _entry_shelf_dir(155) == "LEFT"
