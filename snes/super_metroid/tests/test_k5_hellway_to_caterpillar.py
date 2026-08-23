"""Unit locks for K5 Hellway → Caterpillar wiring."""

from __future__ import annotations

from super_metroid.routes.kpdr.k5 import play_hellway_to_caterpillar
from super_metroid.routes.kpdr.k5.hellway_to_caterpillar import (
    ROOM_CATERPILLAR,
    ROOM_HELLWAY,
)


def test_hellway_to_caterpillar_exports() -> None:
    assert ROOM_HELLWAY == 0xA2F7
    assert ROOM_CATERPILLAR == 0xA322
    assert callable(play_hellway_to_caterpillar)
