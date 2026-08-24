"""Unit locks for K5 Hellway → Caterpillar wiring."""

from __future__ import annotations

from super_metroid.paths import GAME_DIR
from super_metroid.routes.kpdr.k5 import play_hellway_to_caterpillar
from super_metroid.routes.kpdr.k5.hellway_to_caterpillar import (
    ROOM_CATERPILLAR,
    ROOM_HELLWAY,
    _in_right_door_band,
)


def test_hellway_to_caterpillar_exports() -> None:
    assert ROOM_HELLWAY == 0xA2F7
    assert ROOM_CATERPILLAR == 0xA322
    assert callable(play_hellway_to_caterpillar)


def test_hellway_to_caterpillar_is_zero_settle_in_kpdr_probe() -> None:
    """Ice-climb leave is airborne p11; 5f idle eats a Samus Eater."""
    body = (GAME_DIR / "scripts" / "probe" / "kpdr.py").read_text(encoding="utf-8")
    marker = 'zero_settle_segments = {'
    start = body.index(marker)
    end = body.index('}', start)
    assert "hellway-to-caterpillar" in body[start:end]


def test_hellway_right_door_band_rejects_door_slot_underflow() -> None:
    """x=65522 is the Red Tower slot wrap, not Caterpillar's door."""
    assert _in_right_door_band(700)
    assert not _in_right_door_band(39)
    assert not _in_right_door_band(237)
    assert not _in_right_door_band(690)
    assert not _in_right_door_band(65522)
