"""Unit locks for the development-only Moat controller scaffold."""

from __future__ import annotations

from super_metroid.routes.kpdr.moat import ROOM_MOAT, play_moat_cross


def test_moat_scaffold_exports() -> None:
    assert ROOM_MOAT == 0x95FF
    assert callable(play_moat_cross)
