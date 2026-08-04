"""Unit locks for the Alpha Power Bomb room scaffold."""

from __future__ import annotations

from super_metroid.routes.kpdr.alpha_pb import ROOM_ALPHA_PB, play_alpha_pb_collect


def test_alpha_pb_scaffold_exports() -> None:
    assert ROOM_ALPHA_PB == 0xA3AE
    assert callable(play_alpha_pb_collect)
