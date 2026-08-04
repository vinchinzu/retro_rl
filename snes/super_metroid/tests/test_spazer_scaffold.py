"""Unit locks for the early Spazer Beam detour scaffold."""

from __future__ import annotations

from super_metroid.routes.kpdr import spazer


def test_room_constant() -> None:
    assert spazer.ROOM_SPAZER == 0xA447


def test_scaffold_helpers_are_importable() -> None:
    assert callable(spazer.play_below_spazer_to_spazer)
    assert callable(spazer.play_spazer_collect)
    assert callable(spazer.play_spazer_return_to_below)


def test_scaffold_does_not_claim_geometry() -> None:
    assert "scaffold" in spazer.play_below_spazer_to_spazer.__doc__.lower()
    assert "scaffold" in spazer.play_spazer_collect.__doc__.lower()
    assert "scaffold" in spazer.play_spazer_return_to_below.__doc__.lower()


def test_scaffold_exports_are_in___all__() -> None:
    assert "ROOM_SPAZER" in spazer.__all__
    assert "play_below_spazer_to_spazer" in spazer.__all__
    assert "play_spazer_collect" in spazer.__all__
    assert "play_spazer_return_to_below" in spazer.__all__
