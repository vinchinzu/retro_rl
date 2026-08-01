"""Unit locks for the optional Charge Beam return scaffold."""

from __future__ import annotations

from super_metroid.routes.kpdr import charge_return


def test_charge_room_is_big_pink() -> None:
    assert charge_return.ROOM_CHARGE == 0x9D19
    assert charge_return.ROOM_BIG_PINK == charge_return.ROOM_CHARGE


def test_charge_scaffold_helpers_are_importable() -> None:
    assert callable(charge_return.play_charge_beam_collect)
    assert callable(charge_return.play_charge_beam_return)


def test_charge_scaffold_does_not_claim_geometry() -> None:
    assert "scaffold" in charge_return.play_charge_beam_collect.__doc__.lower()
    assert "scaffold" in charge_return.play_charge_beam_return.__doc__.lower()
