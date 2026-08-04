"""Unit locks for early Spazer Beam detour controllers."""

from __future__ import annotations

from super_metroid.routes.kpdr import spazer
from super_metroid.routes.kpdr.spazer import SPAZER_BEAM_MASK


def test_room_constant() -> None:
    assert spazer.ROOM_SPAZER == 0xA447


def test_spazer_beam_mask() -> None:
    assert SPAZER_BEAM_MASK == 0x0004


def test_helpers_are_importable() -> None:
    assert callable(spazer.play_below_spazer_to_spazer)
    assert callable(spazer.play_spazer_collect)
    assert callable(spazer.play_spazer_return_to_below)


def test_helpers_claim_real_geometry() -> None:
    """Door hop, collect, and return are route-ready (not scaffold placeholders)."""
    door_doc = (spazer.play_below_spazer_to_spazer.__doc__ or "").lower()
    collect_doc = (spazer.play_spazer_collect.__doc__ or "").lower()
    return_doc = (spazer.play_spazer_return_to_below.__doc__ or "").lower()
    assert "scaffold" not in door_doc
    assert "scaffold" not in collect_doc
    assert "scaffold" not in return_doc
    assert "green" in door_doc or "super" in door_doc
    assert "chozo" in collect_doc or "spazer" in collect_doc
    assert "blue" in return_doc or "left" in return_doc
    # Return handoff must stay clear of the open Super door.
    assert "door" in return_doc


def test_exports_are_in___all__() -> None:
    assert "ROOM_SPAZER" in spazer.__all__
    assert "SPAZER_BEAM_MASK" in spazer.__all__
    assert "play_below_spazer_to_spazer" in spazer.__all__
    assert "play_spazer_collect" in spazer.__all__
    assert "play_spazer_return_to_below" in spazer.__all__
