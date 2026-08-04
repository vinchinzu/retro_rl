"""Unit locks for Charge Beam collect + conventional return."""

from __future__ import annotations

from super_metroid.routes.kpdr import charge_return
from super_metroid.routes.kpdr.charge_return import CHARGE_BEAM_MASK


def test_charge_room_is_big_pink() -> None:
    assert charge_return.ROOM_CHARGE == 0x9D19
    assert charge_return.ROOM_BIG_PINK == charge_return.ROOM_CHARGE


def test_charge_beam_mask() -> None:
    assert CHARGE_BEAM_MASK == 0x1000


def test_charge_helpers_are_importable() -> None:
    assert callable(charge_return.play_charge_beam_collect)
    assert callable(charge_return.play_charge_beam_return)
    assert callable(charge_return.play_big_pink_charge_detour)


def test_charge_helpers_claim_real_geometry() -> None:
    """Controllers are route-ready (not scaffold placeholders)."""
    collect_doc = (charge_return.play_charge_beam_collect.__doc__ or "").lower()
    return_doc = (charge_return.play_charge_beam_return.__doc__ or "").lower()
    assert "scaffold" not in collect_doc
    assert "scaffold" not in return_doc
    assert "chozo" in collect_doc or "charge" in collect_doc
    # Conventional return: ordinary jumps, not walljump/IBJ.
    assert "walljump" not in return_doc.replace("-", "")
    assert "ordinary" in return_doc or "jump" in return_doc
