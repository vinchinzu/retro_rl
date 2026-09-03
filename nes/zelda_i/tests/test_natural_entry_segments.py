"""Schema + invariant guard for the ``route/natural_entry.py`` SEGMENTS table.

Covers the three recon-scoped rows added in checkpoint 847bce83
(``l8_room_0x7e``, ``l9_rock_hops_0x77_to_0x27``,
``l9_rock_hops_0x27_to_spectacle_rock``): they must parse, be fully
populated, and never claim a natural-entry / status grant.
"""

from __future__ import annotations

from zelda_i.route.natural_entry import (
    SEGMENTS,
    SegmentEntry,
    get_segment,
    missing_natural_entry,
    status_claim_allowed,
)

_CHECKPOINT_847_ROWS = (
    "l8_room_0x7e",
    "l9_rock_hops_0x77_to_0x27",
    "l9_rock_hops_0x27_to_spectacle_rock",
)


def test_every_segment_row_is_fully_populated() -> None:
    for entry in SEGMENTS:
        assert isinstance(entry, SegmentEntry)
        assert isinstance(entry.segment_id, str) and entry.segment_id
        assert isinstance(entry.isolated_clean, bool)
        assert isinstance(entry.assisted_green, bool)
        assert isinstance(entry.natural_entry, bool)
        assert isinstance(entry.blocker, str)
        assert isinstance(entry.predecessor, str) and entry.predecessor
        assert isinstance(entry.status_eligible, bool)


def test_segment_ids_are_unique() -> None:
    ids = [entry.segment_id for entry in SEGMENTS]
    assert len(ids) == len(set(ids))


def test_status_claim_allowed_only_for_l1_complete() -> None:
    allowed = {e.segment_id for e in SEGMENTS if status_claim_allowed(e.segment_id)}
    assert allowed == {"l1_complete"}


def test_checkpoint_847_recon_rows_present_and_scoped() -> None:
    for segment_id in _CHECKPOINT_847_ROWS:
        entry = get_segment(segment_id)
        assert entry is not None, segment_id
        # Recon-fixture rows: never a natural-entry or status promotion.
        assert entry.natural_entry is False
        assert entry.status_eligible is False
        assert status_claim_allowed(segment_id) is False
        assert entry.blocker == "mid_run_state_load"

    assert get_segment("l8_room_0x7e").predecessor == "Level8BushWithCandleFixture"
    assert get_segment("l8_room_0x7e").isolated_clean is True
    assert (
        get_segment("l9_rock_hops_0x77_to_0x27").predecessor
        == "Level9OverworldReconFixture"
    )
    # "Partial" recon: did not reach the interior, so not isolated-clean.
    assert get_segment("l9_rock_hops_0x27_to_spectacle_rock").isolated_clean is False


def test_missing_natural_entry_lists_isolated_or_assisted_rows_only() -> None:
    missing = missing_natural_entry()
    for entry in missing:
        assert entry.natural_entry is False
        assert entry.isolated_clean or entry.assisted_green
    # The two clean recon rows surface here; the "Partial" one does not.
    ids = {e.segment_id for e in missing}
    assert "l8_room_0x7e" in ids
    assert "l9_rock_hops_0x77_to_0x27" in ids
    assert "l9_rock_hops_0x27_to_spectacle_rock" not in ids
