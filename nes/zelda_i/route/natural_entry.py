"""Mechanical catalog of Zelda I natural-entry vs isolated/assisted greens.

Not a live emulator promotion and not a STATUS rewrite. ``status_eligible``
is True only for already-verified M5 Level 1 Clean segments.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SegmentEntry:
    segment_id: str
    isolated_clean: bool
    assisted_green: bool
    natural_entry: bool
    blocker: str  # e.g. "heart_starvation_0x5c", "mid_run_state_load", ""
    predecessor: str
    status_eligible: bool  # always False unless already M5 L1


# Seeded from STATUS / LEVELN_ROUTE. Closing this catalog is not a promote.
SEGMENTS: tuple[SegmentEntry, ...] = (
    SegmentEntry(
        segment_id="l1_complete",
        isolated_clean=True,
        assisted_green=True,
        natural_entry=True,
        blocker="",
        predecessor="power_on",
        status_eligible=True,
    ),
    SegmentEntry(
        segment_id="l2_path_prefix_0x4a",
        isolated_clean=True,
        assisted_green=False,
        natural_entry=True,
        blocker="",
        predecessor="l1_exit_overworld",
        status_eligible=False,
    ),
    SegmentEntry(
        segment_id="l2_door_path_0x3c",
        isolated_clean=True,
        assisted_green=True,
        natural_entry=False,
        blocker="heart_starvation_0x5c",
        predecessor="l2_path_prefix_0x4a",
        status_eligible=False,
    ),
    SegmentEntry(
        segment_id="l2_room_0x6d",
        isolated_clean=True,
        assisted_green=False,
        natural_entry=False,
        blocker="mid_run_state_load",
        predecessor="Level2Entrance",
        status_eligible=False,
    ),
    SegmentEntry(
        segment_id="l2_room_0x6c",
        isolated_clean=True,
        assisted_green=False,
        natural_entry=False,
        blocker="mid_run_state_load",
        predecessor="Level2Entrance",
        status_eligible=False,
    ),
    SegmentEntry(
        segment_id="l2_room_0x7e",
        isolated_clean=True,
        assisted_green=False,
        natural_entry=False,
        blocker="mid_run_state_load",
        predecessor="Level2Entrance",
        status_eligible=False,
    ),
    SegmentEntry(
        segment_id="l2_room_0x6f",
        isolated_clean=True,
        assisted_green=False,
        natural_entry=False,
        blocker="mid_run_state_load",
        predecessor="Level2EastKey",
        status_eligible=False,
    ),
    SegmentEntry(
        segment_id="l2_complete_tf",
        isolated_clean=False,
        assisted_green=True,
        natural_entry=False,
        blocker="mid_run_state_load",
        predecessor="Level2Entrance",
        status_eligible=False,
    ),
    SegmentEntry(
        segment_id="l3_entry_from_l2",
        isolated_clean=False,
        assisted_green=True,
        natural_entry=False,
        blocker="mid_run_state_load",
        predecessor="Level2ExitOverworld",
        status_eligible=False,
    ),
    SegmentEntry(
        segment_id="l4_entry_from_l3",
        isolated_clean=False,
        assisted_green=True,
        natural_entry=False,
        blocker="mid_run_state_load",
        predecessor="Level3ExitOverworld",
        status_eligible=False,
    ),
    SegmentEntry(
        segment_id="l5_entry_from_l4",
        isolated_clean=False,
        assisted_green=True,
        natural_entry=False,
        blocker="mid_run_state_load",
        predecessor="Level4Complete",
        status_eligible=False,
    ),
    SegmentEntry(
        segment_id="l5_room_0x66",
        isolated_clean=True,
        assisted_green=True,
        natural_entry=False,
        blocker="mid_run_state_load",
        predecessor="Level5EntranceFromL4",
        status_eligible=False,
    ),
    SegmentEntry(
        segment_id="l5_east_key_0x77",
        isolated_clean=False,
        assisted_green=True,
        natural_entry=False,
        blocker="mid_run_state_load",
        predecessor="Level5Cleared66",
        status_eligible=False,
    ),
    SegmentEntry(
        segment_id="l5_east_to_whistle_0x04",
        isolated_clean=False,
        assisted_green=True,
        natural_entry=False,
        blocker="mid_run_state_load",
        predecessor="Level5EastKey",
        status_eligible=False,
    ),
    SegmentEntry(
        segment_id="l5_whistle_0x04_to_tf",
        isolated_clean=False,
        assisted_green=True,
        natural_entry=False,
        blocker="not_composed_onto_east_key",
        predecessor="Level5WhistleFrom77",
        status_eligible=False,
    ),
    SegmentEntry(
        segment_id="l9_room_0x51_to_0x41",
        isolated_clean=False,
        assisted_green=False,
        natural_entry=False,
        blocker="dest_no_statue_diamond",
        predecessor="Level9Room51ReconFixture",
        status_eligible=False,
    ),
    SegmentEntry(
        segment_id="l9_room_0x41_suffix",
        isolated_clean=False,
        assisted_green=True,
        natural_entry=False,
        blocker="fixture_only",
        predecessor="l9_room_0x51_to_0x41",
        status_eligible=False,
    ),
)

# L1 Clean M5 family — only these may return True from status_claim_allowed.
_L1_CLEAN_IDS = frozenset({"l1_complete"})


def _index() -> dict[str, SegmentEntry]:
    return {entry.segment_id: entry for entry in SEGMENTS}


def get_segment(segment_id: str) -> SegmentEntry | None:
    return _index().get(segment_id)


def missing_natural_entry() -> tuple[SegmentEntry, ...]:
    """Assisted-green or isolated-clean segments still lacking real-predecessor entry."""
    return tuple(
        entry
        for entry in SEGMENTS
        if not entry.natural_entry and (entry.assisted_green or entry.isolated_clean)
    )


def status_claim_allowed(segment_id: str) -> bool:
    """False for everything except already-verified L1 Clean segments."""
    if segment_id not in _L1_CLEAN_IDS:
        return False
    entry = get_segment(segment_id)
    return bool(entry and entry.status_eligible and entry.natural_entry)


# Optional sibling; catalog stays independent if it is absent.
try:  # pragma: no cover - import probe only
    import zelda_i.route.eligible as _route_eligible  # type: ignore[import-not-found]
except ImportError:
    _route_eligible = None


__all__ = [
    "SEGMENTS",
    "SegmentEntry",
    "get_segment",
    "missing_natural_entry",
    "status_claim_allowed",
]
