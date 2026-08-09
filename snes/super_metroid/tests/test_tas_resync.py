"""Unit tests for tas.resync pure scoring / hit-map helpers (no emulator)."""

from __future__ import annotations

from super_metroid.routes.kpdr.room_ids import (
    ROOM_CLIMB,
    ROOM_LANDING_SITE,
    ROOM_MORPH,
    ROOM_PARLOR,
    ROOM_PIT,
)
from super_metroid.tas.resync import (
    AlignTrial,
    empty_hits,
    score_zebes_progress,
)


def test_empty_hits_keys() -> None:
    h = empty_hits()
    assert set(h) == {"parlor", "climb", "pit", "elev", "morph"}
    assert all(v is None for v in h.values())


def test_align_trial_to_dict_keeps_flat_hit_fields() -> None:
    hits = empty_hits()
    hits["parlor"] = 100
    hits["climb"] = 200
    tr = AlignTrial(
        movie_start=15000,
        pad=0,
        score=99.0,
        unique_rooms=3,
        room_order=["0x91F8", "0x92FD", "0x96BA"],
        hits=hits,
    )
    d = tr.to_dict()
    assert d["hit_parlor"] == 100
    assert d["hit_climb"] == 200
    assert d["hit_pit"] is None
    assert d["hits"]["climb"] == 200
    assert tr.hit_climb == 200
    assert tr.hit_morph is None


def test_score_climb_beats_parlor_bounce() -> None:
    """Landing↔Parlor thrash must score below a Climb hit."""
    parlor_only = score_zebes_progress(
        [ROOM_LANDING_SITE, ROOM_PARLOR],
        {"parlor": 100, "climb": None, "pit": None, "elev": None, "morph": None},
        deaths=0,
    )
    climb_hit = score_zebes_progress(
        [ROOM_LANDING_SITE, ROOM_PARLOR, ROOM_CLIMB],
        {"parlor": 100, "climb": 200, "pit": None, "elev": None, "morph": None},
        deaths=0,
    )
    assert climb_hit > parlor_only
    # Climb should clear a large margin over thrash class.
    assert climb_hit - parlor_only >= 40


def test_score_landing_parlor_thrash_penalized() -> None:
    landing_only = score_zebes_progress(
        [ROOM_LANDING_SITE],
        empty_hits(),
        deaths=0,
    )
    parlor_bounce = score_zebes_progress(
        [ROOM_LANDING_SITE, ROOM_PARLOR],
        {"parlor": 50, "climb": None, "pit": None, "elev": None, "morph": None},
        deaths=0,
    )
    # Parlor still beats pure Landing, but is thrash-penalized.
    assert parlor_bounce > landing_only
    # Without penalty, parlor would be ~2+15+2+5 = 24; with -12-8 ≈ 4.
    # Sanity: thrash score stays well under Climb baseline (~3+15+55+2+5+15).
    assert parlor_bounce < 40


def test_score_goal_bonus_climb() -> None:
    base = score_zebes_progress(
        [ROOM_LANDING_SITE, ROOM_PARLOR, ROOM_CLIMB],
        {"parlor": 1, "climb": 2, "pit": None, "elev": None, "morph": None},
        deaths=0,
    )
    with_goal = score_zebes_progress(
        [ROOM_LANDING_SITE, ROOM_PARLOR, ROOM_CLIMB],
        {"parlor": 1, "climb": 2, "pit": None, "elev": None, "morph": None},
        deaths=0,
        goal="climb",
    )
    assert with_goal >= base + 100


def test_score_legacy_kwargs() -> None:
    s = score_zebes_progress(
        [ROOM_LANDING_SITE, ROOM_PARLOR],
        hit_parlor=10,
        hit_climb=None,
        hit_morph=None,
        deaths=0,
    )
    assert s == score_zebes_progress(
        [ROOM_LANDING_SITE, ROOM_PARLOR],
        {"parlor": 10, "climb": None, "pit": None, "elev": None, "morph": None},
        deaths=0,
    )


def test_score_pit_and_morph_outrank_climb() -> None:
    climb = score_zebes_progress(
        [ROOM_LANDING_SITE, ROOM_PARLOR, ROOM_CLIMB],
        {"parlor": 1, "climb": 2, "pit": None, "elev": None, "morph": None},
    )
    pit = score_zebes_progress(
        [ROOM_LANDING_SITE, ROOM_PARLOR, ROOM_CLIMB, ROOM_PIT],
        {"parlor": 1, "climb": 2, "pit": 3, "elev": None, "morph": None},
        pit_max_x=400,
    )
    morph = score_zebes_progress(
        [
            ROOM_LANDING_SITE,
            ROOM_PARLOR,
            ROOM_CLIMB,
            ROOM_PIT,
            ROOM_MORPH,
        ],
        {"parlor": 1, "climb": 2, "pit": 3, "elev": 4, "morph": 5},
    )
    assert pit > climb
    assert morph > pit


def test_score_deaths_penalize() -> None:
    alive = score_zebes_progress(
        [ROOM_LANDING_SITE, ROOM_PARLOR],
        {"parlor": 1, "climb": None, "pit": None, "elev": None, "morph": None},
        deaths=0,
    )
    dead = score_zebes_progress(
        [ROOM_LANDING_SITE, ROOM_PARLOR],
        {"parlor": 1, "climb": None, "pit": None, "elev": None, "morph": None},
        deaths=2,
    )
    assert alive - dead == 16
