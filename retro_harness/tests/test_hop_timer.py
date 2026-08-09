"""Unit tests for the generic hop timer engine."""

from __future__ import annotations

from dataclasses import dataclass

from retro_harness.hop_timer import HopFrame, HopTimer, OpenHop


@dataclass(frozen=True)
class _Visit:
    loc: int
    dest: int
    entry: int
    leave: int
    exit: int
    seq: int


@dataclass(frozen=True)
class _Disc:
    frame: int
    reason: str
    loc: int
    detail: str


def _make_visit(
    open_hop: OpenHop[int],
    dest: int,
    leave_frame: int,
    exit_frame: int,
    sequence_index: int,
    meta,
    dest_context,
) -> _Visit:
    return _Visit(
        loc=open_hop.location,
        dest=dest,
        entry=open_hop.entry_frame,
        leave=leave_frame,
        exit=exit_frame,
        seq=sequence_index,
    )


def _make_disc(frame: int, reason: str, loc: int, detail: str) -> _Disc:
    return _Disc(frame=frame, reason=reason, loc=loc, detail=detail)


def _timer() -> HopTimer[int, _Visit, _Disc]:
    return HopTimer(
        make_visit=_make_visit,
        make_discontinuity=_make_disc,
        null_location=0,
        leave_context_keys=frozenset({"leave_flag"}),
    )


def test_door_hop_records_frames() -> None:
    t = _timer()
    t.observe_frame(HopFrame(0, 10, "settled"))
    t.observe_frame(HopFrame(5, 10, "settled"))
    t.observe_frame(
        HopFrame(6, 10, "transition", leave_meta={"leave_flag": 1})
    )
    visit = t.observe_frame(HopFrame(20, 11, "settled"))
    assert visit is not None
    assert visit.loc == 10
    assert visit.dest == 11
    assert visit.entry == 0
    assert visit.leave == 6
    assert visit.exit == 20
    assert len(t.visits) == 1


def test_door_hop_span_records_leave_meta_and_next_anchor() -> None:
    t = _timer()
    t.observe_frame(HopFrame(10, 7, "settled", context={"inv": 3}))
    t.observe_frame(
        HopFrame(12, 7, "transition", leave_meta={"leave_flag": 1})
    )
    # Later transition frames must not move leave_frame nor drop leave_meta.
    t.observe_frame(
        HopFrame(13, 7, "transition", leave_meta={"direction": "E"})
    )
    assert t._open is not None
    assert t._open.leave_frame == 12
    assert t._open.context["leave_flag"] == 1
    assert t._open.context["direction"] == "E"
    visit = t.observe_frame(HopFrame(30, 8, "settled"))
    assert visit is not None
    assert visit.loc == 7
    assert visit.dest == 8
    assert visit.entry == 10
    assert visit.leave == 12
    assert visit.exit == 30
    assert visit.seq == 0
    # Fresh open visit is anchored on the destination at the exit frame.
    assert t._open is not None
    assert t._open.location == 8
    assert t._open.source == 7
    assert t._open.entry_frame == 30
    assert len(t.visits) == 1


def test_settled_jump_is_discontinuity() -> None:
    t = _timer()
    t.observe_frame(HopFrame(0, 1, "settled"))
    assert t.observe_frame(HopFrame(3, 9, "settled")) is None
    assert len(t.visits) == 0
    assert len(t.discontinuities) == 1
    assert t.discontinuities[0].reason == "location_jump"
    # Re-anchored on new location.
    assert t._open is not None
    assert t._open.location == 9


def test_bounce_cancels_leave() -> None:
    t = _timer()
    t.observe_frame(HopFrame(0, 5, "settled", context={"inv": 1}))
    t.observe_frame(
        HopFrame(2, 5, "transition", leave_meta={"leave_flag": 7})
    )
    assert t._open is not None and t._open.in_transition
    assert t.observe_frame(HopFrame(4, 5, "settled", context={"inv": 2})) is None
    assert t._open is not None
    assert not t._open.in_transition
    assert "leave_flag" not in t._open.context
    assert t._open.context["inv"] == 2


def test_seamless_adjacent_hop() -> None:
    t = HopTimer(
        make_visit=_make_visit,
        make_discontinuity=_make_disc,
        seamless_allowed=lambda a, b: abs(a - b) == 1,
        null_location=0,
    )
    t.observe_frame(HopFrame(0, 3, "settled"))
    visit = t.observe_frame(HopFrame(10, 4, "settled"))
    assert visit is not None
    assert visit.leave == 10
    assert visit.exit == 10
    assert visit.dest == 4


def test_finalize_session_end() -> None:
    t = _timer()
    t.observe_frame(HopFrame(0, 1, "settled"))
    t.finalize(frame=50)
    assert t._open is None
    assert any(d.reason == "session_end" for d in t.discontinuities)
