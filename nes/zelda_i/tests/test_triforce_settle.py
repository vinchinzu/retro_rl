"""Table-driven TriforceSettleSpec rows (cleanup Phase 2.1)."""

from __future__ import annotations

from zelda_i.overworld.settle import (
    POST_L1_SETTLE,
    POST_L2_SETTLE,
    POST_L3_SETTLE,
    POST_L4_SETTLE,
    POST_L5_SETTLE,
    TRIFORCE_SETTLES,
    PostL4TriforceSettleController,
    PostTriforceSettleController,
    SettlePhase,
    TriforceSettleController,
    TriforceSettleSpec,
    settle_ready,
)
from zelda_i.ram import PLAY_MODE, read_snapshot
from zelda_i.tests.ram_helpers import make_ram

_FANFARE = 18
_SCROLL = 6


def _snap(*, mode: int = PLAY_MODE, level: int = 0, screen: int, tf: int, raft: int = 0):
    return read_snapshot(
        make_ram(
            {},
            mode=mode,
            level=level,
            screen=screen,
            triforce=tf,
            raft=raft,
            x=112,
            y=125,
        )
    )


def test_settle_table_has_five_unique_rows() -> None:
    assert TRIFORCE_SETTLES == (
        POST_L1_SETTLE,
        POST_L2_SETTLE,
        POST_L3_SETTLE,
        POST_L4_SETTLE,
        POST_L5_SETTLE,
    )
    # Row n settles on level n's Triforce bit, and no two rows share an id.
    assert [s.tf_bit for s in TRIFORCE_SETTLES] == [1 << n for n in range(5)]
    assert len({s.spec_id for s in TRIFORCE_SETTLES}) == 5


def test_each_row_idles_fanfare_then_accepts_matching_ow() -> None:
    for spec in TRIFORCE_SETTLES:
        ctl = TriforceSettleController(spec)
        fanfare = _snap(mode=_FANFARE, level=1, screen=0x03, tf=0xFF, raft=1)
        act = ctl.step(fanfare)
        assert act.reason == "settle_wait", spec.spec_id
        assert ctl.success is False
        ready = _snap(
            screen=spec.require_screen,
            tf=spec.tf_bit,
            raft=1 if spec.item == "raft" else 0,
        )
        act = ctl.step(ready)
        assert ctl.success, spec.spec_id
        assert ctl.phase is SettlePhase.DONE
        assert act.reason == "settle_done"
        assert spec.done_note in ctl.notes


def test_raft_rows_reject_zero_raft_others_do_not() -> None:
    no_raft = _snap(screen=POST_L4_SETTLE.require_screen, tf=0x08, raft=0)
    assert not settle_ready(POST_L4_SETTLE, no_raft)
    assert not settle_ready(POST_L3_SETTLE, _snap(screen=0x74, tf=0x04, raft=0))
    assert settle_ready(POST_L1_SETTLE, _snap(screen=0x37, tf=0x01, raft=0))
    assert settle_ready(POST_L5_SETTLE, _snap(screen=0x0B, tf=0x10, raft=0))


def test_wrong_screen_or_missing_bit_or_scroll_is_not_ready() -> None:
    spec = POST_L2_SETTLE
    assert not settle_ready(spec, _snap(screen=0x37, tf=0x02))
    assert not settle_ready(spec, _snap(screen=0x3C, tf=0x01))
    assert not settle_ready(
        spec, _snap(mode=_SCROLL, screen=0x3C, tf=0x02)
    )


def test_timeout_fails_closed() -> None:
    spec = TriforceSettleSpec("tiny", 0x37, 0x01, max_frames=2)
    ctl = TriforceSettleController(spec)
    stuck = _snap(mode=_FANFARE, level=1, screen=0x03, tf=0)
    assert ctl.step(stuck).reason == "settle_wait"
    act = ctl.step(stuck)
    assert act.reason == "timeout"
    assert ctl.phase is SettlePhase.FAILED
    assert ctl.success is False


def test_bound_constructors_keep_historical_no_arg_call() -> None:
    l1 = PostTriforceSettleController()
    l4 = PostL4TriforceSettleController()
    assert l1.spec is POST_L1_SETTLE
    assert l4.spec is POST_L4_SETTLE
