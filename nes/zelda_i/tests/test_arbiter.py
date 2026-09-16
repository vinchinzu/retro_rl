"""The decision ladder as data: order, declines, and the win census.

These are the assertions the implicit ``if act is not None: return act``
chains in ``overworld.path`` and ``overworld.hunt`` cannot carry — precedence
there is source-line order, only observable by driving a whole ``step()`` and
matching ``FrameAction.reason``. Here the order is a number, so the
take_beam-above-stall-escape kind of question is one assert.
"""

from __future__ import annotations

import pytest

from retro_harness.input_script import FrameAction
from zelda_i.overworld.arbiter import STAMP_SEPARATOR, Arbiter, Rung
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot


def _snap(**kwargs) -> ZeldaSnapshot:
    fields = dict(
        mode=PLAY_MODE,
        level=0,
        screen=0x79,
        next_screen=0x79,
        link_x=120,
        link_y=141,
        facing=0,
        sword=1,
        bombs=0,
        rupees=0,
        keys=0,
        health=0x22,  # 3 containers, 2 whole hearts
        heart_partial=0xFF,
        triforce=0,
        compass=0,
        dialog_timer=0,
        colliding_tile=0,
        room_item_id=0,
        room_all_dead=0,
        room_obj_count=0,
        cur_opened_doors=0,
        open_doorway_mask=0,
        objects=(),
    )
    fields.update(kwargs)
    return ZeldaSnapshot(**fields)


def _act(reason: str) -> FrameAction:
    return FrameAction([0, 0, 0, 0, 0, 0, 0, 0], reason)


def _always(reason: str):
    """A rung that claims every frame, like a stall-escape mid-commit."""
    return lambda snap: _act(reason)


def _never(snap: ZeldaSnapshot) -> FrameAction | None:
    """A rung that declines, like ``take_beam`` with no shot available."""
    return None


def _on_screen(screen: int, reason: str):
    return lambda snap: _act(reason) if int(snap.screen) == screen else None


def test_highest_rung_wins_a_contested_frame() -> None:
    arb = Arbiter(
        (
            Rung("stall_escape", 20, _always("escape")),
            Rung("take_beam", 10, _always("beam")),
        )
    )
    act = arb.decide(_snap())
    assert act is not None
    assert act.reason == "beam"
    assert arb.last_winner == "take_beam"


def test_decline_is_skipped_and_the_next_rung_is_consulted() -> None:
    arb = Arbiter((Rung("take_beam", 10, _never), Rung("stall_escape", 20, _always("escape"))))
    act = arb.decide(_snap())
    assert act is not None and act.reason == "escape"
    assert arb.last_winner == "stall_escape"


def test_all_declining_yields_none() -> None:
    arb = Arbiter((Rung("take_beam", 10, _never), Rung("scoop", 20, _never)))
    assert arb.decide(_snap()) is None
    assert arb.last_winner is None
    assert arb.idle_frames == 1


def test_empty_ladder_declines() -> None:
    arb = Arbiter(())
    assert arb.decide(_snap()) is None
    assert arb.census() == {}


def test_census_counts_wins_and_not_declines() -> None:
    arb = Arbiter(
        (
            Rung("hunt_79", 10, _on_screen(0x79, "hunt")),
            Rung("hop", 20, _always("hop")),
            Rung("quiet", 30, _never),
        )
    )
    for screen in (0x79, 0x7A, 0x7A, 0x79):
        arb.decide(_snap(screen=screen, next_screen=screen))
    assert arb.census() == {"hunt_79": 2, "hop": 2, "quiet": 0}
    assert arb.frames == 4
    assert arb.idle_frames == 0


def test_census_lists_every_rung_in_ladder_order() -> None:
    arb = Arbiter((Rung("low", 30, _never), Rung("top", 10, _always("a"))))
    assert list(arb.census()) == ["top", "low"]


def test_idle_frames_count_only_unclaimed_frames() -> None:
    arb = Arbiter((Rung("hunt_79", 10, _on_screen(0x79, "hunt")),))
    arb.decide(_snap(screen=0x79, next_screen=0x79))
    arb.decide(_snap(screen=0x7A, next_screen=0x7A))
    arb.decide(_snap(screen=0x7A, next_screen=0x7A))
    assert arb.census() == {"hunt_79": 1}
    assert (arb.frames, arb.idle_frames) == (3, 2)


def test_reset_clears_accounting_and_keeps_the_ladder() -> None:
    arb = Arbiter((Rung("hop", 20, _always("hop")),))
    arb.decide(_snap())
    arb.reset()
    assert arb.census() == {"hop": 0}
    assert (arb.frames, arb.idle_frames, arb.last_winner) == (0, 0, None)
    assert arb.decide(_snap()) is not None


def test_duplicate_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="take_beam"):
        Arbiter((Rung("take_beam", 10, _never), Rung("take_beam", 20, _always("x"))))


def test_order_is_data_not_insertion_order() -> None:
    beam = Rung("take_beam", 10, _always("beam"))
    escape = Rung("stall_escape", 20, _always("escape"))
    forward = Arbiter((beam, escape))
    backward = Arbiter((escape, beam))
    assert [r.name for r in forward.rungs] == [r.name for r in backward.rungs]
    assert forward.decide(_snap()).reason == backward.decide(_snap()).reason == "beam"
    assert forward.last_winner == backward.last_winner == "take_beam"


def test_equal_priorities_are_broken_by_name_not_insertion() -> None:
    a = Rung("aaa", 10, _always("a"))
    b = Rung("bbb", 10, _always("b"))
    assert Arbiter((b, a)).decide(_snap()).reason == "a"
    assert Arbiter((a, b)).decide(_snap()).reason == "a"


def test_rungs_may_come_from_more_than_one_owner() -> None:
    class _Owner:
        def __init__(self, reason: str, claim: bool) -> None:
            self.reason, self.claim, self.calls = reason, claim, 0

        def act(self, snap: ZeldaSnapshot) -> FrameAction | None:
            self.calls += 1
            return _act(self.reason) if self.claim else None

    path, hunt = _Owner("hop0", True), _Owner("beam", True)
    quiet = _Owner("scoop", False)
    arb = Arbiter(
        (
            Rung("path_hop", 30, path.act),
            Rung("path_scoop", 20, quiet.act),
            Rung("hunt_beam", 10, hunt.act),
        )
    )
    assert arb.decide(_snap()).reason == "beam"
    assert (hunt.calls, quiet.calls, path.calls) == (1, 0, 0)


def test_stamp_is_off_by_default() -> None:
    arb = Arbiter((Rung("hunt_beam", 10, _always("beam_79")),))
    assert arb.decide(_snap()).reason == "beam_79"


def test_stamp_appends_the_winner_without_losing_the_reason() -> None:
    arb = Arbiter((Rung("hunt_beam", 10, _always("beam_79")),), stamp=True)
    assert arb.decide(_snap()).reason == f"beam_79{STAMP_SEPARATOR}hunt_beam"


def test_stamp_does_not_mutate_the_rung_action() -> None:
    shared = _act("beam_79")
    arb = Arbiter((Rung("hunt_beam", 10, lambda snap: shared),), stamp=True)
    stamped = arb.decide(_snap())
    assert stamped is not shared
    assert shared.reason == "beam_79"
    assert stamped.action == shared.action


def test_stamp_names_an_empty_reason() -> None:
    arb = Arbiter((Rung("hunt_beam", 10, lambda snap: _act("")),), stamp=True)
    assert arb.decide(_snap()).reason == "hunt_beam"
