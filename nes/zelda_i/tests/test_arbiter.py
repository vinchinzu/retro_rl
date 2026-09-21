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


# ------------------------------------------------- the wired ladders ---
# Everything above drives the arbiter on its own. Everything below drives
# the two real ladders, because a rung table nothing walks is a rename test.


def _hop_ctl():
    """The one path controller with a hunter, a skirt rung and a live walk."""
    from zelda_i.overworld.shop_p7 import ShopP7WalkController

    return ShopP7WalkController()


def _hop_snap(**kwargs) -> ZeldaSnapshot:
    return _snap(link_x=120, link_y=141, **kwargs)


def _names(arb: Arbiter) -> list[str]:
    return [r.name for r in arb.rungs]


# --- structural: names, uniqueness, and where the beam rung comes from ---


def test_the_hop_ladder_is_the_old_source_order() -> None:
    """The order ``_do_hop`` used to spell as ten ``return``s, as one list."""
    from zelda_i.overworld.path import OverworldPathController

    ctl = _hop_ctl()
    assert _names(ctl.hop_arbiter) == [
        "hop_extra",
        "hop_maze",
        "hop_beam",
        "hop_scoop",
        "hop_occupancy",
        "hop_stall_escape",
        "hop_unstick",
        "hop_edge",
        "hop_hunt",
        "hop_lane",
    ]
    # Every hook sits where the call used to: above everything.
    assert _names(OverworldPathController(hops=()).hop_arbiter)[0] == "hop_extra"


def test_the_hunt_ladder_is_the_old_source_order() -> None:
    from zelda_i.overworld.hunt import ScreenHunter

    assert _names(ScreenHunter().arbiter) == [
        "hunt_strike",
        "hunt_peel",
        "hunt_shield",
        "hunt_duck",
        "hunt_heal",
        "hunt_beam",
        "hunt_transit",
        "hunt_guard",
        "hunt_collect",
        "hunt_chase",
    ]


def test_neither_wired_ladder_repeats_a_name() -> None:
    """``Arbiter.__post_init__`` raises on a duplicate, so this is really an
    assertion that both ladders can be built at all — but it also says the
    census keys are one behaviour each, which is the point of the census."""
    from zelda_i.overworld.hunt import ScreenHunter

    for arb in (_hop_ctl().hop_arbiter, ScreenHunter().arbiter):
        names = _names(arb)
        assert len(names) == len(set(names)), names
        assert list(arb.census()) == names


def test_the_beam_rung_is_the_hunters_own_take_beam() -> None:
    """Retiring the out-of-band call means the rung *is* ``take_beam``."""
    ctl = _hop_ctl()
    beam = next(r for r in ctl.hop_arbiter.rungs if r.name == "hop_beam")
    assert beam.fn == ctl.hunter.take_beam
    # ``_do_hop`` no longer names it.
    import inspect

    src = inspect.getsource(type(ctl).__mro__[1]._do_hop)
    assert "take_beam" not in src, src


def test_a_path_with_no_hunter_has_no_beam_and_no_hunt_rung() -> None:
    from zelda_i.overworld.path import OverworldPathController

    names = _names(OverworldPathController(hops=()).hop_arbiter)
    assert "hop_beam" not in names
    # The hunt rung is still registered (it declines on a hunterless path);
    # the beam is not, because there is no bound method to register.
    assert "hop_hunt" in names


def test_swapping_the_hunter_rebuilds_the_ladder() -> None:
    """The beam rung is a *bound* method, so a stale ladder would fire the
    old hunter's blade."""
    from zelda_i.overworld.hunt import ScreenHunter

    ctl = _hop_ctl()
    first = ctl.hop_arbiter
    assert ctl.hop_arbiter is first
    ctl.hunter = ScreenHunter()
    assert ctl.hop_arbiter is not first
    beam = next(r for r in ctl.hop_arbiter.rungs if r.name == "hop_beam")
    assert beam.fn == ctl.hunter.take_beam


# --- behavioural: these fail if the ORDER changes, not if a name changes ---


def test_the_beam_outranks_the_stall_escape_because_of_its_number() -> None:
    """AGENTS.md Traps: "travelling frames must offer ``take_beam`` above
    stall-escape (600f commits used to zero the weapon)".

    Both rungs claim here. The beam wins, and it wins *because of the
    number*: give the same two rung objects the opposite priorities and the
    escape wins instead. Nothing about the source order changed.
    """
    from dataclasses import replace as _replace

    from zelda_i.overworld.path import HOP_RUNG_BEAM, HOP_RUNG_STALL_ESCAPE

    ctl = _hop_ctl()
    ctl.hunter.take_beam = lambda snap: _act("beam_79")
    ctl._stall_escape = lambda snap, hop: _act("hop2_escape")
    ctl.hop_index = 2
    assert ctl._do_hop(_hop_snap()).reason == "beam_79"
    assert ctl.hop_arbiter.last_winner == "hop_beam"
    assert ctl.hop_arbiter.census()["hop_stall_escape"] == 0

    flipped = Arbiter(
        tuple(
            _replace(
                r,
                priority={
                    "hop_beam": HOP_RUNG_STALL_ESCAPE,
                    "hop_stall_escape": HOP_RUNG_BEAM,
                }.get(r.name, r.priority),
            )
            for r in ctl.hop_arbiter.rungs
        )
    )
    ctl._hop = ctl.hops[2]
    assert flipped.decide(_hop_snap()).reason == "hop2_escape"
    assert flipped.last_winner == "hop_stall_escape"


def test_the_0x79_skirt_is_a_clearance_gate_not_a_precedence_edit() -> None:
    """The gate stays; only its *place* became data.

    C2 read ``0x79 not in hunter.done`` as a precedence edit — the hook
    declining so the hunt below could have the frame — and tried to spell it
    as a rung under ``hop_hunt``. Wiring the ladder proved the reading wrong,
    and this test is the counter-example.

    A completion gate opens **once** and stays open. A rung under the hunt
    opens on every frame the hunt declines, of which there are many while the
    wave is alive. So the two are not the same set, the swap changed
    behaviour in both directions on a frame-perfect chain, and the gate was
    put back. What C2 keeps is that the hook's place is now the number
    ``extra_hop_priority`` rather than the line ``_do_hop`` calls it from.
    """
    from zelda_i.overworld.path import HOP_RUNG_EXTRA

    ctl = _hop_ctl()
    ctl.hop_index = 2  # 0x79 -> 0x7A, the hop the skirt exists for
    assert ctl.extra_hop_priority == HOP_RUNG_EXTRA
    assert _names(ctl.hop_arbiter)[0] == "hop_extra"

    # Wave still alive on 0x79: the gate is shut, and the frame falls all the
    # way through to the hunt — exactly as it did before the ladder existed.
    assert 0x79 not in ctl.hunter.done
    ctl._hunt_action = lambda snap, hop: _act("hunt_79")
    assert ctl._do_hop(_hop_snap()).reason == "hunt_79"

    # The hunt declining does NOT open the gate. This is the whole difference
    # between a clearance gate and a rung below the hunt: under the rung the
    # skirt would take this frame.
    ctl._hunt_action = lambda snap, hop: None
    assert not ctl._do_hop(_hop_snap()).reason.startswith("79_skirt")

    # Chase finished: the gate opens, and now the skirt outranks everything,
    # including a hunt that wants the frame back.
    ctl.hunter.done.add(0x79)
    ctl._hunt_action = lambda snap, hop: _act("hunt_79")
    assert ctl._do_hop(_hop_snap()).reason.startswith("79_skirt")


def test_the_0x79_gate_asks_a_named_query_not_the_done_set() -> None:
    """The read is declared, not a reach into another module's bookkeeping."""
    from zelda_i.overworld.hunt import ScreenHunter
    from zelda_i.overworld.shop_p7 import ShopP7WalkController

    hunter = ScreenHunter()
    assert not hunter.chase_finished(0x79)
    hunter.done.add(0x79)
    assert hunter.chase_finished(0x79)

    # The names the hook loads, from the compiled method rather than its
    # text, so the prose above it can still quote the set it stopped poking.
    loaded = ShopP7WalkController._extra_hop_action.__code__.co_names
    assert "chase_finished" in loaded, loaded
    assert "done" not in loaded, loaded


def test_the_default_hook_still_outranks_the_whole_hop_ladder() -> None:
    """``shop_p7`` moved its own hook; nobody else's moved. ``topup``'s hold
    and the L3/L6/L7/L8 corridor hooks still sit above the maze and the
    beam, which is where their call site used to put them."""
    from zelda_i.overworld.path import HOP_RUNG_EXTRA, OverworldPathController

    ctl = OverworldPathController(hops=_hop_ctl().hops)
    ctl.hop_index = 2
    ctl._extra_hop_action = lambda snap, hop: _act("topup_hold")
    ctl._stall_escape = lambda snap, hop: _act("escape")
    assert ctl.extra_hop_priority == HOP_RUNG_EXTRA
    assert ctl._do_hop(_hop_snap()).reason == "topup_hold"


def test_the_hunt_heal_outranks_the_beam_because_of_its_number() -> None:
    """The beam is a full-health weapon, so on every frame the heal can
    claim, the rung below it is already dead."""
    from zelda_i.overworld.hunt import HUNT_RUNG_BEAM, HUNT_RUNG_HEAL, ScreenHunter

    assert HUNT_RUNG_HEAL < HUNT_RUNG_BEAM
    hunter = ScreenHunter()
    hunter._take_heal = lambda snap, frames: _act("hunt_heal")
    hunter._beam_action = lambda snap, screen: _act("beam_79")
    act = hunter.step(_snap(), 0)
    assert act is not None and act.reason == "hunt_heal"
    assert hunter.arbiter.last_winner == "hunt_heal"


def test_the_hunt_tail_is_a_dispatch_and_a_decline_ends_the_step() -> None:
    """The bottom four rungs are *not* a ladder: exactly one owns the frame
    and its ``None`` is the hunt handing the frame back.

    A transit screen at one heart is the case that would leak: guard's
    condition is true as well, and a naive ladder would let guard charge the
    screen budget behind transit's back.
    """
    from zelda_i.overworld.hunt import ScreenHunter

    hunter = ScreenHunter(transit_screens=frozenset({0x7B}))
    snap = _snap(screen=0x7B, next_screen=0x7B, health=0x20)  # 1 of 3 hearts
    assert int(snap.whole_hearts) <= hunter.min_hearts  # guard would fire
    assert hunter.step(snap, 0) is None  # nothing to collect: back to the path
    assert hunter.census.transit_frames == 1
    assert hunter.census.guard_frames == 0
    assert hunter.screen_frames == 0  # guard's ``_spend`` never ran
    assert hunter.arbiter.last_winner is None
    assert hunter.arbiter.idle_frames == 1


# --- the census ---


def test_the_hop_census_sums_to_the_frames_the_ladder_decided() -> None:
    ctl = _hop_ctl()
    ctl.hop_index = 2
    for _ in range(5):
        ctl._do_hop(_hop_snap())
    report = ctl.report()
    census = report["rung_census"]
    assert sum(census.values()) == report["rung_frames"] == 5
    # ``push`` is the fall-through, not a rung: the frames ``align_and_push``
    # took because the whole ladder declined.
    assert census["push"] == ctl.hop_arbiter.idle_frames


def test_the_hunt_census_sums_to_the_frames_the_ladder_decided() -> None:
    from zelda_i.overworld.hunt import ScreenHunter

    hunter = ScreenHunter()
    for _ in range(4):
        hunter.step(_snap(), 0)
    report = hunter.report()
    census = report["rung_census"]
    assert sum(census.values()) == report["rung_frames"] == 4
    assert census["yielded"] == hunter.arbiter.idle_frames


def test_the_census_credits_the_rung_that_returned_the_action() -> None:
    """The drift the six hand-incremented sites could have: a counter bumped
    in a branch that then hands the frame to somebody else."""
    ctl = _hop_ctl()
    ctl.hop_index = 2
    ctl._hunt_action = lambda snap, hop: _act("hunt_79")
    for _ in range(3):
        ctl._do_hop(_hop_snap())
    census = ctl.report()["rung_census"]
    assert census["hop_hunt"] == 3
    assert census["hop_extra"] == 0  # it ran and declined; it did not win
    assert census["push"] == 0


def test_a_path_reset_clears_the_census_and_keeps_the_ladder() -> None:
    ctl = _hop_ctl()
    ctl.hop_index = 2
    ctl._do_hop(_hop_snap())
    names = _names(ctl.hop_arbiter)
    ctl.reset()
    assert _names(ctl.hop_arbiter) == names
    assert ctl.report()["rung_frames"] == 0


def test_a_hunt_reset_clears_the_census_and_keeps_the_ladder() -> None:
    from zelda_i.overworld.hunt import ScreenHunter

    hunter = ScreenHunter()
    hunter.step(_snap(), 0)
    names = _names(hunter.arbiter)
    hunter.reset()
    assert _names(hunter.arbiter) == names
    assert hunter.arbiter.frames == 0
