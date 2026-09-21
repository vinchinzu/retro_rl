"""Arrive-short top-up at the coast shop. No emulator.

The walk's own stop is the shop screen; the errand's is ``ADDR_BOMBS``. This
stage is what sits between them when the pass banked 19R of a 20R pack
(``pre_l1_beam4``).
"""

from __future__ import annotations

import pytest

from retro_harness.controls import pressed_nes_buttons
from zelda_i.overworld.graph import ScreenHop
from zelda_i.overworld.respawn import RoomHistory, respawn_visits
from zelda_i.overworld.shop_p7 import SHOP_P7_PRICE, SHOP_P7_SCREEN, shop_p7_screens
from zelda_i.overworld.topup import (
    SCREEN_6E_WEST_BAND,
    SCREEN_6F_NORTH_X,
    SHOP_P7_TOPUP_EXCURSIONS,
    TOPUP_MAX_FRAMES,
    Excursion,
    RupeeTopUpController,
    excursion_hops,
    make_shop_p7_topup_controller,
)
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot

NORTH = Excursion(
    "n_5f",
    ScreenHop(0x5F, "UP", align_x=120),
    ScreenHop(SHOP_P7_SCREEN, "DOWN", align_x=120),
)
WEST = Excursion(
    "w_6e",
    ScreenHop(0x6E, "LEFT", align_y=141),
    ScreenHop(SHOP_P7_SCREEN, "RIGHT", align_y=141),
)


def _snap(**kwargs) -> ZeldaSnapshot:
    fields = dict(
        mode=PLAY_MODE,
        level=0,
        screen=SHOP_P7_SCREEN,
        next_screen=SHOP_P7_SCREEN,
        link_x=120,
        link_y=141,
        facing=0,
        sword=1,
        bombs=0,
        rupees=0,
        keys=0,
        health=0x22,
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


def _ctl(excursions=(NORTH, WEST)) -> RupeeTopUpController:
    return make_shop_p7_topup_controller(
        shop_screen=SHOP_P7_SCREEN, price=SHOP_P7_PRICE, excursions=excursions
    )


# ----------------------------------------------------------- the table ---


def test_the_shipped_table_is_the_two_measured_neighbours() -> None:
    """``scratch/probe_6f_neighbours.py`` tag ``n4``: one boot, the real hop
    table to 0x6F, the state saved on arrival, then every candidate
    row/column restored, walked to, pushed, censused and pushed back.

    0x5F crossed from every column 72..232 and came home in 85f; 0x6E crossed
    from y 93..197 and came home in 106f, with 77 / 85 / 205 dead. A painted
    neighbour is how the walk spent 27501 frames on a dead row
    (``shop_p7.SCREEN_7A_EAST_BAND``), so the bands here are the sweep's.
    """
    names = tuple(e.name for e in SHOP_P7_TOPUP_EXCURSIONS)
    assert names == ("north_5f", "west_6e")
    north, west = SHOP_P7_TOPUP_EXCURSIONS
    assert north.out.target == 0x5F and north.out.direction == "UP"
    assert north.out.align_x == SCREEN_6F_NORTH_X == 122
    assert west.out.target == 0x6E and west.out.direction == "LEFT"
    assert west.out.y_band == SCREEN_6E_WEST_BAND == (109, 189)
    # Both come home, and the return is the reverse push on the same lane.
    assert north.back.direction == "DOWN" and north.back.align_x == 122
    assert west.back.direction == "RIGHT" and west.back.y_band == (109, 189)


def test_an_empty_table_is_still_a_legitimate_state() -> None:
    """A shop with no measured neighbour gets a no-op stage, not a stub."""
    assert _ctl(excursions=()).hops == ()


def test_every_excursion_has_to_come_home() -> None:
    """An excursion that ends next door strands the buy stage on the wrong
    screen — and the cave mouth is on 0x6F."""
    with pytest.raises(ValueError):
        excursion_hops(
            SHOP_P7_SCREEN,
            (Excursion("bad", ScreenHop(0x5F, "UP"), ScreenHop(0x5E, "DOWN")),),
        )
    with pytest.raises(ValueError):
        Excursion("round", ScreenHop(0x5F, "UP"), ScreenHop(0x5F, "DOWN"))


def test_the_table_flattens_out_back_out_back() -> None:
    hops = excursion_hops(SHOP_P7_SCREEN, (NORTH, WEST))
    assert tuple(h.target for h in hops) == (0x5F, 0x6F, 0x6E, 0x6F)


def test_the_excursions_are_off_the_walk_not_behind_it() -> None:
    """``overworld.respawn``: ``RoomHistory`` is six slots, so the five screens
    behind 0x6F are all still in the ring on arrival and none of their waves
    come back. A neighbour the walk never entered is the only fresh fight one
    hop away."""
    walked = shop_p7_screens()
    history = RoomHistory()
    for screen in walked:
        history.enter(screen)
    for trip in (NORTH, WEST):
        assert trip.out.target not in walked
        # And the ROM would reopen it: absent from the ring.
        assert trip.out.target not in history.slots
    # The control, and the price of the alternative: every screen is fresh on
    # the way out, and walking back gives nothing until the *sixth* screen —
    # the ring still holds the five behind the shop. Six screens of leevers
    # and a Zora each way is the trip a one-hop neighbour replaces.
    back = tuple(reversed(walked))[1:]
    visits = respawn_visits(walked + back)
    assert all(visits[: len(walked)])
    backtrack = visits[len(walked) :]
    assert not any(backtrack[:5])
    assert backtrack[5] is True
    assert back[5] == 0x7A


# ------------------------------------------------------ the controller ---


def test_a_walk_that_already_has_the_price_stops_on_its_first_frame() -> None:
    """This is what makes the stage safe to leave in the list every pass."""
    ctl = _ctl()
    ctl.step(_snap(rupees=SHOP_P7_PRICE))
    assert ctl.success is True
    assert ctl.phase.name == "DONE"
    assert ctl.hop_index == 0  # the out hop was never taken


def test_being_short_on_the_shop_screen_is_not_a_stop() -> None:
    ctl = _ctl()
    ctl.step(_snap(rupees=SHOP_P7_PRICE - 1))
    assert ctl.success is False


def test_the_money_alone_is_not_a_stop_off_the_shop_screen() -> None:
    """The stage after this one enters a cave mouth on 0x6F: finishing one
    screen north with the money is the same failure as not having it."""
    ctl = _ctl()
    assert ctl._at_stop(_snap(screen=0x5F, rupees=SHOP_P7_PRICE)) is False
    assert ctl._at_stop(_snap(screen=SHOP_P7_SCREEN, rupees=SHOP_P7_PRICE)) is True


def test_an_exhausted_table_finishes_home_short_rather_than_failing() -> None:
    """The buy stage owns "could not afford it" — it is the one that reads
    ``ADDR_BOMBS``. Failing here would hide the rupee count behind a walk
    failure."""
    ctl = _ctl(excursions=())
    ctl.hunt_destination = False  # the destination wave is the walk's errand
    ctl.step(_snap(rupees=3))
    assert ctl.success is True
    assert ctl.phase.name == "DONE"
    assert "topup_short" in " ".join(ctl.notes)


def test_the_top_up_reopens_screens_it_re_enters() -> None:
    """It comes home to 0x6F once per excursion; leaving the screen in
    ``done`` would decline a wave the ROM did put back."""
    ctl = _ctl()
    assert ctl.hunter is not None and ctl.hunter.reopen_on_enter is True
    assert ctl.max_frames == TOPUP_MAX_FRAMES
    assert ctl.evade is True and ctl.scoop_rupees is True


def test_the_back_hop_does_not_retrace_before_the_neighbour_is_fought() -> None:
    """Live t1: hop_index advanced onto the DOWN home hop the first play
    frame of 0x5F. Hunt then skipped because y>200 is the DOWN arrival
    edge, recover allowed DOWN, and the hop retraced in 87f with peak_live
    0. The hold has to sit in extra, above that skip."""
    ctl = _ctl()
    ctl.hop_index = 1  # already on the back hop, as live t1 was
    act = ctl.step(_snap(screen=0x5F, link_x=80, link_y=221, rupees=0))
    assert ctl.success is False
    buttons = pressed_nes_buttons(list(act.action))
    assert "DOWN" not in buttons
    assert "topup_hold" in act.reason
    assert "UP" in buttons  # off the south scroll line, into the wave


def test_the_west_back_hop_does_not_retrace_either() -> None:
    ctl = _ctl()
    ctl.hop_index = 3  # RIGHT home from 0x6E
    act = ctl.step(_snap(screen=0x6E, link_x=240, link_y=141, rupees=0))
    assert ctl.success is False
    buttons = pressed_nes_buttons(list(act.action))
    assert "RIGHT" not in buttons
    assert "topup_hold" in act.reason
    assert "LEFT" in buttons


def test_the_back_hop_runs_once_the_neighbour_is_fought() -> None:
    ctl = _ctl()
    ctl.hop_index = 1
    ctl.hunter.done.add(0x5F)
    ctl.hunter.cleared.add(0x5F)
    act = ctl.step(_snap(screen=0x5F, link_x=80, link_y=221, rupees=0))
    assert ctl.success is False
    assert "DOWN" in pressed_nes_buttons(list(act.action))
