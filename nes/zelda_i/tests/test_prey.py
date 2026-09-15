"""The ROM drop arithmetic the hunt's target selection is built on.

No emulator. Every number here is transcribed from ``Z_04.asm``
(``DropItemRates`` / ``Types0..3`` / ``SetUpDroppedItem``) via
``scratch/drop_mechanics_rom.md``, so a test that disagrees with the ROM is
the thing to fix, not the table.
"""

from __future__ import annotations

import pytest

from zelda_i.dungeon.behaviors import (
    ZORA_CYCLE,
    ZORA_MUZZLE_DWELL,
    ZORA_SHOT_DELAY,
    ZORA_STATE_AIMING,
    ZORA_STATE_FIRING,
    ZORA_STATE_SUBMERGED,
    ZORA_STATE_SUBMERGING,
    zora_shot_eta,
)
from zelda_i.dungeon.ids import (
    OCTOROK_OBJECT_TYPE,
    PEAHAT_OBJECT_TYPE,
    TEKTITE_BLUE_OBJECT_TYPE,
    TEKTITE_OBJECT_TYPE,
    ZORA_OBJECT_TYPE,
)
from zelda_i.overworld.prey import (
    CHASE_FLOOR,
    STREAK_RUPEES,
    THRIFTY_CHASE_RADIUS,
    PreyPolicy,
    drop_row,
    forced_drop_kills,
    kill_value,
    random_rupees,
    streak_forfeit,
)
from zelda_i.ram import ZeldaObject


def _body(type_id: int, x: int = 0, y: int = 0, slot: int = 1) -> ZeldaObject:
    return ZeldaObject(slot=slot, type_id=type_id, x=x, y=y, facing=0, hp=2, state=0)


# ------------------------------------------------------------- the rows ---


@pytest.mark.parametrize(
    "type_id,row",
    [
        (OCTOROK_OBJECT_TYPE, 0),
        (TEKTITE_OBJECT_TYPE, 0),  # red tektite is row 0; only the blue is row 1
        (TEKTITE_BLUE_OBJECT_TYPE, 1),
        (0x09, 2),  # blue octorok: the bomb table
        (PEAHAT_OBJECT_TYPE, 3),  # unlisted types fall to row 3, as the ROM does
        (ZORA_OBJECT_TYPE, 3),
    ],
)
def test_drop_row_matches_the_rom_type_lists(type_id: int, row: int) -> None:
    assert drop_row(type_id) == row


def test_row_one_is_the_only_table_worth_a_rupee_a_kill() -> None:
    """Two 5-rupees in ten columns at 152/256. Everything else is under 0.2."""
    assert random_rupees(1) == pytest.approx(0.891, abs=0.001)
    assert random_rupees(0) == pytest.approx(0.156, abs=0.001)
    assert random_rupees(2) == pytest.approx(0.122, abs=0.001)
    assert random_rupees(3) == pytest.approx(0.081, abs=0.001)


def test_the_streak_is_worth_more_than_row_zero_s_own_drop() -> None:
    """Which is why "skip the red octoroks" cannot mean "walk past them"."""
    assert STREAK_RUPEES > random_rupees(0)
    assert kill_value(OCTOROK_OBJECT_TYPE) == pytest.approx(
        random_rupees(0) + STREAK_RUPEES
    )


def test_only_row_one_clears_the_chase_floor() -> None:
    assert kill_value(TEKTITE_BLUE_OBJECT_TYPE) >= CHASE_FLOOR
    for type_id in (OCTOROK_OBJECT_TYPE, 0x09, PEAHAT_OBJECT_TYPE):
        assert kill_value(type_id) < CHASE_FLOOR


def test_the_fairy_at_sixteen_delays_the_second_forced_rupee() -> None:
    """``$0627 == 16`` is tested first and zeroes ``$0050`` with six kills of
    5-rupee progress on it, so a clean streak pays at 10 and 26, not 10 and 20."""
    assert forced_drop_kills(50) == (10, 26, 36, 46)


def test_a_hit_forfeits_every_kill_already_banked() -> None:
    assert streak_forfeit(0) == 0
    assert streak_forfeit(7) == pytest.approx(7 * STREAK_RUPEES)
    assert streak_forfeit(-3) == 0


# ----------------------------------------------------------- the policy ---


def test_a_zora_is_skipped_at_every_distance() -> None:
    policy = PreyPolicy()
    zora = _body(ZORA_OBJECT_TYPE)
    assert policy.skipped(zora)
    assert policy.chase_radius(zora) == 0
    assert not policy.worth_chasing(zora, 0)


def test_free_reach_outranks_every_gate() -> None:
    """A body at Link's feet is one swing, not a chase — and a cheap kill is
    still a streak tick."""
    policy = PreyPolicy()
    octorok = _body(OCTOROK_OBJECT_TYPE)
    assert policy.worth_chasing(octorok, policy.free_reach, hearts=1, budget_left=0)


def test_short_health_caps_a_cheap_chase_but_not_a_rich_one() -> None:
    policy = PreyPolicy()
    far = THRIFTY_CHASE_RADIUS + 1
    assert policy.worth_chasing(_body(OCTOROK_OBJECT_TYPE), far, hearts=3)
    assert not policy.worth_chasing(_body(OCTOROK_OBJECT_TYPE), far, hearts=2)
    assert policy.worth_chasing(_body(TEKTITE_BLUE_OBJECT_TYPE), far, hearts=2)


def test_a_chase_the_budget_cannot_finish_is_refused() -> None:
    """Link walks ~1 px/frame: past the frames left, the chase ends in a retire."""
    policy = PreyPolicy()
    assert not policy.worth_chasing(
        _body(TEKTITE_BLUE_OBJECT_TYPE), 150, budget_left=100
    )


def test_the_richer_row_scores_higher_at_equal_range() -> None:
    policy = PreyPolicy()
    pad = 80
    assert policy.score(_body(TEKTITE_BLUE_OBJECT_TYPE), pad) > policy.score(
        _body(OCTOROK_OBJECT_TYPE), pad
    )


def test_a_near_cheap_body_can_outscore_a_far_rich_one() -> None:
    """Value over the walk that buys it, not value alone: the ordering is
    rupees per frame, and Link walks one pixel a frame."""
    policy = PreyPolicy()
    assert policy.score(_body(OCTOROK_OBJECT_TYPE), 8) > policy.score(
        _body(TEKTITE_BLUE_OBJECT_TYPE), 180
    )


def test_the_policy_can_be_switched_off_whole() -> None:
    policy = PreyPolicy(enabled=False)
    assert not policy.skipped(_body(ZORA_OBJECT_TYPE))
    assert policy.worth_chasing(_body(OCTOROK_OBJECT_TYPE), 200, hearts=1)


# ------------------------------------------------------- the zora clock ---


def test_the_measured_zora_cycle_is_195_frames() -> None:
    """scratch/zora1.json: four surfacings on 0x59, identical to the frame."""
    assert sum(ZORA_CYCLE.values()) == 195


def test_the_aiming_state_gives_more_warning_than_a_dodge_needs() -> None:
    """15 frames of ObjState 0x02, then the shot 2 frames into 0x03, then 17
    frames of it sitting on the muzzle: 34 before anything travels."""
    zora = _body(ZORA_OBJECT_TYPE)
    aiming = ZeldaObject(**{**zora.__dict__, "state": ZORA_STATE_AIMING})
    assert zora_shot_eta(aiming) == (
        ZORA_CYCLE[ZORA_STATE_AIMING] + ZORA_SHOT_DELAY + ZORA_MUZZLE_DWELL
    )


def test_a_firing_zora_still_has_the_muzzle_dwell_left() -> None:
    zora = _body(ZORA_OBJECT_TYPE)
    firing = ZeldaObject(**{**zora.__dict__, "state": ZORA_STATE_FIRING})
    assert zora_shot_eta(firing) == ZORA_SHOT_DELAY + ZORA_MUZZLE_DWELL


def test_a_submerging_zora_has_nothing_left_to_walk_away_from() -> None:
    zora = _body(ZORA_OBJECT_TYPE)
    for state in (ZORA_STATE_SUBMERGING, ZORA_STATE_SUBMERGED):
        assert zora_shot_eta(ZeldaObject(**{**zora.__dict__, "state": state})) is None


def test_only_a_zora_has_a_shot_eta() -> None:
    assert zora_shot_eta(_body(OCTOROK_OBJECT_TYPE)) is None
