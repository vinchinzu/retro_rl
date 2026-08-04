"""Unit tests for the pure shot planner / search entry point."""

from __future__ import annotations

from hals_golf.tasks.menus import ClubSet, Difficulty, PlayMode
from hals_golf.tasks.profile import MissionProfile
from hals_golf.tasks.shot_policy import (
    DeterministicRoutePolicy,
    HoleInOneSearchPolicy,
    SearchSpec,
    ShotSituation,
    candidate_intents,
    plan_putt,
    plan_shot,
)


def test_candidate_intents_starts_as_deterministic_singleton() -> None:
    profile = MissionProfile(
        play_mode=PlayMode.VS_HAL,
        club_set=ClubSet.METAL,
    )
    situation = ShotSituation(hole=6, strokes=2, rest=32, lie=2)
    intents = candidate_intents(situation, profile)
    assert len(intents) == 1
    assert intents[0] == plan_shot(situation, profile)


def test_metal_h6_chip_in_rest_band() -> None:
    profile = MissionProfile(
        play_mode=PlayMode.VS_HAL,
        club_set=ClubSet.METAL,
    )
    intent = plan_shot(
        ShotSituation(hole=6, strokes=2, rest=32, lie=2),
        profile,
    )
    assert intent.club_downs == 0
    assert intent.power == 28
    assert intent.aim == 4


def test_stroke_play_metal_hole_one_uses_pw_approach() -> None:
    profile = MissionProfile(
        play_mode=PlayMode.STROKE_PLAY,
        club_set=ClubSet.METAL,
        difficulty=Difficulty.AMATEUR,
    )
    approach = plan_shot(
        ShotSituation(hole=1, strokes=1, rest=154, lie=2),
        profile,
    )
    assert approach.power == 42
    assert approach.aim == -2
    assert approach.club_downs == 11
    putt = plan_putt(
        ShotSituation(hole=1, strokes=2, rest=3, lie=6),
        profile,
    )
    assert putt.power == 41


def test_stroke_play_metal_hole_two_uses_sw_approach() -> None:
    profile = MissionProfile(
        play_mode=PlayMode.STROKE_PLAY,
        club_set=ClubSet.METAL,
        difficulty=Difficulty.AMATEUR,
    )
    approach = plan_shot(
        ShotSituation(hole=2, strokes=1, rest=96, lie=2),
        profile,
    )
    assert approach.power == 36
    assert approach.aim == 0
    assert approach.club_downs == 12


def test_stroke_play_metal_worst_holes_use_ported_routes() -> None:
    """H7/H9/H10 were catastrophic on the Jul-20 partial; lock the ports."""
    profile = MissionProfile(
        play_mode=PlayMode.STROKE_PLAY,
        club_set=ClubSet.METAL,
        difficulty=Difficulty.AMATEUR,
    )
    h9_tee = plan_shot(
        ShotSituation(hole=9, strokes=0, rest=145, lie=1),
        profile,
    )
    assert h9_tee.power == 38
    assert h9_tee.aim == 0
    assert h9_tee.club_downs == 11
    h9_putt = plan_putt(
        ShotSituation(hole=9, strokes=1, rest=18, lie=6),
        profile,
    )
    assert h9_putt.power == 20
    h10_second = plan_shot(
        ShotSituation(hole=10, strokes=1, rest=275, lie=2),
        profile,
    )
    assert h10_second.power == 38
    assert h10_second.aim == -6
    assert h10_second.club_downs == 4
    h7_tee = plan_shot(
        ShotSituation(hole=7, strokes=0, rest=520, lie=1),
        profile,
    )
    assert h7_tee.power == 44
    assert h7_tee.aim == -8
    assert h7_tee.club_downs == 0
    h7_second = plan_shot(
        ShotSituation(hole=7, strokes=1, rest=254, lie=2),
        profile,
    )
    assert h7_second.power == 44
    assert h7_second.aim == -4
    assert h7_second.club_downs == 1
    h7_chip = plan_shot(
        ShotSituation(hole=7, strokes=3, rest=23, lie=2),
        profile,
    )
    assert h7_chip.power == 26
    assert h7_chip.aim == -4
    assert h7_chip.club_downs == 0
    h3_finish = plan_shot(
        ShotSituation(hole=3, strokes=2, rest=62, lie=0),
        profile,
    )
    assert h3_finish.power == 38
    assert h3_finish.aim == -2
    assert h3_finish.club_downs == 0


def test_stroke_play_metal_rest_bands_apply_outside_vs_hal() -> None:
    profile = MissionProfile(
        play_mode=PlayMode.STROKE_PLAY,
        club_set=ClubSet.METAL,
        difficulty=Difficulty.AMATEUR,
    )
    h8 = plan_shot(
        ShotSituation(hole=8, strokes=1, rest=104, lie=2),
        profile,
    )
    assert h8.club_downs == 9
    assert h8.power == 38
    assert h8.aim == 0
    h8_tee = plan_shot(
        ShotSituation(hole=8, strokes=0, rest=339, lie=1),
        profile,
    )
    assert h8_tee.power == 44
    assert h8_tee.aim == -8
    h7 = plan_shot(
        ShotSituation(hole=7, strokes=1, rest=205, lie=2),
        profile,
    )
    assert h7.club_downs == 3
    assert h7.power == 44
    assert h7.aim == -4


def test_stroke_play_metal_back_nine_uses_calibrated_tees() -> None:
    profile = MissionProfile(
        play_mode=PlayMode.STROKE_PLAY,
        club_set=ClubSet.METAL,
        difficulty=Difficulty.AMATEUR,
    )
    h12 = plan_shot(ShotSituation(hole=12, strokes=0, rest=408, lie=1), profile)
    assert h12.power == 44 and h12.aim == -4 and h12.club_downs == 0
    h13 = plan_shot(ShotSituation(hole=13, strokes=0, rest=178, lie=1), profile)
    assert h13.power == 38 and h13.aim == -2 and h13.club_downs == 9
    h15 = plan_shot(ShotSituation(hole=15, strokes=0, rest=432, lie=1), profile)
    assert h15.power == 44 and h15.aim == -8
    h16 = plan_shot(ShotSituation(hole=16, strokes=0, rest=528, lie=1), profile)
    assert h16.power == 42 and h16.aim == -5
    h17 = plan_shot(ShotSituation(hole=17, strokes=0, rest=152, lie=1), profile)
    assert h17.power == 34 and h17.aim == -4 and h17.club_downs == 8
    h17_putt = plan_putt(
        ShotSituation(hole=17, strokes=1, rest=13, lie=6),
        profile,
    )
    assert h17_putt.power == 20
    h18 = plan_shot(ShotSituation(hole=18, strokes=0, rest=416, lie=1), profile)
    assert h18.power == 42 and h18.aim == -5 and h18.club_downs == 0
    h18_app = plan_shot(
        ShotSituation(hole=18, strokes=1, rest=169, lie=2),
        profile,
    )
    assert h18_app.power == 44 and h18_app.aim == -5 and h18_app.club_downs == 0


def test_pro_overlay_is_a_noop_until_calibrated() -> None:
    amateur = MissionProfile()
    pro = MissionProfile(difficulty=Difficulty.PRO)
    for situation in (
        ShotSituation(hole=3, strokes=0, rest=505, lie=1),
        ShotSituation(hole=10, strokes=2, rest=114, lie=2),
    ):
        assert plan_shot(situation, pro) == plan_shot(situation, amateur)
    putt = ShotSituation(hole=5, strokes=2, rest=8, lie=6)
    assert plan_putt(putt, pro) == plan_putt(putt, amateur)


def test_deterministic_route_policy_matches_plan_functions() -> None:
    policy = DeterministicRoutePolicy()
    profile = MissionProfile()
    shot_situation = ShotSituation(hole=3, strokes=0, rest=505, lie=1)
    putt_situation = ShotSituation(hole=3, strokes=4, rest=8, lie=6)

    assert policy.plan_shot(shot_situation, profile) == plan_shot(
        shot_situation, profile
    )
    assert policy.plan_putt(putt_situation, profile) == plan_putt(
        putt_situation, profile
    )
    candidates = policy.candidates(shot_situation, profile)
    assert len(candidates) == 1
    assert candidates[0] == plan_shot(shot_situation, profile)


def test_hio_search_base_intent_leads_and_matches_deterministic() -> None:
    policy = HoleInOneSearchPolicy()
    profile = MissionProfile()
    situation = ShotSituation(hole=7, strokes=0, rest=516, lie=1)

    candidates = policy.candidates(situation, profile)

    assert candidates[0] == plan_shot(situation, profile)
    assert len(candidates) > 1


def test_hio_search_expands_a_deduplicated_neighborhood() -> None:
    policy = HoleInOneSearchPolicy()
    profile = MissionProfile()
    situation = ShotSituation(hole=7, strokes=0, rest=516, lie=1)

    candidates = policy.candidates(situation, profile)

    keys = [(c.power, c.aim, c.club_downs) for c in candidates]
    assert len(keys) == len(set(keys))
    assert len(candidates) <= policy.spec.max_candidates


def test_hio_search_respects_max_candidate_cap() -> None:
    spec = SearchSpec(
        power_deltas=(0, -1, 1, -2, 2, -3, 3),
        aim_deltas=(0, -2, 2, -4, 4, -6, 6),
        max_candidates=6,
    )
    policy = HoleInOneSearchPolicy(spec=spec)
    profile = MissionProfile()
    situation = ShotSituation(hole=7, strokes=0, rest=516, lie=1)

    candidates = policy.candidates(situation, profile)

    assert len(candidates) == 6
    assert candidates[0] == plan_shot(situation, profile)


def test_hio_search_only_expands_the_opening_tee_shot() -> None:
    policy = HoleInOneSearchPolicy()
    profile = MissionProfile()

    fairway = ShotSituation(hole=7, strokes=1, rest=200, lie=2)
    assert len(policy.candidates(fairway, profile)) == 1

    non_tee_lie = ShotSituation(hole=7, strokes=0, rest=200, lie=3)
    assert len(policy.candidates(non_tee_lie, profile)) == 1

    tee = ShotSituation(hole=7, strokes=0, rest=516, lie=1)
    assert len(policy.candidates(tee, profile)) > 1


def test_candidate_intents_hio_flag_expands_tee_shot() -> None:
    profile = MissionProfile()
    situation = ShotSituation(hole=7, strokes=0, rest=516, lie=1)

    default = candidate_intents(situation, profile)
    searched = candidate_intents(situation, profile, hole_in_one_search=True)

    assert len(default) == 1
    assert len(searched) > 1
    assert searched[0] == default[0]
