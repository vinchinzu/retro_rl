"""Unit tests for MK1 RAM snapshot and hitbox geometry (no ROM)."""

from __future__ import annotations

from mortal_kombat.ram import (
    LIU_KANG_ID,
    MAX_HEALTH,
    PUNCH_RANGE,
    Screen,
    V3_DIM,
    is_char_select,
    is_fight_ready,
    is_match_lost,
    is_match_won,
    make_test_ram,
    parse_ram,
    rounds_settled,
    snapshot_features,
)


def test_fight_ready_liukang() -> None:
    snap = parse_ram(make_test_ram())
    assert snap.p1_character == LIU_KANG_ID
    assert snap.screen is Screen.FIGHT
    assert is_fight_ready(snap)


def test_continue_screen() -> None:
    snap = parse_ram(make_test_ram(p1_health=0, p2_health=0, timer=0, continue_timer=9))
    assert snap.screen is Screen.CONTINUE


def test_char_select_vs_between_fights() -> None:
    select = parse_ram(
        make_test_ram(p1_health=0, p2_health=0, timer=0, match_counter=0, p1_character=0)
    )
    assert select.screen is Screen.CHAR_SELECT
    # Health leftover 0xA1 on the choose-fighter screen is still select.
    select_hp = parse_ram(make_test_ram(timer=0, match_counter=0, p1_character=0))
    assert select_hp.screen is Screen.CHAR_SELECT
    assert is_char_select(select_hp)
    assert not is_char_select(parse_ram(make_test_ram()))
    vs = parse_ram(
        make_test_ram(p1_health=0, p2_health=0, timer=0, match_counter=2, p1_character=3)
    )
    assert vs.screen is Screen.MENU


def test_bodies_overlap_and_v3_vector() -> None:
    far = parse_ram(make_test_ram(p1_x=40, p2_x=200))
    close = parse_ram(make_test_ram(p1_x=100, p2_x=110))
    assert not far.bodies_overlap
    assert close.bodies_overlap
    assert close.distance_x <= PUNCH_RANGE
    vector, prev = snapshot_features(close, (MAX_HEALTH, MAX_HEALTH))
    assert vector.shape == (V3_DIM,)
    assert prev == (MAX_HEALTH, MAX_HEALTH)
    assert vector[17] == 1.0  # in_range
    assert vector[18] == 1.0  # overlap


def test_facing_points_toward_opponent() -> None:
    snap = parse_ram(make_test_ram(p1_x=80, p2_x=180))
    assert snap.p1.facing == 1
    assert snap.p2.facing == -1


def test_match1_between_rounds_is_not_char_select() -> None:
    """Match 1 keeps match_counter=0; timer-down + leftover health is KO, not select."""
    ko = parse_ram(
        make_test_ram(
            p1_health=80,
            p2_health=0,
            timer=0,
            match_counter=0,
            p1_rounds=1,
            p1_character=LIU_KANG_ID,
        )
    )
    assert ko.screen is Screen.BETWEEN_ROUNDS
    assert not is_char_select(ko)
    intro = parse_ram(
        make_test_ram(
            timer=0,
            match_counter=0,
            p1_rounds=1,
            p1_character=LIU_KANG_ID,
        )
    )
    assert intro.screen is Screen.BETWEEN_ROUNDS
    assert not is_char_select(intro)


def test_timeout_ko_is_not_char_select() -> None:
    snap = parse_ram(
        make_test_ram(
            p1_health=MAX_HEALTH,
            p2_health=1,
            timer=0,
            match_counter=0,
            p1_rounds=0,
            p2_rounds=0,
        )
    )
    assert snap.screen is Screen.BETWEEN_ROUNDS
    assert not is_char_select(snap)


def test_garbage_p2_rounds_clamped() -> None:
    spike = parse_ram(make_test_ram(p2_rounds=5, timer=1, p2_health=1))
    assert spike.p2_rounds == 0
    leftover = parse_ram(
        make_test_ram(p1_rounds=0, p2_rounds=2, timer=153, p1_health=MAX_HEALTH, p2_health=MAX_HEALTH)
    )
    assert leftover.p2_rounds == 0
    vs = parse_ram(
        make_test_ram(
            p1_health=0, p2_health=0, timer=0, match_counter=1, p1_rounds=0, p2_rounds=2
        )
    )
    assert vs.screen is Screen.MENU
    assert not rounds_settled(vs)


def test_match_won_needs_majority() -> None:
    win = parse_ram(make_test_ram(p1_rounds=2, p2_rounds=0))
    split = parse_ram(make_test_ram(p1_rounds=2, p2_rounds=2))
    loss = parse_ram(make_test_ram(p1_rounds=1, p2_rounds=2))
    assert is_match_won(win)
    assert not is_match_won(split)
    assert is_match_lost(loss)
    assert not is_match_won(loss)
