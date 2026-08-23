"""RAM-gated boot controller tests (no ROM)."""

from __future__ import annotations

from mortal_kombat.boot import BootController, Phase
from mortal_kombat.ram import parse_ram, make_test_ram


def test_logos_are_idle_then_start() -> None:
    ctrl = BootController(start_after=100)
    snap = parse_ram(
        make_test_ram(p1_health=0, p2_health=0, timer=0, p1_character=255, match_counter=0)
    )
    phase, buttons = ctrl.decide(snap, frame=10)
    assert phase is Phase.LOGOS
    assert buttons == ()
    phase, buttons = ctrl.decide(snap, frame=120)
    assert phase is Phase.LOGOS
    assert "START" in buttons


def test_char_select_moves_right_until_liukang() -> None:
    ctrl = BootController(start_after=10)
    cage = parse_ram(make_test_ram(timer=0, p1_character=0, match_counter=0))
    phase, buttons = ctrl.decide(cage, frame=50)
    assert phase is Phase.CHAR_SELECT
    assert "DOWN" in buttons
    raiden = parse_ram(make_test_ram(timer=0, p1_character=2, match_counter=0))
    phase, buttons = ctrl.decide(raiden, frame=50)
    assert "RIGHT" in buttons
    liu = parse_ram(make_test_ram(timer=0, p1_character=3, match_counter=0))
    phase, buttons = ctrl.decide(liu, frame=96)
    assert phase is Phase.CHAR_SELECT
    assert "Y" in buttons
    assert "A" in buttons
    subzero = parse_ram(make_test_ram(timer=0, p1_character=5, match_counter=0))
    phase, buttons = ctrl.decide(subzero, frame=50)
    assert "LEFT" in buttons


def test_live_fight_does_not_mash_start() -> None:
    ctrl = BootController()
    snap = parse_ram(make_test_ram(p1_health=120, p2_health=90, timer=70))
    phase, buttons = ctrl.decide(snap, frame=5000)
    assert phase is Phase.FIGHT
    assert buttons == ()


def test_between_rounds_does_not_mash_start() -> None:
    ctrl = BootController()
    ko = parse_ram(
        make_test_ram(p1_health=80, p2_health=0, timer=0, p1_rounds=1, match_counter=0)
    )
    phase, buttons = ctrl.decide(ko, frame=4000)
    assert phase is Phase.FIGHT
    assert buttons == ()


def test_vs_menu_pulses_start() -> None:
    ctrl = BootController()
    vs = parse_ram(
        make_test_ram(
            p1_health=0, p2_health=0, timer=0, match_counter=1, p1_rounds=2
        )
    )
    phase, buttons = ctrl.decide(vs, frame=20)
    assert phase is Phase.VS
    assert "START" in buttons
