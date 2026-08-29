"""One spec table per loop; PNG dumps stay opt-in."""

from __future__ import annotations

from pathlib import Path

import pytest
from retro_harness.ram_state import EnemyState, GameMode, GameState
from tmnt_iv.lab.slash_observe import side
from tmnt_iv.observe import HpDelta, living_hp, policy_input
from tmnt_iv.policy import Stage1Policy
from tmnt_iv.run.bridge import BRIDGE_SPECS, is_fight_ready
from tmnt_iv.run.clean_suite import CLEAN_SPECS, is_live_alleycat, is_live_big_apple, is_live_sewer
from tmnt_iv.run.cli import peek_required_int
from tmnt_iv.run.segment import STAGE_SPECS, maybe_save_png
from tmnt_iv.tests._state import playing


def test_stage_specs_cover_1_through_9() -> None:
    assert set(STAGE_SPECS) == set(range(1, 10))
    for number, spec in STAGE_SPECS.items():
        assert spec.number == number
        assert spec.preferred_states[-1] == "NONE"


def test_bridge_specs_cover_dest_2_and_3() -> None:
    assert set(BRIDGE_SPECS) == {2, 3}
    for dest, spec in BRIDGE_SPECS.items():
        assert spec.dest == dest
        assert spec.target_byte == dest - 1


def test_clean_specs_cover_stage_bytes_0_1_2() -> None:
    assert set(CLEAN_SPECS) == {0, 1, 2}
    for byte, spec in CLEAN_SPECS.items():
        assert spec.stage_byte == byte
        assert spec.suite_states
        assert spec.extra_entry
        assert callable(spec.is_live)


def test_live_predicates_and_fight_ready() -> None:
    apple = GameState(frame=1, mode=GameMode.PLAYING, stage=0, health=80, player_x=100, extras={"event": 0x0A})
    alley = GameState(frame=1, mode=GameMode.PLAYING, stage=1, health=80, player_x=100, extras={"event": 0x0A})
    sewer = GameState(frame=1, mode=GameMode.PLAYING, stage=2, health=80, player_x=100, extras={"event": 0x0A})
    assert is_live_big_apple(apple) and not is_live_big_apple(alley)
    assert is_live_alleycat(alley) and not is_live_alleycat(apple)
    assert is_live_sewer(sewer) and not is_live_sewer(alley)

    empty = GameState(frame=1, mode=GameMode.PLAYING, stage=1, health=80, player_x=100)
    assert not is_fight_ready(empty, min_stage=1)
    ready = GameState(
        frame=1,
        mode=GameMode.PLAYING,
        stage=1,
        health=80,
        player_x=100,
        enemies=(EnemyState(slot=0, x=140, y=160, health=16, active=True),),
    )
    assert is_fight_ready(ready, min_stage=1)
    assert not is_fight_ready(ready, min_stage=2)


def test_maybe_save_png_is_noop_when_disabled() -> None:
    bag: list[str] = []
    maybe_save_png(object(), Path("x.png"), enabled=False, bag=bag)
    assert bag == []


def test_living_hp_rejects_zero_and_sentinels() -> None:
    assert living_hp(1) and living_hp(80) and living_hp(0x60)
    assert not living_hp(0) and not living_hp(0x61) and not living_hp(-1)


def test_hp_delta_counts_live_drops_not_ko_unless_asked() -> None:
    meter = HpDelta.start(80)
    assert meter.note(64) == 16
    assert meter.note(64) == 0
    assert meter.note(0) == 0
    assert meter.damage == 16
    assert meter.min_hp == 64

    with_ko = HpDelta.start(16, count_zero=True)
    assert with_ko.note(0) == 16
    assert with_ko.damage == 16


def test_focus_knob_names_prefixes() -> None:
    from tmnt_iv.grind_knobs import KNOB_BOUNDS, focus_knob_names

    shared = ("attack_range", "standoff", "attack_gap")
    slash = focus_knob_names("slash")
    tank = focus_knob_names("technodrome_tank")
    shredder = focus_knob_names("super_shredder")
    assert all(n.startswith("slash_") or n in shared for n in slash)
    assert all(n.startswith("blocker_") or n in shared for n in tank)
    assert tank == focus_knob_names("tokka_rahzar")
    assert all(n.startswith("shredder_") or n in shared for n in shredder)
    assert slash == focus_knob_names("unknown")
    assert all(n in KNOB_BOUNDS for n in (*slash, *tank, *shredder))


def test_policy_input_unwraps_idle_when_tree_succeeds() -> None:
    action, reason = policy_input(
        Stage1Policy(),
        playing(player_x=64, player_y=192, stage=0, health=80),
    )
    assert action is not None
    assert isinstance(reason, str)
    assert reason


def test_slash_side_left_right_overlap() -> None:
    assert side(10, 40) == "left"
    assert side(40, 10) == "right"
    assert side(16, 16) == "overlap"


def test_peek_required_int_returns_value_and_rest() -> None:
    value, rest = peek_required_int(
        "--stage",
        (1, 2, 3),
        ["--suite", "--stage", "2", "--max-frames", "10"],
        description="Clean probes",
        help="Human stage number 1–3 (RAM byte = N-1).",
        epilog="Pass --stage N --help for that stage's flags.",
    )
    assert value == 2
    assert rest == ["--suite", "--max-frames", "10"]


def test_peek_required_int_exits_when_missing() -> None:
    with pytest.raises(SystemExit):
        peek_required_int(
            "--to",
            (2, 3),
            ["--max-frames", "10"],
            description="Bridge previous clear → fight-ready Stage 2 or 3.",
            help="Destination human stage (2=Alleycat, 3=Sewer).",
        )
