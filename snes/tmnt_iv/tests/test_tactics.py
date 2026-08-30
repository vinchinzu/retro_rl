"""Production-tick traps that keep the run finishing with less damage."""

from __future__ import annotations

from dataclasses import replace

from retro_harness.combat import PreferredFlank
from retro_harness.ram_state import EnemyState
from tmnt_iv.grind_knobs import GrindKnobs
from tmnt_iv.policy import Stage1Policy
from tmnt_iv.tactics.fight import CombatProfile
from tmnt_iv.tactics.raph_air import raph_starbase_jump_action
from tmnt_iv.tactics.slash import SlashTactics
from tmnt_iv.tactics.technodrome import TechnodromeTactics
from tmnt_iv.tests._state import A, B, Y, enemy, playing


def _slash_state(
    *,
    player_x: int,
    slash_x: int,
    animation: int = 0x43,
    player_y: int = 160,
    slash_y: int = 160,
) -> object:
    slash = enemy(slash_x, slash_y, 160, kind=0x50, animation=animation)
    return replace(
        playing(player_x=player_x, player_y=player_y, enemies=(slash,), stage=4, boss_active=True),
        extras={"event": 0x0A, "iframes": 0, "boss_hp": 160, "char_id": 8},
    )


def _poke(profile: CombatProfile) -> tuple[int, int, int, int, int, int, PreferredFlank]:
    return (
        profile.y_tolerance,
        profile.attack_range,
        profile.min_range,
        profile.standoff,
        profile.hold_frames,
        profile.gap_frames,
        profile.flank,
    )


def test_combat_profile_table_pins_stage_rows() -> None:
    foot = enemy(120, 160)
    wave0 = CombatProfile.from_state(playing(stage=0, enemies=(foot,)))
    assert _poke(wave0) == (8, 64, 8, 32, 2, 2, PreferredFlank.NONE)

    alley = CombatProfile.from_state(playing(stage=1, enemies=(foot,)))
    assert _poke(alley) == (6, 65, 0, 36, 1, 2, PreferredFlank.LEFT)

    rat = CombatProfile.from_state(
        playing(stage=2, boss_active=True, enemies=(enemy(120, 200, kind=0x4A),))
    )
    assert _poke(rat) == (32, 120, 8, 24, 2, 1, PreferredFlank.NONE)

    neon = CombatProfile.from_state(playing(stage=7, enemies=(foot,)))
    assert _poke(neon) == (48, 68, 8, 24, 2, 5, PreferredFlank.NONE)

    shredder = CombatProfile.from_state(
        playing(stage=8, boss_active=True, enemies=(enemy(120, 160, kind=0x52),))
    )
    assert _poke(shredder) == (8, 72, 10, 28, 2, 1, PreferredFlank.NONE)


def test_slash_spin_dodge_adx_stays_52_and_never_presses_a() -> None:
    assert GrindKnobs().slash_spin_dodge_adx == 52
    tactics = SlashTactics()
    close = tactics.next(_slash_state(player_x=80, slash_x=120, animation=0xEE))
    assert close is not None
    assert close.reason == "slash_dodge"
    assert close.action[A] == 0

    far = SlashTactics().next(_slash_state(player_x=60, slash_x=120, animation=0xEE))
    assert far is not None
    assert far.reason != "slash_dodge"

    policy = Stage1Policy().tick(
        _slash_state(player_x=100, slash_x=140, player_y=170, slash_y=120)
    )
    assert policy.action is not None
    assert policy.action.reason.startswith("slash_")
    assert policy.action.reason != "jump_slash"
    assert policy.action.action[A] == 0


def test_alleycat_relefts_instead_of_aligning_into_the_pack() -> None:
    state = playing(
        player_x=207,
        player_y=207,
        stage=1,
        enemies=(enemy(185, 192, kind=0x68), enemy(158, 200, slot=1, kind=0x68)),
    )
    action = Stage1Policy().tick(state).action
    assert action is not None
    assert action.reason == "alley_releft"
    assert action.action[6] == 1
    assert action.action[A] == 0


def test_alleycat_right_exits_left_5e_clump_instead_of_releft() -> None:
    state = playing(
        player_x=164,
        player_y=180,
        stage=1,
        enemies=(
            enemy(78, 180, kind=0x5E),
            enemy(96, 180, slot=1, kind=0x5E),
        ),
    )
    action = Stage1Policy().tick(state).action
    assert action is not None
    assert action.reason == "alley_right_exit"
    assert action.action[7] == 1
    assert action.action[6] == 0
    assert action.action[B] == 0
    assert action.action[A] == 0


def test_alleycat_right_kicker_still_relefts() -> None:
    state = playing(
        player_x=199,
        player_y=180,
        stage=1,
        enemies=(enemy(226, 180, kind=0x5E),),
    )
    action = Stage1Policy().tick(state).action
    assert action is not None
    assert action.reason == "alley_releft"
    assert action.action[6] == 1
    assert action.action[A] == 0


def test_alleycat_overlap_5e_still_plants() -> None:
    state = playing(
        player_x=164,
        player_y=180,
        stage=1,
        enemies=(enemy(164, 180, kind=0x5E),),
    )
    action = Stage1Policy().tick(state).action
    assert action is not None
    assert action.reason == "alley_poke"
    assert action.action[Y] == 1
    assert action.action[A] == 0


def test_alleycat_60_wrong_side_still_relefts() -> None:
    state = playing(
        player_x=113,
        player_y=180,
        stage=1,
        enemies=(enemy(80, 180, kind=0x60),),
    )
    action = Stage1Policy().tick(state).action
    assert action is not None
    assert action.reason == "alley_releft"
    assert action.action[6] == 1
    assert action.action[A] == 0


def test_duo_wall_escape_from_right_door() -> None:
    state = playing(
        player_x=224,
        player_y=192,
        stage=3,
        enemies=(enemy(32, 176, 96, kind=0x48), enemy(181, 177, 96, slot=1, kind=0xA0)),
        boss_active=True,
        extras={"event": 0x0A},
    )
    action = Stage1Policy().tick(state).action
    assert action is not None
    assert action.reason == "duo_wall_escape"
    assert action.action[A] == 0
    assert action.action[B] == 1
    assert action.action[6] == 1


def test_technodrome_jump_behind_never_presses_a_or_grounded_by() -> None:
    foot = enemy(110, 160, 12, kind=0x6C)
    state = playing(
        player_x=80,
        player_y=160,
        stage=3,
        enemies=(foot,),
        extras={"event": 0x18, "char_id": 8},
    )
    tactics = TechnodromeTactics()
    reasons: list[str] = []
    for _ in range(40):
        action = tactics.next(state)
        assert action is not None
        reasons.append(action.reason)
        assert action.action[A] == 0
        assert not (action.action[B] and action.action[Y])
        if action.reason == "blocker_retreat":
            break
    assert "blocker_jump_behind" in reasons


def test_form2_dispatch_never_presses_a() -> None:
    boss = EnemyState(slot=0, x=160, y=180, health=190, active=True, kind=0xAE, animation=0xBB)
    state = playing(
        player_x=160,
        player_y=180,
        stage=9,
        enemies=(boss,),
        boss_active=True,
        extras={"event": 0x0A, "iframes": 0, "char_id": 8},
    )
    action = Stage1Policy().tick(state).action
    assert action is not None
    assert action.reason.startswith("shredder_")
    assert action.action[A] == 0


def test_raph_starbase_close_gap_period_is_at_least_4() -> None:
    stack = enemy(40, 190, kind=0xB0)

    def air(frame: int):
        return raph_starbase_jump_action(
            playing(
                player_x=73,
                player_y=176,
                stage=8,
                frame=frame,
                enemies=(stack,),
                extras={"char_id": 8},
            )
        )

    jump = air(0)
    assert jump is not None
    assert jump.reason == "raph_starbase_jump"
    assert jump.action[A] == 0
    for frame in (1, 2, 3):
        gap = air(frame)
        assert gap is not None
        assert gap.reason == "raph_starbase_close_gap"
        assert gap.action[A] == 0
        assert not (gap.action[B] and gap.action[Y])
