"""Alleycat pack tactic: releft, no jumper chase, no 0x76 walk-in."""

from __future__ import annotations

from dataclasses import replace

from retro_harness.ram_state import EnemyState, GameMode, GameState
from tmnt_iv.policy import Stage1Policy
from tmnt_iv.tactics.alleycat import AlleycatPackTactics

_SNES_B = 0
_SNES_Y = 1
_SNES_UP = 4
_SNES_LEFT = 6
_SNES_RIGHT = 7
_SNES_A = 8


def _playing(
    *,
    player_x: int = 80,
    player_y: int = 160,
    enemies: tuple[EnemyState, ...] = (),
    health: int = 80,
    stage: int = 1,
    boss_active: bool = False,
) -> GameState:
    return GameState(
        frame=1,
        mode=GameMode.PLAYING,
        stage=stage,
        player_x=player_x,
        player_y=player_y,
        health=health,
        lives=1,
        enemies=enemies,
        boss_active=boss_active,
        screen_locked=bool(enemies),
    )


def _foot(
    x: int,
    y: int,
    health: int = 16,
    *,
    slot: int = 0,
    kind: int = 0x68,
    animation: int = 0x3B,
) -> EnemyState:
    return EnemyState(
        slot=slot,
        x=x,
        y=y,
        health=health,
        active=True,
        animation=animation,
        kind=kind,
    )


def test_wrong_side_lamp_post_relefts_not_align_up() -> None:
    """Pinned at x≈207 with 0x68 on the left must walk left, not chase Y."""
    state = _playing(
        player_x=207,
        player_y=207,
        enemies=(
            _foot(185, 192, slot=0, animation=0x3B),
            _foot(158, 200, slot=1, animation=0xC5),
        ),
    )
    action = AlleycatPackTactics().next(state)
    assert action is not None
    assert action.reason == "alley_releft"
    assert action.action[_SNES_LEFT] == 1
    assert action.action[_SNES_UP] == 0
    assert action.action[_SNES_B] == 0
    assert action.action[_SNES_A] == 0


def test_sandwiched_jumper_pokes_instead_of_align_up() -> None:
    """Foot both sides + jumper above: plant-poke, do not walk into either."""
    state = _playing(
        player_x=152,
        player_y=207,
        enemies=(
            _foot(104, 197, slot=0, animation=0xC5),
            _foot(219, 213, slot=1, kind=0x62, animation=0xEE),
            _foot(231, 208, slot=2, kind=0x62, animation=0xEE),
        ),
    )
    action = AlleycatPackTactics().next(state)
    assert action is not None
    assert action.reason == "alley_poke"
    assert action.action[_SNES_Y] == 1
    assert action.action[_SNES_UP] == 0
    assert action.action[_SNES_LEFT] == 0
    assert action.action[_SNES_B] == 0


def test_in_range_jumper_above_pokes_not_align_up() -> None:
    """Single 0x68 10px above in slash range: poke from the current lane."""
    state = _playing(
        player_x=150,
        player_y=207,
        enemies=(_foot(180, 197, kind=0x68, animation=0xC5),),
    )
    action = AlleycatPackTactics().next(state)
    assert action is not None
    assert action.reason == "alley_poke"
    assert action.action[_SNES_UP] == 0


def test_five_e_overlap_plants_and_pokes() -> None:
    """Point-blank 0x5E must poke (min_range 0); LEFT-forever is the pack death."""
    state = _playing(
        player_x=199,
        player_y=180,
        enemies=(
            _foot(200, 180, slot=0, kind=0x5E),
            _foot(220, 180, slot=1, kind=0x5E),
        ),
    )
    action = AlleycatPackTactics().next(state)
    assert action is not None
    assert action.reason == "alley_poke"
    assert action.action[_SNES_Y] == 1
    assert action.action[_SNES_LEFT] == 0
    assert action.action[_SNES_B] == 0


def test_cluster_ahead_at_right_holds_and_pokes() -> None:
    """Do not approach_right into a 2+ Foot pile once already at x≥176."""
    state = _playing(
        player_x=180,
        player_y=180,
        enemies=(
            _foot(200, 180, slot=0, kind=0x5E),
            _foot(220, 180, slot=1, kind=0x5E),
        ),
    )
    action = AlleycatPackTactics().next(state)
    assert action is not None
    # 0x5E within 40px on the right is a jump-kick: step left, don't walk in.
    assert action.reason == "alley_releft"
    assert action.action[_SNES_LEFT] == 1
    assert action.action[_SNES_RIGHT] == 0
    assert action.action[_SNES_B] == 0


def test_grabber_ahead_does_not_close() -> None:
    """0x76 inside grab-walk range: plant-poke, never approach_right."""
    state = _playing(
        player_x=126,
        player_y=213,
        enemies=(_foot(155, 218, health=8, kind=0x76, animation=0x96),),
    )
    action = AlleycatPackTactics().next(state)
    assert action is not None
    assert action.reason == "alley_poke"
    assert action.action[_SNES_RIGHT] == 0
    assert action.action[_SNES_B] == 0


def test_grabber_overlap_pokes() -> None:
    state = _playing(
        player_x=160,
        player_y=210,
        enemies=(_foot(165, 212, health=8, kind=0x76),),
    )
    action = AlleycatPackTactics().next(state)
    assert action is not None
    assert action.reason == "alley_poke"
    assert action.action[_SNES_LEFT] == 0
    assert action.action[_SNES_RIGHT] == 0


def test_grabber_wrong_side_spaces_left() -> None:
    state = _playing(
        player_x=180,
        player_y=210,
        enemies=(_foot(160, 212, health=8, kind=0x76),),
    )
    action = AlleycatPackTactics().next(state)
    assert action is not None
    assert action.reason == "alley_grab_space"
    assert action.action[_SNES_LEFT] == 1


def test_established_left_flank_falls_through() -> None:
    """Already on the left poke shoulder: let the shared combat tree run."""
    state = _playing(
        player_x=80,
        player_y=160,
        enemies=(_foot(120, 160, kind=0x5E),),
    )
    assert AlleycatPackTactics().next(state) is None


def test_five_e_right_kicker_walks_left() -> None:
    """24-dmg 0x5E jump-kick from x=226 while planted at 199: step out left."""
    state = _playing(
        player_x=199,
        player_y=188,
        enemies=(
            _foot(84, 186, slot=0, kind=0x5E, animation=0xE8),
            _foot(168, 188, slot=1, kind=0x5E, animation=0x5B),
            _foot(226, 197, slot=2, kind=0x5E, animation=0x86),
        ),
    )
    action = AlleycatPackTactics().next(state)
    assert action is not None
    assert action.reason == "alley_releft"
    assert action.action[_SNES_LEFT] == 1
    assert action.action[_SNES_B] == 0


def test_five_e_sandwich_pokes_instead_of_align_down() -> None:
    """Post-pizza 0x5E sandwich (REACH 24-dmg at x=165) must plant, not Y-chase."""
    state = _playing(
        player_x=165,
        player_y=185,
        enemies=(
            _foot(84, 185, slot=0, kind=0x5E, animation=0x5B),
            _foot(77, 196, slot=1, kind=0x5E, animation=0x86),
            _foot(226, 197, slot=2, kind=0x5E, animation=0x86),
        ),
    )
    action = AlleycatPackTactics().next(state)
    assert action is not None
    assert action.reason == "alley_poke"
    assert action.action[_SNES_UP] == 0
    assert action.action[_SNES_RIGHT] == 0
    assert action.action[_SNES_B] == 0


def test_in_range_wrong_side_pokes_not_releft() -> None:
    """x=160 vs 0x5E at 126/129 is still in poke range — do not walk in."""
    state = _playing(
        player_x=160,
        player_y=188,
        enemies=(
            _foot(126, 194, slot=0, kind=0x5E, animation=0x15),
            _foot(129, 196, slot=1, kind=0x5E, animation=0x15),
        ),
    )
    action = AlleycatPackTactics().next(state)
    assert action is not None
    assert action.reason == "alley_poke"
    assert action.action[_SNES_LEFT] == 0
    assert action.action[_SNES_Y] == 1


def test_five_e_cluster_left_relefts_without_jump() -> None:
    """Out of slash range vs a left 0x5E clump: walk left, never jump."""
    state = _playing(
        player_x=164,
        player_y=188,
        enemies=(
            _foot(76, 188, slot=0, kind=0x5E, animation=0x86),
            _foot(69, 187, slot=1, kind=0x5E, animation=0xE8),
            _foot(96, 188, slot=2, kind=0x5E, animation=0x5B),
        ),
    )
    action = AlleycatPackTactics().next(state)
    assert action is not None
    assert action.reason == "alley_releft"
    assert action.action[_SNES_LEFT] == 1
    assert action.action[_SNES_B] == 0
    assert action.action[_SNES_A] == 0


def test_overlap_beats_farther_left_releft() -> None:
    """Do not walk through a touching 0x5E to chase a farther-left slot."""
    state = _playing(
        player_x=140,
        player_y=188,
        enemies=(
            _foot(145, 195, slot=0, kind=0x5E, animation=0x86),
            _foot(78, 188, slot=1, kind=0x5E, animation=0x5B),
        ),
    )
    action = AlleycatPackTactics().next(state)
    assert action is not None
    assert action.reason == "alley_poke"
    assert action.action[_SNES_LEFT] == 0


def test_metalhead_and_other_stages_idle() -> None:
    tactics = AlleycatPackTactics()
    boss = _playing(
        player_x=160,
        player_y=180,
        enemies=(_foot(180, 180, health=128, kind=0x46),),
        boss_active=True,
    )
    assert tactics.next(boss) is None
    big_apple = replace(
        _playing(enemies=(_foot(120, 160, kind=0x5E),)),
        stage=0,
    )
    assert tactics.next(big_apple) is None


def test_policy_tick_hooks_releft_before_align_up() -> None:
    """Production tick order must not Y-align into the lamp-post pile."""
    state = _playing(
        player_x=207,
        player_y=207,
        enemies=(_foot(185, 192), _foot(158, 200, slot=1)),
    )
    result = Stage1Policy().tick(state)
    assert result.action is not None
    assert result.action.reason == "alley_releft"
    assert result.action.action[_SNES_LEFT] == 1
    assert result.action.action[_SNES_UP] == 0
    assert result.action.action[_SNES_A] == 0
    assert result.action.action[_SNES_B] == 0
