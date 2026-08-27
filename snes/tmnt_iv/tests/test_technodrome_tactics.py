"""Jump-behind opener vs ram-fallback for Technodrome pink Foot."""

from __future__ import annotations

from dataclasses import replace

from retro_harness.ram_state import EnemyState, GameMode, GameState
from tmnt_iv.policy import Stage1Policy
from tmnt_iv.tactics import TechnodromeTactics

_SNES_B = 0
_SNES_Y = 1
_SNES_LEFT = 6
_SNES_RIGHT = 7
_SNES_A = 8


def _playing(
    *,
    player_x: int = 80,
    player_y: int = 160,
    enemies: tuple[EnemyState, ...] = (),
    health: int = 80,
    char_id: int = 8,
    event: int = 0x18,
) -> GameState:
    return GameState(
        frame=1,
        mode=GameMode.PLAYING,
        stage=3,
        player_x=player_x,
        player_y=player_y,
        health=health,
        lives=2,
        enemies=enemies,
        screen_locked=bool(enemies),
        extras={"event": event, "char_id": char_id},
    )


def _foot(
    x: int,
    y: int = 160,
    health: int = 12,
    *,
    slot: int = 0,
    kind: int = 0x6C,
) -> EnemyState:
    return EnemyState(
        slot=slot,
        x=x,
        y=y,
        health=health,
        active=True,
        animation=0x43,
        kind=kind,
    )


def _tank(
    *,
    player_x: int = 80,
    player_y: int = 160,
    enemy_x: int = 110,
    enemy_y: int = 160,
    health: int = 12,
    char_id: int = 8,
) -> GameState:
    return _playing(
        player_x=player_x,
        player_y=player_y,
        enemies=(_foot(enemy_x, enemy_y, health),),
        char_id=char_id,
    )


def test_raph_close_foot_jump_behind_not_retreat() -> None:
    """Raph + adx≤48 + same Y band opens with B+through, not the 40f ram."""
    tactics = TechnodromeTactics()
    action = tactics.next(_tank(player_x=80, enemy_x=110))
    assert action is not None
    assert action.reason == "blocker_jump_behind"
    assert action.action[_SNES_B] == 1
    assert action.action[_SNES_RIGHT] == 1
    assert action.action[_SNES_Y] == 0
    assert action.action[_SNES_A] == 0
    assert tactics._phase == "jb_jump"


def test_raph_close_foot_left_jumps_through_left() -> None:
    """Far side is through the Foot, not away."""
    tactics = TechnodromeTactics()
    action = tactics.next(_tank(player_x=140, enemy_x=100))
    assert action is not None
    assert action.reason == "blocker_jump_behind"
    assert action.action[_SNES_B] == 1
    assert action.action[_SNES_LEFT] == 1


def test_far_foot_uses_ram_retreat() -> None:
    """adx>48 keeps the long retreat → charge ram."""
    tactics = TechnodromeTactics()
    action = tactics.next(_tank(player_x=80, enemy_x=160))
    assert action is not None
    assert action.reason == "blocker_retreat"
    assert action.action[_SNES_LEFT] == 1
    assert action.action[_SNES_B] == 0
    # dx already ≥ retreat_dx, so this first frame also arms the charge.


def test_leo_close_foot_still_rams() -> None:
    """Jump-behind is Raphael-only; Leo keeps production ram."""
    tactics = TechnodromeTactics()
    action = tactics.next(_tank(player_x=80, enemy_x=110, char_id=2))
    assert action is not None
    assert action.reason == "blocker_retreat"
    assert tactics._phase == "retreat"


def test_close_off_y_uses_ram_not_empty_phase_align() -> None:
    """Off-lane close Foot must start ram (retreat/align), not an empty hop wait."""
    tactics = TechnodromeTactics()
    action = tactics.next(_tank(player_x=80, player_y=160, enemy_x=110, enemy_y=200))
    assert action is not None
    assert action.reason in {"technodrome_align", "blocker_retreat"}
    assert tactics._phase == "retreat"
    assert action.reason != "blocker_jump_behind"


def test_corridor_close_foot_still_rams() -> None:
    """Jump-behind is tank-only; Technodrome waves keep the ram."""
    tactics = TechnodromeTactics()
    state = replace(
        _tank(player_x=80, enemy_x=110),
        extras={"event": 0x0A, "char_id": 8},
    )
    action = tactics.next(state)
    assert action is not None
    assert action.reason == "blocker_retreat"
    assert tactics._phase == "retreat"


def test_jump_behind_never_emits_a_or_power_attack() -> None:
    """No A-special and no grounded Y+B across the hop → stun → gap."""
    tactics = TechnodromeTactics()
    state = _tank(player_x=80, enemy_x=110)
    reasons: list[str] = []
    for _ in range(40):
        action = tactics.next(state)
        assert action is not None
        reasons.append(action.reason)
        held = action.action
        assert held[_SNES_A] == 0
        assert not (held[_SNES_B] and held[_SNES_Y])
        if action.reason == "blocker_retreat":
            break
    assert "blocker_jump_behind" in reasons
    assert "blocker_behind_stun" in reasons
    assert "blocker_behind_gap" in reasons


def test_jump_behind_miss_falls_back_to_ram() -> None:
    """No HP drop within the hop budget → retreat/charge ram."""
    tactics = TechnodromeTactics()
    state = _tank(player_x=80, enemy_x=110, health=12)
    reasons: list[str] = []
    for _ in range(80):
        action = tactics.next(state)
        assert action is not None
        reasons.append(action.reason)
        if action.reason == "blocker_retreat":
            break
    assert reasons[0] == "blocker_jump_behind"
    assert "blocker_retreat" in reasons
    assert tactics._phase == "retreat"
    assert tactics._force_ram is True
    # Next opener on the same Foot must stay on ram, not re-hop.
    again = tactics.next(state)
    assert again is not None
    assert again.reason == "blocker_retreat"


def test_jump_behind_stun_then_screen_throw() -> None:
    """HP drop after behind-Y hands off to the existing toward+Y throw."""
    tactics = TechnodromeTactics()
    state = _tank(player_x=80, enemy_x=110, health=12)
    stunned = False
    action = None
    for _ in range(40):
        if tactics._phase == "jb_stun_gap" and not stunned:
            state = _tank(player_x=122, enemy_x=110, health=8)
            stunned = True
        action = tactics.next(state)
        assert action is not None
        if action.reason == "screen_throw":
            break
    assert stunned
    assert action is not None
    assert action.reason in {"screen_throw", "blocker_grab_close"}
    assert tactics._phase in {"grab", "grab_hold"}
    assert action.action[_SNES_A] == 0
    assert not (action.action[_SNES_B] and action.action[_SNES_Y])


def test_charge_never_mixes_jump_or_vertical() -> None:
    """Once the ram charge starts, shrinking dx must not hop or tap UP/DOWN."""
    tactics = TechnodromeTactics()
    far = _tank(player_x=80, enemy_x=160)
    action = None
    for _ in range(45):
        action = tactics.next(far)
        assert action is not None
        if action.reason == "blocker_charge":
            break
    assert action is not None
    assert action.reason == "blocker_charge"
    assert tactics._phase == "charge"
    close = _tank(player_x=80, enemy_x=96)
    for _ in range(10):
        action = tactics.next(close)
        assert action is not None
        assert action.reason == "blocker_charge"
        assert action.action[_SNES_B] == 0
        assert action.action[_SNES_Y] == 0
        assert action.action[4] == 0  # UP
        assert action.action[5] == 0  # DOWN
        assert tactics._phase == "charge"


def test_slot_reuse_drops_stale_grab_into_jump_behind() -> None:
    """Fresh full-HP Foot in the same slot starts a new Raph hop, not a throw."""
    tactics = TechnodromeTactics()
    tactics._in_tank = True
    tactics._target_slot = 0
    tactics._target_kind = 0x6C
    tactics._target_health = 4
    tactics._phase = "grab_gap"
    tactics._timer = 8
    action = tactics.next(_tank(player_x=80, enemy_x=100, health=12))
    assert action is not None
    assert action.reason == "blocker_jump_behind"
    assert tactics._phase == "jb_jump"
    assert tactics._force_ram is False


def test_stage1_policy_raph_tank_uses_jump_behind() -> None:
    """Stage1Policy tick order still owns Technodrome via the extracted class."""
    result = Stage1Policy().tick(_tank(player_x=80, enemy_x=108))
    assert result.action is not None
    assert result.action.reason == "blocker_jump_behind"
    assert result.action.action[_SNES_A] == 0
