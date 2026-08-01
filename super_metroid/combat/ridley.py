"""Development-only full-knowledge Ridley fight strategy.

This is a bounded strategy scaffold for the Lower Norfair Ridley room. It is
explicitly developmentOnly until natural Ridley entry exists on the
continuous chain; it does not warp, place Samus, or write boss/event RAM.
"""

from __future__ import annotations

from dataclasses import dataclass

from super_metroid.combat.features import (
    boss_defeated_in_state,
    features_from_state,
    ridley_catalog,
)
from super_metroid.combat.primitives import ensure_weapon, range_kite_action
from super_metroid.ram import SuperMetroidState
from super_metroid.routes.runtime import ControllerSession, hold

ROOM_RIDLEY = 0xB32E
WEAPON_MISSILES = 1
WEAPON_SUPERS = 2


@dataclass(frozen=True)
class RidleyStrategy:
    """Tunable Super/Missile spray and range-kite parameters."""

    min_range: int = 100
    max_range: int = 220
    jump_range: int = 170
    jump_hold_frames: int = 16
    jump_period: int = 48
    fire_period: int = 3
    max_fight_frames: int = 30_000
    weapon: int = WEAPON_SUPERS


@dataclass(frozen=True)
class RidleyEvidence:
    """Measured result of one development-only Ridley fight attempt."""

    start_frame: int
    body_zero_frame: int | None
    boss_bit_frame: int | None
    end_frame: int
    peak_body_hp: int
    min_body_hp: int
    action_frames: int
    final_body_hp: int
    boss_bit_set: bool
    outcome: str

    def to_dict(self) -> dict[str, object]:
        return {
            "start_frame": self.start_frame,
            "body_zero_frame": self.body_zero_frame,
            "boss_bit_frame": self.boss_bit_frame,
            "end_frame": self.end_frame,
            "peak_body_hp": self.peak_body_hp,
            "min_body_hp": self.min_body_hp,
            "action_frames": self.action_frames,
            "final_body_hp": self.final_body_hp,
            "boss_bit_set": self.boss_bit_set,
            "outcome": self.outcome,
        }


def fight_ridley_action(
    state: SuperMetroidState,
    frame_index: int,
    strategy: RidleyStrategy = RidleyStrategy(),
) -> tuple[str, ...]:
    """Return one pure, RAM-driven Super/Missile spray action for Ridley.

    The selected weapon is controlled by the caller. ``X`` therefore does
    not grant ammo or change inventory state.
    """
    catalog = ridley_catalog()
    features = features_from_state(state, catalog)
    if features.enemy_defeated or state.enemy0_hp == 0:
        return ()
    return range_kite_action(
        state.samus_x,
        features.enemy_x,
        min_range=strategy.min_range,
        max_range=strategy.max_range,
        jump_range=strategy.jump_range,
        frame_index=frame_index,
        jump_period=strategy.jump_period,
        jump_hold_frames=strategy.jump_hold_frames,
        fire_period=strategy.fire_period,
        fire_button="X",
    )


def play_ridley_fight(
    session: ControllerSession,
    *,
    strategy: RidleyStrategy = RidleyStrategy(),
) -> RidleyEvidence:
    """Fight Ridley until HP zero, the boss bit, or a bounded timeout.

    The session must already be in room ``0xB32E``. This is developmentOnly
    evidence until the room is reached naturally from the continuous chain.
    """
    catalog = ridley_catalog()
    start = session.frame
    if session.state.room_id != ROOM_RIDLEY:
        raise RuntimeError(
            f"Ridley fight expected room 0x{ROOM_RIDLEY:04X}, "
            f"got 0x{session.state.room_id:04X}"
        )

    ensure_weapon(session, strategy.weapon)
    initial_hp = session.state.enemy0_hp
    peak_hp = initial_hp
    min_hp = initial_hp
    body_zero_frame: int | None = None
    boss_bit_frame: int | None = None
    prev_hp = initial_hp
    if initial_hp == 0:
        body_zero_frame = start

    for index in range(strategy.max_fight_frames):
        state = session.state
        if boss_defeated_in_state(state, catalog):
            boss_bit_frame = session.frame
            break
        if body_zero_frame is not None:
            break

        names = fight_ridley_action(state, index, strategy)
        if names:
            hold(session, 1, *names, reason="fight_ridley")
        else:
            hold(session, 1, reason="fight_ridley_idle")

        post = session.state
        hp = post.enemy0_hp
        if 0 <= hp <= catalog.max_hp:
            peak_hp = max(peak_hp, hp)
            min_hp = min(min_hp, hp)
        if body_zero_frame is None and hp == 0 and prev_hp > 0:
            body_zero_frame = session.frame
            min_hp = 0
        if boss_bit_frame is None and boss_defeated_in_state(post, catalog):
            boss_bit_frame = session.frame
        if boss_bit_frame is not None or body_zero_frame is not None:
            break
        prev_hp = hp

    final_hp = session.state.enemy0_hp
    boss_set = boss_defeated_in_state(session.state, catalog)
    if boss_set:
        outcome = "ridley_defeated"
    elif body_zero_frame is not None:
        outcome = "ridley_body_zero_no_boss_bit"
    else:
        outcome = "timeout"

    return RidleyEvidence(
        start_frame=start,
        body_zero_frame=body_zero_frame,
        boss_bit_frame=boss_bit_frame,
        end_frame=session.frame,
        peak_body_hp=peak_hp,
        min_body_hp=min_hp,
        action_frames=session.frame - start,
        final_body_hp=final_hp,
        boss_bit_set=boss_set,
        outcome=outcome,
    )
