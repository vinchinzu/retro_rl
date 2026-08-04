"""Development-only Crocomire acid-push strategy scaffold.

Crocomire is a side-path boss and is explicitly not part of the KPDR
continuous spine. The fight is won by pushing the boss into the acid wall;
the catalog HP field is therefore not a defeat condition. This module does
not warp, place Samus, or write boss/event RAM.
"""

from __future__ import annotations

from dataclasses import dataclass

from super_metroid.combat.features import (
    boss_defeated_in_state,
    crocomire_catalog,
)
from super_metroid.ram import SuperMetroidState
from super_metroid.routes.runtime import ControllerSession, hold

ROOM_CROCOMIRE = 0xA98D
WEAPON_MISSILES = 1
WEAPON_SUPERS = 2


@dataclass(frozen=True)
class CrocomireStrategy:
    """Tunable acid-facing push and periodic-fire parameters."""

    push_direction: str = "RIGHT"
    fire_period: int = 3
    fire_hold_frames: int = 1
    max_fight_frames: int = 12_000
    weapon: int = WEAPON_MISSILES


@dataclass(frozen=True)
class CrocomireEvidence:
    """Measured result of one development-only acid-push attempt."""

    start_frame: int
    boss_bit_frame: int | None
    end_frame: int
    action_frames: int
    final_enemy_hp: int
    boss_bit_set: bool
    outcome: str

    def to_dict(self) -> dict[str, object]:
        return {
            "start_frame": self.start_frame,
            "boss_bit_frame": self.boss_bit_frame,
            "end_frame": self.end_frame,
            "action_frames": self.action_frames,
            "final_enemy_hp": self.final_enemy_hp,
            "boss_bit_set": self.boss_bit_set,
            "outcome": self.outcome,
        }


def fight_crocomire_action(
    state: SuperMetroidState,
    frame_index: int,
    strategy: CrocomireStrategy = CrocomireStrategy(),
) -> tuple[str, ...]:
    """Return one pure push/fire action while Crocomire remains active.

    The direction is intentionally strategy-configured: the acid wall is a
    room-specific fact, while the action remains independent of emulator
    ownership and never treats enemy HP zero as victory.
    """
    if boss_defeated_in_state(state, crocomire_catalog()):
        return ()

    names = [strategy.push_direction]
    if (
        strategy.fire_period > 0
        and frame_index % strategy.fire_period < strategy.fire_hold_frames
    ):
        names.append("X")
    return tuple(names)


def play_crocomire_fight(
    session: ControllerSession,
    *,
    strategy: CrocomireStrategy = CrocomireStrategy(),
) -> CrocomireEvidence:
    """Push Crocomire until its boss bit is observed or the fight times out.

    The session must already be in room ``0xA98D``. This is developmentOnly
    evidence until natural Crocomire entry exists; the boss bit is observed,
    never written, by this controller.
    """
    catalog = crocomire_catalog()
    start = session.frame
    if session.state.room_id != ROOM_CROCOMIRE:
        raise RuntimeError(
            f"Crocomire fight expected room 0x{ROOM_CROCOMIRE:04X}, "
            f"got 0x{session.state.room_id:04X}"
        )

    boss_bit_frame: int | None = None
    for index in range(strategy.max_fight_frames):
        state = session.state
        if boss_defeated_in_state(state, catalog):
            boss_bit_frame = session.frame
            break
        names = fight_crocomire_action(state, index, strategy)
        if names:
            hold(session, 1, *names, reason="fight_crocomire")
        else:
            hold(session, 1, reason="fight_crocomire_idle")
        if boss_defeated_in_state(session.state, catalog):
            boss_bit_frame = session.frame
            break

    boss_set = boss_defeated_in_state(session.state, catalog)
    return CrocomireEvidence(
        start_frame=start,
        boss_bit_frame=boss_bit_frame,
        end_frame=session.frame,
        action_frames=session.frame - start,
        final_enemy_hp=session.state.enemy0_hp,
        boss_bit_set=boss_set,
        outcome="pushed" if boss_set else "timeout",
    )
