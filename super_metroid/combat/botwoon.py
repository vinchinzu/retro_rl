"""Development-only full-knowledge Botwoon fight strategy.

This is a strategy scaffold for the Maridia Botwoon room. It is explicitly
developmentOnly until natural Maridia entry exists on the continuous KPDR
chain; it does not warp, place Samus, or write boss/event RAM.
"""

from __future__ import annotations

from dataclasses import dataclass

from super_metroid.combat.features import (
    botwoon_catalog,
    boss_defeated_in_state,
    features_from_state,
)
from super_metroid.combat.primitives import ensure_weapon, spray_action
from super_metroid.ram import SuperMetroidState
from super_metroid.routes.runtime import ControllerSession, hold

ROOM_BOTWOON = 0xD95E
WEAPON_MISSILES = 1
WEAPON_SUPERS = 2


@dataclass(frozen=True)
class BotwoonStrategy:
    """Tunable periodic spray parameters for Botwoon."""

    fire_period: int = 3
    fire_hold_frames: int = 1
    jump_period: int = 48
    jump_hold_frames: int = 16
    max_fight_frames: int = 12_000
    weapon: int = WEAPON_SUPERS


@dataclass(frozen=True)
class BotwoonEvidence:
    """Measured result of one development-only Botwoon fight attempt."""

    start_frame: int
    defeat_frame: int | None
    boss_bit_frame: int | None
    end_frame: int
    peak_enemy_hp: int
    min_enemy_hp: int
    action_frames: int
    final_enemy_hp: int
    boss_bit_set: bool
    outcome: str

    def to_dict(self) -> dict[str, object]:
        return {
            "start_frame": self.start_frame,
            "defeat_frame": self.defeat_frame,
            "boss_bit_frame": self.boss_bit_frame,
            "end_frame": self.end_frame,
            "peak_enemy_hp": self.peak_enemy_hp,
            "min_enemy_hp": self.min_enemy_hp,
            "action_frames": self.action_frames,
            "final_enemy_hp": self.final_enemy_hp,
            "boss_bit_set": self.boss_bit_set,
            "outcome": self.outcome,
        }


def fight_botwoon_action(
    state: SuperMetroidState,
    frame_index: int,
    strategy: BotwoonStrategy = BotwoonStrategy(),
) -> tuple[str, ...]:
    """Return one pure, RAM-driven facing spray action for Botwoon."""
    catalog = botwoon_catalog()
    features = features_from_state(state, catalog)
    if features.enemy_defeated or state.enemy0_hp == 0:
        return ()

    dx = features.enemy_x - state.samus_x
    face = "RIGHT" if dx >= 0 else "LEFT"
    return spray_action(
        frame_index,
        face=face,
        fire_period=strategy.fire_period,
        fire_hold_frames=strategy.fire_hold_frames,
        jump_period=strategy.jump_period,
        jump_hold_frames=strategy.jump_hold_frames,
        fire_button="X",
    )


def play_botwoon_fight(
    session: ControllerSession,
    *,
    strategy: BotwoonStrategy = BotwoonStrategy(),
) -> BotwoonEvidence:
    """Fight Botwoon until HP zero, the boss bit, or a bounded timeout.

    The session must already be in room ``0xD95E``. This is developmentOnly
    evidence until the room is reached naturally from the continuous chain.
    """
    catalog = botwoon_catalog()
    start = session.frame
    if session.state.room_id != ROOM_BOTWOON:
        raise RuntimeError(
            f"Botwoon fight expected room 0x{ROOM_BOTWOON:04X}, "
            f"got 0x{session.state.room_id:04X}"
        )

    ensure_weapon(session, strategy.weapon)
    initial_hp = session.state.enemy0_hp
    peak_hp = initial_hp
    min_hp = initial_hp
    defeat_frame: int | None = start if initial_hp == 0 else None
    boss_bit_frame: int | None = None
    prev_hp = initial_hp

    for index in range(strategy.max_fight_frames):
        state = session.state
        if boss_defeated_in_state(state, catalog):
            boss_bit_frame = session.frame
            break
        if defeat_frame is not None:
            break

        names = fight_botwoon_action(state, index, strategy)
        if names:
            hold(session, 1, *names, reason="fight_botwoon")
        else:
            hold(session, 1, reason="fight_botwoon_idle")

        post = session.state
        hp = post.enemy0_hp
        if 0 <= hp <= catalog.max_hp:
            peak_hp = max(peak_hp, hp)
            min_hp = min(min_hp, hp)
        if defeat_frame is None and hp == 0 and prev_hp > 0:
            defeat_frame = session.frame
            min_hp = 0
        if boss_bit_frame is None and boss_defeated_in_state(post, catalog):
            boss_bit_frame = session.frame
        if boss_bit_frame is not None or defeat_frame is not None:
            break
        prev_hp = hp

    final_hp = session.state.enemy0_hp
    boss_set = boss_defeated_in_state(session.state, catalog)
    if boss_set:
        outcome = "botwoon_defeated"
    elif defeat_frame is not None:
        outcome = "botwoon_body_zero_no_boss_bit"
    else:
        outcome = "timeout"

    return BotwoonEvidence(
        start_frame=start,
        defeat_frame=defeat_frame,
        boss_bit_frame=boss_bit_frame,
        end_frame=session.frame,
        peak_enemy_hp=peak_hp,
        min_enemy_hp=min_hp,
        action_frames=session.frame - start,
        final_enemy_hp=final_hp,
        boss_bit_set=boss_set,
        outcome=outcome,
    )
