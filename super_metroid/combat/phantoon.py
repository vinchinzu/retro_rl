"""Development-only full-knowledge Phantoon fight strategy.

This is a strategy scaffold for the Wrecked Ship boss room. It is explicitly
developmentOnly until natural Phantoon entry exists on the continuous KPDR
chain; it does not warp, place Samus, or write boss/event RAM.
"""

from __future__ import annotations

from dataclasses import dataclass

from super_metroid.combat.features import (
    boss_defeated_in_state,
    features_from_state,
    phantoon_catalog,
)
from super_metroid.combat.primitives import ensure_weapon, range_kite_action
from super_metroid.ram import SuperMetroidState
from super_metroid.routes.runtime import ControllerSession, hold

ROOM_PHANTOON = 0xCD13
WEAPON_MISSILES = 1
WEAPON_SUPERS = 2
PHANTOON_INVISIBLE = "invisible"
PHANTOON_VULNERABLE = "vulnerable"
PHANTOON_DEFEATED = "defeated"


@dataclass(frozen=True)
class PhantoonStrategy:
    """Tunable missile/Super spray and range-kite parameters."""

    min_range: int = 90
    max_range: int = 190
    jump_range: int = 150
    jump_hold_frames: int = 16
    jump_period: int = 48
    fire_period: int = 3
    max_fight_frames: int = 12_000
    weapon: int = WEAPON_MISSILES


@dataclass(frozen=True)
class PhantoonEvidence:
    """Measured result of one development-only Phantoon fight attempt."""

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
    phase_transitions: tuple[tuple[str, int], ...] = ()

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
            "phase_transitions": [
                {"phase": phase, "frame": frame}
                for phase, frame in self.phase_transitions
            ],
        }


def phantoon_phase(state: SuperMetroidState) -> str:
    """Classify the observable Phantoon phase from typed RAM features.

    The stock state surface has no eye/flame animation identifier. A live
    enemy slot with a non-zero spritemap is therefore the vulnerable window;
    a missing/inactive slot is treated as invisible until HP reaches zero.
    This is deliberately a conservative heuristic pending a spritemap probe.
    """
    if state.enemy0_hp == 0:
        return PHANTOON_DEFEATED
    features = features_from_state(state, phantoon_catalog())
    if not features.enemy_active:
        return PHANTOON_INVISIBLE
    return PHANTOON_VULNERABLE


def fight_phantoon_action(
    state: SuperMetroidState,
    frame_index: int,
    strategy: PhantoonStrategy = PhantoonStrategy(),
) -> tuple[str, ...]:
    """Return one pure, RAM-driven spray action for Phantoon.

    The selected weapon is controlled by the caller. ``X`` therefore works
    for either missiles or Supers without granting ammo or changing inventory.
    """
    catalog = phantoon_catalog()
    features = features_from_state(state, catalog)
    if features.enemy_defeated or state.enemy0_hp == 0:
        return ()
    phase = phantoon_phase(state)
    action = range_kite_action(
        state.samus_x,
        features.enemy_x,
        min_range=strategy.min_range,
        max_range=strategy.max_range,
        jump_range=strategy.jump_range,
        frame_index=frame_index,
        jump_period=strategy.jump_period,
        jump_hold_frames=strategy.jump_hold_frames,
        fire_period=strategy.fire_period if phase == PHANTOON_VULNERABLE else 0,
        fire_button="X",
    )
    return action


def play_phantoon_fight(
    session: ControllerSession,
    *,
    strategy: PhantoonStrategy = PhantoonStrategy(),
) -> PhantoonEvidence:
    """Fight Phantoon until HP zero, the boss bit, or a bounded timeout.

    The session must already be in room ``0xCD13``. This is developmentOnly
    evidence until the room is reached naturally from the continuous chain.
    """
    catalog = phantoon_catalog()
    start = session.frame
    if session.state.room_id != ROOM_PHANTOON:
        raise RuntimeError(
            f"Phantoon fight expected room 0x{ROOM_PHANTOON:04X}, "
            f"got 0x{session.state.room_id:04X}"
        )

    ensure_weapon(session, strategy.weapon)
    initial_hp = session.state.enemy0_hp
    peak_hp = initial_hp
    min_hp = initial_hp
    body_zero_frame: int | None = None
    boss_bit_frame: int | None = None
    prev_hp = initial_hp
    current_phase = phantoon_phase(session.state)
    phase_transitions: list[tuple[str, int]] = [(current_phase, start)]
    if initial_hp == 0:
        body_zero_frame = start

    for index in range(strategy.max_fight_frames):
        state = session.state
        if boss_defeated_in_state(state, catalog):
            boss_bit_frame = session.frame
            break
        if body_zero_frame is not None:
            break

        names = fight_phantoon_action(state, index, strategy)
        if names:
            hold(session, 1, *names, reason="fight_phantoon")
        else:
            hold(session, 1, reason="fight_phantoon_idle")

        post = session.state
        next_phase = phantoon_phase(post)
        if next_phase != current_phase:
            phase_transitions.append((next_phase, session.frame))
            current_phase = next_phase
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
        outcome = "phantoon_defeated"
    elif body_zero_frame is not None:
        outcome = "phantoon_body_zero_no_boss_bit"
    else:
        outcome = "timeout"

    return PhantoonEvidence(
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
        phase_transitions=tuple(phase_transitions),
    )
