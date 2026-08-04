"""Development-only Mother Brain strategy scaffold.

This module models the multi-phase fight without owning emulator setup,
progression writes, or the post-fight escape.  It is intentionally deferred
until natural Tourian entry and the escape/ending path are proven.
"""

from __future__ import annotations

from dataclasses import dataclass

from super_metroid.combat.features import (
    boss_defeated_in_state,
    features_from_state,
    mother_brain_catalog,
)
from super_metroid.combat.primitives import spray_action, ensure_weapon
from super_metroid.ram import SuperMetroidState
from super_metroid.routes.runtime import ControllerSession, hold

ROOM_MOTHER_BRAIN = 0xDD58
WEAPON_MISSILES = 1
WEAPON_SUPERS = 2
PHASE_MB1 = "mb1"
PHASE_MB2 = "mb2"
PHASE_MB3 = "mb3"


@dataclass(frozen=True)
class MotherBrainStrategy:
    """Tunable spray parameters for the three Mother Brain phases."""

    fire_period: int = 3
    max_fight_frames: int = 30_000
    weapon: int = WEAPON_MISSILES
    # HP values are the catalog's phase maxima; they label observed phases,
    # not progression state and are not written back to the emulator.
    phase_thresholds: tuple[int, int] = (3_000, 18_000)


@dataclass(frozen=True)
class MotherBrainEvidence:
    """Measured result of one development-only Mother Brain attempt."""

    start_frame: int
    body_zero_frame: int | None
    boss_bit_frame: int | None
    event_frame: int | None
    end_frame: int
    peak_body_hp: int
    min_body_hp: int
    action_frames: int
    final_body_hp: int
    boss_bit_set: bool
    event_set: bool
    phase_timeline: tuple[dict[str, object], ...]
    outcome: str

    def to_dict(self) -> dict[str, object]:
        return {
            "start_frame": self.start_frame,
            "body_zero_frame": self.body_zero_frame,
            "boss_bit_frame": self.boss_bit_frame,
            "event_frame": self.event_frame,
            "end_frame": self.end_frame,
            "peak_body_hp": self.peak_body_hp,
            "min_body_hp": self.min_body_hp,
            "action_frames": self.action_frames,
            "final_body_hp": self.final_body_hp,
            "boss_bit_set": self.boss_bit_set,
            "event_set": self.event_set,
            "phase_timeline": list(self.phase_timeline),
            "outcome": self.outcome,
        }


def mother_brain_phase(
    state: SuperMetroidState,
    strategy: MotherBrainStrategy = MotherBrainStrategy(),
) -> str:
    """Return the phase label implied by the observed enemy HP."""
    first, second = strategy.phase_thresholds
    hp = int(state.enemy0_hp)
    if hp <= first:
        return PHASE_MB1
    if hp <= second:
        return PHASE_MB2
    return PHASE_MB3


def _boss_bit_set(state: SuperMetroidState) -> bool:
    catalog = mother_brain_catalog()
    index = catalog.boss_area_index
    return bool(
        index is not None
        and index < len(state.boss_bits)
        and state.boss_bits[index] & catalog.boss_bit_mask
    )


def _event_set(state: SuperMetroidState) -> bool:
    event_id = mother_brain_catalog().defeat_event_id
    if event_id is None:
        return False
    byte_index = event_id >> 3
    return bool(
        byte_index < len(state.event_flags)
        and state.event_flags[byte_index] & (1 << (event_id & 7))
    )


def fight_mother_brain_action(
    state: SuperMetroidState,
    frame_index: int,
    strategy: MotherBrainStrategy = MotherBrainStrategy(),
) -> tuple[str, ...]:
    """Return one pure spray action, or no action after defeat/event."""
    catalog = mother_brain_catalog()
    if boss_defeated_in_state(state, catalog) or state.enemy0_hp == 0:
        return ()
    features = features_from_state(state, catalog)
    if features.enemy_defeated:
        return ()
    # Phase classification is retained for evidence and future phase-specific
    # policies; this first scaffold uses one spray policy for all phases.
    mother_brain_phase(state, strategy)
    return spray_action(
        frame_index,
        face="RIGHT",
        fire_period=strategy.fire_period,
        fire_hold_frames=1,
        jump_period=0,
        dash_when_not_jumping=True,
        fire_button="X",
    )


def play_mother_brain_fight(
    session: ControllerSession,
    *,
    strategy: MotherBrainStrategy = MotherBrainStrategy(),
) -> MotherBrainEvidence:
    """Run a bounded fight from an already-entered Mother Brain room.

    This is developmentOnly evidence.  It does not implement rainbow-beam or
    hyper-beam special handling, and it does not initiate the escape.
    """
    catalog = mother_brain_catalog()
    start = session.frame
    if session.state.room_id != ROOM_MOTHER_BRAIN:
        raise RuntimeError(
            f"Mother Brain fight expected room 0x{ROOM_MOTHER_BRAIN:04X}, "
            f"got 0x{session.state.room_id:04X}"
        )

    ensure_weapon(session, strategy.weapon)
    initial_hp = int(session.state.enemy0_hp)
    peak_hp = initial_hp
    min_hp = initial_hp
    body_zero_frame: int | None = start if initial_hp == 0 else None
    boss_bit_frame: int | None = start if _boss_bit_set(session.state) else None
    event_frame: int | None = start if _event_set(session.state) else None
    phase_timeline: list[dict[str, object]] = []
    previous_phase: str | None = None
    prev_hp = initial_hp

    for index in range(strategy.max_fight_frames):
        state = session.state
        phase = mother_brain_phase(state, strategy)
        if phase != previous_phase:
            phase_timeline.append({"phase": phase, "frame": session.frame, "hp": int(state.enemy0_hp)})
            previous_phase = phase
        if boss_defeated_in_state(state, catalog):
            boss_bit_frame = boss_bit_frame or session.frame
            if _event_set(state):
                event_frame = event_frame or session.frame
            break
        if state.enemy0_hp == 0 or body_zero_frame is not None:
            break

        names = fight_mother_brain_action(state, index, strategy)
        hold(session, 1, *names, reason="fight_mother_brain" if names else "fight_mother_brain_idle")
        post = session.state
        hp = int(post.enemy0_hp)
        if 0 <= hp <= catalog.phases[-1].max_hp:
            peak_hp = max(peak_hp, hp)
            min_hp = min(min_hp, hp)
        if body_zero_frame is None and hp == 0 and prev_hp > 0:
            body_zero_frame = session.frame
            min_hp = 0
        if boss_bit_frame is None and _boss_bit_set(post):
            boss_bit_frame = session.frame
        if event_frame is None and _event_set(post):
            event_frame = session.frame
        prev_hp = hp
        if body_zero_frame is not None or boss_bit_frame is not None:
            break

    final_hp = int(session.state.enemy0_hp)
    boss_set = _boss_bit_set(session.state)
    event_set = _event_set(session.state)
    if event_set:
        outcome = "mother_brain_defeated_event_set"
    elif boss_set:
        outcome = "mother_brain_defeated"
    elif body_zero_frame is not None:
        outcome = "mother_brain_body_zero_no_event"
    else:
        outcome = "timeout"

    return MotherBrainEvidence(
        start_frame=start,
        body_zero_frame=body_zero_frame,
        boss_bit_frame=boss_bit_frame,
        event_frame=event_frame,
        end_frame=session.frame,
        peak_body_hp=peak_hp,
        min_body_hp=min_hp,
        action_frames=session.frame - start,
        final_body_hp=final_hp,
        boss_bit_set=boss_set,
        event_set=event_set,
        phase_timeline=tuple(phase_timeline),
        outcome=outcome,
    )
