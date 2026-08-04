"""Optional full-knowledge Golden Torizo combat scaffold.

This controller is practice evidence only.  It assumes a natural room entry,
uses the catalogued enemy slot, and never writes inventory, progression, or
boss/event RAM.
"""

from __future__ import annotations

from dataclasses import dataclass

from super_metroid.combat.features import (
    features_from_state,
    golden_torizo_catalog,
)
from super_metroid.combat.primitives import ensure_weapon, range_kite_action
from super_metroid.ram import SuperMetroidState
from super_metroid.routes.runtime import ControllerSession, hold


ROOM_GOLDEN_TORIZO = 0xB283
WEAPON_SUPERS = 2


@dataclass(frozen=True)
class GoldenTorizoStrategy:
    """Tunable Super Missile range-kite parameters for optional practice."""

    min_range: int = 90
    max_range: int = 170
    jump_range: int = 140
    jump_hold_frames: int = 16
    jump_period: int = 48
    fire_period: int = 3
    max_fight_frames: int = 16_000
    weapon: int = WEAPON_SUPERS


@dataclass(frozen=True)
class GoldenTorizoEvidence:
    """Measured result of one bounded Golden Torizo fight attempt."""

    start_frame: int
    body_zero_frame: int | None
    end_frame: int
    peak_body_hp: int
    min_body_hp: int
    action_frames: int
    final_body_hp: int
    outcome: str

    def to_dict(self) -> dict[str, object]:
        return {
            "start_frame": self.start_frame,
            "body_zero_frame": self.body_zero_frame,
            "end_frame": self.end_frame,
            "peak_body_hp": self.peak_body_hp,
            "min_body_hp": self.min_body_hp,
            "action_frames": self.action_frames,
            "final_body_hp": self.final_body_hp,
            "outcome": self.outcome,
        }


def fight_golden_torizo_action(
    state: SuperMetroidState,
    frame_index: int,
    strategy: GoldenTorizoStrategy = GoldenTorizoStrategy(),
) -> tuple[str, ...]:
    """Return one pure RAM-driven Super Missile action."""
    catalog = golden_torizo_catalog()
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


def play_golden_torizo_fight(
    session: ControllerSession,
    *,
    strategy: GoldenTorizoStrategy = GoldenTorizoStrategy(),
) -> GoldenTorizoEvidence:
    """Fight Golden Torizo until enemy HP reaches zero or the budget expires.

    The session must already be in room ``0xB283``.  Golden Torizo has no
    catalogued major-boss bit, so HP-zero is the only defeat signal here.
    """
    catalog = golden_torizo_catalog()
    start = session.frame
    if session.state.room_id != ROOM_GOLDEN_TORIZO:
        raise RuntimeError(
            f"Golden Torizo fight expected room 0x{ROOM_GOLDEN_TORIZO:04X}, "
            f"got 0x{session.state.room_id:04X}"
        )

    ensure_weapon(session, strategy.weapon)
    initial_hp = session.state.enemy0_hp
    peak_hp = initial_hp
    min_hp = initial_hp
    body_zero_frame: int | None = start if initial_hp == 0 else None
    prev_hp = initial_hp

    for index in range(strategy.max_fight_frames):
        if body_zero_frame is not None:
            break
        state = session.state
        names = fight_golden_torizo_action(state, index, strategy)
        if names:
            hold(session, 1, *names, reason="fight_golden_torizo")
        else:
            hold(session, 1, reason="fight_golden_torizo_idle")

        hp = session.state.enemy0_hp
        if 0 <= hp <= catalog.max_hp:
            peak_hp = max(peak_hp, hp)
            min_hp = min(min_hp, hp)
        if body_zero_frame is None and hp == 0 and prev_hp > 0:
            body_zero_frame = session.frame
            min_hp = 0
            break
        prev_hp = hp

    final_hp = session.state.enemy0_hp
    outcome = (
        "golden_torizo_defeated"
        if body_zero_frame is not None
        else "timeout"
    )
    return GoldenTorizoEvidence(
        start_frame=start,
        body_zero_frame=body_zero_frame,
        end_frame=session.frame,
        peak_body_hp=peak_hp,
        min_body_hp=min_hp,
        action_frames=session.frame - start,
        final_body_hp=final_hp,
        outcome=outcome,
    )
