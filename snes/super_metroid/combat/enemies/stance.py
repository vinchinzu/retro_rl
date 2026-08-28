"""Overlay Stance: Engage / Avoid / Absorb / Ignore for this frame.

``choose`` is pure. Charge-shot and solid clamps sit behind it. Hops own
room geometry (filter enemies, skip overlay, pass fire range / takeoff
into the Species action table). Unknown species Ignore. Dead (hp<=0) is
dropped even if the caller passed them.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from super_metroid.combat.enemies.atomic import ice_engage_action
from super_metroid.combat.enemies.scan import Enemy
from super_metroid.combat.enemies.species import Stance, species_of
from super_metroid.combat.enemies.workrobot import avoid_action
from super_metroid.routes.skills.charge_shot import FIRE_RANGE_PX


@dataclass(frozen=True)
class Intent:
    """Hop overlay. Unlisted ids use Species.default_stance."""

    engage: frozenset[int] = frozenset()
    avoid: frozenset[int] = frozenset()
    absorb: frozenset[int] = frozenset()
    ignore: frozenset[int] = frozenset()

    def __post_init__(self) -> None:
        """Reject ambiguous overrides instead of relying on hidden precedence."""
        overrides = (self.engage, self.avoid, self.absorb, self.ignore)
        seen: set[int] = set()
        for group in overrides:
            overlap = seen.intersection(group)
            if overlap:
                ids = ", ".join(f"0x{enemy_id:04X}" for enemy_id in sorted(overlap))
                raise ValueError(f"Intent gives multiple Stances to {ids}")
            seen.update(group)


@dataclass(frozen=True)
class Choice:
    """One overlay frame. ``buttons is None`` means the hop continues."""

    stance: Stance
    target: Enemy | None
    buttons: tuple[str, ...] | None


def stance_for(enemy: Enemy, intent: Intent) -> Stance:
    """Intent override, then Species default (unknown → Ignore)."""
    eid = int(enemy.enemy_id)
    if eid in intent.ignore:
        return Stance.IGNORE
    if eid in intent.absorb:
        return Stance.ABSORB
    if eid in intent.avoid:
        return Stance.AVOID
    if eid in intent.engage:
        return Stance.ENGAGE
    return species_of(eid).default_stance


def _distance(samus_x: int, samus_y: int, enemy: Enemy) -> float:
    dx = abs(int(enemy.x) - int(samus_x))
    dy = abs(int(enemy.y) - int(samus_y))
    return float(dx * dx + dy * dy)


def _engage_buttons(
    samus_x: int,
    samus_y: int,
    facing: int,
    target: Enemy,
    live: tuple[Enemy, ...],
    *,
    movement_type: int,
    charge: int,
    velocity_y: int,
    fire_range_px: int,
    frozen_wait_gap: int | None,
    takeoff_x_min: int | None,
    clamp_solids: bool,
) -> tuple[str, ...] | None:
    del takeoff_x_min
    return ice_engage_action(
        int(samus_x),
        int(samus_y),
        int(facing),
        target,
        live,
        movement_type=int(movement_type),
        charge=int(charge),
        velocity_y=int(velocity_y),
        fire_range_px=int(fire_range_px),
        frozen_wait_gap=frozen_wait_gap,
        clamp=bool(clamp_solids),
    )


def _avoid_buttons(
    samus_x: int,
    samus_y: int,
    facing: int,
    target: Enemy,
    live: tuple[Enemy, ...],
    *,
    movement_type: int,
    charge: int,
    velocity_y: int,
    fire_range_px: int,
    frozen_wait_gap: int | None,
    takeoff_x_min: int | None,
    clamp_solids: bool,
) -> tuple[str, ...] | None:
    del facing, live, movement_type, charge, velocity_y
    del fire_range_px, frozen_wait_gap, clamp_solids
    return avoid_action(
        int(samus_x),
        int(samus_y),
        target,
        takeoff_x_min=takeoff_x_min,
    )


_STANCE_ACTION: dict[
    Stance,
    Callable[..., tuple[str, ...] | None],
] = {
    Stance.ENGAGE: _engage_buttons,
    Stance.AVOID: _avoid_buttons,
}


def choose(
    samus_x: int,
    samus_y: int,
    facing: int,
    enemies: tuple[Enemy, ...],
    intent: Intent,
    *,
    movement_type: int = 0,
    charge: int = 0,
    velocity_y: int = 0,
    fire_range_px: int = FIRE_RANGE_PX,
    frozen_wait_gap: int | None = None,
    takeoff_x_min: int | None = None,
    clamp_solids: bool = False,
) -> Choice:
    """Pure overlay. None buttons = hop movement; empty = idle this frame.

    Actionable Stances are considered Engage, then Avoid, then Absorb. Hops
    must handle live-contact knockback before calling this function.
    """
    action_kw: dict[str, Any] = {
        "movement_type": int(movement_type),
        "charge": int(charge),
        "velocity_y": int(velocity_y),
        "fire_range_px": int(fire_range_px),
        "frozen_wait_gap": frozen_wait_gap,
        "takeoff_x_min": takeoff_x_min,
        "clamp_solids": bool(clamp_solids),
    }
    live = tuple(e for e in enemies if int(e.hp) > 0)
    engaged = tuple(e for e in live if stance_for(e, intent) is Stance.ENGAGE)
    if engaged:
        target = min(engaged, key=lambda e: _distance(samus_x, samus_y, e))
        buttons = _STANCE_ACTION[Stance.ENGAGE](
            int(samus_x),
            int(samus_y),
            int(facing),
            target,
            live,
            **action_kw,
        )
        return Choice(Stance.ENGAGE, target, buttons)

    avoided = tuple(
        sorted(
            (e for e in live if stance_for(e, intent) is Stance.AVOID),
            key=lambda e: _distance(samus_x, samus_y, e),
        )
    )
    if avoided:
        action = _STANCE_ACTION[Stance.AVOID]
        for target in avoided:
            buttons = action(
                int(samus_x),
                int(samus_y),
                int(facing),
                target,
                live,
                **action_kw,
            )
            if buttons is not None:
                return Choice(Stance.AVOID, target, buttons)
        return Choice(Stance.AVOID, avoided[0], None)

    absorbed = tuple(e for e in live if stance_for(e, intent) is Stance.ABSORB)
    if absorbed:
        target = min(absorbed, key=lambda e: _distance(samus_x, samus_y, e))
        return Choice(Stance.ABSORB, target, None)

    return Choice(Stance.IGNORE, None, None)


__all__ = [
    "Choice",
    "Intent",
    "choose",
    "stance_for",
]
