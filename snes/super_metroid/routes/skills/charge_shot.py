"""Charge-release beam shot: position, aim, hold X, release.

Charge equipped: holding X does not fire; release does. Full charge is
``$0CD0 >= 60`` (glow continues to 120). Diagonal aim is shoulder **R**
(up) / **L** (down) — never UP+LEFT/RIGHT. Jump-shot: hold A on the last
charge frames so the projectile spawns in the air.

One-frame builder. Caller holds the returned buttons for 1f and re-reads
charge / pose. Do not tap-release X from a blocked seat.
"""

from __future__ import annotations

from typing import Any

from super_metroid.ram import FACING_LEFT, FACING_RIGHT

# $7E:0CD0 beam-charge counter. Same word Phantoon combat uses.
ADDR_BEAM_CHARGE = 0x0CD0
CHARGE_FULL = 60
# Start the jump this many charge ticks before release so the shot is airborne.
JUMP_LEAD = 12
# $0A1F movement type 0x0E — standing/crouch turnaround. Pose 37/38.
MOVEMENT_TURNING = 14

# Seat: close enough that an unblocked shot can connect.
FIRE_RANGE_PX = 48
# |dx| this small + blob well above → straight UP, not diagonal.
UNDER_BLOB_DX = 16
# dy = target_y - samus_y (negative = blob above).
AIM_UP_DY = -40
AIM_DOWN_DY = 40
JUMP_SHOT_DY = -16


def is_turning(movement_type: int) -> bool:
    """True while Samus is in movement-type 14 (turnaround). Do not fire."""
    return int(movement_type) == MOVEMENT_TURNING


def beam_charge_counter(ram: Any) -> int:
    """Live ``$0CD0`` word. 0 if RAM is missing or short."""
    if ram is None:
        return 0
    try:
        size = len(ram)
    except TypeError:
        return 0
    if size < ADDR_BEAM_CHARGE + 2:
        return 0
    return int(ram[ADDR_BEAM_CHARGE]) | (int(ram[ADDR_BEAM_CHARGE + 1]) << 8)


def session_beam_charge(session: Any) -> int:
    """Charge counter from ``session.env.get_ram``, or 0."""
    env = getattr(session, "env", None)
    get_ram = getattr(env, "get_ram", None) if env is not None else None
    if get_ram is None:
        return 0
    try:
        return beam_charge_counter(get_ram())
    except Exception:  # noqa: BLE001
        return 0


def in_shot_seat(
    samus_x: int,
    samus_y: int,
    target_x: int,
    target_y: int,
    *,
    fire_range_px: int = FIRE_RANGE_PX,
    approach_x_min: int | None = None,
    approach_x_max: int | None = None,
    clamp_slack: int = 8,
) -> bool:
    """True when a shot from here can be aimed at the target.

    Horizontal range, or else the approach clamp (robot / wall) is as close
    as we are allowed — fire from the clamp instead of walking through it.
    """
    del samus_y, target_y
    sx, tx = int(samus_x), int(target_x)
    if abs(tx - sx) <= int(fire_range_px):
        return True
    if approach_x_min is not None and tx < sx and sx <= int(approach_x_min) + clamp_slack:
        return True
    if approach_x_max is not None and tx > sx and sx >= int(approach_x_max) - clamp_slack:
        return True
    return False


def aim_shot_buttons(
    dx: int,
    dy: int,
    *,
    jump: bool = False,
    fire: bool = False,
    include_face: bool = False,
) -> tuple[str, ...]:
    """Aim buttons. **R** = diagonal up, **L** = diagonal down, **UP** under blob.

    ``include_face`` is for the turn/approach frames only. In-seat fire omits
    LEFT/RIGHT so the shot does not walk off the seat.
    """
    names: list[str] = []
    if include_face:
        if dx < 0:
            names.append("LEFT")
        elif dx > 0:
            names.append("RIGHT")
    if dy <= AIM_UP_DY and abs(dx) <= UNDER_BLOB_DX:
        names.append("UP")
    elif dy <= AIM_UP_DY:
        names.append("R")
    elif dy >= AIM_DOWN_DY:
        names.append("L")
    if jump:
        names.append("A")
    if fire:
        names.append("X")
    return tuple(dict.fromkeys(names))


def position_then_charge_action(
    samus_x: int,
    samus_y: int,
    facing: int,
    target_x: int,
    target_y: int,
    *,
    movement_type: int = 0,
    charge: int = 0,
    velocity_y: int = 0,
    fire_range_px: int = FIRE_RANGE_PX,
    approach_x_min: int | None = None,
    approach_x_max: int | None = None,
) -> tuple[str, ...]:
    """One frame: face → walk into seat → charge → jump-lead → release.

    Frozen targets are still shot (Ice-until-dead). Do not walk into a
    clamp (Workrobot / frozen blob): fire from the clamp seat instead.
    Jump-shot holds X through the first airborne frames, then releases.
    """
    sx, sy = int(samus_x), int(samus_y)
    tx, ty = int(target_x), int(target_y)
    dx, dy = tx - sx, ty - sy
    face_left = dx < 0 or (dx == 0 and int(facing) == FACING_LEFT)
    want_facing = FACING_LEFT if face_left else FACING_RIGHT
    face_btn = "LEFT" if face_left else "RIGHT"

    if is_turning(movement_type) or int(facing) != want_facing:
        return (face_btn,)

    seated = in_shot_seat(
        sx,
        sy,
        tx,
        ty,
        fire_range_px=fire_range_px,
        approach_x_min=approach_x_min,
        approach_x_max=approach_x_max,
    )
    if not seated:
        walk: list[str] = [face_btn, "B"]
        if approach_x_min is not None and sx <= int(approach_x_min) and face_left:
            walk = []
        if approach_x_max is not None and sx >= int(approach_x_max) and not face_left:
            walk = []
        if walk:
            walk.append("X")
            return tuple(dict.fromkeys(walk))
        seated = True

    need_jump = dy <= JUMP_SHOT_DY
    airborne = int(velocity_y) != 0
    jumping = need_jump and (int(charge) >= CHARGE_FULL - JUMP_LEAD or airborne)
    # Grounded jump-shot: keep X until vy≠0 so the projectile is airborne.
    firing = int(charge) < CHARGE_FULL or (need_jump and not airborne)
    return aim_shot_buttons(dx, dy, jump=jumping, fire=firing, include_face=False)


__all__ = [
    "ADDR_BEAM_CHARGE",
    "AIM_DOWN_DY",
    "AIM_UP_DY",
    "CHARGE_FULL",
    "FIRE_RANGE_PX",
    "JUMP_LEAD",
    "JUMP_SHOT_DY",
    "MOVEMENT_TURNING",
    "UNDER_BLOB_DX",
    "aim_shot_buttons",
    "beam_charge_counter",
    "in_shot_seat",
    "is_turning",
    "position_then_charge_action",
    "session_beam_charge",
]
