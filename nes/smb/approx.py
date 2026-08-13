"""Thin pure approximate SMB stepper (flat ground only).

``step(obs, action) -> obs`` is deterministic and has no emulator dependency.
It models grounded walk/run, A-edge jump, and rising/falling gravity. No
slopes, pipes, enemies, or collision.

X kinematics were fitted to a live Level1_1 RAM trace:
- 16-bit speed ``(velocity_x, x_force)`` with first-kick ``0x0130``, then
  walk ``+0x98`` / run ``+0xE4``
- position subpixel ``$0400`` advances by ``velocity_x << 4``
"""

from __future__ import annotations

from collections.abc import Sequence

from smb.observation import DEFAULT_GROUND_Y, Observation

__all__ = [
    "JUMP_SPEED",
    "RUN_ACCEL",
    "RUN_MAX",
    "WALK_ACCEL",
    "WALK_MAX",
    "decode_action",
    "idle_action",
    "press",
    "rollout",
    "step",
]

# NES 9-slot: [B, null, Select, Start, Up, Down, Left, Right, A]
_IDX_B = 0
_IDX_LEFT = 6
_IDX_RIGHT = 7
_IDX_A = 8

WALK_ACCEL = 0x98
RUN_ACCEL = 0xE4
FRICTION = 0xD0
FIRST_KICK = 0x0130
WALK_MAX = 0x18
RUN_MAX = 0x28
JUMP_SPEED = -4
GRAVITY_HOLD_STEP = 0x20
GRAVITY_FALL = 0x70
MAX_FALL = 4
PIT_Y = 240


def idle_action() -> tuple[int, ...]:
    return (0,) * 9


def press(*buttons: str) -> tuple[int, ...]:
    """Build a 9-slot NES action from button names (B, A, LEFT, RIGHT, …)."""
    action = [0] * 9
    names = {name.strip().upper() for name in buttons}
    if "B" in names:
        action[_IDX_B] = 1
    if "LEFT" in names:
        action[_IDX_LEFT] = 1
    if "RIGHT" in names:
        action[_IDX_RIGHT] = 1
    if "A" in names:
        action[_IDX_A] = 1
    return tuple(action)


def decode_action(action: Sequence[int]) -> tuple[int, bool, bool]:
    """Return ``(x_dir, run, jump)`` with x_dir in {-1, 0, +1}."""
    buttons = list(action) + [0] * 9
    left = bool(buttons[_IDX_LEFT])
    right = bool(buttons[_IDX_RIGHT])
    if left and not right:
        x_dir = -1
    elif right and not left:
        x_dir = 1
    else:
        x_dir = 0
    return x_dir, bool(buttons[_IDX_B]), bool(buttons[_IDX_A])


def _pack_speed(velocity_x: int, x_force: int) -> int:
    force = int(x_force) & 0xFF
    if velocity_x < 0:
        return -(((-int(velocity_x)) << 8) | force)
    return (int(velocity_x) << 8) | force


def _unpack_speed(speed_16: int) -> tuple[int, int]:
    if speed_16 < 0:
        magnitude = -speed_16
        return -(magnitude >> 8), magnitude & 0xFF
    return speed_16 >> 8, speed_16 & 0xFF


def _clamp_speed(speed_16: int, run: bool) -> int:
    limit = (RUN_MAX if run else WALK_MAX) << 8
    if speed_16 > limit:
        return limit
    if speed_16 < -limit:
        return -limit
    return speed_16


def _update_x_speed(obs: Observation, x_dir: int, run: bool) -> tuple[int, int, int]:
    speed_16 = _pack_speed(obs.velocity_x, obs.x_force)
    facing = obs.facing
    if x_dir > 0:
        facing = 1
        if speed_16 == 0:
            speed_16 = FIRST_KICK
        else:
            speed_16 += RUN_ACCEL if run else WALK_ACCEL
        speed_16 = _clamp_speed(speed_16, run)
    elif x_dir < 0:
        facing = 2
        if speed_16 == 0:
            speed_16 = -FIRST_KICK
        else:
            speed_16 -= RUN_ACCEL if run else WALK_ACCEL
        speed_16 = _clamp_speed(speed_16, run)
    elif speed_16 > 0:
        speed_16 = max(0, speed_16 - FRICTION)
    elif speed_16 < 0:
        speed_16 = min(0, speed_16 + FRICTION)
    velocity_x, x_force = _unpack_speed(speed_16)
    return velocity_x, x_force, facing


def _advance_x(x: int, sub_x: int, velocity_x: int) -> tuple[int, int]:
    total = (int(x) << 8) + (int(sub_x) & 0xFF) + (int(velocity_x) << 4)
    if total < 0:
        return 0, 0
    return total >> 8, total & 0xFF


def _step_air(obs: Observation, jump: bool) -> tuple[int, int, int, int, bool]:
    """Return y, sub_y, velocity_y, vertical_force, on_ground."""
    y = obs.y + obs.velocity_y
    sub_y = obs.sub_y
    velocity_y = obs.velocity_y
    vertical_force = obs.vertical_force
    if jump and velocity_y < 0:
        add = vertical_force if vertical_force else GRAVITY_HOLD_STEP
        vertical_force = min(add + GRAVITY_HOLD_STEP, 0xA0)
    else:
        add = GRAVITY_FALL
        vertical_force = GRAVITY_FALL
    sub_y += add
    if sub_y >= 256:
        y += 1
        sub_y -= 256
        velocity_y = min(velocity_y + 1, MAX_FALL)
    on_ground = False
    if y >= obs.ground_y and velocity_y >= 0:
        y = obs.ground_y
        sub_y = 0
        velocity_y = 0
        vertical_force = 0
        on_ground = True
    return y, sub_y, velocity_y, vertical_force, on_ground


def step(obs: Observation, action: Sequence[int]) -> Observation:
    """Advance one frame. Pure: no RAM, no I/O, no RNG."""
    x_dir, run, jump = decode_action(action)
    velocity_x, x_force, facing = _update_x_speed(obs, x_dir, run)
    x, sub_x = _advance_x(obs.x, obs.sub_x, velocity_x)

    on_ground = obs.on_ground
    y = obs.y
    sub_y = obs.sub_y
    velocity_y = obs.velocity_y
    vertical_force = obs.vertical_force
    if on_ground and jump and not obs.a_held:
        velocity_y = JUMP_SPEED
        on_ground = False
        y = obs.y + JUMP_SPEED
        sub_y = 0
        vertical_force = GRAVITY_HOLD_STEP
    elif not on_ground:
        y, sub_y, velocity_y, vertical_force, on_ground = _step_air(obs, jump)
        # Air still uses the just-updated horizontal speed / position above.

    pose = obs.pose
    dead = obs.dead
    if y >= PIT_Y:
        dead = True
        pose = 0x0B
        y = min(y, 255)

    frame_counter = obs.frame_counter
    if frame_counter is not None:
        frame_counter = (int(frame_counter) + 1) & 0xFF

    return Observation(
        frame=obs.frame + 1,
        x=x,
        y=y,
        pose=pose,
        room=obs.room,
        sub_x=sub_x,
        sub_y=sub_y,
        velocity_x=velocity_x,
        velocity_y=velocity_y,
        energy=obs.energy,
        dead=dead,
        frame_counter=frame_counter,
        enemy0_active=obs.enemy0_active,
        enemy0_type=obs.enemy0_type,
        facing=facing,
        on_ground=on_ground,
        x_force=x_force,
        a_held=jump,
        ground_y=obs.ground_y if obs.ground_y else DEFAULT_GROUND_Y,
        vertical_force=vertical_force,
        oper_mode=obs.oper_mode,
        timer=obs.timer,
    )


def rollout(start: Observation, actions: Sequence[Sequence[int]]) -> list[Observation]:
    """Return ``[start, step(start, a0), …]`` (length ``len(actions)+1``)."""
    frames = [start]
    current = start
    for action in actions:
        current = step(current, action)
        frames.append(current)
    return frames
