"""Thin pure approximate SMB stepper (flat ground only).

``step(obs, action) -> obs`` is deterministic and has no emulator dependency.
It models grounded walk/run, A-edge jump, A-release gravity, and air X
tables. No slopes, pipes, enemies, or collision.

X kinematics were fitted to a live Level1_1 RAM trace:
- 16-bit speed ``(velocity_x, x_force)`` with first-kick ``0x0130``, then
  walk ``+0x98`` / run ``+0xE4``
- position subpixel ``$0400`` advances by ``velocity_x << 4``
- in air, walk tables unless ``|velocity_x| >= 0x19`` (already running)

Y is smbdis ``ImposeGravity`` + ``JumpSwimSub`` A-release:
- ``$0416`` (YMF dummy) += ``$0433`` (Y move-force); carry into Y
- Y += ``velocity_y`` + that carry
- ``$0433`` += ``$0709`` (VerticalForce); carry into ``velocity_y``
- after a 1px rise, A-release copies ``$070A`` into ``$0709``
- land snaps pixel Y and zeros ``velocity_y`` / ``$0433``; keep ``$0416``
  (YMF dummy) and leftover ``$0709``
"""

from __future__ import annotations

from collections.abc import Sequence

from smb.observation import DEFAULT_GROUND_Y, Observation

__all__ = [
    "AIR_RUN_KEEP",
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
AIR_RUN_KEEP = 0x19
JUMP_SPEED = -4
GRAVITY_HOLD_STEP = 0x20
GRAVITY_FALL = 0x70
MAX_FALL = 4
PIT_Y = 240
DIFF_TO_HALT_JUMP = 1


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


def _air_uses_walk_tables(obs: Observation) -> bool:
    """Air X uses walk accel/max until already at run speed (smbdis X_Physics)."""
    return (not obs.on_ground) and abs(int(obs.velocity_x)) < AIR_RUN_KEEP


def _update_x_speed(obs: Observation, x_dir: int, run: bool) -> tuple[int, int, int]:
    if _air_uses_walk_tables(obs):
        run = False
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


def _select_vertical_force(obs: Observation, jump: bool) -> int:
    """JumpSwimSub: keep rising VF while A is held; else copy VerticalForceDown."""
    rising = obs.vertical_force if obs.vertical_force else GRAVITY_HOLD_STEP
    falling = obs.vertical_force_down if obs.vertical_force_down else GRAVITY_FALL
    if obs.velocity_y >= 0:
        return falling
    a_held_continuous = jump and obs.a_held
    if a_held_continuous:
        return rising
    if (int(obs.jump_origin_y) - int(obs.y)) >= DIFF_TO_HALT_JUMP:
        return falling
    return rising


def _impose_gravity(
    y: int,
    sub_y: int,
    velocity_y: int,
    y_move_force: int,
    vertical_force: int,
) -> tuple[int, int, int, int]:
    """smbdis ImposeGravity for the player (downward force only, max fall 4)."""
    sub_total = (int(sub_y) & 0xFF) + (int(y_move_force) & 0xFF)
    sub_y = sub_total & 0xFF
    y = int(y) + int(velocity_y) + (1 if sub_total >= 256 else 0)
    force_total = (int(y_move_force) & 0xFF) + (int(vertical_force) & 0xFF)
    y_move_force = force_total & 0xFF
    velocity_y = int(velocity_y) + (1 if force_total >= 256 else 0)
    if velocity_y >= MAX_FALL and y_move_force >= 0x80:
        velocity_y = MAX_FALL
        y_move_force = 0
    return y, sub_y, velocity_y, y_move_force


def _land_if_needed(
    y: int,
    sub_y: int,
    velocity_y: int,
    y_move_force: int,
    vertical_force: int,
    ground_y: int,
) -> tuple[int, int, int, int, int, bool]:
    on_ground = False
    if y >= ground_y and velocity_y >= 0:
        y = ground_y
        # Game does not wipe Player_YMF_Dummy ($0416) or VerticalForce ($0709).
        velocity_y = 0
        y_move_force = 0
        on_ground = True
    return y, sub_y, velocity_y, y_move_force, vertical_force, on_ground


def _step_air(
    obs: Observation, jump: bool
) -> tuple[int, int, int, int, int, int, bool]:
    """Return y, sub_y, velocity_y, vertical_force, y_move_force, vfd, on_ground."""
    vertical_force = _select_vertical_force(obs, jump)
    vertical_force_down = obs.vertical_force_down if obs.vertical_force_down else GRAVITY_FALL
    y, sub_y, velocity_y, y_move_force = _impose_gravity(
        obs.y, obs.sub_y, obs.velocity_y, obs.y_move_force, vertical_force
    )
    y, sub_y, velocity_y, y_move_force, vertical_force, on_ground = _land_if_needed(
        y, sub_y, velocity_y, y_move_force, vertical_force, obs.ground_y
    )
    return (
        y,
        sub_y,
        velocity_y,
        vertical_force,
        y_move_force,
        vertical_force_down,
        on_ground,
    )


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
    vertical_force_down = obs.vertical_force_down if obs.vertical_force_down else GRAVITY_FALL
    y_move_force = obs.y_move_force
    jump_origin_y = obs.jump_origin_y
    if on_ground and jump and not obs.a_held:
        jump_origin_y = obs.y
        vertical_force = GRAVITY_HOLD_STEP
        vertical_force_down = GRAVITY_FALL
        y, sub_y, velocity_y, y_move_force = _impose_gravity(
            obs.y, 0, JUMP_SPEED, 0, vertical_force
        )
        on_ground = False
    elif not on_ground:
        (
            y,
            sub_y,
            velocity_y,
            vertical_force,
            y_move_force,
            vertical_force_down,
            on_ground,
        ) = _step_air(obs, jump)

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
        vertical_force_down=vertical_force_down,
        y_move_force=y_move_force,
        jump_origin_y=jump_origin_y,
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
