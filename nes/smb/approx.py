"""Thin pure approximate SMB stepper (flat ground only).

``step(player, action, world) -> player`` is deterministic and has no
emulator dependency. It models grounded walk/run, A-edge jump, A-release
gravity, air X tables (including the takeoff frame), and land YMF. No
slopes, pipes, enemies, or collision.

X kinematics follow smbdis ``ImposeFriction`` on a 16-bit two's-complement
``(Player_X_Speed, Player_X_MoveForce)`` word:
- at rest, facing ≠ moving doubles the adder (``$98 << 1`` → first-kick ``$0130``)
- RIGHT adds the adder; LEFT subtracts it (``0 - $0130`` → ``$FED0``, not ``-$0130``)
- then walk ``$98`` / run ``$E4``
- clamp snaps only the high byte (``MaximumRight/LeftSpeed``); leftover ``xf`` stays
- no L/R uses smbdis ``FrictionData`` ``$98`` unless ``|vx| >= $21`` (``$D0``)
- position subpixel ``$0400`` advances by ``velocity_x << 4``
- in air (including the takeoff frame), walk tables unless
  ``|velocity_x| >= 0x19`` (already running)

Y is smbdis ``ImposeGravity`` + ``JumpSwimSub`` A-release:
- ``$0416`` (YMF dummy) += ``$0433`` (Y move-force); carry into Y
- Y += ``velocity_y`` + that carry
- ``$0433`` += ``$0709`` (VerticalForce); carry into ``velocity_y``
- after a 1px rise, A-release copies ``$070A`` into ``$0709``
- land snaps pixel Y and zeros ``velocity_y`` / ``$0433``; keep ``$0416``
  (YMF dummy) and leftover ``$0709``
- takeoff zeros ``$0416`` (InitJS wipes YMF dummy; land does not)

Takeoff (smbdis ``InitJS``) indexes ``JumpMForceData`` / ``FallMForceData`` /
``PlayerYSpdData`` from ``|$0700|`` (``Player_XSpeedAbsolute`` ≈ ``|vx|``).
Land bands only; swim indices 5–6 are not modeled.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from retro_harness.nes import nes_action, nes_idle_action

from smb.observation import FLAT_1_1, PlayerPhysics, World

__all__ = [
    "AIR_RUN_KEEP",
    "BRAKE_FAST",
    "JUMP_FORCE_DOWN",
    "RUN_SPEED_LATCH",
    "JUMP_FORCE_UP",
    "JUMP_SPEED",
    "JUMP_Y_SPEED",
    "RUN_ACCEL",
    "RUN_MAX",
    "WALK_ACCEL",
    "WALK_MAX",
    "decode_action",
    "idle_action",
    "jump_table_index",
    "press",
    "rollout",
    "step",
    "takeoff_vertical",
]

Phase = Literal["ground", "takeoff", "air"]

WALK_ACCEL = 0x98
RUN_ACCEL = 0xE4
FRICTION = 0xD0
# Rest + facing ≠ moving doubles FrictionData[$00] ($98 << 1). LEFT subtracts.
FIRST_KICK = 0x0130
WALK_MAX = 0x18
RUN_MAX = 0x28
AIR_RUN_KEEP = 0x19
# smbdis X_Physics $00 / FrictionData: no L/R uses $98 unless |vx| >= $21
# or RunningSpeed (latched by GetPlayerAnimSpeed when |vx| >= $1C).
BRAKE_FAST = 0x21
RUN_SPEED_LATCH = 0x1C
# smbdis JumpMForceData / FallMForceData / PlayerYSpdData (land 0–4).
JUMP_FORCE_UP = (0x20, 0x20, 0x1E, 0x28, 0x28)
JUMP_FORCE_DOWN = (0x70, 0x70, 0x60, 0x90, 0x90)
JUMP_Y_SPEED = (-4, -4, -4, -5, -5)
JUMP_ABS_VX = (0x09, 0x10, 0x19, 0x1C)
JUMP_SPEED = JUMP_Y_SPEED[0]
MAX_FALL = 4
PIT_Y = 240
DIFF_TO_HALT_JUMP = 1


def idle_action() -> tuple[int, ...]:
    return tuple(int(v) for v in nes_idle_action())


def press(*buttons: str) -> tuple[int, ...]:
    """Build a 9-slot NES action from button names (B, A, LEFT, RIGHT, …)."""
    return tuple(int(v) for v in nes_action(*buttons))


def decode_action(action: Sequence[int]) -> tuple[int, bool, bool]:
    """Return ``(x_dir, run, jump)`` with x_dir in {-1, 0, +1}."""
    from retro_harness.controls import NES_A, NES_B, NES_LEFT, NES_RIGHT

    buttons = list(action) + [0] * 9
    left = bool(buttons[NES_LEFT])
    right = bool(buttons[NES_RIGHT])
    if left and not right:
        x_dir = -1
    elif right and not left:
        x_dir = 1
    else:
        x_dir = 0
    return x_dir, bool(buttons[NES_B]), bool(buttons[NES_A])


def _pack_speed(velocity_x: int, x_force: int) -> int:
    """Unsigned 16-bit word: ``Player_X_Speed`` high, ``Player_X_MoveForce`` low."""
    return ((int(velocity_x) & 0xFF) << 8) | (int(x_force) & 0xFF)


def _unpack_speed(speed_16: int) -> tuple[int, int]:
    word = int(speed_16) & 0xFFFF
    vx_u = word >> 8
    velocity_x = vx_u - 256 if vx_u >= 128 else vx_u
    return velocity_x, word & 0xFF


def _clamp_speed(speed_16: int, run: bool) -> int:
    """Snap ``vx`` to ±max; keep leftover ``xf`` (smbdis does not wipe ``$0705``)."""
    velocity_x, x_force = _unpack_speed(speed_16)
    limit = RUN_MAX if run else WALK_MAX
    if velocity_x >= limit:
        velocity_x = limit
    elif velocity_x < -limit:
        velocity_x = -limit
    return _pack_speed(velocity_x, x_force)


def jump_table_index(velocity_x: int) -> int:
    """smbdis InitJS land index from ``Player_XSpeedAbsolute`` (≈ ``|vx|``)."""
    speed = abs(int(velocity_x))
    for index, limit in enumerate(JUMP_ABS_VX):
        if speed < limit:
            return index
    return len(JUMP_ABS_VX)


def takeoff_vertical(velocity_x: int) -> tuple[int, int, int, int]:
    """Return ``(vy, VerticalForce, VerticalForceDown, Y_MoveForce)`` at takeoff."""
    index = jump_table_index(velocity_x)
    return (
        JUMP_Y_SPEED[index],
        JUMP_FORCE_UP[index],
        JUMP_FORCE_DOWN[index],
        0,
    )


def _phase(player: PlayerPhysics, jump: bool) -> Phase:
    """This frame's motion mode. Takeoff is already air for X_Physics."""
    if player.on_ground and jump and not player.a_held:
        return "takeoff"
    if not player.on_ground:
        return "air"
    return "ground"


def _air_uses_walk_tables(player: PlayerPhysics, jump: bool) -> bool:
    """Air X uses walk accel/max until already at run speed (smbdis X_Physics)."""
    return _phase(player, jump) != "ground" and abs(int(player.velocity_x)) < AIR_RUN_KEEP


def _brake_adder(player: PlayerPhysics) -> int:
    """smbdis FrictionData[$00]: $98 unless RunningSpeed or |vx| >= $21."""
    if player.running_speed or abs(int(player.velocity_x)) >= BRAKE_FAST:
        return FRICTION
    return WALK_ACCEL


def _next_running_speed(player: PlayerPhysics, x_dir: int, jump: bool) -> int:
    """smbdis GetPlayerAnimSpeed: latch |vx| when >= $1C; only on ground."""
    if _phase(player, jump) != "ground":
        return int(player.running_speed)
    abs_vx = abs(int(player.velocity_x))
    if abs_vx >= RUN_SPEED_LATCH:
        return abs_vx
    if x_dir != 0:
        return 0
    return int(player.running_speed)


def _apply_brake(speed_16: int, adder: int) -> int:
    """Add/sub friction; snap through zero (no L/R does not reverse)."""
    velocity_x, _x_force = _unpack_speed(speed_16)
    if velocity_x > 0:
        nxt = (speed_16 - adder) & 0xFFFF
        return 0 if _unpack_speed(nxt)[0] < 0 else nxt
    if velocity_x < 0:
        nxt = (speed_16 + adder) & 0xFFFF
        return 0 if _unpack_speed(nxt)[0] > 0 else nxt
    return speed_16


def _update_x_speed(
    player: PlayerPhysics, x_dir: int, run: bool, jump: bool
) -> tuple[int, int, int]:
    if _air_uses_walk_tables(player, jump):
        run = False
    speed_16 = _pack_speed(player.velocity_x, player.x_force)
    facing = player.facing
    adder = FIRST_KICK if speed_16 == 0 else (RUN_ACCEL if run else WALK_ACCEL)
    if x_dir > 0:
        facing = 1
        speed_16 = _clamp_speed((speed_16 + adder) & 0xFFFF, run)
    elif x_dir < 0:
        facing = 2
        speed_16 = _clamp_speed((speed_16 - adder) & 0xFFFF, run)
    else:
        speed_16 = _apply_brake(speed_16, _brake_adder(player))
    velocity_x, x_force = _unpack_speed(speed_16)
    return velocity_x, x_force, facing


def _advance_x(x: int, sub_x: int, velocity_x: int) -> tuple[int, int]:
    total = (int(x) << 8) + (int(sub_x) & 0xFF) + (int(velocity_x) << 4)
    if total < 0:
        return 0, 0
    return total >> 8, total & 0xFF


def _select_vertical_force(player: PlayerPhysics, jump: bool) -> int:
    """JumpSwimSub: keep rising VF while A is held; else copy VerticalForceDown."""
    rising = int(player.vertical_force)
    falling = int(player.vertical_force_down)
    if player.velocity_y >= 0:
        return falling
    a_held_continuous = jump and player.a_held
    if a_held_continuous:
        return rising
    if (int(player.jump_origin_y) - int(player.y)) >= DIFF_TO_HALT_JUMP:
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
    ground_y: int,
) -> tuple[int, int, int, int, bool]:
    on_ground = False
    if y >= ground_y and velocity_y >= 0:
        y = ground_y
        # Game does not wipe Player_YMF_Dummy ($0416) or VerticalForce ($0709).
        velocity_y = 0
        y_move_force = 0
        on_ground = True
    return y, sub_y, velocity_y, y_move_force, on_ground


def _step_air(
    player: PlayerPhysics, jump: bool, world: World
) -> tuple[int, int, int, int, int, int, bool]:
    """Return y, sub_y, velocity_y, vertical_force, y_move_force, vfd, on_ground."""
    vertical_force = _select_vertical_force(player, jump)
    vertical_force_down = int(player.vertical_force_down)
    y, sub_y, velocity_y, y_move_force = _impose_gravity(
        player.y, player.sub_y, player.velocity_y, player.y_move_force, vertical_force
    )
    y, sub_y, velocity_y, y_move_force, on_ground = _land_if_needed(
        y, sub_y, velocity_y, y_move_force, world.ground_y
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


def step(
    player: PlayerPhysics,
    action: Sequence[int],
    world: World | None = None,
) -> PlayerPhysics:
    """Advance one frame. Pure: no RAM, no I/O, no RNG."""
    floor = world if world is not None else FLAT_1_1
    x_dir, run, jump = decode_action(action)
    velocity_x, x_force, facing = _update_x_speed(player, x_dir, run, jump)
    x, sub_x = _advance_x(player.x, player.sub_x, velocity_x)
    running_speed = _next_running_speed(player, x_dir, jump)
    phase = _phase(player, jump)

    on_ground = player.on_ground
    y = player.y
    sub_y = player.sub_y
    velocity_y = player.velocity_y
    vertical_force = player.vertical_force
    vertical_force_down = player.vertical_force_down
    y_move_force = player.y_move_force
    jump_origin_y = player.jump_origin_y
    if phase == "takeoff":
        jump_origin_y = player.y
        velocity_y, vertical_force, vertical_force_down, y_move_force = takeoff_vertical(
            player.velocity_x
        )
        y, sub_y, velocity_y, y_move_force = _impose_gravity(
            player.y, 0, velocity_y, y_move_force, vertical_force
        )
        on_ground = False
    elif phase == "air":
        (
            y,
            sub_y,
            velocity_y,
            vertical_force,
            y_move_force,
            vertical_force_down,
            on_ground,
        ) = _step_air(player, jump, floor)

    pose = player.pose
    dead = player.dead
    if y >= PIT_Y:
        dead = True
        pose = 0x0B
        y = min(y, 255)

    frame_counter = player.frame_counter
    if frame_counter is not None:
        frame_counter = (int(frame_counter) + 1) & 0xFF

    return PlayerPhysics(
        frame=player.frame + 1,
        x=x,
        y=y,
        pose=pose,
        room=player.room,
        sub_x=sub_x,
        sub_y=sub_y,
        velocity_x=velocity_x,
        velocity_y=velocity_y,
        energy=player.energy,
        dead=dead,
        frame_counter=frame_counter,
        enemy0_active=player.enemy0_active,
        enemy0_type=player.enemy0_type,
        facing=facing,
        on_ground=on_ground,
        x_force=x_force,
        running_speed=running_speed,
        a_held=jump,
        vertical_force=vertical_force,
        vertical_force_down=vertical_force_down,
        y_move_force=y_move_force,
        jump_origin_y=jump_origin_y,
        oper_mode=player.oper_mode,
        timer=player.timer,
    )


def rollout(
    start: PlayerPhysics,
    actions: Sequence[Sequence[int]],
    world: World | None = None,
) -> list[PlayerPhysics]:
    """Return ``[start, step(start, a0), …]`` (length ``len(actions)+1``)."""
    floor = world if world is not None else FLAT_1_1
    frames = [start]
    current = start
    for action in actions:
        current = step(current, action, floor)
        frames.append(current)
    return frames
