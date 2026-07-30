"""Controller-only navigation, combat, pickup, and interaction primitives.

These helpers deliberately operate on observed RAM and controller input.  They
do not write progression state, warp rooms, or assume that every active sprite
slot belongs to the part of a large room currently visible on screen.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

from alttp.ram import AlttpSnapshot
from alttp.startup import action_for, no_action, snapshot_env, step_frames

Buttons: TypeAlias = tuple[str, ...]
ButtonScript: TypeAlias = Sequence[tuple[Buttons, int]]
SnapshotPredicate: TypeAlias = Callable[[AlttpSnapshot], bool]

SPRITE_Y_LOW = 0x0D00
SPRITE_X_LOW = 0x0D10
SPRITE_Y_HIGH = 0x0D20
SPRITE_X_HIGH = 0x0D30
SPRITE_STATE = 0x0DD0
SPRITE_TYPE = 0x0E20
SPRITE_HP = 0x0E50
SPRITE_SLOTS = 16

SPRITE_BLUE_SOLDIER = 0x41
SPRITE_GREEN_SOLDIER = 0x42
SPRITE_KNIGHT = 0x4B
SPRITE_BALL_AND_CHAIN = 0x6A
SPRITE_ZELDA = 0x76
SPRITE_HEART = 0xD8
SPRITE_SMALL_KEY = 0xE4
SPRITE_BIG_KEY = 0xE5
SPRITE_MANTLE = 0xEE

CASTLE_HOSTILE_TYPES = frozenset(
    {
        SPRITE_BLUE_SOLDIER,
        SPRITE_GREEN_SOLDIER,
        SPRITE_KNIGHT,
        SPRITE_BALL_AND_CHAIN,
    }
)

DIRECTION_BUTTONS = frozenset({"UP", "DOWN", "LEFT", "RIGHT"})


@dataclass(frozen=True)
class SpriteSnapshot:
    """One active sprite slot with full 16-bit room coordinates."""

    slot: int
    sprite_type: int
    state: int
    hp: int
    x: int
    y: int

    def distance_to(self, snapshot: AlttpSnapshot) -> int:
        return abs(self.x - snapshot.link_x) + abs(self.y - snapshot.link_y)


@dataclass(frozen=True)
class PrimitiveResult:
    """Observed outcome from a bounded controller primitive."""

    ok: bool
    reason: str
    frames: int
    snapshot: AlttpSnapshot


@dataclass(frozen=True)
class Waypoint:
    """An absolute in-room navigation target."""

    x: int
    y: int
    tolerance: int = 5
    room: int | None = None
    label: str = ""


@dataclass(frozen=True)
class CombatResult:
    """Combat outcome, including the actual slots defeated."""

    ok: bool
    reason: str
    frames: int
    snapshot: AlttpSnapshot
    defeated_slots: tuple[int, ...] = ()


def active_sprites(env: object) -> tuple[SpriteSnapshot, ...]:
    """Read active sprites using the correct 16-bit X/Y coordinates.

    Large indoor rooms can keep sprites for off-screen subrooms active.  The
    high bytes are therefore required; comparing only low bytes aliases those
    sprites onto the current screen and makes combat chase false duplicates.
    """
    ram = np.asarray(env.get_ram(), dtype=np.uint8)  # type: ignore[attr-defined]
    sprites: list[SpriteSnapshot] = []
    for slot in range(SPRITE_SLOTS):
        state = int(ram[SPRITE_STATE + slot])
        if state == 0:
            continue
        sprites.append(
            SpriteSnapshot(
                slot=slot,
                sprite_type=int(ram[SPRITE_TYPE + slot]),
                state=state,
                hp=int(ram[SPRITE_HP + slot]),
                x=(
                    int(ram[SPRITE_X_LOW + slot])
                    | (int(ram[SPRITE_X_HIGH + slot]) << 8)
                ),
                y=(
                    int(ram[SPRITE_Y_LOW + slot])
                    | (int(ram[SPRITE_Y_HIGH + slot]) << 8)
                ),
            )
        )
    return tuple(sprites)


def sprites_of_type(
    env: object,
    sprite_types: Iterable[int],
    *,
    max_distance: int | None = None,
) -> tuple[SpriteSnapshot, ...]:
    """Return active sprites of the requested types, optionally screen-local."""
    wanted = frozenset(int(value) for value in sprite_types)
    snapshot = snapshot_env(env)
    found = tuple(
        sprite for sprite in active_sprites(env) if sprite.sprite_type in wanted
    )
    if max_distance is None:
        return found
    return tuple(
        sprite
        for sprite in found
        if abs(sprite.x - snapshot.link_x) <= max_distance
        and abs(sprite.y - snapshot.link_y) <= max_distance
    )


def run_script(
    env: object,
    script: ButtonScript,
    *,
    stop_when: SnapshotPredicate | None = None,
) -> PrimitiveResult:
    """Run a bounded multi-button script with optional RAM acceptance."""
    frames = 0
    for buttons, hold in script:
        action = (
            no_action()
            if not buttons or buttons == ("NONE",)
            else action_for(*buttons)
        )
        for _ in range(max(0, hold)):
            env.step(action)  # type: ignore[attr-defined]
            frames += 1
            snapshot = snapshot_env(env)
            if stop_when is not None and stop_when(snapshot):
                return PrimitiveResult(True, "acceptance reached", frames, snapshot)
            if snapshot.game_mode == 0x12:
                return PrimitiveResult(False, "Link died", frames, snapshot)
    snapshot = snapshot_env(env)
    return PrimitiveResult(
        stop_when is None or stop_when(snapshot),
        "script complete" if stop_when is None else "acceptance not reached",
        frames,
        snapshot,
    )


def settle_control(env: object, *, max_frames: int = 240) -> PrimitiveResult:
    """Wait for ordinary gameplay control, advancing item/text sequences.

    ``game_mode == 0x07`` alone is not enough: item pickups spend roughly one
    hundred frames in the hold-up pose before opening their message.  Route
    movement must not start while ``link_action == HoldUpItem``.
    """
    frames = 0
    while frames < max_frames:
        snapshot = snapshot_env(env)
        if (
            snapshot.has_control
            and not snapshot.is_text_mode
            and not snapshot.is_hold_up_item
        ):
            return PrimitiveResult(True, "control ready", frames, snapshot)
        if snapshot.game_mode == 0x12:
            return PrimitiveResult(False, "Link died", frames, snapshot)
        if snapshot.is_text_mode:
            button = "A" if (frames // 4) % 2 == 0 else "B"
            step_frames(env, action_for(button), 2)
            step_frames(env, no_action(), 2)
            frames += 4
        else:
            step_frames(env, no_action(), 4)
            frames += 4
    return PrimitiveResult(False, "control timeout", frames, snapshot_env(env))


def move_to(
    env: object,
    waypoint: Waypoint,
    *,
    max_frames: int = 600,
    step_size: int = 3,
    stuck_cycles: int = 28,
) -> PrimitiveResult:
    """Move toward an absolute waypoint with feedback and bounded failure.

    A waypoint is intentionally not a path planner.  Route code supplies
    intermediate points around walls, pits, statues, and ledges.
    """
    frames = 0
    unchanged = 0
    previous_xy: tuple[int, int] | None = None
    while frames < max_frames:
        snapshot = snapshot_env(env)
        if waypoint.room is not None and snapshot.room_base_id != waypoint.room:
            return PrimitiveResult(
                False,
                (
                    f"left room 0x{waypoint.room:02X} before "
                    f"{waypoint.label or 'waypoint'}"
                ),
                frames,
                snapshot,
            )
        dx = waypoint.x - snapshot.link_x
        dy = waypoint.y - snapshot.link_y
        if abs(dx) <= waypoint.tolerance and abs(dy) <= waypoint.tolerance:
            return PrimitiveResult(True, waypoint.label or "waypoint reached", frames, snapshot)
        if snapshot.game_mode == 0x12:
            return PrimitiveResult(False, "Link died", frames, snapshot)

        buttons: list[str] = []
        if abs(dx) > waypoint.tolerance:
            buttons.append("RIGHT" if dx > 0 else "LEFT")
        if abs(dy) > waypoint.tolerance:
            buttons.append("DOWN" if dy > 0 else "UP")
        step_frames(env, action_for(*buttons), step_size)
        frames += step_size

        xy = (snapshot.link_x, snapshot.link_y)
        if xy == previous_xy:
            unchanged += 1
        else:
            unchanged = 0
            previous_xy = xy
        if unchanged >= stuck_cycles:
            return PrimitiveResult(
                False,
                f"stuck before {waypoint.label or 'waypoint'}",
                frames,
                snapshot_env(env),
            )
    return PrimitiveResult(
        False,
        f"timeout before {waypoint.label or 'waypoint'}",
        frames,
        snapshot_env(env),
    )


def move_path(
    env: object,
    waypoints: Sequence[Waypoint],
    *,
    max_frames_per_waypoint: int = 600,
) -> PrimitiveResult:
    """Follow explicit waypoints, stopping on the first failed segment."""
    frames = 0
    for waypoint in waypoints:
        result = move_to(
            env,
            waypoint,
            max_frames=max_frames_per_waypoint,
        )
        frames += result.frames
        if not result.ok:
            return PrimitiveResult(False, result.reason, frames, result.snapshot)
    return PrimitiveResult(True, "path complete", frames, snapshot_env(env))


def move_until(
    env: object,
    buttons: Buttons,
    predicate: SnapshotPredicate,
    *,
    max_frames: int = 600,
    step_size: int = 2,
) -> PrimitiveResult:
    """Hold movement until a RAM predicate succeeds or the bound expires."""
    frames = 0
    action = action_for(*buttons)
    while frames < max_frames:
        snapshot = snapshot_env(env)
        if predicate(snapshot):
            return PrimitiveResult(True, "acceptance reached", frames, snapshot)
        if snapshot.game_mode == 0x12:
            return PrimitiveResult(False, "Link died", frames, snapshot)
        step_frames(env, action, step_size)
        frames += step_size
    snapshot = snapshot_env(env)
    return PrimitiveResult(
        predicate(snapshot),
        "acceptance reached" if predicate(snapshot) else "movement timeout",
        frames,
        snapshot,
    )


def spin_attack(env: object, *, charge_frames: int = 100) -> PrimitiveResult:
    """Charge and release the fighter-sword spin attack."""
    return run_script(
        env,
        (
            (("B",), charge_frames),
            (("NONE",), 24),
        ),
    )


def _face_toward(dx: int, dy: int) -> str:
    if abs(dx) >= abs(dy):
        return "RIGHT" if dx > 0 else "LEFT"
    return "DOWN" if dy > 0 else "UP"


def fight_nearby(
    env: object,
    *,
    sprite_types: Iterable[int] = CASTLE_HOSTILE_TYPES,
    room: int | None = None,
    max_distance: int = 150,
    attack_distance: int = 60,
    stop_when: Callable[[object], bool] | None = None,
    max_cycles: int = 500,
) -> CombatResult:
    """Fight the nearest on-screen target with short movement and sword taps.

    ``stop_when`` should encode progression when a fight produces an item or
    opens a door.  That lets callers stop as soon as the key appears rather
    than chasing an off-screen guard in another subroom.
    """
    wanted = frozenset(int(value) for value in sprite_types)
    start = snapshot_env(env)
    expected_room = start.room_base_id if room is None else room
    frames = 0
    initial_slots = {
        sprite.slot
        for sprite in sprites_of_type(env, wanted, max_distance=max_distance)
    }
    for _ in range(max_cycles):
        snapshot = snapshot_env(env)
        if stop_when is not None and stop_when(env):
            defeated = initial_slots - {
                sprite.slot
                for sprite in sprites_of_type(
                    env, wanted, max_distance=max_distance
                )
            }
            return CombatResult(
                True,
                "combat acceptance reached",
                frames,
                snapshot,
                tuple(sorted(defeated)),
            )
        if snapshot.room_base_id != expected_room:
            return CombatResult(
                False,
                f"left combat room 0x{expected_room:02X}",
                frames,
                snapshot,
            )
        if snapshot.game_mode == 0x12:
            return CombatResult(False, "Link died", frames, snapshot)

        targets = sprites_of_type(env, wanted, max_distance=max_distance)
        if not targets:
            defeated = initial_slots - {
                sprite.slot
                for sprite in sprites_of_type(
                    env, wanted, max_distance=max_distance
                )
            }
            return CombatResult(
                True,
                "no nearby targets",
                frames,
                snapshot,
                tuple(sorted(defeated)),
            )

        target = min(targets, key=lambda sprite: sprite.distance_to(snapshot))
        dx = target.x - snapshot.link_x
        dy = target.y - snapshot.link_y
        if abs(dx) <= attack_distance and abs(dy) <= attack_distance:
            step_frames(env, action_for(_face_toward(dx, dy)), 2)
            step_frames(env, action_for("B"), 4)
            step_frames(env, no_action(), 5)
            frames += 11
            continue

        buttons: list[str] = []
        if abs(dx) > attack_distance - 8:
            buttons.append("RIGHT" if dx > 0 else "LEFT")
        if abs(dy) > attack_distance - 8:
            buttons.append("DOWN" if dy > 0 else "UP")
        step_frames(env, action_for(*buttons), 3)
        frames += 3

    return CombatResult(
        False,
        "combat cycle limit",
        frames,
        snapshot_env(env),
        (),
    )


def collect_nearby(
    env: object,
    sprite_types: Iterable[int],
    *,
    max_distance: int = 180,
    max_frames: int = 480,
) -> PrimitiveResult:
    """Walk directly onto the nearest matching pickup."""
    frames = 0
    wanted = frozenset(int(value) for value in sprite_types)
    while frames < max_frames:
        snapshot = snapshot_env(env)
        items = sprites_of_type(env, wanted, max_distance=max_distance)
        if not items:
            return PrimitiveResult(True, "pickup collected", frames, snapshot)
        target = min(items, key=lambda sprite: sprite.distance_to(snapshot))
        dx = target.x - snapshot.link_x
        dy = target.y - snapshot.link_y
        buttons: list[str] = []
        if abs(dx) > 3:
            buttons.append("RIGHT" if dx > 0 else "LEFT")
        if abs(dy) > 3:
            buttons.append("DOWN" if dy > 0 else "UP")
        step_frames(
            env,
            no_action() if not buttons else action_for(*buttons),
            3,
        )
        frames += 3
        if snapshot.game_mode == 0x12:
            return PrimitiveResult(False, "Link died", frames, snapshot)
    return PrimitiveResult(False, "pickup timeout", frames, snapshot_env(env))


def interact_until(
    env: object,
    predicate: SnapshotPredicate,
    *,
    max_cycles: int = 80,
) -> PrimitiveResult:
    """Alternate A/B interactions and text advancement until accepted."""
    frames = 0
    for cycle in range(max_cycles):
        snapshot = snapshot_env(env)
        if predicate(snapshot):
            return PrimitiveResult(True, "interaction accepted", frames, snapshot)
        if snapshot.game_mode == 0x12:
            return PrimitiveResult(False, "Link died", frames, snapshot)
        button = "A" if cycle % 2 == 0 else "B"
        step_frames(env, action_for(button), 2)
        step_frames(env, no_action(), 3)
        frames += 5
    snapshot = snapshot_env(env)
    return PrimitiveResult(
        predicate(snapshot),
        "interaction accepted" if predicate(snapshot) else "interaction timeout",
        frames,
        snapshot,
    )
