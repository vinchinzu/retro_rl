"""Generic room clear + door exit engine (data-driven from maps/*.json).

SM-style small surface for agents: load a map, clear hostiles, follow a
typed door path, push the edge. Segments stay thin wrappers.

**Isolated edge success:** :func:`run_room_edge` returns ``ok=True`` when the
door destination is reached. Multi-hop segments (e.g. main hall → Zelda)
compose edges and decide their own acceptance.

Prefer this over new per-room 400-line scripts for B1 work.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from alttp import primitives
from alttp.dungeon.nav import (
    DoorLine,
    WalkCommit,
    door_line,
    find_path,
    room_origin,
    room_step,
)
from alttp.dungeon.tilemap import read_room_tiles
from alttp.ram import (
    AlttpSnapshot,
    read_u16,
    room_label,
    snapshot_to_diag,
    zelda_rescued_accepted,
)
from alttp.rollout import PolicyGuard, strike_if_safe
from alttp.room_map import ClearPolicy, KnownDoor, RoomMap, load_room_map
from alttp.room_sense import detect_edge, enemy_boxes, path_blocked_by_enemies
from alttp.route_report import RoutePhaseResult, SegmentResult
from alttp.sprite_ids import vulnerability
from alttp.startup import action_for, no_action, snapshot_env, step_frames
from alttp.walk_physics import LINK_BODY
from retro_harness.input_script import FrameAction

# tilemap_location_calc_mask. CastleMain read $01F8.
_TILE_MASK = 0x00EC
# Door table. A shutter slot is the only type that requires a clear.
# 0x18/0x44/0x48/0x4A are snesrev shutters; 0x36 remaps to one. Object
# 0x18 on room 0x80 is a cell lock and is not in this table.
_DOOR_TYPE = 0x1980
_DOOR_POS = 0x19A0
_DOOR_DIR = 0x19C0
_SHUTTER_TYPES = frozenset({0x18, 0x36, 0x44, 0x48, 0x4A})
# Slot direction is the ROM byte, not the map's LEFT/RIGHT/UP/DOWN.
_SLOT_DIR = {"UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3}


def in_room(snap: AlttpSnapshot, room: int) -> bool:
    return snap.indoors and (not snap.dark_world) and snap.room_base_id == room


def in_origin(snap: AlttpSnapshot, room_map: RoomMap) -> bool:
    """True when Link is at the mapped indoor room or outdoor screen."""
    if room_map.outdoors:
        return (
            not snap.indoors
            and not snap.dark_world
            and room_map.screen_id is not None
            and snap.screen_id == room_map.screen_id
        )
    return in_room(snap, room_map.room_base_id)


def at_door_destination(snap: AlttpSnapshot, door: KnownDoor) -> bool:
    """True when snapshot matches door exit (room or outdoors)."""
    if door.outdoors:
        return not snap.indoors and (
            door.screen_id is None or snap.screen_id == door.screen_id
        )
    if door.to_room is None:
        return False
    return in_room(snap, door.to_room)


# Dungeon submodule 14 (stair anim) can idle ~292–400f after dest room-change.
# Only dest settle uses this; primitives.settle_control default stays 240.
DEST_SETTLE_MAX_FRAMES = 480


def _masked_tile(value: int, mask: int) -> int:
    """Room-local pixel, floored to the 8 px phase ``find_path`` walks."""
    return int(value) & int(mask) & ~7


def _cardinal_toward(snap: AlttpSnapshot, wp: primitives.Waypoint) -> str | None:
    """One axis toward ``wp``. Inside tolerance that axis stays up."""
    dx = wp.x - snap.link_x
    dy = wp.y - snap.link_y
    use_x = abs(dx) > wp.tolerance
    use_y = abs(dy) > wp.tolerance
    if use_x and (not use_y or abs(dx) >= abs(dy)):
        return "RIGHT" if dx > 0 else "LEFT"
    if use_y:
        return "DOWN" if dy > 0 else "UP"
    return None


def _segment_button(ax: int, ay: int, bx: int, by: int) -> str | None:
    dx, dy = bx - ax, by - ay
    if abs(dx) >= abs(dy) and dx != 0:
        return "RIGHT" if dx > 0 else "LEFT"
    if dy != 0:
        return "DOWN" if dy > 0 else "UP"
    return None


def _short_of(x: int, y: int, corner: tuple[int, int], button: str) -> bool:
    """True while ``button`` still aims at ``corner`` rather than back past it."""
    if button == "LEFT":
        return x > corner[0]
    if button == "RIGHT":
        return x < corner[0]
    if button == "UP":
        return y > corner[1]
    if button == "DOWN":
        return y < corner[1]
    return False


def _path_segments(
    env: object, snap: AlttpSnapshot, wp: primitives.Waypoint
) -> list[tuple[str, tuple[int, int]]]:
    """World-pixel corners of one fewest-turn path. Empty when none exists."""
    ram = np.asarray(env.get_ram(), dtype=np.uint8)  # type: ignore[attr-defined]
    mask = read_u16(ram, _TILE_MASK)
    local_x = int(snap.link_x) & mask
    local_y = int(snap.link_y) & mask
    origin = (int(snap.link_x) - local_x, int(snap.link_y) - local_y)
    start = (local_x & ~7, local_y & ~7)
    # A 9-frame commit can carry Link ~16 px past a corner. Don't plan a
    # step back onto an axis that is already that close.
    gx, gy = wp.x, wp.y
    if abs(int(snap.link_x) - gx) <= max(wp.tolerance, 16):
        gx = int(snap.link_x)
    if abs(int(snap.link_y) - gy) <= max(wp.tolerance, 16):
        gy = int(snap.link_y)
    goal = (_masked_tile(gx, mask), _masked_tile(gy, mask))
    path = find_path(read_room_tiles(ram, snap.layer), start, goal, LINK_BODY)
    if not path:
        return []
    world = [(origin[0] + x, origin[1] + y) for x, y in path]
    segments: list[tuple[str, tuple[int, int]]] = []
    for left, right in zip(world, world[1:]):
        button = _segment_button(left[0], left[1], right[0], right[1])
        if button is None:
            continue
        if segments and segments[-1][0] == button:
            segments[-1] = (button, right)
        else:
            segments.append((button, right))
    return segments


def _walk_waypoint(
    env: object,
    wp: primitives.Waypoint,
    *,
    room: int,
    commit: WalkCommit,
    max_frames: int = 500,
) -> tuple[primitives.PrimitiveResult, WalkCommit]:
    """Follow a fewest-turn tile path and commit to each heading.

    A segment is held only until Link passes its corner. ``commit`` survives
    across waypoints so the next leg cannot reverse inside the flutter window.
    """
    frames = 0
    unchanged = 0
    previous: tuple[int, int] | None = None
    snap = snapshot_env(env)
    segments = _path_segments(env, snap, wp)
    index = 0
    replanned = False
    while frames < max_frames:
        snap = snapshot_env(env)
        if wp.room is not None and snap.room_base_id != wp.room:
            return primitives.PrimitiveResult(
                False,
                f"left room 0x{wp.room:02X} before {wp.label or 'waypoint'}",
                frames,
                snap,
            ), commit
        if wp.reached(snap.link_x, snap.link_y):
            return primitives.PrimitiveResult(
                True, wp.label or "waypoint reached", frames, snap
            ), commit
        if snap.game_mode == 0x12:
            return primitives.PrimitiveResult(False, "Link died", frames, snap), commit
        while index < len(segments) and not _short_of(
            snap.link_x, snap.link_y, segments[index][1], segments[index][0]
        ):
            index += 1
        if index < len(segments):
            desired: str | None = segments[index][0]
        else:
            desired = _cardinal_toward(snap, wp)
        button, commit = room_step(commit, desired)
        step_frames(env, no_action() if button is None else action_for(button), 1)
        frames += 1
        xy = (snap.link_x, snap.link_y)
        if xy == previous:
            unchanged += 1
        else:
            unchanged = 0
            previous = xy
        if unchanged == 24 and not replanned:
            segments = _path_segments(env, snapshot_env(env), wp)
            index = 0
            replanned = True
            unchanged = 0
        elif unchanged >= 48:
            return primitives.PrimitiveResult(
                False,
                f"stuck before {wp.label or 'waypoint'}",
                frames,
                snapshot_env(env),
            ), commit
    return primitives.PrimitiveResult(
        False,
        f"timeout before {wp.label or 'waypoint'}",
        frames,
        snapshot_env(env),
    ), commit


def move_path_combat_aware(
    env: object,
    waypoints: list[primitives.Waypoint],
    *,
    room: int,
    policy: ClearPolicy,
    walker: bool = False,
) -> primitives.PrimitiveResult:
    """Follow waypoints; skirmish when a hostile box blocks the next segment.

    ``walker`` follows ``find_path`` and holds each heading for
    ``COMMIT_FRAMES`` instead of retargeting every few frames.
    """
    frames = 0
    commit = WalkCommit()
    for wp in waypoints:
        blocked = path_blocked_by_enemies(
            env,
            wp.x,
            wp.y,
            pad=policy.skirmish_pad,
            max_distance=policy.skirmish_max_distance,
        )
        if blocked is not None:
            fight = primitives.fight_nearby(
                env,
                room=room,
                max_distance=policy.skirmish_max_distance,
                attack_distance=min(48, policy.attack_distance),
                max_cycles=policy.skirmish_max_cycles,
            )
            frames += fight.frames
            sc = primitives.settle_control(env, max_frames=60)
            frames += sc.frames
            if sc.snapshot.game_mode == 0x12:
                return primitives.PrimitiveResult(
                    False, "Link died during corridor skirmish", frames, sc.snapshot
                )
            if sc.snapshot.room_base_id != room:
                return primitives.PrimitiveResult(
                    True, "left room during skirmish", frames, sc.snapshot
                )

        # 0x84 in room 0x55 is not a legend floor, so find_path is empty and
        # the commit walker bonks. Axis move_to is the exit that still works.
        point = load_room_map(f"room_{room:02x}").point(wp.label)
        use_walker = (walker and (point is None or point.role != "axis")
                      and bool(_path_segments(env, snapshot_env(env), wp)))
        if use_walker:
            move, commit = _walk_waypoint(env, wp, room=room, commit=commit)
        else:
            if point is not None and point.role == "axis":
                move = primitives.move_to(env, wp, step_size=1, max_frames=500)
            else:
                move = primitives.move_to(env, wp, max_frames=500)
            if walker:
                commit = WalkCommit()
        frames += move.frames
        snap = move.snapshot
        if snap.game_mode == 0x12:
            return primitives.PrimitiveResult(
                False, f"Link died before {wp.label or 'waypoint'}", frames, snap
            )
        if snap.room_base_id != room:
            return primitives.PrimitiveResult(
                True,
                f"left room during {wp.label or 'waypoint'}",
                frames,
                snap,
            )
        if not move.ok:
            return primitives.PrimitiveResult(
                False,
                f"stuck before {wp.label or 'waypoint'}: {move.reason}",
                frames,
                snap,
            )
    return primitives.PrimitiveResult(True, "path complete", frames, snapshot_env(env))


def door_table_requires_clear(ram: np.ndarray) -> bool:
    """True when a live door slot is a shutter. Open and keyed slots are not."""
    for slot in range(16):
        typ = read_u16(ram, _DOOR_TYPE + 2 * slot)
        pos = read_u16(ram, _DOOR_POS + 2 * slot)
        direc = read_u16(ram, _DOOR_DIR + 2 * slot)
        if typ == 0 and pos == 0 and direc == 0:
            continue
        if (typ & 0xFF) in _SHUTTER_TYPES:
            return True
    return False


def _clear_targets(env: object, snap: AlttpSnapshot, max_distance: int) -> list:
    """Hostiles on Link's layer inside ``max_distance``. Pickups are not targets."""
    found = []
    for sprite in primitives.active_sprites(env):
        if sprite.hp <= 0 or sprite.floor != snap.layer:
            continue
        try:
            row = vulnerability(sprite.sprite_type)
        except (TypeError, ValueError):
            continue
        if row.contact_damage[0] <= 0:
            continue
        if row.ignores_damage and row.contact_class == 0:
            continue
        dist = max(abs(sprite.x - snap.link_x), abs(sprite.y - snap.link_y))
        if dist > max_distance:
            continue
        found.append(sprite)
    return found


class _HostileBfs:
    """One BFS step toward the nearest hostile. The strike is not this step."""

    def __init__(self, room: int, policy: ClearPolicy) -> None:
        self.room = room
        self.policy = policy
        self.commit = WalkCommit()
        self.max_frames = max(1, policy.max_cycles) * 12
        self.success = False
        self.failed = False
        self._env: object | None = None

    def bind_env(self, env: object) -> None:
        self._env = env

    def step(self, snap: AlttpSnapshot) -> FrameAction:
        env = self._env
        targets = _clear_targets(env, snap, self.policy.max_distance)
        if not targets:
            self.success = True
            return FrameAction(list(no_action()), "no targets")
        target = min(
            targets,
            key=lambda sprite: max(
                abs(sprite.x - snap.link_x), abs(sprite.y - snap.link_y)
            ),
        )
        wp = primitives.Waypoint(
            target.x, target.y, tolerance=8, room=self.room, label="hostile"
        )
        segments = _path_segments(env, snap, wp)
        desired = segments[0][0] if segments else _cardinal_toward(snap, wp)
        button, self.commit = room_step(self.commit, desired)
        action = no_action() if button is None else action_for(button)
        return FrameAction([int(value) for value in action], "bfs")


def _play_action(env: object, action: object) -> None:
    step_frames(env, np.asarray(action, dtype=np.int8), 1)


def _clear_shutter(
    env: object,
    room: int,
    policy: ClearPolicy,
    already_past: Callable[[AlttpSnapshot], bool] | None,
) -> tuple[int, str, tuple[int, ...], bool]:
    """BFS toward each hostile and commit ``strike_if_safe`` under ``PolicyGuard``."""
    snap = snapshot_env(env)
    before = {sprite.slot for sprite in _clear_targets(env, snap, policy.max_distance)}
    walker = _HostileBfs(room, policy)
    guard = PolicyGuard(walker)
    guard.bind_env(env)
    frames = 0
    reason = "combat cycle limit"
    ok = False
    while frames < walker.max_frames:
        snap = snapshot_env(env)
        if already_past is not None and already_past(snap):
            reason, ok = "already past room", True
            break
        if snap.game_mode == 0x12:
            reason = "Link died"
            break
        if not in_room(snap, room):
            reason = f"left room 0x{room:02X}"
            break
        targets = _clear_targets(env, snap, policy.max_distance)
        if not targets:
            reason, ok = "no nearby targets", True
            break
        target = min(
            targets,
            key=lambda sprite: max(
                abs(sprite.x - snap.link_x), abs(sprite.y - snap.link_y)
            ),
        )
        if target.hit_timer & 0x80:
            _play_action(env, no_action())
            frames += 1
            continue
        dist = max(abs(target.x - snap.link_x), abs(target.y - snap.link_y))
        if dist <= policy.attack_distance:
            script = strike_if_safe(env, target.slot)
            if script:
                for act in script:
                    _play_action(env, act.action)
                frames += len(script)
                continue
        _play_action(env, guard.step(snap).action)
        frames += 1
    else:
        reason = "combat cycle limit"
    snap = snapshot_env(env)
    left = {sprite.slot for sprite in _clear_targets(env, snap, policy.max_distance)}
    return frames, reason, tuple(sorted(before - left)), ok


def clear_room(
    env: object,
    room_map: RoomMap,
    *,
    phase: str = "clear_room",
    already_past: Callable[[AlttpSnapshot], bool] | None = None,
) -> RoutePhaseResult:
    """Clear only when the door table has a shutter. Otherwise walk alive.

    A required clear approaches with BFS under ``PolicyGuard`` and swings
    with ``strike_if_safe``. Open, keyed, and stair slots do not start it.
    """
    room = room_map.room_base_id
    policy = room_map.clear_policy
    frames = 0
    settle = primitives.settle_control(env)
    frames += settle.frames
    start = settle.snapshot

    if already_past is not None and already_past(start):
        return RoutePhaseResult(
            phase=phase,
            ok=True,
            frames=frames,
            snapshot=start,
            detail="already past room",
            diag=snapshot_to_diag(start),
        )
    if zelda_rescued_accepted(start):
        return RoutePhaseResult(
            phase=phase,
            ok=True,
            frames=frames,
            snapshot=start,
            detail="zelda already rescued",
            diag=snapshot_to_diag(start),
        )
    if not in_room(start, room):
        return RoutePhaseResult(
            phase=phase,
            ok=False,
            frames=frames,
            snapshot=start,
            detail=(
                f"expected room 0x{room:02X}, got "
                f"{room_label(start.room_base_id)} indoors={start.indoors}"
            ),
            diag=snapshot_to_diag(start),
        )

    ram = np.asarray(env.get_ram(), dtype=np.uint8)  # type: ignore[attr-defined]
    if not door_table_requires_clear(ram):
        return RoutePhaseResult(
            phase=phase,
            ok=True,
            frames=frames,
            snapshot=start,
            detail=(
                "walked alive; clearPolicy.required=false "
                "(door table has no shutter)"
            ),
            diag=snapshot_to_diag(start),
        )

    boxes_before = enemy_boxes(env, max_distance=policy.max_distance + 100)
    fought, reason, defeated, fight_ok = _clear_shutter(
        env, room, policy, already_past
    )
    frames += fought
    settle2 = primitives.settle_control(env, max_frames=120)
    frames += settle2.frames
    snap = settle2.snapshot
    remaining = enemy_boxes(env, max_distance=120)
    past = already_past(snap) if already_past is not None else False
    ok = (
        (in_room(snap, room) or past)
        and snap.game_mode != 0x12
        and (fight_ok or reason == "no nearby targets" or len(remaining) == 0)
    )
    return RoutePhaseResult(
        phase=phase,
        ok=ok,
        frames=frames,
        snapshot=snap,
        detail=(
            f"clearPolicy.required=true before={len(boxes_before)} "
            f"defeated={defeated} remaining_near={len(remaining)} "
            f"xy=({snap.link_x},{snap.link_y}) reason={reason}"
        ),
        diag={
            **snapshot_to_diag(snap),
            "enemyBoxesBefore": [b.to_dict() for b in boxes_before],
            "enemyBoxes": [b.to_dict() for b in remaining],
            "combatReason": reason,
        },
    )


def _align_to_door(
    env: object, door: KnownDoor, room: int
) -> primitives.PrimitiveResult:
    """Line up on the live mouth when the map door has no approachXy.

    Picks the matching-direction slot whose centre is closest to Link.
    Unmeasured door types are skipped. No RAM writes.
    """
    snap = snapshot_env(env)
    ram = np.asarray(env.get_ram(), dtype=np.uint8)  # type: ignore[attr-defined]
    origin = room_origin(snap.link_x, snap.link_y)
    tiles = read_room_tiles(ram, snap.layer)
    want = _SLOT_DIR.get(door.direction)
    best: DoorLine | None = None
    best_dist: int | None = None
    if want is not None:
        for slot in range(8):
            typ = read_u16(ram, _DOOR_TYPE + 2 * slot)
            pos = read_u16(ram, _DOOR_POS + 2 * slot)
            direc = read_u16(ram, _DOOR_DIR + 2 * slot)
            if typ == 0 and pos == 0 and direc == 0:
                continue
            if (direc & 0xFF) != want:
                continue
            line = door_line(tiles, pos, direc & 0xFF, typ & 0xFF, origin)
            if line is None:
                continue
            here = snap.link_y if line.axis == "y" else snap.link_x
            dist = abs(here - line.centre)
            if best_dist is None or dist < best_dist:
                best, best_dist = line, dist
    if best is None:
        return primitives.PrimitiveResult(
            False, "no measured doorway for this door", 0, snap
        )
    low, high = ("LEFT", "RIGHT") if best.axis == "x" else ("UP", "DOWN")
    frames = 0
    unchanged = 0
    previous: tuple[int, int] | None = None
    while frames < 180:
        snap = snapshot_env(env)
        if not in_room(snap, room):
            return primitives.PrimitiveResult(
                at_door_destination(snap, door),
                "left during line-up",
                frames,
                snap,
            )
        here = snap.link_y if best.axis == "y" else snap.link_x
        delta = best.centre - here
        if abs(delta) <= best.tolerance:
            return primitives.PrimitiveResult(
                True,
                (
                    f"lined {best.axis}={here} centre={best.centre} "
                    f"tol={best.tolerance} type=0x{best.door_type:02X}"
                ),
                frames,
                snap,
            )
        step_frames(env, action_for(high if delta > 0 else low), 1)
        frames += 1
        snap = snapshot_env(env)
        xy = (snap.link_x, snap.link_y)
        unchanged = unchanged + 1 if xy == previous else 0
        previous = xy
        if unchanged >= 16:
            return primitives.PrimitiveResult(
                False, f"stuck lining up {best.axis} at {xy}", frames, snap
            )
    snap = snapshot_env(env)
    return primitives.PrimitiveResult(
        False, "line-up timeout", frames, snap
    )


def _open_lifted_shutter(
    env: object, room: int, policy: ClearPolicy
) -> primitives.PrimitiveResult:
    """Walk onto a lifted pot's switch, wait out the shutter, clear the hall.

    The lift point is the floor south of the pot. Holding UP contacts it.
    The map clear distance stays put so the closed shutter is not a long
    cardinal bonk. Hall popos are fought after the crossing, south of y=6248.
    """
    frames = 0
    still = 0
    prev: tuple[int, int] | None = None
    for _ in range(36):
        snap = snapshot_env(env)
        if snap.room_base_id != room:
            return primitives.PrimitiveResult(True, "left before lift", frames, snap)
        step_frames(env, action_for("UP"), 1)
        frames += 1
        snap = snapshot_env(env)
        xy = (snap.link_x, snap.link_y)
        still = still + 1 if xy == prev else 0
        prev = xy
        if still >= 4:
            break
    lifted = primitives.lift(env)
    frames += lifted.frames
    if not lifted.ok:
        return primitives.PrimitiveResult(False, lifted.reason, frames, lifted.snapshot)
    thrown = primitives.throw(env)
    frames += thrown.frames
    if not thrown.ok:
        return primitives.PrimitiveResult(False, thrown.reason, frames, thrown.snapshot)
    for _ in range(48):
        snap = snapshot_env(env)
        if snap.room_base_id != room or snap.submodule != 0:
            break
        step_frames(env, action_for("UP"), 1)
        frames += 1
    for _ in range(240):
        snap = snapshot_env(env)
        if snap.room_base_id != room or (snap.game_mode == 0x07 and snap.submodule == 0):
            break
        step_frames(env, no_action(), 1)
        frames += 1
    snap = snapshot_env(env)
    if snap.room_base_id != room:
        return primitives.PrimitiveResult(True, "left during shutter", frames, snap)
    # Passing the center shutter closes the north mouth. It opens again once
    # the hall popos are dead. Stay south of y=6248 so the sword range does
    # not start the push already wedged in the mouth.
    reason = "popo clear timeout"
    ok = False
    for _ in range(max(700, policy.max_cycles)):
        snap = snapshot_env(env)
        if snap.room_base_id != room:
            return primitives.PrimitiveResult(True, "left during popo clear", frames, snap)
        if snap.game_mode == 0x12:
            return primitives.PrimitiveResult(False, "Link died", frames, snap)
        if snap.submodule != 0 or snap.game_mode != 0x07:
            step_frames(env, no_action(), 1)
            frames += 1
            continue
        alive = [
            sp
            for sp in primitives.active_sprites(env)
            if sp.sprite_type == 0x4F and sp.hp > 0 and sp.floor == snap.layer
        ]
        if not alive:
            reason, ok = "popos down", True
            break
        target = min(
            alive,
            key=lambda sp: max(abs(sp.x - snap.link_x), abs(sp.y - snap.link_y)),
        )
        dist = max(abs(target.x - snap.link_x), abs(target.y - snap.link_y))
        if dist <= 32:
            script = strike_if_safe(env, target.slot)
            if script:
                for act in script:
                    step_frames(env, np.asarray(act.action, dtype=np.int8), 1)
                frames += len(script)
                continue
        if snap.link_y < 6248:
            step_frames(env, action_for("DOWN"), 1)
        elif abs(snap.link_x - target.x) > 6:
            step_frames(env, action_for("LEFT" if snap.link_x > target.x else "RIGHT"), 1)
        else:
            step_frames(env, action_for("UP"), 1)
        frames += 1
    snap = snapshot_env(env)
    return primitives.PrimitiveResult(ok or snap.room_base_id != room, reason, frames, snap)


def _leave_path(
    *,
    phase_name: str,
    frames: int,
    snap: AlttpSnapshot,
    door: KnownDoor,
    detail: str,
) -> RoutePhaseResult:
    """Stop a door path when the edge is already done or no longer walkable."""
    return RoutePhaseResult(
        phase=phase_name,
        ok=at_door_destination(snap, door) and snap.game_mode != 0x12,
        frames=frames,
        snapshot=snap,
        detail=detail,
        diag=snapshot_to_diag(snap),
    )


def exit_via_door(
    env: object,
    room_map: RoomMap,
    door: KnownDoor | str,
    *,
    phase: str | None = None,
) -> RoutePhaseResult:
    """Navigate door path (combat-aware) and hold direction through the edge."""
    d = room_map.door(door) if isinstance(door, str) else door
    if d is None:
        snap = snapshot_env(env)
        return RoutePhaseResult(
            phase=phase or "exit_door",
            ok=False,
            frames=0,
            snapshot=snap,
            detail=f"unknown door {door!r}",
            diag=snapshot_to_diag(snap),
        )
    room = room_map.room_base_id
    policy = room_map.clear_policy
    phase_name = phase or f"exit_{d.label}"
    frames = 0
    settle = primitives.settle_control(env)
    frames += settle.frames
    start = settle.snapshot

    def failed(pose: AlttpSnapshot, detail: str) -> RoutePhaseResult:
        return RoutePhaseResult(
            phase=phase_name,
            ok=False,
            frames=frames,
            snapshot=pose,
            detail=detail,
            diag=snapshot_to_diag(pose),
        )

    if at_door_destination(start, d):
        return RoutePhaseResult(
            phase=phase_name,
            ok=True,
            frames=frames,
            snapshot=start,
            detail=(
                f"already at door dest "
                f"{room_label(start.room_base_id)} xy=({start.link_x},{start.link_y})"
            ),
            diag=snapshot_to_diag(start),
        )
    if room_map.outdoors:
        if room_map.screen_id is None:
            return failed(start, "outdoor map has no screen id")
        if not in_origin(start, room_map):
            return failed(
                start,
                f"not on outdoor screen 0x{room_map.screen_id:02X} "
                f"(got 0x{start.screen_id:02X}, indoors={int(start.indoors)})",
            )
        # Reuse the loaded-area outdoor tile walker. It validates that this
        # mapped target belongs to the loaded screen before following it.
        from alttp.overworld import walk_to_screen

        outdoor_waypoints = room_map.waypoints_for_door(d)
        if not outdoor_waypoints:
            return failed(start, f"outdoor door {d.label} has no mapped path")
        for x, y, label, tolerance in outdoor_waypoints:
            approach = walk_to_screen(
                env,
                room_map.screen_id,
                target_xy=(x, y),
                tolerance=min(tolerance, 4),
                guard=True,
            )
            frames += int(approach.get("frames", 0))
            snap = snapshot_env(env)
            if not approach.get("ok") or not in_origin(snap, room_map):
                return failed(
                    snap,
                    f"outdoor waypoint {label} failed: "
                    f"{approach.get('reason', 'left mapped screen')}",
                )
    elif not in_room(start, room):
        return RoutePhaseResult(
            phase=phase_name,
            ok=False,
            frames=frames,
            snapshot=start,
            detail=f"not in room 0x{room:02X} ({room_label(start.room_base_id)})",
            diag=snapshot_to_diag(start),
        )

    wps: list[primitives.Waypoint] = []
    if not room_map.outdoors:
        for x, y, label, tol in room_map.waypoints_for_door(d):
            wps.append(primitives.Waypoint(x, y, tolerance=tol, room=room, label=label))

    lift_wps = [
        wp
        for wp in wps
        if (pt := room_map.point(wp.label)) is not None and pt.role == "lift"
    ]
    if lift_wps:
        path = move_path_combat_aware(
            env, lift_wps, room=room, policy=policy, walker=True
        )
        frames += path.frames
        snap = path.snapshot
        if (
            at_door_destination(snap, d)
            or snap.game_mode == 0x12
            or not in_room(snap, room)
            or not path.ok
        ):
            return _leave_path(
                phase_name=phase_name,
                frames=frames,
                snap=snap,
                door=d,
                detail=path.reason,
            )
        opened = _open_lifted_shutter(env, room, policy)
        frames += opened.frames
        snap = opened.snapshot
        if (
            at_door_destination(snap, d)
            or snap.game_mode == 0x12
            or not in_room(snap, room)
            or not opened.ok
        ):
            return _leave_path(
                phase_name=phase_name,
                frames=frames,
                snap=snap,
                door=d,
                detail=opened.reason,
            )
        skip = {wp.label for wp in lift_wps}
        wps = [wp for wp in wps if wp.label not in skip]

    if wps:
        path = move_path_combat_aware(
            env, wps, room=room, policy=policy, walker=True
        )
        frames += path.frames
        snap = path.snapshot
        if at_door_destination(snap, d):
            return RoutePhaseResult(
                phase=phase_name,
                ok=True,
                frames=frames,
                snapshot=snap,
                detail=f"left during path: {path.reason}",
                diag=snapshot_to_diag(snap),
            )
        if snap.game_mode == 0x12:
            return RoutePhaseResult(
                phase=phase_name,
                ok=False,
                frames=frames,
                snapshot=snap,
                detail=path.reason,
                diag=snapshot_to_diag(snap),
            )
        if not in_room(snap, room):
            return RoutePhaseResult(
                phase=phase_name,
                ok=False,
                frames=frames,
                snapshot=snap,
                detail=(
                    f"left wrong exit → {room_label(snap.room_base_id)} "
                    f"indoors={snap.indoors}"
                ),
                diag=snapshot_to_diag(snap),
            )
        if not path.ok:
            recovery_wps = [
                primitives.Waypoint(x, y, tolerance=tol, room=room, label=label)
                for x, y, label, tol in room_map.recovery_waypoints_for_door(d)
            ]
            if not recovery_wps:
                return RoutePhaseResult(
                    phase=phase_name,
                    ok=False,
                    frames=frames,
                    snapshot=snap,
                    detail=path.reason,
                    diag=snapshot_to_diag(snap),
                )
            recovery = move_path_combat_aware(
                env, recovery_wps, room=room, policy=policy, walker=True
            )
            frames += recovery.frames
            snap = recovery.snapshot
            if at_door_destination(snap, d):
                return RoutePhaseResult(
                    phase=phase_name,
                    ok=True,
                    frames=frames,
                    snapshot=snap,
                    detail=f"left during recovery path: {recovery.reason}",
                    diag=snapshot_to_diag(snap),
                )
            if snap.game_mode == 0x12 or not in_room(snap, room) or not recovery.ok:
                return RoutePhaseResult(
                    phase=phase_name,
                    ok=False,
                    frames=frames,
                    snapshot=snap,
                    detail=(
                        f"primary path failed ({path.reason}); "
                        f"recovery path failed ({recovery.reason})"
                    ),
                    diag=snapshot_to_diag(snap),
                )

    line_note = ""
    if d.approach_xy is None:
        lined = _align_to_door(env, d, room)
        frames += lined.frames
        snap = lined.snapshot
        line_note = lined.reason
        if at_door_destination(snap, d):
            return RoutePhaseResult(
                phase=phase_name,
                ok=True,
                frames=frames,
                snapshot=snap,
                detail=line_note,
                diag=snapshot_to_diag(snap),
            )
        if snap.game_mode == 0x12 or not in_room(snap, room) or not lined.ok:
            return RoutePhaseResult(
                phase=phase_name,
                ok=False,
                frames=frames,
                snapshot=snap,
                detail=line_note,
                diag=snapshot_to_diag(snap),
            )

    # Hold door direction through transition.
    before = snapshot_env(env)
    push_limit = max(
        1,
        (d.push_frames + 2) // 3 if room_map.outdoors else d.push_frames // 3,
    )
    for _ in range(push_limit):
        step_frames(env, action_for(d.direction), 3)
        frames += 3
        snap = snapshot_env(env)
        if at_door_destination(snap, d) or not in_origin(snap, room_map):
            edge = detect_edge(
                before,
                snap,
                expected_room=room,
                frames=frames,
                label=d.label,
                preferred_direction=d.direction,
            )
            ok = at_door_destination(snap, d)
            return RoutePhaseResult(
                phase=phase_name,
                ok=ok,
                frames=frames,
                snapshot=snap,
                detail=(
                    f"{d.direction} push → {room_label(snap.room_base_id)} "
                    f"xy=({snap.link_x},{snap.link_y})"
                    + (f"; {line_note}" if line_note else "")
                ),
                diag={
                    **snapshot_to_diag(snap),
                    "edge": edge.to_dict() if edge else None,
                    "door": d.label,
                    "doorLine": line_note or None,
                },
            )

    snap = snapshot_env(env)
    if not snap.has_control and not snap.indoors:
        # Outdoor entrances may spend the final push frames in an overworld
        # transition module before the indoor room id becomes visible.
        settled = primitives.settle_control(env, max_frames=DEST_SETTLE_MAX_FRAMES)
        frames += settled.frames
        snap = settled.snapshot
        if settled.ok and at_door_destination(snap, d):
            return RoutePhaseResult(
                phase=phase_name,
                ok=True,
                frames=frames,
                snapshot=snap,
                detail=f"{d.direction} push settled in {room_label(snap.room_base_id)}",
                diag=snapshot_to_diag(snap),
            )
    return RoutePhaseResult(
        phase=phase_name,
        ok=False,
        frames=frames,
        snapshot=snap,
        detail=f"door push failed ({d.label}) xy=({snap.link_x},{snap.link_y})",
        diag=snapshot_to_diag(snap),
    )


def run_room_edge(
    env: object,
    map_id: str,
    door_label: str,
    *,
    clear: bool = True,
    source: str = "state_load_dev",
    notes: list[str] | None = None,
    acceptance_fn: Callable[[AlttpSnapshot], dict[str, bool]] | None = None,
    early_check: Callable[
        [AlttpSnapshot, list[RoutePhaseResult], int], SegmentResult | None
    ]
    | None = None,
) -> SegmentResult:
    """Clear (optional) + exit one door. ``ok=True`` when door dest is reached.

    Geometry from ``maps/{map_id}.json``. Multi-hop segments wrap this and
    apply their own full-route acceptance (e.g. Zelda follower).
    """
    room_map = load_room_map(map_id)
    door = room_map.door(door_label)
    if door is None:
        snap = snapshot_env(env)
        return SegmentResult(
            ok=False,
            phase="unknown_door",
            frames=0,
            snapshot=snap,
            source=source,
            blocker=f"door {door_label!r} not in {map_id}",
            notes=notes or [],
        )

    frames = 0
    phases: list[RoutePhaseResult] = []
    base_notes = notes or [
        f"room_engine map={room_map.map_id} door={door_label}",
        f"doors={[d.label for d in room_map.doors]}",
    ]

    def _accept(s: AlttpSnapshot) -> dict[str, bool]:
        if acceptance_fn is not None:
            return acceptance_fn(s)
        return {
            "in_origin_room": in_origin(s, room_map),
            "at_door_dest": at_door_destination(s, door),
        }

    def _edge_ok(
        snap: AlttpSnapshot,
        *,
        phase: str,
        extra_notes: list[str] | None = None,
    ) -> SegmentResult:
        return SegmentResult(
            ok=True,
            phase=phase,
            frames=frames,
            snapshot=snap,
            phases=phases,
            source=source,
            acceptance=_accept(snap),
            blocker="",
            notes=base_notes + (extra_notes or []),
        )

    settle = primitives.settle_control(env)
    frames += settle.frames
    snap = settle.snapshot
    phases.append(
        RoutePhaseResult(
            phase="settle_control",
            ok=snap.has_control,
            frames=settle.frames,
            snapshot=snap,
            detail=f"settled before {map_id}/{door_label}",
            diag=snapshot_to_diag(snap),
        )
    )

    if early_check is not None:
        early = early_check(snap, phases, frames)
        if early is not None:
            return early

    if at_door_destination(snap, door):
        return _edge_ok(
            snap,
            phase=f"via_{door.label}",
            extra_notes=["Already at door destination."],
        )

    if clear and not room_map.outdoors:
        clear_phase = clear_room(
            env,
            room_map,
            phase=f"clear_{room_map.map_id}",
            already_past=lambda s: at_door_destination(s, door),
        )
        frames += clear_phase.frames
        phases.append(clear_phase)
        snap = clear_phase.snapshot
        if at_door_destination(snap, door):
            return _edge_ok(
                snap,
                phase=f"via_{door.label}",
                extra_notes=["Reached door dest during clear."],
            )
        if not clear_phase.ok and snap.game_mode == 0x12:
            return SegmentResult(
                ok=False,
                phase=clear_phase.phase,
                frames=frames,
                snapshot=snap,
                phases=phases,
                source=source,
                acceptance=_accept(snap),
                blocker=clear_phase.detail or "room clear failed",
                notes=base_notes,
            )
        # An open door is walked alive. A clear that times out must not
        # block the exit; fight_nearby on room 0x61 hits its cycle cap.

    exit_phase = exit_via_door(env, room_map, door)
    frames += exit_phase.frames
    phases.append(exit_phase)
    snap = exit_phase.snapshot

    if at_door_destination(snap, door):
        # A room id can change while the door transition still owns input.
        # Settle in the destination before reporting a composable edge success;
        # the next edge must see the real predecessor state, not submodule 14.
        settled = primitives.settle_control(env, max_frames=DEST_SETTLE_MAX_FRAMES)
        frames += settled.frames
        phases.append(
            RoutePhaseResult(
                phase="settle_destination",
                ok=settled.ok,
                frames=settled.frames,
                snapshot=settled.snapshot,
                detail=f"settled after {door.label}",
                diag=snapshot_to_diag(settled.snapshot),
            )
        )
        snap = settled.snapshot
        if not settled.ok or not at_door_destination(snap, door):
            return SegmentResult(
                ok=False,
                phase="settle_destination",
                frames=frames,
                snapshot=snap,
                phases=phases,
                source=source,
                acceptance=_accept(snap),
                blocker=(
                    f"destination did not settle after {door.label}: "
                    f"control={snap.has_control} room={room_label(snap.room_base_id)}"
                ),
                notes=base_notes,
            )
        return _edge_ok(
            snap,
            phase=f"via_{door.label}",
            extra_notes=[
                f"Room complete: exited via {door.label}.",
                f"Landing xy=({snap.link_x},{snap.link_y}).",
            ],
        )

    return SegmentResult(
        ok=False,
        phase=exit_phase.phase if not exit_phase.ok else "room_blocked",
        frames=frames,
        snapshot=snap,
        phases=phases,
        source=source,
        acceptance=_accept(snap),
        blocker=exit_phase.detail or "door exit failed",
        notes=base_notes,
    )
