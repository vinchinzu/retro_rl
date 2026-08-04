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

from alttp import primitives
from alttp.ram import (
    AlttpSnapshot,
    room_label,
    snapshot_to_diag,
    zelda_rescued_accepted,
)
from alttp.room_map import ClearPolicy, KnownDoor, RoomMap, load_room_map
from alttp.room_sense import detect_edge, enemy_boxes, path_blocked_by_enemies
from alttp.route_report import RoutePhaseResult, SegmentResult
from alttp.startup import action_for, snapshot_env, step_frames


def in_room(snap: AlttpSnapshot, room: int) -> bool:
    return snap.indoors and (not snap.dark_world) and snap.room_base_id == room


def at_door_destination(snap: AlttpSnapshot, door: KnownDoor) -> bool:
    """True when snapshot matches door exit (room or outdoors)."""
    if door.outdoors:
        return not snap.indoors
    if door.to_room is None:
        return False
    return in_room(snap, door.to_room)


def move_path_combat_aware(
    env: object,
    waypoints: list[primitives.Waypoint],
    *,
    room: int,
    policy: ClearPolicy,
) -> primitives.PrimitiveResult:
    """Follow waypoints; skirmish when a hostile box blocks the next segment."""
    frames = 0
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

        move = primitives.move_to(env, wp, max_frames=500)
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


def clear_room(
    env: object,
    room_map: RoomMap,
    *,
    phase: str = "clear_room",
    already_past: Callable[[AlttpSnapshot], bool] | None = None,
) -> RoutePhaseResult:
    """Clear hostiles in ``room_map`` using clearPolicy + fight_nearby."""
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

    boxes_before = enemy_boxes(env, max_distance=policy.max_distance + 100)

    def _past(e: object) -> bool:
        if already_past is not None:
            return bool(already_past(snapshot_env(e)))
        return False

    combat = primitives.fight_nearby(
        env,
        room=room,
        max_distance=policy.max_distance,
        attack_distance=policy.attack_distance,
        max_cycles=policy.max_cycles,
        stop_when=_past if already_past is not None else None,
    )
    frames += combat.frames
    settle2 = primitives.settle_control(env, max_frames=120)
    frames += settle2.frames
    snap = settle2.snapshot
    remaining = enemy_boxes(env, max_distance=120)
    past = already_past(snap) if already_past is not None else False
    ok = (
        (in_room(snap, room) or past)
        and snap.game_mode != 0x12
        and (combat.ok or combat.reason == "no nearby targets" or len(remaining) == 0)
    )
    return RoutePhaseResult(
        phase=phase,
        ok=ok,
        frames=frames,
        snapshot=snap,
        detail=(
            f"before={len(boxes_before)} defeated={combat.defeated_slots} "
            f"remaining_near={len(remaining)} xy=({snap.link_x},{snap.link_y}) "
            f"reason={combat.reason}"
        ),
        diag={
            **snapshot_to_diag(snap),
            "enemyBoxesBefore": [b.to_dict() for b in boxes_before],
            "enemyBoxes": [b.to_dict() for b in remaining],
            "combatReason": combat.reason,
        },
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
    if not in_room(start, room):
        return RoutePhaseResult(
            phase=phase_name,
            ok=False,
            frames=frames,
            snapshot=start,
            detail=f"not in room 0x{room:02X} ({room_label(start.room_base_id)})",
            diag=snapshot_to_diag(start),
        )

    wps: list[primitives.Waypoint] = []
    for x, y, label, tol in room_map.waypoints_for_door(d):
        wps.append(primitives.Waypoint(x, y, tolerance=tol, room=room, label=label))

    if wps:
        path = move_path_combat_aware(env, wps, room=room, policy=policy)
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
                env, recovery_wps, room=room, policy=policy
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

    # Hold door direction through transition.
    before = snapshot_env(env)
    push_limit = max(1, d.push_frames // 3)
    for _ in range(push_limit):
        step_frames(env, action_for(d.direction), 3)
        frames += 3
        snap = snapshot_env(env)
        if at_door_destination(snap, d) or not in_room(snap, room):
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
                ),
                diag={
                    **snapshot_to_diag(snap),
                    "edge": edge.to_dict() if edge else None,
                    "door": d.label,
                },
            )

    snap = snapshot_env(env)
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
            "in_origin_room": in_room(s, room_map.room_base_id),
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

    if clear:
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
        if not clear_phase.ok:
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

    exit_phase = exit_via_door(env, room_map, door)
    frames += exit_phase.frames
    phases.append(exit_phase)
    snap = exit_phase.snapshot

    if at_door_destination(snap, door):
        # A room id can change while the door transition still owns input.
        # Settle in the destination before reporting a composable edge success;
        # the next edge must see the real predecessor state, not submodule 2.
        settled = primitives.settle_control(env)
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
