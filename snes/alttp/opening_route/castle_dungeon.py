"""Composable first-Hyrule-Castle dungeon prefix.

This module owns the measured room-edge sequence after the continuous opening
tip.  It deliberately stops at room ``0x50``:

``0x61 main hall → 0x60 west hall → 0x50 NW chamber``.

Both edges are verified from their real predecessor in one clean power-on
chain. The prefix is still *not* a Zelda or Sanctuary clear.
``room_50_east_to_0x01`` and ``room_01_down_to_0x72`` are separate spine
segments (map doors, hold UP at the well). They are not part of this prefix.
``room_72_south_to_0x82`` and ``room_82_west_to_0x81`` are the same: map
geometry, axis holds, not this prefix. The 0x82 tile walker pits from the
real 0x72 landing, so that edge does not call ``run_room_edge``.
``earn_castle_big_key`` fights the full-health guard in room ``0x80`` with
controller inputs; its caller owns predecessor and assist disclosure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from alttp.opening_route.room_engine import (
    DEST_SETTLE_MAX_FRAMES,
    in_room,
    run_room_edge,
)
from alttp.primitives import (
    SPRITE_BALL_AND_CHAIN,
    SPRITE_BIG_KEY,
    SPRITE_SMALL_KEY,
    PrimitiveResult,
    collect_nearby,
    fight_nearby,
    settle_control,
    sprites_of_type,
)
from alttp.ram import (
    HYRULE_CASTLE_B1_EAST_ROOM,
    HYRULE_CASTLE_B1_PIT_ROOM,
    HYRULE_CASTLE_B1_WEST_ROOM,
    HYRULE_CASTLE_MAIN_HALL_ROOM,
    HYRULE_CASTLE_MAIN_WEST_ROOM,
    HYRULE_CASTLE_NW_ROOM,
    ZELDA_CELL_ROOM,
    AlttpSnapshot,
    read_inventory,
    zelda_rescued_accepted,
)
from alttp.room_map import load_room_map
from alttp.route_report import RoutePhaseResult, SegmentResult
from alttp.startup import BootEnv, action_for, no_action, snapshot_env, step_frames

# Pit span. Map pathTolerances (bridge 20) walk off it. One axis at a time.
_AXIS_TOL = 8
# Map attackDistance 50 steps off the north lip. Skirmish distance stays on the map.
_LEDGE_ATTACK_DISTANCE = 40
# Lower-floor guards sit outside skirmishMaxDistance and inside maxDistance.
_LOWER_FIGHT_DISTANCE = 140
_AXIS_STUCK_STEPS = 40


@dataclass(frozen=True)
class DungeonRoomEdge:
    """One measured room-engine edge in the castle dungeon prefix."""

    edge_id: str
    map_id: str
    door_label: str
    source_room: int
    target_room: int
    graph_edge_id: str
    verification: str = "continuous"
    clear: bool = True


MAIN_HALL_TO_NW_PREFIX: tuple[DungeonRoomEdge, ...] = (
    DungeonRoomEdge(
        edge_id="main_hall_west_to_0x60",
        map_id="room_61",
        door_label="west_to_0x60",
        source_room=HYRULE_CASTLE_MAIN_HALL_ROOM,
        target_room=HYRULE_CASTLE_MAIN_WEST_ROOM,
        graph_edge_id="main_hall_west_to_0x60",
    ),
    DungeonRoomEdge(
        edge_id="room_60_north_to_0x50",
        map_id="room_60",
        door_label="north_to_0x50",
        source_room=HYRULE_CASTLE_MAIN_WEST_ROOM,
        target_room=HYRULE_CASTLE_NW_ROOM,
        graph_edge_id="room_60_north_to_0x50",
    ),
)


def evaluate_prefix_acceptance(snapshot: AlttpSnapshot) -> dict[str, bool]:
    """Contract keys for the measured ``0x61 → 0x60 → 0x50`` prefix only."""
    return {
        "fighter_sword_ram": snapshot.has_fighter_sword,
        "main_hall": in_room(snapshot, HYRULE_CASTLE_MAIN_HALL_ROOM),
        "main_west_0x60": in_room(snapshot, HYRULE_CASTLE_MAIN_WEST_ROOM),
        "northwest_0x50": in_room(snapshot, HYRULE_CASTLE_NW_ROOM),
    }


def evaluate_prefix_diagnostics(snapshot: AlttpSnapshot) -> dict[str, bool]:
    """Log-only later-route flags (not prefix exit success)."""
    return {
        "zelda_follower": zelda_rescued_accepted(snapshot),
        "in_sanctuary": snapshot.in_sanctuary,
    }


def _prefix_notes(edges: Sequence[DungeonRoomEdge]) -> list[str]:
    return [
        "First Hyrule Castle dungeon prefix via room_engine map authority.",
        "All listed edges passed from their real predecessor in the clean power-on chain.",
        "No Zelda or Sanctuary claim is made by this prefix.",
        *[
            (
                f"{edge.edge_id}: {edge.map_id}/{edge.door_label} "
                f"(verification={edge.verification})"
            )
            for edge in edges
        ],
    ]


def _prefix_result(
    *,
    ok: bool,
    phase: str,
    frames: int,
    snapshot: AlttpSnapshot,
    source: str,
    phases: list[RoutePhaseResult] | None = None,
    blocker: str = "",
    notes: Sequence[str] = (),
) -> SegmentResult:
    return SegmentResult(
        ok=ok,
        phase=phase,
        frames=frames,
        snapshot=snapshot,
        phases=list(phases or ()),
        source=source,
        acceptance=evaluate_prefix_acceptance(snapshot),
        diagnostics=evaluate_prefix_diagnostics(snapshot),
        blocker=blocker,
        notes=list(notes),
        report_kind="castle_dungeon_prefix",
    )


def run_room_edge_sequence(
    env: BootEnv,
    edges: Sequence[DungeonRoomEdge],
    *,
    source: str = "state_load_dev",
) -> SegmentResult:
    """Run measured room edges in order, failing before a wrong-room action.

    ``run_room_edge`` owns combat, geometry and door pushes.  This thin
    composition layer owns only ordering, room-boundary assertions and a
    cumulative report.  That keeps every future castle-door addition local and
    auditable.
    """
    if not edges:
        snap = snapshot_env(env)
        return _prefix_result(
            ok=False,
            phase="empty_dungeon_prefix",
            frames=0,
            snapshot=snap,
            source=source,
            blocker="no dungeon room edges configured",
        )

    frames = 0
    phases: list[RoutePhaseResult] = []
    notes = _prefix_notes(edges)
    for edge in edges:
        before = snapshot_env(env)
        if not in_room(before, edge.source_room):
            return _prefix_result(
                ok=False,
                phase=f"entry_not_{edge.map_id}",
                frames=frames,
                snapshot=before,
                phases=phases,
                source=source,
                blocker=(
                    f"{edge.edge_id} expected room 0x{edge.source_room:02X}, got "
                    f"0x{before.room_base_id:02X} indoors={before.indoors}"
                ),
                notes=notes,
            )

        result = run_room_edge(
            env,
            edge.map_id,
            edge.door_label,
            clear=edge.clear,
            source=source,
            notes=[
                f"dungeon_edge={edge.edge_id}",
                f"graph_edge={edge.graph_edge_id}",
                f"verification={edge.verification}",
            ],
        )
        frames += result.frames
        phases.extend(result.phases)
        snap = result.snapshot
        if not result.ok:
            return _prefix_result(
                ok=False,
                phase=f"{edge.edge_id}:{result.phase}",
                frames=frames,
                snapshot=snap,
                phases=phases,
                source=source,
                blocker=result.blocker or f"{edge.edge_id} failed",
                notes=notes,
            )
        if not in_room(snap, edge.target_room):
            return _prefix_result(
                ok=False,
                phase=f"{edge.edge_id}:wrong_destination",
                frames=frames,
                snapshot=snap,
                phases=phases,
                source=source,
                blocker=(
                    f"{edge.edge_id} expected destination 0x{edge.target_room:02X}, "
                    f"got 0x{snap.room_base_id:02X} indoors={snap.indoors}"
                ),
                notes=notes,
            )

    final = snapshot_env(env)
    return _prefix_result(
        ok=True,
        phase="castle_dungeon_prefix_complete",
        frames=frames,
        snapshot=final,
        phases=phases,
        source=source,
        notes=notes,
    )


def run_from_main_hall(
    env: BootEnv,
    *,
    source: str = "state_load_dev",
) -> SegmentResult:
    """Run the measured ``0x61 → 0x60 → 0x50`` first-dungeon prefix."""
    return run_room_edge_sequence(env, MAIN_HALL_TO_NW_PREFIX, source=source)


def run_east_to_0x01(
    env: BootEnv,
    *,
    source: str = "state_load_dev",
) -> SegmentResult:
    """``0x50`` east door. Geometry: ``maps/room_50.json`` ``east_to_0x01``."""
    return run_room_edge(
        env,
        "room_50",
        "east_to_0x01",
        clear=True,
        source=source,
    )


def run_down_to_0x72(
    env: BootEnv,
    *,
    source: str = "state_load_dev",
) -> SegmentResult:
    """North-wall well. Map path to the approach, then hold UP (not DOWN)."""
    return run_room_edge(
        env,
        "room_01",
        "down_to_0x72",
        clear=False,
        source=source,
    )


def _dead(snap: AlttpSnapshot) -> bool:
    return snap.hp <= 0 or snap.submodule == 20 or snap.game_mode == 0x12


def _key_sprite(env: object) -> bool:
    return bool(sprites_of_type(env, (SPRITE_SMALL_KEY,), max_distance=200))


def _edge_result(
    *,
    ok: bool,
    phase: str,
    frames: int,
    snapshot: AlttpSnapshot,
    phases: list[RoutePhaseResult],
    source: str,
    dest: int,
    blocker: str = "",
    notes: Sequence[str] = (),
) -> SegmentResult:
    arrived = ok and in_room(snapshot, dest) and not _dead(snapshot)
    return SegmentResult(
        ok=arrived,
        phase=phase,
        frames=frames,
        snapshot=snapshot,
        phases=phases,
        source=source,
        acceptance={"at_door_dest": arrived},
        blocker="" if arrived else blocker or "door dest not reached",
        notes=list(notes),
        report_kind="castle_dungeon_edge",
    )


def _hold_axis(
    env: object,
    *,
    axis: str,
    target: int,
    room: int,
    label: str,
    max_frames: int,
) -> PrimitiveResult:
    """Hold one cardinal. Both axes together walk off the B1 pits."""
    frames = 0
    still = 0
    prev: tuple[int, int] | None = None
    while frames < max_frames:
        snap = snapshot_env(env)
        if _dead(snap):
            return PrimitiveResult(False, f"died before {label}", frames, snap)
        if not in_room(snap, room):
            return PrimitiveResult(
                False, f"left room 0x{room:02X} before {label}", frames, snap
            )
        pos = snap.link_x if axis == "x" else snap.link_y
        if abs(pos - target) <= _AXIS_TOL:
            return PrimitiveResult(True, label, frames, snap)
        if axis == "x":
            button = "RIGHT" if target > snap.link_x else "LEFT"
        else:
            button = "DOWN" if target > snap.link_y else "UP"
        step_frames(env, action_for(button), 2)
        frames += 2
        moved = snapshot_env(env)
        xy = (moved.link_x, moved.link_y)
        if xy == prev:
            still += 1
            if still >= _AXIS_STUCK_STEPS:
                return PrimitiveResult(False, f"stuck before {label}", frames, moved)
        else:
            still = 0
            prev = xy
    return PrimitiveResult(
        False, f"timeout before {label}", frames, snapshot_env(env)
    )


def _axis_to(
    env: object,
    x: int,
    y: int,
    *,
    room: int,
    label: str,
    max_x: int = 400,
    max_y: int = 500,
) -> PrimitiveResult:
    """One cardinal per step. Re-align x if a hit shoves it off the column.

    Holding both axes on the 0x82 bridge walks into the pit. A pure south
    hold after that shove wedges on the corridor wall.
    """
    frames = 0
    still = 0
    prev: tuple[int, int] | None = None
    limit = max_x + max_y
    while frames < limit:
        snap = snapshot_env(env)
        if _dead(snap):
            return PrimitiveResult(False, f"died before {label}", frames, snap)
        if not in_room(snap, room):
            return PrimitiveResult(
                False, f"left room 0x{room:02X} before {label}", frames, snap
            )
        dx = x - snap.link_x
        dy = y - snap.link_y
        if abs(dx) <= _AXIS_TOL and abs(dy) <= _AXIS_TOL:
            return PrimitiveResult(True, label, frames, snap)
        if abs(dx) > _AXIS_TOL:
            button = "RIGHT" if dx > 0 else "LEFT"
        else:
            button = "DOWN" if dy > 0 else "UP"
        step_frames(env, action_for(button), 2)
        frames += 2
        moved = snapshot_env(env)
        xy = (moved.link_x, moved.link_y)
        if xy == prev:
            still += 1
            if still >= _AXIS_STUCK_STEPS:
                return PrimitiveResult(False, f"stuck before {label}", frames, moved)
        else:
            still = 0
            prev = xy
    return PrimitiveResult(
        False, f"timeout before {label}", frames, snapshot_env(env)
    )


def _hold_until(
    env: object,
    buttons: tuple[str, ...],
    predicate: Callable[[AlttpSnapshot], bool],
    *,
    room: int,
    label: str,
    max_frames: int,
) -> PrimitiveResult:
    frames = 0
    action = action_for(*buttons)
    while frames < max_frames:
        snap = snapshot_env(env)
        if _dead(snap):
            return PrimitiveResult(False, f"died before {label}", frames, snap)
        if not in_room(snap, room):
            return PrimitiveResult(
                False, f"left room 0x{room:02X} before {label}", frames, snap
            )
        if predicate(snap):
            return PrimitiveResult(True, label, frames, snap)
        step_frames(env, action, 2)
        frames += 2
    snap = snapshot_env(env)
    if predicate(snap) and not _dead(snap) and in_room(snap, room):
        return PrimitiveResult(True, label, frames, snap)
    return PrimitiveResult(False, f"timeout before {label}", frames, snap)


def _push_leave(
    env: object,
    button: str,
    *,
    origin: int,
    label: str,
    max_frames: int,
) -> PrimitiveResult:
    frames = 0
    action = action_for(button)
    while frames < max_frames:
        snap = snapshot_env(env)
        if _dead(snap):
            return PrimitiveResult(False, f"died before {label}", frames, snap)
        if not in_room(snap, origin):
            return PrimitiveResult(True, label, frames, snap)
        step_frames(env, action, 2)
        frames += 2
    snap = snapshot_env(env)
    if not _dead(snap) and not in_room(snap, origin):
        return PrimitiveResult(True, label, frames, snap)
    return PrimitiveResult(False, f"timeout before {label}", frames, snap)


def _settle_dest(
    env: object,
    *,
    frames: int,
    phases: list[RoutePhaseResult],
    source: str,
    dest: int,
    phase: str,
    notes: list[str],
) -> SegmentResult:
    settled = settle_control(env, max_frames=DEST_SETTLE_MAX_FRAMES)
    frames += settled.frames
    phases.append(
        RoutePhaseResult(
            phase="settle_destination",
            ok=settled.ok,
            frames=settled.frames,
            snapshot=settled.snapshot,
            detail=settled.reason,
        )
    )
    snap = settled.snapshot
    blocker = ""
    if not settled.ok or not in_room(snap, dest) or _dead(snap):
        blocker = (
            f"destination did not settle: control={snap.has_control} "
            f"room=0x{snap.room_base_id:02X} hp={snap.hp} sub={snap.submodule}"
        )
    else:
        notes.append(
            f"Landing xy=({snap.link_x},{snap.link_y}) room=0x{snap.room_base_id:02X}."
        )
    return _edge_result(
        ok=settled.ok and blocker == "",
        phase=phase,
        frames=frames,
        snapshot=snap,
        phases=phases,
        source=source,
        dest=dest,
        blocker=blocker,
        notes=notes,
    )


def _left_origin(result: PrimitiveResult, origin: int) -> bool:
    snap = result.snapshot
    return not result.ok and not _dead(snap) and not in_room(snap, origin)


def run_south_to_0x82(
    env: BootEnv,
    *,
    source: str = "state_load_dev",
) -> SegmentResult:
    """North-ledge ``0x72`` through the key door, then south into ``0x82``.

    Points come from ``maps/room_72.json``. The isolated ``south_to_0x82``
    path starts south of the pit and is not this predecessor.
    """
    room_map = load_room_map("room_72")
    door = room_map.door("south_to_0x82")
    lip = room_map.point("key_door_lip")
    drop = room_map.point("key_ledge_drop")
    approach = room_map.point("south_door_approach")
    notes = [
        "room_72 north ledge: skirmish, small key, key door, south door.",
        "Axis holds only. Graph class of south_to_0x82 stays isolated.",
    ]
    frames = 0
    phases: list[RoutePhaseResult] = []
    if door is None or lip is None or drop is None or approach is None:
        snap = snapshot_env(env)
        return _edge_result(
            ok=False,
            phase="missing_map_point",
            frames=0,
            snapshot=snap,
            phases=phases,
            source=source,
            dest=HYRULE_CASTLE_B1_EAST_ROOM,
            blocker="room_72 map missing key_door_lip, key_ledge_drop, or south door",
            notes=notes,
        )

    def took(result: PrimitiveResult, phase: str) -> PrimitiveResult:
        nonlocal frames
        frames += result.frames
        phases.append(
            RoutePhaseResult(
                phase=phase,
                ok=result.ok,
                frames=result.frames,
                snapshot=result.snapshot,
                detail=result.reason,
            )
        )
        return result

    def fail(phase: str, snap: AlttpSnapshot, blocker: str) -> SegmentResult:
        return _edge_result(
            ok=False,
            phase=phase,
            frames=frames,
            snapshot=snap,
            phases=phases,
            source=source,
            dest=HYRULE_CASTLE_B1_EAST_ROOM,
            blocker=blocker,
            notes=notes,
        )

    opening = took(settle_control(env), "settle_control")
    if not opening.ok:
        return fail("settle_control", opening.snapshot, opening.reason)

    snap = opening.snapshot
    if snap.num_keys < 1:
        policy = room_map.clear_policy
        fight = fight_nearby(
            env,
            room=HYRULE_CASTLE_B1_PIT_ROOM,
            max_distance=policy.skirmish_max_distance,
            attack_distance=_LEDGE_ATTACK_DISTANCE,
            max_cycles=policy.skirmish_max_cycles,
            stop_when=lambda active: _key_sprite(active),
        )
        took(
            PrimitiveResult(fight.ok, fight.reason, fight.frames, fight.snapshot),
            "ledge_fight",
        )
        if _dead(fight.snapshot) or not in_room(fight.snapshot, HYRULE_CASTLE_B1_PIT_ROOM):
            return fail("ledge_fight", fight.snapshot, fight.reason or "ledge fight failed")
        snap = snapshot_env(env)
        waited = 0
        while snap.num_keys < 1 and not _key_sprite(env) and waited < 60:
            step_frames(env, no_action(), 2)
            waited += 2
            snap = snapshot_env(env)
        frames += waited
        if snap.num_keys < 1 and _key_sprite(env):
            picked = took(
                collect_nearby(
                    env,
                    (SPRITE_SMALL_KEY,),
                    max_distance=160,
                    max_frames=240,
                ),
                "collect_key",
            )
            if not picked.ok or _dead(picked.snapshot):
                return fail("collect_key", picked.snapshot, picked.reason)
        held = took(
            settle_control(env, max_frames=DEST_SETTLE_MAX_FRAMES),
            "settle_key",
        )
        snap = held.snapshot
        if not held.ok or snap.num_keys < 1:
            return fail("settle_key", snap, "small key not in inventory")

    walked = took(
        _axis_to(env, lip.x, lip.y, room=HYRULE_CASTLE_B1_PIT_ROOM, label="key_door_lip"),
        "key_door_lip",
    )
    if not walked.ok:
        return fail("key_door_lip", walked.snapshot, walked.reason)

    spent = took(
        _hold_until(
            env,
            ("DOWN", "LEFT"),
            lambda s: s.num_keys == 0,
            room=HYRULE_CASTLE_B1_PIT_ROOM,
            label="key_door_spend",
            max_frames=120,
        ),
        "key_door_spend",
    )
    if not spent.ok:
        return fail("key_door_spend", spent.snapshot, spent.reason)

    # Submodule 4 is the door. Pure DOWN until the lower ledge; a settle here
    # spends the key and leaves Link on the closed lip.
    drop_y = drop.y - 2
    landed = took(
        _hold_until(
            env,
            ("DOWN",),
            lambda s: s.submodule == 0 and s.link_y >= drop_y,
            room=HYRULE_CASTLE_B1_PIT_ROOM,
            label="key_ledge_drop",
            max_frames=480,
        ),
        "key_ledge_drop",
    )
    if not landed.ok:
        return fail("key_ledge_drop", landed.snapshot, landed.reason)

    after_drop = took(settle_control(env), "settle_drop")
    if not after_drop.ok:
        return fail("settle_drop", after_drop.snapshot, after_drop.reason)

    lower = fight_nearby(
        env,
        room=HYRULE_CASTLE_B1_PIT_ROOM,
        max_distance=_LOWER_FIGHT_DISTANCE,
        attack_distance=_LEDGE_ATTACK_DISTANCE,
        max_cycles=room_map.clear_policy.skirmish_max_cycles,
    )
    took(
        PrimitiveResult(lower.ok, lower.reason, lower.frames, lower.snapshot),
        "lower_fight",
    )
    if not lower.ok or _dead(lower.snapshot) or not in_room(
        lower.snapshot, HYRULE_CASTLE_B1_PIT_ROOM
    ):
        return fail("lower_fight", lower.snapshot, lower.reason or "lower fight failed")

    # South-first near the railing pits. Align x, then hold the door direction.
    aligned = took(
        _hold_axis(
            env,
            axis="x",
            target=approach.x,
            room=HYRULE_CASTLE_B1_PIT_ROOM,
            label="south_door_x",
            max_frames=400,
        ),
        "south_door_x",
    )
    if not aligned.ok:
        return fail("south_door_x", aligned.snapshot, aligned.reason)

    pushed = took(
        _push_leave(
            env,
            door.direction,
            origin=HYRULE_CASTLE_B1_PIT_ROOM,
            label="south_to_0x82",
            max_frames=300,
        ),
        "south_to_0x82",
    )
    if not pushed.ok:
        return fail("south_to_0x82", pushed.snapshot, pushed.reason)
    return _settle_dest(
        env,
        frames=frames,
        phases=phases,
        source=source,
        dest=HYRULE_CASTLE_B1_EAST_ROOM,
        phase="via_south_to_0x82",
        notes=notes,
    )


def run_west_to_0x81(
    env: BootEnv,
    *,
    source: str = "state_load_dev",
) -> SegmentResult:
    """``0x82`` west door from the real ``0x72`` landing.

    Axis-walk ``west_to_0x81`` path labels. A room change on that walk is the
    door. Do not fight first and do not call ``run_room_edge``: the tile
    walker pits in the bridge corridor from this landing.
    """
    room_map = load_room_map("room_82")
    door = room_map.door("west_to_0x81")
    notes = [
        "room_82 axis path from the 0x72-south landing. No pre-fight.",
        "Graph class of west_to_0x81 stays isolated.",
    ]
    frames = 0
    phases: list[RoutePhaseResult] = []
    if door is None or door.to_room is None:
        snap = snapshot_env(env)
        return _edge_result(
            ok=False,
            phase="missing_map_point",
            frames=0,
            snapshot=snap,
            phases=phases,
            source=source,
            dest=HYRULE_CASTLE_B1_WEST_ROOM,
            blocker="room_82 map missing west_to_0x81",
            notes=notes,
        )

    def took(result: PrimitiveResult, phase: str) -> PrimitiveResult:
        nonlocal frames
        frames += result.frames
        phases.append(
            RoutePhaseResult(
                phase=phase,
                ok=result.ok,
                frames=result.frames,
                snapshot=result.snapshot,
                detail=result.reason,
            )
        )
        return result

    def fail(phase: str, snap: AlttpSnapshot, blocker: str) -> SegmentResult:
        return _edge_result(
            ok=False,
            phase=phase,
            frames=frames,
            snapshot=snap,
            phases=phases,
            source=source,
            dest=HYRULE_CASTLE_B1_WEST_ROOM,
            blocker=blocker,
            notes=notes,
        )

    opening = took(settle_control(env), "settle_control")
    if not opening.ok:
        return fail("settle_control", opening.snapshot, opening.reason)

    labels = tuple(door.path) + ("west_door_approach",)
    for label in labels:
        point = room_map.point(label)
        if point is None and label == "west_door_approach":
            x, y = door.approach_xy
        elif point is None:
            snap = snapshot_env(env)
            return fail(label, snap, f"room_82 missing point {label}")
        else:
            x, y = point.x, point.y
        walked = took(
            _axis_to(
                env,
                x,
                y,
                room=HYRULE_CASTLE_B1_EAST_ROOM,
                label=label,
            ),
            label,
        )
        if _left_origin(walked, HYRULE_CASTLE_B1_EAST_ROOM):
            return _settle_dest(
                env,
                frames=frames,
                phases=phases,
                source=source,
                dest=HYRULE_CASTLE_B1_WEST_ROOM,
                phase="via_west_to_0x81",
                notes=notes,
            )
        if not walked.ok:
            return fail(label, walked.snapshot, walked.reason)

    pushed = took(
        _push_leave(
            env,
            door.direction,
            origin=HYRULE_CASTLE_B1_EAST_ROOM,
            label="west_to_0x81",
            max_frames=max(door.push_frames, 120),
        ),
        "west_to_0x81",
    )
    if not pushed.ok:
        return fail("west_to_0x81", pushed.snapshot, pushed.reason)
    return _settle_dest(
        env,
        frames=frames,
        phases=phases,
        source=source,
        dest=HYRULE_CASTLE_B1_WEST_ROOM,
        phase="via_west_to_0x81",
        notes=notes,
    )


def earn_castle_big_key(
    env: BootEnv,
    *,
    source: str = "state_load_dev",
    max_decisions: int = 1800,
) -> SegmentResult:
    """Kill the castle guard from its natural arrival and take the big key.

    Face the guard and tap the sword within 20px, including when Link may
    take damage. The caller can supply disclosed HP assistance. This helper
    writes no RAM and performs no restores. The key word at ``$F366`` must
    rise through combat and pickup; Zelda interaction belongs to the caller.
    """
    notes = [
        "Full-health castle guard: controller facing, approach and sword taps.",
        "Prize pickup is collect_nearby; no RAM writes or lookahead restores.",
    ]
    frames = 0
    start_key = int(read_inventory(np.asarray(env.get_ram(), dtype=np.uint8)).big_key)
    phase = "entry"
    blocker = ""
    snap = snapshot_env(env)
    if not in_room(snap, ZELDA_CELL_ROOM) or snap.follower != 0 or start_key:
        blocker = "requires room 0x80, no follower, and no existing big key"
    elif not any(s.hp > 0 for s in sprites_of_type(env, (SPRITE_BALL_AND_CHAIN,))):
        blocker = "castle guard missing at entry"
    else:
        phase = "ball_and_chain"
        fought = fight_nearby(
            env, sprite_types=(SPRITE_BALL_AND_CHAIN,), room=ZELDA_CELL_ROOM,
            max_distance=400, attack_distance=20, max_cycles=max_decisions,
        )
        frames += fought.frames
        notes.append(f"guard combat: {fought.reason} frames={fought.frames}")
        if not fought.ok:
            blocker = fought.reason or "castle guard not defeated"
        elif any(s.hp > 0 for s in sprites_of_type(env, (SPRITE_BALL_AND_CHAIN,))):
            blocker = "castle guard still alive"
        else:
            phase = "prize"

    snap = snapshot_env(env)
    if blocker == "" and phase == "prize":
        for _ in range(240):
            if sprites_of_type(env, (SPRITE_BIG_KEY,)):
                break
            step_frames(env, no_action(), 1)
            frames += 1
        picked = collect_nearby(
            env, (SPRITE_BIG_KEY,), max_distance=400, max_frames=600
        )
        frames += picked.frames
        if not picked.ok:
            blocker = picked.reason or "big key not collected"
        else:
            settled = settle_control(env, max_frames=400)
            frames += settled.frames
            if not settled.ok:
                blocker = settled.reason or "prize did not settle"
            snap = settled.snapshot

    end_key = int(read_inventory(np.asarray(env.get_ram(), dtype=np.uint8)).big_key)
    rose = end_key > start_key
    if blocker == "" and not rose:
        blocker = f"big_key did not rise {start_key} -> {end_key}"
    if blocker == "" and snap.follower != 0:
        blocker = f"follower changed to {snap.follower}"
    if blocker == "" and not in_room(snap, ZELDA_CELL_ROOM):
        blocker = f"left room 0x80 for 0x{snap.room_base_id:02X}"
    notes.append(
        f"big_key {start_key} -> {end_key} "
        f"xy=({snap.link_x},{snap.link_y}) follower={snap.follower}"
    )
    return SegmentResult(
        ok=blocker == "" and rose and snap.follower == 0,
        phase="big_key_in_play" if blocker == "" else phase,
        frames=frames,
        snapshot=snap,
        source=source,
        acceptance={
            "big_key_rose": rose,
            "in_room_80": in_room(snap, ZELDA_CELL_ROOM),
            "follower_clear": snap.follower == 0,
        },
        blocker=blocker,
        notes=notes,
        report_kind="castle_big_key",
    )


def run_from_state(
    state_name: str = "CastleMain",
    *,
    close: bool = True,
) -> SegmentResult:
    """Development diagnostic for the two-edge prefix from main hall."""
    from alttp.opening_route.runner import run_from_state as _run_from_state

    return _run_from_state(state_name, run_from_main_hall, close=close)
