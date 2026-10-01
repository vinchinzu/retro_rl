"""B1 west → B3 keys → Zelda rescue.

The castle prefix is owned by castle_dungeon. The rescue suffix walks
measured room geometry and earns both keys through controller play.
Legacy main-hall diagnostics remain separate from that prefix.
"""

from __future__ import annotations

from alttp.opening_route.castle_dungeon import evaluate_prefix_acceptance
from alttp.opening_route.room_engine import in_room
from alttp.ram import (
    HYRULE_CASTLE_MAIN_HALL_ROOM,
    HYRULE_CASTLE_MAIN_WEST_ROOM,
    AlttpSnapshot,
    snapshot_to_diag,
    zelda_rescued_accepted,
)
from alttp.room_map import load_room_map
from alttp.route_report import RoutePhaseResult, SegmentResult
from alttp.startup import snapshot_env

MAP_ID = "room_61"
WEST_DOOR_LABEL = "west_to_0x60"
ROOM = HYRULE_CASTLE_MAIN_HALL_ROOM
WEST_ROOM = HYRULE_CASTLE_MAIN_WEST_ROOM


def in_main_hall(snap: AlttpSnapshot) -> bool:
    return in_room(snap, ROOM)


def left_main_hall_west(snap: AlttpSnapshot) -> bool:
    """True when indoors in the west-adjacent room measured from 0x61."""
    return in_room(snap, WEST_ROOM)


def near_west_door(snap: AlttpSnapshot, *, tolerance: int | None = None) -> bool:
    """True when in main hall near the west door approach (from map)."""
    if not in_main_hall(snap):
        return False
    door = load_room_map(MAP_ID).door(WEST_DOOR_LABEL)
    if door is None:
        return False
    ax, ay = door.approach_xy
    tol = door.tolerance_for("default", 24) if tolerance is None else tolerance
    return abs(snap.link_x - ax) <= tol and abs(snap.link_y - ay) <= tol


def evaluate_acceptance(snapshot: AlttpSnapshot) -> dict[str, bool]:
    """Diagnostic acceptance for planned Zelda work (not a live segment exit)."""
    acceptance = evaluate_prefix_acceptance(snapshot)
    acceptance.update(
        {
            "main_hall": in_main_hall(snapshot),
            "left_main_hall_west": left_main_hall_west(snapshot),
            "in_zelda_cell": snapshot.in_zelda_cell,
            "zelda_follower": zelda_rescued_accepted(snapshot),
        }
    )
    return acceptance


def run_from_main_hall(
    env: object,
    *,
    source: str = "state_load_dev",
) -> SegmentResult:
    """Planned scaffold only — does not run the dungeon prefix.

    Returns ok only when Zelda is already rescued. Intermediate rooms report
    honest blockers; measured continuous play uses
    :func:`alttp.opening_route.castle_dungeon.run_from_main_hall`.
    """
    snap = snapshot_env(env)
    notes = [
        "main_hall_to_zelda is a planned scaffold, not a live Segment.",
        "Use castle_dungeon_prefix for continuous 0x61 → 0x60 → 0x50.",
        "Zelda follower remains unmeasured from natural entry.",
    ]
    settle_phase = RoutePhaseResult(
        phase="planned_scaffold",
        ok=True,
        frames=0,
        snapshot=snap,
        detail="no route action; Zelda path not measured",
        diag=snapshot_to_diag(snap),
    )
    acc = evaluate_acceptance(snap)
    if zelda_rescued_accepted(snap):
        return SegmentResult(
            ok=True,
            phase="zelda_rescued",
            frames=0,
            snapshot=snap,
            phases=[settle_phase],
            source=source,
            acceptance=acc,
            notes=notes + ["Already at full acceptance."],
        )
    if snap.in_zelda_cell:
        return SegmentResult(
            ok=False,
            phase="in_zelda_cell",
            frames=0,
            snapshot=snap,
            phases=[settle_phase],
            source=source,
            acceptance=acc,
            blocker="in Zelda cell without follower; rescue dialogue not implemented",
            notes=notes,
        )
    if left_main_hall_west(snap):
        return SegmentResult(
            ok=False,
            phase="left_main_hall_west",
            frames=0,
            snapshot=snap,
            phases=[settle_phase],
            source=source,
            acceptance=acc,
            blocker=(
                "room 0x60 is an intermediate continuous checkpoint; "
                "resume via castle_dungeon_prefix"
            ),
            notes=notes,
        )
    return SegmentResult(
        ok=False,
        phase="zelda_path_planned",
        frames=0,
        snapshot=snap,
        phases=[settle_phase],
        source=source,
        acceptance=acc,
        blocker="main hall → Zelda cell is not measured as a natural-entry route",
        notes=notes,
    )


def run_from_state(
    state_name: str = "CastleMain",
    *,
    close: bool = True,
) -> SegmentResult:
    """Development diagnostic for planned Zelda scaffold (no prefix play)."""
    from alttp.opening_route.runner import run_from_state as _run_from_state

    return _run_from_state(state_name, run_from_main_hall, close=close)


def run_from_b1_west(env: object, *, source: str = "state_load_dev") -> SegmentResult:
    """Walk the real stairs and earn both keys through play before rescuing Zelda.

    The cell is in B3 room 0x80, reached through 0x71 and 0x70. The old
    west_to_0x80 map edge is a wall, not a dungeon transition. Combat uses
    the existing tactics; any restore lookahead must be disclosed by the
    caller's audit and verified separately with a forward input replay.
    """
    from alttp.primitives import (
        PrimitiveResult, Waypoint, collect_nearby, fight_nearby, interact_until, move_to,
        move_until, settle_control,
    )
    from alttp.opening_route.castle_dungeon import earn_castle_big_key
    from alttp.route_report import phase_from_primitive

    phases: list[RoutePhaseResult] = []

    def record(label, result):
        phases.append(phase_from_primitive(label, result))
        if not result.ok:
            raise RuntimeError(result.reason)
        return result

    def walk(room, label, tolerance=1):
        point = load_room_map(f"room_{room:02x}").point(label)
        if point is None:
            raise RuntimeError(f"missing geometry room_{room:02x}:{label}")
        waypoint = Waypoint(point.x, point.y, tolerance=tolerance, room=room, label=label)
        moved = move_to(env, waypoint, step_size=1, max_frames=900)
        frames = moved.frames
        for _ in range(2):
            if moved.ok or moved.snapshot.has_control:
                break
            settled = settle_control(env, max_frames=480)
            frames += settled.frames
            if not settled.ok:
                moved = settled
                break
            moved = move_to(env, waypoint, step_size=1, max_frames=900)
            frames += moved.frames
        return record(label, PrimitiveResult(moved.ok, moved.reason, frames, moved.snapshot))

    def push(label, button, predicate, limit=600, settle=True):
        record(label, move_until(env, (button,), predicate, max_frames=limit, step_size=1))
        if settle:
            record(label + "_settle", settle_control(env, max_frames=480))

    def fight(label, distance=180):
        record(label, fight_nearby(
            env, max_distance=distance, attack_distance=50, max_cycles=450,
        ))
        record(label + "_settle", settle_control(env, max_frames=480))

    blocker = ""
    try:
        snap = snapshot_env(env)
        if not in_room(snap, 0x81) or snap.follower != 0 or not snap.has_control:
            raise RuntimeError("rescue requires controllable 0x81 without follower")
        for label in ("rescue_lower_corner", "rescue_lower_north", "rescue_stair_foot"):
            walk(0x81, label)
        push("east_stair", "UP", lambda s: s.layer == 0 and s.has_control)
        walk(0x81, "rescue_stair_head")
        walk(0x81, "rescue_upper_corner", 0)
        walk(0x81, "rescue_upper_south", 0)
        walk(0x81, "rescue_west_stair")
        push("west_stair", "UP", lambda s: s.layer == 1 and s.has_control)
        walk(0x81, "rescue_north_door")
        push("north_to_71", "UP", lambda s: in_room(s, 0x71))
        fight("west_shutter_guard")
        walk(0x71, "rescue_east_door")
        walk(0x71, "rescue_east_guard")
        fight("east_key_guard")
        record("small_key_pickup", collect_nearby(
            env, (0xE4,), max_distance=300, max_frames=600,
        ))
        if snapshot_env(env).num_keys < 1:
            raise RuntimeError("east guard key was not earned")
        walk(0x71, "rescue_alcove_nw")
        # NW diagonal enters the west mouth, then the room scroll settles.
        record("alcove_wrap", move_until(
            env, ("UP", "LEFT"), lambda s: s.link_x <= 688 and s.has_control,
            max_frames=400, step_size=1,
        ))
        walk(0x71, "rescue_east_door")
        walk(0x71, "rescue_key_door")
        record("key_door_settle", settle_control(env, max_frames=480))
        walk(0x71, "rescue_north_landing")
        push("north_lower_stair", "UP", lambda s: s.layer == 1 and s.has_control)
        walk(0x71, "rescue_spiral_align")
        walk(0x71, "rescue_spiral_mouth")
        push("down_to_70", "UP", lambda s: in_room(s, 0x70))
        walk(0x70, "rescue_stair_hall")
        walk(0x70, "rescue_down_stair")
        push("down_to_80", "UP", lambda s: in_room(s, 0x80))
        walk(0x80, "rescue_first_guard")
        fight("cell_entry_guard", 160)
        walk(0x80, "rescue_south_mid")
        walk(0x80, "rescue_ball_approach")
        key = earn_castle_big_key(env, source=source, max_decisions=6000)
        phases.append(RoutePhaseResult(
            phase="earn_big_key", ok=key.ok, frames=key.frames,
            snapshot=key.snapshot, detail=key.blocker or "; ".join(key.notes),
        ))
        if not key.ok:
            raise RuntimeError(key.blocker)
        walk(0x80, "rescue_cell_align")
        # UP into the lock opens the cell and starts Zelda's first message.
        push("cell_unlock", "UP", lambda s: s.is_text_mode, 240, settle=False)
        record("zelda_dialogue", interact_until(
            env, lambda s: s.has_zelda_follower, max_cycles=200,
        ))
        record("rescue_settle", settle_control(env, max_frames=480))
    except RuntimeError as exc:
        blocker = str(exc)
    snap = snapshot_env(env)
    return SegmentResult(
        ok=not blocker and snap.has_zelda_follower and snap.in_zelda_cell,
        phase="zelda_rescued" if not blocker else (phases[-1].phase if phases else "entry"),
        frames=sum(p.frames for p in phases), snapshot=snap, phases=phases,
        source=source,
        acceptance={"zelda_follower": snap.has_zelda_follower, "in_zelda_cell": snap.in_zelda_cell},
        blocker=blocker,
        notes=["Both small key and big key earned through controller play.",
               "Controller-only suffix; audit and replay the power-on prefix separately."],
    )
