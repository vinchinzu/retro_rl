"""Main hall (room 0x61) → Zelda cell / follower.

Thin segment over :mod:`alttp.opening_route.room_engine` + ``maps/room_61.json``.

Measured main-hall leg (2026-07-31, ``CastleMain``):

1. Clear nearby castle hostiles.
2. Navigate side corridor via map door path ``west_to_0x60``.
3. Hold LEFT → room ``0x60``.

Beyond room 0x61 (B1 maze → cell → ``$F3CC==1``) is **not** implemented yet.
Geometry authority is the JSON map — do not redeclare door coords here.
"""

from __future__ import annotations

from alttp.opening_route.room_engine import (
    at_door_destination,
    clear_room,
    exit_via_door,
    in_room,
    run_room_edge,
)
from alttp.ram import (
    HYRULE_CASTLE_MAIN_HALL_ROOM,
    HYRULE_CASTLE_MAIN_WEST_ROOM,
    AlttpSnapshot,
    zelda_rescued_accepted,
)
from alttp.room_map import load_room_map
from alttp.route_report import RoutePhaseResult, SegmentResult

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
    return {
        "fighter_sword_ram": snapshot.has_fighter_sword,
        "main_hall": in_main_hall(snapshot),
        "left_main_hall_west": left_main_hall_west(snapshot),
        "in_zelda_cell": snapshot.in_zelda_cell,
        "zelda_follower": zelda_rescued_accepted(snapshot),
    }


def clear_main_hall(env: object) -> RoutePhaseResult:
    """Clear hostiles in room 0x61 (delegates to room_engine)."""
    room_map = load_room_map(MAP_ID)
    door = room_map.door(WEST_DOOR_LABEL)
    return clear_room(
        env,
        room_map,
        phase="clear_main_hall",
        already_past=lambda s: left_main_hall_west(s)
        or s.in_zelda_cell
        or zelda_rescued_accepted(s)
        or (door is not None and at_door_destination(s, door)),
    )


def exit_main_hall_west(env: object) -> RoutePhaseResult:
    """Approach west door and push LEFT → room 0x60."""
    return exit_via_door(
        env,
        load_room_map(MAP_ID),
        WEST_DOOR_LABEL,
        phase="exit_main_hall_west",
    )


def run_from_main_hall(
    env: object,
    *,
    source: str = "state_load_dev",
) -> SegmentResult:
    """Play main-hall clear + west exit; partial until Zelda follower.

    Isolated edge success (room 0x60) is honest progress from ``run_room_edge``
    (``ok=True``). This multi-hop segment reinterprets that as partial until
    follower acceptance is met.
    """
    room_map = load_room_map(MAP_ID)
    door = room_map.door(WEST_DOOR_LABEL)
    notes = [
        "Main hall via room_engine + maps/room_61.json.",
        f"Known doors: {[d.label for d in room_map.doors]}",
        "verification=isolated for west exit; Zelda follower still planned.",
    ]

    def _early(
        snap: AlttpSnapshot,
        phases: list,
        frames: int,
    ) -> SegmentResult | None:
        if zelda_rescued_accepted(snap):
            return SegmentResult(
                ok=True,
                phase="zelda_rescued",
                frames=frames,
                snapshot=snap,
                phases=phases,
                source=source,
                acceptance=evaluate_acceptance(snap),
                notes=notes + ["Already at full acceptance."],
            )
        if snap.in_zelda_cell:
            return SegmentResult(
                ok=False,
                phase="in_zelda_cell",
                frames=frames,
                snapshot=snap,
                phases=phases,
                source=source,
                acceptance=evaluate_acceptance(snap),
                blocker="in Zelda cell without follower; rescue dialogue not implemented",
                notes=notes,
            )
        return None

    edge = run_room_edge(
        env,
        MAP_ID,
        WEST_DOOR_LABEL,
        clear=True,
        source=source,
        notes=notes,
        acceptance_fn=evaluate_acceptance,
        early_check=_early,
    )

    if zelda_rescued_accepted(edge.snapshot):
        return SegmentResult(
            ok=True,
            phase="zelda_rescued",
            frames=edge.frames,
            snapshot=edge.snapshot,
            phases=edge.phases,
            source=source,
            acceptance=evaluate_acceptance(edge.snapshot),
            notes=list(edge.notes),
        )

    # Multi-hop: isolated west exit is progress but not segment success.
    west = left_main_hall_west(edge.snapshot) or (
        door is not None and at_door_destination(edge.snapshot, door)
    )
    if edge.ok and west:
        return SegmentResult(
            ok=False,
            phase="left_main_hall_west",
            frames=edge.frames,
            snapshot=edge.snapshot,
            phases=edge.phases,
            source=source,
            acceptance=evaluate_acceptance(edge.snapshot),
            blocker="path after room 0x60/0x50 → Zelda cell not implemented",
            notes=list(edge.notes),
        )

    return edge


def run_from_state(
    state_name: str = "CastleMain",
    *,
    close: bool = True,
) -> SegmentResult:
    """Development diagnostic from a main-hall checkpoint state."""
    from alttp.startup import build_boot_env

    env = build_boot_env(state_name)
    try:
        env.reset()  # type: ignore[attr-defined]
        return run_from_main_hall(env, source="state_load_dev")
    finally:
        if close:
            env.close()  # type: ignore[attr-defined]
