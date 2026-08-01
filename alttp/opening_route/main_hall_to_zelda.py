"""Compatibility aggregate: main hall → Zelda follower (still incomplete).

The measured first-dungeon prefix now lives in
:mod:`alttp.opening_route.castle_dungeon`: it composes the continuous
``0x61 → 0x60 → 0x50`` room-engine edges.  This module retains the historical
``main_hall_to_zelda`` API and the truthful Zelda-follower acceptance contract;
it reports the prefix as partial until a measured route reaches Zelda's cell.
"""

from __future__ import annotations

from alttp import primitives
from alttp.opening_route.castle_dungeon import (
    evaluate_prefix_acceptance,
    run_from_main_hall as run_dungeon_prefix_from_main_hall,
)
from alttp.opening_route.room_engine import (
    at_door_destination,
    clear_room,
    exit_via_door,
    in_room,
)
from alttp.ram import (
    HYRULE_CASTLE_MAIN_HALL_ROOM,
    HYRULE_CASTLE_MAIN_WEST_ROOM,
    HYRULE_CASTLE_NW_ROOM,
    AlttpSnapshot,
    snapshot_to_diag,
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
    """Run the measured dungeon prefix; remain partial until Zelda follower.

    ``castle_dungeon`` owns the edge order and room-boundary checks.  Keeping
    this aggregate as a compatibility layer prevents its Zelda-oriented name
    from hiding which room edges actually ran.
    """
    settle = primitives.settle_control(env)
    snap = settle.snapshot
    notes = [
        "Compatibility aggregate over castle_dungeon MAIN_HALL_TO_NW_PREFIX.",
        "Measured continuous prefix: room 0x61 → 0x60 → 0x50.",
        "Zelda follower remains planned and is required for aggregate success.",
    ]

    settle_phase = RoutePhaseResult(
        phase="settle_control",
        ok=snap.has_control,
        frames=settle.frames,
        snapshot=snap,
        detail="settled before main_hall_to_zelda aggregate",
        diag=snapshot_to_diag(snap),
    )
    if zelda_rescued_accepted(snap):
        return SegmentResult(
            ok=True,
            phase="zelda_rescued",
            frames=settle.frames,
            snapshot=snap,
            phases=[settle_phase],
            source=source,
            acceptance=evaluate_acceptance(snap),
            notes=notes + ["Already at full acceptance."],
        )
    if snap.in_zelda_cell:
        return SegmentResult(
            ok=False,
            phase="in_zelda_cell",
            frames=settle.frames,
            snapshot=snap,
            phases=[settle_phase],
            source=source,
            acceptance=evaluate_acceptance(snap),
            blocker="in Zelda cell without follower; rescue dialogue not implemented",
            notes=notes,
        )
    # Preserve the historical partial response for callers resuming at 0x60.
    if left_main_hall_west(snap):
        return SegmentResult(
            ok=False,
            phase="left_main_hall_west",
            frames=settle.frames,
            snapshot=snap,
            phases=[settle_phase],
            source=source,
            acceptance=evaluate_acceptance(snap),
            blocker="room 0x60 is an intermediate prefix checkpoint; resume via castle_dungeon",
            notes=notes,
        )
    if in_room(snap, HYRULE_CASTLE_NW_ROOM):
        return SegmentResult(
            ok=False,
            phase="reached_room_50",
            frames=settle.frames,
            snapshot=snap,
            phases=[settle_phase],
            source=source,
            acceptance=evaluate_acceptance(snap),
            blocker="room 0x50 → Zelda cell is not measured as a natural-entry route",
            notes=notes,
        )

    prefix = run_dungeon_prefix_from_main_hall(env, source=source)
    frames = settle.frames + prefix.frames
    phases = [settle_phase, *prefix.phases]
    final = prefix.snapshot
    if not prefix.ok:
        return SegmentResult(
            ok=False,
            phase=prefix.phase,
            frames=frames,
            snapshot=final,
            phases=phases,
            source=source,
            acceptance=evaluate_acceptance(final),
            blocker=prefix.blocker,
            notes=notes + prefix.notes,
        )
    return SegmentResult(
        ok=False,
        phase="reached_room_50",
        frames=frames,
        snapshot=final,
        phases=phases,
        source=source,
        acceptance=evaluate_acceptance(final),
        blocker="room 0x50 → Zelda cell is not measured as a natural-entry route",
        notes=notes + prefix.notes,
    )


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
