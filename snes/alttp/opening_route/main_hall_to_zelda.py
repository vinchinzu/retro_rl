"""Planned Zelda-path helpers (not a live continuous segment).

The measured first-dungeon prefix lives in
:mod:`alttp.opening_route.castle_dungeon` (``0x61 → 0x60 → 0x50``).

This module keeps map-derived helpers and acceptance keys for the future
main-hall → Zelda cell hop. It is **not** registered in the Segment registry
(a segment must be able to succeed under its exit contract; Zelda follower
is not measured yet).

Use ``castle_dungeon_prefix`` for continuous play through room ``0x50``.
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
    from alttp.startup import build_boot_env

    env = build_boot_env(state_name)
    try:
        env.reset()  # type: ignore[attr-defined]
        return run_from_main_hall(env, source="state_load_dev")
    finally:
        if close:
            env.close()  # type: ignore[attr-defined]
