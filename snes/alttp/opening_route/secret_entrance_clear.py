"""Secret-entrance clear: post-sword room 0x55 → outdoors courtyard pocket.

Segment success is **left_secret_entrance only** (stairs exit outdoors).
Does not claim Zelda rescue — that remains planned after the continuous
tip (``castle_dungeon_prefix`` through room ``0x50``). Later-route
follower/cell/sanctuary flags live under ``diagnostics``, not acceptance.

Composes after ``castle_to_sword`` / ``FighterSword`` predecessor. Clean
intervention only — no progression writes or door warps.

Geometry authority: ``maps/room_55.json`` door ``stairs_to_courtyard`` via
:mod:`alttp.opening_route.room_engine`. Segment code is thin glue
(hold-up dismiss + edge acceptance). Do not reintroduce open-loop
``LEFT×100 + DOWN×250`` macros here.
"""

from __future__ import annotations

from collections.abc import Sequence

from alttp import primitives
from alttp.opening_route.anchors import ROOM_55_SOUTH_Y_MIN
from alttp.opening_route.castle_to_sword import dismiss_hold_up_item
from alttp.opening_route.room_engine import (
    at_door_destination,
    in_room,
    move_path_combat_aware,
    run_room_edge,
)
from alttp.opening_route.runner import PhaseFn
from alttp.ram import (
    SECRET_PASSAGE_ROOM,
    AlttpSnapshot,
    room_label,
    snapshot_to_diag,
    zelda_rescued_accepted,
)
from alttp.room_map import load_room_map
from alttp.route_report import RoutePhaseResult, SegmentResult, segment_result_factory
from alttp.startup import BootEnv, snapshot_env

_REPORT = segment_result_factory("alttp_secret_entrance_clear_report")

MAP_ID = "room_55"
STAIRS_DOOR_LABEL = "stairs_to_courtyard"
ROOM = SECRET_PASSAGE_ROOM

# Soft y cap: deeper south without stair alignment fails to transition.
# Still used by approach helper diagnostics; geometry path is map authority.
SOUTH_CHAMBER_Y_MAX = 2965


def _stairs_door():
    return load_room_map(MAP_ID).door(STAIRS_DOOR_LABEL)


def _stairs_align_xy() -> tuple[int, int]:
    door = _stairs_door()
    if door is None:
        return 2672, 2916
    return door.approach_xy


# Re-export measured stairs constants (map-backed; fallback for import stability).
STAIRS_ALIGN_X, STAIRS_ALIGN_Y = _stairs_align_xy()
STAIRS_ALIGN_TOLERANCE = 6

__all__ = [
    "MAP_ID",
    "STAIRS_DOOR_LABEL",
    "STAIRS_ALIGN_TOLERANCE",
    "STAIRS_ALIGN_X",
    "STAIRS_ALIGN_Y",
    "SOUTH_CHAMBER_Y_MAX",
    "approach_south_chamber",
    "ensure_sword_control",
    "evaluate_acceptance",
    "evaluate_diagnostics",
    "exit_secret_entrance_stairs",
    "left_secret_entrance",
    "run_from_state",
    "run_from_sword",
    "south_chamber_waypoints",
]


def left_secret_entrance(snapshot: AlttpSnapshot) -> bool:
    """True when Link is no longer indoors in the secret-entrance room."""
    if not snapshot.indoors:
        return True
    return snapshot.room_base_id != SECRET_PASSAGE_ROOM


def evaluate_acceptance(snapshot: AlttpSnapshot) -> dict[str, bool]:
    """Contract keys for this segment only.

    Segment ``ok`` uses ``left_secret_entrance``. Later-route Zelda flags live
    in :func:`evaluate_diagnostics` so they never look like exit success.
    """
    return {
        "fighter_sword_ram": snapshot.has_fighter_sword,
        "in_secret_passage": snapshot.in_secret_passage,
        "hold_up_cleared": not snapshot.is_hold_up_item,
        "left_secret_entrance": left_secret_entrance(snapshot),
    }


def evaluate_diagnostics(snapshot: AlttpSnapshot) -> dict[str, bool]:
    """Log-only later-route flags (not part of this segment's exit)."""
    return {
        "zelda_follower": zelda_rescued_accepted(snapshot),
        "in_zelda_cell": snapshot.in_zelda_cell,
        "in_sanctuary": snapshot.in_sanctuary,
    }


def ensure_sword_control(env: BootEnv) -> RoutePhaseResult:
    """Dismiss hold-up-item and require fighter sword + control."""
    # LEFT dismiss first: primitives.settle_control waits for hold-up clear
    # but only advances no_action/text, so active LEFT is required.
    frames = dismiss_hold_up_item(env)
    settle = primitives.settle_control(env)
    frames += settle.frames
    snap = settle.snapshot
    ok = (
        snap.has_fighter_sword
        and snap.has_control
        and (not snap.is_hold_up_item)
        and snap.in_secret_passage
    )
    return RoutePhaseResult(
        phase="ensure_sword_control",
        ok=ok,
        frames=frames,
        snapshot=snap,
        detail=(
            "sword equipped, hold-up cleared, controllable in secret entrance"
            if ok
            else (
                f"sword={snap.has_fighter_sword} hold_up={snap.is_hold_up_item} "
                f"control={snap.has_control} room={room_label(snap.room_base_id)}"
            )
        ),
        diag=snapshot_to_diag(snap),
    )


def south_chamber_waypoints() -> list[primitives.Waypoint]:
    """Map-backed waypoints from sword spawn corridor into south chamber."""
    room_map = load_room_map(MAP_ID)
    door = room_map.door(STAIRS_DOOR_LABEL)
    if door is None:
        return []
    # Only walk as far as the south_chamber point (not stairs align).
    wps: list[primitives.Waypoint] = []
    for label in door.path:
        if label == "stairs_align":
            break
        pt = room_map.point(label)
        if pt is None:
            continue
        wps.append(
            primitives.Waypoint(
                pt.x,
                pt.y,
                tolerance=door.tolerance_for(label),
                room=ROOM,
                label=label,
            )
        )
    return wps


def approach_south_chamber(env: BootEnv) -> RoutePhaseResult:
    """Walk from uncle corridor into the south multi-screen combat chamber.

    Uses map path waypoints (combat-aware), not open-loop frame macros.
    """
    frames = 0
    settle = primitives.settle_control(env)
    frames += settle.frames
    start = settle.snapshot
    if not start.in_secret_passage:
        return RoutePhaseResult(
            phase="approach_south_chamber",
            ok=False,
            frames=frames,
            snapshot=start,
            detail="not in secret entrance",
            diag=snapshot_to_diag(start),
        )
    if start.link_y >= ROOM_55_SOUTH_Y_MIN:
        return RoutePhaseResult(
            phase="approach_south_chamber",
            ok=True,
            frames=frames,
            snapshot=start,
            detail=f"already south chamber xy=({start.link_x},{start.link_y})",
            diag=snapshot_to_diag(start),
        )

    room_map = load_room_map(MAP_ID)
    wps = south_chamber_waypoints()
    if not wps:
        return RoutePhaseResult(
            phase="approach_south_chamber",
            ok=False,
            frames=frames,
            snapshot=start,
            detail=f"no south-chamber path in {MAP_ID}",
            diag=snapshot_to_diag(start),
        )

    path = move_path_combat_aware(
        env, wps, room=ROOM, policy=room_map.clear_policy
    )
    frames += path.frames
    settle = primitives.settle_control(env)
    frames += settle.frames
    snap = settle.snapshot
    ok = (
        snap.in_secret_passage
        and snap.link_y >= ROOM_55_SOUTH_Y_MIN
        and snap.link_y <= SOUTH_CHAMBER_Y_MAX + 20
    )
    return RoutePhaseResult(
        phase="approach_south_chamber",
        ok=ok,
        frames=frames,
        snapshot=snap,
        detail=(
            f"south chamber (map path) xy=({snap.link_x},{snap.link_y}) "
            f"reason={path.reason}"
            if ok
            else (
                f"missed south chamber xy=({snap.link_x},{snap.link_y}) "
                f"reason={path.reason}"
            )
        ),
        diag=snapshot_to_diag(snap),
    )


def exit_secret_entrance_stairs(env: BootEnv) -> RoutePhaseResult:
    """Map-backed stairs exit via room_engine (``stairs_to_courtyard``)."""
    door = _stairs_door()
    if door is None:
        snap = snapshot_env(env)
        return RoutePhaseResult(
            phase="exit_secret_entrance_stairs",
            ok=False,
            frames=0,
            snapshot=snap,
            detail=f"door {STAIRS_DOOR_LABEL!r} missing from {MAP_ID}",
            diag=snapshot_to_diag(snap),
        )

    edge = run_room_edge(
        env,
        MAP_ID,
        STAIRS_DOOR_LABEL,
        clear=True,
        source="secret_entrance_clear",
        notes=[
            f"room_engine map={MAP_ID} door={STAIRS_DOOR_LABEL}",
            "Geometry authority: maps/room_55.json",
        ],
        acceptance_fn=lambda s: {
            "left_secret_entrance": left_secret_entrance(s),
            "at_door_dest": at_door_destination(s, door),
            "in_origin_room": in_room(s, ROOM),
        },
    )
    # Flatten room_engine phases into one stairs phase for segment reports
    # when used as a standalone phase, but preserve detail.
    snap = edge.snapshot
    return RoutePhaseResult(
        phase="exit_secret_entrance_stairs",
        ok=edge.ok and left_secret_entrance(snap),
        frames=edge.frames,
        snapshot=snap,
        detail=edge.blocker or (
            f"exited secret entrance → outdoors "
            f"screen=0x{snap.screen_id:02X} xy=({snap.link_x},{snap.link_y})"
            if left_secret_entrance(snap)
            else f"stairs exit incomplete: {edge.phase}"
        ),
        diag={
            **snapshot_to_diag(snap),
            "roomEnginePhase": edge.phase,
            "roomEngineNotes": list(edge.notes),
        },
    )


def _compose_room_engine_clear(
    env: BootEnv,
    *,
    source: str,
    pre_phases: list[RoutePhaseResult],
    pre_frames: int,
) -> SegmentResult:
    """Hold-up clear then room_engine edge; merge into segment contract."""
    door = _stairs_door()
    edge = run_room_edge(
        env,
        MAP_ID,
        STAIRS_DOOR_LABEL,
        clear=True,
        source=source,
        notes=[
            "Secret-entrance clear = stairs exit outdoors (screen 0x1B pocket).",
            f"room_engine map={MAP_ID} door={STAIRS_DOOR_LABEL}",
            "Next hop: pocket_to_main_hall (bush-cut → door → room 0x61).",
            "Do not claim Zelda rescue until follower_indicator==1.",
        ],
        acceptance_fn=evaluate_acceptance,
    )
    frames = pre_frames + edge.frames
    phases = list(pre_phases) + list(edge.phases)
    snap = edge.snapshot
    ok = edge.ok and left_secret_entrance(snap)
    return _REPORT(
        ok=ok,
        phase="secret_entrance_exited" if ok else (edge.phase or "room_blocked"),
        frames=frames,
        snapshot=snap,
        phases=phases,
        source=source,
        acceptance=evaluate_acceptance(snap),
        diagnostics=evaluate_diagnostics(snap),
        blocker=(
            ""
            if ok
            else (
                edge.blocker
                or "still in secret entrance after room_engine stairs edge"
            )
        ),
        notes=list(edge.notes)
        + (
            [
                "Secret entrance finished (outdoors). Next: "
                "pocket_to_main_hall → B1 → Zelda.",
            ]
            if ok
            else []
        ),
    )


SWORD_CLEAR_PHASES = (
    ensure_sword_control,
    approach_south_chamber,
    exit_secret_entrance_stairs,
)

_SWORD_CLEAR_NOTES = (
    "Secret-entrance clear = stairs exit outdoors (screen 0x1B pocket).",
    f"Geometry: maps/{MAP_ID}.json door {STAIRS_DOOR_LABEL} via room_engine.",
    "Next hop: pocket_to_main_hall (bush-cut → door → room 0x61).",
    "Do not claim Zelda rescue until follower_indicator==1.",
)


def run_from_sword(
    env: BootEnv,
    *,
    source: str = "state_load_dev",
    phases: Sequence[PhaseFn] | None = None,
) -> SegmentResult:
    """Run post-sword secret-entrance clear assuming fighter sword obtained.

    Default path: hold-up clear → room_engine ``stairs_to_courtyard`` edge.
    Segment ``ok`` means ``left_secret_entrance`` only (not Zelda). Pass
    ``phases`` to run a subset (e.g. ensure + approach for unit smokes).
    """
    from alttp.opening_route.runner import run_phases

    if phases is not None:
        return run_phases(
            env,
            list(phases),
            evaluate_acceptance=evaluate_acceptance,
            evaluate_diagnostics=evaluate_diagnostics,
            success_when=left_secret_entrance,
            source=source,
            notes=_SWORD_CLEAR_NOTES,
            success_phase="secret_entrance_exited",
            success_notes=(
                "Secret entrance finished (outdoors). Next: "
                "pocket_to_main_hall → B1 → Zelda.",
            ),
            partial_blocker=(
                "still in secret entrance after phases; stairs exit incomplete"
            ),
            result_factory=_REPORT,
        )

    # Default continuous path: hold-up then single room_engine edge.
    frames = 0
    pre_phases: list[RoutePhaseResult] = []
    control = ensure_sword_control(env)
    frames += control.frames
    pre_phases.append(control)
    if not control.ok:
        snap = control.snapshot
        return _REPORT(
            ok=False,
            phase=control.phase,
            frames=frames,
            snapshot=snap,
            phases=pre_phases,
            source=source,
            acceptance=evaluate_acceptance(snap),
            diagnostics=evaluate_diagnostics(snap),
            blocker=control.detail or "ensure_sword_control failed",
            notes=list(_SWORD_CLEAR_NOTES),
        )
    if left_secret_entrance(control.snapshot):
        snap = control.snapshot
        return _REPORT(
            ok=True,
            phase="secret_entrance_exited",
            frames=frames,
            snapshot=snap,
            phases=pre_phases,
            source=source,
            acceptance=evaluate_acceptance(snap),
            diagnostics=evaluate_diagnostics(snap),
            notes=list(_SWORD_CLEAR_NOTES)
            + ["Already outdoors before room_engine edge."],
        )
    return _compose_room_engine_clear(
        env, source=source, pre_phases=pre_phases, pre_frames=frames
    )


def run_from_state(
    state_name: str = "FighterSword",
    *,
    close: bool = True,
) -> SegmentResult:
    """Development diagnostic from a saved fighter-sword state."""
    from alttp.opening_route.runner import run_from_state as _run_from_state

    return _run_from_state(
        state_name,
        run_from_sword,
        close=close,
        settle=True,
    )
