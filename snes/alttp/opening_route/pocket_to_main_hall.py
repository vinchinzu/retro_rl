"""Courtyard secret pocket → main castle door → room 0x61.

Measured 2026-07-30 headless from continuous tip ``courtyard_secret_pocket``
(after ``secret_entrance_clear`` stairs exit / FighterSword predecessor):

1. **Route** — lift the two bushes that close the hedge pocket, then walk
   south-west into the open courtyard. Sword swings are not used here.
2. **Approach** — overworld walk to the door column. Bushes off that walk
   stay put.
3. **Trigger** — align x≈2040, hold UP → indoors main hall room ``0x61``.

Geometry authority: ``maps/screen_1b_courtyard.json`` door
``main_door_to_0x61``. The pocket lifts are play input from the landing;
approach and trigger coords load from the map. Do not grow
``enter_main_door`` special cases or redeclare door xy in Python.

Do not claim Zelda / Sanctuary from this hop alone.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

from alttp import primitives
from alttp.opening_route.runner import PhaseFn
from alttp.ram import (
    HYRULE_CASTLE_MAIN_HALL_ROOM,
    HYRULE_CASTLE_SCREEN,
    SECRET_PASSAGE_ROOM,
    AlttpSnapshot,
    room_label,
    snapshot_to_diag,
)
from alttp.room_map import KnownDoor, RoomMap, load_room_map
from alttp.route_report import RoutePhaseResult, SegmentResult, segment_result_factory
from alttp.startup import BootEnv, action_for, no_action, snapshot_env, step_frames

_REPORT = segment_result_factory("alttp_pocket_to_main_hall_report")

MAP_ID = "screen_1b_courtyard"
MAIN_DOOR_LABEL = "main_door_to_0x61"

# Open-courtyard / gardens window after pocket escape (route tier).
COURTYARD_OPEN_Y_MIN = 1880
COURTYARD_OPEN_X_MAX = 2220


@lru_cache(maxsize=1)
def courtyard_map() -> RoomMap:
    return load_room_map(MAP_ID)


@lru_cache(maxsize=1)
def main_door() -> KnownDoor:
    door = courtyard_map().door(MAIN_DOOR_LABEL)
    if door is None:
        raise KeyError(f"door {MAIN_DOOR_LABEL!r} missing from {MAP_ID}")
    return door


def _map_point_xy(label: str, default: tuple[int, int]) -> tuple[int, int]:
    pt = courtyard_map().point(label)
    if pt is None:
        return default
    return pt.x, pt.y


# Map-backed measured constants (fallbacks keep import stable if map edited).
COURTYARD_SECRET_POCKET_X, COURTYARD_SECRET_POCKET_Y = _map_point_xy(
    "secret_stairs_pocket", (2248, 1755)
)
COURTYARD_SECRET_POCKET_TOLERANCE = 48
MAIN_DOOR_APPROACH_X, MAIN_DOOR_APPROACH_Y = main_door().approach_xy
MAIN_DOOR_APPROACH_TOLERANCE = main_door().tolerance_for("main_door_approach", 24)


# $0308 bit 0x80. Same byte ``primitives.lift`` treats as the carry flag.
_CARRY_ADDR = 0x0308
_CARRY_BIT = 0x80


def _carrying(env: BootEnv) -> bool:
    return bool(int(env.get_ram()[_CARRY_ADDR]) & _CARRY_BIT)


def _until_still(env: BootEnv, face: str, limit: int) -> int:
    """Hold ``face`` until Link stops, goes indoors, or the limit hits."""
    frames = 0
    still = 0
    previous: tuple[int, int] | None = None
    for _ in range(limit):
        snap = snapshot_env(env)
        if snap.indoors or in_main_hall(snap):
            return frames
        step_frames(env, action_for(face), 1)
        frames += 1
        xy = (snapshot_env(env).link_x, snapshot_env(env).link_y)
        still = still + 1 if xy == previous else 0
        previous = xy
        if still >= 6:
            return frames
    return frames


def _step_until_x(env: BootEnv, x_max: int, *, limit: int = 24) -> int:
    frames = 0
    for _ in range(limit):
        snap = snapshot_env(env)
        if snap.indoors or snap.link_x <= x_max:
            return frames
        step_frames(env, action_for("LEFT"), 1)
        frames += 1
    return frames


def _lift_down(env: BootEnv) -> tuple[int, bool]:
    """Pick up the bush to the south and throw it. Returns frames, connected."""
    frames = 0
    step_frames(env, action_for("DOWN"), 2)
    frames += 2
    lifted = primitives.lift(env)
    frames += lifted.frames
    if not lifted.ok:
        return frames, False
    thrown = primitives.throw(env)
    frames += thrown.frames
    for _ in range(30):
        if not _carrying(env):
            break
        step_frames(env, no_action(), 1)
        frames += 1
    return frames, True


def _reentered_secret(snap: AlttpSnapshot) -> bool:
    return (
        snap.indoors
        and (not snap.dark_world)
        and snap.room_base_id == SECRET_PASSAGE_ROOM
    )


def in_main_hall(snap: AlttpSnapshot) -> bool:
    return (
        snap.indoors
        and (not snap.dark_world)
        and snap.room_base_id == HYRULE_CASTLE_MAIN_HALL_ROOM
    )


def in_open_courtyard(snap: AlttpSnapshot) -> bool:
    """Route-tier: outdoors 0x1B, sword, clearly outside secret-stairs pocket."""
    if snap.indoors or snap.dark_world:
        return False
    if snap.screen_id != HYRULE_CASTLE_SCREEN:
        return False
    if not snap.has_fighter_sword:
        return False
    # Outside the tight landing window OR past the open-courtyard thresholds.
    dx = abs(snap.link_x - COURTYARD_SECRET_POCKET_X)
    dy = abs(snap.link_y - COURTYARD_SECRET_POCKET_Y)
    if dx > COURTYARD_SECRET_POCKET_TOLERANCE or dy > COURTYARD_SECRET_POCKET_TOLERANCE:
        if snap.link_y >= COURTYARD_OPEN_Y_MIN or snap.link_x <= COURTYARD_OPEN_X_MAX:
            return True
    return snap.link_y >= COURTYARD_OPEN_Y_MIN and snap.link_x <= COURTYARD_OPEN_X_MAX


def near_main_door(snap: AlttpSnapshot) -> bool:
    if snap.indoors or snap.dark_world or snap.screen_id != HYRULE_CASTLE_SCREEN:
        return False
    return (
        abs(snap.link_x - MAIN_DOOR_APPROACH_X) <= MAIN_DOOR_APPROACH_TOLERANCE
        and abs(snap.link_y - MAIN_DOOR_APPROACH_Y) <= MAIN_DOOR_APPROACH_TOLERANCE
    )


def evaluate_acceptance(snapshot: AlttpSnapshot) -> dict[str, bool]:
    """On-path progress + exit keys for pocket → main hall."""
    return {
        "fighter_sword_ram": snapshot.has_fighter_sword,
        "outdoors_castle_screen": (
            (not snapshot.indoors)
            and (not snapshot.dark_world)
            and snapshot.screen_id == HYRULE_CASTLE_SCREEN
        ),
        "open_courtyard": in_open_courtyard(snapshot),
        "near_main_door": near_main_door(snapshot),
        "main_hall": in_main_hall(snapshot),
        "reentered_secret": _reentered_secret(snapshot),
    }


def _settle_transition(env: BootEnv, *, max_frames: int = 120) -> tuple[int, AlttpSnapshot]:
    frames = 0
    while frames < max_frames:
        snap = snapshot_env(env)
        if snap.indoors:
            return frames, snap
        if snap.has_control and snap.submodule == 0 and snap.game_mode in (0x07, 0x09):
            return frames, snap
        step_frames(env, no_action(), 2)
        frames += 2
    return frames, snapshot_env(env)


def _leave_pocket(env: BootEnv) -> tuple[int, int]:
    """Lift the pocket bush, walk around the next row, lift one garden bush.

    Returns frames and how many lifts connected. The pocket column is one
    body wide; the grab reaches the bush after an 8px step left of the
    landing x from ``secret_stairs_pocket``.
    """
    frames = 0
    lifts = 0
    frames += _until_still(env, "DOWN", 48)
    frames += _step_until_x(env, COURTYARD_SECRET_POCKET_X - 8)
    used, ok = _lift_down(env)
    frames += used
    if ok:
        lifts += 1
    frames += _until_still(env, "DOWN", 48)
    mark = snapshot_env(env).link_x
    frames += _step_until_x(env, mark - 16)
    frames += _until_still(env, "DOWN", 80)
    if lifts < 2 and not snapshot_env(env).indoors:
        used, ok = _lift_down(env)
        frames += used
        if ok:
            lifts += 1
    return frames, lifts


def _walk_to_door(env: BootEnv) -> tuple[int, bool]:
    """Follow the live overworld grid to the map door approach."""
    from alttp.overworld import find_overworld_path, read_overworld_tiles

    frames = 0
    snap = snapshot_env(env)
    path = find_overworld_path(
        read_overworld_tiles(env.get_ram()),
        (snap.link_x, snap.link_y),
        (MAIN_DOOR_APPROACH_X, MAIN_DOOR_APPROACH_Y),
    )
    if not path:
        return frames, False
    still = 0
    previous: tuple[int, int] | None = None
    for px, py in path:
        for _ in range(16):
            snap = snapshot_env(env)
            if snap.indoors or in_main_hall(snap) or near_main_door(snap):
                return frames, True
            if abs(snap.link_x - px) <= 2 and abs(snap.link_y - py) <= 2:
                break
            if abs(snap.link_x - px) >= abs(snap.link_y - py):
                face = "RIGHT" if px > snap.link_x else "LEFT"
            else:
                face = "DOWN" if py > snap.link_y else "UP"
            step_frames(env, action_for(face), 1)
            frames += 1
            xy = (snapshot_env(env).link_x, snapshot_env(env).link_y)
            still = still + 1 if xy == previous else 0
            previous = xy
            if still < 6:
                continue
            snap = snapshot_env(env)
            if abs(snap.link_x - MAIN_DOOR_APPROACH_X) > 3 and not snap.indoors:
                face = "RIGHT" if snap.link_x < MAIN_DOOR_APPROACH_X else "LEFT"
                step_frames(env, action_for(face), 4)
                frames += 4
                still = 0
                previous = None
                continue
            return frames, near_main_door(snap) or in_main_hall(snap)
    snap = snapshot_env(env)
    return frames, near_main_door(snap) or in_main_hall(snap)


def escape_hedge_pocket(env: BootEnv) -> RoutePhaseResult:
    """Route tier: lift two bushes and walk out of the secret-stairs pocket."""
    frames = 0
    settle = primitives.settle_control(env)
    frames += settle.frames
    start = settle.snapshot
    if in_main_hall(start):
        return RoutePhaseResult(
            phase="escape_hedge_pocket",
            ok=True,
            frames=frames,
            snapshot=start,
            detail="already in main hall",
            diag=snapshot_to_diag(start),
        )
    if start.indoors:
        return RoutePhaseResult(
            phase="escape_hedge_pocket",
            ok=False,
            frames=frames,
            snapshot=start,
            detail=f"expected outdoors pocket, got {room_label(start.room_base_id)}",
            diag=snapshot_to_diag(start),
        )
    if in_open_courtyard(start):
        return RoutePhaseResult(
            phase="escape_hedge_pocket",
            ok=True,
            frames=frames,
            snapshot=start,
            detail=(
                f"already open courtyard xy=({start.link_x},{start.link_y})"
            ),
            diag=snapshot_to_diag(start),
        )

    used, lifts = _leave_pocket(env)
    frames += used
    snap = snapshot_env(env)
    ok = in_open_courtyard(snap) or in_main_hall(snap)
    return RoutePhaseResult(
        phase="escape_hedge_pocket",
        ok=ok,
        frames=frames,
        snapshot=snap,
        detail=(
            f"open courtyard xy=({snap.link_x},{snap.link_y}) lifts={lifts}"
            if ok
            else (
                f"still pocket/hedge xy=({snap.link_x},{snap.link_y}) "
                f"indoors={snap.indoors} lifts={lifts}"
            )
        ),
        diag=snapshot_to_diag(snap),
    )


def approach_main_door(env: BootEnv) -> RoutePhaseResult:
    """Approach tier: walk the overworld grid to the map door approach."""
    frames = 0
    settle = primitives.settle_control(env)
    frames += settle.frames
    start = settle.snapshot
    if in_main_hall(start):
        return RoutePhaseResult(
            phase="approach_main_door",
            ok=True,
            frames=frames,
            snapshot=start,
            detail="already in main hall",
            diag=snapshot_to_diag(start),
        )
    if near_main_door(start):
        return RoutePhaseResult(
            phase="approach_main_door",
            ok=True,
            frames=frames,
            snapshot=start,
            detail=f"already near door xy=({start.link_x},{start.link_y})",
            diag=snapshot_to_diag(start),
        )

    used, reached = _walk_to_door(env)
    frames += used
    snap = snapshot_env(env)
    if not reached and not in_main_hall(snap) and not near_main_door(snap):
        return RoutePhaseResult(
            phase="approach_main_door",
            ok=False,
            frames=frames,
            snapshot=snap,
            detail=f"no walk to the door xy=({snap.link_x},{snap.link_y})",
            diag=snapshot_to_diag(snap),
        )
    snap = snapshot_env(env)
    ok = near_main_door(snap) or in_main_hall(snap)
    return RoutePhaseResult(
        phase="approach_main_door",
        ok=ok,
        frames=frames,
        snapshot=snap,
        detail=(
            f"door approach xy=({snap.link_x},{snap.link_y})"
            if ok
            else f"missed door approach xy=({snap.link_x},{snap.link_y})"
        ),
        diag=snapshot_to_diag(snap),
    )


def enter_main_door(env: BootEnv) -> RoutePhaseResult:
    """Trigger tier: align door approach from map and hold door direction."""
    door = main_door()
    ax, ay = door.approach_xy
    push_dir = door.direction
    push_limit = max(1, door.push_frames // 3)

    frames = 0
    settle = primitives.settle_control(env)
    frames += settle.frames
    start = settle.snapshot
    if in_main_hall(start):
        return RoutePhaseResult(
            phase="enter_main_door",
            ok=True,
            frames=frames,
            snapshot=start,
            detail="already in main hall",
            diag=snapshot_to_diag(start),
        )

    # Align x to map door approach (geometry authority).
    for _ in range(50):
        snap = snapshot_env(env)
        if abs(snap.link_x - ax) <= 3:
            break
        face = "RIGHT" if snap.link_x < ax else "LEFT"
        step_frames(env, action_for(face), 3)
        frames += 3

    for trial in range(8):
        for _ in range(push_limit):
            step_frames(env, action_for(push_dir), 3)
            frames += 3
            t_frames, snap = _settle_transition(env)
            frames += t_frames
            if in_main_hall(snap):
                sc = primitives.settle_control(env, max_frames=120)
                frames += sc.frames
                snap = sc.snapshot
                return RoutePhaseResult(
                    phase="enter_main_door",
                    ok=True,
                    frames=frames,
                    snapshot=snap,
                    detail=(
                        f"entered main hall room 0x61 via {door.label} "
                        f"xy=({snap.link_x},{snap.link_y})"
                    ),
                    diag={**snapshot_to_diag(snap), "door": door.label},
                )
            if _reentered_secret(snap):
                # Wrong door (stairs) — back out and nudge.
                step_frames(env, action_for("DOWN"), 50)
                frames += 50
                sc = primitives.settle_control(env, max_frames=60)
                frames += sc.frames
                break
            if snap.indoors:
                return RoutePhaseResult(
                    phase="enter_main_door",
                    ok=False,
                    frames=frames,
                    snapshot=snap,
                    detail=(
                        f"entered unexpected room {room_label(snap.room_base_id)}"
                    ),
                    diag=snapshot_to_diag(snap),
                )
        # Nudge x and retry (minimal special case — coords still from map).
        step_frames(
            env,
            action_for("LEFT" if trial % 2 == 0 else "RIGHT"),
            6,
        )
        frames += 6

    snap = snapshot_env(env)
    return RoutePhaseResult(
        phase="enter_main_door",
        ok=False,
        frames=frames,
        snapshot=snap,
        detail=(
            f"door trigger timeout ({door.label}) xy=({snap.link_x},{snap.link_y}) "
            f"indoors={snap.indoors}"
        ),
        diag=snapshot_to_diag(snap),
    )


POCKET_PHASES = (
    escape_hedge_pocket,
    approach_main_door,
    enter_main_door,
)

_POCKET_NOTES = (
    "Pocket escape lifts two bushes, then walks. No courtyard sword.",
    f"Geometry: maps/{MAP_ID}.json door {MAIN_DOOR_LABEL}.",
    "Door approach and push come from the map.",
    "Do not claim Zelda until follower_indicator==1.",
)


def run_from_pocket(
    env: BootEnv,
    *,
    source: str = "state_load_dev",
    phases: Sequence[PhaseFn] | None = None,
) -> SegmentResult:
    """Pocket (or open courtyard) → main hall.

    Segment ok means ``main_hall`` (room 0x61). Pass ``phases`` to run a
    subset (replaces former try_escape/try_approach/try_enter flags).
    """
    from alttp.opening_route.runner import run_phases

    return run_phases(
        env,
        list(phases) if phases is not None else list(POCKET_PHASES),
        evaluate_acceptance=evaluate_acceptance,
        success_when=in_main_hall,
        source=source,
        notes=_POCKET_NOTES,
        success_phase="main_hall_entered",
        success_notes=(
            "Main castle door entered (room 0x61). Next: B1 → Zelda cell.",
        ),
        partial_blocker="pocket phases finished without main hall",
        result_factory=_REPORT,
    )


def run_from_sword_through_pocket(
    env: BootEnv,
    *,
    source: str = "state_load_dev",
) -> SegmentResult:
    """Compose secret_entrance_clear (to pocket) then pocket_to_main_hall."""
    from alttp.opening_route.secret_entrance_clear import run_from_sword

    pre = run_from_sword(env, source=source)
    if not pre.ok or not pre.acceptance.get("left_secret_entrance"):
        return _REPORT(
            ok=False,
            phase=pre.phase,
            frames=pre.frames,
            snapshot=pre.snapshot,
            phases=list(pre.phases),
            source=source,
            acceptance={**evaluate_acceptance(pre.snapshot), **pre.acceptance},
            diagnostics=dict(pre.diagnostics),
            blocker=pre.blocker or "failed to reach courtyard pocket",
            notes=list(pre.notes)
            + ["pocket_to_main_hall requires secret-entrance clear first"],
        )
    hall = run_from_pocket(env, source=source)
    # Merge phases / frames.
    return _REPORT(
        ok=hall.ok,
        phase=hall.phase,
        frames=pre.frames + hall.frames,
        snapshot=hall.snapshot,
        phases=list(pre.phases) + list(hall.phases),
        source=source,
        acceptance=hall.acceptance,
        diagnostics={**pre.diagnostics, **hall.diagnostics},
        blocker=hall.blocker,
        notes=list(pre.notes) + list(hall.notes),
    )
