"""Courtyard secret pocket → main castle door → room 0x61.

Measured 2026-07-30 headless from continuous tip ``courtyard_secret_pocket``
(after ``sword_to_zelda`` stairs exit / FighterSword predecessor):

1. **Route** — bush-cut south-west out of the hedge pocket into the open
   courtyard (flower gardens ~x≤2216, y≥1900). Pure walking stays in a
   ~48×64 box; sword swings are required to clear bushes.
2. **Approach** — hard south (cut) to y≈2024, west to x≈2040, north to
   door approach ~(2040, 1790).
3. **Trigger** — align x≈2040, hold UP → indoors main hall room ``0x61``.

Door outdoor landing reverse-measured from ``CastleMain`` exit:
~(2040, 1779). UP re-enters stairs from the pocket (known trap).

Do not claim Zelda / Sanctuary from this hop alone.
"""

from __future__ import annotations

from typing import Any

from alttp import primitives
from alttp.opening_route.anchors import (
    COURTYARD_SECRET_POCKET_TOLERANCE,
    COURTYARD_SECRET_POCKET_X,
    COURTYARD_SECRET_POCKET_Y,
    MAIN_DOOR_APPROACH_TOLERANCE,
    MAIN_DOOR_APPROACH_X,
    MAIN_DOOR_APPROACH_Y,
)
from alttp.ram import (
    HYRULE_CASTLE_MAIN_HALL_ROOM,
    HYRULE_CASTLE_SCREEN,
    SECRET_PASSAGE_ROOM,
    AlttpSnapshot,
    room_label,
    snapshot_to_diag,
)
from alttp.route_report import RoutePhaseResult, SegmentResult
from alttp.startup import action_for, no_action, snapshot_env, step_frames

# Open-courtyard / gardens window after pocket escape (route tier).
COURTYARD_OPEN_Y_MIN = 1880
COURTYARD_OPEN_X_MAX = 2220

# Southern corridor that connects east hedge area to door axis (route).
COURTYARD_SOUTH_CORRIDOR_Y = 2024


def cut_push(
    env: object,
    face: str,
    *,
    walk: int = 12,
    swings: int = 2,
) -> int:
    """Face, swing sword (cut bushes), then walk. Returns frames used."""
    frames = 0
    step_frames(env, action_for(face), 2)
    frames += 2
    for _ in range(swings):
        step_frames(env, action_for("B"), 4)
        step_frames(env, no_action(), 4)
        frames += 8
    step_frames(env, action_for(face), walk)
    frames += walk
    return frames


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


def _settle_transition(env: object, *, max_frames: int = 120) -> tuple[int, AlttpSnapshot]:
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


def escape_hedge_pocket(env: object) -> RoutePhaseResult:
    """Route tier: bush-cut south/west out of secret-stairs hedge pocket."""
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

    # Off the stairs pad south.
    step_frames(env, action_for("DOWN"), 28)
    frames += 28

    # Cut south-west until open-courtyard window.
    for i in range(28):
        snap = snapshot_env(env)
        if _reentered_secret(snap):
            step_frames(env, action_for("DOWN"), 40)
            frames += 40
            sc = primitives.settle_control(env, max_frames=60)
            frames += sc.frames
            continue
        if in_open_courtyard(snap) or in_main_hall(snap):
            break
        frames += cut_push(env, "DOWN", walk=10, swings=3)
        if i % 2 == 1:
            frames += cut_push(env, "LEFT", walk=8, swings=2)

    for _ in range(12):
        snap = snapshot_env(env)
        if in_open_courtyard(snap) or in_main_hall(snap) or snap.link_x <= 2180:
            break
        frames += cut_push(env, "LEFT", walk=10, swings=2)

    # Light skirmish if soldiers nearby.
    fight = primitives.fight_nearby(
        env,
        max_distance=120,
        attack_distance=40,
        max_cycles=200,
        stop_when=lambda e: in_main_hall(snapshot_env(e)),
    )
    frames += fight.frames

    snap = snapshot_env(env)
    ok = in_open_courtyard(snap) or in_main_hall(snap)
    return RoutePhaseResult(
        phase="escape_hedge_pocket",
        ok=ok,
        frames=frames,
        snapshot=snap,
        detail=(
            f"open courtyard xy=({snap.link_x},{snap.link_y})"
            if ok
            else (
                f"still pocket/hedge xy=({snap.link_x},{snap.link_y}) "
                f"indoors={snap.indoors}"
            )
        ),
        diag=snapshot_to_diag(snap),
    )


def approach_main_door(env: object) -> RoutePhaseResult:
    """Approach tier: south corridor → west to door x → north to door y."""
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

    # Hard south (bush-cut) to connecting corridor y≈2024.
    for i in range(200):
        snap = snapshot_env(env)
        if in_main_hall(snap) or near_main_door(snap):
            break
        if _reentered_secret(snap):
            step_frames(env, action_for("DOWN"), 40)
            frames += 40
            continue
        if snap.link_y >= COURTYARD_SOUTH_CORRIDOR_Y:
            break
        before = (snap.link_x, snap.link_y)
        frames += cut_push(env, "DOWN", walk=12, swings=3)
        if (snapshot_env(env).link_x, snapshot_env(env).link_y) == before:
            for face in ("LEFT", "RIGHT"):
                frames += cut_push(env, face, walk=10, swings=2)
                frames += cut_push(env, "DOWN", walk=12, swings=3)
                if snapshot_env(env).link_y > before[1]:
                    break

    # West along corridor to door x.
    for _ in range(120):
        snap = snapshot_env(env)
        if in_main_hall(snap) or near_main_door(snap):
            break
        if snap.link_x <= MAIN_DOOR_APPROACH_X + 8:
            break
        before = (snap.link_x, snap.link_y)
        frames += cut_push(env, "LEFT", walk=14, swings=2)
        if (snapshot_env(env).link_x, snapshot_env(env).link_y) == before:
            for face in ("DOWN", "UP"):
                frames += cut_push(env, face, walk=10, swings=1)
                frames += cut_push(env, "LEFT", walk=12, swings=2)
                if snapshot_env(env).link_x < before[0]:
                    break

    # North to door approach y.
    for _ in range(160):
        snap = snapshot_env(env)
        if in_main_hall(snap) or near_main_door(snap):
            break
        if _reentered_secret(snap):
            step_frames(env, action_for("DOWN"), 40)
            frames += 40
            continue
        dx = MAIN_DOOR_APPROACH_X - snap.link_x
        dy = MAIN_DOOR_APPROACH_Y - snap.link_y
        if abs(dx) <= 16 and abs(dy) <= 16:
            break
        before = (snap.link_x, snap.link_y)
        if abs(dx) > 6:
            face = "RIGHT" if dx > 0 else "LEFT"
            frames += cut_push(env, face, walk=10, swings=1)
        if abs(dy) > 6:
            face = "DOWN" if dy > 0 else "UP"
            frames += cut_push(env, face, walk=10, swings=1)
        if (snapshot_env(env).link_x, snapshot_env(env).link_y) == before:
            for face in ("UP", "LEFT", "RIGHT", "DOWN"):
                frames += cut_push(env, face, walk=10, swings=1)
                if (snapshot_env(env).link_x, snapshot_env(env).link_y) != before:
                    break

    # Clear nearby soldiers before trigger.
    fight = primitives.fight_nearby(
        env,
        max_distance=100,
        max_cycles=150,
        stop_when=lambda e: in_main_hall(snapshot_env(e)),
    )
    frames += fight.frames

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


def enter_main_door(env: object) -> RoutePhaseResult:
    """Trigger tier: align door x and walk UP into room 0x61."""
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

    # Align x to door.
    for _ in range(50):
        snap = snapshot_env(env)
        if abs(snap.link_x - MAIN_DOOR_APPROACH_X) <= 3:
            break
        face = "RIGHT" if snap.link_x < MAIN_DOOR_APPROACH_X else "LEFT"
        step_frames(env, action_for(face), 3)
        frames += 3

    for trial in range(8):
        for _ in range(80):
            step_frames(env, action_for("UP"), 3)
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
                        f"entered main hall room 0x61 "
                        f"xy=({snap.link_x},{snap.link_y})"
                    ),
                    diag=snapshot_to_diag(snap),
                )
            if _reentered_secret(snap):
                # Wrong door (stairs) — back out and nudge.
                step_frames(env, action_for("DOWN"), 50)
                frames += 50
                sc = primitives.settle_control(env, max_frames=60)
                frames += sc.frames
                break
            if snap.indoors:
                # Unexpected indoor — report.
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
        # Nudge x and retry.
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
            f"door trigger timeout xy=({snap.link_x},{snap.link_y}) "
            f"indoors={snap.indoors}"
        ),
        diag=snapshot_to_diag(snap),
    )


def run_from_pocket(
    env: object,
    *,
    source: str = "state_load_dev",
    try_escape: bool = True,
    try_approach: bool = True,
    try_enter: bool = True,
) -> SegmentResult:
    """Pocket (or open courtyard) → main hall.

    Segment ok means ``main_hall`` (room 0x61). Partial progress is reported
    via phases / acceptance without claiming Zelda.
    """
    phases: list[RoutePhaseResult] = []
    total = 0
    notes = [
        "Pocket escape requires bush-cutting (walk-only stays boxed).",
        "South corridor y≈2024 connects east hedges to door axis x≈2040.",
        "Door trigger: align x≈2040, UP → room 0x61.",
        "Do not claim Zelda until follower_indicator==1.",
    ]

    if try_escape:
        esc = escape_hedge_pocket(env)
        phases.append(esc)
        total += esc.frames
        if not esc.ok and not in_main_hall(esc.snapshot):
            acc = evaluate_acceptance(esc.snapshot)
            return SegmentResult(
                ok=False,
                phase=esc.phase,
                frames=total,
                snapshot=esc.snapshot,
                phases=phases,
                source=source,
                acceptance=acc,
                blocker=esc.detail,
                notes=notes,
            )
        if in_main_hall(esc.snapshot):
            acc = evaluate_acceptance(esc.snapshot)
            return SegmentResult(
                ok=True,
                phase="main_hall_entered",
                frames=total,
                snapshot=esc.snapshot,
                phases=phases,
                source=source,
                acceptance=acc,
                blocker="",
                notes=notes,
            )

    if try_approach:
        app = approach_main_door(env)
        phases.append(app)
        total += app.frames
        if not app.ok and not in_main_hall(app.snapshot):
            acc = evaluate_acceptance(app.snapshot)
            return SegmentResult(
                ok=False,
                phase=app.phase,
                frames=total,
                snapshot=app.snapshot,
                phases=phases,
                source=source,
                acceptance=acc,
                blocker=app.detail,
                notes=notes,
            )
        if in_main_hall(app.snapshot):
            acc = evaluate_acceptance(app.snapshot)
            return SegmentResult(
                ok=True,
                phase="main_hall_entered",
                frames=total,
                snapshot=app.snapshot,
                phases=phases,
                source=source,
                acceptance=acc,
                blocker="",
                notes=notes,
            )

    if try_enter:
        ent = enter_main_door(env)
        phases.append(ent)
        total += ent.frames
        snap = ent.snapshot
        acc = evaluate_acceptance(snap)
        if ent.ok and acc["main_hall"]:
            return SegmentResult(
                ok=True,
                phase="main_hall_entered",
                frames=total,
                snapshot=snap,
                phases=phases,
                source=source,
                acceptance=acc,
                blocker="",
                notes=notes
                + [
                    "Main castle door entered (room 0x61). Next: B1 → Zelda cell."
                ],
            )
        return SegmentResult(
            ok=False,
            phase=ent.phase,
            frames=total,
            snapshot=snap,
            phases=phases,
            source=source,
            acceptance=acc,
            blocker=ent.detail,
            notes=notes,
        )

    snap = snapshot_env(env)
    acc = evaluate_acceptance(snap)
    return SegmentResult(
        ok=acc["main_hall"],
        phase="partial",
        frames=total,
        snapshot=snap,
        phases=phases,
        source=source,
        acceptance=acc,
        blocker="" if acc["main_hall"] else "enter skipped",
        notes=notes,
    )


def run_from_sword_through_pocket(
    env: object,
    *,
    source: str = "state_load_dev",
) -> SegmentResult:
    """Compose sword_to_zelda (to pocket) then pocket_to_main_hall."""
    from alttp.opening_route.secret_entrance_clear import run_from_sword

    pre = run_from_sword(env, source=source)
    if not pre.ok or not pre.acceptance.get("left_secret_entrance"):
        return SegmentResult(
            ok=False,
            phase=pre.phase,
            frames=pre.frames,
            snapshot=pre.snapshot,
            phases=list(pre.phases),
            source=source,
            acceptance={**evaluate_acceptance(pre.snapshot), **pre.acceptance},
            blocker=pre.blocker or "failed to reach courtyard pocket",
            notes=list(pre.notes)
            + ["pocket_to_main_hall requires secret-entrance clear first"],
        )
    hall = run_from_pocket(env, source=source)
    # Merge phases / frames.
    return SegmentResult(
        ok=hall.ok,
        phase=hall.phase,
        frames=pre.frames + hall.frames,
        snapshot=hall.snapshot,
        phases=list(pre.phases) + list(hall.phases),
        source=source,
        acceptance=hall.acceptance,
        blocker=hall.blocker,
        notes=list(pre.notes) + list(hall.notes),
    )
