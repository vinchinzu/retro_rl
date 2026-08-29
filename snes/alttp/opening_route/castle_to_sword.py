"""Castle grounds → secret entrance approach → uncle / fighter sword.

Composes with the verified title→castle-grounds predecessor. Does **not**
use progression writes or door warps. State loads are development-only.

Authority:
- stable-retro RAM (``alttp.ram``) is gameplay truth.
- Outdoor approach geometry: ``maps/screen_1b_grounds.json`` door
  ``secret_hole_to_0x55``. Do not redeclare world coords or the former
  open-loop D-pad approach macro in Python.
- Bush-lift / hole-drop remains a measured trigger (candidates below).

Measured 2026-07-29 headless probes (HyruleCastleGrounds predecessor):
- Map path reaches the secret-hole approach (~world 2430,1704 on
  screen 0x1B), near Yaze entrance 0x7D (2432,1696).
- Main south gate is soldier-blocked (text mode 0x0E) until sword.
- Proven bush-lift / hole-drop: face up, short A, wait, walk up into the
  revealed hole → room 0x55; uncle dialogue then yields fighter sword.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import lru_cache

from alttp import primitives
from alttp.opening_route.runner import PhaseFn
from alttp.ram import (
    SECRET_PASSAGE_ROOM,
    AlttpSnapshot,
    castle_entry_accepted,
    secret_passage_accepted,
    snapshot_to_diag,
    uncle_sword_event_accepted,
)
from alttp.room_map import KnownDoor, RoomMap, load_room_map
from alttp.route_report import RoutePhaseResult, SegmentResult, segment_result_factory
from alttp.startup import (
    BootEnv,
    action_for,
    boot_past_title_to_castle,
    build_boot_env,
    no_action,
    snapshot_env,
    step_frames,
)

_REPORT = segment_result_factory("alttp_castle_to_sword_report")

MAP_ID = "screen_1b_grounds"
HOLE_DOOR_LABEL = "secret_hole_to_0x55"


@lru_cache(maxsize=1)
def grounds_map() -> RoomMap:
    return load_room_map(MAP_ID)


@lru_cache(maxsize=1)
def hole_door() -> KnownDoor:
    door = grounds_map().door(HOLE_DOOR_LABEL)
    if door is None:
        raise KeyError(f"door {HOLE_DOOR_LABEL!r} missing from {MAP_ID}")
    return door


def hole_approach_waypoints() -> list[primitives.Waypoint]:
    """Map-backed axis-aligned path from grounds spawn toward the hole."""
    return [
        primitives.Waypoint(x, y, tolerance=tol, label=label)
        for x, y, label, tol in grounds_map().waypoints_for_door(hole_door())
    ]


# Proven bush-lift / hole-drop from the hole-approach position (2026-07-29).
# Face up, lift the bush with A, wait for the lift anim, walk north into the
# hole. Min measured UP walk after A/wait is 40 frames; 56 is a safe margin.
SECRET_HOLE_ENTRY_SCRIPT: tuple[tuple[tuple[str, ...], int], ...] = (
    (("UP",), 2),
    (("A",), 4),
    (("NONE",), 20),
    (("UP",), 56),
)

# Fallback candidates if the proven macro misses (position drift / natural chain).
BUSH_LIFT_CANDIDATES: tuple[tuple[tuple[tuple[str, ...], int], ...], ...] = (
    SECRET_HOLE_ENTRY_SCRIPT,
    ((("UP",), 2), (("A",), 4), (("NONE",), 20), (("UP",), 40)),
    ((("UP",), 2), (("A",), 4), (("NONE",), 20), (("UP",), 80)),
    ((("UP",), 2), (("A",), 8), (("NONE",), 30), (("UP",), 56)),
    ((("A",), 4), (("NONE",), 20), (("UP",), 56)),
    ((("A",), 8), (("NONE",), 40), (("UP",), 60)),
    ((("UP",), 10), (("A",), 4), (("NONE",), 20), (("UP",), 56)),
    ((("LEFT",), 8), (("UP",), 2), (("A",), 4), (("NONE",), 20), (("UP",), 56)),
    ((("RIGHT",), 8), (("UP",), 2), (("A",), 4), (("NONE",), 20), (("UP",), 56)),
    ((("UP", "A"), 8), (("NONE",), 30), (("UP",), 56)),
)

UNCLE_APPROACH_SCRIPT: tuple[tuple[tuple[str, ...], int], ...] = (
    (("UP", "LEFT"), 40),
    (("DOWN", "LEFT"), 40),
    (("LEFT",), 40),
    (("LEFT",), 16),
    (("NONE",), 30),
)


def _at_hole_approach(snap: AlttpSnapshot) -> bool:
    """Tight map approach (not the 48px RAM near_secret_hole window)."""
    if snap.indoors or snap.dark_world:
        return False
    ax, ay = hole_door().approach_xy
    tol = hole_door().tolerance_for("secret_hole_approach", 12)
    return abs(snap.link_x - ax) <= tol and abs(snap.link_y - ay) <= tol


def _arrived(snap: AlttpSnapshot) -> bool:
    return snap.in_secret_passage or snap.indoors or _at_hole_approach(snap)


def _walk_axis_aligned(
    env: BootEnv,
    waypoint: primitives.Waypoint,
    *,
    stop_when: Callable[[AlttpSnapshot], bool] | None = None,
    max_frames: int = 900,
) -> primitives.PrimitiveResult:
    """Cardinal-only walk to a map waypoint.

    Greedy diagonal ``move_to`` clips the south/west hedges on this screen.
    """
    frames = 0
    unchanged = 0
    previous_xy: tuple[int, int] | None = None
    while frames < max_frames:
        snap = snapshot_env(env)
        if stop_when is not None and stop_when(snap):
            return primitives.PrimitiveResult(
                True, waypoint.label or "stopped", frames, snap
            )
        if waypoint.reached(snap.link_x, snap.link_y):
            return primitives.PrimitiveResult(
                True, waypoint.label or "waypoint reached", frames, snap
            )
        dx = waypoint.x - snap.link_x
        dy = waypoint.y - snap.link_y
        if abs(dx) >= abs(dy) and abs(dx) > waypoint.tolerance:
            face = "RIGHT" if dx > 0 else "LEFT"
        else:
            face = "DOWN" if dy > 0 else "UP"
        step_frames(env, action_for(face), 2)
        frames += 2
        xy = (snapshot_env(env).link_x, snapshot_env(env).link_y)
        if xy == previous_xy:
            unchanged += 1
        else:
            unchanged = 0
            previous_xy = xy
        if unchanged < 40:
            continue
        # Nudge the other axis once; hedges often block the longer remainder.
        snap = snapshot_env(env)
        dx = waypoint.x - snap.link_x
        dy = waypoint.y - snap.link_y
        if abs(dx) > waypoint.tolerance:
            step_frames(env, action_for("RIGHT" if dx > 0 else "LEFT"), 8)
            frames += 8
        if abs(dy) > waypoint.tolerance:
            step_frames(env, action_for("DOWN" if dy > 0 else "UP"), 8)
            frames += 8
        if (snapshot_env(env).link_x, snapshot_env(env).link_y) == xy:
            return primitives.PrimitiveResult(
                False,
                f"stuck before {waypoint.label or 'waypoint'}",
                frames,
                snapshot_env(env),
            )
        unchanged = 0
    return primitives.PrimitiveResult(
        False,
        f"timeout before {waypoint.label or 'waypoint'}",
        frames,
        snapshot_env(env),
    )


def approach_secret_hole(env: BootEnv) -> RoutePhaseResult:
    """Walk map path from castle-grounds spawn toward the secret-hole approach."""
    frames = 0
    settle = primitives.settle_control(env)
    frames += settle.frames
    start = settle.snapshot
    if not start.on_castle_grounds and not start.near_secret_hole:
        return RoutePhaseResult(
            phase="approach_secret_hole",
            ok=False,
            frames=frames,
            snapshot=start,
            detail="predecessor is not castle-grounds controllable",
            diag=snapshot_to_diag(start),
        )
    if _arrived(start):
        return RoutePhaseResult(
            phase="approach_secret_hole",
            ok=True,
            frames=frames,
            snapshot=start,
            detail="already at hole approach or indoors",
            diag=snapshot_to_diag(start),
        )

    wps = hole_approach_waypoints()
    if not wps:
        return RoutePhaseResult(
            phase="approach_secret_hole",
            ok=False,
            frames=frames,
            snapshot=start,
            detail=f"no hole path in {MAP_ID}",
            diag=snapshot_to_diag(start),
        )

    last = primitives.PrimitiveResult(True, "path", 0, start)
    for wp in wps:
        snap = snapshot_env(env)
        if _arrived(snap) or wp.reached(snap.link_x, snap.link_y):
            continue
        last = _walk_axis_aligned(env, wp, stop_when=_arrived)
        frames += last.frames
        if _arrived(last.snapshot):
            break
        if not last.ok:
            break

    settle = primitives.settle_control(env)
    frames += settle.frames
    snap = settle.snapshot
    ok = _arrived(snap) or snap.near_secret_hole
    ax, ay = hole_door().approach_xy
    return RoutePhaseResult(
        phase="approach_secret_hole",
        ok=ok,
        frames=frames,
        snapshot=snap,
        detail=(
            "near Yaze 0x7D hole approach"
            if snap.near_secret_hole
            else (
                "entered indoors during approach"
                if snap.indoors
                else (
                    last.reason
                    if not last.ok
                    else f"finished approach off hole tolerance xy=({snap.link_x},{snap.link_y}) target~{ax},{ay}"
                )
            )
        ),
        diag={**snapshot_to_diag(snap), "mapId": MAP_ID, "door": HOLE_DOOR_LABEL},
    )


def attempt_secret_entrance_entry(
    env: BootEnv,
    *,
    candidates: Sequence[Sequence[tuple[tuple[str, ...], int]]] | None = None,
) -> RoutePhaseResult:
    """Try bush-lift / hole-drop macros from the current approach position.

    Tries the proven ``SECRET_HOLE_ENTRY_SCRIPT`` first, then fallbacks.
    Saves emulator state once and restores between candidates (search only;
    not a progression write).
    """
    frames = 0
    settle = primitives.settle_control(env)
    frames += settle.frames
    snap = settle.snapshot
    if snap.in_secret_passage or (
        snap.indoors and secret_passage_accepted(snap)
    ):
        return RoutePhaseResult(
            phase="secret_entrance_entry",
            ok=True,
            frames=frames,
            snapshot=snap,
            detail="already in secret passage",
            diag=snapshot_to_diag(snap),
        )

    macros: Sequence[Sequence[tuple[tuple[str, ...], int]]]
    if candidates is not None:
        macros = candidates
    else:
        macros = BUSH_LIFT_CANDIDATES

    if not hasattr(env, "em"):
        # Single-shot proven macro without restore search.
        script = primitives.run_script(
            env,
            SECRET_HOLE_ENTRY_SCRIPT,
            stop_when=lambda s: s.indoors,
        )
        frames += script.frames
        for _ in range(120):
            step_frames(env, no_action(), 1)
            frames += 1
            snap = snapshot_env(env)
            if snap.indoors:
                break
        settle = primitives.settle_control(env, max_frames=300)
        frames += settle.frames
        snap = settle.snapshot
        ok = snap.in_secret_passage or castle_entry_accepted(snap)
        return RoutePhaseResult(
            phase="secret_entrance_entry",
            ok=ok,
            frames=frames,
            snapshot=snap,
            detail=(
                "entered via proven hole script (no em restore)"
                if ok
                else "proven hole script missed without em restore"
            ),
            diag=snapshot_to_diag(snap),
        )

    approach_state = env.em.get_state()  # type: ignore[attr-defined]
    tried = 0
    for macro in macros:
        tried += 1
        env.em.set_state(approach_state)  # type: ignore[attr-defined]
        script = primitives.run_script(
            env,
            macro,
            stop_when=lambda s: s.indoors,
        )
        frames += script.frames
        # Wait for possible fall / room load.
        for _ in range(120):
            step_frames(env, no_action(), 1)
            frames += 1
            snap = snapshot_env(env)
            if snap.indoors:
                break
        snap = snapshot_env(env)
        if snap.in_secret_passage or castle_entry_accepted(snap):
            settle = primitives.settle_control(env, max_frames=300)
            frames += settle.frames
            snap = settle.snapshot
            return RoutePhaseResult(
                phase="secret_entrance_entry",
                ok=True,
                frames=frames,
                snapshot=snap,
                detail=(
                    "entered via proven SECRET_HOLE_ENTRY_SCRIPT"
                    if tried == 1 and candidates is None
                    else f"entered via candidate #{tried}"
                ),
                diag=snapshot_to_diag(snap),
            )

    env.em.set_state(approach_state)  # type: ignore[attr-defined]
    settle = primitives.settle_control(env)
    frames += settle.frames
    snap = settle.snapshot
    return RoutePhaseResult(
        phase="secret_entrance_entry",
        ok=False,
        frames=frames,
        snapshot=snap,
        detail=(
            f"no bush-lift/hole-drop among {tried} candidates; "
            f"still outdoors near_hole={snap.near_secret_hole} "
            f"xy=({snap.link_x},{snap.link_y})"
        ),
        diag=snapshot_to_diag(snap),
    )


def dismiss_hold_up_item(env: BootEnv, *, max_frames: int = 160) -> int:
    """Clear kPlayerState_HoldUpItem ($5D==21) after sword / chest gets.

    Hold left until the pose ends (~95 frames measured after fighter sword).
    """
    frames = 0
    while frames < max_frames:
        snap = snapshot_env(env)
        if not snap.is_hold_up_item:
            step_frames(env, no_action(), 8)
            return frames + 8
        step_frames(env, action_for("LEFT"), 1)
        frames += 1
    return frames


def advance_uncle_dialogue_for_sword(
    env: BootEnv,
    *,
    max_cycles: int = 400,
) -> RoutePhaseResult:
    """From secret passage, approach uncle and mash dialogue until sword."""
    frames = 0
    settle = primitives.settle_control(env)
    frames += settle.frames
    snap = settle.snapshot
    if snap.has_fighter_sword:
        frames += dismiss_hold_up_item(env)
        snap = snapshot_env(env)
        return RoutePhaseResult(
            phase="uncle_sword",
            ok=True,
            frames=frames,
            snapshot=snap,
            detail="sword already equipped",
            diag=snapshot_to_diag(snap),
        )

    script = primitives.run_script(env, UNCLE_APPROACH_SCRIPT)
    frames += script.frames

    for cycle in range(max_cycles):
        snap = snapshot_env(env)
        if snap.has_fighter_sword:
            frames += dismiss_hold_up_item(env)
            snap = snapshot_env(env)
            return RoutePhaseResult(
                phase="uncle_sword",
                ok=True,
                frames=frames,
                snapshot=snap,
                detail=f"sword after dialogue cycle {cycle}",
                diag=snapshot_to_diag(snap),
            )
        if snap.is_text_mode or not snap.has_control:
            btn = "A" if cycle % 4 < 2 else "B"
            step_frames(env, action_for(btn), 2)
            frames += 2
        else:
            # Explore a bit while occasionally pressing A.
            dirs: tuple[tuple[str, ...], ...] = (
                ("LEFT",),
                ("UP", "LEFT"),
                ("UP",),
                ("DOWN", "LEFT"),
                ("RIGHT",),
            )
            d = dirs[cycle % len(dirs)]
            if cycle % 3 == 0:
                step_frames(env, action_for("A"), 2)
            else:
                step_frames(env, action_for(*d), 4)
            frames += 4 if cycle % 3 else 2

    snap = snapshot_env(env)
    if snap.has_fighter_sword:
        frames += dismiss_hold_up_item(env)
        snap = snapshot_env(env)
    return RoutePhaseResult(
        phase="uncle_sword",
        ok=snap.has_fighter_sword,
        frames=frames,
        snapshot=snap,
        detail="sword not obtained within budget",
        diag=snapshot_to_diag(snap),
    )


def evaluate_acceptance(snapshot: AlttpSnapshot) -> dict[str, bool]:
    """Segment contract + on-path progress keys (no later-route diagnostics)."""
    return {
        "on_castle_grounds": snapshot.on_castle_grounds,
        "near_secret_hole": snapshot.near_secret_hole,
        "castle_entry": castle_entry_accepted(snapshot),
        "secret_passage": secret_passage_accepted(snapshot),
        "uncle_sword": uncle_sword_event_accepted(snapshot),
        "fighter_sword_ram": snapshot.has_fighter_sword,
    }


def run_from_castle_grounds(
    env: BootEnv,
    *,
    source: str = "state_load_dev",
    phases: Sequence[PhaseFn] | None = None,
    include_entry: bool = True,
    include_uncle: bool = True,
) -> SegmentResult:
    """Run the segment assuming env is already on castle grounds.

    Default order: approach → (entry if outdoors) → (uncle if indoors).
    Prefer ``phases=`` for a fixed list; ``include_entry`` / ``include_uncle``
    remain for scripts that only want approach (replaces try_entry/try_uncle).
    """
    phase_rows: list[RoutePhaseResult] = []
    total = 0
    notes: list[str] = []

    if phases is not None:
        from alttp.opening_route.runner import run_phases

        return run_phases(
            env,
            list(phases),
            evaluate_acceptance=evaluate_acceptance,
            success_when=lambda s: bool(s.has_fighter_sword),
            source=source,
            notes=notes,
            success_phase="fighter_sword",
            partial_blocker="castle-to-sword phases finished without fighter sword",
            result_factory=_REPORT,
        )

    approach = approach_secret_hole(env)
    phase_rows.append(approach)
    total += approach.frames
    if not approach.ok and not approach.snapshot.indoors:
        acc = evaluate_acceptance(approach.snapshot)
        return _REPORT(
            ok=False,
            phase=approach.phase,
            frames=total,
            snapshot=approach.snapshot,
            phases=phase_rows,
            source=source,
            acceptance=acc,
            blocker=(
                "failed to reach secret-hole approach from castle grounds "
                f"(xy={approach.snapshot.link_x},{approach.snapshot.link_y})"
            ),
            notes=notes,
        )

    if approach.snapshot.in_secret_passage:
        notes.append("entered secret passage during approach walk")

    if include_entry and not approach.snapshot.in_secret_passage:
        entry = attempt_secret_entrance_entry(env)
        phase_rows.append(entry)
        total += entry.frames

    snap = snapshot_env(env)
    if include_uncle and (snap.in_secret_passage or snap.indoors):
        uncle = advance_uncle_dialogue_for_sword(env)
        phase_rows.append(uncle)
        total += uncle.frames
        snap = uncle.snapshot
    elif include_uncle:
        notes.append("skipped uncle phase: not indoors")

    snap = snapshot_env(env)
    acc = evaluate_acceptance(snap)
    ok = bool(acc["fighter_sword_ram"])
    if ok:
        phase = "fighter_sword"
        blocker = ""
    elif acc["secret_passage"] or acc["castle_entry"]:
        phase = "castle_interior"
        blocker = "indoors but fighter sword not yet collected"
    elif approach.ok:
        phase = "secret_hole_approach"
        blocker = (
            "at secret-hole approach but bush-lift/hole-drop into room "
            f"0x{SECRET_PASSAGE_ROOM:02X} did not complete "
            f"(target world ~{hole_door().approach_xy[0]},{hole_door().approach_xy[1]})"
        )
    else:
        phase = approach.phase
        blocker = approach.detail

    return _REPORT(
        ok=ok,
        phase=phase,
        frames=total,
        snapshot=snap,
        phases=phase_rows,
        source=source,
        acceptance=acc,
        blocker=blocker,
        notes=notes,
    )


def run_natural_chain(
    env: BootEnv | None = None,
    *,
    close: bool = True,
) -> SegmentResult:
    """Title → castle grounds → secret approach → entry/sword attempts."""
    owns = env is None
    if env is None:
        env = build_boot_env()
    try:
        boot = boot_past_title_to_castle(env, close=False)
        if not boot.snapshot.on_castle_grounds:
            acc = evaluate_acceptance(boot.snapshot)
            return _REPORT(
                ok=False,
                phase="boot_to_castle",
                frames=boot.frames,
                snapshot=boot.snapshot,
                phases=[
                    RoutePhaseResult(
                        phase="boot_to_castle",
                        ok=False,
                        frames=boot.frames,
                        snapshot=boot.snapshot,
                        detail="boot_past_title_to_castle missed castle grounds",
                        diag=snapshot_to_diag(boot.snapshot),
                    )
                ],
                source="natural_boot",
                acceptance=acc,
                blocker="natural boot did not reach castle grounds",
            )
        result = run_from_castle_grounds(env, source="natural_boot")
        # Include boot frames in total for the natural chain report.
        result.frames += boot.frames
        result.phases.insert(
            0,
            RoutePhaseResult(
                phase="boot_to_castle",
                ok=True,
                frames=boot.frames,
                snapshot=boot.snapshot,
                detail="verified castle-grounds predecessor",
                diag=snapshot_to_diag(boot.snapshot),
            ),
        )
        return result
    finally:
        if owns and close:
            env.close()


def run_from_state(
    state_name: str = "HyruleCastleGrounds",
    *,
    close: bool = True,
) -> SegmentResult:
    """Development diagnostic from a saved castle-grounds state."""
    from alttp.opening_route.runner import run_from_state as _run_from_state

    return _run_from_state(
        state_name,
        run_from_castle_grounds,
        close=close,
        settle=True,
    )
