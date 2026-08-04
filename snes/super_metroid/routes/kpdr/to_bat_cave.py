"""Bubble Mountain climb: mid-left entry → top-right Super door → Bat Cave.

Phase ladder (docs/tasks/SM-K4.4-PHASE-LADDER.md / HARD_ROOM_SPLITS):
  A lower-left multi-hop → mid pin · A.5 mid re-pin · B+C+D mid loop
  (lip launch + right-structure climb) · E Super door.

Product path is :func:`play_bubble_to_bat_cave` (session only). Dev handoffs
and Phase-C capture live on separate helpers — never on the product API.

Layering (bottom → top)::

  skills.policies.bubble_to_bat
        ↑
  skills.geometry / walljump / runway / door
        ↑
  _to_bat_cave_mid (mid loop)
        ↑
  this product hop module
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import (
    hold,
    require_room,
    select_weapon,
    unmorph,
)
from super_metroid.routes.skills.door import top_super_door
from super_metroid.routes.skills.geometry import (
    BubblePhaseStop,
    BubbleTrack,
    ClimbTrack,
    avoid_wrong_door,
    is_stand_pin_pose,
    is_true_ground,
    new_climb_track,
    on_launch_lip,
    on_mid_iso_pin,
    on_right_shelf,
    on_save_runway,
    phase_c_usable_right_contact,
    phase_d_near_top,
    phase_d_top_band,
    track_state,
)
from super_metroid.routes.skills.policies import bubble_to_bat as P
from super_metroid.routes.runtime import ControllerSession

# Phase ladder public aliases (k4 / tests).
BUBBLE_PHASE_C_X_MIN = P.BUBBLE_PHASE_C_X_MIN
BUBBLE_PHASE_C_Y_MAX = P.BUBBLE_PHASE_C_Y_MAX
BUBBLE_PHASE_C_Y_MIN = P.BUBBLE_PHASE_C_Y_MIN
BUBBLE_PHASE_D_X = P.BUBBLE_PHASE_D_X
BUBBLE_PHASE_D_Y = P.BUBBLE_PHASE_D_Y

ROOM_BAT_CAVE = P.EXIT_ROOM_ID
ROOM_BUBBLE = P.ROOM_ID

MidStart = Literal["launch", "climb"]


def _new_track(
    session: ControllerSession,
    *,
    dump_phase_c: Path | str | None = None,
    stop_at_phase_c: bool = False,
) -> ClimbTrack:
    """Product-local alias for :func:`new_climb_track`."""
    return new_climb_track(
        session,
        label="bubble_to_bat_cave",
        dump_phase_c=dump_phase_c,
        stop_at_phase_c=stop_at_phase_c,
    )


# ---------------------------------------------------------------------------
# Thin hop-local wrappers (historical bubble_* names for tests / probes)
# ---------------------------------------------------------------------------


def bubble_is_true_ground(st: SuperMetroidState, *, max_vy: int = 1) -> bool:
    return is_true_ground(st, poses=P.TRUE_GROUND, max_vy=max_vy)


def bubble_is_stand_pin_pose(st: SuperMetroidState, *, max_vy: int = 1) -> bool:
    return is_stand_pin_pose(st, poses=P.STAND_PIN, max_vy=max_vy)


def bubble_on_mid_iso_pin(st: SuperMetroidState) -> bool:
    return on_mid_iso_pin(st, P)


def bubble_on_launch_lip(st: SuperMetroidState) -> bool:
    return on_launch_lip(st, P)


def bubble_on_right_shelf(st: SuperMetroidState) -> bool:
    return on_right_shelf(st, P)


def bubble_on_save_runway(st: SuperMetroidState) -> bool:
    return on_save_runway(st, P)


def bubble_phase_c_usable_right_contact(st: SuperMetroidState) -> bool:
    return phase_c_usable_right_contact(st, P)


def bubble_phase_d_top_band(st: SuperMetroidState) -> bool:
    return phase_d_top_band(st, P)


def bubble_phase_d_near_top(st: SuperMetroidState, slack: int = 40) -> bool:
    return phase_d_near_top(st, P, slack=slack)


def bubble_track_state(
    session: ControllerSession, track: ClimbTrack, state: SuperMetroidState
) -> None:
    track_state(session, track, state, P)


def bubble_avoid_wrong_door(
    session: ControllerSession, track: ClimbTrack, state: SuperMetroidState
) -> bool:
    return avoid_wrong_door(session, track, state, P)


# ---------------------------------------------------------------------------
# Phase helpers (lower / repin / door). Mid is one call in _to_bat_cave_mid.
# ---------------------------------------------------------------------------


def bubble_land_and_prepare(
    session: ControllerSession,
    track: ClimbTrack,
    *,
    land_frames: int = 40,
) -> None:
    """Land settle + unmorph + beam select. Shorter land for climb/door handoff."""
    label = track.label
    require_room(session, ROOM_BUBBLE, label)

    # Land: if already mid-iso pin class (isolation source), break immediately
    # with |vy|≤2 — a full 40f idle settle from vy≈1 drifts left/down off the
    # save-door platform (x~69 y~427) and forces a failed lower re-climb.
    for _ in range(land_frames):
        state = hold(session, 1, reason=f"{label}_land")
        if bubble_on_mid_iso_pin(state):
            break
        if state.velocity_y == 0 and state.pose in P.STANDING_POSES:
            break
    unmorph(session)
    select_weapon(session, 0)
    track.max_x = max(track.max_x, session.state.samus_x)
    track.min_y = min(track.min_y, session.state.samus_y)


def _bubble_fire_or_mid_pin(state: SuperMetroidState) -> bool:
    """Lower success pin: save fire solid (R16), not mid-iso float.

    Mid-iso float ~(100,365) falls onto the solid lip and steals the fire
    path. Only stop lower when grounded on the save runway (not lip).
    """
    return bubble_on_save_runway(state) and not bubble_on_launch_lip(state)


def bubble_lower_to_mid_pin(session: ControllerSession, track: ClimbTrack) -> None:
    """Phase A (R5): lower-left ledge multi-hop → save fire seat / mid pin."""
    label = track.label

    if _bubble_fire_or_mid_pin(session.state):
        track.mid_reached = True
        bubble_track_state(session, track, session.state)
        return

    # 1a: walk onto lower-left floor shelf (solid place-grid band ~y651).
    for frame in range(140):
        state = session.state
        if state.room_id != ROOM_BUBBLE:
            break
        bubble_track_state(session, track, state)
        if _bubble_fire_or_mid_pin(state):
            track.mid_reached = True
            break
        if bubble_avoid_wrong_door(session, track, state):
            continue
        if state.pose in (137, 138):
            hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_floor_kb")
            continue
        if state.samus_x >= P.FLOOR_SHELF_X and bubble_is_true_ground(state):
            break
        if frame % 12 < 3:
            hold(session, 1, "RIGHT", "B", "X", reason=f"{label}_floor_shot")
        else:
            hold(session, 1, "RIGHT", "B", reason=f"{label}_floor_walk")
    for _ in range(20):
        state = hold(session, 1, reason=f"{label}_floor_settle")
        bubble_track_state(session, track, state)
        if bubble_is_true_ground(state):
            break
        if state.pose in (137, 138):
            hold(session, 1, "A", reason=f"{label}_floor_kb_clear")

    # 1b: multi-hop along left-column shelves to save fire seat (R16).
    shelf_i = 0
    for frame in range(P.LOWER_FRAMES):
        state = session.state
        if state.room_id != ROOM_BUBBLE:
            break
        bubble_track_state(session, track, state)
        if _bubble_fire_or_mid_pin(state):
            track.mid_reached = True
            break
        if bubble_avoid_wrong_door(session, track, state):
            continue
        if state.pose in (137, 138):
            for _ in range(8):
                hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_lower_kb")
            continue
        if state.pose in (27, 28):
            hold(session, 1, "UP", reason=f"{label}_lower_unmorph")
            continue

        x = state.samus_x
        y = state.samus_y
        if x > 250 and y > P.MID_Y:
            hold(session, 1, "LEFT", "B", reason=f"{label}_lower_cavity")
            continue

        shelves = P.LOWER_SHELVES
        while (
            shelf_i < len(shelves) - 1
            and y <= shelves[shelf_i][1] + 12
            and abs(x - shelves[shelf_i][0]) < 40
        ):
            shelf_i += 1
        tx, ty = shelves[shelf_i]

        grounded = bubble_is_true_ground(state)
        if grounded and y > ty + 20:
            for _ in range(8):
                hold(session, 1, "A", reason=f"{label}_lower_charge")
            if x < tx - 10:
                dir_h = "RIGHT"
            elif x > tx + 10:
                dir_h = "LEFT"
            else:
                dir_h = "RIGHT" if x < 115 else "LEFT"
            hop = 28 if (y - ty) > 80 else 36
            for _ in range(hop):
                state = hold(
                    session, 1, dir_h, "B", "A", reason=f"{label}_lower_hop"
                )
                bubble_track_state(session, track, state)
                if state.room_id != ROOM_BUBBLE:
                    break
                if _bubble_fire_or_mid_pin(state):
                    track.mid_reached = True
                    break
            if track.mid_reached or state.room_id != ROOM_BUBBLE:
                break
            continue

        if grounded and y <= ty + 20:
            if abs(x - tx) > 8:
                dir_h = "RIGHT" if x < tx else "LEFT"
                hold(session, 1, dir_h, "B", reason=f"{label}_lower_align")
            else:
                hold(session, 1, reason=f"{label}_lower_idle")
            # R16: solid on final fire shelf counts even before pose settle.
            if _bubble_fire_or_mid_pin(state):
                track.mid_reached = True
                break
            continue

        if x < tx - 5:
            dir_h = "RIGHT"
        elif x > tx + 5:
            dir_h = "LEFT"
        else:
            dir_h = "LEFT" if x > 120 else "RIGHT"
        hold(session, 1, dir_h, "B", "A", reason=f"{label}_lower_air")


def bubble_mid_repin(session: ControllerSession, track: ClimbTrack) -> None:
    """Phase A.5 (R3): standing mid re-pin before open-loop launch.

    R16: save-runway / fire-window solid is a valid pin — do **not** walk right
    toward mid-iso and steal the max-left fire seat.
    """
    label = track.label
    stand_lo, stand_hi = P.MID_STAND_X

    for frame in range(P.MID_REPIN_FRAMES):
        state = session.state
        if state.room_id != ROOM_BUBBLE:
            break
        bubble_track_state(session, track, state)
        if bubble_phase_d_top_band(state):
            track.top_reached = True
            track.standing_mid_pinned = True
            break
        # R16: already on fire solid / save runway → hand off to mid launch.
        if bubble_on_save_runway(state) and not bubble_on_launch_lip(state):
            track.standing_mid_pinned = True
            track.mid_reached = True
            break
        if bubble_avoid_wrong_door(session, track, state):
            continue
        if state.pose in (137, 138):
            for _ in range(10):
                hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_repin_kb")
            continue

        x = state.samus_x
        y = state.samus_y
        if x > P.CAVITY_X_MAX and y > P.TOP_Y:
            hold(session, 1, "LEFT", "B", reason=f"{label}_repin_cap")
            continue

        if bubble_on_mid_iso_pin(state):
            for _ in range(4):
                state = hold(session, 1, reason=f"{label}_repin_settle")
                bubble_track_state(session, track, state)
            if state.room_id == ROOM_BUBBLE and bubble_on_mid_iso_pin(state):
                track.standing_mid_pinned = True
                track.mid_reached = True
                break
            continue

        if y > P.MID_Y + 10:
            if state.velocity_y == 0 and state.pose in P.TRUE_GROUND:
                dir_h = "RIGHT" if x < 140 else "LEFT"
                for _ in range(10):
                    hold(session, 1, "A", reason=f"{label}_repin_charge")
                for _ in range(36):
                    state = hold(
                        session, 1, dir_h, "B", "A", reason=f"{label}_repin_hj"
                    )
                    bubble_track_state(session, track, state)
                    if state.room_id != ROOM_BUBBLE:
                        break
                    if bubble_on_mid_iso_pin(state):
                        break
                continue
            dir_h = "RIGHT" if x < 160 else "LEFT"
            hold(session, 1, dir_h, "B", "A", reason=f"{label}_repin_low_spin")
            continue

        if state.velocity_y == 0 and state.pose in P.STAND_PIN:
            if x < stand_lo:
                hold(session, 1, "RIGHT", "B", reason=f"{label}_repin_walk_r")
            elif x > stand_hi:
                hold(session, 1, "LEFT", "B", reason=f"{label}_repin_walk_l")
            else:
                hold(session, 1, reason=f"{label}_repin_idle")
            continue

        dir_h = "RIGHT" if x < stand_lo else ("LEFT" if x > stand_hi else "RIGHT")
        hold(session, 1, dir_h, "B", reason=f"{label}_repin_air")


def bubble_run_mid(
    session: ControllerSession,
    track: ClimbTrack,
    *,
    start: MidStart = "launch",
) -> None:
    """Phase B→D: single mid-budget loop (lip launch optional + right climb)."""
    from super_metroid.routes.kpdr._to_bat_cave_mid import run_mid_loop

    run_mid_loop(session, track, start=start, policy=P)


def bubble_top_super_door(
    session: ControllerSession, track: ClimbTrack
) -> SuperMetroidState:
    """Phase E: top-right Super door → ordinary Bat Cave."""
    return top_super_door(session, track, policy=P)


# ---------------------------------------------------------------------------
# Product + dev entry points
# ---------------------------------------------------------------------------


def _play_bubble_full(
    session: ControllerSession,
    track: ClimbTrack,
) -> SuperMetroidState:
    """Full pure path: land → lower → repin → mid (once) → Super door."""
    bubble_land_and_prepare(session, track, land_frames=40)
    bubble_lower_to_mid_pin(session, track)
    bubble_mid_repin(session, track)
    bubble_run_mid(session, track, start="launch")
    return bubble_top_super_door(session, track)


def play_bubble_to_bat_cave(session: ControllerSession) -> SuperMetroidState:
    """Bubble Mountain mid-left entry → ordinary Bat Cave via top-right Super door.

    Pure source is the CATH-04 successor at node 3 (≈x39–60 / y634).  Phases:
    (1 R5) lower-left ledge multi-hop → save-door pin band, (1.5 R3) standing
    mid re-pin, (2) single mid loop: solid lip launch + right-structure climb
    to top band without leaving ``0xACB3``, (3) Super-open top-right green
    door into ordinary ``0xB07A``.

    Caps: Morph, Bombs, Missiles, Supers (≥1), Hi-Jump, Varia — **no Speed**.
    Hard-avoid wrong doors (Rising Tide / Save / Missiles Super left; Single
    Chamber right mid).

    Product API: session only. Dev handoffs use
    :func:`play_bubble_climb_from_handoff` / :func:`play_bubble_from_top_door`
    / :func:`play_bubble_to_bat_cave_with_phase_capture`.
    """
    return _play_bubble_full(session, _new_track(session))


def play_bubble_to_bat_cave_with_phase_capture(
    session: ControllerSession,
    *,
    dump_phase_c: Path | str | None = None,
    stop_at_phase_c: bool = False,
) -> SuperMetroidState:
    """Dev: full pure path with Phase-C dump/stop (probe only, not product API)."""
    return _play_bubble_full(
        session,
        _new_track(
            session, dump_phase_c=dump_phase_c, stop_at_phase_c=stop_at_phase_c
        ),
    )


def play_bubble_climb_from_handoff(
    session: ControllerSession,
    *,
    dump_phase_c: Path | str | None = None,
    stop_at_phase_c: bool = False,
) -> SuperMetroidState:
    """Dev: skip lower/repin/launch; enter right-structure climb mid loop."""
    track = _new_track(
        session, dump_phase_c=dump_phase_c, stop_at_phase_c=stop_at_phase_c
    )
    # Pin Phase-C sticky before any settle idle (dump ~(301,429) falls out
    # of y≤430 within a few frames).
    st0 = session.state
    if bubble_phase_c_usable_right_contact(st0) or (
        st0.room_id == ROOM_BUBBLE
        and st0.samus_x >= P.RIGHT_SHELF_X - 20
        and st0.samus_y <= P.PHASE_C_Y_MAX + 40
    ):
        track.phase_c_hit = True
    bubble_land_and_prepare(session, track, land_frames=2)
    track.mid_reached = True
    track.launched = True
    bubble_track_state(session, track, session.state)
    bubble_run_mid(session, track, start="climb")
    return bubble_top_super_door(session, track)


def play_bubble_from_top_door(session: ControllerSession) -> SuperMetroidState:
    """Dev: skip to Super door pressure."""
    track = _new_track(session)
    bubble_land_and_prepare(session, track, land_frames=8)
    track.mid_reached = True
    track.standing_mid_pinned = True
    track.launched = True
    track.top_reached = bubble_phase_d_near_top(session.state, slack=40)
    bubble_track_state(session, track, session.state)
    return bubble_top_super_door(session, track)


def capture_bubble_phase_c(
    session: ControllerSession,
    dump_path: Path | str,
    *,
    stop: bool = True,
) -> SuperMetroidState:
    """Run full path with Phase-C dump (and optional stop)."""
    return play_bubble_to_bat_cave_with_phase_capture(
        session,
        dump_phase_c=dump_path,
        stop_at_phase_c=stop,
    )


__all__ = [
    "BUBBLE_PHASE_C_X_MIN",
    "BUBBLE_PHASE_C_Y_MAX",
    "BUBBLE_PHASE_C_Y_MIN",
    "BUBBLE_PHASE_D_X",
    "BUBBLE_PHASE_D_Y",
    "BubblePhaseStop",
    "BubbleTrack",
    "ClimbTrack",
    "ROOM_BAT_CAVE",
    "ROOM_BUBBLE",
    "bubble_avoid_wrong_door",
    "bubble_is_stand_pin_pose",
    "bubble_is_true_ground",
    "bubble_land_and_prepare",
    "bubble_lower_to_mid_pin",
    "bubble_mid_repin",
    "bubble_on_launch_lip",
    "bubble_on_mid_iso_pin",
    "bubble_on_right_shelf",
    "bubble_on_save_runway",
    "bubble_phase_c_usable_right_contact",
    "bubble_phase_d_near_top",
    "bubble_phase_d_top_band",
    "bubble_run_mid",
    "bubble_top_super_door",
    "bubble_track_state",
    "capture_bubble_phase_c",
    "play_bubble_climb_from_handoff",
    "play_bubble_from_top_door",
    "play_bubble_to_bat_cave",
    "play_bubble_to_bat_cave_with_phase_capture",
]
