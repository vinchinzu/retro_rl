"""Bubble Mountain climb: mid-left entry → top-right Super door → Bat Cave.

Phase ladder (docs/tasks/SM-K4.4-PHASE-LADDER.md / HARD_ROOM_SPLITS):
  A lower-left multi-hop → mid pin · A.5 mid re-pin · B+C+D mid loop
  (lip launch + right-structure climb) · E Super door.

Product path is :func:`play_bubble_to_bat_cave` (session only). Dev handoffs
and Phase-C capture live on separate helpers — never on the product API.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import (
    hold,
    require_room,
    select_weapon,
    unmorph,
    wait_ordinary_room,
)
from super_metroid.routes.kpdr import bubble_mountain_params as P
from super_metroid.routes.kpdr.rooms import (
    ROOM_BAT_CAVE,
    ROOM_BUBBLE,
)
from super_metroid.routes.runtime import ControllerSession

# Public phase-ladder aliases (tests / probe / k4 re-exports).
BUBBLE_PHASE_C_X_MIN = P.PHASE_C_X_MIN
BUBBLE_PHASE_C_Y_MAX = P.PHASE_C_Y_MAX
BUBBLE_PHASE_C_Y_MIN = P.PHASE_C_Y_MIN
BUBBLE_PHASE_D_X = P.PHASE_D_X
BUBBLE_PHASE_D_Y = P.PHASE_D_Y

MidStart = Literal["launch", "climb"]


# ---------------------------------------------------------------------------
# Phase predicates
# ---------------------------------------------------------------------------


class BubblePhaseStop(Exception):
    """Diagnostic early exit when a pure probe stops at a phase pin.

    Probe CLI may treat this as success for capture/recon only — never as
    hop GREEN to Bat Cave / continuous evidence.
    """

    def __init__(
        self,
        phase: str,
        state: SuperMetroidState,
        *,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        self.phase = phase
        self.state = state
        self.metrics = dict(metrics or {})
        super().__init__(
            f"bubble_phase_stop:{phase} room=0x{state.room_id:04X} "
            f"pose={state.pose} xy=({state.samus_x},{state.samus_y}) "
            f"vx={state.velocity_x} vy={state.velocity_y}"
        )


def bubble_phase_c_usable_right_contact(st: SuperMetroidState) -> bool:
    """Phase C: usable right-structure contact at height (not floor thrash)."""
    return (
        int(st.room_id) == ROOM_BUBBLE
        and P.PHASE_C_X_MIN <= int(st.samus_x) <= P.CAVITY_X_MAX
        and P.PHASE_C_Y_MIN <= int(st.samus_y) <= P.PHASE_C_Y_MAX
    )


def bubble_phase_d_top_band(st: SuperMetroidState) -> bool:
    """Phase D: top band before Super door (single source of truth)."""
    return (
        int(st.room_id) == ROOM_BUBBLE
        and int(st.samus_y) <= P.PHASE_D_Y
        and int(st.samus_x) >= P.PHASE_D_X
    )


def bubble_phase_d_near_top(st: SuperMetroidState, slack: int = 40) -> bool:
    """Looser top band for door-start / near-top handoff (slack px)."""
    if bubble_phase_d_top_band(st):
        return True
    return (
        int(st.room_id) == ROOM_BUBBLE
        and int(st.samus_y) <= P.PHASE_D_Y + slack
        and int(st.samus_x) >= P.PHASE_D_X - slack
    )


def bubble_is_true_ground(st: SuperMetroidState, *, max_vy: int = 1) -> bool:
    """True solid ground (charge/reseat/land). Spin apex pose 25 is NOT ground."""
    return abs(int(st.velocity_y)) <= max_vy and int(st.pose) in P.TRUE_GROUND


def bubble_is_stand_pin_pose(st: SuperMetroidState, *, max_vy: int = 1) -> bool:
    """Stand-pin pose class at rest (includes 25/26).

    Use for lip *detection* and approach walk (R6/HEAD). Do **not** use for
    mid-air charge/reseat (R11: pose 25 + vy≈0 is spin apex, not a land).
    """
    return abs(int(st.velocity_y)) <= max_vy and int(st.pose) in P.STAND_PIN


def bubble_on_mid_iso_pin(st: SuperMetroidState) -> bool:
    """Mid-iso pin class (stand_pin poses + velocity + xy band)."""
    stand_lo, stand_hi = P.MID_STAND_X
    return (
        abs(int(st.velocity_y)) <= 2
        and int(st.pose) in P.STAND_PIN
        and stand_lo <= int(st.samus_x) <= stand_hi
        and int(st.samus_y) <= P.MID_Y + 10
    )


def bubble_on_launch_lip(st: SuperMetroidState) -> bool:
    """Solid save-door lip seat (stand-pin poses — R6 / pre-extract HEAD).

    Lip detection must accept pose 25/26; true_ground-only blocked launch on
    natural pure (launched=False thrash). Charge still fires from that seat.
    """
    lip_lo, lip_hi = P.LIP_X
    lip_y_lo, lip_y_hi = P.LIP_Y
    return (
        bubble_is_stand_pin_pose(st)
        and lip_lo <= st.samus_x <= lip_hi
        and lip_y_lo <= st.samus_y <= lip_y_hi
    )


def bubble_on_right_shelf(st: SuperMetroidState) -> bool:
    """Right-structure shelf class (stand-pin; matches pre-extract HEAD)."""
    return (
        bubble_is_stand_pin_pose(st)
        and st.samus_x >= P.RIGHT_SHELF_X
        and st.samus_x <= P.CAVITY_X_MAX
        and st.samus_y <= P.RIGHT_SHELF_Y
        and st.samus_y >= 200
    )


def bubble_on_save_runway(st: SuperMetroidState) -> bool:
    """Save-door outer runway for maprando left climb (no Ice).

    Grounded stand-pin on the mid-left platform outside Save (not inside
    ``0xB0DD``). Human ``bubble_jump_try`` pin ~(27–55, 395); R15 max-left
    fire window down to x≈25.
    """
    x_lo, x_hi = P.SAVE_RUNWAY_X
    y_lo, y_hi = P.SAVE_RUNWAY_Y
    return (
        bubble_is_stand_pin_pose(st)
        and x_lo <= int(st.samus_x) <= x_hi
        and y_lo <= int(st.samus_y) <= y_hi
    )


# ---------------------------------------------------------------------------
# Metrics track (climb outcomes + optional recon dump; no mid-loop control)
# ---------------------------------------------------------------------------


@dataclass
class BubbleTrack:
    """Climb metrics and optional Phase-C recon flags.

    Mid-loop control state (phase / frame budget / height class) lives inside
    :func:`run_bubble_mid_loop` as locals — not on this track.
    """

    label: str = "bubble_to_bat_cave"
    max_x: int = 0
    min_y: int = 0
    mid_reached: bool = False
    top_reached: bool = False
    door_reached: bool = False
    standing_mid_pinned: bool = False
    launched: bool = False
    phase_c_hit: bool = False
    # Recon-only; product path leaves defaults.
    dump_path: Path | None = None
    phase_c_dumped: bool = False
    stop_at_phase_c: bool = False

    def metrics_dict(self) -> dict[str, Any]:
        return {
            "max_x": self.max_x,
            "min_y": self.min_y,
            "mid_reached": self.mid_reached,
            "top_reached": self.top_reached,
            "phase_c_hit": self.phase_c_hit,
            "dump_phase_c": str(self.dump_path) if self.dump_path else None,
            "dumped": self.phase_c_dumped,
        }


def _maybe_dump_bubble_phase_c(
    session: ControllerSession,
    state: SuperMetroidState,
    track: BubbleTrack,
) -> None:
    """Save first Phase-C pin when probe session exposes ``env``."""
    if track.dump_path is None or track.phase_c_dumped:
        return
    if not bubble_phase_c_usable_right_contact(state):
        return
    env = getattr(session, "env", None)
    if env is None:
        return
    from super_metroid.dev.common import save_dev_state

    save_dev_state(env, track.dump_path)
    track.phase_c_dumped = True


def bubble_track_state(
    session: ControllerSession, track: BubbleTrack, state: SuperMetroidState
) -> None:
    """Update climb metrics; optional Phase-C dump/stop for recon tracks."""
    track.max_x = max(track.max_x, state.samus_x)
    track.min_y = min(track.min_y, state.samus_y)
    if state.samus_y <= P.MID_Y and state.samus_x >= 90:
        track.mid_reached = True
    if bubble_phase_d_top_band(state):
        track.top_reached = True
    if bubble_phase_c_usable_right_contact(state):
        if not track.phase_c_hit:
            track.phase_c_hit = True
            _maybe_dump_bubble_phase_c(session, state, track)
            if track.stop_at_phase_c:
                raise BubblePhaseStop("C", state, metrics=track.metrics_dict())


def bubble_avoid_wrong_door(
    session: ControllerSession, track: BubbleTrack, state: SuperMetroidState
) -> bool:
    """Hard-steer away from side doors; return True if a steer was applied."""
    label = track.label
    x, y = state.samus_x, state.samus_y
    # Left doors: Rising Tide ~y624, Save ~y368, Missiles Super ~y112.
    # R16 fire window needs x down to ~25 at y~395 (human pin). Only force
    # RIGHT when deeper than the fire band or not at save height.
    y_lo, y_hi = P.SAVE_RUNWAY_Y
    fire_lo, _fire_hi = P.SAVE_RUNWAY_FIRE_X
    on_save_platform = y_lo <= y <= y_hi
    if x < 22 or (x < 55 and not on_save_platform):
        hold(session, 1, "RIGHT", "B", reason=f"{label}_avoid_left")
        return True
    if on_save_platform and x < fire_lo - 1:
        # Soft nudge off the door shell (x≲24) without abandoning fire seat.
        hold(session, 1, "RIGHT", "B", reason=f"{label}_avoid_save_door")
        return True
    # Right mid Single Chamber ~y368 / x≈496.
    if x > 470 and 300 <= y <= 430:
        hold(session, 1, "LEFT", "B", reason=f"{label}_avoid_sc")
        return True
    return False


def _new_track(
    session: ControllerSession,
    *,
    dump_phase_c: Path | str | None = None,
    stop_at_phase_c: bool = False,
) -> BubbleTrack:
    st = session.state
    return BubbleTrack(
        max_x=st.samus_x,
        min_y=st.samus_y,
        dump_path=Path(dump_phase_c) if dump_phase_c is not None else None,
        stop_at_phase_c=stop_at_phase_c,
    )


# ---------------------------------------------------------------------------
# Phase helpers (lower / repin / door). Mid is one call in bubble_mountain_mid.
# ---------------------------------------------------------------------------


def bubble_land_and_prepare(
    session: ControllerSession,
    track: BubbleTrack,
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


def bubble_lower_to_mid_pin(session: ControllerSession, track: BubbleTrack) -> None:
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


def bubble_mid_repin(session: ControllerSession, track: BubbleTrack) -> None:
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
    track: BubbleTrack,
    *,
    start: MidStart = "launch",
) -> None:
    """Phase B→D: single mid-budget loop (lip launch optional + right climb)."""
    from super_metroid.routes.kpdr.bubble_mountain_mid import run_bubble_mid_loop

    run_bubble_mid_loop(session, track, start=start)


def bubble_top_super_door(
    session: ControllerSession, track: BubbleTrack
) -> SuperMetroidState:
    """Phase E: top-right Super door → ordinary Bat Cave."""
    label = track.label
    if session.state.selected_item != 2:
        select_weapon(session, 2)

    for frame in range(P.DOOR_FRAMES):
        state = session.state
        if state.room_id == ROOM_BAT_CAVE:
            break
        if state.room_id != ROOM_BUBBLE:
            break
        bubble_track_state(session, track, state)
        if bubble_avoid_wrong_door(session, track, state):
            continue
        if state.pose in (137, 138):
            for _ in range(8):
                hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_door_kb")
            continue

        if state.samus_y > 220 or state.samus_x < 280:
            if state.selected_item != 0:
                select_weapon(session, 0)
            dir_h = "RIGHT" if state.samus_x < 320 else "LEFT"
            phase = frame % 16
            if phase < 10:
                hold(session, 1, dir_h, "B", "A", reason=f"{label}_door_climb")
            elif phase < 12:
                hold(session, 1, dir_h, "B", reason=f"{label}_door_rel")
            else:
                opp = "LEFT" if dir_h == "RIGHT" else "RIGHT"
                hold(session, 1, opp, "A", reason=f"{label}_door_wj")
            continue

        track.door_reached = True
        if state.selected_item != 2:
            select_weapon(session, 2)
        phase = frame % 28
        if phase < 4:
            inputs = ("RIGHT", "X")
        elif phase < 14:
            inputs = ("RIGHT",)
        elif phase < 20:
            inputs = ("RIGHT", "B")
        else:
            inputs = ("RIGHT", "B", "A")
        state = hold(session, 1, *inputs, reason=f"{label}_door")
        if state.room_id == ROOM_BAT_CAVE:
            break
    else:
        state = session.state
        raise TimeoutError(
            f"{label}: Bat Cave Super door missed before room "
            f"0x{ROOM_BAT_CAVE:04X}; room=0x{state.room_id:04X} "
            f"pose={state.pose} xy=({state.samus_x},{state.samus_y}) "
            f"door_transition={state.door_transition} max_x={track.max_x} "
            f"min_y={track.min_y} mid_reached={track.mid_reached} "
            f"top_reached={track.top_reached} door_reached={track.door_reached} "
            f"standing_mid_pinned={track.standing_mid_pinned} "
            f"launched={track.launched} phase_c_hit={track.phase_c_hit} "
            f"supers={state.super_missiles} selected={state.selected_item}"
        )

    if session.state.room_id != ROOM_BAT_CAVE:
        state = session.state
        raise TimeoutError(
            f"{label}: left Bubble without ordinary Bat Cave; "
            f"room=0x{state.room_id:04X} pose={state.pose} "
            f"xy=({state.samus_x},{state.samus_y}) max_x={track.max_x} "
            f"min_y={track.min_y} mid_reached={track.mid_reached} "
            f"top_reached={track.top_reached} door_reached={track.door_reached} "
            f"standing_mid_pinned={track.standing_mid_pinned} "
            f"phase_c_hit={track.phase_c_hit}"
        )

    return wait_ordinary_room(
        session,
        ROOM_BAT_CAVE,
        settle_frames=P.TO_BAT_SETTLE_FRAMES,
        label=label,
    )


# ---------------------------------------------------------------------------
# Product + dev entry points
# ---------------------------------------------------------------------------


def _play_bubble_full(
    session: ControllerSession,
    track: BubbleTrack,
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
