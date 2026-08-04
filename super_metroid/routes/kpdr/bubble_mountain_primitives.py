"""Bubble Mountain micro-primitives (entry-guarded, reusable).

Physics (pre-Speed, Hi-Jump equipped) — fold into skills, not superstition:

* Pure horizontal runway speed does **not** strongly raise initial vertical
  velocity of a normal jump or wall-jump. Approximate constants:
  Hi-Jump walljump ~5.33 px/f initial; regular WJ ~4.41 px/f.
* What runway speed *does* buy: better wall-contact positioning/timing,
  longer air time before the next jump, and spin-held horizontal momentum.
* Real “horizontal → vertical” transfer comes from: consecutive wall-jumps
  (chain before vy decays), delayed WJ window, crouch-jump (+8 px start),
  or **enemy damage boost** (~5.25 horizontal KB + hold A+dir) — the last is
  non-repeatable and experiment-only (never product).

Product Phase D (human pin ``bubble_human_runway.state`` ~(27,395) p2):

  prepare (Y-clear, no multi-frame bare RIGHT) → dash±arm-pump (21f fire seat)
  → spin-glide 83f → open-loop **double** WJ → right-spin.
  Single WJ stalls mx200~251 and only “wins” via lucky Geruta/Waver clip.

Caps: Morph + Bombs + Missiles + Supers + Hi-Jump + Varia (no Speed).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, NamedTuple, Sequence

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import (
    POSE_WALL_LATCH,
    WallJumpTiming,
    hold,
    is_wall_latch,
    walljump_once,
)
from super_metroid.routes.kpdr import bubble_mountain as bm
from super_metroid.routes.kpdr import bubble_mountain_params as P
from super_metroid.routes.kpdr.rooms import ROOM_BUBBLE

if TYPE_CHECKING:
    from super_metroid.routes.runtime import ControllerSession

# POSE_WALL_LATCH / WallJumpTiming / is_wall_latch re-exported from
# controller_common (canonical home; Bubble was first consumer).

POSE_KNOCKBACK = frozenset({137, 138})
POSE_STAND_LEFT = frozenset({2, 10})
POSE_STAND_RIGHT = frozenset({1, 9})

# Documented vertical classes (px/frame, approximate; Hi-Jump route).
HIJUMP_WALLJUMP_VY0 = 5.33
REGULAR_WALLJUMP_VY0 = 4.41
DAMAGE_BOOST_HX = 5.25  # Geruta/Waver-class horizontal KB magnitude (approx)


# R15 Phase-D proven consecutive pair (human runway pin open-loop).
R15_WJ1 = WallJumpTiming(
    into="LEFT",
    flip="RIGHT",
    into_frames=P.SAVE_WJ_LEFT_A,
    amid_frames=P.SAVE_WJ_AMID,
    flip_frames=P.SAVE_WJ_RIGHT_A,
    delay_into_frames=0,
)
R15_WJ2 = WallJumpTiming(
    into="LEFT",
    flip="RIGHT",
    into_frames=P.SAVE_WJ2_LEFT_A,
    amid_frames=P.SAVE_WJ2_AMID,
    flip_frames=P.SAVE_WJ2_RIGHT_A,
    delay_into_frames=0,
)
R15_DOUBLE: tuple[WallJumpTiming, WallJumpTiming] = (R15_WJ1, R15_WJ2)

AngleSide = Literal["L", "R"]


def bubble_is_wall_latch(state: SuperMetroidState) -> bool:
    """True when Samus is in wall-latch pose (ready to wall-jump)."""
    return is_wall_latch(state)


def bubble_is_knockback(state: SuperMetroidState) -> bool:
    return int(state.pose) in POSE_KNOCKBACK


def bubble_wall_approach_band(
    state: SuperMetroidState,
    *,
    x_min: int | None = None,
    x_max: int | None = None,
    y_min: int | None = None,
    y_max: int | None = None,
) -> bool:
    """Geometry band approaching the right-structure wall (not a latch)."""
    x0 = P.WJ_APPROACH_X[0] if x_min is None else x_min
    x1 = P.WJ_APPROACH_X[1] if x_max is None else x_max
    y0 = P.WJ_APPROACH_Y[0] if y_min is None else y_min
    y1 = P.WJ_APPROACH_Y[1] if y_max is None else y_max
    return (
        int(state.room_id) == ROOM_BUBBLE
        and x0 <= int(state.samus_x) <= x1
        and y0 <= int(state.samus_y) <= y1
    )


def _track_upd(
    session: ControllerSession,
    track: bm.BubbleTrack,
    state: SuperMetroidState,
    height_box: list[bool] | None,
) -> bool:
    """Update track; return True if caller should stop (top / left room)."""
    bm.bubble_track_state(session, track, state)
    if state.room_id != ROOM_BUBBLE:
        return True
    if height_box is not None and state.samus_y <= P.HEIGHT_CLASS_Y:
        height_box[0] = True
    if bm.bubble_phase_d_top_band(state):
        track.top_reached = True
        return True
    return False


# ---------------------------------------------------------------------------
# Runway / seat skills
# ---------------------------------------------------------------------------


def bubble_stationary_missile_clear(
    session: ControllerSession,
    track: bm.BubbleTrack,
    *,
    face_frames: int | None = None,
    shoot_frames: int | None = None,
    angle_l: bool = True,
) -> None:
    """Face left and spray missiles **without** holding LEFT.

    Entry: grounded on save runway / fire solid (caller checks).
    Clears the pure left-blocker near x37. Do **not** LEFT+X while walking.
    R18: optional L-angle spray (``angle_l``) — pure recon needs it to open the
    walk past ~x43 into the human seat band.
    """
    label = track.label
    face_n = P.SAVE_STATIONARY_FACE if face_frames is None else face_frames
    shoot_n = P.SAVE_STATIONARY_X if shoot_frames is None else shoot_frames
    pose = int(session.state.pose)
    if pose not in POSE_STAND_LEFT and int(session.state.samus_x) >= 35:
        for _ in range(min(face_n, 4)):
            hold(session, 1, "LEFT", reason=f"{label}_stat_x_face")
        for _ in range(3):
            hold(session, 1, reason=f"{label}_stat_x_face_settle")
    for _ in range(shoot_n):
        if angle_l:
            state = hold(session, 1, "X", "L", reason=f"{label}_stat_x_shoot")
        else:
            state = hold(session, 1, "X", reason=f"{label}_stat_x_shoot")
        bm.bubble_track_state(session, track, state)
        if state.room_id != ROOM_BUBBLE:
            return
        if int(state.samus_x) < 24:
            hold(session, 1, "RIGHT", reason=f"{label}_stat_x_abort")
            return


def bubble_walk_brake_to_x(
    session: ControllerSession,
    track: bm.BubbleTrack,
    target_x: int,
    *,
    max_frames: int = 80,
    band: int = 2,
) -> bool:
    """Walk toward ``target_x`` with opposite-dir brake each step; settle."""
    label = track.label
    for _ in range(max_frames):
        state = session.state
        if state.room_id != ROOM_BUBBLE:
            return False
        x = int(state.samus_x)
        if int(state.pose) in POSE_KNOCKBACK:
            hold(session, 1, reason=f"{label}_brake_kb")
            continue
        if abs(x - target_x) <= band and bm.bubble_is_true_ground(state):
            hold(
                session,
                1,
                "RIGHT" if x >= target_x else "LEFT",
                reason=f"{label}_brake_stop",
            )
            for _ in range(10):
                hold(session, 1, reason=f"{label}_brake_settle")
            return abs(int(session.state.samus_x) - target_x) <= band + 2
        if x > target_x:
            hold(session, 1, "LEFT", reason=f"{label}_brake_l")
            hold(session, 1, "RIGHT", reason=f"{label}_brake_r")
        else:
            hold(session, 1, "RIGHT", reason=f"{label}_brake_r")
            hold(session, 1, "LEFT", reason=f"{label}_brake_l")
        hold(session, 1, reason=f"{label}_brake_w")
    return abs(int(session.state.samus_x) - target_x) <= band + 2


def bubble_seat_max_left_fire(
    session: ControllerSession,
    track: bm.BubbleTrack,
    *,
    target_x: int | None = None,
    attempts: int = 3,
) -> bool:
    """R18: seat the human max-left fire band without LEFT+X walk.

    Entry: save-door runway / fire solid (caller checks). Sequence (×attempts):

    1. Stationary missile clear (X without dir) — pure left-blocker ~x37
    2. LEFT+B dash-walk toward ``target_x`` (default ~27); abort if Save door
    3. Brief walk-brake settle + face left (pose 2; RIGHT+B run turns from p2)

    Returns True when within human seat band on true_ground, not knockback.
    """
    label = track.label
    human_lo, human_hi = P.SAVE_HUMAN_SEAT_X
    aim = 27 if target_x is None else target_x
    aim = max(human_lo, min(human_hi, aim))

    if session.state.room_id != ROOM_BUBBLE:
        return False
    if not (
        bm.bubble_on_save_runway(session.state)
        or (
            P.SAVE_RUNWAY_Y[0] <= int(session.state.samus_y) <= P.SAVE_RUNWAY_Y[1]
            and P.SAVE_RUNWAY_X[0] - 5
            <= int(session.state.samus_x)
            <= P.SAVE_RUNWAY_X[1]
        )
    ):
        return False

    for attempt in range(max(1, attempts)):
        if session.state.room_id != ROOM_BUBBLE:
            return False
        x0 = int(session.state.samus_x)
        # Abort if too far left (Save door shell).
        if x0 < human_lo - 2:
            hold(session, 1, "RIGHT", "B", reason=f"{label}_seat_door")
            return False

        # Clear pure left-blocker (~x37–43). L-angle X without dir is required;
        # LEFT+X walk → KB p138. Longer spray on later attempts.
        bubble_stationary_missile_clear(
            session,
            track,
            face_frames=P.SAVE_STATIONARY_FACE + attempt * 2,
            shoot_frames=P.SAVE_STATIONARY_X + attempt * 12,
            angle_l=True,
        )
        if session.state.room_id != ROOM_BUBBLE:
            return False

        # Walk left only (no B) — LEFT+B from x~50 overshoots into Save 0xB0DD.
        stalled = 0
        last_x = int(session.state.samus_x)
        for _ in range(70):
            state = session.state
            if state.room_id != ROOM_BUBBLE:
                return False
            x = int(state.samus_x)
            if int(state.pose) in POSE_KNOCKBACK:
                hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_seat_kb")
                continue
            if x < human_lo:
                hold(session, 1, "RIGHT", reason=f"{label}_seat_door")
                hold(session, 1, reason=f"{label}_seat_door_settle")
                break
            if human_lo <= x <= human_hi and bm.bubble_is_true_ground(state):
                break
            if x > aim:
                hold(session, 1, "LEFT", reason=f"{label}_seat_walk_l")
                # Light brake every other step to keep control near the door.
                if _ % 2 == 1 and int(session.state.samus_x) > human_hi:
                    hold(session, 1, "RIGHT", reason=f"{label}_seat_brake")
            elif x < aim:
                hold(session, 1, "RIGHT", reason=f"{label}_seat_walk_r")
            if abs(int(session.state.samus_x) - last_x) <= 0:
                stalled += 1
            else:
                stalled = 0
                last_x = int(session.state.samus_x)
            if stalled >= 12:
                break

        bubble_walk_brake_to_x(session, track, aim, max_frames=40, band=1)

        # Prefer face-left stand (pose 2/10) — multi-frame bare RIGHT walks off.
        pose = int(session.state.pose)
        if pose in POSE_STAND_RIGHT:
            hold(session, 1, "LEFT", reason=f"{label}_seat_face_l")
            for _ in range(6):
                hold(session, 1, reason=f"{label}_seat_face_settle")
        elif pose not in POSE_STAND_LEFT and pose not in POSE_KNOCKBACK:
            for _ in range(4):
                hold(session, 1, reason=f"{label}_seat_settle")

        x = int(session.state.samus_x)
        st = session.state
        if (
            st.room_id == ROOM_BUBBLE
            and human_lo <= x <= human_hi
            and bm.bubble_is_true_ground(st)
            and int(st.pose) not in POSE_KNOCKBACK
        ):
            return True

    x = int(session.state.samus_x)
    st = session.state
    return (
        st.room_id == ROOM_BUBBLE
        and human_lo <= x <= human_hi
        and bm.bubble_is_true_ground(st)
        and int(st.pose) not in POSE_KNOCKBACK
    )


def bubble_prepare_fire_run(
    session: ControllerSession,
    track: bm.BubbleTrack,
    *,
    y_clear: bool = True,
    y_frames: int = 8,
    crouch: bool = False,
    crouch_frames: int | None = None,
) -> None:
    """Hygiene before RIGHT+B dash from the save-door fire seat.

    **Never** multi-frame bare RIGHT on max-left seat — walks Samus off the
    runway (face-right×6 → lip fall, no p132). RIGHT+B run turns from pose 2.

    Optional crouch (1–2f) starts the jump ~8 px higher when geometry allows;
    on the short human fire seat crouch desyncs the proven arc — off by default.
    """
    label = track.label
    pose = int(session.state.pose)
    if pose in POSE_KNOCKBACK:
        for _ in range(8):
            hold(session, 1, reason=f"{label}_fire_kb_settle")
            if int(session.state.pose) not in POSE_KNOCKBACK:
                break
    x = int(session.state.samus_x)
    human_lo, human_hi = P.SAVE_HUMAN_SEAT_X
    if (
        pose in POSE_STAND_LEFT
        and not (human_lo <= x <= human_hi)
        and x > human_hi
    ):
        hold(session, 1, "RIGHT", reason=f"{label}_fire_face_tap")
        hold(session, 1, reason=f"{label}_fire_face_settle")
    if y_clear:
        for _ in range(y_frames):
            state = hold(session, 1, "Y", reason=f"{label}_save_clear")
            bm.bubble_track_state(session, track, state)
            if state.room_id != ROOM_BUBBLE:
                return
    if crouch:
        n = P.SAVE_CROUCH_FRAMES if crouch_frames is None else crouch_frames
        for _ in range(n):
            hold(session, 1, "DOWN", reason=f"{label}_fire_crouch")


def bubble_runway_dash(
    session: ControllerSession,
    track: bm.BubbleTrack,
    *,
    frames: int | None = None,
    arm_pump: bool | None = None,
    arm_period: int | None = None,
    direction: str = "RIGHT",
) -> None:
    """Ground dash with optional arm-pump (L/R angle spam).

    Without Speed Booster, dash (B) saturates around ~32 frames at value 2.
    Fire-seat product uses ``SAVE_RUN_FRAMES`` (21): longer bare dash from
    x~27 walks off the short runway and loses the jump.

    Arm-pump: each L/R pose change shifts Samus ~1 px forward **on ground
    only** — helps short runways reach the jump point with more built momentum;
    arm-pump pixels do not carry into the air.
    """
    label = track.label
    n = P.SAVE_RUN_FRAMES if frames is None else frames
    pump = P.SAVE_ARM_PUMP if arm_pump is None else arm_pump
    period = P.SAVE_ARM_PUMP_PERIOD if arm_period is None else max(1, arm_period)
    for i in range(n):
        if pump:
            ang: AngleSide = "L" if (i // period) % 2 == 0 else "R"
            state = hold(
                session, 1, direction, "B", ang, reason=f"{label}_run_ap"
            )
        else:
            state = hold(session, 1, direction, "B", reason=f"{label}_run")
        bm.bubble_track_state(session, track, state)
        if state.room_id != ROOM_BUBBLE:
            return


def bubble_spin_glide(
    session: ControllerSession,
    track: bm.BubbleTrack,
    *,
    frames: int | None = None,
    height_box: list[bool] | None = None,
) -> bool:
    """RIGHT+B+A spinjump glide — preserve spin (breaking spin kills hx).

    Returns True if Phase D hit during glide.
    """
    label = track.label
    n = P.SAVE_SPIN_FRAMES if frames is None else frames
    for _ in range(n):
        state = hold(
            session, 1, "RIGHT", "B", "A", reason=f"{label}_spin_glide"
        )
        if _track_upd(session, track, state, height_box):
            return bool(track.top_reached)
    return bool(track.top_reached)


# ---------------------------------------------------------------------------
# Wall-jump skills
# ---------------------------------------------------------------------------


def bubble_wait_wall_ready(
    session: ControllerSession,
    track: bm.BubbleTrack,
    *,
    max_frames: int | None = None,
    into_x: int | None = None,
    height_box: list[bool] | None = None,
) -> bool:
    """Extend **right spin only** until wall-ready (band / latch / Phase D).

    Never presses LEFT+A here — that **is** the wall-jump input and burns the
    first WJ window (turns double into single-class mx200~251 stalls).
    """
    label = track.label
    budget = P.WJ_LATCH_TIMEOUT if max_frames is None else max_frames
    edge = P.WJ_INTO_X if into_x is None else into_x
    for _ in range(budget):
        state = session.state
        if _track_upd(session, track, state, height_box):
            return True
        if bubble_is_wall_latch(state):
            return True
        if int(state.samus_x) >= edge and bubble_wall_approach_band(state):
            return True
        state = hold(
            session, 1, "RIGHT", "B", "A", reason=f"{label}_wj_ready"
        )
        if _track_upd(session, track, state, height_box):
            return True
        if bubble_is_wall_latch(state):
            return True
    st = session.state
    return (
        bubble_is_wall_latch(st)
        or int(st.samus_x) >= edge
        or bool(track.top_reached)
    )


def bubble_wait_wall_latch(
    session: ControllerSession,
    track: bm.BubbleTrack,
    *,
    max_frames: int | None = None,
    into_x: int | None = None,
    height_box: list[bool] | None = None,
) -> bool:
    """Alias: wait for readiness without burning WJ (see ``bubble_wait_wall_ready``)."""
    return bubble_wait_wall_ready(
        session,
        track,
        max_frames=max_frames,
        into_x=into_x,
        height_box=height_box,
    )


def bubble_walljump_approach_coast(
    session: ControllerSession,
    track: bm.BubbleTrack,
    *,
    height_box: list[bool] | None = None,
) -> bool:
    """R15 pre-WJ coast: B+A → release → idle → face into wall (LEFT)."""
    label = track.label
    for _ in range(P.SAVE_APPROACH_BA):
        state = hold(session, 1, "B", "A", reason=f"{label}_dwj_coast")
        if _track_upd(session, track, state, height_box):
            return bool(track.top_reached)
    hold(session, 1, "B", reason=f"{label}_dwj_rel")
    for _ in range(P.SAVE_APPROACH_IDLE):
        hold(session, 1, reason=f"{label}_dwj_idle")
    for _ in range(P.SAVE_APPROACH_TURN):
        hold(session, 1, "LEFT", reason=f"{label}_dwj_turn")
    return bool(track.top_reached)


def bubble_walljump_once(
    session: ControllerSession,
    track: bm.BubbleTrack,
    timing: WallJumpTiming | None = None,
    *,
    height_box: list[bool] | None = None,
    reason: str = "wj",
) -> bool:
    """Execute **one** wall-jump pulse via shared :func:`walljump_once`.

    Bubble track / height-box side effects ride on ``stop_when``. Caller must
    chain ≥2 for Phase D on the save-runway arc.
    """
    label = track.label
    t = timing if timing is not None else R15_WJ1

    def _stop(state: SuperMetroidState) -> bool:
        return _track_upd(session, track, state, height_box)

    walljump_once(
        session,
        t,
        reason=f"{label}_{reason}",
        stop_when=_stop,
    )
    return bool(track.top_reached)


def bubble_consecutive_walljumps(
    session: ControllerSession,
    track: bm.BubbleTrack,
    jumps: Sequence[WallJumpTiming] | None = None,
    *,
    count: int | None = None,
    pre_approach: bool = True,
    extend_spin_ready: bool = False,
    height_box: list[bool] | None = None,
    follow_spin: bool = True,
    follow_frames: int | None = None,
) -> bool:
    """Chain N consecutive wall-jumps (default: R15 **double** = 2).

    Single WJ is **insufficient** for Phase D (mx200 stalls ~251) unless a
    lucky enemy damage clip converts horizontal KB — never product.

    Product path: ``pre_approach=True``, ``extend_spin_ready=False`` (open-loop
    R15 timings). ``extend_spin_ready`` only adds RIGHT+B+A if short of the
    wall — never LEFT+A seek (that burns WJ1).
    """
    label = track.label
    if jumps is None:
        pair = list(R15_DOUBLE)
        n = 2 if count is None else max(2, count)  # never default to single
        while len(pair) < n:
            pair.append(R15_WJ2)
        chain = pair[:n]
    else:
        chain = list(jumps)
        if count is not None:
            chain = chain[: max(1, count)]
        if len(chain) < 2:
            # Explicit single is allowed for experiments only — caller opts in.
            pass

    if pre_approach:
        if bubble_walljump_approach_coast(
            session, track, height_box=height_box
        ):
            return True
    if extend_spin_ready:
        bubble_wait_wall_ready(
            session, track, height_box=height_box
        )
        if track.top_reached or session.state.room_id != ROOM_BUBBLE:
            return bool(track.top_reached)

    for i, timing in enumerate(chain):
        if track.top_reached or session.state.room_id != ROOM_BUBBLE:
            break
        bubble_walljump_once(
            session,
            track,
            timing,
            height_box=height_box,
            reason=f"wj{i + 1}",
        )

    if (
        follow_spin
        and not track.top_reached
        and session.state.room_id == ROOM_BUBBLE
    ):
        n_follow = P.SAVE_WJ_FOLLOW if follow_frames is None else follow_frames
        for _ in range(n_follow):
            state = hold(
                session, 1, "RIGHT", "B", "A", reason=f"{label}_dwj_pd_spin"
            )
            if _track_upd(session, track, state, height_box):
                break
            if (
                state.samus_x >= P.RIGHT_SHELF_X
                and state.samus_y <= P.MIDHIGH_Y
            ):
                break

    return bool(track.top_reached)


def bubble_double_walljump_r15(
    session: ControllerSession,
    track: bm.BubbleTrack,
    *,
    height_class_out: list[bool] | None = None,
    extend_spin_ready: bool = False,
) -> bool:
    """R15 consecutive **double** wall-jump + right-spin Phase D push.

    Open-loop R15 timings (proven on human pin). Optional ``extend_spin_ready``
    only stretches RIGHT+B+A if short of the wall — does **not** LEFT+A-seek.

    Success: ``bubble_phase_d_top_band`` (x≥300 y≤200).
    """
    height_box = height_class_out if height_class_out is not None else [False]
    ok = bubble_consecutive_walljumps(
        session,
        track,
        R15_DOUBLE,
        pre_approach=True,
        extend_spin_ready=extend_spin_ready,
        height_box=height_box,
        follow_spin=True,
    )
    if height_class_out is not None:
        height_class_out[0] = bool(height_box[0])
    return ok


def bubble_walljump_second_left_wall(
    session: ControllerSession,
    track: bm.BubbleTrack,
    *,
    seek_frames: int | None = None,
    into_frames: int = 8,
    flip_frames: int = 16,
    height_box: list[bool] | None = None,
) -> bool:
    """R18 experiment: after right-wall WJ1, seek **left-wall** contact (pose 84).

    Human pin tops via pose **84** (facing-right walljump) at ~(212,157) — a
    left-structure contact after WJ1, not a second pose-132 right-wall latch.
    Pure clear dump matches WJ1 p132 then fails this left-wall bounce
    (subpixel/physics residual; enemy kill does not fix).

    Does **not** LEFT+A-seek the right wall (that burns WJ1). Seeks with
    LEFT+B+A / LEFT spin toward the left column, then RIGHT+A flip.
    """
    label = track.label
    budget = P.WJ2_LEFT_SEEK if seek_frames is None else seek_frames
    for _ in range(budget):
        state = session.state
        if _track_upd(session, track, state, height_box):
            return bool(track.top_reached)
        # Pose 84/83 = walljump anim; 132 right-wall latch is wrong side here.
        pose = int(state.pose)
        if pose in (83, 84) or (
            int(state.samus_x) <= P.WJ2_LEFT_X and int(state.samus_y) <= 200
        ):
            break
        # Drift left-up toward left structure without RIGHT-wall LEFT+A.
        state = hold(
            session, 1, "LEFT", "B", "A", reason=f"{label}_wj2_left_seek"
        )
        if _track_upd(session, track, state, height_box):
            return bool(track.top_reached)
    for _ in range(into_frames):
        state = hold(session, 1, "RIGHT", "A", reason=f"{label}_wj2_left_into")
        if _track_upd(session, track, state, height_box):
            return bool(track.top_reached)
    for _ in range(flip_frames):
        state = hold(
            session, 1, "RIGHT", "B", "A", reason=f"{label}_wj2_left_flip"
        )
        if _track_upd(session, track, state, height_box):
            return bool(track.top_reached)
    return bool(track.top_reached)


def bubble_period_walljump_climb(
    session: ControllerSession,
    track: bm.BubbleTrack,
    *,
    frames: int = 48,
    period: int | None = None,
    into: int | None = None,
    bounce: int | None = None,
    height_box: list[bool] | None = None,
) -> bool:
    """Period-N right-structure WJ climb (bug-layout tolerant height farm).

    On latch, upgrades to a full consecutive double + follow (no single ship).
    """
    label = track.label
    per = P.RIGHT_WJ_PERIOD if period is None else period
    n_into = P.RIGHT_WJ_INTO if into is None else into
    n_bounce = P.RIGHT_WJ_BOUNCE if bounce is None else bounce
    for i in range(frames):
        state = session.state
        if _track_upd(session, track, state, height_box):
            return bool(track.top_reached)
        if bubble_is_wall_latch(state):
            return bubble_consecutive_walljumps(
                session,
                track,
                R15_DOUBLE,
                pre_approach=False,
                extend_spin_ready=False,
                height_box=height_box,
                follow_spin=True,
            )
        ph = i % per
        if ph < n_into:
            state = hold(session, 1, "LEFT", "A", reason=f"{label}_pwj_into")
        elif ph < n_into + n_bounce:
            state = hold(session, 1, "RIGHT", "A", reason=f"{label}_pwj_bounce")
        else:
            state = hold(
                session, 1, "RIGHT", "B", "A", reason=f"{label}_pwj_spin"
            )
        if _track_upd(session, track, state, height_box):
            return bool(track.top_reached)
    return bool(track.top_reached)


def bubble_damage_boost_hold(
    session: ControllerSession,
    track: bm.BubbleTrack,
    *,
    direction: str = "RIGHT",
    frames: int | None = None,
    height_box: list[bool] | None = None,
) -> bool:
    """**Experiment only** — hold A+dir during knockback to convert KB to height.

    Geruta/Waver contact ~5.25 horizontal KB; 5-frame i-frames window. This is
    the non-repeatable transfer that makes single-WJ + bug videos “work.”
    Never gate product Phase D on this.
    """
    label = track.label
    n = P.DMG_BOOST_HOLD_FRAMES if frames is None else frames
    for _ in range(n):
        state = hold(
            session, 1, direction, "A", reason=f"{label}_dmg_boost"
        )
        if _track_upd(session, track, state, height_box):
            return bool(track.top_reached)
        # Also accept pure A+B spin during KB recovery.
        if not bubble_is_knockback(session.state) and int(session.state.pose) not in (
            25,
            26,
            129,
            130,
        ):
            break
    return bool(track.top_reached)


# ---------------------------------------------------------------------------
# Enemy-phase-aware fire (R19)
# ---------------------------------------------------------------------------


class BubbleEnemySnap(NamedTuple):
    """One enemy slot snapshot (WRAM 0x0F78 + slot*0x40)."""

    slot: int
    enemy_id: int
    x: int
    y: int
    hp: int


def _session_env(session: ControllerSession) -> Any | None:
    """Probe/RouteSession expose ``env``; Protocol surface does not require it."""
    return getattr(session, "env", None)


def bubble_read_enemy_slot(
    session: ControllerSession, slot: int
) -> BubbleEnemySnap | None:
    """Read enemy slot id/x/y/hp from session env RAM. None if env missing."""
    env = _session_env(session)
    if env is None:
        return None
    get_ram = getattr(env, "get_ram", None)
    if get_ram is None:
        return None
    ram = get_ram()
    base = 0x0F78 + int(slot) * 0x40

    def u16(off: int) -> int:
        return int(ram[base + off]) | (int(ram[base + off + 1]) << 8)

    return BubbleEnemySnap(
        slot=int(slot),
        enemy_id=u16(0x00),
        x=u16(0x02),
        y=u16(0x06),
        hp=u16(0x14),
    )


def bubble_fire_phase_geometry(
    e4_x: int,
    e4_y: int,
    e6_x: int,
    e6_y: int,
) -> bool:
    """True when Geruta slots 4/6 sit in a proven Phase-D-clear patrol class.

    Pure product open-loop fire hard-caps mx200≈251 unless enemy AI phase
    matches a short window (R18 residual; copying live AI blobs unlocks top;
    zeroing HP alone does not). Two geometric classes from R19 recon:

    * **A** fullpure wait ~89–93 / pure_seat ~108–110
    * **B** fullpure wait ~233–235

    Live isolation also tops at a third geometry, but near-miss boxes false-
    positive on pure seats — not product. Pure function of coordinates.
    """

    def _in(
        x: int, y: int, box: tuple[int, int, int, int]
    ) -> bool:
        x0, x1, y0, y1 = box
        return x0 <= x <= x1 and y0 <= y <= y1

    if _in(e4_x, e4_y, P.FIRE_PHASE_A_E4) and _in(
        e6_x, e6_y, P.FIRE_PHASE_A_E6
    ):
        return True
    if _in(e4_x, e4_y, P.FIRE_PHASE_B_E4) and _in(
        e6_x, e6_y, P.FIRE_PHASE_B_E6
    ):
        return True
    return False


def bubble_fire_phase_clear(session: ControllerSession) -> bool:
    """Read slots 4/6 from env and test :func:`bubble_fire_phase_geometry`.

    Returns False when env is unavailable (skip wait; fire immediately).
    """
    e4 = bubble_read_enemy_slot(session, 4)
    e6 = bubble_read_enemy_slot(session, 6)
    if e4 is None or e6 is None:
        return False
    return bubble_fire_phase_geometry(e4.x, e4.y, e6.x, e6.y)


def bubble_wait_fire_phase(
    session: ControllerSession,
    track: bm.BubbleTrack,
    *,
    max_frames: int | None = None,
) -> bool:
    """Idle on the fire seat until Geruta phase is clear (or budget expires).

    Entry: seated max-left human band (caller checks). Uses pure idle holds —
    **no** LEFT+X / walk that would deseat or knockback. Preserves seat for
    hundreds of frames (R19 recon). Returns True if clear was observed.

    Does **not** patch enemy RAM. Diagnostic enemy copy is never product.
    """
    label = track.label
    budget = P.FIRE_PHASE_MAX_WAIT if max_frames is None else max_frames
    human_lo, human_hi = P.SAVE_HUMAN_SEAT_X
    if budget <= 0:
        return bubble_fire_phase_clear(session)

    if bubble_fire_phase_clear(session):
        return True

    for _ in range(budget):
        state = session.state
        if state.room_id != ROOM_BUBBLE:
            return False
        x = int(state.samus_x)
        y = int(state.samus_y)
        # Abort if we leave the fire seat (Save door / fall / KB).
        if not (
            human_lo - 2 <= x <= human_hi + 4
            and P.SAVE_RUNWAY_Y[0] <= y <= P.SAVE_RUNWAY_Y[1]
        ):
            return False
        if int(state.pose) in POSE_KNOCKBACK:
            hold(session, 1, reason=f"{label}_phase_kb")
            continue
        hold(session, 1, reason=f"{label}_phase_wait")
        bm.bubble_track_state(session, track, session.state)
        if bubble_fire_phase_clear(session):
            return True
    return bubble_fire_phase_clear(session)


# ---------------------------------------------------------------------------
# Full fire-seat recipes
# ---------------------------------------------------------------------------


def bubble_save_runway_fire_recipe(
    session: ControllerSession,
    track: bm.BubbleTrack,
    *,
    y_clear: bool = True,
    crouch: bool = False,
    run_frames: int | None = None,
    arm_pump: bool | None = None,
    spin_frames: int | None = None,
    extend_spin_ready: bool = False,
    wj_count: int = 2,
    phase_wait: bool = True,
    phase_max_frames: int | None = None,
) -> bool:
    """Composable left-platform → right-structure Phase D recipe.

    Product defaults: R19 enemy-phase wait (idle, seat-preserving), no crouch,
    fire-seat run frames, arm-pump off on pure, spin 83, open-loop double WJ.
    Experiment knobs for max-dash (32), crouch, single WJ (``wj_count=1``, not
    for product), extend_spin_ready, ``phase_wait=False`` isolation.
    """
    height_box = [False]
    if phase_wait:
        bubble_wait_fire_phase(
            session, track, max_frames=phase_max_frames
        )
    bubble_prepare_fire_run(
        session, track, y_clear=y_clear, crouch=crouch
    )
    bubble_runway_dash(
        session,
        track,
        frames=run_frames,
        arm_pump=arm_pump,
    )
    if bubble_spin_glide(
        session, track, frames=spin_frames, height_box=height_box
    ):
        return True
    if session.state.room_id != ROOM_BUBBLE:
        return bool(track.top_reached)
    n = max(1, wj_count)
    jumps = list(R15_DOUBLE)
    while len(jumps) < n:
        jumps.append(R15_WJ2)
    return bubble_consecutive_walljumps(
        session,
        track,
        jumps[:n],
        pre_approach=True,
        extend_spin_ready=extend_spin_ready,
        height_box=height_box,
        follow_spin=True,
    )


def bubble_save_runway_open_loop_r15(
    session: ControllerSession,
    track: bm.BubbleTrack,
    *,
    face_right: bool = False,
    y_clear: bool = True,
    arm_pump: bool | None = None,
    extend_spin_ready: bool = False,
) -> bool:
    """Full R15 fire recipe from a seated save-runway pin (product wrapper).

    ``face_right`` is legacy and ignored when False (default). Multi-frame
    face-right walks off max-left — do not use on the human fire seat.
    """
    if face_right:
        label = track.label
        for _ in range(6):
            hold(session, 1, "RIGHT", reason=f"{label}_save_face")
        for _ in range(4):
            hold(session, 1, reason=f"{label}_save_settle")
    return bubble_save_runway_fire_recipe(
        session,
        track,
        y_clear=y_clear,
        crouch=False,
        arm_pump=arm_pump,
        extend_spin_ready=extend_spin_ready,
        wj_count=2,
    )
