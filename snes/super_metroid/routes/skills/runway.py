"""Runway / fire-seat micro-skills and full recipes (policy-driven).

Product Phase D path (human pin ~(27,395) p2):

  prepare (Y-clear, no multi-frame bare RIGHT) → dash±arm-pump
  → spin-glide → open-loop **double** WJ → right-spin.
  Single WJ stalls mx200~251 and only “wins” via lucky enemy clip.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, NamedTuple, Protocol

from super_metroid.routes.controller_common import WallJumpTiming, hold
from super_metroid.routes.skills.geometry import (
    ClimbTrack,
    POSE_KNOCKBACK,
    POSE_STAND_LEFT,
    POSE_STAND_RIGHT,
    is_true_ground,
    on_save_runway,
    track_state,
)
from super_metroid.routes.skills.walljump import (
    _track_upd,
    consecutive_walljumps,
)

if TYPE_CHECKING:
    from super_metroid.routes.runtime import ControllerSession

AngleSide = Literal["L", "R"]


class RunwayPolicy(Protocol):
    """Policy fields used by runway / fire-seat skills."""

    ROOM_ID: int
    TRUE_GROUND: frozenset[int]
    SAVE_STATIONARY_FACE: int
    SAVE_STATIONARY_X: int
    SAVE_HUMAN_SEAT_X: tuple[int, int]
    SAVE_RUNWAY_X: tuple[int, int]
    SAVE_RUNWAY_Y: tuple[int, int]
    SAVE_CROUCH_FRAMES: int
    SAVE_RUN_FRAMES: int
    SAVE_ARM_PUMP: bool
    SAVE_ARM_PUMP_PERIOD: int
    SAVE_SPIN_FRAMES: int
    FIRE_PHASE_MAX_WAIT: int
    FIRE_PHASE_A_E4: tuple[int, int, int, int]
    FIRE_PHASE_A_E6: tuple[int, int, int, int]
    FIRE_PHASE_B_E4: tuple[int, int, int, int]
    FIRE_PHASE_B_E6: tuple[int, int, int, int]
    R15_DOUBLE: tuple[WallJumpTiming, WallJumpTiming]
    R15_WJ2: WallJumpTiming
    # Inherited geometry surface for track_state / on_save_runway
    STAND_PIN: frozenset[int]
    MID_Y: int
    MID_STAND_X: tuple[int, int]
    LIP_X: tuple[int, int]
    LIP_Y: tuple[int, int]
    RIGHT_SHELF_X: int
    RIGHT_SHELF_Y: int
    CAVITY_X_MAX: int
    SAVE_RUNWAY_FIRE_X: tuple[int, int]
    PHASE_C_X_MIN: int
    PHASE_C_Y_MIN: int
    PHASE_C_Y_MAX: int
    PHASE_D_X: int
    PHASE_D_Y: int
    HEIGHT_CLASS_Y: int


def _default_policy() -> RunwayPolicy:
    from super_metroid.routes.skills.policies import bubble_to_bat as pol

    return pol  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Runway / seat skills
# ---------------------------------------------------------------------------


def stationary_missile_clear(
    session: ControllerSession,
    track: ClimbTrack,
    *,
    policy: RunwayPolicy | None = None,
    face_frames: int | None = None,
    shoot_frames: int | None = None,
    angle_l: bool = True,
) -> None:
    """Face left and spray missiles **without** holding LEFT.

    Entry: grounded on save runway / fire solid (caller checks).
    Clears the pure left-blocker near x37. Do **not** LEFT+X while walking.
    """
    pol = policy if policy is not None else _default_policy()
    label = track.label
    face_n = pol.SAVE_STATIONARY_FACE if face_frames is None else face_frames
    shoot_n = pol.SAVE_STATIONARY_X if shoot_frames is None else shoot_frames
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
        track_state(session, track, state, pol)  # type: ignore[arg-type]
        if state.room_id != pol.ROOM_ID:
            return
        if int(state.samus_x) < 24:
            hold(session, 1, "RIGHT", reason=f"{label}_stat_x_abort")
            return


def walk_brake_to_x(
    session: ControllerSession,
    track: ClimbTrack,
    target_x: int,
    *,
    policy: RunwayPolicy | None = None,
    max_frames: int = 80,
    band: int = 2,
) -> bool:
    """Walk toward ``target_x`` with opposite-dir brake each step; settle."""
    pol = policy if policy is not None else _default_policy()
    label = track.label
    for _ in range(max_frames):
        state = session.state
        if state.room_id != pol.ROOM_ID:
            return False
        x = int(state.samus_x)
        if int(state.pose) in POSE_KNOCKBACK:
            hold(session, 1, reason=f"{label}_brake_kb")
            continue
        if abs(x - target_x) <= band and is_true_ground(
            state, poses=pol.TRUE_GROUND
        ):
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


def seat_max_left_fire(
    session: ControllerSession,
    track: ClimbTrack,
    *,
    policy: RunwayPolicy | None = None,
    target_x: int | None = None,
    attempts: int = 3,
) -> bool:
    """Seat the human max-left fire band without LEFT+X walk.

    Entry: save-door runway / fire solid (caller checks). Sequence (×attempts):

    1. Stationary missile clear (X without dir)
    2. LEFT walk toward ``target_x`` (default ~27); abort if Save door
    3. Brief walk-brake settle + face left

    Returns True when within human seat band on true_ground, not knockback.
    """
    pol = policy if policy is not None else _default_policy()
    label = track.label
    human_lo, human_hi = pol.SAVE_HUMAN_SEAT_X
    aim = 27 if target_x is None else target_x
    aim = max(human_lo, min(human_hi, aim))

    if session.state.room_id != pol.ROOM_ID:
        return False
    if not (
        on_save_runway(session.state, pol)  # type: ignore[arg-type]
        or (
            pol.SAVE_RUNWAY_Y[0] <= int(session.state.samus_y) <= pol.SAVE_RUNWAY_Y[1]
            and pol.SAVE_RUNWAY_X[0] - 5
            <= int(session.state.samus_x)
            <= pol.SAVE_RUNWAY_X[1]
        )
    ):
        return False

    for attempt in range(max(1, attempts)):
        if session.state.room_id != pol.ROOM_ID:
            return False
        x0 = int(session.state.samus_x)
        if x0 < human_lo - 2:
            hold(session, 1, "RIGHT", "B", reason=f"{label}_seat_door")
            return False

        stationary_missile_clear(
            session,
            track,
            policy=pol,
            face_frames=pol.SAVE_STATIONARY_FACE + attempt * 2,
            shoot_frames=pol.SAVE_STATIONARY_X + attempt * 12,
            angle_l=True,
        )
        if session.state.room_id != pol.ROOM_ID:
            return False

        stalled = 0
        last_x = int(session.state.samus_x)
        for _ in range(70):
            state = session.state
            if state.room_id != pol.ROOM_ID:
                return False
            x = int(state.samus_x)
            if int(state.pose) in POSE_KNOCKBACK:
                hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_seat_kb")
                continue
            if x < human_lo:
                hold(session, 1, "RIGHT", reason=f"{label}_seat_door")
                hold(session, 1, reason=f"{label}_seat_door_settle")
                break
            if human_lo <= x <= human_hi and is_true_ground(
                state, poses=pol.TRUE_GROUND
            ):
                break
            if x > aim:
                hold(session, 1, "LEFT", reason=f"{label}_seat_walk_l")
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

        walk_brake_to_x(
            session, track, aim, policy=pol, max_frames=40, band=1
        )

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
            st.room_id == pol.ROOM_ID
            and human_lo <= x <= human_hi
            and is_true_ground(st, poses=pol.TRUE_GROUND)
            and int(st.pose) not in POSE_KNOCKBACK
        ):
            return True

    x = int(session.state.samus_x)
    st = session.state
    return (
        st.room_id == pol.ROOM_ID
        and human_lo <= x <= human_hi
        and is_true_ground(st, poses=pol.TRUE_GROUND)
        and int(st.pose) not in POSE_KNOCKBACK
    )


def prepare_fire_run(
    session: ControllerSession,
    track: ClimbTrack,
    *,
    policy: RunwayPolicy | None = None,
    y_clear: bool = True,
    y_frames: int = 8,
    crouch: bool = False,
    crouch_frames: int | None = None,
) -> None:
    """Hygiene before RIGHT+B dash from the save-door fire seat.

    **Never** multi-frame bare RIGHT on max-left seat — walks Samus off the
    runway. RIGHT+B run turns from pose 2.
    """
    pol = policy if policy is not None else _default_policy()
    label = track.label
    pose = int(session.state.pose)
    if pose in POSE_KNOCKBACK:
        for _ in range(8):
            hold(session, 1, reason=f"{label}_fire_kb_settle")
            if int(session.state.pose) not in POSE_KNOCKBACK:
                break
    x = int(session.state.samus_x)
    human_lo, human_hi = pol.SAVE_HUMAN_SEAT_X
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
            track_state(session, track, state, pol)  # type: ignore[arg-type]
            if state.room_id != pol.ROOM_ID:
                return
    if crouch:
        n = pol.SAVE_CROUCH_FRAMES if crouch_frames is None else crouch_frames
        for _ in range(n):
            hold(session, 1, "DOWN", reason=f"{label}_fire_crouch")


def runway_dash(
    session: ControllerSession,
    track: ClimbTrack,
    *,
    policy: RunwayPolicy | None = None,
    frames: int | None = None,
    arm_pump: bool | None = None,
    arm_period: int | None = None,
    direction: str = "RIGHT",
) -> None:
    """Ground dash with optional arm-pump (L/R angle spam)."""
    pol = policy if policy is not None else _default_policy()
    label = track.label
    n = pol.SAVE_RUN_FRAMES if frames is None else frames
    pump = pol.SAVE_ARM_PUMP if arm_pump is None else arm_pump
    period = (
        pol.SAVE_ARM_PUMP_PERIOD if arm_period is None else max(1, arm_period)
    )
    for i in range(n):
        if pump:
            ang: AngleSide = "L" if (i // period) % 2 == 0 else "R"
            state = hold(
                session, 1, direction, "B", ang, reason=f"{label}_run_ap"
            )
        else:
            state = hold(session, 1, direction, "B", reason=f"{label}_run")
        track_state(session, track, state, pol)  # type: ignore[arg-type]
        if state.room_id != pol.ROOM_ID:
            return


def spin_glide(
    session: ControllerSession,
    track: ClimbTrack,
    *,
    policy: RunwayPolicy | None = None,
    frames: int | None = None,
    height_box: list[bool] | None = None,
) -> bool:
    """RIGHT+B+A spinjump glide — preserve spin (breaking spin kills hx).

    Returns True if Phase D hit during glide.
    """
    pol = policy if policy is not None else _default_policy()
    label = track.label
    n = pol.SAVE_SPIN_FRAMES if frames is None else frames
    for _ in range(n):
        state = hold(
            session, 1, "RIGHT", "B", "A", reason=f"{label}_spin_glide"
        )
        if _track_upd(session, track, state, pol, height_box):  # type: ignore[arg-type]
            return bool(track.top_reached)
    return bool(track.top_reached)


# ---------------------------------------------------------------------------
# Enemy-phase-aware fire
# ---------------------------------------------------------------------------


class EnemySnap(NamedTuple):
    """One enemy slot snapshot (WRAM 0x0F78 + slot*0x40)."""

    slot: int
    enemy_id: int
    x: int
    y: int
    hp: int


# Historical name.
BubbleEnemySnap = EnemySnap


def _session_env(session: ControllerSession) -> Any | None:
    """Probe/RouteSession expose ``env``; Protocol surface does not require it."""
    return getattr(session, "env", None)


def read_enemy_slot(
    session: ControllerSession, slot: int
) -> EnemySnap | None:
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

    return EnemySnap(
        slot=int(slot),
        enemy_id=u16(0x00),
        x=u16(0x02),
        y=u16(0x06),
        hp=u16(0x14),
    )


def fire_phase_geometry(
    e4_x: int,
    e4_y: int,
    e6_x: int,
    e6_y: int,
    *,
    policy: RunwayPolicy | None = None,
) -> bool:
    """True when Geruta slots 4/6 sit in a proven Phase-D-clear patrol class."""
    pol = policy if policy is not None else _default_policy()

    def _in(
        x: int, y: int, box: tuple[int, int, int, int]
    ) -> bool:
        x0, x1, y0, y1 = box
        return x0 <= x <= x1 and y0 <= y <= y1

    if _in(e4_x, e4_y, pol.FIRE_PHASE_A_E4) and _in(
        e6_x, e6_y, pol.FIRE_PHASE_A_E6
    ):
        return True
    if _in(e4_x, e4_y, pol.FIRE_PHASE_B_E4) and _in(
        e6_x, e6_y, pol.FIRE_PHASE_B_E6
    ):
        return True
    return False


def fire_phase_clear(
    session: ControllerSession, *, policy: RunwayPolicy | None = None
) -> bool:
    """Read slots 4/6 from env and test :func:`fire_phase_geometry`.

    Returns False when env is unavailable (skip wait; fire immediately).
    """
    pol = policy if policy is not None else _default_policy()
    e4 = read_enemy_slot(session, 4)
    e6 = read_enemy_slot(session, 6)
    if e4 is None or e6 is None:
        return False
    return fire_phase_geometry(e4.x, e4.y, e6.x, e6.y, policy=pol)


def wait_fire_phase(
    session: ControllerSession,
    track: ClimbTrack,
    *,
    policy: RunwayPolicy | None = None,
    max_frames: int | None = None,
) -> bool:
    """Idle on the fire seat until Geruta phase is clear (or budget expires).

    Entry: seated max-left human band (caller checks). Uses pure idle holds —
    **no** LEFT+X / walk that would deseat or knockback.
    """
    pol = policy if policy is not None else _default_policy()
    label = track.label
    budget = pol.FIRE_PHASE_MAX_WAIT if max_frames is None else max_frames
    human_lo, human_hi = pol.SAVE_HUMAN_SEAT_X
    if budget <= 0:
        return fire_phase_clear(session, policy=pol)

    if fire_phase_clear(session, policy=pol):
        return True

    for _ in range(budget):
        state = session.state
        if state.room_id != pol.ROOM_ID:
            return False
        x = int(state.samus_x)
        y = int(state.samus_y)
        if not (
            human_lo - 2 <= x <= human_hi + 4
            and pol.SAVE_RUNWAY_Y[0] <= y <= pol.SAVE_RUNWAY_Y[1]
        ):
            return False
        if int(state.pose) in POSE_KNOCKBACK:
            hold(session, 1, reason=f"{label}_phase_kb")
            continue
        hold(session, 1, reason=f"{label}_phase_wait")
        track_state(session, track, session.state, pol)  # type: ignore[arg-type]
        if fire_phase_clear(session, policy=pol):
            return True
    return fire_phase_clear(session, policy=pol)


# ---------------------------------------------------------------------------
# Full fire-seat recipes
# ---------------------------------------------------------------------------


def save_runway_fire_recipe(
    session: ControllerSession,
    track: ClimbTrack,
    *,
    policy: RunwayPolicy | None = None,
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

    Product defaults: enemy-phase wait, no crouch, fire-seat run frames,
    arm-pump from policy, open-loop double WJ.
    """
    pol = policy if policy is not None else _default_policy()
    height_box = [False]
    if phase_wait:
        wait_fire_phase(
            session, track, policy=pol, max_frames=phase_max_frames
        )
    prepare_fire_run(
        session, track, policy=pol, y_clear=y_clear, crouch=crouch
    )
    runway_dash(
        session,
        track,
        policy=pol,
        frames=run_frames,
        arm_pump=arm_pump,
    )
    if spin_glide(
        session, track, policy=pol, frames=spin_frames, height_box=height_box
    ):
        return True
    if session.state.room_id != pol.ROOM_ID:
        return bool(track.top_reached)
    n = max(1, wj_count)
    jumps = list(pol.R15_DOUBLE)
    while len(jumps) < n:
        jumps.append(pol.R15_WJ2)
    return consecutive_walljumps(
        session,
        track,
        jumps[:n],
        policy=pol,  # type: ignore[arg-type]
        pre_approach=True,
        extend_spin_ready=extend_spin_ready,
        height_box=height_box,
        follow_spin=True,
    )


def save_runway_open_loop(
    session: ControllerSession,
    track: ClimbTrack,
    *,
    policy: RunwayPolicy | None = None,
    face_right: bool = False,
    y_clear: bool = True,
    arm_pump: bool | None = None,
    extend_spin_ready: bool = False,
) -> bool:
    """Full fire recipe from a seated save-runway pin (product wrapper).

    ``face_right`` is legacy and ignored when False (default). Multi-frame
    face-right walks off max-left — do not use on the human fire seat.
    """
    pol = policy if policy is not None else _default_policy()
    if face_right:
        label = track.label
        for _ in range(6):
            hold(session, 1, "RIGHT", reason=f"{label}_save_face")
        for _ in range(4):
            hold(session, 1, reason=f"{label}_save_settle")
    return save_runway_fire_recipe(
        session,
        track,
        policy=pol,
        y_clear=y_clear,
        crouch=False,
        arm_pump=arm_pump,
        extend_spin_ready=extend_spin_ready,
        wj_count=2,
    )


# Historical bubble_* aliases.
bubble_stationary_missile_clear = stationary_missile_clear
bubble_walk_brake_to_x = walk_brake_to_x
bubble_seat_max_left_fire = seat_max_left_fire
bubble_prepare_fire_run = prepare_fire_run
bubble_runway_dash = runway_dash
bubble_spin_glide = spin_glide
bubble_read_enemy_slot = read_enemy_slot
bubble_fire_phase_geometry = fire_phase_geometry
bubble_fire_phase_clear = fire_phase_clear
bubble_wait_fire_phase = wait_fire_phase
bubble_save_runway_fire_recipe = save_runway_fire_recipe
bubble_save_runway_open_loop_r15 = save_runway_open_loop

__all__ = [
    "EnemySnap",
    "BubbleEnemySnap",
    "stationary_missile_clear",
    "walk_brake_to_x",
    "seat_max_left_fire",
    "prepare_fire_run",
    "runway_dash",
    "spin_glide",
    "read_enemy_slot",
    "fire_phase_geometry",
    "fire_phase_clear",
    "wait_fire_phase",
    "save_runway_fire_recipe",
    "save_runway_open_loop",
    "bubble_stationary_missile_clear",
    "bubble_walk_brake_to_x",
    "bubble_seat_max_left_fire",
    "bubble_prepare_fire_run",
    "bubble_runway_dash",
    "bubble_spin_glide",
    "bubble_read_enemy_slot",
    "bubble_fire_phase_geometry",
    "bubble_fire_phase_clear",
    "bubble_wait_fire_phase",
    "bubble_save_runway_fire_recipe",
    "bubble_save_runway_open_loop_r15",
]
