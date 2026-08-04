"""Named climb phases for Kraid return reverse hops (Kihunter / Zeela).

Product hops in :mod:`super_metroid.routes.kpdr.from_kraid` compose these
phases. Geometry and frame budgets are frozen for pure-probe green; edit one
named phase at a time — do not invent mid-loop lineage branches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import (
    ensure_morph,
    hold,
    select_weapon,
    unmorph,
    wait_ordinary_room,
)
from super_metroid.routes.kpdr.rooms import (
    ITEM_HI_JUMP,
    ROOM_BABY_KRAID,
    ROOM_WAREHOUSE,
    ROOM_WAREHOUSE_KIHUNTER,
    ROOM_ZEELA,
)
from super_metroid.routes.skills.door_exit import beam_open_door, period_exit_push
from super_metroid.routes.skills.morph_bomb import (
    align_x,
    morph_bomb_hole_climb,
    morph_roll_to_window,
    morph_upper_plant,
)

if TYPE_CHECKING:
    from super_metroid.routes.runtime import ControllerSession

# ---------------------------------------------------------------------------
# Phase names (Zeela continuous reverse climb)
# ---------------------------------------------------------------------------

ZEELA_PHASE_BOTTOM_ROLL = "bottom_roll"
ZEELA_PHASE_MID_PLATFORM = "mid_platform"
ZEELA_PHASE_BELOW_LIP = "below_platform_lip"
ZEELA_PHASE_WALL_PLANT = "wall_plant"
ZEELA_PHASE_SHOTBLOCK_CLEAR = "shotblock_clear"
ZEELA_PHASE_WALL_REPLANT = "wall_replant"
ZEELA_PHASE_WALL_SPIN_CLIMB = "wall_spin_climb"
ZEELA_PHASE_SHOTBLOCK_CLIMB = "shotblock_wall_climb"  # composer of clear+replant+spin
ZEELA_PHASE_WAREHOUSE_DOOR = "warehouse_door_exit"

# Eye → Baby mid-room (controller-dev reverse; composed from from_kraid)
EYE_PHASE_MID_ROOM = "eye_mid_room_approach"


# ---------------------------------------------------------------------------
# Kihunter guards / predicates
# ---------------------------------------------------------------------------


def kihunter_guard_rooms(
    session: ControllerSession, label: str, *, allow_zeela: bool = False
) -> SuperMetroidState:
    """Fail loud on Baby / wrong-room transitions during the reverse climb."""
    state = session.state
    if state.room_id == ROOM_BABY_KRAID:
        raise TimeoutError(f"{label}: crossed wrong door into Baby Kraid: {state}")
    if state.room_id == ROOM_ZEELA:
        if allow_zeela:
            return state
        raise TimeoutError(
            f"{label}: reached Zeela before true upper Kihunter land: {state}"
        )
    if state.room_id != ROOM_WAREHOUSE_KIHUNTER:
        raise TimeoutError(f"{label}: left source room during climb: {state}")
    return state


def kihunter_upper_band(state: SuperMetroidState) -> bool:
    """True when Samus is in the upper Kihunter band (not mid ledge y≈299)."""
    return (
        state.samus_y < 220
        and state.room_id == ROOM_WAREHOUSE_KIHUNTER
        and state.door_transition == 0
    )


# ---------------------------------------------------------------------------
# Kihunter → Zeela phases
# ---------------------------------------------------------------------------


def kihunter_wall_plant(session: ControllerSession, label: str) -> None:
    """Plant on the hard wall at x≈357 (floor wall blocks walking under hole)."""
    hold(session, 10, reason="kihunter_zeela_entry_release")
    for _ in range(220):
        state = kihunter_guard_rooms(session, label)
        if state.samus_x <= 358:
            break
        hold(session, 1, "LEFT", "B", reason="kihunter_zeela_wall_run")
    for _ in range(40):
        state = kihunter_guard_rooms(session, label)
        if state.samus_x <= 120:
            break
        hold(session, 1, "LEFT", reason="kihunter_zeela_wall_plant")
    hold(session, 4, reason="kihunter_zeela_wall_settle")


def kihunter_mid_ledge(
    session: ControllerSession, label: str, *, best_min_y: list[int]
) -> None:
    """Hi-Jump, drift RIGHT onto mid ledge (y≈299, x≥365)."""
    hold(session, 3, "RIGHT", reason="kihunter_zeela_face_ledge")
    hold(session, 2, reason="kihunter_zeela_face_settle")
    hold(session, 8, "DOWN", reason="kihunter_zeela_crouch_load")
    mid = False
    has_hi_jump = bool(session.state.collected_items & ITEM_HI_JUMP)
    for frame in range(110):
        state = session.state
        # Hi-Jump reaches the y=291 ledge several frames earlier than the
        # historical no-Hi-Jump fixture. Start the rightward landing drift
        # from the live height rather than an absolute loop frame, then cap it
        # at the same x≈367 hand-off used by the fixture route.
        if has_hi_jump and state.samus_y <= 300:
            buttons = ("RIGHT", "A", "B") if state.samus_x < 367 else ()
        elif frame < 30:
            buttons = ("A",)
        elif frame < 45:
            buttons = ("A", "UP", "X")
        elif state.samus_y <= 300:
            buttons = ("RIGHT", "A", "B")
        else:
            buttons = ("RIGHT", "B")
        state = hold(session, 1, *buttons, reason="kihunter_zeela_mid_ledge")
        best_min_y[0] = min(best_min_y[0], state.samus_y)
        kihunter_guard_rooms(session, label)
        if (
            state.samus_y <= 299
            and state.velocity_y == 0
            and state.samus_x >= 365
            and frame > 40
        ):
            mid = True
            break
    hold(session, 16, reason="kihunter_zeela_mid_settle")
    if not mid and not (session.state.samus_y <= 305 and session.state.samus_x >= 360):
        raise TimeoutError(
            f"{label}: mid ledge missed: {session.state}; best_min_y={best_min_y[0]}"
        )


def kihunter_bomb_hole(
    session: ControllerSession, label: str, *, best_min_y: list[int]
) -> None:
    """Align under bomb hole x≈376, morph bomb-jump to upper band, plant."""
    align_x(
        session,
        x_lo=374,
        x_hi=380,
        label="kihunter_zeela_hole",
        max_frames=50,
        settle_frames=6,
        guard=lambda _s: kihunter_guard_rooms(session, label),
        reason="align",
    )
    morph_bomb_hole_climb(
        session,
        label="kihunter_zeela",
        hole_x_lo=372,
        hole_x_hi=382,
        success_y=200,
        peak_y=210,
        settle_y=240,
        firm_y=195,
        max_cycles=90,
        guard=lambda _s: kihunter_guard_rooms(session, label),
        best_min_y=best_min_y,
    )
    morph_upper_plant(
        session,
        label="kihunter_zeela",
        plant_y=190,
        max_bombs=8,
        wait_frames=22,
        settle_frames=10,
        fail_y=230,
        guard=lambda _s: kihunter_guard_rooms(session, label),
        best_min_y=best_min_y,
    )


def kihunter_upper_to_zeela_window(session: ControllerSession, label: str) -> None:
    """Morph-roll left to Zeela down-door window x=96..160, stand, open, drop."""
    morph_roll_to_window(
        session,
        label="kihunter_zeela",
        x_lo=96,
        x_hi=160,
        y_max=230,
        max_frames=500,
        sink_y=210,
        fall_y=300,
        boost_wait=18,
        source_room=ROOM_WAREHOUSE_KIHUNTER,
        forbidden_rooms=frozenset({ROOM_BABY_KRAID}),
    )

    unmorph(session)
    select_weapon(session, 0)
    hold(session, 10, reason="kihunter_zeela_window_stand")
    state = session.state
    if not 90 <= state.samus_x <= 170 or state.samus_y >= 250:
        raise TimeoutError(f"{label}: invalid Zeela door window: {state}")
    hold(session, 8, "LEFT", reason="kihunter_zeela_door_face")
    hold(session, 6, reason="kihunter_zeela_door_release")
    beam_open_door(
        session,
        label="kihunter_zeela",
        shots=6,
        shot_frames=4,
        fuse_frames=14,
        shot_buttons=("DOWN", "X"),
    )

    def _wrong(state: SuperMetroidState) -> None:
        if state.room_id == ROOM_BABY_KRAID:
            raise TimeoutError(
                f"{label}: blue down-door entered Baby Kraid: {session.state}"
            )

    period_exit_push(
        session,
        ROOM_ZEELA,
        label="kihunter_zeela",
        max_frames=520,
        period=30,
        windows=(
            (8, ("DOWN", "A"), "drop"),
            (14, ("DOWN", "A", "B"), "drop_spin"),
            (18, ("DOWN", "X"), "reshot"),
            (30, ("DOWN",), "drop"),
        ),
        on_wrong_room=_wrong,
    )


# ---------------------------------------------------------------------------
# Zeela → Warehouse phases
# ---------------------------------------------------------------------------


def _zeela_guard(
    session: ControllerSession, label: str, state: SuperMetroidState, phase: str
) -> None:
    if state.door_transition and state.samus_y > 250:
        raise TimeoutError(
            f"{label}: floor door transition during {phase}: {state}"
        )
    if state.room_id != ROOM_ZEELA:
        raise TimeoutError(f"{label}: left Zeela during {phase}: {state}")


def zeela_bottom_roll(session: ControllerSession, label: str) -> None:
    """Morph reverse-roll + align second-drop lane."""
    ensure_morph(session)
    for _ in range(900):
        state = hold(session, 1, "LEFT", reason="zeela_warehouse_bottom_roll")
        _zeela_guard(session, label, state, ZEELA_PHASE_BOTTOM_ROLL)
        if state.samus_x <= 160 and state.samus_y >= 350:
            break
    else:
        raise TimeoutError(
            f"{label}: {ZEELA_PHASE_BOTTOM_ROLL} stalled: {session.state}"
        )

    unmorph(session)
    align_x(
        session,
        x_lo=110,
        x_hi=140,
        label="zeela_warehouse_second",
        max_frames=160,
        settle_frames=8,
        guard=lambda s: _zeela_guard(session, label, s, ZEELA_PHASE_BOTTOM_ROLL),
        reason="align",
    )


def zeela_mid_platform(session: ControllerSession, label: str) -> None:
    """Reverse-shot climb onto middle platform (y∈[300,350], x≥90)."""
    select_weapon(session, 0)
    has_hi_jump = bool(session.state.collected_items & ITEM_HI_JUMP)
    mid = False
    for frame in range(700):
        state = session.state
        _zeela_guard(session, label, state, ZEELA_PHASE_MID_PLATFORM)
        mid_x_min = 96 if has_hi_jump else 90
        if (
            300 <= state.samus_y <= 350
            and state.velocity_y == 0
            and state.samus_x >= mid_x_min
            and frame > 50
        ):
            mid = True
            break
        cadence = frame % 28
        if cadence < 5:
            buttons: tuple[str, ...] = ("UP", "X")
        elif state.samus_y <= 360:
            # In/near the hole — drift RIGHT onto the mid platform.
            buttons = ("RIGHT", "A", "B") if cadence >= 16 else ("RIGHT", "A")
        elif cadence < 14:
            buttons = ("A",)
        elif state.samus_x <= 70:
            buttons = ("RIGHT", "A")
        else:
            buttons = ("LEFT", "A")
        hold(session, 1, *buttons, reason="zeela_warehouse_second_reverse_shot")
    hold(session, 12, reason="zeela_warehouse_mid_settle")
    if not mid and not (
        300 <= session.state.samus_y <= 355 and session.state.samus_x >= 85
    ):
        raise TimeoutError(
            f"{label}: {ZEELA_PHASE_MID_PLATFORM} missed: {session.state}"
        )


def zeela_below_platform_lip(session: ControllerSession, label: str) -> None:
    """Walk mid right-edge, crouch-load to below-platform lip."""
    for _ in range(80):
        state = session.state
        _zeela_guard(session, label, state, ZEELA_PHASE_BELOW_LIP)
        if state.samus_y > 360:
            raise TimeoutError(f"{label}: fell off mid platform: {state}")
        if state.samus_x >= 104:
            break
        hold(session, 1, "RIGHT", reason="zeela_warehouse_mid_right_edge")
    hold(session, 8, reason="zeela_warehouse_mid_edge_settle")

    unmorph(session)
    select_weapon(session, 0)
    hold(session, 10, "DOWN", reason="zeela_warehouse_first_crouch_load")
    hold(session, 2, reason="zeela_warehouse_first_crouch_release")
    lip = False
    for frame in range(600):
        state = session.state
        _zeela_guard(session, label, state, ZEELA_PHASE_BELOW_LIP)
        if (
            state.samus_y <= 240
            and state.velocity_y == 0
            and state.samus_x <= 80
            and frame > 40
        ):
            lip = True
            break
        if frame < 18:
            buttons: tuple[str, ...] = ("A",)
        elif frame < 30:
            buttons = ("A", "UP", "X")
        elif state.samus_y < 280:
            buttons = ("LEFT", "A", "B")
        else:
            cadence = frame % 24
            if cadence < 6:
                buttons = ("UP", "X")
            elif cadence < 16:
                buttons = ("A",)
            else:
                buttons = ()
        hold(session, 1, *buttons, reason="zeela_warehouse_first_crouch_climb")
    hold(session, 15, reason="zeela_warehouse_lip_settle")
    if not lip and not (session.state.samus_y <= 250 and session.state.samus_x <= 90):
        raise TimeoutError(
            f"{label}: {ZEELA_PHASE_BELOW_LIP} missed: {session.state}"
        )


def zeela_wall_plant(session: ControllerSession, label: str) -> None:
    """Hop-left from lip land (~x=69) to wall plant band (x≈37, y≈219)."""
    unmorph(session)
    for frame in range(50):
        state = session.state
        _zeela_guard(session, label, state, ZEELA_PHASE_WALL_PLANT)
        if (
            state.samus_x <= 45
            and state.velocity_y == 0
            and 210 <= state.samus_y <= 230
            and frame > 10
        ):
            break
        if frame < 8:
            buttons: tuple[str, ...] = ("A",)
        elif frame < 28:
            buttons = ("LEFT", "A")
        else:
            buttons = ("LEFT",)
        hold(session, 1, *buttons, reason="zeela_warehouse_lip_hop_left")
    hold(session, 20, reason="zeela_warehouse_bp_settle")
    if not (session.state.samus_y <= 230 and session.state.samus_x <= 55):
        raise TimeoutError(
            f"{label}: {ZEELA_PHASE_WALL_PLANT} missed: {session.state}"
        )


def zeela_shotblock_clear(session: ControllerSession, label: str) -> None:
    """Stand on lip + clear reverse-entry shot blocks (UP+X then clear-jump).

    Hard cap without clear is y≈188; this phase must run before wall-spin.
    """
    unmorph(session)
    select_weapon(session, 0)
    for _ in range(20):
        if session.state.pose in (1, 2):
            break
        hold(session, 1, "UP", reason="zeela_warehouse_clear_stand")
    hold(session, 8, reason="zeela_warehouse_clear_stand_settle")
    if session.state.samus_y > 250:
        raise TimeoutError(f"{label}: fell off lip before clear: {session.state}")

    for _ in range(40):
        hold(session, 2, "UP", "X", reason="zeela_warehouse_shotblock_clear")
        hold(session, 4, reason="zeela_warehouse_shotblock_fuse")
    for frame in range(60):
        state = session.state
        _zeela_guard(session, label, state, ZEELA_PHASE_SHOTBLOCK_CLEAR)
        cadence = frame % 8
        if cadence < 3:
            buttons: tuple[str, ...] = ("UP", "X")
        elif cadence < 6:
            buttons = ("A",)
        else:
            buttons = ()
        hold(session, 1, *buttons, reason="zeela_warehouse_clear_jump")


def zeela_wall_replant(session: ControllerSession, label: str) -> None:
    """Re-plant on left wall (x≤35) after shotblock clear, before spin climb."""
    for _ in range(30):
        state = session.state
        _zeela_guard(session, label, state, ZEELA_PHASE_WALL_REPLANT)
        if state.samus_x <= 35:
            break
        if state.samus_y > 250:
            break
        hold(session, 1, "LEFT", "B", reason="zeela_warehouse_wall_plant")
    hold(session, 4, reason="zeela_warehouse_wall_plant_settle")


@dataclass(frozen=True)
class WallSpinResult:
    """Outcome of one :func:`zeela_wall_spin_climb` attempt."""

    reached_top: bool = False
    warehouse_settle: SuperMetroidState | None = None


def zeela_wall_spin_climb(session: ControllerSession, label: str) -> WallSpinResult:
    """Wall-spin cadence to top door band (y≤150 standing).

    ``reached_top`` means grounded top band confirmed in Zeela.
    ``warehouse_settle`` is set when Hi-Jump lineage enters Warehouse mid-climb
    (composer returns that state). Both false means retry / failed attempt.
    """
    for frame in range(900):
        state = session.state
        _zeela_guard(session, label, state, ZEELA_PHASE_WALL_SPIN_CLIMB)
        if (
            state.samus_y <= 150
            and state.velocity_y == 0
            and state.pose in (1, 2, 39, 40, 137, 138)
            and frame > 20
        ):
            hold(session, 8, reason="zeela_warehouse_top_confirm")
            if (
                session.state.samus_y <= 155
                and session.state.velocity_y == 0
                and session.state.room_id == ROOM_ZEELA
            ):
                return WallSpinResult(reached_top=True)
        cadence = frame % 14
        if cadence < 7:
            buttons: tuple[str, ...] = ("LEFT", "A", "B")
        elif cadence < 10:
            buttons = ("RIGHT", "A")
        elif cadence < 12:
            buttons = ("LEFT", "A")
        else:
            buttons = ("LEFT",)
        state = hold(session, 1, *buttons, reason="zeela_warehouse_wall_climb")
        # Hi-Jump continuous lineage may naturally enter Warehouse during
        # the final wall-climb cadence (upper-left door, not floor E-Tank).
        if state.room_id == ROOM_WAREHOUSE:
            if state.samus_y > 250:
                raise TimeoutError(
                    f"{label}: floor door transition during wall climb: {state}"
                )
            return WallSpinResult(
                warehouse_settle=wait_ordinary_room(
                    session,
                    ROOM_WAREHOUSE,
                    settle_frames=320,
                    label=label,
                )
            )
        if session.state.samus_y > 280:
            break
    return WallSpinResult()


def zeela_shotblock_wall_climb(
    session: ControllerSession, label: str
) -> SuperMetroidState | None:
    """Composer: shotblock clear + wall replant + spin climb (up to 3 attempts).

    Returns early ``wait_ordinary_room`` state if Hi-Jump lineage naturally
    enters Warehouse during the climb; otherwise returns None when top is set.
    """
    for _attempt in range(3):
        zeela_shotblock_clear(session, label)
        zeela_wall_replant(session, label)
        spin = zeela_wall_spin_climb(session, label)
        if spin.warehouse_settle is not None:
            return spin.warehouse_settle
        if spin.reached_top:
            return None
        # Still on the lip — retry clear/climb once more.
        if not (session.state.samus_y <= 230 and session.state.samus_x <= 55):
            break
    raise TimeoutError(
        f"{label}: {ZEELA_PHASE_SHOTBLOCK_CLIMB} top band missed: {session.state}"
    )


def eye_mid_room_approach(session: ControllerSession, label: str) -> None:
    """Jump-left past mid-room ledge pin (~x=373) to door approach (x≤140).

    Floor-walk / floor-spin pins mid-room on the Eye reverse hop; clear with
    sustained hop-left + spin + clear-shot cadence. Does not open the hatch.
    """
    for index in range(500):
        state = session.state
        if state.samus_x <= 140:
            break
        phase = index % 30
        if phase < 10:
            hold(session, 1, "LEFT", "A", reason="eye_to_baby_hop")
        elif phase < 18:
            hold(session, 1, "LEFT", "A", "B", reason="eye_to_baby_spin")
        elif phase < 22:
            hold(session, 1, "X", reason="eye_to_baby_clear_shot")
        else:
            hold(session, 1, "LEFT", "B", reason="eye_to_baby_run")
    else:
        raise TimeoutError(
            f"{label}: {EYE_PHASE_MID_ROOM} timed out: {session.state}"
        )


def zeela_warehouse_door_exit(session: ControllerSession, label: str) -> None:
    """Standing LEFT beams into Warehouse 0xA6A1 from top band."""
    unmorph(session)
    select_weapon(session, 0)
    for _ in range(15):
        if session.state.pose in (1, 2):
            break
        hold(session, 1, "UP", reason="zeela_warehouse_door_stand")
    hold(session, 10, reason="zeela_warehouse_door_stand_settle")
    if session.state.samus_y > 170:
        raise TimeoutError(f"{label}: fell before Warehouse door: {session.state}")

    hold(session, 6, "LEFT", reason="zeela_warehouse_door_face")
    hold(session, 4, reason="zeela_warehouse_door_face_release")
    beam_open_door(
        session,
        label="zeela_warehouse",
        shots=10,
        shot_frames=3,
        fuse_frames=12,
        shot_buttons=("LEFT", "X"),
    )

    def _guard_exit(state: SuperMetroidState) -> None:
        if state.door_transition and state.samus_y > 250:
            raise TimeoutError(
                f"{label}: floor door transition at Warehouse exit: {state}"
            )
        if state.room_id != ROOM_ZEELA and state.room_id != ROOM_WAREHOUSE:
            raise TimeoutError(f"{label}: unexpected exit room: {state}")
        if state.room_id == ROOM_ZEELA and state.samus_y > 200:
            raise TimeoutError(f"{label}: fell during Warehouse exit: {state}")

    try:
        period_exit_push(
            session,
            ROOM_WAREHOUSE,
            label="zeela_warehouse",
            max_frames=400,
            period=20,
            windows=(
                (10, ("LEFT", "A"), "exit"),
                (14, ("LEFT", "A", "B"), "exit_spin"),
                (20, ("LEFT",), "exit"),
            ),
            guard=_guard_exit,
        )
    except TimeoutError as exc:
        raise TimeoutError(
            f"{label}: {ZEELA_PHASE_WAREHOUSE_DOOR} timed out: {session.state}"
        ) from exc


__all__ = [
    "ZEELA_PHASE_BOTTOM_ROLL",
    "ZEELA_PHASE_MID_PLATFORM",
    "ZEELA_PHASE_BELOW_LIP",
    "ZEELA_PHASE_WALL_PLANT",
    "ZEELA_PHASE_SHOTBLOCK_CLEAR",
    "ZEELA_PHASE_WALL_REPLANT",
    "ZEELA_PHASE_WALL_SPIN_CLIMB",
    "ZEELA_PHASE_SHOTBLOCK_CLIMB",
    "ZEELA_PHASE_WAREHOUSE_DOOR",
    "EYE_PHASE_MID_ROOM",
    "kihunter_guard_rooms",
    "kihunter_upper_band",
    "kihunter_wall_plant",
    "kihunter_mid_ledge",
    "kihunter_bomb_hole",
    "kihunter_upper_to_zeela_window",
    "zeela_bottom_roll",
    "zeela_mid_platform",
    "zeela_below_platform_lip",
    "zeela_wall_plant",
    "WallSpinResult",
    "zeela_shotblock_clear",
    "zeela_wall_replant",
    "zeela_wall_spin_climb",
    "zeela_shotblock_wall_climb",
    "zeela_warehouse_door_exit",
    "eye_mid_room_approach",
]
