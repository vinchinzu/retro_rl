"""Post-Varia return hops (K3 exit → Kraid → … → Business for K4).

First natural door after Varia PLM: Varia Suit Room left blue door back into
Kraid's Room. Continuous tip ends mid-item pose (~81); this hop recovers to
standing, opens the blue door, and spin-pushes left through the transition.
"""

from __future__ import annotations

from super_metroid.policy import StateRequirement
from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import (
    hold,
    require_room,
    require_state,
    select_weapon,
    unmorph,
    wait_ordinary_room,
)
from super_metroid.routes.kpdr.rooms import (
    ITEM_VARIA,
    ROOM_KRAID,
    ROOM_KRAID_EYE,
    ROOM_VARIA,
)
from super_metroid.routes.runtime import ControllerSession

_STANDING = frozenset({1, 2, 5, 6, 9, 10})
_BAD_AIR = frozenset({81, 164, 0, 155})


def _recover_standing(session: ControllerSession, *, timeout: int = 400) -> None:
    """Leave post-item / mid-air lock poses and land on the Varia floor."""
    unmorph(session)
    for index in range(timeout):
        st = session.state
        if (
            st.pose in _STANDING
            and st.velocity_y == 0
            and st.samus_y >= 130
        ):
            return
        phase = index % 24
        if st.pose in _BAD_AIR or st.pose in (137, 138):
            if phase < 4:
                hold(session, 1, "A", reason="varia_recover_jump")
            elif phase < 12:
                # Nudge right so we do not pin against the left door while locked.
                hold(session, 1, "RIGHT", reason="varia_recover_nudge")
            else:
                hold(session, 1, reason="varia_recover_idle")
            continue
        if phase < 3:
            hold(session, 1, "A", reason="varia_recover_jump")
        elif phase < 10:
            hold(session, 1, "RIGHT", reason="varia_recover_nudge")
        else:
            hold(session, 1, reason="varia_recover_idle")
    raise TimeoutError(
        f"varia_to_kraid: recover standing failed: {session.state}"
    )


def play_varia_to_kraid(session: ControllerSession) -> SuperMetroidState:
    """Exit Varia Suit Room left into Kraid's Room (post-collect return).

    Expects ordinary gameplay in ``0xA6E2`` with Varia already collected
    (continuous tip endpoint or pure post-collect state). Controller-dev until
    composed on continuous ``--to varia`` prefix as a RouteHop.
    """
    require_state(
        session,
        StateRequirement(
            room_id=ROOM_VARIA,
            game_states=frozenset({8}),
            collected_items_mask=ITEM_VARIA,
        ),
        "varia_to_kraid",
    )
    _recover_standing(session)

    # Stay mid-room so door shots clear the left shell without pin-knockback.
    for _ in range(50):
        x = session.state.samus_x
        if 95 <= x <= 130:
            break
        hold(
            session,
            1,
            "RIGHT" if x < 95 else "LEFT",
            reason="varia_center",
        )
    hold(session, 8, reason="varia_center_settle")

    try:
        select_weapon(session, 0)
    except RuntimeError:
        pass

    # Face left, open the blue door with standing beam shots.
    hold(session, 8, "LEFT", reason="varia_face_left")
    hold(session, 6, reason="varia_face_release")
    for _ in range(4):
        hold(session, 4, "X", reason="varia_door_shot")
        hold(session, 18, reason="varia_door_fuse")

    # Spin-push through. Walk-only often pin-locks pose 138 on the door lip.
    for _ in range(360):
        state = hold(session, 1, "LEFT", "B", "A", reason="varia_exit_spin")
        if state.room_id == ROOM_KRAID:
            break
        # Knockback against the lip: brief recover then resume spin.
        if state.pose in (137, 138) and state.samus_x <= 60:
            hold(session, 4, reason="varia_lip_release")
            hold(session, 3, "RIGHT", reason="varia_lip_backoff")
            hold(session, 4, "X", reason="varia_lip_reshot")
            hold(session, 12, reason="varia_lip_fuse")
    else:
        raise TimeoutError(
            f"varia_to_kraid: did not reach 0x{ROOM_KRAID:04X}: {session.state}"
        )

    # Varia left door lands on the right side of Kraid's Room (~x 450+).
    return wait_ordinary_room(
        session,
        ROOM_KRAID,
        settle_frames=240,
        label="varia_to_kraid",
        x_range=(350, 560),
        y_range=(300, 450),
        min_settle_frame=12,
    )


def play_kraid_to_eye_return(session: ControllerSession) -> SuperMetroidState:
    """Attempt the left-door return from Kraid's Room to the Eye Door Room.

    This is a ``controller_dev`` scaffold for the post-boss reverse hop, not
    continuous evidence. It deliberately has a bounded traversal so geometry
    tuning can fail loudly without taking ownership of the emulator or route.
    """
    require_state(
        session,
        StateRequirement(room_id=ROOM_KRAID, game_states=frozenset({8})),
        "kraid_to_eye_return",
    )
    # Keep both the explicit room guard and the ordinary-gameplay requirement
    # visible at this boundary for pure probes and future entry tightening.
    require_room(session, ROOM_KRAID, "kraid_to_eye_return")
    select_weapon(session, 0)
    # Stage just right of the lip, then use one fixed short hop to approach at
    # the higher Y band before settling for the standing door sequence.
    for _ in range(120):
        state = hold(session, 1, "LEFT", reason="kraid_return_approach")
        if state.samus_x <= 220:
            break
    else:
        raise TimeoutError(
            f"kraid_to_eye_return: left door approach timed out: {session.state}"
        )
    hold(session, 24, "LEFT", "A", reason="kraid_return_short_hop")
    hold(session, 20, reason="kraid_return_approach_settle")

    # Spin-pushing pins Samus in pose 138 on this door lip. Back off, stand,
    # and open the blue door with the same standing beam-shot pattern as the
    # Varia-to-Kraid reverse door.
    hold(session, 10, "RIGHT", reason="kraid_return_lip_backoff")
    unmorph(session)
    hold(session, 8, "LEFT", reason="kraid_return_face_left")
    hold(session, 6, reason="kraid_return_release")
    for _ in range(4):
        hold(session, 4, "LEFT", "X", reason="kraid_return_door_shot")
        hold(session, 18, reason="kraid_return_door_fuse")

    # Spin-push through. Walk-only / pinned spin often locks pose 138 on the lip
    # (same pattern as varia_to_kraid left exit).
    for _ in range(720):
        state = hold(
            session,
            1,
            "LEFT",
            "B",
            "A",
            reason="kraid_return_exit",
        )
        if state.room_id == ROOM_KRAID_EYE:
            break
        if state.pose in (137, 138) and state.samus_x <= 80:
            hold(session, 4, reason="kraid_return_lip_release")
            hold(session, 4, "RIGHT", reason="kraid_return_lip_backoff")
            hold(session, 4, "LEFT", reason="kraid_return_reface")
            hold(session, 4, "LEFT", "X", reason="kraid_return_lip_reshot")
            hold(session, 14, reason="kraid_return_lip_fuse")
    else:
        raise TimeoutError(
            f"kraid_to_eye_return: left eye-door exit timed out: {session.state}"
        )
    return wait_ordinary_room(
        session,
        ROOM_KRAID_EYE,
        settle_frames=340,
        label="kraid_to_eye_return",
        x_range=(300, 560),
        y_range=(300, 450),
        min_settle_frame=12,
    )
