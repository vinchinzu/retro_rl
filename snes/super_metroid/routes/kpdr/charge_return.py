"""Big Pink Charge Beam collect + conventional return (no IBJ).

Charge lives in Big Pink ``0x9D19`` at the bottom Chozo (sm-json node 12).
KPDR K1 path (post-Spore Supers): main shaft → mass → Charge Chozo → ordinary
jump climb → GHZ green door.

Human source: ``tasks/charge_human.json`` (2026-08-04):

* Mass-left missile pack (~565,1659) when capacity allows.
* **R-angle arm cannon** opens the Chozo faster than flat beam.
* Simple RIGHT+A staircase return to tunnel lip (y≲1600), not IBJ.
"""

from __future__ import annotations

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import (
    ensure_morph,
    hold,
    require_room,
    select_weapon,
    unmorph,
)
from super_metroid.routes.runtime import ControllerSession

# Charge is a sub-area of Big Pink (same room id).
ROOM_CHARGE = 0x9D19
ROOM_BIG_PINK = ROOM_CHARGE

# collected_beams bit for Charge Beam.
CHARGE_BEAM_MASK = 0x1000

# Item-grab fanfare pose (missile pack + Charge).
_POSE_ITEM_FANFARE = 138

# Mass band after return (soft).
_MASS_Y_MAX = 1685
# Tunnel lip for GHZ exit (human min_y after climb ≈ 1587).
_LIP_Y_MAX = 1605


def _wait_item_fanfare(session: ControllerSession, *, reason: str) -> SuperMetroidState:
    """Hold through pose-138 item fanfare if active.

    Caps wait so a stuck pose-138 without a real collect cannot burn 700f.
    """
    if session.state.pose != _POSE_ITEM_FANFARE:
        return session.state
    for _ in range(240):
        state = hold(session, 1, reason=reason)
        if state.pose != _POSE_ITEM_FANFARE:
            hold(session, 12, reason=f"{reason}_settle")
            return session.state
    # Nudge free of a stuck fanfare pose (PLM already dead / capacity full).
    hold(session, 8, "RIGHT", reason=f"{reason}_unstick")
    hold(session, 12, reason=f"{reason}_unstick_settle")
    return session.state


def _descend_main_to_mass(session: ControllerSession) -> SuperMetroidState:
    """Main-shaft morph ball → lower-left shelf → mass face setup.

    Proven pure path to the bomb-hole. Optionally brushes the mass-left missile
    pack (human ``charge_human`` f255 @ 565,1659) when the PLM is still live.
    """
    require_room(session, ROOM_BIG_PINK, "charge_to_mass")
    missiles_before = int(session.state.missiles)
    max_before = int(session.state.max_missiles)
    ensure_morph(session)

    for _ in range(500):
        state = hold(session, 1, "LEFT", reason="charge_lower_left")
        if state.samus_x <= 560 and state.samus_y >= 1540:
            break
    else:
        raise TimeoutError(f"charge_to_mass: missed lower-left shelf: {session.state}")

    unmorph(session)
    for _ in range(220):
        state = hold(session, 1, "RIGHT", "B", "A", reason="charge_lower_drop")
        if state.samus_x >= 665 and state.samus_y >= 1660:
            break
    else:
        raise TimeoutError(f"charge_to_mass: missed lower mass: {session.state}")

    # Brush mass-left missile pack (human f255 @ 565,1659 → +5 when PLM live).
    # Skip when capacity already full (dev main-shaft fixtures often maxed).
    if session.state.max_missiles <= max_before and session.state.samus_x > 560:
        for _ in range(50):
            state = hold(session, 1, "LEFT", reason="charge_missile_walk")
            if state.missiles > missiles_before or state.max_missiles > max_before:
                break
            if state.samus_x <= 555:
                break
        if (
            session.state.missiles > missiles_before
            or session.state.max_missiles > max_before
        ):
            _wait_item_fanfare(session, reason="charge_missile_fanfare")
        for _ in range(80):
            state = hold(session, 1, "RIGHT", "B", reason="charge_missile_return")
            if state.samus_x >= 680:
                break

    hold(session, 30, "RIGHT", "B", reason="charge_mass_run")
    hold(session, 10, reason="charge_mass_settle")
    hold(session, 12, "LEFT", reason="charge_mass_brake")
    hold(session, 8, "A", reason="charge_mass_vertical")
    return session.state


def _bomb_drop_to_charge_floor(session: ControllerSession) -> SuperMetroidState:
    """From mass face, morph-bomb the right floor hole down to charge depth."""
    ensure_morph(session)
    for i in range(400):
        if i % 25 == 0:
            hold(session, 2, "X", reason="charge_drop_bomb")
            hold(session, 40, reason="charge_drop_fuse")
        direction = "LEFT" if (i // 50) % 2 == 0 else "RIGHT"
        state = hold(session, 1, direction, reason="charge_drop_roll")
        if state.samus_y >= 1850:
            return state
    raise TimeoutError(f"charge_drop: never reached charge depth: {session.state}")


def _approach_chozo_platform(session: ControllerSession) -> SuperMetroidState:
    """From post-drop (~x714/y1852), reach the Chozo platform (~x630/y1915)."""
    unmorph(session)
    hold(session, 30, reason="charge_drop_land")

    # Back right for hop runway on the charge floor.
    for _ in range(50):
        state = hold(session, 1, "RIGHT", reason="charge_runup")
        if state.samus_x >= 690:
            break
    hold(session, 15, reason="charge_runup_settle")

    for _cycle in range(8):
        hold(session, 6, "A", reason="charge_platform_hop")
        for _ in range(30):
            state = hold(session, 1, "LEFT", reason="charge_platform_drift")
            if state.collected_beams & CHARGE_BEAM_MASK:
                return state
            if (
                state.pose in (1, 2, 9, 10)
                and state.samus_y <= 1920
                and state.samus_x <= 640
            ):
                hold(session, 25, reason="charge_platform_settle")
                return session.state
        hold(session, 15, reason="charge_platform_land")
    raise TimeoutError(
        f"charge_platform: never reached Chozo ledge: {session.state}"
    )


def _shoot_and_collect_charge(session: ControllerSession) -> SuperMetroidState:
    """R-angle arm-cannon open + walk into Charge PLM.

    Human: on pedestal hold **R** (arm cannon angle, pose 6) + **X** to crack
    the statue quickly, then touch the orb. Falls back to flat LEFT+X / hop.
    Success requires ``collected_beams & CHARGE`` — pose-138 alone is not enough
    (fanfare can stick without a beam bit).
    """
    if session.state.collected_beams & CHARGE_BEAM_MASK:
        return _wait_item_fanfare(session, reason="charge_item_fanfare")

    hold(session, 40, reason="charge_platform_settle")

    # Slight back-right runway then face statue (left).
    for _ in range(25):
        hold(session, 1, "RIGHT", reason="charge_spin_back")
    hold(session, 8, reason="charge_spin_settle")
    hold(session, 6, "LEFT", reason="charge_face")
    hold(session, 4, reason="charge_face_release")

    # Human technique: R-angle beam (pose 6) while shooting.
    hold(session, 8, "R", reason="charge_angle_hold")
    for _ in range(8):
        hold(session, 8, "X", "R", reason="charge_chozo_angle_shot")
        hold(session, 14, "R", reason="charge_chozo_angle_wait")
        if session.state.collected_beams & CHARGE_BEAM_MASK:
            return _wait_item_fanfare(session, reason="charge_item_fanfare")

    try:
        select_weapon(session, 0)
    except RuntimeError:
        pass

    for _ in range(5):
        hold(session, 2, "LEFT", "X", reason="charge_chozo_shot")
        hold(session, 18, reason="charge_chozo_shot_wait")
        if session.state.collected_beams & CHARGE_BEAM_MASK:
            return _wait_item_fanfare(session, reason="charge_item_fanfare")

    for i in range(100):
        if session.state.collected_beams & CHARGE_BEAM_MASK:
            break
        if i % 20 < 6:
            hold(session, 1, "LEFT", "A", reason="charge_collect_hop")
        else:
            hold(session, 1, "LEFT", reason="charge_collect_walk")
    else:
        raise TimeoutError(
            f"charge_collect: Charge Beam PLM not collected: {session.state}"
        )

    return _wait_item_fanfare(session, reason="charge_item_fanfare")


def play_charge_beam_collect(session: ControllerSession) -> SuperMetroidState:
    """Big Pink main shaft → Charge Beam Chozo collect.

    Expects ordinary Big Pink near the main-shaft handoff (x≲750). Ends with
    Charge Beam collected after fanfare, still in Big Pink near the Chozo.

    Technique: proven pure drop geometry + human **R-angle** Chozo open
    (``tasks/charge_human.json``).
    """
    require_room(session, ROOM_BIG_PINK, "charge_beam_collect")
    if session.state.collected_beams & CHARGE_BEAM_MASK:
        return session.state

    _descend_main_to_mass(session)
    _bomb_drop_to_charge_floor(session)
    _approach_chozo_platform(session)
    return _shoot_and_collect_charge(session)


def _grounded_mass_land(state: SuperMetroidState, *, y_max: int) -> bool:
    """True when standing on a mass-band ledge (not mid-air peak).

    Early return used to fire on airborne y-peaks (vy≠0) and immediately fall
    back into the Charge pit on the next hop.
    """
    if state.samus_y > y_max or state.samus_x < 690:
        return False
    if abs(int(state.velocity_y)) > 1:
        return False
    # Standing / crouch / land poses — not spin (25) or knockback (137/138).
    return int(state.pose) in (1, 2, 9, 10, 39, 40, 41, 42, 81, 101, 129, 130)


def play_charge_beam_return(session: ControllerSession) -> SuperMetroidState:
    """Post-Charge Chozo → grounded mass ledge via ordinary up-right jumps.

    Expects Charge already collected in Big Pink. Climbs the right-side drop
    shaft with simple RIGHT+A staircase (human f1706–1920: x≈700–747 →
    y1955→~1739 grounded). GHZ lip climb lives in ``play_big_pink_to_ghz``.
    """
    require_room(session, ROOM_BIG_PINK, "charge_beam_return")
    if not (session.state.collected_beams & CHARGE_BEAM_MASK):
        raise RuntimeError(
            f"charge_beam_return: Charge not collected "
            f"(beams=0x{session.state.collected_beams:04X})"
        )

    if _grounded_mass_land(session.state, y_max=_MASS_Y_MAX):
        hold(session, 12, reason="charge_return_mass_settle")
        return session.state

    unmorph(session)
    hold(session, 15, reason="charge_return_stand")
    _wait_item_fanfare(session, reason="charge_return_fanfare_clear")

    # Seat under the open shaft (human climb column ≈ x715).
    for _ in range(100):
        state = hold(session, 1, "RIGHT", "B", reason="charge_return_to_shaft")
        if state.samus_x >= 710:
            break
    hold(session, 12, reason="charge_return_shaft_settle")

    # Ordinary run-jump climb. Require grounded land in mass band (not air peak).
    # Human rests at ~747,1739–1755 before the LEFT+A shelf hop.
    best_y = session.state.samus_y
    soft_mass = 1760
    for _cycle in range(50):
        state = session.state
        if _grounded_mass_land(state, y_max=_MASS_Y_MAX):
            hold(session, 16, reason="charge_return_mass_settle")
            return session.state

        hold(session, 8, "RIGHT", "B", reason="charge_return_runup")
        hold(session, 1, "RIGHT", "A", reason="charge_return_jump")
        for _ in range(32):
            state = hold(session, 1, "RIGHT", "A", reason="charge_return_air")
            if state.samus_y < best_y - 2:
                best_y = state.samus_y
        # Land window — only accept grounded mass / soft ledge.
        for _ in range(28):
            state = hold(session, 1, reason="charge_return_land")
            if _grounded_mass_land(state, y_max=soft_mass):
                hold(session, 16, reason="charge_return_mass_settle")
                return session.state

    raise TimeoutError(
        f"charge_beam_return: stuck best_y={best_y}: {session.state}"
    )


def play_big_pink_charge_detour(session: ControllerSession) -> SuperMetroidState:
    """Main-shaft → Charge collect → mass/lip return (skip if already held).

    Used by :func:`play_big_pink_to_ghz` so the continuous K1 path picks up
    Charge on the way to GHZ without a separate tip.
    """
    require_room(session, ROOM_BIG_PINK, "charge_detour")
    if session.state.collected_beams & CHARGE_BEAM_MASK:
        if session.state.samus_y > 1600 and session.state.samus_x < 800:
            return session.state
        _descend_main_to_mass(session)
        return session.state

    play_charge_beam_collect(session)
    return play_charge_beam_return(session)


__all__ = [
    "CHARGE_BEAM_MASK",
    "ROOM_BIG_PINK",
    "ROOM_CHARGE",
    "play_big_pink_charge_detour",
    "play_charge_beam_collect",
    "play_charge_beam_return",
]
