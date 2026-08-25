"""Red Tower → Hellway pure return (K5 hop 12).

Source: ``post_ice_bat_to_red_pure`` ~(206–216, 2443) after Bat→Red dual
**718f**. Climb reverse of ``play_red_tower_to_bat`` descent bands, then RIGHT
into top-right Hellway door ``0xA2F7``.

Hybrid pure (Hi-Jump held on K5 stack)::

  1. Accept Red bottom residual; clear Bat door lip (never RIGHT into 0xA3DD)
  2. Morph + double-bomb IBJ 18/30 centered x≈150 — dual tunnel peak ~y1820
     (do **not** climb_lower first — desyncs IBJ)
  3. Tunnel seat → midplat hop → midplat IBJ dual temporary floor ~y1606
  4. Human ascent RLE first 850f from floor → dual past temp floor ~(122,1459)
     p81 (mid-air peak — not solid; do not force-unmorph)
  5. Spin-left seat ~(37,1499) → alternating period WJ phases dual ~y420
  6. Ice-freeze ripper ladder (morph hop) → top door → RIGHT Hellway

Tape: ``tasks/speed_to_wave_ice_moat_human.json`` f23078–29947 (~6869f Red).
Human mid "platforms" y2255/2159/2023 are **frozen rippers** (Ice held) —
not solid tiles. Temp floor is bombable from above (outbound); climb arrives
on/under lip via IBJ then uses human-matched open-loop + period WJ upper.
"""

from __future__ import annotations

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import (
    hold,
    play_run_shoot_exit,
    require_room,
    select_weapon,
    settle_hold,
    unmorph,
    wait_ordinary_room,
)
from super_metroid.routes.kpdr.k5.geometry import (
    RED_TO_HELLWAY_EXIT_HOLD,
    RED_TO_HELLWAY_EXIT_RUN,
    RED_TO_HELLWAY_EXIT_SETTLE,
    RED_TO_HELLWAY_EXIT_SHOOT,
    RED_TO_HELLWAY_EXIT_SPIN,
    RED_TOP_DOOR_X,
    RED_TOP_DOOR_Y,
)
from super_metroid.routes.kpdr.k5.red_to_hellway_common import (
    _DATA,
    _in_hellway,
    _in_red,
    _unmorph,
)
from super_metroid.routes.kpdr.k5.red_to_hellway_mid import _climb_mid
from super_metroid.routes.kpdr.k5.red_to_hellway_upper import _climb_upper
from super_metroid.routes.kpdr.rooms import ROOM_HELLWAY, ROOM_RED_TOWER
from super_metroid.routes.runtime import ControllerSession


def play_red_to_hellway(session: ControllerSession) -> SuperMetroidState:
    """Red Tower bottom → ordinary Hellway left (K5 hop 12).

    Ice-pin spine: checkpoint climb to ordinary left-door (gs=8, x≤80).
    Do not extra-settle — idle drops the airborne p11 seat into a plant.
    Tape body remains the fallback when the Ice+HJ floor seat is absent.
    """
    label = "red_to_hellway"
    require_room(session, ROOM_RED_TOWER, label)
    from super_metroid.routes.kpdr.k5.red_ice_climb import can_attach_bottom_edge
    from super_metroid.routes.kpdr.k5.red_ice_to_hellway import play_ice_climb_to_hellway
    from super_metroid.routes.rle import load_rle_json, play_script

    if can_attach_bottom_edge(session.state):
        return play_ice_climb_to_hellway(session)

    body_path = _DATA / "red_to_hellway_human_hop.json"
    play_script(
        session,
        load_rle_json(body_path),
        reason=label,
        room_id=ROOM_RED_TOWER,
        stop_when=lambda s: int(s.room_id) != ROOM_RED_TOWER,
    )
    if _in_hellway(session.state):
        return wait_ordinary_room(
            session, ROOM_HELLWAY, settle_frames=RED_TO_HELLWAY_EXIT_SETTLE, label=label
        )
    if int(session.state.room_id) != ROOM_RED_TOWER:
        st = session.state
        raise TimeoutError(
            f"{label}: hop body left Red to 0x{int(st.room_id):04X} "
            f"xy=({st.samus_x},{st.samus_y}) p={st.pose}"
        )
    unmorph(session)
    select_weapon(session, 0)
    hold(session, 6, reason=f"{label}_entry_glide")

    # Clear Bat door lip before any RIGHT spin.
    for _ in range(100):
        st = session.state
        if not _in_red(st):
            break
        if int(st.samus_x) <= 165 and int(st.velocity_y) == 0:
            break
        hold(session, 1, "LEFT", "B", reason=f"{label}_clear_bat")
    settle_hold(session, 6, reason=f"{label}_bottom_settle")

    # Mid IBJ 18/30 c150 is dual-stable from the pure bottom pin itself.
    # Running climb_lower first desyncs enemy/block state and kills the IBJ
    # climb (probe: bottom→peak y1820 dual; post-lower→IBJ stalls ~y1977).
    # Keep lower as recovery inside _climb_mid only.
    _climb_mid(session, f"{label}_mid")
    if _in_hellway(session.state):
        return wait_ordinary_room(
            session, ROOM_HELLWAY, settle_frames=RED_TO_HELLWAY_EXIT_SETTLE, label=label
        )

    _climb_upper(session, f"{label}_upper")
    if _in_hellway(session.state):
        return wait_ordinary_room(
            session, ROOM_HELLWAY, settle_frames=RED_TO_HELLWAY_EXIT_SETTLE, label=label
        )
    if not _in_red(session.state):
        raise TimeoutError(f"{label}: left Red unexpectedly: {session.state}")

    # True morph only before exit — never force-unmorph pose 81/82 residual.
    if int(session.state.pose) in (29, 30, 31, 32):
        _unmorph(session, label)
    # Only reclimb when already in the upper door band (avoid full RLE thrash).
    if RED_TOP_DOOR_Y < int(session.state.samus_y) <= RED_TOP_DOOR_Y + 100:
        _climb_upper(session, f"{label}_reclimb")
        if _in_hellway(session.state):
            return wait_ordinary_room(
                session,
                ROOM_HELLWAY,
                settle_frames=RED_TO_HELLWAY_EXIT_SETTLE,
                label=label,
            )

    # Exit only from the top door band — mid-shaft RIGHT thrash desyncs residual.
    if int(session.state.samus_y) > RED_TOP_DOOR_Y + 120:
        st = session.state
        raise TimeoutError(
            f"{label}: upper residual room=0x{int(st.room_id):04X} "
            f"xy=({st.samus_x},{st.samus_y}) p={st.pose} "
            f"(need y≤{RED_TOP_DOOR_Y + 120} for Hellway exit)"
        )

    for _ in range(100):
        st = session.state
        if _in_hellway(st) or not _in_red(st):
            break
        if int(st.samus_y) > RED_TOP_DOOR_Y + 50:
            hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_top_hop")
            continue
        if int(st.samus_x) < RED_TOP_DOOR_X - 30:
            hold(session, 1, "RIGHT", "B", reason=f"{label}_top_run")
            continue
        break

    if _in_hellway(session.state):
        return wait_ordinary_room(
            session, ROOM_HELLWAY, settle_frames=RED_TO_HELLWAY_EXIT_SETTLE, label=label
        )

    return play_run_shoot_exit(
        session,
        from_room=ROOM_RED_TOWER,
        to_room=ROOM_HELLWAY,
        direction="RIGHT",
        label=label,
        run_frames=RED_TO_HELLWAY_EXIT_RUN,
        shoot_frames=RED_TO_HELLWAY_EXIT_SHOOT,
        spin_frames=RED_TO_HELLWAY_EXIT_SPIN,
        hold_frames=RED_TO_HELLWAY_EXIT_HOLD,
        settle_frames=RED_TO_HELLWAY_EXIT_SETTLE,
    )


__all__ = ["play_red_to_hellway"]
