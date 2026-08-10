"""Bat Room → Red Tower pure return (K5 hop 11).

Source: ``post_ice_below_to_bat_pure`` ~(472, 139) pose 12 right high sill
after Below→Bat dual **485f**. Reverse of ``play_red_tower_to_bat`` bottom
exit (Red RIGHT into Bat left) plus reverse of ``play_bat_to_below_spazer``
platform chain (RIGHT across dry platforms) — LEFT platforms into Red bottom.

Hybrid pure::

  1. Accept Bat right high sill residual (x∈[400,520], y∈[100,180] p12)
  2. Unmorph + beam select + LEFT platform chain (mirror of bat→below RIGHT)
  3. LEFT door into Red Tower bottom ordinary settle (room-id primary)

Tape: ``tasks/speed_to_wave_ice_moat_human.json`` Phase C hop 28→29
(f22367 Bat right sill → f23078 Red bottom ~(19,136) load → y≈2440).
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
)
from super_metroid.routes.kpdr.k5.geometry import (
    BAT_TO_RED_EXIT_HOLD,
    BAT_TO_RED_EXIT_RUN,
    BAT_TO_RED_EXIT_SETTLE,
    BAT_TO_RED_EXIT_SHOOT,
    BAT_TO_RED_EXIT_SPIN,
    BAT_TO_RED_JUMP1,
    BAT_TO_RED_JUMP2,
    BAT_TO_RED_JUMP3,
    BAT_TO_RED_LAND1,
    BAT_TO_RED_LAND2,
    BAT_TO_RED_LAND3,
    BAT_TO_RED_RUNUP1,
    BAT_TO_RED_RUNUP2,
    BAT_TO_RED_RUNUP3,
)
from super_metroid.routes.kpdr.rooms import ROOM_BAT, ROOM_RED_TOWER
from super_metroid.routes.runtime import ControllerSession


def play_bat_to_red(session: ControllerSession) -> SuperMetroidState:
    """Bat left blue door → ordinary Red Tower bottom (reverse of red→bat).

    Expects right high sill after below_to_bat. LEFT platform chain across
    Bat dry platforms into ``0xA253`` bottom — reverse of outbound
    ``play_bat_to_below_spazer`` RIGHT chain and reverse of
    ``play_red_tower_to_bat`` bottom RIGHT exit.
    """
    require_room(session, ROOM_BAT, "bat_to_red")
    unmorph(session)
    select_weapon(session, 0)

    state = session.state
    # Door residual may be mid-air or low; wait to ground on right sill.
    if state.samus_y > 180 or abs(state.velocity_y) > 0:
        for _ in range(60):
            state = hold(session, 1, reason="bat_to_red_land_wait")
            if state.velocity_y == 0 and state.pose in (
                1,
                2,
                9,
                10,
                12,
                25,
                26,
                27,
                28,
                137,
                138,
            ):
                break
        unmorph(session)

    # Right high sill — short glide then LEFT platform chain (mirror bat→below).
    hold(session, 5, reason="bat_to_red_entry_glide")

    # Jump 1: right sill → middle platform (mirror of outbound third jump).
    hold(session, BAT_TO_RED_RUNUP1, "LEFT", "B", reason="bat_to_red_runup1")
    hold(session, BAT_TO_RED_JUMP1, "LEFT", "B", "A", reason="bat_to_red_jump1")
    settle_hold(session, BAT_TO_RED_LAND1, reason="bat_to_red_land1")
    state = session.state
    if not (300 <= state.samus_x <= 420 and state.samus_y <= 200):
        raise TimeoutError(f"bat_to_red: missed middle platform: {state}")

    # Jump 2: middle → first/left-center platform (mirror of outbound second).
    hold(session, BAT_TO_RED_RUNUP2, "LEFT", "B", reason="bat_to_red_runup2")
    hold(session, BAT_TO_RED_JUMP2, "LEFT", "B", "A", reason="bat_to_red_jump2")
    settle_hold(session, BAT_TO_RED_LAND2, reason="bat_to_red_land2")
    state = session.state
    if not (state.samus_x <= 320 and state.samus_y <= 200):
        raise TimeoutError(f"bat_to_red: missed first platform: {state}")

    # Jump 3: first → left ledge / door band (mirror of outbound first long jump).
    # Accept high left sill (y<=155) or low left ledge (~y171–220) like
    # bat_to_below dual-entry heights mirrored.
    hold(session, BAT_TO_RED_RUNUP3, "LEFT", "B", reason="bat_to_red_runup3")
    hold(session, BAT_TO_RED_JUMP3, "LEFT", "B", "A", reason="bat_to_red_jump3")
    settle_hold(session, BAT_TO_RED_LAND3, reason="bat_to_red_land3")
    state = session.state
    if not (state.samus_x <= 240 and state.samus_y <= 240):
        raise TimeoutError(f"bat_to_red: missed left ledge band: {state}")

    # Morph/spin residual after long jump — stand before door exit.
    unmorph(session)
    if session.state.samus_y > 160:
        # Low left ledge: short hop toward high door sill.
        hold(session, 8, "LEFT", "B", reason="bat_to_red_low_runup")
        hold(session, 40, "LEFT", "B", "A", reason="bat_to_red_low_hop")
        settle_hold(session, 30, reason="bat_to_red_low_land")
        unmorph(session)

    return play_run_shoot_exit(
        session,
        from_room=ROOM_BAT,
        to_room=ROOM_RED_TOWER,
        direction="LEFT",
        label="bat_to_red",
        run_frames=BAT_TO_RED_EXIT_RUN,
        shoot_frames=BAT_TO_RED_EXIT_SHOOT,
        spin_frames=BAT_TO_RED_EXIT_SPIN,
        hold_frames=BAT_TO_RED_EXIT_HOLD,
        settle_frames=BAT_TO_RED_EXIT_SETTLE,
    )


__all__ = ["play_bat_to_red"]
