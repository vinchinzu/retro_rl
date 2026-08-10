"""Below Spazer → Bat Room pure return (K5 hop 9).

Source: ``post_ice_west_to_below_pure`` ~(472, 393) pose 82 right-floor residual
after West→Below dual **272f**. Reverse of ``play_bat_to_below_spazer`` door
exit (Bat RIGHT-run into Below left) — floor path LEFT across Below Spazer
into Bat; Spazer already held on K5 stack (beams ``0x1007``).

Hybrid pure::

  1. Accept Below right-floor residual (x∈[400,520], y∈[350,420] p82)
  2. Unmorph + beam select + LEFT floor runner (mirror of below floor→west)
  3. Ordinary Bat settle (room-id primary; right-side residual for Bat→Red)

Tape: ``tasks/speed_to_wave_ice_moat_human.json`` Phase C hop 27→28
(f21858 Below → f22367 Bat ~(20,395)). Pure pin is right-floor after reverse
entry from West (~x=472 y=393); same LEFT door band as outbound Bat entry.
"""

from __future__ import annotations

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import (
    hold,
    require_room,
    select_weapon,
    unmorph,
    wait_ordinary_room,
)
from super_metroid.routes.kpdr.k5.geometry import (
    BELOW_TO_BAT_FRAMES,
    BELOW_TO_BAT_SETTLE,
)
from super_metroid.routes.kpdr.rooms import ROOM_BAT, ROOM_BELOW_SPAZER
from super_metroid.routes.runtime import ControllerSession


def play_below_to_bat(session: ControllerSession) -> SuperMetroidState:
    """Below left blue door → ordinary Bat Room (reverse of bat→below door).

    Expects right-floor Below handoff after west_to_below. LEFT floor/water
    runner into ``0xA3DD`` — mirror of outbound ``play_below_spazer_floor_to_west``
    (RIGHT) and reverse of ``play_bat_to_below_spazer`` RIGHT door exit.
    """
    require_room(session, ROOM_BELOW_SPAZER, "below_to_bat")
    # Door-exit residual may still be in a running pose; brief glide then
    # unmorph so the shared helper does not turn a pose-9/10 into a jump.
    hold(session, 6, reason="below_to_bat_entry_glide")
    unmorph(session)
    select_weapon(session, 0)
    state = session.state
    for frame in range(BELOW_TO_BAT_FRAMES):
        # Mirror floor_to_west: alternate shoot bursts with spin-run through
        # water / Cacatac / Yapping Maw band on the floor path to Bat.
        buttons = ("LEFT", "B", "X") if frame % 35 < 10 else ("LEFT", "B", "A")
        state = hold(session, 1, *buttons, reason="below_to_bat_left")
        if state.room_id == ROOM_BAT:
            break
    else:
        raise TimeoutError(
            f"below_to_bat: Bat Room not reached: {state}"
        )
    return wait_ordinary_room(
        session,
        ROOM_BAT,
        settle_frames=BELOW_TO_BAT_SETTLE,
        label="below_to_bat",
    )


__all__ = ["play_below_to_bat"]
