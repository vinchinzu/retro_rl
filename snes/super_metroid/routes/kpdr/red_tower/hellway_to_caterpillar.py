"""Hellway → Caterpillar reactive return (K5 hop 13).

The movement cadence comes from the human ``speed_to_wave_ice_moat`` tape
hop 29, but is intentionally state-reactive instead of replaying its fragile
1,370-frame open-loop body.  Hold right/run/beam and pulse jump through the
enemy platforms; once the right door seat is reached, land and use the shared
door-exit primitive.
"""

from __future__ import annotations

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import hold, play_run_shoot_exit, require_room
from super_metroid.routes.runtime import ControllerSession

ROOM_HELLWAY = 0xA2F7
ROOM_CATERPILLAR = 0xA322

_DOOR_SEAT_X = 690
_DOOR_SEAT_X_MAX = 800
_TRAVERSE_BUDGET = 2200
_JUMP_PERIOD = 36
_JUMP_HOLD = 24


def _in_right_door_band(x: int) -> bool:
    """True ordinary Hellway right-door band, not a Red Tower slot underflow."""
    return _DOOR_SEAT_X < int(x) < _DOOR_SEAT_X_MAX


def play_hellway_to_caterpillar(session: ControllerSession) -> SuperMetroidState:
    """Cross Hellway right and settle in Caterpillar from the natural K5 pin.

    Public policy: jump the platforms (ceiling-spike d-boosts are optional).
    Do not fall into the floor Samus Eaters.
    https://wiki.supermetroid.run/Hellway

    Dual pins: product tape leave ~(42,153) p29 **2218f** ×2; Ice-climb
    ordinary left-door ~(39,139) p11 **2110f** ×2. Zero-settle the Ice
    pin — idle drops the airborne seat into a plant.
    """
    require_room(session, ROOM_HELLWAY, "hellway_to_caterpillar")

    for frame in range(_TRAVERSE_BUDGET):
        state = session.state
        if int(state.room_id) != ROOM_HELLWAY:
            break
        x = int(state.samus_x)
        if x > 20000:
            # Door-slot fire: room_id flipped, Samus still in the Red Tower
            # slot. RIGHT underflows x to 65522 (>690) and a plant eats the hop.
            hold(session, 1, "RIGHT", reason="hellway_to_caterpillar_door_remap")
            continue
        if _in_right_door_band(x):
            # Human tape reaches this door airborne. Landing makes beam volleys
            # and the crossing deterministic across the two natural dual pins.
            hold(session, 80, reason="hellway_to_caterpillar_door_land")
            return play_run_shoot_exit(
                session,
                from_room=ROOM_HELLWAY,
                to_room=ROOM_CATERPILLAR,
                direction="RIGHT",
                label="hellway_to_caterpillar",
                run_frames=0,
                shoot_frames=24,
                spin_frames=16,
                hold_frames=240,
                settle_frames=200,
            )

        buttons = ["RIGHT", "B", "X"]
        if frame % _JUMP_PERIOD < _JUMP_HOLD:
            buttons.append("A")
        hold(session, 1, *buttons, reason="hellway_to_caterpillar_traverse")

    raise TimeoutError(f"hellway_to_caterpillar: traverse timeout: {session.state}")


__all__ = [
    "ROOM_CATERPILLAR",
    "ROOM_HELLWAY",
    "_in_right_door_band",
    "play_hellway_to_caterpillar",
]
