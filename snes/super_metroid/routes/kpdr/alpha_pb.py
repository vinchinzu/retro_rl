"""Development-only scaffold for the KPDR Alpha Power Bomb room.

The K5 Alpha PB route is not pure-green or continuous evidence yet.  This
module provides a bounded collection attempt from a natural room entry while
the Ice/Speed source state and item-room geometry are captured.
"""

from __future__ import annotations

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import hold, require_room
from super_metroid.routes.runtime import ControllerSession


ROOM_ALPHA_PB = 0xA3AE
COLLECT_ATTEMPT_FRAMES = 240


def play_alpha_pb_collect(
    session: ControllerSession,
    *,
    max_frames: int = COLLECT_ATTEMPT_FRAMES,
) -> SuperMetroidState:
    """Attempt to collect Alpha PB, then fail with a bounded timeout.

    The caller must provide an Alpha PB room entry.  Movement geometry is a
    placeholder: hold right and shoot so a future pure pass can replace this
    one bounded slice without changing the session contract.
    """
    require_room(session, ROOM_ALPHA_PB, "alpha_pb_collect")

    for _ in range(max_frames):
        state = hold(session, 1, "RIGHT", "B", reason="alpha_pb_collect_scaffold")
        if state.max_power_bombs > 0:
            return state

    state = session.state
    raise TimeoutError(
        "alpha_pb_collect: scaffold timeout before PB capacity increased: "
        f"room=0x{state.room_id:04X} pose={state.pose} "
        f"xy=({state.samus_x},{state.samus_y}) max_pb={state.max_power_bombs}"
    )


__all__ = ["COLLECT_ATTEMPT_FRAMES", "ROOM_ALPHA_PB", "play_alpha_pb_collect"]
