"""Natural Alpha Power Bomb room collection from its right-hand entry."""

from __future__ import annotations

import json
from pathlib import Path

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import hold, require_room
from super_metroid.routes.rle import play_snes12_frames
from super_metroid.routes.runtime import ControllerSession


ROOM_ALPHA_PB = 0xA3AE
COLLECT_ATTEMPT_FRAMES = 2200


def play_alpha_pb_collect(
    session: ControllerSession,
    *,
    max_frames: int = COLLECT_ATTEMPT_FRAMES,
) -> SuperMetroidState:
    """Cross the room left and wait through the first-PB collection pose."""
    require_room(session, ROOM_ALPHA_PB, "alpha_pb_collect")

    # Tape-backed room body starts from this exact natural right-door seat.
    # Caterpillar itself remains reactive; the Alpha room contains a long,
    # deterministic series of shot-block jumps and the Chozo pickup pause.
    body_path = (
        Path(__file__).resolve().parents[2]
        / "tasks/warehouse_to_red_human_hops/hop_09_Alpha_Power_Bomb_Room.json"
    )
    payload = json.loads(body_path.read_text(encoding="utf-8"))
    play_snes12_frames(
        session,
        payload["frames"],
        reason="alpha_pb_human_body",
        stop_when=lambda state: state.max_power_bombs > 0,
    )
    if session.state.max_power_bombs > 0:
        return session.state

    for frame in range(max_frames):
        state = session.state
        if state.max_power_bombs > 0:
            return state
        if state.samus_x > 360:
            buttons = ["LEFT", "B"]
            if frame % 44 < 30:
                buttons.append("A")
            hold(session, 1, *buttons, reason="alpha_pb_cross_left")
        else:
            hold(session, 1, "LEFT", reason="alpha_pb_collect_plm")

    state = session.state
    raise TimeoutError(
        "alpha_pb_collect: timeout before PB capacity increased: "
        f"room=0x{state.room_id:04X} pose={state.pose} "
        f"xy=({state.samus_x},{state.samus_y}) max_pb={state.max_power_bombs}"
    )


__all__ = ["COLLECT_ATTEMPT_FRAMES", "ROOM_ALPHA_PB", "play_alpha_pb_collect"]
