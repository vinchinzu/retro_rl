"""Caterpillar return climb to the Red Brinstar elevator room."""

from __future__ import annotations

from pathlib import Path

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import require_room, wait_ordinary_room
from super_metroid.routes.rle import load_rle_json, play_script
from super_metroid.routes.runtime import ControllerSession

ROOM_CATERPILLAR = 0xA322
ROOM_ELEVATOR = 0x962A

_DATA = Path(__file__).resolve().parents[1] / "data"
_CLIMB_RLE = load_rle_json(_DATA / "caterpillar_to_elevator_human_rle.json")


def play_caterpillar_to_elevator(session: ControllerSession) -> SuperMetroidState:
    """Replay the dual-green climb from the natural Alpha PB return seat."""
    require_room(session, ROOM_CATERPILLAR, "caterpillar_to_elevator")
    play_script(
        session,
        _CLIMB_RLE,
        reason="caterpillar_to_elevator_climb",
        room_id=ROOM_CATERPILLAR,
        stop_when=lambda state: int(state.room_id) != ROOM_CATERPILLAR,
    )
    if int(session.state.room_id) != ROOM_ELEVATOR:
        raise TimeoutError(
            "caterpillar_to_elevator: body did not reach elevator: "
            f"{session.state}"
        )
    return wait_ordinary_room(
        session,
        ROOM_ELEVATOR,
        settle_frames=260,
        label="caterpillar_to_elevator",
    )


__all__ = ["ROOM_CATERPILLAR", "ROOM_ELEVATOR", "play_caterpillar_to_elevator"]
