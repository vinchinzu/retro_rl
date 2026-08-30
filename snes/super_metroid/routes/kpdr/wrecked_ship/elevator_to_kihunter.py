"""Red Brinstar elevator room to Crateria Kihunter."""

from __future__ import annotations

from pathlib import Path

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import require_room, wait_ordinary_room
from super_metroid.routes.rle import load_rle_json, play_script
from super_metroid.routes.runtime import ControllerSession

ROOM_ELEVATOR = 0x962A
ROOM_KIHUNTER = 0x948C

_DATA = Path(__file__).resolve().parents[1] / "data"
_ELEVATOR_RLE = load_rle_json(_DATA / "elevator_to_kihunter_human_rle.json")


def play_elevator_to_kihunter(session: ControllerSession) -> SuperMetroidState:
    """Ride up, traverse the connector, and settle in Kihunter."""
    require_room(session, ROOM_ELEVATOR, "elevator_to_kihunter")
    play_script(
        session,
        _ELEVATOR_RLE,
        reason="elevator_to_kihunter_body",
        room_id=ROOM_ELEVATOR,
    )
    return wait_ordinary_room(
        session,
        ROOM_KIHUNTER,
        settle_frames=300,
        label="elevator_to_kihunter",
    )


__all__ = ["ROOM_ELEVATOR", "ROOM_KIHUNTER", "play_elevator_to_kihunter"]
