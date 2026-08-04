"""Development-only Moat crossing scaffold.

The K6 Moat crossing is not part of the continuous route yet.  This module
keeps the two candidate approaches bounded and observable while a natural
Speed + Power Bomb source state is captured.  It owns no emulator state and
does not grant movement abilities or alter progression RAM.
"""

from __future__ import annotations

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import require_room
from super_metroid.routes.runtime import ControllerSession, hold


ROOM_MOAT = 0x95FF
ROOM_WEST_OCEAN = 0x93FE

PLATFORM_ATTEMPT_FRAMES = 90
SPARK_ATTEMPT_FRAMES = 100
MAX_CROSSING_ATTEMPTS = 2


def play_moat_cross(session: ControllerSession) -> SuperMetroidState:
    """Attempt the bounded platform-jump and shinespark Moat approaches.

    The caller must provide a natural/dev Moat entry.  The platform and spark
    sequences are deliberately placeholders pending a source with Speed and
    Power Bombs; failure reports the final live state rather than looping.
    """
    require_room(session, ROOM_MOAT, "moat_cross")

    for attempt in range(MAX_CROSSING_ATTEMPTS):
        # Candidate 1: run and spin-jump across the platforms.  The exact
        # setup is intentionally left for the pure-geometry card.
        hold(
            session,
            PLATFORM_ATTEMPT_FRAMES,
            "RIGHT",
            "B",
            "A",
            reason=f"moat_platform_attempt_{attempt + 1}",
        )
        if session.state.room_id == ROOM_WEST_OCEAN:
            return session.state

        # Candidate 2: build speed and reserve the crouch/spark timing for the
        # same source-state-driven geometry pass.
        hold(
            session,
            SPARK_ATTEMPT_FRAMES,
            "RIGHT",
            "B",
            reason=f"moat_spark_attempt_{attempt + 1}",
        )
        if session.state.room_id == ROOM_WEST_OCEAN:
            return session.state

    raise TimeoutError(
        "moat_cross: bounded platform/spark attempts did not reach West Ocean: "
        f"{session.state}"
    )
