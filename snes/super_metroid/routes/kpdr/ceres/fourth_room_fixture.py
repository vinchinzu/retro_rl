"""Fourth Ceres room (Scientist → Ridley) room-gated fixture.

This fixture represents the room-gated arm-pump from Scientist 0xE021 → Ridley
0xE0B5 using the product helper `_ceres_arm_pump_until` from arm_pump.py.

Unlike rooms 1-3 which have fixed tape extracted from
`_ceres_outbound_to_scientist_spans`, this segment is **room-gated** with
variable length. Product code (`routes.kpdr.ceres.outbound.play_ceres_to_ridley_door`
lines 91-99) uses `_ceres_arm_pump_until` with `done=lambda s: s.room_id == ROOM_CERES_RIDLEY`.

Flat 0xE06B is passed through but not a stop point in the product helper.
The done condition checks for Ridley 0xE0B5.

Implementation:
- Calls `_ceres_arm_pump_until` with product parameters
- Direction: RIGHT, max_frames: 900, stuck_jump_after: 40
- Reason: "ceres_out_flat_band" (same as product)
- Done: room_id == ROOM_CERES_RIDLEY (same as product)
- Variable frame count depending on entry conditions

Policy:
- Predictor (StubPredictor / sm_rev_predict) = search speed only
- Emulator (stable-retro / SMEDIT snes9x) = ground truth
- Room-clear claims require emulator validation

Validation:
- Start state: env var `SM_CERES_SCIENTIST_STATE` (path to .state file on disk)
- Tests skip without ROM_AVAILABLE or missing start state
- Never commit .state or ROM blobs to repo

Note:
    This is NOT a fixed tape. It invokes the product helper `_ceres_arm_pump_until`.
    Frame count is variable and determined by room-gated adaptive behavior.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from super_metroid.routes.kpdr.ceres.arm_pump import _ceres_arm_pump_until
from super_metroid.routes.kpdr.room_ids import (
    ROOM_CERES_RIDLEY,
    ROOM_CERES_SCIENTIST,
)
from super_metroid.routes.runtime import RouteSession

__all__ = [
    "CeresFourthRoomFixture",
    "play_ceres_fourth_room",
    "validate_ceres_fourth_room_emulator",
]


@dataclass(frozen=True)
class CeresFourthRoomFixture:
    """Fourth Ceres room fixture (room-gated, variable length).

    Represents Scientist 0xE021 → Ridley 0xE0B5 using product helper
    `_ceres_arm_pump_until`. Frame count is variable; done condition checks
    for ROOM_CERES_RIDLEY.

    Emulator validation uses env var SM_CERES_SCIENTIST_STATE for start state.
    """

    from_room_id: int
    to_room_id: int
    frames_consumed: int
    helper_source: str
    emulator_validated: bool = False
    emulator_success: bool = False
    emulator_final_room: int | None = None
    emulator_final_x: int | None = None
    emulator_final_y: int | None = None

    @property
    def room_clear(self) -> bool:
        """True if emulator validation confirmed room clear (ground truth).

        Never claim room-clear without emulator validation.
        """
        return self.emulator_validated and self.emulator_success


def play_ceres_fourth_room(session: RouteSession) -> CeresFourthRoomFixture:
    """Play Ceres Scientist → Ridley using product helper (room-gated).

    Invokes `_ceres_arm_pump_until` with the same parameters as
    `routes.kpdr.ceres.outbound.play_ceres_to_ridley_door` (lines 92-99):
    - Direction: RIGHT
    - Max frames: 900
    - Done: room_id == ROOM_CERES_RIDLEY
    - Stuck jump after: 40
    - Reason: "ceres_out_flat_band"

    Args:
        session: RouteSession starting at Scientist 0xE021

    Returns:
        CeresFourthRoomFixture with frames_consumed from helper

    Raises:
        TimeoutError: If Ridley room not reached within max_frames

    Note:
        This is NOT a fixed tape. Frame count is variable depending on entry
        conditions. The helper is adaptive and room-gated.
    """
    if session.state.room_id != ROOM_CERES_SCIENTIST:
        raise ValueError(
            f"Expected to start at Scientist 0x{ROOM_CERES_SCIENTIST:X}, "
            f"got room 0x{session.state.room_id:X}"
        )

    # Product helper call (same as play_ceres_to_ridley_door)
    frames = _ceres_arm_pump_until(
        session,
        "RIGHT",
        reason="ceres_out_flat_band",
        max_frames=900,
        done=lambda s: s.room_id == ROOM_CERES_RIDLEY,
        stuck_jump_after=40,
    )

    return CeresFourthRoomFixture(
        from_room_id=ROOM_CERES_SCIENTIST,
        to_room_id=ROOM_CERES_RIDLEY,
        frames_consumed=frames,
        helper_source=(
            "_ceres_arm_pump_until from routes.kpdr.ceres.arm_pump "
            "(same parameters as play_ceres_to_ridley_door)"
        ),
        emulator_validated=False,
    )


def validate_ceres_fourth_room_emulator(
    start_state_path: Path | str | None = None,
) -> CeresFourthRoomFixture:
    """Validate fourth room on real emulator (ground truth).

    Runs `play_ceres_fourth_room` on stable-retro / SMEDIT snes9x from
    Scientist start state. This is the authoritative validation path for
    room-clear claims.

    Args:
        start_state_path: Path to Ceres Scientist start state (optional)
            If None, uses env var SM_CERES_SCIENTIST_STATE

    Returns:
        CeresFourthRoomFixture with emulator validation results

    Raises:
        FileNotFoundError: If ROM or start state not available
        RuntimeError: If emulator fails to load
        ValueError: If start state path not provided and env var not set
        TimeoutError: If Ridley room not reached within max_frames

    Note:
        Tests skip validation if:
        - ROM_AVAILABLE is False
        - SM_CERES_SCIENTIST_STATE env var not set
        - Start state file does not exist
    """
    if start_state_path is None:
        start_state_path = os.environ.get("SM_CERES_SCIENTIST_STATE")
        if not start_state_path:
            raise ValueError(
                "start_state_path not provided and "
                "SM_CERES_SCIENTIST_STATE not set"
            )

    start_state_path = Path(start_state_path)
    if not start_state_path.exists():
        raise FileNotFoundError(f"Start state not found: {start_state_path}")

    from super_metroid.assist import UnlimitedResourcesAssist
    from super_metroid.dev.common import make_dev_env
    from super_metroid.progression import MORPH_GRAPH
    from super_metroid.routes.runtime import RouteSession
    from retro_harness.env import read_state_bytes

    env = make_dev_env()
    try:
        env.reset()
        env.em.set_state(read_state_bytes(start_state_path))
        session = RouteSession(
            env, writer=None, assist=UnlimitedResourcesAssist(), graph=MORPH_GRAPH
        )
        try:
            fixture = play_ceres_fourth_room(session)
            success = session.state.room_id == ROOM_CERES_RIDLEY
        except TimeoutError:
            success = False
            fixture = CeresFourthRoomFixture(
                from_room_id=ROOM_CERES_SCIENTIST,
                to_room_id=ROOM_CERES_RIDLEY,
                frames_consumed=0,
                helper_source="timeout",
                emulator_validated=False,
            )
        return CeresFourthRoomFixture(
            from_room_id=fixture.from_room_id,
            to_room_id=fixture.to_room_id,
            frames_consumed=fixture.frames_consumed,
            helper_source=fixture.helper_source,
            emulator_validated=True,
            emulator_success=success,
            emulator_final_room=session.state.room_id,
            emulator_final_x=int(session.state.samus_x),
            emulator_final_y=int(session.state.samus_y),
        )
    finally:
        env.close()
