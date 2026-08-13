"""First Ceres room (Elevator → Falling Tile) hop/tape fixture.

Search-based fixture for Ceres Elevator → Falling Tile room transition using
the hop planning layer. This is the first room of the Ceres → Morph → Bomb
overnight bar sequence.

Policy:
- Search with StubPredictor (or sm_rev_predict if available) for speed
- Validate on real emulator (stable-retro / SMEDIT snes9x) for ground truth
- Mini/stub results are heuristics only; emulator wins
- Room-clear claims require emulator validation

This module provides:
- Fixture data structure for the searched tape
- Search function using TrajectoryEvaluator
- Emulator validation path
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from super_metroid.emulator_validation import (
    EmulatorValidationResult,
    validate_trajectory_on_emulator,
)
from super_metroid.hop_planning import (
    HopCandidate,
    TrajectoryEvaluator,
    evaluate_takeoff_trajectory,
)
from super_metroid.physics_sim import FrameInput, PhysicsPredictor, StubPredictor
from super_metroid.routes.kpdr.room_ids import (
    ROOM_CERES_ELEVATOR,
    ROOM_CERES_FALLING,
)
from super_metroid.takeoff import TakeoffWindow

__all__ = [
    "CeresFirstRoomFixture",
    "search_ceres_first_room",
    "validate_ceres_first_room",
    "CERES_ELEVATOR_START_X",
    "CERES_ELEVATOR_START_Y",
    "CERES_FALLING_TARGET_X",
    "CERES_FALLING_TARGET_Y",
]

# Ceres Elevator starting position (first control after intro)
CERES_ELEVATOR_START_X = 200
CERES_ELEVATOR_START_Y = 180

# Ceres Falling Tile room target (right side entry)
CERES_FALLING_TARGET_X = 50
CERES_FALLING_TARGET_Y = 200


@dataclass(frozen=True)
class CeresFirstRoomFixture:
    """First Ceres room hop/tape fixture.

    Contains the searched input sequence for Ceres Elevator → Falling Tile
    room transition with predictor and (optional) emulator validation results.
    """

    from_room_id: int
    to_room_id: int
    start_x: int
    start_y: int
    target_x: int
    target_y: int
    inputs: tuple[FrameInput, ...]
    predictor_name: str
    predictor_feasible: bool
    predictor_final_x: int
    predictor_final_y: int
    predictor_frames: int
    emulator_validated: bool = False
    emulator_success: bool = False
    emulator_final_room: int | None = None
    emulator_final_x: int | None = None
    emulator_final_y: int | None = None

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            **asdict(self),
            "inputs": [
                {"buttons": inp.buttons, "frame": i}
                for i, inp in enumerate(self.inputs)
            ],
        }

    @property
    def room_clear(self) -> bool:
        """True if emulator validation confirmed room clear.

        Never claim room-clear from predictor alone.
        """
        return self.emulator_validated and self.emulator_success


def search_ceres_first_room(
    predictor: PhysicsPredictor | None = None,
    *,
    max_search_frames: int = 300,
) -> CeresFirstRoomFixture:
    """Search for Ceres Elevator → Falling Tile trajectory.

    Uses hop planning layer with StubPredictor (default) or custom predictor
    to find a feasible input sequence. Does NOT validate on emulator.

    Args:
        predictor: Physics predictor (defaults to StubPredictor)
        max_search_frames: Maximum frames to search

    Returns:
        CeresFirstRoomFixture with predictor results (emulator_validated=False)

    Note:
        This is a simplified greedy search for the first room fixture.
        Production search would use A* or similar with multiple candidates.
    """
    if predictor is None:
        predictor = StubPredictor(name="ceres-first-room-search")

    evaluator = TrajectoryEvaluator(predictor)

    # Define takeoff window (starting momentum range)
    takeoff = TakeoffWindow(
        momentum_range=(CERES_ELEVATOR_START_X, CERES_ELEVATOR_START_X + 20),
        direction="RIGHT",
    )

    # Simple greedy search: RIGHT movement for N frames
    # (Real search would try multiple candidates, A/B jumps, etc.)
    best_candidate: HopCandidate | None = None
    best_distance = float("inf")

    for frames in range(10, max_search_frames, 10):
        # Try straight RIGHT movement
        inputs = [FrameInput(buttons=0x80) for _ in range(frames)]  # RIGHT=0x80

        candidate = evaluate_takeoff_trajectory(
            takeoff,
            CERES_ELEVATOR_START_X,
            CERES_ELEVATOR_START_Y,
            inputs,
            target_x_range=(
                CERES_FALLING_TARGET_X - 50,
                CERES_FALLING_TARGET_X + 50,
            ),
            target_y_range=(
                CERES_FALLING_TARGET_Y - 50,
                CERES_FALLING_TARGET_Y + 50,
            ),
        )

        # Calculate distance to target
        dx = candidate.final_x - CERES_FALLING_TARGET_X
        dy = candidate.final_y - CERES_FALLING_TARGET_Y
        distance = (dx * dx + dy * dy) ** 0.5

        if distance < best_distance:
            best_distance = distance
            best_candidate = candidate

        # Stop if we found a feasible solution
        if candidate.feasible:
            best_candidate = candidate
            break

    if best_candidate is None:
        # Fallback: return a minimal fixture
        inputs = tuple(FrameInput(buttons=0x80) for _ in range(30))
        return CeresFirstRoomFixture(
            from_room_id=ROOM_CERES_ELEVATOR,
            to_room_id=ROOM_CERES_FALLING,
            start_x=CERES_ELEVATOR_START_X,
            start_y=CERES_ELEVATOR_START_Y,
            target_x=CERES_FALLING_TARGET_X,
            target_y=CERES_FALLING_TARGET_Y,
            inputs=inputs,
            predictor_name=predictor.name(),
            predictor_feasible=False,
            predictor_final_x=CERES_ELEVATOR_START_X,
            predictor_final_y=CERES_ELEVATOR_START_Y,
            predictor_frames=30,
        )

    return CeresFirstRoomFixture(
        from_room_id=ROOM_CERES_ELEVATOR,
        to_room_id=ROOM_CERES_FALLING,
        start_x=CERES_ELEVATOR_START_X,
        start_y=CERES_ELEVATOR_START_Y,
        target_x=CERES_FALLING_TARGET_X,
        target_y=CERES_FALLING_TARGET_Y,
        inputs=best_candidate.inputs,
        predictor_name=predictor.name(),
        predictor_feasible=best_candidate.feasible,
        predictor_final_x=best_candidate.final_x,
        predictor_final_y=best_candidate.final_y,
        predictor_frames=best_candidate.frames,
    )


def validate_ceres_first_room(
    fixture: CeresFirstRoomFixture,
    start_state_path: Path | str,
) -> CeresFirstRoomFixture:
    """Validate fixture on real emulator (ground truth).

    Takes a searched fixture and runs it on stable-retro / SMEDIT snes9x.
    This is the authoritative validation path for room-clear claims.

    Args:
        fixture: Searched fixture to validate
        start_state_path: Path to Ceres Elevator start state

    Returns:
        Updated fixture with emulator validation results

    Raises:
        FileNotFoundError: If ROM is not available
        RuntimeError: If emulator fails to load

    Note:
        Requires ROM and stable-retro. Tests should use
        @pytest.mark.skipif(not ROM_AVAILABLE) to skip without ROM.
    """
    result = validate_trajectory_on_emulator(
        start_state_path,
        fixture.inputs,
        target_room_id=fixture.to_room_id,
        target_x_range=(fixture.target_x - 50, fixture.target_x + 50),
        target_y_range=(fixture.target_y - 50, fixture.target_y + 50),
    )

    return CeresFirstRoomFixture(
        from_room_id=fixture.from_room_id,
        to_room_id=fixture.to_room_id,
        start_x=fixture.start_x,
        start_y=fixture.start_y,
        target_x=fixture.target_x,
        target_y=fixture.target_y,
        inputs=fixture.inputs,
        predictor_name=fixture.predictor_name,
        predictor_feasible=fixture.predictor_feasible,
        predictor_final_x=fixture.predictor_final_x,
        predictor_final_y=fixture.predictor_final_y,
        predictor_frames=fixture.predictor_frames,
        emulator_validated=True,
        emulator_success=result.success,
        emulator_final_room=result.final_room_id,
        emulator_final_x=result.final_x,
        emulator_final_y=result.final_y,
    )
