"""SMB next-frame claims over ``approx.step``, graded against observed pose.

Search in the stepper (free). Live halt is the first missed per-frame claim.
Residual R(τ) stays the search-model keep/reject, not this halt.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from retro_harness.predict import Grade, first_miss_index, grade_claims
from smb.approx import step
from smb.observation import Observation, PlayerPhysics, World

__all__ = [
    "grade_player",
    "grade_trajectory",
    "halt_plan",
    "player_claim",
    "player_fields",
    "predict_step",
]


def player_fields(player: PlayerPhysics | Observation) -> dict[str, Any]:
    return {
        "x": int(player.x),
        "y": int(player.y),
        "pose": int(player.pose),
        "room": int(player.room),
        "dead": bool(player.dead),
    }


def player_claim(player: PlayerPhysics | Observation) -> str:
    """Exact after-pose claim for one frame."""
    return (
        f"x={int(player.x)}; y={int(player.y)}; "
        f"pose={int(player.pose)}; room={int(player.room)}"
    )


def predict_step(
    player: PlayerPhysics,
    action: Sequence[int],
    world: World | None = None,
) -> tuple[PlayerPhysics, str]:
    """Roll the pure stepper one frame and return (next, claim)."""
    nxt = step(player, action, world)
    return nxt, player_claim(nxt)


def grade_player(prediction: str, observed: PlayerPhysics | Observation) -> Grade:
    """Grade a stepper claim against an observed pose (after-only exact fields)."""
    return grade_claims(prediction, {}, player_fields(observed))


def grade_trajectory(
    predicted: Sequence[PlayerPhysics | Observation],
    observed: Sequence[PlayerPhysics | Observation],
) -> list[Grade]:
    """Zip predicted vs observed poses and grade each claimed frame."""
    return [
        grade_player(player_claim(pred), obs)
        for pred, obs in zip(predicted, observed)
    ]


def halt_plan(grades: Sequence[Grade]) -> bool:
    """True when live execution must stop (first missed per-frame claim)."""
    return first_miss_index(grades) is not None
