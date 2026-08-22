"""Zelda RAM claims on top of ``retro_harness.predict``."""

from __future__ import annotations

from typing import Any, Mapping

from retro_harness.predict import Grade, grade_claims
from zelda_i.ram import ZeldaSnapshot
from zelda_i.walk_physics import predicted_xy

__all__ = [
    "grade_snapshots",
    "grade_walk",
    "snapshot_fields",
    "walk_claim",
]


def snapshot_fields(snap: ZeldaSnapshot) -> dict[str, Any]:
    """Pose fields a walk/door prediction can name."""
    return {
        "x": int(snap.link_x),
        "y": int(snap.link_y),
        "screen": int(snap.screen),
        "room": int(snap.screen),
        "mode": int(snap.mode),
        "level": int(snap.level),
    }


def walk_claim(direction: str) -> str:
    """One-pixel cardinal move claim."""
    nx, ny = predicted_xy(0, 0, direction)
    return f"move {nx},{ny}"


def grade_snapshots(
    prediction: str,
    before: ZeldaSnapshot | Mapping[str, Any],
    after: ZeldaSnapshot | Mapping[str, Any],
) -> Grade:
    left = before if isinstance(before, Mapping) else snapshot_fields(before)
    right = after if isinstance(after, Mapping) else snapshot_fields(after)
    return grade_claims(prediction, left, right)


def grade_walk(
    direction: str,
    before: ZeldaSnapshot | Mapping[str, Any],
    after: ZeldaSnapshot | Mapping[str, Any],
) -> Grade:
    """Grade a cardinal walk. A miss means the cell ahead is blocked."""
    return grade_snapshots(walk_claim(direction), before, after)
