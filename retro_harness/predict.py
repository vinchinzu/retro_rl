"""Falsifiable next-frame claims. Games grade RAM; the grammar is shared.

An act without a prediction is a refuse (free). A miss names the belief that
broke. Planned sequences halt at the first miss — never batch exploration.
Unknown clauses miss. Approx without ± uses tol 4 (`x≈120`); games that care
write `x≈120±4` explicitly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

__all__ = [
    "Grade",
    "MissingPrediction",
    "first_miss_index",
    "grade_claims",
    "parse_claims",
]

_MOVE = re.compile(r"^move\s+(-?\d+)\s*,\s*(-?\d+)$", re.IGNORECASE)
_APPROX = re.compile(
    r"^([A-Za-z_][A-Za-z0-9_]*)\s*≈\s*(.+?)(?:\s*±\s*(\d+))?$"
)
_EXACT = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)$")

_POSE_KEYS = ("x", "y", "screen", "room", "mode", "level", "pose")
_DEFAULT_APPROX_TOL = 4


class MissingPrediction(ValueError):
    """Harness refuse: no claim arrived with the act."""


@dataclass(frozen=True)
class Grade:
    """One prediction, split into held vs missed clauses."""

    ok: bool
    held: tuple[str, ...]
    missed: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "held": list(self.held),
            "missed": list(self.missed),
        }


def parse_claims(text: str) -> tuple[str, ...]:
    """Split a claim string. Empty input is a refuse, not a noop."""
    if text is None or not str(text).strip():
        raise MissingPrediction("act requires a prediction")
    parts = tuple(part.strip() for part in str(text).split(";") if part.strip())
    if not parts:
        raise MissingPrediction("act requires a prediction")
    return parts


def _parse_value(text: str) -> Any:
    raw = text.strip()
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(raw, 0)
    except ValueError:
        return raw


def _pose_changed(before: Mapping[str, Any], after: Mapping[str, Any]) -> bool:
    for key in _POSE_KEYS:
        left, right = before.get(key), after.get(key)
        if left is None and right is None:
            continue
        if left != right:
            return True
    return False


def _grade_one(
    claim: str, before: Mapping[str, Any], after: Mapping[str, Any]
) -> bool:
    key = claim.strip()
    lowered = key.lower()
    if lowered == "noop":
        return not _pose_changed(before, after)
    if lowered == "change":
        return _pose_changed(before, after)

    move = _MOVE.match(key)
    if move is not None:
        dx, dy = int(move.group(1)), int(move.group(2))
        bx, by = before.get("x"), before.get("y")
        ax, ay = after.get("x"), after.get("y")
        if bx is None or by is None or ax is None or ay is None:
            return False
        return int(ax) - int(bx) == dx and int(ay) - int(by) == dy

    approx = _APPROX.match(key)
    if approx is not None:
        name = approx.group(1)
        want = _parse_value(approx.group(2))
        tol = int(approx.group(3) or _DEFAULT_APPROX_TOL)
        got = after.get(name)
        if got is None or not isinstance(want, int):
            return False
        try:
            return abs(int(got) - want) <= tol
        except (TypeError, ValueError):
            return False

    exact = _EXACT.match(key)
    if exact is not None:
        name = exact.group(1)
        want = _parse_value(exact.group(2))
        got = after.get(name)
        if isinstance(want, bool):
            return bool(got) is want
        if isinstance(want, int):
            try:
                return int(got) == want
            except (TypeError, ValueError):
                return False
        return got == want

    return False


def grade_claims(
    text: str,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> Grade:
    """Grade each clause on its own. One wrong part is a miss."""
    held: list[str] = []
    missed: list[str] = []
    for claim in parse_claims(text):
        if _grade_one(claim, before, after):
            held.append(claim)
        else:
            missed.append(claim)
    return Grade(ok=not missed, held=tuple(held), missed=tuple(missed))


def first_miss_index(grades: Sequence[Grade]) -> int | None:
    """Index of the first missed grade, or None if the plan held."""
    for index, grade in enumerate(grades):
        if not grade.ok:
            return index
    return None
