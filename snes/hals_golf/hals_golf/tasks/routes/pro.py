"""Pro-difficulty route overlays (stubs pending calibration).

Pro tees are longer and greens faster than Amateur, so these tables will
eventually diverge. Until each hole is calibrated they stay empty and merge as
no-ops on top of the verified Amateur / VS HAL tables — Amateur remains the
source of truth. Do not invent Pro numbers here; an empty overlay keeps the
Amateur clear intact.
"""

from __future__ import annotations

# One-based hole -> stroke index -> (power, signed aim).
PRO_HOLE_SHOT_PLANS: dict[int, dict[int, tuple[int, int]]] = {}

# One-based hole -> stroke index -> DOWN taps from the default 1W.
PRO_HOLE_CLUB_PLANS: dict[int, dict[int, int]] = {}

# One-based hole -> REST yards -> putting-meter power.
PRO_HOLE_PUTT_PLANS: dict[int, dict[int, int]] = {}

__all__ = [
    "PRO_HOLE_SHOT_PLANS",
    "PRO_HOLE_CLUB_PLANS",
    "PRO_HOLE_PUTT_PLANS",
]
