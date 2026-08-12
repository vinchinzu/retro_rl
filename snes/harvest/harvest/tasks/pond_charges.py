"""Scripted pond corridor charge action builders.

East→south past fence wall end, gap-south fallback, west→south-lip to F0.
Geometry constants used by the charge scripts live here.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from harvest.tasks.nav import make_action
from harvest.tasks.pond_policy import PRIMARY_POND_STAND


# ── corridor geometry ────────────────────────────────────────────────

# Fence wall ends at x=29 on y=31; past end is x≥31 for empty-handed south.
FENCE_WALL_END_X = 31
# Soft-block band just south of wall before F0 lip.
SOFT_BLOCK_Y_BAND = (32, 33)
# South lip corridor latitude for F0 approach.
SOUTH_LIP_Y = 34

# Fallback corridor crumbs if map_config import ever fails at call sites.
_FALLBACK_POST_GAP: Tuple[Tuple[int, int], ...] = (
    (13, 32),
    (16, 32),
    (20, 32),
    (24, 32),
    (28, 32),
    PRIMARY_POND_STAND,
)


# ── scripted charge action builders ──────────────────────────────────

def build_east_south_corridor_charge(
    player: Tuple[int, int],
    charge_count: int = 0,
) -> List[np.ndarray]:
    """Scripted east past fence wall end, then south into pond band.

    Empty-handed south through y=31 gap soft-blocks on (13,31). Fence wall
    is x=11–29 on y=31 — RIGHT only to x=29 then DOWN walks into remaining
    posts (power-on residual: charge lands (29,30) still_north). Must clear
    past the wall end (x≥30/31) then DOWN. Soft-block band at ~(25,30)
    also blocks pure densify.

    Power-on rr-5go9: pure RIGHT+DOWN from (29,30) never advances — DOWN
    hits the last fence post at (29,31) and RIGHT soft-blocks on the lip.
    Split legs: east-only while x<31 (RIGHT bursts + N/S micro-wiggle);
    south-only once x≥31. Completion re-queues until south of wall.
    """
    actions: List[np.ndarray] = []
    # Nudge off gap row if needed.
    if player[1] >= 31:
        actions.extend([make_action(up=True) for _ in range(20)])
        actions.extend([make_action() for _ in range(6)])

    target_x = FENCE_WALL_END_X
    n = charge_count
    # Never climb north of y≈29 — power-on residual ran into mountain
    # band ~(36,24) and east_pond kept going further east (rr-5go9).
    if player[0] >= target_x and player[1] <= 31:
        # Already past fence end longitude: pure south (no more RIGHT).
        actions.extend(
            [make_action(down=True, b=True) for _ in range(240)]
        )
        actions.extend([make_action() for _ in range(12)])
    elif player[0] < target_x:
        # East-only leg under fence wall. Mix walk+run RIGHT with tiny
        # N/S wiggle — never long UP (mountain drift).
        need_right = max(target_x - player[0], 2)
        bursts = max(need_right, 3 if n == 0 else 5)
        for _burst in range(bursts):
            # Walk-speed first (power-on B-run soft-block at ~(29,30)).
            if n >= 1 or player[0] >= 28:
                actions.extend(
                    [make_action(right=True) for _ in range(40)]
                )
            actions.extend(
                [make_action(right=True, b=True) for _ in range(48)]
            )
            actions.extend([make_action() for _ in range(3)])
            # Tiny vertical wiggle only (4 frames) — not a climb.
            actions.extend([make_action(up=True) for _ in range(3)])
            actions.extend(
                [make_action(right=True, b=True) for _ in range(28)]
            )
            actions.extend([make_action(down=True) for _ in range(3)])
            actions.extend(
                [make_action(right=True, b=True) for _ in range(28)]
            )
        actions.extend([make_action() for _ in range(6)])
        # Short south probe — full south when re-queue sees x≥31.
        actions.extend(
            [make_action(down=True, b=True) for _ in range(24 if n == 0 else 60)]
        )
        actions.extend([make_action() for _ in range(6)])
    else:
        # Past fence end (x≥31), already south-ish: reinforce south.
        actions.extend(
            [make_action(down=True, b=True) for _ in range(220)]
        )
        actions.extend([make_action() for _ in range(12)])
    return actions


def build_gap_south_fallback(
    player: Tuple[int, int],
) -> List[np.ndarray]:
    """When east past fence end is sealed, try open gap then south.

    Power-on residual: RIGHT from (29,30) never advances (soft edge) while
    dry fixture east-crawls freely. Walk west to the cleared gap column
    (~x=12–16), then long DOWN with L/R wiggle to soft-break (13,31).
    """
    actions: List[np.ndarray] = []
    # West toward typical corridor_only clear columns.
    need_left = max(player[0] - 14, 2)
    actions.extend(
        [make_action(left=True, b=True) for _ in range(28 * need_left)]
    )
    actions.extend([make_action() for _ in range(6)])
    # Align y=29–30 north of gap then charge south with wiggle.
    if player[1] > 30:
        actions.extend(
            [make_action(up=True) for _ in range(20)]
        )
        actions.extend([make_action() for _ in range(4)])
    elif player[1] < 29:
        actions.extend(
            [make_action(down=True) for _ in range(20)]
        )
        actions.extend([make_action() for _ in range(4)])
    # Long south with brief L/R wiggles to break (13,31) soft-block.
    for _ in range(6):
        actions.extend(
            [make_action(down=True, b=True) for _ in range(40)]
        )
        actions.extend([make_action(left=True) for _ in range(8)])
        actions.extend(
            [make_action(down=True, b=True) for _ in range(40)]
        )
        actions.extend([make_action(right=True) for _ in range(8)])
    actions.extend([make_action() for _ in range(10)])
    return actions



def build_west_south_lip_charge(
    player: Tuple[int, int],
) -> Tuple[List[np.ndarray], str]:
    """Scripted approach to F0 south lip from south-of-wall / soft-block.

    ROM (dry fixture GREEN): soft-block (28,32) → LEFT → south → east →
    north; second charge from ~(29,35) → near pond → multihop fill.

    Power-on residual: fence open lands ~(18,35). East-first with **capped
    UP** — long UP at low x walks back through the y=31 gap to ~(20,30).

    Returns (actions, band_name).
    """
    actions: List[np.ndarray] = []
    # East of pond south-of-wall: hard LEFT toward F0. Require y≥32 so we
    # never treat mountain band ~(36,24) as east_pond (rr-5go9 residual).
    if player[0] >= 34 and player[1] >= 32:
        band = "east_pond"
        # Prefer south lip y≈34–35 corridor while going west.
        if player[1] < 33:
            actions.extend(
                [make_action(down=True, b=True) for _ in range(40)]
            )
            actions.extend([make_action() for _ in range(4)])
        elif player[1] > 36:
            actions.extend(
                [make_action(up=True, b=True) for _ in range(24 * min(player[1] - 34, 5))]
            )
            actions.extend([make_action() for _ in range(4)])
        need_left = max(player[0] - 32, 4)
        actions.extend(
            [make_action(left=True, b=True) for _ in range(36 * need_left)]
        )
        actions.extend([make_action() for _ in range(4)])
        # Walk-speed LEFT if B-run soft-blocks (power-on (41,32)).
        actions.extend(
            [make_action(left=True) for _ in range(80)]
        )
        actions.extend([make_action() for _ in range(4)])
        actions.extend(
            [make_action(left=True, b=True) for _ in range(80)]
        )
        actions.extend([make_action() for _ in range(4)])
        if player[1] > 34:
            actions.extend(
                [make_action(up=True, b=True) for _ in range(20 * min(player[1] - 34, 4))]
            )
        elif player[1] < 34:
            actions.extend(
                [make_action(down=True, b=True) for _ in range(20 * min(34 - player[1], 4))]
            )
        actions.extend([make_action() for _ in range(8)])
    elif player[0] >= 34 and player[1] < 32:
        # Mountain / north-east drift: pure south toward pond latitude.
        band = "east_north_drift"
        actions.extend(
            [make_action(down=True, b=True) for _ in range(200)]
        )
        actions.extend([make_action() for _ in range(6)])
        actions.extend(
            [make_action(left=True, b=True) for _ in range(80)]
        )
        actions.extend([make_action() for _ in range(8)])
    elif player[0] >= 27 and 32 <= player[1] <= 33:
        band = "soft"
        # Soft-block band ~(28,32)/(29,32): RIGHT/DOWN often freeze.
        # Power-on rr-qc9r: LEFT 36 + UP landed (25,34) then south_far UP
        # climbed back to (29,32) — forever (25,34)↔(29,32) oscillation.
        # Fix: DOWN off soft-block first (no long LEFT), pure RIGHT on
        # y≥34 corridor. UP only after x≥31 and only if y>34.
        actions.extend(
            [make_action(down=True, b=True) for _ in range(100)]
        )
        actions.extend([make_action() for _ in range(6)])
        # Brief LEFT wiggle only (soft-break) — not a west retreat.
        actions.extend(
            [make_action(left=True, b=True) for _ in range(12)]
        )
        actions.extend([make_action() for _ in range(3)])
        actions.extend(
            [make_action(down=True, b=True) for _ in range(80)]
        )
        actions.extend([make_action() for _ in range(4)])
        # Long east toward pond longitude (x≥32) on south lip.
        actions.extend(
            [make_action(right=True, b=True) for _ in range(280)]
        )
        actions.extend([make_action() for _ in range(6)])
        # Walk-speed RIGHT if B-run soft-blocks mid-corridor.
        actions.extend(
            [make_action(right=True) for _ in range(60)]
        )
        actions.extend([make_action() for _ in range(4)])
        actions.extend(
            [make_action(right=True, b=True) for _ in range(80)]
        )
        actions.extend([make_action() for _ in range(6)])
        # Tiny UP only after east progress is expected — re-queue if short.
        # Cap 16: long UP re-enters y=31 gap or soft band (rr-5go9/qc9r).
        actions.extend(
            [make_action(up=True, b=True) for _ in range(16)]
        )
        actions.extend([make_action() for _ in range(4)])
        actions.extend(
            [make_action(right=True, b=True) for _ in range(48)]
        )
        actions.extend([make_action() for _ in range(8)])
    elif player[0] <= 26 and player[1] >= 32:
        band = "south_far"
        # Power-on ~(18,35)/(23,33)/(25,34): stay y≥34 while running east.
        # rr-qc9r: trailing UP 20 from y=34 climbed into soft (29,32).
        # When already on lip y=34–36: pure RIGHT only — no UP.
        if player[1] < 34:
            actions.extend(
                [make_action(down=True, b=True) for _ in range(80)]
            )
            actions.extend([make_action() for _ in range(4)])
        elif player[1] > 36:
            climb = min(player[1] - 35, 4)
            if climb > 0:
                actions.extend(
                    [make_action(up=True, b=True) for _ in range(24 * climb)]
                )
                actions.extend([make_action() for _ in range(4)])
        need_east = max(0, 32 - player[0])
        # Primary east corridor: long B-run + walk-speed recover.
        actions.extend(
            [make_action(right=True, b=True) for _ in range(40 * max(need_east, 8))]
        )
        actions.extend([make_action() for _ in range(6)])
        actions.extend(
            [make_action(right=True) for _ in range(80)]
        )
        actions.extend([make_action() for _ in range(4)])
        actions.extend(
            [make_action(right=True, b=True) for _ in range(120)]
        )
        actions.extend([make_action() for _ in range(6)])
        # Tiny south wiggle only (not UP) to break soft edges at y=34.
        if player[1] <= 34:
            actions.extend(
                [make_action(down=True) for _ in range(12)]
            )
            actions.extend([make_action() for _ in range(3)])
        actions.extend(
            [make_action(right=True, b=True) for _ in range(100)]
        )
        actions.extend([make_action() for _ in range(8)])
        # UP only if we started deep south (y>36 was climbed above).
        # Never UP when already on F0 latitude y=34–35.
    elif player[1] >= 34:
        band = "south"
        # South lip corridor x→32. rr-qc9r: UP at low x re-enters soft band;
        # long pure RIGHT from (29,35) overshoots to (36,36). Scale RIGHT
        # to remaining east only; near F0 (x≥28) use short bursts.
        if player[1] >= 38:
            actions.extend(
                [make_action(up=True, b=True) for _ in range(24 * min(player[1] - 35, 6))]
            )
            actions.extend([make_action() for _ in range(6)])
        if player[0] >= 32:
            # x≥32: align y to 34 then act/multihop.
            if player[1] > 34:
                actions.extend(
                    [make_action(up=True, b=True) for _ in range(24 * min(player[1] - 34, 4))]
                )
            elif player[1] < 34:
                actions.extend(
                    [make_action(down=True, b=True) for _ in range(24 * min(34 - player[1], 3))]
                )
            actions.extend([make_action() for _ in range(8)])
            actions.extend(
                [make_action(right=True, b=True) for _ in range(16)]
            )
            actions.extend([make_action() for _ in range(6)])
        elif player[0] >= 28:
            # Near stand (~(28–31,34–35)): short east only, no long run.
            need_east = max(32 - player[0], 1)
            actions.extend(
                [make_action(right=True, b=True) for _ in range(28 * need_east)]
            )
            actions.extend([make_action() for _ in range(4)])
            actions.extend(
                [make_action(right=True) for _ in range(24 * need_east)]
            )
            actions.extend([make_action() for _ in range(4)])
            if player[1] > 34:
                actions.extend(
                    [make_action(up=True, b=True) for _ in range(16)]
                )
                actions.extend([make_action() for _ in range(4)])
            actions.extend(
                [make_action(right=True, b=True) for _ in range(20)]
            )
            actions.extend([make_action() for _ in range(6)])
        else:
            # Far west on lip (x≤27): longer east, but cap before overshoot.
            need_east = max(32 - player[0], 3)
            actions.extend(
                [make_action(right=True, b=True) for _ in range(36 * min(need_east, 6))]
            )
            actions.extend([make_action() for _ in range(4)])
            actions.extend(
                [make_action(right=True) for _ in range(40)]
            )
            actions.extend([make_action() for _ in range(4)])
            actions.extend(
                [make_action(right=True, b=True) for _ in range(80)]
            )
            actions.extend([make_action() for _ in range(6)])
            if player[1] <= 35:
                actions.extend(
                    [make_action(down=True) for _ in range(12)]
                )
                actions.extend([make_action() for _ in range(3)])
                actions.extend(
                    [make_action(right=True, b=True) for _ in range(48)]
                )
                actions.extend([make_action() for _ in range(6)])
    else:
        band = "generic"
        # y=32–33 mid: south then east (gap-safe). Avoid long LEFT to gap.
        # No trailing UP — re-queue south/soft bands handle F0 latitude.
        if player[0] < 30:
            actions.extend(
                [make_action(down=True, b=True) for _ in range(80)]
            )
            actions.extend([make_action() for _ in range(4)])
            actions.extend(
                [make_action(right=True, b=True) for _ in range(40 * max(32 - player[0], 4))]
            )
            actions.extend([make_action() for _ in range(4)])
        else:
            actions.extend(
                [make_action(down=True, b=True) for _ in range(60)]
            )
            actions.extend([make_action() for _ in range(4)])
            actions.extend(
                [make_action(right=True, b=True) for _ in range(160)]
            )
            actions.extend([make_action() for _ in range(4)])
        actions.extend(
            [make_action(down=True, b=True) for _ in range(40)]
        )
        actions.extend([make_action() for _ in range(4)])
        actions.extend(
            [make_action(right=True, b=True) for _ in range(140)]
        )
        actions.extend([make_action() for _ in range(8)])

    return actions, band


__all__ = [
    "FENCE_WALL_END_X",
    "SOFT_BLOCK_Y_BAND",
    "SOUTH_LIP_Y",
    "build_east_south_corridor_charge",
    "build_gap_south_fallback",
    "build_west_south_lip_charge",
]
