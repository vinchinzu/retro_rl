"""Stroke-play metal-club route overlays.

Metal woods / irons change carry distances versus the standard bag. These
tables override Amateur stroke-play plans when ``ClubSet.METAL`` is active.
Until a hole is calibrated the overlay stays empty so standard routes remain
the fallback. VS HAL keeps its own ``VS_HAL_METAL_*`` tables.

Durable calibration notes live in ``docs/metal_stroke.md``. Update that file
and the ``METAL_STROKE_*`` memory constants whenever a clear improves.
"""

from __future__ import annotations

from hals_golf.tasks.routes.tables import AMATEUR_PARS

# Same card as Amateur; re-exported so metal diagnostics can import one module.
METAL_STROKE_PARS = AMATEUR_PARS

# Best recorded metal stroke-play scores (per-hole upgrades through 2026-07-21).
# Verified Title→course_complete (2026-07-21 video): total=70, card below with
# H4=6 / H10=7 as the live over-par holes.
METAL_STROKE_BEST_PARTIAL: tuple[int, ...] = (
    3,
    3,
    4,  # H3 birdie (soft driver finish from ~62y)
    2,  # H4 single-hole best; Title clear scored 6
    3,
    3,
    5,
    4,
    2,
    7,
    5,
    3,  # H12 birdie
    3,  # H13 par
    4,  # H14 par
    3,  # H15 birdie
    5,  # H16 birdie
    2,  # H17 eagle (MetalTee17 + putt 13→20)
    4,  # H18 par (post-H17 tee 42/-5 → 44/-5)
)

# Remaining over-par holes on the verified Title clear — calibration priority.
METAL_STROKE_WORST_HOLES: tuple[int, ...] = (4, 10)

# One-based hole -> stroke index -> (power, signed aim).
METAL_HOLE_SHOT_PLANS: dict[int, dict[int, tuple[int, int]]] = {
    1: {
        # Metal driver reaches ~154y fairway; stock 5I stalls at 111y.
        # Full PW left stops three yards from the cup.
        1: (42, -2),
    },
    2: {
        # Metal driver reaches ~96y; stock 3W never finds the green.
        # Soft SW stops one yard from the cup for birdie.
        1: (36, 0),
    },
    3: {
        # MetalTee3: 44/-2 → ~238y, 4W 42/-6 → ~62y (lie=0). Stock 7I 38/-2
        # hangs (no rest change). Soft driver 38/-2 → 1y green (MetalH3_fw62).
        0: (44, -2),
        1: (42, -6),
        2: (38, -2),
    },
    4: {
        # Metal PW finds fringe; controlled 9I reaches a puttable green.
        0: (38, 0),
    },
    5: {
        # VS HAL metal: soft SW left from ~47y instead of bouncing back.
        2: (26, -8),
    },
    6: {
        # Stock 3W finds sand. Driver left to ~142y, then PW left to a
        # two-yard birdie putt.
        0: (42, -4),
        1: (42, -4),
    },
    7: {
        # Live MetalTee7 calibration (stroke tees ≠ VS HAL):
        # driver 44/-8 → ~254y, 3W 44/-4 → ~144y, driver 44/-2 → ~23y
        # fairway, soft driver chip 26/-4 → 2y green (par putt).
        0: (44, -8),
        1: (44, -4),
        2: (44, -2),
        3: (26, -4),
    },
    8: {
        # MetalTee8: Amateur 42/-4 finds bunker at 101y (inescapable).
        # Extra-full driver left reaches ~104y fairway; soft 8I to 5y green.
        0: (44, -8),
        1: (38, 0),
        2: (31, -5),
    },
    9: {
        # MetalTee9: soft PW (38/0) lands 18y on the green (lie=6). Harder PW
        # finds fairway/bunker; 9I barely moves (~201y). Was 66 with stock PW.
        0: (38, 0),
    },
    10: {
        # Ported from VS HAL metal eagle corridor: 5I opener, controlled 3I
        # left, 9I finish for the longer leave variant.
        0: (42, -12),
        1: (38, -6),
        2: (38, 0),
    },
    11: {
        # MetalH11_fw219: Amateur 3W 42/-11 fails to move the ball and racks
        # up infinite strokes. Extra-full 3W left reaches ~58y fairway.
        2: (44, -8),
        # VS HAL metal SW finish avoids fringe loop from ~90y.
        3: (32, -4),
    },
    12: {
        # MetalTee12: stock/VS (42,0) fails to move (hung clear). Extra-full
        # driver left reaches ~136y fairway; soft 7I left to a 7y green.
        0: (44, -4),
        1: (38, -2),
        2: (26, -8),
    },
    13: {
        # MetalTee13: Amateur 38/0 finds bunker/rough with metal 8I.
        # Same power aimed left reaches a 21y green for birdie.
        0: (38, -2),
    },
    15: {
        # MetalTee15: Amateur 42/0 fails to move. Extra-full driver left to
        # ~161y rough; 3W left reaches an 8y green.
        0: (44, -8),
        1: (42, -4),
        2: (38, 3),
    },
    16: {
        # MetalTee16: Amateur 42/0 finds trouble. Driver -5 → ~276y, driver
        # +4 → ~149y, driver -4 → ~22y fairway, soft chip -8 → 5y green.
        0: (42, -5),
        1: (42, 4),
        2: (44, -4),
        3: (26, -8),
    },
    17: {
        # Post-H16 MetalTee17: 7I 34/-4 → 13y green (36/-2 was 18–21y).
        0: (34, -4),
    },
    18: {
        # Post-H17-eagle MetalTee18: 44/0 fails to move. Driver 42/-5 →
        # ~169y fairway; driver 44/-5 → 24y green (MetalH18_fw169).
        0: (42, -5),
        1: (44, -5),
        2: (26, -6),
    },
}

# One-based hole -> stroke index -> DOWN taps from the default metal 1W.
METAL_HOLE_CLUB_PLANS: dict[int, dict[int, int]] = {
    1: {1: 11},  # PW
    2: {1: 12},  # SW
    3: {0: 0, 1: 2, 2: 0},  # driver, 4W, soft driver chip to green
    4: {0: 10},  # 9I
    5: {2: 12},  # SW
    6: {0: 0, 1: 11},  # driver, PW
    7: {0: 0, 1: 1, 2: 0, 3: 0},  # driver, 3W, driver, soft driver chip
    8: {1: 9, 2: 12},  # 8I to green; SW recovery
    9: {0: 11},  # soft PW to green
    10: {0: 6, 1: 4, 2: 10},  # 5I, 3I, 9I
    11: {2: 1, 3: 12},  # 3W escape from ~219y; SW finish
    12: {0: 0, 1: 8, 2: 0},  # driver, 7I to green; soft driver chip recovery
    13: {0: 9},  # soft 8I
    15: {0: 0, 1: 1},  # driver, 3W to green
    16: {0: 0, 1: 0, 2: 0, 3: 0},  # four drivers / soft chip
    17: {0: 8},  # soft 7I
    18: {0: 0, 1: 0, 2: 12},  # driver, driver approach, SW recovery
}

# One-based hole -> REST yards -> putting-meter power.
METAL_HOLE_PUTT_PLANS: dict[int, dict[int, int]] = {
    1: {3: 41},
    2: {1: 42},
    3: {1: 42, 5: 42, 7: 40},
    6: {2: 42},
    7: {2: 42, 3: 42, 4: 42},
    8: {5: 42, 4: 42, 8: 40},
    9: {18: 20, 17: 22, 16: 24, 15: 26},
    11: {5: 42, 4: 42, 7: 40},
    12: {7: 40, 5: 42, 4: 42},
    13: {21: 28, 22: 28, 18: 20, 17: 22, 16: 24, 6: 40},
    15: {8: 40, 1: 42},
    16: {5: 42, 4: 42, 15: 13},
    17: {13: 20, 18: 22, 21: 28, 24: 28, 28: 26, 4: 42},
    18: {24: 28, 26: 28, 30: 26, 22: 28, 23: 28, 1: 42},
}

__all__ = [
    "METAL_HOLE_SHOT_PLANS",
    "METAL_HOLE_CLUB_PLANS",
    "METAL_HOLE_PUTT_PLANS",
    "METAL_STROKE_BEST_PARTIAL",
    "METAL_STROKE_PARS",
    "METAL_STROKE_WORST_HOLES",
]
