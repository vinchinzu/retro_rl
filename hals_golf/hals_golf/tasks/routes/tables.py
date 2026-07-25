"""Deterministic Amateur / VS HAL route tables."""

from __future__ import annotations

# VS HAL is a fixed front-12 match even though stroke play uses all 18.
VS_HAL_MATCH_HOLES = 12

# Amateur stroke-play scorecard. Keeping this beside the deterministic route
# lets scorecard diagnostics identify regressions on an individual hole.
AMATEUR_PARS = (4, 4, 5, 3, 4, 4, 5, 4, 3, 4, 5, 4, 3, 4, 4, 5, 3, 4)

# Deterministic routing for hazardous doglegs. Keys are one-based holes, then
# the on-screen stroke number before the shot; values are (power, signed aim).
# Hole 3 crosses water repeatedly if every lie uses the default straight line.
HOLE_SHOT_PLANS: dict[int, dict[int, tuple[int, int]]] = {
    2: {
        # Driver to the fairway, then 3W reaches an 11-yard birdie putt.
        0: (42, 0),
        1: (42, 0),
    },
    3: {
        # The direct driver line clips water. Lay up left with an 8I onto the
        # first fairway, then use a 3W and 3I to carry the lake and river.
        # A 5W leaves a short 6I chip-in for par.
        0: (32, -20),
        1: (42, -4),
        2: (42, -6),
        3: (42, 1),
        4: (26, -7),
    },
    4: {
        # PW carries the par-three water and stops four yards from the cup.
        0: (39, 1),
    },
    5: {
        # Two controlled drivers stay on the narrow fairway. A short pitch
        # angled right leaves a two-yard par tap-in.
        0: (40, 0),
        1: (40, 0),
        2: (27, 5),
        # VS HAL often leaves a longer third; keep a soft pitch in the table.
        3: (28, 4),
        4: (26, 3),
    },
    6: {
        # 3W avoids the tee hazard. Two controlled drivers then reach the cup
        # in three; retain the splash as a recovery route after any miss.
        0: (42, 0),
        1: (40, 0),
        2: (35, 0),
        3: (28, 0),
    },
    7: {
        0: (40, 0),
        1: (42, -10),
        2: (40, 0),
        3: (36, -2),
    },
    8: {
        # Driver left avoids the tee bunker; a controlled 8I reaches four
        # yards for a birdie tap-in.
        0: (42, -4),
        1: (40, 2),
        # Retain the old short approach after any miss.
        2: (31, -5),
    },
    9: {
        # PW reaches a 17-yard putt. Soft 39/0 birdies here but desyncs H10's
        # water route on the full-clear timeline, so keep the stable par line.
        0: (40, 0),
    },
    10: {
        # A full 5I left reaches the lower fairway, 3I left clears the lake,
        # and a controlled 7I leaves a five-yard par putt.
        0: (42, -8),
        1: (42, -8),
        2: (37, -1),
    },
    11: {
        0: (40, 0),
        1: (42, 0),
        2: (42, -11),
        # Full driver left reaches an 11-yard par putt.
        3: (42, -4),
    },
    12: {
        # Full power from the Amateur 400y tee is out of bounds and returns
        # to the same spot with a penalty stroke. This lands safely at 192y.
        0: (40, 0),
        1: (42, 0),
        2: (37, -1),
    },
    13: {
        # Live full-clear wind: 5I distance fallback stops ~85y and racks up
        # a double-digit hole. Controlled 8I reaches an 18y birdie putt.
        0: (38, 0),
    },
    14: {
        # Driver to the fairway, then 9I reaches a two-yard birdie tap-in.
        0: (42, 0),
        1: (41, 1),
        # Retain the old bunker recovery after a miss.
        2: (38, 0),
    },
    15: {
        # Full driver and 5W set up a controlled driver hole-out for birdie.
        0: (42, 0),
        1: (42, 0),
        2: (38, 3),
    },
    16: {
        # Straight driver finds fairway on the live H13-birdie timeline;
        # aim -1 finds rough and starts a long recovery loop.
        0: (42, 0),
        1: (42, -5),
        2: (42, -4),
        # Retain the old recovery route after a miss.
        3: (40, 0),
        4: (25, -8),
    },
    17: {
        # A soft 7I reaches four yards on the short par three.
        0: (36, 0),
    },
    18: {
        # Full driver power takes a penalty on this layout. A full 5W finds
        # the fairway, then a full 3W left clears the bunker to the green.
        0: (42, 0),
        1: (42, -5),
        # A soft SW left reaches the green from the resulting 47-yard lie.
        2: (26, -6),
    },
}

# VS HAL Amateur tees are longer on Hole 3; keep stroke-play plans intact and
# override only this mode's routing (calibrated from VsHal_Hole3_* states).
VS_HAL_HOLE_SHOT_PLANS: dict[int, dict[int, tuple[int, int]]] = {
    2: {
        # Keep the 3W approach; do not demote to a mid-iron from ~124y.
        1: (42, 0),
    },
    5: {
        # (42, -2) finds rough; (40, 0) finds sand. Controlled driver left
        # keeps fairway (~260), 3I to ~100, 9I right onto the green.
        0: (40, -2),
        1: (42, 0),
        2: (36, 5),
    },
    6: {
        # Full 3W finds sand; controlled 3W stays fairway (~201), then a
        # driver reaches ~123.  Full driver reaches the 32y fairway and a
        # soft driver right chips in for par.
        0: (40, 0),
        1: (40, 0),
        2: (42, 0),
        3: (28, 4),
    },
    8: {
        # Driver left, 8I to ~110y, full SW right chips in (verified).
        0: (42, -4),
        1: (40, 2),
        2: (40, 2),
    },
    9: {
        # Full-course wind: 40/+2 finds sand and starts a 26<->15y loop.
        # One frame softer reaches the 25y green for a stable par/tie.
        0: (39, 2),
    },
    11: {
        # From ~118y, 7I reaches a 17y green putt. Fringe chip-ins softlock
        # Hal's turn; always land on the green then putt.
        3: (38, 0),
        4: (38, 0),
    },
    12: {
        # Keep the short-of-OB tee; VS HAL matches stroke-play routing.
        0: (40, 0),
    },
    13: {
        # The generic driver needs a second approach and loses 3-2.  A
        # controlled 8I reaches the green directly for the two-shot tie.
        0: (38, 0),
    },
    3: {
        # Standard-club layup, then 3I/3I/9I to the green.
        0: (32, -20),
        1: (42, -6),
        2: (42, 0),
        3: (36, 0),
    },
    7: {
        # Longer tee / wind: bunker at ~330y, then ~260y approach.
        1: (38, -12),  # deep-sand driver escape (see bunker override)
        2: (40, -6),  # 5I into green corridor
        3: (36, -2),
        4: (32, -2),
    },
    10: {
        # Stroke-play 5I/-8 finds trees on VS HAL. More left keeps the
        # fairway (~307), 3I clears the lake (~169), driver left to ~50y,
        # then SW onto the green.
        0: (42, -12),
        1: (42, -8),
        2: (42, -8),
    },
}

VS_HAL_HOLE_CLUB_PLANS: dict[int, dict[int, int]] = {
    2: {1: 1},  # 3W
    3: {0: 9, 1: 4, 2: 4, 3: 10},  # 8I, 3I, 3I, 9I
    5: {0: 0, 1: 4, 2: 10},  # driver, 3I, 9I
    6: {0: 1, 1: 0, 2: 0, 3: 0},  # 3W, then three drivers
    7: {1: 0, 2: 6, 3: 8, 4: 10},  # driver sand, 5I, 7I, 9I
    8: {0: 0, 1: 9, 2: 12},  # driver, 8I, SW chip-in
    9: {0: 11},  # PW
    10: {0: 6, 1: 4, 2: 0},  # 5I, 3I, driver
    11: {3: 8, 4: 8},  # 7I
    13: {0: 9},  # 8I
}

# Metal woods and long irons have materially different ranges. Keep their
# calibrated routes separate so older standard-club save states remain valid.
VS_HAL_METAL_HOLE_SHOT_PLANS: dict[int, dict[int, tuple[int, int]]] = {
    4: {
        # Metal PW comes up on the fringe and costs two more strokes.  A
        # controlled 9I reaches the nine-yard green for a par-saving putt.
        0: (38, 0),
    },
    3: {
        # Driver left finds the 282y fairway, 4W left reaches 149y, and a
        # controlled 7I left finishes on the 13y green.
        0: (42, -6),
        1: (42, -6),
        2: (38, -2),
    },
    5: {
        # The first two standard slots reach 47y safely. A soft SW left lands
        # five yards from the cup instead of bouncing backward to 51y.
        2: (26, -8),
    },
    6: {
        # Controlled 3W to 183y, full 4I to 65y, then SW right into the cup.
        0: (40, 0),
        1: (42, 0),
        2: (28, 6),
    },
    7: {
        # Extra-full driver clears the first corner, 3W reaches 82y, and a
        # soft SW leaves a three-yard tap-in instead of the 220y sand lock.
        0: (44, -4),
        1: (42, 0),
        2: (34, -3),
    },
    10: {
        # Preserve the safe 5I opener.  When the full match leaves 275y, a
        # controlled 3I left holes out for eagle; the older 4W is blocked and
        # moves only eleven yards.  Keep the 9I finish for the 138y variant.
        0: (42, -12),
        1: (38, -6),
        2: (38, 0),
    },
    11: {
        # The standard 7I leaves a long fringe loop with metal ranges. SW
        # left from 91y stops one yard from the cup for a match-saving five.
        3: (32, -4),
    },
    12: {
        # Driver stays fairway at 143y, 8I lays up to 24y, then a soft driver
        # left holes out in three (the full-power standard opener finds sand).
        0: (42, 0),
        1: (36, 0),
        2: (26, -8),
    },
}

VS_HAL_METAL_HOLE_CLUB_PLANS: dict[int, dict[int, int]] = {
    4: {0: 10},  # 9I to nine-yard green
    3: {0: 0, 1: 2, 2: 8},  # driver, 4W, 7I
    5: {2: 12},  # SW to five-yard putt
    6: {0: 1, 1: 5, 2: 12},  # 3W, 4I, SW chip-in
    7: {0: 0, 1: 1, 2: 12},  # driver, 3W, SW
    10: {0: 6, 1: 4, 2: 10},  # 5I, 3I eagle, 9I variant finish
    11: {3: 12},  # SW to one-yard tap-in
    12: {0: 0, 1: 9, 2: 0},  # driver, 8I, driver chip-in
}

# Number of DOWN taps from the default 1W on the in-shot club card.
HOLE_CLUB_PLANS: dict[int, dict[int, int]] = {
    2: {1: 1},  # 3W
    3: {0: 9, 1: 1, 2: 4, 3: 2, 4: 7},  # 8I, 3W, 3I, 5W, 6I
    4: {0: 11},  # PW
    6: {0: 1},  # 3W
    8: {1: 9},  # 8I
    9: {0: 11},  # PW
    10: {0: 6, 1: 4, 2: 8},  # 5I, 3I, 7I
    11: {2: 1},  # 3W
    13: {0: 9},  # 8I
    14: {1: 10, 2: 4},  # 9I; 3I recovery
    15: {1: 2},  # 5W
    16: {1: 2, 2: 11, 4: 6},  # 5W, PW; 5I recovery chip-in
    17: {0: 8},  # 7I
    18: {0: 2, 1: 1, 2: 12},  # 5W, 3W, SW
}

# Greens have different slopes, so equal REST values can require different
# meter timings. These overrides are calibrated from deterministic tee states.
HOLE_PUTT_PLANS: dict[int, dict[int, int]] = {
    1: {
        # VS HAL / wind often leaves a longer first green putt than stroke play.
        18: 40,
        17: 40,
        16: 39,
        15: 38,
        13: 36,
        11: 34,
    },
    2: {
        # VS HAL 3W approach often leaves ~17y on the green.
        17: 40,
        16: 39,
        18: 41,
        19: 21,
        # 34 from the stroke-play 11y leave dumps into the greenside bunker;
        # 38-42 hole for birdie.
        11: 40,
        # Crawl leftovers after a soft first putt.
        15: 40,
        14: 40,
        13: 40,
        12: 42,
        1: 42,
    },
    6: {
        18: 40,
        15: 38,
        14: 37,
        12: 36,
        10: 36,
    },
    3: {
        8: 39,  # Holes the par putt instead of stopping four yards short.
        # VS HAL longer approach leaves 10–20y; calibrate those lags.
        20: 40,
        18: 43,
        13: 44,
        12: 43,
        11: 42,
        10: 41,
        9: 40,
    },
    4: {
        # Generic 52-frame lag from 17y crawls; 36 holes for birdie.
        17: 36,
        16: 36,
        15: 36,
        13: 40,
        12: 40,
        2: 42,
    },
    5: {
        8: 38,  # Holes the birdie putt on this green's slope.
        # 36 from 10y overshoots to ~17y; 38 holes. Longer lags to 10y.
        10: 38,
        12: 38,
        14: 40,
        16: 41,
        18: 42,
        19: 40,
        20: 40,
        23: 40,
    },
    7: {
        # Generic 52-frame lag crawls; 36 holes the common ~20y leave.
        20: 36,
        19: 36,
        18: 36,
        17: 36,
        16: 36,
        14: 36,
        13: 20,  # Holes the par putt on this green's slope.
        8: 39,
    },
    8: {
        # Same long-lag trap as H7; 28 holes the ~23y leave.
        23: 28,
        21: 28,
        19: 28,
        17: 36,
        16: 36,
        14: 36,
        13: 36,
        12: 40,
        11: 40,
        9: 38,
    },
    9: {
        17: 40,
        19: 21,
        22: 18,
        23: 20,
        24: 21,
        # VS HAL PW right leaves ~25y; 38 lags to a tap-in (22 crawled).
        25: 38,
        # Soft PW tee leave: 40 holes the eight-yard birdie.
        8: 40,
        5: 43,
        4: 42,
        7: 40,
        10: 40,
        11: 40,
    },  # Wind-dependent tee results: hole or lag plus tap-in.
    13: {
        # Live-clear 8I leave: 20 holes for birdie. H14 uses REST bands when
        # the early finish leaves a longer fairway (~172y vs ~163y).
        18: 20,
        22: 20,
        17: 22,
        16: 24,
        # Soft PW tee leave: 40 holes the six-yard birdie.
        6: 40,
        4: 40,
    },
    15: {
        # Soft SW leave after the longer H13-birdie fairway approach.
        1: 42,
    },
    11: {
        11: 37,
        16: 18,
        # 40 softlocks the Hal transition; 36/18/24 reach Hole 12 tee.
        17: 36,
        15: 38,
        13: 36,
        27: 36,
        # Longer VS HAL leaves were crawling on generic 52.
        22: 28,
        21: 28,
        20: 28,
        19: 28,
        18: 36,
    },  # Par-putt timings plus VS HAL longer lags.
    16: {
        15: 13,  # Holes the birdie putt on this green's slope.
        14: 13,
        13: 13,
        12: 12,
        11: 12,
        10: 12,
    },
    18: {
        15: 16,
        29: 25,
        8: 39,
        1: 42,
    },  # Final-timeline par putt; retain old-route recovery timings.
}

# Match-play wind and turn timing create distinct landing coordinates on a
# few greens.  Keep these calibrations out of the verified stroke-play route.
VS_HAL_HOLE_PUTT_PLANS: dict[int, dict[int, int]] = {
    # The metal route reaches this green in two.  Generic power 45 stops five
    # yards short; 42 holes the eight-yard putt for birdie and an early lead.
    1: {8: 42},
    # Metal H2's PW recovery reaches 23y; 20 holes this uphill coordinate.
    2: {23: 20},
    # Holes the common H3 approach instead of allowing Hal's turn to be
    # mistaken for three more player putts.
    3: {20: 24},
    # The VS HAL 17-yard coordinate holes at 18; stroke play retains 36.
    4: {17: 18},
    # Metal H10 reaches this slope at 14y; the generic 51-frame roll stops
    # eleven yards away, while 18 holes the first putt for par.
    10: {14: 18},
    # Full-course VS HAL reaches H12 in two and holes this for birdie.
    12: {23: 23},
}
