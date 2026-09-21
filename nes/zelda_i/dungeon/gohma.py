"""Shared Gohma sensors: the eye clock and the arrow lead (L6 0x1C, L8 0x1E).

There is one Gohma. `level6/gohma.py` and `level8/magic_key.py` fight it in
two different rooms, and until this module they carried two byte-identical
copies of every sensor it needs: the `$03C7` eye tracker, the eight-sample
strafe history, and the lead-and-clamp that turns that history into a column
to stand in. `magic_key` already imported the *constants* from `level6.gohma`;
only the code was doubled. Same shape as `dungeon/gleeok.py`, which owns the
sensors L4 and L6 share while the fight controllers stay on the levels.

**Sensors only.** No controller, no dispatcher, no room geometry: the firing
column (`STAND_Y`, `LINK_X_MIN/MAX`, `COLUMN_X_MIN/MAX`) is per-room and stays
on the level that owns the room. Everything here is a pure function of RAM.

The arithmetic is reproduced exactly, down to the sample window and the
minimum-sample gate. Both chains are frame-perfect and a different velocity
read is a different arrow.
"""

from __future__ import annotations

from typing import Any

from zelda_i.dungeon.ids import GOHMA_BLUE_OBJECT_TYPE, GOHMA_OBJECT_TYPE
from zelda_i.ram import ZeldaObject, ZeldaSnapshot

__all__ = [
    "ARROW_SPEED",
    "EYE_ADDR",
    "EYE_CLOCK_CAP",
    "EYE_EDGE_WINDOW",
    "EYE_SHUT",
    "GOHMA_TYPES",
    "GX_MIN_SAMPLES",
    "GX_WINDOW",
    "LEAD_CLAMP",
    "advance_eye",
    "aim_column",
    "arrow_aim_x",
    "eye_fresh_open",
    "gohma_live",
    "read_eye",
    "strafe_vx",
]

# 0x33 HP 96 (three wooden arrows), 0x34 HP 32. Colour is not asserted:
# `dungeon/ids.py` labels 0x33 red and 0x34 blue and the ROM HP suggests the
# labels are swapped (`scratch/enemy_constants_rom.md` section 7.3). Both are
# accepted so a re-observation is never a false miss.
GOHMA_TYPES = frozenset({GOHMA_OBJECT_TYPE, GOHMA_BLUE_OBJECT_TYPE})

# --- The eye ---------------------------------------------------------------
# RAM 0x03C7 reads 0xC0 for the ~17f closed blink and a lower value
# (0x70 / 0x60 / 0x58 ...) while open, on a ~65f cycle. An arrow only damages
# Gohma if it arrives while the eye is open, and firing on any fixed multiple
# of the cycle aliases straight onto the closed blink (L6 v1-v4 and the first
# reactive pass: 20+ arrows, 0 connects). Fire on the *rising edge* only.
EYE_ADDR = 0x03C7
EYE_SHUT = 0xC0
EYE_EDGE_WINDOW = 16
# The open-frame counter saturates rather than growing without bound; it is
# only ever compared against EYE_EDGE_WINDOW.
EYE_CLOCK_CAP = 9999

# --- The arrow -------------------------------------------------------------
ARROW_SPEED = 2.8  # px/frame up the column (L6: y~168 fire -> kill ~18f)
LEAD_CLAMP = 12
# Samples of the body's x kept for the strafe read, and the minimum before the
# read is trusted. Gohma strafes on a ~260f cycle; fewer than GX_MIN_SAMPLES
# reads 0.0 rather than a one-frame difference.
GX_WINDOW = 8
GX_MIN_SAMPLES = 4


def gohma_live(snap: ZeldaSnapshot) -> list:
    """Gohma body slots 1-12. TYPE presence: HP may read 0 mid-fight."""
    return [
        obj
        for obj in snap.objects
        if 1 <= obj.slot <= 12 and int(obj.type_id) in GOHMA_TYPES
    ]


def read_eye(env: Any | None) -> int | None:
    """``$03C7``, or None when there is no env (unit tests, replay)."""
    if env is None:
        return None
    try:
        return int(env.get_ram()[EYE_ADDR])
    except Exception:  # pragma: no cover - defensive
        return None


def advance_eye(eye_open_since: int, byte: int | None) -> int:
    """One frame of the eye clock: -1 while shut, else frames since it opened.

    An unreadable byte counts as shut — never fire on a sensor that is not
    answering.
    """
    if byte is None or byte == EYE_SHUT:
        return -1
    if eye_open_since < 0:
        return 0
    return min(EYE_CLOCK_CAP, eye_open_since + 1)


def eye_fresh_open(eye_open_since: int) -> bool:
    """Eye left the blink within EYE_EDGE_WINDOW frames (an arrow will land)."""
    return 0 <= eye_open_since <= EYE_EDGE_WINDOW


def strafe_vx(hist: list[int], gx: int) -> float:
    """Push ``gx`` onto ``hist`` (in place, GX_WINDOW deep) and read px/frame.

    0.0 until GX_MIN_SAMPLES samples exist. Mean over the kept window, which is
    the same estimator ``dungeon.tracking.ObjectTracker`` uses — but *not* the
    same tracker: this history is only advanced on the frames the fight reaches
    an aim decision, while the tracker samples every observed frame.
    """
    hist.append(int(gx))
    del hist[:-GX_WINDOW]
    if len(hist) < GX_MIN_SAMPLES:
        return 0.0
    return (hist[-1] - hist[0]) / (len(hist) - 1)


def aim_column(
    gx: int,
    gvx: float,
    body_y: int,
    link_y: int,
    bounds: tuple[int, int],
) -> int:
    """Column to stand in, from an already-sampled strafe read.

    Leads the strafe by the arrow's flight time, clamps the lead to
    LEAD_CLAMP, then clamps the column into the room's walkable ``bounds``.
    Split from ``strafe_vx`` because L6 samples the history *before* its climb
    branch can return, and the sample cadence is part of the measurement.
    """
    flight = max(1.0, (int(link_y) - int(body_y)) / ARROW_SPEED)
    lead = int(round(gvx * flight))
    lead = max(-LEAD_CLAMP, min(LEAD_CLAMP, lead))
    lo, hi = bounds
    return max(lo, min(hi, int(gx) + lead))


def arrow_aim_x(
    hist: list[int],
    body: ZeldaObject,
    link_y: int,
    bounds: tuple[int, int],
) -> int:
    """``strafe_vx`` then ``aim_column`` — for callers that do both at once."""
    gx = int(body.x)
    return aim_column(gx, strafe_vx(hist, gx), int(body.y), link_y, bounds)
