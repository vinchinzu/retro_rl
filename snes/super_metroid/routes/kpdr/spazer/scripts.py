"""Open-loop RLE tables for Early Spazer (guide-shaped, not gold human paste).

Source guides: ``tasks/spazer_from_charge_chunks.json``,
``tasks/spazer_top_drop_human.json``. Never train on excluded thrash frames
f9403–11066.
"""

from __future__ import annotations

from super_metroid.routes.rle import RleScript

# Floor→standing mid — guide ``floor_to_mid`` compressed phases (~205f).
# aim-up clear → shaft position → spin-left → RIGHT+A crest → mid land.
# Closed-loop Cacatac clear runs *before* this script.
FLOOR_MID_RLE: RleScript = (
    # Phase: aim-up clear (guide shape)
    (4, ("UP",)),
    (7, ("UP", "X")),
    (3, ("UP",)),
    (7, ("UP", "X")),
    (6, ("UP",)),
    (6, ()),
    # Phase: in-shaft position (x≈48–55)
    (8, ("LEFT",)),
    (6, ()),
    (5, ("RIGHT",)),
    (21, ()),
    # Phase: spin-left into left wall, then RIGHT+A crest over mid lip
    (1, ("B", "LEFT")),
    (37, ("B", "LEFT", "A")),
    (3, ("B", "A")),
    (1, ()),
    (3, ("RIGHT",)),
    (55, ("RIGHT", "A")),
    (1, ("A",)),
    # Phase: land settle standing mid
    (31, ()),
)

# Top return handoff ~(380,155) → floor ~(43,395) — guide top-drop slice.
# Morph+X = bombs on shelf. No RIGHT Super re-entry.
TOP_MID_RLE: RleScript = (
    (12, ()),
    (13, ("RIGHT",)),
    (13, ()),
    (2, ("LEFT",)),
    (23, ("B", "LEFT")),
    (20, ("B", "LEFT", "A")),
    (1, ("LEFT", "A")),
    (18, ("LEFT",)),
    (8, ()),
    (5, ("DOWN",)),
    (5, ()),
    (6, ("DOWN",)),
    (16, ()),
    (49, ("LEFT",)),
    (9, ("LEFT", "X")),
    (87, ("LEFT",)),
    (7, ("LEFT", "X")),
    (105, ("LEFT",)),
    (80, ()),
    (7, ("X",)),
    (15, ()),
    (3, ("RIGHT",)),
    (3, ("UP", "RIGHT")),
    (40, ()),
)

# Solid top node4 ~(91,91) → Super door lip — morph tunnel (bombs=X).
# Mainline door approach (not a residual card).
TOP_DOOR_APPROACH_RLE: RleScript = (
    (4, ()),
    (5, ("X",)),
    (5, ()),
    (6, ("DOWN",)),
    (6, ()),
    (7, ("DOWN",)),
    (12, ()),
    (9, ("X",)),
    (2, ()),
    (131, ("RIGHT",)),
    (10, ("RIGHT", "X")),
    (131, ("RIGHT",)),
    (5, ("UP", "RIGHT")),
    (13, ("RIGHT",)),
    (34, ("RIGHT", "A")),
    (5, ("RIGHT",)),
    (10, ("RIGHT", "A")),
    (7, ("RIGHT",)),
    (14, ()),
    (40, ("RIGHT", "B")),
)

__all__ = [
    "FLOOR_MID_RLE",
    "RleScript",
    "TOP_DOOR_APPROACH_RLE",
    "TOP_MID_RLE",
]
