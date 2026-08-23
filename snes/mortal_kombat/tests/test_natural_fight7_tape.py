"""ROM-free integrity tests for the natural-entry Match 7 continuation tape."""

from mortal_kombat.natural_fight7_tape import (
    NATURAL_FIGHT7_FRAMES,
    NATURAL_FIGHT7_RLE,
)
from mortal_kombat.scripts.replay_natural_fight1 import buttons_from_mask
from mortal_kombat.scripts.replay_natural_fight6 import (
    NATURAL_THROUGH_FIGHT6_FRAMES,
    NATURAL_THROUGH_FIGHT6_RLE,
)
from mortal_kombat.scripts.replay_natural_fight7 import (
    NATURAL_THROUGH_FIGHT7_FRAMES,
    NATURAL_THROUGH_FIGHT7_RLE,
)


def test_natural_fight7_tape_frame_count_and_masks() -> None:
    assert sum(count for _mask, count in NATURAL_FIGHT7_RLE) == NATURAL_FIGHT7_FRAMES
    assert NATURAL_FIGHT7_FRAMES == 4_751
    assert all(0 <= mask < 2**12 for mask, _count in NATURAL_FIGHT7_RLE)
    assert all(count > 0 for _mask, count in NATURAL_FIGHT7_RLE)


def test_natural_through_fight7_concatenates_predecessor() -> None:
    assert NATURAL_THROUGH_FIGHT7_FRAMES == NATURAL_THROUGH_FIGHT6_FRAMES + NATURAL_FIGHT7_FRAMES
    assert NATURAL_THROUGH_FIGHT7_FRAMES == 41_503
    assert NATURAL_THROUGH_FIGHT7_RLE[: len(NATURAL_THROUGH_FIGHT6_RLE)] == NATURAL_THROUGH_FIGHT6_RLE
    assert NATURAL_THROUGH_FIGHT7_RLE[len(NATURAL_THROUGH_FIGHT6_RLE) :] == NATURAL_FIGHT7_RLE


def test_buttons_from_mask_still_decodes_twelve_snes_buttons() -> None:
    buttons = buttons_from_mask((1 << 0) | (1 << 7) | (1 << 11))
    assert buttons.shape == (12,)
    assert buttons.dtype.name == "int8"
    assert set(buttons.nonzero()[0]) == {0, 7, 11}
