"""ROM-free integrity tests for the natural-entry Match 5 continuation tape."""

from mortal_kombat.natural_fight5_tape import (
    NATURAL_FIGHT5_FRAMES,
    NATURAL_FIGHT5_RLE,
)
from mortal_kombat.scripts.replay_natural_fight1 import buttons_from_mask
from mortal_kombat.scripts.replay_natural_fight4 import (
    NATURAL_THROUGH_FIGHT4_FRAMES,
    NATURAL_THROUGH_FIGHT4_RLE,
)
from mortal_kombat.scripts.replay_natural_fight5 import (
    NATURAL_THROUGH_FIGHT5_FRAMES,
    NATURAL_THROUGH_FIGHT5_RLE,
)


def test_natural_fight5_tape_frame_count_and_masks() -> None:
    assert sum(count for _mask, count in NATURAL_FIGHT5_RLE) == NATURAL_FIGHT5_FRAMES
    assert NATURAL_FIGHT5_FRAMES == 4_619
    assert all(0 <= mask < 2**12 for mask, _count in NATURAL_FIGHT5_RLE)
    assert all(count > 0 for _mask, count in NATURAL_FIGHT5_RLE)


def test_natural_through_fight5_concatenates_predecessor() -> None:
    assert NATURAL_THROUGH_FIGHT5_FRAMES == NATURAL_THROUGH_FIGHT4_FRAMES + NATURAL_FIGHT5_FRAMES
    assert NATURAL_THROUGH_FIGHT5_FRAMES == 29_783
    assert NATURAL_THROUGH_FIGHT5_RLE[: len(NATURAL_THROUGH_FIGHT4_RLE)] == NATURAL_THROUGH_FIGHT4_RLE
    assert NATURAL_THROUGH_FIGHT5_RLE[len(NATURAL_THROUGH_FIGHT4_RLE) :] == NATURAL_FIGHT5_RLE


def test_buttons_from_mask_still_decodes_twelve_snes_buttons() -> None:
    buttons = buttons_from_mask((1 << 0) | (1 << 7) | (1 << 11))
    assert buttons.shape == (12,)
    assert buttons.dtype.name == "int8"
    assert set(buttons.nonzero()[0]) == {0, 7, 11}
