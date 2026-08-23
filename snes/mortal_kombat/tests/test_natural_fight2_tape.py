"""ROM-free integrity tests for the natural-entry Match 2 continuation tape."""

from mortal_kombat.natural_fight1_tape import NATURAL_FIGHT1_FRAMES, NATURAL_FIGHT1_RLE
from mortal_kombat.natural_fight2_tape import (
    NATURAL_FIGHT2_FRAMES,
    NATURAL_FIGHT2_RLE,
)
from mortal_kombat.scripts.replay_natural_fight1 import buttons_from_mask
from mortal_kombat.scripts.replay_natural_fight2 import (
    NATURAL_THROUGH_FIGHT2_FRAMES,
    NATURAL_THROUGH_FIGHT2_RLE,
)


def test_natural_fight2_tape_frame_count_and_masks() -> None:
    assert sum(count for _mask, count in NATURAL_FIGHT2_RLE) == NATURAL_FIGHT2_FRAMES
    assert NATURAL_FIGHT2_FRAMES == 5_055
    assert all(0 <= mask < 2**12 for mask, _count in NATURAL_FIGHT2_RLE)
    assert all(count > 0 for _mask, count in NATURAL_FIGHT2_RLE)


def test_natural_through_fight2_concatenates_predecessor() -> None:
    assert NATURAL_THROUGH_FIGHT2_FRAMES == NATURAL_FIGHT1_FRAMES + NATURAL_FIGHT2_FRAMES
    assert NATURAL_THROUGH_FIGHT2_FRAMES == 12_918
    assert NATURAL_THROUGH_FIGHT2_RLE[: len(NATURAL_FIGHT1_RLE)] == NATURAL_FIGHT1_RLE
    assert NATURAL_THROUGH_FIGHT2_RLE[len(NATURAL_FIGHT1_RLE) :] == NATURAL_FIGHT2_RLE


def test_buttons_from_mask_still_decodes_twelve_snes_buttons() -> None:
    buttons = buttons_from_mask((1 << 0) | (1 << 7) | (1 << 11))
    assert buttons.shape == (12,)
    assert buttons.dtype.name == "int8"
    assert set(buttons.nonzero()[0]) == {0, 7, 11}
