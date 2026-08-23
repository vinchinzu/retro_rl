"""ROM-free integrity tests for the natural-entry Match 3 continuation tape."""

from mortal_kombat.natural_fight3_tape import (
    NATURAL_FIGHT3_FRAMES,
    NATURAL_FIGHT3_RLE,
)
from mortal_kombat.scripts.replay_natural_fight1 import buttons_from_mask
from mortal_kombat.scripts.replay_natural_fight2 import (
    NATURAL_THROUGH_FIGHT2_FRAMES,
    NATURAL_THROUGH_FIGHT2_RLE,
)
from mortal_kombat.scripts.replay_natural_fight3 import (
    NATURAL_THROUGH_FIGHT3_FRAMES,
    NATURAL_THROUGH_FIGHT3_RLE,
)


def test_natural_fight3_tape_frame_count_and_masks() -> None:
    assert sum(count for _mask, count in NATURAL_FIGHT3_RLE) == NATURAL_FIGHT3_FRAMES
    assert NATURAL_FIGHT3_FRAMES == 5_159
    assert all(0 <= mask < 2**12 for mask, _count in NATURAL_FIGHT3_RLE)
    assert all(count > 0 for _mask, count in NATURAL_FIGHT3_RLE)


def test_natural_through_fight3_concatenates_predecessor() -> None:
    assert NATURAL_THROUGH_FIGHT3_FRAMES == NATURAL_THROUGH_FIGHT2_FRAMES + NATURAL_FIGHT3_FRAMES
    assert NATURAL_THROUGH_FIGHT3_FRAMES == 18_077
    assert NATURAL_THROUGH_FIGHT3_RLE[: len(NATURAL_THROUGH_FIGHT2_RLE)] == NATURAL_THROUGH_FIGHT2_RLE
    assert NATURAL_THROUGH_FIGHT3_RLE[len(NATURAL_THROUGH_FIGHT2_RLE) :] == NATURAL_FIGHT3_RLE


def test_buttons_from_mask_still_decodes_twelve_snes_buttons() -> None:
    buttons = buttons_from_mask((1 << 0) | (1 << 7) | (1 << 11))
    assert buttons.shape == (12,)
    assert buttons.dtype.name == "int8"
    assert set(buttons.nonzero()[0]) == {0, 7, 11}
