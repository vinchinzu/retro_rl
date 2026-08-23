"""ROM-free integrity tests for the natural-entry Match 1 tape."""

from mortal_kombat.natural_fight1_tape import (
    NATURAL_FIGHT1_FRAMES,
    NATURAL_FIGHT1_RLE,
)
from mortal_kombat.scripts.replay_natural_fight1 import buttons_from_mask


def test_natural_fight1_tape_frame_count_and_masks() -> None:
    assert sum(count for _mask, count in NATURAL_FIGHT1_RLE) == NATURAL_FIGHT1_FRAMES
    assert NATURAL_FIGHT1_FRAMES == 7_863
    assert all(0 <= mask < 2**12 for mask, _count in NATURAL_FIGHT1_RLE)
    assert all(count > 0 for _mask, count in NATURAL_FIGHT1_RLE)


def test_buttons_from_mask_decodes_twelve_snes_buttons() -> None:
    buttons = buttons_from_mask((1 << 0) | (1 << 7) | (1 << 11))
    assert buttons.shape == (12,)
    assert buttons.dtype.name == "int8"
    assert set(buttons.nonzero()[0]) == {0, 7, 11}
