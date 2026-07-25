"""Unit tests for continuous scene-advance helpers."""

from __future__ import annotations

from great_waldo_search.scene_advance import (
    ADVANCE_AFTER_SCENE,
    ADVANCE_TO_SCENE3,
    ADVANCE_TO_SCENE4,
    ADVANCE_TO_SCENE5,
    is_favorable_scroll_layout,
)


def test_continuous_advance_pre_idle_values() -> None:
    """Full-run path uses the bit-exact continuous pre_idle timings."""
    assert ADVANCE_TO_SCENE3.pre_idle == 1
    assert ADVANCE_TO_SCENE4.pre_idle == 2
    assert ADVANCE_TO_SCENE5.pre_idle == 0
    assert ADVANCE_AFTER_SCENE[2] is ADVANCE_TO_SCENE3
    assert ADVANCE_AFTER_SCENE[3] is ADVANCE_TO_SCENE4
    assert ADVANCE_AFTER_SCENE[4] is ADVANCE_TO_SCENE5


def test_favorable_scroll_layout_rejects_soft_and_wrong_lands() -> None:
    """Scene3 accepts ~160 and rejects soft ~206 / wrong ~32 lands."""
    assert is_favorable_scroll_layout(3, 158)
    assert is_favorable_scroll_layout(3, 160)
    assert not is_favorable_scroll_layout(3, 206)
    assert not is_favorable_scroll_layout(3, 32)
    assert is_favorable_scroll_layout(1, 999)
