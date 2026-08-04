from __future__ import annotations

from tmnt_iv.scripts.record_segmented_showcase import showcase_clips


def test_showcase_is_stage_ordered_and_disclosed_as_segmented() -> None:
    clips = showcase_clips()

    assert clips[0].state == "Stage1"
    assert clips[-1].state == "Boss9_phase2_low"
    assert any(clip.state == "Stage9_Clear" for clip in clips)
    assert all(clip.max_frames > 0 for clip in clips)
    assert all(clip.note for clip in clips)

