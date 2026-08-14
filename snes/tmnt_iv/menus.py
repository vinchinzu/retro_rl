"""Menu / boot helpers for reaching Stage 1 gameplay."""

from __future__ import annotations

from collections.abc import Iterator

from retro_harness.input_script import FrameAction, idle_frames, mash_button, mash_start


def title_advance_frames(
    *,
    pulses: int = 8,
    hold: int = 10,
    gap: int = 45,
) -> list[FrameAction]:
    """START pulses to skip logos / title into mode / char select."""
    return mash_start(pulses=pulses, hold=hold, gap=gap)


def character_confirm_frames(
    *,
    hold: int = 12,
    gap: int = 35,
    pulses: int = 3,
) -> list[FrameAction]:
    """Confirm the default turtle (Leonardo) on character select."""
    return mash_button(
        "START",
        pulses=pulses,
        hold=hold,
        gap=gap,
        hold_reason="char_confirm",
        wait_reason="char_wait",
    )


def boot_to_stage1_script() -> Iterator[FrameAction]:
    """Yield a conservative title → char-select → stage-open script."""
    yield from title_advance_frames()
    yield from idle_frames(120, "post_title_idle")
    yield from character_confirm_frames()
    yield from idle_frames(360, "stage_open_wait")
