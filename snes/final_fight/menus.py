"""Menu / boot helpers for reaching Stage 1 gameplay."""

from __future__ import annotations

from collections.abc import Iterator

from retro_harness.input_script import FrameAction, idle_frames, mash_button, mash_start


def title_advance_frames(
    *,
    pulses: int = 6,
    hold: int = 10,
    gap: int = 40,
) -> list[FrameAction]:
    """START pulses to skip logos / title into character select."""
    return mash_start(pulses=pulses, hold=hold, gap=gap)


def character_confirm_frames(
    *,
    hold: int = 12,
    gap: int = 30,
    pulses: int = 2,
) -> list[FrameAction]:
    """Confirm default character (Cody) on the select screen."""
    return mash_button(
        "START",
        pulses=pulses,
        hold=hold,
        gap=gap,
        hold_reason="char_confirm",
        wait_reason="char_wait",
    )


def boot_to_stage1_script() -> Iterator[FrameAction]:
    """Yield a conservative title → character-select → start script."""
    yield from title_advance_frames()
    yield from idle_frames(90, "post_title_idle")
    yield from character_confirm_frames()
    yield from idle_frames(300, "stage_open_wait")
