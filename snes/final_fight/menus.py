"""Menu / boot helpers for reaching Stage 1 gameplay."""

from __future__ import annotations

from collections.abc import Iterator

from retro_harness.actions import buttons, idle_action
from retro_harness.input_script import FrameAction, mash_start


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
    out: list[FrameAction] = []
    for _ in range(pulses):
        for _ in range(hold):
            out.append(
                FrameAction(action=buttons("START"), reason="char_confirm")
            )
        for _ in range(gap):
            out.append(FrameAction(action=idle_action(), reason="char_wait"))
    return out


def boot_to_stage1_script() -> Iterator[FrameAction]:
    """Yield a conservative title → character-select → start script."""
    yield from title_advance_frames()
    # Settle after title transition.
    for _ in range(90):
        yield FrameAction(action=idle_action(), reason="post_title_idle")
    yield from character_confirm_frames()
    # Wait through stage open / map / spawn.
    for _ in range(300):
        yield FrameAction(action=idle_action(), reason="stage_open_wait")
