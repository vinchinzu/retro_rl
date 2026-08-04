"""Menu / boot helpers for reaching Stage 1 gameplay."""

from __future__ import annotations

from collections.abc import Iterator

from retro_harness.actions import buttons, idle_action
from retro_harness.input_script import FrameAction, mash_start


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
    out: list[FrameAction] = []
    for _ in range(pulses):
        for _ in range(hold):
            out.append(
                FrameAction(
                    action=buttons("START"), reason="char_confirm"
                )
            )
        for _ in range(gap):
            out.append(
                FrameAction(action=idle_action(), reason="char_wait")
            )
    return out


def boot_to_stage1_script() -> Iterator[FrameAction]:
    """Yield a conservative title → char-select → stage-open script."""
    yield from title_advance_frames()
    for _ in range(120):
        yield FrameAction(action=idle_action(), reason="post_title_idle")
    yield from character_confirm_frames()
    # Stage intro / spawn settle.
    for _ in range(360):
        yield FrameAction(action=idle_action(), reason="stage_open_wait")
