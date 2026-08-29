"""Menu / boot helpers for reaching Stage 1 gameplay."""

from __future__ import annotations

from collections.abc import Iterator

from retro_harness.actions import buttons, idle_action
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


# Frame-accurate real-menu boot. Two DOWN presses enter Options, RIGHT
# changes Level to Hard, two UP presses return to 1 Player, RIGHT bursts
# select Raphael, and the last START confirms him. Do not merge this with
# boot_to_stage1_script — that plan mashes START onto default Leonardo.
RAPH_HARD_BOOT_ACTIONS: dict[int, tuple[str, ...]] = {
    300: ("START",),
    700: ("DOWN",),
    720: ("DOWN",),
    750: ("START",),
    950: ("RIGHT",),
    1000: ("START",),
    1200: ("UP",),
    1220: ("UP",),
    1250: ("START",),
    1440: ("RIGHT",),
    1441: ("RIGHT",),
    1442: ("RIGHT",),
    1443: ("RIGHT",),
    1444: ("RIGHT",),
    1452: ("RIGHT",),
    1453: ("RIGHT",),
    1454: ("RIGHT",),
    1455: ("RIGHT",),
    1456: ("RIGHT",),
    1464: ("RIGHT",),
    1465: ("RIGHT",),
    1466: ("RIGHT",),
    1467: ("RIGHT",),
    1468: ("RIGHT",),
    1490: ("START",),
}
RAPH_HARD_BOOT_LAST = max(RAPH_HARD_BOOT_ACTIONS)


def raph_hard_boot_action(frame: int) -> list[int]:
    """Return the scheduled Raphael Hard menu input for one power-on frame."""
    names = RAPH_HARD_BOOT_ACTIONS.get(frame)
    return buttons(*names) if names else idle_action()
