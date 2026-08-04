"""Menu helpers for reaching Mission 1 gameplay."""

from __future__ import annotations

from collections.abc import Iterator

from retro_harness.input_script import FrameAction, mash_start


def boot_to_stage1_script() -> Iterator[FrameAction]:
    """Pulse START through logos, title, and the default 1P mode.

    The boot probe stops consuming this script as soon as live player RAM is
    detected, so later pulses cannot pause Mission 1.
    """
    yield from mash_start(pulses=24, hold=5, gap=70)
