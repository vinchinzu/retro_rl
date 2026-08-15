"""Deterministic reset-to-Level-1 sequence for Zelda I (NES)."""

from __future__ import annotations

from collections.abc import Iterator

from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.input_script import FrameAction

# Safety cap; compact path is ready ~199–210f. Fallback period loop if needed.
BOOT_MAX_FRAMES = 2000
# Fallback open-loop period (used only after the compact first-slot sequence).
# Period 50 still clears is_level1_ready; 40 fails menu debounce.
BOOT_PERIOD = 50

# --- Compact first-slot boot (TAS-adapted, chatterbox all-items #4767M) ---
# Title START after settle, START on empty slot 1 → name entry, one-letter
# name via SELECT cursor + A, START to confirm, START to begin game.
# No SELECT on the file-select screen (stays on first option = first quest).
# Name is one glyph, never "ZELDA" (that starts second quest).
# Verified ready at frame ~199–200 on PRG1 / fceumm.
BOOT_FILE_SLOT = 1
BOOT_QUEST = 1  # first playthrough; not the post-credits / ZELDA quest
_TITLE_SETTLE = 37
_FILE_GAP = 20
_NAME_PRE_GAP = 9
_POST_NAME_GAP = 15
_FIRST_PLAYTHROUGH_LOAD_WAIT = 180


def _idle(n: int, reason: str = "boot_wait") -> Iterator[FrameAction]:
    for _ in range(n):
        yield FrameAction(nes_idle_action(), reason)


def _press(button: str, frames: int = 1, reason: str = "boot") -> Iterator[FrameAction]:
    for _ in range(frames):
        yield FrameAction(nes_action(button), reason)


def boot_compact_first_slot_script() -> Iterator[FrameAction]:
    """Yield the fast power-on path: first file slot + short name + begin.

    First option on a blank file select is empty slot 1 (register + first
    quest). No file-menu SELECT cycling. Name entry moves the letter cursor
    with SELECT (not bare D-pad); we pick one letter then START to confirm.
    """
    yield from _idle(_TITLE_SETTLE)
    yield from _press("START", 1, "boot_title_start")
    yield from _idle(_FILE_GAP)
    # Empty slot 1 → name registration (mode 14).
    yield from _press("START", 1, "boot_slot1")
    yield from _idle(_NAME_PRE_GAP)
    # Name grid: SELECT steps the letter cursor; A commits one glyph; START ends.
    # Timing matches TAS (SEL+DOWN, SEL+DOWN, A, SELECT, START).
    yield FrameAction(nes_action("SELECT", "DOWN"), "boot_name_cursor")
    yield FrameAction(nes_idle_action(), "boot_wait")
    yield FrameAction(nes_action("SELECT", "DOWN"), "boot_name_cursor")
    yield FrameAction(nes_action("A"), "boot_name_letter")
    yield FrameAction(nes_action("SELECT"), "boot_name_cursor")
    yield FrameAction(nes_action("START"), "boot_name_confirm")
    yield from _idle(_POST_NAME_GAP)
    # Begin game on the newly registered first slot.
    yield from _press("START", 1, "boot_begin")


def boot_fallback_period_script() -> Iterator[FrameAction]:
    """Slower open-loop START/A/SELECT cycle if the compact path misses."""
    for frame in range(1, BOOT_MAX_FRAMES + 1):
        slot = frame % BOOT_PERIOD
        if 6 <= slot < 14:
            yield FrameAction(nes_action("START"), "boot_start")
        elif 22 <= slot < 28:
            yield FrameAction(nes_action("A"), "boot_confirm")
        elif 33 <= slot < 37:
            yield FrameAction(nes_action("SELECT"), "boot_select")
        else:
            yield FrameAction(nes_idle_action(), "boot_wait")


def boot_first_playthrough_script() -> Iterator[FrameAction]:
    """Power-on → first file option → first quest. No file-menu SELECT.

    The Survival spine uses this path only. Period fallback can press SELECT
    on the file menu and leave slot 1; that is not a fixed game choice.
    """
    yield from boot_compact_first_slot_script()
    yield from _idle(_FIRST_PLAYTHROUGH_LOAD_WAIT)


def boot_to_level1_script() -> Iterator[FrameAction]:
    """Yield title/file inputs toward overworld play (compact, then fallback)."""
    compact = list(boot_compact_first_slot_script())
    yield from compact
    # Pad with idles through the mode-3→5 overworld load (~100f), then fallback.
    # boot_to_ready early-exits on is_level1_ready, so this is only a safety net.
    remaining = BOOT_MAX_FRAMES - len(compact)
    if remaining <= 0:
        return
    # Prefer quiet wait first (game is loading); then period fallback.
    quiet = min(_FIRST_PLAYTHROUGH_LOAD_WAIT, remaining)
    yield from _idle(quiet)
    remaining -= quiet
    for i, fa in enumerate(boot_fallback_period_script()):
        if i >= remaining:
            break
        yield fa
