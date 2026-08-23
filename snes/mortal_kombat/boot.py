"""RAM-gated power-on → Liu Kang fight-ready.

No fixed-length mash. Each frame reads WRAM and picks START / d-pad / confirm
from the current screen class. Used by the continuous tournament runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from retro_harness.actions import snes_action
from mortal_kombat.ram import (
    LIU_KANG_ID,
    FightSnapshot,
    Screen,
    is_char_select,
    is_fight_ready,
    parse_ram,
)

_START_AFTER = 600  # skip Acclaim logos before mashing START
_PULSE = 20
_HOLD = 6


class Phase(IntEnum):
    """Boot state machine. FIGHT is the success terminal."""

    LOGOS = 0
    CHAR_SELECT = 1
    VS = 2
    FIGHT = 3
    CONTINUE = 4


@dataclass
class BootController:
    """Pure RAM policy: snapshot + frame → next phase and SNES buttons."""

    character: int = LIU_KANG_ID
    start_after: int = _START_AFTER

    def decide(self, snap: FightSnapshot, frame: int) -> tuple[Phase, tuple[str, ...]]:
        if snap.screen is Screen.FIGHT or is_fight_ready(snap, character=self.character):
            return Phase.FIGHT, ()
        if snap.screen is Screen.CREDITS:
            return Phase.FIGHT, ()
        if snap.screen is Screen.CONTINUE:
            return Phase.CONTINUE, _pulse(frame, "START")
        # Intra-match KO / "FIGHT!" intro auto-advances. START here pauses.
        if snap.screen is Screen.BETWEEN_ROUNDS:
            return Phase.FIGHT, ()
        if snap.screen is Screen.MENU:
            return Phase.VS, _pulse(frame, "START")
        if is_char_select(snap):
            return self._char_select(snap, frame)
        if frame < self.start_after:
            return Phase.LOGOS, ()
        return Phase.LOGOS, _pulse(frame, "START")

    def _char_select(
        self, snap: FightSnapshot, frame: int
    ) -> tuple[Phase, tuple[str, ...]]:
        """Top row: Cage, Kano, Raiden, Liu Kang. ``p1_character`` tracks the cursor."""
        if snap.p1_character == self.character:
            return Phase.CHAR_SELECT, _pulse(frame, "Y", "A", period=24, hold=8)
        # Observed cursor graph (hold ~8f): Cage --DOWN--> Kano --DOWN-->
        # Raiden --RIGHT--> Liu Kang. RIGHT from Cage falls to the bottom row.
        if snap.p1_character == 2:
            direction = "RIGHT"
        elif snap.p1_character in (0, 1):
            direction = "DOWN"
        else:
            direction = "LEFT"
        return Phase.CHAR_SELECT, _pulse(frame, direction, period=50, hold=8)


def _pulse(frame: int, *buttons: str, period: int = _PULSE, hold: int = _HOLD) -> tuple[str, ...]:
    if frame % period < hold:
        return buttons
    return ()


def action_from_buttons(names: tuple[str, ...]) -> np.ndarray:
    """Map named SNES buttons to a 12-button array."""
    if not names:
        return snes_action(dtype=np.int8)
    return snes_action(*names, dtype=np.int8)


def boot_to_fight(env, *, max_frames: int = 9000, character: int = LIU_KANG_ID) -> FightSnapshot:
    """Drive ``env`` from power-on (or any menu) until Liu Kang fight-ready.

    Raises ``TimeoutError`` if the fight never settles.
    """
    ctrl = BootController(character=character)
    last = parse_ram(env.unwrapped.get_ram())
    for frame in range(max_frames):
        last = parse_ram(env.unwrapped.get_ram())
        phase, names = ctrl.decide(last, frame)
        env.step(action_from_buttons(names))
        if phase is Phase.FIGHT and is_fight_ready(last, character=character):
            return last
    raise TimeoutError(
        f"boot did not reach Liu Kang fight in {max_frames} frames "
        f"(screen={last.screen.name} char={last.p1_character} "
        f"hp={last.p1_health}/{last.p2_health} timer={last.timer})"
    )
