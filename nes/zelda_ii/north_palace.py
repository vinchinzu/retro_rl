"""Leave North Palace from Level1 (first side-scroll → overworld).

M3 isolated segment: hold LEFT until engine mode ``$0736`` is overworld
play (5). North Palace is two pages; the west door fires ~308f from
``Level1``, overworld play ~331f. Idle during 1–4 / 16 transitions so
Link stays on the palace tile instead of walking into the moat.
"""

from __future__ import annotations

from dataclasses import dataclass

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action

from zelda_ii.ram import TRANSITION_MODES, read_snapshot

SEGMENT_MAX_FRAMES = 900
SETTLE_FRAMES = 8


@dataclass
class LeavePalacePolicy:
    """Walk west out of North Palace; idle on transition and overworld."""

    def tick(self, ram) -> FrameAction:
        snap = read_snapshot(ram)
        if snap.dead:
            return FrameAction(nes_idle_action(), "dead")
        if snap.overworld:
            return FrameAction(nes_idle_action(), "clear_hold")
        if snap.engine_mode in TRANSITION_MODES:
            return FrameAction(nes_idle_action(), "transition")
        return FrameAction(nes_action("LEFT"), "walk_left")
