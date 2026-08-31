"""Level 6 dark 0x29 clear leftover: finish west of the center block.

East leftover (184,144) cannot LEFT (tile 244). Occupancy chase of the
last east wizzrobe is why clear29 landed there. Peel LEFT from the north
mouth, then fight/patrol only west of x=64. Do not occupancy-DOWN or
LEFT at y=144/165.
"""

from __future__ import annotations

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action
from zelda_i.dungeon.engine import (
    DungeonRoomSpec,
    GenericDungeonRoomController,
)
from zelda_i.level6.dungeon import CLEAR29_WEST_X, ROOM_29_SPEC
from zelda_i.ram import ZeldaObject, ZeldaSnapshot

__all__ = [
    "Level6Clear29Controller",
    "make_clear29_controller",
]


class Level6Clear29Controller(GenericDungeonRoomController):
    """West-aisle 0x29 clear. Never chase east of CLEAR29_WEST_X."""

    def __init__(self, spec: DungeonRoomSpec | None = None) -> None:
        super().__init__(spec if spec is not None else ROOM_29_SPEC)

    def _combat(
        self, snap: ZeldaSnapshot, live: tuple[ZeldaObject, ...]
    ) -> FrameAction:
        if int(snap.link_x) >= CLEAR29_WEST_X:
            self.combat_frames += 1
            off_wall = self._off_wall_step(snap)
            if off_wall is not None:
                return off_wall
            dash = self.spec.combat.inland_dash
            if dash > 0 and self.combat_frames <= dash:
                if self.spec.combat.occupancy_patrol:
                    self.walker.last_dir = self.spec.entry.direction
                return self._swing(
                    self.spec.entry.direction,
                    "inland_dash",
                    period=self.spec.combat.engage_attack_period,
                    hold=self.spec.combat.engage_attack_hold,
                )
            if self.spec.combat.occupancy_patrol:
                self.walker.last_dir = "LEFT"
            return FrameAction(nes_action("LEFT"), "west_peel")
        west = tuple(obj for obj in live if obj.x < CLEAR29_WEST_X)
        return super()._combat(snap, west)


def make_clear29_controller() -> Level6Clear29Controller:
    """Occupancy-patrol 0x29 so leftover is west of x=64."""
    return Level6Clear29Controller()
