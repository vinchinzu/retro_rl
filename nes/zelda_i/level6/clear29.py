"""Level 6 dark 0x29 clear leftover: north-west of the island.

East leftover (184,144) cannot LEFT. SW leftover (56,157) cannot
UP/DOWN/RIGHT (south29 BLOCKED 6/6). LEFT at north mouth y=77 is the
door channel (red 3). DOWN inland from (120,77), then LEFT at y=109.
Fight the whole room, but never walk the west aisle south of the door
band (y=141). CLEAR_ONLY is not success until leftover is x<64 and
y<=133 (historical (55,133)).
"""

from __future__ import annotations

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.dungeon.engine import (
    DungeonPhase,
    DungeonRoomSpec,
    GenericDungeonRoomController,
)
from zelda_i.level6.dungeon import CLEAR29_WEST_X, ROOM_29_SPEC
from zelda_i.level6.overworld import LEVEL6, LEVEL6_DARK_29_ROOM
from zelda_i.ram import ZeldaObject, ZeldaSnapshot
from zelda_i.spine.hops import play_ready

__all__ = [
    "CLEAR29_COMBAT_Y",
    "CLEAR29_HANDOFF_Y",
    "Level6Clear29Controller",
    "clear29_handoff_ok",
    "make_clear29_controller",
]

# Historical leftover (55,133). Island SW trap is y=157.
CLEAR29_HANDOFF_Y = 133
CLEAR29_COMBAT_Y = 141
NORTH_BAND_Y = 109


def leftover_ok(snap: ZeldaSnapshot) -> bool:
    """West of the center block and north of the SW island trap."""
    return int(snap.link_x) < CLEAR29_WEST_X and int(snap.link_y) <= CLEAR29_HANDOFF_Y


def clear29_handoff_ok(snap: ZeldaSnapshot, **_: object) -> bool:
    """Spine stop: cleared 0x29 at a successor-safe leftover."""
    return play_ready(
        snap,
        level=LEVEL6,
        screen=LEVEL6_DARK_29_ROOM,
        spec=ROOM_29_SPEC,
        rod=True,
        tf_eq=0x1F,
    ) and leftover_ok(snap)


def _is_dir(action: FrameAction, direction: str) -> bool:
    return list(action.action) == list(nes_action(direction))


class Level6Clear29Controller(GenericDungeonRoomController):
    """Full-room 0x29 clear. Leftover must be x<64 and y<=133."""

    def __init__(self, spec: DungeonRoomSpec | None = None) -> None:
        super().__init__(spec if spec is not None else ROOM_29_SPEC)

    def _hold(self, direction: str, reason: str) -> FrameAction:
        self.combat_frames += 1
        if self.spec.combat.occupancy_patrol:
            self.walker.last_dir = direction
        return FrameAction(nes_action(direction), reason)

    def _combat(
        self, snap: ZeldaSnapshot, live: tuple[ZeldaObject, ...]
    ) -> FrameAction:
        x, y = int(snap.link_x), int(snap.link_y)
        # Red 3: LEFT at y=77 is the door channel. DOWN inland first.
        if x >= CLEAR29_WEST_X and y < NORTH_BAND_Y:
            return self._hold("DOWN", "north_inland")
        if x >= CLEAR29_WEST_X and y <= NORTH_BAND_Y:
            return self._hold("LEFT", "west_peel")
        # West aisle south of the door band is the irrecoverable SW trap.
        if x < CLEAR29_WEST_X and y > CLEAR29_COMBAT_Y:
            return self._hold("UP", "north_handoff")
        action = super()._combat(snap, live)
        if (
            x < CLEAR29_WEST_X
            and y >= CLEAR29_COMBAT_Y
            and _is_dir(action, "DOWN")
        ):
            if self.spec.combat.occupancy_patrol:
                self.walker.last_dir = "DOWN"
            return self._swing(
                "DOWN",
                "south_hold",
                period=self.spec.combat.engage_attack_period,
                hold=self.spec.combat.engage_attack_hold,
            )
        return action

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        action = super().step(snap)
        if not self.success or leftover_ok(snap):
            return action
        self.success = False
        if self.phase is DungeonPhase.DONE:
            self._set_phase(DungeonPhase.FIGHT, "unsafe_leftover")
        y = int(snap.link_y)
        if y > CLEAR29_HANDOFF_Y:
            return FrameAction(nes_action("UP"), "north_handoff")
        if int(snap.link_x) >= CLEAR29_WEST_X:
            return FrameAction(nes_action("LEFT"), "west_peel")
        return FrameAction(nes_idle_action(), "handoff_wait")


def make_clear29_controller() -> Level6Clear29Controller:
    """Clear 0x29; leftover is x<64 and y<=133."""
    return Level6Clear29Controller()
