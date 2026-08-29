"""Aggressive attack-and-reposition policy for Mission 1."""

from __future__ import annotations

from retro_harness.actions import buttons, idle_action
from retro_harness.bot_runner import BehaviorNode, NodeStatus, TickResult
from retro_harness.ram_state import GameMode, GameState
from retro_harness.input_script import FrameAction

Y_TOLERANCE = 5
ATTACK_RANGE_X = 22
EDGE_ATTACK_RANGE_X = 48
ATTACK_CYCLE = 14
# Mission 1's top-floor hall starts at its right edge after the elevator.
LEFTWARD_AREAS = frozenset({0x13})
AIRPORT_STAIRS_AREA = 0x15
AIRPORT_RUNWAY_AREA = 0x16
GYM_STAIRS_AREA = 0x19
# Runway actors shift between several screen-X projections while the plane
# scrolls.  A wider poke band prevents left/right oscillation around Billy.
AREA_ATTACK_RANGES = {0x16: 72, 0x17: 32}
BLOCK_COUNTER_AREAS = frozenset({0x19})


class Stage1Policy(BehaviorNode):
    """Fight rendered actors, including HP0 fighters, then walk right.

    Super Double Dragon maps B to block.  This first policy intentionally
    never emits B: it alternates punch (Y) and kick (A), with short gaps and
    lane repositioning.
    """

    name = "Stage1Policy"

    def __init__(self) -> None:
        self._combat_frame = 0
        self._stairs_frame = 0
        self._counter_frame = 0

    def reset(self) -> None:
        """Reset attack cadence for a fresh segment attempt."""
        self._combat_frame = 0
        self._stairs_frame = 0
        self._counter_frame = 0

    def _descend_airport_stairs(self, state: GameState) -> FrameAction:
        """Snake down Mission 2's spiral passage between combat locks."""
        # Long holds are intentional: the walkway wraps screen X while the
        # hidden spiral progress continues.  Short edge-based reversals never
        # reach the next landing.
        phase = self._stairs_frame % 2200
        self._stairs_frame += 1
        if phase < 800:
            return FrameAction(buttons("LEFT"), "stairs_left")
        if phase < 1200:
            return FrameAction(buttons("DOWN"), "stairs_down")
        if phase < 1800:
            return FrameAction(buttons("RIGHT"), "stairs_right")
        return FrameAction(buttons("DOWN"), "stairs_down")

    def _climb_gym_stairs(self, state: GameState) -> FrameAction:
        """Walk left to Mission 3's gym stairs, then up the diagonal."""
        # The dummy sits on the right; the staircase climbs up-left.  Walk
        # right never leaves 0x19.  Long holds match the airport spiral.
        phase = self._stairs_frame % 1600
        self._stairs_frame += 1
        if phase < 500:
            return FrameAction(buttons("LEFT"), "gym_left")
        if phase < 1100:
            return FrameAction(buttons("UP", "LEFT"), "gym_up_left")
        return FrameAction(buttons("UP"), "gym_up")

    def _attack(self) -> FrameAction:
        phase = self._combat_frame % ATTACK_CYCLE
        self._combat_frame += 1
        if phase < 4:
            return FrameAction(buttons("Y"), "punch")
        if 7 <= phase < 11:
            return FrameAction(buttons("A"), "kick")
        return FrameAction(idle_action(), "attack_gap")

    def _block_counter(self) -> FrameAction:
        """Break the Chin brothers' low-HP counter loop."""
        phase = self._counter_frame % 24
        self._counter_frame += 1
        if phase < 14:
            return FrameAction(buttons("B"), "counter_block")
        if phase < 20:
            return FrameAction(buttons("Y"), "counter_punch")
        return FrameAction(idle_action(), "counter_gap")

    def tick(self, state: GameState) -> TickResult:
        if state.player_dead or state.health == 0:
            action = FrameAction(idle_action(), "ko_wait")
        elif state.mode is not GameMode.PLAYING:
            action = FrameAction(idle_action(), "transition_wait")
        else:
            target = state.nearest_enemy()
            if target is None:
                self._combat_frame = 0
                if state.stage == AIRPORT_STAIRS_AREA:
                    action = self._descend_airport_stairs(state)
                elif state.stage == GYM_STAIRS_AREA:
                    action = self._climb_gym_stairs(state)
                elif state.stage == AIRPORT_RUNWAY_AREA:
                    action = FrameAction(
                        buttons("DOWN", "RIGHT"), "runway_down_right"
                    )
                elif state.stage in LEFTWARD_AREAS:
                    action = FrameAction(buttons("LEFT"), "walk_left")
                else:
                    action = FrameAction(buttons("RIGHT"), "walk_right")
            else:
                self._stairs_frame = 0
                dy = target.y - state.player_y
                dx = target.x - state.player_x
                attack_range = AREA_ATTACK_RANGES.get(
                    state.stage, ATTACK_RANGE_X
                )
                if (
                    state.player_x <= 35
                    and dx < 0
                    or state.player_x >= 220
                    and dx > 0
                ):
                    attack_range = EDGE_ATTACK_RANGE_X
                if abs(dy) > Y_TOLERANCE:
                    name = "UP" if dy < 0 else "DOWN"
                    action = FrameAction(
                        buttons(name),
                        "align_up" if name == "UP" else "align_down",
                    )
                elif abs(dx) > attack_range:
                    name = "RIGHT" if dx > 0 else "LEFT"
                    action = FrameAction(
                        buttons(name),
                        "approach_right" if name == "RIGHT" else "approach_left",
                    )
                else:
                    if (
                        state.stage in BLOCK_COUNTER_AREAS
                        and target.health <= 10
                    ):
                        action = self._block_counter()
                    else:
                        action = self._attack()
        return TickResult(status=NodeStatus.RUNNING, action=action, reason=self.name)


def build_stage1_tree() -> Stage1Policy:
    """Return the Mission 1 behavior root (API parity with ladder games)."""
    return Stage1Policy()
