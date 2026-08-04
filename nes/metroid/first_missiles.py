"""Natural Maru Mari → first-missiles policy.

Target: missile capacity ``$687A > 0`` (and usually missiles-enabled
``$010E != 0``).

The verified prefix now runs from the morph pedestal through all three
east-corridor blue doors and onto the third stable platform in the west
shaft.  The remaining frontier is the enemy-populated upper shaft, bridge,
east-shaft descent, and missile pickup.

Probe notes (2026-07-28)
------------------------

* The reliable return is the low morph tunnel, not the visually tempting
  upper reversal: morph on the pedestal, roll east through (1,14)–(3,14),
  unmorph at x≈76, then jump into the real skree-start room.
* Door transitions must stop on their first controllable frame.  Continuing
  to hold RIGHT after the transition shifts enemy timing and costs health.
* The long tunnel and third-door room require fixed short jump/shot cadences.
  The verified prefix reaches the west shaft with 6 energy and no state load.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.input_script import FrameAction

from metroid.ram import is_missiles_obtained, read_missile_capacity, read_snapshot

SEGMENT_MAX_FRAMES = 12_000

Buttons = tuple[str, ...]
Span = tuple[Buttons, int, str]

_EAST_CORRIDOR_SPANS: tuple[Span, ...] = (
    (
        (("RIGHT", "A"), 60, "east_jump"),
        (("RIGHT",), 10, "east_release"),
    )
    * 10
)

_FIRST_DOOR_SPANS: tuple[Span, ...] = (
    ((), 30, "first_door_land"),
    (("LEFT",), 50, "first_door_backoff"),
    (("RIGHT", "A"), 40, "first_door_jump"),
    ((), 80, "first_door_platform_land"),
    (("RIGHT",), 10, "first_door_face"),
    (("B",), 1, "first_door_shoot"),
    ((), 3, "first_door_open_wait"),
)

_BRIDGE_TUNNEL_SPANS: tuple[Span, ...] = (
    (("DOWN",), 10, "bridge_morph"),
    (("RIGHT",), 155, "bridge_roll"),
    (("UP",), 1, "second_door_unmorph"),
    ((), 3, "second_door_stand"),
    (("B",), 1, "second_door_shoot"),
    ((), 3, "second_door_open_wait"),
)

_NATURAL_PEDESTAL_SPANS: tuple[Span, ...] = (
    (("LEFT",), 40, "natural_pedestal_backoff"),
    (("UP",), 1, "natural_pedestal_unmorph"),
    (("LEFT", "A"), 20, "natural_pedestal_jump"),
    ((), 10, "natural_pedestal_land"),
    (("DOWN",), 20, "natural_pedestal_remorph"),
)

_MISSILE_CORRIDOR_SPANS: tuple[Span, ...] = (
    (
        (("RIGHT", "A"), 60, "missile_corridor_jump"),
        (("RIGHT",), 10, "missile_corridor_release"),
    )
    * 17
    + (
        (("RIGHT", "A"), 10, "missile_corridor_last_jump"),
        ((), 100, "long_tunnel_land"),
    )
)

# Starts in morph form at map (10,14), x≈196/y≈200.  The short A taps alter
# enemy phase without leaving morph form until the final jump out of the wall.
_LONG_TUNNEL_SPANS: tuple[Span, ...] = (
    ((), 55, "long_tunnel_enemy_wait"),
    (("RIGHT",), 27, "long_tunnel_roll_1"),
    (("RIGHT", "A"), 3, "long_tunnel_tap_1"),
    (("RIGHT",), 9, "long_tunnel_roll_2"),
    (("RIGHT", "A"), 3, "long_tunnel_tap_2"),
    (("RIGHT",), 6, "long_tunnel_roll_3"),
    (("RIGHT", "A"), 30, "long_tunnel_jump_1"),
    (("RIGHT",), 48, "long_tunnel_roll_4"),
    (("RIGHT", "A"), 3, "long_tunnel_tap_3"),
    (("RIGHT",), 9, "long_tunnel_roll_5"),
    (("RIGHT", "A"), 3, "long_tunnel_tap_4"),
    (("RIGHT",), 9, "long_tunnel_roll_6"),
    (("A",), 3, "long_tunnel_unmorph"),
    (("RIGHT",), 30, "long_tunnel_exit"),
)

_THIRD_DOOR_SPANS: tuple[Span, ...] = (
    ((("B",), 1, "third_room_opening_shot"),)
    + ((("B",), 1, "third_room_shoot"), ((), 9, "third_room_shot_gap")) * 9
    + (
        (("RIGHT", "A"), 90, "third_door_jump"),
        ((), 80, "third_door_platform_land"),
        (("RIGHT",), 5, "third_door_face"),
        (("B",), 1, "third_door_shoot"),
        ((), 3, "third_door_open_wait"),
    )
)

_WEST_SHAFT_ENTRY_SPANS: tuple[Span, ...] = (
    (("RIGHT", "A"), 20, "west_shaft_jump_1"),
    ((), 40, "west_shaft_land_1"),
)

_WEST_SHAFT_AFTER_MORPH_SPANS: tuple[Span, ...] = (
    (("RIGHT", "A"), 44, "west_shaft_jump_2"),
    ((), 19, "west_shaft_land_2"),
    (("RIGHT",), 12, "west_shaft_edge"),
    (("A",), 28, "west_shaft_jump_3"),
    (("LEFT",), 12, "west_shaft_drift"),
    ((), 5, "west_shaft_land_3"),
)

_WEST_SHAFT_NATURAL_SPANS: tuple[Span, ...] = (
    (("RIGHT",), 1, "west_shaft_natural_align"),
    (("RIGHT", "A"), 44, "west_shaft_jump_2"),
    ((), 19, "west_shaft_land_2"),
    (("A",), 32, "west_shaft_jump_3"),
    (("LEFT",), 28, "west_shaft_drift"),
    ((), 1, "west_shaft_land_3"),
)


class MissilesPhase(Enum):
    MORPH_EXIT = auto()
    NATURAL_PEDESTAL = auto()
    RETURN_STAND = auto()
    EAST_CORRIDOR = auto()
    FIRST_DOOR = auto()
    FIRST_DOOR_ENTER = auto()
    BRIDGE_TUNNEL = auto()
    SECOND_DOOR_ENTER = auto()
    MISSILE_CORRIDOR = auto()
    LONG_TUNNEL_ENTRY = auto()
    LONG_TUNNEL = auto()
    THIRD_DOOR = auto()
    THIRD_DOOR_ENTER = auto()
    WEST_SHAFT = auto()
    FRONTIER = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class FirstMissilesController:
    """Stateful frame policy for the natural morph → missiles route."""

    phase: MissilesPhase = MissilesPhase.MORPH_EXIT
    frames: int = 0
    phase_frames: int = 0
    span_index: int = 0
    span_progress: int = 0
    last_x: int | None = None
    stable_x_frames: int = 0
    shaft_variant: str | None = None
    notes: list[str] = field(default_factory=list)
    success: bool = False
    # Level1 is useful for corridor regression probes.  It has no morph ball,
    # so that mode cannot finish the bridge tunnel.
    start_from_corridor: bool = False

    @property
    def terminal(self) -> bool:
        return self.phase in {
            MissilesPhase.FRONTIER,
            MissilesPhase.DONE,
            MissilesPhase.FAILED,
        }

    def reset(self) -> None:
        self.phase = (
            MissilesPhase.EAST_CORRIDOR
            if self.start_from_corridor
            else MissilesPhase.MORPH_EXIT
        )
        self.frames = 0
        self.phase_frames = 0
        self.span_index = 0
        self.span_progress = 0
        self.last_x = None
        self.stable_x_frames = 0
        self.shaft_variant = None
        self.notes.clear()
        self.success = False

    def _set_phase(self, phase: MissilesPhase, note: str = "") -> None:
        if phase is self.phase:
            return
        self.phase = phase
        self.phase_frames = 0
        self.span_index = 0
        self.span_progress = 0
        self.last_x = None
        self.stable_x_frames = 0
        self.shaft_variant = None
        if note:
            self.notes.append(note)

    def _run_spans(self, spans: tuple[Span, ...]) -> FrameAction | None:
        if self.span_index >= len(spans):
            return None
        buttons, hold, label = spans[self.span_index]
        action = nes_action(*buttons) if buttons else nes_idle_action()
        self.span_progress += 1
        if self.span_progress >= hold:
            self.span_index += 1
            self.span_progress = 0
        return FrameAction(action, label)

    def step(self, env: Any) -> FrameAction:
        self.frames += 1
        self.phase_frames += 1
        snap = read_snapshot(env.get_ram(), env=env)

        if is_missiles_obtained(env) or snap.missile_capacity > 0:
            self.success = True
            self._set_phase(MissilesPhase.DONE, "missiles_obtained")
            return FrameAction(nes_idle_action(), "done")

        if self.frames >= SEGMENT_MAX_FRAMES:
            self._set_phase(MissilesPhase.FAILED, "timeout")
            return FrameAction(nes_idle_action(), "timeout")

        if self.terminal:
            return FrameAction(nes_idle_action(), self.phase.name.lower())

        # Horizontal room scrolls and door animations use additional game-mode
        # values while still consuming held input.  Gating on mode 3 inserts
        # idle frames and changes both jump distance and enemy timing.
        if snap.engine_mode != 0:
            return FrameAction(nes_idle_action(), "mode_wait")

        if self.phase is MissilesPhase.MORPH_EXIT:
            return self._morph_exit(snap)
        if self.phase is MissilesPhase.NATURAL_PEDESTAL:
            return self._natural_pedestal(snap)
        if self.phase is MissilesPhase.RETURN_STAND:
            return self._return_stand(snap)
        if self.phase is MissilesPhase.EAST_CORRIDOR:
            return self._east_corridor(snap)
        if self.phase is MissilesPhase.FIRST_DOOR:
            return self._first_door(snap)
        if self.phase is MissilesPhase.FIRST_DOOR_ENTER:
            return self._enter_east(snap, 6, MissilesPhase.BRIDGE_TUNNEL)
        if self.phase is MissilesPhase.BRIDGE_TUNNEL:
            return self._bridge_tunnel(snap)
        if self.phase is MissilesPhase.SECOND_DOOR_ENTER:
            return self._enter_east(snap, 7, MissilesPhase.MISSILE_CORRIDOR)
        if self.phase is MissilesPhase.MISSILE_CORRIDOR:
            return self._missile_corridor(snap)
        if self.phase is MissilesPhase.LONG_TUNNEL_ENTRY:
            return self._long_tunnel_entry(snap)
        if self.phase is MissilesPhase.LONG_TUNNEL:
            return self._long_tunnel(snap)
        if self.phase is MissilesPhase.THIRD_DOOR:
            return self._third_door(snap)
        if self.phase is MissilesPhase.THIRD_DOOR_ENTER:
            return self._enter_east(snap, 11, MissilesPhase.WEST_SHAFT)
        if self.phase is MissilesPhase.WEST_SHAFT:
            return self._west_shaft(snap)
        return FrameAction(nes_idle_action(), "unexpected_phase")

    def _morph_exit(self, snap) -> FrameAction:
        """Roll through the low morph tunnel and reach the east opening."""
        if snap.map_x >= 4:
            self._set_phase(MissilesPhase.EAST_CORRIDOR, "skree_start_reached")
            return self._east_corridor(snap)

        if snap.samus_x == self.last_x:
            self.stable_x_frames += 1
        else:
            self.last_x = snap.samus_x
            self.stable_x_frames = 0

        if (
            snap.map_x == 1
            and snap.samus_status == 3
            and 135 <= snap.samus_x <= 145
            and snap.samus_y >= 184
            and self.stable_x_frames >= 3
        ):
            self._set_phase(
                MissilesPhase.NATURAL_PEDESTAL,
                "natural_pedestal_realign",
            )
            return self._natural_pedestal(snap)

        if (
            snap.map_x == 3
            and snap.samus_status == 3
            and 65 <= snap.samus_x <= 90
            and snap.samus_y >= 190
            and self.stable_x_frames >= 3
        ):
            # Exactly one UP frame unmorphs without climbing into the wall.
            self._set_phase(MissilesPhase.RETURN_STAND, "return_tunnel_cleared")
            return FrameAction(nes_action("UP"), "return_unmorph")

        if snap.samus_status != 3 and snap.map_x <= 2:
            return FrameAction(nes_action("DOWN"), "return_morph")
        return FrameAction(nes_action("RIGHT"), "return_roll")

    def _natural_pedestal(self, snap) -> FrameAction:
        action = self._run_spans(_NATURAL_PEDESTAL_SPANS)
        if action is not None:
            return action
        self._set_phase(MissilesPhase.MORPH_EXIT, "natural_pedestal_aligned")
        return self._morph_exit(snap)

    def _return_stand(self, snap) -> FrameAction:
        if snap.map_x >= 4 and snap.samus_x >= 140:
            self._set_phase(MissilesPhase.EAST_CORRIDOR, "skree_start_reached")
            return self._east_corridor(snap)
        return FrameAction(nes_action("RIGHT", "A"), "return_jump_out")

    def _east_corridor(self, snap) -> FrameAction:
        # Align both natural and Level1 probe entries at the verified cadence
        # origin before consuming the fixed spans.
        if (
            self.span_index == 0
            and self.span_progress == 0
            and (snap.map_x < 4 or (snap.map_x == 4 and snap.samus_x < 140))
        ):
            return FrameAction(nes_action("RIGHT", "A"), "east_align")
        action = self._run_spans(_EAST_CORRIDOR_SPANS)
        if action is not None:
            return action
        self._set_phase(MissilesPhase.FIRST_DOOR, "first_door_wall")
        return self._first_door(snap)

    def _first_door(self, snap) -> FrameAction:
        action = self._run_spans(_FIRST_DOOR_SPANS)
        if action is not None:
            return action
        self._set_phase(MissilesPhase.FIRST_DOOR_ENTER, "first_door_open")
        return self._enter_east(snap, 6, MissilesPhase.BRIDGE_TUNNEL)

    def _enter_east(
        self,
        snap,
        target_map_x: int,
        next_phase: MissilesPhase,
    ) -> FrameAction:
        if snap.map_x == target_map_x and snap.in_door == 0:
            self._set_phase(next_phase, f"entered_map_{target_map_x}")
            return self.step_action_for_phase(snap)
        return FrameAction(nes_action("RIGHT"), f"enter_map_{target_map_x}")

    def step_action_for_phase(self, snap) -> FrameAction:
        """Dispatch after an in-frame phase change without rereading RAM."""
        if self.phase is MissilesPhase.BRIDGE_TUNNEL:
            return self._bridge_tunnel(snap)
        if self.phase is MissilesPhase.MISSILE_CORRIDOR:
            return self._missile_corridor(snap)
        if self.phase is MissilesPhase.LONG_TUNNEL:
            return self._long_tunnel(snap)
        if self.phase is MissilesPhase.WEST_SHAFT:
            return self._west_shaft(snap)
        return FrameAction(nes_idle_action(), "phase_boundary")

    def _bridge_tunnel(self, snap) -> FrameAction:
        action = self._run_spans(_BRIDGE_TUNNEL_SPANS)
        if action is not None:
            return action
        self._set_phase(MissilesPhase.SECOND_DOOR_ENTER, "second_door_open")
        return self._enter_east(snap, 7, MissilesPhase.MISSILE_CORRIDOR)

    def _missile_corridor(self, snap) -> FrameAction:
        action = self._run_spans(_MISSILE_CORRIDOR_SPANS)
        if action is not None:
            return action
        self._set_phase(MissilesPhase.LONG_TUNNEL_ENTRY, "long_tunnel_wall")
        return self._long_tunnel_entry(snap)

    def _long_tunnel_entry(self, snap) -> FrameAction:
        action = self._run_spans(((("DOWN",), 10, "long_tunnel_morph"),))
        if action is not None:
            return action
        if snap.map_x == 10 and snap.samus_x >= 196:
            self._set_phase(MissilesPhase.LONG_TUNNEL, "long_tunnel_base")
            return self._long_tunnel(snap)
        return FrameAction(nes_action("RIGHT"), "long_tunnel_roll_in")

    def _long_tunnel(self, snap) -> FrameAction:
        action = self._run_spans(_LONG_TUNNEL_SPANS)
        if action is not None:
            return action
        self._set_phase(MissilesPhase.THIRD_DOOR, "long_tunnel_cleared")
        return self._third_door(snap)

    def _third_door(self, snap) -> FrameAction:
        action = self._run_spans(_THIRD_DOOR_SPANS)
        if action is not None:
            return action
        self._set_phase(MissilesPhase.THIRD_DOOR_ENTER, "third_door_open")
        return self._enter_east(snap, 11, MissilesPhase.WEST_SHAFT)

    def _west_shaft(self, snap) -> FrameAction:
        if self.shaft_variant is None:
            action = self._run_spans(_WEST_SHAFT_ENTRY_SPANS)
            if action is not None:
                return action
            self.shaft_variant = "natural" if snap.samus_x <= 65 else "after_morph"
            self.span_index = 0
            self.span_progress = 0

        spans = (
            _WEST_SHAFT_NATURAL_SPANS
            if self.shaft_variant == "natural"
            else _WEST_SHAFT_AFTER_MORPH_SPANS
        )
        action = self._run_spans(spans)
        if action is not None:
            return action
        if (
            snap.map_cell == (11, 13)
            and 90 <= snap.samus_x <= 125
            and snap.samus_y == 225
            and snap.samus_status == 0
            and snap.health_units > 0
        ):
            self._set_phase(MissilesPhase.FRONTIER, "west_shaft_upper_platform")
            return FrameAction(nes_idle_action(), "frontier")
        self._set_phase(MissilesPhase.FAILED, "west_shaft_landing_failed")
        return FrameAction(nes_idle_action(), "west_shaft_failed")

    def report(self) -> dict[str, object]:
        return {
            "phase": self.phase.name,
            "frames": self.frames,
            "success": self.success,
            "terminal": self.terminal,
            "span_index": self.span_index,
            "notes": list(self.notes),
        }


def missiles_segment_success(env: Any) -> bool:
    return is_missiles_obtained(env) or read_missile_capacity(env) > 0
