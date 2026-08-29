"""Level 1 mode-9 0x7F: walk onto the bow, exit stairs, return play 0x23.

Two-ladder passage (L6 rod analog). Warp pose (128,141) spits to the left
ladder (48,93). RIGHT at y=141 from x=48 is the pit (tile 250). DOWN to
y=189, RIGHT under the pit, UP the east ladder, LEFT onto (136,141).
Return via the left ladder (48,93). Do not poke ADDR_BOW. Clean M5 skips
this hop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.hop_controller import (
    CellarCross,
    HopController,
    WAIT_SCROLL_B,
    axis_dir,
    cellar_cross_dir,
)
from zelda_i.level1_bow import BOW22_MAX_FRAMES, LEVEL1_BOW_ROOM, make_bow22_controller
from zelda_i.level1_bow_cellar import (
    BOW_CELLAR_MAX_FRAMES,
    LEVEL1_BOW_CELLAR_ROOM,
    level1_bow_cellar_stages,
    make_bow_cellar_controller,
)
from zelda_i.level1_bow_rejoin import (
    REJOIN_MAX_FRAMES,
    make_bow_rejoin_controller,
)
from zelda_i.level1_finish import ROOM_KEY_GORIYA, level1_triforce_stages
from zelda_i.ram import PASSAGE_MODE, PLAY_MODE, ZeldaSnapshot
from zelda_i.screen_glance import BOW_PICKUP_LEAVE, GlanceLeftover, grade_controller

__all__ = [
    "BOW_PEDESTAL",
    "BOW_PICKUP_MAX_FRAMES",
    "EAST_X",
    "EXIT_STAIRS",
    "FLOOR_Y",
    "SETTLE_FRAMES",
    "WEST_X",
    "Level1BowPickupController",
    "level1_bow_detour_stages",
    "level1_bow_pickup_glance",
    "level1_bow_pickup_stages",
    "level1_bow_pickup_success",
    "level1_survival_tf_stages",
    "make_bow_pickup_controller",
]

LEVEL1 = 1
ALIGN = 2
SETTLE_FRAMES = 8
WEST_X = 48
FLOOR_Y = 189
EAST_X = 192
BOW_PEDESTAL = (136, 141)
# Left-ladder CheckWarps: x=48 is a multiple of $10; y=93 is $10k+$D.
EXIT_STAIRS = (WEST_X, 93)
EAST_DOOR = (224, 141)
# South-mouth RIGHT past x=208 is tile 223 (SE brick). Climb the east
# column to the door band, then RIGHT. x=208 UP from y=141 is the NE statue.
EAST_COLUMN = 208
CLIP_Y = 181
STAIRS_TILES = range(0x70, 0x74)
BOW_PICKUP_MAX_FRAMES = 6000
SAMPLE_PERIOD = 12
BOW_CROSS = CellarCross(
    west_x=WEST_X,
    east_x=EAST_X,
    floor_y=FLOOR_Y,
    mouth_y=BOW_PEDESTAL[1],
    tol=ALIGN,
)


class PickupPhase(Enum):
    SETTLE = auto()
    HUNT = auto()
    EXIT_STAIRS = auto()
    EAST_22 = auto()
    DONE = auto()
    FAILED = auto()


def make_bow_pickup_controller() -> "Level1BowPickupController":
    """Hunt the 0x7F pedestal, then stairs out to play 0x23. No poke."""
    return Level1BowPickupController()


def level1_bow_detour_stages():
    """KEY-LEFT + cellar + pickup/return + rejoin for backtrack44."""
    return (
        ("level1_bow_0x22", make_bow22_controller(), BOW22_MAX_FRAMES),
        ("level1_bow_cellar", make_bow_cellar_controller(), BOW_CELLAR_MAX_FRAMES),
        ("level1_bow_pickup", make_bow_pickup_controller(), BOW_PICKUP_MAX_FRAMES),
        ("level1_bow_rejoin", make_bow_rejoin_controller(), REJOIN_MAX_FRAMES),
    )


def level1_bow_pickup_stages():
    """Prefix through cellar stairs, then pickup + return to play 0x23."""
    return (
        *level1_bow_cellar_stages(),
        ("level1_bow_pickup", make_bow_pickup_controller(), BOW_PICKUP_MAX_FRAMES),
    )


def level1_survival_tf_stages():
    """Survival L1 TF with the bow detour after clear23_key. Not Clean M5."""
    from zelda_i.level1_east_dungeon import (
        ROOM_44_SURVIVAL_SPEC,
        Room44SurvivalController,
    )

    stages: list[Any] = []
    for item in level1_triforce_stages(natural_entry=True, survival=True):
        if item[0] == "clear44":
            spec = ROOM_44_SURVIVAL_SPEC
            stages.append(
                (item[0], Room44SurvivalController(spec), spec.max_frames)
            )
        else:
            stages.append(item)
        if item[0] == "clear23_key":
            stages.extend(level1_bow_detour_stages())
    return tuple(stages)


def level1_bow_pickup_success(snap: ZeldaSnapshot) -> bool:
    """Play 0x23 with walked ADDR_BOW. Reject cellar and 0x22."""
    return (
        snap.level == LEVEL1
        and snap.mode == PLAY_MODE
        and not snap.transitioning
        and snap.screen == ROOM_KEY_GORIYA
        and int(snap.bow) != 0
    )


def level1_bow_pickup_glance(controller) -> GlanceLeftover:
    """Play 0x23 west-mouth leftover after the bow cellar return."""
    return grade_controller(controller, BOW_PICKUP_LEAVE)


def _exit_stage(xy: tuple[int, int]) -> tuple[tuple[int, int], str, bool]:
    """Back to the left ladder. Do not LEFT at y=141 across the pit."""
    x, y = xy
    if abs(x - WEST_X) <= ALIGN and y > EXIT_STAIRS[1] + ALIGN:
        return EXIT_STAIRS, "exit_up", True
    if y < FLOOR_Y - ALIGN and abs(x - WEST_X) > ALIGN:
        if abs(x - EAST_X) > ALIGN:
            return (EAST_X, y), "exit_to_east", False
        return (EAST_X, FLOOR_Y), "exit_down_east", True
    if x > WEST_X + ALIGN:
        return (WEST_X, FLOOR_Y), "exit_west", False
    return EXIT_STAIRS, "exit_up", True


@dataclass
class Level1BowPickupController(HopController):
    """Settle mode 9, walk pedestal until ADDR_BOW, stairs, east to 0x23."""

    spec_id: str = "level1_bow_pickup"
    room: int = LEVEL1_BOW_CELLAR_ROOM
    max_frames: int = BOW_PICKUP_MAX_FRAMES
    wait_modes: tuple[int, ...] = WAIT_SCROLL_B
    done_reason: str = "arrived_23_bow"
    phase_frames: int = 0
    phase: PickupPhase = PickupPhase.SETTLE
    samples: list[dict[str, Any]] = field(default_factory=list)
    leftover: dict[str, Any] = field(default_factory=dict)
    spawned: bool = False

    def _set_phase(self, phase: PickupPhase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            if note:
                self.notes.append(note)

    def timeout_note(self, snap: ZeldaSnapshot) -> str:
        return (
            f"timeout_{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
            f"_mode={snap.mode}_bow={int(snap.bow)}"
        )

    def on_arrive(self, snap: ZeldaSnapshot) -> str:
        return f"bow_{snap.screen:02x}_{snap.link_x}_{snap.link_y}_mode={snap.mode}"

    def _fields(self, snap: ZeldaSnapshot) -> dict[str, Any]:
        return {
            "x": int(snap.link_x),
            "y": int(snap.link_y),
            "mode": int(snap.mode),
            "screen": int(snap.screen),
            "tile": int(snap.colliding_tile),
            "bow": int(snap.bow),
            "arrows": int(snap.arrows),
            "keys": int(snap.keys),
            "bombs": int(snap.bombs),
            "triforce": int(snap.triforce),
            "phase": self.phase.name,
        }

    def emit(
        self, snap: ZeldaSnapshot, action: FrameAction, *, force: bool = False
    ) -> FrameAction:
        if force or self.frames <= 2 or self.frames % SAMPLE_PERIOD == 0:
            self.samples.append(
                {
                    "frame": self.frames,
                    **self._fields(snap),
                    "reason": action.reason,
                }
            )
        self.leftover = self._fields(snap)
        return action

    def mark_fail(self, note: str, reason: str | None = None) -> FrameAction:
        self._set_phase(PickupPhase.FAILED, note)
        return super().mark_fail(note, reason)

    def mark_done(self, snap: ZeldaSnapshot, note: str | None = None) -> FrameAction:
        self._set_phase(PickupPhase.DONE, note or self.on_arrive(snap))
        return super().mark_done(snap, note)

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        return level1_bow_pickup_success(snap)

    def _walk(
        self,
        snap: ZeldaSnapshot,
        dest: tuple[int, int],
        reason: str,
        *,
        y_first: bool,
    ) -> FrameAction:
        btn = axis_dir(
            (int(snap.link_x), int(snap.link_y)), dest, y_first=y_first, tol=ALIGN
        )
        if btn is None:
            return FrameAction(nes_idle_action(), f"{reason}_idle")
        return FrameAction(nes_action(btn), reason)

    def _hunt(self, snap: ZeldaSnapshot) -> FrameAction:
        x, y = int(snap.link_x), int(snap.link_y)
        px, py = BOW_PEDESTAL
        # Cardinal UP at (192,189) did not climb (tile 243). L6 rod: clip
        # the east-ladder south face with RIGHT+UP; LEFT+UP if overshot.
        if y >= CLIP_Y and x >= EAST_X - 16:
            if x > EAST_X + ALIGN:
                return FrameAction(nes_action("LEFT", "UP"), "east_clip")
            return FrameAction(nes_action("RIGHT", "UP"), "east_clip")
        if abs(x - px) <= ALIGN and abs(y - py) <= ALIGN:
            return FrameAction(nes_idle_action(), "bow_stand_idle")
        if x >= EAST_X - 16:
            if y > py + ALIGN:
                return FrameAction(nes_action("UP"), "east_climb")
            if x > px + ALIGN:
                return FrameAction(nes_action("LEFT"), "pedestal")
            return FrameAction(nes_idle_action(), "bow_stand_idle")
        on_floor = y >= FLOOR_Y - ALIGN
        if on_floor or x <= WEST_X + 16:
            btn = cellar_cross_dir((x, y), BOW_CROSS, on_floor=on_floor)
            reason = {
                "DOWN": "west_floor",
                "RIGHT": "floor_east",
                "LEFT": "floor_east",
                "UP": "east_climb",
            }[btn]
            return FrameAction(nes_action(btn), reason)
        if x > px + ALIGN:
            return FrameAction(nes_action("LEFT"), "pedestal")
        return FrameAction(nes_idle_action(), "bow_stand_idle")

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        if snap.level == 0:
            return self.mark_fail(
                f"ow_early_{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
            )
        if snap.level != LEVEL1:
            return self.mark_fail(f"left_level_{snap.level}")

        if int(snap.bow) != 0 and self.phase in {
            PickupPhase.SETTLE,
            PickupPhase.HUNT,
        }:
            self._set_phase(
                PickupPhase.EXIT_STAIRS,
                f"got_bow_{snap.link_x}_{snap.link_y}_mode={snap.mode}",
            )

        if self.phase is PickupPhase.SETTLE:
            if snap.mode != PASSAGE_MODE:
                return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
            if int(snap.link_x) > WEST_X + 16:
                return FrameAction(nes_idle_action(), "wait_spawn")
            if not self.spawned:
                self.spawned = True
                self.phase_frames = 0
                self.notes.append(f"spit_{snap.link_x}_{snap.link_y}")
            if self.phase_frames < SETTLE_FRAMES or not snap.is_updating_mode:
                return FrameAction(nes_idle_action(), "settle_cellar")
            self._set_phase(
                PickupPhase.HUNT,
                f"hunt_{snap.link_x}_{snap.link_y}",
            )

        if self.phase is PickupPhase.HUNT:
            if snap.mode != PASSAGE_MODE:
                return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
            return self._hunt(snap)

        if self.phase is PickupPhase.EXIT_STAIRS:
            if snap.mode == PLAY_MODE and snap.screen == LEVEL1_BOW_ROOM:
                self._set_phase(
                    PickupPhase.EAST_22,
                    f"back_22_{snap.link_x}_{snap.link_y}",
                )
            elif snap.mode == PASSAGE_MODE:
                x, y = int(snap.link_x), int(snap.link_y)
                tile = int(snap.colliding_tile)
                # Left-ladder leftover (48,93) tile 0x6F never CheckWarps.
                # Hold UP until stairs $70–$73, then idle (X/Y already aligned).
                if abs(x - WEST_X) <= ALIGN and y <= EXIT_STAIRS[1] + ALIGN:
                    if tile in STAIRS_TILES:
                        return FrameAction(nes_idle_action(), "exit_warp")
                    return FrameAction(nes_action("UP"), "exit_up")
                # Cardinal DOWN at (192,141) is the pit (tile 250). Drop off
                # the east column with LEFT+DOWN (L6 exit analog).
                if y < FLOOR_Y - ALIGN and x >= EAST_X - 16:
                    return FrameAction(nes_action("LEFT", "DOWN"), "exit_drop")
                dest, reason, y_first = _exit_stage((x, y))
                return self._walk(snap, dest, reason, y_first=y_first)
            else:
                return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")

        if self.phase is PickupPhase.EAST_22:
            if snap.mode == PASSAGE_MODE:
                self._set_phase(
                    PickupPhase.EXIT_STAIRS,
                    f"reentered_cellar_{snap.link_x}_{snap.link_y}",
                )
                return FrameAction(nes_idle_action(), "reentered_cellar")
            if snap.mode != PLAY_MODE or snap.screen != LEVEL1_BOW_ROOM:
                return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
            x, y = int(snap.link_x), int(snap.link_y)
            # Live: y-first to 141 from (96,173) stepped on center stairs
            # tile 0x71 and CheckWarped back into 0x7F. Peel south first.
            # Live: RIGHT at y=189 past x=208 is SE brick tile 223.
            if x < EAST_COLUMN - ALIGN:
                if y < FLOOR_Y - ALIGN:
                    return FrameAction(nes_action("DOWN"), "east_peel")
                return FrameAction(nes_action("RIGHT"), "east_22")
            if y > EAST_DOOR[1] + ALIGN:
                return FrameAction(nes_action("UP"), "east_column")
            if x < EAST_DOOR[0] - ALIGN:
                return FrameAction(nes_action("RIGHT"), "east_22")
            return FrameAction(nes_action("RIGHT"), "enter_23")

        return FrameAction(nes_idle_action(), "failed")

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.phase_frames += 1
        return super().step(snap)

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "phase": self.phase.name,
            "notes": list(self.notes),
            "samples": list(self.samples),
            "policy": (
                "Wait left-ladder spit x=48; DOWN y=189; RIGHT under the "
                "pit to x=192; UP; LEFT onto (136,141); ADDR_BOW then east "
                "column DOWN, floor LEFT, left-ladder UP (48,93); play 0x22 "
                "RIGHT into 0x23; no poke"
            ),
            "leftover": dict(self.leftover),
            "spec_id": self.spec_id,
            "room": self.room,
        }
