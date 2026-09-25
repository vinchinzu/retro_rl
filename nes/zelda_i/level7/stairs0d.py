"""Level 7 play 0x0D walk-on of the hidden staircase into cellar 0x7B.

Room 0x0D is a ring. Measured with `zelda_i.dungeon.tilemap` (the cart-WRAM
`$6530` room tile map), interior 16x16 cells:

```
        x=  32  48  64  80  96 112 128 144 160 176 192 208
 y= 96       .   .   .   .   .   .   .   .   .   .   .   S   <- stairs (post-push)
 y=112       .   #   #   #   #   #   #   #   #   #   #   .
 y=128       .   .   .   .   .   .   .   .   .   .   #   .
 y=144       .   .   .   .   .   .   .   .   .   .   B   .   <- west door at (16,144)
 y=160       .   .   .   .   .   .   .   .   .   .   #   .
 y=176       .   #   #   #   #   #   #   #   #   #   #   .
 y=192       .   .   .   .   .   .   .   .   .   .   .   .
```

`x=32` and `x=208` are the **only** two crossings of the `y=112` / `y=176`
solid bands. Every earlier sitting clamped to `INLAND_X = (64, 192)` — a
stale wallmaster-grab guard carried over from the *uncleared* recon — and so
excluded both corridors; that guard, not the terrain, was the blocker.

`x=192` is never a corridor: static blocks at `(192,128)` and `(192,160)`
sandwich the pushable `0x68` at `(192,144)`. The residual's "(192,133) plug,
tile 179" is just the static block at `(192,128)`.

The ROM secret is `block_stairs` (AttrE low nibble 5): before the push
`stair_cells()` is empty; the verified 16px RIGHT push slides the block to
the `(208,144)` cell and writes the staircase quad `70 72 / 71 73` into
`(208, 96)`. **The 0x68 object's RAM `x`/`y` is then repointed to the
stairs**, which is where the old "the block snaps to (208,96)" belief came
from — the block itself really moves one tile RIGHT.

Route: RIGHT push, west along the `y=128` row (never `y=144`, the west door
at `(16,144)` exits to 0x79), UP the `x=32` column, east along the `y=96`
row onto `(208,93)` -> mode 9 cellar 0x7B, right/B ladder spawn `(192,93)`.

No position/door/key/Triforce/mode/facing writes; `position_writes` stays 0.
OccupancyWalker is banned in this room — do not import it here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.combat import nearest_enemy
from zelda_i.dungeon.behaviors import WALLMASTER_TYPE
from zelda_i.dungeon.hop_controller import HopController, WAIT_SCROLL_B
from zelda_i.dungeon.ids import CLOCK_DROP_OBJECT_TYPE, CLOCK_DROP_STATE
from zelda_i.level7.stairs import NOSE_CELLAR_ROM, TIP_OF_NOSE_ROM
from zelda_i.ram import PASSAGE_MODE, PLAY_MODE, ZeldaObject, ZeldaSnapshot

__all__ = [
    "ALIGN",
    "BLOCK_OBJECT_TYPE",
    "CELLAR_ROOM",
    "DOOR_ROW_TOL",
    "DOOR_ROW_Y",
    "EAST_COLUMN_X",
    "PUSH_MOVED_PX",
    "RAM_CLAIM",
    "ROOM",
    "ROW_Y",
    "STAIR_CELL",
    "STAIRS_0D_MAX_FRAMES",
    "STALL_FRAMES",
    "TIP_BLOCK_STAND",
    "TIP_BLOCK_XY",
    "TIP_BLOCK_CELL_AFTER_RIGHT",
    "WEST_COLUMN_X",
    "WEST_DOOR_GUARD_X",
    "Level7Stairs0DController",
    "ROOM_0D_BLOCK",
    "ROOM_0D_BLOCK_AFTER_RIGHT",
    "ROOM_0D_BLOCK_CELL_AFTER_RIGHT",
    "ROOM_0D_BLOCK_STAND",
    "ROOM_0D_NORTH_ARM",
    "ROOM_0D_STAIR_CELL",
    "ROOM_0D_STAIR_WARP_HYP",
    "Room0DClearController",
    "Stairs0DPhase",
    "level7_stairs0d_stages",
    "level7_stairs0d_success",
    "make_stairs0d_controller",
    "room_0d_clear_step",
]

LEVEL7 = 7
ROOM = TIP_OF_NOSE_ROM  # 0x0D
CELLAR_ROOM = NOSE_CELLAR_ROM  # 0x7B
BLOCK_OBJECT_TYPE = 0x68
STAIRS_0D_MAX_FRAMES = 2400
_SAMPLE_PERIOD = 12
ALIGN = 1
PUSH_MOVED_PX = 16

# Link's stored y for each 16x16 cell row (cell top - 3).
ROW_Y: dict[int, int] = {
    96: 93, 112: 109, 128: 125, 144: 141, 160: 157, 176: 173, 192: 189,
}
WEST_COLUMN_X = 32
EAST_COLUMN_X = 208
# The one row that holds the west doorway; LEFT here leaves 0x0D for 0x79.
DOOR_ROW_Y = ROW_Y[144]
DOOR_ROW_TOL = 8
# Refuse LEFT this far east of the west wall while on the door row.
WEST_DOOR_GUARD_X = WEST_COLUMN_X + 16
# Where the travel legs turn: west of the y=128 row's east end.
TURN_WEST_X = 160
TIP_BLOCK_XY = (192, 144)
TIP_BLOCK_STAND = (176, 144)
# Tile-map truth after the RIGHT push (object RAM instead reports the stairs).
TIP_BLOCK_CELL_AFTER_RIGHT = (208, 144)
STAIR_CELL = (208, 96)
PHASE_MAX_FRAMES = 700
# Frames of no movement inside one single-axis leg before the hop fails.
STALL_FRAMES = 60

# Written before the first live trial (rr-8t4.3, 2026-09-04).
RAM_CLAIM = (
    "From Level7Interior0DClearedReconFixture (play 0x0D mode 5 (63,149)): "
    "verified 16px RIGHT push of the 0x68 at (192,144), then LEFT to x<=160, "
    "UP to the y=128 row (Link y=125), LEFT to x=32, UP to y=93, RIGHT along "
    "the y=96 row to x=208. Stepping onto (208,93) enters cellar 0x7B mode 9. "
    "Miss if LEFT pins at x>34, or UP at x=32 pins at y>=101, or Link stands "
    "at (208,93) in play 0x0D with no mode change. Never LEFT on the y=141 "
    "door row at x<=32 (that exits to 0x79)."
)


def tip_block(snap: ZeldaSnapshot) -> ZeldaObject | None:
    """The 0x68 nearest the known `(192,144)` cell. Ignore other movers."""
    blocks = [
        obj
        for obj in snap.objects
        if 1 <= int(obj.slot) <= 12 and int(obj.type_id) == BLOCK_OBJECT_TYPE
    ]
    if not blocks:
        return None
    bx, by = TIP_BLOCK_XY
    return min(
        blocks, key=lambda o: abs(int(o.x) - bx) + abs(int(o.y) - by)
    )


def level7_stairs0d_success(snap: ZeldaSnapshot) -> bool:
    """Mode 9 in cellar 0x7B. A play room is never success for this hop."""
    return (
        int(snap.level) == LEVEL7
        and int(snap.mode) == PASSAGE_MODE
        and int(snap.screen) == CELLAR_ROOM
    )


class Stairs0DPhase(Enum):
    """One cardinal per phase.

    Never mix axis corrections inside a phase: Zelda re-snaps Link's `y` when
    he walks horizontally, so an "align y then press RIGHT" loop oscillates
    forever instead of moving (burned once, `20260904_S1`).
    """

    PUSH_UP = auto()
    PUSH_EAST = auto()
    PUSH_ALIGN = auto()
    PUSH_HOLD = auto()
    PEEL_WEST = auto()
    NORTH_ROW = auto()
    WEST_COLUMN = auto()
    NORTH_COLUMN = auto()
    EAST_TOP = auto()
    DONE = auto()
    FAILED = auto()


@dataclass(kw_only=True)
class Level7Stairs0DController(HopController):
    """0x0D RIGHT push then ring walk onto the `(208,96)` staircase."""

    spec_id: str = "level7_tip_of_nose_stairs"
    max_frames: int = STAIRS_0D_MAX_FRAMES
    require_level: int = LEVEL7
    wait_modes: tuple[int, ...] = WAIT_SCROLL_B
    done_reason: str = "cellar_0x7b"
    dest: int = CELLAR_ROOM
    route_eligible: bool = False
    phase: Stairs0DPhase = Stairs0DPhase.PUSH_UP
    phase_frames: int = 0
    writes: int = 0
    block_x0: int | None = None
    block_y0: int | None = None
    stall_xy: tuple[int, int] | None = None
    stall_frames: int = 0
    leftover: dict[str, Any] = field(default_factory=dict)
    samples: list[dict[str, Any]] = field(default_factory=list)
    position_assist: dict[str, Any] = field(
        default_factory=lambda: {"position_writes": 0, "progression_writes": 0}
    )

    @property
    def stage_id(self) -> str:
        return self.spec_id

    def _set_phase(self, phase: Stairs0DPhase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            self.stall_xy = None
            self.stall_frames = 0
            if note:
                self._note(note)

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        return level7_stairs0d_success(snap)

    def on_arrive(self, snap: ZeldaSnapshot) -> str:
        return f"cellar_0x{snap.screen:02x}_{snap.link_x}_{snap.link_y}"

    def mark_fail(self, note: str, reason: str | None = None) -> FrameAction:
        self._set_phase(Stairs0DPhase.FAILED, note)
        return super().mark_fail(note, reason)

    def mark_done(
        self, snap: ZeldaSnapshot, note: str | None = None
    ) -> FrameAction:
        self._set_phase(Stairs0DPhase.DONE, note or self.on_arrive(snap))
        return super().mark_done(snap, note)

    def emit(
        self, snap: ZeldaSnapshot, action: FrameAction, *, force: bool = False
    ) -> FrameAction:
        block = tip_block(snap)
        self.leftover = {
            "x": int(snap.link_x),
            "y": int(snap.link_y),
            "mode": int(snap.mode),
            "screen": int(snap.screen),
            "tile": int(snap.colliding_tile),
            "phase": self.phase.name,
            "bx": None if block is None else int(block.x),
            "by": None if block is None else int(block.y),
            "keys": int(snap.keys),
            "bombs": int(snap.bombs),
            "candle": int(snap.candle),
            "triforce": int(snap.triforce),
        }
        if force or self.frames <= 2 or self.frames % _SAMPLE_PERIOD == 0:
            self.samples.append(
                {
                    "frame": self.frames,
                    "x": int(snap.link_x),
                    "y": int(snap.link_y),
                    "mode": int(snap.mode),
                    "screen": int(snap.screen),
                    "phase": self.phase.name,
                    "reason": action.reason,
                    "tile": int(snap.colliding_tile),
                    "bx": None if block is None else int(block.x),
                    "by": None if block is None else int(block.y),
                }
            )
        return action

    def timeout_note(self, snap: ZeldaSnapshot) -> str:
        return (
            f"timeout_0x{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
            f"_mode={snap.mode}_phase={self.phase.name}"
        )

    def _left(self, snap: ZeldaSnapshot, reason: str) -> FrameAction:
        """LEFT with the west-door guard. Never step into `(16,144)`.

        One cell of margin: a leg that drifted onto the door row fails at
        `x=48` instead of walking out of 0x0D into 0x79. The live route
        crosses west on the y=128 row, so this never fires in the green run.
        """
        if (
            abs(int(snap.link_y) - DOOR_ROW_Y) <= DOOR_ROW_TOL
            and int(snap.link_x) <= WEST_DOOR_GUARD_X
        ):
            return self.mark_fail(
                f"west_door_row_left_{snap.link_x}_{snap.link_y}"
            )
        return FrameAction(nes_action("LEFT"), reason)

    def _block_moved(self, snap: ZeldaSnapshot) -> bool:
        """True once the 0x68 has left its start cell.

        After the push the object's RAM x/y is repointed to the revealed
        stairs `(208,96)`, so this reads as a jump, not a clean 16px slide.
        """
        block = tip_block(snap)
        if block is None or self.block_x0 is None:
            return False
        return (
            abs(int(block.x) - int(self.block_x0)) >= PUSH_MOVED_PX
            or abs(int(block.y) - int(self.block_y0 or 0)) >= PUSH_MOVED_PX
        )

    def _stalled(self, snap: ZeldaSnapshot) -> bool:
        xy = (int(snap.link_x), int(snap.link_y))
        if xy == self.stall_xy:
            self.stall_frames += 1
        else:
            self.stall_xy = xy
            self.stall_frames = 0
        return self.stall_frames >= STALL_FRAMES

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        if int(snap.mode) != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if int(snap.screen) != ROOM:
            return self.mark_fail(f"left_0x0d_to_0x{snap.screen:02x}")
        if self.phase_frames > PHASE_MAX_FRAMES:
            return self.mark_fail(
                f"phase_{self.phase.name.lower()}_over_budget"
                f"_{snap.link_x}_{snap.link_y}"
            )
        x, y = int(snap.link_x), int(snap.link_y)
        block = tip_block(snap)
        if block is None:
            return self.mark_fail("no_block_0x68")
        if self.block_x0 is None:
            self.block_x0, self.block_y0 = int(block.x), int(block.y)
            self._note(f"block_{self.block_x0}_{self.block_y0}")
        stand_x = int(self.block_x0) - 16
        stand_y = int(self.block_y0)

        if self.phase is Stairs0DPhase.PUSH_UP:
            # Rise onto the block's row before travelling east; a mid-row y
            # would re-snap on the first RIGHT and oscillate.
            if y > ROW_Y[144]:
                return FrameAction(nes_action("UP"), "push_row_up")
            self._set_phase(Stairs0DPhase.PUSH_EAST, f"push_row_{x}_{y}")

        if self.phase is Stairs0DPhase.PUSH_EAST:
            if x < stand_x:
                if self._stalled(snap):
                    return self.mark_fail(f"push_east_stall_{x}_{y}")
                return FrameAction(nes_action("RIGHT"), "push_approach")
            self._set_phase(Stairs0DPhase.PUSH_ALIGN, f"push_stand_{x}_{y}")

        if self.phase is Stairs0DPhase.PUSH_ALIGN:
            if y < stand_y - ALIGN:
                return FrameAction(nes_action("DOWN"), "push_align_down")
            if y > stand_y + ALIGN:
                return FrameAction(nes_action("UP"), "push_align_up")
            self._set_phase(Stairs0DPhase.PUSH_HOLD, f"push_aligned_{x}_{y}")

        if self.phase is Stairs0DPhase.PUSH_HOLD:
            if not self._block_moved(snap):
                return FrameAction(nes_action("RIGHT"), "push_block")
            self._set_phase(
                Stairs0DPhase.PEEL_WEST,
                f"pushed_to_{int(block.x)}_{int(block.y)}",
            )

        if self.phase is Stairs0DPhase.PEEL_WEST:
            if x > TURN_WEST_X:
                if self._stalled(snap):
                    return self.mark_fail(f"peel_west_stall_{x}_{y}")
                return self._left(snap, "peel_west")
            self._set_phase(Stairs0DPhase.NORTH_ROW, f"peeled_{x}_{y}")

        if self.phase is Stairs0DPhase.NORTH_ROW:
            if y > ROW_Y[128]:
                if self._stalled(snap):
                    return self.mark_fail(f"north_row_stall_{x}_{y}")
                return FrameAction(nes_action("UP"), "up_to_row128")
            self._set_phase(Stairs0DPhase.WEST_COLUMN, f"row128_{x}_{y}")

        if self.phase is Stairs0DPhase.WEST_COLUMN:
            if x > WEST_COLUMN_X:
                if self._stalled(snap):
                    return self.mark_fail(f"west_column_stall_{x}_{y}")
                return self._left(snap, "west_column")
            self._set_phase(Stairs0DPhase.NORTH_COLUMN, f"west_{x}_{y}")

        if self.phase is Stairs0DPhase.NORTH_COLUMN:
            if y > ROW_Y[96]:
                if self._stalled(snap):
                    return self.mark_fail(f"north_column_stall_{x}_{y}")
                return FrameAction(nes_action("UP"), "north_column")
            self._set_phase(Stairs0DPhase.EAST_TOP, f"top_row_{x}_{y}")

        if self.phase is Stairs0DPhase.EAST_TOP:
            if self._stalled(snap):
                return self.mark_fail(f"east_top_stall_{x}_{y}")
            return FrameAction(nes_action("RIGHT"), "east_top_row")

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
            "spec_id": self.spec_id,
            "stage_id": self.spec_id,
            "room": ROOM,
            "dest": self.dest,
            "dest_screen": self.dest,
            "door": "STAIRS",
            "policy": RAM_CLAIM,
            "evidence": "fixture-live",
            "route_eligible": False,
            "natural_entry": False,
            "writes": int(self.writes),
            "stair_cell": list(STAIR_CELL),
            "block_after_right": list(TIP_BLOCK_CELL_AFTER_RIGHT),
            "leftover": dict(self.leftover),
            "position_assist": dict(self.position_assist),
        }


def make_stairs0d_controller() -> Level7Stairs0DController:
    """0x0D RIGHT push then ring walk onto `(208,96)`. No position writes."""
    return Level7Stairs0DController()


def level7_stairs0d_stages() -> tuple[tuple[str, Any, int], ...]:
    """One stage: 0x0D leftover -> live push -> ring walk -> cellar 0x7B."""
    ctl = make_stairs0d_controller()
    return (("level7_tip_of_nose_stairs", ctl, ctl.max_frames),)


_SWING_PERIOD = 8
_SWING_HOLD = 4
_0D_RING_X = 36  # the outer ring: a Wallmaster emerges at once
ROOM_0D = ROOM

ROOM0D_CLEAR_MAX_FRAMES = 30000
# 0x68 west-face stand. A 16px RIGHT slide is autonomous once it starts
# (1px/2f at y=144) and the 16th pixel snaps the object to (208,96) on
# the NE stair hole. Stairs dest $EB still unobserved (0d_push_v20/v22).
# SOLVED 2026-09-04 (rr-8t4.3): the walk-on lives in `level7.stairs0d`.
# 0x0D is a ring; x=32 and x=208 are the only crossings of the y=112 /
# y=176 solid bands, and x=192 is never a corridor (static blocks at
# (192,128)/(192,160) sandwich the pushable). ROOM_0D_NORTH_ARM (176,117)
# is boxed because the whole y=112 band is solid, not because of a snap.
ROOM_0D_BLOCK = (192, 144)
ROOM_0D_BLOCK_STAND = (176, 144)
# RAM-observed value of the 0x68 object after the RIGHT push. It is an
# artifact: the tile map shows the block quad at (208,144) and the freshly
# revealed staircase at (208,96), and the object's x/y is repointed to the
# stairs. `ROOM_0D_BLOCK_CELL_AFTER_RIGHT` is the real post-push cell.
ROOM_0D_BLOCK_AFTER_RIGHT = (208, 96)
ROOM_0D_BLOCK_CELL_AFTER_RIGHT = (208, 144)
ROOM_0D_STAIR_CELL = (208, 96)
ROOM_0D_NORTH_ARM = (176, 117)
# Live 2/2: stepping onto (208,93) after the push enters cellar 0x7B mode 9.
ROOM_0D_STAIR_WARP_HYP = (208, 93)
# Plus-corner 0x27 park here until they peel to the west wall one-at-a-time.
_0D_STATUE_XY = frozenset({(128, 125), (128, 157), (160, 125), (160, 157)})
_0D_NUDGE = (52, 117)
# Wait beside the column the Wallmaster climbs, not 44 px inland: from
# (96,141) most withdrew into the wall before reaching Link. Over 12 offsets
# x 3 Blue Ring power-on pins (14/10/3): 11876f (sd 5298) -> 3597f (sd 1682),
# 36/36, 0 hearts; (48,117) and (64,117) were within noise.
_0D_HOME = (56, 117)


def _0d_spawners(snap: ZeldaSnapshot) -> tuple[tuple[ZeldaObject, ...], tuple[ZeldaObject, ...]]:
    live: list[ZeldaObject] = []
    parked: list[ZeldaObject] = []
    for obj in snap.objects:
        if not (1 <= int(obj.slot) <= 12):
            continue
        if (int(obj.type_id) & 0xFF) != WALLMASTER_TYPE or int(obj.hp) <= 0:
            continue
        xy = (int(obj.x), int(obj.y))
        if xy in _0D_STATUE_XY:
            continue
        in_floor = 12 <= int(obj.x) <= 200 and 80 <= int(obj.y) <= 200
        if int(obj.state) != 0 and in_floor:
            live.append(obj)
        else:
            parked.append(obj)
    return tuple(live), tuple(parked)


@dataclass(kw_only=True)
class Room0DClearController(HopController):
    """0x0D TIP_OF_NOSE: spawn/kill 5 wallmasters.  x=32 any y grabs.

    Plus-corner 0x27 peel to the west wall one-at-a-time (not statues).
    Nudge x≈52 y=117, slash LEFT at x=48, peel inland after each kill.
    2/2 room_all_dead (0d_wm_v10 / 0d_cleared).  Recon-wired only.
    The stairs walk-on that follows is `level7.stairs0d` (live 2/2):
    16px RIGHT on the 0x68 (192,144) reveals the staircase at the
    (208,96) cell, then the ring walk x=32 UP / y=96 east reaches it.
    """

    spec_id: str = "level7_room0d_clear"
    max_frames: int = ROOM0D_CLEAR_MAX_FRAMES
    require_level: int = LEVEL7
    done_reason: str = "left_0x0d_cleared"
    dest: int | None = ROOM_0D
    _phase: str = "to_nudge"
    _phase_n: int = 0

    @property
    def stage_id(self) -> str:
        return self.spec_id

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        return (
            snap.screen == ROOM_0D
            and snap.mode == PLAY_MODE
            and not snap.transitioning
            and int(snap.room_all_dead) != 0
        )

    def on_arrive(self, snap: ZeldaSnapshot) -> str:
        return f"cleared_0x{snap.screen:02x}_{snap.link_x}_{snap.link_y}"

    def timeout_note(self, snap: ZeldaSnapshot) -> str:
        return (
            f"timeout_0x{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
            f"_dead={snap.room_all_dead}_phase={self._phase}"
        )

    def scroll_action(self, snap: ZeldaSnapshot) -> FrameAction:
        return FrameAction(nes_action("RIGHT"), "clear0d_scroll")

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        if snap.screen != ROOM_0D:
            return self.mark_fail(f"unexpected_room_0x{snap.screen:02x}")
        return room_0d_clear_step(snap, self)

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "spec_id": self.spec_id,
            "stage_id": self.spec_id,
            "dest_screen": self.dest,
            "evidence": "fixture-live",
            "route_eligible": False,
            "door": "CLEAR",
        }


def room_0d_clear_step(snap: ZeldaSnapshot, ctl: Room0DClearController) -> FrameAction:
    """One frame of 0x0D wallmaster clear.  Stay off x=32 / x=208."""
    x, y = int(snap.link_x), int(snap.link_y)
    live, parked = _0d_spawners(snap)
    incoming = tuple(o for o in parked if int(o.state) != 0)
    if x < 44 and ctl._phase != "nudge":
        ctl._phase = "retreat"
        ctl._phase_n = 0
        return FrameAction(nes_action("RIGHT"), "clear0d_grab_peel")
    if live:
        ctl._phase = "fight"
        tgt = nearest_enemy(snap.link_x, snap.link_y, live)
        if tgt is None:
            return FrameAction(nes_idle_action(), "wm_missing")
        stand_x = max(48, int(tgt.x) + 18)
        if x < 48:
            return FrameAction(nes_action("RIGHT"), "clear0d_fight_peel")
        if abs(y - int(tgt.y)) > 10:
            return FrameAction(
                nes_action("DOWN" if y < int(tgt.y) else "UP"), "clear0d_fight_y"
            )
        if abs(x - stand_x) > 3:
            return FrameAction(
                nes_action("RIGHT" if x < stand_x else "LEFT"), "clear0d_fight_x"
            )
        face = "LEFT" if int(tgt.x) < x else "RIGHT"
        if ctl.frames % _SWING_PERIOD < _SWING_HOLD:
            return FrameAction(nes_action(face, "A"), "clear0d_slash")
        return FrameAction(nes_action(face), "clear0d_face")
    if any(
        int(o.type_id) == CLOCK_DROP_OBJECT_TYPE and int(o.state) == CLOCK_DROP_STATE
        for o in snap.objects
    ):
        # A taken clock ($066C) stops the ring spawning until Link leaves the
        # room: two Wallmasters stayed parked through 517 nudges (30000f,
        # natural_credits_43r). A dead Wallmaster drops it on the nudge line;
        # wait at home for it to expire instead of walking over it.
        ctl._phase = "retreat"
        ctl._phase_n = 0
    if incoming or ctl._phase == "retreat":
        ctl._phase = "retreat"
        ctl._phase_n += 1
        if abs(x - _0D_HOME[0]) > 4:
            btn = "RIGHT" if x < _0D_HOME[0] else "LEFT"
            return FrameAction(nes_action(btn), "clear0d_home_x")
        if abs(y - _0D_HOME[1]) > 4:
            return FrameAction(
                nes_action("UP" if y > _0D_HOME[1] else "DOWN"), "clear0d_home_y"
            )
        if ctl._phase_n >= 40:
            ctl._phase = "to_nudge"
            ctl._phase_n = 0
        return FrameAction(nes_idle_action(), "clear0d_home")
    if ctl._phase == "nudge":
        # A Wallmaster emerges the frame Link stands on the outer ring (x<=36
        # here; x>=40 never, measured by pinning Link along y=117). The x<44
        # grab peel above used to stop every nudge at ~x=42, so 0x0D lapped
        # nudge/home ~100 times (11352f, Blue Ring power-on 14) waiting on
        # luck. Touch the ring, then retreat and fight what comes out.
        # Holding the ring 3 or 8 frames was slower over 36 offsets.
        ctl._phase_n += 1
        if x <= _0D_RING_X or ctl._phase_n >= 72:
            ctl._phase = "retreat"
            ctl._phase_n = 0
            return FrameAction(nes_action("RIGHT"), "clear0d_nudge_done")
        return FrameAction(nes_action("LEFT"), "clear0d_nudge")
    tx, ty = _0D_NUDGE
    if abs(x - tx) <= 6 and abs(y - ty) <= 6:
        ctl._phase = "nudge"
        ctl._phase_n = 0
        return FrameAction(nes_action("LEFT"), "clear0d_nudge_start")
    if abs(y - ty) > 4:
        return FrameAction(
            nes_action("UP" if y > ty else "DOWN"), "clear0d_to_nudge_y"
        )
    return FrameAction(
        nes_action("LEFT" if x > tx else "RIGHT"), "clear0d_to_nudge_x"
    )

