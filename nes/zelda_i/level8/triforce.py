"""L8 0x3C north shutter dest hop + 0x2C shard walk-on.

From play 0x3C (32,181) SW floor. North shutter is RAM-open (doors 12).
UP inland first (do not DOWN / 0x4C). x-align 120 on the north band,
UP push. OccupancyWalker banned. Dest $EB is live **0x2C** (T2/T3),
room_item 0x1B. ROM 0x2C was hypothesis; now RAM. route_eligible=false.
Not on L8_THROUGH. Shard walk is UP x=120 onto the statue-square TF.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.anchors import TF_BIT_L8
from zelda_i.dungeon.door_hop import (
    HopFail,
    RoomHopController,
    RoomHopSpec,
    door_band_goal,
)
from zelda_i.dungeon.hop_controller import HopController, WAIT_SCROLL_B
from zelda_i.dungeon.ops import DOOR_TARGETS
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot

__all__ = [
    "FANFARE_MODE",
    "NORTH_3C_GATE",
    "NORTH_3C_DEST",
    "NORTH_3C_DEST_POSE",
    "NORTH_3C_MAX_FRAMES",
    "NORTH_3C_ORIGIN",
    "NORTH_3C_ORIGIN_POSE",
    "NORTH_BAND_Y",
    "NORTH_DOOR",
    "RAM_CLAIM",
    "ROOM_ITEM_TF",
    "SHARD_XY",
    "SOUTH_FAIL",
    "TF_ROOM_HYP",
    "Level8North3CController",
    "Level8Shard2CController",
    "Level8OWLeaveController",
    "OW_LEAVE_POSE",
    "OW_LEAVE_SCREEN",
    "make_north_3c_controller",
    "make_ow_leave_controller",
    "make_shard_2c_controller",
    "north_3c_step",
    "shard_2c_step",
]

NORTH_3C_ORIGIN = 0x3C
NORTH_3C_ORIGIN_POSE = (32, 181)
NORTH_DOOR = DOOR_TARGETS["UP"]  # (120, 93)
# T1 boxed UP on west-wall diamond (32,133) tile 179. Align on the
# door-row, then UP the center aisle (Gleeok stand was walkable).
NORTH_BAND_Y = 141
SOUTH_FAIL = 0x4C  # open bomb hole; do not DOWN
# T2/T3 live dest $EB=0x2C (120,205) south mouth, room_item 0x1B.
# ROM/walkthrough also said 0x2C; lock is the live trial, not the ROM.
NORTH_3C_DEST = 0x2C
NORTH_3C_DEST_POSE = (120, 205)
TF_ROOM_HYP = NORTH_3C_DEST
ROOM_ITEM_TF = 0x1B
SHARD_XY = (120, 141)
FANFARE_MODE = 18
CELLAR_FAIL = (0x0F, 0x2F)
NORTH_3C_MAX_FRAMES = 4000
SHARD_MAX_FRAMES = 4000
# rr-6o7.3: after the shard fanfare Link is returned to OW `0x6D` `(96,93)`
# mode 5, TF `0xFF` -- the measured power-on `--through level8` leave
# (`level8_ow_leave_settle` stage: 567f to 30 consecutive OW-0x6D frames) and
# the L9 predecessor pose.  Matches the old `Level8PostShardOWReconFixture`.
OW_LEAVE_SCREEN = 0x6D
OW_LEAVE_POSE = (96, 93)
OW_LEAVE_SETTLE_FRAMES = 30
OW_LEAVE_MAX_FRAMES = 3000
_DOOR_TOL = 4
_SAMPLE_PERIOD = 12
RAM_CLAIM = (
    "From play 0x3C leftover (32,181), UP inland (do not exit south), "
    "x-align 120, UP through the open north shutter, first settled play "
    "$EB is RAM (hyp TF, NOT 0x4C). Keys 8→8 bombs 5→5 MK 1 TF still "
    "0x7F until the shard is walked onto. Deaths 0. progression_writes=0."
)


def north_3c_step(snap: ZeldaSnapshot) -> FrameAction:
    """UP inland first, x-align 120 on the north band, UP push. Never DOWN / 0x4C."""
    x, y = int(snap.link_x), int(snap.link_y)
    if y > NORTH_BAND_Y:
        return FrameAction(nes_action("UP"), "north_inland")
    gx, gy = door_band_goal("UP", (x, y), NORTH_DOOR)
    if abs(x - gx) > _DOOR_TOL:
        btn = "RIGHT" if x < gx else "LEFT"
        return FrameAction(nes_action(btn), "north_align")
    if y > gy + _DOOR_TOL:
        return FrameAction(nes_action("UP"), "north_approach")
    return FrameAction(nes_action("UP"), "north_push")


def _leftover(snap: ZeldaSnapshot) -> dict[str, Any]:
    return {
        "x": int(snap.link_x),
        "y": int(snap.link_y),
        "mode": int(snap.mode),
        "screen": int(snap.screen),
        "tile": int(snap.colliding_tile),
        "keys": int(snap.keys),
        "bombs": int(snap.bombs),
        "triforce": int(snap.triforce),
        "doors": int(snap.cur_opened_doors),
        "item": int(snap.room_item_id),
        "hc": int(snap.heart_containers),
    }


NORTH_3C_GATE = RoomHopSpec(
    spec_id="level8_north_3c",
    origin=NORTH_3C_ORIGIN,
    door="UP",
    done_reason="left_0x3c_north",
    step=north_3c_step,
    level=8,
    max_frames=NORTH_3C_MAX_FRAMES,
    sample_period=_SAMPLE_PERIOD,
    leftover_fn=_leftover,
    fails=(
        HopFail(CELLAR_FAIL, "cellar_0x{screen:02x}", on_passage=True),
        HopFail((SOUTH_FAIL,), "south_0x4c"),
    ),
    scroll_button="UP",
    scroll_reason="north_scroll",
    settle_button="UP",
    settle_reason="north_settle",
    report_extra=(
        ("tf_room_hyp", TF_ROOM_HYP),
        ("assumed_0x2c", False),
        ("policy", RAM_CLAIM),
    ),
)


@dataclass(kw_only=True)
class Level8North3CController(RoomHopController):
    """0x3C leftover → north shutter UP. Dest is RAM; fail 0x4C / cellar."""

    spec: RoomHopSpec = NORTH_3C_GATE


def make_north_3c_controller(
    *, dest: int | None = NORTH_3C_DEST
) -> Level8North3CController:
    return Level8North3CController(dest=dest)


def shard_2c_step(snap: ZeldaSnapshot) -> FrameAction:
    """x-align 120, UP onto the center shard. Never DOWN out the south door."""
    x, y = int(snap.link_x), int(snap.link_y)
    tx, ty = SHARD_XY
    if abs(x - tx) > _DOOR_TOL:
        btn = "RIGHT" if x < tx else "LEFT"
        return FrameAction(nes_action(btn), "shard_align")
    if y > ty + _DOOR_TOL:
        return FrameAction(nes_action("UP"), "shard_approach")
    return FrameAction(nes_action("UP"), "shard_stand")


@dataclass(kw_only=True)
class Level8Shard2CController(HopController):
    """Play 0x2C south mouth → walk onto 0x1B. Fanfare or a TF rising edge.

    ``tf_in`` latches the shard bit on the first stepped frame: a state that
    already carries TF ``0x80`` (any post-shard pin, e.g. the fixture-lineage
    ``Level8PostShardOWReconFixture``) fails closed instead of reporting a
    zero-input success.
    """

    spec_id: str = "level8_shard_2c"
    max_frames: int = SHARD_MAX_FRAMES
    require_level: int | None = None
    wait_modes: tuple[int, ...] = WAIT_SCROLL_B
    done_reason: str = "tf_0x80"
    route_eligible: bool = False
    leftover: dict[str, Any] = field(default_factory=dict)
    writes: int = 0
    tf_in: int | None = field(default=None, init=False)

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        if self.tf_in is None or int(self.tf_in) & TF_BIT_L8:
            return False
        if snap.mode == FANFARE_MODE:
            return True
        return bool(int(snap.triforce) & TF_BIT_L8)

    def on_arrive(self, snap: ZeldaSnapshot) -> str:
        return (
            f"tf_0x{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
            f"_mode={snap.mode}_tf=0x{snap.triforce:02x}"
        )

    def emit(
        self, snap: ZeldaSnapshot, action: FrameAction, *, force: bool = False
    ) -> FrameAction:
        if force or not self.leftover or self.frames % _SAMPLE_PERIOD == 0:
            self.leftover = _leftover(snap)
        return action

    def guard(self, snap: ZeldaSnapshot) -> FrameAction | None:
        if self.tf_in is None:
            self.tf_in = int(snap.triforce)
            if int(self.tf_in) & TF_BIT_L8:
                return self.mark_fail("l8_shard_already_taken")
        blocked = HopController.guard(self, snap)
        if blocked is not None:
            return blocked
        if (
            snap.level == 8
            and snap.mode == PLAY_MODE
            and not snap.transitioning
            and snap.screen != NORTH_3C_DEST
        ):
            return self.mark_fail(f"left_0x2c_to_0x{snap.screen:02x}")
        return None

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        if snap.mode == FANFARE_MODE:
            return FrameAction(nes_idle_action(), "wait_fanfare")
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.level != 8:
            return FrameAction(nes_idle_action(), f"wait_level_{snap.level}")
        return shard_2c_step(snap)

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "spec_id": self.spec_id,
            "evidence": "fixture-live",
            "route_eligible": False,
            "natural_entry": False,
            "writes": int(self.writes),
            "tf_in": self.tf_in,
            "leftover": dict(self.leftover),
        }


def make_shard_2c_controller() -> Level8Shard2CController:
    return Level8Shard2CController()


@dataclass(kw_only=True)
class Level8OWLeaveController(HopController):
    """Idle the shard fanfare until Link settles on OW ``0x6D`` play.

    The chapter epilogue: ``Level8Shard2CController`` stops at the fanfare
    (mode 18), but ``level8_clear_stop`` needs the settled overworld leave.
    Press nothing -- the fanfare and the dungeon->OW scroll run themselves.
    Fails closed if Link settles on any OW screen other than ``0x6D``.
    """

    spec_id: str = "level8_ow_leave_settle"
    max_frames: int = OW_LEAVE_MAX_FRAMES
    require_level: int | None = None
    wait_modes: tuple[int, ...] = WAIT_SCROLL_B
    done_reason: str = "settled_ow_0x6d"
    route_eligible: bool = False
    settle_frames: int = OW_LEAVE_SETTLE_FRAMES
    leftover: dict[str, Any] = field(default_factory=dict)
    writes: int = 0
    _ow_since: int = field(default=0, init=False)

    def _on_ow_6d(self, snap: ZeldaSnapshot) -> bool:
        return (
            snap.level == 0
            and snap.screen == OW_LEAVE_SCREEN
            and snap.mode == PLAY_MODE
            and not snap.transitioning
            and bool(int(snap.triforce) & TF_BIT_L8)
        )

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        self._ow_since = self._ow_since + 1 if self._on_ow_6d(snap) else 0
        return self._ow_since >= self.settle_frames

    def on_arrive(self, snap: ZeldaSnapshot) -> str:
        return (
            f"ow_0x{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
            f"_tf=0x{snap.triforce:02x}"
        )

    def emit(
        self, snap: ZeldaSnapshot, action: FrameAction, *, force: bool = False
    ) -> FrameAction:
        if force or not self.leftover or self.frames % _SAMPLE_PERIOD == 0:
            self.leftover = _leftover(snap)
        return action

    def guard(self, snap: ZeldaSnapshot) -> FrameAction | None:
        blocked = HopController.guard(self, snap)
        if blocked is not None:
            return blocked
        if (
            snap.level == 0
            and snap.mode == PLAY_MODE
            and not snap.transitioning
            and snap.screen != OW_LEAVE_SCREEN
        ):
            return self.mark_fail(
                f"l8_ow_leave_wrong_screen_0x{snap.screen:02x}"
            )
        if snap.level not in (0, 8):
            return self.mark_fail(f"l8_ow_leave_wrong_level_{snap.level}")
        return None

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        return FrameAction(nes_idle_action(), f"ow_leave_wait_m{snap.mode}")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "spec_id": self.spec_id,
            "evidence": "fixture-live",
            "route_eligible": False,
            "natural_entry": False,
            "writes": int(self.writes),
            "leftover": dict(self.leftover),
        }


def make_ow_leave_controller() -> Level8OWLeaveController:
    return Level8OWLeaveController()
