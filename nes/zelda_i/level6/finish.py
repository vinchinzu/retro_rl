"""Level 6 post-Gohma finish: heart 0x1A, north 0x0C, shard TF 0x20.

Live leftover: Gohma play 0x1C ``(120,189)`` → heart ``(120,149)`` hc 7→8
→ north shutter play 0x0C ``(120,205)`` → fanfare ``(120,149)`` TF 0x3F.
Do not poke TF/doors. Isolated BFS banned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.dungeon.hop_controller import (
    HopController,
    WAIT_SCROLL_B,
    dungeon_align_then_push,
)
from zelda_i.level6.gohma import gohma_live
from zelda_i.level6.occupancy import occupancy_new_miss, record_l6_walk
from zelda_i.level6.overworld import (
    LEVEL6,
    LEVEL6_GOHMA_ROOM,
    LEVEL6_GOHMA_WING_2C_ROOM,
    LEVEL6_TF_ROOM,
    LEVEL6_TRIFORCE_BIT,
    SCREEN_LEVEL6_ENTRANCE,
)
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot
from zelda_i.walk.physics import OccupancyWalker

__all__ = [
    "DOOR_NORTH",
    "EXIT_MAX_FRAMES",
    "FANFARE_MODE",
    "FINISH_MAX_FRAMES",
    "HEART_XY",
    "SHARD_XY",
    "Level6ExitController",
    "Level6HeartController",
    "Level6North0cController",
    "Level6ShardController",
    "level6_exit_success",
    "level6_heart_success",
    "level6_north0c_success",
    "level6_success",
    "make_exit_controller",
    "make_heart_controller",
    "make_north0c_controller",
    "make_shard_controller",
    "north_shutter_open",
]

FANFARE_MODE = 18
DOOR_NORTH = 0x08
FINISH_MAX_FRAMES = 4000
EXIT_MAX_FRAMES = 2500
SAMPLE_PERIOD = 16
HEART_XY = (120, 141)
SHARD_XY = (120, 141)
POST_HEART_CONTAINERS = 8
INCOMING_TF = 0x1F
LEAVING_TF = INCOMING_TF | LEVEL6_TRIFORCE_BIT  # 0x3F


def north_shutter_open(snap: ZeldaSnapshot) -> bool:
    """North bit on either live door field. Do not poke."""
    return bool(
        (int(snap.cur_opened_doors) | int(snap.open_doorway_mask)) & DOOR_NORTH
    )


def _finish_leftover(snap: ZeldaSnapshot, base: dict[str, int]) -> dict[str, int]:
    return {
        **base,
        "health": int(snap.health),
        "heart_containers": int(snap.heart_containers),
        "room_item_id": int(snap.room_item_id),
        "cur_opened_doors": int(snap.cur_opened_doors),
        "open_doorway_mask": int(snap.open_doorway_mask),
    }


@dataclass
class _FinishHop(HopController):
    """Occupancy dest hop with heart/door leftover. Subclass arrived/policy."""

    samples: list[dict[str, Any]] = field(default_factory=list)
    leftover: dict[str, Any] = field(default_factory=dict)
    walker: OccupancyWalker = field(default_factory=OccupancyWalker)
    max_frames: int = FINISH_MAX_FRAMES
    wait_modes: tuple[int, ...] = WAIT_SCROLL_B

    def emit(
        self, snap: ZeldaSnapshot, action: FrameAction, *, force: bool = False
    ) -> FrameAction:
        self.leftover = _finish_leftover(
            snap,
            record_l6_walk(
                self.samples,
                snap,
                reason=action.reason,
                frames=self.frames,
                period=SAMPLE_PERIOD,
                misses=self.walker.misses,
                force=force,
            ),
        )
        return action

    def timeout_note(self, snap: ZeldaSnapshot) -> str:
        return (
            f"timeout_{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
            f"_mode={snap.mode}_hc={snap.heart_containers}_tf={snap.triforce:02x}"
        )

    def _leave_guard(self, snap: ZeldaSnapshot) -> FrameAction | None:
        if snap.level != LEVEL6:
            return self.mark_fail(f"left_level_{snap.level}")
        return None

    def _path(self, snap: ZeldaSnapshot, dest: tuple[int, int]) -> FrameAction:
        xy = (int(snap.link_x), int(snap.link_y))
        occupancy_new_miss(self.walker, xy, allow_first=True)
        direction = self.walker.next_dir(xy, dest)
        if direction is None:
            return FrameAction(nes_idle_action(), "occupancy_stand")
        return FrameAction(nes_action(direction), "occupancy_path")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "samples": list(self.samples),
            "leftover": dict(self.leftover),
            "misses": self.walker.misses,
            "spec_id": self.spec_id,
            "room": getattr(self, "room", None),
        }


@dataclass
class Level6HeartController(_FinishHop):
    """Occupancy to room-center heart. Stops on incoming containers +1."""

    spec_id: str = "level6_heart_0x1c"
    room: int = LEVEL6_GOHMA_ROOM
    done_reason: str = "heart_got"
    incoming_containers: int | None = None

    def on_arrive(self, snap: ZeldaSnapshot) -> str:
        return (
            f"heart_{snap.link_x}_{snap.link_y}"
            f"_hc={self.incoming_containers}->{snap.heart_containers}"
            f"_health=0x{snap.health:02x}"
        )

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        # The container, not full hearts: only the Survival refill fills
        # them, and a last-heart run stood on the taken heart 4000 frames.
        if self.incoming_containers is None:
            return False
        return snap.heart_containers > self.incoming_containers

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        left = self._leave_guard(snap)
        if left is not None:
            return left
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.screen != self.room:
            return self.mark_fail(f"left_0x{self.room:02x}_to_0x{snap.screen:02x}")
        if snap.triforce != INCOMING_TF:
            return self.mark_fail(f"tf_changed_0x{snap.triforce:02x}")
        if self.incoming_containers is None:
            self.incoming_containers = int(snap.heart_containers)
            self.notes.append(f"incoming_hc={self.incoming_containers}")
        if gohma_live(snap):
            return self.mark_fail("gohma_still_live")
        return self._path(snap, HEART_XY)


@dataclass
class Level6North0cController(_FinishHop):
    """Wait for the natural north shutter, then cardinal x-align UP to 0x0C."""

    spec_id: str = "level6_north_0x0c"
    room: int = LEVEL6_GOHMA_ROOM
    dest: int = LEVEL6_TF_ROOM
    done_reason: str = "entered_0c"

    def on_arrive(self, snap: ZeldaSnapshot) -> str:
        return f"arrived_{snap.screen:02x}_{snap.link_x}_{snap.link_y}"

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        return (
            snap.level == LEVEL6
            and snap.mode == PLAY_MODE
            and not snap.transitioning
            and snap.screen == self.dest
        )

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        left = self._leave_guard(snap)
        if left is not None:
            return left
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.screen == LEVEL6_GOHMA_WING_2C_ROOM:
            return self.mark_fail("back_0x2c")
        if snap.screen != self.room:
            return self.mark_fail(f"left_0x{self.room:02x}_to_0x{snap.screen:02x}")
        if snap.triforce != INCOMING_TF:
            return self.mark_fail(f"tf_changed_0x{snap.triforce:02x}")
        if not north_shutter_open(snap):
            return FrameAction(nes_idle_action(), "wait_shutter")
        return dungeon_align_then_push(
            snap, push_dir="UP", target_x=HEART_XY[0], reason="north0c"
        )


@dataclass
class Level6ShardController(_FinishHop):
    """Occupancy onto the 0x0C shard. TF 0x1F|0x20. Fanfare is success."""

    spec_id: str = "level6_triforce_0x20"
    room: int = LEVEL6_TF_ROOM
    done_reason: str = "tf_0x20"
    wait_modes: tuple[int, ...] = WAIT_SCROLL_B

    def on_arrive(self, snap: ZeldaSnapshot) -> str:
        return (
            f"tf_{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
            f"_mode={snap.mode}_tf=0x{snap.triforce:02x}"
        )

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        return int(snap.triforce) == LEAVING_TF

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        if snap.mode == FANFARE_MODE:
            return FrameAction(nes_idle_action(), "wait_fanfare")
        left = self._leave_guard(snap)
        if left is not None:
            return left
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.screen == LEVEL6_GOHMA_ROOM:
            return self.mark_fail("back_0x1c")
        if snap.screen == LEVEL6_GOHMA_WING_2C_ROOM:
            return self.mark_fail("back_0x2c")
        if snap.screen != self.room:
            return self.mark_fail(f"left_0x{self.room:02x}_to_0x{snap.screen:02x}")
        return self._path(snap, SHARD_XY)


@dataclass
class Level6ExitController(_FinishHop):
    """Idle through the shard fanfare until the engine returns Link to the
    overworld.

    Zelda 1 auto-warps out of a dungeon after a Triforce piece — back onto the
    dungeon's overworld entrance tile (L1 → OW ``0x37`` ~(112,125), L4 →
    island ``0x45``). This stage only measures where L6 lands; it never walks.
    Measured (`--through level6-exit` 1/1): OW ``0x22`` ``(112,125)``, mode 5,
    TF ``0x3F`` — the Dragon mouth tile, which the L7 bait/pond route depends
    on. ``(112,125)`` is not the "mode 16 → dungeon" trap: that only fires on
    a fresh UP into the mouth, not on emerging onto it.
    """

    spec_id: str = "level6_exit_ow"
    room: int = LEVEL6_TF_ROOM
    done_reason: str = "ow_return"
    max_frames: int = EXIT_MAX_FRAMES

    def on_arrive(self, snap: ZeldaSnapshot) -> str:
        return (
            f"ow_{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
            f"_mode={snap.mode}_tf=0x{snap.triforce:02x}"
        )

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        return (
            snap.level == 0
            and snap.mode == PLAY_MODE
            and not snap.transitioning
            and snap.screen == SCREEN_LEVEL6_ENTRANCE
            and int(snap.triforce) == LEAVING_TF
        )

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        # The fanfare return is automatic; hold still. Only fail if Link ends
        # up back in an L6 play room other than the shard room 0x0C.
        if (
            snap.level == LEVEL6
            and snap.mode == PLAY_MODE
            and not snap.transitioning
            and snap.screen != self.room
        ):
            return self.mark_fail(f"l6_play_0x{snap.screen:02x}_no_ow_return")
        return FrameAction(nes_idle_action(), f"wait_return_mode_{snap.mode}")


def make_heart_controller() -> Level6HeartController:
    return Level6HeartController()


def make_north0c_controller() -> Level6North0cController:
    return Level6North0cController()


def make_shard_controller() -> Level6ShardController:
    return Level6ShardController()


def make_exit_controller() -> Level6ExitController:
    return Level6ExitController()


def _play_1c(snap: ZeldaSnapshot) -> bool:
    return (
        snap.level == LEVEL6
        and snap.mode == PLAY_MODE
        and not snap.transitioning
        and snap.screen == LEVEL6_GOHMA_ROOM
        and int(snap.triforce) == INCOMING_TF
        and int(snap.bow) >= 1
        and int(snap.rod) >= 1
    )


def level6_heart_success(snap: ZeldaSnapshot) -> bool:
    """Play 0x1C, heart +1 from 7, TF still 0x1F (hearts need not be full)."""
    if not _play_1c(snap):
        return False
    if snap.heart_containers < POST_HEART_CONTAINERS:
        return False
    return not gohma_live(snap)


def level6_north0c_success(snap: ZeldaSnapshot) -> bool:
    """Play 0x0C after the natural shutter; TF still 0x1F; heart kept."""
    if snap.level != LEVEL6 or snap.triforce != INCOMING_TF:
        return False
    if snap.mode != PLAY_MODE or snap.transitioning or snap.screen != LEVEL6_TF_ROOM:
        return False
    return snap.heart_containers >= POST_HEART_CONTAINERS


def level6_success(snap: ZeldaSnapshot) -> bool:
    """Shard 0x20 with incoming 0x1F bits. Fanfare in 0x0C is the leave."""
    if snap.level != LEVEL6 or int(snap.triforce) != LEAVING_TF:
        return False
    if snap.screen != LEVEL6_TF_ROOM:
        return False
    if snap.mode not in (PLAY_MODE, FANFARE_MODE):
        return False
    return snap.heart_containers >= POST_HEART_CONTAINERS


def level6_exit_success(snap: ZeldaSnapshot) -> bool:
    """Post-fanfare engine return: OW play on the Dragon entrance screen 0x22
    with the full L6 clear bits (TF 0x3F) and the heart kept."""
    if snap.level != 0 or int(snap.triforce) != LEAVING_TF:
        return False
    if snap.mode != PLAY_MODE or snap.transitioning:
        return False
    if snap.screen != SCREEN_LEVEL6_ENTRANCE:
        return False
    return snap.heart_containers >= POST_HEART_CONTAINERS
