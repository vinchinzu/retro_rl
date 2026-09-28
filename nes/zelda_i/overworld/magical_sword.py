"""The Magical Sword before Level 6: two coast hearts, then the 0x21 grave.

Clean L6 dies on the White Sword with 10 containers; a what-if pin with 12
and the Magical Sword finishes it (rr-fwny). The grave's Old Man gates on 12
containers and the spine reaches L5 with 9, so the ladder heart 0x5F and the
raft heart 0x2F are walked right after L4's 0x67 rupee rock (Stepladder and
Raft owned), and L5's heart makes 12 before the L6 walk passes 0x21.

Measured 2026-09-25 from ``C8c_return_4a`` (post-L4, 0x67), no writes:

- 0x67 -> 0x77 -> the pre-L1 bomb-shop coast (``SHOP_P7_HOPS``) -> 0x6F ->
  UP at x=120 -> 0x5F. 0x6F's north mouth is x 80..128; ``align_x=122`` is
  off the 8 px turn lattice and flipped Link 120/128 for 17k frames.
- 0x5F: the heart is on a stepping stone at (192,144). Shore (128,141),
  water 144..159, stone 160, water 176..191, heart stone 192: hold RIGHT and
  the stepladder bridges both gaps (108 frames).
- 0x4F -> 0x3F. The dock tip is (96,117); UP from it rafts to island 0x2F.
- 0x2F: take-any cave at x=96 over the landing; heart on the right,
  (152,149) as at 0x7B/0x2C. Containers +1 in 399 frames.
- Back: cave exit, DOWN on x=96 rafts to 0x3F, landing at the dock's top
  (96,61). The hunter's peel there walked Link back onto the raft for 15k
  frames, so the first hop off the dock runs without ``defend``/``evade``.
  Then the coast west to 0x78 and ``RUPEES_67_BACK_HOPS`` to 0x4A.
- Grave (from ``C8c_enter_level6`` at 0x33, 12-container what-if): 0x32 ->
  0x31 -> UP into 0x21. Twelve graves; slot 11's tile object 0x65 marks the
  one at (144,144). Stand (144,157) below it, hold UP: it slides, the stairs
  lead to the Old Man, the sword is the centre item (466 frames).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.dungeon.hop_controller import mouth_step, room_step
from zelda_i.overworld.gather_segments import (
    HEART_L8_ITEM_XY,
    CaveExitController,
    HopWalkController,
)
from zelda_i.overworld.graph import ScreenHop
from zelda_i.overworld.shop_p7 import SHOP_P7_HOPS
from zelda_i.overworld.zd_map import mirror_screen_hops
from zelda_i.ram import CAVE_MODE, PLAY_MODE, ZeldaSnapshot
from zelda_i.spine.hops import LatchedPlan, gated_stages

__all__ = [
    "coast_heart_plan",
    "grave_plan",
    "COAST_BACK_HOPS",
    "COAST_TO_5F_HOPS",
    "GraveSwordController",
    "LadderHeartController",
    "RaftRideController",
    "TakeAnyHeartController",
    "coast_heart_stages",
    "magical_sword_stages",
]

RUPEES_67_SCREEN = 0x67
LADDER_HEART_SCREEN = 0x5F
LADDER_HEART_SHORE = (128, 141)
LADDER_HEART_STONE_X = 184
RAFT_DOCK_SCREEN = 0x3F
RAFT_DOCK_TIP = (96, 117)
RAFT_ISLAND_SCREEN = 0x2F
RAFT_CAVE_MOUTH = (96, 141)
# The Magical Sword grave gates on this many containers; L5's heart is one.
SWORD_CONTAINERS = 12
# A heart already gone leaves Link on its stone or item tile: give up after.
EMPTY_ITEM_FRAMES = 60
MAGICAL_SWORD = 3
GRAVE_SCREEN = 0x21
GRAVE_STAND = (144, 157)
GRAVE_PUSH_MAX = 120
CAVE_CENTRE_ITEM = (120, 149)

COAST_TO_5F_HOPS: tuple[ScreenHop, ...] = (
    ScreenHop(0x77, "DOWN", align_x=112),
    *SHOP_P7_HOPS,
    ScreenHop(LADDER_HEART_SCREEN, "UP", align_x=120),
)
COAST_TO_DOCK_HOPS: tuple[ScreenHop, ...] = (
    ScreenHop(0x4F, "UP"),
    ScreenHop(RAFT_DOCK_SCREEN, "UP"),
)
RAFT_BACK_HOPS: tuple[ScreenHop, ...] = (
    ScreenHop(RAFT_DOCK_SCREEN, "DOWN", align_x=RAFT_DOCK_TIP[0]),
)
OFF_DOCK_HOPS: tuple[ScreenHop, ...] = (ScreenHop(0x4F, "DOWN"),)
GRAVE_FROM_SCREENS = (0x33, 0x32)
TO_GRAVE_HOPS: tuple[ScreenHop, ...] = (
    ScreenHop(0x32, "LEFT", align_y=141),
    ScreenHop(0x31, "LEFT"),
    ScreenHop(GRAVE_SCREEN, "UP"),
)
FROM_GRAVE_HOPS: tuple[ScreenHop, ...] = (
    ScreenHop(0x31, "DOWN"),
    ScreenHop(0x32, "RIGHT"),
)


def _coast_back_hops() -> tuple[ScreenHop, ...]:
    from zelda_i.level5.overworld import RUPEES_67_BACK_HOPS

    # West along the coast to 0x78, then the 0x67 return's 0x68 climb.
    west = mirror_screen_hops(0x77, SHOP_P7_HOPS)[:-1]
    assert west[-1].target == 0x78 and RUPEES_67_BACK_HOPS[2].target == 0x68
    return (
        ScreenHop(LADDER_HEART_SCREEN, "DOWN"),
        ScreenHop(0x6F, "DOWN", align_x=120),
        *west,
        *RUPEES_67_BACK_HOPS[2:],
    )


COAST_BACK_HOPS: tuple[ScreenHop, ...] = _coast_back_hops()


def coast_heart_plan() -> LatchedPlan:
    """Decided once, on the detour's first frame; every leg reads it."""

    def skip(snap: ZeldaSnapshot) -> str | None:
        if not snap.ladder or not snap.raft:
            return "no_ladder_or_raft"
        if snap.heart_containers + 1 >= SWORD_CONTAINERS:
            return "containers_enough"
        if snap.level != 0 or snap.screen != RUPEES_67_SCREEN:
            return f"not_from_0x{RUPEES_67_SCREEN:02x}"
        return None

    return LatchedPlan("coast_hearts", skip)


def grave_plan() -> LatchedPlan:
    """The L6 walk turns into 0x21 only with 12 containers and no sword yet."""

    def skip(snap: ZeldaSnapshot) -> str | None:
        if snap.sword >= MAGICAL_SWORD:
            return "sword_owned"
        if snap.heart_containers < SWORD_CONTAINERS:
            return f"containers_{snap.heart_containers}"
        if snap.level != 0 or snap.screen not in GRAVE_FROM_SCREENS:
            return f"not_from_0x{snap.screen:02x}"
        return None

    return LatchedPlan("magical_sword", skip)


@dataclass
class _ContainerStage:
    """Shared bookkeeping: succeed once the container byte rises."""

    max_frames: int = 3000
    frames: int = 0
    success: bool = False
    failed: bool = False
    containers0: int = -1
    notes: list[str] = field(default_factory=list)
    _env: Any = field(default=None, repr=False)

    def bind_env(self, env: Any) -> None:
        self._env = env

    def _gained(self, snap: ZeldaSnapshot) -> bool:
        if self.containers0 < 0:
            self.containers0 = snap.heart_containers
        return snap.heart_containers > self.containers0

    def _tick(self, snap: ZeldaSnapshot) -> FrameAction | None:
        self.frames += 1
        if snap.mode == 17:
            self.failed = True
            self.notes.append("link_death")
            return FrameAction(nes_idle_action(), "link_death")
        if self.frames >= self.max_frames:
            self.failed = True
            self.notes.append(f"timeout_{snap.screen:02x}_{snap.link_x}_{snap.link_y}")
            return FrameAction(nes_idle_action(), "timeout")
        return None

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "containers": [self.containers0, self.containers0 + int(self.success)],
            "notes": list(self.notes),
        }


@dataclass
class LadderHeartController(_ContainerStage):
    """0x5F: lattice to the shore, hold RIGHT over both stepladder gaps."""

    crossing: bool = False
    returning: bool = False
    on_stone: int = 0

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        if (done := self._tick(snap)) is not None:
            return done
        x, y = int(snap.link_x), int(snap.link_y)
        if self._gained(snap) and not self.returning:
            self.returning = True
            self.notes.append("ladder_heart")
        if self.returning:
            # The stone is an island: only the ladder reaches the shore.
            if x <= LADDER_HEART_SHORE[0]:
                self.success = True
                return FrameAction(nes_idle_action(), "ladder_ashore")
            return FrameAction(nes_action("LEFT"), "ladder_left")
        if (x, y) == LADDER_HEART_SHORE:
            self.crossing = True
        if self.crossing:
            if x >= LADDER_HEART_STONE_X:
                self.on_stone += 1
                if self.on_stone > EMPTY_ITEM_FRAMES:
                    self.returning = True
                    self.notes.append("no_heart_on_stone")
            return FrameAction(nes_action("RIGHT"), "ladder_right")
        step = room_step(snap, LADDER_HEART_SHORE, tol=0, env=self._env)
        return FrameAction(nes_action(step) if step else nes_idle_action(), "ladder_shore")


@dataclass
class RaftRideController:
    """Lattice to a dock tip, then hold ``direction`` until ``dest`` plays."""

    dock_tip: tuple[int, int] = RAFT_DOCK_TIP
    direction: str = "UP"
    dest: int = RAFT_ISLAND_SCREEN
    max_frames: int = 3000
    frames: int = 0
    success: bool = False
    failed: bool = False
    _env: Any = field(default=None, repr=False)

    def bind_env(self, env: Any) -> None:
        self._env = env

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        if snap.level == 0 and snap.screen == self.dest and snap.mode == PLAY_MODE and not snap.transitioning:
            self.success = True
            return FrameAction(nes_idle_action(), "raft_arrived")
        if snap.mode == 17 or self.frames >= self.max_frames:
            self.failed = True
            return FrameAction(nes_idle_action(), "raft_failed")
        if snap.transitioning or snap.mode != PLAY_MODE:
            return FrameAction(nes_action(self.direction), "raft_ride")
        x, y = int(snap.link_x), int(snap.link_y)
        if x == self.dock_tip[0] and abs(y - self.dock_tip[1]) <= 2:
            return FrameAction(nes_action(self.direction), "raft_board")
        step = room_step(snap, self.dock_tip, tol=0, env=self._env)
        return FrameAction(nes_action(step or self.direction), "raft_dock")

    def report(self) -> dict[str, Any]:
        return {"success": self.success, "failed": self.failed, "frames": self.frames}


@dataclass
class TakeAnyHeartController(_ContainerStage):
    """Walk into an open take-any cave and touch the right-hand heart."""

    mouth: tuple[int, int] = RAFT_CAVE_MOUTH
    item: tuple[int, int] = HEART_L8_ITEM_XY
    at_item: int = 0

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        if (done := self._tick(snap)) is not None:
            return done
        if self._gained(snap):
            self.success = True
            self.notes.append("take_any_heart")
            return FrameAction(nes_idle_action(), "take_any_heart")
        if snap.mode == CAVE_MODE:
            step = room_step(snap, self.item, tol=1, env=self._env)
            if step is None:
                self.at_item += 1
                if self.at_item > EMPTY_ITEM_FRAMES:
                    self.success = True
                    self.notes.append("no_heart_in_cave")
            return FrameAction(nes_action(step) if step else nes_idle_action(), "take_any_item")
        if snap.transitioning or snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"take_any_wait_{snap.mode}")
        step = mouth_step(snap, self.mouth[0], self.mouth[1], direction="UP", env=self._env)
        return FrameAction(nes_action(step), "take_any_mouth")


def coast_heart_stages() -> tuple[tuple[str, Any, int], ...]:
    """0x67 -> ladder heart 0x5F -> raft heart 0x2F -> 0x4A. Skips as one."""
    plan = coast_heart_plan()
    legs: tuple[tuple[str, Any], ...] = (
        ("walk_coast_5f", HopWalkController(hops=COAST_TO_5F_HOPS, max_frames=16000)),
        ("ladder_heart_5f", LadderHeartController()),
        ("walk_dock_3f", HopWalkController(hops=COAST_TO_DOCK_HOPS, max_frames=4000)),
        ("raft_2f", RaftRideController()),
        ("raft_heart_2f", TakeAnyHeartController()),
        ("exit_cave_2f", CaveExitController(clear=0)),
        ("raft_3f", HopWalkController(hops=RAFT_BACK_HOPS, max_frames=3000)),
        (
            "off_dock_3f",
            HopWalkController(hops=OFF_DOCK_HOPS, max_frames=3000, defend=False, evade=False),
        ),
        ("walk_coast_back", HopWalkController(hops=COAST_BACK_HOPS, max_frames=24000)),
    )
    return gated_stages(plan, legs)


@dataclass
class GraveSwordController:
    """0x21: push the marked grave UP, take the stairs, touch the centre item."""

    max_frames: int = 3000
    frames: int = 0
    pushed: int = 0
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)
    _env: Any = field(default=None, repr=False)

    def bind_env(self, env: Any) -> None:
        self._env = env

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        if snap.sword >= MAGICAL_SWORD:
            self.success = True
            return FrameAction(nes_idle_action(), "magical_sword")
        if snap.mode == 17 or self.frames >= self.max_frames:
            self.failed = True
            self.notes.append(f"failed_{snap.screen:02x}_{snap.link_x}_{snap.link_y}")
            return FrameAction(nes_idle_action(), "grave_failed")
        if snap.mode == CAVE_MODE:
            # The Old Man's text holds Link; presses during it are harmless.
            step = room_step(snap, CAVE_CENTRE_ITEM, tol=1, env=self._env)
            return FrameAction(nes_action(step or "UP"), "grave_cave_item")
        if snap.transitioning or snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"grave_wait_{snap.mode}")
        x, y = int(snap.link_x), int(snap.link_y)
        if self.pushed or (x, y) == GRAVE_STAND:
            # Past the stand the grave has slid: the same UP walks the stairs.
            self.pushed += 1
            if self.pushed > GRAVE_PUSH_MAX and (x, y) == GRAVE_STAND:
                self.failed = True
                self.notes.append("grave_did_not_move")
            return FrameAction(nes_action("UP"), "grave_push")
        step = room_step(snap, GRAVE_STAND, tol=0, env=self._env)
        return FrameAction(nes_action(step) if step else nes_idle_action(), "grave_stand")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "pushed": self.pushed,
            "notes": list(self.notes),
        }


def magical_sword_stages() -> tuple[tuple[str, Any, int], ...]:
    """0x33/0x32 -> 0x21 grave -> Magical Sword -> back to 0x32. Skips as one."""
    plan = grave_plan()
    legs: tuple[tuple[str, Any], ...] = (
        ("walk_grave_21", HopWalkController(hops=TO_GRAVE_HOPS, max_frames=6000)),
        ("magical_sword_21", GraveSwordController()),
        ("exit_cave_21", CaveExitController(clear=0)),
        ("return_32_from_21", HopWalkController(hops=FROM_GRAVE_HOPS, max_frames=6000)),
    )
    return gated_stages(plan, legs)
