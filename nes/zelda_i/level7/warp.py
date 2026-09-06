"""Recorder overworld warp (H1): whirlwind carry to a completed-dungeon door.

Blowing the already-owned Recorder (``ADDR_WHISTLE`` / ``$065C`` >= 1, B-slot 5
through ``dungeon.pause_select``) on a **non-entrance** overworld screen starts a
whirlwind-carry cutscene (mode 5→6→7→4→5, Link auto-walks off the edge with no
player input) that drops him on the door screen of a completed dungeon, cycling
by facing.  Live cycle facing DOWN from ``0x24`` (``scratch/pond/
probe_recorder_warp_cycle.py``, ``probe_recorder_warp_full_route.py``)::

    0x22 (L6) → 0x0B (L5) → 0x45 (L4) → 0x74 (L3) → 0x3C (L2)

This is why the mountain-locked post-L6 pocket is escapable at all: the L4
island door ``0x45`` sits one screen NORTH of ``0x55``, which is already on the
green ``LEVEL7_POND_APPROACH_HOPS`` chain to the Demon pond ``0x42``
(bead ``rr-8t4.1``).  ``0x74 → 0x64`` UP is DEAD (``0x74``'s whole north edge is
mountain across all 32 tile columns).

Natural play: the Recorder is owned from L5 and the Raft from L3, so no
capability is assumed that the measured L6 leave (TF ``0x3F``) does not hold.
No RAM pokes — ``writes=0``.

The blow loop is **screen-checked**, not a hardcoded count: recon measured a
deterministic 8 blows to ``0x45`` (3 blows per dungeon-advance after the first,
landings on the intermediate door screens included), but blows fired from a
door screen can no-op, so the controller blows until ``$EB`` reads the target or
``MAX_BLOWS`` is spent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.dungeon.pause_select import B_SLOT_RECORDER, PauseSelectController
from zelda_i.ram import ADDR_WHISTLE, PLAY_MODE, ZeldaSnapshot, read_u8

__all__ = [
    "BLOW_PRESSES",
    "FACE_FRAMES",
    "MAX_BLOWS",
    "SETTLE_STABLE_FRAMES",
    "WARP_MAX_FRAMES",
    "RecorderWarpController",
    "WarpPhase",
    "make_recorder_warp_controller",
]

DEATH_MODE = 17
CAVE_MODE = 16
# probe_recorder_warp_full_route.py, 3/3 identical: 20f of facing input, 12
# B presses, then a 90f xy-stable settle before the next blow.
FACE_FRAMES = 20
BLOW_PRESSES = 12
SETTLE_STABLE_FRAMES = 90
SETTLE_MAX_FRAMES = 3000
# Recon needed 8; the cap leaves margin for a no-op blow off a door screen.
MAX_BLOWS = 12
WARP_MAX_FRAMES = 40_000
WARP_FACING = "DOWN"


class WarpPhase(Enum):
    SELECT = auto()
    FACE = auto()
    BLOW = auto()
    SETTLE = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class RecorderWarpController:
    """Pause-select the Recorder, then blow until ``target_screen`` is live.

    ``launch_screen`` must be a non-entrance overworld screen — entrance screens
    suppress the warp (``probe_recorder_warp.py``: 8 blows on ``0x22``, zero
    change).  Success is ``target_screen`` settled in mode 5 on the overworld.
    """

    target_screen: int
    launch_screen: int
    facing: str = WARP_FACING
    max_blows: int = MAX_BLOWS
    max_frames: int = WARP_MAX_FRAMES
    phase: WarpPhase = WarpPhase.SELECT
    frames: int = 0
    phase_frames: int = 0
    blows: int = 0
    success: bool = False
    landings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    _env: Any = field(default=None, init=False, repr=False)
    _select: PauseSelectController = field(init=False, repr=False)
    _stable: int = field(default=0, init=False, repr=False)
    _last_xy: tuple[int, int] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._select = PauseSelectController(want=B_SLOT_RECORDER, name="recorder")

    @property
    def failed(self) -> bool:
        return self.phase is WarpPhase.FAILED

    def bind_env(self, env: Any) -> None:
        self._env = env
        self._select.bind_env(env)

    def _note(self, note: str) -> None:
        if note not in self.notes:
            self.notes.append(note)

    def _set_phase(self, phase: WarpPhase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            if note:
                self._note(note)

    def _fail(self, reason: str) -> FrameAction:
        self._set_phase(WarpPhase.FAILED, reason)
        return FrameAction(nes_idle_action(), reason)

    def _finish(self, note: str) -> FrameAction:
        self.success = True
        self._set_phase(WarpPhase.DONE, note)
        return FrameAction(nes_idle_action(), note)

    def _run_select(self, snap: ZeldaSnapshot) -> FrameAction:
        action = self._select.drive(snap)
        for note in self._select.notes:
            self._note(note)
        if self._select.failed:
            return self._fail(self._select.fail_reason or "warp_select_failed")
        if action is None:
            self._set_phase(WarpPhase.FACE, "recorder_ready")
            return FrameAction(nes_action(self.facing), "warp_face")
        return action

    def _settled(self, snap: ZeldaSnapshot) -> bool:
        xy = (int(snap.link_x), int(snap.link_y))
        if snap.mode == PLAY_MODE and not snap.transitioning and xy == self._last_xy:
            self._stable += 1
        else:
            self._stable = 0
        self._last_xy = xy
        return self._stable >= SETTLE_STABLE_FRAMES

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.phase_frames += 1
        if self.success or self.failed:
            return FrameAction(nes_idle_action(), "done")
        if self._env is None:
            return self._fail("warp_env_not_bound")
        if snap.mode == DEATH_MODE:
            return self._fail("death")
        if self.frames > self.max_frames:
            return self._fail("warp_budget_exhausted")
        if snap.level != 0:
            return self._fail(f"warp_left_overworld_L{snap.level}")
        if snap.mode == CAVE_MODE:
            return self._fail("warp_entered_cave")
        if int(read_u8(self._env.get_ram(), ADDR_WHISTLE)) < 1:
            return self._fail("warp_requires_whistle")

        if self.phase is WarpPhase.SELECT:
            if int(snap.screen) != self.launch_screen:
                return self._fail(f"warp_not_on_launch_0x{int(snap.screen):02x}")
            return self._run_select(snap)
        if self.phase is WarpPhase.FACE:
            if self.phase_frames >= FACE_FRAMES:
                self._set_phase(WarpPhase.BLOW, "warp_blow")
                return FrameAction(nes_action("B"), "warp_blow")
            return FrameAction(nes_action(self.facing), "warp_face")
        if self.phase is WarpPhase.BLOW:
            if self.phase_frames >= BLOW_PRESSES:
                self.blows += 1
                self._stable = 0
                self._last_xy = None
                self._set_phase(WarpPhase.SETTLE, "warp_settle")
                return FrameAction(nes_idle_action(), "warp_settle")
            return FrameAction(nes_action("B"), "warp_blow")
        if self.phase is WarpPhase.SETTLE:
            if self.phase_frames > SETTLE_MAX_FRAMES:
                return self._fail("warp_settle_timeout")
            if not self._settled(snap):
                return FrameAction(nes_idle_action(), "warp_settle")
            screen = int(snap.screen)
            self.landings.append(f"0x{screen:02x}")
            if screen == self.target_screen:
                return self._finish(f"warp_landed_0x{screen:02x}")
            if self.blows >= self.max_blows:
                return self._fail(f"warp_target_unreached_0x{screen:02x}")
            self._set_phase(WarpPhase.FACE, "warp_next_blow")
            return FrameAction(nes_action(self.facing), "warp_face")
        return FrameAction(nes_idle_action(), "done")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "spec_id": "level7_recorder_warp",
            "launch_screen": f"0x{self.launch_screen:02X}",
            "target_screen": f"0x{self.target_screen:02X}",
            "facing": self.facing,
            "blows": self.blows,
            "landings": list(self.landings),
            "phase": self.phase.name,
            "normal_pause_input": True,
            "writes": 0,
            "route_eligible": False,
            "evidence": "fixture-live",
            "notes": list(self.notes),
        }


def make_recorder_warp_controller(
    *, target_screen: int, launch_screen: int
) -> RecorderWarpController:
    """Fresh warp controller (never share instances across trials)."""
    return RecorderWarpController(
        target_screen=target_screen, launch_screen=launch_screen
    )
