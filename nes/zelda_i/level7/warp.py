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
from zelda_i.dungeon.hop_controller import LatticeDoorWalker
from zelda_i.dungeon.pause_select import B_SLOT_RECORDER, PauseSelectController
from zelda_i.ram import (
    ADDR_OBJ_TYPE,
    ADDR_WHIRLWIND_SUMMONED,
    ADDR_WHISTLE,
    PLAY_MODE,
    ZeldaSnapshot,
    read_u8,
)

__all__ = [
    "BLOW_PRESSES",
    "FACE_FRAMES",
    "MAX_BLOWS",
    "SETTLE_STABLE_FRAMES",
    "WARP_DOOR_LEVELS",
    "WARP_MAX_FRAMES",
    "RecorderWarpController",
    "WarpPhase",
    "make_recorder_warp_controller",
]

DEATH_MODE = 17
CAVE_MODE = 16
# probe_recorder_warp_full_route.py, 3/3 identical: 20f of facing input, 12
# B presses, then a 90f xy-stable settle before the next blow. The settle also
# waits for $0508 to clear: the whirlwind needs ~272f to reach Link, and a
# stand-still settle alone re-faced (moved) Link under it, so it missed him on
# 0x0B and $0508 stayed 1 until a screen change (rr-p8rg, Spine_fail probe).
FACE_FRAMES = 20
BLOW_PRESSES = 12
SETTLE_STABLE_FRAMES = 90
SETTLE_MAX_FRAMES = 3000
# A landing on a dungeon's door screen can carry Link in (0x24 -> 0x22 walked
# into L6, natural_credits_poweron47). Walk back out its south door for at
# most this many frames, then the landing reads as that door screen.
DUNGEON_EXIT_MAX_FRAMES = 600
# Recon needed 8; the cap leaves margin for a no-op blow off a door screen.
MAX_BLOWS = 12
# The whirlwind is object $2E while it crosses. $0508 set with no $2E on screen
# in play mode means it passed Link by: an enemy knocked him off its row
# (resl7_warpfix, a red Leever on 0x0B, 8.5 hearts). $0508 then stays set
# until a screen load, so the recovery walks off the screen and blows again.
WHIRLWIND_OBJECT_TYPE = 0x2E
WHIRLWIND_MISS_FRAMES = 30
LEAVE_DIRECTIONS = ("DOWN", "UP", "LEFT", "RIGHT")
LEAVE_DIRECTION_FRAMES = 600
LEAVE_WALK_IN_FRAMES = 32
WARP_MAX_FRAMES = 40_000
WARP_FACING = "DOWN"
# Whirlwind destinations by dungeon (live cycle above; 0x37 → UP → 0x3C
# measured from Spine_fail). Facing DOWN summons the next lower owned level,
# UP the next higher, and the summon is spent even when the carry misses Link
# (0x0B DOWN miss → next DOWN lands 0x74, not 0x45: resl7_warpfix2).
WARP_DOOR_LEVELS: dict[int, int] = {
    0x37: 1,
    0x3C: 2,
    0x74: 3,
    0x45: 4,
    0x0B: 5,
    0x22: 6,
}


_FACING_CODE = {"UP": 0x08, "DOWN": 0x04, "RIGHT": 0x01, "LEFT": 0x02}


class WarpPhase(Enum):
    SELECT = auto()
    FACE = auto()
    BLOW = auto()
    SETTLE = auto()
    LEAVE = auto()
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
    _missed: int = field(default=0, init=False, repr=False)
    _leave: LatticeDoorWalker = field(default_factory=LatticeDoorWalker, init=False, repr=False)
    _leave_dir: int = field(default=0, init=False, repr=False)
    _leave_from: int = field(default=0, init=False, repr=False)
    _walk_in: int = field(default=0, init=False, repr=False)
    misses: int = 0
    dungeon_exits: int = 0
    _exit_frames: int = 0
    facings: list[str] = field(default_factory=list)
    _cursor: int | None = field(default=None, init=False, repr=False)
    _blow_face: str = field(default=WARP_FACING, init=False, repr=False)

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
            return self._to_face("recorder_ready")
        return action

    def _settled(self, snap: ZeldaSnapshot) -> bool:
        xy = (int(snap.link_x), int(snap.link_y))
        carried = int(read_u8(self._env.get_ram(), ADDR_WHIRLWIND_SUMMONED)) != 0
        if (
            snap.mode == PLAY_MODE
            and not snap.transitioning
            and not carried
            and xy == self._last_xy
        ):
            self._stable += 1
        else:
            self._stable = 0
        self._last_xy = xy
        return self._stable >= SETTLE_STABLE_FRAMES

    def _owned_levels(self, snap: ZeldaSnapshot) -> list[int]:
        return sorted(
            lvl for lvl in WARP_DOOR_LEVELS.values() if int(snap.triforce) & (1 << (lvl - 1))
        )

    def _choose_facing(self) -> str:
        """Toward the target level from the last summoned one.

        Unknown cursor (first blow off a plain screen) or cursor already on the
        target (a spent miss) keeps ``facing``; the next landing re-anchors it.
        """
        want = WARP_DOOR_LEVELS.get(self.target_screen)
        if self._cursor is None or want is None or want == self._cursor:
            return self.facing
        return "DOWN" if want < self._cursor else "UP"

    def _spend_cursor(self, snap: ZeldaSnapshot) -> None:
        """A missed carry still used its summon: step the cursor with it."""
        owned = self._owned_levels(snap)
        if self._cursor is None or self._cursor not in owned:
            self._cursor = None
            return
        i = owned.index(self._cursor)
        i = i - 1 if self._blow_face == "DOWN" else i + 1
        self._cursor = owned[i % len(owned)]

    def _to_face(self, note: str) -> FrameAction:
        self._blow_face = self._choose_facing()
        self._set_phase(WarpPhase.FACE, note)
        return FrameAction(nes_action(self._blow_face), "warp_face")

    def _whirlwind_missed(self, snap: ZeldaSnapshot) -> bool:
        ram = self._env.get_ram()
        crossing = any(
            int(read_u8(ram, ADDR_OBJ_TYPE + slot)) == WHIRLWIND_OBJECT_TYPE
            for slot in range(1, 12)
        )
        if (
            snap.mode == PLAY_MODE
            and not snap.transitioning
            and int(read_u8(ram, ADDR_WHIRLWIND_SUMMONED))
            and not crossing
        ):
            self._missed += 1
        else:
            self._missed = 0
        return self._missed >= WHIRLWIND_MISS_FRAMES

    def _run_leave(self, snap: ZeldaSnapshot) -> FrameAction:
        """Lattice-walk off ``_leave_from``; the screen load clears $0508."""
        if int(snap.screen) != self._leave_from:
            if snap.mode == PLAY_MODE and not snap.transitioning:
                # Blows on the arrival edge no-oped 5x on 0x1B (resl7_warpfix4);
                # step clear of it first.
                self._walk_in += 1
                if self._walk_in < LEAVE_WALK_IN_FRAMES:
                    step = LEAVE_DIRECTIONS[min(self._leave_dir, len(LEAVE_DIRECTIONS) - 1)]
                    return FrameAction(nes_action(step), "warp_leave_walk_in")
                self._note(f"warp_left_to_0x{int(snap.screen):02x}")
                return self._to_face("warp_reblow")
            return FrameAction(nes_idle_action(), "warp_leave_scroll")
        if not snap.mode == PLAY_MODE or snap.transitioning:
            return FrameAction(nes_idle_action(), "warp_leave_scroll")
        while self._leave_dir < len(LEAVE_DIRECTIONS):
            direction = LEAVE_DIRECTIONS[self._leave_dir]
            if self.phase_frames <= LEAVE_DIRECTION_FRAMES * (self._leave_dir + 1):
                act = self._leave.action(self._env, snap, direction, "warp_leave")
                if act is not None:
                    return act
            self._leave_dir += 1
            self._leave = LatticeDoorWalker()
        return self._fail(f"warp_leave_boxed_0x{self._leave_from:02x}")

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
            if self.phase is WarpPhase.SETTLE and self._exit_frames < DUNGEON_EXIT_MAX_FRAMES:
                if self._exit_frames == 0:
                    self.dungeon_exits += 1
                    self._note(f"warp_carried_into_L{snap.level}")
                self._exit_frames += 1
                return FrameAction(nes_action("DOWN"), "warp_dungeon_exit")
            return self._fail(f"warp_left_overworld_L{snap.level}")
        self._exit_frames = 0
        if snap.mode == CAVE_MODE:
            return self._fail("warp_entered_cave")
        if int(read_u8(self._env.get_ram(), ADDR_WHISTLE)) < 1:
            return self._fail("warp_requires_whistle")

        if self.phase is WarpPhase.SELECT:
            if int(snap.screen) != self.launch_screen:
                return self._fail(f"warp_not_on_launch_0x{int(snap.screen):02x}")
            return self._run_select(snap)
        if self.phase is WarpPhase.FACE:
            # Turn, do not walk: 20f of UP on the L3 landing walked Link into
            # the 0x74 door (resl7_warpfix3). One press turns him.
            turned = int(snap.facing) == _FACING_CODE[self._blow_face]
            if (turned and self.phase_frames > 1) or self.phase_frames >= FACE_FRAMES:
                self.facings.append(self._blow_face)
                self._set_phase(WarpPhase.BLOW, "warp_blow")
                return FrameAction(nes_action("B"), "warp_blow")
            return FrameAction(nes_action(self._blow_face), "warp_face")
        if self.phase is WarpPhase.BLOW:
            if self.phase_frames >= BLOW_PRESSES:
                self.blows += 1
                self._stable = 0
                self._last_xy = None
                self._set_phase(WarpPhase.SETTLE, "warp_settle")
                return FrameAction(nes_idle_action(), "warp_settle")
            return FrameAction(nes_action("B"), "warp_blow")
        if self.phase is WarpPhase.LEAVE:
            return self._run_leave(snap)
        if self.phase is WarpPhase.SETTLE:
            if self._whirlwind_missed(snap):
                self.misses += 1
                self._spend_cursor(snap)
                self._leave_from = int(snap.screen)
                self._leave_dir = 0
                self._walk_in = 0
                self._leave = LatticeDoorWalker()
                self._set_phase(WarpPhase.LEAVE, "warp_whirlwind_missed_leave")
                return self._run_leave(snap)
            if self.phase_frames > SETTLE_MAX_FRAMES:
                if read_u8(self._env.get_ram(), ADDR_WHIRLWIND_SUMMONED):
                    return self._fail("warp_whirlwind_missed")
                return self._fail("warp_settle_timeout")
            if not self._settled(snap):
                return FrameAction(nes_idle_action(), "warp_settle")
            screen = int(snap.screen)
            self.landings.append(f"0x{screen:02x}")
            self._cursor = WARP_DOOR_LEVELS.get(screen, self._cursor)
            if screen == self.target_screen:
                return self._finish(f"warp_landed_0x{screen:02x}")
            if self.blows >= self.max_blows:
                return self._fail(f"warp_target_unreached_0x{screen:02x}")
            return self._to_face("warp_next_blow")
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
            "facings": list(self.facings),
            "blows": self.blows,
            "misses": self.misses,
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
