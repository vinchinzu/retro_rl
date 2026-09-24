"""Level 7 Demon pond: post-L6 walk, pause-select Recorder, drain into 0x79.

Scratch probes live in ``scratch/pond/``. Drain recipe, re-measured live from
the naturally-arrived pin ``OW_L7PondNatural`` (``scratch/pond/
capture_pond_natural_pin.py`` + ``scratch/pond/probe_dump_pond_drained.py``):
OW ``$EB=0x42`` south shore ``(128,221)`` → stand ``(128,189)`` → 12×B + idle
~240 for the song → stairs ``(96,144)`` tile ``0x70`` → L7 play ``0x79``
``(120,205)``.  Requires already-owned Whistle (``ADDR_WHISTLE`` / ``$065C``
>= 1).  Never poke whistle or ``$0656`` selected_item.

The stair coordinates above replace an earlier ``(96,132)`` tile-114 guess
from ``scratch/pond/probe_l7_pond_drain.py`` drain_v2 / ``LEVEL7_ROUTE.md``.
That recon was measured from the ``OW_L7Pond`` pin with a **poked** Whistle
and a different approach vector (``PostSwordStart`` geometry walk, not a
natural post-warp arrival); its stair position never matched the drained
``$6530`` tile map from a real arrival and the live drain always failed
``stairs_not_found``. The re-measured cell (``$6530`` cols 12-13, rows
10-11) is confirmed by walking onto it and observing mode 16 → L7 play
``0x79`` ``(120,205)``.

Pause-select B-slot 5 through ``dungeon.pause_select``. Snapshot has no
``selected_item``; read ``ADDR_SELECTED_ITEM`` after ``bind_env``.

Drain occupancy is waypoint-only (overworld south shore sits outside dungeon
bounds). Halt on the first occupancy miss (do not batch). No RAM writes.
``route_eligible=False``.
"""

from __future__ import annotations

from zelda_i.dungeon.hop_controller import room_step

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.anchors import SCREEN_LEVEL6_ENTRANCE
from zelda_i.dungeon.pause_select import (
    B_SLOT_RECORDER,
    PauseSelectController,
)
from zelda_i.level7.overworld import (
    POST_L6_TO_POND_HOPS,
    at_l6_cave_mouth,
    bait_32_north_action,
    make_pond_53_walker,
    on_level7_pond_hyp,
    pond_suffix_extra_hop_action,
)
from zelda_i.overworld.graph import ScreenHop
from zelda_i.overworld.path import OverworldPathController
from zelda_i.overworld.stitch import UNMEASURED_HANDOFF, OverworldHandoff
from zelda_i.ram import (
    ADDR_WHISTLE,
    PLAY_MODE,
    ZeldaSnapshot,
    read_u8,
)

__all__ = [
    "BLOW_STAND",
    "DEST",
    "DEST_XY",
    "LEVEL7",
    "POND_SCREEN",
    "POND_STAIR_TILE",
    "SOUTH_SHORE",
    "STAIR_CANDIDATES",
    "STAIRS_XY",
    "WHISTLE_B_SLOT",
    "ApproachPhase",
    "Level7PondDrainController",
    "POST_L6_TO_POND_HOPS",
    "PondPhase",
    "PostLevel6OverworldController",
    "make_pond_drain_controller",
    "make_post_l6_overworld_controller",
]

LEVEL7 = 7
DEATH_MODE = 17
POND_SCREEN = 0x42
DEST = 0x79
DEST_XY = (120, 205)
SOUTH_SHORE = (128, 221)
BLOW_STAND = (128, 189)
STAIRS_XY = (96, 144)
POND_STAIR_TILE = 0x70
WHISTLE_B_SLOT = B_SLOT_RECORDER
# STAIRS_XY is the measured top-left corner of the drained 2x2 stair quad
# ($6530 cols 12-13, rows 10-11); the remaining candidates are the other
# three corners of that same quad, kept as fallback if the direct approach
# ever lands a few px off (probe_dump_pond_drained.py, live 2/2).
STAIR_CANDIDATES: tuple[tuple[int, int], ...] = (
    STAIRS_XY,
    (104, 144),
    (96, 152),
    (104, 152),
)
POND_MAX_FRAMES = 8000
APPROACH_MAX_FRAMES = 40_000
STAND_SETTLE_FRAMES = 8
BLOW_PRESSES = 12
BLOW_WAIT_FRAMES = 240
ARRIVE_TOL = 3
STAIR_DWELL_FRAMES = 12
STUCK_FRAMES = 16
WAIT_MODES = (2, 3, 4, 6, 7, 9, 10, 11, 16)


class PondPhase(Enum):
    SELECT = auto()
    WALK = auto()
    STAND_SETTLE = auto()
    BLOW = auto()
    BLOW_WAIT = auto()
    STAIRS = auto()
    DONE = auto()
    FAILED = auto()


def _toward(
    xy: tuple[int, int], dest: tuple[int, int], *, tol: int
) -> str | None:
    """Larger-delta cardinal, matching probe ``_seek`` / stair walk."""
    x, y = xy
    tx, ty = dest
    dx, dy = tx - x, ty - y
    if abs(dx) <= tol and abs(dy) <= tol:
        return None
    if abs(dx) >= abs(dy) and abs(dx) > tol:
        return "RIGHT" if dx > 0 else "LEFT"
    return "DOWN" if dy > 0 else "UP"


@dataclass
class Level7PondDrainController:
    """Drain live OW ``0x42`` with owned Whistle and enter play ``0x79``."""

    max_frames: int = POND_MAX_FRAMES
    phase: PondPhase = PondPhase.SELECT
    frames: int = 0
    phase_frames: int = 0
    blow_presses: int = 0
    stair_index: int = 0
    success: bool = False
    failed: bool = False
    blew: bool = False
    screen_in: int | None = None
    level_in: int | None = None
    leftover: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)
    _env: Any = field(default=None, init=False, repr=False)
    _select: PauseSelectController = field(init=False, repr=False)
    _stuck: int = field(default=0, init=False, repr=False)
    _dwell: int = field(default=0, init=False, repr=False)
    _last_xy: tuple[int, int] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._select = PauseSelectController(
            want=WHISTLE_B_SLOT, name="recorder"
        )

    def bind_env(self, env: Any) -> None:
        self._env = env
        self._select.bind_env(env)

    @property
    def cursor_moves(self) -> int:
        return self._select.cursor_moves

    def _note(self, note: str) -> None:
        if note not in self.notes:
            self.notes.append(note)

    def _set_phase(self, phase: PondPhase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            if note:
                self._note(note)

    def _fail(self, reason: str) -> FrameAction:
        self.failed = True
        self._set_phase(PondPhase.FAILED, reason)
        return FrameAction(nes_idle_action(), reason)

    def _finish(self, snap: ZeldaSnapshot, reason: str = "entered_0x79") -> FrameAction:
        self.success = True
        self.leftover = {
            "level": int(snap.level),
            "screen": int(snap.screen),
            "mode": int(snap.mode),
            "x": int(snap.link_x),
            "y": int(snap.link_y),
        }
        self._set_phase(PondPhase.DONE, reason)
        return FrameAction(nes_idle_action(), reason)

    def _begin_blow(self, note: str) -> FrameAction:
        self.blow_presses = 1
        self._set_phase(PondPhase.BLOW, note)
        return FrameAction(nes_action("B"), "whistle_blow")

    def _walk_to_stand(self, snap: ZeldaSnapshot) -> FrameAction:
        # ROM lattice: the greedy axis step flipped 112<->114 at y=205 for
        # 3200 frames once Link reached the shore off the stand column.
        direction = room_step(snap, BLOW_STAND, tol=ARRIVE_TOL)
        if direction is not None:
            axis = "y" if direction in ("UP", "DOWN") else "x"
            return FrameAction(nes_action(direction), f"stand_{axis}")
        self._set_phase(PondPhase.STAND_SETTLE, "at_stand")
        return FrameAction(nes_idle_action(), "stand_arrive")

    def _stairs_cell(self) -> tuple[int, int]:
        return STAIR_CANDIDATES[min(self.stair_index, len(STAIR_CANDIDATES) - 1)]

    def _advance_stair_candidate(self) -> None:
        missed = self._stairs_cell()
        self._note(f"stairs_miss_{missed[0]}_{missed[1]}")
        self.stair_index += 1
        self._dwell = 0
        self._stuck = 0
        self._last_xy = None

    def _walk_stairs(self, snap: ZeldaSnapshot) -> FrameAction:
        xy = (int(snap.link_x), int(snap.link_y))
        dest = self._stairs_cell()
        direction = _toward(xy, dest, tol=ARRIVE_TOL)
        if direction is None:
            self._dwell += 1
            if self._dwell >= STAIR_DWELL_FRAMES:
                if self.stair_index + 1 >= len(STAIR_CANDIDATES):
                    return self._fail("stairs_not_found")
                self._advance_stair_candidate()
            return FrameAction(nes_action("UP"), "stairs_step")
        if self._last_xy == xy:
            self._stuck += 1
            if self._stuck >= STUCK_FRAMES:
                return self._fail("occupancy_miss")
        else:
            self._stuck = 0
        self._last_xy = xy
        tag = "stairs_seek" if self.stair_index else "stairs"
        axis = "y" if direction in ("UP", "DOWN") else "x"
        return FrameAction(nes_action(direction), f"{tag}_{axis}")

    def _after_select(self, snap: ZeldaSnapshot) -> FrameAction:
        if self.phase is PondPhase.WALK:
            return self._walk_to_stand(snap)
        if self.phase is PondPhase.STAND_SETTLE:
            if self.phase_frames >= STAND_SETTLE_FRAMES:
                return self._begin_blow("whistle_ready")
            return FrameAction(nes_idle_action(), "stand_settle")
        if self.phase is PondPhase.BLOW:
            if self.blow_presses >= BLOW_PRESSES:
                self._set_phase(PondPhase.BLOW_WAIT, "whistle_wait")
                return FrameAction(nes_idle_action(), "whistle_wait")
            self.blow_presses += 1
            return FrameAction(nes_action("B"), "whistle_blow")
        if self.phase is PondPhase.BLOW_WAIT:
            if self.phase_frames >= BLOW_WAIT_FRAMES:
                self.blew = True
                self._set_phase(PondPhase.STAIRS, "seek_stairs")
                return self._walk_stairs(snap)
            return FrameAction(nes_idle_action(), "whistle_wait")
        if self.phase is PondPhase.STAIRS:
            return self._walk_stairs(snap)
        return FrameAction(nes_idle_action(), "done")

    def _run_select(self, snap: ZeldaSnapshot) -> FrameAction | None:
        action = self._select.drive(snap)
        for note in self._select.notes:
            self._note(note)
        if self._select.failed:
            return self._fail(self._select.fail_reason)
        if action is None:
            if snap.level != 0 or int(snap.screen) != POND_SCREEN:
                return self._fail("pause_close_contract_mismatch")
            self._set_phase(PondPhase.WALK, "recorder_ready")
            return self._after_select(snap)
        return action

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.phase_frames += 1
        if self.success or self.failed:
            return FrameAction(nes_idle_action(), "done")
        if self._env is None:
            return self._fail("pond_drain_env_not_bound")
        ram = self._env.get_ram()
        whistle = int(read_u8(ram, ADDR_WHISTLE))
        if self.screen_in is None:
            self.screen_in = int(snap.screen)
            self.level_in = int(snap.level)
            self._note(f"screen_in_L{snap.level}_0x{snap.screen:02x}")
        if whistle < 1:
            return self._fail("pond_requires_whistle")
        if snap.mode == DEATH_MODE:
            return self._fail("death")
        if self.frames > self.max_frames:
            return self._fail("budget_exhausted")
        dest_settled = (
            snap.level == LEVEL7
            and snap.mode == PLAY_MODE
            and not snap.transitioning
            and int(snap.screen) == DEST
        )
        if dest_settled:
            started_elsewhere = not (
                self.level_in == LEVEL7 and self.screen_in == DEST
            )
            if started_elsewhere:
                return self._finish(snap)
            return self._fail("already_in_entry_without_drain")
        if snap.transitioning or snap.mode in WAIT_MODES:
            if self.phase is PondPhase.STAIRS:
                return FrameAction(nes_idle_action(), "stairs_scroll")
            return FrameAction(nes_idle_action(), "wait_scroll")
        if snap.mode == PLAY_MODE and not snap.transitioning:
            if snap.level == 0:
                if int(snap.screen) != POND_SCREEN:
                    return self._fail(f"left_ow_0x{snap.screen:02x}")
            elif snap.level != LEVEL7:
                return self._fail(f"left_level_{snap.level}")
            elif int(snap.screen) != DEST:
                return self._fail(
                    f"left_room_L{snap.level}_0x{snap.screen:02x}"
                )
        if self.phase is PondPhase.SELECT:
            return self._run_select(snap)
        return self._after_select(snap)

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "spec_id": "level7_pond_drain_entry",
            "pond_screen": f"0x{POND_SCREEN:02X}",
            "dest": f"0x{DEST:02X}",
            "blow_stand": list(BLOW_STAND),
            "stairs_xy": list(STAIRS_XY),
            "stair_tile": POND_STAIR_TILE,
            "stair_index": self.stair_index,
            "blew": self.blew,
            "phase": self.phase.name,
            "cursor_moves": self.cursor_moves,
            "blow_presses": self.blow_presses,
            "screen_in": self.screen_in,
            "level_in": self.level_in,
            "leftover": dict(self.leftover) if self.leftover else None,
            "normal_pause_input": True,
            "writes": 0,
            "route_eligible": False,
            "evidence": "fixture-live",
            "notes": list(self.notes),
        }


def make_pond_drain_controller() -> Level7PondDrainController:
    """Fresh 0x42 whistle-drain + 0x79 entry controller (never share instances)."""
    return Level7PondDrainController()


class ApproachPhase(Enum):
    HOP = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class PostLevel6OverworldController(OverworldPathController):
    """Measured L6 leave -> greened pond-prefix hops.  Success only on 0x42.

    Refuses every frame until the shared ``OverworldHandoff`` verifies.  Once it
    does, walks ``POST_L6_TO_POND_HOPS``.  A partial table fails closed on the
    last greened screen via ``_after_hops``; OW ``0x42`` mode 5 is SUCCESS.
    Never walk UP into the 0x22 cave mouth (mode 16 → L6).
    """

    handoff: OverworldHandoff = UNMEASURED_HANDOFF
    hops: tuple[ScreenHop, ...] = POST_L6_TO_POND_HOPS
    # The spine now stops this stage on the warp launch screen 0x24 and hands
    # off to level7.warp; the pond 0x42 default keeps the standalone recon
    # walk (and its tests) unchanged.
    dest_screen: int = POND_SCREEN
    phase: ApproachPhase = ApproachPhase.HOP
    max_frames: int = APPROACH_MAX_FRAMES
    require_sword: bool = True
    _env: Any = field(default=None, init=False, repr=False)
    _handoff_checked: bool = field(default=False, init=False, repr=False)
    _left_mouth: bool = field(default=False, init=False, repr=False)
    _pond53_walk: Any = field(default=None, init=False, repr=False)

    @property
    def failed(self) -> bool:
        return self.phase is ApproachPhase.FAILED

    def bind_env(self, env: Any) -> None:
        self._env = env

    def _fail_now(self, reason: str) -> FrameAction:
        self._set_phase(ApproachPhase.FAILED, reason)
        return FrameAction(nes_idle_action(), reason)

    def _on_dest(self, snap: ZeldaSnapshot) -> bool:
        if self.dest_screen == POND_SCREEN:
            return on_level7_pond_hyp(snap)
        return (
            snap.level == 0
            and snap.mode == PLAY_MODE
            and int(snap.screen) == self.dest_screen
        )

    def _after_hops(self, snap: ZeldaSnapshot) -> FrameAction:
        if self._on_dest(snap):
            return self._finish(f"post_l6_dest_0x{self.dest_screen:02x}")
        return self._fail_now("post_l6_path_exhausted_unmeasured")

    def _on_hop_advanced(
        self, snap: ZeldaSnapshot, completed_hop: ScreenHop
    ) -> FrameAction:
        del completed_hop
        if self._on_dest(snap):
            return self._finish(f"post_l6_dest_0x{self.dest_screen:02x}")
        if self.hop_index >= len(self.hops):
            return self._after_hops(snap)
        return FrameAction(nes_idle_action(), "hop_advance")

    def _pond53_walker(self):
        if self._pond53_walk is None:
            self._pond53_walk = make_pond_53_walker()
        return self._pond53_walk

    def _extra_hop_action(
        self, snap: ZeldaSnapshot, hop: ScreenHop
    ) -> FrameAction | None:
        # DELETED 2026-09-04: hop.target == 0x21 (pond_22_to_21_action) and
        # hop.target == 0x25 (bait_24_east_action) special-cases. Neither
        # target is in POST_L6_TO_POND_HOPS -- confirmed dead code (this hop
        # table never produces those targets, so the branches never fired).
        # pond_22_to_21_action / make_pond_22_walker stay as standalone,
        # directly-tested recon functions in overworld.py (0x22 west edge
        # is proven dead, docs/LEVEL7_ROUTE.md); bait_24_east_action stays
        # live in OverworldToBaitShopController for POST_L6_TO_BAIT_HOPS.
        extra = pond_suffix_extra_hop_action(
            snap, hop, swing=self._swing, pond53_walker=self._pond53_walker()
        )
        if extra is not None:
            return extra
        if hop.target == 0x33:
            act = bait_32_north_action(snap, swing=self._swing)
            if act is not None:
                return act
        # A stall goes to the ladder's unstick rung (a lattice step to the
        # exit band). The idle that stood here outranked every rung that
        # could move Link, and idling only grows the stuck count.
        return None

    def _reentry_refusal(self, snap: ZeldaSnapshot) -> str | None:
        if snap.level == 6:
            return "l6_dungeon_enter"
        if snap.mode == 16 and snap.screen == SCREEN_LEVEL6_ENTRANCE:
            return "l6_cave_mouth_enter"
        if snap.in_cave:
            return "unexpected_cave"
        if not at_l6_cave_mouth(snap):
            self._left_mouth = True
        elif self._left_mouth:
            return "l6_cave_mouth_reentry"
        return None

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        if self.failed:
            return FrameAction(nes_idle_action(), "failed")
        if not self._handoff_checked:
            if self._env is None:
                return self._fail_now("entry_controller_env_not_bound")
            mismatch = self.handoff.mismatch(snap, self._env.get_ram())
            if mismatch is not None:
                return self._fail_now(mismatch)
            if not self.hops:
                return self._fail_now("post_l6_path_unmeasured")
            self._handoff_checked = True
            self.notes.append("post_l6_handoff_accepted")
        reentry = self._reentry_refusal(snap)
        if reentry is not None:
            return self._fail_now(reentry)
        return super().step(snap)

    def report(self) -> dict[str, Any]:
        out = super().report()
        out.update(
            {
                "evidence": self.handoff.evidence,
                "route_eligible": self.handoff.route_eligible,
                "dest_screen": f"0x{self.dest_screen:02X}",
                "failed": self.failed,
                "writes": 0,
            }
        )
        return out


def make_post_l6_overworld_controller(
    *,
    handoff: OverworldHandoff = UNMEASURED_HANDOFF,
    hops: tuple[ScreenHop, ...] = POST_L6_TO_POND_HOPS,
    dest_screen: int = POND_SCREEN,
) -> PostLevel6OverworldController:
    return PostLevel6OverworldController(
        handoff=handoff, hops=hops, dest_screen=dest_screen
    )
