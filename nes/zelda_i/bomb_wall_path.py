"""Parameterized bomb-wall traverse controller for Zelda I dungeons.

One phase machine covers every L2 (and future) bomb-wall hop. Geometry comes
from ``level2_puzzles.BombWall`` (or any compatible stand/face/opens_to object).
Optional pre-clear uses a ``DungeonRoomSpec``; optional south-band approach is
a flag (Dodongo 0x1e).

Do not add per-room phase machines — configure a ``BombWallController``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Protocol

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.dungeon import DungeonPhase, DungeonRoomSpec, GenericDungeonRoomController
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot

BOMB_N_STAND_TOL = 4
BOMB_N_WAIT_BLAST = 100
BOMB_N_STEP_BACK = 6
BOMB_N_MAX_FRAMES = 16000
# Inventory select poke lives in dungeon_ops (not this path engine).


class BombWallLike(Protocol):
    """Minimal geometry contract (satisfied by ``level2_puzzles.BombWall``)."""

    room: int
    stand: tuple[int, int]
    face: str
    opens_to: int


class BombWallPhase(Enum):
    """Shared phases for all bomb-wall traverses."""

    SETTLE = auto()
    CLEAR = auto()
    SOUTH_BAND = auto()
    TO_STAND = auto()
    FACE = auto()
    PLACE = auto()
    WAIT = auto()
    PUSH = auto()
    DONE = auto()
    FAILED = auto()


# Back-compat aliases (tests / scripts import these names).
BombNorthPhase = BombWallPhase
BoomBombNorthPhase = BombWallPhase
PostBoomBombNorthPhase = BombWallPhase
BombNorth1EPhase = BombWallPhase

ClearWhen = Callable[[ZeldaSnapshot], bool]


@dataclass
class BombWallController:
    """Frame policy: optional clear → stand → face → place bomb → wait → push.

    Success when play-ready on ``wall.opens_to`` at ``level``.
    """

    wall: BombWallLike
    level: int = 2
    # Optional room clear before bombing.
    clear_spec: DungeonRoomSpec | None = None
    clear_when: ClearWhen | None = None
    # Approach: walk south first (Dodongo 0x1e pocket).
    south_band_first: bool = False
    south_band_y: int = 170
    south_band_max_frames: int = 80
    # After south band, center x on the stand column (0x1e live: x≈120).
    south_center_max_frames: int = 200
    # Optional diamond-safe approach (0x1e: south then east column).
    approach_waypoints: tuple[tuple[int, int], ...] = ()
    approach_index: int = 0
    approach_tol: int = 4
    # Face / place / wait tuning.
    face_frames: int = 4
    step_back: int = BOMB_N_STEP_BACK
    wait_blast: int = BOMB_N_WAIT_BLAST
    # If True, hold face direction while waiting (no step-back).
    wait_hold_face: bool = False
    # Fail if bomb count did not drop during WAIT (strict L2 6f/5f policy).
    require_bomb_consumed: bool = True
    stand_tol: int = BOMB_N_STAND_TOL
    stand_timeout: int = 2500
    push_timeout: int = 700
    max_frames: int = BOMB_N_MAX_FRAMES
    # Runtime state
    phase: BombWallPhase = BombWallPhase.SETTLE
    frames: int = 0
    phase_frames: int = 0
    success: bool = False
    notes: list[str] = field(default_factory=list)
    bombs_before_place: int | None = None
    bombs_after_place: int | None = None
    clear_controller: GenericDungeonRoomController | None = None

    @property
    def stand(self) -> tuple[int, int]:
        return self.wall.stand

    @property
    def face(self) -> str:
        return self.wall.face.upper()

    @property
    def from_room(self) -> int:
        return int(self.wall.room)

    @property
    def to_room(self) -> int:
        return int(self.wall.opens_to)

    def _set_phase(self, phase: BombWallPhase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            if note:
                self.notes.append(note)

    def _fail(self, note: str) -> FrameAction:
        self._set_phase(BombWallPhase.FAILED, note)
        return FrameAction(nes_idle_action(), note)

    def _at_stand(self, snap: ZeldaSnapshot) -> bool:
        tx, ty = self.stand
        return abs(snap.link_x - tx) <= self.stand_tol and abs(
            snap.link_y - ty
        ) <= self.stand_tol

    def _goto_stand(self, snap: ZeldaSnapshot) -> FrameAction:
        """Walk to bomb stand. Prefer y-band near stand then x, else dominant axis.

        0x1e south-band approach is x-first (live): diamond mid-y blocks UP
        before the stand column is centered.
        """
        tx, ty = self.stand
        dx = tx - snap.link_x
        dy = ty - snap.link_y
        if self.south_band_first:
            if abs(dx) > self.stand_tol:
                return FrameAction(
                    nes_action("RIGHT" if dx > 0 else "LEFT"), "stand_x"
                )
            if abs(dy) > self.stand_tol:
                return FrameAction(
                    nes_action("UP" if dy < 0 else "DOWN"), "stand_y"
                )
            return FrameAction(nes_idle_action(), "stand_ready")
        if abs(snap.link_y - ty) <= 12 and abs(dx) > self.stand_tol:
            if abs(dy) > self.stand_tol:
                return FrameAction(
                    nes_action("UP" if dy < 0 else "DOWN"), "stand_band_y"
                )
            return FrameAction(
                nes_action("RIGHT" if dx > 0 else "LEFT"), "stand_band_x"
            )
        if abs(dx) > self.stand_tol and abs(dx) >= abs(dy):
            return FrameAction(
                nes_action("RIGHT" if dx > 0 else "LEFT"), "stand_x"
            )
        if abs(dy) > self.stand_tol:
            return FrameAction(
                nes_action("UP" if dy < 0 else "DOWN"), "stand_y"
            )
        return FrameAction(nes_idle_action(), "stand_ready")

    def _push_dir(self, snap: ZeldaSnapshot) -> FrameAction:
        """Align to stand x (for UP/DOWN faces) or y (for LEFT/RIGHT) then push."""
        face = self.face
        if face in ("UP", "DOWN"):
            cx = self.stand[0]
            x_tol = 3 if (
                (face == "UP" and snap.link_y <= 110)
                or (face == "DOWN" and snap.link_y >= 180)
            ) else 6
            if abs(snap.link_x - cx) > x_tol:
                return FrameAction(
                    nes_action("RIGHT" if snap.link_x < cx else "LEFT"),
                    "push_align_x",
                )
            return FrameAction(nes_action(face), f"push_{face.lower()}")
        # LEFT / RIGHT: align y then push
        cy = self.stand[1]
        if abs(snap.link_y - cy) > 6:
            return FrameAction(
                nes_action("DOWN" if snap.link_y < cy else "UP"),
                "push_align_y",
            )
        return FrameAction(nes_action(face), f"push_{face.lower()}")

    def _needs_clear(self, snap: ZeldaSnapshot) -> bool:
        if self.clear_when is not None:
            return bool(self.clear_when(snap))
        if self.clear_spec is not None:
            return bool(self.clear_spec.live_enemies(snap))
        return False

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.phase_frames += 1

        if self.phase is BombWallPhase.DONE:
            return FrameAction(nes_idle_action(), "done")
        if self.phase is BombWallPhase.FAILED:
            return FrameAction(nes_idle_action(), "failed")

        if snap.mode == 17:
            return self._fail("link_death")
        if self.frames >= self.max_frames:
            return self._fail("timeout")

        # Success: play-ready on destination room.
        if (
            snap.level == self.level
            and snap.screen == self.to_room
            and snap.mode == PLAY_MODE
        ):
            self.success = True
            self._set_phase(
                BombWallPhase.DONE, f"entered_0x{self.to_room:02x}"
            )
            return FrameAction(nes_idle_action(), "done")

        if snap.level != self.level:
            return FrameAction(nes_idle_action(), f"wait_level_{self.level}")

        if snap.transitioning or snap.mode in (4, 6, 7, 16):
            if self.phase is BombWallPhase.PUSH or snap.screen == self.to_room:
                return FrameAction(
                    nes_action(self.face), f"scroll_{self.face.lower()}"
                )
            return FrameAction(nes_idle_action(), f"settle_mode_{snap.mode}")

        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")

        if self.phase is BombWallPhase.SETTLE:
            if snap.screen != self.from_room:
                return self._fail(f"wrong_room_0x{snap.screen:02x}")
            if self._needs_clear(snap):
                if self.clear_spec is None:
                    return self._fail("clear_needed_but_no_spec")
                self.clear_controller = GenericDungeonRoomController(
                    self.clear_spec
                )
                self.clear_controller.phase = DungeonPhase.FIGHT
                self._set_phase(
                    BombWallPhase.CLEAR, f"need_clear_0x{self.from_room:02x}"
                )
            elif self.south_band_first:
                if snap.bombs <= 0:
                    return self._fail("no_bombs")
                self._set_phase(BombWallPhase.SOUTH_BAND, "south_band_first")
            else:
                if snap.bombs <= 0:
                    return self._fail("no_bombs")
                self._set_phase(BombWallPhase.TO_STAND, "already_clear")

        if self.phase is BombWallPhase.CLEAR:
            assert self.clear_controller is not None
            action = self.clear_controller.step(snap)
            if self.clear_controller.success:
                if snap.bombs <= 0:
                    return self._fail("no_bombs_after_clear")
                if self.south_band_first:
                    self._set_phase(BombWallPhase.SOUTH_BAND, "cleared_south")
                self._set_phase(BombWallPhase.TO_STAND, "cleared")
                return FrameAction(nes_idle_action(), "clear_done")
            if self.clear_controller.phase is DungeonPhase.FAILED:
                return self._fail("clear_failed")
            return action

        if self.phase is BombWallPhase.SOUTH_BAND:
            if self.approach_waypoints:
                if self.approach_index >= len(self.approach_waypoints):
                    self._set_phase(BombWallPhase.TO_STAND, "to_bomb_stand")
                    return self._goto_stand(snap)
                wx, wy = self.approach_waypoints[self.approach_index]
                atol = self.approach_tol
                if (
                    abs(snap.link_x - wx) <= atol
                    and abs(snap.link_y - wy) <= atol
                ):
                    self.approach_index += 1
                    self.notes.append(f"approach_{self.approach_index}")
                    return FrameAction(nes_idle_action(), "approach_next")
                # First waypoint is south-band (y-first); later wps keep x locked.
                y_first = self.approach_index == 0
                if y_first and abs(snap.link_y - wy) > atol:
                    return FrameAction(
                        nes_action("DOWN" if snap.link_y < wy else "UP"),
                        "approach_y",
                    )
                if abs(snap.link_x - wx) > atol:
                    return FrameAction(
                        nes_action("RIGHT" if snap.link_x < wx else "LEFT"),
                        "approach_x",
                    )
                if abs(snap.link_y - wy) > atol:
                    return FrameAction(
                        nes_action("DOWN" if snap.link_y < wy else "UP"),
                        "approach_y",
                    )
                return FrameAction(nes_idle_action(), "approach_hold")
            # Only dive south from a north pocket. Mid-y (96,141) is already
            # walkable; extra DOWN walks into a dead column (live 0x1e).
            if (
                snap.link_y < 130
                and snap.link_y < self.south_band_y
                and self.phase_frames <= self.south_band_max_frames
            ):
                return FrameAction(nes_action("DOWN"), "south_band")
            tx = self.stand[0]
            if abs(snap.link_x - tx) > self.stand_tol:
                if self.phase_frames > (
                    self.south_band_max_frames + self.south_center_max_frames
                ):
                    self._set_phase(BombWallPhase.TO_STAND, "to_bomb_stand")
                    return self._goto_stand(snap)
                return FrameAction(
                    nes_action("RIGHT" if snap.link_x < tx else "LEFT"),
                    "south_center_x",
                )
            self._set_phase(BombWallPhase.TO_STAND, "to_bomb_stand")
            return self._goto_stand(snap)

        if self.phase is BombWallPhase.TO_STAND:
            if snap.bombs <= 0:
                return self._fail("no_bombs")
            if self._at_stand(snap):
                self._set_phase(BombWallPhase.FACE, "at_bomb_stand")
                if self.south_band_first or self.wait_hold_face:
                    # 1e / post-boom style: start facing immediately.
                    return FrameAction(nes_action(self.face), f"face_{self.face.lower()}")
            elif self.phase_frames > self.stand_timeout:
                return self._fail("stand_timeout")
            else:
                return self._goto_stand(snap)

        if self.phase is BombWallPhase.FACE:
            if self.phase_frames < self.face_frames:
                return FrameAction(nes_action(self.face), f"face_{self.face.lower()}")
            self._set_phase(BombWallPhase.PLACE, "faced")
            # fall through to place same frame for 6f/5f style; 1e places on FACE end

        if self.phase is BombWallPhase.PLACE:
            if snap.bombs <= 0:
                return self._fail("no_bombs_at_place")
            self.bombs_before_place = int(snap.bombs)
            self._set_phase(BombWallPhase.WAIT, "placed_bomb")
            return FrameAction(nes_action(self.face, "B"), "place_bomb")

        if self.phase is BombWallPhase.WAIT:
            if self.bombs_after_place is None and self.bombs_before_place is not None:
                if snap.bombs < self.bombs_before_place:
                    self.bombs_after_place = int(snap.bombs)
                    self.notes.append(
                        f"bomb_used_{self.bombs_before_place}->{snap.bombs}"
                    )
            if self.wait_hold_face:
                # 1e / post-boom: hold face through blast, no step-back.
                if self.phase_frames < self.wait_blast:
                    return FrameAction(
                        nes_action(self.face), "wait_blast"
                    )
                # Soft record bomb use if we never saw a drop (don't fail).
                if (
                    self.bombs_after_place is None
                    and self.bombs_before_place is not None
                ):
                    self.bombs_after_place = int(snap.bombs)
                self._set_phase(BombWallPhase.PUSH, "blast_done")
                return self._push_dir(snap)

            # Strict 6f/5f: step back then idle wait; require bomb consumed.
            if self.phase_frames < self.step_back:
                back = {
                    "UP": "DOWN",
                    "DOWN": "UP",
                    "LEFT": "RIGHT",
                    "RIGHT": "LEFT",
                }[self.face]
                return FrameAction(nes_action(back), "step_back")
            if self.phase_frames < self.wait_blast:
                return FrameAction(nes_idle_action(), "wait_blast")
            if (
                self.require_bomb_consumed
                and self.bombs_after_place is None
                and self.bombs_before_place is not None
                and snap.bombs >= self.bombs_before_place
            ):
                return self._fail("bomb_not_consumed")
            self._set_phase(BombWallPhase.PUSH, "blast_done")
            return FrameAction(nes_action(self.face), "push_start")

        if self.phase is BombWallPhase.PUSH:
            if self.phase_frames > self.push_timeout:
                return self._fail("push_timeout")
            return self._push_dir(snap)

        return FrameAction(nes_idle_action(), "idle")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "phase": self.phase.name,
            "frames": self.frames,
            "notes": list(self.notes),
            "bombs_before_place": self.bombs_before_place,
            "bombs_after_place": self.bombs_after_place,
            "stand": list(self.stand),
            "wall_room": f"0x{self.from_room:02x}",
            "opens_to": f"0x{self.to_room:02x}",
            "clear": (
                self.clear_controller.report()
                if self.clear_controller is not None
                else None
            ),
        }


__all__ = [
    "BOMB_N_MAX_FRAMES",
    "BOMB_N_STAND_TOL",
    "BOMB_N_STEP_BACK",
    "BOMB_N_WAIT_BLAST",
    "BombNorth1EPhase",
    "BombNorthPhase",
    "BombWallController",
    "BombWallPhase",
    "BoomBombNorthPhase",
    "PostBoomBombNorthPhase",
]
