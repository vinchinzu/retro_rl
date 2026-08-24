"""Level 4 0x30/0x31/0x32/0x60 path controllers (clear, key-right, stepladder).

Room specs and stop predicates remain in ``level4_dungeon``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.dungeon import DungeonPhase, GenericDungeonRoomController
from zelda_i.level4_north30 import (
    Level4North30Controller,
    North30Phase,
    make_north_30_controller,
)
from zelda_i.level4_occupancy import ROOM_60_WAYPOINTS
from zelda_i.level4_dungeon import (
    KEY_30_EAST_Y,
    KEY_30_EAST_Y_TOL,
    LADDER_60_PICKUP_XY,
    LEVEL4,
    PUSH_32_DIR,
    PUSH_32_STAND,
    ROOM_30_SPEC,
    ROOM_32_SPEC,
    ROOM_L4_EAST_31,
    ROOM_L4_EAST_32,
    ROOM_L4_NORTH_30,
    ROOM_L4_STEPLADDER,
    STAIRS_32_APPROACH,
)
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot

# 0x31 maze → east door band then free RIGHT (hold4 BFS; rr-resv).
# hold=6/q=8 starves connectivity from clear pose ~(128,133); hold4/q4 reaches east.
MAZE_31_HOLD = 4
MAZE_31_CELL_Q = 4

PUSH_32_HOLD = 200
STAIRS_32_PUSH = "UP"
STAIRS_32_PUSH_FRAMES = 120
# Isolated leftover ~(48,69). Token replay of that BFS is not a spine path.
MAZE_60_HOLD = 4
MAZE_60_SPAWN_XY = (48, 69)
MAZE_60_SETTLE = 30
# v13: UP at x=152 y=189 is solid; UP+LEFT slides west. Clip on the stairs
# column (x=160) with UP, then LEFT+UP once y drops (do not LEFT on y=189).
CLIP_60: tuple[tuple[int, str, str], ...] = (
    (189, "UP", "RIGHT"),
    (173, "UP", "LEFT"),
    (157, "LEFT", "UP"),
    (173, "LEFT", "UP"),
    (157, "UP", "LEFT"),
)
CLIP_60_BUDGET = 48
CLIP_60_OPEN_X = 54
CLIP_60_EXIT_X = 176
MAZE_60_TO_LADDER: tuple[str, ...] = (
    "UP",
    "UP",
    "UP",
    "UP",
    "DOWN",
    "DOWN",
    "DOWN",
    "DOWN",
    "DOWN",
    "DOWN",
    "DOWN",
    "DOWN",
    "DOWN",
    "DOWN",
    "DOWN",
    "DOWN",
    "DOWN",
    "DOWN",
    "DOWN",
    "DOWN",
    "DOWN",
    "DOWN",
    "RIGHT",
    "UP",
    "RIGHT",
    "UP",
    "RIGHT",
    "UP",
    "RIGHT",
    "UP",
    "RIGHT",
    "UP",
    "RIGHT",
    "UP",
    "RIGHT",
    "UP",
    "RIGHT",
    "UP",
    "RIGHT",
    "UP",
    "RIGHT",
    "UP",
    "RIGHT",
    "UP",
    "RIGHT",
    "UP",
    "RIGHT",
    "UP",
    "RIGHT",
    "UP",
    "RIGHT",
    "UP",
    "RIGHT",
    "UP",
    "UP",
    "UP",
    "UP",
    "UP",
    "UP",
    "UP",
    "UP",
    "UP",
    "LEFT",
    "UP",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
)

# --- Post-ladder residual (rr-05fz live 2026-08-10) ---
# Item-pickup freeze on Level4Stepladder: need idle before movement works.
POST_LADDER_ITEM_SETTLE = 150
# mode-9 0x60 exit: clear 4× Keese then hold4 multi-grid BFS → 0x32 play.
# Live sample path from keese-clear pose ~(112,141) (rr-05fz):
EXIT_60_HOLD = 4
EXIT_60_SAMPLE_PATH: tuple[str, ...] = (
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "DOWN",
    "RIGHT",
    "DOWN",
    "DOWN",
    "DOWN",
    "DOWN",
    "DOWN",
    "DOWN",
    "DOWN",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "UP",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "UP",
    "UP",
    "UP",
    "UP",
    "UP",
    "UP",
    "UP",
    "UP",
    "UP",
    "UP",
    "UP",
    "UP",
    "UP",
    "UP",
    "UP",
    "UP",
    "UP",
    "UP",
    "UP",
    "UP",
    "UP",
    "UP",
    "UP",
    "UP",
    "UP",
)
# Post-ladder 0x32 → free LEFT to 0x31 (BFS around pushed 0x68 block; y≈141 west).
WEST_31_HOLD = 4
WEST_31_SAMPLE_PATH: tuple[str, ...] = (
    "LEFT",
    "LEFT",
    "DOWN",
    "DOWN",
    "LEFT",
    "DOWN",
    "DOWN",
    "DOWN",
    "DOWN",
    "DOWN",
    "DOWN",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
)


# --- 0x30 Vire clear from north band (rr-n1wn) ---

# Link cannot walk north of y≈128 (solid wall). Vires fly above and dip into
# the walkable band — clear by east-west patrol on the north band facing UP.
_NORTH_BAND_Y = 133
_NORTH_BAND_Y_MAX = 148
_CLEAR30_PATROL_X: tuple[int, ...] = (40, 80, 120, 160, 200, 160, 120, 80)


class Clear30Phase(Enum):
    TO_BAND = auto()
    FIGHT = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class Level4Clear30Controller:
    """Clear 3× Vire on 0x30 from the north walkable band (ignore 0x2b).

    Live (rr-n1wn): walkable cells y∈[128,208]; flyers above wall need UP
    slashes when they share x or dip into the band. Generic mid-room chase
    starves damage.
    """

    max_frames: int = 20000
    phase: Clear30Phase = Clear30Phase.TO_BAND
    frames: int = 0
    phase_frames: int = 0
    combat_frames: int = 0
    patrol_index: int = 0
    max_live_enemies: int = 0
    last_live_enemies: int = 0
    success: bool = False
    notes: list[str] = field(default_factory=list)

    def _set_phase(self, phase: Clear30Phase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            if note:
                self.notes.append(note)

    def _fail(self, note: str) -> FrameAction:
        self._set_phase(Clear30Phase.FAILED, note)
        return FrameAction(nes_idle_action(), note)

    def _live(self, snap: ZeldaSnapshot) -> tuple:
        return ROOM_30_SPEC.live_enemies(snap)

    def _swing(self, direction: str, reason: str) -> FrameAction:
        # period 6 hold 3 — same as ROOM_30_SPEC combat tuning
        if (self.combat_frames % 6) < 3:
            return FrameAction(nes_action(direction, "A"), f"{reason}_slash")
        return FrameAction(nes_action(direction), reason)

    def _fight_step(self, snap: ZeldaSnapshot) -> FrameAction:
        from zelda_i.combat import should_swing_at

        self.combat_frames += 1
        live = self._live(snap)
        self.last_live_enemies = len(live)
        self.max_live_enemies = max(self.max_live_enemies, len(live))

        if (
            not live
            and self.max_live_enemies >= ROOM_30_SPEC.expected_enemy_count
        ):
            self.success = True
            self._set_phase(Clear30Phase.DONE, "room_cleared")
            return FrameAction(nes_idle_action(), "done")

        if not live:
            # Seen empty before expected count — keep patrolling briefly.
            return FrameAction(nes_action("UP"), "wait_spawn")

        # Prefer targets in/near the walkable band; else any Vire for x-align.
        band = [o for o in live if o.y >= 112]
        targets = band if band else list(live)
        nearest = min(
            targets,
            key=lambda o: abs(o.x - snap.link_x) + abs(o.y - snap.link_y),
        )
        dx = nearest.x - snap.link_x
        dy = nearest.y - snap.link_y

        # Stay on north band.
        if snap.link_y > _NORTH_BAND_Y_MAX:
            return FrameAction(nes_action("UP"), "return_north_band")

        # Flyer above: face UP and slash when roughly under them.
        above = nearest.y < snap.link_y - 6
        if above and abs(dx) <= 28:
            return self._swing("UP", "slash_up_flyer")
        if above and abs(dx) > 8:
            direction = "RIGHT" if dx > 0 else "LEFT"
            # Keep a light UP bias so we don't drift south while aligning.
            if snap.link_y > _NORTH_BAND_Y + 4:
                return FrameAction(nes_action("UP"), "reband_while_align")
            return FrameAction(nes_action(direction), "align_x_flyer")

        # Target in band: close then slash.
        if abs(dy) > 10 and nearest.y >= 112:
            direction = "DOWN" if dy > 0 else "UP"
        elif abs(dx) > 8:
            direction = "RIGHT" if dx > 0 else "LEFT"
        else:
            direction = "UP" if above or abs(dy) <= 10 else (
                "DOWN" if dy > 0 else "UP"
            )

        if should_swing_at(
            snap.link_x, snap.link_y, direction, (nearest,)
        ) or (abs(dx) <= 16 and abs(dy) <= 28):
            return self._swing(direction, "engage")

        # No close target — east-west patrol on the band.
        tx = _CLEAR30_PATROL_X[self.patrol_index % len(_CLEAR30_PATROL_X)]
        if abs(snap.link_x - tx) <= 6:
            self.patrol_index += 1
            tx = _CLEAR30_PATROL_X[self.patrol_index % len(_CLEAR30_PATROL_X)]
        if snap.link_y > _NORTH_BAND_Y + 6:
            return FrameAction(nes_action("UP"), "patrol_reband")
        direction = "RIGHT" if snap.link_x < tx else "LEFT"
        return FrameAction(nes_action(direction), "patrol_band")

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.phase_frames += 1

        if self.phase is Clear30Phase.DONE:
            return FrameAction(nes_idle_action(), "done")
        if self.phase is Clear30Phase.FAILED:
            return FrameAction(nes_idle_action(), "failed")
        if snap.mode == 17:
            return self._fail("link_death")
        if self.frames >= self.max_frames:
            return self._fail("timeout")
        if snap.level != LEVEL4:
            return FrameAction(nes_idle_action(), "wait_level4")
        if snap.transitioning or snap.mode in (4, 6, 7):
            return FrameAction(nes_action("UP"), "scroll")
        if snap.mode == 8:
            return FrameAction(nes_idle_action(), "hurt_freeze")
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.screen != ROOM_L4_NORTH_30:
            return self._fail(f"wrong_room_0x{snap.screen:02x}")

        live = self._live(snap)
        self.last_live_enemies = len(live)
        self.max_live_enemies = max(self.max_live_enemies, len(live))
        if (
            not live
            and self.max_live_enemies >= ROOM_30_SPEC.expected_enemy_count
        ):
            self.success = True
            self._set_phase(Clear30Phase.DONE, "room_cleared")
            return FrameAction(nes_idle_action(), "done")

        if self.phase is Clear30Phase.TO_BAND:
            if snap.link_y <= _NORTH_BAND_Y_MAX and abs(snap.link_x - 120) <= 40:
                self._set_phase(Clear30Phase.FIGHT, "on_north_band")
            else:
                if abs(snap.link_x - 120) > 6 and snap.link_y > 160:
                    return FrameAction(
                        nes_action("RIGHT" if snap.link_x < 120 else "LEFT"),
                        "center_x_south",
                    )
                if snap.link_y > _NORTH_BAND_Y:
                    return FrameAction(nes_action("UP"), "walk_north_band")
                self._set_phase(Clear30Phase.FIGHT, "on_north_band")

        return self._fight_step(snap)

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "phase": self.phase.name,
            "frames": self.frames,
            "combat_frames": self.combat_frames,
            "max_live_enemies": self.max_live_enemies,
            "last_live_enemies": self.last_live_enemies,
            "notes": list(self.notes),
            "segment": "level4_clear_0x30",
            "patrol_x": list(_CLEAR30_PATROL_X),
            "north_band_y": _NORTH_BAND_Y,
        }


def make_room_30_clear_controller() -> Level4Clear30Controller:
    """Clear 0x30 Vires from north band (ignore invuln 0x2b; rr-n1wn)."""
    return Level4Clear30Controller()


class KeyRight31Phase(Enum):
    CLEAR = auto()
    ALIGN = auto()
    PUSH = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class Level4KeyRight31Controller:
    """From 0x30 with ≥1 key: optional clear Vires, then KEY-RIGHT into 0x31.

    Live (rr-n1wn): hold RIGHT @ y≈141; keys 1→0; enter west door ~(16,141).
    0x31 has 5× Vire ``0x12``. Free N/E/W sealed; KEY-LEFT none; DOWN→0x40.
    """

    clear_vires: bool = True
    max_frames: int = 25000
    phase: KeyRight31Phase = KeyRight31Phase.CLEAR
    frames: int = 0
    phase_frames: int = 0
    success: bool = False
    keys_before: int | None = None
    notes: list[str] = field(default_factory=list)
    _clear: Level4Clear30Controller | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.clear_vires:
            self._clear = Level4Clear30Controller()
        else:
            self.phase = KeyRight31Phase.ALIGN

    def _set_phase(self, phase: KeyRight31Phase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            if note:
                self.notes.append(note)

    def _fail(self, note: str) -> FrameAction:
        self._set_phase(KeyRight31Phase.FAILED, note)
        return FrameAction(nes_idle_action(), note)

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.phase_frames += 1

        if self.phase is KeyRight31Phase.DONE:
            return FrameAction(nes_idle_action(), "done")
        if self.phase is KeyRight31Phase.FAILED:
            return FrameAction(nes_idle_action(), "failed")
        if snap.mode == 17:
            return self._fail("link_death")
        if self.frames >= self.max_frames:
            return self._fail("timeout")

        if (
            snap.level == LEVEL4
            and snap.screen == ROOM_L4_EAST_31
            and snap.mode == PLAY_MODE
            and not snap.transitioning
        ):
            self.success = True
            self._set_phase(KeyRight31Phase.DONE, "entered_0x31")
            return FrameAction(nes_idle_action(), "done")

        if snap.level != LEVEL4:
            return FrameAction(nes_idle_action(), "wait_level4")
        if snap.transitioning or snap.mode in (4, 6, 7):
            return FrameAction(nes_action("RIGHT"), "scroll_right")
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")

        if self.phase is KeyRight31Phase.CLEAR:
            if snap.screen != ROOM_L4_NORTH_30:
                return self._fail(f"clear_wrong_room_0x{snap.screen:02x}")
            assert self._clear is not None
            live = ROOM_30_SPEC.live_enemies(snap)
            # Pre-cleared checkpoint (Level4Room30Cleared) or fight just finished.
            if not live and (
                self._clear.max_live_enemies >= 3
                or self._clear.success
                or self.phase_frames <= 2
            ):
                self.keys_before = snap.keys
                note = (
                    "cleared_0x30"
                    if self._clear.max_live_enemies >= 3 or self._clear.success
                    else "precleared_0x30"
                )
                self._set_phase(KeyRight31Phase.ALIGN, note)
            else:
                return self._clear.step(snap)

        if snap.screen not in (ROOM_L4_NORTH_30, ROOM_L4_EAST_31):
            return self._fail(f"wrong_room_0x{snap.screen:02x}")

        if self.keys_before is None:
            self.keys_before = snap.keys
        if self.keys_before is not None and self.keys_before < 1 and snap.keys < 1:
            return self._fail("no_keys")

        if abs(snap.link_y - KEY_30_EAST_Y) > KEY_30_EAST_Y_TOL:
            self._set_phase(KeyRight31Phase.ALIGN, "align_y")
            return FrameAction(
                nes_action("UP" if snap.link_y > KEY_30_EAST_Y else "DOWN"),
                "align_y",
            )
        self._set_phase(KeyRight31Phase.PUSH, "push_key_right")
        return FrameAction(nes_action("RIGHT"), "push_key_right")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "phase": self.phase.name,
            "frames": self.frames,
            "notes": list(self.notes),
            "segment": "level4_key_right_0x31",
            "target_room": f"0x{ROOM_L4_EAST_31:02x}",
            "key_y": KEY_30_EAST_Y,
            "keys_before": self.keys_before,
        }


def make_key_right_31_controller(
    *, clear_vires: bool = True
) -> Level4KeyRight31Controller:
    """0x30 → KEY-RIGHT @y141 → 0x31 (5× Vire). Optionally clear Vires first."""
    return Level4KeyRight31Controller(clear_vires=clear_vires)



class StepladderPhase(Enum):
    CLEAR = auto()
    ALIGN_PUSH = auto()
    PUSH = auto()
    APPROACH_STAIRS = auto()
    ENTER_STAIRS = auto()
    SETTLE_STAIRS = auto()
    PATH = auto()
    HUNT = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class Level4StepladderController:
    """0x32 clear → push left block → stairs 0x60 → ADDR_LADDER (rr-tib8).

    Live dual-green: stand ~(120,141) hold LEFT; approach ~(208,96) hold UP into
    mode-9 0x60; occupancy waypoints along stairs-column x=160 UP to pedestal.
    """

    clear_first: bool = True
    max_frames: int = 35000
    phase: StepladderPhase = StepladderPhase.CLEAR
    frames: int = 0
    phase_frames: int = 0
    path_index: int = 0
    hold_left: int = 0
    probe_i: int = 0
    success: bool = False
    notes: list[str] = field(default_factory=list)
    samples: list[dict[str, Any]] = field(default_factory=list)
    _clear: GenericDungeonRoomController | None = field(default=None, repr=False)
    _hunt_i: int = 0
    _last_xy: tuple[int, int] | None = None
    _stall: int = 0

    def __post_init__(self) -> None:
        if self.clear_first:
            self._clear = GenericDungeonRoomController(ROOM_32_SPEC)
            self._clear.phase = DungeonPhase.FIGHT
        else:
            self.phase = StepladderPhase.ALIGN_PUSH

    def _set_phase(self, phase: StepladderPhase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            if note:
                self.notes.append(note)

    def _fail(self, note: str) -> FrameAction:
        self._set_phase(StepladderPhase.FAILED, note)
        return FrameAction(nes_idle_action(), note)

    def _sample(self, snap: ZeldaSnapshot, reason: str) -> None:
        sample = {
            "frame": self.frames,
            "x": int(snap.link_x),
            "y": int(snap.link_y),
            "phase": self.phase.name,
            "probe_i": self.probe_i,
            "reason": reason,
            "stall": self._stall,
        }
        if (
            not self.samples
            or self.samples[-1]["reason"] != reason
            or self.frames - self.samples[-1]["frame"] >= 250
        ):
            self.samples.append(sample)

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        # Controllers only get snap (no ADDR_LADDER field). Runner confirms
        # ``level4_stepladder_success``; we mark success near pedestal after path.
        self.frames += 1
        self.phase_frames += 1
        xy = (int(snap.link_x), int(snap.link_y))
        if self._last_xy == xy:
            self._stall += 1
        else:
            self._stall = 0
            self._last_xy = xy

        if self.phase is StepladderPhase.DONE:
            return FrameAction(nes_idle_action(), "done")
        if self.phase is StepladderPhase.FAILED:
            return FrameAction(nes_idle_action(), "failed")
        if snap.mode == 17:
            return self._fail("link_death")
        if self.frames >= self.max_frames:
            self._sample(snap, "timeout")
            return self._fail(f"timeout_{snap.link_x}_{snap.link_y}")

        if snap.transitioning or snap.mode in (4, 6, 7):
            return FrameAction(nes_action("UP"), "scroll")
        if snap.mode == 8:
            return FrameAction(nes_idle_action(), "hurt_freeze")

        if self.phase is StepladderPhase.CLEAR:
            if snap.screen != ROOM_L4_EAST_32:
                return self._fail(f"clear_wrong_room_0x{snap.screen:02x}")
            assert self._clear is not None
            live = ROOM_32_SPEC.live_enemies(snap)
            if not live and (
                self._clear.max_live_enemies >= 4
                or self._clear.success
                or self.phase_frames <= 2
            ):
                note = (
                    "cleared_0x32"
                    if self._clear.max_live_enemies >= 4 or self._clear.success
                    else "precleared_0x32"
                )
                self._set_phase(StepladderPhase.ALIGN_PUSH, note)
            else:
                return self._clear.step(snap)

        if self.phase is StepladderPhase.ALIGN_PUSH:
            if snap.screen != ROOM_L4_EAST_32:
                return self._fail(f"push_wrong_room_0x{snap.screen:02x}")
            if snap.mode not in (PLAY_MODE, 5):
                return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
            tx, ty = PUSH_32_STAND
            # Statue/block solids around center x∈[80,160] y≈128 block straight
            # south from the clear patrol band — detour west or east first.
            if (
                snap.link_y < ty - 6
                and 72 <= snap.link_x <= 168
                and abs(snap.link_x - tx) < 48
            ):
                # Prefer west aisle (also lines up for LEFT push).
                side_x = 48 if snap.link_x <= 128 else 192
                if abs(snap.link_x - side_x) > 6:
                    return FrameAction(
                        nes_action("RIGHT" if snap.link_x < side_x else "LEFT"),
                        "push_detour_x",
                    )
                if abs(snap.link_y - ty) > 4:
                    return FrameAction(
                        nes_action("DOWN" if snap.link_y < ty else "UP"),
                        "push_detour_y",
                    )
            dx, dy = tx - snap.link_x, ty - snap.link_y
            if abs(dx) <= 4 and abs(dy) <= 4:
                self._set_phase(StepladderPhase.PUSH, "at_push_stand")
            elif abs(dy) > 4 and (abs(dx) <= 8 or abs(dy) >= abs(dx)):
                return FrameAction(
                    nes_action("DOWN" if dy > 0 else "UP"), "align_push_y"
                )
            else:
                return FrameAction(
                    nes_action("RIGHT" if dx > 0 else "LEFT"), "align_push_x"
                )

        if self.phase is StepladderPhase.PUSH:
            if snap.screen != ROOM_L4_EAST_32:
                if snap.screen == ROOM_L4_STEPLADDER or snap.mode == 9:
                    self._set_phase(StepladderPhase.SETTLE_STAIRS, "stairs_mid_push")
                    return FrameAction(nes_idle_action(), "stairs_mid_push")
                return self._fail(f"push_left_room_0x{snap.screen:02x}")
            if self.phase_frames >= PUSH_32_HOLD:
                self._set_phase(StepladderPhase.APPROACH_STAIRS, "push_held")
            else:
                return FrameAction(nes_action(PUSH_32_DIR), "push_left_block")

        if self.phase is StepladderPhase.APPROACH_STAIRS:
            if snap.screen == ROOM_L4_STEPLADDER or snap.mode == 9:
                self._set_phase(StepladderPhase.SETTLE_STAIRS, "entered_stairs")
                return FrameAction(nes_idle_action(), "entered_stairs")
            if snap.screen != ROOM_L4_EAST_32:
                return self._fail(f"stairs_wrong_room_0x{snap.screen:02x}")
            if snap.mode not in (PLAY_MODE, 5):
                return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
            tx, ty = STAIRS_32_APPROACH
            # After left-block push, route NE: prefer east aisle then north.
            if snap.link_y > ty + 8 and snap.link_x < 180:
                if snap.link_x < 176:
                    return FrameAction(nes_action("RIGHT"), "stairs_east_first")
                return FrameAction(nes_action("UP"), "stairs_north_aisle")
            dx, dy = tx - snap.link_x, ty - snap.link_y
            if abs(dx) <= 4 and abs(dy) <= 4:
                self._set_phase(StepladderPhase.ENTER_STAIRS, "at_stairs_approach")
            elif abs(dy) > 4 and (abs(dx) <= 12 or abs(dy) >= abs(dx)):
                return FrameAction(
                    nes_action("DOWN" if dy > 0 else "UP"), "stairs_align_y"
                )
            else:
                return FrameAction(
                    nes_action("RIGHT" if dx > 0 else "LEFT"), "stairs_align_x"
                )

        if self.phase is StepladderPhase.ENTER_STAIRS:
            if snap.screen == ROOM_L4_STEPLADDER or snap.mode == 9:
                self._set_phase(StepladderPhase.SETTLE_STAIRS, "entered_0x60")
                return FrameAction(nes_idle_action(), "entered_0x60")
            if self.phase_frames >= STAIRS_32_PUSH_FRAMES:
                return self._fail("stairs_timeout")
            return FrameAction(nes_action(STAIRS_32_PUSH), "enter_stairs_up")

        if self.phase is StepladderPhase.SETTLE_STAIRS:
            # Idle through mode-9 scroll; scripted path only from NW spawn band.
            if snap.transitioning or snap.mode in (4, 6, 7):
                return FrameAction(nes_idle_action(), "stairs_scroll_settle")
            if snap.screen != ROOM_L4_STEPLADDER and snap.mode != 9:
                return self._fail(f"settle_wrong_room_0x{snap.screen:02x}")
            if self.phase_frames < MAZE_60_SETTLE:
                return FrameAction(nes_idle_action(), "stairs_idle_settle")
            sx, sy = MAZE_60_SPAWN_XY
            # NE (~208,93) may resettle NW; west-aisle leftover walks to spawn.
            if abs(snap.link_x - sx) <= 24:
                if self.phase_frames > MAZE_60_SETTLE + 240:
                    self._set_phase(StepladderPhase.HUNT, "spawn_join_timeout")
                    return FrameAction(nes_idle_action(), "spawn_join_timeout")
                if abs(snap.link_x - sx) > 6:
                    return FrameAction(
                        nes_action("RIGHT" if snap.link_x < sx else "LEFT"),
                        "join_spawn_x",
                    )
                if abs(snap.link_y - sy) > 4:
                    return FrameAction(
                        nes_action("DOWN" if snap.link_y < sy else "UP"),
                        "join_spawn_y",
                    )
                self._set_phase(StepladderPhase.PATH, "path_from_spawn")
                self.path_index = 0
                self.hold_left = 0
                self.probe_i = 0
                return FrameAction(nes_idle_action(), "path_from_spawn")
            if self.phase_frames < MAZE_60_SETTLE + 180:
                return FrameAction(nes_idle_action(), "wait_nw_resettle")
            self._set_phase(
                StepladderPhase.HUNT,
                f"hunt_from_nonspawn_{snap.link_x}_{snap.link_y}",
            )
            return FrameAction(nes_idle_action(), "hunt_from_nonspawn")

        if self.phase is StepladderPhase.PATH:
            if snap.mode in (4, 6, 7) or snap.transitioning:
                return FrameAction(nes_idle_action(), "path_settle")
            if snap.screen == ROOM_L4_EAST_32 and snap.mode == PLAY_MODE:
                self._sample(snap, "path_exited_to_0x32")
                return self._fail("path_exited_to_0x32")
            if snap.screen != ROOM_L4_STEPLADDER and snap.mode != 9:
                return self._fail(f"path_wrong_room_0x{snap.screen:02x}")
            if xy[0] >= CLIP_60_EXIT_X:
                self._sample(snap, f"clip_exit_{xy[0]}_{xy[1]}")
                return self._fail(f"clip_exit_{xy[0]}_{xy[1]}")
            tx, ty = LADDER_60_PICKUP_XY
            if abs(xy[0] - tx) <= 6 and abs(xy[1] - ty) <= 6:
                self._set_phase(StepladderPhase.HUNT, "at_pedestal")
                return FrameAction(nes_idle_action(), "at_pedestal")
            if 150 <= xy[1] <= 164 and xy[0] > CLIP_60_OPEN_X:
                self.path_index = max(self.path_index, 1)
            if abs(xy[1] - 158) <= 4 and xy[0] <= CLIP_60_OPEN_X and self._stall >= CLIP_60_BUDGET:
                self._sample(snap, "gap158_solid")
                return self._fail(f"gap158_solid_{xy[0]}_{xy[1]}")
            if xy[1] >= 165 and xy[0] >= 164 and self._stall >= CLIP_60_BUDGET:
                self._sample(snap, "stairs_up_solid")
                return self._fail(f"stairs_up_solid_{xy[0]}_{xy[1]}")
            if self.probe_i >= len(CLIP_60):
                self._sample(snap, "clips_done")
                return self._fail(f"clips_exhausted_{xy[0]}_{xy[1]}")
            if self._stall >= CLIP_60_BUDGET:
                gy, a, b = CLIP_60[self.probe_i]
                if abs(xy[1] - gy) > 4:
                    d = "DOWN" if xy[1] < gy else "UP"
                    return FrameAction(nes_action(d), "clip_aisle_y")
                self._sample(snap, f"clip_{a}_{b}")
                self.hold_left += 1
                if self.hold_left >= CLIP_60_BUDGET:
                    self._sample(snap, f"clip_miss_{self.probe_i}_{xy[0]}_{xy[1]}")
                    self.probe_i += 1
                    self.hold_left = 0
                    return FrameAction(nes_idle_action(), "clip_next")
                return FrameAction(nes_action(a, b), "clip_se_diag")
            if xy[0] > CLIP_60_OPEN_X:
                self._sample(snap, "strip_open")
            return self._follow_60_waypoints(xy)

        if self.phase is StepladderPhase.HUNT:
            if snap.mode in (4, 6, 7) or snap.transitioning:
                return FrameAction(nes_idle_action(), "hunt_settle")
            if snap.screen == ROOM_L4_EAST_32 and snap.mode == PLAY_MODE:
                return self._fail("hunt_exited_to_0x32")
            tx, ty = LADDER_60_PICKUP_XY
            dx, dy = tx - snap.link_x, ty - snap.link_y
            if abs(dx) <= 6 and abs(dy) <= 6:
                self._hunt_i += 1
                if self._hunt_i > 20:
                    self.success = True
                    self._set_phase(StepladderPhase.DONE, "ladder_pedestal")
                    return FrameAction(nes_idle_action(), "done")
                return FrameAction(nes_idle_action(), "hunt_idle")
            if snap.link_y >= 165:
                if snap.link_x > 54:
                    return FrameAction(nes_action("LEFT"), "hunt_south_back_west")
                return FrameAction(nes_action("UP"), "hunt_south_back_north")
            if snap.link_x >= 168 and snap.link_y >= 150:
                return FrameAction(nes_action("LEFT"), "hunt_avoid_exit")
            # SE corridor: stay west of the exit; UP toward the island.
            if snap.link_y >= 165 and snap.link_x < CLIP_60_EXIT_X:
                if snap.link_x < 168:
                    return FrameAction(nes_action("RIGHT"), "hunt_se_east")
                if snap.link_x >= 174:
                    return FrameAction(nes_action("LEFT"), "hunt_se_off_exit")
                return FrameAction(nes_action("UP"), "hunt_se_up")
            if snap.link_x > CLIP_60_OPEN_X:
                if abs(dx) > 6:
                    return FrameAction(
                        nes_action("RIGHT" if dx > 0 else "LEFT"), "hunt_x"
                    )
                if abs(dy) > 6:
                    return FrameAction(
                        nes_action("DOWN" if dy > 0 else "UP"), "hunt_y"
                    )
            if snap.link_x <= CLIP_60_OPEN_X:
                return FrameAction(nes_action("DOWN"), "hunt_aisle_south")
            if abs(dy) > 8:
                return FrameAction(
                    nes_action("DOWN" if dy > 0 else "UP"), "hunt_y_first"
                )
            if abs(dx) > 6:
                return FrameAction(
                    nes_action("RIGHT" if dx > 0 else "LEFT"), "hunt_x"
                )
            return FrameAction(nes_action("DOWN" if dy > 0 else "UP"), "hunt_y")

        return FrameAction(nes_idle_action(), "idle")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "phase": self.phase.name,
            "frames": self.frames,
            "path_index": self.path_index,
            "probe_i": self.probe_i,
            "notes": list(self.notes),
            "segment": "level4_stepladder",
            "push_stand": list(PUSH_32_STAND),
            "stairs_approach": list(STAIRS_32_APPROACH),
            "ladder_xy": list(LADDER_60_PICKUP_XY),
            "path_len": len(MAZE_60_TO_LADDER),
            "samples": list(self.samples),
        }

    def _follow_60_waypoints(self, xy: tuple[int, int]) -> FrameAction:
        """y=158 gap: between west-brick and south-water, then UP to pedestal."""
        if self.path_index >= len(ROOM_60_WAYPOINTS):
            self._set_phase(StepladderPhase.HUNT, "waypoints_done")
            return FrameAction(nes_idle_action(), "waypoints_done")
        if xy[0] <= CLIP_60_OPEN_X:
            if xy[1] < 155:
                return FrameAction(nes_action("DOWN"), "join_gap158_y")
            if xy[1] > 161:
                return FrameAction(nes_action("UP"), "join_gap158_y")
        wx, wy = ROOM_60_WAYPOINTS[self.path_index]
        if abs(xy[0] - wx) <= 4 and abs(xy[1] - wy) <= 4:
            self.path_index += 1
            return FrameAction(nes_idle_action(), "wp_next")
        if abs(xy[1] - wy) > 4:
            d = "DOWN" if xy[1] < wy else "UP"
            return FrameAction(nes_action(d), "wp_y")
        d = "RIGHT" if xy[0] < wx else "LEFT"
        return FrameAction(nes_action(d), "wp_x")


def make_stepladder_controller(*, clear_first: bool = True) -> Level4StepladderController:
    """0x32 → push left block → 0x60 → ADDR_LADDER (rr-tib8)."""
    return Level4StepladderController(clear_first=clear_first)
