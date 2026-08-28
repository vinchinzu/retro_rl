#!/usr/bin/env python3
"""Truth-table 1-2 UG floor pipes B/C / past C for the flag exit.

Boot HappyLee 1-2 once, stay on the UG floor (never the y64 ceiling / warp
room), mid-fall DOWN each candidate with the plant hidden, and halt at the
first world=0 outdoor flag. On success, idle to 1-3 control and write the
all_exits pin. Does not touch the warp any% line.

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
  uv run python -m smb.scripts.probe_1_2_flag
```
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from retro_harness.env import read_state_bytes
from retro_harness.segment_runner import save_rgb_png
from smb.paths import RECORDINGS_DIR
from smb.play_record import write_stage_pin
from smb.ram import (
    PLAYER_STATE_AUTO_WALK,
    PLAYER_STATE_FLAGPOLE,
    player_on_ground,
    read_enemy_slots,
    read_snapshot,
)
from smb.tas.chain import reach_surface_after_hl_1_1
from smb.tas.fm2 import parse_fm2
from smb.tas.replay import IDLE, get_state, make_level1_env, set_state, to_action9
from smb.tas.stages import CONTROL_X_MAX, DEFAULT_FM2, HL_1_2_FM2_START, is_1_3_control

# NES-9: B=0, D=5, L=6, R=7, A=8
RB = [1, 0, 0, 0, 0, 0, 0, 1, 0]
RBA = [1, 0, 0, 0, 0, 0, 0, 1, 1]
RD = [0, 0, 0, 0, 0, 1, 0, 1, 0]
D = [0, 0, 0, 0, 0, 1, 0, 0, 0]
L = [0, 0, 0, 0, 0, 0, 1, 0, 0]
A = [0, 0, 0, 0, 0, 0, 0, 0, 1]
# HL lift jump: A-only 19f then idle coast, land ~ (2620, 128).
HL_LIFT_A_HOLD = 19

PIPE_A_X = 2856
PIPE_B_X = 2920
PIPE_C_X = 2984
PIPE_HALF = 16
PIPE_SLOP = 12
ENEMY_TYPE_PIRANHA = 13
PLANT_HIDDEN_Y = 158
FLOOR_Y_MIN = 148
CEILING_Y_MAX = 80  # standing on the y64 ceiling; jumps/lifts sit lower
PLAYER_STATE_PIPE_ENTER = 3
# Last grounded UG pose before HL jumps the end stairs (~x2550).
UG_MIN_X_FOR_CORRIDOR = 2000
OUT_DIR = RECORDINGS_DIR / "segments_all_exits"
EVIDENCE = OUT_DIR / "evidence"
CACHE_STATE = OUT_DIR / "hl_1_2_floor_corridor.state"

# Halt / reject labels used by the truth table and tests.
OUTDOOR_FLAG = "outdoor_flag"
WARP = "warp"
STILL_UG = "still_ug"
DEATH = "death"
NO_ENTER = "no_enter"
CEILING = "ceiling"


def plant_hidden(
    enemies: list[dict[str, int]],
    pipe_x: int,
    *,
    slop: int = 24,
) -> bool:
    """True when no piranha is up at *pipe_x* (absent or y >= hidden line)."""
    plants = [
        enemy
        for enemy in enemies
        if int(enemy.get("type", -1)) == ENEMY_TYPE_PIRANHA
        and abs(int(enemy["x"]) - pipe_x) <= slop
    ]
    if not plants:
        return True
    return all(int(enemy["y"]) >= PLANT_HIDDEN_Y for enemy in plants)


def aligned_with_pipe(player_x: int, pipe_x: int, *, slop: int = PIPE_SLOP) -> bool:
    return abs(int(player_x) - int(pipe_x)) <= slop


def is_pipe_transition(snap: Any) -> bool:
    """Vertical/side pipe load blanks pose to (0,0) / state 0, 2, 3, 7."""
    ps = int(snap.player_state)
    if ps in (0, 2, 3, 7):
        return True
    return int(snap.player_x) == 0 and int(snap.player_y) == 0


def is_ceiling(snap: Any) -> bool:
    """True on the y64 ceiling run (warp), not a jump apex or pipe transition."""
    if is_pipe_transition(snap):
        return False
    y = int(snap.player_y)
    if y < 64:
        return True
    return y < CEILING_Y_MAX and bool(snap.grounded)


def is_floor(snap: Any) -> bool:
    return int(snap.player_y) >= FLOOR_Y_MIN


def sky_is_overworld(rgb: np.ndarray | None) -> bool:
    """True on blue overworld (flag area or the pipe-wipe into it), not UG black."""
    if rgb is None or getattr(rgb, "ndim", 0) != 3 or rgb.shape[0] < 56:
        return False
    band = rgb[32:56, :, :]
    return float(band[:, :, 2].mean()) > 120.0


def classify_destination(
    *,
    world: int,
    player_state: int,
    outdoor_sky: bool,
    dying: bool,
    ceiling: bool = False,
) -> str:
    """Halt label for one pipe trial. Outdoor world-0 flag wins; warp rejects."""
    if dying:
        return DEATH
    if int(world) != 0:
        return WARP
    if int(player_state) == PLAYER_STATE_FLAGPOLE:
        return OUTDOOR_FLAG
    if int(player_state) == PLAYER_STATE_AUTO_WALK:
        return OUTDOOR_FLAG
    if outdoor_sky:
        return OUTDOOR_FLAG
    if ceiling:
        return CEILING
    return STILL_UG


def snap_row(snap: Any, **extra: Any) -> dict[str, Any]:
    row = {
        "world": int(snap.world),
        "level": int(snap.level),
        "dash_level": int(snap.dash_level),
        "area_pointer": int(snap.area_pointer),
        "x": int(snap.player_x),
        "y": int(snap.player_y),
        "xs": int(snap.x_speed),
        "ys": int(snap.y_speed),
        "ps": int(snap.player_state),
        "timer": int(snap.timer),
        "lives": int(snap.lives),
        "grounded": bool(snap.grounded),
        "in_air": bool(snap.in_air),
    }
    row.update(extra)
    return row


def log(*parts: object) -> None:
    print(*parts, flush=True)


def _rgb(result: Any, env: Any) -> np.ndarray | None:
    if isinstance(result, (tuple, list)) and result:
        frame = result[0]
        if isinstance(frame, np.ndarray) and frame.ndim == 3:
            return frame
    try:
        rendered = env.render()
    except Exception:
        return None
    if isinstance(rendered, np.ndarray):
        return rendered
    return None


def step(env: Any, buttons: list[int]) -> tuple[np.ndarray | None, Any]:
    rgb = _rgb(env.step(to_action9(buttons)), env)
    return rgb, read_snapshot(env.get_ram())


@dataclass
class TrialResult:
    candidate: str
    outcome: str
    frames: int
    entered_pipe: bool
    snap: dict[str, Any] = field(default_factory=dict)
    screenshot: str = ""
    notes: str = ""


class FlagProbe:
    """One emulator, one HL boot, then restore-from-corridor trials."""

    def __init__(self, *, evidence: Path = EVIDENCE) -> None:
        self.env: Any | None = None
        self.corridor: Any | None = None
        self.lives0 = 2
        self.ug_area = 0
        self.evidence = evidence
        self.boot_log: dict[str, Any] = {}
        self.trials: list[TrialResult] = []

    def close(self) -> None:
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
            self.env = None

    def _env(self) -> Any:
        assert self.env is not None
        return self.env

    def _shot(self, name: str, rgb: np.ndarray | None) -> str:
        if rgb is None:
            return ""
        path = save_rgb_png(rgb, self.evidence / name)
        return path.name

    def _abort_label(self, snap: Any, rgb: np.ndarray | None) -> str | None:
        if snap.dying or snap.lives < self.lives0:
            return DEATH
        if int(snap.world) != 0:
            return WARP
        if is_ceiling(snap):
            return CEILING
        label = classify_destination(
            world=int(snap.world),
            player_state=int(snap.player_state),
            outdoor_sky=sky_is_overworld(rgb),
            dying=bool(snap.dying),
            ceiling=is_ceiling(snap),
        )
        if label in {OUTDOOR_FLAG, WARP, DEATH}:
            return label
        return None

    def load_corridor_cache(self) -> dict[str, Any] | None:
        """Reuse a prior HL floor-corridor state (same process or --cache file)."""
        if not CACHE_STATE.is_file():
            return None
        env = make_level1_env()
        self.env = env
        data = read_state_bytes(CACHE_STATE)
        env.em.set_state(data)
        env.reset()
        env.em.set_state(data)
        snap = read_snapshot(env.get_ram())
        rgb = env.render()
        if not isinstance(rgb, np.ndarray):
            rgb = None
        if (
            int(snap.world) != 0
            or int(snap.player_x) < UG_MIN_X_FOR_CORRIDOR
            or snap.dying
            or not player_on_ground(env.get_ram())
        ):
            log(f"cache unusable {snap_row(snap)} phys={player_on_ground(env.get_ram())}")
            return None
        if is_ceiling(snap):
            log(f"cache is ceiling {snap_row(snap)}")
            return None
        self.corridor = get_state(env)
        self.lives0 = int(snap.lives)
        self.ug_area = int(snap.area_pointer)
        shot = self._shot("corridor_from_cache.png", rgb)
        log(f"corridor from cache {snap_row(snap)} shot={shot}")
        return {"corridor": snap_row(snap), "screenshot": shot, "from_cache": True}

    def boot_hl_to_corridor(self, *, fm2_path: Path = DEFAULT_FM2) -> dict[str, Any]:
        """HL 1-1 → surface → FM2 1-2; keep last floor pose at the plant pipes."""
        env = make_level1_env()
        self.env = env
        t0 = time.time()
        leave, wait, ctrl = reach_surface_after_hl_1_1(env)
        log(f"surface control leave_1_1={leave} wait={wait} {snap_row(ctrl)}")
        fm2 = parse_fm2(fm2_path).frames
        body = fm2[HL_1_2_FM2_START:]
        last_floor: Any | None = None
        last_floor_snap: dict[str, Any] | None = None
        ug_at: int | None = None
        rgb = None
        snap = read_snapshot(env.get_ram())
        self.lives0 = int(snap.lives)
        for i, frame in enumerate(body):
            rgb, snap = step(env, list(frame[:9]))
            if ug_at is None and int(snap.level) == 2 and int(snap.world) == 0:
                ug_at = i + 1
                self.ug_area = int(snap.area_pointer)
                log(f"UG enter f{ug_at} {snap_row(snap)}")
            abort = self._abort_label(snap, rgb)
            if abort in {DEATH, WARP}:
                log(f"boot abort {abort} f{i + 1} {snap_row(snap)}")
                break
            if (
                ug_at is not None
                and player_on_ground(env.get_ram())
                and is_floor(snap)
                and int(snap.player_x) >= UG_MIN_X_FOR_CORRIDOR
                and int(snap.world) == 0
                and not snap.dying
            ):
                last_floor = get_state(env)
                last_floor_snap = snap_row(snap, fm2_i=i + 1, motion=int(env.get_ram()[0x001D]))
            if ug_at is not None and is_ceiling(snap) and int(snap.player_x) >= 2200:
                log(f"HL left floor for ceiling f{i + 1} {snap_row(snap)}")
                break
            if ug_at is not None and i % 100 == 0:
                log(f"  hl f{i + 1} {snap_row(snap)}")
            if i + 1 >= 1800:
                break
        if last_floor is None:
            raise RuntimeError("HL 1-2 never stood on the UG floor past x=2000")
        set_state(env, last_floor)
        self.corridor = last_floor
        snap = read_snapshot(env.get_ram())
        rgb = env.render()
        if not isinstance(rgb, np.ndarray):
            rgb, snap = step(env, [0] * 9)
            set_state(env, last_floor)
            snap = read_snapshot(env.get_ram())
        self.lives0 = int(snap.lives)
        self.ug_area = int(snap.area_pointer)
        shot = self._shot("corridor_from_hl.png", rgb if isinstance(rgb, np.ndarray) else None)
        CACHE_STATE.parent.mkdir(parents=True, exist_ok=True)
        CACHE_STATE.write_bytes(self.corridor)
        self.boot_log = {
            "leave_1_1": leave,
            "ctrl_wait": wait,
            "ug_at": ug_at,
            "corridor": last_floor_snap,
            "elapsed_s": round(time.time() - t0, 1),
            "screenshot": shot,
        }
        log(f"corridor cached {last_floor_snap} shot={shot} {self.boot_log['elapsed_s']}s")
        return self.boot_log

    def restore_corridor(self) -> Any:
        assert self.corridor is not None
        set_state(self._env(), self.corridor)
        return read_snapshot(self._env().get_ram())

    def _capped(self, max_frames: int, buttons_fn: Any) -> tuple[str | None, Any, np.ndarray | None, int]:
        """Step up to *max_frames*; return (halt_label, snap, rgb, frames)."""
        rgb = None
        snap = read_snapshot(self._env().get_ram())
        stall_x = int(snap.player_x)
        stall_n = 0
        for i in range(max_frames):
            buttons = buttons_fn(snap, i, stall_n)
            rgb, snap = step(self._env(), buttons)
            halt = self._abort_label(snap, rgb)
            if halt is not None:
                return halt, snap, rgb, i + 1
            if int(snap.player_x) == stall_x:
                stall_n += 1
            else:
                stall_x = int(snap.player_x)
                stall_n = 0
        return None, snap, rgb, max_frames

    def jump_hold_sweep(self, *, holds: range = range(8, 21)) -> list[dict[str, Any]]:
        """A-only holds from the physics-grounded lift (HL uses 19, then idle)."""
        rows: list[dict[str, Any]] = []
        for hold in holds:
            self.restore_corridor()
            rgb = None
            snap = read_snapshot(self._env().get_ram())
            halt: str | None = None
            for _ in range(hold):
                rgb, snap = step(self._env(), A)
                if snap.dying or int(snap.world) != 0 or int(snap.player_y) >= 240:
                    halt = DEATH if snap.dying or int(snap.player_y) >= 240 else WARP
                    break
            if halt is None:
                for coast in range(50):
                    rgb, snap = step(self._env(), list(IDLE))
                    if snap.dying or int(snap.world) != 0 or int(snap.player_y) >= 240:
                        halt = DEATH if snap.dying or int(snap.player_y) >= 240 else WARP
                        break
                    if sky_is_overworld(rgb):
                        halt = OUTDOOR_FLAG
                        break
                    if player_on_ground(self._env().get_ram()) and coast >= 1:
                        break
            if halt is None:
                halt = classify_destination(
                    world=int(snap.world),
                    player_state=int(snap.player_state),
                    outdoor_sky=sky_is_overworld(rgb),
                    dying=bool(snap.dying),
                    ceiling=is_ceiling(snap),
                )
            row = snap_row(snap, hold=hold, halt=halt, phys=int(player_on_ground(self._env().get_ram())))
            rows.append(row)
            log(f"  A-hold={hold} halt={halt} {row}")
            if halt == OUTDOOR_FLAG or (player_on_ground(self._env().get_ram()) and 90 <= int(snap.player_y) <= 160):
                self._shot(f"jump_hold_{hold:02d}.png", rgb)
        return rows

    def lift_jump_then_down(self, *, hold_a: int = HL_LIFT_A_HOLD) -> tuple[str, Any, np.ndarray | None, bool]:
        """HL-style A-only jump onto the exit platform, then DOWN on the pipe."""
        entered = False
        rgb = None
        snap = read_snapshot(self._env().get_ram())
        for _ in range(hold_a):
            rgb, snap = step(self._env(), A)
            halt = self._abort_label(snap, rgb)
            if halt in {DEATH, WARP, OUTDOOR_FLAG}:
                return halt, snap, rgb, entered
        for _ in range(50):
            rgb, snap = step(self._env(), list(IDLE))
            halt = self._abort_label(snap, rgb)
            if halt in {DEATH, WARP, OUTDOOR_FLAG}:
                return halt, snap, rgb, entered
            if player_on_ground(self._env().get_ram()):
                break
        self._shot(f"land_hold_{hold_a:02d}.png", rgb)
        log(f"  landed {snap_row(snap)} phys={player_on_ground(self._env().get_ram())}")
        # Brake and settle on the short exit pipe (right lip against the wall).
        for _ in range(25):
            if int(snap.x_speed) > 8:
                rgb, snap = step(self._env(), L)
            else:
                rgb, snap = step(self._env(), RB if int(snap.player_x) < 2648 else list(IDLE))
            halt = self._abort_label(snap, rgb)
            if halt in {DEATH, WARP, OUTDOOR_FLAG}:
                return halt, snap, rgb, entered
        platform = get_state(self._env())
        self._shot("on_exit_pipe.png", rgb)
        log(f"  on_pipe {snap_row(snap)}")
        if is_pipe_transition(snap):
            halt, snap, rgb = self.watch_pipe_exit()
            return halt, snap, rgb, True
        for hop in (3, 4, 5, 6, 8, 10):
            set_state(self._env(), platform)
            entered = False
            for _ in range(hop):
                rgb, snap = step(self._env(), A)
            for _ in range(50):
                rgb, snap = step(self._env(), D)
                if int(snap.player_state) == PLAYER_STATE_PIPE_ENTER or is_pipe_transition(snap):
                    entered = True
                halt = self._abort_label(snap, rgb)
                if halt in {DEATH, WARP, OUTDOOR_FLAG}:
                    log(f"  hop={hop} → {halt} entered={entered} {snap_row(snap)}")
                    self._shot(f"hop_{hop:02d}_{halt}.png", rgb)
                    return halt, snap, rgb, entered
                if entered:
                    break
            if entered:
                halt, snap, rgb = self.watch_pipe_exit()
                log(f"  hop={hop} entered → {halt} {snap_row(snap)}")
                self._shot(f"hop_{hop:02d}_{halt}.png", rgb)
                return halt, snap, rgb, True
            log(f"  hop={hop} miss {snap_row(snap)}")
        halt = classify_destination(
            world=int(snap.world),
            player_state=int(snap.player_state),
            outdoor_sky=sky_is_overworld(rgb),
            dying=bool(snap.dying),
            ceiling=is_ceiling(snap),
        )
        return halt, snap, rgb, entered

    def advance_floor(self, *, max_frames: int = 700, allow_down: bool = True) -> tuple[str | None, Any, np.ndarray | None, int]:
        """From the end-of-UG lift: ride/hop onto the brick platform, DOWN the exit pipe.

        A long jump from this lift clears the wall into the warp ceiling — keep A
        to a tap, and abort at y < 80.
        """
        jump_hold = 0

        def buttons(snap: Any, i: int, stall_n: int) -> list[int]:
            nonlocal jump_hold
            if i in {10, 30, 60, 90, 140} or i % 80 == 0:
                rgb_now = self._env().render()
                if isinstance(rgb_now, np.ndarray):
                    self._shot(f"floor_run_f{i:03d}.png", rgb_now)
                log(f"    f{i} {snap_row(snap)} stall={stall_n}")
            if int(snap.player_state) == PLAYER_STATE_PIPE_ENTER:
                return D
            y = int(snap.player_y)
            x = int(snap.player_x)
            enemies = read_enemy_slots(self._env().get_ram())
            falling = (not snap.grounded) and int(snap.y_speed) > 0
            if allow_down and falling:
                mouths = [PIPE_A_X, PIPE_B_X, PIPE_C_X] + [
                    int(enemy["x"])
                    for enemy in enemies
                    if int(enemy.get("type", -1)) == ENEMY_TYPE_PIRANHA
                ]
                for mouth in mouths:
                    if aligned_with_pipe(x, mouth, slop=18) and plant_hidden(enemies, mouth):
                        return D
                # Exit pipe has no plant slot; DOWN while falling onto the platform lip.
                if x >= 2560 and y >= 100:
                    return D
            if y < 120:
                jump_hold = 0
                if stall_n >= 20 and snap.grounded:
                    return D  # seated on the exit pipe
                return RB
            if jump_hold > 0 and not snap.grounded:
                jump_hold -= 1
                return RBA
            if not snap.grounded:
                return RB
            # Elevated lift/platform: tap A only when stalled, never a full jump.
            if stall_n >= 14:
                jump_hold = 4
                return RBA
            return RB

        halt, snap, rgb, frames = self._capped(max_frames, buttons)
        if halt is None and int(snap.player_state) == PLAYER_STATE_PIPE_ENTER:
            halt, snap, rgb, extra = self._capped(220, lambda *_: D)
            frames += extra
        log(f"  advance_floor halt={halt} {snap_row(snap)} f{frames}")
        return halt, snap, rgb, frames

    def jump_over(self, target_x: int, *, hold_a: int = 16, max_frames: int = 180) -> str | None:
        """Run-jump past *target_x* without DOWN. None = still in UG on floor."""

        def buttons(snap: Any, i: int, stall_n: int) -> list[int]:
            if int(snap.player_x) >= target_x + PIPE_HALF and is_floor(snap):
                return list(IDLE)
            if not snap.grounded:
                return RB
            if stall_n >= 12 or int(snap.player_x) < target_x + PIPE_HALF:
                return RBA if i % (hold_a + 8) < hold_a else RB
            return RB

        halt, snap, rgb, frames = self._capped(max_frames, buttons)
        if halt is not None:
            return halt
        if int(snap.player_x) < target_x:
            log(f"jump_over short x={snap.player_x} want>={target_x} f{frames}")
        return None

    def wait_plant_hidden(self, pipe_x: int, *, max_wait: int = 90) -> str | None:
        """Idle a few tiles left of the pipe until the plant ducks. Halt or None."""

        def buttons(snap: Any, i: int, stall_n: int) -> list[int]:
            enemies = read_enemy_slots(self._env().get_ram())
            if plant_hidden(enemies, pipe_x) and int(snap.player_x) <= pipe_x - 36:
                return list(IDLE)
            if int(snap.player_x) > pipe_x - 36:
                return L
            if int(snap.player_x) < pipe_x - 64:
                return RB
            return list(IDLE)

        halt, _snap, _rgb, _frames = self._capped(max_wait, buttons)
        return halt

    def mid_fall_down(self, pipe_x: int, *, max_frames: int = 160) -> tuple[str, Any, np.ndarray | None, bool]:
        """Approach from the left, jump, DOWN while falling over the pipe mouth."""
        entered = False

        def buttons(snap: Any, i: int, stall_n: int) -> list[int]:
            nonlocal entered
            if int(snap.player_state) == PLAYER_STATE_PIPE_ENTER:
                entered = True
                return D
            enemies = read_enemy_slots(self._env().get_ram())
            hidden = plant_hidden(enemies, pipe_x)
            falling = (not snap.grounded) and int(snap.y_speed) > 0
            over = aligned_with_pipe(int(snap.player_x), pipe_x, slop=PIPE_SLOP + 4)
            if falling and over and hidden:
                return D
            if not snap.grounded:
                return RB
            if int(snap.player_x) < pipe_x - 20:
                return RBA if hidden else RB
            if int(snap.player_x) > pipe_x + 8:
                return L
            return RBA if hidden else RB

        halt, snap, rgb, frames = self._capped(max_frames, buttons)
        if int(snap.player_state) == PLAYER_STATE_PIPE_ENTER:
            entered = True
        if halt is None and entered:
            halt, snap, rgb, extra = self._capped(200, lambda *_: D)
            frames += extra
        if halt is None:
            halt = classify_destination(
                world=int(snap.world),
                player_state=int(snap.player_state),
                outdoor_sky=sky_is_overworld(rgb),
                dying=bool(snap.dying),
                ceiling=is_ceiling(snap),
            )
            if halt == STILL_UG and not entered:
                halt = NO_ENTER
        log(f"  mid_fall x={pipe_x} → {halt} entered={entered} {snap_row(snap)} f{frames}")
        return halt, snap, rgb, entered

    def watch_pipe_exit(self, *, max_frames: int = 360) -> tuple[str, Any, np.ndarray | None]:
        halt, snap, rgb, _frames = self._capped(max_frames, lambda *_: D)
        if halt is None:
            halt = classify_destination(
                world=int(snap.world),
                player_state=int(snap.player_state),
                outdoor_sky=sky_is_overworld(rgb),
                dying=bool(snap.dying),
                ceiling=is_ceiling(snap),
            )
        return halt, snap, rgb

    def run_trial(self, candidate: str) -> TrialResult:
        self.restore_corridor()
        log(f"trial {candidate}")
        halt: str | None = None
        rgb = None
        snap = read_snapshot(self._env().get_ram())
        entered = False
        notes = ""
        if candidate == "floor_run":
            halt, snap, rgb, entered = self.lift_jump_then_down()
            notes = f"hl_lift_A{HL_LIFT_A_HOLD}"
        elif candidate == "B":
            halt = self.wait_plant_hidden(PIPE_B_X)
            if halt is None:
                halt, snap, rgb, entered = self.mid_fall_down(PIPE_B_X)
        elif candidate == "C":
            halt = self.jump_over(PIPE_B_X)
            if halt is None:
                halt = self.wait_plant_hidden(PIPE_C_X)
            if halt is None:
                halt, snap, rgb, entered = self.mid_fall_down(PIPE_C_X)
            else:
                notes = notes or "halt before C mouth"
        elif candidate == "past_C":
            halt = self.jump_over(PIPE_B_X)
            if halt is None:
                halt = self.jump_over(PIPE_C_X)
            if halt is None:
                halt, snap, rgb, frames = self._scan_past_c()
                notes = f"scan_frames={frames}"
            else:
                notes = "halt jumping B/C"
        else:
            raise ValueError(candidate)

        if halt in {None, STILL_UG} and entered:
            halt, snap, rgb = self.watch_pipe_exit()
        if halt is None:
            halt = STILL_UG
        shot = self._shot(f"trial_{candidate}_{halt}.png", rgb)
        result = TrialResult(
            candidate=candidate,
            outcome=halt,
            frames=0,
            entered_pipe=entered,
            snap=snap_row(snap),
            screenshot=shot,
            notes=notes,
        )
        self.trials.append(result)
        log(f"  result {candidate} {halt} shot={shot} {notes}")
        return result

    def _scan_past_c(self) -> tuple[str | None, Any, np.ndarray | None, int]:
        """Walk right of C; DOWN any further plant pipe; cap so walls cannot spin."""
        found_pipe: int | None = None

        def buttons(snap: Any, i: int, stall_n: int) -> list[int]:
            nonlocal found_pipe
            enemies = read_enemy_slots(self._env().get_ram())
            plants = [
                enemy
                for enemy in enemies
                if int(enemy.get("type", -1)) == ENEMY_TYPE_PIRANHA
                and int(enemy["x"]) > PIPE_C_X + 24
            ]
            if plants and found_pipe is None:
                found_pipe = int(plants[0]["x"])
                log(f"  past-C plant pipe at x={found_pipe}")
            if found_pipe is not None:
                falling = (not snap.grounded) and int(snap.y_speed) > 0
                hidden = plant_hidden(enemies, found_pipe)
                if falling and aligned_with_pipe(int(snap.player_x), found_pipe) and hidden:
                    return D
                if int(snap.player_x) < found_pipe - 16:
                    return RBA if hidden or stall_n >= 8 else RB
                if not snap.grounded:
                    return RB
                return RBA
            if stall_n >= 18 and snap.grounded:
                return RBA
            if not snap.grounded:
                return RB
            return RB

        halt, snap, rgb, frames = self._capped(220, buttons)
        if halt is None and found_pipe is not None:
            halt, snap, rgb, entered = self.mid_fall_down(found_pipe)
            if entered and halt in {None, STILL_UG, NO_ENTER}:
                halt, snap, rgb = self.watch_pipe_exit()
        return halt, snap, rgb, frames

    def finish_to_1_3(self) -> dict[str, Any]:
        """From outdoor 1-2, hop off the exit pipe, flag, wait 1-3 control."""
        env = self._env()
        rgb = None
        snap = read_snapshot(env.get_ram())
        t0 = time.time()
        stall_x = int(snap.player_x)
        stall_n = 0
        frames = 0
        halt: str | None = None
        for _ in range(1600):
            frames += 1
            if is_1_3_control(snap):
                halt = "1_3_control"
                break
            if snap.dying or int(snap.world) != 0:
                halt = DEATH if snap.dying else WARP
                break
            if int(snap.player_state) in (PLAYER_STATE_FLAGPOLE, PLAYER_STATE_AUTO_WALK):
                buttons = list(IDLE)
            elif stall_n >= 8 and (snap.grounded or player_on_ground(env.get_ram())):
                buttons = RBA
            else:
                buttons = RB
            rgb, snap = step(env, buttons)
            if int(snap.player_x) == stall_x:
                stall_n += 1
            else:
                stall_x = int(snap.player_x)
                stall_n = 0
            if frames in {30, 90, 180, 360}:
                self._shot(f"flag_run_f{frames:04d}.png", rgb)
                log(f"  flag_run f{frames} {snap_row(snap)}")
        shot = self._shot("1_3_control.png" if is_1_3_control(snap) else "1_3_wait_fail.png", rgb)
        report = {
            "ok": is_1_3_control(snap),
            "frames": frames,
            "elapsed_s": round(time.time() - t0, 1),
            "snap": snap_row(snap),
            "screenshot": shot,
            "halt": halt,
        }
        if not report["ok"]:
            log(f"1-3 control missed {report}")
            return report
        pin = write_stage_pin(
            task_name="all_exits_v1",
            stage_id="1-3",
            state_bytes=get_state(env),
            snap=snap,
            frame=frames,
            rta_frames=frames,
            kind="control",
        )
        report["pin"] = str(pin)
        log(f"wrote 1-3 pin {pin} {snap_row(snap)}")
        return report


def run_probe(*, fresh: bool = False, sweep: bool = False) -> dict[str, Any]:
    probe = FlagProbe()
    report: dict[str, Any] = {"ok": False, "trials": []}
    try:
        boot = None if fresh else probe.load_corridor_cache()
        if boot is None:
            if probe.env is not None:
                probe.close()
            boot = probe.boot_hl_to_corridor()
        report["boot"] = boot
        if sweep:
            report["jump_sweep"] = probe.jump_hold_sweep()
        for candidate in ("floor_run", "B", "C", "past_C"):
            trial = probe.run_trial(candidate)
            report["trials"].append(asdict(trial))
            if trial.outcome == OUTDOOR_FLAG:
                report["winner"] = candidate
                report["finish"] = probe.finish_to_1_3()
                report["ok"] = bool(report["finish"].get("ok"))
                break
            if trial.outcome == WARP:
                log(f"{candidate} went to warp — skip remaining in that room")
        return report
    finally:
        probe.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="ignore cached corridor state and boot HL 1-2 again",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="A-hold sweep from the lift before the flag-pipe trial",
    )
    args = parser.parse_args(argv)
    t0 = time.time()
    report = run_probe(fresh=args.fresh, sweep=args.sweep)
    report["elapsed_s"] = round(time.time() - t0, 1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "probe_1_2_flag_report.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    log(f"report {path} ok={report.get('ok')} winner={report.get('winner')}")
    print(json.dumps({k: report[k] for k in ("ok", "winner", "elapsed_s") if k in report}, indent=2))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
