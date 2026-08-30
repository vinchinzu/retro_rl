"""Flight recorder and freeze/stall abort for TMNT IV trials.

Enemyless frozen X for ``FREEZE_ABORT_FRAMES`` is still an infinite
dumpster/rail loop. Living-enemy progress stalls (Alleycat bridge
``combat_stall_escape`` / ``edge_press``) fire on a shorter watchdog
instead of waiting for an empty screen.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from retro_harness.env import save_state
from retro_harness.ram_state import GameState
from tmnt_iv.paths import GAME, GAME_DIR, RECORDINGS_DIR

FREEZE_ABORT_FRAMES = 12_000
_RING = 180
_CYCLE_SPAN = 24


@dataclass(frozen=True)
class RecorderAbort:
    """Stall or enemyless-X freeze with a recoverable window."""

    kind: str
    reason: str
    last_progress: Any
    first_cycle: Any
    window_summary: list[dict[str, Any]]
    frames: int


@dataclass
class FrameSample:
    """One bounded ring entry."""

    frame: int
    reason: str
    player_x: int
    player_y: int
    anim: int
    iframes: int
    camera: int
    progress: int
    event: int
    stage: int
    hazards: tuple[Any, ...]
    pickups: tuple[Any, ...]
    enemies: tuple[tuple[int, int, int, int], ...]  # kind, pose, anim, hp


def progress_key(state: GameState) -> tuple[Any, ...]:
    """Stage-aware progress: streets/camera, locked HP, boss event, Neon camera."""
    event = int(state.extras.get("event", -1))
    progress = int(state.extras.get("progress_x", state.camera_x))
    if state.boss_active:
        return ("boss", state.stage, event, int(state.extras.get("boss_hp", 0)))
    living = state.living_enemies
    if living:
        hp = sum(int(e.health) for e in living)
        return ("fight", state.stage, progress, hp)
    return ("scroll", state.stage, progress, state.player_x)


def _enemy_tuple(state: GameState) -> tuple[tuple[int, int, int, int], ...]:
    rows: list[tuple[int, int, int, int]] = []
    for enemy in state.enemies:
        rows.append(
            (int(enemy.kind), int(enemy.x), int(enemy.animation), int(enemy.health))
        )
    return tuple(rows)


def _can_dump(env: Any, obs: Any) -> bool:
    if obs is None or env is None:
        return False
    em = getattr(env, "em", None)
    return em is not None and hasattr(em, "get_state")


def dump_abort_snapshot(env: Any, obs: Any, state: GameState, frame: int) -> str | None:
    """Write PNG+state when the emulator can save; skip in ROM-free tests."""
    if not _can_dump(env, obs):
        return None
    from PIL import Image

    shot = RECORDINGS_DIR / (
        f"scratch_freeze_s{state.stage}_x{state.player_x}_f{frame}.png"
    )
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(obs, dtype=np.uint8)).save(shot)
    save_state(
        env,
        GAME_DIR,
        GAME,
        f"ScratchFreeze_s{state.stage}_x{state.player_x}",
    )
    return str(shot)


@dataclass
class FlightRecorder:
    """Bounded frame ring plus freeze/stall watchdogs."""

    freeze_abort_frames: int = FREEZE_ABORT_FRAMES
    stall_frames: int = 2400
    samples: deque[FrameSample] = field(default_factory=lambda: deque(maxlen=_RING))
    freeze_frames: int = 0
    last_progress: Any = None
    first_cycle: Any = None
    _freeze_x: int = -1
    _freeze_stage: int = -1
    _stall_key: Any = None
    _stall_count: int = 0
    _cycle_hits: dict[tuple[Any, ...], int] = field(default_factory=dict)

    def observe(
        self,
        *,
        freeze_armed: bool,
        live: bool,
        state: GameState,
        frame: int,
        reason: str,
        damage: int,
    ) -> RecorderAbort | None:
        """Record one frame; return an abort when freeze or combat stall fires."""
        key = progress_key(state)
        self.last_progress = key
        sample = FrameSample(
            frame=frame,
            reason=reason,
            player_x=state.player_x,
            player_y=state.player_y,
            anim=int(state.extras.get("anim", -1)),
            iframes=int(state.extras.get("iframes", -1)),
            camera=state.camera_x,
            progress=int(state.extras.get("progress_x", state.camera_x)),
            event=int(state.extras.get("event", -1)),
            stage=state.stage,
            hazards=tuple(state.extras.get("hazards") or ()),
            pickups=tuple(state.extras.get("pickups") or ()),
            enemies=_enemy_tuple(state),
        )
        self.samples.append(sample)
        cycle_id = (reason, key)
        seen = self._cycle_hits.get(cycle_id, 0) + 1
        self._cycle_hits[cycle_id] = seen
        if self.first_cycle is None and seen >= _CYCLE_SPAN:
            self.first_cycle = {"reason": reason, "progress": key, "count": seen}

        abort: RecorderAbort | None = None
        if live and state.living_enemies and self.stall_frames > 0:
            if key == self._stall_key:
                self._stall_count += 1
            else:
                self._stall_key = key
                self._stall_count = 0
            if self._stall_count >= self.stall_frames:
                abort = self._abort(
                    "stall",
                    (
                        f"progress stall {self._stall_count}f at frame {frame}: "
                        f"stage={state.stage} key={key} reason={reason} dmg={damage}"
                    ),
                )
        else:
            self._stall_key = None
            self._stall_count = 0

        if freeze_armed:
            if state.player_x == self._freeze_x and state.stage == self._freeze_stage:
                self.freeze_frames += 1
            else:
                self._freeze_x = state.player_x
                self._freeze_stage = state.stage
                self.freeze_frames = 0
            if self.freeze_frames in {2_000, 5_000} or (
                self.freeze_frames >= 2_000 and self.freeze_frames % 5_000 == 0
            ):
                print(
                    f"FREEZE {self.freeze_frames}f  frame={frame} "
                    f"stage={state.stage} "
                    f"p=({state.player_x},{state.player_y}) "
                    f"cam={state.camera_x} reason={reason} "
                    f"dmg={damage}",
                    flush=True,
                )
            if abort is None and self.freeze_frames >= self.freeze_abort_frames:
                abort = self._abort(
                    "freeze",
                    (
                        f"frozen X for {self.freeze_frames}f at frame {frame}: "
                        f"stage={state.stage} "
                        f"p=({state.player_x},{state.player_y}) "
                        f"cam={state.camera_x} reason={reason} "
                        f"dmg={damage}"
                    ),
                )
        else:
            self._freeze_x = -1
            self._freeze_stage = -1
            self.freeze_frames = 0
        return abort

    def window(self, n: int = 12) -> list[dict[str, Any]]:
        """Latest ring samples as JSON-friendly dicts."""
        out: list[dict[str, Any]] = []
        for sample in list(self.samples)[-n:]:
            out.append(
                {
                    "frame": sample.frame,
                    "reason": sample.reason,
                    "player": (sample.player_x, sample.player_y),
                    "anim": sample.anim,
                    "iframes": sample.iframes,
                    "camera": sample.camera,
                    "progress": sample.progress,
                    "event": sample.event,
                    "enemies": [list(row) for row in sample.enemies],
                }
            )
        return out

    def _abort(self, kind: str, reason: str) -> RecorderAbort:
        return RecorderAbort(
            kind=kind,
            reason=reason,
            last_progress=self.last_progress,
            first_cycle=self.first_cycle,
            window_summary=self.window(),
            frames=self.stall_frames if kind == "stall" else self.freeze_frames,
        )


class FreezeWatch:
    """Facade over ``FlightRecorder`` for the enemyless frozen-X abort."""

    def __init__(self, abort_frames: int = FREEZE_ABORT_FRAMES) -> None:
        self.abort_frames = abort_frames
        self.recorder = FlightRecorder(freeze_abort_frames=abort_frames, stall_frames=0)
        self.frames = 0

    def tick(
        self,
        *,
        armed: bool,
        state: GameState,
        frame: int,
        reason: str,
        damage: int,
        obs: np.ndarray,
        env: Any,
    ) -> None:
        """Advance the freeze counter; dump PNG+state and raise on abort."""
        abort = self.recorder.observe(
            freeze_armed=armed,
            live=armed,
            state=state,
            frame=frame,
            reason=reason,
            damage=damage,
        )
        self.frames = self.recorder.freeze_frames
        if abort is None:
            return
        shot = dump_abort_snapshot(env, obs, state, frame)
        detail = abort.reason if shot is None else f"{abort.reason} shot={shot}"
        raise RuntimeError(detail)
