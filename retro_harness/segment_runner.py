"""Reusable headless segment-run helpers for oneshot SNES agents.

Pure stop/success heuristics and report I/O live here. Emulator loops and
game-specific RAM adapters stay in each game's scripts.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from retro_harness.ram_state import GameMode, GameState


class SegmentOutcome(Enum):
    """Why a segment run stopped."""

    SUCCESS = auto()
    TIMEOUT = auto()
    DEATH = auto()
    ERROR = auto()


def configure_headless() -> None:
    """Force SDL dummy drivers for headless emulator runs."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("SDL_SOFTWARE_RENDERER", "1")


def player_is_down(
    state: GameState,
    *,
    max_live_health: int | None = 128,
) -> bool:
    """True when the player is dead or outside the live-health band.

    Args:
        state: Normalized game state.
        max_live_health: When an int (default 128), PLAYING health outside
            ``1..max_live_health`` counts as down — beat-em-ups with corpse
            HP wrap should keep this default. When ``None``, only
            ``player_dead`` or ``health <= 0`` counts (platformers).
    """
    if state.player_dead:
        return True
    if max_live_health is None:
        return state.health <= 0
    if state.mode is not GameMode.PLAYING:
        return False
    return not (0 < state.health <= max_live_health)


def save_rgb_png(obs: np.ndarray, path: Path) -> Path:
    """Write an rgb_array observation to a PNG; create parents as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(obs).save(path)
    return path


def write_json_report(path: Path, report: dict[str, Any]) -> Path:
    """Write a JSON segment report with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def enemy_health_sum(state: GameState) -> int:
    """Sum HP across living enemies."""
    return sum(e.health for e in state.living_enemies)


def is_screen_clear(
    state: GameState,
    *,
    start_camera_x: int,
    had_enemies: bool,
    clear_hold_frames: int,
    clear_frames_seen: int,
    camera_unlock_delta: int = 8,
) -> bool:
    """Heuristic: first wave cleared / camera unlocked.

    Success when the segment started with enemies and now has none, plus one
    of: level complete, GO! flashing, camera scroll, or sustained clear hold.

    Prefers typed ``GameState.go_flashing`` / ``area_clear`` fields; also
    reads legacy ``extras`` keys for one-release back-compat.
    """
    if not had_enemies:
        return False
    area_clear = state.area_clear or bool(state.extras.get("area_clear"))
    if state.level_complete or area_clear:
        return True
    if state.living_enemies:
        return False
    go_flashing = state.go_flashing or bool(state.extras.get("go_flashing"))
    cam_delta = state.camera_x - start_camera_x
    if go_flashing or cam_delta >= camera_unlock_delta:
        return True
    if not state.screen_locked and clear_frames_seen >= clear_hold_frames:
        return True
    return False


@dataclass
class SegmentTracker:
    """Accumulate per-run metrics and evaluate stop conditions."""

    max_frames: int = 3600
    clear_hold_frames: int = 30
    camera_unlock_delta: int = 8
    start_camera_x: int = 0
    start_health: int = 0
    start_enemy_count: int = 0
    start_enemy_hp: int = 0
    had_enemies: bool = False
    frames: int = 0
    kills: int = 0
    clear_frames_seen: int = 0
    peak_enemy_count: int = 0
    min_health: int = 0
    reason_counts: dict[str, int] = field(default_factory=dict)
    _prev_living: int = 0
    _prev_enemy_hp: int = 0

    def begin(self, state: GameState) -> None:
        """Seed tracker from the first observed state."""
        living = len(state.living_enemies)
        hp_sum = enemy_health_sum(state)
        self.start_camera_x = state.camera_x
        self.start_health = state.health
        self.start_enemy_count = living
        self.start_enemy_hp = hp_sum
        self.had_enemies = living > 0
        self.peak_enemy_count = living
        self.min_health = state.health
        self._prev_living = living
        self._prev_enemy_hp = hp_sum
        self.frames = 0
        self.kills = 0
        self.clear_frames_seen = 0
        self.reason_counts = {}

    def note_reason(self, reason: str) -> None:
        """Count a policy reason string for the report."""
        key = reason or "unknown"
        self.reason_counts[key] = self.reason_counts.get(key, 0) + 1

    def update(self, state: GameState) -> SegmentOutcome | None:
        """Feed one frame; return an outcome when the run should stop."""
        self.frames += 1
        living = len(state.living_enemies)
        hp_sum = enemy_health_sum(state)
        self.peak_enemy_count = max(self.peak_enemy_count, living)
        self.min_health = min(self.min_health, state.health)

        if living < self._prev_living:
            self.kills += self._prev_living - living
        self._prev_living = living
        self._prev_enemy_hp = hp_sum

        if player_is_down(state):
            return SegmentOutcome.DEATH

        if living == 0 and self.had_enemies:
            self.clear_frames_seen += 1
        else:
            self.clear_frames_seen = 0

        if is_screen_clear(
            state,
            start_camera_x=self.start_camera_x,
            had_enemies=self.had_enemies,
            clear_hold_frames=self.clear_hold_frames,
            clear_frames_seen=self.clear_frames_seen,
            camera_unlock_delta=self.camera_unlock_delta,
        ):
            return SegmentOutcome.SUCCESS

        if self.frames >= self.max_frames:
            return SegmentOutcome.TIMEOUT
        return None

    def damage_dealt(self, state: GameState) -> int:
        """Enemy HP removed since begin (ignores despawn edge cases)."""
        remaining = enemy_health_sum(state)
        return max(0, self.start_enemy_hp - remaining)

    def to_report(
        self,
        *,
        outcome: SegmentOutcome,
        final: GameState,
        screenshots: list[str],
        start_state: str,
        extras: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build a JSON-serializable segment report."""
        report: dict[str, Any] = {
            "outcome": outcome.name.lower(),
            "success": outcome is SegmentOutcome.SUCCESS,
            "frames": self.frames,
            "start_state": start_state,
            "kills": self.kills,
            "damage_dealt": self.damage_dealt(final),
            "start_health": self.start_health,
            "end_health": final.health,
            "min_health": self.min_health,
            "start_enemy_count": self.start_enemy_count,
            "end_enemy_count": len(final.living_enemies),
            "peak_enemy_count": self.peak_enemy_count,
            "start_camera_x": self.start_camera_x,
            "end_camera_x": final.camera_x,
            "camera_delta": final.camera_x - self.start_camera_x,
            "player_x": final.player_x,
            "player_y": final.player_y,
            "go_flashing": final.go_flashing
            or bool(final.extras.get("go_flashing")),
            "screen_locked": final.screen_locked,
            "level_complete": final.level_complete,
            "reason_counts": dict(sorted(self.reason_counts.items())),
            "screenshots": screenshots,
        }
        if extras:
            report["extras"] = extras
        return report


def snapshot_state(state: GameState) -> dict[str, Any]:
    """Compact GameState dict for debugging reports."""
    return {
        "frame": state.frame,
        "mode": state.mode.name,
        "player_x": state.player_x,
        "player_y": state.player_y,
        "health": state.health,
        "lives": state.lives,
        "camera_x": state.camera_x,
        "enemies": len(state.living_enemies),
        "enemy_hp": enemy_health_sum(state),
        "screen_locked": state.screen_locked,
        "go_flashing": state.go_flashing
        or bool(state.extras.get("go_flashing")),
        "level_complete": state.level_complete,
        "player_dead": state.player_dead,
        "boss_active": state.boss_active,
        "stage": state.stage,
        "room": state.room,
    }


@dataclass
class WaveRecord:
    """One fight → clear cycle inside a multi-wave chain."""

    index: int
    start_frame: int
    start_camera_x: int
    start_enemy_count: int
    start_health: int
    start_enemy_hp: int = 0
    start_lives: int = 0
    end_frame: int | None = None
    end_camera_x: int | None = None
    end_health: int | None = None
    end_lives: int | None = None
    damage_dealt: int = 0
    player_damage: int = 0
    peak_enemy_count: int = 0
    kills: int = 0

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable wave summary."""
        return {
            "index": self.index,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "frames": (
                None
                if self.end_frame is None
                else self.end_frame - self.start_frame
            ),
            "start_camera_x": self.start_camera_x,
            "end_camera_x": self.end_camera_x,
            "start_enemy_count": self.start_enemy_count,
            "peak_enemy_count": self.peak_enemy_count,
            "kills": self.kills,
            "start_health": self.start_health,
            "end_health": self.end_health,
            "start_lives": self.start_lives,
            "end_lives": self.end_lives,
            "start_enemy_hp": self.start_enemy_hp,
            "damage_dealt": self.damage_dealt,
            "player_damage": self.player_damage,
        }


@dataclass
class WaveChainTracker:
    """Track multiple fight→clear→walk cycles until a terminal stop.

    Unlike ``SegmentTracker``, a single wave clear does not end the run.
    Success when ``target_waves`` clears are reached, the level completes,
    or (optionally) a boss becomes combat-active.
    """

    max_frames: int = 12000
    clear_hold_frames: int = 30
    camera_unlock_delta: int = 8
    target_waves: int | None = None
    stop_on_boss: bool = True
    start_camera_x: int = 0
    start_health: int = 0
    start_enemy_count: int = 0
    start_enemy_hp: int = 0
    frames: int = 0
    kills: int = 0
    damage_dealt_total: int = 0
    peak_enemy_count: int = 0
    min_health: int = 0
    waves_cleared: int = 0
    waves: list[WaveRecord] = field(default_factory=list)
    reason_counts: dict[str, int] = field(default_factory=dict)
    boss_reached: bool = False
    _in_wave: bool = False
    _wave: WaveRecord | None = None
    _wave_start_camera: int = 0
    _wave_had_enemies: bool = False
    _clear_frames_seen: int = 0
    _prev_living: int = 0
    _prev_enemy_hp: int = 0

    def begin(self, state: GameState) -> None:
        """Seed tracker from the first observed state."""
        living = len(state.living_enemies)
        hp_sum = enemy_health_sum(state)
        self.start_camera_x = state.camera_x
        self.start_health = state.health
        self.start_enemy_count = living
        self.start_enemy_hp = hp_sum
        self.peak_enemy_count = living
        self.min_health = state.health
        self.frames = 0
        self.kills = 0
        self.damage_dealt_total = 0
        self.waves_cleared = 0
        self.waves = []
        self.reason_counts = {}
        self.boss_reached = False
        self._in_wave = False
        self._wave = None
        self._clear_frames_seen = 0
        self._prev_living = living
        self._prev_enemy_hp = hp_sum
        if living > 0:
            self._start_wave(state)

    def note_reason(self, reason: str) -> None:
        """Count a policy reason string for the report."""
        key = reason or "unknown"
        self.reason_counts[key] = self.reason_counts.get(key, 0) + 1

    def _start_wave(self, state: GameState) -> None:
        living = len(state.living_enemies)
        self._in_wave = True
        self._wave_had_enemies = True
        self._wave_start_camera = state.camera_x
        self._clear_frames_seen = 0
        self._wave = WaveRecord(
            index=len(self.waves) + 1,
            start_frame=self.frames,
            start_camera_x=state.camera_x,
            start_enemy_count=living,
            start_health=state.health,
            start_enemy_hp=enemy_health_sum(state),
            start_lives=state.lives,
            peak_enemy_count=living,
        )

    def _finish_wave(self, state: GameState) -> None:
        assert self._wave is not None
        self._wave.end_frame = self.frames
        self._wave.end_camera_x = state.camera_x
        self._wave.end_health = state.health
        self._wave.end_lives = state.lives
        # Player HP can wrap (corpse bytes > 128); only count real chip.
        if (
            0 < self._wave.start_health <= 128
            and 0 < state.health <= 128
            and state.health < self._wave.start_health
        ):
            self._wave.player_damage = (
                self._wave.start_health - state.health
            )
        self.waves.append(self._wave)
        self.waves_cleared += 1
        self._in_wave = False
        self._wave = None
        self._wave_had_enemies = False
        self._clear_frames_seen = 0

    def update(self, state: GameState) -> SegmentOutcome | None:
        """Feed one frame; return an outcome when the chain should stop."""
        self.frames += 1
        living = len(state.living_enemies)
        hp_sum = enemy_health_sum(state)
        self.peak_enemy_count = max(self.peak_enemy_count, living)
        if 0 < state.health <= 128:
            self.min_health = min(self.min_health, state.health)

        if hp_sum < self._prev_enemy_hp:
            dealt = self._prev_enemy_hp - hp_sum
            self.damage_dealt_total += dealt
            if self._wave is not None:
                self._wave.damage_dealt += dealt
        if living < self._prev_living:
            gained = self._prev_living - living
            self.kills += gained
            if self._wave is not None:
                self._wave.kills += gained
        self._prev_living = living
        self._prev_enemy_hp = hp_sum

        if self._wave is not None:
            self._wave.peak_enemy_count = max(
                self._wave.peak_enemy_count, living
            )

        if player_is_down(state):
            return SegmentOutcome.DEATH

        # TCRF: boss status 01 = present undrawn, 03 = drawn. Either means
        # Damnd/Thrasher (or later bosses) has spawned into the boss slot.
        boss_status = int(state.extras.get("boss_status", 0))
        if state.boss_active and boss_status >= 1:
            self.boss_reached = True
            if self.stop_on_boss:
                if self._in_wave and living == 0:
                    self._finish_wave(state)
                return SegmentOutcome.SUCCESS

        if state.level_complete:
            if self._in_wave:
                self._finish_wave(state)
            return SegmentOutcome.SUCCESS

        if living > 0 and not self._in_wave:
            self._start_wave(state)

        if self._in_wave:
            if living == 0 and self._wave_had_enemies:
                self._clear_frames_seen += 1
            else:
                self._clear_frames_seen = 0
            cam_delta = state.camera_x - self._wave_start_camera
            # Without camera unlock / GO!, require a longer empty hold so
            # off-screen filter flicker does not false-clear a lock.
            hold = self.clear_hold_frames
            go_flashing = state.go_flashing or bool(
                state.extras.get("go_flashing")
            )
            if cam_delta < self.camera_unlock_delta and not go_flashing:
                hold = max(hold, 90)
            killed = self._wave is not None and self._wave.kills > 0
            if killed and is_screen_clear(
                state,
                start_camera_x=self._wave_start_camera,
                had_enemies=self._wave_had_enemies,
                clear_hold_frames=hold,
                clear_frames_seen=self._clear_frames_seen,
                camera_unlock_delta=self.camera_unlock_delta,
            ):
                self._finish_wave(state)
                if (
                    self.target_waves is not None
                    and self.waves_cleared >= self.target_waves
                ):
                    return SegmentOutcome.SUCCESS

        if self.frames >= self.max_frames:
            return SegmentOutcome.TIMEOUT
        return None

    def to_report(
        self,
        *,
        outcome: SegmentOutcome,
        final: GameState,
        screenshots: list[str],
        start_state: str,
        extras: dict[str, Any] | None = None,
        saved_states: list[str] | None = None,
    ) -> dict[str, Any]:
        """Build a JSON-serializable multi-wave chain report."""
        report: dict[str, Any] = {
            "outcome": outcome.name.lower(),
            "success": outcome is SegmentOutcome.SUCCESS,
            "frames": self.frames,
            "start_state": start_state,
            "kills": self.kills,
            "damage_dealt": self.damage_dealt_total,
            "waves_cleared": self.waves_cleared,
            "screens_cleared": self.waves_cleared,
            "waves": [w.to_dict() for w in self.waves],
            "boss_reached": self.boss_reached,
            "start_health": self.start_health,
            "end_health": final.health,
            "min_health": self.min_health,
            "start_enemy_count": self.start_enemy_count,
            "end_enemy_count": len(final.living_enemies),
            "peak_enemy_count": self.peak_enemy_count,
            "start_camera_x": self.start_camera_x,
            "end_camera_x": final.camera_x,
            "camera_delta": final.camera_x - self.start_camera_x,
            "player_x": final.player_x,
            "player_y": final.player_y,
            "go_flashing": final.go_flashing
            or bool(final.extras.get("go_flashing")),
            "screen_locked": final.screen_locked,
            "level_complete": final.level_complete,
            "boss_active": final.boss_active,
            "boss_hp": final.extras.get("boss_hp"),
            "boss_status": final.extras.get("boss_status"),
            "stage": final.stage,
            "room": final.room,
            "reason_counts": dict(sorted(self.reason_counts.items())),
            "screenshots": screenshots,
            "saved_states": saved_states or [],
        }
        if extras:
            report["extras"] = extras
        return report


__all__ = [
    "SegmentOutcome",
    "SegmentTracker",
    "WaveChainTracker",
    "WaveRecord",
    "configure_headless",
    "enemy_health_sum",
    "is_screen_clear",
    "player_is_down",
    "save_rgb_png",
    "snapshot_state",
    "write_json_report",
]
