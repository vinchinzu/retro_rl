"""Headless sequence evaluator (fitness function) for platformer optimization.

Generalized from the DKC optimizer evaluator. Takes a LevelConfig instead
of hardcoded constants, uses RAMSchema for reads, and delegates progress
tracking to a ProgressTracker.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from platformer_common.actions import (
    DEFAULT_PLATFORMER_ACTIONS,
    action_index_to_buttons,
)
from platformer_common.level_config import LevelConfig
from platformer_common.progress import ProgressTracker, make_progress_tracker
from retro_harness.env import make_env
from retro_harness.ram_state import RAMSchema


@dataclass
class EvalResult:
    """Result of evaluating an action sequence."""

    fitness: float = 0.0
    completed: bool = False
    died: bool = False
    total_frames: int = 0
    timer_frames: int = 0
    max_x: float = 0.0
    max_progress: float = 0.0
    final_x: float = 0.0
    final_y: float = 0.0
    level_id_at_end: int = 0
    early_terminated: bool = False
    gameplay_start_frame: int = 0
    bonus_frames: int = 0


class Evaluator:
    """Fast headless evaluator using emulator state caching.

    Death detection is aggressive: terminates immediately when lives drop
    OR camera resets (death animation), whichever comes first.
    """

    def __init__(self, config: LevelConfig, *, start_state: str | None = None) -> None:
        self.config = config
        self._start_state_override = start_state
        self._action_table = config.action_table or DEFAULT_PLATFORMER_ACTIONS
        self._schema = config.ram_schema
        self._tracker: ProgressTracker = make_progress_tracker(config)
        self._env = None
        self._cached_state = None
        self._initial_values: dict[str, int] | None = None
        self._initial_camera_x: float = 0.0
        # Pre-compute the set of level_ids that are "this level"
        self._main_level_ids: set[int] = {config.target_level_id} | set(config.level_id_aliases)

    @classmethod
    def from_level_id(cls, level_id: str, **kwargs) -> Evaluator:
        """Create an evaluator by looking up a registered level."""
        from platformer_common.level_config import get_level_config

        return cls(get_level_config(level_id), **kwargs)

    def _ensure_env(self):
        if self._env is None:
            state = self._start_state_override or self.config.start_state
            self._env = make_env(
                game=self.config.game_name,
                state=state,
                game_dir=self.config.game_dir,
                render_mode="rgb_array",
            )
            self._env.reset()
            self._cached_state = self._env.em.get_state()
            ram = self._env.get_ram()
            self._initial_values = self._read_ram(ram)
            self._initial_camera_x = float(self._initial_values.get("camera_x", 0))

    def _read_ram(self, ram: np.ndarray) -> dict[str, int]:
        """Read all configured RAM fields, then apply computed values."""
        values = self._schema.read(ram)
        return self.config.apply_computed(values)

    def _read_timer_total(self, values: dict[str, int]) -> int:
        """Compute total timer from minutes + frames fields."""
        frames = values.get("timer_frames", 0)
        minutes = values.get("timer_minutes", 0)
        return minutes * 60 * 60 + frames

    def is_dead(self, values: dict[str, int], in_sub_level: bool = False) -> bool:
        """Check death via configured signals.

        Args:
            in_sub_level: If True, skip camera_reset signal (bonus rooms
                          cause camera jumps that look like death resets).
        """
        for signal in self.config.death_signals:
            if signal == "lives_drop":
                if self._initial_values and "lives" in values and "lives" in self._initial_values:
                    if values["lives"] < self._initial_values["lives"]:
                        return True

            elif signal == "camera_reset":
                if in_sub_level:
                    continue  # camera jumps to bonus room map position
                camera_x = float(values.get("camera_x", 0))
                threshold = self.config.camera_reset_threshold
                if self._initial_camera_x > threshold and camera_x < self._initial_camera_x - threshold:
                    return True

            elif signal == "health_zero":
                health = values.get("health", 1)
                if health <= 0:
                    return True

            elif signal == "level_change":
                level_id = values.get("level_id", self.config.target_level_id)
                if level_id != self.config.target_level_id and level_id != 0:
                    return True

        return False

    def evaluate(
        self,
        actions: list[int] | list[list[int]],
        early_terminate: bool = True,
    ) -> EvalResult:
        """Evaluate an action sequence and return fitness result.

        *actions* can be a list of action indices (int) **or** a list of raw
        12-element button arrays (list[int]).  Raw buttons are used as-is,
        avoiding the lossy action-table mapping.

        Terminates IMMEDIATELY on death. Progress tracked via configured
        ProgressTracker.
        """
        self._ensure_env()
        assert self._env is not None and self._cached_state is not None

        # Detect whether actions are indices or raw button arrays
        raw_mode = len(actions) > 0 and isinstance(actions[0], list)

        # Fast reset via cached state
        self._env.em.set_state(self._cached_state)
        self._tracker.reset()

        # Seed tracker with initial values so progress is relative to state start
        # (matches old evaluator behavior: initial_camera_x captured before any steps)
        if self._initial_values:
            self._tracker.update(self._initial_values)

        result = EvalResult()
        max_x = 0.0
        frames_since_progress = 0
        prev_progress = 0.0
        start_timer: Optional[int] = None
        gameplay_started = False
        bonus_frames = 0

        # Workaround: stable-retro ignores SNES Select for SM weapon toggle.
        # Replicate the rising-edge RAM write from the play session so
        # recordings that used Select replay correctly.
        _select_prev = False
        _select_val = 0
        _has_selected_item: Optional[bool] = None

        for frame_idx, action in enumerate(actions):
            if raw_mode:
                buttons = list(action)  # type: ignore[arg-type]
            else:
                buttons = action_index_to_buttons(action, self._action_table)  # type: ignore[arg-type]
            action_size = self._env.action_space.shape[0]
            if len(buttons) < action_size:
                buttons = buttons + [0] * (action_size - len(buttons))
            elif len(buttons) > action_size:
                buttons = buttons[:action_size]

            # Select toggle workaround (rising edge → RAM write)
            if len(buttons) > 2 and buttons[2]:  # SNES Select is button index 2
                if not _select_prev:
                    if _has_selected_item is None:
                        try:
                            self._env.unwrapped.data.lookup_value("selected_item")
                            _has_selected_item = True
                        except Exception:
                            _has_selected_item = False
                    if _has_selected_item:
                        _select_val ^= 1
                        try:
                            self._env.unwrapped.data.set_value("selected_item", _select_val)
                        except Exception:
                            pass
                _select_prev = True
            else:
                _select_prev = False

            self._env.step(np.array(buttons, dtype=np.int8))

            ram = self._env.get_ram()
            values = self._read_ram(ram)

            camera_x = float(values.get("camera_x", 0))
            x = float(values.get("player_x", 0))
            y = float(values.get("player_y", 0))
            level_id = values.get("level_id", 0)
            timer = self._read_timer_total(values)

            if start_timer is None and timer > 0:
                start_timer = timer

            # Track when gameplay starts
            # For camera-based games, wait for camera scroll; for player-position
            # games (SM), start immediately since camera_x is absent/zero.
            if not gameplay_started:
                if self._initial_camera_x > 0:
                    if camera_x > self._initial_camera_x:
                        gameplay_started = True
                else:
                    gameplay_started = True

            # Sub-level detection: freeze progress + stall when in bonus room.
            # level_id_aliases are part of the same level (not sub-levels).
            in_sub_level = (level_id != 0 and level_id not in self._main_level_ids)

            if not in_sub_level:
                # Update progress tracker (only in main level)
                progress = self._tracker.update(values)
                if progress > prev_progress:
                    prev_progress = progress
                    frames_since_progress = 0
                else:
                    frames_since_progress += 1

                # Track screen-relative x for reporting (main level only)
                if 0 < x < 30000 and x > max_x:
                    max_x = x
            else:
                bonus_frames += 1
                frames_since_progress += 1  # penalize time in bonus rooms

            # === COMPLETION CHECK (before death, since level transitions
            #     can trigger camera_reset / level_change death signals) ===
            if self.config.completion_signal == "level_id_change":
                if level_id not in self._main_level_ids and level_id != 0:
                    # Check if this is a real completion or bonus room
                    is_real_completion = (
                        self._tracker.max_progress >= self.config.completion_min_progress
                        and (not self.config.completion_level_ids
                             or level_id in self.config.completion_level_ids)
                        and level_id not in self.config.completion_exclude_ids
                    )
                    if is_real_completion:
                        result.completed = True
                        result.total_frames = frame_idx + 1
                        if start_timer is not None:
                            result.timer_frames = timer - start_timer
                        result.max_x = max_x
                        result.max_progress = self._tracker.max_progress
                        result.final_x = x
                        result.final_y = y
                        result.level_id_at_end = level_id
                        result.bonus_frames = bonus_frames
                        result.fitness = self.config.completion_bonus - frame_idx
                        return result
                    # else: bonus room / mid-level transition, continue playing

            # === DEATH DETECTION (immediate) ===
            if gameplay_started and self.is_dead(values, in_sub_level=in_sub_level):
                result.died = True
                result.total_frames = frame_idx + 1
                result.max_x = max_x
                result.max_progress = self._tracker.max_progress
                result.final_x = x
                result.final_y = y
                result.level_id_at_end = level_id
                result.bonus_frames = bonus_frames
                result.fitness = (
                    self._tracker.max_progress * self.config.progress_weight
                    - self.config.death_penalty
                )
                return result

            # Early termination if stuck
            if early_terminate and frames_since_progress > self.config.max_stall_frames:
                # For backtrack-aware trackers, also check if truly stalled
                if not hasattr(self._tracker, 'is_stalled') or not self._tracker.is_stalled:
                    # Only terminate on frame stall, not backtrack stall
                    pass
                result.early_terminated = True
                break

        # Didn't complete (or early terminated)
        ram = self._env.get_ram()
        values = self._read_ram(ram)
        result.total_frames = frame_idx + 1 if actions else 0
        result.max_x = max_x
        result.max_progress = self._tracker.max_progress
        result.final_x = float(values.get("player_x", 0))
        result.final_y = float(values.get("player_y", 0))
        result.level_id_at_end = values.get("level_id", 0)
        result.bonus_frames = bonus_frames
        result.fitness = self._tracker.max_progress * self.config.progress_weight
        return result

    def evaluate_trace(self, actions: list[int] | list[list[int]]) -> EvalResult:
        """Like evaluate(early_terminate=False) but prints level_id changes."""
        self._ensure_env()
        assert self._env is not None and self._cached_state is not None

        self._env.em.set_state(self._cached_state)

        raw_mode = len(actions) > 0 and isinstance(actions[0], list)
        action_table = self._action_table
        prev_lid = self.config.target_level_id

        _select_prev = False
        _select_val = 0
        _has_selected_item: bool | None = None

        for frame_idx, action in enumerate(actions):
            if raw_mode:
                buttons = list(action)  # type: ignore[arg-type]
            else:
                buttons = action_index_to_buttons(action, action_table)  # type: ignore[arg-type]
            action_size = self._env.action_space.shape[0]
            if len(buttons) < action_size:
                buttons = buttons + [0] * (action_size - len(buttons))
            elif len(buttons) > action_size:
                buttons = buttons[:action_size]

            # Select toggle workaround (same as evaluate)
            if len(buttons) > 2 and buttons[2]:
                if not _select_prev:
                    if _has_selected_item is None:
                        try:
                            self._env.unwrapped.data.lookup_value("selected_item")
                            _has_selected_item = True
                        except Exception:
                            _has_selected_item = False
                    if _has_selected_item:
                        _select_val ^= 1
                        try:
                            self._env.unwrapped.data.set_value("selected_item", _select_val)
                        except Exception:
                            pass
                _select_prev = True
            else:
                _select_prev = False

            self._env.step(np.array(buttons, dtype=np.int8))

            ram = self._env.get_ram()
            values = self._read_ram(ram)
            lid = values.get("level_id", 0)
            cx = float(values.get("camera_x", 0))
            lives = values.get("lives", 0)

            if lid != prev_lid:
                print(f"  frame {frame_idx:>5}: level_id 0x{prev_lid:04X} -> 0x{lid:04X}  camera_x={cx:.0f}  lives={lives}")
                prev_lid = lid

        # Run normal evaluate for the actual result
        return self.evaluate(actions, early_terminate=False)

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None
            self._cached_state = None

    def __del__(self):
        self.close()
