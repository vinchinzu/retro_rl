"""Per-plot water step helpers for CropWaterTask (rr-ds3 extract).

Residual crop-walk recovery, step retarget/reorder, walkable overlays, and
water-phase start live here so the mono only composes them.
"""

from __future__ import annotations

from harvest.tasks.crop_fsm import CropState, PlotPhase

from typing import List, Optional, Set, Tuple

import numpy as np

from harvest.tasks.nav import get_tile_at, tile_dist
from harvest.tasks.crop_geometry import (
    WATER_REFILL_THRESHOLD,
    build_water_steps,
    is_rainy_weather,
    plot_tiles,
    refill_action_sequence,
    tile_can_be_water_target,
    water_action_sequence,
    _water_step_variants,
)
from harvest.core.tile_catalog import Tool
from harvest.tasks.farm_clearer import use_tool
from retro_harness import TaskResult


class CropWaterOpsMixin:
    """Water-phase helpers for CropWaterTask."""

    def _try_residual_crop_walk_recovery(self, ram: np.ndarray) -> bool:
        """After partial water success, allow walking crop tiles for last dry.

        Gate: plot watered, can spent, re-scan pass, or reorder-cap hit.
        Residual (12,26) often needs stand on wet (13,26) face left.
        """
        if self._plot_phase != PlotPhase.WATER:
            return False
        # Never residual-act with an empty can (unit + ROM pass re-scan).
        if self._water_level(ram) < 1:
            return False
        can_spent = (
            self._pre_water_level >= 0
            and self._water_level(ram) < self._pre_water_level
        )
        if (
            self._plot_watered < 1
            and self._pass_number < 2
            and not can_spent
            and getattr(self, "_water_step_retries", 0) < 3
        ):
            return False
        if getattr(self, "_water_crop_walk_recoveries", 0) >= 3:
            return False
        self._water_crop_walk_recoveries = (
            getattr(self, "_water_crop_walk_recoveries", 0) + 1
        )
        self._allow_crop_walkable = True
        self._pathfinder.temp_blocked.clear()
        if self._plot_index < len(self._plots):
            self._pathfinder.extra_walkable = set(
                plot_tiles(self._plots[self._plot_index], include_center=True)
            )
        player = self._navigator.current_tile
        # Adjacent face-water when already next to residual dry.
        if self._target_tile is not None and tile_dist(player, self._target_tile) == 1:
            face = self._face_from_approach(player, self._target_tile)
            self._approach_tile = player
            self._face_direction = face
            if self._water_index < len(self._water_steps):
                self._water_steps[self._water_index] = (
                    self._target_tile,
                    player,
                    face,
                )
            print(
                f"[CROP] Residual adjacent water at {player} → "
                f"{self._target_tile} face={face}"
            )
            self._state = CropState.ACT
            self._navigator.path = []
            self._steps_on_target = 0
            return True
        # Prefer stand on already-wet crop neighbor of residual dry.
        if self._target_tile is not None:
            for face, (dx, dy) in (
                ("left", (1, 0)),
                ("right", (-1, 0)),
                ("up", (0, 1)),
                ("down", (0, -1)),
            ):
                stand = (self._target_tile[0] + dx, self._target_tile[1] + dy)
                if stand not in self._current_plot_tiles() and stand != player:
                    continue
                # Path with full plot walkable.
                path = self._pathfinder.find_path(
                    ram,
                    player,
                    stand,
                    walkable_override=set(self._current_plot_tiles()) | {stand},
                )
                if path is not None or stand == player:
                    self._approach_tile = stand
                    self._face_direction = face
                    if self._water_index < len(self._water_steps):
                        self._water_steps[self._water_index] = (
                            self._target_tile,
                            stand,
                            face,
                        )
                    if stand == player:
                        print(
                            f"[CROP] Residual act on plot stand {stand} "
                            f"→ {self._target_tile} face={face}"
                        )
                        self._state = CropState.ACT
                        self._navigator.path = []
                        self._steps_on_target = 0
                        return True
                    print(
                        f"[CROP] Residual crop-stand {stand} face={face} "
                        f"for {self._target_tile}"
                    )
                    self._state = CropState.NAVIGATE
                    self._navigator.path = path or []
                    self._navigator.stasis = 0
                    self._steps_on_target = 0
                    return True
        retargeted = self._retarget_current_water_step(
            ram
        ) or self._reorder_remaining_water_steps(ram)
        if self._plot_index < len(self._plots):
            self._pathfinder.extra_walkable |= set(
                plot_tiles(self._plots[self._plot_index], include_center=True)
            )
        print(
            f"[CROP] Residual crop-walk recovery "
            f"n={self._water_crop_walk_recoveries} retarget={retargeted} "
            f"target={self._target_tile} stand={self._approach_tile}"
        )
        self._state = CropState.NAVIGATE
        self._navigator.path = []
        self._navigator.stasis = 0
        self._steps_on_target = 0
        return True

    def _retarget_current_water_step(self, ram: np.ndarray) -> bool:
        if self._plot_phase != PlotPhase.WATER or self._target_tile is None:
            return False
        best = self._best_water_variant(ram, self._target_tile, self._navigator.current_tile)
        if best is None:
            return False
        stand, face, _score = best
        changed = stand != self._approach_tile or face != self._face_direction
        self._approach_tile = stand
        self._face_direction = face
        if self._water_index < len(self._water_steps):
            self._water_steps[self._water_index] = (self._target_tile, stand, face)
        self._set_water_walkable()
        return changed

    def _reorder_remaining_water_steps(self, ram: np.ndarray) -> bool:
        if self._plot_phase != PlotPhase.WATER or self._water_index >= len(self._water_steps):
            return False

        current_tile = self._navigator.current_tile
        prefix = self._water_steps[:self._water_index]
        remaining_scored = []
        for offset, (target, _stand, _face) in enumerate(self._water_steps[self._water_index:]):
            best = self._best_water_variant(ram, target, current_tile)
            if best is None:
                continue
            stand, face, score = best
            remaining_scored.append(((score, offset), (target, stand, face)))

        if not remaining_scored:
            return False

        reordered = [step for _score, step in sorted(remaining_scored, key=lambda item: item[0])]
        changed = reordered != self._water_steps[self._water_index:]
        self._water_steps = prefix + reordered
        self._target_tile, self._approach_tile, self._face_direction = self._water_steps[self._water_index]
        return changed

    def _reprioritize_water_step(self, ram: np.ndarray, *, reason: str) -> bool:
        # Cap reorders on the same residual dry — thrash between (12,25)/(12,27)
        # never advances and blocks residual crop-walk recovery.
        retries = getattr(self, "_water_step_retries", 0)
        if retries >= 3:
            return False
        if not self._reorder_remaining_water_steps(ram):
            return False
        self._water_step_retries = retries + 1
        print(
            f"[CROP] REORDER water tiles ({reason}) "
            f"target={self._target_tile} stand={self._approach_tile} "
            f"face={self._face_direction} retry={self._water_step_retries}/3"
        )
        self._state = CropState.NAVIGATE
        self._navigator.path = []
        self._steps_on_target = 0
        return True

    def _begin_water_phase(self, ram: np.ndarray, allow_unknown_tiles: bool = False):
        """Set up per-tile watering for current plot using WATER_PLAN_CENTER."""
        if self._plot_index >= len(self._plots):
            return
        center = self._plots[self._plot_index]
        cx, cy = center
        if is_rainy_weather(ram):
            print(
                f"[CROP] Plot {self._plot_index + 1}/{len(self._plots)} "
                f"center=({cx},{cy}) rain; skipping manual watering"
            )
            self._advance_plot(ram)
            return

        self._plot_phase = PlotPhase.WATER
        self._plot_watered = 0
        self._plot_skipped = 0
        self._allow_unknown_water_tiles = allow_unknown_tiles
        self._allow_crop_walkable = allow_unknown_tiles
        self._water_verify_retries = 0
        self._last_water_level_before = -1
        self._last_water_tile_before = -1
        # Build concrete per-tile watering steps. Resume states keep stands on
        # the notch/perimeter when possible, but allow inside-plot recovery.
        self._water_steps = build_water_steps(
            ram,
            center,
            allow_crop_walkable=self._allow_crop_walkable,
            allow_unknown_tiles=allow_unknown_tiles,
            include_fresh_tilled=allow_unknown_tiles,
            start_tile=self._navigator.current_tile,
            skip_tiles=set(self.skip_water_tiles),
        )
        self._water_index = 0

        water_lvl = self._water_level(ram)
        self._pre_water_level = water_lvl  # plot-level: track starting level
        mode = "unknown-ok" if allow_unknown_tiles else "dry-only"
        print(f"[CROP] Plot {self._plot_index + 1}/{len(self._plots)} center=({cx},{cy}) phase=WATER can={water_lvl} mode={mode} steps={len(self._water_steps)}")

        if not self._water_steps:
            print(f"[CROP] Plot {self._plot_index + 1}/{len(self._plots)} has no reachable water targets")
            self._advance_plot(ram)
            return

        self._reorder_remaining_water_steps(ram)
        # Navigate to first stand position
        target, stand, face = self._water_steps[0]
        self._target_tile = target
        self._approach_tile = stand
        self._face_direction = face
        self._set_water_walkable()
        self._state = CropState.NAVIGATE
        self._navigator.path = []
        self._steps_on_target = 0

    def _advance_water_step(self, ram: np.ndarray):
        """Move to the next water step, or finish the plot."""
        self._water_verify_retries = 0
        self._water_step_retries = 0
        self._last_water_level_before = -1
        self._last_water_tile_before = -1
        self._water_index += 1
        # Do not carry stasis blocks into the next stand attempt.
        self._pathfinder.temp_blocked.clear()
        if self._water_index >= len(self._water_steps):
            # All tiles attempted — plot-level verification
            center = self._plots[self._plot_index]
            cx, cy = center
            lvl = self._water_level(ram)
            water_used = max(0, self._pre_water_level - lvl) if self._pre_water_level >= 0 else 0
            actual_watered = self._plot_watered
            actual_skipped = self._plot_skipped

            self.watered_count += actual_watered
            self.skipped_water += actual_skipped

            tile_ids = []
            for dy in range(-1, 2):
                row = []
                for dx in range(-1, 2):
                    tid = get_tile_at(ram, cx + dx, cy + dy)
                    row.append(f"0x{tid:02X}")
                tile_ids.append(" ".join(row))
            print(f"[CROP] Plot {self._plot_index + 1}/{len(self._plots)} WATER DONE: "
                  f"{actual_watered}/{len(self._water_steps)} watered (used {water_used} water, can={lvl})")
            print(f"[CROP]   3x3 tiles: [{tile_ids[0]}] [{tile_ids[1]}] [{tile_ids[2]}]")
            if actual_skipped > 0:
                print(f"[CROP] WARNING: Plot {self._plot_index + 1} incomplete ({actual_skipped} skipped)")
            self._pre_water_level = -1
            self._advance_plot(ram)
        else:
            self._reorder_remaining_water_steps(ram)
            # Navigate to next stand position
            target, stand, face = self._water_steps[self._water_index]
            self._target_tile = target
            self._approach_tile = stand
            self._face_direction = face
            self._set_water_walkable()
            self._state = CropState.NAVIGATE
            self._navigator.path = []
            self._steps_on_target = 0

    def _best_water_variant(
        self,
        ram: np.ndarray,
        target: Tuple[int, int],
        current_tile: Tuple[int, int],
    ) -> Optional[Tuple[Tuple[int, int], str, int]]:
        """Pick the best currently-reachable adjacent stand for a water target."""
        if self._plot_index >= len(self._plots):
            return None

        center = self._plots[self._plot_index]
        variants = _water_step_variants(
            ram,
            center,
            target,
            allow_crop_walkable=self._allow_crop_walkable,
        )
        plot_set = self._current_plot_tiles()
        best: Optional[Tuple[Tuple[int, int], str, int]] = None
        for stand, face in variants:
            # Residual recovery needs path *through* wet crop neighbors, not
            # only the stand cell itself.
            if self._allow_crop_walkable and plot_set:
                walkable_override = set(plot_set)
                walkable_override.add(stand)
            elif self._allow_crop_walkable and stand in plot_set:
                walkable_override = {stand}
            else:
                walkable_override = None
            path = self._pathfinder.find_path(
                ram, current_tile, stand, walkable_override=walkable_override
            )
            if path is None:
                continue
            # Strongly prefer perimeter/notch stands over standing on crops.
            score = tile_dist(current_tile, stand)
            if stand in plot_set:
                score += 32
            candidate = (stand, face, score)
            if best is None or candidate[2] < best[2]:
                best = candidate
        return best

    @staticmethod
    def _face_from_approach(approach: Tuple[int, int], target: Tuple[int, int]) -> str:
        """Derive face direction from stand tile toward target tile."""
        dx = target[0] - approach[0]
        dy = target[1] - approach[1]
        if abs(dx) >= abs(dy):
            return "right" if dx > 0 else "left"
        return "down" if dy > 0 else "up"

    def _set_water_walkable(self) -> None:
        """Allow crop stands; full 3x3 when residual crop-walk recovery is armed.

        Intermediate wet tiles (0x55) must stay walkable while pathing onto a
        residual stand (e.g. (13,26) for dry (12,26)) — otherwise follow_path
        blocks mid-plot and re-adds temp_blocked.
        """
        self._pathfinder.extra_walkable.clear()
        if self._plot_phase != PlotPhase.WATER:
            return
        plot = self._current_plot_tiles()
        if self._allow_crop_walkable and plot:
            if getattr(self, "_water_crop_walk_recoveries", 0) > 0:
                self._pathfinder.extra_walkable = set(plot)
            elif self._approach_tile is not None and self._approach_tile in plot:
                self._pathfinder.extra_walkable.add(self._approach_tile)

    def _set_crop_walkable(self):
        """Mark current plot's 3x3 tiles as walkable on the pathfinder.

        Freshly planted crops are walkable in-game for the first few days.
        Sets pathfinder.extra_walkable so both find_path and follow_path work.
        """
        self._pathfinder.extra_walkable.clear()
        if self._plot_index < len(self._plots):
            center = self._plots[self._plot_index]
            self._pathfinder.extra_walkable = set(plot_tiles(center, include_center=True))

    def _clear_crop_walkable(self):
        """Remove crop walkable overrides from pathfinder."""
        self._pathfinder.extra_walkable.clear()

    def _current_plot_tiles(self) -> Set[Tuple[int, int]]:
        if self._plot_index >= len(self._plots):
            return set()
        return set(plot_tiles(self._plots[self._plot_index], include_center=True))

    def _act_water(self, ram: np.ndarray) -> Optional[TaskResult]:
        """Water current tile using navigator-precise positioning."""
        if self._tool_mgr.current != Tool.WATERING_CAN:
            self._tool_mgr.start_search()
            self._state = CropState.TOOL_SWITCH
            return None

        # Skip tiles that don't need watering (dried tilled, untilled, etc.)
        if self._water_index < len(self._water_steps):
            target = self._water_steps[self._water_index][0]
            tid = get_tile_at(ram, target[0], target[1])
            if target in self.skip_water_tiles or not tile_can_be_water_target(
                tid,
                allow_unknown=self._allow_unknown_water_tiles,
                include_fresh_tilled=self._allow_unknown_water_tiles,
            ):
                print(f"[CROP] SKIP water tile {self._water_index + 1}/{len(self._water_steps)} target={target} "
                      f"tid=0x{tid:02X} (not a dry target)")
                self._advance_water_step(ram)
                return None

        water_lvl = self._water_level(ram)

        # Empty can: always refill before attempting water (ToolUsed early-outs at 0).
        if water_lvl < 1 and not self._refill_exhausted:
            print(f"[CROP] Empty can (level={water_lvl}), need refill before watering")
            self._start_refill(ram)
            return None

        # Count only waterable remaining tiles for partial-can refill check
        waterable_remaining = 0
        for i in range(self._water_index, len(self._water_steps)):
            t = self._water_steps[i][0]
            if t in self.skip_water_tiles:
                continue
            if tile_can_be_water_target(
                get_tile_at(ram, t[0], t[1]),
                allow_unknown=self._allow_unknown_water_tiles,
                include_fresh_tilled=self._allow_unknown_water_tiles,
            ):
                waterable_remaining += 1

        if water_lvl < waterable_remaining and not self._refill_exhausted:
            print(f"[CROP] Water level={water_lvl} < {waterable_remaining} waterable remaining, need refill")
            self._start_refill(ram)
            return None

        if water_lvl < 1 and self._refill_exhausted:
            # Empty and can't refill — skip remaining tiles
            remaining = len(self._water_steps) - self._water_index
            print(f"[CROP] Empty can, no refill, skipping {remaining} remaining tiles")
            self.skipped_water += remaining
            self._plot_skipped += remaining
            self._water_index = len(self._water_steps)
            self._advance_water_step(ram)
            return None

        face = self._face_direction or "down"

        if self.debug or self._water_index == 0:
            target = self._water_steps[self._water_index][0] if self._water_index < len(self._water_steps) else None
            print(f"[CROP] WATER tile {self._water_index + 1}/{len(self._water_steps)} target={target} face={face} can={water_lvl}")

        self._last_water_level_before = water_lvl
        self._last_water_tile_before = tid
        self._action_queue.extend(water_action_sequence(face, cooldown=60, face_frames=1))
        self._state = CropState.VERIFY
        return None

    def _act_refill(self, ram: np.ndarray) -> Optional[TaskResult]:
        """Refill watering can at pond."""
        if self._tool_mgr.current != Tool.WATERING_CAN:
            self._tool_mgr.start_search()
            self._state = CropState.TOOL_SWITCH
            return None

        face = self._refill_pond_face or "down"
        # Record level right before action (not during _start_refill which is pre-navigation)
        self._refill_level_before = self._water_level(ram)
        print(f"[CROP] REFILL facing {face} can={self._refill_level_before}")

        self._action_queue.extend(refill_action_sequence(face))
        self._state = CropState.VERIFY
        return None

