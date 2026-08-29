"""Per-plot water step helpers for CropWaterTask (rr-ds3 extract).

Residual crop-walk recovery, step retarget/reorder, walkable overlays,
water-phase start, and center/act/verify/tool-switch arms live here so
the composer only dispatches them.
"""

from __future__ import annotations

from harvest.tasks.crop_planter import CropState, ON_APPROACH_PHASES, PlotPhase

from typing import Optional, Set, Tuple

import numpy as np

from harvest.tasks.nav import get_tile_at, tile_dist
from harvest.tasks.crop_geometry import (
    build_water_steps,
    count_tilled,
    is_rainy_weather,
    plot_tiles,
    pond_access_blocking_fences,
    refill_action_sequence,
    tile_can_be_water_target,
    tile_is_watered,
    water_action_sequence,
    _water_step_variants,
)
from harvest.core.carry import SEED_ITEM
from harvest.core.tile_catalog import ADDR_INPUT_LOCK, Tool
from harvest.tasks.farm_ops import cycle_tool
from retro_harness import TaskResult, TaskStatus


class CropWaterOpsMixin:
    """Water-phase helpers plus act/verify/center/tool_switch for CropWaterTask."""

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

    def _handle_center(self, ram: np.ndarray) -> Optional[TaskResult]:
        if self._approach_tile is None:
            self._state = CropState.DETECT
            return None
        tol = 1 if self._plot_phase in ON_APPROACH_PHASES else 2
        center_action = self._navigator.center_on_tile(self._approach_tile, tolerance=tol)
        if center_action is None:
            if self._plot_phase == PlotPhase.STAGE_POND:
                # Staged — hand off to fence clear (skip re-stage).
                fences = pond_access_blocking_fences(ram)
                print(
                    f"[CROP] Pond stage reached at {self._navigator.current_tile}; "
                    f"starting fence clear (wall n={len(fences)})"
                )
                if fences and self._try_open_pond_access(ram, fences, skip_stage=True):
                    return None
                self._plot_phase = PlotPhase.WATER
                self._start_refill(ram)
                return None
            self._state = CropState.ACT
        else:
            self._action_queue.append(center_action)
        return None

    def _handle_act(self, ram: np.ndarray) -> Optional[TaskResult]:
        if self._action_queue:
            return None

        # Wait for input lock to clear and player to settle
        input_lock = int(ram[ADDR_INPUT_LOCK]) if ADDR_INPUT_LOCK < len(ram) else 0
        if input_lock != 1 or self._navigator.stasis < 6:
            return None

        # Position check: must be on the correct tile
        player = self._navigator.current_tile
        if self._plot_phase in ON_APPROACH_PHASES:
            # Must be ON approach tile for plant/water/hoe
            if player != self._approach_tile:
                print(f"[CROP] {self._plot_phase.upper()} pos mismatch: at ({player[0]},{player[1]}) need ({self._approach_tile[0]},{self._approach_tile[1]}), re-navigate")
                self._state = CropState.NAVIGATE
                self._navigator.path = []
                return None
        else:
            # Refill: on or adjacent to approach tile
            if tile_dist(player, self._approach_tile) > 1:
                self._state = CropState.NAVIGATE
                self._navigator.path = []
                return None

        # Re-center drift correction
        tol = 1 if self._plot_phase in ON_APPROACH_PHASES else 2
        center_action = self._navigator.center_on_tile(self._approach_tile, tolerance=tol)
        if center_action is not None:
            self._action_queue.append(center_action)
            return None

        if self._plot_phase == PlotPhase.STAGE_POND:
            fences = pond_access_blocking_fences(ram)
            if fences and self._try_open_pond_access(ram, fences, skip_stage=True):
                return None
            self._plot_phase = PlotPhase.WATER
            self._start_refill(ram)
            return None

        act_handlers = {
            PlotPhase.PLANT: self._act_plant,
            PlotPhase.HOE: self._act_hoe,
            PlotPhase.WATER: self._act_water,
            PlotPhase.REFILL: self._act_refill,
        }
        handler = act_handlers.get(self._plot_phase)
        if handler is not None:
            return handler(ram)
        return None

    def _handle_verify(self, ram: np.ndarray) -> Optional[TaskResult]:
        if self._action_queue:
            return None

        input_lock = int(ram[ADDR_INPUT_LOCK]) if ADDR_INPUT_LOCK < len(ram) else 0
        if input_lock != 1:
            return None

        if self._plot_phase == PlotPhase.HOE:
            self._advance_hoe_step(ram)
            return None

        if self._plot_phase == PlotPhase.PLANT:
            center = self._plots[self._plot_index]
            tilled_remaining = count_tilled(ram, center)
            # Don't retry - plant action fires once per plot.  If position
            # and tool were correct (checked in _handle_act), seeds were used.
            # Tile data may lag behind the animation; retrying wastes seeds.
            self.planted_count += 1
            if tilled_remaining == 0:
                print(f"[CROP] PLANT OK plot {self._plot_index + 1} planted={self.planted_count}")
            else:
                print(f"[CROP] PLANT OK plot {self._plot_index + 1} planted={self.planted_count} ({tilled_remaining} tiles still updating)")
            if self._is_establish_only:
                # Day-plan plant pass: seeds+hoe only; water after can re-fetch.
                self._advance_plot(ram)
            else:
                self._begin_water_phase(ram, allow_unknown_tiles=True)

        elif self._plot_phase == PlotPhase.WATER:
            if self._water_index >= len(self._water_steps):
                self._advance_water_step(ram)
                return None

            target = self._water_steps[self._water_index][0]
            lvl_after = self._water_level(ram)
            tid_after = get_tile_at(ram, target[0], target[1])
            used_water = self._last_water_level_before >= 0 and lvl_after < self._last_water_level_before
            tile_watered = tile_is_watered(tid_after)

            if used_water or tile_watered:
                self._plot_watered += 1
                self._water_verify_retries = 0
                self._advance_water_step(ram)
            else:
                self._water_verify_retries += 1
                if self._water_verify_retries >= 2:
                    self.skipped_water += 1
                    self._plot_skipped += 1
                    print(
                        f"[CROP] SKIP water tile {self._water_index + 1}/{len(self._water_steps)} "
                        f"target={target} tid=0x{tid_after:02X} can={lvl_after} (verify failed)"
                    )
                    self._water_verify_retries = 0
                    self._advance_water_step(ram)
                else:
                    print(
                        f"[CROP] RETRY water tile {self._water_index + 1}/{len(self._water_steps)} "
                        f"target={target} tid=0x{tid_after:02X} can={lvl_after}"
                    )
                    self._state = CropState.CENTER

        elif self._plot_phase == PlotPhase.REFILL:
            lvl_after = self._water_level(ram)
            if lvl_after > self._refill_level_before:
                # Refill succeeded — navigate back to current water step
                self.refill_count += 1
                print(f"[CROP] REFILL OK can={lvl_after} (was {self._refill_level_before}) refills={self.refill_count}")
                self._pre_water_level = lvl_after  # reset for plot-level verification
                self._refill_search_level = -1  # reset search tracking
                self._plot_phase = PlotPhase.WATER
                self._set_water_walkable()
                if self._water_index < len(self._water_steps):
                    target, stand, face = self._water_steps[self._water_index]
                    self._target_tile = target
                    self._approach_tile = stand
                    self._face_direction = face
                else:
                    center = self._plots[self._plot_index]
                    self._target_tile = center
                    self._approach_tile = center
                self._state = CropState.NAVIGATE
                self._navigator.path = []
                self._steps_on_target = 0
            else:
                # Refill failed — mark tile and neighbors as bad, try another
                bad = self._refill_pond_tile
                self._bad_refill_tiles.add(bad)
                # If water was CONSUMED (level decreased), this area is actively
                # harmful — mark a 2-tile radius as bad to skip nearby tiles
                if lvl_after < self._refill_level_before:
                    for dy in range(-2, 3):
                        for dx in range(-2, 3):
                            self._bad_refill_tiles.add((bad[0] + dx, bad[1] + dy))
                    print(f"[CROP] REFILL FAILED at ({bad[0]},{bad[1]}) can={lvl_after} (was {self._refill_level_before}), "
                          f"water consumed! blacklisted neighborhood, trying next")
                else:
                    print(f"[CROP] REFILL FAILED at ({bad[0]},{bad[1]}) can={lvl_after} (was {self._refill_level_before}), trying next")
                self._plot_phase = PlotPhase.WATER
                self._start_refill(ram)  # try another water edge

        if self._state == CropState.DONE:
            return self._terminal_result()

        return None

    def _handle_tool_switch(self, ram: np.ndarray) -> Optional[TaskResult]:
        """Cycle tools to find the needed one."""
        if self._plot_phase == PlotPhase.PLANT:
            wanted = SEED_ITEM.get(self.seed_type, SEED_ITEM["potato"])
        elif self._plot_phase == PlotPhase.HOE:
            wanted = int(Tool.HOE)
        else:
            wanted = int(Tool.WATERING_CAN)

        self._tool_mgr.update(ram)
        current = self._tool_mgr.current

        if current == wanted:
            if self.debug:
                print(f"[CROP] Found tool 0x{wanted:02X}")
            self._state = CropState.CENTER
            return None

        # d2_farm_plant: spent seed bag leaves selected=0 with can in backpack.
        # One X selects it — do not wait for a full cycle or declare missing.
        if self._tool_mgr.needs_swap(wanted):
            if self.debug:
                print(
                    f"[CROP] Swap to 0x{wanted:02X} "
                    f"(selected=0x{current:02X} backpack=0x{self._tool_mgr.backpack:02X})"
                )
            self._action_queue.extend(cycle_tool())
            return None

        self._tool_mgr.record()

        if self._tool_mgr.cycle_complete() and not self._tool_mgr.has(wanted):
            if self._plot_phase == PlotPhase.PLANT:
                center = self._plots[self._plot_index] if self._plot_index < len(self._plots) else None
                print(f"[CROP] Seed tool 0x{wanted:02X} not found, skipping plant plot at {center}")
                self._advance_plot(ram)
                return None
            if self._plot_phase == PlotPhase.HOE:
                center = self._plots[self._plot_index] if self._plot_index < len(self._plots) else None
                print(f"[CROP] Hoe 0x{wanted:02X} not found, skipping establish plot at {center}")
                self._advance_plot(ram)
                return None
            # Watering can missing after plant is common (only 2 carry slots).
            # Report partial success so the day plan can re-fetch the can and
            # run a second CROP_WATER pass instead of aborting the whole day.
            if self.planted_count > 0 or self.watered_count > 0:
                msg = (
                    f"planted={self.planted_count} watered={self.watered_count} "
                    f"refills={self.refill_count}; tool 0x{wanted:02X} not in carry pair"
                )
                print(f"[CROP] Partial complete: {msg}")
                return TaskResult(status=TaskStatus.SUCCESS, reason=msg)
            # Water-only pass without the can: keep the explicit reason so the
            # day plan can recover via ENSURE_WATERING_CAN / recovery task.
            if self._is_water_only and wanted == int(Tool.WATERING_CAN):
                print("[CROP] Watering can not in carry pair (water-only pass)")
                return TaskResult(
                    status=TaskStatus.FAILURE,
                    reason="watering can not in carry pair",
                )
            print(f"[CROP] Tool 0x{wanted:02X} not found in inventory")
            return TaskResult(status=TaskStatus.FAILURE, reason=f"tool 0x{wanted:02X} not in inventory")

        self._action_queue.extend(cycle_tool())
        return None

