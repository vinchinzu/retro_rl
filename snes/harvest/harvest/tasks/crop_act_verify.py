"""Center / act / verify / tool-switch arms for CropWaterTask (rr-ds3)."""

from __future__ import annotations

from typing import Optional

import numpy as np

from retro_harness import TaskResult, TaskStatus

from harvest.core.carry import SEED_ITEM
from harvest.core.tile_catalog import ADDR_INPUT_LOCK, Tool
from harvest.tasks.crop_fsm import CropState, PlotPhase, ON_APPROACH_PHASES
from harvest.tasks.crop_geometry import (
    count_tilled,
    pond_access_blocking_fences,
    tile_is_watered,
)
from harvest.tasks.farm_clearer import cycle_tool
from harvest.tasks.nav import get_tile_at, tile_dist


class CropActVerifyMixin:
    """Act/verify/center/tool_switch handlers for CropWaterTask."""

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

