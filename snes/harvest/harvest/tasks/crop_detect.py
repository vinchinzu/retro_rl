"""Detect / start-plot / advance-plot handlers for CropWaterTask (rr-ds3)."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from retro_harness import TaskResult, TaskStatus

from harvest.core.carry import seed_in_carry_pair as seed_item_in_carry_pair
from harvest.tasks.crop_fsm import CropState, PlotPhase
from harvest.tasks.crop_geometry import (
    _count_crop_tiles,
    _merge_plot_centers,
    count_tilled,
    detect_crop_resume_plots,
    detect_plots,
    tile_needs_watering,
)
from harvest.tasks.nav import get_tile_at, tile_dist
from harvest.tasks.water_refill import crop_completion_status


class CropDetectMixin:
    """Detect and plot-lifecycle methods for CropWaterTask."""

    def _count_dry_crop_tiles(self, ram: np.ndarray) -> int:
        """Count dry crop / waterable tiles in task bounds (for acceptance)."""
        x0, y0, x1, y1 = self.bounds
        n = 0
        for ty in range(y0, y1 + 1):
            for tx in range(x0, x1 + 1):
                tid = get_tile_at(ram, tx, ty)
                if tile_needs_watering(tid):
                    n += 1
        return n

    def _snapshot_start_acceptance(self, ram: np.ndarray) -> None:
        """Capture dry-tile / seed-stock facts once at first detect."""
        if self._acceptance_snapped:
            return
        self._dry_crop_tiles_at_start = self._count_dry_crop_tiles(ram)
        self._had_seed_stock_at_start = self._has_plantable_seed_stock(ram)
        self._acceptance_snapped = True

    def _terminal_result(self, *, rain: bool = False) -> TaskResult:
        """Map plant/water counters to SUCCESS / no_work SUCCESS / FAILURE."""
        status, reason = crop_completion_status(
            work_mode=self.work_mode,
            planted=self.planted_count,
            watered=self.watered_count,
            dry_at_start=self._dry_crop_tiles_at_start,
            refill_exhausted=self._refill_exhausted,
            had_seed_stock=self._had_seed_stock_at_start,
            rain=rain,
        )
        extra = ""
        if self.skipped_water:
            extra += f" skipped={self.skipped_water}"
        if self.refill_count:
            extra += f" refills={self.refill_count}"
        if self._pass_number:
            extra += f" passes={self._pass_number}"
        full_reason = reason + extra
        print(f"[CROP] Complete ({status}): {full_reason}")
        if status == "failure":
            return TaskResult(status=TaskStatus.FAILURE, reason=full_reason)
        return TaskResult(status=TaskStatus.SUCCESS, reason=full_reason)

    def _has_plantable_seed_stock(self, ram: np.ndarray) -> bool:
        """True when seeds are in hand or counted in inventory for this crop."""
        if seed_item_in_carry_pair(ram, self.seed_type):
            return True
        try:
            from harvest.planner.day_plan_status import ram_seed_count

            return int(ram_seed_count(ram, self.seed_type)) > 0
        except Exception:
            return False

    def _plan_bounds_around(
        self,
        anchor: Tuple[int, int],
        radius: int = 12,
    ) -> Tuple[int, int, int, int]:
        """Clamp planning to a neighborhood around ``anchor`` inside task bounds."""
        x_min, y_min, x_max, y_max = self.bounds
        ax, ay = anchor
        return (
            max(x_min, ax - radius),
            max(y_min, ay - radius),
            min(x_max, ax + radius),
            min(y_max, ay + radius),
        )

    def _plan_bounds_near_player(self, start: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Clamp planning to a viewport-reachable neighborhood around the player.

        Full-farm plans often pick distant centers that BFS cannot reach through
        stale off-screen tiles. Keep new plots within ~12 tiles of the player
        and inside the task bounds.
        """
        return self._plan_bounds_around(start, radius=12)

    def _handle_detect(self, ram: np.ndarray) -> Optional[TaskResult]:
        """Scan for crop plots."""
        self._snapshot_start_acceptance(ram)
        resume_plots = detect_crop_resume_plots(ram, self.bounds)
        if resume_plots:
            supplemental = detect_plots(ram, self.bounds)
            self._plots = _merge_plot_centers(resume_plots, supplemental)
        else:
            self._plots = detect_plots(ram, self.bounds)
        if not self._plots and self._is_water_only and self._dry_crop_tiles_at_start > 0:
            # Partial plant (west-pocket tilled=2 / 1–3 crop tiles) misses the
            # default resume min_count=4. Water keep-alive must still find those
            # dry singles/pairs or every day ends water fail: dry_crops=N watered=0
            # without attempting can/refill (rr-5in continuous residual).
            sparse = detect_crop_resume_plots(ram, self.bounds, min_count=1)
            if sparse:
                print(
                    f"[CROP] Sparse water plots (min_count=1): {sparse} "
                    f"dry={self._dry_crop_tiles_at_start}"
                )
                self._plots = sparse
            else:
                singles = self._singleton_dry_crop_centers(ram)
                if singles:
                    print(
                        f"[CROP] Singleton dry water targets: {singles} "
                        f"dry={self._dry_crop_tiles_at_start}"
                    )
                    self._plots = singles
        if not self._plots:
            # Virgin soil: plan + hoe + plant instead of silently succeeding.
            # Water-only pass never opens new plots (no seeds/hoe in carry).
            can_plant = (
                not self._is_water_only
                and self._has_plantable_seed_stock(ram)
            )
            if self._pass_number == 1 and can_plant:
                planned = self._plan_new_plot_centers(ram)
                if planned:
                    self._plots = planned
                else:
                    print("[CROP] No plots detected and no plantable plan")
                    return self._terminal_result()
            elif self._pass_number == 1:
                return self._terminal_result()
            else:
                self._state = CropState.DONE
                return None
        current_tile = self._navigator.current_tile
        self._plots.sort(key=lambda center: (tile_dist(current_tile, center), center[1], center[0]))
        self._plot_index = 0
        pass_label = f"(pass {self._pass_number})" if self._pass_number > 1 else ""
        print(
            f"[CROP] Detected {len(self._plots)} plots: {self._plots} "
            f"mode={self.work_mode} {pass_label}"
        )
        self._start_plot(ram)
        return None

    def _start_plot(self, ram: np.ndarray):
        """Begin processing the current plot."""
        if self._plot_index >= len(self._plots):
            return
        center = self._plots[self._plot_index]
        self._set_crop_walkable()  # allow pathfinding through crop tiles
        tilled = count_tilled(ram, center)
        crop_tiles = _count_crop_tiles(ram, center[0], center[1])

        # Water-only: never hoe/plant; water established crops or skip the plot.
        if self._is_water_only:
            if crop_tiles > 0:
                self._begin_water_phase(ram, allow_unknown_tiles=False)
            else:
                print(
                    f"[CROP] Plot {self._plot_index + 1}/{len(self._plots)} "
                    f"center=({center[0]},{center[1]}) water-only with no crops; skip"
                )
                self._advance_plot(ram)
            return

        # Seed bag plants a 3x3 from the center notch. Plant once enough ring
        # tiles are hoed (full 8 is ideal; partial still uses the bag).
        if crop_tiles == 0 and tilled >= 4:
            self._plot_phase = PlotPhase.PLANT
            self._target_tile = center
            self._approach_tile = center  # stand ON center to plant
            self._face_direction = "down"
            self._set_crop_walkable()
            self._state = CropState.NAVIGATE
            self._navigator.path = []  # force re-path
            self._steps_on_target = 0
            print(f"[CROP] Plot {self._plot_index + 1}/{len(self._plots)} center=({center[0]},{center[1]}) phase=PLANT tilled={tilled}")
        elif crop_tiles == 0 and tilled < 4:
            # Untilled soil: hoe the 8 ring tiles, then plant from center.
            self._begin_hoe_phase(ram)
        else:
            if crop_tiles > 0 and tilled > 0:
                print(
                    f"[CROP] Plot {self._plot_index + 1}/{len(self._plots)} "
                    f"has {crop_tiles} crop tiles and {tilled} open tilled tiles; skip seeding partial plot"
                )
            if self._is_establish_only:
                # Plant pass leaves watering for the later can pass.
                print(
                    f"[CROP] Plot {self._plot_index + 1}/{len(self._plots)} "
                    f"already established; establish-only skips water"
                )
                self._advance_plot(ram)
            else:
                self._begin_water_phase(ram, allow_unknown_tiles=False)

    def _advance_plot(self, ram: np.ndarray):
        """Move to the next plot, or trigger a re-scan pass, or finish."""
        self._clear_crop_walkable()
        self._plot_index += 1
        if self._plot_index >= len(self._plots):
            # Water: re-scan when tiles were skipped.
            # Establish: one retry pass so a rejected center can fall back to
            # another nearby tillable plot (rejected centers are remembered).
            can_retry_establish = (
                self._is_establish_only
                and self._pass_number < 2
                and self.planted_count == 0
                and bool(self._rejected_plan_centers)
            )
            can_retry_water = (
                not self._is_establish_only
                and self._pass_number < 3
                and self.skipped_water > 0
            )
            if can_retry_establish or can_retry_water:
                prev_skip = self.skipped_water
                self._pass_number += 1
                self._state = CropState.DETECT
                self._pathfinder.temp_blocked.clear()
                self._refill_exhausted = False
                if can_retry_establish:
                    print(
                        f"[CROP] Establish pass {self._pass_number - 1} planted=0; "
                        f"retry with rejected={sorted(self._rejected_plan_centers)}"
                    )
                else:
                    print(
                        f"[CROP] Pass {self._pass_number - 1} complete ({prev_skip} skipped), "
                        f"starting pass {self._pass_number}..."
                    )
            else:
                self._state = CropState.DONE
        else:
            self._start_plot(ram)

