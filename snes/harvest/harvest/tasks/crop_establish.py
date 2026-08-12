"""Hoe / plant phase helpers for CropWaterTask (rr-ds3 extract).

Establish-only and full-mode plant ceremony arms.
"""

from __future__ import annotations

from harvest.tasks.crop_fsm import CropState, PlotPhase

from typing import List, Optional, Tuple

import numpy as np

from retro_harness import TaskResult

from harvest.core.carry import SEED_ITEM
from harvest.core.tile_catalog import Tool
from harvest.tasks.farm_clearer import use_tool
from harvest.tasks.crop_geometry import (
    DEFAULT_CROP_BOUNDS,
    DRIED_TILLED,
    FRESH_TILLED,
    TILLABLE_TILES,
    UNTILLED,
    WATERED_TILLED,
    _merge_plot_centers,
    count_tilled,
    detect_crop_resume_plots,
    detect_plots,
    hoe_action_sequence,
    hoe_plan,
    is_crop_tile,
    is_dry_crop_tile,
)
from harvest.tasks.nav import WALKABLE_TILES, get_tile_at, make_action, tile_dist


class CropEstablishMixin:
    """Hoe and plant phase methods for CropWaterTask."""

    def _begin_hoe_phase(self, ram: np.ndarray) -> None:
        """Hoe untilled ring tiles for the current planned plot center."""
        center = self._plots[self._plot_index]
        cx, cy = center
        self._plot_phase = PlotPhase.HOE
        self._water_steps = []
        self._water_index = 0
        for target, stand, face in hoe_plan(center):
            tid = get_tile_at(ram, target[0], target[1])
            if tid in TILLABLE_TILES or tid == DRIED_TILLED or tid == UNTILLED:
                self._water_steps.append((target, stand, face))
            elif tid not in {FRESH_TILLED, WATERED_TILLED}:
                # Unknown/blocked — still try if soil-like low IDs.
                if tid in {0x00, 0x01, 0x02}:
                    self._water_steps.append((target, stand, face))
        print(
            f"[CROP] Plot {self._plot_index + 1}/{len(self._plots)} "
            f"center=({cx},{cy}) phase=HOE steps={len(self._water_steps)}"
        )
        if not self._water_steps:
            # Nothing to hoe; try plant or water.
            tilled = count_tilled(ram, center)
            if tilled >= 4:
                self._plot_phase = PlotPhase.PLANT
                self._target_tile = center
                self._approach_tile = center
                self._state = CropState.NAVIGATE
                self._navigator.path = []
                self._steps_on_target = 0
                print(f"[CROP] HOE skipped; planting with tilled={tilled}")
                return
            print(f"[CROP] HOE found no tillable tiles at ({cx},{cy}); skipping plot")
            self._rejected_plan_centers.add(center)
            self._advance_plot(ram)
            return
        target, stand, face = self._water_steps[0]
        self._target_tile = target
        self._approach_tile = stand
        self._face_direction = face
        self._clear_crop_walkable()
        self._state = CropState.NAVIGATE
        self._navigator.path = []
        self._steps_on_target = 0

    def _advance_hoe_step(self, ram: np.ndarray) -> None:
        self._water_index += 1
        self._steps_on_target = 0
        self._navigator.path = []
        if self._water_index >= len(self._water_steps):
            center = self._plots[self._plot_index]
            tilled = count_tilled(ram, center)
            print(f"[CROP] HOE complete plot {self._plot_index + 1} tilled={tilled}")
            if tilled < 2:
                # No reachable till work — reject this planned center and move on.
                self._rejected_plan_centers.add(center)
                print(f"[CROP] Rejecting planned center {center} after failed hoe")
                self._advance_plot(ram)
                return
            if tilled < 4:
                print(
                    f"[CROP] Partial hoe tilled={tilled}; still attempting plant "
                    f"(seed bag covers tilled tiles)"
                )
            self._plot_phase = PlotPhase.PLANT
            self._target_tile = center
            self._approach_tile = center
            self._face_direction = "down"
            self._state = CropState.NAVIGATE
            self._navigator.path = []
            return
        target, stand, face = self._water_steps[self._water_index]
        self._target_tile = target
        self._approach_tile = stand
        self._face_direction = face
        self._state = CropState.NAVIGATE

    def _act_hoe(self, ram: np.ndarray) -> Optional[TaskResult]:
        """Hoe one untilled ring tile for the current planned plot."""
        if self._tool_mgr.current != int(Tool.HOE):
            self._tool_mgr.start_search()
            self._state = CropState.TOOL_SWITCH
            return None
        face = self._face_direction or "down"
        target = self._target_tile
        tid = get_tile_at(ram, target[0], target[1]) if target else 0xFF
        print(
            f"[CROP] HOE tile {self._water_index + 1}/{len(self._water_steps)} "
            f"target={target} face={face} tid=0x{tid:02X}"
        )
        self._action_queue.extend(hoe_action_sequence(face))
        self._state = CropState.VERIFY
        return None

    def _act_plant(self, ram: np.ndarray) -> Optional[TaskResult]:
        """Plant seeds at current plot center."""
        seed_item = SEED_ITEM.get(self.seed_type, SEED_ITEM["potato"])
        if self._tool_mgr.current != seed_item:
            self._tool_mgr.start_search()
            self._state = CropState.TOOL_SWITCH
            return None

        center = self._plots[self._plot_index]
        player = self._navigator.current_tile
        # Debug: dump 3x3 tile IDs around center
        cx, cy = center
        tile_ids = []
        for dy in range(-1, 2):
            row = []
            for dx in range(-1, 2):
                tid = get_tile_at(ram, cx + dx, cy + dy)
                row.append(f"0x{tid:02X}")
            tile_ids.append(" ".join(row))
        print(f"[CROP] PLANT at ({cx},{cy}) player=({player[0]},{player[1]}) seed=0x{seed_item:02X}")
        print(f"[CROP]   3x3 tiles: [{tile_ids[0]}] [{tile_ids[1]}] [{tile_ids[2]}]")

        # Face → settle → Y → long cooldown.  Plant animation takes ~150f
        # so use 90f cooldown to ensure tile data updates before verify.
        self._action_queue.extend([make_action(down=True) for _ in range(4)])  # face down
        self._action_queue.extend([make_action() for _ in range(6)])           # settle
        self._action_queue.extend(use_tool(frames=20, cooldown=90))            # Y + long cooldown
        self._state = CropState.VERIFY
        return None

    def _plan_new_plot_centers(self, ram: np.ndarray) -> List[Tuple[int, int]]:
        """Use crop_planner to place new 3x3 plots on tillable soil.

        Prefer the early-spring field anchor (crop_planner.DEFAULT_START_TILE)
        so we do not plant near shipping/south stream when NAV lands there.
        Fall back to player-local then full-farm bounds.
        """
        try:
            from harvest.planner.crop_planner import (
                DEFAULT_START_TILE,
                CropPlanningConfig,
                plan_crop_field,
            )
            from harvest.planner.day_plan_status import read_world_date
        except Exception as exc:
            print(f"[CROP] Crop planner unavailable: {exc}")
            return []

        season, day = read_world_date(ram)
        start = self._navigator.current_tile
        preferred = DEFAULT_START_TILE
        # Prefer player-local first so BFS can reach hoe stands after NAV_CROP.
        # Preferred-field / full-farm plans often pick east of the x=32 fence
        # (e.g. 35,27) which is unreachable from the early-spring west pocket.
        attempts: List[Tuple[str, Tuple[int, int, int, int], int]] = [
            ("player_local", self._plan_bounds_near_player(start), 1),
            ("preferred_field", self._plan_bounds_around(preferred, radius=14), 1),
            ("full_farm", self.bounds, 1),
        ]
        plan = None
        centers: List[Tuple[int, int]] = []
        used_label = ""
        used_bounds = attempts[0][1]
        for label, bounds, max_bags in attempts:
            config = CropPlanningConfig(
                season=int(season),
                day=int(day),
                seed_type=self.seed_type,
                max_seed_bags=max_bags,
                bounds=bounds,
                start_tile=start,
                # Strongly prefer nearby plots over slightly higher remote scores.
                route_weight=40,
            )
            plan = plan_crop_field(ram, config)
            centers = [
                plot.center
                for plot in plan.plots
                if plot.center not in self._rejected_plan_centers
            ][:1]
            if centers:
                used_label = label
                used_bounds = bounds
                break
        # Planner access checks are strict (watering stands) and full-farm
        # scores often pick east/south of the early-spring fence pocket
        # (unreachable via viewport BFS). Prefer a nearby tillable 3x3 the hoe
        # can actually reach.
        fallback = self._fallback_local_till_center(ram, start)
        if fallback is not None:
            if not centers:
                print(
                    f"[CROP] Planner empty; fallback till center {fallback} "
                    f"near player {start}"
                )
                return [fallback]
            planned = centers[0]
            planned_dist = abs(planned[0] - start[0]) + abs(planned[1] - start[1])
            fallback_dist = abs(fallback[0] - start[0]) + abs(fallback[1] - start[1])
            if planned_dist > 12 and fallback_dist + 4 < planned_dist:
                print(
                    f"[CROP] Prefer fallback till {fallback} (dist={fallback_dist}) "
                    f"over planner {planned} (dist={planned_dist}, zone={used_label})"
                )
                return [fallback]
        elif centers:
            # No nearby till fallback: drop unreachable remote planner centers.
            planned = centers[0]
            planned_dist = abs(planned[0] - start[0]) + abs(planned[1] - start[1])
            if planned_dist > 12:
                print(
                    f"[CROP] Drop remote planner center {planned} "
                    f"(dist={planned_dist}); no local fallback"
                )
                centers = []
        if centers and plan is not None:
            print(
                f"[CROP] Planned {len(centers)} new {plan.crop_name} plot(s) "
                f"layout={plan.layout_name} zone={used_label} bounds={used_bounds}: "
                f"{centers}"
            )
        else:
            print("[CROP] Crop planner found no placeable plots")
        return centers

    def _fallback_local_till_center(
        self,
        ram: np.ndarray,
        start: Tuple[int, int],
    ) -> Optional[Tuple[int, int]]:
        """Pick a nearby 3x3 of untilled soil when the formal planner finds none.

        Early spring west pocket has open dirt the planner rejects (missing
        watering-access stands). Hoe+plant still works if we stand on the
        center notch. Only accept centers reachable via a short hop path so we
        do not plant south/east of the live-map fence pocket.
        """
        px, py = start
        best: Optional[Tuple[int, int]] = None
        best_key: Optional[Tuple[int, int, int]] = None
        for cy in range(max(2, py - 8), min(62, py + 9)):
            for cx in range(max(2, px - 8), min(62, px + 9)):
                center = (cx, cy)
                if center in self._rejected_plan_centers:
                    continue
                tillable = 0
                hard_block = 0
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        tx, ty = cx + dx, cy + dy
                        tid = get_tile_at(ram, tx, ty)
                        if tid in TILLABLE_TILES or tid in {
                            0x00,
                            0x01,
                            0x02,
                            FRESH_TILLED,
                            WATERED_TILLED,
                        }:
                            tillable += 1
                        elif tid in WALKABLE_TILES:
                            # path tile inside plot — can still hoe around
                            tillable += 1
                        else:
                            hard_block += 1
                # Allow a rock/debris in the notch (seen at 12,25) if enough soil.
                if tillable < 6 or hard_block > 2:
                    continue
                # Prefer centers we can path to, or at least path to a hoe stand.
                stand_ok = False
                for _target, stand, _face in hoe_plan(center):
                    if stand == start:
                        stand_ok = True
                        break
                    stand_path = self._pathfinder.find_path(
                        ram, start, stand, max_steps=12
                    )
                    if stand_path and stand_path[-1] == stand:
                        stand_ok = True
                        break
                if not stand_ok:
                    # Center path is enough when stands fail only due to hop cap.
                    if start != center:
                        path = self._pathfinder.find_path(
                            ram, start, center, max_steps=12
                        )
                        if not path or path[-1] != center:
                            continue
                dist = abs(cx - px) + abs(cy - py)
                key = (dist, -tillable, cy, cx)
                if best_key is None or key < best_key:
                    best_key = key
                    best = center
        return best

