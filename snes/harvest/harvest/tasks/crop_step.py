"""Main step loop + per-target timeout policy for CropWaterTask (rr-ds3)."""

from __future__ import annotations

from typing import Optional

import numpy as np

from retro_harness import ActionResult, TaskResult, TaskStatus, WorldState

from harvest.core.carry import SEED_ITEM, seed_in_carry_pair as seed_item_in_carry_pair
from harvest.core.tile_catalog import ADDR_INPUT_LOCK
from harvest.tasks.crop_fsm import CropState, PlotPhase, POND_ACCESS_PHASES
from harvest.tasks.crop_geometry import is_main_pond_stand, is_rainy_weather
from harvest.tasks.nav import make_action


class CropStepMixin:
    """``step`` implementation for CropWaterTask."""

    def step(self, world: WorldState) -> TaskResult:
        self._navigator.update(world.ram)
        self._tool_mgr.update(world.ram)
        self._total_steps += 1
        self._steps_on_target += 1

        if (
            self._total_steps == 1
            and is_rainy_weather(world.ram)
            and not self._is_water_only
            and not seed_item_in_carry_pair(world.ram, self.seed_type)
        ):
            wanted = SEED_ITEM.get(self.seed_type, SEED_ITEM["potato"])
            # Rain waters existing crops; without seeds there is no plant work either.
            # Still run detect in case established plots need nothing — but if no
            # seeds and rain, short-circuit so day plan can finish.
            # Water-only mode still scans (rain already watered; detect will no-op).
            print(f"[CROP] Rain and seed tool 0x{wanted:02X} not in carry pair; no crop work needed")
            self._snapshot_start_acceptance(world.ram)
            return self._terminal_result(rain=True)

        # Do not fail early when the watering can is out of the 2-slot carry pair.
        # Day plan often leaves seeds in-hand after ENSURE_CROP_SEEDS; we still
        # need to hoe/plant, then cycle to the can for watering.

        if self.debug and self._total_steps % self.debug_interval == 0:
            cur = self._navigator.current_tile
            print(f"[CROP] step={self._total_steps} phase={self._plot_phase} state={self._state} "
                  f"pos={cur} plot={self._plot_index}/{len(self._plots)} "
                  f"planted={self.planted_count} watered={self.watered_count} can={self._water_level(world.ram)}")

        # Timeout per target. Multi-hop refill gets a longer budget (corridor
        # from west pocket is 15–25 tiles + fence open overhead). Fence-open /
        # stage_pond own their own subtask budgets — do not abort them via
        # crop per-target timeout (that was resetting to detect mid-clear).
        if self._plot_phase in POND_ACCESS_PHASES:
            # Soft-cap fence thrash. Only early-bail when gap is open AND hands
            # are empty — otherwise we interrupt mid-carry before local_drop
            # (ROM: gap opens on lift, then 900f timeout left the bot stuck
            # carrying on the gap tile).
            carrying = self._player_carrying(world.ram)
            gap_open = self._pond_corridor_gap_open(world.ram)
            fence_budget = (
                900
                if gap_open and not carrying
                else max(self.max_steps_per_target * 3, 4000)
            )
            if self._steps_on_target > fence_budget:
                print(
                    f"[CROP] Fence/stage soft-timeout phase={self._plot_phase} "
                    f"budget={fence_budget}; forcing multi-hop or refill search"
                )
                self._fence_subtask = None
                self._steps_on_target = 0
                # Drop carried post first — multi-hop while carrying soft-locks
                # south-through-gap at the cleared fence tile.
                if self._ensure_hands_empty_for_refill(world.ram):
                    self._pending_multihop_after_drop = True
                    self._plot_phase = PlotPhase.REFILL
                    self._state = CropState.NAVIGATE
                    return TaskResult(
                        status=TaskStatus.RUNNING,
                        action=ActionResult(self._action_queue.popleft()),
                    )
                if self._pond_corridor_gap_open(world.ram) or self._fence_open_attempts > 0:
                    if self._commit_multihop_main_pond(
                        world.ram, self._water_level(world.ram)
                    ):
                        return TaskResult(
                            status=TaskStatus.RUNNING,
                            action=ActionResult(make_action()),
                        )
                self._plot_phase = PlotPhase.WATER
                self._start_refill(world.ram)
                return TaskResult(
                    status=TaskStatus.RUNNING,
                    action=ActionResult(make_action()),
                )
            # Fall through to normal step handling without target timeout.
            pass
        refill_budget = (
            max(self.max_steps_per_target * 3, 3600)
            if self._plot_phase == PlotPhase.REFILL
            else self.max_steps_per_target
        )
        if (
            self._plot_phase not in POND_ACCESS_PHASES
            and self._steps_on_target > refill_budget
            and self._target_tile is not None
        ):
            self._failed_tiles.add(self._target_tile)
            self._failures += 1
            self._action_queue.clear()
            if self._plot_phase == PlotPhase.WATER:
                if self._reprioritize_water_step(world.ram, reason="timeout"):
                    return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
                if self._try_residual_crop_walk_recovery(world.ram):
                    return TaskResult(
                        status=TaskStatus.RUNNING, action=ActionResult(make_action())
                    )
                self.skipped_water += 1
                self._plot_skipped += 1
                print(f"[CROP] SKIP water tile {self._water_index + 1}/{len(self._water_steps)} (timeout) target={self._target_tile}")
                self._advance_water_step(world.ram)
            elif self._plot_phase == PlotPhase.HOE:
                print(
                    f"[CROP] SKIP hoe tile {self._water_index + 1}/{len(self._water_steps)} "
                    f"(timeout) target={self._target_tile}"
                )
                self._advance_hoe_step(world.ram)
            elif self._plot_phase == PlotPhase.REFILL:
                player = self._navigator.current_tile
                print(
                    f"[CROP] Refill timed out at {player} "
                    f"stand={self._refill_pond_tile} best_dist="
                    f"{getattr(self, '_refill_best_dist', '?')}"
                )
                # Densify thrash: scripted charge before more multihop.
                pond = self._refill_pond_tile
                pond_ok = pond is None or (pond[0] >= 30 and pond[1] >= 30)
                if (
                    player[1] <= 31
                    and 18 <= player[0] <= 32
                    and pond_ok
                    and getattr(self._corridor, "east_south_charges", 0) < 6
                ):
                    self._queue_east_south_corridor_charge(player)
                    self._steps_on_target = 0
                    self._corridor.refill_densify_stalls = 0
                    return TaskResult(
                        status=TaskStatus.RUNNING,
                        action=ActionResult(make_action()),
                    )
                if (
                    player[1] >= 32
                    and player[0] <= 31
                    and pond_ok
                    and getattr(self._corridor, "south_lip_charges", 0) < 12
                ):
                    self._queue_west_south_lip_charge(player)
                    self._steps_on_target = 0
                    self._corridor.refill_densify_stalls = 0
                    return TaskResult(
                        status=TaskStatus.RUNNING,
                        action=ActionResult(make_action()),
                    )
                # Soft: try multi-hop re-commit once more before blacklisting.
                if (
                    getattr(self, "_refill_multihop", False)
                    and getattr(self, "_refill_nav_failures", 0) < 6
                    and (
                        self._pond_corridor_gap_open(world.ram)
                        or self._fence_open_attempts > 0
                    )
                ):
                    self._refill_nav_failures = getattr(self, "_refill_nav_failures", 0) + 1
                    self._steps_on_target = 0
                    if self._commit_multihop_main_pond(
                        world.ram, self._water_level(world.ram)
                    ):
                        return TaskResult(
                            status=TaskStatus.RUNNING,
                            action=ActionResult(make_action()),
                        )
                if self._refill_pond_tile and not is_main_pond_stand(
                    self._refill_pond_tile
                ):
                    self._bad_refill_tiles.add(self._refill_pond_tile)
                # Navigate back to current water step
                self._plot_phase = PlotPhase.WATER
                self._refill_multihop = False
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
            elif self._plot_phase == PlotPhase.PLANT:
                center = self._plots[self._plot_index] if self._plot_index < len(self._plots) else None
                if center is not None:
                    self._rejected_plan_centers.add(center)
                print(f"[CROP] Plant timeout at {center}; skipping plot")
                self._advance_plot(world.ram)
            else:
                self._target_tile = None
                self._state = CropState.DETECT
            if self._failures >= self.max_failures:
                return TaskResult(status=TaskStatus.FAILURE, reason="too many target timeouts")

        # Drain action queue
        if self._action_queue:
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(self._action_queue.popleft()))

        # Dialog dismissal
        input_lock = int(world.ram[ADDR_INPUT_LOCK]) if ADDR_INPUT_LOCK < len(world.ram) else 1
        if input_lock != 1:
            action = make_action(a=True) if self._total_steps % 2 == 0 else make_action()
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action), reason="dialog")

        # Check if all plots done
        if self._state == CropState.DONE:
            return self._terminal_result()

        # Outer CropState dispatch. FENCE_OPEN takes WorldState (subtask);
        # other arms take RAM only.
        if self._state == CropState.FENCE_OPEN:
            result = self._handle_fence_open(world)
            if result is not None:
                return result

        handlers = {
            CropState.DETECT: self._handle_detect,
            CropState.NAVIGATE: self._handle_navigate,
            CropState.CENTER: self._handle_center,
            CropState.ACT: self._handle_act,
            CropState.VERIFY: self._handle_verify,
            CropState.TOOL_SWITCH: self._handle_tool_switch,
        }

        handler = handlers.get(self._state)
        if handler:
            result = handler(world.ram)
            if result is not None:
                return result

        if self._action_queue:
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(self._action_queue.popleft()))

        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

