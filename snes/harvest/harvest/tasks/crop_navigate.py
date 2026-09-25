"""Navigate / stuck-recovery functions for CropWaterTask.

Multi-phase path follow, stasis recovery, and the refill-phase dispatcher.
Not a mixin. CropWaterTask calls these when a corridor refill or a pre-armed
navigate state is active.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from retro_harness import ActionResult, TaskResult, TaskStatus

from harvest.core.carry import SEED_ITEM, seed_in_carry_pair as seed_item_in_carry_pair
from harvest.core.tile_catalog import ADDR_INPUT_LOCK
from harvest.tasks.crop_planter import CropState, POND_ACCESS_PHASES, PlotPhase
from harvest.tasks.crop_geometry import (
    count_tilled,
    is_main_pond_stand,
    is_rainy_weather,
    pond_access_blocking_fences,
)
from harvest.tasks.nav import make_action, tile_dist
from harvest.tasks.pond_hop import (
    decide_after_east_south_charge,
    decide_after_gap_reseat,
    decide_after_multihop_drop,
    decide_after_south_lip_charge,
)



def _handle_navigate(self, ram: np.ndarray) -> Optional[TaskResult]:
    # Finish local drop before multi-hop after fence open.
    # Charge-completion policy: pond_hop.decide_after_*.
    if getattr(self, "_pending_multihop_after_drop", False):
        if self._player_carrying(ram):
            if not self._action_queue:
                self._queue_local_drop()
            return None  # step() drains queue
        self._pending_multihop_after_drop = False
        player = self._navigator.current_tile
        decision = decide_after_multihop_drop(
            player,
            south_lip_charges=getattr(self._corridor, "south_lip_charges", 0),
        )
        return self._apply_corridor_nav_decision(ram, player, decision)

    # Drain N/E nudge after gap drop, then multi-hop.
    if getattr(self, "_pending_gap_reseat", False):
        if self._action_queue:
            return None
        self._pending_gap_reseat = False
        player = self._navigator.current_tile
        decision = decide_after_gap_reseat(player)
        return self._apply_corridor_nav_decision(ram, player, decision)

    # East→south corridor charge (or legacy gap-charge) completion.
    if getattr(self._corridor, "pending_gap_charge", False):
        if self._action_queue:
            return None
        self._corridor.pending_gap_charge = False
        player = self._navigator.current_tile
        decision = decide_after_east_south_charge(
            player,
            east_south_charges=getattr(self._corridor, "east_south_charges", 0),
            south_lip_charges=getattr(self._corridor, "south_lip_charges", 0),
            east_south_stuck_at=getattr(self._corridor, "east_south_stuck_at", None),
            gap_south_tried=getattr(self._corridor, "gap_south_tried", False),
        )
        return self._apply_corridor_nav_decision(ram, player, decision)

    # West→south-lip charge completion (after (28,32) soft-block / south thrash).
    if getattr(self._corridor, "pending_south_lip_charge", False):
        if self._action_queue:
            return None
        self._corridor.pending_south_lip_charge = False
        player = self._navigator.current_tile
        decision = decide_after_south_lip_charge(
            player,
            south_lip_charges=getattr(self._corridor, "south_lip_charges", 0),
            east_south_charges=getattr(self._corridor, "east_south_charges", 0),
        )
        return self._apply_corridor_nav_decision(ram, player, decision)

    # Mid-refill: if we somehow started carrying, drop before walking.
    if (
        self._plot_phase == PlotPhase.REFILL
        and self._player_carrying(ram)
        and not self._action_queue
    ):
        self._queue_local_drop()
        return None

    if self._target_tile is None or self._approach_tile is None:
        self._state = CropState.DETECT
        return None

    if self._plot_phase == PlotPhase.WATER:
        if not self._navigator.path and self._retarget_current_water_step(ram):
            self._navigator.path = []
            self._navigator.stasis = 0
        self._set_water_walkable()

    # Arrived?
    if self._navigator.current_tile == self._approach_tile:
        self._state = CropState.CENTER
        return None
    # Refill multi-hop: adjacent to stand is enough to act (center tolerance).
    if (
        self._plot_phase == PlotPhase.REFILL
        and self._approach_tile is not None
        and tile_dist(self._navigator.current_tile, self._approach_tile) <= 1
    ):
        self._state = CropState.CENTER
        return None

    # Track multi-hop progress toward the ultimate refill stand.
    if self._plot_phase == PlotPhase.REFILL and self._approach_tile is not None:
        d = tile_dist(self._navigator.current_tile, self._approach_tile)
        if d < getattr(self, "_refill_best_dist", 999):
            self._refill_best_dist = d
            # Progress resets the per-target timeout so multi-hop F0 can
            # cross ~20 tiles without false "Refill timed out".
            self._steps_on_target = 0
            self._pathfinder.temp_blocked.clear()

    # Stuck recovery
    if self._navigator.stasis > self.stasis_repath and self._navigator.path:
        # Refill multi-hop: do not permanently block the next east cell —
        # stasis on the north lip often means animation lock, not a wall.
        # Water already in plant pocket (y≤30, ≤2 from stand): never block
        # the notch/stand — that was the residual (12,26) failure mode
        # (temp_blocked the only approach cell).
        player_now = self._navigator.current_tile
        water_near_stand = (
            self._plot_phase == PlotPhase.WATER
            and self._approach_tile is not None
            and player_now[1] <= 30
            and tile_dist(player_now, self._approach_tile) <= 2
        )
        if (
            (
                self._plot_phase == PlotPhase.REFILL
                and getattr(self, "_refill_multihop", False)
            )
            or water_near_stand
        ):
            self._pathfinder.temp_blocked.clear()
        else:
            self._pathfinder.temp_blocked.add(self._navigator.path[0])
        path = self._find_nav_path(ram, self._navigator.current_tile, self._approach_tile)
        if path:
            self._navigator.path = path
            self._navigator.stasis = 0
        else:
            self._failures += 1
            self._failed_tiles.add(self._target_tile)
            if self._plot_phase == PlotPhase.WATER:
                # ROM: post-F0 return north through y=31 gap soft-blocks
                # pure-up from ~(13,33). Nudge right then up (lands y≤30).
                player = self._navigator.current_tile
                if (
                    player[1] >= 32
                    and self._approach_tile is not None
                    and self._approach_tile[1] <= 30
                    and not self._action_queue
                    and getattr(self, "_water_north_returns", 0) < 2
                ):
                    self._water_north_returns = (
                        getattr(self, "_water_north_returns", 0) + 1
                    )
                    self._action_queue.extend(
                        [make_action(right=True, b=True) for _ in range(40)]
                    )
                    self._action_queue.extend(
                        [make_action(up=True, b=True) for _ in range(160)]
                    )
                    self._action_queue.extend([make_action() for _ in range(8)])
                    self._navigator.stasis = 0
                    self._navigator.path = []
                    print(
                        f"[CROP] Water return north charge from {player} "
                        f"(n={self._water_north_returns})"
                    )
                    return None
                # Near stand / target: snap face-water or repath without block.
                if (
                    self._target_tile is not None
                    and tile_dist(player, self._target_tile) == 1
                ):
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
                        f"[CROP] Near-target water act at {player} → "
                        f"{self._target_tile} face={face}"
                    )
                    self._state = CropState.ACT
                    self._navigator.path = []
                    self._navigator.stasis = 0
                    return None
                if water_near_stand:
                    # One more repath after clear; don't skip residual yet.
                    self._pathfinder.temp_blocked.clear()
                    path2 = self._find_nav_path(
                        ram, player, self._approach_tile
                    )
                    if path2 is not None:
                        self._navigator.path = path2
                        self._navigator.stasis = 0
                        return None
                    # Force act if already on approach.
                    if player == self._approach_tile:
                        self._state = CropState.ACT
                        self._navigator.path = []
                        return None
                if self._reprioritize_water_step(ram, reason="stuck nav"):
                    return None
                if self._try_residual_crop_walk_recovery(ram):
                    return None
                self.skipped_water += 1
                self._plot_skipped += 1
                print(f"[CROP] SKIP water tile {self._water_index + 1}/{len(self._water_steps)} (stuck nav) target={self._target_tile}")
                self._advance_water_step(ram)
            elif self._plot_phase == PlotPhase.HOE:
                print(
                    f"[CROP] SKIP hoe tile {self._water_index + 1}/{len(self._water_steps)} "
                    f"(stuck nav) target={self._target_tile}"
                )
                self._advance_hoe_step(ram)
            elif self._plot_phase == PlotPhase.PLANT:
                center = self._plots[self._plot_index] if self._plot_index < len(self._plots) else None
                tilled = count_tilled(ram, center) if center else 0
                if center is not None and tilled >= 4:
                    # Close enough: attempt plant even if exact center nav struggled.
                    print(f"[CROP] Plant nav stuck at center {center}; forcing plant attempt tilled={tilled}")
                    self._state = CropState.ACT
                else:
                    if center is not None:
                        self._rejected_plan_centers.add(center)
                    print(f"[CROP] Plant nav stuck; skipping plot {center}")
                    self._advance_plot(ram)
            elif self._plot_phase == PlotPhase.REFILL:
                self._recover_refill_nav(ram, reason="stuck nav")
            elif self._plot_phase == PlotPhase.STAGE_POND:
                print("[CROP] Stage pond stuck; trying fence open from here")
                fences = pond_access_blocking_fences(ram)
                if fences and self._try_open_pond_access(ram, fences, skip_stage=True):
                    return None
                self._plot_phase = PlotPhase.WATER
                self._start_refill(ram)
            else:
                self._state = CropState.DETECT
            if self._failures >= self.max_failures:
                return TaskResult(status=TaskStatus.FAILURE, reason="too many nav failures")
            return None

    # Try to path if no current path
    if not self._navigator.path:
        path = self._find_nav_path(ram, self._navigator.current_tile, self._approach_tile)
        if path:
            self._navigator.path = path
            self._navigator.stasis = 0
        else:
            self._failures += 1
            if self._plot_phase == PlotPhase.WATER:
                if self._reprioritize_water_step(ram, reason="no path"):
                    return None
                if self._try_residual_crop_walk_recovery(ram):
                    return None
                self.skipped_water += 1
                self._plot_skipped += 1
                print(f"[CROP] SKIP water tile {self._water_index + 1}/{len(self._water_steps)} (no path) target={self._target_tile}")
                self._advance_water_step(ram)
            elif self._plot_phase == PlotPhase.HOE:
                print(
                    f"[CROP] SKIP hoe tile {self._water_index + 1}/{len(self._water_steps)} "
                    f"(no path) target={self._target_tile}"
                )
                self._advance_hoe_step(ram)
            elif self._plot_phase == PlotPhase.PLANT:
                center = self._plots[self._plot_index] if self._plot_index < len(self._plots) else None
                tilled = count_tilled(ram, center) if center else 0
                if center is not None and tilled >= 4 and tile_dist(self._navigator.current_tile, center) <= 1:
                    print(f"[CROP] Plant path missing but adjacent to {center}; forcing plant")
                    self._approach_tile = self._navigator.current_tile
                    self._state = CropState.ACT
                elif center is not None and tilled >= 4:
                    print(f"[CROP] No path to plant center {center}; retrying with crop walkable")
                    self._set_crop_walkable()
                    self._navigator.path = []
                    self._steps_on_target = 0
                else:
                    if center is not None:
                        self._rejected_plan_centers.add(center)
                    print(f"[CROP] No path to plant center {center}; skipping plot")
                    self._advance_plot(ram)
            elif self._plot_phase == PlotPhase.REFILL:
                self._recover_refill_nav(ram, reason="no path")
            elif self._plot_phase == PlotPhase.STAGE_POND:
                print("[CROP] No path to pond stage; trying fence open from here")
                fences = pond_access_blocking_fences(ram)
                if fences and self._try_open_pond_access(ram, fences, skip_stage=True):
                    return None
                self._plot_phase = PlotPhase.WATER
                self._start_refill(ram)
            else:
                self._state = CropState.DETECT
            return None

    action = self._navigator.follow_path(ram)
    if action is not None:
        if self._plot_phase == PlotPhase.WATER:
            action = action.copy()
            if (
                self._navigator.current_tile in self._current_plot_tiles()
                or tile_dist(self._navigator.current_tile, self._approach_tile) <= 1
            ):
                action[0] = 0  # slow down only once we're threading the plot edge
        self._action_queue.append(action)
    return None


def _dispatch_legacy(self, world: WorldState) -> TaskResult:
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
