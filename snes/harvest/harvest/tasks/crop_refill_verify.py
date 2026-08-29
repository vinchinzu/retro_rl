"""Refill thrash / path recover / corridor-decision arms for CropRefillMixin.

Mechanical extract from ``crop_refill.py`` (nav densify thrash, soft recover,
charge-completion dispatch). Methods keep ``self`` access to task fields.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from retro_harness import TaskResult

from harvest.tasks.crop_planter import CropState, PlotPhase
from harvest.tasks.nav import VIEWPORT_HOP_TILES, tile_dist
from harvest.tasks.pond_hop import (
    KIND_ACT_AT_STAND,
    KIND_ARM_F0_AND_LIP,
    KIND_COMMIT_MULTIHOP_MAYBE_ACT_OR_REFILL,
    KIND_COMMIT_MULTIHOP_OR_REFILL,
    KIND_QUEUE_EAST_SOUTH,
    KIND_QUEUE_GAP_SOUTH,
    KIND_QUEUE_WEST_SOUTH_LIP,
    KIND_TRY_MULTIHOP_CONTINUE,
    KIND_TRY_MULTIHOP_MAYBE_ACT_CONTINUE,
    PRIMARY_POND_FACE,
    PRIMARY_POND_STAND,
    CorridorNavDecision,
    ThrashChargeKind,
    ThrashCounters,
    decide_after_south_lip_charge,
    evaluate_corridor_thrash,
)
from harvest.tasks.crop_geometry import is_main_pond_stand


class CropRefillVerifyMixin:
    """Pathfind thrash, refill recover, and corridor charge-decision apply."""

    def _find_nav_path(
        self,
        ram: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> Optional[List[Tuple[int, int]]]:
        """Pathfind with viewport hop fallback for distant crop targets.

        Live farm tiles go stale outside the loaded viewport. Without
        ``max_steps``, ``find_path`` returns None for goals ~12+ tiles away
        and establish/water immediately skip every hoe stand. Hop toward the
        goal so multi-hop navigation can close the gap.
        """
        # Refill stands are often 15–25 tiles from west plots after fence open;
        # allow a longer hop so multi-hop can keep closing without false exhaust.
        # Densify intermediate goals so hop-toward does not thrash at ~(25,30).
        hop = VIEWPORT_HOP_TILES + 3 if self._plot_phase == PlotPhase.REFILL else VIEWPORT_HOP_TILES
        nav_goal = goal
        if self._plot_phase == PlotPhase.REFILL and getattr(self, "_refill_multihop", False):
            # Densify thrash → scripted charges. Rules live in pond_hop
            # (past-fence pure-south, N/S/E stall regions, near-F0 short lip).
            thrash = evaluate_corridor_thrash(
                start,
                goal,
                ThrashCounters(
                    east_south_charges=getattr(
                        self._corridor, "east_south_charges", 0
                    ),
                    south_lip_charges=getattr(
                        self._corridor, "south_lip_charges", 0
                    ),
                    refill_densify_stalls=getattr(
                        self._corridor, "refill_densify_stalls", 0
                    ),
                    refill_densify_last=getattr(
                        self._corridor, "refill_densify_last", None
                    ),
                ),
            )
            self._corridor.refill_densify_stalls = thrash.refill_densify_stalls
            self._corridor.refill_densify_last = thrash.refill_densify_last
            if thrash.fire_charge:
                if thrash.log:
                    print(thrash.log)
                if thrash.charge is ThrashChargeKind.WEST_SOUTH_LIP:
                    self._queue_west_south_lip_charge(start)
                else:
                    self._queue_east_south_corridor_charge(start)
                return None  # navigate drains charge queue
            nav_goal = self._refill_hop_goal(ram, start, goal)
            if nav_goal != goal:
                print(
                    f"[CROP] Refill densify hop {start} → {nav_goal} "
                    f"(ultimate={goal})"
                )
        path = self._pathfinder.find_path(
            ram,
            start,
            nav_goal,
            max_steps=hop,
        )
        # Reject regressive truncated hops (long west-around routes).
        if (
            path
            and self._plot_phase == PlotPhase.REFILL
            and getattr(self, "_refill_multihop", False)
        ):
            end = path[-1]
            if tile_dist(end, goal) >= tile_dist(start, goal) and end != goal:
                # Try an explicit densify target once more.
                alt = self._refill_hop_goal(ram, start, goal)
                if alt != nav_goal and alt != start:
                    alt_path = self._pathfinder.find_path(
                        ram, start, alt, max_steps=hop
                    )
                    if alt_path and tile_dist(alt_path[-1], goal) < tile_dist(
                        start, goal
                    ):
                        print(
                            f"[CROP] Refill reject regressive hop end={end}; "
                            f"using densify {alt}"
                        )
                        return alt_path
                return None
        return path

    def _recover_refill_nav(self, ram: np.ndarray, *, reason: str) -> None:
        """Recover from mid-refill path loss without hard-exhausting once.

        Multi-hop pond approaches often lose full BFS mid-route when the
        viewport rolls; blacklisting the stand + reselecting (or snapping when
        already adjacent to a corridor stand) is enough to continue. Only after
        several soft fails do we mark refill exhausted.
        """
        player = self._navigator.current_tile
        try:
            from harvest.maps.map_config import farm_pond_refill_stands
            stands = farm_pond_refill_stands()
        except Exception:
            stands = (((32, 34), "up"), ((33, 30), "down"))

        for stand, face in stands:
            if tile_dist(player, stand) <= 1:
                print(
                    f"[CROP] Refill snap to nearby stand {stand} face={face} "
                    f"({reason})"
                )
                self._refill_pond_tile = stand
                self._refill_pond_face = face
                self._target_tile = stand
                self._approach_tile = stand
                self._face_direction = face
                self._state = CropState.ACT
                self._navigator.path = []
                self._steps_on_target = 0
                self._refill_multihop = False
                return

        # Track multi-hop progress: if we closed distance, soft-retry without
        # blacklisting the ultimate F0 stand (viewport hop thrash otherwise).
        ultimate = self._refill_pond_tile or self._approach_tile
        if ultimate is not None:
            cur_dist = tile_dist(player, ultimate)
            best = getattr(self, "_refill_best_dist", 999)
            if cur_dist < best:
                self._refill_best_dist = cur_dist
                print(
                    f"[CROP] Refill multi-hop progress dist {best}→{cur_dist} "
                    f"at {player} ({reason}); repath"
                )
                self._state = CropState.NAVIGATE
                self._navigator.path = []
                self._navigator.stasis = 0
                self._pathfinder.temp_blocked.clear()
                return

        # Multi-hop densify soft retry before burning a failure slot.
        if getattr(self, "_refill_multihop", False) and ultimate is not None:
            hop_goal = self._refill_hop_goal(ram, player, ultimate)
            if hop_goal != ultimate and hop_goal != player:
                path = self._pathfinder.find_path(
                    ram,
                    player,
                    hop_goal,
                    max_steps=VIEWPORT_HOP_TILES + 3,
                )
                if path:
                    print(
                        f"[CROP] Refill densify recover → {hop_goal} "
                        f"from {player} ({reason})"
                    )
                    self._navigator.path = path
                    self._navigator.stasis = 0
                    self._state = CropState.NAVIGATE
                    return

        # Only blacklist non-progress failures; keep main pond stands longer.
        if self._refill_pond_tile is not None and not is_main_pond_stand(
            self._refill_pond_tile
        ):
            self._bad_refill_tiles.add(self._refill_pond_tile)
        elif self._refill_pond_tile is not None:
            # Main pond: only blacklist after repeated no-progress stalls.
            fails = getattr(self, "_refill_nav_failures", 0)
            if fails >= 3:
                self._bad_refill_tiles.add(self._refill_pond_tile)

        self._refill_nav_failures = getattr(self, "_refill_nav_failures", 0) + 1
        if self._refill_nav_failures >= 8:
            print(
                f"[CROP] Refill nav failed {self._refill_nav_failures}x "
                f"({reason}); exhausting"
            )
            self._refill_exhausted = True
            self._refill_multihop = False
            self._plot_phase = PlotPhase.WATER
            self._set_water_walkable()
            if self._water_index < len(self._water_steps):
                target, stand, face = self._water_steps[self._water_index]
                self._target_tile = target
                self._approach_tile = stand
                self._face_direction = face
            elif self._plots and self._plot_index < len(self._plots):
                center = self._plots[self._plot_index]
                self._target_tile = center
                self._approach_tile = center
            self._state = CropState.NAVIGATE
            self._navigator.path = []
            return

        print(
            f"[CROP] Refill repath {self._refill_nav_failures}/8 after {reason} "
            f"(stand={self._refill_pond_tile} pos={player})"
        )
        # Prefer re-commit multi-hop to nearest remaining stand over full search
        # which may re-enter fence-open when wall residue remains.
        if self._pond_corridor_gap_open(ram) or self._fence_open_attempts > 0:
            if self._commit_multihop_main_pond(ram, self._water_level(ram)):
                return
        self._start_refill(ram)

    def _apply_corridor_nav_decision(
        self,
        ram: np.ndarray,
        player: Tuple[int, int],
        decision: CorridorNavDecision,
        *,
        north_band_multihop_tried: bool = False,
        near_f0_multihop_tried: bool = False,
    ) -> Optional[TaskResult]:
        """Apply a pond_hop charge-completion decision (navigate thrash).

        Decision body lives in ``pond_hop.decide_after_*``.
        """
        if decision.log:
            for line in decision.log.split("\n"):
                if line:
                    print(line)
        if decision.set_east_south_stuck_at is not None:
            self._corridor.east_south_stuck_at = decision.set_east_south_stuck_at

        kind = decision.kind
        if kind == KIND_QUEUE_EAST_SOUTH:
            self._queue_east_south_corridor_charge(player)
            return None
        if kind == KIND_QUEUE_GAP_SOUTH:
            self._queue_gap_south_fallback(player)
            return None
        if kind == KIND_QUEUE_WEST_SOUTH_LIP:
            self._queue_west_south_lip_charge(player)
            return None
        if kind == KIND_ARM_F0_AND_LIP:
            stand = decision.stand or PRIMARY_POND_STAND
            face = decision.face or PRIMARY_POND_FACE
            self._queue_west_south_lip_charge(player)
            self._refill_pond_tile = stand
            self._refill_pond_face = face
            self._approach_tile = stand
            self._target_tile = stand
            self._face_direction = face
            self._refill_multihop = True
            self._plot_phase = PlotPhase.REFILL
            return None
        if kind == KIND_ACT_AT_STAND:
            stand = decision.stand or PRIMARY_POND_STAND
            face = decision.face or PRIMARY_POND_FACE
            self._refill_pond_tile = stand
            self._refill_pond_face = face
            self._approach_tile = stand
            self._target_tile = stand
            self._face_direction = face
            self._plot_phase = PlotPhase.REFILL
            self._state = CropState.ACT
            self._navigator.path = []
            return None
        if kind == KIND_TRY_MULTIHOP_CONTINUE:
            if self._commit_multihop_main_pond(ram, self._water_level(ram)):
                return None
            nxt = decide_after_south_lip_charge(
                player,
                south_lip_charges=getattr(self._corridor, "south_lip_charges", 0),
                east_south_charges=getattr(self._corridor, "east_south_charges", 0),
                north_band_multihop_tried=True,
                near_f0_multihop_tried=near_f0_multihop_tried,
            )
            # Avoid re-printing the header line already emitted.
            nxt = CorridorNavDecision(
                kind=nxt.kind,
                log="\n".join(
                    ln
                    for ln in nxt.log.split("\n")
                    if ln and not ln.startswith("[CROP] South-lip charge done")
                ),
                stand=nxt.stand,
                face=nxt.face,
                set_east_south_stuck_at=nxt.set_east_south_stuck_at,
            )
            return self._apply_corridor_nav_decision(
                ram,
                player,
                nxt,
                north_band_multihop_tried=True,
                near_f0_multihop_tried=near_f0_multihop_tried,
            )
        if kind == KIND_TRY_MULTIHOP_MAYBE_ACT_CONTINUE:
            if self._commit_multihop_main_pond(ram, self._water_level(ram)):
                if (
                    self._refill_pond_tile is not None
                    and tile_dist(player, self._refill_pond_tile) <= 1
                ):
                    self._approach_tile = self._refill_pond_tile
                    self._target_tile = self._refill_pond_tile
                    self._state = CropState.ACT
                    self._navigator.path = []
                return None
            nxt = decide_after_south_lip_charge(
                player,
                south_lip_charges=getattr(self._corridor, "south_lip_charges", 0),
                east_south_charges=getattr(self._corridor, "east_south_charges", 0),
                north_band_multihop_tried=True,
                near_f0_multihop_tried=True,
            )
            nxt = CorridorNavDecision(
                kind=nxt.kind,
                log="\n".join(
                    ln
                    for ln in nxt.log.split("\n")
                    if ln and not ln.startswith("[CROP] South-lip charge done")
                ),
                stand=nxt.stand,
                face=nxt.face,
                set_east_south_stuck_at=nxt.set_east_south_stuck_at,
            )
            return self._apply_corridor_nav_decision(
                ram,
                player,
                nxt,
                north_band_multihop_tried=True,
                near_f0_multihop_tried=True,
            )
        if kind == KIND_COMMIT_MULTIHOP_OR_REFILL:
            if self._commit_multihop_main_pond(ram, self._water_level(ram)):
                return None
            self._plot_phase = PlotPhase.WATER
            self._start_refill(ram)
            return None
        if kind == KIND_COMMIT_MULTIHOP_MAYBE_ACT_OR_REFILL:
            if self._commit_multihop_main_pond(ram, self._water_level(ram)):
                if (
                    self._refill_pond_tile is not None
                    and tile_dist(player, self._refill_pond_tile) <= 1
                ):
                    self._approach_tile = self._refill_pond_tile
                    self._target_tile = self._refill_pond_tile
                    self._state = CropState.ACT
                    self._navigator.path = []
                return None
            self._plot_phase = PlotPhase.WATER
            self._start_refill(ram)
            return None
        # Unknown kind — fail safe to refill restart.
        self._plot_phase = PlotPhase.WATER
        self._start_refill(ram)
        return None
