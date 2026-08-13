"""Pond approach arms for CropRefillMixin: multihop commits + fence open.

Mechanical extract from ``crop_refill.py`` so the public mixin stays under
the soft ~1000 LOC budget. Methods keep ``self`` access to task fields.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from retro_harness import TaskResult, TaskStatus, WorldState

from harvest.tasks.crop_fsm import CropState, PlotPhase
from harvest.tasks.nav import VIEWPORT_HOP_TILES, WALKABLE_TILES, get_tile_at, tile_dist
from harvest.tasks.water_refill import (
    REFILL_NONFILL_WATER_TILES,
    REFILL_PREFERRED_WATER_TILES,
    select_staging_stand,
)
from harvest.tasks.pond_corridor import PRIMARY_POND_FACE, PRIMARY_POND_STAND
from harvest.tasks.crop_geometry import (
    edge_water_tile_id,
    find_pond_edges,
    is_bad_refill_stand,
    is_main_pond_stand,
)


class CropRefillPondMixin:
    """Multihop stand commits and pond-access fence-open for refill."""

    def _commit_multihop_preferred_edge(
        self,
        ram: np.ndarray,
        current_lvl: int,
    ) -> bool:
        """Multi-hop to a preferred fill edge when exact path is viewport-false.

        Prioritize north-band F9 (and other preferred water north of y=25) so
        empty-can refill from the west plant pocket does not require the y=31
        fence corridor. Returns True if a stand was committed.
        """
        player = self._navigator.current_tile
        edges = find_pond_edges(
            ram,
            self.refill_bounds or self.bounds,
            water_tiles=REFILL_PREFERRED_WATER_TILES,
            exclude_bad_stands=True,
        )
        if self._bad_refill_tiles:
            edges = [(t, f) for t, f in edges if t not in self._bad_refill_tiles]
        edges = [(t, f) for t, f in edges if not is_bad_refill_stand(t)]
        edges = [
            (t, f)
            for t, f in edges
            if edge_water_tile_id(ram, t, f) in REFILL_PREFERRED_WATER_TILES
        ]
        if not edges:
            return False

        # Prefer F9/near-pocket north stands (x≤35, y≤20) before distant FA/FB.
        def sort_key(edge: Tuple[Tuple[int, int], str]) -> Tuple[int, int, int]:
            tile, face = edge
            wid = edge_water_tile_id(ram, tile, face)
            # 0=F9 near, 1=other north preferred, 2=south FC, 3=far east FA/FB
            if wid == 0xF9 or (tile[1] <= 20 and tile[0] <= 35):
                rank = 0
            elif tile[1] <= 25:
                rank = 1
            elif tile[1] >= 45:
                rank = 2
            else:
                rank = 3
            dist = abs(tile[0] - player[0]) + abs(tile[1] - player[1])
            return (rank, dist, wid if wid >= 0 else 999)

        edges = sorted(edges, key=sort_key)
        hop_budget = VIEWPORT_HOP_TILES + 5
        # ROM 2026-08-10: north F9 is sealed from the west plant pocket by the
        # y=13–14 fence bar. Manhattan-improving hops to ~(21,23) are false
        # positives — multihop thrash never reaches F9. Only commit when a full
        # path exists or a hop *nearly arrives* (end within 3 of stand).
        for tile, face in edges[:24]:
            if is_main_pond_stand(tile):
                continue
            wid = edge_water_tile_id(ram, tile, face)
            full = self._pathfinder.find_path(ram, player, tile)
            if full is not None:
                self._commit_refill_nav(
                    ram,
                    tile,
                    face,
                    current_lvl,
                    source="preferred_edge_multihop",
                    water_id=wid,
                    multihop=len(full) > VIEWPORT_HOP_TILES,
                )
                return True
            hop = self._pathfinder.find_path(
                ram, player, tile, max_steps=hop_budget
            )
            if hop is None:
                continue
            end = hop[-1] if hop else player
            # Require a hop that nearly arrives. Manhattan "improved" hops to
            # sealed F9/FA/FB islands thrash forever (dry-fixture false positive).
            nearly = end == tile or tile_dist(end, tile) <= 3
            if not nearly:
                continue
            print(
                f"[CROP] Multi-hop preferred edge ({tile[0]},{tile[1]}) "
                f"face={face} water=0x{wid:02X} end={end} nearly=True"
            )
            self._commit_refill_nav(
                ram,
                tile,
                face,
                current_lvl,
                source="preferred_edge_multihop",
                water_id=wid,
                multihop=True,
            )
            return True
        return False

    def _commit_multihop_main_pond(
        self,
        ram: np.ndarray,
        current_lvl: int,
    ) -> bool:
        """Commit to a main-pond F0 stand for multi-hop navigate after gap open.

        True full-path reachability is often false under live viewport even
        when the y=31 wall has a gap — partial hops + densified north-lip
        waypoints close the distance. Prefer north-lip stands when the player
        is still on y≤30 (post-fence-open stall ~tile 25,30).
        """
        player = self._navigator.current_tile
        try:
            from harvest.maps.map_config import farm_pond_refill_stands
            stands = list(farm_pond_refill_stands())
        except Exception:
            stands = [((32, 34), "up"), ((33, 30), "down"), ((32, 30), "down")]

        candidates = [
            (s, f)
            for s, f in stands
            if s not in self._bad_refill_tiles and not is_bad_refill_stand(s)
        ]
        if not candidates:
            candidates = [(s, f) for s, f in stands if not is_bad_refill_stand(s)]
        if not candidates:
            return False

        # Drop stands that are non-walkable on the live map (dry fixture often
        # has 0x05 fence residue on north-lip (32,30)/(34,30)).
        walkable_cands: List[Tuple[Tuple[int, int], str]] = []
        for s, f in candidates:
            tid = int(get_tile_at(ram, s[0], s[1]))
            if tid in WALKABLE_TILES or tid in (0xA0, 0xA1, 0xA8, 0x01, 0x02, 0x07):
                walkable_cands.append((s, f))
            elif s == player:
                walkable_cands.append((s, f))
        if walkable_cands:
            candidates = walkable_cands

        # After gap open: prefer south-lip stands (32/33,34). North-lip crawl
        # soft-blocks at ~(25,30) (can't walk north around 0xFF).
        south = [(s, f) for s, f in candidates if s[1] >= 33]
        if south:
            candidates = south

        candidates.sort(
            key=lambda sf: abs(sf[0][0] - player[0]) + abs(sf[0][1] - player[1])
        )
        stand, face = candidates[0]
        wid = edge_water_tile_id(ram, stand, face)
        # Stale viewport often returns 0 / dirt far from pond; still commit.
        if wid in REFILL_NONFILL_WATER_TILES:
            for alt_stand, alt_face in candidates[1:]:
                alt_wid = edge_water_tile_id(ram, alt_stand, alt_face)
                if alt_wid not in REFILL_NONFILL_WATER_TILES:
                    stand, face, wid = alt_stand, alt_face, alt_wid
                    break

        water_id = wid if wid in REFILL_PREFERRED_WATER_TILES else 0xF0
        self._commit_refill_nav(
            ram,
            stand,
            face,
            current_lvl,
            source="main_pond_multihop",
            water_id=water_id,
            multihop=True,
        )
        return True

    def _pond_access_staging_tiles(self) -> Tuple[Tuple[int, int], ...]:
        try:
            from harvest.maps.map_config import FARM_POND_ACCESS_STAGING_TILES
            return FARM_POND_ACCESS_STAGING_TILES
        except Exception:
            return (
                (11, 29),
                (12, 29),
                (10, 28),
                (11, 28),
                (15, 29),
                (18, 30),
                (20, 30),
            )

    def _try_stage_pond_access(self, ram: np.ndarray) -> bool:
        """Nav to a free stand north of the fence wall before clearing fences.

        ROM trap: after planting in the west pocket the bot often stands on
        (13,27) where pure-south movement soft-blocks (tile IDs still look
        walkable). Staging west/left first makes FenceClearLoopTask pathable.
        """
        if getattr(self, "_pond_staged", False):
            return False
        player = self._navigator.current_tile
        staging_tiles = self._pond_access_staging_tiles()

        def _hop(start: Tuple[int, int], goal: Tuple[int, int]):
            return self._find_nav_path(ram, start, goal)

        target = select_staging_stand(player, _hop, staging_tiles=staging_tiles)
        if target is None:
            return False
        if target.stand == player:
            self._pond_staged = True
            return False

        self._pond_staged = True
        self._pending_fence_open = True
        self._plot_phase = PlotPhase.STAGE_POND
        self._target_tile = target.stand
        self._approach_tile = target.stand
        self._face_direction = target.face
        self._state = CropState.NAVIGATE
        self._navigator.path = []
        self._steps_on_target = 0
        print(
            f"[CROP] Staging pond access via ({target.stand[0]},{target.stand[1]}) "
            f"from {player} (named corridor)"
        )
        return True

    def _try_open_pond_access(
        self,
        ram: np.ndarray,
        fences: List[Tuple[int, int]],
        *,
        skip_stage: bool = False,
    ) -> bool:
        """Start a limited fence-clear subtask to open the y=31 pond corridor.

        Returns True if a subtask/nav was started (caller should wait). Fence
        toss targets the main pond lip, so success both opens the path and
        parks the player at a fill stand.
        """
        if getattr(self, "_fence_open_attempts", 0) >= 2:
            return False
        if not fences:
            return False

        # Stage out of the plant pocket first — otherwise FenceClearLoopTask
        # plans a pure-south path that game physics never accepts.
        if not skip_stage and self._try_stage_pond_access(ram):
            return True

        try:
            from harvest.tasks.fence_flow import FenceClearLoopTask
        except Exception as exc:
            print(f"[CROP] Fence open unavailable: {exc}")
            return False

        self._fence_open_attempts = getattr(self, "_fence_open_attempts", 0) + 1
        self._pending_fence_open = False
        # Prefer fences nearest the player on the access row.
        player = self._navigator.current_tile
        fences_sorted = sorted(
            fences,
            key=lambda t: abs(t[0] - player[0]) + abs(t[1] - player[1]),
        )
        print(
            f"[CROP] Opening pond access: corridor_only clear 2 fences "
            f"(nearest={fences_sorted[0]}, wall n={len(fences)}, from={player})"
        )
        # corridor_only: local-drop (no pond toss thrash). Clear 2 adjacent
        # fences — single-tile gap soft-blocks south transit empty-handed;
        # two-wide gap is walkable on the dry fixture.
        task = FenceClearLoopTask(
            max_fences=2,
            max_steps_per_fence=1600,
            corridor_only=True,
        )
        # Lightweight world for reset
        from types import SimpleNamespace
        world = SimpleNamespace(ram=ram, info={}, obs=None)
        task.reset(world)
        self._fence_subtask = task
        self._plot_phase = PlotPhase.OPEN_POND
        self._state = CropState.FENCE_OPEN
        self._navigator.path = []
        # Fence subtask must not inherit stage/water steps_on_target budget.
        self._steps_on_target = 0
        return True

    def _handle_fence_open(self, world: WorldState) -> Optional[TaskResult]:
        """Drive FenceClearLoopTask until corridor opens, then resume refill."""
        task = self._fence_subtask
        if task is None:
            self._state = CropState.DETECT
            self._plot_phase = PlotPhase.WATER
            return None

        result = task.step(world)
        status = result.status
        if status == TaskStatus.RUNNING:
            return result
        if status == TaskStatus.SUCCESS:
            cleared = getattr(task, "cleared_count", 0)
            player = self._navigator.current_tile
            print(
                f"[CROP] Pond access open (cleared={cleared} fences) "
                f"at {player}; multi-hop F0"
            )
            self._fence_subtask = None
            # Carry-south success: already south of wall with/without post.
            if player[1] >= 32:
                if self._ensure_hands_empty_for_refill(world.ram):
                    self._pending_multihop_after_drop = True
                    self._plot_phase = PlotPhase.REFILL
                    self._state = CropState.NAVIGATE
                    return None
                # Far from F0 south of wall: scripted lip charge beats densify
                # thrash at ~(24,34)↔(24,35) (power-on residual rr-o00y).
                if (
                    tile_dist(player, PRIMARY_POND_STAND) > 3
                    and player[0] <= 28
                    and getattr(self._corridor, "south_lip_charges", 0) < 2
                ):
                    print(
                        f"[CROP] Fence open south at {player}; "
                        f"west→south-lip to F0 (skip densify)"
                    )
                    self._queue_west_south_lip_charge(player)
                    self._plot_phase = PlotPhase.REFILL
                    self._state = CropState.NAVIGATE
                    # Commit ultimate stand so act can fill after charge.
                    self._refill_pond_tile = PRIMARY_POND_STAND
                    self._refill_pond_face = PRIMARY_POND_FACE
                    self._approach_tile = PRIMARY_POND_STAND
                    self._target_tile = PRIMARY_POND_STAND
                    self._face_direction = PRIMARY_POND_FACE
                    self._refill_multihop = True
                    return None
                lvl = self._water_level(world.ram)
                if self._commit_multihop_main_pond(world.ram, lvl):
                    return None
            # Still north: drop any post, then scripted east→south past fence
            # end (x≥31). Densify alone sticks at ~(25,30).
            if self._ensure_hands_empty_for_refill(world.ram):
                self._pending_multihop_after_drop = True
                self._plot_phase = PlotPhase.REFILL
                self._state = CropState.NAVIGATE
                return None
            self._queue_east_south_corridor_charge(player)
            self._plot_phase = PlotPhase.REFILL
            self._state = CropState.NAVIGATE
            return None
        # Failure / blocked — try multi-hop if gap partial, else full search.
        print(f"[CROP] Fence open failed: {result.reason}; retrying refill search")
        self._fence_subtask = None
        if self._ensure_hands_empty_for_refill(world.ram):
            self._pending_multihop_after_drop = True
            self._plot_phase = PlotPhase.REFILL
            self._state = CropState.NAVIGATE
            return None
        lvl = self._water_level(world.ram)
        if self._pond_corridor_gap_open(world.ram):
            if self._commit_multihop_main_pond(world.ram, lvl):
                return None
        self._plot_phase = PlotPhase.WATER
        self._start_refill(world.ram)
        return None
