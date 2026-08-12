"""Can-refill search, pond access, multi-hop, and corridor thrash for crops.

Mixin extracted from ``CropWaterTask`` (rr-ds3) so refill policy no longer
lives as nested thrash inside the crop mono. Methods keep ``self`` access
to task fields; behavior is an intentional move of prior private methods.
"""

from __future__ import annotations

from harvest.tasks.crop_fsm import CropState, PlotPhase

from typing import List, Optional, Tuple

import numpy as np

from retro_harness import Task, TaskResult, TaskStatus, WorldState

from harvest.tasks.nav import (
    VIEWPORT_HOP_TILES,
    WALKABLE_TILES,
    get_tile_at,
    make_action,
    tile_dist,
)
from harvest.tasks.water_refill import (
    REFILL_NONFILL_WATER_TILES,
    REFILL_PREFERRED_WATER_TILES,
    corridor_needs_fence_open,
    order_preferred_edges,
    select_main_pond_refill,
    select_staging_stand,
)
from harvest.tasks.pond_corridor import (
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
    build_east_south_corridor_charge,
    build_gap_south_fallback,
    build_west_south_lip_charge,
    compute_refill_hop_goal,
    decide_after_south_lip_charge,
    pond_corridor_gap_open as pond_corridor_gap_is_open,
)
from harvest.tasks.pond_thrash import (
    ThrashChargeKind,
    ThrashCounters,
    evaluate_corridor_thrash,
)
from harvest.tasks.crop_geometry import (
    edge_water_tile_id,
    find_pond_edges,
    is_bad_refill_stand,
    is_main_pond_stand,
    pond_access_blocking_fences,
    refill_stand_band,
)


class CropRefillMixin:
    """Refill / pond-access methods for CropWaterTask."""

    def _select_preferred_refill_edge(
        self,
        ram: np.ndarray,
        player: Tuple[int, int],
    ) -> Optional[Tuple[Tuple[int, int], str, int, str]]:
        """Pick a pathable preferred fill stand (F0/F9–FD).

        Returns (stand, face, water_id, path_mode) or None. Prefers full BFS,
        then viewport-hop. Does not start fence-open — caller decides that.
        """
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
            return None

        edges = order_preferred_edges(
            edges,
            player,
            water_id_for=lambda t, f: edge_water_tile_id(ram, t, f),
        )
        check_n = min(len(edges), 40)
        hop_fallback: Optional[Tuple[Tuple[int, int], str, int]] = None
        for tile, face in edges[:check_n]:
            water_id = edge_water_tile_id(ram, tile, face)
            full = self._pathfinder.find_path(ram, player, tile)
            if full is not None:
                return (tile, face, water_id, "full")
            # Viewport-near only: hop must actually reach the stand tile.
            # Partial hops toward a fenced-off pond/stream are not reachability.
            if hop_fallback is None:
                hop = self._pathfinder.find_path(
                    ram, player, tile, max_steps=VIEWPORT_HOP_TILES
                )
                if hop is not None and (not hop or hop[-1] == tile):
                    hop_fallback = (tile, face, water_id)
        if hop_fallback is not None:
            tile, face, water_id = hop_fallback
            return (tile, face, water_id, "hop")
        return None

    def _start_refill(self, ram: np.ndarray):
        """Navigate to a CheckToolSuccess fill stand to refill the watering can.

        Order (Clean, no RAM poke):
          1) Named main-pond corridor when pathable
          2) Any other preferred edge (F9 north / FC south / …) while pathable
             — critical: do this *before* fence-open so west-pocket empty-can
             can fill at F9 without burning the day on y=31 fence toss stalls
          3) Stage + open y=31 fence only when no preferred water is reachable
          4) Exhaust
        Non-fill stream IDs F1/F2/F7/F8 are never chosen.
        """
        current_lvl = self._water_level(ram)

        # Track when refill search starts; detect water leaking during search
        if self._refill_search_level < 0:
            self._refill_search_level = current_lvl
        elif current_lvl < self._refill_search_level:
            leaked = self._refill_search_level - current_lvl
            print(f"[CROP] Refill search leaked {leaked} water (was {self._refill_search_level}, now {current_lvl})")
            for bad in list(self._bad_refill_tiles):
                for dy in range(-3, 4):
                    for dx in range(-3, 4):
                        self._bad_refill_tiles.add((bad[0] + dx, bad[1] + dy))
            self._refill_search_level = current_lvl

        player = self._navigator.current_tile

        # rr-qc9r: after a failed refill tile, charge counters stay maxed so
        # subsequent water tiles only densify-thrash. Soft-reset when not
        # mid-scripted-charge so each empty-can attempt gets a fresh budget.
        if (
            not getattr(self._corridor, "pending_south_lip_charge", False)
            and not getattr(self._corridor, "pending_gap_charge", False)
            and not self._action_queue
            and (
                getattr(self._corridor, "south_lip_charges", 0) >= 6
                or getattr(self._corridor, "east_south_charges", 0) >= 4
            )
        ):
            print(
                f"[CROP] Soft-reset refill charges at {player} "
                f"(lip={getattr(self._corridor, 'south_lip_charges', 0)} "
                f"es={getattr(self._corridor, 'east_south_charges', 0)})"
            )
            self._corridor.soft_reset_charges()

        def _full_path(start: Tuple[int, int], goal: Tuple[int, int]):
            return self._pathfinder.find_path(ram, start, goal)

        def _reachable_path(start: Tuple[int, int], goal: Tuple[int, int]):
            """True reachability only — not a partial hop toward a blocked goal.

            Viewport hop is used later for navigation after a stand is chosen.
            Using hop as the select_main_pond path oracle falsely treats the
            main pond as pathable when the y=31 wall is still up (hop ends
            north of the wall but is non-None).
            """
            full = self._pathfinder.find_path(ram, start, goal)
            if full is not None:
                return full
            # Goal inside live viewport: hop that actually reaches the stand.
            hop = self._pathfinder.find_path(
                ram, start, goal, max_steps=VIEWPORT_HOP_TILES
            )
            if hop is not None and (not hop or hop[-1] == goal):
                return hop
            return None

        # 1) Named main-pond corridor (FARM_MAIN_POND_STANDS) — only when the
        # stand still faces CheckToolSuccess-valid water on the live map.
        def _pond_fill_ok(stand: Tuple[int, int], face: str) -> bool:
            return edge_water_tile_id(ram, stand, face) in REFILL_PREFERRED_WATER_TILES

        pond = select_main_pond_refill(
            player, _reachable_path, bad_stands=self._bad_refill_tiles
        )
        if pond is not None and not _pond_fill_ok(pond.stand, pond.face):
            pond = None

        if pond is not None:
            wid = edge_water_tile_id(ram, pond.stand, pond.face)
            self._commit_refill_nav(
                ram,
                pond.stand,
                pond.face,
                current_lvl,
                source=pond.source,
                water_id=wid if wid >= 0 else 0xF0,
            )
            return

        blocking = pond_access_blocking_fences(ram)

        # 2) Preferred-edge search BEFORE fence-open. North F9 spur is often
        # pathable from the west plant pocket without clearing y=31; south FC
        # after partial clear too. Fence toss used to starve this path.
        chosen = self._select_preferred_refill_edge(ram, player)
        if chosen is not None:
            tile, face, water_id, path_mode = chosen
            if path_mode == "hop":
                print(
                    f"[CROP] Refill using hop path to "
                    f"({tile[0]},{tile[1]}) water=0x{water_id:02X}"
                )
            self._commit_refill_nav(
                ram,
                tile,
                face,
                current_lvl,
                source="preferred_edge",
                water_id=water_id,
            )
            return

        # 2b) Multi-hop preferred edges (esp. north F9) when full/exact-hop
        # reachability fails under viewport. Dry fixture has F9 at ~(26,12)
        # but full BFS from west pocket is None — without this we burn the day
        # on y=31 fence toss that cannot south-transit empty-handed.
        if self._commit_multihop_preferred_edge(ram, current_lvl):
            return

        # 3) Corridor gap already open but full BFS still fails under viewport
        # staleness — multi-hop to F0 *before* spending another fence-open.
        # ROM trap: after local_drop gap at ~(25,30), preferred edges and full
        # pond BFS are still false; a second FenceClearLoopTask burns the day.
        if self._pond_corridor_gap_open(ram) or self._fence_open_attempts > 0:
            if self._commit_multihop_main_pond(ram, current_lvl):
                return

        # 4) Wall still sealed — open y=31 corridor for main pond.
        if blocking and corridor_needs_fence_open(
            player,
            _full_path,
            blocking_fences=blocking,
            bad_stands=self._bad_refill_tiles,
        ):
            if self._try_open_pond_access(ram, list(blocking)):
                return
        elif blocking and self._try_open_pond_access(ram, list(blocking)):
            return

        # 5) Exhaust — nothing preferred pathable and fence open declined.
        self._refill_exhausted = True
        remaining = len(self._water_steps) - self._water_index
        print(
            f"[CROP] No reachable preferred water edge"
            f"{f' (fences={len(blocking)})' if blocking else ''}"
            f", skipping {remaining} tiles"
        )
        self.skipped_water += remaining
        self._plot_skipped += remaining
        self._water_index = len(self._water_steps)
        self._advance_water_step(ram)

    def _player_carrying(self, ram: np.ndarray) -> bool:
        """True when player is carrying a liftable (fence/bush/rock)."""
        try:
            from harvest.tasks.fence_flow import ACTION_CARRYING_BIT, ADDR_PLAYER_STATE
        except Exception:
            ACTION_CARRYING_BIT = 0x02
            ADDR_PLAYER_STATE = 0xD2
        if ADDR_PLAYER_STATE >= len(ram):
            return False
        return bool(int(ram[ADDR_PLAYER_STATE]) & ACTION_CARRYING_BIT)

    def _queue_local_drop(self) -> None:
        """Queue a multi-face local drop so multi-hop can walk after fence open."""
        self._action_queue.clear()
        for face in ("down", "left", "right", "up"):
            self._action_queue.extend([make_action(**{face: True}) for _ in range(6)])
            self._action_queue.extend(
                [make_action(**{face: True, "a": True}) for _ in range(12)]
            )
            self._action_queue.extend([make_action() for _ in range(8)])
        print("[CROP] Queued local drop (carrying blocks pond multi-hop)")

    def _queue_east_south_corridor_charge(
        self,
        player: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Scripted east past fence wall end, then south into pond band.

        Body lives in ``pond_corridor.build_east_south_corridor_charge``.
        """
        if player is None:
            player = self._navigator.current_tile
        n = getattr(self._corridor, "east_south_charges", 0)
        actions = build_east_south_corridor_charge(player, n)
        self._action_queue.clear()
        self._action_queue.extend(actions)
        self._corridor.pending_gap_charge = True
        self._corridor.east_south_charges = n + 1
        # Scripted charges burn frames outside BFS progress — reset target clock.
        self._steps_on_target = 0
        print(
            f"[CROP] Queue east→south corridor charge from {player} "
            f"(past fence end x≥31 then south) n={self._corridor.east_south_charges}"
        )

    def _queue_gap_south_fallback(
        self,
        player: Optional[Tuple[int, int]] = None,
    ) -> None:
        """When east past fence end is sealed, try open gap then south.

        Body lives in ``pond_corridor.build_gap_south_fallback``.
        """
        if player is None:
            player = self._navigator.current_tile
        actions = build_gap_south_fallback(player)
        self._action_queue.clear()
        self._action_queue.extend(actions)
        self._corridor.gap_south_tried = True
        self._corridor.pending_gap_charge = True
        self._corridor.east_south_charges = getattr(self._corridor, "east_south_charges", 0) + 1
        self._steps_on_target = 0
        print(
            f"[CROP] Queue gap-south fallback from {player} "
            f"(west to gap then south wiggle) n={self._corridor.east_south_charges}"
        )

    def _queue_west_south_lip_charge(
        self,
        player: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Scripted approach to F0 south lip from south-of-wall / soft-block.

        Body lives in ``pond_corridor.build_west_south_lip_charge``.
        """
        if player is None:
            player = self._navigator.current_tile
        actions, band = build_west_south_lip_charge(player)
        self._action_queue.clear()
        self._action_queue.extend(actions)
        self._corridor.pending_south_lip_charge = True
        self._corridor.south_lip_charges = getattr(self._corridor, "south_lip_charges", 0) + 1
        # Scripted charges burn frames outside BFS progress — reset target clock.
        self._steps_on_target = 0
        print(
            f"[CROP] Queue west→south-lip charge from {player} toward F0 stand "
            f"(band={band}) n={self._corridor.south_lip_charges}"
        )

    def _ensure_hands_empty_for_refill(self, ram: np.ndarray) -> bool:
        """If carrying, queue drop and return True (caller should wait)."""
        if not self._player_carrying(ram):
            return False
        if self._action_queue:
            return True
        self._queue_local_drop()
        return True

    def _pond_corridor_gap_open(self, ram: np.ndarray) -> bool:
        """True when the y=31 wall has a usable gap (not sealed, not unknown).

        Predicate lives in ``pond_corridor.pond_corridor_gap_open``.
        """
        n = len(pond_access_blocking_fences(ram))
        return pond_corridor_gap_is_open(
            n, getattr(self, "_fence_open_attempts", 0)
        )

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

    def _refill_hop_goal(
        self,
        ram: np.ndarray,
        player: Tuple[int, int],
        ultimate: Tuple[int, int],
    ) -> Tuple[int, int]:
        """Densify multi-hop refill: nearest corridor waypoint that closes dist.

        Body lives in ``pond_corridor.compute_refill_hop_goal``.
        """
        best_seen = getattr(self, "_refill_best_dist", tile_dist(player, ultimate))
        hop_budget = VIEWPORT_HOP_TILES + 3

        def find_path(
            start: Tuple[int, int],
            goal: Tuple[int, int],
            max_steps: Optional[int] = None,
        ):
            if max_steps is None:
                return self._pathfinder.find_path(ram, start, goal)
            return self._pathfinder.find_path(
                ram, start, goal, max_steps=max_steps
            )

        return compute_refill_hop_goal(
            player,
            ultimate,
            find_path,
            best_seen=best_seen,
            hop_budget=hop_budget,
        )

    def _commit_refill_nav(
        self,
        ram: np.ndarray,
        stand: Tuple[int, int],
        face: str,
        current_lvl: int,
        *,
        source: str,
        water_id: int = 0xF0,
        multihop: bool = False,
    ) -> None:
        """Begin navigate/act for a chosen refill stand."""
        player = self._navigator.current_tile
        self._refill_pond_tile = stand
        self._refill_pond_face = face
        self._refill_level_before = current_lvl
        self._clear_crop_walkable()
        self._plot_phase = PlotPhase.REFILL
        self._target_tile = stand
        self._approach_tile = stand
        self._face_direction = face
        self._state = CropState.NAVIGATE
        self._navigator.path = []
        self._steps_on_target = 0
        dist = abs(stand[0] - player[0]) + abs(stand[1] - player[1])
        # Multi-hop when far or after fence-gap (viewport cannot full-BFS).
        self._refill_multihop = bool(multihop) or dist > VIEWPORT_HOP_TILES
        self._refill_best_dist = dist
        band = refill_stand_band(stand)
        print(
            f"[CROP] Refill at ({stand[0]},{stand[1]}) facing {face} "
            f"water=0x{water_id:02X} source={source} dist={dist} band={band} "
            f"can={current_lvl} multihop={self._refill_multihop}"
        )

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
            # Densify thrash → scripted charges. Rules live in pond_thrash
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
        """Apply a pond_corridor charge-completion decision (navigate thrash).

        Decision body lives in ``pond_corridor.decide_after_*``.
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


