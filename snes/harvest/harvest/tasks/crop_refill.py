"""Can-refill search, pond access, multi-hop, and corridor thrash for crops.

Mixin extracted from ``CropWaterTask`` (rr-ds3) so refill policy no longer
lives as nested thrash inside the crop mono. Methods keep ``self`` access
to task fields; behavior is an intentional move of prior private methods.

Pond approach / fence-open live in ``crop_refill_pond``; densify thrash,
recover, and corridor-decision apply live in ``crop_refill_verify``.
``CropRefillMixin`` remains the public surface for ``CropWaterTask``.
"""

from __future__ import annotations

from harvest.tasks.crop_fsm import CropState, PlotPhase

from typing import Optional, Tuple

import numpy as np

from harvest.tasks.nav import (
    VIEWPORT_HOP_TILES,
    make_action,
    tile_dist,
)
from harvest.tasks.water_refill import (
    REFILL_PREFERRED_WATER_TILES,
    corridor_needs_fence_open,
    order_preferred_edges,
    select_main_pond_refill,
)
from harvest.tasks.pond_corridor import (
    pond_corridor_gap_open as pond_corridor_gap_is_open,
    build_east_south_corridor_charge,
    build_gap_south_fallback,
    build_west_south_lip_charge,
    compute_refill_hop_goal,
)
from harvest.tasks.crop_geometry import (
    edge_water_tile_id,
    find_pond_edges,
    is_bad_refill_stand,
    pond_access_blocking_fences,
    refill_stand_band,
)
from harvest.tasks.crop_refill_pond import CropRefillPondMixin
from harvest.tasks.crop_refill_verify import CropRefillVerifyMixin


class CropRefillMixin(CropRefillPondMixin, CropRefillVerifyMixin):
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
