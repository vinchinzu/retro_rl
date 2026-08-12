"""Multi-hop densify goal selection for pond can refill.

``compute_refill_hop_goal`` picks the next intermediate waypoint when a
direct path to the F0 stand is unreliable (viewport soft-blocks, fence
residue, south-lip thrash).
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

from harvest.maps.map_config import (
    FARM_POND_MULTIHOP_WAYPOINTS,
    FARM_POND_POST_GAP_CORRIDOR,
)
from harvest.tasks.nav import VIEWPORT_HOP_TILES, tile_dist


# find_path(start, goal, max_steps=None) → path list or None
PathFn = Callable[..., Optional[Sequence[Tuple[int, int]]]]


def compute_refill_hop_goal(
    player: Tuple[int, int],
    ultimate: Tuple[int, int],
    find_path: PathFn,
    *,
    best_seen: Optional[int] = None,
    hop_budget: Optional[int] = None,
) -> Tuple[int, int]:
    """Densify multi-hop refill: nearest corridor waypoint that closes dist.

    Without intermediates, hop-toward F0 from ~(25,30) stalls when the
    next live walkable cell is not strictly closer in raw BFS (viewport
    dirt IDs / fence residue). Named north-lip chain keeps progress.

    Monotonic: never pick a waypoint farther from the ultimate stand than
    the best distance already achieved — that was the (24,30)→(15,30)
    thrash on the dry fixture.

    ``find_path(start, goal, max_steps=None)`` should return a path list
    ending at/near goal, or None if unreachable.
    """
    dist_u = tile_dist(player, ultimate)
    if hop_budget is None:
        hop_budget = VIEWPORT_HOP_TILES + 3
    if best_seen is None:
        best_seen = dist_u

    # Only accept the ultimate when a *true* short path exists.
    # rr-qc9r: south of wall at x≤30, BFS invents 4–7 tile paths to F0
    # that live physics never walks — force intermediate densify below
    # whenever still west of stand on the south lip.
    full = find_path(player, ultimate)
    if full is not None and len(full) <= hop_budget:
        south_corridor_far = (
            player[1] >= 33
            and player[0] <= 30
            and ultimate[0] >= 32
            and ultimate[1] >= 33
            and dist_u > 2
        )
        if not south_corridor_far:
            return ultimate

    post_gap = FARM_POND_POST_GAP_CORRIDOR
    chain = FARM_POND_MULTIHOP_WAYPOINTS

    # Near pond but off-stand (e.g. (33,38) after lip overshoot): hop to lip.
    # East-of-pond thrash (41,32): BFS invents path to (32,34) that never
    # moves — force intermediate west crumbs first.
    if (
        player[0] >= 30
        and player[1] >= 30
        and ultimate[1] >= 30
        and ultimate[0] >= 30
        and tile_dist(player, ultimate) > 1
    ):
        # Far east of pond: step west in small hops (not full 9-tile leap).
        if player[0] >= 36:
            west_step = (max(player[0] - 4, 33), min(max(player[1], 33), 35))
            if west_step != player:
                print(
                    f"[CROP] East-pond densify west step {player} → {west_step}"
                )
                return west_step
        near_crumbs: List[Tuple[int, int]] = [
            (min(player[0] - 2, 34), player[1]) if player[0] > 34 else (32, 34),
            (34, 34),
            (33, 34),
            (32, 34),
            (33, 35),
            (32, 35),
            (32, 36),
            (30, 34),
            (31, 34),
        ]
        for wp in near_crumbs:
            if wp == player:
                continue
            # Don't densify further east.
            if wp[0] > player[0]:
                continue
            hop = find_path(player, wp, max_steps=hop_budget + 4
            )
            if hop is None:
                continue
            end = hop[-1]
            if end == player:
                continue
            # Reject ends that don't improve x toward pond when east of it.
            if player[0] > 34 and end[0] >= player[0]:
                continue
            if tile_dist(end, ultimate) < dist_u or (
                end[0] < player[0]
                or (end[1] < player[1] and abs(end[0] - ultimate[0]) <= 2)
            ):
                print(
                    f"[CROP] Near-pond densify {player} → {wp} (end={end})"
                )
                return wp
        if player[0] > 34:
            return (max(player[0] - 3, 32), min(max(player[1], 33), 35))
        return ultimate

    # ROM trap: after east→south wall cross, player often lands (28,32)
    # where RIGHT/DOWN soft-block. BFS wants (29,32)→pond and thrash.
    # Route WEST then south corridor: (24,32)→(24,35)→(29,36)→(32,34).
    # Power-on residual: densify oscillated (24,34)↔(24,35) forever because
    # pure-south was accepted even when dist-to-F0 grew. Prefer east.
    if (
        player[1] >= 32
        and ultimate[1] >= 33
        and player[0] <= 30
        and tile_dist(player, ultimate) > 1
    ):
        # On soft-block ~(28,32): force SOUTH first (not west). rr-qc9r:
        # west-first to (24,32) then densify landed the (25,34)↔(29,32)
        # oscillation. Stay below fence and push east on y≥34.
        if player[0] >= 27 and player[1] <= 33:
            for wp in ((player[0], 34), (28, 34), (30, 34), (29, 35), (26, 34)):
                if wp == player:
                    continue
                hop = find_path(player, wp, max_steps=hop_budget + 2
                )
                if hop is not None and hop[-1] != player:
                    end = hop[-1]
                    if end[1] > player[1] or end[0] >= player[0]:
                        print(
                            f"[CROP] Pond soft-block (28,32) band: south/east "
                            f"{player} → {wp} (end={end})"
                        )
                        return wp
            forced_s = (min(player[0] + 2, 32), min(player[1] + 2, 35))
            print(
                f"[CROP] Pond soft-block densify force south/east "
                f"{player} → {forced_s}"
            )
            return forced_s
        # On y≥34 corridor: prefer short EAST hops over direct F0 / pure N/S.
        # Power-on densify (25,34)→(32,34) is 7 tiles — viewport edge thrash.
        # Dry residual: score preferred (29,35)→(29,34) pure-north (d=1)
        # forever — must require east gain when west of stand (rr-qc9r).
        if player[1] >= 34 and player[0] < 32:
            # Cap intermediate hop to +4 east so we chain rather than
            # leap the full remaining distance (viewport false paths).
            max_east = min(player[0] + 4, 32)
            near_east = [
                (min(player[0] + 3, max_east), 34),
                (min(player[0] + 2, max_east), 34),
                (min(player[0] + 4, max_east), 34),
                (min(player[0] + 3, max_east), player[1]),
                (min(player[0] + 2, max_east), player[1]),
                (min(player[0] + 4, max_east), player[1]),
                (min(player[0] + 3, max_east), 35),
                (min(30, max_east), 34) if player[0] < 30 else (31, 34),
                (min(31, max_east), 34) if player[0] < 31 else (32, 34),
            ]
            # Only allow ultimate F0 when already within 4 tiles.
            if dist_u <= 4:
                near_east.extend([(32, 34), (32, 35), (31, 34)])
            best_wp: Optional[Tuple[int, int]] = None
            best_score: Optional[Tuple[int, int, int, int]] = None
            for wp in near_east:
                if wp == player:
                    continue
                d_to_wp = tile_dist(player, wp)
                if d_to_wp > hop_budget or d_to_wp > 5:
                    continue
                # Must aim east of current x (no pure N/S / west).
                if wp[0] <= player[0]:
                    continue
                # Cap east leap when far — chain intermediates.
                if wp[0] - player[0] > 4 and dist_u > 4:
                    continue
                hop = find_path(player, wp, max_steps=hop_budget + 2
                )
                if hop is None:
                    continue
                end = hop[-1]
                if end == player or end[0] <= player[0]:
                    continue
                if end[0] - player[0] > 4 and dist_u > 4:
                    continue
                end_dist = tile_dist(end, ultimate)
                east_gain = end[0] - player[0]
                # Prefer: more east (within cap), closer to F0, short hop.
                score = (-east_gain, end_dist, d_to_wp, abs(end[1] - 34))
                if best_score is None or score < best_score:
                    best_score = score
                    best_wp = wp
            if best_wp is not None:
                print(
                    f"[CROP] South-lip densify {player} → {best_wp} "
                    f"(east-prefer, ultimate={ultimate})"
                )
                return best_wp
            forced = (min(player[0] + 3, 32), 34)
            print(
                f"[CROP] South-lip densify fallback east {player} → {forced}"
            )
            return forced
        south_lip_crumbs: List[Tuple[int, int]] = [
            # East-first corridor (avoid N/S thrash at fixed x=24).
            (26, 35),
            (28, 35),
            (29, 36),
            (30, 36),
            (32, 36),
            (32, 35),
            (32, 34),
            (33, 34),
            (26, 34),
            (28, 34),
            (24, 35),
            (24, 34),
            (24, 33),
            (24, 32),
        ]
        best_wp = None
        best_score = None
        for wp in south_lip_crumbs:
            if wp == player:
                continue
            d_to_wp = tile_dist(player, wp)
            if d_to_wp > hop_budget + 4:
                continue
            # Prefer crumbs that improve toward ultimate without re-entering
            # the (28,32) east-lock from west of it.
            if wp[0] >= 28 and wp[1] <= 33 and player[0] <= 27:
                continue
            hop = find_path(player, wp, max_steps=hop_budget + 4
            )
            if hop is None:
                continue
            end = hop[-1]
            if end == player:
                continue
            end_dist = tile_dist(end, ultimate)
            # Reject pure N/S oscillation at same (or west) x when far from F0.
            if (
                end[0] <= player[0]
                and end_dist >= dist_u
                and abs(player[0] - ultimate[0]) > 2
            ):
                continue
            # Accept only if we close dist to F0 or move east toward it.
            if not (end_dist < dist_u or end[0] > player[0]):
                continue
            # Prefer: better end dist, more east, shorter hop.
            score = (end_dist, -end[0], d_to_wp)
            if best_score is None or score < best_score:
                best_score = score
                best_wp = wp
        if best_wp is not None:
            print(
                f"[CROP] South-lip densify {player} → {best_wp} "
                f"(east-prefer, ultimate={ultimate})"
            )
            return best_wp
        # BFS crumbs blocked (stale viewport): return scripted east target.
        if player[0] < 30:
            forced = (min(player[0] + 4, 32), max(player[1], 34))
            print(
                f"[CROP] South-lip densify fallback east {player} → {forced}"
            )
            return forced

    # ROM trap: multi-hop to main-pond south lip from north of y=31.
    # Empty-handed south through a y=31 gap soft-blocks on (13,31) y≈505
    # (BFS invents (12,32) path that game physics rejects). NEVER densify
    # south through the gap empty-handed.
    #
    # ROM-verified routes after gap open (Y1_Test_Crops_Planted_Dry):
    #   1) Carry-south while holding a post (FenceClearLoop corridor_only)
    #   2) East past fence wall end (x≥30/31) then pure south (empty OK)
    # Prefer (2) for post-drop multi-hop; never charge gap from y≤31 empty.
    if player[1] <= 31 and ultimate[1] >= 32:
        # Past fence end (x≥31): force south — never self-hop (31,29) thrash
        # (power-on residual rr-5go9). Reject west-regressive truncated
        # hops (BFS end=(20,30) from (31,29) is false viewport progress).
        if player[0] >= 31:
            south_targets: List[Tuple[int, int]] = [
                (player[0], 32),
                (31, 32),
                (32, 32),
                (32, 33),
                (32, 34),
                (30, 32),
                (31, 33),
            ]
            for wp in south_targets:
                if wp == player:
                    continue
                hop = find_path(player, wp, max_steps=hop_budget + 4
                )
                if hop is None:
                    continue
                end = hop[-1]
                if end == player:
                    continue
                # Must not walk west of start toward plant pocket.
                if end[0] < player[0] - 1:
                    continue
                if end[1] > player[1] or tile_dist(end, ultimate) < dist_u:
                    print(
                        f"[CROP] Past-fence densify south {player} → {wp} "
                        f"(end={end}, ultimate={ultimate})"
                    )
                    return wp
            # Scripted: stay east, push south one tile at a time.
            forced_s = (player[0], min(player[1] + 2, 34))
            if forced_s != player:
                print(
                    f"[CROP] Past-fence densify force south {player} → {forced_s}"
                )
                return forced_s

        # East-crawl: y=30 lip past fence end x≥31 then south to pond lip.
        east_crumbs: List[Tuple[int, int]] = [
            (min(player[0] + 4, 32), min(player[1], 30)),
            (20, min(player[1], 30)),
            (24, 30),
            (26, 30),
            (28, 30),
            (30, 30),
            (31, 30),
            (32, 30),
            (31, 32),
            (30, 32),
            (30, 33),
            (32, 33),
            (32, 34),
        ]
        # On the gap row y=31 (west of fence end): step N/E off soft-block.
        if player[1] == 31 and player[0] < 30:
            for wp in (
                (min(player[0] + 2, 28), 30),
                (player[0], 30),
                (player[0] + 1, 30),
                (player[0] - 1, 30),
                (20, 30),
                (24, 30),
                (28, 30),
            ):
                if wp == player or wp[0] < 0 or wp[1] < 0:
                    continue
                hop = find_path(player, wp, max_steps=hop_budget + 2
                )
                if hop is not None:
                    print(
                        f"[CROP] Gap soft-block escape {player} → {wp} "
                        f"(east-crawl, never south through gap)"
                    )
                    return wp
            return (min(player[0] + 3, 28), 30)

        for wp in east_crumbs:
            if wp == player:
                continue
            # Only accept crumbs that improve toward ultimate or push east/south.
            if wp[0] < player[0] and wp[1] <= player[1]:
                continue
            d_to_wp = tile_dist(player, wp)
            if d_to_wp > hop_budget + 4:
                continue
            hop = find_path(player, wp, max_steps=hop_budget + 4
            )
            if hop is None:
                continue
            end = hop[-1] if hop else player
            # Reject hops that only walk onto the gap tile (y=31, x≈12–16).
            if end[1] == 31 and end[0] <= 18:
                continue
            if end == player:
                continue
            if end[1] >= 32 or end[0] > player[0] or tile_dist(end, ultimate) < dist_u:
                print(
                    f"[CROP] East-crawl densify {player} → {wp} "
                    f"(end={end}, ultimate={ultimate})"
                )
                return wp

        # Fallback: walk east on current lip past fence wall end (never
        # south through gap empty-handed). Never return player (self-hop).
        for wx in (32, 31, 30, 28, 26, 24, 22, 20, 18):
            wp = (wx, min(player[1], 30))
            if wp == player or wp[0] <= player[0]:
                continue
            hop = find_path(player, wp, max_steps=hop_budget + 2
            )
            if hop is not None:
                return wp
        forced_e = (min(player[0] + 4, 32), min(player[1], 30))
        if forced_e == player:
            forced_e = (player[0], min(player[1] + 2, 32))
        return forced_e

    # North F9 multi-hop only when already north of the y=13 fence bar
    # (player y≤16) or east of it (x≥20). From west plant pocket F9 is
    # sealed — do not densify into potato/y=13 thrash.
    if ultimate[1] <= 20 and (player[1] <= 16 or player[0] >= 20):
        north_crumbs: List[Tuple[int, int]] = [
            (20, 16),
            (22, 14),
            (24, 13),
            (25, 13),
            (player[0], max(ultimate[1], player[1] - 4)),
            (min(ultimate[0], player[0] + 3), max(ultimate[1], player[1] - 3)),
            (ultimate[0] - 1, ultimate[1]),
            ultimate,
        ]
        for wp in north_crumbs:
            if wp == player or wp[0] < 0 or wp[1] < 0:
                continue
            hop = find_path(player, wp, max_steps=hop_budget + 2
            )
            if hop is None:
                continue
            end = hop[-1]
            if end[1] < player[1] or tile_dist(end, ultimate) < dist_u:
                return wp

    # Default: next breadcrumb closer to ultimate (south corridor first).
    best: Optional[Tuple[int, int]] = None
    best_goal_dist = dist_u
    best_wp_dist = 999
    progress_cap = min(dist_u, best_seen + 1)
    for wp in chain:
        if wp == player:
            continue
        d_to_goal = tile_dist(wp, ultimate)
        if d_to_goal >= progress_cap:
            continue
        d_to_wp = tile_dist(player, wp)
        if d_to_wp > hop_budget or d_to_wp < 1:
            continue
        hop = find_path(player, wp, max_steps=hop_budget
        )
        if hop is None:
            continue
        end = hop[-1] if hop else player
        if tile_dist(end, ultimate) >= dist_u:
            continue
        if (
            best is None
            or d_to_wp < best_wp_dist
            or (d_to_wp == best_wp_dist and d_to_goal < best_goal_dist)
        ):
            best = wp
            best_goal_dist = d_to_goal
            best_wp_dist = d_to_wp

    if best is not None:
        return best
    return ultimate



__all__ = [
    "PathFn",
    "compute_refill_hop_goal",
]
