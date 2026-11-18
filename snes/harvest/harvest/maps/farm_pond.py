"""Farm pond / water-source constants and helpers.

ROM-mapped spring-farm facts for can refill, fence access, and no-go tiles.
"""

import os
from typing import Dict, FrozenSet, Optional, Tuple

from harvest.core.tile_catalog import MOUNTAIN_TILEMAP_IDS

# Farm coordinates whose visual/collision behavior is not represented by the
# metatile alone.  Several well-body tiles render as 0xA1, which is walkable in
# other farm contexts, so keep the coordinate-specific facts here.

# Horse barn sprite on farm 0x00. Live IDs are 0xA0/0xA3 (path), so occupancy
# comes from the trimmed horse_barn_edges slice: cells the farmer pushed into
# and never stood on. Stand-on stasis tiles stay walkable — treating those as
# no-go boxed (16,20) against 0xD8 and blocked the human south leave.
HORSE_BARN_WALL_TILES: FrozenSet[Tuple[int, int]] = frozenset({
    (16, 18), (17, 18), (18, 19), (19, 18),
    (18, 20), (18, 21),
})
# Cow-barn body west of the dirt the tape stood on (x=31 y=18–21). Push-into
# was x=30, not the leftover x=31 column.
COW_BARN_EAST_FACE_TILES: FrozenSet[Tuple[int, int]] = frozenset(
    (30, y) for y in range(18, 22)
)
# Trimmed horse_barn_edges: first step off (17,20) is south onto (17,21), then
# west around y=23. Dump NE stones at (46,16) face-up into 0xFA (46,15).
EAST_SPUR_FA_STAND: Tuple[int, int] = (46, 16)
EAST_SPUR_FA_FACE: str = "up"
# Water cells north of the stand. Approach from west dirt (45,16), not (46,14).
EAST_SPUR_FA_WATER: FrozenSet[Tuple[int, int]] = frozenset({(46, 14), (46, 15)})
EAST_SPUR_FA_APPROACH: Tuple[int, int] = (45, 16)
# Live leftover checkpoint: x=48 y=16 is 0xA8 bank (walkable ID, push-blocks
# from the north). Keep it no-go so a y=15 hug cannot south onto it; y=16
# east→west repair skirts via y=17.
EAST_SPUR_FA_EAST_LANE_X: int = 47
EAST_SPUR_FA_A8_BANK: FrozenSet[Tuple[int, int]] = frozenset(
    (x, 16) for x in range(48, 51)
)
# y=13→14 is a physical wall at x=46–50 even when IDs are 0x01/0xA0/0xA1.
# LEFT from (48,13) reaches (47,13) then DOWN slides back onto the 0xA0 bank
# (the 400k leftover oscillation). First open DOWN is x=51 (4f live).
EAST_SPUR_FA_SOUTH_OPEN_X: int = 51
# West of the spur: (45,14)/(45,15) are 0xA1 (walkable ID, push-blocks).
# Live leftover after the x=51 dump sat at (44,14); RIGHT is blocked.
# Drop to y=16 at x=44, then east onto the approach.
EAST_SPUR_FA_WEST_DROP_X: int = 44
EAST_SPUR_FA_WEST_LIP: FrozenSet[Tuple[int, int]] = frozenset(
    {(45, 14), (45, 15)}
)
# South-stream 0xFC at (14,49)/(14,50). Live leftover 1-stone pin (16,48):
# DOWN at x=16 is open (8f to (16,49)). DOWN from (15,48) onto 0xA1 (15,49)
# slides east back to (16,48). DOWN from (13,48) onto (13,49) is a wall.
# Pickup BFS treated those A1 banks as dirt and never reached (12,55).
SOUTH_STREAM_FC_BANKS: FrozenSet[Tuple[int, int]] = frozenset({
    (13, 49), (15, 49),
    (13, 50), (15, 50),
})
# The second well at (37,48) has walkable-looking A1/A3/A8 rim tiles.
# The cover sweep stalled at (572,761), trying to reach (36,48) through
# that rim, on every evening from Y3 Spring D9 onward in the 851 ending.
SOUTH_WELL_BODY: FrozenSet[Tuple[int, int]] = frozenset(
    (x, y) for y in range(47, 50) for x in range(36, 39)
)
HORSE_BARN_LEAVE_TILE: Tuple[int, int] = (17, 21)

FARM_NO_GO_TILES: FrozenSet[Tuple[int, int]] = frozenset({
    # Shipping-bin ditch / F2 water fringe (does NOT refill the can).
    (9, 26), (9, 27), (9, 28),
    (11, 26), (11, 27), (11, 28),
    # House frontage.
    (8, 12), (9, 12), (10, 12),
    # Well body.  These are visually solid even when the live tile ID is 0xA1.
    (15, 26), (16, 26), (17, 26),
    (15, 27), (16, 27), (17, 27),
    # Tool-shed body (ROM-measured 2026-09-28 by teleport+hold probes on the
    # Summer D30 farm): live ids read 0xA1 but the farmer stops. The door
    # column (26,28)-(26,29) and its flanking (25,29) stay open.
    (24, 26), (25, 26), (26, 26), (27, 26), (28, 26),
    (24, 27), (25, 27), (26, 27), (27, 27), (28, 27),
    (24, 28), (25, 28), (27, 28), (28, 28),
    (24, 29), (25, 29), (27, 29), (28, 29),
}) | HORSE_BARN_WALL_TILES | COW_BARN_EAST_FACE_TILES | SOUTH_STREAM_FC_BANKS | SOUTH_WELL_BODY

# The x=11 column between the F2 ditch and the west ring is in FARM_NO_GO_TILES
# but is open ground: a teleport to (11,26) and DOWN walked to (11,30) and on
# through the fence gap to (11,32) on the Summer D5 power-on farm. It is the
# only way into the bin stand (11,30) once the west ring is grown and a weed
# regrows on (12,30), and on Summer D27 the only way from the yard to the
# pasture (the y=31 gap had moved to x=11-12). Only the harvest ship walk and the
# pasture route allow it (``Pathfinder.allow_tiles``): opened for every farm
# path it broke the west pocket's hoe-stand navigation (40 CROP_ESTABLISH
# failures in a power-on year, against 6).
SHIP_BIN_LANE_TILES: FrozenSet[Tuple[int, int]] = frozenset({(11, 26), (11, 27), (11, 28)})
# Standing on the lane or at the bin, a grown west ring and the fence leave the
# lane as the only way out. A walk that *starts* on these tiles may use it: a
# sealed RETURN_HOME or can fetch after a ship walk cost whole days of watering
# (36 "no_path sealed" events at (182,478) in one Summer).
SHIP_BIN_STAND_TILES: FrozenSet[Tuple[int, int]] = SHIP_BIN_LANE_TILES | frozenset({(11, 29), (11, 30)})

# ── Farm water sources (ROM-mapped, spring farm tilemap 0x00) ──────────────
# CheckToolSuccess farm fill only when tile-in-front *property* is F0/F9–FD
# (can → 0x14). Raw F1/F2/F7/F8 look like water but do not fill.
#
# Main pond (F0) water cells ~(31–34, 31–33). Human refill stand (go_to_water_source):
# south lip (32,34)/(33,34) face up. North lip (33,30) face down also fills.
#
# Early-spring west plant pocket (y≤30, x≈10–25) is cut off from the pond by a
# solid 0x05 fence wall on y=31 (x=11–29). Clearing ≥1 fence opens full BFS to
# F0/FC/FD/FB. Until then only non-fill F1/F8 north stream is pathable.
FARM_MAIN_POND_WATER_BOUNDS: Tuple[int, int, int, int] = (31, 31, 34, 33)
FARM_MAIN_POND_STANDS: Tuple[Tuple[Tuple[int, int], str], ...] = (
    ((32, 34), "up"),    # human go_to_water_source_end
    ((33, 34), "up"),
    ((33, 30), "down"),  # Y1_Near_Pond (ROM fill 0→20)
    ((34, 30), "down"),
    ((32, 30), "down"),
)
# Fence row that walls west field off from the main pond / south farm.
FARM_POND_ACCESS_FENCE_ROW: int = 31
FARM_POND_ACCESS_FENCE_X_RANGE: Tuple[int, int] = (11, 29)
# Staging stands just north of that wall. West plant-pocket stands (e.g.
# (13,27) after potato plant) soft-block pure-south movement even when live
# tile IDs look walkable — stage west/left before FenceClearLoopTask.
# (11,28) is also in FARM_NO_GO_TILES (shipping-bin ditch), which reads like a
# contradiction worth cleaning up — but POND_CORRIDOR_TILES in crop_planner is
# built from this tuple, so removing it is NOT the no-op it looks like. Left
# in deliberately; see docs/tasks/rr-20w-run16-defects.md.
FARM_POND_ACCESS_STAGING_TILES: Tuple[Tuple[int, int], ...] = (
    (11, 29),
    (12, 29),
    (10, 28),
    (11, 28),
    (15, 29),
    (18, 30),
    (20, 30),
)

# Densified multi-hop chain after a y=31 gap opens. Viewport BFS only sees
# ~7 tiles. ROM trap: north-lip y=30 east of x≈25 soft-blocks north and hits
# 0xFF — do **not** crawl the north lip. After a gap opens, go **south**
# through the wall then east on y≈32–34 to the south-lip F0 stands.
FARM_POND_POST_GAP_CORRIDOR: Tuple[Tuple[int, int], ...] = (
    # Through / just south of the y=31 gap (common clear at x≈12–15)
    (12, 32),
    (13, 32),
    (14, 32),
    (15, 32),
    (16, 32),
    (18, 32),
    (20, 32),
    (22, 32),
    (24, 32),
    (26, 32),
    (28, 32),
    (30, 32),
    (30, 33),
    (32, 33),
    (32, 34),
    (33, 34),
)
# North-lip crumbs only for when already south-of-gap routing fails.
FARM_POND_MULTIHOP_WAYPOINTS: Tuple[Tuple[int, int], ...] = (
    *FARM_POND_POST_GAP_CORRIDOR,
    (15, 29),
    (18, 30),
    (20, 30),
    (22, 30),
    (24, 30),
    (28, 29),
    (28, 30),
    (32, 30),
    (33, 30),
)

# Explicit west-pocket → main-pond corridor for empty-can refill.
# Prefer this route over generic water-edge search when the player is still
# north of the y=31 fence wall in the early-spring plant pocket.
FARM_POND_REFILL_CORRIDOR: Tuple[str, ...] = (
    "stage_west_of_fence",  # FARM_POND_ACCESS_STAGING_TILES
    "open_fence_row_y31",   # clear ≥1 fence on FARM_POND_ACCESS_FENCE_ROW
    "fill_at_main_pond",    # FARM_MAIN_POND_STANDS (F0 CheckToolSuccess)
)


def farm_pond_refill_primary_stand() -> Tuple[Tuple[int, int], str]:
    """Primary verified fill stand (south lip face up)."""
    return FARM_MAIN_POND_STANDS[0]


def farm_pond_refill_stands() -> Tuple[Tuple[Tuple[int, int], str], ...]:
    """Ordered preferred fill stands for the named pond corridor."""
    return FARM_MAIN_POND_STANDS


def player_in_west_plant_pocket(tile: Tuple[int, int]) -> bool:
    """True when player is north of the y=31 fence wall in the west/mid field."""
    x, y = tile
    return y <= 30 and x <= 28


# Tillable yard north of the y=31 fence (x=11–29) and west of the wall end.
# House/north-stream y<14 is out: this is the hoe/plant pocket only.
# Inclusive (x_min, y_min, x_max, y_max).
WEST_PLANT_POCKET_BOUNDS: Tuple[int, int, int, int] = (3, 14, 28, 30)

# Hoe/plant stand from ``d2_farm_plant``: 8 potatoes around (13,28), bag spent
# at that notch (0x07→0x54). Prefer this over DEFAULT_START_TILE (15,29).
WEST_POCKET_PLANT_CENTER: Tuple[int, int] = (13, 28)

# Second potato ring (rr-20w.3, D3). Beside the D2 rows, east of the well
# body (x15-17 y26-27). All 8 crop tiles land on solid soil y27-29 — off the
# y30 fence lip and off the well — with watering stands leaning toward the
# pond gap, never the shipping alcove or the D2 crops. This is the winner of
# ``crop_planner.plan_crop_field`` on live ``Y1_D3_PostShop`` RAM once
# fence-lip (y30) rings are excluded (score 170, route_cost 45, profit 440).
SECOND_POCKET_PLANT_CENTER: Tuple[int, int] = (19, 28)

# North face of that ring. (17, 29) is 0x02, but on LN_herd_y2_spring the
# stump 0x09–0x0C sits on (18, 28)–(19, 29), so the hop from the west
# pocket push-faces (18, 28), (18, 29), and (18, 32). (19, 26) is untilled
# 0x01 and the walk from (13, 26) stays on 0x01/0x02 north of the well.
SECOND_POCKET_HOE_STAND: Tuple[Tuple[int, int], str] = ((19, 26), "down")

# East-of-pond ring sites (rr-rings). The farm is NOT stable: the monthly tile
# check regrows the original clutter (fences on y=31, the (18-19, 28-29)
# stump, rocks, stumps, weeds, stones) over every bare tile, so Summer D1 looks
# like Spring D1, not like the cleared D2 farm (ROM ``PR_Y1_Summer_D1``, saved
# as ``LN_y1econ_rings_summer_d1_farm``). Sites are therefore picked on that
# pristine layout by ``scratch/site_study3.py``: the 5x5 footprint (ring, notch
# and every HOE_PLAN stand) holds only bare soil plus at most four weeds or
# stones (lifted by the plant-stand clearer), no rock, stump or fence; every
# hoe stand is walkable from the yard with the viewport BFS rules, and fully
# grown rings (solid crops) leave all pond stands reachable. A ring on the
# pond's south lip (32-33, 35) would seal its only approach, and the pasture
# (``grass_planter.PASTURE_BOUNDS`` = x13-27, y33-48) owns the field south of
# the yard. Walking tiles to the nearest pond stand: (33,37) 3, (41,33) 8,
# (44,29) 11. The yard ring (19,28) is dead in Summer (stump) and skipped by
# ``_ring_bare_capacity`` until the stump is gone.
SOUTH_POCKET_PLANT_CENTERS: Tuple[Tuple[int, int], ...] = (
    (33, 37),
    (41, 33),
    (44, 29),
)

# West-strip block south-west of the bin, opened by the fully cleared D2 farm
# (``FARM_CLEAR_D2``): the farm array of ``LN_d2_perfect_poweron``, Spring D24
# and Summer D1 holds zero debris, so these 5x5 footprints stay bare soil all
# Year 1. Two-tile lanes (x6-7, y37-38) and the x=11 bin column stay open, so
# the y=37 dirt row of ``_FARM_SOUTH_FIELD_TO_WEST_GATE`` and (6,33) still
# walk. A crop here is 6-12 tiles from the bin stand (11,30); the south field
# is 22-35. The south stream (14,49) fills the can. Rings beyond five exceed
# one stamina bar: ``CROP_WATER`` reports ``stamina_low`` and the day plan
# splices a hot-spring refill.
WEST_STRIP_PLANT_CENTERS: Tuple[Tuple[int, int], ...] = (
    (9, 35),
    (4, 35),
    (9, 40),
    (4, 40),
)

# Packed lattice south of the bin (``HARVEST_RING_LAYOUT=packed``). Rings sit 4 tiles apart on both axes, so
# the single lane between two rings is shared: every crop tile touches a lane, a lane serves the ring on each
# side, and eight crops cost 16 tiles of ground instead of the 25 of an isolated 5x5 site. Lanes run on
# x = 2, 6, 10 ... 26 and y = 33, 37, 41, 45, 49. That keeps the two tiles the hard-coded spa and shop walks
# use (``map_routes._FARM_SOUTH_WEST_TO_WEST_GATE``: the (6,37) lane crossing and (6,33)), the x=2 edge
# lane, and the open rows y=31-33 between the bin lane and the first ring row. Four rows of six, plus the
# two pocket rings north of y=31, are 26 rings (208 crops). ``PACKED_RING_ORDER`` is closest-first by
# walking distance from the bin stand (11,30) with every prefix still reachable, so a smaller N is always
# the best N; computed on the cleared D2 farm array (``LN_d2_perfect_poweron``). Mean walk to a crop's lane
# stand: 9.7 tiles at 9 rings, 13.4 at 18, 15.6 at 24 (the legacy nine average 18.0).
PACKED_COLUMNS: Tuple[int, ...] = (4, 8, 12, 16, 20, 24)
PACKED_ROWS: Tuple[int, ...] = (35, 39, 43, 47)
PACKED_RING_ORDER: Tuple[Tuple[int, int], ...] = (
    (12, 35), (8, 35), (16, 35), (12, 39), (4, 35), (8, 39), (20, 35), (16, 39),
    (12, 43), (4, 39), (8, 43), (24, 35), (20, 39), (16, 43), (12, 47), (4, 43),
    (8, 47), (24, 39), (20, 43), (16, 47), (4, 47), (24, 43), (20, 47), (24, 47),
)

# Keep the verified D2 opening, then establish by distance to the shipping
# stand. Nearby west-strip plots precede remote eastern plots. Sites are data:
# targets, bag counts and the splice limit all follow this ordering.
LEGACY_POCKET_PLANT_CENTERS: Tuple[Tuple[int, int], ...] = (
    WEST_POCKET_PLANT_CENTER,
    *sorted(
        (SECOND_POCKET_PLANT_CENTER, *WEST_STRIP_PLANT_CENTERS, *SOUTH_POCKET_PLANT_CENTERS),
        key=lambda center: abs(center[0] - 11) + abs(center[1] - 30),
    ),
)
PACKED_POCKET_PLANT_CENTERS: Tuple[Tuple[int, int], ...] = (
    WEST_POCKET_PLANT_CENTER,
    SECOND_POCKET_PLANT_CENTER,
    *PACKED_RING_ORDER,
)
# Notch blocks (``HARVEST_RING_LAYOUT=blocks``): 3x3 plots with the centre planted and one side cell
# left bare, ``xxx/xxo/xxx`` (east) or ``xxx/oxx/xxx`` (west), on the pitch-4 lattice in mirrored pairs
# whose notches face across the shared lane. Rows are the ``block_calendar`` plan derived from the
# cleared D2 farm (``LN_d2_perfect_poweron``, ``plan_blocks(pin_walkable(), core=12, late=2)``), nearest
# first: (centre, notch, role, Spring last-sow day, Summer crop, Summer bag day). ``core`` blocks stop
# potatoes on D25/D26 so they ripen on Summer D1/D2 and take corn that morning; the two ``late`` blocks
# sow potatoes to D29/D30 (ripe Summer D5/D6), then take the Summer bag. (13,28) keeps the ring shape
# the pin's D2 potatoes hoed (hoed soil never reverts).
BLOCK_PLAN_ROWS: Tuple[Tuple[Tuple[int, int], str, str, int, str, int], ...] = (
    ((13, 28), "ring", "core", 25, "corn", 1),
    ((8, 35), "west", "core", 26, "corn", 2),
    ((12, 35), "east", "core", 25, "corn", 1),
    ((4, 35), "east", "core", 26, "corn", 2),
    ((8, 39), "west", "core", 25, "corn", 1),
    ((16, 35), "west", "core", 26, "corn", 2),
    ((12, 39), "east", "core", 25, "corn", 1),
    ((19, 28), "west", "core", 26, "corn", 2),
    ((4, 39), "east", "core", 25, "corn", 1),
    ((8, 43), "west", "core", 26, "corn", 2),
    ((20, 35), "east", "core", 25, "corn", 1),
    ((16, 39), "west", "core", 26, "corn", 2),
    ((12, 43), "east", "late", 29, "tomato", 5),
    ((4, 43), "east", "late", 30, "tomato", 6),
)
# Year-2 plots: the next six lattice blocks by pickup-to-bin frames (``block_calendar`` ranking over the
# cleared farm; a field-side bin or the sprinkler would justify more, see docs/plan.md). Year 1 never sows
# them (``calendar_bag``), because its calendar is tuned for the first fourteen. Year 2 starts with the
# first fourteen hoed and every block open on Spring D1, so all twenty sow potatoes from D1 to their last
# day (alternating D25/D26, ripe Summer D1/D2) and take corn that morning, like the Year-1 core blocks.
Y2_BLOCK_PLAN_ROWS: Tuple[Tuple[Tuple[int, int], str, str, int, str, int], ...] = (
    ((8, 47), "west", "core", 26, "corn", 2),
    ((24, 35), "west", "core", 25, "corn", 1),
    ((20, 39), "east", "core", 26, "corn", 2),
    ((16, 43), "west", "core", 25, "corn", 1),
    ((12, 47), "east", "core", 26, "corn", 2),
    ((4, 47), "east", "core", 25, "corn", 1),
)
Y2_ONLY_CENTERS = frozenset(row[0] for row in Y2_BLOCK_PLAN_ROWS)   # Year 2 sows them only with ``HARVEST_Y2_BLOCKS=1`` (unproven: see docs/plan.md)
_NOTCH_LAYOUT = {"east": "eight_east_notch", "west": "eight_west_notch", "ring": "eight_tile_ring"}

RING_LAYOUT_ENV = "HARVEST_RING_LAYOUT"
RING_LAYOUTS: Dict[str, Tuple[Tuple[int, int], ...]] = {
    "legacy": LEGACY_POCKET_PLANT_CENTERS,
    "packed": PACKED_POCKET_PLANT_CENTERS,
    # Closest-first prefixes, for finding how many rings a day can pay for (every prefix stays reachable).
    "packed12": PACKED_POCKET_PLANT_CENTERS[:12],
    "packed18": PACKED_POCKET_PLANT_CENTERS[:18],
    "blocks": tuple(row[0] for row in BLOCK_PLAN_ROWS),
}


def is_packed_layout(name: Optional[str] = None) -> bool:
    """The lattice layouts (packed rings or notch blocks): they share the lane walks and the tour."""
    return (name or ring_layout_name()).startswith(("packed", "blocks"))
RING_LAYOUT_DEFAULT = "legacy"


def ring_layout_name() -> str:
    """The layout this process runs. Read once at import, like the site data it selects."""
    name = os.environ.get(RING_LAYOUT_ENV, RING_LAYOUT_DEFAULT).strip().lower()
    if name not in RING_LAYOUTS:
        raise ValueError(f"unknown {RING_LAYOUT_ENV}={name!r}; choose from {', '.join(RING_LAYOUTS)}")
    return name


POCKET_PLANT_CENTERS: Tuple[Tuple[int, int], ...] = RING_LAYOUTS[ring_layout_name()]
# Extend the legacy west strip only after the first economy year. These bare-ground sites preserve
# x=6 and the bin column, avoid the pasture, and add 24 plants without moving the opening's plots.
SCORE_WEST_STRIP_CENTERS = ((9, 45), (4, 45), (9, 50))
# Year 3 Spring is the last window for potatoes and turnips (496 each is 31 points apiece) and twelve sites
# only reached 386 and 390 in ``fixD``: the sites, not the stamina, were the limit (the daily watering was
# ~90 tiles of two bars). These bare 5x5 sites sit east and south of the west strip on the pitch-5 grid, clear
# of the pasture patch (x 14-21, y 33-41), the lanes (x=6, y=33, y=37) and the stump at (13-15, 49-50), nearest
# the bin first. Year 3 only, so the verified Year 1-2 timeline is untouched.
SCORE_Y3_EXTRA_CENTERS = ((14, 45), (24, 35), (19, 45), (24, 40), (24, 45), (19, 50), (4, 50), (14, 55),
                          (24, 50), (9, 55), (29, 40), (29, 45))
Y3_EXTRA_SITES_ENV = "HARVEST_Y3_EXTRA_SITES"
Y3_SUMMER_SITES_ENV = "HARVEST_Y3_SUMMER_SITES"      # "1" keeps the extra sites through Summer (an A/B)
Y3_FIRST_YEAR = 2          # ``year`` is 0-based: Year 3


Y3_SUMMER_EXTRA_ENV = "HARVEST_Y3_SUMMER_EXTRA_SITES"   # how many extra sites Summer Y3 keeps (default 0: only the twelve)


def y3_summer_extra_site_count() -> int:
    """Extra Year-3 sites kept through Summer for corn and tomatoes (an A/B; 0 = the twelve base sites only)."""
    try:
        return max(0, int(os.environ.get(Y3_SUMMER_EXTRA_ENV, "0")))
    except ValueError:
        return 0


def y3_extra_site_count() -> int:
    """How many of ``SCORE_Y3_EXTRA_CENTERS`` Year 3 sows (env switch for an A/B; 0 restores twelve sites)."""
    try:
        wanted = int(os.environ.get(Y3_EXTRA_SITES_ENV, str(len(SCORE_Y3_EXTRA_CENTERS))))
    except ValueError:
        wanted = len(SCORE_Y3_EXTRA_CENTERS)
    return max(0, min(len(SCORE_Y3_EXTRA_CENTERS), wanted))
_BLOCK_ROWS: Dict[Tuple[int, int], tuple] = (
    {row[0]: row for row in BLOCK_PLAN_ROWS + Y2_BLOCK_PLAN_ROWS} if ring_layout_name() == "blocks" else {}
)


def active_plot_centers(year: int = 0, season: Optional[int] = None) -> Tuple[Tuple[int, int], ...]:
    """The plots this year sows: ``POCKET_PLANT_CENTERS``, plus the Year-2 blocks from Year 2 under ``blocks``.

    The Year-3 extra sites are for the last Spring's potatoes and turnips: pass ``season`` and from Summer on they
    drop out, so their ground is no longer reserved from the whole-field cover (twelve sites grow all the corn and
    tomatoes the ending needs; the extra eight to twelve only made the final fill's backlog larger).
    """
    if int(year) >= 1 and ring_layout_name() == "blocks" and os.environ.get("HARVEST_Y2_BLOCKS", "0") == "1":
        return POCKET_PLANT_CENTERS + tuple(row[0] for row in Y2_BLOCK_PLAN_ROWS)
    from harvest.planner.run_strategy import active

    if int(year) >= 1 and ring_layout_name() == "legacy" and active().balance_crops:
        spring = season is None or int(season) == 0 or os.environ.get(Y3_SUMMER_SITES_ENV, "0") == "1"
        extra = SCORE_Y3_EXTRA_CENTERS[:y3_extra_site_count()] if int(year) >= Y3_FIRST_YEAR and spring else ()
        if not extra and int(year) >= Y3_FIRST_YEAR:
            extra = SCORE_Y3_EXTRA_CENTERS[:min(y3_extra_site_count(), y3_summer_extra_site_count())]
        return POCKET_PLANT_CENTERS + SCORE_WEST_STRIP_CENTERS + extra
    return POCKET_PLANT_CENTERS


def ram_year(ram) -> int:
    from harvest.core.ram_catalog import read_ram_value

    return int(read_ram_value(ram, "year", raw=True) or 0)


def plot_layout_name(center: Tuple[int, int], year: int = 0) -> str:
    """``CROP_LAYOUTS`` name of the plot at ``center``: a notch block, else the 8-around ring."""
    row = _BLOCK_ROWS.get(tuple(center))
    if row and int(year) >= 1:
        return "seven_east_access" if row[1] == "east" else "seven_west_access"
    return _NOTCH_LAYOUT[row[1]] if row else "eight_tile_ring"


def plot_crop_tiles(center: Tuple[int, int], year: int = 0) -> Tuple[Tuple[int, int], ...]:
    """The 8 cells a bag sows for the plot at ``center`` (the notch or the ring centre stays bare)."""
    from harvest.planner.crop_planner import CROP_LAYOUTS

    return tuple(CROP_LAYOUTS[plot_layout_name(center, year)].crop_tiles(center))


def count_plot_tiles(ram, center: Tuple[int, int], ids: FrozenSet[int]) -> int:
    from harvest.tasks.nav import get_tile_at

    return sum(1 for tx, ty in plot_crop_tiles(center, ram_year(ram)) if int(get_tile_at(ram, tx, ty)) in ids)


def calendar_bag(center: Tuple[int, int], season: int, day: int, year: int = 0) -> Optional[str]:
    """The block calendar's bag for ``center`` today; ``"any"`` for plots outside the calendar."""
    row = _BLOCK_ROWS.get(tuple(center))
    if row is None:
        return "any"
    if row[0] in Y2_ONLY_CENTERS and int(year) < 1:
        return None
    from harvest.planner.block_calendar import BlockPlan, sow_crop

    _c, notch, role, last_sow, crop, plant_day = row
    plan = BlockPlan(center=row[0], notch=notch, role=role, spring_last_sow=last_sow,
                     summer_crop=crop, summer_plant_day=plant_day)
    return sow_crop(plan, season, day)


def crop_working_tiles() -> FrozenSet[Tuple[int, int]]:
    """Keep grass outside crop footprints, perimeter stands and the bin lane."""
    return frozenset(
        (cx + dx, cy + dy)
        for cx, cy in active_plot_centers(1)
        for dy in range(-2, 3) for dx in range(-2, 3)
    ) | SHIP_BIN_STAND_TILES


# Tile IDs a fresh potato seed can still be sown onto: bare farm dirt plus
# freshly / recently tilled soil. A ring that is 7/8 crops + 1 unreachable
# structure tile has 0 of these left -> no real capacity.
_POCKET_BARE_TILE_IDS: FrozenSet[int] = frozenset({0x00, 0x01, 0x02})
# Weed (0x03) and stone (0x04) are 1x1 debris the establish skill's
# ``clear_plant_stands`` lifts away; they do not box a ring in.
_POCKET_LIFTABLE_TILE_IDS: FrozenSet[int] = frozenset({0x03, 0x04})
# Ripe tiles in a ring that make it count as "about to be empty".
_RIPE_REPLANT_MIN = 6
# Ripe potato (0x60/0x61) and turnip (0x6E/0x6F): they are replanted, not regrown.
_ONE_SHOT_RIPE_TILES: FrozenSet[int] = frozenset({0x60, 0x61, 0x6E, 0x6F})


SOON_RIPE_ENV = "HARVEST_Y3_SOON_RIPE"
SPRING_LAST_SOW = 26     # ``crop_planner.SPRING_LAST_SOW_DAY``: a potato sown later ripens past Summer D2


def shop_closed_days_ahead(ram) -> int:
    """Consecutive days from tomorrow on that the seed shop is shut: Sat+Sun (a Friday) or a festival eve."""
    from harvest.calendar_schedule import festival_on
    from harvest.core.ram_catalog import read_ram_value

    weekday = int(read_ram_value(ram, "weekday"))
    year, season, day = (int(read_ram_value(ram, key)) for key in ("year", "season", "day"))
    closed = 0
    for ahead in (1, 2, 3):
        shut = (weekday + ahead) % 7 in (0, 6) or (day + ahead <= 30 and festival_on(year, season, day + ahead) is not None)
        if not shut:
            break
        closed += 1
    return closed


def soon_ripe_nights(ram) -> int:
    """Nights a Year-3 Spring shop trip looks ahead: rings that ripen while the shop is shut need their bag today.

    ``fixD`` Y3 Spring D24 (Friday) bought nothing, six turnip rings ripened on D25 and the shop was shut D25-D26,
    so they sat empty until the last sowing day (D26) had passed. Zero outside Year 3 Spring and past the last
    day a bag can still ripen (``SPRING_LAST_SOW``).
    """
    from harvest.core.ram_catalog import read_ram_value

    if os.environ.get(SOON_RIPE_ENV, "1") == "0":
        return 0
    if int(read_ram_value(ram, "year", raw=True) or 0) < Y3_FIRST_YEAR or int(read_ram_value(ram, "season")) != 0:
        return 0
    nights = shop_closed_days_ahead(ram)
    return nights if nights and int(read_ram_value(ram, "day")) + nights <= SPRING_LAST_SOW else 0


def soon_ripe_tiles(nights: int) -> FrozenSet[int]:
    """Potato (0x60/0x61) and turnip (0x6E/0x6F) ids within ``nights`` watered nights of ripe (+2 ids a night)."""
    return frozenset(range(0x60 - 2 * nights, 0x62)) | frozenset(range(0x6E - 2 * nights, 0x70))


def _ring_bare_capacity(ram, center: Tuple[int, int]) -> int:
    """Count 8-ring tiles a seed bag could still be sown onto."""
    from harvest.tasks.crop_geometry import (
        FRESH_TILLED,
        WATERED_TILLED,
        plot_tiles,
    )
    from harvest.tasks.nav import get_tile_at

    sowable = _POCKET_BARE_TILE_IDS | _POCKET_LIFTABLE_TILE_IDS | {FRESH_TILLED, WATERED_TILLED}
    return count_plot_tiles(ram, center, frozenset(sowable))


def pocket_plant_target(ram) -> Optional[Tuple[int, int]]:
    """First pocket center that still needs *and can receive* a full bag.

    A center qualifies only when its 8-ring is (a) not already fully planted
    and (b) still has at least ``PLOT_RING_SIZE - count_ring_planted`` bare /
    tilled tiles a seed can land on. A ring boxed in by structure tiles it
    can never till has zero real capacity and is skipped (treated as done).
    Returns ``None`` when no center qualifies.
    """
    targets = pocket_plant_targets(ram)
    return targets[0] if targets else None


def pocket_plant_targets(ram, *, replant_ripe: bool = False) -> Tuple[Tuple[int, int], ...]:
    """Every pocket center that still needs *and can receive* a full bag.

    ``pocket_plant_target`` answers "which ring next"; this answers "how many
    bags does the farm want today". On a harvest day every ring comes back
    empty at once, so a one-bag shop trip replants one of them and the rest
    idle until tomorrow's trip.

    ``replant_ripe`` is for the morning shop decision, which runs before the
    harvest: a ripe ring of one-shot Spring crops (potato/turnip) is about to
    be empty, so it already wants its bag. That holds in Spring (the replant)
    and in Summer (the last Spring sowings ripen on D1, and corn planted by D3
    still gets six harvests). Ripe Summer crops regrow and never count.
    """
    if ram is None:
        return ()
    from harvest.tasks.crop_skills import PLANTED_OR_WET, PLOT_RING_SIZE
    from harvest.core.ram_catalog import read_ram_value

    from harvest.planner import run_strategy
    from harvest.tasks.nav import get_tile_at

    targets = []
    season = int(read_ram_value(ram, "season"))
    day = int(read_ram_value(ram, "day") or 0)
    year = int(read_ram_value(ram, "year", raw=True) or 0)
    summer = season == 1
    # A strategy that caps shipping leaves a capped one-shot crop on the plant, so its ring
    # never empties and a morning bag for it would sit unplanted.
    replant_ripe = replant_ripe and run_strategy.one_shot_replant_ok(ram)
    if season == 0 and not run_strategy.one_shot_sowing_ok(ram):
        return ()       # the Spring potatoes already planted or shipped reach the cap
    soon_nights = soon_ripe_nights(ram) if replant_ripe and season == 0 else 0
    soon = soon_ripe_tiles(soon_nights) if soon_nights else frozenset()
    for center in active_plot_centers(year, season):
        if calendar_bag(center, season, day, year) is None:
            continue        # the block calendar: past its last Spring sowing, or no Summer bag yet
        size = len(plot_crop_tiles(center, year))
        planted = count_plot_tiles(ram, center, PLANTED_OR_WET)
        if planted >= size:
            if (
                replant_ripe
                and season <= 1
                and count_plot_tiles(ram, center, _ONE_SHOT_RIPE_TILES) >= _RIPE_REPLANT_MIN
                and not run_strategy.within_ship_cap(ram, [
                    t for t in plot_crop_tiles(center, year) if get_tile_at(ram, *t) in _ONE_SHOT_RIPE_TILES
                ])[1]
            ):
                targets.append(center)
            elif soon and count_plot_tiles(ram, center, soon) >= _RIPE_REPLANT_MIN:
                targets.append(center)      # ripens while the shop is shut: the bag is bought before the weekend
            continue
        # A bag sown from the centre spends a use on every occupied cell, so
        # one or two open cells never earn the seed back (Summer corn delays
        # livestock too). Partly harvested rings wait for the rest to ripen.
        if size - planted < 3:
            continue
        if not ring_center_reachable(ram, center):
            continue
        if _ring_bare_capacity(ram, center) >= size - planted:
            targets.append(center)
    return tuple(targets)


def ring_center_reachable(ram, center: Tuple[int, int]) -> bool:
    """True when the sowing stand (the centre) opens onto a walkable side cell.

    The bag is sown standing on the centre. Corners do not count: the farmer
    walks in four directions, so a ring whose open cells are only corners
    walls its centre in, and ``nav_pocket_plant`` timed out on five straight
    Spring days (D19-D24, nine rings), each aborting the day's watering.
    """
    from harvest.core.tile_catalog import FARM_WALKABLE
    from harvest.tasks.nav import get_tile_at

    cx, cy = center
    layout = plot_layout_name(center, ram_year(ram))
    if layout != "eight_tile_ring":
        # A notch block sows from its hoed centre and is entered through the bare notch.
        from harvest.tasks.crop_geometry import FRESH_TILLED, WATERED_TILLED

        notch = (cx + (1 if layout in {"eight_east_notch", "seven_east_access"} else -1), cy)
        centre_ok = int(get_tile_at(ram, cx, cy)) in FARM_WALKABLE | {FRESH_TILLED, WATERED_TILLED}
        return centre_ok and int(get_tile_at(ram, *notch)) in FARM_WALKABLE
    if int(get_tile_at(ram, cx, cy)) not in FARM_WALKABLE:
        return False
    return any(
        int(get_tile_at(ram, cx + dx, cy + dy)) in FARM_WALKABLE
        for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0))
    )


# Hoe swing and watering-can use each cost 2 stamina (ROM, see
# docs/SPRING_ECONOMY.md 2.2) and the farmer starts a day with 100. A ring is 8
# hoe swings once and 8 waterings every day, so stamina, not frames, caps the
# ring count: six rings cost ~96 stamina of watering alone (ROM
# ``LN_y1econ_rings_d3_6planted_full``: 47 waters, 7.1k frames, 100 -> 2).
STAMINA_PER_TOOL_USE = 2


def pocket_water_reserve(ram) -> int:
    """Stamina the standing crops still need today (2 per dry growing tile)."""
    from harvest.tasks.crop_geometry import WATERABLE_TILES

    dry = sum(
        count_plot_tiles(ram, center, frozenset(WATERABLE_TILES))
        for center in active_plot_centers(ram_year(ram))
    )
    return dry * STAMINA_PER_TOOL_USE


# One hot-spring soak refills the bar. ``CROP_WATER`` reports ``stamina_low``
# when it cannot fund every dry tile and the day plan splices one soak, so a
# day with the spa allowed can pay for about two bars of tool use.
SPA_REFILL_STAMINA = 100


def ring_establish_affordable(ram, center: Tuple[int, int], *, spa_refills: int = 0) -> bool:
    """True when today's stamina covers this ring's hoe swings *and* the
    watering the farm already owes.

    A new ring's own first watering may slip a day (a crop only stalls), but
    the standing crops are not starved to open another ring. Replanting after
    a Spring harvest leaves watered soil (0x08) and costs no hoe at all.
    ``spa_refills`` soaks still allowed today add ``SPA_REFILL_STAMINA`` each.
    """
    from harvest.core.ram_catalog import read_ram_value
    from harvest.tasks.crop_skills import HOED_OR_PLANTED, PLOT_RING_SIZE

    stamina = int(read_ram_value(ram, "stamina") or 0) + max(0, spa_refills) * SPA_REFILL_STAMINA
    unhoed = len(plot_crop_tiles(center, ram_year(ram))) - count_plot_tiles(ram, center, HOED_OR_PLANTED)
    need = max(0, unhoed) * STAMINA_PER_TOOL_USE + pocket_water_reserve(ram)
    return stamina >= need


def next_unplanted_pocket_center(ram) -> Optional[Tuple[int, int]]:
    """First pocket center that needs planting and can receive it, else ``None``.

    Delegates to :func:`pocket_plant_target`. There is deliberately no
    "fall back to the last center" behaviour: a ring with no bare capacity
    must be accepted as done rather than retried forever.
    """
    return pocket_plant_target(ram)


def planted_pocket_centers(ram) -> Tuple[Tuple[int, int], ...]:
    """Every pocket center whose 8-ring is planted (water targets, in order)."""
    from harvest.tasks.crop_skills import PLANTED_OR_WET

    return tuple(
        center
        for center in active_plot_centers(ram_year(ram))
        if count_plot_tiles(ram, center, PLANTED_OR_WET) > 0
    )


def pocket_water_center(ram) -> Tuple[int, int]:
    """The pocket ring a single CROP_WATER pocket pass should water.

    The most-recently established planted ring; the west pocket when nothing
    is planted yet.
    """
    planted = planted_pocket_centers(ram)
    return planted[-1] if planted else WEST_POCKET_PLANT_CENTER


# Named water pockets: (name, water_tile_id, fills_can, sample_cells)
FARM_WATER_POCKETS: Tuple[Tuple[str, int, bool, Tuple[Tuple[int, int], ...]], ...] = (
    ("main_pond", 0xF0, True, ((32, 32), (33, 32), (31, 32), (34, 32))),
    ("north_stream_f1", 0xF1, False, ((13, 16), (14, 16))),
    ("shipping_ditch_f2", 0xF2, False, ((8, 29), (9, 29), (8, 30), (9, 30))),
    ("north_pool_f7", 0xF7, False, ((24, 5), (26, 5), (28, 6))),
    ("north_stream_f8", 0xF8, False, ((18, 22),)),
    ("north_spur_f9", 0xF9, True, ((26, 12), (26, 13))),
    ("east_spur_fa", 0xFA, True, ((46, 14), (46, 15))),
    ("east_spur_fb", 0xFB, True, ((49, 36), (49, 37))),
    ("south_stream_fc", 0xFC, True, ((14, 49), (14, 50))),
    ("southeast_fd", 0xFD, True, ((41, 54), (41, 55))),
)

FARM_TILEMAP_IDS: FrozenSet[int] = frozenset({0x00, 0x01, 0x02, 0x03})
FARM_TILEMAP_NAMES: Dict[int, str] = {
    0x00: "farm",
    0x01: "farm_summer",
    0x02: "farm_fall",
    0x03: "farm_winter",
}

MOUNTAIN_NO_GO_TILES: FrozenSet[Tuple[int, int]] = frozenset({
    (20, 44),  # solid cliff/rock directly north of south entry tile (20, 45)
})

NO_GO_TILES_BY_TILEMAP: Dict[int, FrozenSet[Tuple[int, int]]] = {
    **{tilemap_id: FARM_NO_GO_TILES for tilemap_id in FARM_TILEMAP_IDS},
    **{tilemap_id: MOUNTAIN_NO_GO_TILES for tilemap_id in MOUNTAIN_TILEMAP_IDS},
}

