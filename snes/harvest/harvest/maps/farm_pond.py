"""Farm pond / water-source constants and helpers.

ROM-mapped spring-farm facts for can refill, fence access, and no-go tiles.
"""

from typing import Dict, FrozenSet, Tuple

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
}) | HORSE_BARN_WALL_TILES | COW_BARN_EAST_FACE_TILES

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

NO_GO_TILES_BY_TILEMAP: Dict[int, FrozenSet[Tuple[int, int]]] = {
    tilemap_id: FARM_NO_GO_TILES for tilemap_id in FARM_TILEMAP_IDS
}

