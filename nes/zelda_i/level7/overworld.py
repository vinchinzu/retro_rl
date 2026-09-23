"""Level 7 (Demon) overworld approach and gated entry helpers.

Gated by Whistle (L5) for pond drain and Bait/Food for hungry Goriya.
The start-to-pond hop table is usable without either item so geometry can be
verified independently; opening the pond still requires the naturally earned
Whistle.

See ``docs/LEVEL7_ROUTE.md``.
"""

from __future__ import annotations

from zelda_i.dungeon.hop_controller import ow_edge_band_step

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_idle_action
from zelda_i.overworld.common import (
    EDGE_EAST_X,
    EDGE_NORTH_Y,
    EDGE_SOUTH_Y,
    EDGE_WEST_X,
)
from zelda_i.overworld.graph import ScreenHop, path_screens_from_hops
from zelda_i.overworld.path import OverworldPathController
from zelda_i.ram import (
    ADDR_CANDLE,
    ADDR_FOOD,
    ADDR_WHISTLE,
    PLAY_MODE,
    ZeldaSnapshot,
    read_u8,
)
from zelda_i.walk.physics import OccupancyGrid, OccupancyWalker

from zelda_i.anchors import (
    SCREEN_BRACELET_ARMOS,
    SCREEN_LEVEL4_ENTRANCE,
    SCREEN_LEVEL6_ENTRANCE,
    SCREEN_LEVEL7_BAIT_SHOP_HYP,
    SCREEN_LEVEL7_ENTRY_ROOM,
    SCREEN_LEVEL7_POND_HYP,
    SCREEN_MAGICAL_SWORD_GRAVE,
    TF_BIT_L7 as LEVEL7_TRIFORCE_BIT,
)

SOURCE_HYPOTHESIS = True
LEVEL7 = 7
# Source / Data Crystal style: candle 1=blue, 2=red (confirm live).
CANDLE_RED_PLANNED = 2

# Executable pond approach: reuse the live west-forest path through 0x55,
# join 0x64, then use the source pond suffix directly from 0x54.  Mapping the
# pond does not require the optional bait-shop detour.
LEVEL7_POND_APPROACH_HOPS: tuple[ScreenHop, ...] = (
    ScreenHop(0x78, "RIGHT", align_y=140),
    ScreenHop(0x68, "UP", align_x=48),
    ScreenHop(0x58, "UP", align_x=48),
    ScreenHop(0x57, "LEFT", y_band_lo=148, y_band_hi=162),
    ScreenHop(0x56, "LEFT", y_band_lo=148, y_band_hi=162),
    ScreenHop(0x55, "LEFT", align_y=133),
    ScreenHop(0x65, "DOWN", align_x=112),
    ScreenHop(0x64, "LEFT", align_y=141),
    ScreenHop(0x54, "UP"),
)

# DEAD SPUR — do not use as PostLevel6OverworldController default.
# 0x22↓0x32 is the L6 approach reverse into the mountain-locked pocket
# 0x22/0x32/0x33/0x23/0x24/0x25. No south walk to shop 0x34 or pond 0x42.
# Kept for bait-micro tests (bait_32_north_action, bait_24_east_action).
POST_L6_TO_BAIT_HOPS: tuple[ScreenHop, ...] = (
    ScreenHop(0x32, "DOWN", align_x=112),
    ScreenHop(0x33, "RIGHT", align_y=141),
    ScreenHop(0x23, "UP", align_x=208),
    ScreenHop(SCREEN_BRACELET_ARMOS, "RIGHT", align_y=141),
    ScreenHop(0x25, "RIGHT", align_y=141),
)
POST_L6_TO_BAIT_SCREENS: tuple[int, ...] = path_screens_from_hops(
    SCREEN_LEVEL6_ENTRANCE, POST_L6_TO_BAIT_HOPS
)
# Dead belief: (112,125) on 0x22 is the L6 leave. Mode 16 → dungeon.
L6_CAVE_MOUTH_X = 112
L6_CAVE_MOUTH_Y = 125
L6_CAVE_MOUTH_TOL = 16
# y=141 is the edge of the cave-mouth reentry box (125+16). Travel south of
# it so Ghini knockback along the west wall cannot count as l6_cave_mouth.
POND_22_WEST_Y = 157
POND_22_WEST_GOAL: tuple[int, int] = (EDGE_WEST_X + 4, POND_22_WEST_Y)
# Live miss l7_p22w7: 0x22 LEFT toward Magical Sword grave 0x21 is west
# mountain. Leftover 0x22 (90,165) tile 38, occupancy 41 misses, west edge
# unreachable. PNG: recordings/l7_p22w7_final.png. 0x22 is a boxed
# mountain graveyard: north is the L6 cave (UP = mode 16), west mountain,
# south corridor is the only walk-off (live L6 reverse 0x22↓0x32).
# Dead: 0x22 south leftover (120,221) UP is tile 216 (east wall of x=112).
# Dead: l7_p22w5 (97,141) is inside the cave-mouth reentry box (tol=16).
POST_L6_22_WEST_HOP = ScreenHop(
    SCREEN_MAGICAL_SWORD_GRAVE, "LEFT", align_y=POND_22_WEST_Y
)
# Greened prefix (not the dead 0x25 pocket): L6 reverse 0x22↓0x32→0x33↑0x23
# →0x24 then 0x24 UP x=160 → 0x14 (l7_p24n) then 0x14 LEFT y=165-189 → 0x13
# (l7_p14w 1/1, leftover 0x13 (240,189)). Do not RIGHT 0x24→0x25.
# Next: 0x13 LEFT at the south sand (y=189, south of the Armos) toward 0x12
# (l7_p13w 1/1, leftover 0x12 (240,189)). Do not DOWN at (240,189)
# (south mountain). 0x12 LEFT y=165-189 is DEAD (l7_p12w timeout, leftover
# 0x12 (16,181) tile 218 west wall). _after_hops succeeds only on pond 0x42.
#
# 2026-09-04 sitting (H1/H2 recon, rr-8t4.1 residual; scratch/pond/probe_*):
#
# H1 -- Recorder overworld warp is REAL (searched + confirmed live, contrary
# to the "nothing happens" first read): blowing the already-owned Recorder
# (B-slot 5, `dungeon.pause_select`) on any *non-entrance* overworld screen
# triggers a whirlwind-carry cutscene (mode 5→6→7→4→5, Link auto-walks off
# the screen edge with no player input) to the entrance of a *completed*
# dungeon, cycling forward/backward through 1..6 by UP/DOWN facing
# (`probe_recorder_warp_cycle.py`). Confirmed cycle from 0x24 facing DOWN:
# 0x22(L6)→0x0B(L5)→0x45(L4)→0x74(L3)→0x3C(L2) (`rw_cycle_down2`, 4082f,
# writes=0). It does **not** fire reliably from the dragon-mouth screen
# 0x22 itself (`probe_recorder_warp.py`, 8 blows, zero change) -- entrance
# screens appear to suppress it. Each destination is a dungeon *door*
# screen, not the pond approach band, and advancing the cycle needs 1-3
# blows per step once a screen's exit is boulder/mountain-blocked (e.g. it
# stalls bouncing against 0x3C's east wall for several blows). **Not
# adopted**: no observed destination lands on the green pond band, and the
# per-blow advance is not perfectly deterministic. If a future sitting
# finds a completed-dungeon entrance screen that is itself on
# `LEVEL7_POND_APPROACH_HOPS`, this is worth revisiting.
#
# H2 -- 0x12's `$6530` tile map (`probe_dump_ow_tilemap.py`) shows a
# non-mountain gap at tile-cols 6-7 (x=48-63) running the whole north edge.
# **LIVE 2/2 byte-identical**: 0x12 UP at x=48 (tol 3) → screen 0x02
# `(48,61)` mode 7, f=2424 total (`probe_12_north_gap.py --gap-x 48`,
# `l7_12north_confirm1`/`confirm2`, writes=0). x=56 (one tile column over)
# is DEAD (wall). Chasing this further west/south: `0x32` LEFT with a
# y-band (128,183) plus the existing `bait_32_north_action` x=112 realign
# (the arrival point (120,61) is one tile off the true x=112 gap column and
# freezes DOWN forever otherwise -- this is the documented
# `l7_bait_from_l6 leftover (120,61)` bug, previously only wired for
# `hop.target == 0x33`) reaches **0x31** `(240,133)`; 0x31 DOWN
# `align_x=120` reaches **0x41** `(120,61)` (`probe_32_west_hop.py`,
# `hop_1_31`/`hop_2_41` notes). **DEAD end: 0x41 → 0x42 RIGHT** is a solid,
# full-height wall (tile `0xc4-0xc7` for all 22 tile-rows of 0x41's east
# edge, confirmed both by the `$6530` dump and a live 30,000-frame stuck
# test parked at `(128,141)` with `--infinite-life`, never crossing).
# 0x32's own south edge is independently reconfirmed here as 100% mountain
# across all 32 columns (row21), matching the prior sweep-based "no south
# exit" finding but now from a full tile-map read, not a sample sweep.
# **Net result: the entire west-of-L6 column (0x12→0x02, and
# 0x32→0x31→0x41) is now fully mapped and does not reach the pond band.**
# H3 (the long way round via the L5→L6 approach reversed) was never needed:
# **H1 IS the route** (2026-09-05, rr-8t4.1).  See `level7/warp.py` and
# `WARP_JOIN_TO_POND_HOPS` below -- the warp cycle's L4 island door `0x45`
# is one screen NORTH of `0x55`, which is already on the green
# `LEVEL7_POND_APPROACH_HOPS`.  Post-L6 → pond `0x42` is live 2/2
# (`scratch/pond/probe_recorder_warp_full_route.py`, tags `rw_full_route_t2`
# / `_t3`, 4913f each, writes=0).  `0x74 → 0x64` UP is DEAD: `0x74`'s whole
# north edge is mountain across all 32 tile columns (`$6530` dump plus a
# live 10-column sweep, every candidate stuck at y=85).
# POST_L6_TO_POND_HOPS below is kept as the *walk-only* prefix record; the
# spine now walks only its first four hops (`POST_L6_TO_WARP_HOPS`) and
# warps out of the pocket from `0x24`.
POST_L6_TO_POND_HOPS: tuple[ScreenHop, ...] = (
    ScreenHop(0x32, "DOWN", align_x=112),
    ScreenHop(0x33, "RIGHT", align_y=141),
    ScreenHop(0x23, "UP", align_x=208),
    ScreenHop(SCREEN_BRACELET_ARMOS, "RIGHT", align_y=141),
    ScreenHop(0x14, "UP", align_x=160),
    ScreenHop(0x13, "LEFT", y_band_lo=165, y_band_hi=189),
    ScreenHop(0x12, "LEFT", y_band_lo=165, y_band_hi=189),
)
POST_L6_TO_POND_SCREENS: tuple[int, ...] = path_screens_from_hops(
    SCREEN_LEVEL6_ENTRANCE, POST_L6_TO_POND_HOPS
)
BAIT_APPROACH_MAX_FRAMES = 15_000
# l7_p22w5 leftover (97,141): occupancy knockback re-entered the mouth box
# (tol=16). Block the cave opening so BFS sweeps the west rock instead.
POND_22_CAVE_BLOCKED: frozenset[tuple[int, int]] = frozenset(
    (x, y)
    for x in range(L6_CAVE_MOUTH_X - 24, L6_CAVE_MOUTH_X + 25)
    for y in range(EDGE_NORTH_Y, L6_CAVE_MOUTH_Y + 1)
)
# Live L6 0x32↑0x22 is align_x=112. Fixture leftover arrived 0x32 (120,61):
# recover_off_edge DOWN at x=120 is the east wall of that corridor (tile 216).
BAIT_32_CORRIDOR_X = 112
BAIT_32_NORTH_Y = 80
# l7_bait_33up leftover 0x24 (16,189): DOWN is south mountain, not 0x34.
# l7_bait_24east leftover (0,141): occupancy xmin=14 trapped the west edge.
# l7_bait_24sand leftover (25,181): occupancy boxed in the SW mountain corner.
# l7_bait_24belt leftover (160,189): DOWN at the north-ladder x is mountain.
# l7_bait_24se leftover (208,189): DOWN at the SE mouth is mountain (tile 206).
# Live L6 0x24→0x23 is LEFT @ y=141; 0x25 is that band reversed. Never DOWN
# at y=189.
BAIT_24_EAST_Y = 141
BAIT_24_Y_TOL = 4
# 0x64 north gap to 0x54 (probe_64_north_to_54): open column at x≈60.
BAIT_64_GAP_X = 60
# 0x52 boulder field (probe_52_wall): climb the open west column x≈48 to the
# mid-band y≈120, traverse RIGHT to x≈132, then UP funnels Link through the
# boulder-wall gap (~x128) into the x≈112 north gap to the pond 0x42.
POND_52_CLIMB_X = 48
POND_52_GAP_X = 132
POND_52_MIDBAND_Y = 122

LEVEL7_POND_HOPS: tuple[ScreenHop, ...] = LEVEL7_POND_APPROACH_HOPS + (
    ScreenHop(0x53, "LEFT", align_y=141),
    # 0x53 west: central y≈141 is tree-blocked; lower gap is y≈189.
    # v9 leftover (224,173): align_y-first DOWN is blocked (hop10_ay).
    ScreenHop(0x52, "LEFT", align_y=189),
    ScreenHop(SCREEN_LEVEL7_POND_HYP, "UP", align_x=112),
)
LEVEL7_POND_SCREENS: tuple[int, ...] = path_screens_from_hops(0x77, LEVEL7_POND_HOPS)

# --- H1 Recorder-warp escape from the mountain-locked post-L6 pocket ---
# The spine walks only the first four POST_L6_TO_POND_HOPS (0x22↓0x32→0x33
# ↑0x23→0x24), then blows the owned Recorder on 0x24 -- a NON-entrance
# screen; door screens suppress the warp -- facing DOWN until the cycle
# lands on the L4 island door 0x45 (`level7.warp.RecorderWarpController`).
WARP_LAUNCH_SCREEN = 0x24
POST_L6_TO_WARP_HOPS: tuple[ScreenHop, ...] = POST_L6_TO_POND_HOPS[:4]
POST_L6_TO_WARP_SCREENS: tuple[int, ...] = path_screens_from_hops(
    SCREEN_LEVEL6_ENTRANCE, POST_L6_TO_WARP_HOPS
)
WARP_ISLAND_SCREEN = SCREEN_LEVEL4_ENTRANCE
# The whirlwind drops Link on 0x45 at (128,141) -- the raft-dock column.
# x=128 is a clear open corridor the full height of 0x55 (probe_55_dump.py,
# 74f raw DOWN, no realign).  Do NOT reuse the stock LEVEL7_POND_HOPS
# `ScreenHop(0x65, "DOWN", align_x=112)` here: 112 assumes the *east*
# 0x56→0x55 arrival band and drags Link LEFT into the mid-screen house/tree
# obstacle (tile cols 14-17, rows 8-11), which parked rw_full_route_t1 at
# (128,103) for a full 30,000f budget.  Keep x=128 through 0x55→0x65.
POND_55_DOCK_X = 128
WARP_JOIN_TO_POND_HOPS: tuple[ScreenHop, ...] = (
    ScreenHop(0x55, "DOWN", align_x=POND_55_DOCK_X),
    ScreenHop(0x65, "DOWN", align_x=POND_55_DOCK_X),
) + LEVEL7_POND_HOPS[7:]
WARP_JOIN_TO_POND_SCREENS: tuple[int, ...] = path_screens_from_hops(
    WARP_ISLAND_SCREEN, WARP_JOIN_TO_POND_HOPS
)
# rr-8t4.4: peel the live warp-join at 0x54 and walk north to the Armos
# bait shop. Shop 0x34 is entered from the south. Gaps from OVERWORLD_DOORS
# recon (single-run, not 2/2): 0x54→0x44 x≈116, 0x44→0x34 x≈132.
SHOP_54_NORTH_X = 116
SHOP_44_NORTH_X = 132
SHOP_NORTH_X_TOL = 6
WARP_JOIN_TO_SHOP_HOPS: tuple[ScreenHop, ...] = WARP_JOIN_TO_POND_HOPS[:4] + (
    ScreenHop(0x44, "UP", align_x=SHOP_54_NORTH_X),
    ScreenHop(SCREEN_LEVEL7_BAIT_SHOP_HYP, "UP", align_x=SHOP_44_NORTH_X),
)
WARP_JOIN_TO_SHOP_SCREENS: tuple[int, ...] = path_screens_from_hops(
    WARP_ISLAND_SCREEN, WARP_JOIN_TO_SHOP_HOPS
)

# 0x53 east-edge vertical travel is the v9 miss.  Leave the east column
# (x>192) before descending to the hypothesized west gap, then LEFT to 0x52.
POND_53_INLAND_X = 192
POND_53_WEST_GAP_Y = 189
POND_53_Y_TOL = 4
POND_53_SEED_BLOCKED: frozenset[tuple[int, int]] = frozenset({(224, 174)})


def pond_suffix_extra_hop_action(
    snap: ZeldaSnapshot,
    hop: ScreenHop,
    *,
    swing,
    pond53_walker: OccupancyWalker,
) -> FrameAction | None:
    """Live 0x64 / 0x53 / 0x52 / shop-north micros. Only fires for hops in the table."""
    # 0x65→0x64 arrives on the east ledge at ~(232,109).  UP is blocked
    # there: descend to the open middle band, cross to the north gap and
    # climb.  probe_64_north_to_54: x≈60 is a clean open column to 0x54;
    # x≤40 stalls at y≈93; x≈120 is under the central tree isle.
    if hop.target == 0x54 and snap.screen == 0x64:
        if snap.link_x > BAIT_64_GAP_X + 6 and snap.link_y < 116:
            return swing("DOWN", "64_east_ledge_down")
        if snap.link_x > BAIT_64_GAP_X + 6:
            return swing("LEFT", "64_cross_to_north")
        if snap.link_x < BAIT_64_GAP_X - 6:
            return swing("RIGHT", "64_north_ax")
        return swing("UP", "64_north")
    if hop.target == 0x52 and snap.screen == 0x53:
        return pond_53_to_52_action(snap, walker=pond53_walker, swing=swing)
    if hop.target == SCREEN_LEVEL7_POND_HYP and snap.screen == 0x52:
        # ROM lattice to the north gap. The hand climb below pressed RIGHT
        # into a rock at (48,117) for 6608 frames (R17).
        step = ow_edge_band_step(None, snap, "UP", POND_52_GAP_X - 24, POND_52_GAP_X + 4)
        if step is not None:
            return swing(step, "52_lattice_north")
        # 0x52 rock field (probe_52_wall): climb the open west column x≈48
        # from the bottom corridor to the mid-band y≈120, traverse RIGHT to
        # x≈132, then a UP push funnels Link through the boulder-wall gap
        # (~x128) into the x≈112 north gap to pond 0x42.
        if snap.link_y > POND_52_MIDBAND_Y:
            if snap.link_x > POND_52_CLIMB_X + 6:
                return swing("LEFT", "52_to_climb_column")
            if snap.link_x < POND_52_CLIMB_X - 6:
                return swing("RIGHT", "52_climb_ax")
            return swing("UP", "52_climb")
        if snap.link_x < POND_52_GAP_X - 4:
            return swing("RIGHT", "52_traverse_midband")
        return swing("UP", "52_gap_up")
    if hop.target == 0x44 and snap.screen == 0x54:
        if abs(snap.link_x - SHOP_54_NORTH_X) > SHOP_NORTH_X_TOL:
            btn = "LEFT" if snap.link_x > SHOP_54_NORTH_X else "RIGHT"
            return swing(btn, "54_shop_ax")
        return swing("UP", "54_shop_north")
    if hop.target == SCREEN_LEVEL7_BAIT_SHOP_HYP and snap.screen == 0x44:
        if abs(snap.link_x - SHOP_44_NORTH_X) > SHOP_NORTH_X_TOL:
            btn = "LEFT" if snap.link_x > SHOP_44_NORTH_X else "RIGHT"
            return swing(btn, "44_shop_ax")
        return swing("UP", "44_shop_north")
    return None


class Level7NavPhase(Enum):
    HOP = auto()
    DONE = auto()
    FAILED = auto()


def bait_32_north_action(snap: ZeldaSnapshot, *, swing) -> FrameAction | None:
    """Leave the 0x32 north mouth via the live x=112 corridor, then DOWN.

    l7_bait_from_l6 leftover (120,61): off_north DOWN never left the cell.
    """
    if snap.screen != 0x32 or snap.link_y >= BAIT_32_NORTH_Y:
        return None
    if abs(snap.link_x - BAIT_32_CORRIDOR_X) > 5:
        btn = "LEFT" if snap.link_x > BAIT_32_CORRIDOR_X else "RIGHT"
        return swing(btn, "32_north_ax")
    return swing("DOWN", "32_north_down")


def bait_24_east_action(snap: ZeldaSnapshot, *, swing) -> FrameAction | None:
    """Live y=141 band RIGHT toward 0x25. Never DOWN at the south wall.

    l7_bait_33up leftover (16,189): DOWN is south mountain.
    l7_bait_24east leftover (0,141): occupancy xmin=14 trapped the west edge.
    l7_bait_24sand leftover (25,181): occupancy boxed in the SW corner.
    l7_bait_24belt leftover (160,189): DOWN at the north-ladder x is mountain.
    l7_bait_24se leftover (208,189): DOWN is SE mountain. UP to the band.
    """
    if snap.screen != SCREEN_BRACELET_ARMOS:
        return None
    if snap.link_x < EDGE_WEST_X:
        return swing("RIGHT", "24_west_inland")
    if snap.link_y > BAIT_24_EAST_Y + BAIT_24_Y_TOL:
        return swing("UP", "24_east_band")
    if snap.link_y < BAIT_24_EAST_Y - BAIT_24_Y_TOL:
        return swing("DOWN", "24_east_band")
    return None


def at_l6_cave_mouth(snap: ZeldaSnapshot) -> bool:
    """True on the 0x22 dungeon mouth that starts mode-16 L6 enter."""
    return (
        snap.level == 0
        and snap.screen == SCREEN_LEVEL6_ENTRANCE
        and abs(snap.link_x - L6_CAVE_MOUTH_X) <= L6_CAVE_MOUTH_TOL
        and abs(snap.link_y - L6_CAVE_MOUTH_Y) <= L6_CAVE_MOUTH_TOL
    )


def pond_22_to_21_action(
    snap: ZeldaSnapshot,
    *,
    walker: OccupancyWalker,
    swing,
) -> FrameAction | None:
    """Leave the 0x22 cave mouth without UP, then occupancy-walk west to 0x21.

    Measured leave (112,125) is the L6 mouth: UP is mode 16. Occupancy miss →
    block cell → replan; no path → stand. At the west edge defer to hop LEFT.
    """
    if snap.screen != SCREEN_LEVEL6_ENTRANCE:
        return None
    if snap.link_x <= EDGE_WEST_X + 6:
        return None
    # Measured leave (112,125): UP is L6 enter. Step off the mouth first.
    if at_l6_cave_mouth(snap) or (
        abs(snap.link_x - L6_CAVE_MOUTH_X) <= L6_CAVE_MOUTH_TOL
        and snap.link_y <= L6_CAVE_MOUTH_Y + 4
    ):
        return swing("DOWN", "22_leave_mouth")
    # South leftover (120,221): UP at x=120 is tile 216 (east wall of the
    # x=112 corridor). Climb the corridor to the west band before occupancy
    # so swing-stalls cannot poison the only north cell.
    if snap.link_y > POND_22_WEST_Y + 8:
        if abs(snap.link_x - L6_CAVE_MOUTH_X) > 5:
            btn = "LEFT" if snap.link_x > L6_CAVE_MOUTH_X else "RIGHT"
            return swing(btn, "22_south_corridor_ax")
        return swing("UP", "22_south_corridor_up")
    xy = (int(snap.link_x), int(snap.link_y))
    walker.observe(xy)
    direction = walker.next_dir(xy)
    if direction is None:
        # Occupancy 1px/swing/knockback can drop the path before the west
        # rock is actually mapped. Keep pushing LEFT until the grid is dense.
        if walker.misses < 40:
            return swing("LEFT", "22_west")
        return FrameAction(nes_idle_action(), "22_no_path_stand")
    if direction == "UP" and abs(snap.link_x - L6_CAVE_MOUTH_X) <= L6_CAVE_MOUTH_TOL:
        return swing("LEFT", "22_avoid_cave_up")
    return swing(direction, "22_west")


def make_pond_22_walker() -> OccupancyWalker:
    return OccupancyWalker(
        grid=OccupancyGrid(
            blocked=set(POND_22_CAVE_BLOCKED),
            xmin=EDGE_WEST_X,
            xmax=EDGE_EAST_X,
            ymin=EDGE_NORTH_Y,
            ymax=EDGE_SOUTH_Y,
        ),
        goal=POND_22_WEST_GOAL,
    )


def make_pond_53_walker() -> OccupancyWalker:
    return OccupancyWalker(
        grid=OccupancyGrid(
            blocked=set(POND_53_SEED_BLOCKED),
            xmin=EDGE_WEST_X,
            xmax=EDGE_EAST_X,
            ymin=EDGE_NORTH_Y,
            ymax=EDGE_SOUTH_Y,
        ),
        goal=(EDGE_WEST_X + 4, POND_53_WEST_GAP_Y),
    )


def pond_53_to_52_action(
    snap: ZeldaSnapshot,
    *,
    walker: OccupancyWalker,
    swing,
) -> FrameAction | None:
    """LEFT inland from the east edge before descending toward 0x52.

    v9 live miss: ``hop10_ay`` DOWN from ``(224,173)`` never left 0x53.
    l7_dnp_pond_53 leftover ``(176,205)``: once inland and at/below the
    hypothesized gap, push LEFT instead of occupancy-UP back to y=189.
    Occupancy miss → block cell → replan; no path → stand.
    """
    xy = (int(snap.link_x), int(snap.link_y))
    walker.observe(xy)
    if snap.link_x > POND_53_INLAND_X:
        return swing("LEFT", "53_inland_left")
    if snap.link_y >= POND_53_WEST_GAP_Y - POND_53_Y_TOL:
        return None
    direction = walker.next_dir(xy)
    if direction is None:
        return FrameAction(nes_idle_action(), "53_no_path_stand")
    reason = (
        "53_descend_west_gap"
        if direction in {"DOWN", "UP"}
        else "53_left_west_gap"
    )
    return swing(direction, reason)


@dataclass
class OverworldToLevel7PondController(OverworldPathController):
    """Walk from the start screen to the Demon pond on screen ``0x42``.

    Geometry-only: Whistle is not required.  Drain/entry is a later chapter.
    """

    phase: Level7NavPhase = Level7NavPhase.HOP
    hops: tuple[ScreenHop, ...] = LEVEL7_POND_HOPS
    require_sword: bool = True
    _pond53_walk: OccupancyWalker | None = field(
        default=None, init=False, repr=False
    )

    @property
    def failed(self) -> bool:
        return self.phase is Level7NavPhase.FAILED

    def end_screen(self) -> int:
        return self.hops[-1].target

    def _pond53_walker(self) -> OccupancyWalker:
        if self._pond53_walk is None:
            self._pond53_walk = make_pond_53_walker()
        return self._pond53_walk

    def _extra_hop_action(
        self, snap: ZeldaSnapshot, hop: ScreenHop
    ) -> FrameAction | None:
        return pond_suffix_extra_hop_action(
            snap, hop, swing=self._swing, pond53_walker=self._pond53_walker()
        )


@dataclass
class OverworldToBaitShopController(OverworldPathController):
    """Fixture-live walk: post-L6 south leftover through ``0x25``.

    ``0x24`` south is mountain. Shop ``0x34`` is still unobserved. Spine
    chapters stay fail-closed (``verified=false``). Do not enter the 0x22 cave
    mouth.
    """

    phase: Level7NavPhase = Level7NavPhase.HOP
    hops: tuple[ScreenHop, ...] = POST_L6_TO_BAIT_HOPS
    require_sword: bool = True
    max_frames: int = BAIT_APPROACH_MAX_FRAMES
    need_rupees: int = 60
    evidence: str = "hypothesis"
    route_eligible: bool = False

    @property
    def failed(self) -> bool:
        return self.phase is Level7NavPhase.FAILED

    def end_screen(self) -> int:
        return self.hops[-1].target

    def _extra_hop_action(
        self, snap: ZeldaSnapshot, hop: ScreenHop
    ) -> FrameAction | None:
        if hop.target == 0x33:
            return bait_32_north_action(snap, swing=self._swing)
        if hop.target == 0x25:
            return bait_24_east_action(snap, swing=self._swing)
        return None

    def _refuse(self, reason: str) -> FrameAction:
        self.success = False
        return self._fail(reason)

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        if self.failed:
            return FrameAction(nes_idle_action(), "failed")
        if snap.level == 6:
            return self._refuse("l6_dungeon_enter")
        if at_l6_cave_mouth(snap):
            return self._refuse("l6_cave_mouth")
        if snap.mode == 16 and snap.screen == SCREEN_LEVEL6_ENTRANCE:
            return self._refuse("l6_cave_mouth_enter")
        if snap.in_cave:
            return self._refuse("unexpected_cave")
        return super().step(snap)

    def report(self) -> dict[str, Any]:
        out = super().report()
        out.update(
            {
                "evidence": self.evidence,
                "route_eligible": self.route_eligible,
                "failed": self.failed,
                "end_screen": hex(self.end_screen()),
                "writes": 0,
            }
        )
        return out


def has_whistle(ram) -> bool:
    return bool(read_u8(ram, ADDR_WHISTLE))


def has_food(ram) -> bool:
    """Bait / Food inventory (hungry Goriya)."""
    return bool(read_u8(ram, ADDR_FOOD))


def has_red_candle(ram) -> bool:
    """Planned: candle byte == 2 means Red Candle (verify live)."""
    return read_u8(ram, ADDR_CANDLE) >= CANDLE_RED_PLANNED


def required_caps_for_entry() -> frozenset[str]:
    """Whistle required to open pond stairs (source). Food needed mid-dungeon."""
    return frozenset({"whistle"})


def required_caps_for_clear() -> frozenset[str]:
    return frozenset({"whistle", "food", "red_candle"})


def missing_entry_caps(ram) -> list[str]:
    missing: list[str] = []
    if not has_whistle(ram):
        missing.append("whistle")
    return missing


def missing_clear_caps(ram) -> list[str]:
    missing = missing_entry_caps(ram)
    if not has_food(ram):
        missing.append("food")
    return missing


def on_level7_pond_hyp(snap: ZeldaSnapshot) -> bool:
    return (
        snap.level == 0
        and snap.mode == PLAY_MODE
        and snap.screen == SCREEN_LEVEL7_POND_HYP
    )


def on_level7_bait_shop_hyp(snap: ZeldaSnapshot) -> bool:
    return (
        snap.level == 0
        and snap.mode == PLAY_MODE
        and snap.screen == SCREEN_LEVEL7_BAIT_SHOP_HYP
    )


def level7_overworld_stop(_snap: ZeldaSnapshot) -> bool:
    """Exact overworld geometry stop: controllable on the pond screen."""
    return on_level7_pond_hyp(_snap)


def planning_report() -> dict[str, Any]:
    return {
        "level": LEVEL7,
        "name": "The Demon",
        "status": "pond_controller_partial",
        "source_hypothesis": SOURCE_HYPOTHESIS,
        "required_entry_caps": sorted(required_caps_for_entry()),
        "required_clear_caps": sorted(required_caps_for_clear()),
        "triforce_bit": LEVEL7_TRIFORCE_BIT,
        "ram": {
            "whistle": hex(ADDR_WHISTLE),
            "food": hex(ADDR_FOOD),
            "candle": hex(ADDR_CANDLE),
        },
        "screens_hypothesized": {
            "bait_shop": hex(SCREEN_LEVEL7_BAIT_SHOP_HYP),
            "pond": hex(SCREEN_LEVEL7_POND_HYP),
        },
        "bait_shop_hops_from_post_l6": [
            {"target": hex(h.target), "dir": h.direction} for h in POST_L6_TO_BAIT_HOPS
        ],
        "pond_hops_from_post_l6": [
            {"target": hex(h.target), "dir": h.direction} for h in POST_L6_TO_POND_HOPS
        ],
        "pond_hops_from_start": [
            {"target": hex(h.target), "dir": h.direction} for h in LEVEL7_POND_HOPS
        ],
        "pond_53_micro": {
            "inland_x": POND_53_INLAND_X,
            "west_gap_y": POND_53_WEST_GAP_Y,
            "seed_blocked": [list(c) for c in sorted(POND_53_SEED_BLOCKED)],
            "evidence": "fixture-live leftover 0x53 (224,173) hop10_ay",
        },
        "bait_24_micro": {
            "east_y": BAIT_24_EAST_Y,
            "next_hop": hex(0x25),
            "evidence": "fixture-live 0x24→0x25 RIGHT @ y=141 leftover 0x25 (0,141)",
        },
        "live": {
            "pond_screen": hex(SCREEN_LEVEL7_POND_HYP),
            "entry_room": hex(SCREEN_LEVEL7_ENTRY_ROOM),
            "entry_xy": [120, 205],
            "pond_stairs_xy": [96, 132],
            "boss_room": None,
            "evidence": "fixture-live Level7Entrance pin (whistle poke recon)",
        },
        "docs": "nes/zelda_i/docs/LEVEL7_ROUTE.md",
    }
