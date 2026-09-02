"""Level 7 (Demon) overworld approach and gated entry helpers.

Gated by Whistle (L5) for pond drain and Bait/Food for hungry Goriya.
The start-to-pond hop table is usable without either item so geometry can be
verified independently; opening the pond still requires the naturally earned
Whistle.

See ``docs/LEVEL7_ROUTE.md``.
"""

from __future__ import annotations

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
    SCREEN_LEVEL6_ENTRANCE,
    SCREEN_LEVEL7_BAIT_SHOP_HYP,
    SCREEN_LEVEL7_POND_HYP,
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

# Post-L6 leftover play 0x22 (120,221) → Armos bait shop 0x34.
# 0x22↓0x32 and 0x32→0x33 @ y=141 reverse the live L6 door walk.
# Dead: 0x33 RIGHT @ y=141 → 0x34 (l7_bait_32ax leftover (208,141) east mountain).
# Live L6 reverse: 0x33↑0x23 @ x=208, 0x23→0x24 @ y=141.
# 0x24 south sand east is live through (208,189). DOWN at 16/160/208 is mountain.
# Fixture-live: 0x24→0x25 RIGHT @ y=141 (l7_bait_25 leftover 0x25 (0,141)).
# Shop 0x34 stays unobserved. Next: inland off the west mouth, then DOWN.
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
BAIT_APPROACH_MAX_FRAMES = 15_000
# Dead belief: (112,125) on 0x22 is the L6 leave. Mode 16 → dungeon.
L6_CAVE_MOUTH_X = 112
L6_CAVE_MOUTH_Y = 125
L6_CAVE_MOUTH_TOL = 16
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

# 0x53 east-edge vertical travel is the v9 miss.  Leave the east column
# (x>192) before descending to the hypothesized west gap, then LEFT to 0x52.
POND_53_INLAND_X = 192
POND_53_WEST_GAP_Y = 189
POND_53_Y_TOL = 4
POND_53_SEED_BLOCKED: frozenset[tuple[int, int]] = frozenset({(224, 174)})
_POND_53_HOP_INDEX = 10


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
            self._pond53_walk = OccupancyWalker(
                grid=OccupancyGrid(
                    blocked=set(POND_53_SEED_BLOCKED),
                    xmin=EDGE_WEST_X,
                    xmax=EDGE_EAST_X,
                    ymin=EDGE_NORTH_Y,
                    ymax=EDGE_SOUTH_Y,
                ),
                goal=(EDGE_WEST_X + 4, POND_53_WEST_GAP_Y),
            )
        return self._pond53_walk

    def _extra_hop_action(
        self, snap: ZeldaSnapshot, hop: ScreenHop
    ) -> FrameAction | None:
        # 0x65→0x64 arrives on the east ledge at ~(232,109).  UP is blocked
        # there: descend to the open middle band, cross to the north gap and
        # climb.  probe_64_north_to_54: x≈60 is a clean open column to 0x54;
        # x≤40 stalls at y≈93; x≈120 is under the central tree isle.
        if hop.target == 0x54 and snap.screen == 0x64:
            if snap.link_x > BAIT_64_GAP_X + 6 and snap.link_y < 116:
                return self._swing("DOWN", "64_east_ledge_down")
            if snap.link_x > BAIT_64_GAP_X + 6:
                return self._swing("LEFT", "64_cross_to_north")
            if snap.link_x < BAIT_64_GAP_X - 6:
                return self._swing("RIGHT", "64_north_ax")
            return self._swing("UP", "64_north")
        if hop.target == 0x52 and snap.screen == 0x53:
            return pond_53_to_52_action(
                snap, walker=self._pond53_walker(), swing=self._swing
            )
        if hop.target == SCREEN_LEVEL7_POND_HYP and snap.screen == 0x52:
            # 0x52 rock field (probe_52_wall): climb the open west column x≈48
            # from the bottom corridor to the mid-band y≈120, traverse RIGHT to
            # x≈132, then a UP push funnels Link through the boulder-wall gap
            # (~x128) into the x≈112 north gap to pond 0x42.
            if snap.link_y > POND_52_MIDBAND_Y:
                if snap.link_x > POND_52_CLIMB_X + 6:
                    return self._swing("LEFT", "52_to_climb_column")
                if snap.link_x < POND_52_CLIMB_X - 6:
                    return self._swing("RIGHT", "52_climb_ax")
                return self._swing("UP", "52_climb")
            if snap.link_x < POND_52_GAP_X - 4:
                return self._swing("RIGHT", "52_traverse_midband")
            return self._swing("UP", "52_gap_up")
        return None


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
            "pond_screen": None,
            "entry_room": None,
            "boss_room": None,
        },
        "docs": "nes/zelda_i/docs/LEVEL7_ROUTE.md",
    }
