"""Early overworld item-gate hop tables (assisted pathing).

Maps **geometry-only** screen transitions for early OW capabilities:

1. Blue Candle shop (near start) — L8 bush burn gate
2. White Sword cave — 5 heart **containers** gate
3. Bomb shop (east coast) — early bombs capability

All hops are **source-planned** (Zelda Dungeon Gathering + GameFAQs shop
index letter coords decoded to our ``(row<<4)|col`` screen ids) unless a
probe promotes ``verification`` to ``assisted`` / ``observed``.

Strategy: path under ``--infinite-life`` first; combat farm residual is
``rr-38p`` / ``rr-ccx``. No Clean STATUS claims.

Heart-gate note
---------------
``UnlimitedHealthAssist`` only refills the **low nibble** of ``ADDR_HEALTH``
(filled hearts). The high nibble (containers − 1) is never increased.
White Sword Old Man requires **5 heart containers**, not full fill — so
infinite-life alone does **not** unlock white sword. Need L1 heart and/or
OW bomb-heart caves (see ``WHITE_SWORD_MIN_CONTAINERS``).

Bomb capacity upgrades (8→12→16 for 100R) are **inside** L5 / L7, not OW
mouths — this module maps the early **bomb shop** for inventory bombs.

See ``docs/OVERWORLD_DOORS.md`` (Key overworld capabilities) and
``docs/plan.md`` ZOW notes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Literal

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.overworld import (
    LEVEL2_5C_MAZE_WAYPOINTS,
    SCREEN_START,
    ScreenHop,
    is_5c_maze_hop,
    path_screens_from_hops,
)
from zelda_i.nav_common import unstick_wiggle
from zelda_i.ow_path import OverworldPathController
from zelda_i.ram import (
    ADDR_CANDLE,
    ADDR_SWORD,
    PLAY_MODE,
    ZeldaSnapshot,
    read_snapshot,
    read_u8,
)

Verification = Literal["planned", "assisted", "observed"]

# ---------------------------------------------------------------------------
# Anchors (decoded from walkthrough letter maps + hop arithmetic)
# ---------------------------------------------------------------------------

# Blue Candle open shop near start (GameFAQs G-7; IGN “N then W of start”).
# 0x67 is a dead-end grove — do **not** route west off 0x67. Detour east first.
SCREEN_CANDLE_SHOP_NEAR = 0x66  # col 6, row 6
# Mountain open shop with Blue Candle (GameFAQs M-1) — near white sword area.
SCREEN_CANDLE_SHOP_MOUNTAIN = 0x0C  # col 12, row 0
# Primary early candle target for L8 prep (near-start is cheaper path).
SCREEN_CANDLE_SHOP = SCREEN_CANDLE_SHOP_NEAR

# White sword: GameFAQs K-1 / ZD Gathering 1.3 → **0x0A** (source cave).
# Live region stop: L5 mouth **0x0B** after Lost Hills (OW west off 0x0B is
# sealed — LEFT/UP enter L5). Cave tile residual rr-38p.
SCREEN_WHITE_SWORD_CAVE = 0x0A  # source/planned cave screen
SCREEN_WHITE_SWORD_REGION = 0x0B  # live assisted settle (L5 mouth)
WHITE_SWORD_MIN_CONTAINERS = 5
SWORD_WHITE = 2

# Early bomb shops (GameFAQs regular shops with Bombs @20R):
#   K-5 → **0x4A** (live path via L2 prefix; preferred early stop)
#   P-7 → **0x6F** (ZD right×8 up×1; 0x5E has no east/south live — residual)
SCREEN_BOMB_SHOP = 0x4A  # primary early (K-5)
SCREEN_BOMB_SHOP_COAST = 0x6F  # P-7 coast residual
BOMB_SHOP_PRICE_SOURCE = 20
CANDLE_SHOP_PRICE_SOURCE = 60

# Bomb capacity upgrades are dungeon Old Man shops (not OW mouths).
BOMB_CAPACITY_UPGRADE_LOCATIONS_SOURCE = ("level5", "level7")  # 100R each

SEGMENT_MAX_FRAMES = 45000
SWORD_SWING_PERIOD = 10
SWORD_SWING_FRAMES = 3
STUCK_THRESHOLD = 50


# ---------------------------------------------------------------------------
# Hop tables — verification starts planned
# ---------------------------------------------------------------------------

# Near-start Blue Candle: avoid 0x67 trap (no west corridor).
# Live recon (2026-08-06): 0x56 has **no south exit** to 0x66. Approach via
# L3-style west then south into 0x65, then east @y≈140–170 into 0x66.
# 0x77 E@y≈140 → 0x78 N@x≈48 → 0x68 N@x≈48 → 0x58
#   W@y≈155 → 0x57 W → 0x56 W → 0x55 S@x≈112 → 0x65 E@y≈140 → **0x66**
CANDLE_SHOP_NEAR_HOPS: tuple[ScreenHop, ...] = (
    ScreenHop(0x78, "RIGHT", align_y=140),
    ScreenHop(0x68, "UP", align_x=48),
    ScreenHop(0x58, "UP", align_x=48),
    ScreenHop(0x57, "LEFT", y_band_lo=148, y_band_hi=162),
    ScreenHop(0x56, "LEFT", y_band_lo=148, y_band_hi=162),
    ScreenHop(0x55, "LEFT", align_y=133),
    ScreenHop(0x65, "DOWN", align_x=112),
    # Live: east corridor opens near y≈86 (center wall blocks y≈70 and y≈173).
    ScreenHop(SCREEN_CANDLE_SHOP_NEAR, "RIGHT", align_y=88),
)
CANDLE_SHOP_NEAR_SCREENS: tuple[int, ...] = path_screens_from_hops(
    SCREEN_START, CANDLE_SHOP_NEAR_HOPS
)
CANDLE_SHOP_NEAR_VERIFICATION: Verification = "assisted"  # ig3 2026-08-06

# White Sword region: L2-style east to 0x4A, L5 hops into Lost Hills 0x1B,
# free pocket + ↑×4 → **0x0B** (live). Source cave **0x0A** residual (no OW
# west off 0x0B). Heart gate still applies at the cave.
WHITE_SWORD_PREFIX_HOPS: tuple[ScreenHop, ...] = (
    ScreenHop(0x78, "RIGHT", align_y=140),
    ScreenHop(0x68, "UP", align_x=48),
    ScreenHop(0x58, "UP", align_x=48),
    ScreenHop(0x59, "RIGHT", y_band_lo=148, y_band_hi=162),
    ScreenHop(0x49, "UP", align_x=112),
    ScreenHop(0x4A, "RIGHT", align_y=141),
)
# L5 mid-east hops from 0x4A (same geometry as level5_overworld.LEVEL5_PATH_HOPS).
WHITE_SWORD_HOPS: tuple[ScreenHop, ...] = WHITE_SWORD_PREFIX_HOPS + (
    ScreenHop(0x3A, "UP", align_x=112),
    ScreenHop(0x3B, "RIGHT", align_y=140),
    ScreenHop(0x2B, "UP", align_x=48),
    ScreenHop(0x2C, "RIGHT", align_y=85),
    ScreenHop(0x1C, "UP", align_x=48),
    ScreenHop(0x1B, "LEFT", align_y=140),
)
WHITE_SWORD_SCREENS: tuple[int, ...] = path_screens_from_hops(
    SCREEN_START, WHITE_SWORD_HOPS
)
WHITE_SWORD_VERIFICATION: Verification = "assisted"  # region 0x0B live ig9
WHITE_SWORD_LOST_HILLS_UPS = 4
SCREEN_LOST_HILLS = 0x1B
SCREEN_LEVEL5_DOOR = SCREEN_WHITE_SWORD_REGION

# 0x58 bush-grid waypoints (from overworld_nav L1) for hop 0x58→0x48 UP.
SCREEN_58_BUSH = 0x58
WHITE_SWORD_58_WAYPOINTS: tuple[tuple[int, int], ...] = (
    (48, 160),
    (80, 157),
    (112, 157),
)
MAZE_WAYPOINT_TOL_58 = 6


def is_58_bush_hop(hop: ScreenHop) -> bool:
    """True for the 0x58→0x48 north hop that needs bush waypoints."""
    return hop.target == 0x48 and hop.direction == "UP"

# Mountain candle shop (M-1 / 0x0C): source-only; exact path TBD after 0x0A
# cave residual. Placeholder single hop for table registration (not neighbor-
# validated against start — use candle_shop_near for early candle).
CANDLE_SHOP_MOUNTAIN_FROM_WS_HOPS: tuple[ScreenHop, ...] = (
    ScreenHop(SCREEN_CANDLE_SHOP_MOUNTAIN, "RIGHT", align_y=140),
)
CANDLE_SHOP_MOUNTAIN_FROM_WS_SCREENS: tuple[int, ...] = path_screens_from_hops(
    SCREEN_WHITE_SWORD_REGION, CANDLE_SHOP_MOUNTAIN_FROM_WS_HOPS
)
CANDLE_SHOP_MOUNTAIN_VERIFICATION: Verification = "planned"
# Alias: same as near-start until mountain path is live-probed.
CANDLE_SHOP_MOUNTAIN_HOPS: tuple[ScreenHop, ...] = CANDLE_SHOP_NEAR_HOPS
CANDLE_SHOP_MOUNTAIN_SCREENS: tuple[int, ...] = CANDLE_SHOP_NEAR_SCREENS

# Bomb shop 0x4A (K-5): same early corridor as L2 prefix from start.
# 0x77 E → 0x78 N → 0x68 N → 0x58 E → 0x59 N → 0x49 E → **0x4A**
BOMB_SHOP_HOPS: tuple[ScreenHop, ...] = (
    ScreenHop(0x78, "RIGHT", align_y=140),
    ScreenHop(0x68, "UP", align_x=48),
    ScreenHop(0x58, "UP", align_x=48),
    ScreenHop(0x59, "RIGHT", y_band_lo=148, y_band_hi=162),
    ScreenHop(0x49, "UP", align_x=112),
    ScreenHop(SCREEN_BOMB_SHOP, "RIGHT", align_y=141),
)
BOMB_SHOP_SCREENS: tuple[int, ...] = path_screens_from_hops(
    SCREEN_START, BOMB_SHOP_HOPS
)
BOMB_SHOP_VERIFICATION: Verification = "assisted"  # shared with L2 path geometry

# Coast P-7 residual (not wired as default — 0x5E sealed east/south live).
BOMB_SHOP_COAST_HOPS: tuple[ScreenHop, ...] = (
    ScreenHop(0x78, "RIGHT", align_y=140),
    ScreenHop(0x68, "UP", align_x=48),
    ScreenHop(0x58, "UP", align_x=48),
    ScreenHop(0x59, "RIGHT", y_band_lo=148, y_band_hi=162),
    ScreenHop(0x5A, "RIGHT", y_band_lo=120, y_band_hi=145),
    ScreenHop(0x5B, "RIGHT", y_band_lo=130, y_band_hi=150),
    ScreenHop(0x5C, "RIGHT", y_band_lo=80, y_band_hi=95),
    ScreenHop(0x5D, "RIGHT", y_band_lo=120, y_band_hi=140),
    ScreenHop(0x5E, "RIGHT", y_band_lo=120, y_band_hi=150),
    # Residual: live 0x5E has no E/S — alternate to 0x6F TBD.
    ScreenHop(SCREEN_BOMB_SHOP_COAST, "DOWN", align_x=64),
)


# Shared maze geometry with L2 door / L8 bush paths.
ITEM_GATE_5C_MAZE_WAYPOINTS: tuple[tuple[int, int], ...] = LEVEL2_5C_MAZE_WAYPOINTS


# ---------------------------------------------------------------------------
# Gate descriptors (for docs / reports / unit structure tests)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ItemGateRoute:
    """One named OW item-gate hop route."""

    name: str
    start: int
    end: int
    hops: tuple[ScreenHop, ...]
    screens: tuple[int, ...]
    verification: Verification
    requires_note: str
    price_source: int | None = None
    min_heart_containers: int | None = None
    uses_5c_maze: bool = False
    source: str = "zeldadungeon_gathering+gamefaqs_shop_index"


ITEM_GATE_ROUTES: dict[str, ItemGateRoute] = {
    "candle_shop_near": ItemGateRoute(
        name="candle_shop_near",
        start=SCREEN_START,
        end=SCREEN_CANDLE_SHOP_NEAR,
        hops=CANDLE_SHOP_NEAR_HOPS,
        screens=CANDLE_SHOP_NEAR_SCREENS,
        verification=CANDLE_SHOP_NEAR_VERIFICATION,
        requires_note=f"~{CANDLE_SHOP_PRICE_SOURCE} rupees; open cave shop",
        price_source=CANDLE_SHOP_PRICE_SOURCE,
    ),
    "candle_shop_mountain": ItemGateRoute(
        name="candle_shop_mountain",
        start=SCREEN_START,
        # Until mountain path is live, reuse near-start candle hops (0x66).
        end=SCREEN_CANDLE_SHOP_NEAR,
        hops=CANDLE_SHOP_MOUNTAIN_HOPS,
        screens=CANDLE_SHOP_MOUNTAIN_SCREENS,
        verification=CANDLE_SHOP_MOUNTAIN_VERIFICATION,
        requires_note=(
            f"placeholder=near 0x66; true M-1 0x0C residual "
            f"(~{CANDLE_SHOP_PRICE_SOURCE}R)"
        ),
        price_source=CANDLE_SHOP_PRICE_SOURCE,
    ),
    "white_sword": ItemGateRoute(
        name="white_sword",
        start=SCREEN_START,
        end=SCREEN_WHITE_SWORD_REGION,  # live stop 0x0B; cave 0x0A residual
        hops=WHITE_SWORD_HOPS,
        # screens list ends on 0x1B (Lost Hills); controller continues to 0x0B.
        screens=WHITE_SWORD_SCREENS,
        verification=WHITE_SWORD_VERIFICATION,
        requires_note=(
            f"region 0x0B live; cave 0x0A planned; "
            f"{WHITE_SWORD_MIN_CONTAINERS} heart *containers* for Old Man "
            "(infinite-life fill does not grant containers)"
        ),
        min_heart_containers=WHITE_SWORD_MIN_CONTAINERS,
        uses_5c_maze=False,
    ),
    "bomb_shop": ItemGateRoute(
        name="bomb_shop",
        start=SCREEN_START,
        end=SCREEN_BOMB_SHOP,
        hops=BOMB_SHOP_HOPS,
        screens=BOMB_SHOP_SCREENS,
        verification=BOMB_SHOP_VERIFICATION,
        requires_note=(
            f"~{BOMB_SHOP_PRICE_SOURCE}R open cave on 0x4A (K-5); "
            "coast 0x6F residual"
        ),
        price_source=BOMB_SHOP_PRICE_SOURCE,
        uses_5c_maze=False,
    ),
}


def route_for(name: str) -> ItemGateRoute:
    try:
        return ITEM_GATE_ROUTES[name]
    except KeyError as exc:
        known = ", ".join(sorted(ITEM_GATE_ROUTES))
        raise KeyError(f"unknown item gate route {name!r}; known: {known}") from exc


def white_sword_containers_ok(snap: ZeldaSnapshot) -> bool:
    """True when heart *containers* meet the Old Man gate (not fill)."""
    return snap.heart_containers >= WHITE_SWORD_MIN_CONTAINERS


def white_sword_heart_gate_blocks(snap: ZeldaSnapshot) -> bool:
    """True when containers alone block white sword (assist fill irrelevant)."""
    return not white_sword_containers_ok(snap)


def has_candle(ram) -> bool:
    return read_u8(ram, ADDR_CANDLE) != 0


def has_white_sword(ram) -> bool:
    return read_u8(ram, ADDR_SWORD) >= SWORD_WHITE


def hops_are_neighbors(hops: tuple[ScreenHop, ...], start: int) -> bool:
    """Structural check: each hop target is a grid neighbor of the previous screen."""
    from zelda_i.overworld import neighbor_screens

    current = start
    for hop in hops:
        neighbors = neighbor_screens(current)
        if hop.target not in neighbors.values():
            return False
        current = hop.target
    return True


# ---------------------------------------------------------------------------
# Thin hop controller (geometry walk only; no shop buy / cave enter)
# ---------------------------------------------------------------------------


class ItemGateNavPhase(Enum):
    HOP = auto()
    FREE_POCKET = auto()
    LOST_HILLS = auto()
    TO_WHITE_SWORD = auto()
    DONE = auto()
    FAILED = auto()


# Lost Hills pocket free (from level5_overworld)
_POCKET_FREE_Y = 172
_POCKET_FREE_X = 100


@dataclass
class ItemGateHopController(OverworldPathController):
    """Walk a single item-gate hop table; stop on end screen (OW play).

    Does not enter caves, buy items, or check inventory success — pathing
    only. Use ``require_sword`` default True when starting from PostSwordStart.

    White-sword route: hops end on Lost Hills ``0x1B``; then FREE_POCKET +
    LOST_HILLS (↑×4) to region stop ``0x0B`` (L5 mouth). Source cave ``0x0A``
    is residual (no OW west off 0x0B live).
    """

    phase: ItemGateNavPhase = ItemGateNavPhase.HOP
    route_name: str = "candle_shop_near"
    hops: tuple[ScreenHop, ...] = CANDLE_SHOP_NEAR_HOPS
    maze_waypoints: tuple[tuple[int, int], ...] = ()
    maze_hop_pred: Any = None
    max_frames: int = SEGMENT_MAX_FRAMES
    swing_period: int = SWORD_SWING_PERIOD
    swing_hold: int = SWORD_SWING_FRAMES
    stuck_threshold: int = STUCK_THRESHOLD
    require_sword: bool = True
    stop_y_lo: int = 40
    stop_y_hi: int = 210
    hills_ups: int = 0
    hills_in_scroll: bool = False
    pocket_stage: int = 0

    def __post_init__(self) -> None:
        route = route_for(self.route_name)
        self.hops = route.hops
        if route.uses_5c_maze:
            self.maze_waypoints = ITEM_GATE_5C_MAZE_WAYPOINTS
            if self.maze_hop_pred is None:
                self.maze_hop_pred = is_5c_maze_hop
        self.door_screen = route.end

    def reset(self) -> None:
        super().reset()
        route = route_for(self.route_name)
        self.hops = route.hops
        self.hills_ups = 0
        self.hills_in_scroll = False
        self.pocket_stage = 0

    def end_screen(self) -> int:
        if self.route_name == "white_sword":
            return SCREEN_WHITE_SWORD_REGION
        return self.hops[-1].target if self.hops else -1

    def _at_stop(self, snap: ZeldaSnapshot) -> bool:
        if self.route_name == "white_sword":
            return (
                snap.level == 0
                and snap.mode == PLAY_MODE
                and snap.screen == SCREEN_WHITE_SWORD_REGION
                and self.stop_y_lo < snap.link_y < self.stop_y_hi
                and (not self.require_sword or snap.has_sword)
            )
        return super()._at_stop(snap)

    def _wants_post_hop(self) -> bool:
        return self.route_name == "white_sword"

    def _on_hop_advanced(
        self, snap: ZeldaSnapshot, completed_hop: ScreenHop
    ) -> FrameAction:
        if (
            self.route_name == "white_sword"
            and self.hop_index >= len(self.hops)
        ):
            self._set_phase(ItemGateNavPhase.FREE_POCKET, "lost_hills_pocket")
            self.hills_ups = 0
            self.hills_in_scroll = False
            self.pocket_stage = 0
            return FrameAction(nes_idle_action(), "pocket_ready")
        return super()._on_hop_advanced(snap, completed_hop)

    def _after_hops(self, snap: ZeldaSnapshot) -> FrameAction:
        if self.route_name == "white_sword":
            if self.phase is ItemGateNavPhase.HOP:
                self._set_phase(ItemGateNavPhase.FREE_POCKET, "hops_to_pocket")
            if self.phase is ItemGateNavPhase.FREE_POCKET:
                return self._free_pocket_step(snap)
            if self.phase is ItemGateNavPhase.LOST_HILLS:
                return self._lost_hills_step(snap)
        return super()._after_hops(snap)

    def _before_play(self, snap: ZeldaSnapshot) -> FrameAction | None:
        if self.route_name == "white_sword":
            if self.phase is ItemGateNavPhase.FREE_POCKET:
                return self._free_pocket_step(snap)
            if self.phase is ItemGateNavPhase.LOST_HILLS:
                return self._lost_hills_step(snap)
        return None

    def _handle_transition(self, snap: ZeldaSnapshot) -> FrameAction:
        if self.route_name == "white_sword" and self.phase in (
            ItemGateNavPhase.FREE_POCKET,
            ItemGateNavPhase.LOST_HILLS,
        ):
            if self.phase is ItemGateNavPhase.LOST_HILLS:
                return self._lost_hills_step(snap)
            return self._free_pocket_step(snap)
        return super()._handle_transition(snap)

    def _free_pocket_step(self, snap: ZeldaSnapshot) -> FrameAction:
        """Leave 0x1C east ledge into Lost Hills main path (L5 geometry)."""
        if snap.screen == SCREEN_WHITE_SWORD_REGION and snap.mode == PLAY_MODE:
            return self._finish("white_sword_region")
        if snap.screen != SCREEN_LOST_HILLS:
            self._set_phase(ItemGateNavPhase.LOST_HILLS, "left_pocket_screen")
            return FrameAction(nes_idle_action(), "pocket_sc")
        if snap.link_x <= 120:
            self.notes.append("pocket_already_free")
            self._set_phase(ItemGateNavPhase.LOST_HILLS, "pocket_free")
            return FrameAction(nes_idle_action(), "pocket_done")
        if self.stuck > self.stuck_threshold:
            action, self.stuck = unstick_wiggle(self.stuck, reason="pocket_unstick")
            return action
        # East ledge often blocks pure DOWN at x≈240 — step left first if glued.
        if snap.link_x >= 200 and snap.link_y < _POCKET_FREE_Y - 2:
            if self.phase_frames % 40 < 20:
                return self._swing("LEFT", "pocket_unwedge")
            return self._swing("DOWN", "pocket_down")
        if self.pocket_stage == 0:
            if snap.link_y >= _POCKET_FREE_Y - 2:
                self.pocket_stage = 1
                self.notes.append("pocket_down")
                return self._swing("LEFT", "pocket_left")
            return self._swing("DOWN", "pocket_down")
        if snap.link_x <= _POCKET_FREE_X:
            self.notes.append("pocket_free")
            self._set_phase(ItemGateNavPhase.LOST_HILLS, "pocket_free")
            return FrameAction(nes_idle_action(), "pocket_done")
        return self._swing("LEFT", "pocket_left")

    def _lost_hills_step(self, snap: ZeldaSnapshot) -> FrameAction:
        """Four UP transitions on 0x1B → 0x0B (same rule as L5)."""
        if snap.screen == SCREEN_WHITE_SWORD_REGION and snap.mode == PLAY_MODE:
            self.notes.append(f"hills_ups_{self.hills_ups}_to_door")
            return self._finish("white_sword_region")
        if snap.mode in (6, 7) or snap.transitioning:
            self.hills_in_scroll = True
            return FrameAction(nes_action("UP"), "hills_scroll")
        if self.hills_in_scroll and snap.mode == PLAY_MODE:
            self.hills_in_scroll = False
            if snap.screen == SCREEN_LOST_HILLS:
                self.hills_ups += 1
                self.notes.append(f"hills_wrap_{self.hills_ups}")
            elif snap.screen == SCREEN_WHITE_SWORD_REGION:
                self.hills_ups += 1
                self.notes.append(f"hills_door_{self.hills_ups}")
                return self._finish("white_sword_region")
        if self.stuck > self.stuck_threshold:
            action, self.stuck = unstick_wiggle(self.stuck, reason="hills_unstick")
            return action
        if abs(snap.link_x - 112) > 6:
            btn = "RIGHT" if snap.link_x < 112 else "LEFT"
            return self._swing(btn, "hills_ax")
        return self._swing("UP", "hills_up")

    def _extra_hop_action(
        self, snap: ZeldaSnapshot, hop: ScreenHop
    ) -> FrameAction | None:
        # Climb 0x5B north corridor before east into 0x5C (same as L8).
        if (
            hop.target == 0x5C
            and hop.direction == "RIGHT"
            and snap.screen == 0x5B
            and snap.link_y > 100
        ):
            return self._swing("UP", "5b_north_corridor")
        # After 0x58 north hop completes, ensure we leave south edge before
        # align_x (align_and_push skips align when y>=205).
        if (
            hop.direction == "UP"
            and hop.align_x is not None
            and snap.link_y >= 200
            and snap.screen != SCREEN_58_BUSH
        ):
            return self._swing("UP", "climb_south_edge")
        return None

    def report(self) -> dict[str, Any]:
        base = super().report()
        base["route_name"] = self.route_name
        base["end_screen"] = self.end_screen()
        route = route_for(self.route_name)
        base["verification"] = route.verification
        base["requires_note"] = route.requires_note
        if self.route_name == "white_sword":
            base["hills_ups"] = self.hills_ups
        return base


def screen_reached(ram, screen: int, *, require_sword: bool = True) -> bool:
    snap = read_snapshot(ram)
    if not (
        snap.level == 0
        and snap.mode == PLAY_MODE
        and snap.screen == screen
        and 40 < snap.link_y < 210
    ):
        return False
    if require_sword and not snap.has_sword:
        return False
    return True


def gate_report_snapshot(snap: ZeldaSnapshot, ram=None) -> dict[str, Any]:
    """Inventory + heart-gate fields for probe reports."""
    candle = read_u8(ram, ADDR_CANDLE) if ram is not None else None
    return {
        "screen": snap.screen,
        "screen_hex": f"0x{snap.screen:02x}",
        "mode": snap.mode,
        "level": snap.level,
        "x": snap.link_x,
        "y": snap.link_y,
        "sword": snap.sword,
        "bombs": snap.bombs,
        "rupees": snap.rupees,
        "health": snap.health,
        "heart_containers": snap.heart_containers,
        "filled_hearts_nibble": snap.filled_hearts,
        "white_sword_containers_ok": white_sword_containers_ok(snap),
        "white_sword_heart_gate_blocks": white_sword_heart_gate_blocks(snap),
        "candle": candle,
    }


__all__ = [
    "SCREEN_CANDLE_SHOP",
    "SCREEN_CANDLE_SHOP_NEAR",
    "SCREEN_CANDLE_SHOP_MOUNTAIN",
    "SCREEN_WHITE_SWORD_CAVE",
    "SCREEN_WHITE_SWORD_REGION",
    "SCREEN_BOMB_SHOP",
    "SCREEN_BOMB_SHOP_COAST",
    "WHITE_SWORD_MIN_CONTAINERS",
    "SWORD_WHITE",
    "BOMB_SHOP_PRICE_SOURCE",
    "CANDLE_SHOP_PRICE_SOURCE",
    "BOMB_CAPACITY_UPGRADE_LOCATIONS_SOURCE",
    "CANDLE_SHOP_NEAR_HOPS",
    "CANDLE_SHOP_NEAR_SCREENS",
    "CANDLE_SHOP_MOUNTAIN_HOPS",
    "CANDLE_SHOP_MOUNTAIN_SCREENS",
    "CANDLE_SHOP_MOUNTAIN_FROM_WS_HOPS",
    "WHITE_SWORD_HOPS",
    "WHITE_SWORD_SCREENS",
    "WHITE_SWORD_58_WAYPOINTS",
    "is_58_bush_hop",
    "BOMB_SHOP_HOPS",
    "BOMB_SHOP_SCREENS",
    "ITEM_GATE_5C_MAZE_WAYPOINTS",
    "ITEM_GATE_ROUTES",
    "ItemGateRoute",
    "ItemGateHopController",
    "ItemGateNavPhase",
    "route_for",
    "white_sword_containers_ok",
    "white_sword_heart_gate_blocks",
    "has_candle",
    "has_white_sword",
    "hops_are_neighbors",
    "screen_reached",
    "gate_report_snapshot",
    "SEGMENT_MAX_FRAMES",
]
