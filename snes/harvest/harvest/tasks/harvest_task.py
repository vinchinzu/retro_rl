"""Dynamic crop harvest + ship task driven by the save state's ripe tiles."""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import numpy as np


from retro_harness import ActionResult, Task, TaskResult, TaskStatus, WorldState

from harvest.tasks.crop_planter import DEFAULT_CROP_BOUNDS, is_crop_tile, is_mature_crop_tile
from harvest.core.tile_catalog import ADDR_INPUT_LOCK, mature_tile_crop
from harvest.planner import run_strategy
from harvest.maps.farm_pond import SHIP_BIN_LANE_TILES
from harvest.tasks.nav import (
    MAP_WIDTH,
    TILE_SIZE,
    WALKABLE_TILES,
    Navigator,
    Pathfinder,
    get_pos_from_ram,
    get_tile_at,
    make_action,
    tile_dist,
)
from harvest.tasks.farm_ops import TileScanner

from harvest.core.animal_status import read_held_item
from harvest.tasks import horse as saddle
from harvest.core.harvest_state import HarvestStateDocument
from harvest.core.ram_catalog import field_spec, live_wram_base, read_ram_u24

ADDR_PLAYER_STATE = field_spec("player_state").address
ACTION_CARRYING_BIT = 0x02
ADDR_SHIPPING_MONEY = field_spec("shipping_money").address
READY_VISIBLE_MIN = 0xA0
SHIP_STAND_TILE = (11, 30)
SHIP_FACE = "left"
# The bin's F2 tiles are (8-9, 29-30) and the throw reads two tiles ahead, so four stands ship: east
# (11,29)/(11,30) facing left and south (8,32)/(9,32) facing up (ROM probe on
# ``LN_saddle_whatif_harvest_y2d22``: each paid 120G a corn). The south pair is one tile from the y=33
# lane of the ring lattice.
SHIP_FALLBACKS: Tuple[Tuple[Tuple[int, int], str], ...] = (
    ((11, 30), "left"),
    ((8, 32), "up"),
    ((9, 32), "up"),
    ((11, 29), "left"),
)
BIN_STANDS: Tuple[Tuple[int, int], ...] = tuple(stand for stand, _face in SHIP_FALLBACKS)


def bin_distance(tile: Tuple[int, int]) -> int:
    """Manhattan tiles from ``tile`` to the nearest bin stand."""
    return min(abs(tile[0] - sx) + abs(tile[1] - sy) for sx, sy in BIN_STANDS)


def nearest_bin_stand(tile: Tuple[int, int]) -> Tuple[int, int]:
    return min(BIN_STANDS, key=lambda st: abs(tile[0] - st[0]) + abs(tile[1] - st[1]))
# CLEAR_FIELD / lift debris IDs (not shippable). Shipping bin rejects these and
# HARVEST_ROUTE used to thrash ship_verify until timeout (rr-9xyy D15/D16).
# 0x03 weed (tile id often mirrors held), 0x04 small stone, 0x09 stump or large
# weed carry, 0x0D large rock (power-on CLEAR residual).
DEBRIS_HELD_ITEMS = frozenset({0x03, 0x04, 0x05, 0x06, *range(0x09, 0x11)})
CLEAR_HANDS_ATTEMPT_LIMIT = 4
# Consecutive BFS-less/blocked frames before the next stand (rr-20w.3.2).
# Budgets are generous: landing the crop beats a fast skip.
TARGET_NAV_STALL_LIMIT = 600
# Hard per-target ceiling when BFS "progress" never actually arrives.
TARGET_NAV_TOTAL_LIMIT = 6000
SHIP_NAV_STALL_LIMIT = 600
# Ceiling on one crop's walk to the bin across every stand. Past it the crop is
# tossed and the route fails: a farmer who cannot reach the bin otherwise
# wanders until the 24k-frame phase watchdog and skips the whole watering day.
SHIP_NAV_TOTAL_LIMIT = 4000

# Cross-day failure memory. The planner re-plans HARVEST_ROUTE every morning
# from the live map, so a ripe tile no stand can reach (or that never picks
# up) was retried at full price every day: one Fall tile ate all 30 days of
# the season. A tile that failed ``HARVEST_TILE_FAIL_LIMIT`` routes is left
# out of the target list until it stops being a ripe tile (picked, changed).
HARVEST_TILE_FAIL_LIMIT = 2

# ROM ``Dropedonsaleplace`` (bank_81 $8193BF): a crop dropped in the bin at
# 17:00 or later is consumed and never added to shipping_money; 27 Summer corn
# (3,240G) went that way from the Spring-timer pin. A ripe crop keeps on the
# plant, so a pick whose walk cannot reach the bin before 17:00 waits for the
# next morning. The ROM clock (``UpdateTime``, bank_82) counts 60 ticks a minute
# byte and 15 minute bytes an hour, one tick a frame outdoors, so 17:00 is tick
# 15,300 and the window is in frames. The old check read the minute byte as
# 0-59: near the end of an hour it let a far crop start a walk that landed after
# 17:00 (five west-strip crops in v2_d3) and at the top of one it gave up early.
# Running is 2 px a frame (8 a tile) with or without a crop; turns and lane
# detours make ~9.5 a Manhattan tile (ship_bench). Lift ~23 + throw ~36 frames.
SHIP_CLOSE_HOUR = 17
SHIP_CLOSE_TICK = SHIP_CLOSE_HOUR * 15 * 60
SHIP_FRAMES_PER_TILE = 9.5
SHIP_PICK_THROW_FRAMES = 60
SHIP_MARGIN_FRAMES = 90


SHIP_LOG_SCHEMA = "harvest.ship_log/1"

# Fast press. The pick reads the tile one ahead of the farmer's raw pixel and the bin throw the tile two
# ahead (ROM ``CalculateTileinFront``, bank_81 $81D14E: pixel +/- 16 per tile on the facing axis, the
# other axis untouched), so a press is valid on the first frame the pixel is inside the stand tile.
# The centred press walked the last 7-9 px at 1 px a frame (no B) on both ends of every crop and padded
# each A with a face/idle prefix: ~25 of a crop's frames, measured on ``LN_saddle_whatif_harvest_y2d22``
# (``harvest.scripts.ship_bench``). The lift runs ~23 frames from the A edge (``player_action`` 4) and the
# throw ~36 (5) whatever the pad does, so the verify waits on the animation. ``HARVEST_FAST_PRESS=0``
# restores the centred press for an A/B.
FAST_PRESS = os.environ.get("HARVEST_FAST_PRESS", "1") != "0"
# Off the facing axis the farmer runs (2 px a frame) until this close to the tile centre: a turn pressed at
# a tile edge beside an open lane can slide him one pixel into the lane, and the pick then reads the lane.
PRESS_PERP_TOLERANCE = 4
FAST_IDLE_FRAMES = 1
PRESS_TURN_LIMIT = 8
LIFT_WAIT_FRAMES = 40
THROW_WAIT_FRAMES = 60
FACE_CODE = {"down": 0, "up": 1, "right": 2, "left": 3}


def clock_tick(ram: np.ndarray) -> int:
    """ROM clock in its own ticks: 60 a minute unit, 15 units an hour (``UpdateTime``, bank_82)."""
    from harvest.core.ram_catalog import read_ram_value

    hour = int(read_ram_value(ram, "hour") or 0)
    minute = int(read_ram_value(ram, "minute") or 0)
    second = int(read_ram_value(ram, "second") or 0)
    return (hour * 15 + minute) * 60 + second


def ship_eta_frames(here: Tuple[int, int], stand: Tuple[int, int], ship_stand: Tuple[int, int]) -> float:
    """Frames (= clock ticks outdoors) from ``here`` to a pick at ``stand`` and the throw at the bin."""
    walk = abs(here[0] - stand[0]) + abs(here[1] - stand[1])
    walk += abs(stand[0] - ship_stand[0]) + abs(stand[1] - ship_stand[1])
    return walk * SHIP_FRAMES_PER_TILE + SHIP_PICK_THROW_FRAMES + SHIP_MARGIN_FRAMES


def ship_window_fits(ram: np.ndarray, here: Tuple[int, int], stand: Tuple[int, int],
                     ship_stand: Tuple[int, int] = SHIP_STAND_TILE) -> bool:
    """True when a crop picked from ``stand`` reaches the bin before 17:00."""
    return clock_tick(ram) + ship_eta_frames(here, stand, ship_stand) < SHIP_CLOSE_TICK
_TILE_FAILURES: dict[Tuple[int, int], int] = {}


def reset_harvest_failure_memory() -> None:
    _TILE_FAILURES.clear()


def note_harvest_tile_failure(tile: Tuple[int, int]) -> int:
    _TILE_FAILURES[tile] = _TILE_FAILURES.get(tile, 0) + 1
    return _TILE_FAILURES[tile]


def drop_failed_harvest_tiles(
    target_tiles: List[Tuple[int, int]],
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Split live ripe tiles into (kept, parked) by the failure memory.

    Memory for tiles that are no longer ripe is forgotten here, so a tile
    that is picked or replaced starts clean.
    """
    live = set(target_tiles)
    for tile in [t for t in _TILE_FAILURES if t not in live]:
        del _TILE_FAILURES[tile]
    kept = [t for t in target_tiles if _TILE_FAILURES.get(t, 0) < HARVEST_TILE_FAIL_LIMIT]
    parked = [t for t in target_tiles if t not in kept]
    return kept, parked


def unparked_harvest_tiles(tiles: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """``tiles`` less those the failure memory has parked (read only).

    The day plan asks this before it schedules a harvest: a Fall ring-centre
    tile no stand can reach reads ripe every morning, and ``NAV_CROP`` plus
    ``HARVEST_ROUTE`` for it ate hours (5,000+ frames on Fall D5) before the
    mushroom trip, which then started at 15:00 and failed the 16:00 gate.
    """
    return [t for t in tiles if _TILE_FAILURES.get(t, 0) < HARVEST_TILE_FAIL_LIMIT]


def is_carrying(ram: np.ndarray) -> bool:
    idx = ADDR_PLAYER_STATE + live_wram_base(ram)
    return bool(int(ram[idx]) & ACTION_CARRYING_BIT) if idx < len(ram) else False


LIFT_ACTIONS = frozenset({4, 5})   # player_action: lift, throw


def lift_in_progress(ram: np.ndarray) -> bool:
    """True while the lift or throw animation runs (the pad is ignored and the carry bit lags)."""
    from harvest.core.ram_catalog import read_ram_value

    return int(read_ram_value(ram, "player_action") or 0) in LIFT_ACTIONS


def is_debris_held(ram: np.ndarray) -> bool:
    """True when hands hold clear-field debris (not a crop for the shipping bin)."""
    held = int(read_held_item(ram))
    return held in DEBRIS_HELD_ITEMS


def read_shipping_money(ram: np.ndarray) -> int:
    return read_ram_u24(ram, ADDR_SHIPPING_MONEY) * 10


def is_ripe_crop_tile(tile_id: int) -> bool:
    return is_mature_crop_tile(tile_id)


Tile = Tuple[int, int]


def _ripe_tiles(bounds: Tuple[int, int, int, int], ripe) -> List[Tile]:
    left, top, right, bottom = bounds
    return [(x, y) for y in range(top, bottom + 1) for x in range(left, right + 1) if ripe(x, y)]


def _live_harvestable_crop_tiles_from_ram(
    ram: np.ndarray,
    *,
    bounds: Tuple[int, int, int, int] = DEFAULT_CROP_BOUNDS,
) -> List[Tile]:
    return _ripe_tiles(bounds, lambda x, y: is_ripe_crop_tile(get_tile_at(ram, x, y)))


def _load_document(state_name: Optional[str]) -> Optional[HarvestStateDocument]:
    if not state_name:
        return None
    try:
        return HarvestStateDocument.load(state_name)
    except FileNotFoundError:
        return None


def _ripe_in_save(document: HarvestStateDocument):
    def ripe(x: int, y: int) -> bool:
        tile = document.farm_tile(x, y)
        return is_ripe_crop_tile(tile.persistent_value) and tile.visible_value >= READY_VISIBLE_MIN
    return ripe


def state_harvestable_crop_tiles(
    state_name: Optional[str],
    *,
    bounds: Tuple[int, int, int, int] = DEFAULT_CROP_BOUNDS,
) -> List[Tile]:
    """Return ripe crop tiles using explicit pickable crop tile IDs."""
    document = _load_document(state_name)
    if document is None:
        return []
    candidates = _ripe_tiles(bounds, _ripe_in_save(document))
    if candidates:
        return candidates
    ram_array = getattr(document, "ram_array", None)
    if callable(ram_array):
        return _live_harvestable_crop_tiles_from_ram(np.asarray(ram_array(), dtype=np.uint8), bounds=bounds)
    return []


def live_harvestable_crop_tiles(
    ram: np.ndarray,
    state_name: Optional[str],
    *,
    bounds: Tuple[int, int, int, int] = DEFAULT_CROP_BOUNDS,
) -> List[Tile]:
    """Ripe in the save and still ripe on the live map; the live map alone without a save."""
    document = _load_document(state_name)
    in_save = None if document is None else _ripe_in_save(document)
    if in_save is None or not _ripe_tiles(bounds, in_save):
        return _live_harvestable_crop_tiles_from_ram(ram, bounds=bounds)
    return _ripe_tiles(bounds, lambda x, y: in_save(x, y) and is_ripe_crop_tile(get_tile_at(ram, x, y)))


@dataclass(frozen=True)
class HarvestStep:
    target: Tuple[int, int]
    stand: Tuple[int, int]
    face: str
    group: int = 0
    # Other walkable (stand, face) pairs adjacent to ``target``, ranked after
    # the primary. A pose-fragile approach to the primary stand (rr-20w.3.2:
    # a genuinely walkable tile that still snags on a corner/fence collision
    # from one direction) gets a second try from a different angle instead of
    # stalling the whole harvest route on one target.
    alt_stands: Tuple[Tuple[Tuple[int, int], str], ...] = ()


def _target_groups(target_tiles: List[Tuple[int, int]]) -> dict[Tuple[int, int], int]:
    """Group adjacent crop cells so harvesting clears one plot before moving on."""
    targets = set(target_tiles)
    groups: dict[Tuple[int, int], int] = {}
    group_id = 0
    for start in target_tiles:
        if start in groups:
            continue
        stack = [start]
        groups[start] = group_id
        while stack:
            tx, ty = stack.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nxt = (tx + dx, ty + dy)
                    if nxt in targets and nxt not in groups:
                        groups[nxt] = group_id
                        stack.append(nxt)
        group_id += 1
    return groups


def _stand_boxed_by_plot(ram: np.ndarray, stand: Tuple[int, int]) -> bool:
    """True when crops on three sides seal a walkable notch (ring center)."""
    sx, sy = stand
    crops = 0
    for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
        if is_crop_tile(get_tile_at(ram, sx + dx, sy + dy)):
            crops += 1
    return crops >= 3


def _open_harvest_stand(ram: np.ndarray, stand: Tuple[int, int], no_go: set) -> bool:
    if stand in no_go:
        return False
    if get_tile_at(ram, *stand) not in WALKABLE_TILES:
        return False
    return not _stand_boxed_by_plot(ram, stand)


def build_harvest_steps(
    ram: np.ndarray,
    target_tiles: List[Tuple[int, int]],
    *,
    ship_stand: Tuple[int, int] = SHIP_STAND_TILE,
) -> List[HarvestStep]:
    """Build adjacent stand tiles for each ripe crop tile."""
    from harvest.maps.farm_pond import FARM_NO_GO_TILES

    face_to_delta = {
        "up": (0, 1),
        "down": (0, -1),
        "left": (1, 0),
        "right": (-1, 0),
    }
    groups = _target_groups(target_tiles)
    target_set = set(target_tiles)
    no_go = set(FARM_NO_GO_TILES)
    steps: List[HarvestStep] = []

    for target in target_tiles:
        choices: List[Tuple[Tuple[int, int, int, int], Tuple[int, int], str]] = []
        vertical_crops: List[Tuple[Tuple[int, int, int, int], Tuple[int, int], str]] = []
        for face, (dx, dy) in face_to_delta.items():
            stand = (target[0] + dx, target[1] + dy)
            sx, sy = stand
            if not (0 <= sx < MAP_WIDTH and 0 <= sy < MAP_WIDTH):
                continue
            score = (
                0,
                bin_distance(stand) if ship_stand == SHIP_STAND_TILE
                else abs(ship_stand[0] - sx) + abs(ship_stand[1] - sy),
                abs(target[1] - ship_stand[1]),
                abs(target[0] - ship_stand[0]),
            )
            if _open_harvest_stand(ram, stand, no_go):
                choices.append((score, stand, face))
            elif (
                face in ("up", "down")
                and stand in target_set
                and is_crop_tile(get_tile_at(ram, sx, sy))
            ):
                # Ripe vertical neighbor. It opens once that ring tile is picked.
                vertical_crops.append((score, stand, face))

        if not choices:
            choices = vertical_crops
        if choices:
            choices.sort(key=lambda item: item[0])
            _best_score, best_stand, best_face = choices[0]
            alt_stands = tuple((stand, face) for _score, stand, face in choices[1:])
            steps.append(
                HarvestStep(
                    target=target,
                    stand=best_stand,
                    face=best_face,
                    group=groups.get(target, 0),
                    alt_stands=alt_stands,
                )
            )

    steps.sort(
        key=lambda step: (
            bin_distance(step.stand) if ship_stand == SHIP_STAND_TILE
            else abs(step.stand[0] - ship_stand[0]) + abs(step.stand[1] - ship_stand[1]),
            step.group,
            step.stand[1],
            step.stand[0],
        )
    )
    return steps


# Virgin plant / water days: ship-area fallback (136,520) made the planner
# pick unreachable south plots. Prefer the open early-spring field anchor
# used by crop_planner.DEFAULT_START_TILE (15, 29).
PREFERRED_PLANT_TILE: Tuple[int, int] = (15, 29)
PREFERRED_PLANT_PX: Tuple[int, int] = (
    PREFERRED_PLANT_TILE[0] * TILE_SIZE + 8,
    PREFERRED_PLANT_TILE[1] * TILE_SIZE + 8,
)


def crop_nav_target_px(
    ram: np.ndarray,
    state_name: Optional[str],
    *,
    bounds: Tuple[int, int, int, int] = DEFAULT_CROP_BOUNDS,
    fallback_px: Tuple[int, int] = PREFERRED_PLANT_PX,
) -> Tuple[int, int]:
    """Choose the nearest crop-adjacent stand for the pre-crop NAV_CROP phase."""
    target_tiles = state_harvestable_crop_tiles(state_name, bounds=bounds)
    if not target_tiles:
        target_tiles = _live_harvestable_crop_tiles_from_ram(ram, bounds=bounds)
    if target_tiles:
        steps = build_harvest_steps(ram, target_tiles)
        if steps:
            pos = get_pos_from_ram(ram)
            current_tile = (pos.x // TILE_SIZE, pos.y // TILE_SIZE)
            step = min(
                steps,
                key=lambda item: (
                    abs(item.stand[0] - current_tile[0])
                    + abs(item.stand[1] - current_tile[1]),
                    item.group,
                    item.stand[1],
                    item.stand[0],
                ),
            )
            return (step.stand[0] * TILE_SIZE + 8, step.stand[1] * TILE_SIZE + 8)

    # Planting / watering days: aim at the nearest detected plot center.
    from harvest.tasks.crop_planter import detect_plots

    plots = detect_plots(ram, bounds)
    if plots:
        pos = get_pos_from_ram(ram)
        current_tile = (pos.x // TILE_SIZE, pos.y // TILE_SIZE)
        center = min(
            plots,
            key=lambda tile: (
                abs(tile[0] - current_tile[0]) + abs(tile[1] - current_tile[1]),
                tile[1],
                tile[0],
            ),
        )
        return (center[0] * TILE_SIZE + 8, center[1] * TILE_SIZE + 8)

    return fallback_px


@dataclass
class HarvestTask(Task):
    """Harvest ripe crops from the current save layout and ship them one by one."""

    name: str = "harvest"
    state_name: Optional[str] = None
    bounds: Tuple[int, int, int, int] = DEFAULT_CROP_BOUNDS
    ship_stand: Tuple[int, int] = SHIP_STAND_TILE
    ship_face: str = SHIP_FACE
    ship_fallbacks: Tuple[Tuple[Tuple[int, int], str], ...] = SHIP_FALLBACKS
    # Frames are not the scarce resource for this bot (a day burns a flat
    # ~12-20k frames regardless of work done, then idles) — completeness
    # beats speed. 60000 gives every target room to exhaust its full stand
    # list (see TARGET_NAV_TOTAL_LIMIT) without the whole route dying first.
    timeout: int = 60000

    _scanner: TileScanner = field(default_factory=TileScanner, init=False)
    _pathfinder: Pathfinder = field(init=False)
    _navigator: Navigator = field(init=False)
    _phase: str = field(default="select", init=False)
    _steps: List[HarvestStep] = field(default_factory=list, init=False)
    _current: Optional[HarvestStep] = field(default=None, init=False)
    _action_queue: Deque[np.ndarray] = field(default_factory=deque, init=False)
    _step_count: int = field(default=0, init=False)
    _verify_count: int = field(default=0, init=False)
    _ship_money_before: int = field(default=0, init=False)
    _ship_options: List[Tuple[Tuple[int, int], str]] = field(default_factory=list, init=False)
    _ship_option_index: int = field(default=0, init=False)
    _target_live_before: int = field(default=-1, init=False)
    _initial_target_count: int = field(default=0, init=False)
    _unreachable_count: int = field(default=0, init=False)
    _unreachable_targets: List[Tuple[int, int]] = field(default_factory=list, init=False)
    _active_group: Optional[int] = field(default=None, init=False)
    _clear_hands_attempts: int = field(default=0, init=False)
    # Per-target stand fallback (rr-20w.3.2): the candidate (stand, face)
    # pairs left to try for ``_current``, and how far into them we are.
    _stand_choices: List[Tuple[Tuple[int, int], str]] = field(default_factory=list, init=False)
    _stand_choice_index: int = field(default=0, init=False)
    _current_stand: Tuple[int, int] = field(default=(0, 0), init=False)
    _current_face: str = field(default="down", init=False)
    _nav_stall_steps: int = field(default=0, init=False)
    _target_nav_steps: int = field(default=0, init=False)
    _ship_nav_steps: int = field(default=0, init=False)
    _abort_reason: Optional[str] = field(default=None, init=False)
    _deferred_since_pick: set[Tuple[int, int]] = field(default_factory=set, init=False)
    _defer_target: Optional[Tuple[int, int]] = field(default=None, init=False)
    harvested_count: int = field(default=0, init=False)
    shipped_count: int = field(default=0, init=False)
    skipped_count: int = field(default=0, init=False)
    # Saddle shipping (``harvest.tasks.horse``): the adult horse takes a thrown crop into the
    # bin total. ``_park`` leaves it beside far ripe crops; ``_horse_stands`` are the open
    # tiles around it that ``_reset_ship_options`` offered; ``_saddle_off`` is set when a
    # horse throw did not pay, so the rest of the route ships at the bin.
    _park: Optional[saddle.ParkHorse] = field(default=None, init=False)
    _saddle_off: bool = field(default=False, init=False)
    _horse_stands: dict = field(default_factory=dict, init=False)
    _horse_at: Optional[Tuple[int, int]] = field(default=None, init=False)
    _horse_hold: int = field(default=0, init=False)
    _on_horse_stand: int = field(default=0, init=False)
    _ship_at_horse: bool = field(default=False, init=False)
    horse_shipped: int = field(default=0, init=False)
    park_outcome: str = field(default="", init=False)
    _park_tried: bool = field(default=False, init=False)
    # Per-crop pickup-to-ship records (schema ``SHIP_LOG_SCHEMA``), one per bin drop, in drop order.
    # Frames are ``world.frame`` (the emulator frame the planner runs at); ``tick`` is the ROM clock in
    # 1/60 game-minute units ((hour*15 + minute)*60 + second), which runs every outdoor frame before 18:00.
    ship_log: List[dict] = field(default_factory=list, init=False)
    _rec: Optional[dict] = field(default=None, init=False)
    _frame: int = field(default=0, init=False)
    _ram: Optional[np.ndarray] = field(default=None, init=False)
    # A fast press that did not lift the crop is retried once from the centred stand.
    _recentre: bool = field(default=False, init=False)
    _push_from: Optional[Tuple[int, int]] = field(default=None, init=False)
    _press_face: str = field(default="down", init=False)
    _press_turns: int = field(default=0, init=False)
    _press_then: str = field(default="select", init=False)
    _press_legacy: dict = field(default_factory=dict, init=False)
    fast_misses: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._pathfinder = Pathfinder(self._scanner)
        self._pathfinder.allow_tiles = set(SHIP_BIN_LANE_TILES)
        self._navigator = Navigator(self._pathfinder)

    def reset(self, world: WorldState) -> None:
        self._phase = "select"
        self._current = None
        self._action_queue.clear()
        self._step_count = 0
        self._verify_count = 0
        self._target_live_before = -1
        self._ship_options = []
        self._ship_option_index = 0
        self._active_group = None
        self._clear_hands_attempts = 0
        self._stand_choices = []
        self._stand_choice_index = 0
        self._current_stand = (0, 0)
        self._current_face = "down"
        self._target_nav_steps = 0
        self._ship_nav_steps = 0
        self._abort_reason = None
        self._defer_target = None
        self._deferred_since_pick.clear()
        self.harvested_count = 0
        self.shipped_count = 0
        self.skipped_count = 0
        self._clear_navigation_state()

        target_tiles = live_harvestable_crop_tiles(world.ram, self.state_name, bounds=self.bounds)
        target_tiles, parked = drop_failed_harvest_tiles(target_tiles)
        if parked:
            print(f"[HARVEST] Parked {len(parked)} tile(s) that failed {HARVEST_TILE_FAIL_LIMIT} routes: {parked}")
        target_tiles, held = run_strategy.within_ship_cap(world.ram, target_tiles)
        if held:
            print(f"[HARVEST] Left {len(held)} ripe {'/'.join(run_strategy.crop_kinds(world.ram, held))} "
                  f"tile(s) on the plant at the {run_strategy.active().name} ship cap: {held}")
        self._steps = build_harvest_steps(world.ram, target_tiles, ship_stand=self.ship_stand)
        reachable_targets = {step.target for step in self._steps}
        self._unreachable_targets = [t for t in target_tiles if t not in reachable_targets]
        self._initial_target_count = len(target_tiles)
        self._unreachable_count = len(self._unreachable_targets)
        for tile in self._unreachable_targets:
            note_harvest_tile_failure(tile)

        extra_walkable = set()
        for step in self._steps:
            for stand, _face in ((step.stand, step.face), *step.alt_stands):
                tid = get_tile_at(world.ram, *stand)
                if tid in WALKABLE_TILES and not is_crop_tile(tid):
                    extra_walkable.add(stand)
        self._pathfinder.extra_walkable = extra_walkable
        # rr-20w.3.2: report the raw detected count against the reachable
        # count so a target dropped for "no valid adjacent stand" is visible
        # immediately instead of surfacing as a silent day-end shortfall.
        print(
            f"[HARVEST] Detected {len(target_tiles)} ripe crop targets from live map "
            f"({len(self._steps)} reachable"
            + (f", unreachable={self._unreachable_targets}" if self._unreachable_targets else "")
            + ")"
        )
        self._park, self._saddle_off, self._horse_stands = None, False, {}
        self._horse_at, self._ship_at_horse, self.horse_shipped, self.park_outcome = None, False, 0, ""
        self._park_tried = False
        self.ship_log, self._rec = [], None
        self._recentre, self.fast_misses = False, 0

    def _park_due(self, ram: np.ndarray) -> bool:
        """True once only far crops are left, with enough day to ride out and ship them."""
        if self._park_tried or self._saddle_off or not self._steps or saddle.saddle_usable(ram) is None:
            return False
        if saddle.clock_minute(ram) > saddle.PARK_LATEST_MINUTE:
            return False
        return all(
            tile_dist(step.stand, self.ship_stand) >= saddle.PARK_FAR_FROM_BIN for step in self._steps
        )

    def _plan_park(self, ram: np.ndarray) -> bool:
        """Leave the horse beside the far ripe crops when the walking saved clearly pays."""
        self._park_tried = True
        horse = saddle.saddle_usable(ram)
        if horse is None or is_carrying(ram):
            return False
        self._navigator.update(ram)
        plan = saddle.plan_park(
            [step.stand for step in self._steps],
            bin_stand=self.ship_stand,
            here=self._navigator.current_tile,
            horse=horse,
            open_tile=saddle.default_open_tile(ram),
            crop_adjacent=saddle.crop_adjacent(ram),
            minutes_left=saddle.SHIP_CLOSE_MINUTE - saddle.clock_minute(ram),
        )
        if plan is None:
            return False
        self._park = saddle.ParkHorse(plan)
        self._phase = "park"
        print(
            f"[HARVEST] PARK_HORSE plan: horse at {horse} -> {plan.park} saves ~{plan.saving_tiles} tiles "
            f"for ~{plan.cost_tiles} (bin {self.ship_stand})"
        )
        return True

    def _step_park(self, world: WorldState) -> TaskResult:
        action, outcome = self._park.step(world.ram, self._navigate_to_tile)
        if outcome is None:
            return self._running(action)
        self.park_outcome = outcome
        print(f"[HARVEST] PARK_HORSE {outcome}")
        self._park = None
        self._clear_navigation_state()
        self._phase = "select"
        return self._running(make_action())

    def _horse_unusable(self, ram: np.ndarray, why: str) -> TaskResult:
        """Ship the rest of the route at the bin."""
        print(f"[HARVEST] saddle shipping off: {why}")
        self._saddle_off, self._ship_at_horse = True, False
        self._ship_options, self._ship_option_index, self._horse_stands = [], 0, {}
        self._clear_navigation_state()
        self._phase = "ship_nav"
        return self._running()

    def _step_horse_face(self, world: WorldState) -> TaskResult:
        """Face the horse until its saddle bit is up, then throw: a throw before it pays nothing."""
        ram = world.ram
        stand, face = self._current_ship_option(ram)
        if not is_carrying(ram):
            self._phase = "ship_verify"
            return self._running()
        if saddle.farm_horse(ram) != self._horse_at:
            self._ship_options, self._phase = [], "ship_nav"
            self._clear_navigation_state()
            return self._running()
        if saddle.saddle_ready(ram):
            self._ship_money_before = read_shipping_money(ram)
            self._action_queue.extend(saddle.throw_actions())
            self._verify_count, self._phase = 0, "ship_verify"
            print(f"[HARVEST] SHIP at the horse {self._horse_at} from stand={stand} money={self._ship_money_before}")
            return self._running()
        self._horse_hold += 1
        if self._horse_hold > saddle.FACE_FRAMES:
            return self._horse_unusable(ram, "the horse never raised its saddle bit")
        return self._running(make_action(**{face: True}))

    def _window_ship_stand(self, ram: np.ndarray, pick_stand: Tuple[int, int]) -> Tuple[int, int]:
        """The ship stand a pick at ``pick_stand`` will use: the nearer of the bin and the horse."""
        bin_stand = min((stand for stand, _face in self.ship_fallbacks),
                        key=lambda st: abs(st[0] - pick_stand[0]) + abs(st[1] - pick_stand[1]),
                        default=self.ship_stand)
        horse = None if self._saddle_off else saddle.saddle_usable(ram)
        if horse is None:
            return bin_stand
        stands = [stand for stand, _face in saddle.horse_stands(ram, horse)]
        if not stands:
            return bin_stand
        near = min(stands, key=lambda stand: abs(stand[0] - pick_stand[0]) + abs(stand[1] - pick_stand[1]))
        return near if tile_dist(near, pick_stand) < tile_dist(bin_stand, pick_stand) else bin_stand

    def _crop_ship_pending(self) -> bool:
        """True after a successful pick that still needs a bin drop."""
        return self.harvested_count > self.shipped_count

    def _should_ship_carry(self, ram: np.ndarray) -> bool:
        """Ship crop carries; never treat CLEAR debris as bin cargo (rr-9xyy)."""
        if not is_carrying(ram):
            return False
        if is_debris_held(ram):
            return False
        if self._crop_ship_pending():
            return True
        # Non-debris held item (crop) without counter — still try the bin.
        return int(read_held_item(ram)) != 0

    def _should_clear_hands(self, ram: np.ndarray) -> bool:
        """Drop debris (or stuck empty-id carry) before picking ripe crops."""
        if not is_carrying(ram):
            return False
        if is_debris_held(ram):
            return True
        # Carry bit with no held id after failed ship: free hands for picks.
        return int(read_held_item(ram)) == 0 and not self._crop_ship_pending()

    def _begin_clear_hands(self, ram: np.ndarray, *, reason: str) -> None:
        held = int(read_held_item(ram))
        self._clear_navigation_state()
        self._action_queue.clear()
        self._clear_hands_attempts += 1
        # Function-local: transitions imports day_plan_status, which imports this module.
        from harvest.planner.tasks.transitions import multi_face_toss_actions, toss_held_actions

        if self._clear_hands_attempts <= 1:
            self._action_queue.extend(toss_held_actions(face="down", step_away=True))
        else:
            self._action_queue.extend(multi_face_toss_actions(prefer_south=True))
        self._phase = "clear_hands"
        print(
            f"[HARVEST] Clear hands before harvest "
            f"(held=0x{held:02X} attempt={self._clear_hands_attempts} {reason})"
        )

    def _abandon_carry(self, ram: np.ndarray, *, reason: str) -> TaskResult:
        """Toss the crop that cannot reach the bin, then fail the route."""
        self._abort_reason = (
            f"{reason}: harvested={self.harvested_count} shipped={self.shipped_count} "
            f"skipped={self.skipped_count} remaining={[s.target for s in self._steps]}"
        )
        print(f"[HARVEST] {self._abort_reason}")
        return self._toss_now(ram, reason=reason)

    def can_start(self, world: WorldState) -> bool:
        return True

    @property
    def progress_text(self) -> str:
        total = self._initial_target_count
        return (
            f"harvested={self.harvested_count} shipped={self.shipped_count} "
            f"skipped={self.skipped_count} unreachable={self._unreachable_count} "
            f"remaining={len(self._steps)} total={total}"
        )

    def _completion_result(self) -> TaskResult:
        reason = (
            f"harvested={self.harvested_count} shipped={self.shipped_count} "
            f"skipped={self.skipped_count} unreachable={self._unreachable_count}"
        )
        if self._initial_target_count == 0:
            return TaskResult(status=TaskStatus.SUCCESS, reason=reason)
        if (
            self.skipped_count > 0
            or self._unreachable_count > 0
            or self.harvested_count < self._initial_target_count
            or self.shipped_count < self.harvested_count
        ):
            return TaskResult(status=TaskStatus.FAILURE, reason=f"incomplete harvest: {reason}")
        return TaskResult(status=TaskStatus.SUCCESS, reason=reason)

    def _queue_press_a(
        self,
        face: str,
        *,
        face_frames: int = 2,
        hold_frames: int = 14,
        settle_frames: int = 10,
    ) -> None:
        self._action_queue.extend(make_action(**{face: True}) for _ in range(face_frames))
        self._action_queue.extend(make_action() for _ in range(2))
        self._action_queue.extend(make_action(a=True) for _ in range(hold_frames))
        self._action_queue.extend(make_action() for _ in range(settle_frames))

    def _queue_press(self, ram: np.ndarray, face: str, **legacy) -> None:
        """Face ``face`` and tap A (``FAST_PRESS``): the ``press`` phase turns until the ROM faces it.

        A one-frame turn after a run sometimes does not register (``LN_blocks_whatif_spring_d8``: the
        farmer stayed facing down at the bin, threw the potato on the ground in 17 frames, paid 0), so
        A waits for ``player_direction``.
        """
        if not FAST_PRESS or self._recentre:
            self._queue_press_a(face, **legacy)
            return
        self._press_face, self._press_turns, self._press_legacy = face, 0, legacy
        self._press_then, self._phase = self._phase, "press"

    def _step_press(self, world: WorldState) -> TaskResult:
        from harvest.core.ram_catalog import read_ram_value

        face = self._press_face
        if int(read_ram_value(world.ram, "player_direction") or 0) == FACE_CODE[face]:
            self._action_queue.extend(make_action() for _ in range(FAST_IDLE_FRAMES))
            self._action_queue.extend(make_action(a=True) for _ in range(2))
            self._phase = self._press_then
            return self._running(self._pop_action())
        self._press_turns += 1
        if self._press_turns > PRESS_TURN_LIMIT:
            self._queue_press_a(face, **self._press_legacy)       # the centred press, as before
            self._phase = self._press_then
            return self._running(self._pop_action())
        return self._running(make_action(**{face: True}))

    def _dialog_pulse_action(self) -> np.ndarray:
        return make_action(a=True) if self._step_count % 2 == 0 else make_action()

    def _pop_action(self) -> Optional[np.ndarray]:
        if self._action_queue:
            return self._action_queue.popleft()
        return None

    @staticmethod
    def _running(action: Optional[np.ndarray] = None) -> TaskResult:
        return TaskResult(status=TaskStatus.RUNNING, action=None if action is None else ActionResult(action))

    def _toss_now(self, ram: np.ndarray, *, reason: str) -> TaskResult:
        """Queue the toss tape and play its first frame."""
        self._begin_clear_hands(ram, reason=reason)
        return self._running(self._pop_action())

    def _back_to_select(self) -> None:
        self._recentre = False
        self._current = None
        self._clear_navigation_state()
        self._phase = "select"

    def _note(self, key: str) -> None:
        """Stamp ``f_<key>``/``tick_<key>`` on the open ship record."""
        if self._rec is not None:
            self._rec[f"f_{key}"] = self._frame
            self._rec[f"tick_{key}"] = clock_tick(self._ram) if self._ram is not None else None

    def _picked(self, phase: str, how: str) -> TaskResult:
        self._recentre = False
        self._note("pick")
        if self._rec is not None:
            self._rec.update(stand=list(self._current_stand), face=self._current_face,
                             stand_tries=self._stand_choice_index + 1)
        self.harvested_count += 1
        self._deferred_since_pick.clear()
        self._clear_navigation_state()
        self._phase = phase
        print(f"[HARVEST] PICK OK target={self._current.target} {how}")
        return self._running()

    def _close_ship_record(self, ram: np.ndarray, paid_g: int) -> None:
        """File the open pickup-to-ship record at the bin drop."""
        rec, self._rec = self._rec, None
        if rec is None:
            return
        self._note_into(rec, "ship", ram)
        rec["paid_g"] = int(max(0, paid_g))
        rec["at_horse"] = bool(self._ship_at_horse)
        self.ship_log.append(rec)

    def _note_into(self, rec: dict, key: str, ram: np.ndarray) -> None:
        rec[f"f_{key}"] = self._frame
        rec[f"tick_{key}"] = clock_tick(ram)

    def _clear_navigation_state(self) -> None:
        self._push_from = None
        self._navigator.path = []
        self._navigator.stasis = 0
        self._pathfinder.temp_blocked.clear()
        self._nav_stall_steps = 0

    def _stand_open(self, ram: np.ndarray, step: HarvestStep) -> bool:
        """True when a stand is walkable ground, not a ripe neighbor."""
        for stand, _face in ((step.stand, step.face), *step.alt_stands):
            tid = int(get_tile_at(ram, *stand))
            if tid in WALKABLE_TILES and not is_ripe_crop_tile(tid):
                return True
        return False

    def _choose_next_step(self, ram: np.ndarray | None = None) -> Optional[HarvestStep]:
        if not self._steps:
            return None
        if self._active_group is not None and not any(step.group == self._active_group for step in self._steps):
            self._active_group = None
        current = self._navigator.current_tile
        if self._active_group is None:
            self._active_group = min(
                {step.group for step in self._steps},
                key=lambda group: min(
                    abs(step.stand[0] - current[0]) + abs(step.stand[1] - current[1])
                    for step in self._steps
                    if step.group == group
                ),
            )
        group_indices = [idx for idx, step in enumerate(self._steps) if step.group == self._active_group]
        pool = group_indices
        if ram is not None:
            opened = [idx for idx in group_indices if self._stand_open(ram, self._steps[idx])]
            # Ring first. A center whose only stands are ripe tiles waits.
            if opened:
                pool = opened
            held = self._defer_target
            if held is not None:
                alt = [idx for idx in pool if self._steps[idx].target != held]
                pool = alt or pool
                if not alt:
                    self._defer_target = None
        best_idx = min(
            pool,
            key=lambda idx: abs(self._steps[idx].stand[0] - current[0]) + abs(self._steps[idx].stand[1] - current[1]),
        )
        step = self._steps[best_idx]
        for idx in pool:
            other = self._steps[idx]
            if other.target == step.stand and other.stand != step.target:
                best_idx = idx
                break
        return self._steps.pop(best_idx)

    def _reset_ship_options(self, ram: np.ndarray) -> None:
        options: List[Tuple[Tuple[int, int], str]] = []
        for stand, face in self.ship_fallbacks:
            if stand not in [item[0] for item in options]:
                options.append((stand, face))
        if self.ship_stand not in [item[0] for item in options]:
            options.append((self.ship_stand, self.ship_face))

        self._horse_stands, self._horse_at = {}, None
        horse = None if self._saddle_off else saddle.saddle_usable(ram)
        if horse is not None:
            self._horse_at = horse
            for stand, face in saddle.horse_stands(ram, horse):
                self._horse_stands[stand] = face
                options.append((stand, face))

        current = self._navigator.current_tile
        scored: List[Tuple[Tuple[int, int, int, int], Tuple[Tuple[int, int], str]]] = []
        for index, (stand, face) in enumerate(options):
            path = self._pathfinder.find_path(ram, current, stand)
            if path is not None:
                score = (0, len(path), abs(stand[0] - current[0]) + abs(stand[1] - current[1]), index)
            else:
                score = (1, abs(stand[0] - current[0]) + abs(stand[1] - current[1]), 0, index)
            scored.append((score, (stand, face)))

        self._ship_options = [option for _score, option in sorted(scored, key=lambda item: item[0])]
        self._ship_option_index = 0

    def _current_ship_option(self, ram: np.ndarray) -> Tuple[Tuple[int, int], str]:
        if not self._ship_options:
            self._reset_ship_options(ram)
        if not self._ship_options:
            return self.ship_stand, self.ship_face
        self._ship_option_index = min(self._ship_option_index, len(self._ship_options) - 1)
        return self._ship_options[self._ship_option_index]

    def _try_next_ship_option(self, ram: np.ndarray) -> bool:
        if not self._ship_options:
            self._reset_ship_options(ram)
        self._ship_option_index += 1
        if self._ship_option_index >= len(self._ship_options):
            return False
        self._clear_navigation_state()
        self._verify_count = 0
        self._phase = "ship_nav"
        stand, face = self._ship_options[self._ship_option_index]
        print(f"[HARVEST] Retrying ship from stand={stand} face={face}")
        return True

    def _fallback_action(self, goal: Tuple[int, int]) -> np.ndarray:
        current = self._navigator.current_tile
        dx = goal[0] - current[0]
        dy = goal[1] - current[1]
        if abs(dx) >= abs(dy):
            primary = "right" if dx > 0 else "left"
            secondary = "down" if dy > 0 else "up"
        else:
            primary = "down" if dy > 0 else "up"
            secondary = "right" if dx > 0 else "left"

        opposites = {"up": "down", "down": "up", "left": "right", "right": "left"}
        stasis = self._navigator.stasis
        if stasis < 45:
            direction = primary
        elif stasis < 90:
            direction = secondary
        elif stasis < 135:
            direction = opposites[primary]
        else:
            direction = opposites[secondary]
        return make_action(**{direction: True, "b": True})

    def _fast_arrival(self, goal: Tuple[int, int], face: Optional[str], *, contact: bool) -> Optional[np.ndarray]:
        """In the stand tile: None when a press is valid now, else a run step (B held).

        The axis across ``face`` is trimmed to ``PRESS_PERP_TOLERANCE``. A pick (``contact``) also needs the
        farmer pressed against the crop: the lift ignores A one pixel short of the block (ROM probe, branch
        of ``LN_saddle_whatif_harvest_y2d22``: facing down from y=568 never lifts, y=569 lifts on the A
        edge). So run at it until a frame moves nothing. The throw reads two tiles ahead and needs no contact.
        """
        pos = self._navigator.current_pos
        cx, cy = goal[0] * TILE_SIZE + 8, goal[1] * TILE_SIZE + 8
        if face in ("up", "down") and abs(pos.x - cx) > PRESS_PERP_TOLERANCE:
            self._push_from = None
            return make_action(**{"right" if pos.x < cx else "left": True, "b": True})
        if face in ("left", "right") and abs(pos.y - cy) > PRESS_PERP_TOLERANCE:
            self._push_from = None
            return make_action(**{"down" if pos.y < cy else "up": True, "b": True})
        if not contact:
            return None
        here = (pos.x, pos.y)
        if self._push_from == here:
            return None
        self._push_from = here
        return make_action(**{face: True, "b": True})

    def _navigate_to_tile(
        self, ram: np.ndarray, goal: Tuple[int, int], *, press_face: Optional[str] = None, contact: bool = False,
    ) -> Tuple[Optional[np.ndarray], bool]:
        """Action, or None when centered. True means this frame used fallback.

        ``press_face`` asks for a fast arrival (``FAST_PRESS``): inside the tile is enough.
        """
        if press_face is not None and FAST_PRESS and self._navigator.current_tile == goal:
            return self._fast_arrival(goal, press_face, contact=contact), False
        if self._navigator.current_tile == goal or self._navigator.at_tile(goal):
            return self._navigator.center_on_tile(goal, tolerance=1), False

        if self._navigator.stasis > 120 and self._navigator.path:
            self._pathfinder.temp_blocked.add(self._navigator.path[0])
            self._navigator.path = []

        if not self._navigator.path:
            path = self._pathfinder.find_path(
                ram,
                self._navigator.current_tile,
                goal,
                max_steps=7,
            )
            if path is None:
                return self._fallback_action(goal), True
            self._navigator.path = path

        action = self._navigator.follow_path(ram)
        if action is None:
            return self._fallback_action(goal), True
        return self._run_centring(action), False

    def _run_centring(self, action: np.ndarray) -> np.ndarray:
        """Hold B on a path-centring step that is 4+ px off the tile centre (2 px a frame, no overshoot).

        ``Navigator.follow_path`` centres across the walk at 1 px a frame before each turn: ~11 of a
        near-block crop's ~124 frames on the Spring D8 pin.
        """
        if not FAST_PRESS or action[0]:
            return action
        pos, tile = self._navigator.current_pos, self._navigator.current_tile
        off_x, off_y = abs(pos.x - (tile[0] * TILE_SIZE + 8)), abs(pos.y - (tile[1] * TILE_SIZE + 8))
        horizontal, vertical = bool(action[6] or action[7]), bool(action[4] or action[5])
        if (horizontal and not vertical and off_x >= 4) or (vertical and not horizontal and off_y >= 4):
            action = action.copy()
            action[0] = 1
        return action

    def _stand_provably_unreachable(self, ram: np.ndarray) -> bool:
        """True when BFS over the loaded viewport cannot reach the stand.

        A fully grown ring seals its own centre (every neighbour is a crop
        tile), and the old walk-until-timeout discovery cost ~6000f per stand:
        Summer D30 spent all 24k watchdog frames on one corn target. Only
        judged inside the viewport hop range, where BFS is not stale.
        """
        cur = self._navigator.current_tile
        stand = self._current_stand
        if cur == stand or tile_dist(cur, stand) > 7:
            return False
        return self._pathfinder.find_path(ram, cur, stand) is None

    def _note_open_stands(self, ram: np.ndarray) -> None:
        for stand, _face in self._stand_choices:
            tid = get_tile_at(ram, *stand)
            if tid in WALKABLE_TILES and not is_crop_tile(tid):
                self._pathfinder.extra_walkable.add(stand)

    def _step_target_nav(self, world: WorldState) -> TaskResult:
        self._note_open_stands(world.ram)
        self._target_nav_steps += 1
        sealed = self._target_nav_steps == 1 and self._stand_provably_unreachable(world.ram)
        if sealed:
            action, blocked = self._fallback_action(self._current_stand), True
        else:
            fast = None if self._recentre else self._current_face
            action, blocked = self._navigate_to_tile(world.ram, self._current_stand, press_face=fast, contact=True)
        self._nav_stall_steps = self._nav_stall_steps + 1 if blocked else 0
        # Re-check a sealed stand while blocked instead of idling out the stall.
        if (
            not sealed
            and blocked
            and self._nav_stall_steps in (3, 40, 120)
            and self._stand_provably_unreachable(world.ram)
        ):
            sealed = True
        if action is not None and (
            sealed
            or self._nav_stall_steps > TARGET_NAV_STALL_LIMIT
            or self._target_nav_steps > TARGET_NAV_TOTAL_LIMIT
        ):
            if self._advance_stand_choice():
                return self._running()
            step = self._current
            neighbor_left = step is not None and any(
                abs(other.target[0] - step.target[0]) + abs(other.target[1] - step.target[1]) == 1
                for other in self._steps
            )
            if neighbor_left and step.target not in self._deferred_since_pick:
                # Ring is still queued. Retry the center after a neighbor is gone.
                self._defer_target = step.target
                self._deferred_since_pick.add(step.target)
                self._steps.append(step)
                self._back_to_select()
                return self._running()
            self.skipped_count += 1
            note_harvest_tile_failure(self._current.target)
            print(
                f"[HARVEST] SKIP target={self._current.target} "
                f"(every stand unreachable: {self._stand_choices})"
            )
            self._back_to_select()
            return self._running()
        if action is not None:
            return self._running(action)
        self._target_live_before = get_tile_at(world.ram, *self._current.target)
        self._verify_count = 0
        self._phase = "target_verify"
        self._queue_press(world.ram, self._current_face)
        return self._running()

    def _advance_stand_choice(self) -> bool:
        """Switch ``_current`` to its next candidate (stand, face).

        Returns False once every candidate has been tried, so the caller can
        skip the target instead of bumping into the same wall forever.
        """
        if self._stand_choice_index + 1 >= len(self._stand_choices):
            return False
        self._stand_choice_index += 1
        self._current_stand, self._current_face = self._stand_choices[self._stand_choice_index]
        self._clear_navigation_state()
        self._target_nav_steps = 0
        print(
            f"[HARVEST] Target {self._current.target} stand unreachable, "
            f"retry stand={self._current_stand} face={self._current_face} "
            f"({self._stand_choice_index + 1}/{len(self._stand_choices)})"
        )
        return True

    def step(self, world: WorldState) -> TaskResult:
        self._step_count += 1
        self._frame = int(getattr(world, "frame", 0) or 0)
        self._ram = world.ram
        self._navigator.update(world.ram)
        if self._step_count > self.timeout:
            remaining = [step.target for step in self._steps]
            for tile in remaining + ([self._current.target] if self._current is not None else []):
                note_harvest_tile_failure(tile)
            reason = (
                f"harvest timeout harvested={self.harvested_count} "
                f"shipped={self.shipped_count} skipped={self.skipped_count} "
                f"unreachable={self._unreachable_targets} remaining={remaining}"
            )
            print(f"[HARVEST] {reason}")
            return TaskResult(status=TaskStatus.FAILURE, reason=reason)

        carrying = is_carrying(world.ram)

        action = self._pop_action()
        if action is not None:
            return self._running(action)

        if self._phase == "press":
            return self._step_press(world)
        if self._phase == "park":
            return self._step_park(world)
        if self._phase == "horse_face":
            return self._step_horse_face(world)
        if saddle.is_riding(world.ram):
            # An A beside the horse mounts it; a rider cannot pick, so get off (a pulse, never a hold).
            self._clear_navigation_state()
            return self._running(make_action(a=True) if self._step_count % 60 < 3 else make_action())

        if self._phase == "clear_hands":
            if not carrying and int(read_held_item(world.ram)) == 0:
                self._clear_hands_attempts = 0
                if self._abort_reason is not None:
                    return TaskResult(status=TaskStatus.FAILURE, reason=self._abort_reason)
                self._phase = "select"
                return self._running()
            if self._clear_hands_attempts >= CLEAR_HANDS_ATTEMPT_LIMIT:
                return TaskResult(
                    status=TaskStatus.FAILURE,
                    reason=(
                        f"clear hands timeout held=0x{int(read_held_item(world.ram)):02X} "
                        f"attempts={self._clear_hands_attempts}"
                    ),
                )
            return self._toss_now(world.ram, reason="retry")

        if self._phase == "select":
            if carrying:
                if self._should_clear_hands(world.ram):
                    # Debris left by CLEAR_FIELD (held=0x0D) is not bin-shippable.
                    return self._toss_now(world.ram, reason="non_crop_carry")
                if self._should_ship_carry(world.ram):
                    self._clear_navigation_state()
                    self._phase = "ship_nav"
                else:
                    # Carry animation in progress — wait one frame.
                    return self._running(make_action())
            else:
                if lift_in_progress(world.ram):
                    return self._running(make_action())     # a crop is still rising into the hands
                if self._park_due(world.ram) and self._plan_park(world.ram):
                    return self._running()
                self._current = self._choose_next_step(world.ram)
                if self._current is None:
                    return self._completion_result()
                if self.state_name is None and not ship_window_fits(
                    world.ram, self._navigator.current_tile, self._current.stand,
                    self._window_ship_stand(world.ram, self._current.stand),
                ):
                    left = len(self._steps) + 1
                    self._steps, self._current = [], None
                    reason = (
                        f"ship window closed: {left} ripe left for tomorrow "
                        f"harvested={self.harvested_count} shipped={self.shipped_count}"
                    )
                    print(f"[HARVEST] {reason}")
                    return TaskResult(status=TaskStatus.SUCCESS, reason=reason)
                self._clear_navigation_state()
                self._phase = "target_nav"
                self._ship_options = []
                self._ship_option_index = 0
                self._stand_choices = [
                    (self._current.stand, self._current.face),
                    *self._current.alt_stands,
                ]
                self._stand_choice_index = 0
                self._current_stand, self._current_face = self._stand_choices[0]
                self._target_nav_steps = 0
                tile = int(get_tile_at(world.ram, *self._current.target))
                self._rec = {
                    "target": list(self._current.target), "crop": mature_tile_crop(tile) or "?",
                    "from": list(self._navigator.current_tile),
                    "f_select": self._frame, "tick_select": clock_tick(world.ram),
                }
                print(
                    f"[HARVEST] Target {self._current.target} "
                    f"stand={self._current_stand} face={self._current_face}"
                )

        if self._phase == "target_nav":
            return self._step_target_nav(world)

        if self._phase == "target_verify":
            if carrying:
                return self._picked("ship_nav", "carrying=1")
            target_tile_id = get_tile_at(world.ram, *self._current.target)
            tile_changed_after_pick = (
                self._target_live_before >= 0
                and target_tile_id != self._target_live_before
                and (not is_crop_tile(target_tile_id) or target_tile_id < self._target_live_before)
            )
            if (
                (self.state_name is not None and tile_changed_after_pick)
                or (self.state_name is None and not is_crop_tile(target_tile_id))
            ):
                # A potato's tile turns to soil on the A edge, ~23 frames before the lift sets the carry
                # bit. Leaving now walks off with the crop rising and the next A drops it.
                if lift_in_progress(world.ram) and self._verify_count < LIFT_WAIT_FRAMES:
                    self._verify_count += 1
                    return self._running(make_action())
                return self._picked("ship_nav" if is_carrying(world.ram) else "select", "tile changed")
            self._verify_count += 1
            fast = FAST_PRESS and not self._recentre
            if fast and self._verify_count > LIFT_WAIT_FRAMES:
                # Missed from the tile edge: retry once centred, as the old press did.
                self.fast_misses += 1
                self._recentre = True
                self._clear_navigation_state()
                self._phase = "target_nav"
                print(f"[HARVEST] fast pick missed target={self._current.target}; retry centred")
                return self._running(make_action())
            if not fast and self._verify_count > 12:
                self.skipped_count += 1
                note_harvest_tile_failure(self._current.target)
                print(f"[HARVEST] SKIP target={self._current.target} (not ready or no pickup)")
                self._back_to_select()
            return self._running(make_action())

        if self._phase == "ship_nav":
            if not carrying:
                self._current = None
                self._phase = "select"
                return self._running()
            ship_stand, ship_face = self._current_ship_option(world.ram)
            if ship_stand in self._horse_stands and self._step_count % 15 == 0 and saddle.farm_horse(world.ram) != self._horse_at:
                self._ship_options = []  # it wandered: choose again against where it is now
                ship_stand, ship_face = self._current_ship_option(world.ram)
            bin_face = None if ship_stand in self._horse_stands else ship_face
            action, blocked = self._navigate_to_tile(world.ram, ship_stand, press_face=bin_face)
            if ship_stand in self._horse_stands:
                # Centring beside the horse never finishes (its body is in the way): the tile is enough.
                self._on_horse_stand = self._on_horse_stand + 1 if self._navigator.current_tile == ship_stand else 0
                if self._on_horse_stand >= saddle.ON_STAND_FRAMES:
                    action = None
            self._nav_stall_steps = self._nav_stall_steps + 1 if blocked else 0
            self._ship_nav_steps += 1
            if ship_stand in self._horse_stands and (
                self._nav_stall_steps > saddle.HORSE_STALL_STEPS or self._ship_nav_steps > saddle.HORSE_NAV_STEPS
            ):
                return self._horse_unusable(world.ram, "the horse is out of reach")
            if action is not None:
                if self._nav_stall_steps > SHIP_NAV_STALL_LIMIT and self._try_next_ship_option(world.ram):
                    return self._running()
                if self._nav_stall_steps > SHIP_NAV_STALL_LIMIT or self._ship_nav_steps > SHIP_NAV_TOTAL_LIMIT:
                    return self._abandon_carry(world.ram, reason="bin unreachable")
                return self._running(action)
            if ship_stand in self._horse_stands:
                self._ship_at_horse, self._horse_hold, self._phase = True, 0, "horse_face"
                return self._running()
            self._ship_at_horse = False
            self._ship_money_before = read_shipping_money(world.ram)
            self._note("throw")
            if self._rec is not None:
                self._rec.update(ship_stand=list(ship_stand), ship_face=ship_face)
            self._verify_count = 0
            self._phase = "ship_verify"
            self._queue_press(world.ram, ship_face, hold_frames=14, settle_frames=12)
            print(f"[HARVEST] SHIP from stand={ship_stand} face={ship_face} money={self._ship_money_before}")
            return self._running()

        if self._phase == "ship_verify":
            carrying = is_carrying(world.ram)
            ship_money = read_shipping_money(world.ram)
            input_lock = int(world.ram[ADDR_INPUT_LOCK]) if ADDR_INPUT_LOCK < len(world.ram) else 1
            if input_lock != 1:
                self._verify_count += 1
                if self._verify_count > 180:
                    return TaskResult(status=TaskStatus.FAILURE, reason="ship dialog timeout")
                return TaskResult(
                    status=TaskStatus.RUNNING,
                    action=ActionResult(self._dialog_pulse_action()),
                    reason="dialog",
                )
            if not carrying:
                # Bin drop succeeds immediately; shipping_money often settles at 5pm.
                if ship_money > self._ship_money_before:
                    self.shipped_count += max(1, (ship_money - self._ship_money_before) // 80)
                    self.horse_shipped += 1 if self._ship_at_horse else 0
                    print(
                        f"[HARVEST] SHIP OK money={ship_money} "
                        f"(+{ship_money - self._ship_money_before}) shipped={self.shipped_count}"
                    )
                else:
                    self.shipped_count += 1
                    if self._ship_at_horse:
                        self._saddle_off = True  # the crop went in unpaid: no more horse throws
                    from harvest.core.ram_catalog import read_ram_value

                    late = ("at 17:00+ the ROM pays nothing"
                            if int(read_ram_value(world.ram, "hour") or 0) >= SHIP_CLOSE_HOUR
                            else "no bin increase")
                    print(
                        f"[HARVEST] SHIP UNPAID (bin drop; {late}) "
                        f"money={ship_money} shipped={self.shipped_count}"
                    )
                self._close_ship_record(world.ram, ship_money - self._ship_money_before)
                self._ship_nav_steps = 0
                self._back_to_select()
                return self._running()
            self._verify_count += 1
            if self._verify_count > (THROW_WAIT_FRAMES if FAST_PRESS else 20):
                if self._try_next_ship_option(world.ram):
                    return self._running()
                # Defense in depth: debris misrouted into ship_verify → toss, not fail.
                if not self._crop_ship_pending() or is_debris_held(world.ram):
                    return self._toss_now(world.ram, reason="ship_verify_debris")
                print(
                    f"[HARVEST] ship verify timeout target="
                    f"{self._current.target if self._current else None} "
                    f"harvested={self.harvested_count} shipped={self.shipped_count} "
                    f"skipped={self.skipped_count} remaining={[s.target for s in self._steps]}"
                )
                return TaskResult(status=TaskStatus.FAILURE, reason="ship verify timeout")
            return self._running(make_action())

        return TaskResult(status=TaskStatus.FAILURE, reason=f"unknown phase {self._phase}")
