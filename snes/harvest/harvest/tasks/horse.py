"""Saddlebag shipping: an adult horse takes a thrown crop straight into the shipping total.

ROM facts (probed on research pins ``LN_saddle_whatif_*``; decomp bank_81 ``CODE_819195`` and
bank_84 ``Event_UNKPlayerPos``):

- A throw (``player_action`` 5) that hits a sprite pays the item's bin price into ``shipping_money``
  and bumps its crop counter only while ``$7F1F5C & 0x0010`` is set and the hour is below 17. With
  the bit clear the crop is consumed and nothing is paid.
- The bit is raised by the horse's event about five frames after an adult horse (``horse_age``
  0x15) has the farmer on the tile north, south or west of it, facing it. East did not work.
- The same A, pressed with empty hands while the bit is up, mounts the horse (``player_state``
  0x11). A second A dismounts one tile south of the horse, and the horse stays where it stood.
  Riding is not faster than walking, so the only use of the ride is to park the horse.
- The horse starts every morning at tile (17,20) and wanders a few pixels. The bin is closer to
  every ring except the far east ones, so ``plan_park`` moves the horse next to those.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Tuple

import numpy as np

from harvest.core.npc_catalog import FARM_HORSE_SPRITES as HORSE_SPRITES
from harvest.core.ram_catalog import read_ram_value
from harvest.tasks.nav import WALKABLE_TILES, get_tile_at, make_action

Tile = Tuple[int, int]

HORSE_ADULT_AGE = 0x15
SADDLE_BIT = 0x0010  # $7F1F5C
RIDING_BIT = 0x10  # player_state, on top of the walking bit
CLOSE_MINUTE = 16 * 60 + 55  # the ROM pays nothing from 17:00; a throw takes ~2 game minutes
SHIP_CLOSE_MINUTE = 17 * 60
FARM_TILEMAP = 0x00

# N, W, S, E of the horse's tile: (dx, dy, direction that faces the horse). East is last: it never
# raised the bit in the probe, so it is only a fallback.
HORSE_STANDS: Tuple[Tuple[int, int, str], ...] = ((0, -1, "down"), (-1, 0, "right"), (0, 1, "up"), (1, 0, "left"))

ON_STAND_FRAMES = 6  # on the stand tile this long counts as arrived: the horse's body keeps the farmer off the exact centre
HORSE_STALL_STEPS = 150  # blocked this long on the way to a horse stand: ship at the bin
HORSE_NAV_STEPS = 600  # or this long in all (it wanders; ROM probe: it was 1,000 frames away)
FACE_FRAMES = 90  # hold the direction this long for the bit before giving up on a stand
THROW_SETTLE_FRAMES = 3
THROW_HOLD_FRAMES = 14
PARK_TOTAL_FRAMES = 4500
PARK_STALL_FRAMES = 600
ARRIVE_FRAMES = 20  # frames on the park tile before the rider gets off
MOUNT_TRIES = 3
DISMOUNT_TRIES = 4
# Parking is for the crops the bin walk cannot afford. The A/B over Y2 Summer D13-D30 (sb0/sb1)
# showed an up-front park loses: the harvest starts about 12:00, the near rings are cheaper at
# the bin than any far ring at the horse, and the ride ate the window. So park only once the
# near crops are done and this much of the day is left.
PARK_LATEST_MINUTE = 14 * 60 + 30
PARK_FAR_FROM_BIN = 16  # every remaining pick stand is at least this far from the bin
PARK_MINUTES_PER_TILE = 0.6
PARK_MINUTES_PER_CROP = 17.0  # measured: ~235 frames a crop beside the horse, 14.5 frames a minute
PARK_MIN_CROPS = 6
PARK_MIN_SAVING_TILES = 80  # walking tiles saved over the pass, both directions
PARK_MIN_RATIO = 2.0  # saving over the cost of fetching, riding and mounting
MOUNT_COST_TILES = 14  # ~130 frames of mount and dismount, in walking tiles
PARK_SEARCH_RADIUS = 3
# A horse beside a ring can stand on a watering stand later in the day. Prefer a park tile
# with no crop around it unless that costs more than this many tiles over all the picks.
CROP_ADJACENT_PENALTY = 30


def saddle_enabled() -> bool:
    """Opt in with ``HARVEST_SADDLE=1``. Off by default: the A/B over Y2 Summer D13-D30 (one draw
    each, same pin) was 144,310G and 485 corn without it, 135,400G / 402 with the up-front park
    and 141,130G / 451 with the late park, because the harvest starts about 12:00 and the near
    rings are cheaper at the bin than any ring at the horse (see docs/Y1_BENCH.md)."""
    return os.environ.get("HARVEST_SADDLE", "0") == "1"


def _field(ram: np.ndarray, key: str) -> int:
    return int(read_ram_value(ram, key, raw=True))


def horse_adult(ram: np.ndarray) -> bool:
    return _field(ram, "horse_age") == HORSE_ADULT_AGE


def saddle_ready(ram: np.ndarray) -> bool:
    """The horse is letting a thrown crop through right now."""
    return bool(_field(ram, "event_flags_1f5c") & SADDLE_BIT)


def is_riding(ram: np.ndarray) -> bool:
    return bool(_field(ram, "player_state") & RIDING_BIT)


def _tile_of(ram: np.ndarray) -> Tile:
    from harvest.tasks.nav import get_pos_from_ram

    pos = get_pos_from_ram(ram)
    return (pos.x // 16, pos.y // 16)


def clock_minute(ram: np.ndarray) -> int:
    return _field(ram, "hour") * 60 + _field(ram, "minute")


def before_close(ram: np.ndarray) -> bool:
    return clock_minute(ram) <= CLOSE_MINUTE


def farm_horse(ram: np.ndarray) -> Optional[Tile]:
    """Tile of the horse sprite on the farm, or None (indoors, mounted, or not out)."""
    from harvest.core.npc_catalog import iter_game_objects

    if _field(ram, "tilemap") != FARM_TILEMAP:
        return None
    for obj in iter_game_objects(ram):
        if not obj.is_player and obj.sprite_table_idx in HORSE_SPRITES:
            return (obj.pixel[0] // 16, obj.pixel[1] // 16)
    return None


def saddle_usable(ram: np.ndarray) -> Optional[Tile]:
    """The horse's tile when a crop thrown at it would pay, else None."""
    if not saddle_enabled() or not horse_adult(ram) or is_riding(ram) or not before_close(ram):
        return None
    return farm_horse(ram)


def horse_stands(ram: np.ndarray, horse: Tile, *, open_tile: Optional[Callable[[Tile], bool]] = None) -> list[Tuple[Tile, str]]:
    """Walkable (stand, facing) pairs around the horse, best first."""
    open_tile = open_tile or default_open_tile(ram)
    out = []
    for dx, dy, face in HORSE_STANDS:
        stand = (horse[0] + dx, horse[1] + dy)
        if open_tile(stand):
            out.append((stand, face))
    return out


def default_open_tile(ram: np.ndarray) -> Callable[[Tile], bool]:
    from harvest.maps.farm_pond import FARM_NO_GO_TILES
    from harvest.tasks.crop_planter import is_crop_tile

    def open_tile(tile: Tile) -> bool:
        if tile in FARM_NO_GO_TILES or not (0 <= tile[0] < 64 and 0 <= tile[1] < 64):
            return False
        tid = get_tile_at(ram, *tile)
        return tid in WALKABLE_TILES and not is_crop_tile(tid)

    return open_tile


def crop_adjacent(ram: np.ndarray) -> Callable[[Tile], bool]:
    """True for a tile with a crop on any of its eight neighbours."""
    from harvest.tasks.crop_planter import is_crop_tile

    def adjacent(tile: Tile) -> bool:
        return any(
            is_crop_tile(get_tile_at(ram, tile[0] + dx, tile[1] + dy))
            for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy
        )

    return adjacent


def _dist(a: Tile, b: Tile) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


@dataclass(frozen=True)
class ParkPlan:
    park: Tile  # the tile the farmer dismounts on; the horse stays here, the farmer ends one south
    saving_tiles: int
    cost_tiles: int


def plan_park(
    pick_stands: Sequence[Tile],
    *,
    bin_stand: Tile,
    here: Tile,
    horse: Tile,
    open_tile: Callable[[Tile], bool],
    crop_adjacent: Optional[Callable[[Tile], bool]] = None,
    minutes_left: Optional[int] = None,
) -> Optional[ParkPlan]:
    """Where to leave the horse so the ripe crops ship from beside it, or None if it does not pay.

    Every ripe crop is a round trip to the bin (twice its distance) or to the horse. The best
    park tile minimises the sum of the shorter of the two. It pays when the walking saved is
    clearly more than fetching the horse, riding to the field and mounting twice.
    """
    stands = list(pick_stands)
    if len(stands) < 4:
        return None
    baseline = sum(_dist(s, bin_stand) for s in stands)
    candidates: set[Tile] = set()
    for sx, sy in stands:
        for dx in range(-PARK_SEARCH_RADIUS, PARK_SEARCH_RADIUS + 1):
            for dy in range(-PARK_SEARCH_RADIUS, PARK_SEARCH_RADIUS + 1):
                tile = (sx + dx, sy + dy)
                if open_tile(tile) and open_tile((tile[0], tile[1] + 1)):
                    candidates.add(tile)
    best: Optional[Tuple[int, int, Tile]] = None
    for tile in sorted(candidates):
        # The farmer stands one tile from the horse when throwing, so count that step.
        total = sum(min(_dist(s, bin_stand), _dist(s, tile) + 1) for s in stands)
        penalty = CROP_ADJACENT_PENALTY if crop_adjacent is not None and (
            crop_adjacent(tile) or crop_adjacent((tile[0], tile[1] + 1))
        ) else 0
        if best is None or total + penalty < best[0] + best[1]:
            best = (total, penalty, tile)
    if best is None:
        return None
    saving = 2 * (baseline - best[0])
    cost = _dist(here, horse) + _dist(horse, best[2]) + MOUNT_COST_TILES
    if saving < PARK_MIN_SAVING_TILES or saving < PARK_MIN_RATIO * cost:
        return None
    if minutes_left is not None:
        crops_after_ride = (minutes_left - PARK_MINUTES_PER_TILE * cost) / PARK_MINUTES_PER_CROP
        if min(crops_after_ride, len(stands)) < PARK_MIN_CROPS:
            return None
    return ParkPlan(park=best[2], saving_tiles=saving, cost_tiles=cost)


WalkFn = Callable[[np.ndarray, Tile], Tuple[Optional[np.ndarray], bool]]


@dataclass
class ParkHorse:
    """Walk to the horse, mount it, ride to ``plan.park`` and dismount.

    ``step`` returns ``(action, outcome)``. ``outcome`` stays None until it is over: ``"done"``
    (the horse stands at the park tile) or ``"abandon: ..."`` (the farmer is on foot again and
    the harvest ships at the bin). The walking itself is the caller's, so it shares the harvest
    task's pathfinder, stall handling and fallback moves.
    """

    plan: ParkPlan
    state: str = "approach"
    _frames: int = field(default=0, init=False)
    _stall: int = field(default=0, init=False)
    _hold: int = field(default=0, init=False)
    _face: str = field(default="down", init=False)
    _mount_tries: int = field(default=0, init=False)
    _dismount_tries: int = field(default=0, init=False)
    _result: str = field(default="done", init=False)
    _on_park: int = field(default=0, init=False)
    _on_stand: int = field(default=0, init=False)

    def _abandon(self, ram: np.ndarray, why: str) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """Give up. A rider gets off first so the harvest finds the farmer on foot."""
        self._result = f"abandon: {why}"
        if is_riding(ram):
            self.state, self._hold = "dismount", 0
            return make_action(), None
        return None, self._result

    def step(self, ram: np.ndarray, walk: WalkFn) -> Tuple[Optional[np.ndarray], Optional[str]]:
        self._frames += 1
        riding = is_riding(ram)
        if self._frames > PARK_TOTAL_FRAMES and self.state != "dismount":
            return self._abandon(ram, "park timeout")
        if self.state == "approach":
            if riding:
                self.state = "ride"
                return make_action(), None
            return self._approach(ram, walk)
        if self.state == "face":
            return self._face_horse(ram)
        if self.state == "mount":
            return self._mount(ram)
        if self.state == "ride":
            return self._ride(ram, walk)
        return self._dismount(ram)

    def _approach(self, ram: np.ndarray, walk: WalkFn) -> Tuple[Optional[np.ndarray], Optional[str]]:
        horse = farm_horse(ram)
        if horse is None:
            return None, "abandon: no horse on the farm"
        stands = horse_stands(ram, horse)
        if not stands:
            return None, "abandon: no open tile beside the horse"
        stand, self._face = stands[0]
        action, blocked = walk(ram, stand)
        self._stall = self._stall + 1 if blocked else 0
        self._on_stand = self._on_stand + 1 if _tile_of(ram) == stand else 0
        if self._stall > PARK_STALL_FRAMES:
            return self._abandon(ram, "cannot reach the horse")
        if action is not None and self._on_stand < ON_STAND_FRAMES:
            return action, None
        self.state, self._hold = "face", 0
        return make_action(), None

    def _face_horse(self, ram: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[str]]:
        if saddle_ready(ram):
            self.state, self._hold = "mount", 0
            return make_action(), None
        self._hold += 1
        if self._hold > FACE_FRAMES:
            return self._abandon(ram, "the horse never took the saddle bit")
        return make_action(**{self._face: True}), None

    def _mount(self, ram: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[str]]:
        if is_riding(ram):
            self.state, self._stall = "ride", 0
            return make_action(), None
        self._hold += 1
        # 3 neutral frames, a 3-frame A, then 60 frames for the mount animation.
        phase = self._hold % 66
        if self._hold > 66 * MOUNT_TRIES:
            return self._abandon(ram, "the horse would not take a rider")
        if 3 <= phase < 6:
            return make_action(a=True), None
        return make_action(), None

    def _ride(self, ram: np.ndarray, walk: WalkFn) -> Tuple[Optional[np.ndarray], Optional[str]]:
        if not is_riding(ram):
            return None, "abandon: lost the mount"
        action, blocked = walk(ram, self.plan.park)
        self._stall = self._stall + 1 if blocked else 0
        # The centring move after the last tile never reports "centred" on a mounted farmer
        # (ROM probe: 3,900 frames of it), so being on the park tile for a moment is arrival.
        self._on_park = self._on_park + 1 if _tile_of(ram) == self.plan.park else 0
        if action is not None and self._stall <= PARK_STALL_FRAMES and self._on_park < ARRIVE_FRAMES:
            return action, None
        self.state, self._hold = "dismount", 0
        return make_action(), None

    def _dismount(self, ram: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[str]]:
        if not is_riding(ram):
            return None, self._result
        self._hold += 1
        if self._hold > 90 * DISMOUNT_TRIES:
            return None, "abandon: could not dismount"
        idle = _field(ram, "player_action") == 0 and _field(ram, "input_lock") == 1
        if idle and self._hold % 90 < 3:
            return make_action(a=True), None
        return make_action(), None


def throw_actions() -> list[np.ndarray]:
    """A held crop at a horse whose bit is up: settle, a held A (the throw), settle again."""
    return (
        [make_action()] * THROW_SETTLE_FRAMES
        + [make_action(a=True)] * THROW_HOLD_FRAMES
        + [make_action()] * 12
    )


__all__ = [
    "HORSE_SPRITES",
    "ParkHorse",
    "ParkPlan",
    "SHIP_CLOSE_MINUTE",
    "clock_minute",
    "crop_adjacent",
    "default_open_tile",
    "farm_horse",
    "horse_adult",
    "horse_stands",
    "is_riding",
    "plan_park",
    "saddle_enabled",
    "saddle_ready",
    "saddle_usable",
    "throw_actions",
]
