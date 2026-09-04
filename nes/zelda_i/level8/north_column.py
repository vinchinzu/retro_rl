"""Fixture-live Level 8 north-column one-frame policies (0x7E → 0x1E).

Promotes the settled interior chain into spine-composable controllers.
Not route eligible; not a power-on claim.  Local sword-clear specs are
never registered as ``DungeonRoomSpec`` rows.  Bomb walls use
``BombWallController``; north key/shutter doors use ``exit_door``
geometry (``DOOR_TARGETS['UP']``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.dungeon.bomb_wall import BombWallController, BombWallPhase
from zelda_i.dungeon.engine import (
    AliveRule,
    CombatTuning,
    DoorRoute,
    DungeonPhase,
    DungeonRoomSpec,
    GenericDungeonRoomController,
    RewardKind,
    RewardSpec,
)
from zelda_i.dungeon.hop_controller import HopController, WAIT_SCROLL_B
from zelda_i.dungeon.ids import MANHANDLA_OBJECT_TYPE
from zelda_i.dungeon.ops import B_ITEM_BOMB, DOOR_TARGETS
from zelda_i.level8.dungeon import LEVEL8
from zelda_i.ram import ADDR_SELECTED_ITEM, PLAY_MODE, ZeldaSnapshot, read_u8

# Live recon rooms.  0x0C is unregistered in dungeon.ids (0x0B is "darknut");
# colour is a walkthrough correlation, not an observation.
TYPE_0C = 0x0C
SMALL_KEY_ITEM = 0x19
ROOM_ENTRY = 0x7E
ROOM_MANHANDLA = 0x6E
ROOM_DARKNUT_KEY = 0x5E
ROOM_SHUTTER = 0x4E
ROOM_BLUE_DARKNUTS = 0x3E
ROOM_MAP_MANHANDLA = 0x2E
ROOM_GOHMA = 0x1E

NORTH_MANHANDLA_ROOMS = frozenset({ROOM_ENTRY, ROOM_MANHANDLA, ROOM_DARKNUT_KEY})
DARKNUT_KEY_ROOMS = frozenset(
    {
        ROOM_DARKNUT_KEY,
        ROOM_SHUTTER,
        ROOM_BLUE_DARKNUTS,
        ROOM_MAP_MANHANDLA,
        ROOM_GOHMA,
    }
)

NORTH_DOOR = DOOR_TARGETS["UP"]  # (120, 93)
BOMB_NORTH_STAND = (120, 105)
BOMB_NORTH_APPROACH_3E: tuple[tuple[int, int], ...] = ((120, 109),)
CENTER_KEY_STAND = (120, 141)
# Peel off the 0x2E centre column (map 0x17). Door push is separate so an
# overshoot north of y=93 is not walked back south.
MAP_SKIP_WAYPOINTS: tuple[tuple[int, int], ...] = ((88, 109), (120, 109))
MANHANDLA_SETTLE_FRAMES = 120
DARKNUT_SETTLE_FRAMES = 8
KEY_FREEZE_FRAMES = 40

_SWORD_PATROL: tuple[tuple[int, int], ...] = (
    (64, 109),
    (120, 109),
    (176, 109),
    (176, 141),
    (176, 173),
    (120, 173),
    (64, 173),
    (64, 141),
    (120, 141),
    (100, 125),
    (140, 157),
    (80, 157),
    (160, 125),
)


class BombWall6ENorth:
    """0x6E Manhandla north wall → 0x5E. Stand (120,105) face UP."""

    room = ROOM_MANHANDLA
    stand = BOMB_NORTH_STAND
    face = "UP"
    opens_to = ROOM_DARKNUT_KEY


class BombWall3ENorth:
    """0x3E north wall → 0x2E. Approach y=109 (statue row at y~141)."""

    room = ROOM_BLUE_DARKNUTS
    stand = BOMB_NORTH_STAND
    face = "UP"
    opens_to = ROOM_MAP_MANHANDLA


def _sword_clear_spec(
    room: int, types: tuple[int, ...], spec_id: str
) -> DungeonRoomSpec:
    """Local fight_clear engine row. Not registered; not on L8_THROUGH."""
    return DungeonRoomSpec(
        spec_id=spec_id,
        source_room=room,
        room_id=room,
        entry=DoorRoute("DOWN", ((120, 205),)),
        enemy_types=types,
        expected_enemy_count=1,
        alive_rule=AliveRule.TYPE_AND_HP,
        combat=CombatTuning(
            patrol=_SWORD_PATROL,
            engage_distance=48,
            attack_phase=2,
            patrol_attack_period=6,
            patrol_attack_hold=3,
            engage_attack_period=5,
            engage_attack_hold=3,
        ),
        reward=RewardSpec(kind=RewardKind.CLEAR_ONLY, settle_all_dead=0),
        max_frames=5000,
        level=LEVEL8,
    )


CLEAR_6E_SPEC = _sword_clear_spec(
    ROOM_MANHANDLA, (MANHANDLA_OBJECT_TYPE,), "l8_clear_0x6e_manhandla"
)
CLEAR_5E_SPEC = _sword_clear_spec(
    ROOM_DARKNUT_KEY, (TYPE_0C,), "l8_clear_0x5e_0x0c"
)
CLEAR_3E_SPEC = _sword_clear_spec(
    ROOM_BLUE_DARKNUTS, (TYPE_0C,), "l8_clear_0x3e_0x0c"
)
CLEAR_2E_SPEC = _sword_clear_spec(
    ROOM_MAP_MANHANDLA, (MANHANDLA_OBJECT_TYPE,), "l8_clear_0x2e_manhandla"
)


def _live_of(snap: ZeldaSnapshot, types: tuple[int, ...]) -> tuple:
    want = frozenset(types)
    return tuple(
        obj
        for obj in snap.objects
        if 1 <= obj.slot <= 12 and obj.type_id in want and obj.hp > 0
    )


def _goto(
    snap: ZeldaSnapshot, tx: int, ty: int, *, reason: str, tol: int = 4
) -> FrameAction | None:
    """One x-then-y step toward (tx, ty). None when inside *tol* (ops.goto)."""
    if abs(snap.link_x - tx) > tol:
        btn = "RIGHT" if snap.link_x < tx else "LEFT"
        return FrameAction(nes_action(btn), f"{reason}_x")
    if abs(snap.link_y - ty) > tol:
        btn = "DOWN" if snap.link_y < ty else "UP"
        return FrameAction(nes_action(btn), f"{reason}_y")
    return None


def _north_door(snap: ZeldaSnapshot, *, reason: str = "north_door") -> FrameAction:
    """Align x to the UP door, walk north to the plane, then hold UP.

    Never walk south after overshooting y=93 — that oscillates on the door
    tile (live 0x7E y=87/89). Matches exit_door UP geometry.
    """
    tx, ty = NORTH_DOOR
    if abs(snap.link_x - tx) > 4:
        btn = "RIGHT" if snap.link_x < tx else "LEFT"
        return FrameAction(nes_action(btn), f"{reason}_x")
    if snap.link_y > ty + 4:
        return FrameAction(nes_action("UP"), f"{reason}_y")
    return FrameAction(nes_action("UP"), f"{reason}_push")


class _SelectPhase(Enum):
    OPEN = auto()
    OPEN_SETTLE = auto()
    CYCLE = auto()
    CURSOR_SETTLE = auto()
    CLOSE = auto()
    CLOSE_SETTLE = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class _PauseSelectBombs:
    """Pause-cycle already-owned bombs onto B. No RAM poke."""

    phase: _SelectPhase = _SelectPhase.OPEN
    phase_frames: int = 0
    cursor_moves: int = 0
    failed: bool = False
    notes: list[str] = field(default_factory=list)

    def step(self, ram: Any) -> FrameAction | None:
        selected = int(read_u8(ram, ADDR_SELECTED_ITEM))
        if self.phase is _SelectPhase.DONE:
            return None
        if self.phase is _SelectPhase.FAILED:
            return FrameAction(nes_idle_action(), "pause_select_failed")
        if self.phase is _SelectPhase.OPEN:
            if selected == B_ITEM_BOMB:
                self.phase = _SelectPhase.DONE
                return None
            self.phase = _SelectPhase.OPEN_SETTLE
            self.phase_frames = 0
            self.notes.append("pause_open")
            return FrameAction(nes_action("START"), "pause_open")
        self.phase_frames += 1
        if self.phase is _SelectPhase.OPEN_SETTLE:
            if self.phase_frames >= 20:
                self.phase = _SelectPhase.CYCLE
                self.phase_frames = 0
            return FrameAction(nes_idle_action(), "pause_settle")
        if self.phase is _SelectPhase.CYCLE:
            if selected == B_ITEM_BOMB:
                self.phase = _SelectPhase.CLOSE
                self.phase_frames = 0
                return FrameAction(nes_idle_action(), "cursor_ready")
            if self.cursor_moves >= 8:
                self.failed = True
                self.phase = _SelectPhase.FAILED
                self.notes.append("bomb_cursor_not_found")
                return FrameAction(nes_idle_action(), "bomb_cursor_not_found")
            self.cursor_moves += 1
            self.phase = _SelectPhase.CURSOR_SETTLE
            self.phase_frames = 0
            return FrameAction(nes_action("RIGHT"), "pause_next_item")
        if self.phase is _SelectPhase.CURSOR_SETTLE:
            if self.phase_frames >= 8:
                self.phase = _SelectPhase.CYCLE
                self.phase_frames = 0
            return FrameAction(nes_idle_action(), "pause_cursor_settle")
        if self.phase is _SelectPhase.CLOSE:
            self.phase = _SelectPhase.CLOSE_SETTLE
            self.phase_frames = 0
            return FrameAction(nes_action("START"), "pause_close")
        if self.phase is _SelectPhase.CLOSE_SETTLE:
            if self.phase_frames < 24:
                return FrameAction(nes_idle_action(), "pause_resume")
            if selected != B_ITEM_BOMB:
                self.failed = True
                self.phase = _SelectPhase.FAILED
                self.notes.append("pause_close_bombs_unselected")
                return FrameAction(nes_idle_action(), "pause_close_bombs_unselected")
            self.phase = _SelectPhase.DONE
            return None
        return None


@dataclass(kw_only=True)
class _NorthColumnBase(HopController):
    """Shared L8 north-column guards: fixture-live, no writes, known rooms."""

    wait_modes: tuple[int, ...] = WAIT_SCROLL_B
    require_level: int = LEVEL8
    writes: int = 0
    evidence: str = "fixture-live"
    route_eligible: bool = False
    _env: Any = field(default=None, init=False, repr=False)
    _room: int | None = field(default=None, init=False)
    _room_frames: int = field(default=0, init=False)
    _traveled: bool = field(default=False, init=False)
    _clear: GenericDungeonRoomController | None = field(default=None, init=False, repr=False)
    _wall: BombWallController | None = field(default=None, init=False, repr=False)
    _select: _PauseSelectBombs | None = field(default=None, init=False, repr=False)
    _map_wp: int = field(default=0, init=False)
    _key_wait: int = field(default=0, init=False)
    _keys_in: int | None = field(default=None, init=False)

    def bind_env(self, env: Any) -> None:
        self._env = env

    def scroll_action(self, snap: ZeldaSnapshot) -> FrameAction:
        del snap
        return FrameAction(nes_action("UP"), "north_scroll")

    def _track_room(self, snap: ZeldaSnapshot) -> None:
        if self._room is None:
            self._room = int(snap.screen)
            return
        if int(snap.screen) != self._room:
            self._room = int(snap.screen)
            self._room_frames = 0
            self._traveled = True
            self._clear = None
            self._wall = None
            self._select = None
            self._map_wp = 0
            self._key_wait = 0
            self._keys_in = None
            return
        self._room_frames += 1

    def _spawn_wait(self, snap: ZeldaSnapshot, frames: int) -> FrameAction | None:
        if self._clear is not None and self._clear.max_live_enemies > 0:
            return None
        if self._traveled and self._room_frames < frames:
            return FrameAction(
                nes_idle_action(), f"settle_0x{snap.screen:02x}"
            )
        return None

    def _fight(self, snap: ZeldaSnapshot, spec: DungeonRoomSpec) -> FrameAction:
        if self._clear is None:
            self._clear = GenericDungeonRoomController(spec)
            self._clear.phase = DungeonPhase.FIGHT
        action = self._clear.step(snap)
        if self._clear.phase is DungeonPhase.FAILED:
            note = (
                self._clear.notes[-1]
                if self._clear.notes
                else f"clear_0x{snap.screen:02x}_failed"
            )
            if "left_target_room" in note or action.reason == "left_target_room":
                return self.mark_fail(
                    f"clear_0x{snap.screen:02x}_first_departure_guard"
                )
            return self.mark_fail(note)
        return action

    def _maybe_select_bombs(self, snap: ZeldaSnapshot) -> FrameAction | None:
        del snap
        if self._env is None:
            return None
        if self._select is None:
            self._select = _PauseSelectBombs()
        action = self._select.step(self._env.get_ram())
        if self._select.failed:
            return self.mark_fail(
                self._select.notes[-1] if self._select.notes else "pause_select_failed"
            )
        return action

    def _bomb(
        self,
        snap: ZeldaSnapshot,
        wall: Any,
        *,
        approach: tuple[tuple[int, int], ...] = (),
    ) -> FrameAction:
        select = self._maybe_select_bombs(snap)
        if select is not None:
            return select
        if snap.bombs <= 0:
            return self.mark_fail(f"no_bombs_0x{snap.screen:02x}")
        if self._wall is None:
            self._wall = BombWallController(
                wall=wall,
                level=LEVEL8,
                approach_waypoints=approach,
                max_frames=8000,
            )
        action = self._wall.step(snap)
        if self._wall.phase is BombWallPhase.FAILED:
            note = (
                self._wall.notes[-1]
                if self._wall.notes
                else f"bomb_north_0x{snap.screen:02x}_failed"
            )
            return self.mark_fail(note)
        return action

    def _north_key(self, snap: ZeldaSnapshot, *, reason: str) -> FrameAction:
        if snap.keys <= 0:
            return self.mark_fail(f"no_keys_0x{snap.screen:02x}")
        return _north_door(snap, reason=reason)

    def report(self) -> dict[str, Any]:
        out = super().report()
        out.update(
            {
                "evidence": self.evidence,
                "route_eligible": self.route_eligible,
                "natural_entry": False,
                "writes": self.writes,
                "notes": list(self.notes),
            }
        )
        return out


@dataclass(kw_only=True)
class Level8NorthManhandlaController(_NorthColumnBase):
    """0x7E UP → clear 0x6E sword-only → bomb-N (120,105) → 0x5E."""

    spec_id: str = "level8_north_manhandla_bomb"
    max_frames: int = 20_000
    done_reason: str = "arrived_0x5e"

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        return (
            snap.level == LEVEL8
            and snap.screen == ROOM_DARKNUT_KEY
            and snap.mode == PLAY_MODE
            and not snap.transitioning
        )

    def on_arrive(self, snap: ZeldaSnapshot) -> str:
        return f"arrived_0x{snap.screen:02x}_{snap.link_x}_{snap.link_y}"

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        self._track_room(snap)
        if snap.screen == ROOM_ENTRY:
            return _north_door(snap, reason="free_north_0x7e")
        if snap.screen == ROOM_MANHANDLA:
            if _live_of(snap, (MANHANDLA_OBJECT_TYPE,)):
                return self._fight(snap, CLEAR_6E_SPEC)
            wait = self._spawn_wait(snap, MANHANDLA_SETTLE_FRAMES)
            if wait is not None:
                return wait
            return self._bomb(snap, BombWall6ENorth())
        return self.mark_fail(f"l8_north_unknown_room_0x{snap.screen:02x}")


@dataclass(kw_only=True)
class Level8DarknutKeyController(_NorthColumnBase):
    """0x5E clear/key + shutter-N 0x4E + key-N 0x3E + bomb-N 0x2E + key-N 0x1E.

    0x4E mixed census is not walked.  0x2E map (room_item 0x17) is skipped
    on the y=109 band.  Stops at settled 0x1E; does not fight the 0x33 body.
    """

    spec_id: str = "level8_darknut_key_up"
    max_frames: int = 30_000
    done_reason: str = "arrived_0x1e"

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        return (
            snap.level == LEVEL8
            and snap.screen == ROOM_GOHMA
            and snap.mode == PLAY_MODE
            and not snap.transitioning
        )

    def on_arrive(self, snap: ZeldaSnapshot) -> str:
        return f"arrived_0x{snap.screen:02x}_{snap.link_x}_{snap.link_y}"

    def _map_skip_north(self, snap: ZeldaSnapshot) -> FrameAction:
        while self._map_wp < len(MAP_SKIP_WAYPOINTS):
            tx, ty = MAP_SKIP_WAYPOINTS[self._map_wp]
            action = _goto(snap, tx, ty, reason=f"map_skip_{self._map_wp}", tol=5)
            if action is not None:
                return action
            self._map_wp += 1
        return _north_door(snap, reason="north_key_0x2e")

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        self._track_room(snap)
        room = int(snap.screen)
        if room == ROOM_DARKNUT_KEY:
            if self._keys_in is None:
                self._keys_in = int(snap.keys)
            if _live_of(snap, (TYPE_0C,)):
                return self._fight(snap, CLEAR_5E_SPEC)
            wait = self._spawn_wait(snap, DARKNUT_SETTLE_FRAMES)
            if wait is not None:
                return wait
            # room_item 0x19 can linger after the natural pickup; latch on
            # the key-count rise (live 9→10) so we do not walk back south.
            if snap.keys <= self._keys_in and snap.room_item_id == SMALL_KEY_ITEM:
                action = _goto(
                    snap, CENTER_KEY_STAND[0], CENTER_KEY_STAND[1],
                    reason="center_key", tol=3,
                )
                if action is not None:
                    return action
                self._key_wait += 1
                if self._key_wait < KEY_FREEZE_FRAMES:
                    return FrameAction(nes_idle_action(), "center_key_freeze")
            return _north_door(snap, reason="shutter_north_0x5e")
        if room == ROOM_SHUTTER:
            # Mixed 0x4E census is not a clear target; north key is live.
            return self._north_key(snap, reason="key_north_0x4e")
        if room == ROOM_BLUE_DARKNUTS:
            if _live_of(snap, (TYPE_0C,)):
                return self._fight(snap, CLEAR_3E_SPEC)
            wait = self._spawn_wait(snap, DARKNUT_SETTLE_FRAMES)
            if wait is not None:
                return wait
            return self._bomb(
                snap, BombWall3ENorth(), approach=BOMB_NORTH_APPROACH_3E
            )
        if room == ROOM_MAP_MANHANDLA:
            if _live_of(snap, (MANHANDLA_OBJECT_TYPE,)):
                return self._fight(snap, CLEAR_2E_SPEC)
            wait = self._spawn_wait(snap, MANHANDLA_SETTLE_FRAMES)
            if wait is not None:
                return wait
            return self._map_skip_north(snap)
        return self.mark_fail(f"l8_north_unknown_room_0x{snap.screen:02x}")


def make_north_manhandla_controller() -> Level8NorthManhandlaController:
    return Level8NorthManhandlaController()


def make_darknut_key_controller() -> Level8DarknutKeyController:
    return Level8DarknutKeyController()


__all__ = [
    "BOMB_NORTH_APPROACH_3E",
    "BOMB_NORTH_STAND",
    "BombWall3ENorth",
    "BombWall6ENorth",
    "DARKNUT_KEY_ROOMS",
    "Level8DarknutKeyController",
    "Level8NorthManhandlaController",
    "MAP_SKIP_WAYPOINTS",
    "NORTH_MANHANDLA_ROOMS",
    "ROOM_BLUE_DARKNUTS",
    "ROOM_DARKNUT_KEY",
    "ROOM_ENTRY",
    "ROOM_GOHMA",
    "ROOM_MANHANDLA",
    "ROOM_MAP_MANHANDLA",
    "ROOM_SHUTTER",
    "TYPE_0C",
    "make_darknut_key_controller",
    "make_north_manhandla_controller",
]
