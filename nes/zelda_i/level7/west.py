"""Level 7 red-candle mainline dest hops and recon east-spur leftovers.

``Room68NorthController`` through ``Room49UpController`` / ``Room09DownController``
are the candle-path dest hops with no other owner. ``Room6CEastController``,
``Room68DownController``, and ``Room58NorthController`` are recon-only spurs
used by hops factories and scratch probes. Shared combat helpers live in
``level7.path``.
"""

from __future__ import annotations

from zelda_i.dungeon.hop_controller import lattice_door_step

from dataclasses import dataclass, field
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.combat import nearest_enemy, should_swing_at
from zelda_i.dungeon.behaviors import EnemyKind, engagement_hint
from zelda_i.dungeon.hop_controller import (
    HopController,
    LatticeDoorWalker,
    dungeon_align_then_push,
)
from zelda_i.level7.graph import (
    BOMB_UPGRADE,
    DIGDOGGER_2,
    DODONGOS_UPGRADE,
    GORIYA_BUBBLE,
    GORIYA_COMPASS,
    LEVEL7_ROOM_BY_ID,
    ROPES_KEY,
    STALFOS_KEY,
    WEST_LOCK_SKIP,
)
from zelda_i.level7.path import (
    DOOR_Y_TOL,
    ENTRY_SCREEN,
    LEVEL7,
    NORTH_X_TOL,
    ROOM_08,
    ROOM_18,
    ROOM_69,
    ROOM_6A,
    ROOM_6B,
    _SWING_HOLD,
    _SWING_PERIOD,
    _goriya_fight,
    _projectiles,
    live_goriyas,
    live_keese,
)
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot

# 0x6C (DIGDOGGER_1): entry (16,141) west mouth, digdogger 0x38 + statue 0x55.
# East dest is live $EB=0x6D (STALFOS_KEY — stalfos 0x2a + small_key 0x19,
# a dead-end).  Traverse rides the y=141 band east; a digdogger bump nudges
# Link through the east door.  2/2 byte-identical
# (recordings/6c_right_v1/v2.json).
ROOM_6C = 0x6C
ROOM_6C_DOOR_Y = 141
ROOM_6C_EAST_PLANE = 224
ROOM6C_EAST_MAX_FRAMES = 4000
# 0x68 (KEESE_TRAPS): dark, 4 blade traps 0x49 (corners) + 4 keese 0x1b.
# Reached via the 0x69 west bomb wall (Link enters ~(208,93) NE).  The north
# door to live $EB=0x58 (DODONGOS_UPGRADE) is OPEN — align x=120 on the top,
# push UP.  2/2 byte-identical (recordings/68_up_v1/v2.json).
ROOM_68 = 0x68
ROOM_68_NORTH_X = 120
ROOM_68_TOP_BAND_Y = 93
ROOM68_NORTH_MAX_FRAMES = 4000
# 0x68 DOWN: OPEN south door to live $EB=0x78 (ROPES_KEY — ropes 0x28 +
# small_key 0x19 on the floor, dead-end).  Blade traps 0x49 in the four
# corners; OccupancyWalker poisons on knockback so this is a waypoint
# micro: peel west to x=160 (off the east trap column), drop to y=141
# (between trap rows y~93 / y~189), align x=120, push DOWN.  If knocked
# onto the y~189 trap row off-x, rise first.  2/2 byte-identical
# (recordings/68_down_v3/v4.json, arrived frame 297).
ROOM_68_SOUTH_X = 120
ROOM_68_SAFE_X = 160
ROOM_68_MID_Y = 141
ROOM_68_TRAP_ROW_Y = 189
ROOM68_DOWN_MAX_FRAMES = 4000
# 0x58 (DODONGOS_UPGRADE): dark, 3x invulnerable roamers 0x31 (hp 240) —
# dodge, do NOT try to kill.  Link spawns bottom (120,205).  The EAST door
# to live $EB=0x59 (GORIYA_COMPASS) is OPEN (keys unchanged).  A central
# structure walls the y=141 band west of x~129, so the route climbs the
# east-open column: (120,165) → (200,165) → (200,141) → push RIGHT.
# 2/2 byte-identical (recordings/58_east_v2/v3.json).
ROOM_58 = 0x58
ROOM_58_EAST_COLUMN_X = 200
ROOM_58_MID_Y = 165
ROOM_58_DOOR_Y = 141
ROOM58_EAST_MAX_FRAMES = 4000
# 0x58 NORTH: KEY door (keys 4→3) to live $EB=0x48 (BOMB_UPGRADE — bubble
# 0x40 + 0x4f, old-man "I BET YOU'D LIKE TO HAVE -100" bomb-capacity,
# dead-end).  A central 2-block mass walls the x=120 column around y=141,
# so east-around: climb y=165, RIGHT to x=160, UP to y=93, align x=120,
# push UP.  Dodge 3x invuln 0x31; do not fight.  Do NOT write max_bombs
# (upgrade is a 100-rupee purchase, unobserved).  2/2 byte-identical
# (recordings/58_north_v2/v3.json, arrived frame 337).
ROOM_58_NORTH_X = 120
ROOM_58_NORTH_EAST_X = 160
ROOM_58_NORTH_MID_Y = 165
ROOM_58_NORTH_TOP_Y = 93
ROOM58_NORTH_MAX_FRAMES = 4000
# 0x59 (GORIYA_COMPASS): lit; goriya 0x05 + 0x06 + boomerang 0x5c.  Entry
# (16,141) W mouth.  The kill-clear opens the UP door bit (cur_opened_doors
# bit 3 = UP) but the naive clear boxes Link at (48,125).  A central mass
# fills ~x100..190 / y118..165; the route is a perimeter waypoint micro:
# rise the west side to the y~100 open band, go west to x~44, rise to the
# y~64 top band, cross to x=120, push UP.  Dest is live $EB=0x49
# (GORIYA_BUBBLE — goriya 0x05 + keese 0x1b + bubble residual 0x2b, entry
# (120,205) S mouth).  2/2 byte-identical (recordings/59_up_v2/v3.json,
# arrived frame 2329).  RIGHT (KILL_CLEAR) -> COMPASS stays a hyp dead-end.
ROOM_59 = 0x59
ROOM_59_WEST_MOUTH = (16, 141)
ROOM_59_MID_BAND_Y = 100
ROOM_59_TOP_BAND_Y = 93
ROOM_59_WEST_COLUMN_X = 44
ROOM_59_NORTH_X = 120
ROOM_59_NORTH_PLANE_Y = 93
ROOM59_UP_MAX_FRAMES = 5000
# 0x49 (GORIYA_BUBBLE): entry (120,205) S mouth. Kill-clear goriya 0x05 then
# walk UP across the full-width water moat (~y120, tile 0xF4) at x=120.
# Stepladder required (recon fixture pokes ADDR_LADDER). Dest live $EB=0x39
# (DIGDOGGER_2: digdogger 0x38 + statue 0x55). 2/2 (49_up_v1/v2).
ROOM_49 = 0x49
ROOM_49_NORTH_X = 120
ROOM_49_NORTH_PLANE_Y = 93
ROOM_49_NORTH_BAND_Y = 109  # land north of the moat (walkable east-west)
ROOM_49_MOAT_SOUTH_Y = 133  # first solid row without the Stepladder
ROOM49_UP_MAX_FRAMES = 7000


def east_of_room6c_ram_id() -> int | None:
    """Live ``$EB`` of the room east of ``0x6C`` (STALFOS_KEY), or None."""
    return LEVEL7_ROOM_BY_ID[STALFOS_KEY].ram_id


def north_of_room68_ram_id() -> int | None:
    """Live ``$EB`` of the room north of ``0x68`` (DODONGOS_UPGRADE), or None."""
    return LEVEL7_ROOM_BY_ID[DODONGOS_UPGRADE].ram_id


def east_of_room58_ram_id() -> int | None:
    """Live ``$EB`` of the room east of ``0x58`` (GORIYA_COMPASS), or None."""
    return LEVEL7_ROOM_BY_ID[GORIYA_COMPASS].ram_id


def north_of_room59_ram_id() -> int | None:
    """Live ``$EB`` of the room north of ``0x59`` (GORIYA_BUBBLE), or None."""
    return LEVEL7_ROOM_BY_ID[GORIYA_BUBBLE].ram_id


def north_of_room49_ram_id() -> int | None:
    """Live ``$EB`` of the room north of ``0x49`` (DIGDOGGER_2), or None."""
    return LEVEL7_ROOM_BY_ID[DIGDOGGER_2].ram_id


def south_of_room68_ram_id() -> int | None:
    """Live ``$EB`` of the room south of ``0x68`` (ROPES_KEY), or None."""
    return LEVEL7_ROOM_BY_ID[ROPES_KEY].ram_id


def north_of_room58_ram_id() -> int | None:
    """Live ``$EB`` of the room north of ``0x58`` (BOMB_UPGRADE), or None."""
    return LEVEL7_ROOM_BY_ID[BOMB_UPGRADE].ram_id


@dataclass(kw_only=True)
class _WestHop(HopController):
    """Shared dest / arrive / report for L7 west hops. Not a new dispatcher."""

    dest: int | None = None
    origin: frozenset[int] = frozenset()
    door: str = ""

    @property
    def stage_id(self) -> str:
        return self.spec_id

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        if (
            snap.level != LEVEL7
            or snap.mode != PLAY_MODE
            or snap.transitioning
            or snap.screen in self.origin
        ):
            return False
        if self.dest is not None:
            return snap.screen == self.dest
        return True

    def on_arrive(self, snap: ZeldaSnapshot) -> str:
        return f"arrived_0x{snap.screen:02x}_{snap.link_x}_{snap.link_y}"

    def timeout_note(self, snap: ZeldaSnapshot) -> str:
        extra = self._timeout_extra()
        note = (
            f"timeout_0x{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
            f"_mode={snap.mode}"
        )
        return f"{note}_{extra}" if extra else note

    def _timeout_extra(self) -> str:
        return ""

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "spec_id": self.spec_id,
            "stage_id": self.spec_id,
            "dest_screen": self.dest,
            "evidence": "fixture-live",
            "route_eligible": False,
            "door": self.door,
        }


def room_6c_east_step(
    snap: ZeldaSnapshot,
    *,
    dest: int | None = None,
    stuck: int = 0,
    bump: int = 0,
) -> FrameAction:
    """One frame of 0x6C west mouth → live east dest 0x6D (STALFOS_KEY).

    Ride the ``y=141`` band east; the digdogger ``0x38`` may briefly block —
    a short UP bump (managed by the controller) nudges Link past it and the
    east door.
    """
    if snap.level != LEVEL7:
        return FrameAction(nes_idle_action(), "wait_level7")
    if snap.transitioning:
        return FrameAction(nes_action("RIGHT"), "east6c_scroll")
    if snap.mode != PLAY_MODE:
        return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
    if dest is not None and snap.screen == dest:
        return FrameAction(nes_idle_action(), "east6c_arrived")
    if snap.screen != ROOM_6C:
        return FrameAction(nes_idle_action(), f"unexpected_room_0x{snap.screen:02x}")

    y = int(snap.link_y)
    if bump > 0:
        return FrameAction(nes_action("UP"), "east6c_bump")
    if y > ROOM_6C_DOOR_Y + DOOR_Y_TOL:
        return FrameAction(nes_action("UP"), "east6c_drop_up")
    if y < ROOM_6C_DOOR_Y - DOOR_Y_TOL:
        return FrameAction(nes_action("DOWN"), "east6c_drop_down")
    return FrameAction(nes_action("RIGHT"), "east6c_push")


@dataclass(kw_only=True)
class Room6CEastController(_WestHop):
    """0x6C west mouth → east door to live dest 0x6D (STALFOS_KEY)."""

    spec_id: str = "level7_room6c_east"
    max_frames: int = ROOM6C_EAST_MAX_FRAMES
    require_level: int = LEVEL7
    done_reason: str = "left_0x6c"
    dest: int | None = field(default_factory=east_of_room6c_ram_id)
    origin: frozenset[int] = frozenset(
        {ENTRY_SCREEN, ROOM_69, ROOM_6A, ROOM_6B, ROOM_6C}
    )
    door: str = "RIGHT"
    _last_x: int | None = None
    _stuck: int = 0
    _bump: int = 0

    def _timeout_extra(self) -> str:
        return f"stuck={self._stuck}"

    def scroll_action(self, snap: ZeldaSnapshot) -> FrameAction:
        if snap.screen == ROOM_6B:
            return self.mark_fail("west_backtrack")
        return FrameAction(nes_action("RIGHT"), "east6c_scroll")

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        x = int(snap.link_x)
        if self._bump > 0:
            self._bump -= 1
        elif self._last_x is not None and x == self._last_x:
            self._stuck += 1
            if self._stuck > 6:
                self._bump = 10
                self._stuck = 0
        else:
            self._stuck = 0
        self._last_x = x
        action = room_6c_east_step(snap, dest=self.dest, bump=self._bump)
        if action.reason.startswith("unexpected_room"):
            if snap.screen == ROOM_6B:
                return self.mark_fail("west_backtrack")
            return self.mark_fail(action.reason)
        return action


def room_68_north_step(
    snap: ZeldaSnapshot,
    *,
    dest: int | None = None,
) -> FrameAction:
    """One frame of 0x68 (KEESE_TRAPS) → OPEN north door → live dest 0x58.

    Link enters ~(208,93) from the 0x69 west bomb wall.  Route: ride the
    top band west to ``x=120``, push UP.  Blade traps 0x49 / keese only chip;
    assist soaks it.
    """
    if snap.level != LEVEL7:
        return FrameAction(nes_idle_action(), "wait_level7")
    if snap.transitioning:
        return FrameAction(nes_action("UP"), "north68_scroll")
    if snap.mode != PLAY_MODE:
        return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
    if dest is not None and snap.screen == dest:
        return FrameAction(nes_idle_action(), "north68_arrived")
    if snap.screen != ROOM_68:
        return FrameAction(nes_idle_action(), f"unexpected_room_0x{snap.screen:02x}")

    x, y = int(snap.link_x), int(snap.link_y)
    if y > ROOM_68_TOP_BAND_Y + DOOR_Y_TOL and abs(x - ROOM_68_NORTH_X) > NORTH_X_TOL:
        return FrameAction(nes_action("UP"), "north68_rise")
    if abs(x - ROOM_68_NORTH_X) > NORTH_X_TOL:
        btn = "LEFT" if x > ROOM_68_NORTH_X else "RIGHT"
        return FrameAction(nes_action(btn), "north68_align_x")
    return FrameAction(nes_action("UP"), "north68_push")


@dataclass(kw_only=True)
class Room68NorthController(_WestHop):
    """0x68 KEESE_TRAPS → OPEN north door to live dest 0x58 (DODONGOS_UPGRADE).

    Recon-wired only.  Reached via the 0x69 west bomb wall.
    """

    spec_id: str = "level7_room68_north"
    max_frames: int = ROOM68_NORTH_MAX_FRAMES
    require_level: int = LEVEL7
    done_reason: str = "left_0x68_north"
    dest: int | None = field(default_factory=north_of_room68_ram_id)
    origin: frozenset[int] = frozenset(
        {ENTRY_SCREEN, ROOM_69, ROOM_6A, ROOM_6B, ROOM_68}
    )
    door: str = "UP"

    def scroll_action(self, snap: ZeldaSnapshot) -> FrameAction:
        return FrameAction(nes_action("UP"), "north68_scroll")

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        action = room_68_north_step(snap, dest=self.dest)
        if action.reason.startswith("unexpected_room"):
            return self.mark_fail(action.reason)
        return action


@dataclass(kw_only=True)
class Room58EastController(_WestHop):
    """0x58 (DODONGOS_UPGRADE) → OPEN east door to live dest 0x59.

    3x invulnerable 0x31 roamers are dodged (assist soaks chip damage).
    Waypoint micro up the east-open column: (120,165) → (200,165) →
    (200,141) → push RIGHT.  Recon-wired only.
    """

    spec_id: str = "level7_room58_east"
    max_frames: int = ROOM58_EAST_MAX_FRAMES
    require_level: int = LEVEL7
    done_reason: str = "left_0x58"
    dest: int | None = field(default_factory=east_of_room58_ram_id)
    origin: frozenset[int] = frozenset(
        {ENTRY_SCREEN, ROOM_69, ROOM_68, ROOM_58}
    )
    door: str = "RIGHT"
    _phase: str = "climb"

    def _timeout_extra(self) -> str:
        return f"phase={self._phase}"

    def scroll_action(self, snap: ZeldaSnapshot) -> FrameAction:
        return FrameAction(nes_action("RIGHT"), "east58_scroll")

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        if snap.screen != ROOM_58:
            return self.mark_fail(f"unexpected_room_0x{snap.screen:02x}")
        x, y = int(snap.link_x), int(snap.link_y)
        if self._phase == "climb":
            if y > ROOM_58_MID_Y + DOOR_Y_TOL:
                return FrameAction(nes_action("UP"), "east58_climb")
            self._phase = "cross"
        if self._phase == "cross":
            if x < ROOM_58_EAST_COLUMN_X - NORTH_X_TOL:
                if abs(y - ROOM_58_MID_Y) > DOOR_Y_TOL:
                    return FrameAction(
                        nes_action("UP" if y > ROOM_58_MID_Y else "DOWN"),
                        "east58_cross_y",
                    )
                return FrameAction(nes_action("RIGHT"), "east58_cross_x")
            self._phase = "drop"
        if self._phase == "drop":
            if abs(y - ROOM_58_DOOR_Y) > DOOR_Y_TOL:
                return FrameAction(
                    nes_action("UP" if y > ROOM_58_DOOR_Y else "DOWN"), "east58_drop"
                )
            self._phase = "push"
        return dungeon_align_then_push(
            snap,
            push_dir="RIGHT",
            target_y=ROOM_58_DOOR_Y,
            y_tol=DOOR_Y_TOL,
            door_plane=224,
            reason="east58",
        )


@dataclass(kw_only=True)
class Room59UpController(_WestHop):
    """0x59 (GORIYA_COMPASS) west mouth: kill-clear the goriya 0x05/0x06,
    then the perimeter waypoint micro around the central mass to the UP door
    -> live dest 0x49 (GORIYA_BUBBLE).

    The kill-clear sets ``cur_opened_doors`` bit 3 (UP) but boxes Link at
    ``(48,125)``.  Phases: rise to the ``y~100`` open west band, go west to
    ``x~44``, rise to the ``y~64`` top band, cross to ``x=120``, push UP.
    2/2 byte-identical (recordings/59_up_v2/v3.json).  Recon-wired only.
    """

    spec_id: str = "level7_room59_up"
    max_frames: int = ROOM59_UP_MAX_FRAMES
    require_level: int = LEVEL7
    done_reason: str = "left_0x59_north"
    dest: int | None = field(default_factory=north_of_room59_ram_id)
    origin: frozenset[int] = frozenset(
        {ENTRY_SCREEN, ROOM_69, ROOM_68, ROOM_58, ROOM_59}
    )
    door: str = "UP"
    saw_goriya: bool = False
    _phase: str = "clear"
    _door: LatticeDoorWalker = field(default_factory=LatticeDoorWalker, init=False, repr=False)

    def _timeout_extra(self) -> str:
        return f"phase={self._phase}_saw={int(self.saw_goriya)}"

    def scroll_action(self, snap: ZeldaSnapshot) -> FrameAction:
        return FrameAction(nes_action("UP"), "up59_scroll")

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        if snap.screen != ROOM_59:
            if snap.screen in {ROOM_58, ENTRY_SCREEN}:
                return self.mark_fail("west_backtrack")
            return self.mark_fail(f"unexpected_room_0x{snap.screen:02x}")

        live = live_goriyas(snap)
        if live:
            self.saw_goriya = True
            target = nearest_enemy(snap.link_x, snap.link_y, live)
            if target is None:
                return FrameAction(nes_idle_action(), "goriya_missing")
            return _goriya_fight(snap, target, frames=self.frames)
        if not self.saw_goriya:
            return FrameAction(nes_idle_action(), "spawn_wait")
        # ROM lattice to the north door first. The hand phases below press
        # UP from wherever the clear left Link: from (64,133) that is the
        # central block (Blue Ring power-on 4 resume, 5000f timeout).
        door = self._door.action(None, snap, "UP", "up59_lattice")
        if door is not None:
            return door

        x, y = int(snap.link_x), int(snap.link_y)
        if self._phase == "clear":
            self._phase = "rise1"
        if self._phase == "rise1":
            if y > ROOM_59_MID_BAND_Y + DOOR_Y_TOL:
                return FrameAction(nes_action("UP"), "up59_rise1")
            self._phase = "west"
        if self._phase == "west":
            if x > ROOM_59_WEST_COLUMN_X + NORTH_X_TOL:
                return FrameAction(nes_action("LEFT"), "up59_west")
            self._phase = "rise2"
        if self._phase == "rise2":
            if y > ROOM_59_TOP_BAND_Y + DOOR_Y_TOL:
                return FrameAction(nes_action("UP"), "up59_rise2")
            self._phase = "cross"
        if self._phase == "cross":
            if abs(x - ROOM_59_NORTH_X) > NORTH_X_TOL:
                btn = "LEFT" if x > ROOM_59_NORTH_X else "RIGHT"
                return FrameAction(nes_action(btn), "up59_cross")
            self._phase = "push"
        return dungeon_align_then_push(
            snap,
            push_dir="UP",
            target_x=ROOM_59_NORTH_X,
            x_tol=NORTH_X_TOL,
            reason="up59",
        )


def room_68_down_step(
    snap: ZeldaSnapshot,
    *,
    dest: int | None = None,
) -> FrameAction:
    """One frame of 0x68 (KEESE_TRAPS) → OPEN south door → live dest 0x78.

    Peel off the east trap column, drop between trap rows, align x=120,
    push DOWN.  OccupancyWalker poisons on blade-trap knockback — waypoints.
    """
    if snap.level != LEVEL7:
        return FrameAction(nes_idle_action(), "wait_level7")
    if snap.transitioning:
        return FrameAction(nes_action("DOWN"), "south68_scroll")
    if snap.mode != PLAY_MODE:
        return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
    if dest is not None and snap.screen == dest:
        return FrameAction(nes_idle_action(), "south68_arrived")
    if snap.screen != ROOM_68:
        return FrameAction(nes_idle_action(), f"unexpected_room_0x{snap.screen:02x}")

    x, y = int(snap.link_x), int(snap.link_y)
    if abs(x - ROOM_68_SOUTH_X) > NORTH_X_TOL and abs(y - ROOM_68_TRAP_ROW_Y) <= 8:
        return FrameAction(nes_action("UP"), "south68_off_trap")
    if x > ROOM_68_SAFE_X + NORTH_X_TOL:
        return FrameAction(nes_action("LEFT"), "south68_peel")
    if y < ROOM_68_MID_Y - DOOR_Y_TOL:
        return FrameAction(nes_action("DOWN"), "south68_drop")
    if abs(x - ROOM_68_SOUTH_X) > NORTH_X_TOL:
        btn = "LEFT" if x > ROOM_68_SOUTH_X else "RIGHT"
        return FrameAction(nes_action(btn), "south68_align_x")
    return FrameAction(nes_action("DOWN"), "south68_push")


@dataclass(kw_only=True)
class Room68DownController(_WestHop):
    """0x68 KEESE_TRAPS → OPEN south door to live dest 0x78 (ROPES_KEY).

    Recon-wired only.  0x78 is a dead-end (ropes 0x28 + floor key 0x19).
    """

    spec_id: str = "level7_room68_down"
    max_frames: int = ROOM68_DOWN_MAX_FRAMES
    require_level: int = LEVEL7
    done_reason: str = "left_0x68_south"
    dest: int | None = field(default_factory=south_of_room68_ram_id)
    origin: frozenset[int] = frozenset(
        {ENTRY_SCREEN, ROOM_69, ROOM_6A, ROOM_6B, ROOM_68}
    )
    door: str = "DOWN"

    def scroll_action(self, snap: ZeldaSnapshot) -> FrameAction:
        return FrameAction(nes_action("DOWN"), "south68_scroll")

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        action = room_68_down_step(snap, dest=self.dest)
        if action.reason.startswith("unexpected_room"):
            return self.mark_fail(action.reason)
        return action


def room_58_north_step(
    snap: ZeldaSnapshot,
    *,
    dest: int | None = None,
) -> FrameAction:
    """One frame of 0x58 → KEY north door → live dest 0x48 (BOMB_UPGRADE).

    East-around the central 2-block mass, then the x=120 channel.  Dodge
    0x31; do not fight.  Key spend is natural (door KEY).
    """
    if snap.level != LEVEL7:
        return FrameAction(nes_idle_action(), "wait_level7")
    if snap.transitioning:
        return FrameAction(nes_action("UP"), "north58_scroll")
    if snap.mode != PLAY_MODE:
        return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
    if dest is not None and snap.screen == dest:
        return FrameAction(nes_idle_action(), "north58_arrived")
    if snap.screen != ROOM_58:
        return FrameAction(nes_idle_action(), f"unexpected_room_0x{snap.screen:02x}")

    x, y = int(snap.link_x), int(snap.link_y)
    if y <= ROOM_58_NORTH_TOP_Y + DOOR_Y_TOL:
        if abs(x - ROOM_58_NORTH_X) > NORTH_X_TOL:
            btn = "LEFT" if x > ROOM_58_NORTH_X else "RIGHT"
            return FrameAction(nes_action(btn), "north58_align_x")
        return FrameAction(nes_action("UP"), "north58_push")
    if y > ROOM_58_NORTH_MID_Y + DOOR_Y_TOL:
        return FrameAction(nes_action("UP"), "north58_climb")
    if x < ROOM_58_NORTH_EAST_X - NORTH_X_TOL:
        return FrameAction(nes_action("RIGHT"), "north58_east")
    return FrameAction(nes_action("UP"), "north58_rise")


@dataclass(kw_only=True)
class Room58NorthController(_WestHop):
    """0x58 DODONGOS_UPGRADE → KEY north door to live dest 0x48.

    Recon-wired only.  0x48 is a dead-end old-man bomb-capacity room
    (100 rupees).  Do not write max_bombs.
    """

    spec_id: str = "level7_room58_north"
    max_frames: int = ROOM58_NORTH_MAX_FRAMES
    require_level: int = LEVEL7
    done_reason: str = "left_0x58_north"
    dest: int | None = field(default_factory=north_of_room58_ram_id)
    origin: frozenset[int] = frozenset(
        {ENTRY_SCREEN, ROOM_69, ROOM_68, ROOM_58}
    )
    door: str = "UP"

    def scroll_action(self, snap: ZeldaSnapshot) -> FrameAction:
        return FrameAction(nes_action("UP"), "north58_scroll")

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        action = room_58_north_step(snap, dest=self.dest)
        if action.reason.startswith("unexpected_room"):
            return self.mark_fail(action.reason)
        return action


def room_49_up_step(
    snap: ZeldaSnapshot,
    *,
    dest: int | None = None,
    saw_goriya: bool = False,
    frames: int = 0,
) -> FrameAction:
    """One frame of 0x49 kill-clear → north door across the water moat.

    The Stepladder lets Link walk UP at ``x=120`` through tile ``0xF4``.
    The north exit is OPEN-like (``cur_opened_doors`` stays 0); goriya
    clear is still required so they do not box the south mouth.
    """
    if snap.level != LEVEL7:
        return FrameAction(nes_idle_action(), "wait_level7")
    if snap.transitioning:
        return FrameAction(nes_action("UP"), "up49_scroll")
    if snap.mode != PLAY_MODE:
        return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
    if dest is not None and snap.screen == dest:
        return FrameAction(nes_idle_action(), "up49_arrived")
    if snap.screen != ROOM_49:
        return FrameAction(nes_idle_action(), f"unexpected_room_0x{snap.screen:02x}")

    live = live_goriyas(snap)
    if live:
        target = nearest_enemy(snap.link_x, snap.link_y, live)
        if target is None:
            return FrameAction(nes_idle_action(), "goriya_missing")
        return _goriya_fight(snap, target, frames=frames)
    if not saw_goriya:
        return FrameAction(nes_idle_action(), "spawn_wait")

    x, y = int(snap.link_x), int(snap.link_y)
    keese = live_keese(snap)
    if keese:
        target = nearest_enemy(x, y, keese)
        if target is not None:
            hint = engagement_hint(
                EnemyKind.KEESE, snap, target, projectiles=_projectiles(snap)
            )
            if should_swing_at(x, y, hint.face, (target,), hint=hint):
                if frames % _SWING_PERIOD < _SWING_HOLD:
                    return FrameAction(nes_action(hint.face, "A"), "keese_slash")
                return FrameAction(nes_action(hint.face), "keese_face")
        # Never chase keese onto the water — 49_ctl_v3 pinned at (64,117).
        on_water = ROOM_49_NORTH_BAND_Y < y <= ROOM_49_MOAT_SOUTH_Y
        if on_water:
            return FrameAction(nes_action("UP"), "keese_off_water")

    # Stepladder crosses the moat on the facing axis only — do not strafe
    # on the water. Align x on south land, hold UP across, then align north.
    if y > ROOM_49_NORTH_BAND_Y + DOOR_Y_TOL:
        if y > ROOM_49_MOAT_SOUTH_Y and abs(x - ROOM_49_NORTH_X) > NORTH_X_TOL:
            btn = "LEFT" if x > ROOM_49_NORTH_X else "RIGHT"
            return FrameAction(nes_action(btn), "up49_south_align")
        if int(snap.ladder) < 1:
            return FrameAction(nes_idle_action(), "moat_requires_ladder")
        btn = "UP"
        if keese and frames % _SWING_PERIOD < _SWING_HOLD:
            return FrameAction(nes_action(btn, "A"), "up49_cross_slash")
        return FrameAction(nes_action(btn), "up49_cross")
    if keese:
        # North land, keese still up: face them but stay off the water.
        target = nearest_enemy(x, y, keese)
        if target is not None and int(target.y) <= ROOM_49_NORTH_BAND_Y + DOOR_Y_TOL:
            hint = engagement_hint(
                EnemyKind.KEESE, snap, target, projectiles=_projectiles(snap)
            )
            return FrameAction(nes_action(hint.face), "keese_chase_land")
        if abs(x - ROOM_49_NORTH_X) > NORTH_X_TOL:
            btn = "LEFT" if x > ROOM_49_NORTH_X else "RIGHT"
            return FrameAction(nes_action(btn), "keese_wait_align")
        return FrameAction(nes_action("UP", "A"), "keese_door_slash")
    return dungeon_align_then_push(
        snap,
        push_dir="UP",
        target_x=ROOM_49_NORTH_X,
        x_tol=NORTH_X_TOL,
        reason="up49",
    )


@dataclass(kw_only=True)
class Room49UpController(_WestHop):
    """0x49 (GORIYA_BUBBLE) south mouth: kill-clear goriya 0x05, then UP
    across the water moat at x=120 (Stepladder) to live dest 0x39.

    2/2 byte-identical (recordings/49_up_v1/v2.json).  Recon-wired only.
    Requires ADDR_LADDER=1 on the recon fixture.
    """

    spec_id: str = "level7_room49_up"
    max_frames: int = ROOM49_UP_MAX_FRAMES
    require_level: int = LEVEL7
    done_reason: str = "left_0x49_north"
    dest: int | None = field(default_factory=north_of_room49_ram_id)
    origin: frozenset[int] = frozenset(
        {ENTRY_SCREEN, ROOM_69, ROOM_68, ROOM_58, ROOM_59, ROOM_49}
    )
    door: str = "UP"
    saw_goriya: bool = False

    def _timeout_extra(self) -> str:
        return f"saw={int(self.saw_goriya)}"

    def scroll_action(self, snap: ZeldaSnapshot) -> FrameAction:
        return FrameAction(nes_action("UP"), "up49_scroll")

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        if snap.screen != ROOM_49:
            if snap.screen in {ROOM_59, ENTRY_SCREEN}:
                return self.mark_fail("south_backtrack")
            return self.mark_fail(f"unexpected_room_0x{snap.screen:02x}")

        if live_goriyas(snap):
            self.saw_goriya = True
        action = room_49_up_step(
            snap, dest=self.dest, saw_goriya=self.saw_goriya, frames=self.frames
        )
        if action.reason == "moat_requires_ladder":
            return self.mark_fail("moat_requires_ladder")
        if action.reason.startswith("unexpected_room"):
            if snap.screen in {ROOM_59, ENTRY_SCREEN}:
                return self.mark_fail("south_backtrack")
            return self.mark_fail(action.reason)
        return action


ROOM_09 = 0x09
ROOM_09_SOUTH_Y = 189
ROOM_09_SOUTH_X = 120
ROOM09_DOWN_MAX_FRAMES = 8000


def south_of_room09_ram_id() -> int | None:
    """Live ``$EB`` of the room south of ``0x09`` (WEST_LOCK_SKIP)."""
    return LEVEL7_ROOM_BY_ID[WEST_LOCK_SKIP].ram_id


def room_09_down_step(
    snap: ZeldaSnapshot,
    *,
    dest: int | None = None,
    saw_goriya: bool = False,
    frames: int = 0,
) -> FrameAction:
    """One frame of 0x09 kill-clear → south shutter.

    Dead: south is OPEN on spawn. The shutter walks only after the goriya
    0x05/0x06 clear; ``cur_opened_doors`` stays LEFT. Drop to ``y=189``
    (west of the statue row), align ``x=120``, push DOWN.
    """
    if snap.level != LEVEL7:
        return FrameAction(nes_idle_action(), "wait_level7")
    if snap.transitioning:
        return FrameAction(nes_action("DOWN"), "down09_scroll")
    if snap.mode != PLAY_MODE:
        return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
    if dest is not None and snap.screen == dest:
        return FrameAction(nes_idle_action(), "down09_arrived")
    if snap.screen != ROOM_09:
        return FrameAction(nes_idle_action(), f"unexpected_room_0x{snap.screen:02x}")

    live = live_goriyas(snap)
    if live:
        target = nearest_enemy(snap.link_x, snap.link_y, live)
        if target is None:
            return FrameAction(nes_idle_action(), "goriya_missing")
        return _goriya_fight(snap, target, frames=frames)
    if not saw_goriya:
        return FrameAction(nes_idle_action(), "spawn_wait")

    # ROM lattice to the south door and through it; the straight drop below
    # hit the statue row at (152,125) for 7254 frames (R19 resume).
    step = lattice_door_step(None, snap, "DOWN")
    if step is not None:
        return FrameAction(nes_action(step), "down09_lattice")
    x, y = int(snap.link_x), int(snap.link_y)
    if y < ROOM_09_SOUTH_Y - DOOR_Y_TOL:
        return FrameAction(nes_action("DOWN"), "down09_drop")
    return dungeon_align_then_push(
        snap,
        push_dir="DOWN",
        target_x=ROOM_09_SOUTH_X,
        x_tol=NORTH_X_TOL,
        reason="down09",
    )


@dataclass(kw_only=True)
class Room09DownController(_WestHop):
    """0x09 (GORIYA_POST_RUPEE) west mouth: kill-clear, south shutter
    to live dest 0x19 (WEST_LOCK_SKIP).  2/2 (09_down_v2/v3).
    Recon-wired only.
    """

    spec_id: str = "level7_room09_down"
    max_frames: int = ROOM09_DOWN_MAX_FRAMES
    require_level: int = LEVEL7
    done_reason: str = "left_0x09_south"
    dest: int | None = field(default_factory=south_of_room09_ram_id)
    origin: frozenset[int] = frozenset(
        {ENTRY_SCREEN, ROOM_18, ROOM_08, ROOM_09}
    )
    door: str = "DOWN"
    saw_goriya: bool = False

    def _timeout_extra(self) -> str:
        return f"saw={int(self.saw_goriya)}"

    def scroll_action(self, snap: ZeldaSnapshot) -> FrameAction:
        return FrameAction(nes_action("DOWN"), "down09_scroll")

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        if live_goriyas(snap):
            self.saw_goriya = True
        action = room_09_down_step(
            snap, dest=self.dest, saw_goriya=self.saw_goriya, frames=self.frames
        )
        if action.reason.startswith("unexpected_room"):
            if snap.screen == ROOM_08:
                return self.mark_fail("west_backtrack")
            return self.mark_fail(action.reason)
        return action


__all__ = [
    "ROOM_49",
    "ROOM_58",
    "ROOM_59",
    "ROOM_68",
    "ROOM_6C",
    "ROOM_09",
    "Room6CEastController",
    "Room68NorthController",
    "Room68DownController",
    "Room49UpController",
    "Room09DownController",
    "Room58EastController",
    "Room58NorthController",
    "Room59UpController",
    "east_of_room6c_ram_id",
    "east_of_room58_ram_id",
    "north_of_room68_ram_id",
    "south_of_room68_ram_id",
    "north_of_room58_ram_id",
    "south_of_room09_ram_id",
    "north_of_room49_ram_id",
    "north_of_room59_ram_id",
    "room_49_up_step",
    "room_09_down_step",
    "room_6c_east_step",
    "room_68_north_step",
    "room_68_down_step",
    "room_58_north_step",
]
