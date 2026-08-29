"""Level 2 Survival-spine path: live 0x7d through Magical Boomerang 0x4f.

Room specs and stop predicates stay in ``level2_dungeon``. This module owns
the continuous-spine orchestration and the named nav controllers that
``GenericDungeonRoomController`` + spec waypoints cannot do (backtrack,
west-entry 0x6e, diamond-east key door).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum, auto
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.dungeon.engine import (
    DungeonPhase,
    GenericDungeonRoomController,
    RewardKind,
    RewardSpec,
)
from zelda_i.level2.bomb_path import (
    make_bomb_north_controller,
    make_boom_bomb_north_controller,
)
from zelda_i.level2.dungeon import (
    ROOM_4F_SPEC,
    ROOM_6C_SPEC,
    ROOM_6D_SPEC,
    ROOM_6E_SPEC,
    ROOM_6F_SPEC,
    ROOM_7E_SPEC,
    ROOM_L2_COMPASS,
    ROOM_L2_EAST_KEY,
    ROOM_L2_EAST_OF_ROPES,
    ROOM_L2_ENTRY,
    ROOM_L2_ROPES,
    ROOM_L2_WEST_KEY,
)
from zelda_i.dungeon.hop_controller import DEATH_MODE, HopController, dungeon_align_then_push
from zelda_i.overworld.common import (
    DIAMOND_BAND_6E,
    DIAMOND_BAND_7D,
    DIAMOND_WALL_X,
    DOOR_Y_DEFAULT,
    diamond_east_phase,
    track_stuck,
    unstick_wiggle,
)
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot

DOOR_X = 120
DOOR_Y = 141
# Live 2026-08-15: ALIGN_TOL=6 pushed RIGHT at y=135 into the 0x6c wall and
# timed out at (128, 133). East/west mouths need |y-141|≤2 (L3 uses ≤4).
ALIGN_TOL = 2
STUCK_THRESHOLD = 24
BACKTRACK_7D_MAX_FRAMES = 3000
ENTER_6E_WEST_MAX_FRAMES = 4000
ENTER_6F_KEY_MAX_FRAMES = 4000
# Mid-room box used to leave the 0x7d east alcove after 0x7e LEFT.
_MID_X = (70, 180)
_MID_Y = (110, 175)

Hop = tuple[int, str, int]

# Isolated 7e walks onto the key mid-fight from keys=0. The spine already
# holds the west key, so Generic idles at |delta|≤5 on target (136,141) and
# misses a 5px-off floor key (live timeout at (141,141), keys still 1).
# Isolated 6e fight with engage=64 chases a rope into the north trench
# (live (80, 93)). Patrol-only mid box keeps the key-door start in band.
# Bow-splice l6_gohma_bow_v1 leftover 0x6c (96,141) last_live=2 8000f.
# Isolated patrol does not chase corner Ropes; occupancy chase does.
# v2 cleared in 480f then idled (138,141) 2px off key target (136,141).
ROOM_6C_SPINE_SPEC = replace(
    ROOM_6C_SPEC,
    spec_id="level2_room6c_west_key_spine",
    combat=replace(
        ROOM_6C_SPEC.combat,
        occupancy_patrol=True,
    ),
    reward=RewardSpec(
        kind=RewardKind.FIXED_INVENTORY,
        inventory_field="keys",
        target=(136, 141),
        waypoints=(
            (136, 141),
            (120, 141),
            (136, 125),
            (152, 141),
            (136, 157),
            (104, 141),
            (168, 141),
        ),
        reward_while_live=True,
    ),
)
ROOM_6E_SPINE_SPEC = replace(
    ROOM_6E_SPEC,
    spec_id="level2_room6e_ropes_spine",
    combat=replace(
        ROOM_6E_SPEC.combat,
        engage_distance=28,
    ),
)
# Bow-splice l6_gohma_bow_v3 leftover 0x6f (96,117) north of diamonds.
# v4 peeled to (96,141) then RIGHT into the west diamond face (skip_1).
# West aisle x=96 is open N-S; south-around y=173 then east.
ROOM_6F_SPINE_SPEC = replace(
    ROOM_6F_SPEC,
    spec_id="level2_room6f_compass_spine",
    reward=RewardSpec(
        kind=RewardKind.FIXED_INVENTORY,
        inventory_field="compass",
        target=(208, 101),
        waypoints=(
            (96, 173),
            (120, 173),
            (160, 173),
            (192, 173),
            (192, 141),
            (192, 101),
            (208, 101),
            (200, 109),
        ),
    ),
)
ROOM_7E_SPINE_SPEC = replace(
    ROOM_7E_SPEC,
    spec_id="level2_room7e_east_key_spine",
    reward=RewardSpec(
        kind=RewardKind.FIXED_INVENTORY,
        inventory_field="keys",
        target=(136, 141),
        waypoints=(
            (136, 141),
            (120, 141),
            (136, 125),
            (152, 141),
            (136, 157),
            (104, 141),
            (168, 141),
        ),
        reward_while_live=True,
    ),
)


class Level2NavPhase(Enum):
    WALK = auto()
    DONE = auto()
    FAILED = auto()


def level2_boom_success(snap: ZeldaSnapshot) -> bool:
    """Magical Boomerang owned (mid-dungeon gate before the TF suffix)."""
    return int(snap.magical_boomerang) != 0


def level2_through_success(snap: ZeldaSnapshot) -> bool:
    """``through=level2`` stop: Moon triforce shard 0x02, not merely boom."""
    return (int(snap.triforce) & 0x02) != 0


def _in_mid_room(snap: ZeldaSnapshot) -> bool:
    return _MID_X[0] <= snap.link_x <= _MID_X[1] and _MID_Y[0] <= snap.link_y <= _MID_Y[1]


def _door_push(snap: ZeldaSnapshot, direction: str) -> FrameAction:
    if direction in ("LEFT", "RIGHT"):
        return dungeon_align_then_push(
            snap, push_dir=direction, target_y=DOOR_Y, y_tol=ALIGN_TOL, reason="door"
        )
    return dungeon_align_then_push(
        snap, push_dir=direction, target_x=DOOR_X, x_tol=ALIGN_TOL, reason="door"
    )


@dataclass
class L2NavBase(HopController):
    """Death / timeout / dest / hurt-freeze / last-dir scroll."""

    dest_room: int = 0
    phase: Level2NavPhase = Level2NavPhase.WALK
    wait_modes: tuple[int, ...] = ()
    _last_dir: str = "RIGHT"

    def timeout_note(self, snap: ZeldaSnapshot) -> str:
        del snap
        return "timeout"

    def scroll_action(self, snap: ZeldaSnapshot) -> FrameAction:
        del snap
        return FrameAction(nes_action(self._last_dir), "room_scroll")

    def _set_phase(self, phase: Level2NavPhase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            if note:
                self.notes.append(note)

    def mark_fail(self, note: str, reason: str | None = None) -> FrameAction:
        self._set_phase(Level2NavPhase.FAILED, note)
        return super().mark_fail(note, reason)

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        return snap.screen == self.dest_room and snap.mode == PLAY_MODE

    def on_arrive(self, snap: ZeldaSnapshot) -> str:
        del snap
        return "arrived"

    def mark_done(self, snap: ZeldaSnapshot, note: str | None = None) -> FrameAction:
        self._set_phase(Level2NavPhase.DONE, note or self.on_arrive(snap))
        return super().mark_done(snap, note)

    def guard(self, snap: ZeldaSnapshot) -> FrameAction | None:
        if self.success:
            return FrameAction(nes_idle_action(), "done")
        if self.failed or self.frames >= self.max_frames:
            return self.mark_fail(self.timeout_note(snap))
        if snap.mode == DEATH_MODE:
            return self.mark_fail("link_death")
        if snap.mode == 8:
            return FrameAction(nes_idle_action(), "hurt_freeze")
        if snap.transitioning:
            return self.scroll_action(snap)
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        return None

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "phase": self.phase.name,
            "frames": self.frames,
            "dest_room": self.dest_room,
            "notes": list(self.notes),
        }


@dataclass
class Level2RoomWalkController(L2NavBase):
    """Ordered dungeon door hops. Optional diamond-free on one source room."""

    dest_room: int = 0
    hops: tuple[Hop, ...] = ()
    max_frames: int = BACKTRACK_7D_MAX_FRAMES
    diamond_free_room: int | None = None
    _stuck: int = 0
    _last_x: int = -1
    _last_y: int = -1
    _last_screen: int = -1

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        if (
            self.diamond_free_room is not None
            and snap.screen == self.diamond_free_room
            and not _in_mid_room(snap)
        ):
            # East alcove of 0x7d: LEFT on y=141 hits the diamond (live timeout
            # at (208,141)). Drop to band y=157 first, then LEFT into mid.
            if snap.screen == ROOM_L2_ENTRY and snap.link_x >= 176:
                if abs(snap.link_y - DIAMOND_BAND_7D) > 4:
                    self._last_dir = "DOWN" if snap.link_y < DIAMOND_BAND_7D else "UP"
                    return FrameAction(nes_action(self._last_dir), "alcove_band")
                self._last_dir = "LEFT"
                return FrameAction(nes_action("LEFT"), "alcove_left")
            action, _ = diamond_east_phase(snap, phase="free", band_y=157)
            self._last_dir = "LEFT"
            return action

        if self._stuck > STUCK_THRESHOLD:
            action, self._stuck = unstick_wiggle(self._stuck, reason="door_unstick")
            return action

        for from_room, direction, _to_room in self.hops:
            if snap.screen == from_room:
                self._last_dir = direction
                return _door_push(snap, direction)
        return FrameAction(nes_idle_action(), f"wait_room_0x{snap.screen:02x}")

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self._stuck, self._last_x, self._last_y, self._last_screen = track_stuck(
            snap,
            last_x=self._last_x,
            last_y=self._last_y,
            last_screen=self._last_screen,
            stuck=self._stuck,
        )
        return super().step(snap)


@dataclass
class Level2BacktrackTo7dController(Level2RoomWalkController):
    """0x6c RIGHT → 0x6d DOWN → entry 0x7d (after west key)."""

    dest_room: int = ROOM_L2_ENTRY
    hops: tuple[Hop, ...] = (
        (ROOM_L2_WEST_KEY, "RIGHT", ROOM_L2_ROPES),
        (ROOM_L2_ROPES, "DOWN", ROOM_L2_ENTRY),
    )
    max_frames: int = BACKTRACK_7D_MAX_FRAMES


# Reverse 0x7d east alcove (isolated 2/2): LEFT on y=141 from the door mouth
# (x≈224) is solid. LEFT×6, UP×12, LEFT×20 until x≤150, then center + UP.
_REVERSE_ALCOVE = ("LEFT",) * 6 + ("UP",) * 12 + ("LEFT",) * 20
_REVERSE_ALCOVE_X = 150


@dataclass
class Level2WestEnter6eController(L2NavBase):
    """0x7e LEFT → 0x7d reverse-diamond → 0x6d RIGHT → 0x6e west mouth.

    Isolated 2/2 (``run_level2_clear6f._nav_east_key_to_6e``). 0x7e UP lands
    south y≈181; geom ``south_band_y181_then_y141`` sticks at (200, 157).
    Door-mouth DOWN to band 157 also sticks (live (224, 141)).
    """

    dest_room: int = ROOM_L2_EAST_OF_ROPES
    max_frames: int = ENTER_6E_WEST_MAX_FRAMES
    alcove_cycle: int = 0
    inland_frames: int = 0
    _last_dir: str = "LEFT"

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        if snap.screen != self.dest_room or snap.mode != PLAY_MODE:
            return False
        # Isolated nav holds RIGHT through the west door. Stopping at
        # (16, 141) leaves Link in the mouth; keep-mid then locks.
        return not (snap.link_x < 80 and self.inland_frames < 90)

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        if snap.screen == self.dest_room and snap.mode == PLAY_MODE:
            self.inland_frames += 1
            self._last_dir = "RIGHT"
            return FrameAction(nes_action("RIGHT"), "west_inland_x")
        if snap.screen == ROOM_L2_EAST_KEY:
            self._last_dir = "LEFT"
            return _door_push(snap, "LEFT")
        if snap.screen == ROOM_L2_ENTRY:
            if snap.link_x > _REVERSE_ALCOVE_X:
                direction = _REVERSE_ALCOVE[self.alcove_cycle % len(_REVERSE_ALCOVE)]
                self.alcove_cycle += 1
                self._last_dir = direction
                return FrameAction(nes_action(direction), "alcove_cycle")
            self._last_dir = "UP"
            return _door_push(snap, "UP")
        if snap.screen == ROOM_L2_ROPES:
            self._last_dir = "RIGHT"
            return _door_push(snap, "RIGHT")
        return FrameAction(nes_idle_action(), f"wait_room_0x{snap.screen:02x}")

    def report(self) -> dict[str, Any]:
        out = super().report()
        out["alcove_cycle"] = self.alcove_cycle
        return out


@dataclass
class Level2Clear6eController:
    """Clear 3 ropes. Isolated ``_clear_6e_keep_mid``: idle, then FIGHT.

    Keep-mid matches the 2/2 script (x<56 / y<105 / y>185) only after the
    120f settle so west-door knockback does not lock DOWN at (88, 93).
    """

    inner: GenericDungeonRoomController = field(
        default_factory=lambda: GenericDungeonRoomController(ROOM_6E_SPINE_SPEC)
    )
    settle_frames: int = 0
    settle_max: int = 120

    def __post_init__(self) -> None:
        self.inner.phase = DungeonPhase.FIGHT

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        if snap.mode == PLAY_MODE and snap.screen == ROOM_L2_EAST_OF_ROPES:
            if self.settle_frames < self.settle_max:
                self.settle_frames += 1
                return FrameAction(nes_idle_action(), "settle_6e")
            if snap.link_x < 40:
                return FrameAction(nes_action("RIGHT"), "keep_mid_x")
            if snap.link_y > 195:
                return FrameAction(nes_action("UP"), "keep_mid_s")
        return self.inner.step(snap)

    @property
    def success(self) -> bool:
        return self.inner.success

    @property
    def phase(self):
        return self.inner.phase

    @property
    def spec(self):
        return self.inner.spec

    def report(self) -> dict[str, Any]:
        return self.inner.report()


@dataclass
class Level2Enter6fKeyController(L2NavBase):
    """0x6e key-RIGHT: mid-band y≈113 → wall x≥200 → vertical y=141 → RIGHT.

    Isolated 2/2 (``run_level2_clear6f._enter_6f_key_door``). Do not LEFT at
    the wall (re-enters the diamond) and do not climb the east wall from
    y≈181 (stuck at (200, 157)). Fails honestly when keys==0.
    """

    dest_room: int = ROOM_L2_COMPASS
    from_room: int = ROOM_L2_EAST_OF_ROPES
    band_y: int = DIAMOND_BAND_6E
    door_y: int = DOOR_Y_DEFAULT
    wall_x: int = DIAMOND_WALL_X
    require_keys: int = 1
    max_frames: int = ENTER_6F_KEY_MAX_FRAMES
    door_phase: str = "band"
    _last_dir: str = "RIGHT"

    def on_arrive(self, snap: ZeldaSnapshot) -> str:
        del snap
        return "key_door_entered"

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        if snap.screen != self.from_room:
            return FrameAction(nes_idle_action(), f"wait_room_0x{snap.screen:02x}")
        if snap.keys < self.require_keys and self.door_phase != "push":
            return self.mark_fail("no_keys")

        x, y = snap.link_x, snap.link_y
        # Live post-clear leftover (64, 93): north corridor RIGHT to
        # x≥208, DOWN to door y, RIGHT through the key door (1/1).
        # South pocket (y>165) must not use this: x≥200 + UP is the
        # east-wall climb that sticks at (200, 157).
        if self.door_phase == "band" and (y < 110 or (x >= 200 and y <= 165)):
            if x < 208:
                return FrameAction(nes_action("RIGHT"), "north_east")
            if y < 137:
                return FrameAction(nes_action("DOWN"), "north_door_y")
            if y > 160:
                return FrameAction(nes_action("UP"), "north_door_y")
            self.door_phase = "push"
        action, next_phase = diamond_east_phase(
            snap,
            phase=self.door_phase,
            band_y=self.band_y,
            door_y=self.door_y,
            wall_x=self.wall_x,
            cycle=self.frames,
        )
        self.door_phase = next_phase
        return action

    def report(self) -> dict[str, Any]:
        out = super().report()
        out["door_phase"] = self.door_phase
        return out


def level2_to_boom_stages():
    """Controller table: live 0x7d through Magical Boomerang 0x4f.

    Bomb stages fail with ``no_bombs`` when inventory is 0. The Survival
    spine tops up owned bomb/key counts before these stages (ASSIST_CONTRACT
    shortcut until a farm pass). This table itself does not poke.
    """
    bomb_6f = make_bomb_north_controller()
    # Isolated Level2_5F is (120, 189). Gel-clear patrol parks mid-diamond
    # (v12 stand_timeout (106, 117)). Skip clear; walk south hole UP to stand.
    bomb_5f = make_boom_bomb_north_controller(clear_gels=False)
    return (
        ("clear6d", GenericDungeonRoomController(ROOM_6D_SPEC), ROOM_6D_SPEC.max_frames),
        (
            "clear6c_key",
            GenericDungeonRoomController(ROOM_6C_SPINE_SPEC),
            ROOM_6C_SPINE_SPEC.max_frames,
        ),
        (
            "backtrack_7d",
            Level2BacktrackTo7dController(),
            BACKTRACK_7D_MAX_FRAMES,
        ),
        (
            "clear7e_key",
            GenericDungeonRoomController(ROOM_7E_SPINE_SPEC),
            ROOM_7E_SPINE_SPEC.max_frames,
        ),
        (
            "enter_6e_west",
            Level2WestEnter6eController(),
            ENTER_6E_WEST_MAX_FRAMES,
        ),
        ("clear6e", Level2Clear6eController(), ROOM_6E_SPINE_SPEC.max_frames),
        (
            "enter_6f_key",
            Level2Enter6fKeyController(),
            ENTER_6F_KEY_MAX_FRAMES,
        ),
        (
            "clear6f_compass",
            GenericDungeonRoomController(ROOM_6F_SPINE_SPEC),
            ROOM_6F_SPINE_SPEC.max_frames,
        ),
        ("bomb_north_6f", bomb_6f, bomb_6f.max_frames),
        ("bomb_north_5f", bomb_5f, bomb_5f.max_frames),
        (
            "clear4f_boom",
            GenericDungeonRoomController(ROOM_4F_SPEC),
            ROOM_4F_SPEC.max_frames,
        ),
    )


__all__ = [
    "BACKTRACK_7D_MAX_FRAMES",
    "ENTER_6E_WEST_MAX_FRAMES",
    "ENTER_6F_KEY_MAX_FRAMES",
    "L2NavBase",
    "Level2BacktrackTo7dController",
    "Level2Clear6eController",
    "Level2Enter6fKeyController",
    "Level2NavPhase",
    "Level2RoomWalkController",
    "Level2WestEnter6eController",
    "ROOM_6C_SPINE_SPEC",
    "ROOM_6E_SPINE_SPEC",
    "ROOM_6F_SPINE_SPEC",
    "ROOM_7E_SPINE_SPEC",
    "level2_boom_success",
    "level2_through_success",
    "level2_to_boom_stages",
]
