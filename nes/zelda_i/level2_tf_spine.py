"""Level 2 Survival-spine suffix: Magical Boomerang → Dodongo → TF 0x02.

Room specs stay in ``level2_dungeon``. Bomb-wall factories stay in
``level2_bomb_path``. This module is the controller table for
``run_survival_spine --through level2`` after boom.

Inventory counts (bombs/keys) and B-slot select are applied by the spine
via ``dungeon_ops.apply_owned_inventory`` — documented Survival shortcut,
not Clean, never an undiscovered item.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum, auto
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.dungeon import (
    DungeonPhase,
    GenericDungeonRoomController,
    RewardKind,
    RewardSpec,
)
from zelda_i.level2_bomb_path import (
    Level2BombNorth1eSpineController,
    make_post_boom_bomb_north_controller,
)
from zelda_i.level2_boss_combat import (
    DODONGO_FIGHT_MAX_FRAMES,
    DODONGO_TYPE,
    goto_action,
    mouth_target,
)
from zelda_i.level2_boss_tf import (
    Level2PostBossTfController,
    TF_COLLECT_MAX_FRAMES,
    make_post_boss_tf_controller,
)
from zelda_i.level2_dungeon import (
    ROOM_1E_SPEC,
    ROOM_2E_SPEC,
    ROOM_3E_MOLDORM_SPEC,
    ROOM_3F_SPEC,
)
from zelda_i.level2_enter_1e import ENTER_1E_MAX_FRAMES, Level2Enter1eController
from zelda_i.level2_puzzles import DOOR_UP, LEVEL2_TRIFORCE_BIT, ROOM_L2_TF
from zelda_i.level2_spine import Level2RoomWalkController
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot

# Isolated complete used --poke-bombs 16. Same budget, documented.
SPINE_TF_BOMB_POKE = 16
SPINE_TF_KEY_POKE = 2
SOUTH_BAND_UP_MAX_FRAMES = 4000
SOUTH_BAND_Y = 189
DOOR_X = 120

# Isolated clear_types used min_n=1 / 4. Spec expected counts can miss a spawn.
ROOM_3F_SPINE_SPEC = replace(ROOM_3F_SPEC, expected_enemy_count=1)
ROOM_3E_SPINE_SPEC = replace(
    ROOM_3E_MOLDORM_SPEC,
    spec_id="level2_room3e_moldorm_spine",
    expected_enemy_count=1,
    reward=RewardSpec(kind=RewardKind.CLEAR_ONLY),
)
ROOM_2E_SPINE_SPEC = replace(
    ROOM_2E_SPEC,
    spec_id="level2_room2e_ropes_spine",
    expected_enemy_count=1,
    required_open_doors=DOOR_UP,
    combat=replace(
        ROOM_2E_SPEC.combat,
        engage_distance=80,
        attack_phase=0,
    ),
)
ROOM_1E_SPINE_SPEC = replace(ROOM_1E_SPEC, expected_enemy_count=1)


def level2_boom_owned(snap: ZeldaSnapshot) -> bool:
    return int(snap.magical_boomerang) != 0


def level2_through_success(snap: ZeldaSnapshot) -> bool:
    """``through=level2`` stop: Moon triforce shard, room 0x0d west of Dodongo."""
    return (int(snap.triforce) & LEVEL2_TRIFORCE_BIT) != 0


def _fight(spec) -> GenericDungeonRoomController:
    ctl = GenericDungeonRoomController(spec)
    ctl.phase = DungeonPhase.FIGHT
    return ctl


@dataclass
class Level2ClearDoorController:
    """Fight until a kill-door bit opens. Isolated 0x2e stops on UP, not 0 live."""

    inner: GenericDungeonRoomController
    door_bit: int = DOOR_UP
    room_id: int = 0x2E
    success: bool = False
    notes: list[str] = field(default_factory=list)

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        if (
            snap.mode == PLAY_MODE
            and snap.screen == self.room_id
            and snap.cur_opened_doors & self.door_bit
        ):
            self.success = True
            if "door_open" not in self.notes:
                self.notes.append("door_open")
            return FrameAction(nes_idle_action(), "done")
        action = self.inner.step(snap)
        if self.inner.success:
            self.success = True
        return action

    @property
    def phase(self):
        if self.success:
            return self.inner.phase
        return self.inner.phase

    @property
    def spec(self):
        return self.inner.spec

    def report(self) -> dict[str, Any]:
        payload = self.inner.report()
        payload["door_bit"] = self.door_bit
        payload["notes"] = list(self.notes) + list(payload.get("notes") or [])
        payload["success"] = self.success
        return payload


class SouthCenterPhase(Enum):
    WALK = auto()
    DONE = auto()
    FAILED = auto()


WEST_AISLE_X = 64
EAST_AISLE_X = 176


@dataclass
class Level2ToSouthCenterController:
    """Walk to (120, 189) via a side aisle. 0x1e NW pocket DOWN is solid."""

    room_id: int = 0x1E
    dest: tuple[int, int] = (120, 189)
    max_frames: int = SOUTH_BAND_UP_MAX_FRAMES
    phase: SouthCenterPhase = SouthCenterPhase.WALK
    frames: int = 0
    success: bool = False
    notes: list[str] = field(default_factory=list)
    _last_dir: str = "DOWN"
    _stuck: int = 0
    _last_xy: tuple[int, int] = (-1, -1)

    def _fail(self, note: str) -> FrameAction:
        self.phase = SouthCenterPhase.FAILED
        self.notes.append(note)
        return FrameAction(nes_idle_action(), note)

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        xy = (snap.link_x, snap.link_y)
        self._stuck = self._stuck + 1 if xy == self._last_xy else 0
        self._last_xy = xy
        if snap.mode == 17:
            return self._fail("link_death")
        if self.frames >= self.max_frames:
            return self._fail("timeout")
        if snap.mode == 8:
            return FrameAction(nes_idle_action(), "hurt_freeze")
        if snap.transitioning:
            return FrameAction(nes_action(self._last_dir), "room_scroll")
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.screen != self.room_id:
            return FrameAction(nes_idle_action(), f"wait_room_0x{snap.screen:02x}")
        x, y = snap.link_x, snap.link_y
        tx, ty = self.dest
        if abs(x - tx) <= 4 and abs(y - ty) <= 4:
            self.success = True
            self.phase = SouthCenterPhase.DONE
            return FrameAction(nes_idle_action(), "done")
        if self._stuck > 14:
            return FrameAction(nes_idle_action(), "south_wait")
        # 0x1e west column DOWN from the north band is solid (v6 (48,93),
        # v7 (72,93)). Isolated used the east aisle; do that from anywhere
        # north or mid-diamond.
        if y <= 117 and x < EAST_AISLE_X:
            self._last_dir = "RIGHT"
            return FrameAction(nes_action("RIGHT"), "north_to_east")
        if 72 < x < EAST_AISLE_X and 117 < y < 181:
            self._last_dir = "RIGHT"
            return FrameAction(nes_action("RIGHT"), "diamond_to_east")
        if x >= EAST_AISLE_X:
            if y < ty:
                self._last_dir = "DOWN"
                return FrameAction(nes_action("DOWN"), "east_south")
            self._last_dir = "LEFT"
            return FrameAction(nes_action("LEFT"), "south_align_x")
        if x <= 72:
            self._last_dir = "RIGHT"
            return FrameAction(nes_action("RIGHT"), "leave_west_pocket")
        if y < ty:
            self._last_dir = "DOWN"
            return FrameAction(nes_action("DOWN"), "south_band")
        self._last_dir = "RIGHT" if x < tx else "LEFT"
        return FrameAction(nes_action(self._last_dir), "south_align_x")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "phase": self.phase.name,
            "frames": self.frames,
            "room_id": self.room_id,
            "notes": list(self.notes),
        }


class SouthBandUpPhase(Enum):
    WALK = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class Level2SouthBandUpController:
    """South-band then x=120 UP. Mid-y diamond traps a naive center UP.

    Isolated ``enter_up`` (LEVEL2_ROUTE): DOWN to y≈189, align x, hold UP.
    """

    dest_room: int
    south_y: int = SOUTH_BAND_Y
    door_x: int = DOOR_X
    max_frames: int = SOUTH_BAND_UP_MAX_FRAMES
    phase: SouthBandUpPhase = SouthBandUpPhase.WALK
    frames: int = 0
    success: bool = False
    notes: list[str] = field(default_factory=list)
    _last_dir: str = "UP"
    _stuck: int = 0
    _last_xy: tuple[int, int] = (-1, -1)

    def _fail(self, note: str) -> FrameAction:
        self.phase = SouthBandUpPhase.FAILED
        self.notes.append(note)
        return FrameAction(nes_idle_action(), note)

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        xy = (snap.link_x, snap.link_y)
        self._stuck = self._stuck + 1 if xy == self._last_xy else 0
        self._last_xy = xy
        if snap.mode == 17:
            return self._fail("link_death")
        if self.frames >= self.max_frames:
            return self._fail("timeout")
        if snap.screen == self.dest_room and snap.mode == PLAY_MODE:
            self.success = True
            self.phase = SouthBandUpPhase.DONE
            return FrameAction(nes_idle_action(), "done")
        if snap.mode == 8:
            return FrameAction(nes_idle_action(), "hurt_freeze")
        if snap.transitioning:
            return FrameAction(nes_action(self._last_dir), "room_scroll")
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        x, y = snap.link_x, snap.link_y
        if self._stuck > 14:
            # v5 LEFT solid. v6 DOWN solid. Same leftover (96,141) gutter.
            # North band y<=117 is the documented free strip.
            self._last_dir = "UP"
            return FrameAction(nes_action("UP"), "diamond_unstick_north")
        # Diamond rooms (0x3e / 0x2e): v1 (120,185); v2 (154,141); v3 (175,109)
        # was still inside the old free box and held RIGHT. Side aisle north,
        # then door-column UP. North band y<=117 is not "diamond".
        if 72 < x < 168 and 117 < y < 181:
            self._last_dir = "LEFT" if x <= self.door_x else "RIGHT"
            return FrameAction(nes_action(self._last_dir), "diamond_free")
        if x <= 72 or x >= 168:
            if y > 117:
                self._last_dir = "UP"
                return FrameAction(nes_action("UP"), "side_north")
            self._last_dir = "RIGHT" if x < self.door_x else "LEFT"
            return FrameAction(nes_action(self._last_dir), "north_align_x")
        if y <= 117:
            if abs(x - self.door_x) > 2:
                self._last_dir = "RIGHT" if x < self.door_x else "LEFT"
                return FrameAction(nes_action(self._last_dir), "north_align_x")
            self._last_dir = "UP"
            return FrameAction(nes_action("UP"), "push_up")
        if y < self.south_y:
            self._last_dir = "DOWN"
            return FrameAction(nes_action("DOWN"), "south_band")
        if abs(x - self.door_x) > 2:
            self._last_dir = "RIGHT" if x < self.door_x else "LEFT"
            return FrameAction(nes_action(self._last_dir), "south_align_x")
        self._last_dir = "UP"
        return FrameAction(nes_action("UP"), "push_up")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "phase": self.phase.name,
            "frames": self.frames,
            "dest_room": self.dest_room,
            "notes": list(self.notes),
        }


class DodongoPhase(Enum):
    SETTLE = auto()
    FIGHT = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class Level2DodongoController:
    """Bomb-in-mouth Dodongo. No inventory poke — spine tops up first."""

    max_frames: int = DODONGO_FIGHT_MAX_FRAMES
    settle_frames: int = 90
    dodongo_type: int = DODONGO_TYPE
    clamp_x: tuple[int, int] = (48, 192)
    clamp_y: tuple[int, int] = (105, 185)
    mouth_tol: int = 12
    mouth_offset: int = 12
    phase: DodongoPhase = DodongoPhase.SETTLE
    frames: int = 0
    success: bool = False
    notes: list[str] = field(default_factory=list)
    bombs_used: int = 0
    place_cd: int = 0
    place_face: str = "UP"
    last_hp: int | None = None
    last_slot: int | None = None
    hits_est: int = 0

    def _fail(self, note: str) -> FrameAction:
        self.phase = DodongoPhase.FAILED
        self.notes.append(note)
        return FrameAction(nes_idle_action(), note)

    def _living(self, snap: ZeldaSnapshot) -> list[Any]:
        return [
            o
            for o in snap.objects
            if o.type_id == self.dodongo_type and 1 <= o.slot <= 10 and o.hp > 0
        ]

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        if snap.mode == 17:
            return self._fail("link_death")
        if self.frames >= self.max_frames:
            return self._fail("timeout")
        if (int(snap.triforce) & LEVEL2_TRIFORCE_BIT) != 0:
            self.success = True
            self.phase = DodongoPhase.DONE
            return FrameAction(nes_idle_action(), "done")
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if self.phase is DodongoPhase.SETTLE:
            if self.frames < self.settle_frames:
                return FrameAction(nes_idle_action(), "settle_0e")
            self.phase = DodongoPhase.FIGHT

        living = self._living(snap)
        dodos = [
            o
            for o in snap.objects
            if o.type_id == self.dodongo_type and 1 <= o.slot <= 10
        ]
        if not living and snap.room_all_dead >= 20:
            self.success = True
            self.phase = DodongoPhase.DONE
            self.notes.append("dodongo_dead")
            return FrameAction(nes_idle_action(), "done")
        if not living:
            if self.frames > 200 and snap.room_all_dead >= 20 and not dodos:
                self.success = True
                self.phase = DodongoPhase.DONE
                self.notes.append("dodongo_dead_settle")
                return FrameAction(nes_idle_action(), "done")
            wander = ("UP", "RIGHT", "DOWN", "LEFT")[self.frames // 20 % 4]
            return FrameAction(nes_action(wander, "A"), "dodo_search")

        if self.place_cd > 0:
            self.place_cd -= 1
            if self.place_cd > 50:
                retreat = {
                    "UP": "DOWN",
                    "DOWN": "UP",
                    "LEFT": "RIGHT",
                    "RIGHT": "LEFT",
                }.get(self.place_face, "DOWN")
                return FrameAction(nes_action(retreat), "dodo_retreat")
            if self.place_cd > 20:
                return FrameAction(nes_action(self.place_face, "A"), "dodo_cover")
            return FrameAction(nes_idle_action(), "dodo_wait_blast")

        d = min(living, key=lambda o: abs(o.x - snap.link_x) + abs(o.y - snap.link_y))
        if self.last_slot != d.slot:
            self.last_hp = None
            self.last_slot = d.slot
        if self.last_hp is not None and d.hp < self.last_hp:
            self.hits_est += 1
        self.last_hp = d.hp

        tx, ty, face = mouth_target(d, self.mouth_offset)
        if face in ("LEFT", "RIGHT"):
            ty = d.y
            tx = max(self.clamp_x[0], min(self.clamp_x[1], tx))
        else:
            tx = d.x
            ty = max(self.clamp_y[0], min(self.clamp_y[1], ty))
        dist = abs(snap.link_x - d.x) + abs(snap.link_y - d.y)
        at_mouth = (
            abs(snap.link_x - tx) <= self.mouth_tol
            and abs(snap.link_y - ty) <= self.mouth_tol
        )
        if snap.bombs <= 0:
            return self._fail("out_of_bombs")
        if at_mouth or dist <= 24:
            if dist > 14:
                act, _ = goto_action(snap, d.x, d.y, tol=8)
                return FrameAction(act, "dodo_close")
            self.place_face = face
            self.place_cd = 95
            self.bombs_used += 1
            return FrameAction(nes_action(face, "B"), "dodo_place")
        act, _ = goto_action(snap, tx, ty, tol=6)
        return FrameAction(act, "dodo_approach")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "phase": self.phase.name,
            "frames": self.frames,
            "bombs_used_est": self.bombs_used,
            "hits_est": self.hits_est,
            "poke": False,
            "notes": list(self.notes),
        }


@dataclass
class Level2TfCollectController:
    """HC → LEFT 0x0d → south-band waypoints. Success on ``tf & 0x02``."""

    inner: Level2PostBossTfController = field(
        default_factory=make_post_boss_tf_controller
    )

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        return self.inner.step(snap, tf_value=int(snap.triforce))

    @property
    def success(self) -> bool:
        return self.inner.success

    @property
    def phase(self):
        return self.inner.phase

    @property
    def max_frames(self) -> int:
        return self.inner.max_frames

    def report(self) -> dict[str, Any]:
        return self.inner.report()


def level2_tf_stages():
    """Controller table: boom room 0x4f through TF bit 0x02 in 0x0d.

    Path: 0x4f bomb-N → 0x3f → LEFT 0x3e → UP 0x2e → UP 0x1e → bomb-N
    0x0e Dodongo → LEFT 0x0d. TF is WEST of the boss, not east.
    """
    bomb_4f = make_post_boom_bomb_north_controller()
    # v8 leftover (120, 117): waypoint (120, 93) is the closed bomb wall.
    # v9 west peel reached (96, 101); cardinal RIGHT solid. Spine wrapper
    # peels west then RIGHT+UP clips to stand. Isolated default unchanged.
    bomb_1e = Level2BombNorth1eSpineController()
    return (
        ("bomb_north_4f", bomb_4f, bomb_4f.max_frames),
        ("clear3f", _fight(ROOM_3F_SPINE_SPEC), ROOM_3F_SPINE_SPEC.max_frames),
        (
            "enter_3e",
            Level2RoomWalkController(
                dest_room=0x3E,
                hops=((0x3F, "LEFT", 0x3E),),
                max_frames=SOUTH_BAND_UP_MAX_FRAMES,
            ),
            SOUTH_BAND_UP_MAX_FRAMES,
        ),
        ("clear3e", _fight(ROOM_3E_SPINE_SPEC), ROOM_3E_SPINE_SPEC.max_frames),
        (
            "enter_2e",
            Level2SouthBandUpController(dest_room=0x2E),
            SOUTH_BAND_UP_MAX_FRAMES,
        ),
        (
            "clear2e",
            Level2ClearDoorController(
                inner=_fight(ROOM_2E_SPINE_SPEC),
                door_bit=DOOR_UP,
                room_id=0x2E,
            ),
            ROOM_2E_SPINE_SPEC.max_frames,
        ),
        (
            "enter_1e",
            Level2Enter1eController(),
            ENTER_1E_MAX_FRAMES,
        ),
        ("clear1e", _fight(ROOM_1E_SPINE_SPEC), ROOM_1E_SPINE_SPEC.max_frames),
        ("bomb_north_1e", bomb_1e, bomb_1e.max_frames),
        ("fight_dodongo", Level2DodongoController(), DODONGO_FIGHT_MAX_FRAMES),
        ("collect_tf", Level2TfCollectController(), TF_COLLECT_MAX_FRAMES),
    )


__all__ = [
    "DOOR_X",
    "DodongoPhase",
    "EAST_AISLE_X",
    "Level2ClearDoorController",
    "Level2DodongoController",
    "Level2Enter1eController",
    "Level2SouthBandUpController",
    "Level2ToSouthCenterController",
    "Level2TfCollectController",
    "WEST_AISLE_X",
    "ROOM_1E_SPINE_SPEC",
    "ROOM_2E_SPINE_SPEC",
    "ROOM_3E_SPINE_SPEC",
    "ROOM_3F_SPINE_SPEC",
    "SOUTH_BAND_UP_MAX_FRAMES",
    "SOUTH_BAND_Y",
    "SPINE_TF_BOMB_POKE",
    "SPINE_TF_KEY_POKE",
    "SouthBandUpPhase",
    "level2_boom_owned",
    "level2_tf_stages",
    "level2_through_success",
]
