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
from zelda_i.combat import direction_to_facing
from zelda_i.dungeon.behaviors import face_toward
from zelda_i.dungeon.hop_controller import room_step
from zelda_i.overworld.hunt import sword_stand
from zelda_i.dungeon.engine import (
    DungeonPhase,
    GenericDungeonRoomController,
    RewardKind,
    RewardSpec,
)
from zelda_i.level2.bomb_path import (
    Level2BombNorth1eSpineController,
    make_post_boom_bomb_north_controller,
)
from zelda_i.level2.boss_combat import (
    DODONGO_FIGHT_MAX_FRAMES,
    DODONGO_TYPE,
    FACE_E,
    FACE_N,
    FACE_S,
    FACE_W,
    goto_action,
    in_front_of_mouth,
    mouth_path_clear,
    mouth_target,
)
from zelda_i.level2.boss_tf import (
    Level2PostBossTfController,
    TF_COLLECT_MAX_FRAMES,
    make_post_boss_tf_controller,
)
from zelda_i.level2.dungeon import (
    ROOM_1E_SPEC,
    ROOM_2E_SPEC,
    ROOM_3E_MOLDORM_SPEC,
    ROOM_3F_SPEC,
)
from zelda_i.level2.enter_1e import ENTER_1E_MAX_FRAMES, Level2Enter1eController
from zelda_i.level2.puzzles import DOOR_UP, LEVEL2_TRIFORCE_BIT
from zelda_i.dungeon.pause_select import B_SLOT_BOMBS, PauseSelectController
from zelda_i.level2.spine import Level2RoomWalkController
from zelda_i.ram import ADDR_OBJ_STATE, ADDR_SELECTED_ITEM, PLAY_MODE, ZeldaSnapshot, read_u8

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


# ObjState 1: swallowed a bomb. It stands still ~97 frames; a second
# swallow kills it (LH40/LH43/BlueRingFull14 pins).
DODONGO_BLOATED = 1
# The Dodongo ObjState the sword can cut: 2, stunned by a blast beside the
# head. Measured on three pins: in 1 (swallowed a bomb) no side takes a cut,
# and in 0 (walking) none does either.
DODONGO_STUNNED = frozenset({2})
# Link's bomb slots and the ObjStates of a bomb on the floor (fuse, flash,
# blast). A swallowed bomb leaves none, so the strike need not wait out the
# post-placement retreat: that retreat spent 45 of a 97-frame bloat
# (natural_credits_poweron48).
BOMB_SLOTS = (0x10, 0x11)
# How far Link may drift off a reached mouth stand before walking it again.
DODONGO_STAND_SLACK = 8
BOMB_ARMED_STATES = range(18, 21)
# Floor a stun stand is clamped into (x0, x1, y0, y1).
DODONGO_STAND_BOX = (32, 208, 93, 189)


def mouth_stand(dodo: Any) -> tuple[tuple[int, int], str] | None:
    """Where Link stands to drop a bomb against the mouth, and his facing.

    The body is 16x16 facing N/S and 32x16 facing E/W (it turns at x=192
    against the x=224 wall); a bomb lands 16 px ahead of Link. None when
    that stand is off the floor (the mouth is at a wall).
    """
    x, y, f = int(dodo.x), int(dodo.y), int(dodo.facing)
    if f & FACE_N:
        spot = ((x, y - 32), "DOWN")
    elif f & FACE_S:
        spot = ((x, y + 30), "UP")
    elif f & FACE_W:
        spot = ((x - 32, y), "RIGHT")
    elif f & FACE_E:
        spot = ((x + 48, y), "LEFT")
    else:
        return None
    (sx, sy), _ = spot
    x0, x1, y0, y1 = DODONGO_STAND_BOX
    return spot if x0 <= sx <= x1 and y0 <= sy <= y1 else None


@dataclass
class Level2DodongoController:
    """Bomb-in-mouth Dodongo. Do not occupancy-grade the moving boss."""

    max_frames: int = DODONGO_FIGHT_MAX_FRAMES
    settle_frames: int = 90
    dodongo_type: int = DODONGO_TYPE
    clamp_x: tuple[int, int] = (48, 192)
    clamp_y: tuple[int, int] = (105, 185)
    mouth_tol: int = 12
    mouth_offset: int = 16
    contact: int = 14
    stable_face_frames: int = 8
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
    face_hist: dict[int, int] = field(default_factory=dict)
    stable_n: dict[int, int] = field(default_factory=dict)
    select_item: int | None = B_SLOT_BOMBS
    _env: Any = field(default=None, init=False, repr=False)
    _select: PauseSelectController | None = field(default=None, init=False, repr=False)
    _at_mouth: bool = field(default=False, init=False, repr=False)
    _full_hp: int = field(default=0, init=False, repr=False)

    def bind_env(self, env: Any) -> None:
        self._env = env
        if self._select is not None:
            self._select.bind_env(env)

    def _fail(self, note: str) -> FrameAction:
        self.phase = DodongoPhase.FAILED
        self.notes.append(note)
        return FrameAction(nes_idle_action(), note)

    def _stand(self, reason: str) -> FrameAction:
        return FrameAction(nes_idle_action(), reason)

    def _bomb_armed(self) -> bool:
        if self._env is None:
            return False
        ram = self._env.get_ram()
        return any(int(ram[ADDR_OBJ_STATE + slot]) in BOMB_ARMED_STATES for slot in BOMB_SLOTS)

    def _living(self, snap: ZeldaSnapshot) -> list[Any]:
        return [
            o
            for o in snap.objects
            if o.type_id == self.dodongo_type and 1 <= o.slot <= 10 and o.hp > 0
        ]

    def _track_face(self, living: list[Any]) -> None:
        for o in living:
            if self.face_hist.get(o.slot) == o.facing:
                self.stable_n[o.slot] = self.stable_n.get(o.slot, 0) + 1
            else:
                self.stable_n[o.slot] = 0
            self.face_hist[o.slot] = o.facing

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        if snap.mode == 17:
            return self._fail("link_death")
        if self.frames >= self.max_frames:
            return self._fail("timeout")
        if (int(snap.triforce) & LEVEL2_TRIFORCE_BIT) != 0:
            self.success = True
            self.phase = DodongoPhase.DONE
            return self._stand("done")
        if snap.mode != PLAY_MODE:
            return self._stand(f"wait_mode_{snap.mode}")
        if self.phase is DodongoPhase.SETTLE:
            if self.frames < self.settle_frames:
                return self._stand("settle_0e")
            self.phase = DodongoPhase.FIGHT

        # Before the B-item select: with the bag empty the ROM moves B off
        # bombs and the select answered "done" for 12300 frames.
        live = self._living(snap)
        self._full_hp = max([self._full_hp] + [int(o.hp) for o in live])
        if (
            snap.bombs <= 0
            and self.place_cd <= 0
            and live
            and not any(int(o.state) in DODONGO_STUNNED for o in live)
            # A cut one dies ~19 frames later with state back at 0 (S48 pin:
            # hp 240->208, then this check fired one frame before the kill).
            and all(int(o.hp) >= self._full_hp for o in live)
        ):
            return self._fail("out_of_bombs")
        if (
            self.select_item is not None
            and self._env is not None
            and snap.mode == PLAY_MODE
            and not snap.transitioning
            and snap.bombs > 0  # empty bag: nothing to select, the pause cycled forever
        ):
            curr = int(read_u8(self._env.get_ram(), ADDR_SELECTED_ITEM))
            if curr != int(self.select_item):
                if self._select is None:
                    self._select = PauseSelectController(want=int(self.select_item), name="bombs")
                    self._select.bind_env(self._env)
                act = self._select.drive(snap)
                if act is not None:
                    return act

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
            return self._stand("done")
        if not living:
            if self.frames > 200 and snap.room_all_dead >= 20 and not dodos:
                self.success = True
                self.phase = DodongoPhase.DONE
                self.notes.append("dodongo_dead_settle")
                return self._stand("done")
            wander = ("UP", "RIGHT", "DOWN", "LEFT")[self.frames // 20 % 4]
            return FrameAction(nes_action(wander, "A"), "dodo_search")

        self._track_face(living)
        bloated = [o for o in living if int(o.state) == DODONGO_BLOATED]
        if not bloated:
            self._at_mouth = False
        if bloated and snap.bombs > 0 and not self._bomb_armed():
            # Two swallowed bombs kill it, and it stands still through the
            # ~97-frame bloat (state 1): set the next bomb in front of the
            # mouth now. It swallows it on resuming, or the blast stuns it
            # for the sword. The three green pins all did exactly this; the
            # red ones walked off and came back after it moved.
            d = min(bloated, key=lambda o: abs(o.x - snap.link_x) + abs(o.y - snap.link_y))
            spot = mouth_stand(d)
            if spot is not None:
                (sx, sy), face = spot
                # Latched once reached: the turn press walks Link a couple of
                # px toward the body, and re-walking the off-lattice stand
                # swapped UP/DOWN every frame through the bloat (S48 pin).
                off = max(abs(int(snap.link_x) - sx), abs(int(snap.link_y) - sy))
                if off > DODONGO_STAND_SLACK:
                    self._at_mouth = False
                if not self._at_mouth:
                    step = room_step(snap, (sx, sy), tol=4, env=self._env)
                    if step is not None:
                        return FrameAction(nes_action(step), "dodo_bloat_walk")
                    self._at_mouth = True
                if int(snap.facing) != direction_to_facing(face):
                    return FrameAction(nes_action(face), "dodo_bloat_face")
                self.place_face = face
                self.place_cd = 95
                self.bombs_used += 1
                return FrameAction(nes_action(face, "B"), "dodo_bloat_place")
        stunned = [o for o in living if int(o.state) in DODONGO_STUNNED]
        if stunned and not self._bomb_armed():
            # Stunned, the sword finishes it: one white-sword cut from a stun
            # pin, dead 19 frames later. Placing on through the stun spent 7
            # bombs and timed out (natural_credits_poweron46).
            self.place_cd = 0
            d = min(stunned, key=lambda o: abs(o.x - snap.link_x) + abs(o.y - snap.link_y))
            stand = sword_stand(
                int(snap.link_x), int(snap.link_y), d, DODONGO_STAND_BOX, avoid_muzzle=True
            )
            # tol 4: a stand 20 px off the body is rarely a lattice node
            # (y=136 beside a body at 156); the nearest node is <= 4 px away.
            step = room_step(snap, stand, tol=4, env=self._env)
            if step is not None:
                return FrameAction(nes_action(step), "dodo_stun_walk")
            face = face_toward(int(snap.link_x), int(snap.link_y), int(d.x), int(d.y))
            if self.frames % 8 < 4:
                return FrameAction(nes_action(face, "A"), "dodo_stun_slash")
            return FrameAction(nes_action(face), "dodo_stun_face")
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
            return self._stand("dodo_wait_blast")

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
        # The bomb lands 16 px ahead of Link, so Link stands where the bloat
        # stand is (a body length off the mouth), not 16 px off the body: the
        # NC65 pin stood 9 px from a south-facing Dodongo, every bomb dropped
        # on its back, and all 7 were spent. 66/66 saved pins with this.
        spot = mouth_stand(d)
        if spot is not None:
            (tx, ty), face = spot
            at_mouth = (
                abs(snap.link_x - tx) <= DODONGO_STAND_SLACK
                and abs(snap.link_y - ty) <= DODONGO_STAND_SLACK
            )
        front = in_front_of_mouth(snap.link_x, snap.link_y, d)
        path_ok = mouth_path_clear(d)
        stable = self.stable_n.get(d.slot, 0) >= self.stable_face_frames
        if snap.bombs <= 0:
            if int(d.hp) < self._full_hp:
                return self._stand("dodo_hurt_wait")
            return self._fail("out_of_bombs")
        if dist < self.contact and not (at_mouth and front):
            dx, dy = d.x - snap.link_x, d.y - snap.link_y
            if abs(dx) >= abs(dy):
                dest = (snap.link_x - 24 if dx > 0 else snap.link_x + 24, snap.link_y)
            else:
                dest = (snap.link_x, snap.link_y - 24 if dy > 0 else snap.link_y + 24)
            act, _ = goto_action(snap, dest[0], dest[1], tol=4)
            return FrameAction(act, "dodo_standoff")
        if at_mouth and front and path_ok and stable:
            # B uses Link's facing from the previous frame. Turn first, then
            # re-check the moving mouth before committing a bomb.
            if int(snap.facing) != direction_to_facing(face):
                return FrameAction(nes_action(face), "dodo_face")
            self.place_face = face
            self.place_cd = 95
            self.bombs_used += 1
            return FrameAction(nes_action(face, "B"), "dodo_place")
        if at_mouth and front:
            return self._stand("dodo_wait_mouth")
        if not path_ok:
            return self._stand("dodo_wait_mouth")
        act, _ = goto_action(snap, tx, ty, tol=4)
        return FrameAction(act, "dodo_approach")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "phase": self.phase.name,
            "frames": self.frames,
            "bombs_used_est": self.bombs_used,
            "hits_est": self.hits_est,
            "occupancy_misses": 0,
            "occupancy_blocked": 0,
            "poke": False,
            "route_eligible": False,
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
