"""L3 dest hops: Raft reverse (Survival) and Entrance→TF Clean suffix.

Directed dest (RAM)::

    0x5b BOMB_RIGHT@(192,141) → 0x5c
    clear Darknuts (doors R|L) → RIGHT y≈141 → 0x5d
    clear Zol/Gel/Keese (ignore 0x2b) → UP → 0x4d Manhandla 0x3c
    bombs → HC → UP 0x3d → TF bit 0x04

Survival still enters from Level3Raft via exit_passage + 0x69 UP + bomb-R 0x59.
Clean Entrance→TF skips Raft and uses ``level3_entrance_tf_stages``.
No ``idle(n)`` as a hop; door dest is leftover-relative ``door_band_goal``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import save_rgb_png
from zelda_i.dungeon.bomb_wall import BombWallController
from zelda_i.dungeon.door_hop import DoorHopController, DoorHopSpec
from zelda_i.dungeon.engine import DungeonPhase, GenericDungeonRoomController
from zelda_i.dungeon.hop_controller import (
    HopController,
    WAIT_SCROLL_B,
    dungeon_align_then_push,
)
from zelda_i.dungeon.ops import (
    live_killables,
    poke_bombs,
    room_fields,
)
from zelda_i.dungeon.pause_select import B_SLOT_BOMBS, PauseSelectController, drink_if_low
from zelda_i.dungeon.tilemap import has_room_tile_map
from zelda_i.level3.boss_combat import (
    BOMB_NORTH_STANDS,
    Level3BossCombatMixin,
    PREP_CLEAR_TYPES,
    UP_APPROACHES,
    exit_raft_passage,
    prep_5d_still_killable,
)
from zelda_i.anchors import TF_BIT_L3 as LEVEL3_TRIFORCE_BIT
from zelda_i.level3.occupancy import seed_block_cells
from zelda_i.level3.dungeon import (
    DARKNUT_OBJECT_TYPE,
    INVULN_MOVER_0X2B,
    MANHANDLA_OBJECT_TYPE,
    ROOM_5C_SPEC,
    ROOM_5D_SPEC,
    ROOM_69_SPEC,
    ROOM_L3_BOSS,
    ROOM_L3_BOSS_PREP,
    ROOM_L3_BOMB_SHORTCUT,
    ROOM_L3_COMPASS,
    ROOM_L3_DARKNUTS,
    ROOM_L3_SOUTH_DARKNUTS,
    ROOM_L3_TF,
    ROOM_L3_WEST_DARKNUTS,
    level3_manhandla_live,
)
from zelda_i.level3.geometry import (
    BOMB_STAND_59_RIGHT,
    BOMB_STAND_5B_RIGHT,
    DOOR_5C_RIGHT_Y,
    NORTH_DOOR_X,
    NORTH_DOOR_X_TOL,
    PASSAGE_EXIT_WAYPOINTS,
)
from zelda_i.level3.overworld import LEVEL3
from zelda_i.level3.raft_path import SPAWN_SETTLE_FRAMES
from zelda_i.paths import RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_LINK_FACING,
    ADDR_SELECTED_ITEM,
    PLAY_MODE,
    ZeldaSnapshot,
    read_snapshot,
    read_u8,
)
from zelda_i.screen_glance import leftover_from_snapshot

BOSS_PATH_PHASES: tuple[str, ...] = (
    "exit_passage",
    "up_69",
    "bomb_59",
    "right_5a",
    "bomb_5b",
    "clear_5c",
    "right_5d",
    "clear_prep",
    "open_up",
    "manhandla",
    "collect_tf",
    "done",
    "failed",
)

BOSS_PATH_MAX_FRAMES = 120_000
BOMB_5B_MAX_FRAMES = 8000
MANHANDLA_MAX_FRAMES = 16000
# Clean 0x4d: stay south of the waist, bomb the flower centroid, retreat.
# Serial red 4 chased north to (82,101) after two bombs (accelerated heads).
MANHANDLA_FIGHT_Y_MIN = 141
MANHANDLA_FIGHT_Y_MAX = 173
MANHANDLA_FIGHT_X_MIN = 56
MANHANDLA_FIGHT_X_MAX = 184
MANHANDLA_CONTACT = 32
MANHANDLA_BOMB_MIN = 24
MANHANDLA_BOMB_MAX = 52
MANHANDLA_BOMB_CD = 72
MANHANDLA_RETREAT = 36
MANHANDLA_SWORD = 22
EAST_DOOR = (208, 141)
NORTH_DOOR = (NORTH_DOOR_X, 93)
# 0x5c diamond waist (96,141) pockets occupancy. Spec-declare $6530
# BLOCK_TILES on the dest hop so inferred misses are not forgotten.


def _l3_door(
    spec_id: str,
    room: int,
    goal: tuple[int, int],
    hold_dir: str,
    policy: str,
    dest_room: int,
    **kw: Any,
) -> DoorHopSpec:
    kw.setdefault("level", LEVEL3)
    kw.setdefault("fail_ow", True)
    kw.setdefault("wait_modes", WAIT_SCROLL_B)
    return DoorHopSpec(spec_id, room, goal, hold_dir, policy, dest_room=dest_room, **kw)


UP_69_SPEC = _l3_door(
    "level3_up_0x69",
    ROOM_L3_SOUTH_DARKNUTS,
    NORTH_DOOR,
    "UP",
    "occupancy x=120 UP; dest 0x59",
    ROOM_L3_WEST_DARKNUTS,
)
RIGHT_5A_SPEC = _l3_door(
    "level3_right_0x5a",
    ROOM_L3_COMPASS,
    EAST_DOOR,
    "RIGHT",
    "occupancy y=141 RIGHT; dest 0x5b",
    ROOM_L3_DARKNUTS,
    push_at_goal=True,
    align="y",
    cardinal_hold=True,
)
RIGHT_5C_SPEC = _l3_door(
    "level3_right_0x5c",
    ROOM_L3_BOMB_SHORTCUT,
    EAST_DOOR,
    "RIGHT",
    "occupancy dest 0x5d; $6530 BLOCK_TILES seed (no cardinal_hold, no y-align)",
    ROOM_L3_BOSS_PREP,
    push_at_goal=True,
    align="dest",
)
UP_5D_SPEC = _l3_door(
    "level3_up_0x5d",
    ROOM_L3_BOSS_PREP,
    NORTH_DOOR,
    "UP",
    "occupancy dest 0x4d; $6530 BLOCK_TILES seed (no south_band)",
    ROOM_L3_BOSS,
    align="dest",
)
UP_4D_SPEC = _l3_door(
    "level3_up_0x4d",
    ROOM_L3_BOSS,
    NORTH_DOOR,
    "UP",
    "occupancy x=120 UP; dest 0x3d",
    ROOM_L3_TF,
    south_band=True,
)


@dataclass
class L3DoorHopController(DoorHopController):
    """Occupancy dest hop. Shared engine; L3 has no rod so skip the L6 gate.

    Dest hops seed spec-declared BLOCK_TILES from cart-WRAM ``$6530`` on
    bind_env / first play so 0x5c diamonds and the 0x5d center plus survive
    occupancy forget. North-door dest (120,93) and band y=109 stay open.
    """

    env: Any = field(default=None, repr=False)
    _blocks_seeded: bool = field(default=False, init=False, repr=False)

    def bind_env(self, env: Any) -> None:
        self.env = env
        ram = env.get_ram()
        snap = read_snapshot(ram)
        if (
            snap.screen == self.spec.room
            and snap.mode == PLAY_MODE
            and not snap.transitioning
        ):
            self._seed_blocks(ram)

    def _seed_blocks(self, ram: Any) -> None:
        if self._blocks_seeded:
            return
        if not has_room_tile_map(ram):
            return
        n = seed_block_cells(self.walker.grid, ram)
        gx, gy = self.spec.goal
        for y in range(gy - 8, gy + 9):
            self.walker.grid.blocked.discard((gx, y))
        self.walker.grid.blocked.discard(self.goal)
        self.walker.grid.blocked.discard((gx, int(self.spec.north_band_y)))
        self.walker.path = None
        self._blocks_seeded = True
        if n:
            self.notes.append(f"seed_blocks_{n}")

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        if (
            not self._blocks_seeded
            and self.env is not None
            and snap.screen == self.spec.room
            and snap.mode == PLAY_MODE
        ):
            self._seed_blocks(self.env.get_ram())
        return super().policy(snap)

    def _dest(self, snap: ZeldaSnapshot) -> FrameAction | None:
        spec = self.spec
        if snap.screen == spec.room:
            return None
        if snap.mode != PLAY_MODE or snap.transitioning:
            return None
        xy = f"{snap.link_x}_{snap.link_y}"
        if spec.dest_room is not None and snap.screen != spec.dest_room:
            return self._fail(snap, f"wrong_room_{snap.screen:02x}_{xy}")
        if spec.success_fn(
            snap, not_room=spec.room, dest_room=spec.dest_room, passage_ok=False
        ):
            return self._mark_success(snap)
        return None


@dataclass(frozen=True)
class _L3BombWall:
    room: int
    stand: tuple[int, int]
    face: str
    opens_to: int


L3_WALL_5B = _L3BombWall(
    ROOM_L3_DARKNUTS, BOMB_STAND_5B_RIGHT, "RIGHT", ROOM_L3_BOMB_SHORTCUT
)
L3_WALL_59 = _L3BombWall(
    ROOM_L3_WEST_DARKNUTS, BOMB_STAND_59_RIGHT, "RIGHT", ROOM_L3_COMPASS
)


def make_l3_bomb_5b() -> BombWallController:
    """0x5b bomb-RIGHT → 0x5c. Pause-select bombs; never poke count.

    Leftover after dest_6b clear is inland (live (176,125) stand_timeout).
    y-first to the waist then RIGHT to (192,141).
    """
    return BombWallController(
        wall=L3_WALL_5B,
        level=LEVEL3,
        select_item=B_SLOT_BOMBS,
        wait_hold_face=True,
        require_bomb_consumed=False,
        max_frames=BOMB_5B_MAX_FRAMES,
        approach_waypoints=(BOMB_STAND_5B_RIGHT,),
    )


def make_l3_bomb_59() -> BombWallController:
    """Post-Raft 0x59 bomb-RIGHT → 0x5a (walk sealed)."""
    return BombWallController(
        wall=L3_WALL_59,
        level=LEVEL3,
        select_item=B_SLOT_BOMBS,
        wait_hold_face=True,
        require_bomb_consumed=False,
        max_frames=BOMB_5B_MAX_FRAMES,
    )


@dataclass
class Level3SpawnClearController:
    """Wait for spawn (RAM live or settle window), then dest-clear + doors."""

    spec: Any
    spawn_max: int = SPAWN_SETTLE_FRAMES
    frames: int = 0
    spawn_frames: int = 0
    saw_live: bool = False
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)
    leftover: dict[str, Any] = field(default_factory=dict)
    combat: Any = field(init=False)
    max_frames: int = field(init=False)
    route_eligible: bool = False

    def __post_init__(self) -> None:
        self.combat = GenericDungeonRoomController(self.spec)
        self.max_frames = int(self.spec.max_frames) + int(self.spawn_max)

    def bind_env(self, env: Any) -> None:
        ram = env.get_ram()
        walker = getattr(self.combat, "walker", None)
        if walker is not None and has_room_tile_map(ram):
            n = seed_block_cells(walker.grid, ram)
            if n:
                self.notes.append(f"seed_blocks_{n}")

    def _doors_ok(self, snap: ZeldaSnapshot) -> bool:
        need = int(self.spec.required_open_doors or 0)
        if not need:
            return True
        return (int(snap.cur_opened_doors) & need) == need

    def _cleared(self, snap: ZeldaSnapshot) -> bool:
        if (
            snap.screen != self.spec.room_id
            or snap.mode != PLAY_MODE
            or snap.transitioning
        ):
            return False
        if self.spec.live_enemies(snap):
            return False
        if not self.saw_live and self.spawn_frames < self.spawn_max:
            return False
        return self._doors_ok(snap)

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.leftover = leftover_from_snapshot(snap)
        if self.success:
            return FrameAction(nes_idle_action(), "done")
        if self.failed:
            return FrameAction(nes_idle_action(), "failed")
        if snap.mode == 17:
            self.failed = True
            self.notes.append("link_death")
            return FrameAction(nes_idle_action(), "link_death")
        if self.frames >= self.max_frames:
            self.failed = True
            self.notes.append("timeout")
            return FrameAction(nes_idle_action(), "timeout")
        live = self.spec.live_enemies(snap)
        if live:
            self.saw_live = True
            action = self.combat.step(snap)
            if self.combat.phase is DungeonPhase.FAILED:
                self.failed = True
                self.notes.append("clear_failed")
            return action
        if not self.saw_live and self.spawn_frames < self.spawn_max:
            self.spawn_frames += 1
            return FrameAction(nes_idle_action(), "spawn_wait")
        if self._cleared(snap):
            self.success = True
            self.notes.append(f"cleared_0x{self.spec.room_id:02x}")
            return FrameAction(nes_idle_action(), "done")
        if self.combat.success or self.combat.phase is DungeonPhase.DONE:
            return FrameAction(nes_idle_action(), "wait_doors")
        return self.combat.step(snap)

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "leftover": dict(self.leftover),
            "saw_live": self.saw_live,
            "spec_id": self.spec.spec_id,
            "route_eligible": False,
        }


@dataclass
class Level3ManhandlaController(HopController):
    """0x4d leftover: south-band bomb centroid, HC, UP 0x3d. Dest TF 0x04."""

    spec_id: str = "level3_manhandla_tf"
    room: int = ROOM_L3_BOSS
    max_frames: int = MANHANDLA_MAX_FRAMES
    wait_modes: tuple[int, ...] = WAIT_SCROLL_B
    done_reason: str = "tf04"
    leftover: dict[str, Any] = field(default_factory=dict)
    samples: list[dict[str, Any]] = field(default_factory=list)
    bomb_cd: int = 0
    retreat_frames: int = 0
    retreat_dir: str = "DOWN"
    saw_heads: bool = False
    hc0: int | None = None
    env: Any | None = None
    _select: PauseSelectController | None = field(default=None, repr=False)
    route_eligible: bool = False
    writes: int = 0

    def bind_env(self, env: Any) -> None:
        self.env = env

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        return bool(int(snap.triforce) & LEVEL3_TRIFORCE_BIT)

    def on_arrive(self, snap: ZeldaSnapshot) -> str:
        return f"tf04_{snap.screen:02x}_{snap.link_x}_{snap.link_y}"

    def emit(
        self, snap: ZeldaSnapshot, action: FrameAction, *, force: bool = False
    ) -> FrameAction:
        if force or self.frames <= 2 or self.frames % 16 == 0:
            self.leftover = leftover_from_snapshot(snap)
            self.samples.append(
                {
                    "frame": self.frames,
                    "x": int(snap.link_x),
                    "y": int(snap.link_y),
                    "screen": int(snap.screen),
                    "reason": action.reason,
                    "bombs": int(snap.bombs),
                    "triforce": int(snap.triforce),
                    "health": int(snap.health),
                    "heads": int(self.saw_heads),
                }
            )
        return action

    def _selected(self) -> int | None:
        if self.env is None:
            return B_SLOT_BOMBS
        return int(read_u8(self.env.get_ram(), ADDR_SELECTED_ITEM))

    def _select_bombs(self, snap: ZeldaSnapshot) -> FrameAction:
        if self.env is None:
            return FrameAction(nes_action("B"), "place_bomb")
        if self._select is None:
            self._select = PauseSelectController(want=B_SLOT_BOMBS, name="bombs")
            self._select.bind_env(self.env)
        driven = self._select.drive(snap)
        if self._select.failed:
            return self.mark_fail(self._select.fail_reason or "bombs_not_selected")
        if driven is not None:
            return driven
        return FrameAction(nes_idle_action(), "bombs_selected")

    def _fight(self, snap: ZeldaSnapshot, heads: list) -> FrameAction:
        ram = self.env.get_ram() if self.env else None

        if self.bomb_cd > 0:
            self.bomb_cd -= 1
        if self.retreat_frames > 0:
            self.retreat_frames -= 1

        # Determine Center and Velocity
        slot5 = next((o for o in heads if o.slot == 5), None)
        if slot5:
            cx, cy = int(slot5.x), int(slot5.y)
            facing = slot5.facing if ram is None else read_u8(ram, ADDR_LINK_FACING + 5)
        else:
            cx = sum(int(h.x) for h in heads) // len(heads)
            cy = sum(int(h.y) for h in heads) // len(heads)
            facing = heads[0].facing if ram is None else read_u8(ram, ADDR_LINK_FACING + heads[0].slot)

        dx = (1 if facing & 0x01 else 0) - (1 if facing & 0x02 else 0)
        dy = (1 if facing & 0x04 else 0) - (1 if facing & 0x08 else 0)
        speed = 0.5 if len(heads) >= 4 else 1.5
        pred_cx = max(MANHANDLA_FIGHT_X_MIN + 8, min(MANHANDLA_FIGHT_X_MAX - 8, cx + int(round(48 * speed * dx))))

        # North-of-waist guard: never chase north into the waist
        if snap.link_y < MANHANDLA_FIGHT_Y_MIN:
            return FrameAction(nes_action("DOWN"), "stay_south")

        # 1. Fireball avoidance (Highest priority!)
        dangerous_fb = [
            p for p in snap.objects
            if p.type_id == 0x56 and abs(p.x - snap.link_x) <= 16 and -8 <= (snap.link_y - p.y) <= 36
        ]
        if dangerous_fb:
            nfb = min(dangerous_fb, key=lambda p: abs(p.x - snap.link_x) + abs(p.y - snap.link_y))
            dodge = ("RIGHT" if snap.link_x <= MANHANDLA_FIGHT_X_MAX - 16 else "LEFT") if nfb.x <= snap.link_x else ("LEFT" if snap.link_x >= MANHANDLA_FIGHT_X_MIN + 16 else "RIGHT")
            return FrameAction(nes_action(dodge), "dodge_fireball")

        # 2. Post-bomb retreat handling: while a bomb is ticking, retreat away from centroid
        if self.retreat_frames > 0:
            if snap.link_y >= MANHANDLA_FIGHT_Y_MAX:
                toward = "LEFT" if snap.link_x > NORTH_DOOR_X else "RIGHT"
                return FrameAction(nes_action(toward), "retreat_bomb")
            return FrameAction(nes_action("DOWN"), "retreat_bomb")

        # 3. Initial climb from south door
        if snap.link_y > MANHANDLA_FIGHT_Y_MAX:
            return FrameAction(nes_action("UP"), "climb")

        # 4. Immediate Head Contact Avoidance
        nearest_head = min(heads, key=lambda o: abs(o.x - snap.link_x) + abs(o.y - snap.link_y))
        dist_nearest = abs(nearest_head.x - snap.link_x) + abs(nearest_head.y - snap.link_y)
        dhx, dhy = nearest_head.x - snap.link_x, nearest_head.y - snap.link_y
        face_head = "UP" if dhy < -4 else ("DOWN" if dhy > 4 else ("RIGHT" if dhx > 0 else "LEFT"))

        if dist_nearest < MANHANDLA_CONTACT:
            if nearest_head.y <= snap.link_y and snap.link_y < MANHANDLA_FIGHT_Y_MAX:
                return FrameAction(nes_action("DOWN"), "combat_backstep")
            dodge = ("LEFT" if snap.link_x >= MANHANDLA_FIGHT_X_MIN + 16 else "RIGHT") if nearest_head.x >= snap.link_x else ("RIGHT" if snap.link_x <= MANHANDLA_FIGHT_X_MAX - 16 else "LEFT")
            return FrameAction(nes_action(dodge), "combat_backstep")

        # 5. Centroid Interception / Bomb Placement
        if snap.bombs > 0 and self.bomb_cd <= 0:
            if self._selected() != B_SLOT_BOMBS:
                return self._select_bombs(snap)
            can_bomb = (dy > 0 and 124 <= cy <= 136 and abs(snap.link_x - pred_cx) <= 8 and snap.link_y >= 165) or \
                       (dy == 0 and 140 <= cy <= 160 and abs(snap.link_x - pred_cx) <= 12 and snap.link_y >= cy + 12) or \
                       (len(heads) <= 2 and 24 <= dist_nearest <= MANHANDLA_BOMB_MAX)
            if can_bomb:
                self.bomb_cd, self.retreat_frames = 75, 55
                b_dir = "UP" if len(heads) > 2 else face_head
                return FrameAction(nes_action(b_dir, "B"), "place_bomb")

        # 6. Sword fallback when out of bombs
        if snap.bombs <= 0 and dist_nearest <= 28:
            return FrameAction(nes_action(face_head, "A"), "sword_slash")

        # 7. Intercept Positioning (Position Link at pred_cx in south band y=168..173)
        target_x = pred_cx
        target_y = min(MANHANDLA_FIGHT_Y_MAX, max(165, cy + 32))
        if abs(snap.link_x - target_x) > 6:
            return FrameAction(nes_action("RIGHT" if snap.link_x < target_x else "LEFT"), "approach")
        if abs(snap.link_y - target_y) > 4:
            return FrameAction(nes_action("DOWN" if snap.link_y < target_y else "UP"), "approach")
        return FrameAction(nes_action("UP"), "approach")

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        if snap.level != LEVEL3:
            return self.mark_fail(f"left_level_{snap.level}")
        if snap.screen == ROOM_L3_TF:
            return dungeon_align_then_push(
                snap, push_dir="UP", target_x=NORTH_DOOR_X, target_y=141, reason="tf"
            )
        if snap.screen != ROOM_L3_BOSS:
            return self.mark_fail(f"left_0x{self.room:02x}_to_0x{snap.screen:02x}")
        if self.hc0 is None:
            self.hc0 = int(snap.heart_containers)
        heads = level3_manhandla_live(snap)
        if heads:
            self.saw_heads = True
            return self._fight(snap, heads)
        if not self.saw_heads:
            if snap.link_y > MANHANDLA_FIGHT_Y_MAX:
                return FrameAction(nes_action("UP"), "climb")
            return FrameAction(nes_idle_action(), "spawn_wait")
        if int(snap.heart_containers) <= self.hc0:
            return dungeon_align_then_push(
                snap,
                push_dir="UP",
                target_x=NORTH_DOOR_X,
                target_y=141,
                reason="hc",
            )
        if abs(snap.link_x - NORTH_DOOR_X) > NORTH_DOOR_X_TOL:
            btn = "LEFT" if snap.link_x > NORTH_DOOR_X else "RIGHT"
            return FrameAction(nes_action(btn), "north_align")
        return FrameAction(nes_action("UP"), "north_push")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "samples": list(self.samples),
            "leftover": dict(self.leftover),
            "spec_id": self.spec_id,
            "route_eligible": False,
            "writes": int(self.writes),
            "policy": "south-band centroid bomb; retreat; HC; UP 0x3d; dest TF 0x04",
        }


def level3_boss_suffix_stages():
    """Cleared 0x5b leftover → bomb-R 0x5c → 0x5d → Manhandla → TF 0x04."""
    return (
        ("bomb_5b", make_l3_bomb_5b(), BOMB_5B_MAX_FRAMES),
        (
            "clear_5c",
            Level3SpawnClearController(ROOM_5C_SPEC),
            ROOM_5C_SPEC.max_frames + SPAWN_SETTLE_FRAMES,
        ),
        ("right_5d", L3DoorHopController(RIGHT_5C_SPEC), RIGHT_5C_SPEC.max_frames),
        (
            "clear_5d",
            Level3SpawnClearController(ROOM_5D_SPEC),
            ROOM_5D_SPEC.max_frames + SPAWN_SETTLE_FRAMES,
        ),
        ("up_4d", L3DoorHopController(UP_5D_SPEC), UP_5D_SPEC.max_frames),
        ("manhandla_tf", Level3ManhandlaController(), MANHANDLA_MAX_FRAMES),
    )


@dataclass
class _PlayWait(HopController):
    room: int = 0
    wait_modes: tuple[int, ...] = WAIT_SCROLL_B
    done_reason: str = "play_ready"

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        return (
            snap.screen == self.room
            and snap.mode == PLAY_MODE
            and not snap.transitioning
        )

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        del snap
        return FrameAction(nes_idle_action(), "wait_play")


@dataclass
class Level3BossPathController(Level3BossCombatMixin):
    """Assisted Survival: Level3Raft → Manhandla → TF bit 0x04.

    Phases (see ``BOSS_PATH_PHASES``)::

        exit_passage → up_69 → bomb_59 → right_5a → bomb_5b → clear_5c
        → right_5d → clear_prep → open_up → manhandla → collect_tf → done

    Hybrid: methods take ``env`` / ``assist`` / ``total`` frame counter.
    """

    frames: int = 0
    phase: str = "exit_passage"
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)
    traps: list[str] = field(default_factory=list)
    path_log: list[dict] = field(default_factory=list)
    max_frames: int = BOSS_PATH_MAX_FRAMES
    poke_bombs: int | None = None  # RECON opt-in; durable default off
    # Bombs Manhandla may not spend: Link leaves them for the next walls.
    manhandla_bomb_reserve: int = 0
    tag: str = "l3_to_boss"
    continuous_mode: bool = False  # one-way policy: never restore emulator state
    # Outcome flags
    reached_5d: bool = False
    reached_4d: bool = False
    boss_beaten: bool = False
    tf04: bool = False
    manhandla_confirmed: bool = False
    dmg_events: int = 0
    last_error: str | None = None
    state_restores: int = 0
    # Sub-reports
    path_to_5d_report: dict | None = None
    gate_5d_report: dict | None = None
    fight_report: dict | None = None

    def _set_phase(self, phase: str, note: str = "") -> None:
        if phase != self.phase:
            self.phase = phase
            if note:
                self.notes.append(note)

    def _fail(self, error: str) -> dict[str, Any]:
        self.failed = True
        self.last_error = error
        self._set_phase("failed", error)
        return {"ok": False, "error": error}

    def _maybe_poke(self, env: Any, *, note: bool = True) -> None:
        if self.poke_bombs is None:
            return
        msg = poke_bombs(env, self.poke_bombs)
        if note:
            self.notes.append(f"RECON poke {msg}")

    def _restore_state(self, env: Any, state: Any) -> None:
        """Recon retry primitive; forbidden by construction on the spine."""
        if self.continuous_mode:
            raise RuntimeError("continuous_mode forbids emulator state restore")
        env.em.set_state(state)
        self.state_restores += 1

    def _drive_hop(
        self,
        env: Any,
        assist: Any | None,
        total: list[int],
        controller: Any,
        *,
        max_frames: int | None = None,
    ) -> Any:
        bind = getattr(controller, "bind_env", None)
        if callable(bind):
            bind(env)
        limit = int(max_frames or getattr(controller, "max_frames", 4000) or 4000)
        for _ in range(limit):
            drink_if_low(env, assist, total)
            snap = read_snapshot(env.get_ram())
            action = controller.step(snap)
            env.step(action.action)
            total[0] += 1
            if assist is not None:
                assist.apply_env(env, frame=total[0])
            if getattr(controller, "success", False) or getattr(
                controller, "failed", False
            ):
                break
            phase = getattr(controller, "phase", None)
            pname = getattr(phase, "name", phase)
            if isinstance(pname, str) and pname.upper() in {"FAILED", "DONE"}:
                break
            if snap.mode == 17:
                break
        return controller

    def _hop_fail(
        self,
        error: str,
        path_log: list[dict],
        traps: list[str],
        notes: list[str],
        env: Any,
        hop: Any | None = None,
    ) -> dict[str, Any]:
        snap = read_snapshot(env.get_ram())
        out = self._fail(error)
        out.update(
            {
                "path_log": path_log,
                "final": room_fields(snap, env.get_ram()),
                "traps": traps,
                "notes": notes,
                "hop": hop.report() if hop is not None and hasattr(hop, "report") else None,
            }
        )
        self.path_log.extend(path_log)
        return out

    def path_to_5d(
        self,
        env: Any,
        assist: Any | None,
        total: list[int],
    ) -> dict[str, Any]:
        """Directed dest hops: Level3Raft / 0x0f → play 0x5d."""
        path_log: list[dict] = []
        traps: list[str] = []
        notes: list[str] = []
        self._set_phase("exit_passage")

        ex = exit_raft_passage(env, assist, total)
        path_log.append(
            {
                "step": "passage_exit",
                "ok": ex.get("ok"),
                "to": (ex.get("after") or {}).get("sc"),
                "error": ex.get("error"),
            }
        )
        if not ex.get("ok"):
            out = self._fail("passage_exit_failed")
            out["path_log"] = path_log
            out["exit"] = ex
            return out
        obs, *_ = env.step(nes_idle_action())
        save_rgb_png(obs, RECORDINGS_DIR / f"{self.tag}_exit_0x69.png")
        total[0] += 1
        self._maybe_poke(env)

        snap = read_snapshot(env.get_ram())
        if snap.screen == ROOM_L3_SOUTH_DARKNUTS:
            self._set_phase("up_69", "entered_0x69")
            live_dn = live_killables(snap, (DARKNUT_OBJECT_TYPE,))
            if live_dn:
                # The respawned Darknuts on the way back up: the engine clear
                # (flank strike), not ``fight_clear``'s centre patrol, which
                # took 3.5-4.7 hearts here on two of eight Clean offsets.
                clr = self._drive_hop(env, assist, total, Level3SpawnClearController(ROOM_69_SPEC))
                path_log.append(
                    {
                        "step": "clear_69",
                        "ok": clr.success,
                        "frames": clr.frames,
                    }
                )
            hop = L3DoorHopController(UP_69_SPEC)
            self._drive_hop(env, assist, total, hop)
            path_log.append(
                {
                    "step": "69_up",
                    "ok": hop.success,
                    "to": hop.leftover.get("screen") if hop.leftover else None,
                }
            )
            if not hop.success:
                return self._hop_fail(
                    "failed_69_up", path_log, traps, notes, env, hop
                )

        self._set_phase("bomb_59")
        if self.poke_bombs is not None and read_snapshot(env.get_ram()).bombs < 2:
            poke_bombs(env, self.poke_bombs)
        hop = make_l3_bomb_59()
        self._drive_hop(env, assist, total, hop)
        path_log.append({"step": "59_bomb_right", "ok": hop.success})
        if not hop.success:
            traps.append("0x59 walk-RIGHT sealed post-Raft (expected)")
            return self._hop_fail(
                "failed_59_bomb_right", path_log, traps, notes, env, hop
            )

        self._set_phase("right_5a")
        hop = L3DoorHopController(RIGHT_5A_SPEC)
        self._drive_hop(env, assist, total, hop)
        path_log.append({"step": "5a_right", "ok": hop.success})
        if not hop.success:
            return self._hop_fail(
                "failed_5a_right", path_log, traps, notes, env, hop
            )

        self._set_phase("bomb_5b")
        if self.poke_bombs is not None and read_snapshot(env.get_ram()).bombs < 2:
            poke_bombs(env, self.poke_bombs)
        hop = make_l3_bomb_5b()
        self._drive_hop(env, assist, total, hop)
        path_log.append({"step": "5b_bomb_right", "ok": hop.success})
        if not hop.success:
            return self._hop_fail(
                "failed_5b_bomb_right", path_log, traps, notes, env, hop
            )

        self._set_phase("clear_5c")
        snap = read_snapshot(env.get_ram())
        if snap.screen == ROOM_L3_BOMB_SHORTCUT:
            if self.poke_bombs is not None:
                poke_bombs(env, self.poke_bombs)
            clr = Level3SpawnClearController(ROOM_5C_SPEC)
            self._drive_hop(env, assist, total, clr)
            path_log.append(
                {
                    "step": "clear_5c",
                    "ok": clr.success,
                    "frames": clr.frames,
                    "saw_live": clr.saw_live,
                }
            )
            if not clr.success:
                return self._hop_fail(
                    "failed_5c_clear", path_log, traps, notes, env, clr
                )
            self._set_phase("right_5d")
            hop = L3DoorHopController(RIGHT_5C_SPEC)
            self._drive_hop(env, assist, total, hop)
            path_log.append({"step": "5c_right", "ok": hop.success})
            if not hop.success:
                obs, *_ = env.step(nes_idle_action())
                total[0] += 1
                save_rgb_png(obs, RECORDINGS_DIR / f"{self.tag}_failed_0x5c.png")
                traps.append("0x5c RIGHT dest hop missed 0x5d")
                return self._hop_fail(
                    "failed_5c_right", path_log, traps, notes, env, hop
                )

        wait = _PlayWait(room=ROOM_L3_BOSS_PREP, max_frames=120)
        self._drive_hop(env, assist, total, wait)
        snap = read_snapshot(env.get_ram())
        ok = snap.screen == ROOM_L3_BOSS_PREP and snap.level == LEVEL3
        obs, *_ = env.step(nes_idle_action())
        total[0] += 1
        save_rgb_png(obs, RECORDINGS_DIR / f"{self.tag}_prep_0x5d.png")
        if ok:
            self.reached_5d = True
            self._set_phase("clear_prep", "arrived_0x5d")
        else:
            self._fail("not_at_5d")
        result = {
            "ok": ok,
            "path_log": path_log,
            "traps": traps,
            "notes": notes,
            "final": room_fields(snap, env.get_ram()),
            "mode_at_5d": snap.mode,
        }
        self.path_log.extend(path_log)
        self.traps.extend(traps)
        self.notes.extend(notes)
        self.path_to_5d_report = result
        self.frames = total[0]
        return result

    def confirm_manhandla(self, env: Any) -> list:
        """Record live Manhandla heads on current snapshot."""
        snap = read_snapshot(env.get_ram())
        heads = level3_manhandla_live(snap)
        self.manhandla_confirmed = len(heads) > 0
        if heads:
            self.notes.append(
                f"Manhandla type 0x3c: {len(heads)} live heads "
                f"hps={[o.hp for o in heads]}"
            )
        return heads

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "phase": self.phase,
            "frames": self.frames,
            "notes": list(self.notes),
            "traps": list(self.traps),
            "path_log": list(self.path_log),
            "reached_5d": self.reached_5d,
            "reached_4d": self.reached_4d,
            "boss_beaten": self.boss_beaten,
            "tf04": self.tf04,
            "manhandla_confirmed": self.manhandla_confirmed,
            "dmg_events": self.dmg_events,
            "last_error": self.last_error,
            "phases": list(BOSS_PATH_PHASES),
            "poke_bombs": self.poke_bombs,
            "continuous_mode": self.continuous_mode,
            "state_restores": self.state_restores,
            "path": (
                "0x0f exit→0x69 UP dest→0x59 BOMB_R dest→0x5a R dest→0x5b "
                "BOMB_R dest→0x5c clear R dest→0x5d"
            ),
            "intervention_class": "survival",
            "track": "assisted",
            "geometry": {
                "bomb_stand_59": list(BOMB_STAND_59_RIGHT),
                "bomb_stand_5b": list(BOMB_STAND_5B_RIGHT),
                "door_5c_right_y": DOOR_5C_RIGHT_Y,
                "passage_exit_waypoints": [list(w) for w in PASSAGE_EXIT_WAYPOINTS],
                "prep_clear_types": [f"0x{t:02x}" for t in PREP_CLEAR_TYPES],
                "invuln_ignored": f"0x{INVULN_MOVER_0X2B:02x}",
                "manhandla_type": f"0x{MANHANDLA_OBJECT_TYPE:02x}",
            },
        }


__all__ = [
    "BOMB_NORTH_STANDS",
    "BOSS_PATH_MAX_FRAMES",
    "BOSS_PATH_PHASES",
    "L3DoorHopController",
    "Level3BossPathController",
    "Level3ManhandlaController",
    "Level3SpawnClearController",
    "PREP_CLEAR_TYPES",
    "RIGHT_5A_SPEC",
    "RIGHT_5C_SPEC",
    "UP_4D_SPEC",
    "UP_5D_SPEC",
    "UP_69_SPEC",
    "UP_APPROACHES",
    "exit_raft_passage",
    "level3_boss_suffix_stages",
    "make_l3_bomb_59",
    "make_l3_bomb_5b",
    "prep_5d_still_killable",
]
