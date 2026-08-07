"""Level 2 Dodongo path combat: room clear, bomb-N 0x1e, Dodongo fight.

Re-exported via ``level2_boss_path`` for public API stability.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.dungeon import (
    AliveRule,
    CombatTuning,
    DoorRoute,
    DungeonPhase,
    DungeonRoomSpec,
    GenericDungeonRoomController,
    RewardKind,
    RewardSpec,
)
from zelda_i.dungeon_ops import (
    ADDR_SELECTED_ITEM,
    B_ITEM_BOMB,
    ensure_bomb,
    poke_bombs as ops_poke_bombs,
)
from zelda_i.level2_dungeon import (
    BOMB_N_STAND_TOL,
    BOMB_N_WAIT_BLAST,
    LEVEL_2,
)
from zelda_i.level2_puzzles import (
    BOMB_WALL_1E_NORTH,
    DOOR_UP,
    L2_BOSS_EXIT_DOOR_Y,
    BombWall,
    is_at_bomb_stand,
)
from zelda_i.ram import (
    ADDR_TRIFORCE,
    PLAY_MODE,
    ZeldaSnapshot,
    read_snapshot,
    read_u8,
)

# Local room / type constants (mirrored for combat module independence).
ROOM_1E: int = 0x1E
ROOM_0E: int = 0x0E
DODONGO_TYPE: int = 0x32
KEESE_TYPE: int = 0x1B
BOMB_WALL_1E: BombWall = BOMB_WALL_1E_NORTH
BOMB_STAND_1E: tuple[int, int] = BOMB_WALL_1E.stand
FACE_E, FACE_W, FACE_S, FACE_N = 0x01, 0x02, 0x04, 0x08
BOMB_1E_MAX_FRAMES: int = 8000
DODONGO_FIGHT_MAX_FRAMES: int = 14000

IGNORE_TYPES: frozenset[int] = frozenset(
    {0x55, 0x49, 0x4E, 0x5C, 0x4A, 0x60, 0x61, 0x62, 0x63}
)
TYPE_ONLY: frozenset[int] = frozenset({0x15, 0x1B, 0x41})

_WIDE_PATROL: tuple[tuple[int, int], ...] = (
    (120, 189),
    (64, 189),
    (64, 141),
    (64, 109),
    (120, 109),
    (176, 109),
    (176, 141),
    (176, 189),
    (120, 141),
    (120, 173),
)

LEVEL2_TF_BIT: int = 0x02


def triforce_bit_02(tf_value: int) -> bool:
    return bool(int(tf_value) & LEVEL2_TF_BIT)


def mouth_target(dodo: Any) -> tuple[int, int, str]:
    """Stand in front of Dodongo snout; return (x, y, face_when_placing)."""
    f = int(dodo.facing)
    if f & FACE_E:
        return dodo.x + 12, dodo.y, "LEFT"
    if f & FACE_W:
        return dodo.x - 12, dodo.y, "RIGHT"
    if f & FACE_S:
        return dodo.x, dodo.y + 12, "UP"
    if f & FACE_N:
        return dodo.x, dodo.y - 12, "DOWN"
    return dodo.x, dodo.y, "UP"


def goto_action(
    snap: ZeldaSnapshot, tx: int, ty: int, tol: int = 6
) -> tuple[Any, bool]:
    """Axis-aligned walk toward (tx, ty). Returns (nes_action, at_target)."""
    if abs(snap.link_x - tx) > tol:
        return nes_action("RIGHT" if snap.link_x < tx else "LEFT"), False
    if abs(snap.link_y - ty) > tol:
        return nes_action("DOWN" if snap.link_y < ty else "UP"), False
    return nes_idle_action(), True


def idle(env: Any, n: int = 1) -> None:
    """L2 env idle (no assist counter). Delegates to dungeon_ops with null assist."""
    from zelda_i.dungeon_ops import idle as ops_idle

    ops_idle(env, None, [0], frames=n)


def ensure_bomb_selected(env: Any) -> None:
    """Prefer selected_item = bomb (0x02). Best-effort poke via dungeon_ops."""
    ensure_bomb(env)


def poke_bombs(env: Any, n: int = 16) -> str:
    """Assisted recon: top up bombs (not Clean)."""
    return ops_poke_bombs(env, n)


def live_objects(
    snap: ZeldaSnapshot, types: frozenset[int] | None = None
) -> list[Any]:
    out: list[Any] = []
    for o in snap.objects:
        if not (1 <= o.slot <= 10):
            continue
        if o.type_id in (0, 0xFF) or o.type_id in IGNORE_TYPES:
            continue
        if types is not None and o.type_id not in types:
            continue
        if o.type_id in TYPE_ONLY or o.hp > 0:
            out.append(o)
    return out


def sample_snapshot(snap: ZeldaSnapshot, ram: Any, *, event: str) -> dict[str, Any]:
    live = live_objects(snap)
    types = Counter(o.type_id for o in live)
    dodos = [
        o for o in snap.objects if o.type_id == DODONGO_TYPE and 1 <= o.slot <= 10
    ]
    return {
        "event": event,
        "mode": snap.mode,
        "sc": f"0x{snap.screen:02x}",
        "xy": [snap.link_x, snap.link_y],
        "keys": snap.keys,
        "bombs": snap.bombs,
        "doors": snap.cur_opened_doors,
        "mask": snap.open_doorway_mask,
        "all_dead": snap.room_all_dead,
        "room_item": snap.room_item_id,
        "live": len(live),
        "types": {f"0x{k:02x}": v for k, v in types.items()},
        "dodongo": [
            {"slot": o.slot, "x": o.x, "y": o.y, "hp": o.hp, "facing": o.facing}
            for o in dodos
        ],
        "tf": int(read_u8(ram, ADDR_TRIFORCE)),
        "tf02": triforce_bit_02(read_u8(ram, ADDR_TRIFORCE)),
    }


def enter_up(env: Any, dest: int, *, budget: int = 1600) -> bool:
    """UP door on L2: south band lateral then pure UP (diamond-safe)."""
    for _ in range(100):
        s = read_snapshot(env.get_ram())
        if s.mode != PLAY_MODE:
            env.step(nes_idle_action())
            continue
        if s.link_y >= 180:
            break
        env.step(nes_action("DOWN"))
    last = (-1, -1)
    stuck = 0
    for _ in range(budget):
        s = read_snapshot(env.get_ram())
        if s.screen == dest and s.mode == PLAY_MODE:
            return True
        if s.mode != PLAY_MODE:
            env.step(
                nes_action("UP")
                if s.transitioning or s.mode in (6, 7)
                else nes_idle_action()
            )
            continue
        x, y = s.link_x, s.link_y
        if (x, y) == last:
            stuck += 1
        else:
            stuck, last = 0, (x, y)
        if stuck > 14:
            env.step(nes_action(("DOWN", "RIGHT", "DOWN", "LEFT", "UP")[stuck % 5]))
            continue
        if abs(x - 120) > 2:
            if y < 175:
                env.step(nes_action("DOWN"))
            else:
                env.step(nes_action("RIGHT" if x < 120 else "LEFT"))
            continue
        env.step(nes_action("UP"))
    return read_snapshot(env.get_ram()).screen == dest


def enter_left(env: Any, dest: int, *, budget: int = 700) -> bool:
    for _ in range(budget):
        s = read_snapshot(env.get_ram())
        if s.screen == dest and s.mode == PLAY_MODE:
            return True
        if s.mode != PLAY_MODE:
            env.step(
                nes_action("LEFT")
                if s.transitioning or s.mode in (6, 7)
                else nes_idle_action()
            )
            continue
        if abs(s.link_y - L2_BOSS_EXIT_DOOR_Y) > 4:
            env.step(
                nes_action("DOWN" if s.link_y < L2_BOSS_EXIT_DOOR_Y else "UP")
            )
        else:
            env.step(nes_action("LEFT"))
    return read_snapshot(env.get_ram()).screen == dest


def wait_types(
    env: Any, types: tuple[int, ...], *, n: int = 1, budget: int = 200
) -> int:
    c = 0
    for _ in range(budget):
        live = live_objects(read_snapshot(env.get_ram()), frozenset(types))
        c = len(live)
        if c >= n:
            return c
        env.step(nes_idle_action())
    return c


def clear_types(
    env: Any,
    types: tuple[int, ...],
    *,
    max_frames: int = 14000,
    min_n: int = 1,
) -> dict[str, Any]:
    """Generic clear via GenericDungeonRoomController + stuck recovery."""
    wait_types(env, types, n=min_n, budget=200)
    snap = read_snapshot(env.get_ram())
    live = live_objects(snap, frozenset(types))
    if not live:
        return {"success": True, "already_clear": True, "frames": 0}
    rule = AliveRule.TYPE if set(types) <= TYPE_ONLY else AliveRule.TYPE_AND_HP
    spec = DungeonRoomSpec(
        spec_id=f"clear_0x{snap.screen:02x}",
        source_room=snap.screen,
        room_id=snap.screen,
        entry=DoorRoute("UP", ((120, 141),)),
        enemy_types=types,
        expected_enemy_count=max(min_n, len(live)),
        alive_rule=rule,
        combat=CombatTuning(
            patrol=_WIDE_PATROL,
            engage_distance=80,
            attack_phase=2,
            patrol_attack_period=6,
            patrol_attack_hold=3,
            engage_attack_period=4,
            engage_attack_hold=3,
        ),
        reward=RewardSpec(kind=RewardKind.CLEAR_ONLY),
        max_frames=max_frames,
        level=LEVEL_2,
    )
    ctrl = GenericDungeonRoomController(spec)
    ctrl.phase = DungeonPhase.FIGHT
    f = 0
    last_xy = (-1, -1)
    stuck = 0
    for f in range(max_frames):
        s = read_snapshot(env.get_ram())
        live_n = len(live_objects(s, frozenset(types)))
        if live_n == 0 and s.room_all_dead >= 10:
            return {
                **ctrl.report(),
                "frames": f + 1,
                "success": True,
                "notes": list(ctrl.notes) + ["zero_live"],
            }
        xy = (s.link_x, s.link_y)
        if xy == last_xy:
            stuck += 1
        else:
            stuck, last_xy = 0, xy
        if stuck > 40:
            ensure_bomb_selected(env)
            env.step(nes_action(("DOWN", "LEFT", "RIGHT", "UP")[stuck % 4], "A"))
            if stuck % 50 == 49 and s.bombs > 0:
                env.step(nes_action("UP", "B"))
            continue
        env.step(ctrl.step(s).action)
        if ctrl.success or ctrl.phase in (DungeonPhase.FAILED, DungeonPhase.DONE):
            if live_n == 0:
                break
            ctrl.phase = DungeonPhase.FIGHT
            ctrl.success = False
    return {**ctrl.report(), "frames": f + 1}

class BombNorth1EPhase(Enum):
    """Phase machine for 0x1e bomb-north → Dodongo 0x0e."""

    SETTLE = auto()
    SOUTH_BAND = auto()
    TO_STAND = auto()
    FACE = auto()
    PLACE = auto()
    WAIT = auto()
    PUSH = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class Level2BombNorth1EController:
    """From cleared 0x1e: bomb north wall @ catalog stand → enter 0x0e.

    Uses ``BOMB_WALL_1E_NORTH`` stand (120, 101). Walk-UP after Goriya clear
    is solid (doors bit UP|DOWN=12 is a red herring).
    """

    phase: BombNorth1EPhase = BombNorth1EPhase.SETTLE
    frames: int = 0
    phase_frames: int = 0
    success: bool = False
    notes: list[str] = field(default_factory=list)
    bombs_before_place: int | None = None
    bombs_after_place: int | None = None
    max_frames: int = BOMB_1E_MAX_FRAMES
    stand: tuple[int, int] = BOMB_STAND_1E
    stand_tol: int = BOMB_N_STAND_TOL
    wait_blast: int = BOMB_N_WAIT_BLAST
    wall: BombWall = BOMB_WALL_1E

    def _set_phase(self, phase: BombNorth1EPhase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            if note:
                self.notes.append(note)

    def _fail(self, note: str) -> FrameAction:
        self._set_phase(BombNorth1EPhase.FAILED, note)
        return FrameAction(nes_idle_action(), note)

    def _at_stand(self, snap: ZeldaSnapshot) -> bool:
        return is_at_bomb_stand(
            snap.link_x, snap.link_y, self.wall, tol=self.stand_tol
        )

    def _goto_stand(self, snap: ZeldaSnapshot) -> FrameAction:
        tx, ty = self.stand
        dx = tx - snap.link_x
        dy = ty - snap.link_y
        if abs(snap.link_y - ty) <= 12 and abs(dx) > self.stand_tol:
            if abs(dy) > self.stand_tol:
                return FrameAction(
                    nes_action("UP" if dy < 0 else "DOWN"), "stand_band_y"
                )
            return FrameAction(
                nes_action("RIGHT" if dx > 0 else "LEFT"), "stand_band_x"
            )
        if abs(dx) > self.stand_tol and abs(dx) >= abs(dy):
            return FrameAction(
                nes_action("RIGHT" if dx > 0 else "LEFT"), "stand_x"
            )
        if abs(dy) > self.stand_tol:
            return FrameAction(
                nes_action("UP" if dy < 0 else "DOWN"), "stand_y"
            )
        return FrameAction(nes_idle_action(), "stand_ready")

    def _push_north(self, snap: ZeldaSnapshot) -> FrameAction:
        cx = self.stand[0]
        x_tol = 3 if snap.link_y <= 110 else 6
        if abs(snap.link_x - cx) > x_tol:
            return FrameAction(
                nes_action("RIGHT" if snap.link_x < cx else "LEFT"),
                "push_align_x",
            )
        return FrameAction(nes_action("UP"), "push_north")

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.phase_frames += 1

        if self.phase is BombNorth1EPhase.DONE:
            return FrameAction(nes_idle_action(), "done")
        if self.phase is BombNorth1EPhase.FAILED:
            return FrameAction(nes_idle_action(), "failed")

        if snap.mode == 17:
            return self._fail("link_death")
        if self.frames >= self.max_frames:
            return self._fail("timeout")

        if (
            snap.level == LEVEL_2
            and snap.screen == ROOM_0E
            and snap.mode == PLAY_MODE
        ):
            self.success = True
            self._set_phase(BombNorth1EPhase.DONE, "entered_0x0e")
            return FrameAction(nes_idle_action(), "done")

        if snap.level != LEVEL_2:
            return FrameAction(nes_idle_action(), f"wait_level_{LEVEL_2}")

        if snap.transitioning or snap.mode in (4, 6, 7, 16):
            if self.phase is BombNorth1EPhase.PUSH or snap.screen == ROOM_0E:
                return FrameAction(nes_action("UP"), "scroll_north")
            return FrameAction(nes_idle_action(), f"settle_mode_{snap.mode}")

        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")

        if self.phase is BombNorth1EPhase.SETTLE:
            if snap.screen != ROOM_1E:
                return self._fail(f"wrong_room_0x{snap.screen:02x}")
            if snap.bombs <= 0:
                return self._fail("no_bombs")
            # South-band detour first (north pockets can pin stand approach).
            self._set_phase(BombNorth1EPhase.SOUTH_BAND, "south_band_first")
            return FrameAction(nes_action("DOWN"), "south_band")

        if self.phase is BombNorth1EPhase.SOUTH_BAND:
            if snap.link_y >= 170 or self.phase_frames > 80:
                self._set_phase(BombNorth1EPhase.TO_STAND, "to_bomb_stand")
                return self._goto_stand(snap)
            return FrameAction(nes_action("DOWN"), "south_band")

        if self.phase is BombNorth1EPhase.TO_STAND:
            if self._at_stand(snap):
                self._set_phase(BombNorth1EPhase.FACE, "at_bomb_stand")
                return FrameAction(nes_action("UP"), "face_up")
            if self.phase_frames > 2500:
                return self._fail("stand_timeout")
            return self._goto_stand(snap)

        if self.phase is BombNorth1EPhase.FACE:
            if self.phase_frames < 6:
                return FrameAction(nes_action("UP"), "face_up")
            self._set_phase(BombNorth1EPhase.PLACE, "faced_up")
            self.bombs_before_place = int(snap.bombs)
            return FrameAction(nes_action("UP", "B"), "place_bomb")

        if self.phase is BombNorth1EPhase.PLACE:
            self.bombs_after_place = int(snap.bombs)
            if (
                self.bombs_before_place is not None
                and self.bombs_after_place < self.bombs_before_place
            ):
                self.notes.append(
                    f"bomb_used_{self.bombs_before_place}->{self.bombs_after_place}"
                )
            self._set_phase(BombNorth1EPhase.WAIT, "placed_bomb")
            return FrameAction(nes_action("UP"), "wait_blast")

        if self.phase is BombNorth1EPhase.WAIT:
            if self.phase_frames < self.wait_blast:
                return FrameAction(nes_action("UP"), "wait_blast")
            self._set_phase(BombNorth1EPhase.PUSH, "blast_done")
            return self._push_north(snap)

        if self.phase is BombNorth1EPhase.PUSH:
            if self.phase_frames > 700:
                return self._fail("push_timeout")
            return self._push_north(snap)

        return FrameAction(nes_idle_action(), "idle")

    def report(self) -> dict[str, Any]:
        return {
            "phase": self.phase.name,
            "frames": self.frames,
            "success": self.success,
            "notes": list(self.notes),
            "stand": list(self.stand),
            "wall_room": f"0x{self.wall.room:02x}",
            "opens_to": f"0x{self.wall.opens_to:02x}",
            "bombs_before_place": self.bombs_before_place,
            "bombs_after_place": self.bombs_after_place,
        }


def make_bomb_north_1e_controller() -> Level2BombNorth1EController:
    """Factory for isolated 0x1e → 0x0e bomb-north segment."""
    return Level2BombNorth1EController()


def bomb_north_1e_wall(env: Any, *, dest: int = ROOM_0E) -> dict[str, Any]:
    """Run bomb-N 0x1e controller until enter dest or fail."""
    ctrl = make_bomb_north_1e_controller()
    s0 = read_snapshot(env.get_ram())
    b0, sc0 = s0.bombs, s0.screen
    ensure_bomb_selected(env)
    for _ in range(ctrl.max_frames):
        s = read_snapshot(env.get_ram())
        env.step(ctrl.step(s).action)
        if ctrl.success or ctrl.phase is BombNorth1EPhase.FAILED:
            break
    s = read_snapshot(env.get_ram())
    return {
        "ok": s.screen == dest,
        "bombs": f"{b0}->{s.bombs}",
        "from": f"0x{sc0:02x}",
        "to": f"0x{s.screen:02x}",
        "xy": [s.link_x, s.link_y],
        "stand": list(BOMB_STAND_1E),
        "controller": ctrl.report(),
    }

def fight_dodongo(
    env: Any,
    assist: Any | None = None,
    *,
    max_frames: int = DODONGO_FIGHT_MAX_FRAMES,
    apply_assist: Callable[[Any, int], None] | None = None,
) -> dict[str, Any]:
    """Bomb-in-mouth policy for type 0x32 Dodongo.

    Walkthrough: drop bomb nearly in mouth; 2 successful mouths kill.
    Assisted bomb top-up OK. Not Clean STATUS.
    """
    log: list[dict[str, Any]] = []
    bombs_used = 0
    hits_est = 0
    last_hp = None
    place_cd = 0
    poke_notes = [poke_bombs(env, 16)]
    ensure_bomb_selected(env)
    f = 0
    for f in range(max_frames):
        if assist is not None and f % 15 == 0:
            if apply_assist is not None:
                apply_assist(env, f)
            else:
                assist.apply_env(env, frame=f)
        s = read_snapshot(env.get_ram())
        if s.bombs < 2 and assist is not None:
            poke_bombs(env, 12)
            ensure_bomb_selected(env)
        if s.mode != PLAY_MODE:
            env.step(nes_idle_action())
            continue
        if triforce_bit_02(read_u8(env.get_ram(), ADDR_TRIFORCE)):
            log.append(sample_snapshot(s, env.get_ram(), event="tf_mid_fight"))
            break
        dodos = [
            o
            for o in s.objects
            if o.type_id == DODONGO_TYPE and 1 <= o.slot <= 10
        ]
        living = [o for o in dodos if o.hp > 0]
        if not living and not dodos and s.room_all_dead >= 20:
            log.append(sample_snapshot(s, env.get_ram(), event="dodongo_dead"))
            break
        if not living:
            env.step(nes_action(("UP", "RIGHT", "DOWN", "LEFT")[f // 20 % 4], "A"))
            if f > 200 and s.room_all_dead >= 20 and not dodos:
                log.append(
                    sample_snapshot(s, env.get_ram(), event="dodongo_dead_settle")
                )
                break
            continue

        d = living[0]
        if last_hp is not None and d.hp < last_hp:
            hits_est += 1
            log.append(
                {
                    "event": "hp_drop",
                    "hp": f"{last_hp}->{d.hp}",
                    "f": f,
                    "xy": [d.x, d.y],
                    "facing": d.facing,
                }
            )
        last_hp = d.hp

        tx, ty, face = mouth_target(d)
        tx = max(48, min(192, tx))
        ty = max(105, min(185, ty))
        dist = abs(s.link_x - d.x) + abs(s.link_y - d.y)
        at_mouth = abs(s.link_x - tx) <= 12 and abs(s.link_y - ty) <= 12

        if place_cd > 0:
            place_cd -= 1
            if place_cd > 50:
                retreat = {
                    "UP": "DOWN",
                    "DOWN": "UP",
                    "LEFT": "RIGHT",
                    "RIGHT": "LEFT",
                }.get(face, "DOWN")
                env.step(nes_action(retreat))
            elif place_cd > 20:
                env.step(nes_action(face, "A"))
            else:
                env.step(nes_idle_action())
            continue

        if (at_mouth or dist <= 24) and s.bombs > 0:
            ensure_bomb_selected(env)
            if dist > 14:
                act, _ = goto_action(s, d.x, d.y, tol=8)
                env.step(act)
                continue
            env.step(nes_action(face))
            env.step(nes_action(face, "B"))
            bombs_used += 1
            place_cd = 95
            if bombs_used <= 8 or bombs_used % 4 == 0:
                log.append(
                    sample_snapshot(s, env.get_ram(), event=f"placed_f{f}")
                    | {
                        "face": face,
                        "target": [tx, ty],
                        "dodo": [d.x, d.y, d.facing, d.hp],
                        "dist": dist,
                    }
                )
            continue

        act, _ = goto_action(s, tx, ty, tol=6)
        env.step(act)

    s = read_snapshot(env.get_ram())
    alive = [o for o in s.objects if o.type_id == DODONGO_TYPE and o.hp > 0]
    return {
        "success": len(alive) == 0,
        "frames": f + 1,
        "bombs_used_est": bombs_used,
        "hits_est": hits_est,
        "poke_notes": poke_notes,
        "final": sample_snapshot(s, env.get_ram(), event="fight_end"),
        "log": log[-40:],
    }


__all__ = [
    "ADDR_SELECTED_ITEM",
    "B_ITEM_BOMB",
    "BOMB_1E_MAX_FRAMES",
    "BOMB_STAND_1E",
    "BOMB_WALL_1E",
    "BombNorth1EPhase",
    "DODONGO_FIGHT_MAX_FRAMES",
    "DODONGO_TYPE",
    "FACE_E",
    "FACE_N",
    "FACE_S",
    "FACE_W",
    "IGNORE_TYPES",
    "KEESE_TYPE",
    "Level2BombNorth1EController",
    "TYPE_ONLY",
    "bomb_north_1e_wall",
    "clear_types",
    "ensure_bomb_selected",
    "enter_left",
    "enter_up",
    "fight_dodongo",
    "goto_action",
    "idle",
    "live_objects",
    "make_bomb_north_1e_controller",
    "mouth_target",
    "poke_bombs",
    "sample_snapshot",
    "triforce_bit_02",
    "wait_types",
]
