"""Assisted: Level2Boom → Dodongo 0x0e bomb-mouth → triforce bit 0x02 (rr-n5i).

Path (live 2026-08-07):
  0x4f bomb-N → 0x3f → LEFT Moldorm 0x3e → UP ropes 0x2e clear → UP Goriya 0x1e
  clear → **bomb-N @(120,101)** → boss **0x0e** (type 0x32 Dodongo)
  → 2× bomb-in-mouth → Heart → RIGHT TF room → ADDR_TRIFORCE & 0x02

Walk-UP on 0x1e after clear is **solid** (doors bit UP|DOWN=12 is a red herring;
physical open is bomb wall). Prefer ``--infinite-life`` first pass; not Clean STATUS.

Examples::

    uv run python nes/zelda_i/scripts/run_level2_dodongo.py --infinite-life --trials 1
    uv run python nes/zelda_i/scripts/run_level2_dodongo.py --infinite-life --save-state
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_NES = _REPO_ROOT / "nes"
for _p in (_REPO_ROOT, _NES):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from retro_harness.env import make_env, save_state
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon import (
    AliveRule,
    CombatTuning,
    DoorRoute,
    DungeonPhase,
    DungeonRoomSpec,
    GenericDungeonRoomController,
    GORIYA_OBJECT_TYPE,
    MOLDORM_OBJECT_TYPE,
    RewardKind,
    RewardSpec,
    ROPE_OBJECT_TYPE,
)
from zelda_i.dungeon_trace import write_state_provenance
from zelda_i.level2_dungeon import (
    Level2PostBoomBombNorthController,
    PostBoomBombNorthPhase,
    ROOM_1E_SPEC,
    ROOM_2E_SPEC,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_TRIFORCE,
    PLAY_MODE,
    ZeldaSnapshot,
    read_snapshot,
    read_u8,
)

# Live 2026-08-07 (probe_level2_1e_up)
ROOM_3F = 0x3F
ROOM_3E = 0x3E
ROOM_2E = 0x2E
ROOM_1E = 0x1E
ROOM_0E = 0x0E  # Dodongo boss
ROOM_TF = 0x0D  # expected east of boss (walkthrough); verified live if open
DODONGO_TYPE = 0x32
KEESE = 0x1B
BOMB_STAND_1E = (120, 101)
ADDR_SELECTED_ITEM = 0x0656
B_ITEM_BOMB = 0x02

# Facing bits (ADDR_LINK_FACING / obj facing)
FACE_E, FACE_W, FACE_S, FACE_N = 0x01, 0x02, 0x04, 0x08

IGNORE = frozenset({0x55, 0x49, 0x4E, 0x5C, 0x4A, 0x60, 0x61, 0x62, 0x63})
TYPE_ONLY = frozenset({0x15, 0x1B, 0x41})


def _idle(env, n: int = 1) -> None:
    for _ in range(n):
        env.step(nes_idle_action())


def _ensure_bomb(env) -> None:
    try:
        mem = env.unwrapped.data.memory
        if hasattr(mem, "set_byte"):
            mem.set_byte(ADDR_SELECTED_ITEM, B_ITEM_BOMB)
            return
    except Exception:
        pass
    try:
        env.unwrapped.data.set_value("selected_item", B_ITEM_BOMB)
    except Exception:
        pass


def _live(snap: ZeldaSnapshot, types: frozenset[int] | None = None) -> list:
    out = []
    for o in snap.objects:
        if not (1 <= o.slot <= 10):
            continue
        if o.type_id in (0, 0xFF) or o.type_id in IGNORE:
            continue
        if types is not None and o.type_id not in types:
            continue
        if o.type_id in TYPE_ONLY or o.hp > 0:
            out.append(o)
    return out


def _sample(snap: ZeldaSnapshot, ram, *, event: str) -> dict:
    live = _live(snap)
    types = Counter(o.type_id for o in live)
    dodos = [o for o in snap.objects if o.type_id == DODONGO_TYPE and 1 <= o.slot <= 10]
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
        "tf02": bool(read_u8(ram, ADDR_TRIFORCE) & 0x02),
    }


def _goto(snap: ZeldaSnapshot, tx: int, ty: int, tol: int = 6):
    if abs(snap.link_x - tx) > tol:
        return nes_action("RIGHT" if snap.link_x < tx else "LEFT"), False
    if abs(snap.link_y - ty) > tol:
        return nes_action("DOWN" if snap.link_y < ty else "UP"), False
    return nes_idle_action(), True


def _enter_up(env, dest: int, *, budget: int = 1600) -> bool:
    """UP door on L2 0x2e: get to x=120 via south band, pure UP.

    Live traps (2026-08-07):
    - Diamond solids block some mid-y laterals (LEFT@y141 x≤96 dead).
    - South band y≈189 free for lateral; DOWN from mid usually open.
    - Door needs |x−120|≤2 (x=96 reaches y=93 but stays on screen).
    """
    # Phase 0: hold DOWN to south band (escape north pockets)
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
    for _i in range(budget):
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
            # Prefer south band for lateral; if already south, slide
            if y < 175:
                env.step(nes_action("DOWN"))
            else:
                env.step(nes_action("RIGHT" if x < 120 else "LEFT"))
            continue
        # x aligned — climb
        env.step(nes_action("UP"))
    return read_snapshot(env.get_ram()).screen == dest


def _enter_left(env, dest: int, *, budget: int = 700) -> bool:
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
        if abs(s.link_y - 141) > 4:
            env.step(nes_action("DOWN" if s.link_y < 141 else "UP"))
        else:
            env.step(nes_action("LEFT"))
    return read_snapshot(env.get_ram()).screen == dest


def _enter_right(env, dest: int | None = None, *, budget: int = 700) -> bool:
    start = read_snapshot(env.get_ram()).screen
    for _ in range(budget):
        s = read_snapshot(env.get_ram())
        if dest is not None and s.screen == dest and s.mode == PLAY_MODE:
            return True
        if dest is None and s.screen != start and s.mode == PLAY_MODE:
            return True
        if s.mode != PLAY_MODE:
            env.step(
                nes_action("RIGHT")
                if s.transitioning or s.mode in (6, 7)
                else nes_idle_action()
            )
            continue
        if abs(s.link_y - 141) > 4:
            env.step(nes_action("DOWN" if s.link_y < 141 else "UP"))
        else:
            env.step(nes_action("RIGHT"))
    s = read_snapshot(env.get_ram())
    if dest is None:
        return s.screen != start
    return s.screen == dest


def _wait_types(env, types: tuple[int, ...], *, n: int = 1, budget: int = 200) -> int:
    c = 0
    for _ in range(budget):
        live = _live(read_snapshot(env.get_ram()), frozenset(types))
        c = len(live)
        if c >= n:
            return c
        env.step(nes_idle_action())
    return c


def _clear_types(env, types: tuple[int, ...], *, max_frames: int = 14000, min_n: int = 1) -> dict:
    _wait_types(env, types, n=min_n, budget=200)
    snap = read_snapshot(env.get_ram())
    live = _live(snap, frozenset(types))
    if not live:
        return {"success": True, "already_clear": True, "frames": 0}
    rule = AliveRule.TYPE if set(types) <= TYPE_ONLY else AliveRule.TYPE_AND_HP
    # Wider patrol + south band so L2 diamond pockets don't pin Link north.
    patrol = (
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
    spec = DungeonRoomSpec(
        spec_id=f"clear_0x{snap.screen:02x}",
        source_room=snap.screen,
        room_id=snap.screen,
        entry=DoorRoute("UP", ((120, 141),)),
        enemy_types=types,
        expected_enemy_count=max(min_n, len(live)),
        alive_rule=rule,
        combat=CombatTuning(
            patrol=patrol,
            engage_distance=80,
            attack_phase=2,
            patrol_attack_period=6,
            patrol_attack_hold=3,
            engage_attack_period=4,
            engage_attack_hold=3,
        ),
        reward=RewardSpec(kind=RewardKind.CLEAR_ONLY),
        max_frames=max_frames,
        level=2,
    )
    ctrl = GenericDungeonRoomController(spec)
    ctrl.phase = DungeonPhase.FIGHT
    f = 0
    last_xy = (-1, -1)
    stuck = 0
    for f in range(max_frames):
        s = read_snapshot(env.get_ram())
        live_n = len(_live(s, frozenset(types)))
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
            # unstick + bomb residual (ropes die to bombs)
            _ensure_bomb(env)
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


def _bomb_north_wall(env, stand=(120, 101), dest: int = ROOM_0E) -> dict:
    sx, sy = stand
    s0 = read_snapshot(env.get_ram())
    b0, sc0 = s0.bombs, s0.screen
    # walk to bomb stand (south-band detour then north stand)
    for _ in range(80):
        s = read_snapshot(env.get_ram())
        if s.link_y >= 170:
            break
        env.step(nes_action("DOWN"))
    for _ in range(200):
        s = read_snapshot(env.get_ram())
        if abs(s.link_x - 120) <= 4:
            break
        env.step(nes_action("RIGHT" if s.link_x < 120 else "LEFT"))
    for _ in range(350):
        s = read_snapshot(env.get_ram())
        act, at = _goto(s, sx, sy, tol=3)
        env.step(act)
        if at:
            break
    _ensure_bomb(env)
    for _ in range(5):
        env.step(nes_action("UP"))
    env.step(nes_action("UP", "B"))
    _idle(env, 2)
    for _ in range(90):
        env.step(nes_action("UP"))
    for _ in range(250):
        s = read_snapshot(env.get_ram())
        if s.screen == dest and s.mode == PLAY_MODE:
            break
        if s.mode != PLAY_MODE:
            env.step(
                nes_action("UP")
                if s.transitioning or s.mode in (6, 7)
                else nes_idle_action()
            )
            continue
        if abs(s.link_x - 120) > 2:
            env.step(nes_action("RIGHT" if s.link_x < 120 else "LEFT"))
        else:
            env.step(nes_action("UP"))
    s = read_snapshot(env.get_ram())
    return {
        "ok": s.screen == dest,
        "bombs": f"{b0}->{s.bombs}",
        "from": f"0x{sc0:02x}",
        "to": f"0x{s.screen:02x}",
        "xy": [s.link_x, s.link_y],
        "stand": list(stand),
    }


def _poke_bombs(env, n: int = 16) -> str:
    """Assisted recon: top up bombs (not Clean)."""
    try:
        env.unwrapped.data.set_value("bombs", int(n) & 0xFF)
        return f"bombs={n}"
    except Exception as exc:
        return f"poke_fail={exc!r}"


def _mouth_target(dodo) -> tuple[int, int, str]:
    """Stand in front of snout; return (x, y, face_when_placing)."""
    f = int(dodo.facing)
    # Place bomb ~12px in front of body center toward facing
    if f & FACE_E:
        return dodo.x + 12, dodo.y, "LEFT"  # Link W of snout, face E into mouth
    if f & FACE_W:
        return dodo.x - 12, dodo.y, "RIGHT"
    if f & FACE_S:
        return dodo.x, dodo.y + 12, "UP"
    if f & FACE_N:
        return dodo.x, dodo.y - 12, "DOWN"
    # unknown facing: stand on body and face any
    return dodo.x, dodo.y, "UP"


def _fight_dodongo(env, assist: UnlimitedHealthAssist | None, *, max_frames: int = 12000) -> dict:
    """Bomb-in-mouth / path bomb policy (Zelda Dungeon L2).

    Walkthrough: get close, drop bomb nearly in mouth while Dodongo walks at you;
    2 successful mouths kill. Also try side bomb + sword. Assisted bomb top-up OK.
    """
    log = []
    bombs_used = 0
    hits_est = 0
    last_hp = None
    place_cd = 0
    poke_notes = [_poke_bombs(env, 16)]
    _ensure_bomb(env)
    for f in range(max_frames):
        if assist is not None and f % 15 == 0:
            assist.apply_env(env, frame=f)
        # top up if dry (assisted)
        s = read_snapshot(env.get_ram())
        if s.bombs < 2 and assist is not None:
            _poke_bombs(env, 12)
            _ensure_bomb(env)
        if s.mode != PLAY_MODE:
            env.step(nes_idle_action())
            continue
        if read_u8(env.get_ram(), ADDR_TRIFORCE) & 0x02:
            log.append(_sample(s, env.get_ram(), event="tf_mid_fight"))
            break
        dodos = [
            o
            for o in s.objects
            if o.type_id == DODONGO_TYPE and 1 <= o.slot <= 10
        ]
        # Require sustained clear: type gone AND room_all_dead≥20 (explosion
        # briefly blanks type while HP still mid-fight — do not early-exit).
        living = [o for o in dodos if o.hp > 0]
        if not living and not dodos and s.room_all_dead >= 20:
            log.append(_sample(s, env.get_ram(), event="dodongo_dead"))
            break
        if not living:
            # stunned / exploding — keep swording and wait, do not leave yet
            env.step(nes_action(("UP", "RIGHT", "DOWN", "LEFT")[f // 20 % 4], "A"))
            if f > 200 and s.room_all_dead >= 20 and not dodos:
                log.append(_sample(s, env.get_ram(), event="dodongo_dead_settle"))
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

        tx, ty, face = _mouth_target(d)
        tx = max(48, min(192, tx))
        ty = max(105, min(185, ty))
        dist = abs(s.link_x - d.x) + abs(s.link_y - d.y)
        at_mouth = abs(s.link_x - tx) <= 12 and abs(s.link_y - ty) <= 12

        if place_cd > 0:
            place_cd -= 1
            # after place: step back then sword (side-bomb+sword method)
            if place_cd > 50:
                retreat = {
                    "UP": "DOWN",
                    "DOWN": "UP",
                    "LEFT": "RIGHT",
                    "RIGHT": "LEFT",
                }.get(face, "DOWN")
                env.step(nes_action(retreat))
            elif place_cd > 20:
                env.step(nes_action(face, "A"))  # sword if stunned
            else:
                env.step(nes_idle_action())
            continue

        # Drop bomb when adjacent / on mouth line
        if (at_mouth or dist <= 24) and s.bombs > 0:
            _ensure_bomb(env)
            # Prefer placing ON the dodongo body: walk onto it then B
            if dist > 14:
                act, _ = _goto(s, d.x, d.y, tol=8)
                env.step(act)
                continue
            env.step(nes_action(face))
            env.step(nes_action(face, "B"))
            bombs_used += 1
            place_cd = 95
            if bombs_used <= 8 or bombs_used % 4 == 0:
                log.append(
                    _sample(s, env.get_ram(), event=f"placed_f{f}")
                    | {
                        "face": face,
                        "target": [tx, ty],
                        "dodo": [d.x, d.y, d.facing, d.hp],
                        "dist": dist,
                    }
                )
            continue

        # Intercept path: stand ahead of facing vector
        act, _ = _goto(s, tx, ty, tol=6)
        env.step(act)

    s = read_snapshot(env.get_ram())
    alive = [o for o in s.objects if o.type_id == DODONGO_TYPE and o.hp > 0]
    return {
        "success": len(alive) == 0,
        "frames": f + 1,
        "bombs_used_est": bombs_used,
        "hits_est": hits_est,
        "poke_notes": poke_notes,
        "final": _sample(s, env.get_ram(), event="fight_end"),
        "log": log[-40:],
    }


def _collect_and_tf(env, assist: UnlimitedHealthAssist | None, *, budget: int = 4000) -> dict:
    """After Dodongo: collect heart, leave via open door (live: LEFT bit=2), TF center.

    Live 2026-08-07: post-kill doors=LEFT only (0x02), heart RoomItemId 0x1A on
    0x0e; walkthrough "east" is residual vs live LEFT→0x0d candidate.
    """
    log = []
    heart_touched = False
    for f in range(budget):
        if assist is not None and f % 20 == 0:
            assist.apply_env(env, frame=20000 + f)
        s = read_snapshot(env.get_ram())
        tf = read_u8(env.get_ram(), ADDR_TRIFORCE)
        if tf & 0x02:
            log.append(_sample(s, env.get_ram(), event="tf_got"))
            return {"ok": True, "frames": f + 1, "log": log, "final": log[-1]}
        if s.mode != PLAY_MODE:
            # triforce fanfare / mode 18 settle
            env.step(nes_idle_action())
            continue

        if s.screen == ROOM_0E:
            doors = s.cur_opened_doors
            # 1) Touch heart at center if still present
            if not heart_touched and f < 400:
                act, at = _goto(s, 120, 141, tol=8)
                env.step(act)
                if at:
                    heart_touched = True
                if f % 80 == 0:
                    log.append(_sample(s, env.get_ram(), event=f"heart_f{f}"))
                continue
            # 2) Prefer open door bits: LEFT=2 first (live), then RIGHT=1, UP, DOWN
            if doors & 0x02:  # LEFT
                if abs(s.link_y - 141) > 4:
                    env.step(nes_action("DOWN" if s.link_y < 141 else "UP"))
                else:
                    env.step(nes_action("LEFT"))
            elif doors & 0x01:  # RIGHT
                if abs(s.link_y - 141) > 4:
                    env.step(nes_action("DOWN" if s.link_y < 141 else "UP"))
                else:
                    env.step(nes_action("RIGHT"))
            elif doors & 0x08:  # UP
                if abs(s.link_x - 120) > 2:
                    env.step(nes_action("RIGHT" if s.link_x < 120 else "LEFT"))
                else:
                    env.step(nes_action("UP"))
            else:
                # try RIGHT then LEFT brute even if bit lag
                if f % 200 < 100:
                    env.step(nes_action("RIGHT") if abs(s.link_y - 141) <= 4 else (
                        nes_action("DOWN" if s.link_y < 141 else "UP")
                    ))
                else:
                    env.step(nes_action("LEFT") if abs(s.link_y - 141) <= 4 else (
                        nes_action("DOWN" if s.link_y < 141 else "UP")
                    ))
            if f % 100 == 0:
                log.append(_sample(s, env.get_ram(), event=f"boss_exit_f{f}"))
            continue

        # TF room (live 0x0d via LEFT from boss): free from east entry alcove
        # then stand on pedestal (center). Stuck at x≈208 needs UP/DOWN free.
        x, y = s.link_x, s.link_y
        if abs(x - 120) > 8 or abs(y - 141) > 8:
            if x >= 190 and abs(y - 141) <= 6:
                # east door alcove — step vertically free then LEFT
                env.step(nes_action("UP" if f % 40 < 20 else "DOWN"))
            elif abs(x - 120) > 8:
                env.step(nes_action("RIGHT" if x < 120 else "LEFT"))
            else:
                env.step(nes_action("DOWN" if y < 141 else "UP"))
        else:
            env.step(nes_idle_action())
        if f % 60 == 0:
            log.append(_sample(s, env.get_ram(), event=f"tf_room_f{f}"))
    s = read_snapshot(env.get_ram())
    return {
        "ok": bool(read_u8(env.get_ram(), ADDR_TRIFORCE) & 0x02),
        "frames": budget,
        "log": log[-30:],
        "final": _sample(s, env.get_ram(), event="tf_fail"),
    }


def run_once(
    *,
    start_state: str = "Level2Boom",
    infinite_life: bool = True,
    tag: str = "level2_dodongo",
    save_checkpoint: bool = False,
) -> dict:
    configure_headless()
    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True) if infinite_life else None
    track = "assisted" if infinite_life else "clean"
    timeline: list = []
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        env.reset()
        obs, *_ = env.step(nes_idle_action())
        if assist is not None:
            assist.apply_env(env, frame=0)
        timeline.append(_sample(read_snapshot(env.get_ram()), env.get_ram(), event="boot"))

        # 0x4f → 0x3f bomb
        ctrl = Level2PostBoomBombNorthController()
        for f in range(ctrl.max_frames):
            if assist is not None:
                assist.apply_env(env, frame=f)
            env.step(ctrl.step(read_snapshot(env.get_ram())).action)
            if ctrl.success or ctrl.phase is PostBoomBombNorthPhase.FAILED:
                break
        timeline.append(_sample(read_snapshot(env.get_ram()), env.get_ram(), event="3f"))
        if read_snapshot(env.get_ram()).screen != ROOM_3F:
            return _fail(env, timeline, tag, track, "no_3f")

        _clear_types(env, (KEESE,), max_frames=5000, min_n=1)
        if not _enter_left(env, ROOM_3E):
            return _fail(env, timeline, tag, track, "no_3e")
        timeline.append(_sample(read_snapshot(env.get_ram()), env.get_ram(), event="3e"))

        cr = _clear_types(env, (MOLDORM_OBJECT_TYPE,), max_frames=16000, min_n=1)
        timeline.append({"event": "moldorm", **cr})
        for _ in range(250):
            s = read_snapshot(env.get_ram())
            act, _ = _goto(s, 120, 141, 8)
            env.step(act)
        timeline.append(_sample(read_snapshot(env.get_ram()), env.get_ram(), event="moldorm_key"))

        if not _enter_up(env, ROOM_2E):
            return _fail(env, timeline, tag, track, "no_2e")
        timeline.append(_sample(read_snapshot(env.get_ram()), env.get_ram(), event="2e"))

        # ropes: robust clear (ROOM_2E_SPEC flaky when Link pins north pocket)
        cr = _clear_types(env, (ROPE_OBJECT_TYPE,), max_frames=18000, min_n=4)
        for _ in range(90):
            env.step(nes_idle_action())
        s = read_snapshot(env.get_ram())
        timeline.append(
            {
                "event": "ropes",
                **cr,
                "doors": s.cur_opened_doors,
                "ropes_left": len(_live(s, frozenset({ROPE_OBJECT_TYPE}))),
                "xy": [s.link_x, s.link_y],
            }
        )
        if not (s.cur_opened_doors & 0x08):
            # one more mop pass
            cr2 = _clear_types(env, (ROPE_OBJECT_TYPE,), max_frames=10000, min_n=1)
            for _ in range(60):
                env.step(nes_idle_action())
            s = read_snapshot(env.get_ram())
            timeline.append(
                {
                    "event": "ropes_mop2",
                    **cr2,
                    "doors": s.cur_opened_doors,
                    "ropes_left": len(_live(s, frozenset({ROPE_OBJECT_TYPE}))),
                }
            )
        if not (read_snapshot(env.get_ram()).cur_opened_doors & 0x08):
            return _fail(env, timeline, tag, track, "no_2e_up_door")

        if not _enter_up(env, ROOM_1E, budget=1600):
            return _fail(env, timeline, tag, track, "no_1e")
        timeline.append(_sample(read_snapshot(env.get_ram()), env.get_ram(), event="1e"))

        _wait_types(env, (GORIYA_OBJECT_TYPE,), n=3, budget=200)
        c1 = GenericDungeonRoomController(ROOM_1E_SPEC)
        c1.phase = DungeonPhase.FIGHT
        for f in range(ROOM_1E_SPEC.max_frames):
            if assist is not None and f % 20 == 0:
                assist.apply_env(env, frame=6000 + f)
            env.step(c1.step(read_snapshot(env.get_ram())).action)
            if c1.success or c1.phase in (DungeonPhase.FAILED, DungeonPhase.DONE):
                break
        timeline.append({"event": "goriya", **c1.report(), "frames": f + 1})
        for _ in range(50):
            env.step(nes_idle_action())
        timeline.append(_sample(read_snapshot(env.get_ram()), env.get_ram(), event="1e_cleared"))
        obs, *_ = env.step(nes_idle_action())
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_1e_cleared.png")

        # CRITICAL: bomb wall north, not walk-UP
        bomb = _bomb_north_wall(env, BOMB_STAND_1E, ROOM_0E)
        timeline.append({"event": "bomb_1e_n", **bomb})
        if not bomb.get("ok"):
            return _fail(env, timeline, tag, track, "no_boss_bomb")
        timeline.append(_sample(read_snapshot(env.get_ram()), env.get_ram(), event="0e"))
        obs, *_ = env.step(nes_idle_action())
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_boss_entry.png")

        if save_checkpoint:
            save_state(env, GAME_DIR, GAME, "Level2_0E")

        # settle spawn
        for _ in range(90):
            if assist is not None:
                assist.apply_env(env, frame=8000)
            env.step(nes_idle_action())
        timeline.append(_sample(read_snapshot(env.get_ram()), env.get_ram(), event="0e_settle"))

        fight = _fight_dodongo(env, assist, max_frames=14000)
        timeline.append({"event": "dodongo_fight", **{k: v for k, v in fight.items() if k != "log"}})
        timeline.extend(fight.get("log") or [])
        obs, *_ = env.step(nes_idle_action())
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_after_fight.png")

        tf_report = _collect_and_tf(env, assist, budget=3000)
        timeline.append({"event": "tf_phase", "ok": tf_report["ok"], "frames": tf_report["frames"]})
        timeline.extend(tf_report.get("log") or [])

        ram = env.get_ram()
        snap = read_snapshot(ram)
        tf = int(read_u8(ram, ADDR_TRIFORCE))
        ok = bool(tf & 0x02)
        obs, *_ = env.step(nes_idle_action())
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_{'ok' if ok else 'fail'}.png")

        ck = None
        prov = None
        if ok and save_checkpoint:
            ck_path = save_state(env, GAME_DIR, GAME, "Level2Complete")
            ck = str(ck_path)
            prov = str(
                write_state_provenance(
                    ck_path,
                    source_state_path=GAME_DIR
                    / "custom_integrations"
                    / GAME
                    / f"{start_state}.state",
                    request={
                        "segment": "level2_dodongo_tf02",
                        "bead": "rr-n5i",
                        "track": track,
                    },
                )
            )

        out = {
            "bead": "rr-n5i",
            "result": "TF_02" if ok else ("DODONGO_DEAD" if fight.get("success") else "PARTIAL"),
            "ok": ok,
            "track": track,
            "intervention_class": "survival" if infinite_life else "clean",
            "triforce": tf,
            "triforce_bit_0x02": ok,
            "boss_room": "0x0e",
            "dodongo_type": "0x32",
            "bomb_wall_1e": {"stand": list(BOMB_STAND_1E), "face": "UP", "to": "0x0e"},
            "fight": {k: v for k, v in fight.items() if k != "log"},
            "timeline": timeline,
            "final": _sample(snap, ram, event="final"),
            "checkpoint": ck,
            "provenance": prov,
            "natural_entry": False,
            "evidence": [
                f"recordings/{tag}.json",
                f"recordings/{tag}_boss_entry.png",
                f"recordings/{tag}_{'ok' if ok else 'fail'}.png",
            ],
        }
        write_json_report(RECORDINGS_DIR / f"{tag}.json", out)
        return out
    finally:
        env.close()


def _fail(env, timeline, tag, track, reason: str) -> dict:
    s = read_snapshot(env.get_ram())
    out = {
        "bead": "rr-n5i",
        "result": "FAIL",
        "ok": False,
        "reason": reason,
        "track": track,
        "timeline": timeline,
        "final": _sample(s, env.get_ram(), event="fail"),
        "triforce_bit_0x02": False,
    }
    write_json_report(RECORDINGS_DIR / f"{tag}.json", out)
    try:
        obs = env.render()
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_fail_{reason}.png")
    except Exception:
        pass
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--from-state", default="Level2Boom")
    p.add_argument("--infinite-life", action="store_true", default=True)
    p.add_argument("--no-infinite-life", action="store_true")
    p.add_argument("--tag", default="level2_dodongo")
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--save-state", action="store_true")
    args = p.parse_args()
    inf = not args.no_infinite_life
    results = []
    for t in range(args.trials):
        tag = args.tag if args.trials == 1 else f"{args.tag}_t{t}"
        r = run_once(
            start_state=args.from_state,
            infinite_life=inf,
            tag=tag,
            save_checkpoint=args.save_state and t == 0,
        )
        results.append(r)
        print(
            f"trial{t}: result={r.get('result')} ok={r.get('ok')} "
            f"tf={r.get('triforce')} fight={r.get('fight', {}).get('success')} "
            f"final_sc={(r.get('final') or {}).get('sc')}"
        )
    n_ok = sum(1 for r in results if r.get("ok"))
    print(f"summary: {n_ok}/{len(results)} TF 0x02")
    if args.trials > 1:
        write_json_report(
            RECORDINGS_DIR / f"{args.tag}_summary.json",
            {"ok": n_ok, "trials": len(results), "results": [r.get("result") for r in results]},
        )


if __name__ == "__main__":
    main()
