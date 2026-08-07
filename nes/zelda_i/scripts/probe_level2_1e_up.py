"""Focused residual: 0x1e Goriya clear → physical UP → Dodongo (rr-n5i).

Prior recon: after clear ``cur_opened_doors=12`` (UP|DOWN) but walk-UP stuck
near y≈117 with ``open_doorway_mask`` lagging / solid coll. Visual door open.

Strategy (assisted): chain Level2Boom → 0x3f → Moldorm → 0x2e clear → 0x1e
clear, then systematically try UP unlock policies. Does **not** claim Clean.

Examples::

    uv run python nes/zelda_i/scripts/probe_level2_1e_up.py --infinite-life
    uv run python nes/zelda_i/scripts/probe_level2_1e_up.py --infinite-life --tag l2_1e_up
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
    MOLDORM_OBJECT_TYPE,
    RewardKind,
    RewardSpec,
    ROPE_OBJECT_TYPE,
)
from zelda_i.level2_dungeon import (
    Level2PostBoomBombNorthController,
    PostBoomBombNorthPhase,
    ROOM_1E_SPEC,
    ROOM_L2_GORIYA_BOMBS,
    ROOM_L2_NORTH_OF_4E,
    ROOM_L2_ROPES_UNLOCK,
    ROOM_L2_TRAPS_KEESE,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_CUR_OPENED_DOORS,
    ADDR_OPEN_DOORWAY_MASK,
    ADDR_TRIFORCE,
    PLAY_MODE,
    ZeldaSnapshot,
    read_snapshot,
    read_u8,
)

LEVEL_2 = 2
ROOM_4F = 0x4F
ROOM_3F = ROOM_L2_TRAPS_KEESE
ROOM_3E = ROOM_L2_NORTH_OF_4E
ROOM_2E = ROOM_L2_ROPES_UNLOCK
ROOM_1E = ROOM_L2_GORIYA_BOMBS

DOOR_RIGHT = 0x01
DOOR_LEFT = 0x02
DOOR_DOWN = 0x04
DOOR_UP = 0x08

ADDR_SELECTED_ITEM = 0x0656
B_ITEM_BOMB = 0x02

DOOR_TARGETS = {
    "RIGHT": (208, 141),
    "LEFT": (32, 141),
    "UP": (120, 93),
    "DOWN": (120, 205),
}

KEESE = 0x1B
IGNORE = frozenset({0x55, 0x49, 0x4E, 0x5C, 0x4A})
DROPS = frozenset({0x60, 0x61, 0x62, 0x63})
TYPE_ONLY = frozenset({0x15, 0x1B, 0x41})


def _bits(raw: int) -> dict:
    return {
        "R": bool(raw & DOOR_RIGHT),
        "L": bool(raw & DOOR_LEFT),
        "D": bool(raw & DOOR_DOWN),
        "U": bool(raw & DOOR_UP),
        "raw": int(raw),
    }


def _live(snap: ZeldaSnapshot) -> list:
    out = []
    for o in snap.objects:
        if not (1 <= o.slot <= 10):
            continue
        if o.type_id in (0, 0xFF) or o.type_id in DROPS or o.type_id in IGNORE:
            continue
        if o.type_id in TYPE_ONLY or o.hp > 0:
            out.append(o)
    return out


def _sample(snap: ZeldaSnapshot, ram, *, event: str = "s") -> dict:
    live = _live(snap)
    types = Counter(o.type_id for o in live)
    return {
        "event": event,
        "mode": snap.mode,
        "sc": f"0x{snap.screen:02x}",
        "xy": [snap.link_x, snap.link_y],
        "facing": snap.facing,
        "keys": snap.keys,
        "bombs": snap.bombs,
        "room_all_dead": snap.room_all_dead,
        "doors": snap.cur_opened_doors,
        "doors_bits": _bits(snap.cur_opened_doors),
        "mask": snap.open_doorway_mask,
        "mask_bits": _bits(snap.open_doorway_mask),
        "live": len(live),
        "types": {f"0x{k:02x}": v for k, v in types.items()},
        "tf": int(read_u8(ram, ADDR_TRIFORCE)),
        "sel": int(read_u8(ram, ADDR_SELECTED_ITEM)),
    }


def _idle(env, n: int = 1) -> None:
    for _ in range(n):
        env.step(nes_idle_action())


def _goto_xy(snap: ZeldaSnapshot, tx: int, ty: int, tol: int = 4):
    if abs(snap.link_x - tx) > tol:
        return nes_action("RIGHT" if snap.link_x < tx else "LEFT"), False
    if abs(snap.link_y - ty) > tol:
        return nes_action("DOWN" if snap.link_y < ty else "UP"), False
    return nes_idle_action(), True


def _push_door(snap: ZeldaSnapshot, direction: str, x_tol: int = 2):
    tx, ty = DOOR_TARGETS[direction]
    if direction in ("LEFT", "RIGHT"):
        if abs(snap.link_y - ty) > 4:
            return nes_action("DOWN" if snap.link_y < ty else "UP")
        return nes_action(direction)
    if abs(snap.link_x - tx) > x_tol:
        return nes_action("RIGHT" if snap.link_x < tx else "LEFT")
    return nes_action(direction)


def _walk_to(env, tx: int, ty: int, *, budget: int = 400, tol: int = 4) -> bool:
    """Walk to (tx,ty). Prefer open mid bands: drop to y≈141/165 before long
    lateral moves (north wall at y≈117–125 traps lateral against diamond solids).
    """
    stuck = 0
    last = (-1, -1)
    for i in range(budget):
        snap = read_snapshot(env.get_ram())
        if snap.mode != PLAY_MODE:
            env.step(nes_idle_action())
            continue
        x, y = snap.link_x, snap.link_y
        if abs(x - tx) <= tol and abs(y - ty) <= tol:
            return True
        if (x, y) == last:
            stuck += 1
        else:
            stuck = 0
            last = (x, y)
        # Unstick: cycle cardinals
        if stuck > 18:
            env.step(nes_action(("UP", "RIGHT", "DOWN", "LEFT")[stuck % 4]))
            continue
        # If far in x and currently on a high/low wall band, drop/rise to open y first
        if abs(x - tx) > tol:
            if y < 130 and ty >= 130:
                env.step(nes_action("DOWN"))
                continue
            if y > 170 and ty <= 170:
                env.step(nes_action("UP"))
                continue
            # Prefer travel on y≈141 band for long lateral
            if abs(x - tx) > 16 and abs(y - 141) > 10 and 100 < y < 190:
                env.step(nes_action("DOWN" if y < 141 else "UP"))
                continue
            env.step(nes_action("RIGHT" if x < tx else "LEFT"))
            continue
        env.step(nes_action("DOWN" if y < ty else "UP"))
    return False


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


def _poke_u8(env, addr: int, val: int) -> str:
    try:
        mem = env.unwrapped.data.memory
        if hasattr(mem, "set_byte"):
            mem.set_byte(addr, int(val) & 0xFF)
            return f"set_byte(0x{addr:04x},{val})"
        if hasattr(mem, "assign"):
            mem.assign(addr, "|u1", int(val) & 0xFF)
            return f"assign(0x{addr:04x},{val})"
    except Exception as exc:
        return f"fail:{exc!r}"
    return "no_mem_api"


def _wait_enemies(
    env,
    enemy_types: tuple[int, ...],
    *,
    min_count: int = 1,
    budget: int = 180,
) -> int:
    """Idle until target types spawn (room entry lag ~100f common)."""
    n = 0
    for _ in range(budget):
        live = _live(read_snapshot(env.get_ram()))
        n = sum(1 for o in live if o.type_id in enemy_types)
        if n >= min_count:
            return n
        env.step(nes_idle_action())
    return n


def _clear_types(
    env,
    enemy_types: tuple[int, ...],
    *,
    max_frames: int = 14000,
    wait_spawn: int = 180,
    min_count: int = 1,
) -> dict:
    spawned = _wait_enemies(
        env, enemy_types, min_count=min_count, budget=wait_spawn
    )
    snap = read_snapshot(env.get_ram())
    live = _live(snap)
    if not any(o.type_id in enemy_types for o in live):
        return {
            "success": True,
            "already_clear": True,
            "frames": 0,
            "spawned": spawned,
        }
    rule = AliveRule.TYPE if set(enemy_types) <= TYPE_ONLY else AliveRule.TYPE_AND_HP
    spec = DungeonRoomSpec(
        spec_id=f"probe_clear_0x{snap.screen:02x}",
        source_room=snap.screen,
        room_id=snap.screen,
        entry=DoorRoute("UP", ((120, 141),)),
        enemy_types=enemy_types,
        expected_enemy_count=max(min_count, len(live)),
        alive_rule=rule,
        combat=CombatTuning(
            patrol=(
                (64, 109),
                (120, 109),
                (176, 109),
                (176, 141),
                (176, 173),
                (120, 173),
                (64, 173),
                (64, 141),
                (120, 141),
            ),
            engage_distance=72,
            attack_phase=2,
            patrol_attack_period=8,
            patrol_attack_hold=3,
            engage_attack_period=6,
            engage_attack_hold=3,
        ),
        reward=RewardSpec(kind=RewardKind.CLEAR_ONLY),
        max_frames=max_frames,
        level=LEVEL_2,
    )
    controller = GenericDungeonRoomController(spec)
    controller.phase = DungeonPhase.FIGHT
    frames = 0
    for frames in range(max_frames):
        action = controller.step(read_snapshot(env.get_ram()))
        env.step(action.action)
        if controller.success or controller.phase in (
            DungeonPhase.FAILED,
            DungeonPhase.DONE,
        ):
            break
    return {**controller.report(), "frames": frames + 1}


def _free_to_center(env, *, budget: int = 600) -> bool:
    """Reach ~(120,141) with stuck-wiggle (diamond solids on L2 floors)."""
    last = (-1, -1)
    stuck = 0
    for i in range(budget):
        s = read_snapshot(env.get_ram())
        if s.mode != PLAY_MODE:
            env.step(nes_idle_action())
            continue
        x, y = s.link_x, s.link_y
        if abs(x - 120) <= 4 and abs(y - 141) <= 10:
            return True
        if (x, y) == last:
            stuck += 1
        else:
            stuck = 0
            last = (x, y)
        if stuck > 12:
            # cycle including diagonals via alternating
            env.step(nes_action(("LEFT", "DOWN", "RIGHT", "DOWN", "UP", "LEFT")[stuck % 6]))
            continue
        # Prefer open mid corridor: drop from north pocket first
        if y < 130 and abs(x - 120) > 6:
            # try left/right toward center while periodically DOWN
            if stuck > 4 or i % 5 == 0:
                env.step(nes_action("DOWN"))
            else:
                env.step(nes_action("RIGHT" if x < 120 else "LEFT"))
            continue
        if abs(x - 120) > 4:
            env.step(nes_action("RIGHT" if x < 120 else "LEFT"))
            continue
        env.step(nes_action("DOWN" if y < 141 else "UP"))
    s = read_snapshot(env.get_ram())
    return abs(s.link_x - 120) <= 6 and abs(s.link_y - 141) <= 16


def _enter_dir(
    env, direction: str, dest: int, *, budget: int = 900, x_tol: int = 2
) -> bool:
    # Re-center first: clear/fight often ends in a north pocket (y≈109–117)
    # where pure lateral/DOWN scrapes diamond solids.
    _free_to_center(env, budget=min(700, max(400, budget // 2)))
    for i in range(budget):
        s = read_snapshot(env.get_ram())
        if s.screen == dest and s.mode == PLAY_MODE:
            return True
        if s.mode != PLAY_MODE:
            env.step(
                nes_action(direction)
                if s.transitioning or s.mode in (6, 7)
                else nes_idle_action()
            )
            continue
        if direction == "UP":
            # Off-x → free-center style; on-x → pure UP (verified 0x2e→0x1e).
            if abs(s.link_x - 120) > x_tol:
                if s.link_y < 140:
                    env.step(nes_action("DOWN" if i % 3 else (
                        "RIGHT" if s.link_x < 120 else "LEFT"
                    )))
                else:
                    env.step(nes_action("RIGHT" if s.link_x < 120 else "LEFT"))
                continue
            env.step(nes_action("UP"))
            continue
        env.step(_push_door(s, direction, x_tol=x_tol))
    s = read_snapshot(env.get_ram())
    return s.screen == dest and s.mode == PLAY_MODE


def _max_north_scan(env, *, x_list: list[int], push: int = 80) -> list[dict]:
    """From mid-room, try UP at each x; record min y and whether room changes."""
    results = []
    home = read_snapshot(env.get_ram())
    home_sc = home.screen
    for x in x_list:
        _walk_to(env, x, 141, budget=250)
        min_y = read_snapshot(env.get_ram()).link_y
        min_xy = [x, min_y]
        left = False
        for _ in range(push):
            s = read_snapshot(env.get_ram())
            if s.screen != home_sc:
                left = True
                min_xy = [s.link_x, s.link_y]
                break
            if abs(s.link_x - x) > 2:
                env.step(nes_action("RIGHT" if s.link_x < x else "LEFT"))
            else:
                env.step(nes_action("UP"))
            if s.link_y < min_y:
                min_y = s.link_y
                min_xy = [s.link_x, s.link_y]
        s = read_snapshot(env.get_ram())
        results.append(
            {
                "align_x": x,
                "left_room": left,
                "end_sc": f"0x{s.screen:02x}",
                "min_y": min_y,
                "min_xy": min_xy,
                "end_xy": [s.link_x, s.link_y],
                "doors": s.cur_opened_doors,
                "mask": s.open_doorway_mask,
            }
        )
        if left:
            # stay for caller
            break
        # re-center
        _walk_to(env, 120, 141, budget=200)
    return results


def _try_policy(env, name: str, fn) -> dict:
    snap0 = read_snapshot(env.get_ram())
    before = _sample(snap0, env.get_ram(), event=f"{name}_before")
    detail = fn()
    snap1 = read_snapshot(env.get_ram())
    after = _sample(snap1, env.get_ram(), event=f"{name}_after")
    ok = snap1.screen != snap0.screen and snap1.mode in (PLAY_MODE, 6, 7)
    return {
        "policy": name,
        "ok": ok,
        "before": before,
        "after": after,
        "detail": detail,
    }


def chain_to_1e(env, assist: UnlimitedHealthAssist | None, log: list) -> bool:
    """Level2Boom → bombN 0x3f → LEFT 0x3e clear+key → UP 0x2e clear → UP 0x1e."""
    # bomb north 0x4f → 0x3f
    ctrl = Level2PostBoomBombNorthController()
    for f in range(ctrl.max_frames):
        if assist is not None:
            assist.apply_env(env, frame=f)
        act = ctrl.step(read_snapshot(env.get_ram()))
        env.step(act.action)
        if ctrl.success or ctrl.phase is PostBoomBombNorthPhase.FAILED:
            break
    snap = read_snapshot(env.get_ram())
    log.append(_sample(snap, env.get_ram(), event="after_bomb_4f"))
    if snap.screen != ROOM_3F:
        return False

    # clear keese (optional) then LEFT → 0x3e
    _clear_types(env, (KEESE,), max_frames=6000)
    if not _enter_dir(env, "LEFT", ROOM_3E):
        log.append(_sample(read_snapshot(env.get_ram()), env.get_ram(), event="fail_3e"))
        return False
    log.append(_sample(read_snapshot(env.get_ram()), env.get_ram(), event="entered_3e"))

    # Moldorm clear + wander for key
    cr = _clear_types(
        env, (MOLDORM_OBJECT_TYPE,), max_frames=16000, min_count=1, wait_spawn=60
    )
    log.append({"event": "moldorm_clear", **cr})
    # pick key if present
    for _ in range(400):
        s = read_snapshot(env.get_ram())
        if s.room_item_id in (0, 0x03) and s.keys >= 1:
            # still try center wander
            pass
        act, at = _goto_xy(s, 120, 141, tol=8)
        env.step(act)
        # small patrol for fixed key
        if _ % 40 < 10:
            env.step(nes_action("UP"))
        elif _ % 40 < 20:
            env.step(nes_action("RIGHT"))
        elif _ % 40 < 30:
            env.step(nes_action("DOWN"))
        else:
            env.step(nes_action("LEFT"))
    log.append(_sample(read_snapshot(env.get_ram()), env.get_ram(), event="after_moldorm"))

    if not _enter_dir(env, "UP", ROOM_2E):
        log.append(_sample(read_snapshot(env.get_ram()), env.get_ram(), event="fail_2e"))
        return False
    log.append(_sample(read_snapshot(env.get_ram()), env.get_ram(), event="entered_2e"))

    cr = _clear_types(
        env, (ROPE_OBJECT_TYPE,), max_frames=16000, min_count=4, wait_spawn=200
    )
    log.append({"event": "ropes_clear", **cr})
    # Wait for kill-door UP bit (doors often 0→8)
    for _ in range(90):
        s = read_snapshot(env.get_ram())
        if s.cur_opened_doors & DOOR_UP:
            break
        env.step(nes_idle_action())
    log.append(_sample(read_snapshot(env.get_ram()), env.get_ram(), event="after_ropes"))

    if not _enter_dir(env, "UP", ROOM_1E, x_tol=2, budget=1200):
        # retry with looser tol / longer
        if not _enter_dir(env, "UP", ROOM_1E, x_tol=6, budget=1200):
            log.append(
                _sample(read_snapshot(env.get_ram()), env.get_ram(), event="fail_1e")
            )
            return False
    log.append(_sample(read_snapshot(env.get_ram()), env.get_ram(), event="entered_1e"))
    return True


def run_probe(
    *,
    start_state: str = "Level2Boom",
    infinite_life: bool = True,
    tag: str = "l2_1e_up",
    save_checkpoint: bool = False,
) -> dict:
    configure_headless()
    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True) if infinite_life else None
    track = "assisted" if infinite_life else "clean"
    log: list = []
    policies: list = []
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        obs, *_ = env.step(nes_idle_action())
        if assist is not None:
            assist.apply_env(env, frame=0)
        _idle(env, 15)
        log.append(_sample(read_snapshot(env.get_ram()), env.get_ram(), event="boot"))

        if not chain_to_1e(env, assist, log):
            snap = read_snapshot(env.get_ram())
            save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_chain_fail.png")
            out = {
                "bead": "rr-n5i",
                "result": "CHAIN_FAIL",
                "track": track,
                "log": log,
                "final": _sample(snap, env.get_ram(), event="final"),
            }
            write_json_report(RECORDINGS_DIR / f"{tag}.json", out)
            return out

        # Clear Goriya 0x1e (wait spawn ~100f)
        if assist is not None:
            assist.apply_env(env, frame=5000)
        from zelda_i.dungeon import GORIYA_OBJECT_TYPE

        _wait_enemies(env, (GORIYA_OBJECT_TYPE,), min_count=3, budget=200)
        log.append(
            _sample(read_snapshot(env.get_ram()), env.get_ram(), event="1e_spawned")
        )
        # Prefer ROOM_1E_SPEC controller
        ctrl = GenericDungeonRoomController(ROOM_1E_SPEC)
        ctrl.phase = DungeonPhase.FIGHT
        f = 0
        for f in range(ROOM_1E_SPEC.max_frames):
            if assist is not None and f % 30 == 0:
                assist.apply_env(env, frame=5000 + f)
            act = ctrl.step(read_snapshot(env.get_ram()))
            obs, *_ = env.step(act.action)
            if ctrl.success or ctrl.phase in (DungeonPhase.FAILED, DungeonPhase.DONE):
                break
        log.append({"event": "goriya_clear", **ctrl.report(), "frames": f + 1})
        # Wait for door open animation / mask
        door_timeline = []
        for i in range(120):
            env.step(nes_idle_action())
            if i % 10 == 0:
                s = read_snapshot(env.get_ram())
                door_timeline.append(
                    {
                        "i": i,
                        "doors": s.cur_opened_doors,
                        "mask": s.open_doorway_mask,
                        "all_dead": s.room_all_dead,
                        "xy": [s.link_x, s.link_y],
                    }
                )
        log.append({"event": "door_timeline", "samples": door_timeline})
        post = _sample(read_snapshot(env.get_ram()), env.get_ram(), event="post_clear")
        log.append(post)
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_postclear.png")

        # --- Policy A: strict x=120 UP push (L3-style |dx|≤2) ---
        def pol_strict_up():
            _walk_to(env, 120, 141)
            min_y = 255
            for _ in range(200):
                s = read_snapshot(env.get_ram())
                if s.screen != ROOM_1E:
                    return {"left": True, "min_y": min_y, "sc": f"0x{s.screen:02x}"}
                min_y = min(min_y, s.link_y)
                env.step(_push_door(s, "UP", x_tol=2))
            s = read_snapshot(env.get_ram())
            return {
                "left": s.screen != ROOM_1E,
                "min_y": min_y,
                "sc": f"0x{s.screen:02x}",
                "xy": [s.link_x, s.link_y],
            }

        policies.append(_try_policy(env, "strict_x120_up", pol_strict_up))
        if policies[-1]["ok"]:
            return _finish_ok(env, assist, log, policies, tag, track, save_checkpoint)

        # --- Policy B: x-scan max-north ---
        def pol_xscan():
            xs = list(range(100, 145, 2))
            return {"scan": _max_north_scan(env, x_list=xs, push=100)}

        policies.append(_try_policy(env, "x_scan_north", pol_xscan))
        if policies[-1]["ok"]:
            return _finish_ok(env, assist, log, policies, tag, track, save_checkpoint)
        save_rgb_png(
            env.render() if hasattr(env, "render") else obs,
            RECORDINGS_DIR / f"{tag}_after_xscan.png",
        )
        # grab frame
        obs, *_ = env.step(nes_idle_action())
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_after_xscan.png")

        # --- Policy C: diagonal UP+LEFT / UP+RIGHT at north wall ---
        def pol_diagonal():
            results = []
            for combo in (
                ("UP", "LEFT"),
                ("UP", "RIGHT"),
                ("LEFT", "UP"),
                ("RIGHT", "UP"),
            ):
                _walk_to(env, 120, 125)
                for _ in range(30):
                    env.step(nes_action(combo[0]))
                for _ in range(120):
                    s = read_snapshot(env.get_ram())
                    if s.screen != ROOM_1E:
                        results.append(
                            {
                                "combo": combo,
                                "ok": True,
                                "sc": f"0x{s.screen:02x}",
                                "xy": [s.link_x, s.link_y],
                            }
                        )
                        return {"results": results}
                    # alternate
                    env.step(nes_action(combo[_ % 2]))
                s = read_snapshot(env.get_ram())
                results.append(
                    {
                        "combo": combo,
                        "ok": False,
                        "sc": f"0x{s.screen:02x}",
                        "xy": [s.link_x, s.link_y],
                        "min_attempt": True,
                    }
                )
                _walk_to(env, 120, 141)
            return {"results": results}

        policies.append(_try_policy(env, "diagonal_clip", pol_diagonal))
        if policies[-1]["ok"]:
            return _finish_ok(env, assist, log, policies, tag, track, save_checkpoint)

        # --- Policy D: key at north door (boss door?) ---
        def pol_key_north():
            keys0 = read_snapshot(env.get_ram()).keys
            _walk_to(env, 120, 109)
            # face UP and push (game auto-uses key on locked door)
            for _ in range(200):
                s = read_snapshot(env.get_ram())
                if s.screen != ROOM_1E:
                    return {
                        "left": True,
                        "keys0": keys0,
                        "keys1": s.keys,
                        "sc": f"0x{s.screen:02x}",
                    }
                env.step(_push_door(s, "UP", x_tol=2))
            s = read_snapshot(env.get_ram())
            return {
                "left": s.screen != ROOM_1E,
                "keys0": keys0,
                "keys1": s.keys,
                "doors": s.cur_opened_doors,
                "mask": s.open_doorway_mask,
                "xy": [s.link_x, s.link_y],
            }

        policies.append(_try_policy(env, "key_north_push", pol_key_north))
        if policies[-1]["ok"]:
            return _finish_ok(env, assist, log, policies, tag, track, save_checkpoint)

        # --- Policy E: bomb north wall stands ---
        def pol_bomb_n():
            stands = [
                (120, 101),
                (120, 97),
                (120, 109),
                (112, 101),
                (128, 101),
                (120, 117),
                (104, 101),
                (136, 101),
            ]
            outs = []
            for sx, sy in stands:
                s0 = read_snapshot(env.get_ram())
                b0 = s0.bombs
                d0 = s0.cur_opened_doors
                m0 = s0.open_doorway_mask
                _walk_to(env, sx, sy, tol=3)
                _ensure_bomb(env)
                for _ in range(4):
                    env.step(nes_action("UP"))
                env.step(nes_action("UP", "B"))
                _idle(env, 2)
                for _ in range(90):
                    env.step(nes_action("UP"))
                left = False
                for _ in range(200):
                    s = read_snapshot(env.get_ram())
                    if s.screen != ROOM_1E:
                        left = True
                        break
                    env.step(_push_door(s, "UP", x_tol=2))
                s = read_snapshot(env.get_ram())
                outs.append(
                    {
                        "stand": [sx, sy],
                        "ok": left,
                        "bombs": f"{b0}->{s.bombs}",
                        "doors": f"{d0}->{s.cur_opened_doors}",
                        "mask": f"{m0}->{s.open_doorway_mask}",
                        "sc": f"0x{s.screen:02x}",
                        "xy": [s.link_x, s.link_y],
                    }
                )
                if left:
                    return {"stands": outs}
                _walk_to(env, 120, 141)
            return {"stands": outs}

        policies.append(_try_policy(env, "bomb_north", pol_bomb_n))
        if policies[-1]["ok"]:
            return _finish_ok(env, assist, log, policies, tag, track, save_checkpoint)
        obs, *_ = env.step(nes_idle_action())
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_after_bombn.png")

        # --- Policy F: re-enter room from 0x2e then UP ---
        def pol_reentry():
            # go DOWN to 0x2e
            ok_down = _enter_dir(env, "DOWN", ROOM_2E, budget=600)
            s = read_snapshot(env.get_ram())
            if not ok_down:
                return {
                    "down_ok": False,
                    "sc": f"0x{s.screen:02x}",
                    "doors": s.cur_opened_doors,
                    "mask": s.open_doorway_mask,
                }
            _idle(env, 30)
            s2e = _sample(s, env.get_ram(), event="on_2e")
            ok_up = _enter_dir(env, "UP", ROOM_1E, budget=600, x_tol=2)
            if not ok_up:
                return {"down_ok": True, "reenter_1e": False, "on_2e": s2e}
            _idle(env, 40)
            s1e = read_snapshot(env.get_ram())
            sample_re = _sample(s1e, env.get_ram(), event="reentered_1e")
            # immediate UP
            for _ in range(250):
                s = read_snapshot(env.get_ram())
                if s.screen != ROOM_1E and s.mode in (PLAY_MODE, 6, 7):
                    return {
                        "down_ok": True,
                        "reenter_1e": True,
                        "up_ok": True,
                        "sc": f"0x{s.screen:02x}",
                        "reenter_sample": sample_re,
                        "xy": [s.link_x, s.link_y],
                        "doors": s.cur_opened_doors,
                        "mask": s.open_doorway_mask,
                    }
                env.step(_push_door(s, "UP", x_tol=2))
            s = read_snapshot(env.get_ram())
            return {
                "down_ok": True,
                "reenter_1e": True,
                "up_ok": False,
                "reenter_sample": sample_re,
                "final": _sample(s, env.get_ram(), event="reentry_up_fail"),
            }

        policies.append(_try_policy(env, "reentry_then_up", pol_reentry))
        if policies[-1]["ok"]:
            return _finish_ok(env, assist, log, policies, tag, track, save_checkpoint)
        obs, *_ = env.step(nes_idle_action())
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_after_reentry.png")

        # --- Policy G: RAM poke mask = doors, then UP ---
        def pol_mask_poke():
            s = read_snapshot(env.get_ram())
            # ensure on 1e
            if s.screen != ROOM_1E:
                _enter_dir(env, "UP", ROOM_1E, budget=400)
                s = read_snapshot(env.get_ram())
            doors = s.cur_opened_doors
            notes = [
                _poke_u8(env, ADDR_OPEN_DOORWAY_MASK, doors | DOOR_UP | DOOR_DOWN),
                _poke_u8(env, ADDR_CUR_OPENED_DOORS, doors | DOOR_UP | DOOR_DOWN),
            ]
            _idle(env, 5)
            for _ in range(250):
                s = read_snapshot(env.get_ram())
                if s.screen != ROOM_1E:
                    return {
                        "ok": True,
                        "notes": notes,
                        "sc": f"0x{s.screen:02x}",
                        "xy": [s.link_x, s.link_y],
                    }
                env.step(_push_door(s, "UP", x_tol=2))
            s = read_snapshot(env.get_ram())
            return {
                "ok": False,
                "notes": notes,
                "doors": s.cur_opened_doors,
                "mask": s.open_doorway_mask,
                "xy": [s.link_x, s.link_y],
                "sample": _sample(s, env.get_ram(), event="mask_poke_end"),
            }

        policies.append(_try_policy(env, "mask_poke_up", pol_mask_poke))
        if policies[-1]["ok"]:
            return _finish_ok(env, assist, log, policies, tag, track, save_checkpoint)

        # --- Policy H: bomb RIGHT residual (old man) then other path? ---
        def pol_bomb_r_and_explore():
            outs = []
            for sx, sy in ((200, 141), (184, 141), (176, 141), (208, 141)):
                s0 = read_snapshot(env.get_ram())
                if s0.screen != ROOM_1E:
                    break
                b0 = s0.bombs
                _walk_to(env, sx, sy, tol=4)
                _ensure_bomb(env)
                for _ in range(4):
                    env.step(nes_action("RIGHT"))
                env.step(nes_action("RIGHT", "B"))
                for _ in range(90):
                    env.step(nes_action("RIGHT"))
                left = False
                for _ in range(200):
                    s = read_snapshot(env.get_ram())
                    if s.screen != ROOM_1E:
                        left = True
                        break
                    env.step(_push_door(s, "RIGHT", x_tol=4))
                s = read_snapshot(env.get_ram())
                outs.append(
                    {
                        "stand": [sx, sy],
                        "ok": left,
                        "bombs": f"{b0}->{s.bombs}",
                        "sc": f"0x{s.screen:02x}",
                        "xy": [s.link_x, s.link_y],
                        "doors": s.cur_opened_doors,
                        "mask": s.open_doorway_mask,
                    }
                )
                if left:
                    break
            return {"bomb_r": outs}

        policies.append(_try_policy(env, "bomb_right", pol_bomb_r_and_explore))

        # --- Policy I: free explore grid after clear for any exit ---
        def pol_grid():
            edges = []
            start_sc = read_snapshot(env.get_ram()).screen
            if start_sc != ROOM_1E:
                return {"skip": True, "sc": f"0x{start_sc:02x}"}
            for y in range(101, 190, 16):
                for x in range(48, 200, 16):
                    _walk_to(env, x, y, budget=120, tol=6)
                    for d in ("UP", "RIGHT", "DOWN", "LEFT"):
                        for _ in range(40):
                            s = read_snapshot(env.get_ram())
                            if s.screen != start_sc:
                                edges.append(
                                    {
                                        "from_xy": [x, y],
                                        "dir": d,
                                        "to": f"0x{s.screen:02x}",
                                        "xy": [s.link_x, s.link_y],
                                        "doors": s.cur_opened_doors,
                                        "mask": s.open_doorway_mask,
                                    }
                                )
                                # return if possible
                                opp = {
                                    "UP": "DOWN",
                                    "DOWN": "UP",
                                    "LEFT": "RIGHT",
                                    "RIGHT": "LEFT",
                                }[d]
                                for _ in range(200):
                                    s2 = read_snapshot(env.get_ram())
                                    if s2.screen == start_sc:
                                        break
                                    env.step(nes_action(opp))
                                break
                            env.step(nes_action(d))
                        else:
                            continue
                        break
            return {"edges": edges}

        policies.append(_try_policy(env, "grid_explore", pol_grid))
        if policies[-1]["ok"]:
            return _finish_ok(env, assist, log, policies, tag, track, save_checkpoint)

        obs, *_ = env.step(nes_idle_action())
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_final.png")
        snap = read_snapshot(env.get_ram())
        out = {
            "bead": "rr-n5i",
            "result": "UP_RESIDUAL",
            "track": track,
            "goal": "0x1e UP → Dodongo → TF 0x02",
            "log": log,
            "policies": policies,
            "final": _sample(snap, env.get_ram(), event="final"),
            "triforce_bit_0x02": bool(read_u8(env.get_ram(), ADDR_TRIFORCE) & 0x02),
            "evidence": [
                f"recordings/{tag}.json",
                f"recordings/{tag}_postclear.png",
                f"recordings/{tag}_final.png",
            ],
        }
        write_json_report(RECORDINGS_DIR / f"{tag}.json", out)
        return out
    finally:
        env.close()


def _finish_ok(env, assist, log, policies, tag, track, save_checkpoint) -> dict:
    snap = read_snapshot(env.get_ram())
    obs, *_ = env.step(nes_idle_action())
    save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_opened.png")
    boss_sc = snap.screen
    log.append(_sample(snap, env.get_ram(), event="entered_north_of_1e"))

    # Try to fight Dodongo with bombs if we landed somewhere new
    dodongo_log = []
    if snap.screen != ROOM_1E:
        # idle settle
        for _ in range(60):
            env.step(nes_idle_action())
            if assist is not None:
                assist.apply_env(env, frame=9000)
        snap = read_snapshot(env.get_ram())
        dodongo_log.append(_sample(snap, env.get_ram(), event="boss_settle"))
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_boss_room.png")

        # bomb spam near center toward large enemies
        _ensure_bomb(env)
        for i in range(800):
            if assist is not None and i % 20 == 0:
                assist.apply_env(env, frame=10000 + i)
            s = read_snapshot(env.get_ram())
            # target live non-ignore
            live = _live(s)
            if live:
                t = live[0]
                act, at = _goto_xy(s, t.x, t.y, tol=16)
                if at or abs(s.link_x - t.x) < 24:
                    # face and bomb
                    if abs(s.link_x - t.x) > abs(s.link_y - t.y):
                        face = "RIGHT" if t.x > s.link_x else "LEFT"
                    else:
                        face = "DOWN" if t.y > s.link_y else "UP"
                    if i % 40 < 6:
                        env.step(nes_action(face, "B"))
                    else:
                        env.step(nes_action(face))
                else:
                    env.step(act)
            else:
                # wander + bomb
                d = ("UP", "RIGHT", "DOWN", "LEFT")[i // 40 % 4]
                if i % 50 < 4:
                    env.step(nes_action(d, "B"))
                else:
                    env.step(nes_action(d))
            if read_u8(env.get_ram(), ADDR_TRIFORCE) & 0x02:
                break
            # heart / clear?
            if i % 100 == 99:
                dodongo_log.append(
                    _sample(
                        read_snapshot(env.get_ram()),
                        env.get_ram(),
                        event=f"boss_t{i}",
                    )
                )

        # After clear, try RIGHT for TF
        for _ in range(400):
            s = read_snapshot(env.get_ram())
            if read_u8(env.get_ram(), ADDR_TRIFORCE) & 0x02:
                break
            if s.mode != PLAY_MODE:
                env.step(nes_idle_action())
                continue
            # center then RIGHT door
            if s.screen == boss_sc:
                env.step(_push_door(s, "RIGHT", x_tol=4))
            else:
                # walk to center of TF room
                act, at = _goto_xy(s, 120, 141, tol=8)
                env.step(act if not at else nes_idle_action())
        tf = int(read_u8(env.get_ram(), ADDR_TRIFORCE))
        snap = read_snapshot(env.get_ram())
        dodongo_log.append(_sample(snap, env.get_ram(), event="post_boss"))
        obs, *_ = env.step(nes_idle_action())
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_post_boss.png")

        ck = None
        if save_checkpoint and (tf & 0x02):
            ck = str(save_state(env, GAME_DIR, GAME, "Level2Complete"))

        out = {
            "bead": "rr-n5i",
            "result": "OPENED" if not (tf & 0x02) else "TF_02",
            "track": track,
            "boss_room": f"0x{boss_sc:02x}",
            "triforce": tf,
            "triforce_bit_0x02": bool(tf & 0x02),
            "log": log,
            "policies": policies,
            "dodongo": dodongo_log,
            "checkpoint": ck,
            "winning_policy": next(
                (p["policy"] for p in policies if p.get("ok")), None
            ),
        }
        write_json_report(RECORDINGS_DIR / f"{tag}.json", out)
        return out

    out = {
        "bead": "rr-n5i",
        "result": "OPENED_NO_SCREEN?",
        "track": track,
        "log": log,
        "policies": policies,
        "final": _sample(snap, env.get_ram(), event="final"),
    }
    write_json_report(RECORDINGS_DIR / f"{tag}.json", out)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--from-state", default="Level2Boom")
    p.add_argument("--infinite-life", action="store_true", default=True)
    p.add_argument("--no-infinite-life", action="store_true")
    p.add_argument("--tag", default="l2_1e_up")
    p.add_argument("--save-state", action="store_true")
    args = p.parse_args()
    inf = not args.no_infinite_life
    report = run_probe(
        start_state=args.from_state,
        infinite_life=inf,
        tag=args.tag,
        save_checkpoint=args.save_state,
    )
    print(
        f"result={report.get('result')} "
        f"tf02={report.get('triforce_bit_0x02')} "
        f"win={report.get('winning_policy')} "
        f"final={report.get('final', {}).get('sc')}"
    )
    for pol in report.get("policies") or []:
        print(
            f"  policy {pol['policy']}: ok={pol['ok']} "
            f"detail_keys={list((pol.get('detail') or {}).keys())}"
        )


if __name__ == "__main__":
    main()
