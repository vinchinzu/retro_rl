"""Level 5 whistle / bomb / cellar inbound path.

Bomb-west 0x65→0x64 center stairs → cellar 0x07 other mouth →
0x06 key-west → 0x05 clear+block stairs → 0x04 Recorder → left mouth back to 0x05.

Room specs and stop predicates remain in ``level5_dungeon``.
Import from ``zelda_i.level5_path`` (public facade).
"""

from __future__ import annotations

from retro_harness.nes import nes_action, nes_idle_action

from zelda_i.dungeon_ids import DARKNUT_OBJECT_TYPE
from zelda_i.level5_dungeon import LEVEL_5, ROOM_L5_GIBDO_66, ROOM_L5_WEST_65
from zelda_i.level5_path import _step, walk_axis
from zelda_i.ram import PLAY_MODE

ROOM_L5_BLUE_64 = 0x64
ROOM_L5_CELLAR_07 = 0x07
ROOM_L5_PASSAGE_06 = 0x06
ROOM_L5_WHISTLE_05 = 0x05
ROOM_L5_WHISTLE_ITEM = 0x04
BLUE_DARKNUT_TYPE = 0x0C
BOMB_WEST_STAND = (40, 141)
CENTER_STAIRS = (120, 141)
CELLAR_MODES = (9, 10, 11, 16)
# 0x06 diamond: 0x68 rests (96,144). Push UP → (96,128). Stairs stand (96,133).
# Center 0x70–0x73 tiles are decorative and do not warp. South key is 0x16, not return.
ROOM_06_BLOCK_X = 96
ROOM_06_BLOCK_REST_Y = 144
ROOM_06_BLOCK_PUSHED_Y = 128
ROOM_06_STAIRS_X = 96
ROOM_06_STAIRS_Y = 133


def _cellar_walk_axis(env, assist, total: list[int], axis: str, target: int, max_f: int = 700) -> bool:
    """Axis walk that survives recorder fanfare and aborts on a real 0x04 leave."""
    last = None
    stall = 0
    read_snapshot = __import__("zelda_i.ram", fromlist=["read_snapshot"]).read_snapshot
    for _ in range(max_f):
        snap = read_snapshot(env.get_ram())
        if snap.mode == PLAY_MODE and snap.screen != ROOM_L5_WHISTLE_ITEM:
            return True
        if axis == "x":
            if abs(snap.link_x - target) <= 1:
                return True
            action = nes_action("RIGHT" if snap.link_x < target else "LEFT")
        else:
            if abs(snap.link_y - target) <= 1:
                return True
            action = nes_action("DOWN" if snap.link_y < target else "UP")
        _step(env, assist, total, action)
        snap2 = read_snapshot(env.get_ram())
        pos = (snap2.link_x, snap2.link_y)
        if pos == last:
            stall += 1
            if stall >= 160:
                return False
        else:
            stall = 0
        last = pos
    return False


def select_b_item_menu(env, assist, total: list[int], want: int) -> dict:
    """Pause-cycle B items. want=1 bombs, want=5 recorder. No RAM poke."""
    from zelda_i.ram import ADDR_SELECTED_ITEM, read_u8

    selected0 = int(read_u8(env.get_ram(), ADDR_SELECTED_ITEM))
    seen = [selected0]
    if selected0 == want:
        return {"used": False, "selected": selected0, "seen": seen}
    _step(env, assist, total, nes_action("START"))
    idle = __import__("zelda_i.dungeon_ops", fromlist=["idle"]).idle
    idle(env, assist, total, 20)
    chosen = selected0
    for _ in range(8):
        _step(env, assist, total, nes_action("RIGHT"))
        idle(env, assist, total, 8)
        cur = int(read_u8(env.get_ram(), ADDR_SELECTED_ITEM))
        seen.append(cur)
        if cur == want:
            chosen = cur
            break
    _step(env, assist, total, nes_action("START"))
    idle(env, assist, total, 24)
    return {
        "used": True,
        "selected_before": selected0,
        "selected_after": int(read_u8(env.get_ram(), ADDR_SELECTED_ITEM)),
        "seen": seen,
        "preferred": chosen,
    }


def bomb_west_from_65(env, assist, total: list[int]) -> dict:
    """Bomb the west wall of cleared 0x65. One bomb. Dest must become 0x64.

    Live 0x65 has a center diamond: y=109 then x=32 then y=141, not y=141 first.
    Hold LEFT through the west scroll even while SCREEN still reads 0x65.
    """
    from zelda_i.dungeon_ops import goto, idle, push_dir
    from zelda_i.ram import read_snapshot as _rs

    walk_axis(env, assist, total, "y", 109, max_f=400)
    walk_axis(env, assist, total, "x", 32, max_f=400)
    walk_axis(env, assist, total, "y", 141, max_f=400)
    walk_axis(env, assist, total, "x", 32, max_f=200)
    goto(env, assist, total, 32, 141, tol=3, max_f=300)
    for _ in range(8):
        _step(env, assist, total, nes_action("LEFT"))
    idle(env, assist, total, 8)
    menu = select_b_item_menu(env, assist, total, 1)
    snap = _rs(env.get_ram())
    bombs0 = int(snap.bombs)
    room0 = int(snap.screen)
    _step(env, assist, total, nes_action("LEFT", "B"))
    for _ in range(16):
        _step(env, assist, total, nes_action("RIGHT"))
    idle(env, assist, total, 100)
    for _ in range(360):
        snap = _rs(env.get_ram())
        if snap.mode == PLAY_MODE and snap.screen == ROOM_L5_BLUE_64:
            break
        _step(env, assist, total, nes_action("LEFT"))
    idle(env, assist, total, 24)
    for _ in range(240):
        snap = _rs(env.get_ram())
        if snap.mode == PLAY_MODE and snap.screen == ROOM_L5_BLUE_64:
            break
        if snap.mode in (6, 7, 4, 16):
            _step(env, assist, total, nes_action("LEFT"))
        else:
            _step(env, assist, total, nes_idle_action())
    idle(env, assist, total, 16)
    snap = _rs(env.get_ram())
    return {
        "path": "bomb_west_from_65",
        "menu": menu,
        "bombs_in": bombs0,
        "bombs_out": int(snap.bombs),
        "bombs_spent": bombs0 - int(snap.bombs),
        "dest": snap.screen,
        "xy": [snap.link_x, snap.link_y],
        "mode": snap.mode,
        "success": (
            snap.level == LEVEL_5
            and snap.screen == ROOM_L5_BLUE_64
            and snap.mode == PLAY_MODE
        ),
    }


def _in_cellar(snap) -> bool:
    return snap.mode in CELLAR_MODES


def bomb_east_from_65(env, assist, total: list[int]) -> dict:
    """Bomb the east wall of cleared 0x65. One bomb. Dest must become 0x66.

    North shutter is one-way (0x55 S=open / 0x65 N=shutter). Diamond: y=109
    then east, not y=141 first.
    """
    from zelda_i.dungeon_ops import goto, idle
    from zelda_i.ram import read_snapshot as _rs

    walk_axis(env, assist, total, "y", 109, max_f=400)
    walk_axis(env, assist, total, "x", 208, max_f=500)
    walk_axis(env, assist, total, "y", 141, max_f=400)
    walk_axis(env, assist, total, "x", 224, max_f=200)
    goto(env, assist, total, 224, 141, tol=3, max_f=300)
    for _ in range(8):
        _step(env, assist, total, nes_action("RIGHT"))
    idle(env, assist, total, 8)
    menu = select_b_item_menu(env, assist, total, 1)
    snap = _rs(env.get_ram())
    bombs0 = int(snap.bombs)
    room0 = int(snap.screen)
    _step(env, assist, total, nes_action("RIGHT", "B"))
    for _ in range(16):
        _step(env, assist, total, nes_action("LEFT"))
    idle(env, assist, total, 100)
    for _ in range(360):
        snap = _rs(env.get_ram())
        if snap.mode == PLAY_MODE and snap.screen == ROOM_L5_GIBDO_66:
            break
        _step(env, assist, total, nes_action("RIGHT"))
    idle(env, assist, total, 24)
    for _ in range(240):
        snap = _rs(env.get_ram())
        if snap.mode == PLAY_MODE and snap.screen == ROOM_L5_GIBDO_66:
            break
        if snap.mode in (6, 7, 4, 16):
            _step(env, assist, total, nes_action("RIGHT"))
        else:
            _step(env, assist, total, nes_idle_action())
    idle(env, assist, total, 16)
    snap = _rs(env.get_ram())
    return {
        "path": "bomb_east_from_65",
        "menu": menu,
        "bombs_in": bombs0,
        "bombs_out": int(snap.bombs),
        "bombs_spent": bombs0 - int(snap.bombs),
        "dest": snap.screen,
        "xy": [snap.link_x, snap.link_y],
        "mode": snap.mode,
        "success": snap.level == LEVEL_5 and snap.screen == ROOM_L5_GIBDO_66 and snap.mode == PLAY_MODE,
    }


def take_center_stairs_64(env, assist, total: list[int]) -> dict:
    """Walk the south (then north) gap onto visible center stairs in 0x64.

    Do not hunt the east bomb hole. Success = cellar/stairs mode or room 0x07.
    """
    from zelda_i.dungeon_ops import idle
    from zelda_i.ram import read_snapshot as _rs

    log = []
    snap = _rs(env.get_ram())
    start = {"xy": [snap.link_x, snap.link_y], "mode": snap.mode, "room": snap.screen}

    def done(snap) -> bool:
        if _in_cellar(snap):
            return True
        return snap.level == LEVEL_5 and snap.screen == ROOM_L5_CELLAR_07

    paths = (
        (("y", 189), ("x", 80), ("y", 141), ("x", 120)),
        (("y", 189), ("x", 96), ("y", 149), ("x", 120), ("y", 141)),
        (("y", 189), ("x", 64), ("y", 141), ("x", 120)),
        (("y", 93), ("x", 80), ("y", 141), ("x", 120)),
        (("y", 189), ("x", 120), ("y", 141)),
        (("y", 173), ("x", 120), ("y", 141), ("x", 120)),
    )
    for name_i, steps in enumerate(paths):
        if done(_rs(env.get_ram())):
            break
        if _rs(env.get_ram()).screen != ROOM_L5_BLUE_64:
            break
        for axis, tgt in steps:
            walk_axis(env, assist, total, axis, tgt, max_f=360)
            snap = _rs(env.get_ram())
            log.append(
                {
                    "path": name_i,
                    "step": f"{axis}:{tgt}",
                    "xy": [snap.link_x, snap.link_y],
                    "mode": snap.mode,
                    "room": snap.screen,
                }
            )
            if done(snap):
                break
        idle(env, assist, total, 20)
        snap = _rs(env.get_ram())
        if done(snap):
            break
        # Nudge onto the tile; never hold LEFT (east bomb hole → 0x65).
        for direction in ("UP", "DOWN", "RIGHT"):
            for _ in range(16):
                snap = _rs(env.get_ram())
                if done(snap) or snap.screen != ROOM_L5_BLUE_64:
                    break
                _step(env, assist, total, nes_action(direction))
            idle(env, assist, total, 8)
            if done(_rs(env.get_ram())):
                break
        if done(_rs(env.get_ram())):
            break

    for _ in range(200):
        snap = _rs(env.get_ram())
        if done(snap) or (snap.mode == PLAY_MODE and snap.screen != ROOM_L5_BLUE_64):
            break
        _step(env, assist, total, nes_idle_action())
    idle(env, assist, total, 16)
    snap = _rs(env.get_ram())
    return {
        "path": "south_gap_center_stairs",
        "start": start,
        "log": log,
        "dest": snap.screen,
        "mode": snap.mode,
        "xy": [snap.link_x, snap.link_y],
        "cellar": _in_cellar(snap),
        "success": done(snap) and snap.screen != ROOM_L5_WEST_65,
    }


# Live L5 cellar 0x07: left mouth spawn ~(48,93); floor y=189; right climb x=192.
L5_CELLAR_FLOOR_Y = 189
L5_CELLAR_LEFT_X = 48
L5_CELLAR_RIGHT_X = 192


def cellar_other_mouth(env, assist, total: list[int]) -> dict:
    """From L5 cellar 0x07, take the opposite mouth to room 0x06. No pokes."""
    from zelda_i.dungeon_ops import idle, push_dir
    from zelda_i.ram import read_snapshot as _rs

    # Stair-enter sits at (128,141) ~90f, then remaps to a ladder (48,93) or (192,93).
    for _ in range(180):
        snap = _rs(env.get_ram())
        if snap.mode in CELLAR_MODES and (snap.link_x <= 64 or snap.link_x >= 176):
            break
        _step(env, assist, total, nes_idle_action())
    idle(env, assist, total, 12)
    snap = _rs(env.get_ram())
    start = {"xy": [snap.link_x, snap.link_y], "room": snap.screen, "mode": snap.mode}
    # Left column is the 0x64 return. Floor-cross to x=192 then UP → 0x06.
    if snap.link_x <= 128:
        side = "right"
        tx = L5_CELLAR_RIGHT_X
    else:
        side = "left"
        tx = L5_CELLAR_LEFT_X
    walk_axis(env, assist, total, "y", L5_CELLAR_FLOOR_Y, max_f=400)
    walk_axis(env, assist, total, "x", tx, max_f=500)
    room0 = _rs(env.get_ram()).screen
    push_dir(env, assist, total, "UP", frames=200)
    idle(env, assist, total, 20)
    for _ in range(240):
        snap = _rs(env.get_ram())
        if snap.mode == PLAY_MODE and snap.screen != room0:
            break
        if snap.mode == PLAY_MODE and snap.screen == ROOM_L5_PASSAGE_06:
            break
        _step(env, assist, total, nes_idle_action())
    idle(env, assist, total, 16)
    snap = _rs(env.get_ram())
    return {
        "path": "cellar_other_mouth",
        "start": start,
        "chose_side": side,
        "target_x": tx,
        "dest": snap.screen,
        "mode": snap.mode,
        "xy": [snap.link_x, snap.link_y],
        "success": snap.level == LEVEL_5 and snap.screen == ROOM_L5_PASSAGE_06 and snap.mode == PLAY_MODE,
    }


def key_west_to(env, assist, total: list[int], expect: int) -> dict:
    """Spend a key at the west door. No door/key poke."""
    from zelda_i.dungeon_ops import goto, idle, push_dir
    from zelda_i.ram import read_snapshot as _rs

    snap = _rs(env.get_ram())
    keys0 = int(snap.keys)
    room0 = int(snap.screen)
    walk_axis(env, assist, total, "y", 141, max_f=400)
    walk_axis(env, assist, total, "x", 32, max_f=500)
    goto(env, assist, total, 32, 141, tol=3, max_f=300)
    push_dir(env, assist, total, "LEFT", frames=240)
    idle(env, assist, total, 16)
    snap = _rs(env.get_ram())
    if snap.screen != room0:
        for _ in range(240):
            snap = _rs(env.get_ram())
            if snap.mode == PLAY_MODE:
                break
            _step(env, assist, total, nes_idle_action())
        idle(env, assist, total, 16)
    snap = _rs(env.get_ram())
    return {
        "path": "key_west",
        "keys_in": keys0,
        "keys_out": int(snap.keys),
        "key_spent": int(snap.keys) < keys0,
        "dest": snap.screen,
        "xy": [snap.link_x, snap.link_y],
        "mode": snap.mode,
        "success": snap.level == LEVEL_5 and snap.screen == expect and snap.mode == PLAY_MODE,
    }


def fight_blue_darknuts(env, assist, total: list[int], room: int, expected: int, source: int) -> dict:
    """Reuse GenericDungeonRoomController + ROOM_5B_SPEC / ROOM_59 combat."""
    from dataclasses import replace

    from zelda_i.dungeon import (
        DoorRoute,
        DungeonPhase,
        GenericDungeonRoomController,
        RewardKind,
        RewardSpec,
    )
    from zelda_i.level3_dungeon import ROOM_59_SPEC, ROOM_5B_SPEC
    from zelda_i.ram import read_snapshot as _rs

    spec = replace(
        ROOM_5B_SPEC,
        spec_id=f"level5_room{room:02x}_blue_darknuts",
        source_room=source,
        room_id=room,
        entry=DoorRoute("LEFT", ((224, 141),)),
        enemy_types=(BLUE_DARKNUT_TYPE, DARKNUT_OBJECT_TYPE),
        expected_enemy_count=expected,
        required_open_doors=0,
        reward=RewardSpec(kind=RewardKind.CLEAR_ONLY, settle_all_dead=0),
        combat=ROOM_59_SPEC.combat,
        exit_routes=(DoorRoute("RIGHT", ((208, 141),)),),
        max_frames=28000,
        level=LEVEL_5,
    )
    ctl = GenericDungeonRoomController(spec)
    start_n = None
    last_n = None
    progress = []
    for _ in range(spec.max_frames):
        snap = _rs(env.get_ram())
        if snap.mode == PLAY_MODE and snap.screen == room:
            live = spec.live_enemies(snap)
            if start_n is None:
                start_n = len(live)
                last_n = start_n
                progress.append({"f": ctl.frames, "n": start_n})
            elif len(live) != last_n:
                last_n = len(live)
                progress.append({"f": ctl.frames, "n": last_n})
        action = ctl.step(snap)
        _step(env, assist, total, action.action)
        if ctl.success or ctl.phase is DungeonPhase.FAILED:
            break
    snap = _rs(env.get_ram())
    live = [
        o
        for o in snap.objects
        if 1 <= o.slot <= 12 and o.type_id in (BLUE_DARKNUT_TYPE, DARKNUT_OBJECT_TYPE) and o.hp > 0
    ] if snap.mode == PLAY_MODE else []
    return {
        "ok": bool(ctl.success) and not live,
        "frames": ctl.frames,
        "start_n": 0 if start_n is None else start_n,
        "end_n": len(live),
        "progress": progress,
        "spec_id": spec.spec_id,
        "xy": [snap.link_x, snap.link_y],
        "room": snap.screen,
    }


def push_block_stairs(env, assist, total: list[int], room: int) -> dict:
    """Push 0x68 then stand on revealed stairs. Never treat a door exit as stairs."""
    from zelda_i.dungeon_ops import idle, push_dir
    from zelda_i.level9_stairs import BLOCK_STAIRS_X, BLOCK_STAIRS_Y, PUSHABLE_BLOCK
    from zelda_i.ram import read_snapshot as _rs

    snap = _rs(env.get_ram())
    blocks = [
        o for o in snap.objects if 1 <= o.slot <= 12 and o.type_id == PUSHABLE_BLOCK
    ]
    log = []
    dest = None

    def left_ok(snap) -> bool:
        if _in_cellar(snap):
            return True
        return snap.screen != room and snap.mode in (*CELLAR_MODES, PLAY_MODE) and snap.screen != ROOM_L5_WEST_65

    targets = [(b.x, b.y) for b in blocks] + [
        (96, 144),
        (112, 144),
        (80, 144),
        (120, 144),
        (128, 144),
    ]
    seen = set()
    for tx, ty in targets:
        key = (tx // 8, ty // 8)
        if key in seen:
            continue
        seen.add(key)
        snap = _rs(env.get_ram())
        if left_ok(snap):
            dest = {"room": snap.screen, "mode": snap.mode, "xy": [snap.link_x, snap.link_y]}
            break
        walk_axis(env, assist, total, "y", ty, max_f=280)
        walk_axis(env, assist, total, "x", tx + 16, max_f=280)
        rec = {"stand": [tx, ty], "dirs": []}
        for direction in ("LEFT", "UP", "DOWN", "RIGHT"):
            push_dir(env, assist, total, direction, frames=90)
            idle(env, assist, total, 8)
            snap = _rs(env.get_ram())
            rec["dirs"].append(
                {
                    "dir": direction,
                    "xy": [snap.link_x, snap.link_y],
                    "mode": snap.mode,
                    "room": snap.screen,
                }
            )
            if left_ok(snap):
                dest = {"room": snap.screen, "mode": snap.mode, "xy": [snap.link_x, snap.link_y]}
                break
        log.append(rec)
        if dest is not None:
            break
        for sx, sy in ((tx, ty), (BLOCK_STAIRS_X, BLOCK_STAIRS_Y), CENTER_STAIRS, (120, 125)):
            walk_axis(env, assist, total, "y", sy, max_f=200)
            walk_axis(env, assist, total, "x", sx, max_f=200)
            idle(env, assist, total, 10)
            snap = _rs(env.get_ram())
            if left_ok(snap):
                dest = {"room": snap.screen, "mode": snap.mode, "xy": [snap.link_x, snap.link_y]}
                break
        if dest is not None:
            break
    snap = _rs(env.get_ram())
    return {
        "blocks_seen": [{"slot": b.slot, "x": b.x, "y": b.y} for b in blocks],
        "dest": dest,
        "end": {"room": snap.screen, "mode": snap.mode, "xy": [snap.link_x, snap.link_y]},
        "log": log,
        "success": dest is not None,
    }



def take_whistle_04(env, assist, total: list[int]) -> dict:
    """Cellar 0x04: floor y=189, short ladder x=176, left on y=141 to the Recorder."""
    from zelda_i.dungeon_ops import idle
    from zelda_i.ram import ADDR_WHISTLE, read_snapshot as _rs, read_u8

    w0 = int(read_u8(env.get_ram(), ADDR_WHISTLE))
    walk_axis(env, assist, total, "y", 189, max_f=400)
    walk_axis(env, assist, total, "x", 176, max_f=400)
    for _ in range(80):
        snap = _rs(env.get_ram())
        if int(read_u8(env.get_ram(), ADDR_WHISTLE)) > w0:
            break
        if snap.link_y <= 141 and abs(snap.link_x - 176) <= 4:
            break
        _step(env, assist, total, nes_action("UP"))
    idle(env, assist, total, 8)
    walk_axis(env, assist, total, "y", 141, max_f=200)
    walk_axis(env, assist, total, "x", 128, max_f=300)
    idle(env, assist, total, 12)
    w1 = int(read_u8(env.get_ram(), ADDR_WHISTLE))
    if w1 <= w0:
        walk_axis(env, assist, total, "x", 144, max_f=200)
        walk_axis(env, assist, total, "x", 120, max_f=200)
        idle(env, assist, total, 10)
        w1 = int(read_u8(env.get_ram(), ADDR_WHISTLE))
    snap = _rs(env.get_ram())
    return {
        "in": w0,
        "out": w1,
        "got": w1 > w0,
        "xy": [snap.link_x, snap.link_y],
        "room": snap.screen,
        "mode": snap.mode,
    }


def hunt_whistle(env, assist, total: list[int]) -> dict:
    """Walk item stands until ADDR_WHISTLE becomes 1.

    Room 0x04 is a side-scroll item cellar: top-down stands stay on the
    floor (y=189). Use take_whistle_04 (right short ladder -> y=141).
    """
    from zelda_i.dungeon_ops import idle
    from zelda_i.ram import ADDR_WHISTLE, read_snapshot as _rs, read_u8

    w0 = int(read_u8(env.get_ram(), ADDR_WHISTLE))
    snap0 = _rs(env.get_ram())
    room0 = snap0.screen
    hits = []
    if room0 == ROOM_L5_WHISTLE_ITEM or snap0.mode in CELLAR_MODES:
        cellar = take_whistle_04(env, assist, total)
        hits.append({"via": "take_whistle_04", "xy": cellar.get("xy"), "value": cellar.get("out")})
        if cellar.get("got"):
            return {"in": w0, "out": cellar["out"], "got": True, "hits": hits, "via": "take_whistle_04"}
    stands = (
        (120, 141),
        (136, 141),
        (104, 141),
        (120, 125),
        (120, 157),
        (80, 141),
        (160, 141),
        (120, 109),
        (64, 117),
        (176, 117),
        (96, 165),
        (144, 165),
    )
    for tx, ty in stands:
        snap = _rs(env.get_ram())
        if snap.screen != room0 and snap.mode == PLAY_MODE:
            break
        walk_axis(env, assist, total, "y", ty, max_f=220)
        walk_axis(env, assist, total, "x", tx, max_f=220)
        idle(env, assist, total, 10)
        w1 = int(read_u8(env.get_ram(), ADDR_WHISTLE))
        snap = _rs(env.get_ram())
        hits.append({"stand": [tx, ty], "xy": [snap.link_x, snap.link_y], "value": w1})
        if w1 > w0:
            break
    w1 = int(read_u8(env.get_ram(), ADDR_WHISTLE))
    return {"in": w0, "out": w1, "got": w1 > w0, "hits": hits}


# Live 0x04 item cellar: isolated recorder alcove ~y=141, x≈112–176.
# Short ladder at x=176 drops to pit y=189. Left mouth stairs at x=48
# (spawn 48,65) return to play 0x05. Do not walk left on the alcove —
# the platform does not connect to the left column.
WHISTLE_04_LADDER_X = 176
WHISTLE_04_PIT_Y = 189
WHISTLE_04_MOUTH_X = 48
WHISTLE_04_MOUTH_Y = 65


def exit_whistle_04(env, assist, total: list[int]) -> dict:
    """Leave cellar 0x04: alcove x=176 DOWN → pit y=189 → left mouth x=48 UP → 0x05.

    Failed probes walked LEFT/UP on the recorder alcove (y=141). That platform
    does not connect to the left column. Drop the short ladder first.
    """
    from zelda_i.dungeon_ops import idle, push_dir
    from zelda_i.ram import ADDR_WHISTLE, read_snapshot as _rs, read_u8

    snap = _rs(env.get_ram())
    start = {
        "xy": [snap.link_x, snap.link_y],
        "mode": snap.mode,
        "room": snap.screen,
        "whistle": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
    }
    log = [dict(start, tag="start")]

    def left_ok(s) -> bool:
        return s.mode == PLAY_MODE and s.screen != ROOM_L5_WHISTLE_ITEM

    def rec(tag: str) -> None:
        s = _rs(env.get_ram())
        log.append(
            {
                "tag": tag,
                "xy": [s.link_x, s.link_y],
                "mode": s.mode,
                "room": s.screen,
                "tile": int(s.colliding_tile),
            }
        )

    # Recorder item-get holds Link overhead. Tap RIGHT (toward the ladder)
    # until RAM x/y changes — idle-only is not enough; walk_axis aborts at 40.
    idle(env, assist, total, 40)
    thawed = False
    for n in range(10):
        x0 = _rs(env.get_ram()).link_x
        y0 = _rs(env.get_ram()).link_y
        for i in range(24):
            _step(env, assist, total, nes_action("RIGHT"))
            snap = _rs(env.get_ram())
            if snap.link_x != x0 or snap.link_y != y0:
                thawed = True
                log.append({"tag": "thawed", "burst": n, "f": i, "xy": [snap.link_x, snap.link_y]})
                break
        if thawed:
            break
        idle(env, assist, total, 32)
    rec("unstick")

    def drop_ladder() -> None:
        _cellar_walk_axis(env, assist, total, "y", 141, max_f=240)
        _cellar_walk_axis(env, assist, total, "x", WHISTLE_04_LADDER_X, max_f=700)
        rec("ladder")
        for _ in range(280):
            snap = _rs(env.get_ram())
            if left_ok(snap) or snap.link_y >= WHISTLE_04_PIT_Y - 2:
                break
            _step(env, assist, total, nes_action("DOWN"))
        idle(env, assist, total, 8)
        _cellar_walk_axis(env, assist, total, "y", WHISTLE_04_PIT_Y, max_f=400)
        rec("pit")

    # Alcove (y≈141) only drops at the short ladder x=176.
    for attempt in range(3):
        snap = _rs(env.get_ram())
        if left_ok(snap) or snap.link_y >= 170:
            break
        drop_ladder()
        snap = _rs(env.get_ram())
        if snap.link_y < 170 and abs(snap.link_x - WHISTLE_04_LADDER_X) > 4:
            log.append({"tag": f"retry_ladder_{attempt}", "xy": [snap.link_x, snap.link_y]})

    snap = _rs(env.get_ram())
    # Live RAM: only the pit (y>=170) connects to the left mouth. Do not
    # walk LEFT on the alcove — that stalls at x≈112, y=141.
    if not left_ok(snap) and snap.link_y >= 170:
        _cellar_walk_axis(env, assist, total, "x", WHISTLE_04_MOUTH_X, max_f=700)
        rec("left_col")
        # Hold UP from the pit. Do not walk_axis to y=65 — that overshoots
        # into 0x05's north door after the mouth fires.
        push_dir(env, assist, total, "UP", frames=280)
        idle(env, assist, total, 12)
        for _ in range(280):
            snap = _rs(env.get_ram())
            if left_ok(snap):
                break
            if abs(snap.link_x - WHISTLE_04_MOUTH_X) > 4:
                _step(
                    env,
                    assist,
                    total,
                    nes_action("LEFT" if snap.link_x > WHISTLE_04_MOUTH_X else "RIGHT"),
                )
            else:
                _step(env, assist, total, nes_action("UP"))
        idle(env, assist, total, 20)
    rec("after_up")
    snap = _rs(env.get_ram())
    return {
        "path": "alcove_ladder176_pit189_left48",
        "start": start,
        "log": log,
        "dest": snap.screen,
        "mode": snap.mode,
        "xy": [snap.link_x, snap.link_y],
        "whistle": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
        "success": (
            snap.level == LEVEL_5
            and snap.mode == PLAY_MODE
            and snap.screen == ROOM_L5_WHISTLE_05
        ),
        "left_cellar": left_ok(snap),
        "thawed": thawed,
    }



leave_whistle_cellar = exit_whistle_04
walk_out_of_04 = exit_whistle_04
