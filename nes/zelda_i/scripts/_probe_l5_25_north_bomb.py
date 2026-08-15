"""Bomb NORTH wall of L5 0x25 from Level5Cleared25. One bomb. No pokes.

Reuse dungeon_ops.goto + the 0x66 west-bomb menu/place/push pattern
(stand 120,101 face UP = BOMB_N_STAND). Only claim dest if room id changes.

If dest opens: dump objects/doors/item; stairs down; Darknuts/item walk
until 0x065C=1 or clearly not whistle. Honest checkpoint if whistle or dest.

If still 0x25: from Level5Cleared25 walk west into 0x24, then SOUTH only
(do NOT fight type 0x38). Stop.
"""
from __future__ import annotations

import zipfile
from dataclasses import replace

from retro_harness.env import make_env, reset_obs, state_path, write_state_bytes
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.door_graph.core import DoorDir, dirs_from_mask
from zelda_i.dungeon import (
    DoorRoute,
    DungeonPhase,
    DungeonRoomSpec,
    GenericDungeonRoomController,
    RewardKind,
    RewardSpec,
)
from zelda_i.dungeon_ids import DARKNUT_OBJECT_TYPE, object_name, room_item_name
from zelda_i.dungeon_ops import goto, idle, push_dir
from zelda_i.dungeon_trace import compact_snapshot, write_state_provenance
from zelda_i.level2_bomb_path import BOMB_N_STAND
from zelda_i.level3_dungeon import ROOM_5B_SPEC, ROOM_59_SPEC
from zelda_i.level5_dungeon import LEVEL_5, POLS_VOICE_OBJECT_TYPE
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR, SHARED_ROM_ZIP
from zelda_i.ram import (
    ADDR_CANDLE,
    ADDR_SELECTED_ITEM,
    ADDR_WHISTLE,
    PLAY_MODE,
    ZeldaSnapshot,
    read_snapshot,
    read_u8,
)

STATE = "Level5Cleared25"
ROOM_25 = 0x25
ROOM_24 = 0x24
BOSS_38 = 0x38
STAND = BOMB_N_STAND  # (120, 101)
FACE = "UP"
MAX_FIGHT_FRAMES = 20000
CELLAR_MODES = (9, 10, 11, 16)
STAIR_TILE_LO = 0x70
STAIR_TILE_HI = 0x73
BLACK_MOUTH_TILE = 0x24
PUSHABLE_BLOCK = 0x68

ROM_Q1_NS = 0x18700
ROM_Q1_EW = 0x18780
ROM_Q1_FLAGS = 0x18980
DOOR_CODES = {
    0: "open",
    1: "wall",
    2: "false",
    3: "false2",
    4: "bomb",
    5: "key",
    6: "key2",
    7: "shutter",
}
SECRET_CODES = {
    0: "none",
    1: "all_dead",
    2: "ringleader",
    3: "last_boss",
    4: "block_door",
    5: "block_stairs",
    6: "money_or_life",
    7: "foes_item",
}

ITEM_WAYPOINTS: tuple[tuple[int, int], ...] = (
    (120, 141),
    (96, 117),
    (144, 165),
    (80, 141),
    (160, 141),
    (120, 157),
    (120, 125),
    (64, 117),
    (176, 117),
    (64, 165),
    (176, 165),
    (112, 141),
    (128, 141),
    (120, 109),
    (120, 173),
)


def rom_room(room: int) -> dict:
    with zipfile.ZipFile(SHARED_ROM_ZIP) as zf:
        data = zf.read(zf.namelist()[0])

    def b(dc: int) -> int:
        return data[dc + 0x10]

    ns = b(ROM_Q1_NS + room)
    ew = b(ROM_Q1_EW + room)
    flags = b(ROM_Q1_FLAGS + room)
    n, s = (ns >> 5) & 7, (ns >> 2) & 7
    w, e = (ew >> 5) & 7, (ew >> 2) & 7
    secret = flags & 7
    return {
        "room": f"0x{room:02x}",
        "N": DOOR_CODES.get(n, str(n)),
        "S": DOOR_CODES.get(s, str(s)),
        "W": DOOR_CODES.get(w, str(w)),
        "E": DOOR_CODES.get(e, str(e)),
        "secret": SECRET_CODES.get(secret, str(secret)),
        "ns_hex": f"0x{ns:02x}",
        "ew_hex": f"0x{ew:02x}",
        "flags_hex": f"0x{flags:02x}",
    }


def decode_doors(mask: int) -> dict:
    value = int(mask) & 0x0F
    return {
        "raw": value,
        "raw_hex": f"0x{value:02x}",
        "east": bool(value & DoorDir.RIGHT),
        "west": bool(value & DoorDir.LEFT),
        "south": bool(value & DoorDir.DOWN),
        "north": bool(value & DoorDir.UP),
        "open": sorted(d.name for d in dirs_from_mask(value)),
    }


def inv_block(ram) -> dict:
    snap = read_snapshot(ram)
    return {
        "selected_item_0x0656": int(read_u8(ram, ADDR_SELECTED_ITEM)),
        "whistle_0x065C": int(read_u8(ram, ADDR_WHISTLE)),
        "candle_0x065B": int(read_u8(ram, ADDR_CANDLE)),
        "bombs": int(snap.bombs),
        "keys": int(snap.keys),
    }


def dump_live(snap: ZeldaSnapshot, ram) -> dict:
    compact = compact_snapshot(snap)
    compact["doors"] = decode_doors(snap.cur_opened_doors)
    compact["doorway_mask"] = decode_doors(snap.open_doorway_mask)
    compact["room_hex"] = f"0x{snap.screen:02x}"
    compact["next_room_hex"] = f"0x{snap.next_screen:02x}"
    compact["inventory"] = inv_block(ram)
    compact["whistle_0x065C"] = int(read_u8(ram, ADDR_WHISTLE))
    compact["colliding_tile"] = snap.colliding_tile
    compact["objects"] = [
        {
            "slot": obj.slot,
            "type_id": obj.type_id,
            "type_hex": f"0x{obj.type_id:02x}",
            "type_name": object_name(obj.type_id),
            "x": obj.x,
            "y": obj.y,
            "hp": obj.hp,
            "state": obj.state,
            "facing": obj.facing,
        }
        for obj in snap.objects
        if 1 <= obj.slot <= 12 and obj.type_id not in (0, 0xFF)
    ]
    return compact


def live_pols(snap: ZeldaSnapshot) -> list:
    if snap.mode != PLAY_MODE:
        return []
    return [
        obj
        for obj in snap.objects
        if 1 <= obj.slot <= 12
        and obj.type_id == POLS_VOICE_OBJECT_TYPE
        and obj.hp > 0
    ]


def live_darknuts(snap: ZeldaSnapshot) -> list:
    if snap.mode != PLAY_MODE:
        return []
    return [
        obj
        for obj in snap.objects
        if 1 <= obj.slot <= 12
        and obj.type_id == DARKNUT_OBJECT_TYPE
        and obj.hp > 0
    ]


def stair_tile(tile: int) -> bool:
    return STAIR_TILE_LO <= int(tile) <= STAIR_TILE_HI or int(tile) == BLACK_MOUTH_TILE


def open_env(state: str = STATE):
    env = make_env(GAME, state, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True)
    obs, _ = reset_obs(env)
    obs, *_ = env.step(nes_idle_action())
    assist.apply_env(env, frame=0)
    return env, assist, obs


def step(env, assist, total, action):
    obs, *_ = env.step(action)
    total[0] += 1
    assist.apply_env(env, frame=total[0])
    return obs


def walk_axis(env, assist, total, axis: str, target: int, max_f: int = 400) -> bool:
    last = None
    stall = 0
    for _ in range(max_f):
        snap = read_snapshot(env.get_ram())
        if axis == "x":
            if abs(snap.link_x - target) <= 1:
                return True
            step(env, assist, total, nes_action("RIGHT" if snap.link_x < target else "LEFT"))
        else:
            if abs(snap.link_y - target) <= 1:
                return True
            step(env, assist, total, nes_action("DOWN" if snap.link_y < target else "UP"))
        snap2 = read_snapshot(env.get_ram())
        pos = (snap2.link_x, snap2.link_y)
        if pos == last:
            stall += 1
            if stall >= 40:
                return False
        else:
            stall = 0
        last = pos
    return False


NORTH_HOLE_PATHS = (
    # Raised-tile row at y=117: only x=32 / x=208 reach the north door plane.
    ("west_wall_north", (("y", 125), ("x", 32), ("y", 93), ("x", 120))),
    ("east_wall_north", (("y", 125), ("x", 208), ("y", 93), ("x", 120))),
)

WEST_24_PATHS = (
    ("y141_west", (("y", 141), ("x", 32))),
    ("south189_west", (("y", 189), ("x", 32), ("y", 141))),
    ("north109_west", (("y", 109), ("x", 80), ("y", 141), ("x", 32))),
    ("east208_south189_west", (("x", 208), ("y", 189), ("x", 32), ("y", 141))),
    ("south173_west64", (("y", 173), ("x", 64), ("y", 141), ("x", 32))),
)


def push_north_hole(env, assist, total, room0: int) -> dict:
    """After the north bomb hole opens, path around raised tiles and push UP."""
    log = []
    for name, steps in NORTH_HOLE_PATHS:
        snap = read_snapshot(env.get_ram())
        if snap.screen != room0:
            return {
                "changed": True,
                "path": name,
                "dest": f"0x{snap.screen:02x}",
                "log": log,
            }
        rec = {"path": name, "steps": []}
        for axis, tgt in steps:
            ok = walk_axis(env, assist, total, axis, tgt, max_f=450)
            snap = read_snapshot(env.get_ram())
            rec["steps"].append(
                {
                    "axis": axis,
                    "tgt": tgt,
                    "ok": ok,
                    "xy": [snap.link_x, snap.link_y],
                    "room": f"0x{snap.screen:02x}",
                    "tile": snap.colliding_tile,
                }
            )
            if snap.screen != room0:
                rec["changed_mid"] = True
                log.append(rec)
                wait_play(env, assist, total, snap.screen, max_f=240)
                return {
                    "changed": True,
                    "path": name,
                    "dest": f"0x{snap.screen:02x}",
                    "log": log,
                }
        for _ in range(220):
            snap = read_snapshot(env.get_ram())
            if snap.screen != room0 and snap.mode in (PLAY_MODE, 6, 7, 4):
                break
            step(env, assist, total, nes_action("UP"))
        idle(env, assist, total, 16)
        snap = read_snapshot(env.get_ram())
        if snap.screen != room0:
            wait_play(env, assist, total, snap.screen, max_f=240)
            rec["xy"] = [snap.link_x, snap.link_y]
            rec["dest"] = f"0x{snap.screen:02x}"
            log.append(rec)
            print("NORTH_HOLE", name, "dest", rec["dest"], flush=True)
            return {
                "changed": True,
                "path": name,
                "dest": rec["dest"],
                "log": log,
            }
        rec["xy"] = [snap.link_x, snap.link_y]
        rec["doors"] = int(snap.cur_opened_doors) & 0x0F
        log.append(rec)
        print("NORTH_HOLE", name, "still", f"0x{snap.screen:02x}", "xy", rec["xy"], flush=True)
    return {"changed": False, "path": None, "dest": None, "log": log}


def wait_play(env, assist, total, room: int, *, max_f: int = 360) -> bool:
    for _ in range(max_f):
        snap = read_snapshot(env.get_ram())
        if (
            snap.level == LEVEL_5
            and snap.screen == room
            and snap.mode == PLAY_MODE
            and not snap.transitioning
        ):
            idle(env, assist, total, 16)
            snap = read_snapshot(env.get_ram())
            return (
                snap.screen == room
                and snap.mode == PLAY_MODE
                and not snap.transitioning
            )
        step(env, assist, total, nes_idle_action())
    return False


def select_bombs_menu(env, assist, total) -> dict:
    """Pause-menu select bombs. No selected-item / bomb-count poke."""
    selected0 = int(read_u8(env.get_ram(), ADDR_SELECTED_ITEM))
    if selected0 == 1:
        return {"used": False, "selected": selected0, "reason": "already_bombs_1"}
    step(env, assist, total, nes_action("START"))
    idle(env, assist, total, 20)
    for _ in range(3):
        step(env, assist, total, nes_action("RIGHT"))
        idle(env, assist, total, 6)
    step(env, assist, total, nes_action("START"))
    idle(env, assist, total, 24)
    return {
        "used": True,
        "selected_before": selected0,
        "selected_after": int(read_u8(env.get_ram(), ADDR_SELECTED_ITEM)),
    }


def save_ckpt(env, name: str, request: dict, trial: dict) -> str:
    path = write_state_bytes(state_path(GAME_DIR, GAME, name), env.em.get_state())
    write_state_provenance(
        path,
        source_state_path=GAME_DIR / "custom_integrations" / GAME / f"{STATE}.state",
        request=request,
        selected_trial=trial,
        natural_entry=False,
    )
    return name


def take_stairs(env, assist, total, tag: str) -> dict:
    ram = env.get_ram()
    snap = read_snapshot(ram)
    room0 = snap.screen
    mode0 = snap.mode
    for direction in ("UP", "DOWN", "LEFT", "RIGHT"):
        push_dir(env, assist, total, direction, frames=140)
        idle(env, assist, total, 16)
        snap = read_snapshot(env.get_ram())
        if snap.screen != room0 or snap.mode != mode0:
            break
    extra = 0
    while extra < 220:
        snap = read_snapshot(env.get_ram())
        if snap.mode in (PLAY_MODE, 9, 11) and not snap.transitioning:
            idle(env, assist, total, 20)
            break
        step(env, assist, total, nes_idle_action())
        extra += 1
    ram = env.get_ram()
    snap = read_snapshot(ram)
    png = RECORDINGS_DIR / f"{tag}.png"
    obs, *_ = env.step(nes_idle_action())
    total[0] += 1
    assist.apply_env(env, frame=total[0])
    save_rgb_png(obs, png)
    dump = dump_live(snap, ram)
    return {
        "success": snap.screen != room0 or snap.mode != mode0,
        "from_room": f"0x{room0:02x}",
        "from_mode": mode0,
        "dest_room": f"0x{snap.screen:02x}",
        "mode": snap.mode,
        "xy": [snap.link_x, snap.link_y],
        "whistle_0x065C": int(read_u8(ram, ADDR_WHISTLE)),
        "dump": dump,
        "screenshot": str(png.resolve()),
        "pokes": False,
    }


def walk_for_whistle(env, assist, total) -> dict:
    ram = env.get_ram()
    w0 = int(read_u8(ram, ADDR_WHISTLE))
    room0 = read_snapshot(ram).screen
    hits = []
    for tx, ty in ITEM_WAYPOINTS:
        snap = read_snapshot(env.get_ram())
        if snap.screen != room0 and snap.mode == PLAY_MODE:
            break
        walk_axis(env, assist, total, "y", ty, max_f=220)
        walk_axis(env, assist, total, "x", tx, max_f=220)
        idle(env, assist, total, 8)
        ram = env.get_ram()
        w1 = int(read_u8(ram, ADDR_WHISTLE))
        snap = read_snapshot(ram)
        rec = {
            "stand": [tx, ty],
            "xy": [snap.link_x, snap.link_y],
            "whistle": w1,
            "tile": snap.colliding_tile,
            "mode": snap.mode,
            "room": f"0x{snap.screen:02x}",
            "stair_tile": stair_tile(snap.colliding_tile),
        }
        hits.append(rec)
        if w1 > w0 or snap.mode in CELLAR_MODES or stair_tile(snap.colliding_tile):
            break
    ram = env.get_ram()
    w1 = int(read_u8(ram, ADDR_WHISTLE))
    snap = read_snapshot(ram)
    return {
        "whistle_in": w0,
        "whistle_out": w1,
        "whistle_got": w1 > w0,
        "end_mode": snap.mode,
        "end_room": f"0x{snap.screen:02x}",
        "end_tile": int(snap.colliding_tile),
        "stair_tile": stair_tile(snap.colliding_tile),
        "stands": hits,
    }


def fight_darknuts(env, assist, total, room: int) -> dict:
    spec = replace(
        ROOM_5B_SPEC,
        spec_id=f"level5_room{room:02x}_darknuts_reuse5b59",
        source_room=ROOM_25,
        room_id=room,
        entry=DoorRoute("UP", ((120, 205),)),
        enemy_types=(DARKNUT_OBJECT_TYPE,),
        expected_enemy_count=max(1, len(live_darknuts(read_snapshot(env.get_ram())))),
        required_open_doors=0,
        reward=RewardSpec(kind=RewardKind.CLEAR_ONLY, settle_all_dead=0),
        combat=ROOM_59_SPEC.combat,
        exit_routes=(DoorRoute("DOWN", ((120, 205),)),),
        max_frames=MAX_FIGHT_FRAMES,
        level=LEVEL_5,
    )
    ctl = GenericDungeonRoomController(spec)
    start_n = None
    last_n = None
    progress = []
    for _ in range(spec.max_frames):
        snap = read_snapshot(env.get_ram())
        if snap.mode == PLAY_MODE and snap.screen == room:
            live = spec.live_enemies(snap)
            if start_n is None:
                start_n = len(live)
                last_n = start_n
                progress.append({"f": ctl.frames, "n": start_n, "hps": [o.hp for o in live]})
            elif len(live) != last_n:
                last_n = len(live)
                progress.append({"f": ctl.frames, "n": last_n, "hps": [o.hp for o in live]})
                print(f"DN_KILL n={last_n} f={ctl.frames}", flush=True)
        action = ctl.step(snap)
        step(env, assist, total, action.action)
        if ctl.success or ctl.phase is DungeonPhase.FAILED:
            break
    snap = read_snapshot(env.get_ram())
    live = live_darknuts(snap) if snap.mode == PLAY_MODE else []
    return {
        "ok": bool(ctl.success) and not live,
        "frames": ctl.frames,
        "start_n": 0 if start_n is None else start_n,
        "end_n": len(live),
        "progress": progress,
        "controller": ctl.report(),
        "reused": "GenericDungeonRoomController + ROOM_5B_SPEC + ROOM_59_SPEC.combat",
        "spec_id": spec.spec_id,
    }


def handle_dest(env, assist, total, dest_room: int) -> dict:
    """Dump dest; stairs down; Darknuts/item walk until whistle or not."""
    ram = env.get_ram()
    snap = read_snapshot(ram)
    dump0 = dump_live(snap, ram)
    png = RECORDINGS_DIR / "l5_25_north_bomb_dest.png"
    obs, *_ = env.step(nes_idle_action())
    total[0] += 1
    assist.apply_env(env, frame=total[0])
    save_rgb_png(obs, png)
    dest_json = RECORDINGS_DIR / "l5_25_north_bomb_dest.json"
    body = {
        "via": "0x25 NORTH bomb",
        "room": f"0x{dest_room:02x}",
        "rom": rom_room(dest_room),
        "dump": dump0,
        "screenshot": str(png.resolve()),
        "pokes": False,
        "status_claim": None,
    }
    write_json_report(dest_json, body)
    print(
        "DEST",
        f"0x{dest_room:02x}",
        "mode",
        snap.mode,
        "item",
        snap.room_item_id,
        room_item_name(snap.room_item_id),
        "objs",
        [(o["type_hex"], o["type_name"], o["hp"]) for o in dump0.get("objects") or []],
        "doors",
        dump0.get("doors"),
        "whistle",
        dump0.get("whistle_0x065C"),
        flush=True,
    )

    types = {o["type_id"] for o in dump0.get("objects") or []}
    notes = []
    stairs = None
    fight = None
    walk = None
    ckpt = None

    entered_name = f"Level5Entered{dest_room:02X}"
    ckpt = save_ckpt(
        env,
        entered_name,
        {
            "segment": entered_name,
            "predecessor_entry": True,
            "start_state": STATE,
            "via": "0x25 NORTH bomb",
            "key_poke": False,
            "door_poke": False,
            "bomb_count_poke": False,
            "selected_item_poke": False,
        },
        {
            "success": True,
            "room": dest_room,
            "bombs": int(snap.bombs),
            "keys": int(snap.keys),
            "whistle_0x065C": int(read_u8(ram, ADDR_WHISTLE)),
        },
    )
    notes.append(f"saved_{entered_name}")

    if snap.mode in CELLAR_MODES or stair_tile(snap.colliding_tile):
        notes.append("stairs_on_arrive")
        stairs = take_stairs(env, assist, total, "l5_25_north_bomb_stairs")
        print("STAIRS", stairs.get("dest_room"), "mode", stairs.get("mode"), flush=True)

    snap = read_snapshot(env.get_ram())
    if live_darknuts(snap):
        notes.append(f"darknuts_{len(live_darknuts(snap))}")
        fight = fight_darknuts(env, assist, total, int(snap.screen))
        idle(env, assist, total, 20)
        print("DARKNUTS", fight.get("ok"), "end_n", fight.get("end_n"), flush=True)

    snap = read_snapshot(env.get_ram())
    blocks = [
        o
        for o in snap.objects
        if 1 <= o.slot <= 12 and o.type_id == PUSHABLE_BLOCK
    ]
    if blocks:
        notes.append(f"pushable_{len(blocks)}")
        blk = blocks[0]
        goto(env, assist, total, blk.x, blk.y, tol=3, max_f=400)
        for direction in ("LEFT", "UP", "RIGHT", "DOWN"):
            for _ in range(40):
                step(env, assist, total, nes_action(direction))
            idle(env, assist, total, 8)
            snap = read_snapshot(env.get_ram())
            if snap.mode in CELLAR_MODES or stair_tile(snap.colliding_tile):
                notes.append(f"block_push_{direction}_stairs")
                break

    snap = read_snapshot(env.get_ram())
    if snap.mode in CELLAR_MODES or stair_tile(snap.colliding_tile):
        if stairs is None:
            stairs = take_stairs(env, assist, total, "l5_25_north_bomb_stairs")
            print("STAIRS2", stairs.get("dest_room"), "mode", stairs.get("mode"), flush=True)

    walk = walk_for_whistle(env, assist, total)
    print(
        "WALK_WHISTLE",
        walk.get("whistle_in"),
        "->",
        walk.get("whistle_out"),
        "stair",
        walk.get("stair_tile"),
        flush=True,
    )
    if walk.get("stair_tile") or walk.get("end_mode") in CELLAR_MODES:
        if stairs is None:
            stairs = take_stairs(env, assist, total, "l5_25_north_bomb_stairs")
            walk2 = walk_for_whistle(env, assist, total)
            walk["after_stairs"] = walk2
            print("WALK_AFTER_STAIRS", walk2.get("whistle_out"), flush=True)

    ram = env.get_ram()
    snap = read_snapshot(ram)
    whistle = int(read_u8(ram, ADDR_WHISTLE))
    if whistle:
        ckpt = save_ckpt(
            env,
            "Level5Whistle",
            {
                "segment": "Level5Whistle",
                "predecessor_entry": True,
                "start_state": STATE,
                "via": "0x25 NORTH bomb dest",
                "key_poke": False,
                "door_poke": False,
                "bomb_count_poke": False,
                "selected_item_poke": False,
            },
            {
                "success": True,
                "room": int(snap.screen),
                "whistle_0x065C": whistle,
                "bombs": int(snap.bombs),
                "keys": int(snap.keys),
            },
        )
        notes.append("saved_Level5Whistle")

    final = dump_live(snap, ram)
    return {
        "arrive": dump0,
        "rom": rom_room(dest_room),
        "screenshot": str(png.resolve()),
        "dump_path": str(dest_json.resolve()),
        "types": sorted(f"0x{t:02x}" for t in types),
        "notes": notes,
        "stairs": stairs,
        "darknuts": fight,
        "walk": {k: v for k, v in walk.items() if k != "stands"} if walk else None,
        "checkpoint": ckpt,
        "final": final,
        "whistle_0x065C": whistle,
        "pokes": False,
    }


def dump_24_south() -> dict:
    """From Level5Cleared25 walk west into 0x24, then SOUTH. Do not fight 0x38."""
    env, assist, obs = open_env(STATE)
    total = [1]
    try:
        idle(env, assist, total, 16)
        start = dump_live(read_snapshot(env.get_ram()), env.get_ram())
        keys0 = int(start["keys"])
        used = None
        for name, steps in WEST_24_PATHS:
            for axis, tgt in steps:
                walk_axis(env, assist, total, axis, tgt, max_f=450)
            snap = read_snapshot(env.get_ram())
            if abs(snap.link_x - 32) <= 8 and abs(snap.link_y - 141) <= 8:
                used = name
                break
        print("WEST24_PATH", used, "xy", [read_snapshot(env.get_ram()).link_x, read_snapshot(env.get_ram()).link_y], flush=True)
        push_dir(env, assist, total, "LEFT", frames=240)
        idle(env, assist, total, 16)
        snap = read_snapshot(env.get_ram())
        if snap.screen == ROOM_24:
            wait_play(env, assist, total, ROOM_24, max_f=240)
        idle(env, assist, total, 12)
        at24 = dump_live(read_snapshot(env.get_ram()), env.get_ram())
        print(
            "AT24",
            at24.get("room_hex"),
            "mode",
            at24.get("mode"),
            "mask",
            at24.get("doorway_mask"),
            "objs",
            [(o["type_hex"], o["hp"]) for o in at24.get("objects") or []],
            flush=True,
        )
        # SOUTH only. East-wall then south mouth — do not engage 0x38.
        walk_axis(env, assist, total, "x", 208, max_f=300)
        walk_axis(env, assist, total, "y", 205, max_f=400)
        walk_axis(env, assist, total, "x", 120, max_f=300)
        room0 = read_snapshot(env.get_ram()).screen
        push_dir(env, assist, total, "DOWN", frames=220)
        idle(env, assist, total, 16)
        snap = read_snapshot(env.get_ram())
        if snap.screen != room0:
            wait_play(env, assist, total, snap.screen, max_f=240)
        idle(env, assist, total, 20)
        ram = env.get_ram()
        snap = read_snapshot(ram)
        after = dump_live(snap, ram)
        png = RECORDINGS_DIR / "l5_24_south.png"
        obs, *_ = env.step(nes_idle_action())
        total[0] += 1
        assist.apply_env(env, frame=total[0])
        save_rgb_png(obs, png)
        changed = after.get("room") != at24.get("room")
        fought = any(
            o.get("type_id") == BOSS_38 and o.get("hp", 240) < 240
            for o in after.get("objects") or []
        )
        rec = {
            "from_state": STATE,
            "pokes": False,
            "status_claim": None,
            "fought_0x38": fought,
            "keys_in": keys0,
            "keys_out": after.get("keys"),
            "start25": {
                "room": start.get("room_hex"),
                "keys": start.get("keys"),
                "bombs": start.get("bombs"),
            },
            "at24": at24,
            "after": after,
            "changed": changed,
            "dest_room": after.get("room_hex") if changed else None,
            "rom24": rom_room(ROOM_24),
            "rom_dest": rom_room(int(after["room"])) if changed else None,
            "screenshot": str(png.resolve()),
            "whistle_0x065C": after.get("whistle_0x065C"),
        }
        write_json_report(RECORDINGS_DIR / "l5_24_south.json", rec)
        print(
            "SOUTH24",
            "changed",
            changed,
            "dest",
            rec.get("dest_room"),
            "fought",
            fought,
            flush=True,
        )
        return rec
    finally:
        env.close()


def main() -> dict:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    commands = [
        "PYTHONPATH=nes:. uv run python nes/zelda_i/scripts/_probe_l5_25_north_bomb.py  "
        "# Level5Cleared25 NORTH bomb 0x25, infinite-life, dungeon_ops.goto + "
        "0x66-west menu/place/push, stand (120,101) UP, no pokes"
    ]
    rom25 = rom_room(ROOM_25)
    rom15 = rom_room(0x15)
    rom24 = rom_room(ROOM_24)
    print("ROM25", rom25, flush=True)
    print("ROM15", rom15, flush=True)
    print("ROM24", rom24, flush=True)

    env, assist, obs = open_env(STATE)
    total = [1]
    dest_handle = None
    south24 = None
    ckpt = None
    try:
        idle(env, assist, total, 24)
        ram = env.get_ram()
        start_snap = read_snapshot(ram)
        start = dump_live(start_snap, ram)
        n_pols = len(live_pols(start_snap))
        print(
            "START",
            start.get("room_hex"),
            "mode",
            start.get("mode"),
            start.get("mode_name"),
            "xy",
            [start.get("x"), start.get("y")],
            "keys",
            start.get("keys"),
            "bombs",
            start.get("bombs"),
            "pols",
            n_pols,
            "whistle",
            start.get("whistle_0x065C"),
            "selected",
            start.get("inventory", {}).get("selected_item_0x0656"),
            flush=True,
        )
        pose_ok = (
            start_snap.level == LEVEL_5
            and start_snap.screen == ROOM_25
            and start_snap.mode == PLAY_MODE
            and n_pols == 0
        )

        # South-band first so we do not wedge in the raised-tile pocket at (120,117).
        walk_axis(env, assist, total, "y", 173, max_f=400)
        walk_axis(env, assist, total, "x", 120, max_f=400)
        walk_axis(env, assist, total, "y", 117, max_f=400)
        walk_axis(env, assist, total, "x", 120, max_f=200)
        for _ in range(8):
            step(env, assist, total, nes_action(FACE))
        idle(env, assist, total, 8)
        before = dump_live(read_snapshot(env.get_ram()), env.get_ram())
        obs, *_ = env.step(nes_idle_action())
        total[0] += 1
        assist.apply_env(env, frame=total[0])
        before_png = RECORDINGS_DIR / "l5_25_north_bomb_before.png"
        save_rgb_png(obs, before_png)

        menu = select_bombs_menu(env, assist, total)
        bombs0 = int(read_snapshot(env.get_ram()).bombs)
        room0 = int(read_snapshot(env.get_ram()).screen)

        step(env, assist, total, nes_action(FACE, "B"))
        # Step back south immediately so the blast does not pin Link in the tile pocket.
        for _ in range(24):
            step(env, assist, total, nes_action("DOWN"))
        idle(env, assist, total, 100)
        bombs1 = int(read_snapshot(env.get_ram()).bombs)
        door_open = bool(read_snapshot(env.get_ram()).cur_opened_doors & 0x08)
        print(
            "BLAST bombs",
            bombs0,
            "->",
            bombs1,
            "north_door",
            door_open,
            "xy",
            [read_snapshot(env.get_ram()).link_x, read_snapshot(env.get_ram()).link_y],
            flush=True,
        )
        hole = push_north_hole(env, assist, total, room0)
        idle(env, assist, total, 16)
        if read_snapshot(env.get_ram()).screen != room0:
            wait_play(env, assist, total, read_snapshot(env.get_ram()).screen, max_f=240)
        idle(env, assist, total, 16)

        after = dump_live(read_snapshot(env.get_ram()), env.get_ram())
        obs, *_ = env.step(nes_idle_action())
        total[0] += 1
        assist.apply_env(env, frame=total[0])
        after_png = RECORDINGS_DIR / "l5_25_north_bomb.png"
        save_rgb_png(obs, after_png)

        dest_changed = after.get("room") != before.get("room")
        dest_room = after.get("room") if dest_changed else None
        dest_hex = after.get("room_hex") if dest_changed else None
        print(
            "BOMB25N dest_changed",
            dest_changed,
            "from",
            before.get("room_hex"),
            "to",
            after.get("room_hex"),
            "bombs",
            bombs0,
            "->",
            after.get("bombs"),
            flush=True,
        )

        if dest_changed and dest_room is not None:
            dest_handle = handle_dest(env, assist, total, int(dest_room))
            ckpt = dest_handle.get("checkpoint")
        env.close()
        env = None

        if not dest_changed:
            south24 = dump_24_south()

        whistle = (after.get("whistle_0x065C") or 0)
        if dest_handle:
            whistle = dest_handle.get("whistle_0x065C", whistle)

        report = {
            "ok": dest_changed,
            "status_claim": None,
            "from_state": STATE,
            "pokes": False,
            "commands": commands,
            "helper_reused": "dungeon_ops.goto + 0x66 west bomb menu/place/push + BOMB_N_STAND",
            "pose_ok": pose_ok,
            "start": {
                "room": start.get("room_hex"),
                "mode": start.get("mode"),
                "mode_name": start.get("mode_name"),
                "xy": [start.get("x"), start.get("y")],
                "keys": start.get("keys"),
                "bombs": start.get("bombs"),
                "pols_live": n_pols,
                "whistle_0x065C": start.get("whistle_0x065C"),
                "selected_item_0x0656": start.get("inventory", {}).get("selected_item_0x0656"),
            },
            "rom25": rom25,
            "rom15": rom15,
            "rom24": rom24,
            "stand": list(STAND),
            "face": FACE,
            "menu": menu,
            "before": before,
            "after": after,
            "bombs_in": bombs0,
            "bombs_out": int(after.get("bombs") or bombs1),
            "bombs_spent": bombs0 - int(after.get("bombs") or bombs1),
            "one_bomb": True,
            "north_hole": hole,
            "dest_changed": dest_changed,
            "dest_north_of_0x25": dest_hex if dest_changed else "sealed",
            "dest_handle": dest_handle,
            "south24": south24,
            "checkpoint": ckpt,
            "whistle_0x065C": whistle,
            "before_screenshot": str(before_png.resolve()),
            "screenshot": str(after_png.resolve()),
        }
        write_json_report(RECORDINGS_DIR / "l5_25_north_bomb.json", report)
        return report
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    r = main()
    print("CMD", r["commands"])
    print("POSE", r.get("start"))
    print("ROM25", r.get("rom25"))
    print("BOMBS", r.get("bombs_in"), "->", r.get("bombs_out"), "spent", r.get("bombs_spent"))
    print("DEST_NORTH", r.get("dest_north_of_0x25"), "changed", r.get("dest_changed"))
    dh = r.get("dest_handle") or {}
    if dh:
        print("DEST_OBJS", (dh.get("arrive") or {}).get("objects"))
        print("DEST_ITEM", (dh.get("arrive") or {}).get("room_item_id"), (dh.get("arrive") or {}).get("room_item_name"))
        print("DEST_DOORS", (dh.get("arrive") or {}).get("doors"))
        print("DARKNUTS", dh.get("darknuts"))
        print("STAIRS", None if not dh.get("stairs") else {k: v for k, v in dh["stairs"].items() if k != "dump"})
    print("WHISTLE", r.get("whistle_0x065C"))
    if r.get("south24"):
        s = r["south24"]
        print("SOUTH24 dest", s.get("dest_room"), "fought", s.get("fought_0x38"))
    print("CKPT", r.get("checkpoint"))
    print("status_claim", None)
