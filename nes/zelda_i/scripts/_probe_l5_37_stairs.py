"""Hunt L5 stairs from Level5Cleared37 ladder, then 0x27/0x26 0x68 push.

PRIORITY: 0x15 is a hint pocket — leave it. Whistle still missing.
1. 0x37 pit/ladder (Level5Cleared37): walk the ladder/stairs, dump dest.
2. 0x27 and 0x26: push any 0x68 after clear.
3. 0x24 south only if still no stairs (mask had S). Do NOT fight 0x38.

No pokes, no candle invent, no Clean STATUS, no east67, no 0x65 bombs,
no 0x24 combat, no more 0x15.
Reuse dungeon_ops / stairs helpers / push_dir.
"""
from __future__ import annotations

import zipfile

from retro_harness.env import make_env, reset_obs, state_path, write_state_bytes
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.door_graph.core import DoorDir, dirs_from_mask
from zelda_i.dungeon_ids import object_name, room_item_name
from zelda_i.dungeon_ops import goto, idle, push_dir
from zelda_i.dungeon_trace import compact_snapshot, write_state_provenance
from zelda_i.level9_stairs import (
    BLOCK_PUSH_STANDS,
    BLOCK_STAIRS_X,
    BLOCK_STAIRS_Y,
    CELLAR_MODE,
    ITEM_CELLAR_MODE,
    PUSHABLE_BLOCK,
    STAIR_STANDS,
    dest_report,
    on_stair_tile,
    on_warp_tile,
    pushable_block,
    stair_transition_modes,
    walk_to_step,
)
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

STATE_37 = "Level5Cleared37"
STATE_27 = "Level5Cleared27"
STATE_26 = "Level5Cleared26"
STATE_25 = "Level5Cleared25"
ROOM_37 = 0x37
ROOM_27 = 0x27
ROOM_26 = 0x26
ROOM_25 = 0x25
ROOM_24 = 0x24
BOSS_38 = 0x38
LEVEL_5 = 5

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

# 0x37 ladder center + nearby stands. Avoid x=56 pinch; cross at y≈109.
LADDER_STANDS: tuple[tuple[int, int], ...] = (
    (120, 141),
    (120, 125),
    (120, 157),
    (112, 141),
    (128, 141),
    (120, 109),
    (120, 173),
    (96, 141),
    (144, 141),
    (120, 117),
    (104, 141),
    (136, 141),
    (80, 141),
    (160, 141),
    (120, 93),
    (120, 189),
    (BLOCK_STAIRS_X, BLOCK_STAIRS_Y),
) + tuple(STAIR_STANDS[:8])

ITEM_STANDS: tuple[tuple[int, int], ...] = (
    (120, 141),
    (120, 125),
    (120, 109),
    (96, 141),
    (144, 141),
    (80, 141),
    (160, 141),
    (120, 157),
    (64, 117),
    (176, 117),
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
    compact["colliding_tile"] = int(snap.colliding_tile)
    compact["stair_tile"] = bool(on_stair_tile(snap))
    compact["warp_tile"] = bool(on_warp_tile(snap))
    compact["cellar_mode"] = bool(stair_transition_modes(snap.mode))
    compact["dest"] = dest_report(snap)
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
    compact["blocks_0x68"] = [
        o for o in compact["objects"] if o["type_id"] == PUSHABLE_BLOCK
    ]
    return compact


def open_env(state: str):
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
        if stair_transition_modes(snap.mode):
            return True
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
            if stall >= 36:
                return False
        else:
            stall = 0
        last = pos
    return False


def to_ladder_center(env, assist, total) -> dict:
    """0x37: avoid x=56 pinch; cross at y≈109 then x=120, center 120,141."""
    snap = read_snapshot(env.get_ram())
    start = [snap.link_x, snap.link_y]
    log = [{"at": start, "step": "start"}]
    if abs(snap.link_x - 120) <= 2 and abs(snap.link_y - 141) <= 2:
        return {"ok": True, "via": "already_centered", "xy": start, "log": log}
    # If on the pinched column, go y=109 first.
    if snap.link_x < 80:
        walk_axis(env, assist, total, "y", 109, max_f=300)
        snap = read_snapshot(env.get_ram())
        log.append({"at": [snap.link_x, snap.link_y], "via": "y109_from_pinch"})
    walk_axis(env, assist, total, "y", 109, max_f=250)
    walk_axis(env, assist, total, "x", 120, max_f=400)
    walk_axis(env, assist, total, "y", 141, max_f=300)
    snap = read_snapshot(env.get_ram())
    ok = abs(snap.link_x - 120) <= 3 and abs(snap.link_y - 141) <= 3
    if not ok:
        goto(env, assist, total, 120, 141, tol=3, max_f=500)
        snap = read_snapshot(env.get_ram())
        ok = abs(snap.link_x - 120) <= 4 and abs(snap.link_y - 141) <= 4
    log.append({"at": [snap.link_x, snap.link_y], "ok": ok})
    return {"ok": ok, "via": "y109_x120", "xy": [snap.link_x, snap.link_y], "log": log}


def wait_settle(env, assist, total, max_f: int = 240) -> None:
    for _ in range(max_f):
        snap = read_snapshot(env.get_ram())
        if snap.mode in (PLAY_MODE, CELLAR_MODE, ITEM_CELLAR_MODE) and not snap.transitioning:
            idle(env, assist, total, 16)
            return
        step(env, assist, total, nes_idle_action())


def left_room(snap: ZeldaSnapshot, room0: int, mode0: int) -> bool:
    if stair_transition_modes(snap.mode):
        return True
    if snap.mode != mode0:
        return True
    if snap.screen != room0 and snap.mode not in (6, 7):
        return True
    if snap.level == 0:
        return True
    return False


def walk_stands(env, assist, total, stands, room0: int, mode0: int) -> dict:
    hits = []
    took = False
    dest = None
    for tx, ty in stands:
        snap = read_snapshot(env.get_ram())
        if left_room(snap, room0, mode0):
            took = True
            dest = dump_live(snap, env.get_ram())
            break
        # y-first like 0x37 unstick / walk_to_step
        for _ in range(360):
            snap = read_snapshot(env.get_ram())
            if left_room(snap, room0, mode0):
                took = True
                dest = dump_live(snap, env.get_ram())
                break
            frame = walk_to_step(snap, tx, ty, y_first=True, tol=2)
            if frame.reason == "walk_arrived":
                idle(env, assist, total, 10)
                snap = read_snapshot(env.get_ram())
                rec = {
                    "stand": [tx, ty],
                    "xy": [snap.link_x, snap.link_y],
                    "tile": int(snap.colliding_tile),
                    "stair_tile": bool(on_stair_tile(snap)),
                    "mode": snap.mode,
                    "room": f"0x{snap.screen:02x}",
                    "whistle": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
                }
                hits.append(rec)
                if left_room(snap, room0, mode0) or on_stair_tile(snap) or on_warp_tile(snap):
                    took = left_room(snap, room0, mode0) or on_stair_tile(snap)
                    dest = dump_live(snap, env.get_ram())
                break
            step(env, assist, total, frame.action)
        if took:
            break
        # Nudge on the stand.
        for direction in ("UP", "DOWN", "LEFT", "RIGHT"):
            for _ in range(18):
                snap = read_snapshot(env.get_ram())
                if left_room(snap, room0, mode0):
                    took = True
                    dest = dump_live(snap, env.get_ram())
                    break
                step(env, assist, total, nes_action(direction))
            if took:
                break
        if took:
            break
    return {"took": took, "hits": hits, "dest": dest}


def take_stairs_if_any(env, assist, total, tag: str, room0: int, mode0: int) -> dict:
    snap = read_snapshot(env.get_ram())
    if not (left_room(snap, room0, mode0) or on_stair_tile(snap) or on_warp_tile(snap)):
        return {"took": False}
    for direction in ("UP", "DOWN", "LEFT", "RIGHT"):
        push_dir(env, assist, total, direction, frames=140)
        idle(env, assist, total, 12)
        snap = read_snapshot(env.get_ram())
        if left_room(snap, room0, mode0):
            break
    wait_settle(env, assist, total, max_f=280)
    idle(env, assist, total, 20)
    ram = env.get_ram()
    snap = read_snapshot(ram)
    dump = dump_live(snap, ram)
    png = RECORDINGS_DIR / f"{tag}.png"
    obs, *_ = env.step(nes_idle_action())
    total[0] += 1
    assist.apply_env(env, frame=total[0])
    save_rgb_png(obs, png)
    json_path = RECORDINGS_DIR / f"{tag}.json"
    body = {
        "via": tag,
        "pokes": False,
        "status_claim": None,
        "dump": dump,
        "dest": dest_report(snap),
        "screenshot": str(png.resolve()),
        "whistle_0x065C": dump.get("whistle_0x065C"),
        "rom": rom_room(int(snap.screen)),
    }
    write_json_report(json_path, body)
    print(
        "DEST",
        tag,
        "room",
        dump.get("room_hex"),
        "mode",
        snap.mode,
        "level",
        snap.level,
        "item",
        snap.room_item_id,
        room_item_name(snap.room_item_id),
        "whistle",
        dump.get("whistle_0x065C"),
        "objs",
        [(o["type_hex"], o["type_name"], o["hp"]) for o in dump.get("objects") or []],
        flush=True,
    )
    return {
        "took": True,
        "dump": dump,
        "dest": dest_report(snap),
        "screenshot": str(png.resolve()),
        "dump_path": str(json_path.resolve()),
        "whistle_0x065C": dump.get("whistle_0x065C"),
    }


def hunt_item(env, assist, total) -> dict:
    w0 = int(read_u8(env.get_ram(), ADDR_WHISTLE))
    room0 = read_snapshot(env.get_ram()).screen
    hits = []
    for tx, ty in ITEM_STANDS:
        snap = read_snapshot(env.get_ram())
        if snap.screen != room0 and snap.mode == PLAY_MODE:
            break
        walk_axis(env, assist, total, "y", ty, max_f=220)
        walk_axis(env, assist, total, "x", tx, max_f=220)
        idle(env, assist, total, 8)
        w1 = int(read_u8(env.get_ram(), ADDR_WHISTLE))
        snap = read_snapshot(env.get_ram())
        hits.append(
            {
                "stand": [tx, ty],
                "xy": [snap.link_x, snap.link_y],
                "whistle": w1,
                "tile": int(snap.colliding_tile),
                "mode": snap.mode,
            }
        )
        if w1 > w0:
            break
    w1 = int(read_u8(env.get_ram(), ADDR_WHISTLE))
    return {"whistle_in": w0, "whistle_out": w1, "got": w1 > w0, "hits": hits}


def save_whistle(env, source_state: str) -> str:
    name = "Level5Whistle"
    path = write_state_bytes(state_path(GAME_DIR, GAME, name), env.em.get_state())
    ram = env.get_ram()
    snap = read_snapshot(ram)
    write_state_provenance(
        path,
        source_state_path=GAME_DIR / "custom_integrations" / GAME / f"{source_state}.state",
        request={
            "segment": "Level5Whistle",
            "predecessor_entry": True,
            "start_state": source_state,
            "via": "0x37 ladder / 0x27-0x26 push / 0x24 south",
            "key_poke": False,
            "door_poke": False,
            "bomb_count_poke": False,
            "selected_item_poke": False,
        },
        selected_trial={
            "success": True,
            "room": int(snap.screen),
            "mode": int(snap.mode),
            "whistle_0x065C": int(read_u8(ram, ADDR_WHISTLE)),
            "xy": [snap.link_x, snap.link_y],
        },
        natural_entry=False,
    )
    return name


def push_blocks(env, assist, total, room0: int) -> dict:
    snap = read_snapshot(env.get_ram())
    mode0 = snap.mode
    blocks = [
        o
        for o in snap.objects
        if 1 <= o.slot <= 12 and o.type_id == PUSHABLE_BLOCK
    ]
    helper = pushable_block(snap)
    log = []
    took = False
    dest = None
    if helper is not None and not any(b.slot == helper.slot for b in blocks):
        blocks.append(helper)
    targets = [(b.x, b.y, f"obj_{b.slot}") for b in blocks]
    # Typical post-clear block stands even if no 0x68 object yet.
    for tx, ty in ((120, 141), (96, 141), (144, 141), (80, 141), (160, 141)) + BLOCK_PUSH_STANDS[:4]:
        targets.append((tx, ty, "stand"))
    seen = set()
    for tx, ty, kind in targets:
        key = (tx // 8, ty // 8)
        if key in seen:
            continue
        seen.add(key)
        snap = read_snapshot(env.get_ram())
        if left_room(snap, room0, mode0):
            took = True
            dest = dump_live(snap, env.get_ram())
            break
        walk_axis(env, assist, total, "y", ty, max_f=280)
        walk_axis(env, assist, total, "x", tx, max_f=280)
        rec = {"kind": kind, "stand": [tx, ty], "dirs": []}
        for direction in ("LEFT", "UP", "RIGHT", "DOWN"):
            push_dir(env, assist, total, direction, frames=90)
            idle(env, assist, total, 8)
            snap = read_snapshot(env.get_ram())
            rec["dirs"].append(
                {
                    "dir": direction,
                    "xy": [snap.link_x, snap.link_y],
                    "tile": int(snap.colliding_tile),
                    "stair": bool(on_stair_tile(snap)),
                    "mode": snap.mode,
                    "room": f"0x{snap.screen:02x}",
                    "blocks": [
                        {"x": o.x, "y": o.y}
                        for o in snap.objects
                        if 1 <= o.slot <= 12 and o.type_id == PUSHABLE_BLOCK
                    ],
                }
            )
            if left_room(snap, room0, mode0) or on_stair_tile(snap):
                took = True
                dest = dump_live(snap, env.get_ram())
                break
        log.append(rec)
        if took:
            break
        # Walk the revealed-stairs stand after a push.
        walk_axis(env, assist, total, "y", BLOCK_STAIRS_Y, max_f=200)
        walk_axis(env, assist, total, "x", BLOCK_STAIRS_X, max_f=200)
        idle(env, assist, total, 8)
        snap = read_snapshot(env.get_ram())
        if left_room(snap, room0, mode0) or on_stair_tile(snap):
            took = True
            dest = dump_live(snap, env.get_ram())
            break
    snap = read_snapshot(env.get_ram())
    return {
        "blocks_seen": [
            {"slot": b.slot, "x": b.x, "y": b.y, "hp": b.hp} for b in blocks
        ],
        "took": took,
        "dest": dest,
        "end": dump_live(snap, env.get_ram()),
        "log": log,
    }


def shot(env, assist, total, name: str) -> str:
    png = RECORDINGS_DIR / f"{name}.png"
    obs, *_ = env.step(nes_idle_action())
    total[0] += 1
    assist.apply_env(env, frame=total[0])
    save_rgb_png(obs, png)
    return str(png.resolve())


def hunt_37() -> dict:
    env, assist, obs = open_env(STATE_37)
    total = [1]
    try:
        idle(env, assist, total, 16)
        ram = env.get_ram()
        start = dump_live(read_snapshot(ram), ram)
        print(
            "START37",
            start.get("room_hex"),
            "mode",
            start.get("mode"),
            "xy",
            [start.get("x"), start.get("y")],
            "tile",
            start.get("colliding_tile"),
            "whistle",
            start.get("whistle_0x065C"),
            "blocks",
            start.get("blocks_0x68"),
            "objs",
            [(o["type_hex"], o["type_name"]) for o in start.get("objects") or []],
            flush=True,
        )
        center = to_ladder_center(env, assist, total)
        print("CENTER", center, flush=True)
        snap = read_snapshot(env.get_ram())
        room0, mode0 = snap.screen, snap.mode
        walked = walk_stands(env, assist, total, LADDER_STANDS, room0, mode0)
        print(
            "WALK37 took",
            walked.get("took"),
            "hits",
            len(walked.get("hits") or []),
            "stair_hits",
            [h for h in (walked.get("hits") or []) if h.get("stair_tile")],
            flush=True,
        )
        dest = None
        item = None
        ckpt = None
        if walked.get("took") or (
            walked.get("dest")
            and (
                walked["dest"].get("cellar_mode")
                or walked["dest"].get("stair_tile")
            )
        ):
            dest = take_stairs_if_any(env, assist, total, "l5_37_stairs_dest", room0, mode0)
            snap = read_snapshot(env.get_ram())
            if stair_transition_modes(snap.mode) or snap.screen != room0:
                item = hunt_item(env, assist, total)
                print("ITEM37", item, flush=True)
                if item.get("got"):
                    ckpt = save_whistle(env, STATE_37)
        after = dump_live(read_snapshot(env.get_ram()), env.get_ram())
        png = shot(env, assist, total, "l5_37_stairs")
        whistle = int(read_u8(env.get_ram(), ADDR_WHISTLE))
        body = {
            "from_state": STATE_37,
            "pokes": False,
            "status_claim": None,
            "rom37": rom_room(ROOM_37),
            "start": start,
            "center": center,
            "walk": {
                "took": walked.get("took"),
                "hits": walked.get("hits"),
            },
            "dest": dest,
            "item": item,
            "after": after,
            "screenshot": png,
            "whistle_0x065C": whistle,
            "checkpoint": ckpt,
            "first_stairs": bool(dest and dest.get("took")),
        }
        write_json_report(RECORDINGS_DIR / "l5_37_stairs.json", body)
        return body
    finally:
        env.close()


def hunt_push(state: str, room: int, tag: str) -> dict:
    env, assist, obs = open_env(state)
    total = [1]
    try:
        idle(env, assist, total, 12)
        ram = env.get_ram()
        start = dump_live(read_snapshot(ram), ram)
        print(
            f"START{room:02X}",
            start.get("room_hex"),
            "xy",
            [start.get("x"), start.get("y")],
            "tile",
            start.get("colliding_tile"),
            "blocks",
            start.get("blocks_0x68"),
            "whistle",
            start.get("whistle_0x065C"),
            "objs",
            [(o["type_hex"], o["type_name"], o["x"], o["y"]) for o in start.get("objects") or []],
            flush=True,
        )
        snap = read_snapshot(env.get_ram())
        room0, mode0 = snap.screen, snap.mode
        pushed = push_blocks(env, assist, total, room0)
        print(
            f"PUSH{room:02X}",
            "blocks",
            pushed.get("blocks_seen"),
            "took",
            pushed.get("took"),
            flush=True,
        )
        dest = None
        item = None
        ckpt = None
        if pushed.get("took"):
            dest = take_stairs_if_any(env, assist, total, f"{tag}_dest", room0, mode0)
            snap = read_snapshot(env.get_ram())
            if stair_transition_modes(snap.mode) or snap.screen != room0:
                item = hunt_item(env, assist, total)
                if item.get("got"):
                    ckpt = save_whistle(env, state)
        after = dump_live(read_snapshot(env.get_ram()), env.get_ram())
        png = shot(env, assist, total, tag)
        whistle = int(read_u8(env.get_ram(), ADDR_WHISTLE))
        body = {
            "from_state": state,
            "pokes": False,
            "status_claim": None,
            "rom": rom_room(room),
            "start": start,
            "push": {
                "blocks_seen": pushed.get("blocks_seen"),
                "took": pushed.get("took"),
                "log": pushed.get("log"),
            },
            "dest": dest,
            "item": item,
            "after": after,
            "screenshot": png,
            "whistle_0x065C": whistle,
            "checkpoint": ckpt,
        }
        write_json_report(RECORDINGS_DIR / f"{tag}.json", body)
        return body
    finally:
        env.close()


def hunt_24_south() -> dict:
    env, assist, obs = open_env(STATE_25)
    total = [1]
    try:
        idle(env, assist, total, 12)
        start = dump_live(read_snapshot(env.get_ram()), env.get_ram())
        print("START25", start.get("room_hex"), [start.get("x"), start.get("y")], flush=True)
        # West into 0x24. Align y=141; detour south if pinched.
        walk_axis(env, assist, total, "y", 141, max_f=300)
        walk_axis(env, assist, total, "x", 32, max_f=500)
        if read_snapshot(env.get_ram()).screen != ROOM_24:
            walk_axis(env, assist, total, "y", 189, max_f=300)
            walk_axis(env, assist, total, "x", 32, max_f=500)
            walk_axis(env, assist, total, "y", 141, max_f=300)
        push_dir(env, assist, total, "LEFT", frames=180)
        idle(env, assist, total, 16)
        snap = read_snapshot(env.get_ram())
        if snap.screen == ROOM_24:
            wait_settle(env, assist, total)
        idle(env, assist, total, 12)
        at24 = dump_live(read_snapshot(env.get_ram()), env.get_ram())
        print(
            "AT24",
            at24.get("room_hex"),
            "mask",
            at24.get("doorway_mask"),
            "objs",
            [(o["type_hex"], o["hp"]) for o in at24.get("objects") or []],
            flush=True,
        )
        # SOUTH only. East wall then south mouth — do not engage 0x38.
        walk_axis(env, assist, total, "x", 208, max_f=300)
        walk_axis(env, assist, total, "y", 205, max_f=400)
        walk_axis(env, assist, total, "x", 120, max_f=300)
        room0 = read_snapshot(env.get_ram()).screen
        push_dir(env, assist, total, "DOWN", frames=220)
        idle(env, assist, total, 16)
        snap = read_snapshot(env.get_ram())
        if snap.screen != room0:
            wait_settle(env, assist, total)
        idle(env, assist, total, 20)
        after = dump_live(read_snapshot(env.get_ram()), env.get_ram())
        png = shot(env, assist, total, "l5_24_south")
        changed = after.get("room") != at24.get("room")
        fought = any(
            o.get("type_id") == BOSS_38 and o.get("hp", 240) < 240
            for o in after.get("objects") or []
        )
        dest = None
        item = None
        ckpt = None
        if changed and (
            after.get("cellar_mode")
            or after.get("stair_tile")
            or stair_transition_modes(after.get("mode") or 0)
        ):
            dest = take_stairs_if_any(
                env, assist, total, "l5_24_south_dest", room0, PLAY_MODE
            )
            item = hunt_item(env, assist, total)
            if item.get("got"):
                ckpt = save_whistle(env, STATE_25)
        rec = {
            "from_state": STATE_25,
            "pokes": False,
            "status_claim": None,
            "fought_0x38": fought,
            "start25": {"room": start.get("room_hex"), "keys": start.get("keys")},
            "at24": at24,
            "after": after,
            "changed": changed,
            "dest_room": after.get("room_hex") if changed else None,
            "dest": dest,
            "item": item,
            "rom24": rom_room(ROOM_24),
            "rom_dest": rom_room(int(after["room"])) if changed else None,
            "screenshot": png,
            "whistle_0x065C": after.get("whistle_0x065C"),
            "checkpoint": ckpt,
        }
        write_json_report(RECORDINGS_DIR / "l5_24_south.json", rec)
        print(
            "SOUTH24 changed",
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
        "PYTHONPATH=nes:. uv run python nes/zelda_i/scripts/_probe_l5_37_stairs.py  "
        "# Level5Cleared37 walk ladder; Level5Cleared27/26 push 0x68; "
        "0x24 south only if no stairs. infinite-life, no pokes"
    ]
    print("ROM37", rom_room(ROOM_37), flush=True)
    print("ROM27", rom_room(ROOM_27), flush=True)
    print("ROM26", rom_room(ROOM_26), flush=True)
    print("ROM24", rom_room(ROOM_24), flush=True)

    r37 = hunt_37()
    whistle = int(r37.get("whistle_0x065C") or 0)
    first = None
    if r37.get("first_stairs") or (r37.get("dest") and r37["dest"].get("took")):
        first = {
            "via": "0x37 ladder",
            "dest": (r37.get("dest") or {}).get("dest")
            or (r37.get("after") or {}).get("dest"),
            "room": (r37.get("after") or {}).get("room_hex"),
            "mode": (r37.get("after") or {}).get("mode"),
        }

    r27 = None
    r26 = None
    r24 = None
    if whistle == 0:
        r27 = hunt_push(STATE_27, ROOM_27, "l5_27_push")
        whistle = int(r27.get("whistle_0x065C") or 0)
        if first is None and r27.get("dest") and r27["dest"].get("took"):
            first = {
                "via": "0x27 push",
                "dest": r27["dest"].get("dest"),
                "room": (r27.get("after") or {}).get("room_hex"),
                "mode": (r27.get("after") or {}).get("mode"),
            }
    if whistle == 0:
        r26 = hunt_push(STATE_26, ROOM_26, "l5_26_push")
        whistle = int(r26.get("whistle_0x065C") or 0)
        if first is None and r26.get("dest") and r26["dest"].get("took"):
            first = {
                "via": "0x26 push",
                "dest": r26["dest"].get("dest"),
                "room": (r26.get("after") or {}).get("room_hex"),
                "mode": (r26.get("after") or {}).get("mode"),
            }
    if whistle == 0 and first is None:
        r24 = hunt_24_south()
        whistle = int(r24.get("whistle_0x065C") or 0)
        if first is None and r24.get("changed"):
            first = {
                "via": "0x24 south",
                "dest": (r24.get("dest") or {}).get("dest")
                if r24.get("dest")
                else None,
                "room": r24.get("dest_room"),
                "mode": (r24.get("after") or {}).get("mode"),
            }

    ckpt = None
    for row in (r37, r27, r26, r24):
        if row and row.get("checkpoint"):
            ckpt = row["checkpoint"]

    report = {
        "commands": commands,
        "pokes": False,
        "status_claim": None,
        "0x37_ladder": {
            "took": bool(r37.get("first_stairs")),
            "dest_room": (r37.get("after") or {}).get("room_hex"),
            "mode": (r37.get("after") or {}).get("mode"),
            "whistle": r37.get("whistle_0x065C"),
            "center": r37.get("center"),
        },
        "0x27_push": None
        if r27 is None
        else {
            "blocks": (r27.get("push") or {}).get("blocks_seen"),
            "took": (r27.get("push") or {}).get("took"),
            "whistle": r27.get("whistle_0x065C"),
        },
        "0x26_push": None
        if r26 is None
        else {
            "blocks": (r26.get("push") or {}).get("blocks_seen"),
            "took": (r26.get("push") or {}).get("took"),
            "whistle": r26.get("whistle_0x065C"),
        },
        "0x24_south": None
        if r24 is None
        else {
            "dest": r24.get("dest_room"),
            "fought_0x38": r24.get("fought_0x38"),
            "whistle": r24.get("whistle_0x065C"),
        },
        "FIRST_STAIRS_DEST": first,
        "whistle_0x065C": whistle,
        "checkpoint": ckpt,
    }
    write_json_report(RECORDINGS_DIR / "l5_37_stairs_summary.json", report)
    print("FIRST_STAIRS", first, flush=True)
    print("WHISTLE", whistle, flush=True)
    print("CKPT", ckpt, flush=True)
    return report


if __name__ == "__main__":
    r = main()
    print("SUMMARY", r)
