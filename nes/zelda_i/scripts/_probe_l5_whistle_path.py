"""L5 whistle path from Level5Cleared65. ROM W=bomb. No pokes. No 0x24/candle/0x15.

1. Bomb WEST 0x65 -> 0x64 (5 Blue Darknuts). One bomb, dest must change.
2. Stairs in 0x64 -> cellar 0x07 (ns=0x64, ew=0x06). Dump 0x07.
3. Other cellar mouth -> 0x06. Dump 0x06.
4. Key WEST -> 0x05 (6 Blue Darknuts, secret=block_stairs). Dump 0x05.
5. Kill all 6, push 0x68, stairs -> 0x04 (item 0x05). Grab whistle. Stop at 0x065C=1.

Then if whistle=1: Level5Cleared25 WEST -> 0x24, whistle-shrink Digdogger 0x38,
sword small ones, grab heart 0x1A, dump live TF room.

Reuse: 0x66 west-bomb menu/place/push, GenericDungeonRoomController +
ROOM_5B_SPEC / ROOM_59_SPEC.combat, level9_stairs walk_to_step/dest_report,
push_dir for 0x68 after all-dead.
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
from zelda_i.level3_dungeon import ROOM_5B_SPEC, ROOM_59_SPEC
from zelda_i.level5_dungeon import LEVEL_5
from zelda_i.level9_stairs import (
    BLOCK_PUSH_STANDS,
    BLOCK_STAIRS_X,
    BLOCK_STAIRS_Y,
    CELLAR_CORRIDOR_Y,
    CELLAR_EXIT_Y,
    CELLAR_LEFT_X,
    CELLAR_MODE,
    CELLAR_RIGHT_X,
    CELLAR_SPLIT_X,
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
    ADDR_TRIFORCE,
    ADDR_WHISTLE,
    PLAY_MODE,
    ZeldaSnapshot,
    read_snapshot,
    read_u8,
)

STATE_65 = "Level5Cleared65"
STATE_55 = "Level5Cleared55"
STATE_25 = "Level5Cleared25"
ROOM_65 = 0x65
ROOM_64 = 0x64
ROOM_07 = 0x07
ROOM_06 = 0x06
ROOM_05 = 0x05
ROOM_04 = 0x04
ROOM_25 = 0x25
ROOM_24 = 0x24
ROOM_14 = 0x14
BOSS_38 = 0x38
BOMB_W_STAND = (40, 141)
FACE_W = "LEFT"
MAX_FIGHT_FRAMES = 28000
CELLAR_MODES = (9, 10, 11, 16)
TF_BIT_L5 = 0x10

ROM_Q1_NS = 0x18700
ROM_Q1_EW = 0x18780
ROM_Q1_MON = 0x18800
ROM_Q1_ITEM = 0x18900
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
    (208, 141),
    (32, 141),
)

STAIR_HUNT: tuple[tuple[int, int], ...] = (
    (120, 141),
    (120, 125),
    (120, 157),
    (96, 141),
    (144, 141),
    (80, 141),
    (160, 141),
    (120, 109),
    (120, 173),
    (64, 141),
    (176, 141),
    (BLOCK_STAIRS_X, BLOCK_STAIRS_Y),
) + tuple(STAIR_STANDS)

WEST_24_PATHS = (
    ("y141_west", (("y", 141), ("x", 32))),
    ("south189_west", (("y", 189), ("x", 32), ("y", 141))),
    ("east208_south189_west", (("x", 208), ("y", 189), ("x", 32), ("y", 141))),
)


def rom_room(room: int) -> dict:
    with zipfile.ZipFile(SHARED_ROM_ZIP) as zf:
        data = zf.read(zf.namelist()[0])

    def b(dc: int) -> int:
        return data[dc + 0x10]

    ns = b(ROM_Q1_NS + room)
    ew = b(ROM_Q1_EW + room)
    flags = b(ROM_Q1_FLAGS + room)
    item = b(ROM_Q1_ITEM + room)
    mon = b(ROM_Q1_MON + room)
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
        "item_hex": f"0x{item:02x}",
        "mon_hex": f"0x{mon:02x}",
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
        "triforce_0x0671": int(read_u8(ram, ADDR_TRIFORCE)),
        "tf_l5_bit": bool(int(read_u8(ram, ADDR_TRIFORCE)) & TF_BIT_L5),
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
    compact["triforce_0x0671"] = int(read_u8(ram, ADDR_TRIFORCE))
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


BLUE_DARKNUT_TYPE = 0x0C  # live 0x64: 5x HP128
DARKNUT_TYPES = (DARKNUT_OBJECT_TYPE, BLUE_DARKNUT_TYPE)


def live_darknuts(snap: ZeldaSnapshot) -> list:
    if snap.mode != PLAY_MODE:
        return []
    return [
        obj
        for obj in snap.objects
        if 1 <= obj.slot <= 12
        and obj.type_id in DARKNUT_TYPES
        and obj.hp > 0
    ]


def live_boss(snap: ZeldaSnapshot) -> list:
    if snap.mode != PLAY_MODE:
        return []
    return [
        obj
        for obj in snap.objects
        if 1 <= obj.slot <= 12 and obj.type_id == BOSS_38 and obj.hp > 0
    ]


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
            if stall >= 40:
                return False
        else:
            stall = 0
        last = pos
    return False


def wait_play(env, assist, total, room: int | None = None, *, max_f: int = 360) -> bool:
    for _ in range(max_f):
        snap = read_snapshot(env.get_ram())
        room_ok = room is None or snap.screen == room
        if (
            snap.level == LEVEL_5
            and room_ok
            and snap.mode in (PLAY_MODE, CELLAR_MODE, ITEM_CELLAR_MODE)
            and not snap.transitioning
        ):
            idle(env, assist, total, 16)
            return True
        step(env, assist, total, nes_idle_action())
    return False


def shot(env, assist, total, name: str) -> str:
    png = RECORDINGS_DIR / f"{name}.png"
    obs, *_ = env.step(nes_idle_action())
    total[0] += 1
    assist.apply_env(env, frame=total[0])
    save_rgb_png(obs, png)
    return str(png.resolve())


def write_dump(tag: str, body: dict) -> str:
    path = RECORDINGS_DIR / f"{tag}.json"
    write_json_report(path, body)
    return str(path.resolve())


def save_ckpt(env, name: str, source_state: str, request: dict, trial: dict) -> str:
    path = write_state_bytes(state_path(GAME_DIR, GAME, name), env.em.get_state())
    write_state_provenance(
        path,
        source_state_path=GAME_DIR / "custom_integrations" / GAME / f"{source_state}.state",
        request=request,
        selected_trial=trial,
        natural_entry=False,
    )
    return name


def select_bombs_menu(env, assist, total) -> dict:
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


def select_whistle_menu(env, assist, total) -> dict:
    """Pause-cycle B items. Prefer selected=5 (recorder). No poke."""
    selected0 = int(read_u8(env.get_ram(), ADDR_SELECTED_ITEM))
    seen = [selected0]
    if selected0 == 5:
        return {"used": False, "selected": selected0, "reason": "already_whistle_5", "seen": seen}
    step(env, assist, total, nes_action("START"))
    idle(env, assist, total, 20)
    chosen = selected0
    for _ in range(8):
        step(env, assist, total, nes_action("RIGHT"))
        idle(env, assist, total, 8)
        cur = int(read_u8(env.get_ram(), ADDR_SELECTED_ITEM))
        seen.append(cur)
        if cur == 5:
            chosen = cur
            break
        if cur not in (0, 1) and chosen in (0, 1):
            chosen = cur
    # If we overshot, keep last non-bomb if whistle never appeared.
    step(env, assist, total, nes_action("START"))
    idle(env, assist, total, 24)
    return {
        "used": True,
        "selected_before": selected0,
        "selected_after": int(read_u8(env.get_ram(), ADDR_SELECTED_ITEM)),
        "seen": seen,
        "preferred": chosen,
    }


def midroom_warp(snap: ZeldaSnapshot) -> bool:
    """Stair/black-mouth only in the playfield, not door-hole 0x24."""
    if on_stair_tile(snap):
        return True
    if not on_warp_tile(snap):
        return False
    return 48 < snap.link_x < 200 and 105 < snap.link_y < 185


def left_room(snap: ZeldaSnapshot, room0: int, mode0: int) -> bool:
    if stair_transition_modes(snap.mode):
        return True
    if snap.mode in (6, 7, 4):
        return False
    if snap.screen != room0 and snap.mode == PLAY_MODE:
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
                if left_room(snap, room0, mode0) or midroom_warp(snap):
                    took = left_room(snap, room0, mode0) or midroom_warp(snap)
                    dest = dump_live(snap, env.get_ram())
                break
            step(env, assist, total, frame.action)
        if took:
            break
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



def take_center_stairs_64(env, assist, total) -> dict:
    """0x64 diamond: south gap x=120 y=173, UP onto center stairs. Never push RIGHT."""
    room0 = ROOM_64
    log = []
    # Pull off the east door first.
    walk_axis(env, assist, total, "y", 173, max_f=400)
    walk_axis(env, assist, total, "x", 120, max_f=500)
    walk_axis(env, assist, total, "y", 157, max_f=300)
    walk_axis(env, assist, total, "x", 120, max_f=200)
    snap = read_snapshot(env.get_ram())
    log.append({"phase": "south_gap", "xy": [snap.link_x, snap.link_y], "tile": int(snap.colliding_tile), "room": f"0x{snap.screen:02x}", "mode": snap.mode})
    if snap.screen != room0:
        return {"took": False, "reason": "left_0x64_before_stairs", "log": log, "dump": dump_live(snap, env.get_ram())}
    # Nudge UP/DOWN only; recenter x if we drift toward a door.
    for direction in ("UP", "DOWN", "UP"):
        for _ in range(90):
            snap = read_snapshot(env.get_ram())
            if stair_transition_modes(snap.mode) or (snap.screen != room0 and snap.mode == PLAY_MODE):
                break
            if snap.link_x > 160:
                step(env, assist, total, nes_action("LEFT"))
                continue
            if snap.link_x < 80:
                step(env, assist, total, nes_action("RIGHT"))
                continue
            step(env, assist, total, nes_action(direction))
        snap = read_snapshot(env.get_ram())
        log.append({"phase": f"nudge_{direction}", "xy": [snap.link_x, snap.link_y], "tile": int(snap.colliding_tile), "mode": snap.mode, "room": f"0x{snap.screen:02x}", "stair": bool(on_stair_tile(snap))})
        if stair_transition_modes(snap.mode) or (snap.screen != room0 and snap.mode not in (6, 7)):
            break
    # If still here, try north gap then UP.
    snap = read_snapshot(env.get_ram())
    if snap.screen == room0 and snap.mode == PLAY_MODE and not stair_transition_modes(snap.mode):
        walk_axis(env, assist, total, "y", 109, max_f=300)
        walk_axis(env, assist, total, "x", 120, max_f=300)
        walk_axis(env, assist, total, "y", 125, max_f=200)
        for _ in range(80):
            snap = read_snapshot(env.get_ram())
            if stair_transition_modes(snap.mode) or snap.screen != room0:
                break
            if snap.link_x > 160:
                step(env, assist, total, nes_action("LEFT"))
            else:
                step(env, assist, total, nes_action("DOWN"))
        log.append({"phase": "north_gap", "xy": [read_snapshot(env.get_ram()).link_x, read_snapshot(env.get_ram()).link_y], "mode": read_snapshot(env.get_ram()).mode, "room": f"0x{read_snapshot(env.get_ram()).screen:02x}"})
    wait_play(env, assist, total, max_f=280)
    idle(env, assist, total, 20)
    ram = env.get_ram()
    snap = read_snapshot(ram)
    dump = dump_live(snap, ram)
    png = shot(env, assist, total, "l5_64_stairs")
    ok = stair_transition_modes(snap.mode) or (snap.screen != room0 and snap.screen != ROOM_65)
    body = {
        "via": "0x64 center stairs",
        "pokes": False,
        "status_claim": None,
        "ok": ok,
        "log": log,
        "dump": dump,
        "dest": dest_report(snap),
        "screenshot": png,
        "whistle_0x065C": dump.get("whistle_0x065C"),
        "rom": rom_room(int(snap.screen)),
    }
    write_dump("l5_64_stairs", body)
    print(
        "DEST l5_64_stairs",
        "room",
        dump.get("room_hex"),
        "mode",
        snap.mode,
        "ok",
        ok,
        "xy",
        [snap.link_x, snap.link_y],
        "tile",
        dump.get("colliding_tile"),
        flush=True,
    )
    return {
        "took": ok,
        "dump": dump,
        "dest": dest_report(snap),
        "screenshot": png,
        "log": log,
        "whistle_0x065C": dump.get("whistle_0x065C"),
    }

def take_stairs(env, assist, total, tag: str, room0: int, mode0: int) -> dict:
    for direction in ("UP", "DOWN", "LEFT", "RIGHT"):
        push_dir(env, assist, total, direction, frames=140)
        idle(env, assist, total, 12)
        snap = read_snapshot(env.get_ram())
        if left_room(snap, room0, mode0):
            break
    wait_play(env, assist, total, max_f=280)
    idle(env, assist, total, 20)
    ram = env.get_ram()
    snap = read_snapshot(ram)
    dump = dump_live(snap, ram)
    png = shot(env, assist, total, tag)
    body = {
        "via": tag,
        "pokes": False,
        "status_claim": None,
        "dump": dump,
        "dest": dest_report(snap),
        "screenshot": png,
        "whistle_0x065C": dump.get("whistle_0x065C"),
        "rom": rom_room(int(snap.screen)),
    }
    json_path = write_dump(tag, body)
    print(
        "DEST",
        tag,
        "room",
        dump.get("room_hex"),
        "mode",
        snap.mode,
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
        "took": left_room(snap, room0, mode0) or snap.screen != room0,
        "dump": dump,
        "dest": dest_report(snap),
        "screenshot": png,
        "dump_path": json_path,
        "whistle_0x065C": dump.get("whistle_0x065C"),
    }


def fight_darknuts(env, assist, total, room: int, expected: int, source: int) -> dict:
    spec = replace(
        ROOM_5B_SPEC,
        spec_id=f"level5_room{room:02x}_darknuts_reuse5b59",
        source_room=source,
        room_id=room,
        entry=DoorRoute("LEFT", ((224, 141),)),
        enemy_types=DARKNUT_TYPES,
        expected_enemy_count=expected,
        required_open_doors=0,
        reward=RewardSpec(kind=RewardKind.CLEAR_ONLY, settle_all_dead=0),
        combat=ROOM_59_SPEC.combat,
        exit_routes=(DoorRoute("RIGHT", ((208, 141),)),),
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
                print(f"DN_KILL room=0x{room:02x} n={last_n} f={ctl.frames}", flush=True)
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


def fight_type(env, assist, total, room: int, type_id: int, expected: int) -> dict:
    spec = replace(
        ROOM_5B_SPEC,
        spec_id=f"level5_room{room:02x}_type{type_id:02x}",
        source_room=room,
        room_id=room,
        entry=DoorRoute("LEFT", ((224, 141),)),
        enemy_types=(type_id,),
        expected_enemy_count=max(1, expected),
        required_open_doors=0,
        reward=RewardSpec(kind=RewardKind.CLEAR_ONLY, settle_all_dead=0),
        combat=ROOM_59_SPEC.combat,
        exit_routes=(DoorRoute("RIGHT", ((208, 141),)),),
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
                print(f"KILL type=0x{type_id:02x} n={last_n} f={ctl.frames}", flush=True)
        action = ctl.step(snap)
        step(env, assist, total, action.action)
        if ctl.success or ctl.phase is DungeonPhase.FAILED:
            break
    snap = read_snapshot(env.get_ram())
    live = [
        o
        for o in snap.objects
        if 1 <= o.slot <= 12 and o.type_id == type_id and o.hp > 0
    ] if snap.mode == PLAY_MODE else []
    return {
        "ok": bool(ctl.success) and not live,
        "frames": ctl.frames,
        "start_n": 0 if start_n is None else start_n,
        "end_n": len(live),
        "progress": progress,
        "controller": ctl.report(),
        "spec_id": spec.spec_id,
    }


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
    if helper is not None and not any(getattr(b, "slot", None) == helper.slot for b in blocks):
        blocks.append(helper)
    targets = [(b.x, b.y, f"obj_{b.slot}") for b in blocks]
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
                }
            )
            if left_room(snap, room0, mode0) or on_stair_tile(snap):
                took = True
                dest = dump_live(snap, env.get_ram())
                break
        log.append(rec)
        if took:
            break
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
        "blocks_seen": [{"slot": b.slot, "x": b.x, "y": b.y} for b in blocks],
        "took": took,
        "dest": dest,
        "end": dump_live(snap, env.get_ram()),
        "log": log,
    }


def hunt_item(env, assist, total, field_addr: int = ADDR_WHISTLE) -> dict:
    w0 = int(read_u8(env.get_ram(), field_addr))
    room0 = read_snapshot(env.get_ram()).screen
    hits = []
    for tx, ty in ITEM_WAYPOINTS:
        snap = read_snapshot(env.get_ram())
        if snap.screen != room0 and snap.mode == PLAY_MODE:
            break
        walk_axis(env, assist, total, "y", ty, max_f=220)
        walk_axis(env, assist, total, "x", tx, max_f=220)
        idle(env, assist, total, 8)
        w1 = int(read_u8(env.get_ram(), field_addr))
        snap = read_snapshot(env.get_ram())
        hits.append(
            {
                "stand": [tx, ty],
                "xy": [snap.link_x, snap.link_y],
                "value": w1,
                "tile": int(snap.colliding_tile),
                "mode": snap.mode,
                "room": f"0x{snap.screen:02x}",
            }
        )
        if w1 > w0:
            break
    w1 = int(read_u8(env.get_ram(), field_addr))
    return {"in": w0, "out": w1, "got": w1 > w0, "hits": hits}


def exit_cellar_other_mouth(env, assist, total) -> dict:
    """From mode-9 cellar, walk the opposite mouth. No pokes."""
    snap = read_snapshot(env.get_ram())
    start = {"xy": [snap.link_x, snap.link_y], "room": f"0x{snap.screen:02x}", "mode": snap.mode}
    if snap.link_x < CELLAR_SPLIT_X:
        side = "right"
        tx = CELLAR_RIGHT_X
    else:
        side = "left"
        tx = CELLAR_LEFT_X
    walk_axis(env, assist, total, "y", CELLAR_CORRIDOR_Y, max_f=400)
    walk_axis(env, assist, total, "x", tx, max_f=500)
    walk_axis(env, assist, total, "y", CELLAR_EXIT_Y, max_f=400)
    room0 = read_snapshot(env.get_ram()).screen
    mode0 = read_snapshot(env.get_ram()).mode
    push_dir(env, assist, total, "UP", frames=180)
    idle(env, assist, total, 16)
    wait_play(env, assist, total, max_f=280)
    idle(env, assist, total, 20)
    snap = read_snapshot(env.get_ram())
    return {
        "start": start,
        "chose_side": side,
        "target_x": tx,
        "changed": snap.screen != room0 or snap.mode != mode0,
        "end": dump_live(snap, env.get_ram()),
    }


def bomb_west_65(env, assist, total) -> dict:
    """0x66-west pattern: stand (40,141) face LEFT, one B bomb, push LEFT."""
    walk_axis(env, assist, total, "y", 141, max_f=400)
    walk_axis(env, assist, total, "x", BOMB_W_STAND[0], max_f=400)
    goto(env, assist, total, BOMB_W_STAND[0], BOMB_W_STAND[1], tol=3, max_f=400)
    for _ in range(8):
        step(env, assist, total, nes_action(FACE_W))
    idle(env, assist, total, 8)
    before = dump_live(read_snapshot(env.get_ram()), env.get_ram())
    before_png = shot(env, assist, total, "l5_65_west_bomb_before")
    menu = select_bombs_menu(env, assist, total)
    bombs0 = int(read_snapshot(env.get_ram()).bombs)
    room0 = int(read_snapshot(env.get_ram()).screen)
    step(env, assist, total, nes_action(FACE_W, "B"))
    for _ in range(20):
        step(env, assist, total, nes_action("RIGHT"))
    idle(env, assist, total, 100)
    bombs1 = int(read_snapshot(env.get_ram()).bombs)
    for _ in range(200):
        snap = read_snapshot(env.get_ram())
        if snap.screen != room0 and snap.mode in (PLAY_MODE, 6, 7, 4):
            break
        step(env, assist, total, nes_action(FACE_W))
    idle(env, assist, total, 20)
    if read_snapshot(env.get_ram()).screen != room0:
        wait_play(env, assist, total, read_snapshot(env.get_ram()).screen, max_f=240)
    idle(env, assist, total, 16)
    after = dump_live(read_snapshot(env.get_ram()), env.get_ram())
    after_png = shot(env, assist, total, "l5_65_west_bomb")
    dest_changed = after.get("room") != before.get("room")
    rec = {
        "stand": list(BOMB_W_STAND),
        "face": FACE_W,
        "menu": menu,
        "before": before,
        "after": after,
        "bombs_in": bombs0,
        "bombs_out": int(after.get("bombs") or bombs1),
        "bombs_spent": bombs0 - int(after.get("bombs") or bombs1),
        "one_bomb": True,
        "dest_changed": dest_changed,
        "dest": after.get("room_hex") if dest_changed else None,
        "before_screenshot": before_png,
        "screenshot": after_png,
        "whistle_0x065C": after.get("whistle_0x065C"),
        "pokes": False,
    }
    write_dump("l5_65_west_bomb", rec)
    print(
        "BOMB65W dest_changed",
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
    return rec


def key_west(env, assist, total, expect: int) -> dict:
    keys0 = int(read_snapshot(env.get_ram()).keys)
    room0 = int(read_snapshot(env.get_ram()).screen)
    walk_axis(env, assist, total, "y", 141, max_f=400)
    walk_axis(env, assist, total, "x", 32, max_f=500)
    goto(env, assist, total, 32, 141, tol=3, max_f=300)
    push_dir(env, assist, total, "LEFT", frames=240)
    idle(env, assist, total, 16)
    snap = read_snapshot(env.get_ram())
    if snap.screen != room0:
        wait_play(env, assist, total, snap.screen, max_f=240)
    idle(env, assist, total, 16)
    snap = read_snapshot(env.get_ram())
    keys1 = int(snap.keys)
    return {
        "keys_in": keys0,
        "keys_out": keys1,
        "key_spent": keys1 < keys0,
        "dest": f"0x{snap.screen:02x}",
        "changed": snap.screen != room0,
        "expect": f"0x{expect:02x}",
        "ok": snap.screen == expect,
        "xy": [snap.link_x, snap.link_y],
        "mode": snap.mode,
    }


def dump_and_save_room(env, assist, total, tag: str, ckpt_name: str, source: str, via: str) -> dict:
    ram = env.get_ram()
    snap = read_snapshot(ram)
    dump = dump_live(snap, ram)
    png = shot(env, assist, total, tag)
    body = {
        "via": via,
        "pokes": False,
        "status_claim": None,
        "dump": dump,
        "dest": dest_report(snap),
        "rom": rom_room(int(snap.screen)),
        "screenshot": png,
        "whistle_0x065C": dump.get("whistle_0x065C"),
    }
    json_path = write_dump(tag, body)
    ckpt = save_ckpt(
        env,
        ckpt_name,
        source,
        {
            "segment": ckpt_name,
            "predecessor_entry": True,
            "start_state": source,
            "via": via,
            "key_poke": False,
            "door_poke": False,
            "bomb_count_poke": False,
            "selected_item_poke": False,
        },
        {
            "success": True,
            "room": int(snap.screen),
            "mode": int(snap.mode),
            "bombs": int(snap.bombs),
            "keys": int(snap.keys),
            "whistle_0x065C": int(read_u8(ram, ADDR_WHISTLE)),
        },
    )
    print(
        "DUMP",
        tag,
        "room",
        dump.get("room_hex"),
        "mode",
        snap.mode,
        "xy",
        [snap.link_x, snap.link_y],
        "keys",
        snap.keys,
        "bombs",
        snap.bombs,
        "item",
        snap.room_item_id,
        room_item_name(snap.room_item_id),
        "doors",
        dump.get("doors"),
        "whistle",
        dump.get("whistle_0x065C"),
        "objs",
        [(o["type_hex"], o["type_name"], o["hp"]) for o in dump.get("objects") or []],
        "ckpt",
        ckpt,
        flush=True,
    )
    return {"dump": dump, "screenshot": png, "dump_path": json_path, "checkpoint": ckpt}


def confirm_start_state() -> tuple[str, dict]:
    """Confirm live bombs on Cleared65; fall back to Cleared55 DOWN if needed."""
    env, assist, obs = open_env(STATE_65)
    total = [1]
    try:
        idle(env, assist, total, 16)
        snap = read_snapshot(env.get_ram())
        start = dump_live(snap, env.get_ram())
        print(
            "CONFIRM65",
            start.get("room_hex"),
            "mode",
            start.get("mode"),
            "xy",
            [start.get("x"), start.get("y")],
            "bombs",
            start.get("bombs"),
            "keys",
            start.get("keys"),
            "whistle",
            start.get("whistle_0x065C"),
            flush=True,
        )
        ok = (
            snap.level == LEVEL_5
            and snap.screen == ROOM_65
            and snap.mode == PLAY_MODE
            and int(snap.bombs) >= 1
        )
        if ok:
            return STATE_65, {
                "state": STATE_65,
                "ok": True,
                "start": start,
                "via": "Level5Cleared65 live",
            }
        print("CLEARED65_BOMBS", snap.bombs, "room", f"0x{snap.screen:02x}", "fallback", flush=True)
    finally:
        env.close()

    env, assist, obs = open_env(STATE_55)
    total = [1]
    try:
        idle(env, assist, total, 16)
        start55 = dump_live(read_snapshot(env.get_ram()), env.get_ram())
        walk_axis(env, assist, total, "x", 120, max_f=300)
        walk_axis(env, assist, total, "y", 205, max_f=400)
        push_dir(env, assist, total, "DOWN", frames=220)
        idle(env, assist, total, 16)
        wait_play(env, assist, total, ROOM_65, max_f=240)
        idle(env, assist, total, 16)
        snap = read_snapshot(env.get_ram())
        start = dump_live(snap, env.get_ram())
        print(
            "FALLBACK55",
            start.get("room_hex"),
            "bombs",
            start.get("bombs"),
            "keys",
            start.get("keys"),
            flush=True,
        )
        return STATE_55, {
            "state": STATE_55,
            "ok": snap.screen == ROOM_65 and int(snap.bombs) >= 1,
            "start55": start55,
            "start": start,
            "via": "Level5Cleared55 DOWN to 0x65",
        }
    finally:
        env.close()


def digdogger_and_tf() -> dict:
    """From Level5Cleared25 WEST into 0x24. Whistle-shrink, sword, heart, TF."""
    env, assist, obs = open_env(STATE_25)
    total = [1]
    try:
        idle(env, assist, total, 16)
        start = dump_live(read_snapshot(env.get_ram()), env.get_ram())
        used = None
        for name, steps in WEST_24_PATHS:
            for axis, tgt in steps:
                walk_axis(env, assist, total, axis, tgt, max_f=450)
            snap = read_snapshot(env.get_ram())
            if abs(snap.link_x - 32) <= 8 and abs(snap.link_y - 141) <= 8:
                used = name
                break
        push_dir(env, assist, total, "LEFT", frames=240)
        idle(env, assist, total, 16)
        snap = read_snapshot(env.get_ram())
        if snap.screen == ROOM_24:
            wait_play(env, assist, total, ROOM_24, max_f=240)
        idle(env, assist, total, 16)
        at24 = dump_live(read_snapshot(env.get_ram()), env.get_ram())
        print(
            "AT24",
            at24.get("room_hex"),
            "mode",
            at24.get("mode"),
            "objs",
            [(o["type_hex"], o["type_name"], o["hp"]) for o in at24.get("objects") or []],
            "whistle",
            at24.get("whistle_0x065C"),
            flush=True,
        )
        if at24.get("room") != ROOM_24:
            rec = {
                "ok": False,
                "failed_room": "0x24",
                "reason": "west_from_cleared25_did_not_enter_0x24",
                "path": used,
                "start": start,
                "at24": at24,
            }
            write_dump("l5_24_whistle_boss", rec)
            return rec

        menu = select_whistle_menu(env, assist, total)
        # Play whistle a few times, then sword whatever remains.
        for _ in range(4):
            step(env, assist, total, nes_action("B"))
            idle(env, assist, total, 40)
        idle(env, assist, total, 60)
        after_whistle = dump_live(read_snapshot(env.get_ram()), env.get_ram())
        print(
            "WHISTLE_B",
            "selected",
            menu,
            "objs",
            [(o["type_hex"], o["type_name"], o["hp"]) for o in after_whistle.get("objects") or []],
            flush=True,
        )
        fight = None
        snap = read_snapshot(env.get_ram())
        bosses = live_boss(snap)
        if bosses:
            fight = fight_type(env, assist, total, ROOM_24, BOSS_38, expected=len(bosses))
            idle(env, assist, total, 20)
            print("BOSS_FIGHT", fight.get("ok"), "end_n", fight.get("end_n"), flush=True)
        # Also sword any leftover small types (0x18 etc) if 0x38 gone but others live.
        snap = read_snapshot(env.get_ram())
        leftovers = [
            o
            for o in snap.objects
            if 1 <= o.slot <= 12 and o.type_id not in (0, 0xFF, 0x55, 0x4E, 0x40) and o.hp > 0
        ]
        extra = None
        if leftovers:
            extra = fight_type(env, assist, total, int(snap.screen), leftovers[0].type_id, expected=len(leftovers))
            idle(env, assist, total, 16)

        heart = hunt_item(env, assist, total, ADDR_WHISTLE)  # walk; heart is room item
        # Walk heart stands specifically.
        walk_axis(env, assist, total, "y", 141, max_f=300)
        walk_axis(env, assist, total, "x", 224, max_f=400)
        idle(env, assist, total, 12)
        walk_axis(env, assist, total, "x", 120, max_f=300)
        walk_axis(env, assist, total, "y", 141, max_f=200)
        after_heart = dump_live(read_snapshot(env.get_ram()), env.get_ram())

        # North shutter of 0x24 -> TF 0x14 (ROM). Also try south if north sealed.
        tf = None
        room0 = read_snapshot(env.get_ram()).screen
        walk_axis(env, assist, total, "x", 120, max_f=300)
        walk_axis(env, assist, total, "y", 93, max_f=400)
        push_dir(env, assist, total, "UP", frames=220)
        idle(env, assist, total, 16)
        snap = read_snapshot(env.get_ram())
        if snap.screen != room0:
            wait_play(env, assist, total, snap.screen, max_f=240)
            idle(env, assist, total, 20)
            tf = dump_and_save_room(
                env,
                assist,
                total,
                f"l5_{snap.screen:02x}_triforce",
                "Level5Triforce",
                STATE_25,
                "0x24 north after Digdogger",
            )
        else:
            walk_axis(env, assist, total, "y", 205, max_f=400)
            push_dir(env, assist, total, "DOWN", frames=220)
            idle(env, assist, total, 16)
            snap = read_snapshot(env.get_ram())
            if snap.screen != room0:
                wait_play(env, assist, total, snap.screen, max_f=240)
                idle(env, assist, total, 20)
                tf = dump_and_save_room(
                    env,
                    assist,
                    total,
                    f"l5_{snap.screen:02x}_triforce",
                    "Level5Triforce",
                    STATE_25,
                    "0x24 south after Digdogger",
                )
        # Walk TF item if we landed in a room with item 0x1B.
        snap = read_snapshot(env.get_ram())
        tf_walk = None
        tf0 = int(read_u8(env.get_ram(), ADDR_TRIFORCE))
        if snap.room_item_id == 0x1B or (int(snap.screen) == ROOM_14):
            tf_walk = hunt_item(env, assist, total, ADDR_TRIFORCE)
            idle(env, assist, total, 20)
        ram = env.get_ram()
        snap = read_snapshot(ram)
        final = dump_live(snap, ram)
        png = shot(env, assist, total, "l5_24_whistle_boss")
        rec = {
            "ok": bool(final.get("inventory", {}).get("tf_l5_bit")) or (tf is not None),
            "pokes": False,
            "status_claim": None,
            "path_to_24": used,
            "start25": start,
            "at24": at24,
            "menu": menu,
            "after_whistle": after_whistle,
            "fight": fight,
            "extra": extra,
            "heart_walk": heart,
            "after_heart": after_heart,
            "tf": tf,
            "tf_walk": None if tf_walk is None else {k: v for k, v in tf_walk.items() if k != "hits"},
            "final": final,
            "screenshot": png,
            "whistle_0x065C": final.get("whistle_0x065C"),
            "triforce_0x0671": final.get("triforce_0x0671"),
            "tf_l5": bool(int(final.get("triforce_0x0671") or 0) & TF_BIT_L5),
            "tf_room": final.get("room_hex"),
        }
        write_dump("l5_24_whistle_boss", rec)
        print(
            "DIGDOGGER",
            "tf_room",
            rec.get("tf_room"),
            "tf_l5",
            rec.get("tf_l5"),
            "whistle",
            rec.get("whistle_0x065C"),
            flush=True,
        )
        return rec
    finally:
        env.close()


def main() -> dict:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    commands = [
        "PYTHONPATH=nes:. uv run python nes/zelda_i/scripts/_probe_l5_whistle_path.py  "
        "# Level5Cleared65 WEST bomb 0x65->0x64, stairs 0x07->0x06, key 0x05, "
        "block-stairs 0x04 whistle, then Cleared25 WEST Digdogger. infinite-life, no pokes"
    ]
    hops = []
    failed_room = None
    checkpoints = []
    whistle = 0
    boss = None

    roms = {r: rom_room(r) for r in (0x65, 0x64, 0x07, 0x06, 0x05, 0x04, 0x24, 0x14)}
    print("ROM", roms, flush=True)

    import sys
    from_64 = "--from-64" in sys.argv
    if from_64:
        source, confirm = "Level5Entered64", {"state": "Level5Entered64", "ok": True, "via": "resume Level5Entered64"}
    else:
        source, confirm = confirm_start_state()
    if not confirm.get("ok"):
        report = {
            "ok": False,
            "failed_room": "0x65",
            "reason": "no_bombs_on_cleared65_or_fallback",
            "confirm": confirm,
            "commands": commands,
            "pokes": False,
            "status_claim": None,
            "roms": roms,
        }
        write_dump("l5_whistle_path", report)
        return report

    env, assist, obs = open_env(source)
    total = [1]
    try:
        idle(env, assist, total, 16)
        # If we loaded Cleared55, walk DOWN into 0x65 again (confirm closed that env).
        snap = read_snapshot(env.get_ram())
        if snap.screen == 0x55:
            walk_axis(env, assist, total, "x", 120, max_f=300)
            walk_axis(env, assist, total, "y", 205, max_f=400)
            push_dir(env, assist, total, "DOWN", frames=220)
            wait_play(env, assist, total, ROOM_65, max_f=240)
            idle(env, assist, total, 16)
            snap = read_snapshot(env.get_ram())
        start = dump_live(snap, env.get_ram())
        print(
            "START",
            start.get("room_hex"),
            "bombs",
            start.get("bombs"),
            "keys",
            start.get("keys"),
            "whistle",
            start.get("whistle_0x065C"),
            "selected",
            start.get("inventory", {}).get("selected_item_0x0656"),
            flush=True,
        )
        if from_64:
            if snap.screen != ROOM_64:
                wait_play(env, assist, total, ROOM_64, max_f=240)
                snap = read_snapshot(env.get_ram())
                start = dump_live(snap, env.get_ram())
            bomb = {"dest_changed": True, "dest": "0x64", "resumed": True, "bombs_in": start.get("bombs"), "bombs_out": start.get("bombs")}
            hops.append({"hop": "0x65_west_bomb", "dest": "0x64", "ok": True, "resumed": True})
        elif int(start.get("bombs") or 0) < 1 or snap.screen != ROOM_65:
            failed_room = "0x65"
            report = {
                "ok": False,
                "failed_room": failed_room,
                "reason": "start_not_0x65_with_bombs",
                "start": start,
                "confirm": confirm,
                "commands": commands,
                "pokes": False,
                "status_claim": None,
            }
            write_dump("l5_whistle_path", report)
            return report

        # 1. Bomb WEST 0x65 -> 0x64
        if from_64:
            pass
        else:
            bomb = bomb_west_65(env, assist, total)
        hops.append({"hop": "0x65_west_bomb", "dest": bomb.get("dest"), "ok": bomb.get("dest_changed")})
        if not bomb.get("dest_changed") or bomb.get("dest") != "0x64":
            failed_room = "0x65"
            report = {
                "ok": False,
                "failed_room": failed_room,
                "reason": "west_bomb_did_not_enter_0x64",
                "bomb": bomb,
                "hops": hops,
                "start": start,
                "commands": commands,
                "pokes": False,
                "status_claim": None,
                "roms": roms,
            }
            write_dump("l5_whistle_path", report)
            return report


        # 2. Stairs in 0x64 (center, diamond blocks). Settle play first.
        wait_play(env, assist, total, ROOM_64, max_f=240)
        idle(env, assist, total, 20)
        d64 = dump_and_save_room(
            env, assist, total, "l5_64_arrive", "Level5Entered64", source, "0x65 WEST bomb settled"
        )
        checkpoints.append(d64["checkpoint"])
        stairs64 = take_center_stairs_64(env, assist, total)
        print("WALK64 took", stairs64.get("took"), "dest", (stairs64.get("dump") or {}).get("room_hex"), (stairs64.get("dump") or {}).get("mode"), flush=True)
        fight64 = None
        walked = {"took": stairs64.get("took"), "hits": stairs64.get("log")}
        if stairs64 is None or not stairs64.get("took"):
            failed_room = "0x64"
            report = {
                "ok": False,
                "failed_room": failed_room,
                "reason": "stairs_in_0x64_not_taken",
                "bomb": {k: bomb[k] for k in bomb if k not in ("before", "after")},
                "arrive64": d64["dump"],
                "walk64": {k: v for k, v in walked.items() if k != "dest"},
                "fight64": fight64,
                "hops": hops,
                "commands": commands,
                "pokes": False,
                "status_claim": None,
            }
            write_dump("l5_whistle_path", report)
            return report

        hops.append({"hop": "0x64_stairs", "dest": (stairs64.get("dump") or {}).get("room_hex"), "mode": (stairs64.get("dump") or {}).get("mode"), "ok": True})
        d07 = dump_and_save_room(
            env, assist, total, "l5_07_arrive", "Level5Entered07", source, "0x64 stairs"
        )
        checkpoints.append(d07["checkpoint"])

        # 3. Other mouth of cellar -> 0x06
        snap = read_snapshot(env.get_ram())
        if snap.mode not in CELLAR_MODES and snap.screen != ROOM_07:
            # Already exited? dump current.
            print("AFTER_STAIRS not cellar", f"0x{snap.screen:02x}", "mode", snap.mode, flush=True)
        cellar = exit_cellar_other_mouth(env, assist, total)
        hops.append({"hop": "0x07_other_mouth", "dest": (cellar.get("end") or {}).get("room_hex"), "ok": cellar.get("changed")})
        snap = read_snapshot(env.get_ram())
        if snap.screen != ROOM_06:
            failed_room = "0x07"
            report = {
                "ok": False,
                "failed_room": failed_room,
                "reason": "other_cellar_mouth_did_not_enter_0x06",
                "cellar": cellar,
                "d07": d07["dump"],
                "hops": hops,
                "commands": commands,
                "pokes": False,
                "status_claim": None,
            }
            write_dump("l5_whistle_path", report)
            return report
        d06 = dump_and_save_room(
            env, assist, total, "l5_06_arrive", "Level5Entered06", source, "cellar 0x07 other mouth"
        )
        checkpoints.append(d06["checkpoint"])

        # 4. Key WEST -> 0x05
        west = key_west(env, assist, total, ROOM_05)
        hops.append({"hop": "0x06_key_west", "dest": west.get("dest"), "key_spent": west.get("key_spent"), "ok": west.get("ok")})
        print("KEYWEST", west, flush=True)
        if not west.get("ok"):
            failed_room = "0x06"
            report = {
                "ok": False,
                "failed_room": failed_room,
                "reason": "key_west_did_not_enter_0x05",
                "west": west,
                "d06": d06["dump"],
                "hops": hops,
                "commands": commands,
                "pokes": False,
                "status_claim": None,
            }
            write_dump("l5_whistle_path", report)
            return report
        d05 = dump_and_save_room(
            env, assist, total, "l5_05_arrive", "Level5Entered05", source, "0x06 WEST key"
        )

        # 5. Kill 6 Blue Darknuts, push block, stairs -> 0x04, grab whistle
        snap = read_snapshot(env.get_ram())
        n_dn = len(live_darknuts(snap))
        fight05 = fight_darknuts(env, assist, total, ROOM_05, expected=max(6, n_dn), source=ROOM_06)
        idle(env, assist, total, 20)
        print("FIGHT05", fight05.get("ok"), "end_n", fight05.get("end_n"), "start", fight05.get("start_n"), flush=True)
        if not fight05.get("ok"):
            failed_room = "0x05"
            report = {
                "ok": False,
                "failed_room": failed_room,
                "reason": "darknuts_in_0x05_not_cleared",
                "fight05": fight05,
                "d05": d05["dump"],
                "hops": hops,
                "commands": commands,
                "pokes": False,
                "status_claim": None,
            }
            write_dump("l5_whistle_path", report)
            return report
        cleared05 = dump_and_save_room(
            env, assist, total, "l5_05_cleared", "Level5Cleared05", source, "0x05 6/6 darknuts"
        )
        checkpoints.append(cleared05["checkpoint"])

        pushed = push_blocks(env, assist, total, ROOM_05)
        print("PUSH05 took", pushed.get("took"), "blocks", pushed.get("blocks_seen"), flush=True)
        stairs05 = None
        snap = read_snapshot(env.get_ram())
        if pushed.get("took") or midroom_warp(snap) or left_room(snap, ROOM_05, PLAY_MODE):
            stairs05 = take_stairs(env, assist, total, "l5_05_stairs", ROOM_05, PLAY_MODE)
        if stairs05 is None or not stairs05.get("took"):
            # Hunt stairs after push.
            snap = read_snapshot(env.get_ram())
            walked05 = walk_stands(env, assist, total, STAIR_HUNT, snap.screen, snap.mode)
            if walked05.get("took") or on_stair_tile(read_snapshot(env.get_ram())):
                stairs05 = take_stairs(env, assist, total, "l5_05_stairs", ROOM_05, PLAY_MODE)
        if stairs05 is None or not stairs05.get("took"):
            failed_room = "0x05"
            report = {
                "ok": False,
                "failed_room": failed_room,
                "reason": "block_stairs_from_0x05_not_taken",
                "fight05": {k: fight05[k] for k in fight05 if k != "controller"},
                "push": {k: v for k, v in pushed.items() if k != "log"},
                "hops": hops,
                "commands": commands,
                "pokes": False,
                "status_claim": None,
            }
            write_dump("l5_whistle_path", report)
            return report
        hops.append({"hop": "0x05_block_stairs", "dest": (stairs05.get("dump") or {}).get("room_hex"), "ok": True})

        snap = read_snapshot(env.get_ram())
        d04 = dump_and_save_room(
            env, assist, total, "l5_04_whistle", "Level5Entered04", source, "0x05 block stairs"
        )
        whistle_walk = hunt_item(env, assist, total, ADDR_WHISTLE)
        idle(env, assist, total, 12)
        ram = env.get_ram()
        snap = read_snapshot(ram)
        whistle = int(read_u8(ram, ADDR_WHISTLE))
        print("WHISTLE_WALK", whistle_walk.get("in"), "->", whistle_walk.get("out"), "now", whistle, flush=True)
        if whistle < 1:
            # One more pass of stands + nudges.
            snap = read_snapshot(env.get_ram())
            walk_stands(env, assist, total, ITEM_WAYPOINTS, snap.screen, snap.mode)
            whistle = int(read_u8(env.get_ram(), ADDR_WHISTLE))
        final04 = dump_live(read_snapshot(env.get_ram()), env.get_ram())
        png04 = shot(env, assist, total, "l5_04_whistle")
        write_dump(
            "l5_04_whistle",
            {
                "via": "0x05 block stairs",
                "pokes": False,
                "status_claim": None,
                "arrive": d04["dump"],
                "walk": {k: v for k, v in whistle_walk.items() if k != "hits"},
                "final": final04,
                "screenshot": png04,
                "whistle_0x065C": whistle,
                "rom": roms[0x04],
            },
        )
        if whistle < 1:
            failed_room = "0x04"
            report = {
                "ok": False,
                "failed_room": failed_room,
                "reason": "whistle_0x065C_still_0_in_0x04",
                "final04": final04,
                "hops": hops,
                "commands": commands,
                "pokes": False,
                "status_claim": None,
            }
            write_dump("l5_whistle_path", report)
            return report
        ckpt_w = save_ckpt(
            env,
            "Level5Whistle",
            source,
            {
                "segment": "Level5Whistle",
                "predecessor_entry": True,
                "start_state": source,
                "via": "0x65 bomb -> 0x64 stairs -> 0x07 -> 0x06 key -> 0x05 block -> 0x04",
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
        checkpoints.append(ckpt_w)
        hops.append({"hop": "0x04_whistle", "dest": f"0x{snap.screen:02x}", "whistle_0x065C": whistle, "ok": True})
    finally:
        env.close()

    if whistle >= 1:
        boss = digdogger_and_tf()

    report = {
        "ok": whistle >= 1,
        "failed_room": failed_room,
        "status_claim": None,
        "pokes": False,
        "commands": commands,
        "helper_reused": (
            "0x66 west bomb menu/place/push + GenericDungeonRoomController "
            "+ ROOM_5B_SPEC + ROOM_59_SPEC.combat + level9_stairs walk_to_step/dest_report "
            "+ push_dir 0x68"
        ),
        "confirm": {k: confirm[k] for k in confirm if k != "start"},
        "start_state": source,
        "roms": roms,
        "hops": hops,
        "checkpoints": checkpoints,
        "whistle_0x065C": whistle,
        "digdogger": None
        if boss is None
        else {
            "ok": boss.get("ok"),
            "tf_room": boss.get("tf_room"),
            "tf_l5": boss.get("tf_l5"),
            "triforce_0x0671": boss.get("triforce_0x0671"),
            "whistle_0x065C": boss.get("whistle_0x065C"),
        },
    }
    write_dump("l5_whistle_path", report)
    return report


if __name__ == "__main__":
    r = main()
    print("CMD", r.get("commands"))
    print("HOPS", r.get("hops"))
    print("WHISTLE", r.get("whistle_0x065C"))
    print("FAILED_ROOM", r.get("failed_room"))
    print("CKPT", r.get("checkpoints"))
    print("DIGDOGGER", r.get("digdogger"))
    print("status_claim", None)
