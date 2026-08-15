"""Clear L5 0x34 6x Gibdo from Level5Cleared25 via 0x24 (no Digdogger fight).

Start: Level5Cleared25. WEST into 0x24 (do not fight type 0x38), SOUTH into 0x34.
Wait play mode 5. Reuse GenericDungeonRoomController + ROOM_66_SPEC Gibdo combat.
6x Gibdo 0x30 HP=112, item 0x03, ROM N=shutter S/W/E=wall secret=all_dead.

After 6/6:
- Dump cur_opened_doors, open_doorway_mask, N/S/E/W
- Dump tiles under/near Link and room (0x68, 0x70-0x73, cellar)
- If stairs appear, walk them, dump dest, check 0x065C
- Level5Cleared34 only if 6/6 dead
- Screenshot

Then re-scan Level5Cleared37 / 27 / 26 for 0x68 / stair tiles (load, scan, no re-clear).

If whistle still 0: leave dungeon (near-entrance 0x66→0x76 south) and attempt
blue candle. Report exact blocker if shop path / rupees are not ready.
No pokes, no Clean STATUS, no east67, no 0x65 bombs, no 0x15, no 0x38 fight.
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
from zelda_i.dungeon_ids import object_name
from zelda_i.dungeon_ops import idle, push_dir
from zelda_i.dungeon_trace import compact_snapshot, write_state_provenance
from zelda_i.level5_dungeon import (
    GIBDO_OBJECT_TYPE,
    LEVEL_5,
    ROOM_66_SPEC,
    ROOM_ITEM_NONE,
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

STATE = "Level5Cleared25"
ROOM_25 = 0x25
ROOM_24 = 0x24
ROOM_34 = 0x34
ROOM_66 = 0x66
ROOM_76 = 0x76
BOSS_38 = 0x38
MAX_FIGHT_FRAMES = 28000
PUSHABLE_BLOCK = 0x68
STAIR_TILE_LO = 0x70
STAIR_TILE_HI = 0x73
BLACK_MOUTH_TILE = 0x24
CELLAR_MODES = (9, 10, 11, 16)
CANDLE_SHOP_PRICE = 60

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

WEST_24_PATHS = (
    ("y141_west", (("y", 141), ("x", 32))),
    ("south189_west", (("y", 189), ("x", 32), ("y", 141))),
    ("north109_west", (("y", 109), ("x", 80), ("y", 141), ("x", 32))),
    ("east208_south189_west", (("x", 208), ("y", 189), ("x", 32), ("y", 141))),
    ("south173_west64", (("y", 173), ("x", 64), ("y", 141), ("x", 32))),
)

TILE_STANDS: tuple[tuple[int, int], ...] = (
    (120, 141),
    (96, 141),
    (144, 141),
    (80, 141),
    (160, 141),
    (64, 141),
    (176, 141),
    (48, 141),
    (192, 141),
    (112, 141),
    (128, 141),
    (120, 125),
    (120, 157),
    (120, 109),
    (120, 173),
    (80, 125),
    (160, 125),
    (80, 157),
    (160, 157),
    (64, 117),
    (176, 117),
    (64, 165),
    (176, 165),
    (96, 117),
    (144, 117),
    (96, 165),
    (144, 165),
    (120, 93),
    (120, 189),
    (208, 141),
    (32, 141),
    (208, 96),
    (48, 96),
)

BLOCK_PUSHES: tuple[tuple[int, int, str], ...] = (
    (120, 141, "LEFT"),
    (120, 141, "RIGHT"),
    (120, 141, "UP"),
    (120, 141, "DOWN"),
    (96, 141, "LEFT"),
    (144, 141, "RIGHT"),
    (80, 141, "LEFT"),
    (160, 141, "RIGHT"),
    (120, 125, "UP"),
    (120, 157, "DOWN"),
)

RESCAN_ROOMS = (
    ("Level5Cleared37", 0x37, "l5_37_rescan"),
    ("Level5Cleared27", 0x27, "l5_27_rescan"),
    ("Level5Cleared26", 0x26, "l5_26_rescan"),
)


def make_34_spec() -> DungeonRoomSpec:
    """ROOM_66_SPEC combat / liveness, retargeted to 0x34 6x Gibdo."""
    return replace(
        ROOM_66_SPEC,
        spec_id="level5_room34_gibdos_reuse66",
        source_room=ROOM_24,
        room_id=ROOM_34,
        entry=DoorRoute("DOWN", ((120, 93),)),
        expected_enemy_count=6,
        required_open_doors=0,
        reward=RewardSpec(kind=RewardKind.CLEAR_ONLY),
        room_item_id=ROOM_ITEM_NONE,
        exit_routes=(DoorRoute("UP", ((120, 93),)),),
        max_frames=MAX_FIGHT_FRAMES,
        level=LEVEL_5,
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
        "rupees": int(snap.rupees),
    }


def stair_tile(tile: int) -> bool:
    return STAIR_TILE_LO <= int(tile) <= STAIR_TILE_HI or int(tile) == BLACK_MOUTH_TILE


def interesting_tile(tile: int) -> bool:
    t = int(tile)
    return t == PUSHABLE_BLOCK or stair_tile(t)


def dump_live(snap: ZeldaSnapshot, ram) -> dict:
    compact = compact_snapshot(snap)
    compact["doors"] = decode_doors(snap.cur_opened_doors)
    compact["doorway_mask"] = decode_doors(snap.open_doorway_mask)
    compact["room_hex"] = f"0x{snap.screen:02x}"
    compact["next_room_hex"] = f"0x{snap.next_screen:02x}"
    compact["inventory"] = inv_block(ram)
    compact["whistle_0x065C"] = int(read_u8(ram, ADDR_WHISTLE))
    compact["candle_0x065B"] = int(read_u8(ram, ADDR_CANDLE))
    compact["colliding_tile"] = int(snap.colliding_tile)
    compact["stair_tile"] = stair_tile(snap.colliding_tile)
    compact["cellar_mode"] = snap.mode in CELLAR_MODES
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


def live_gibdos(snap: ZeldaSnapshot) -> list:
    if snap.mode != PLAY_MODE:
        return []
    return [
        obj
        for obj in snap.objects
        if 1 <= obj.slot <= 12 and obj.type_id == GIBDO_OBJECT_TYPE and obj.hp > 0
    ]


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
        if snap.mode in CELLAR_MODES:
            return True
        if axis == "x":
            if abs(snap.link_x - target) <= 1:
                return True
            step(env, assist, total, nes_action("RIGHT" if snap.link_x < target else "LEFT"))
        else:
            if abs(snap.link_y - target) <= 1:
                return True
            step(env, assist, total, nes_action("DOWN" if snap.link_y < target else "UP"))
        pos = (snap.link_x, snap.link_y)
        if pos == last:
            stall += 1
            if stall >= 40:
                return False
        else:
            stall = 0
        last = pos
    return False


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


def align_door(env, assist, total, tx: int, ty: int = 141, frames: int = 24) -> list[int]:
    for _ in range(frames):
        snap = read_snapshot(env.get_ram())
        if abs(snap.link_x - tx) <= 2 and abs(snap.link_y - ty) <= 2:
            break
        if abs(snap.link_y - ty) > 2:
            step(env, assist, total, nes_action("DOWN" if snap.link_y < ty else "UP"))
        else:
            step(env, assist, total, nes_action("LEFT" if snap.link_x > tx else "RIGHT"))
    snap = read_snapshot(env.get_ram())
    return [snap.link_x, snap.link_y]


def shot(env, assist, total, name: str):
    png = RECORDINGS_DIR / f"{name}.png"
    obs, *_ = env.step(nes_idle_action())
    total[0] += 1
    assist.apply_env(env, frame=total[0])
    save_rgb_png(obs, png)
    return str(png.resolve())


def walk_25_west_24_south_34(env, assist, total) -> dict:
    """Proven 0x25 WEST → 0x24 (no fight) → SOUTH → 0x34."""
    log = []
    snap = read_snapshot(env.get_ram())
    start = dump_live(snap, env.get_ram())
    used = None
    for name, steps in WEST_24_PATHS:
        for axis, tgt in steps:
            ok = walk_axis(env, assist, total, axis, tgt, max_f=450)
            snap = read_snapshot(env.get_ram())
            rec = {
                "path": name,
                "step": f"axis:{axis}:{tgt}",
                "ok": ok,
                "xy": [snap.link_x, snap.link_y],
                "room": f"0x{snap.screen:02x}",
            }
            log.append(rec)
            print("NAV25", rec, flush=True)
        snap = read_snapshot(env.get_ram())
        if abs(snap.link_x - 32) <= 8 and abs(snap.link_y - 141) <= 8:
            used = name
            break
    align_door(env, assist, total, 32, 141, frames=28)
    room0 = read_snapshot(env.get_ram()).screen
    push_dir(env, assist, total, "LEFT", frames=240)
    idle(env, assist, total, 16)
    snap = read_snapshot(env.get_ram())
    if snap.screen == ROOM_24:
        wait_play(env, assist, total, ROOM_24, max_f=240)
    idle(env, assist, total, 12)
    at24 = dump_live(read_snapshot(env.get_ram()), env.get_ram())
    fought = any(
        o.get("type_id") == BOSS_38 and o.get("hp", 240) < 240
        for o in at24.get("objects") or []
    )
    print(
        "AT24",
        at24.get("room_hex"),
        "mode",
        at24.get("mode"),
        "mask",
        at24.get("doorway_mask"),
        "fought",
        fought,
        flush=True,
    )
    # SOUTH only. East-wall then south mouth — do not engage 0x38.
    walk_axis(env, assist, total, "x", 208, max_f=300)
    walk_axis(env, assist, total, "y", 205, max_f=400)
    walk_axis(env, assist, total, "x", 120, max_f=300)
    align_door(env, assist, total, 120, 205, frames=20)
    room1 = read_snapshot(env.get_ram()).screen
    push_dir(env, assist, total, "DOWN", frames=220)
    idle(env, assist, total, 16)
    snap = read_snapshot(env.get_ram())
    if snap.screen != room1:
        wait_play(env, assist, total, snap.screen, max_f=240)
    idle(env, assist, total, 20)
    after = dump_live(read_snapshot(env.get_ram()), env.get_ram())
    fought_after = fought or any(
        o.get("type_id") == BOSS_38 and o.get("hp", 240) < 240
        for o in after.get("objects") or []
    )
    return {
        "start25": start,
        "west_path": used,
        "log": log,
        "at24": at24,
        "after": after,
        "changed": after.get("room") != at24.get("room"),
        "dest_room": after.get("room_hex"),
        "fought_0x38": fought_after,
        "from_room0": f"0x{room0:02x}",
    }


def fight_room(env, assist, total, spec: DungeonRoomSpec) -> dict:
    ctl = GenericDungeonRoomController(spec)
    obs = None
    start_live = None
    last_n = None
    progress = []
    for _ in range(spec.max_frames):
        snap = read_snapshot(env.get_ram())
        if snap.mode == PLAY_MODE and snap.screen == spec.room_id:
            live = spec.live_enemies(snap)
            if start_live is None:
                start_live = live
                last_n = len(live)
                progress.append({"f": ctl.frames, "n": len(live), "hps": [o.hp for o in live]})
            elif len(live) != last_n:
                last_n = len(live)
                progress.append({"f": ctl.frames, "n": len(live), "hps": [o.hp for o in live]})
                print(
                    f"KILL n={len(live)} f={ctl.frames} hps={[o.hp for o in live]}",
                    flush=True,
                )
        action = ctl.step(snap)
        obs = step(env, assist, total, action.action)
        if ctl.success or ctl.phase is DungeonPhase.FAILED:
            break
    snap = read_snapshot(env.get_ram())
    live = spec.live_enemies(snap) if snap.mode == PLAY_MODE else ()
    start_n = 0 if start_live is None else len(start_live)
    return {
        "obs": obs,
        "ok": bool(ctl.success) and snap.screen == spec.room_id and not live,
        "frames": ctl.frames,
        "start_n": start_n,
        "end_n": len(live),
        "kills": start_n - len(live),
        "end_hps": [o.hp for o in live],
        "progress": progress,
        "controller": ctl.report(),
        "spec_id": spec.spec_id,
        "controller_cls": "GenericDungeonRoomController",
        "reused": "GenericDungeonRoomController + ROOM_66_SPEC combat",
        "max_frames": spec.max_frames,
        "combat": {
            "patrol": list(spec.combat.patrol),
            "engage_distance": spec.combat.engage_distance,
            "engage_attack_period": spec.combat.engage_attack_period,
            "engage_attack_hold": spec.combat.engage_attack_hold,
            "patrol_attack_period": spec.combat.patrol_attack_period,
            "patrol_attack_hold": spec.combat.patrol_attack_hold,
            "attack_phase": spec.combat.attack_phase,
        },
    }


def scan_tiles(env, assist, total, room: int, tag: str) -> dict:
    """Walk stands; record colliding tiles. No position pokes."""
    ram = env.get_ram()
    snap = read_snapshot(ram)
    room0 = snap.screen
    mode0 = snap.mode
    blocks = [
        {
            "slot": obj.slot,
            "x": obj.x,
            "y": obj.y,
            "hp": obj.hp,
            "state": obj.state,
        }
        for obj in snap.objects
        if 1 <= obj.slot <= 12 and obj.type_id == PUSHABLE_BLOCK
    ]
    whistle0 = int(read_u8(ram, ADDR_WHISTLE))
    notes = []
    if blocks:
        notes.append(f"pushable_blocks_{len(blocks)}")
    else:
        notes.append("no_0x68_object")
    if stair_tile(snap.colliding_tile):
        notes.append(f"already_on_stair_tile_0x{int(snap.colliding_tile):02x}")
    if snap.mode in CELLAR_MODES:
        notes.append(f"already_cellar_mode_{snap.mode}")

    pushes = []
    candidates = list(BLOCK_PUSHES)
    for blk in blocks:
        candidates.insert(0, (blk["x"], blk["y"], "LEFT"))
        candidates.insert(1, (blk["x"], blk["y"], "UP"))
    seen = set()
    for tx, ty, direction in candidates:
        key = (tx, ty, direction)
        if key in seen:
            continue
        seen.add(key)
        if len(pushes) >= 8:
            break
        snap = read_snapshot(env.get_ram())
        if snap.screen != room0 or snap.mode in CELLAR_MODES:
            notes.append("left_room_during_push")
            break
        walk_axis(env, assist, total, "y", ty, max_f=220)
        walk_axis(env, assist, total, "x", tx, max_f=220)
        for _ in range(36):
            step(env, assist, total, nes_action(direction))
        idle(env, assist, total, 8)
        ram = env.get_ram()
        snap = read_snapshot(ram)
        rec = {
            "stand": [tx, ty],
            "dir": direction,
            "xy": [snap.link_x, snap.link_y],
            "tile": int(snap.colliding_tile),
            "tile_hex": f"0x{int(snap.colliding_tile):02x}",
            "mode": snap.mode,
            "room": f"0x{snap.screen:02x}",
            "whistle": int(read_u8(ram, ADDR_WHISTLE)),
            "stair_tile": stair_tile(snap.colliding_tile),
            "interesting": interesting_tile(snap.colliding_tile),
        }
        pushes.append(rec)
        print("PUSH", tag, rec, flush=True)
        if snap.screen != room0 or snap.mode in CELLAR_MODES or stair_tile(snap.colliding_tile):
            notes.append("push_changed_room_or_stairs")
            break

    stands = []
    tile_hits = []
    for tx, ty in TILE_STANDS:
        snap = read_snapshot(env.get_ram())
        if snap.screen != room0 and snap.mode == PLAY_MODE:
            notes.append("left_room_during_stands")
            break
        if snap.mode in CELLAR_MODES:
            notes.append("cellar_during_stands")
            break
        walk_axis(env, assist, total, "y", ty, max_f=200)
        walk_axis(env, assist, total, "x", tx, max_f=200)
        idle(env, assist, total, 6)
        ram = env.get_ram()
        snap = read_snapshot(ram)
        rec = {
            "stand": [tx, ty],
            "xy": [snap.link_x, snap.link_y],
            "tile": int(snap.colliding_tile),
            "tile_hex": f"0x{int(snap.colliding_tile):02x}",
            "mode": snap.mode,
            "room": f"0x{snap.screen:02x}",
            "stair_tile": stair_tile(snap.colliding_tile),
            "interesting": interesting_tile(snap.colliding_tile),
        }
        stands.append(rec)
        if interesting_tile(snap.colliding_tile) or snap.mode in CELLAR_MODES:
            tile_hits.append(rec)
            notes.append(f"hit_tile_0x{int(snap.colliding_tile):02x}_at_{tx}_{ty}")
            for direction in ("UP", "DOWN", "LEFT", "RIGHT"):
                room_before = snap.screen
                mode_before = snap.mode
                push_dir(env, assist, total, direction, frames=80)
                idle(env, assist, total, 12)
                snap = read_snapshot(env.get_ram())
                if snap.screen != room_before or snap.mode != mode_before:
                    notes.append(
                        f"stairs_took_{direction}_to_0x{snap.screen:02x}_m{snap.mode}"
                    )
                    break
            break

    ram = env.get_ram()
    snap = read_snapshot(ram)
    whistle1 = int(read_u8(ram, ADDR_WHISTLE))
    dest = None
    if snap.screen != room0 or snap.mode in CELLAR_MODES:
        dest = {
            "room": f"0x{snap.screen:02x}",
            "mode": snap.mode,
            "xy": [snap.link_x, snap.link_y],
            "whistle_0x065C": whistle1,
            "dump": dump_live(snap, ram),
        }
    tiles_seen = sorted({s["tile"] for s in stands} | {p["tile"] for p in pushes})
    stair_hits = [s for s in stands if s.get("stair_tile")] + [
        p for p in pushes if p.get("stair_tile")
    ]
    block_tiles = [s for s in stands if s.get("tile") == PUSHABLE_BLOCK]
    return {
        "tag": tag,
        "room": f"0x{room:02x}",
        "room0": f"0x{room0:02x}",
        "mode0": mode0,
        "blocks_0x68": blocks,
        "block_tiles_0x68": block_tiles,
        "stair_hits": stair_hits,
        "tile_hits": tile_hits,
        "tiles_seen": [f"0x{t:02x}" for t in tiles_seen],
        "tiles_seen_int": tiles_seen,
        "has_0x68": bool(blocks) or bool(block_tiles),
        "has_stairs_tile": bool(stair_hits),
        "cellar": snap.mode in CELLAR_MODES,
        "whistle_in": whistle0,
        "whistle_out": whistle1,
        "whistle_got": whistle1 > whistle0,
        "end_tile": int(snap.colliding_tile),
        "end_mode": snap.mode,
        "end_room": f"0x{snap.screen:02x}",
        "notes": notes,
        "pushes": pushes,
        "stands": stands,
        "dest": dest,
        "pokes": False,
    }


def take_stairs_if_open(env, assist, total, tag: str) -> dict | None:
    ram = env.get_ram()
    snap = read_snapshot(ram)
    if snap.mode not in CELLAR_MODES and not stair_tile(snap.colliding_tile):
        return None
    room0 = snap.screen
    mode0 = snap.mode
    for direction in ("UP", "DOWN"):
        push_dir(env, assist, total, direction, frames=160)
        idle(env, assist, total, 20)
        snap = read_snapshot(env.get_ram())
        if snap.screen != room0 or snap.mode != mode0:
            break
    extra = 0
    while extra < 200:
        snap = read_snapshot(env.get_ram())
        if snap.mode in (PLAY_MODE, 9, 11) and not snap.transitioning:
            idle(env, assist, total, 20)
            break
        step(env, assist, total, nes_idle_action())
        extra += 1
    ram = env.get_ram()
    snap = read_snapshot(ram)
    png = shot(env, assist, total, f"{tag}_stairs")
    dump = dump_live(snap, ram)
    body = {
        "via": f"{tag} stairs/hole",
        "ok": snap.screen != room0 or snap.mode != mode0,
        "dump": dump,
        "screenshot": png,
        "whistle_0x065C": int(read_u8(ram, ADDR_WHISTLE)),
        "status_claim": None,
        "pokes": False,
    }
    write_json_report(RECORDINGS_DIR / f"{tag}_stairs.json", body)
    return {
        "direction": "STAIRS",
        "success": snap.screen != room0 or snap.mode != mode0,
        "dest_room": f"0x{snap.screen:02x}",
        "mode": snap.mode,
        "xy": [snap.link_x, snap.link_y],
        "whistle_0x065C": int(read_u8(ram, ADDR_WHISTLE)),
        "dump": dump,
        "screenshot": png,
    }


def rescan_cleared(state: str, room: int, tag: str) -> dict:
    """Load an already-cleared checkpoint and scan tiles. Do not re-clear."""
    env = None
    total = [1]
    try:
        env, assist, obs = open_env(state)
        idle(env, assist, total, 24)
        ram = env.get_ram()
        snap = read_snapshot(ram)
        start = dump_live(snap, ram)
        print(
            "RESCAN_START",
            state,
            start.get("room_hex"),
            "mode",
            start.get("mode"),
            "xy",
            [start.get("x"), start.get("y")],
            "blocks",
            len(start.get("blocks_0x68") or []),
            "tile",
            start.get("colliding_tile"),
            flush=True,
        )
        pose_ok = snap.level == LEVEL_5 and snap.screen == room and snap.mode == PLAY_MODE
        scan = None
        stairs = None
        if pose_ok:
            scan = scan_tiles(env, assist, total, room, tag)
            stairs = take_stairs_if_open(env, assist, total, tag)
        png = shot(env, assist, total, tag)
        ram = env.get_ram()
        snap = read_snapshot(ram)
        final = dump_live(snap, ram)
        body = {
            "from_state": state,
            "expected_room": f"0x{room:02x}",
            "pose_ok": pose_ok,
            "re_cleared": False,
            "pokes": False,
            "status_claim": None,
            "start": start,
            "scan": scan,
            "stairs": stairs,
            "final": final,
            "has_0x68": bool((scan or {}).get("has_0x68") or start.get("blocks_0x68")),
            "has_stairs_tile": bool((scan or {}).get("has_stairs_tile")),
            "cellar": bool((scan or {}).get("cellar") or final.get("cellar_mode")),
            "whistle_0x065C": final.get("whistle_0x065C"),
            "screenshot": png,
            "rom": rom_room(room),
        }
        write_json_report(RECORDINGS_DIR / f"{tag}.json", body)
        print(
            "RESCAN",
            state,
            "0x68",
            body["has_0x68"],
            "stairs",
            body["has_stairs_tile"],
            "cellar",
            body["cellar"],
            "whistle",
            body["whistle_0x065C"],
            flush=True,
        )
        return body
    finally:
        if env is not None:
            env.close()


def leave_dungeon_from_66() -> dict:
    """Existing near-entrance exit: Level5Cleared66 SOUTH → 0x76 SOUTH → OW."""
    env = None
    total = [1]
    try:
        env, assist, obs = open_env("Level5Cleared66")
        idle(env, assist, total, 24)
        start = dump_live(read_snapshot(env.get_ram()), env.get_ram())
        log = []
        # 0x66 south to 0x76
        walk_axis(env, assist, total, "x", 120, max_f=400)
        walk_axis(env, assist, total, "y", 205, max_f=400)
        align_door(env, assist, total, 120, 205, frames=20)
        room0 = read_snapshot(env.get_ram()).screen
        push_dir(env, assist, total, "DOWN", frames=240)
        idle(env, assist, total, 16)
        snap = read_snapshot(env.get_ram())
        if snap.screen != room0:
            wait_play(env, assist, total, snap.screen, max_f=240)
        at76 = dump_live(read_snapshot(env.get_ram()), env.get_ram())
        log.append({"via": "66_south", "dest": at76.get("room_hex"), "level": at76.get("level")})
        print("LEAVE66", at76.get("room_hex"), "level", at76.get("level"), flush=True)
        # 0x76 south mouth to OW
        if at76.get("room") == ROOM_76 and at76.get("level") == LEVEL_5:
            walk_axis(env, assist, total, "x", 120, max_f=300)
            walk_axis(env, assist, total, "y", 205, max_f=400)
            align_door(env, assist, total, 120, 205, frames=20)
            room1 = read_snapshot(env.get_ram()).screen
            push_dir(env, assist, total, "DOWN", frames=280)
            idle(env, assist, total, 20)
            extra = 0
            while extra < 240:
                snap = read_snapshot(env.get_ram())
                if snap.level == 0 and not snap.transitioning:
                    idle(env, assist, total, 20)
                    break
                step(env, assist, total, nes_idle_action())
                extra += 1
        ram = env.get_ram()
        snap = read_snapshot(ram)
        after = dump_live(snap, ram)
        png = shot(env, assist, total, "l5_leave_ow")
        on_ow = snap.level == 0
        rupees = int(snap.rupees)
        candle = int(read_u8(ram, ADDR_CANDLE))
        blockers = []
        if not on_ow:
            blockers.append("did_not_reach_overworld")
        if rupees < CANDLE_SHOP_PRICE:
            blockers.append(f"rupees_{rupees}_lt_{CANDLE_SHOP_PRICE}")
        blockers.append("no_ready_OW_path_L5_0x0B_to_candle_shop_0x5E")
        blockers.append("rr-38p_early_OW_candle_bead_not_composed_from_L5_exit")
        rec = {
            "from_state": "Level5Cleared66",
            "helper": "walk 0x66 SOUTH → 0x76 SOUTH mouth (existing entrance reverse)",
            "pokes": False,
            "status_claim": None,
            "start66": start,
            "at76": at76,
            "after": after,
            "on_overworld": on_ow,
            "ow_screen": f"0x{snap.screen:02x}" if on_ow else None,
            "level": snap.level,
            "rupees": rupees,
            "candle_0x065B": candle,
            "whistle_0x065C": int(read_u8(ram, ADDR_WHISTLE)),
            "shop_price": CANDLE_SHOP_PRICE,
            "got_candle": candle == 1,
            "blockers": blockers,
            "log": log,
            "screenshot": png,
        }
        write_json_report(RECORDINGS_DIR / "l5_leave_ow.json", rec)
        print(
            "LEAVE_OW",
            "on_ow",
            on_ow,
            "screen",
            rec.get("ow_screen"),
            "rupees",
            rupees,
            "candle",
            candle,
            "blockers",
            blockers,
            flush=True,
        )
        return rec
    finally:
        if env is not None:
            env.close()


def main() -> dict:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    commands = [
        "PYTHONPATH=nes:. uv run python nes/zelda_i/scripts/_probe_l5_34_clear.py  "
        "# Level5Cleared25 WEST 0x24 (no fight) SOUTH 0x34, infinite-life, "
        "GenericDungeonRoomController+ROOM_66_SPEC, 28000f; rescan 37/27/26; "
        "leave 66→76 if no whistle"
    ]
    rom34 = rom_room(ROOM_34)
    rom24 = rom_room(ROOM_24)
    print("ROM34", rom34, flush=True)
    print("ROM24", rom24, flush=True)

    env = None
    hop = None
    fight = None
    scan = None
    stairs = None
    saved = None
    mid_dump = None
    arrive_dump = None
    start_dump = None
    post_clear_doors = None
    dead = False
    total = [1]
    try:
        env, assist, obs = open_env(STATE)
        idle(env, assist, total, 24)
        ram = env.get_ram()
        start_snap = read_snapshot(ram)
        start_dump = dump_live(start_snap, ram)
        print(
            "START",
            hex(start_snap.screen),
            [start_snap.link_x, start_snap.link_y],
            "keys",
            start_snap.keys,
            "rupees",
            start_snap.rupees,
            "whistle",
            start_dump.get("whistle_0x065C"),
            flush=True,
        )

        if start_snap.screen != ROOM_34:
            hop = walk_25_west_24_south_34(env, assist, total)
            print(
                "HOP",
                hop.get("changed"),
                hop.get("dest_room"),
                "fought_0x38",
                hop.get("fought_0x38"),
                flush=True,
            )

        ready = wait_play(env, assist, total, ROOM_34, max_f=360)
        ram = env.get_ram()
        arrive_snap = read_snapshot(ram)
        arrive_dump = dump_live(arrive_snap, ram)
        arrive_png = shot(env, assist, total, "l5_34_arrive")
        n_gib = len(live_gibdos(arrive_snap))
        print(
            "READY",
            ready,
            "room",
            hex(arrive_snap.screen),
            "mode",
            arrive_snap.mode,
            "xy",
            (arrive_snap.link_x, arrive_snap.link_y),
            "gibdos",
            n_gib,
            "item",
            arrive_snap.room_item_id,
            "doors",
            hex(arrive_snap.cur_opened_doors),
            "mask",
            hex(arrive_snap.open_doorway_mask),
            flush=True,
        )

        if not ready or arrive_snap.screen != ROOM_34 or arrive_snap.mode != PLAY_MODE:
            report = {
                "ok": False,
                "reason": "never_play_mode_5_in_0x34",
                "status_claim": None,
                "pokes": False,
                "fought_0x38": bool(hop and hop.get("fought_0x38")),
                "commands": commands,
                "controller_reused": "GenericDungeonRoomController + ROOM_66_SPEC",
                "rom34": rom34,
                "rom24": rom24,
                "start": start_dump,
                "arrive": arrive_dump,
                "hop": hop,
                "ready": ready,
                "checkpoint": None,
                "whistle_0x065C": arrive_dump.get("whistle_0x065C"),
                "screenshot": arrive_png,
            }
            write_json_report(RECORDINGS_DIR / "l5_34_clear.json", report)
            return report

        spec = make_34_spec()
        fight = fight_room(env, assist, total, spec)
        idle(env, assist, total, 30)
        extra = 0
        while extra < 90:
            snap = read_snapshot(env.get_ram())
            if snap.mode == PLAY_MODE and snap.screen == ROOM_34:
                break
            step(env, assist, total, nes_idle_action())
            extra += 1
        # all_dead secret: shutter / stairs / block can spawn after last kill
        idle(env, assist, total, 50)

        ram = env.get_ram()
        mid_snap = read_snapshot(ram)
        mid_dump = dump_live(mid_snap, ram)
        dead = (
            mid_snap.screen == ROOM_34
            and mid_snap.mode == PLAY_MODE
            and not live_gibdos(mid_snap)
        )
        doors_end = decode_doors(mid_snap.cur_opened_doors)
        mask_end = decode_doors(mid_snap.open_doorway_mask)
        post_clear_doors = {
            "from_room": "0x34",
            "all_6_dead": dead,
            "cur_opened_doors": doors_end,
            "open_doorway_mask": mask_end,
            "N": rom34["N"],
            "S": rom34["S"],
            "E": rom34["E"],
            "W": rom34["W"],
            "secret": rom34["secret"],
            "room_all_dead": mid_snap.room_all_dead,
            "keys": int(mid_snap.keys),
            "bombs": int(mid_snap.bombs),
            "rupees": int(mid_snap.rupees),
            "room_item_id": mid_snap.room_item_id,
        }
        print(
            "MID_CLEAR dead",
            dead,
            "frames",
            fight["frames"],
            "kills",
            fight["kills"],
            "end_n",
            fight["end_n"],
            "doors",
            doors_end,
            "mask",
            mask_end,
            "all_dead",
            mid_snap.room_all_dead,
            flush=True,
        )

        state_bytes = env.em.get_state()
        png = shot(env, assist, total, "l5_34_clear")

        if dead:
            scan = scan_tiles(env, assist, total, ROOM_34, "l5_34")
            stairs = take_stairs_if_open(env, assist, total, "l5_34")
            png = shot(env, assist, total, "l5_34_clear")
            ram = env.get_ram()
            mid_snap = read_snapshot(ram)
            mid_dump = dump_live(mid_snap, ram)
            write_json_report(
                RECORDINGS_DIR / "l5_34_tiles.json",
                {
                    "from_room": "0x34",
                    "all_6_dead": dead,
                    "scan": scan,
                    "stairs": stairs,
                    "dump": mid_dump,
                    "pokes": False,
                },
            )

        if dead:
            path = write_state_bytes(
                state_path(GAME_DIR, GAME, "Level5Cleared34"),
                state_bytes,
            )
            write_state_provenance(
                path,
                source_state_path=(
                    GAME_DIR / "custom_integrations" / GAME / f"{STATE}.state"
                ),
                request={
                    "segment": "Level5Cleared34",
                    "predecessor_entry": True,
                    "start_state": STATE,
                    "key_poke": False,
                    "door_poke": False,
                    "bomb_count_poke": False,
                    "selected_item_poke": False,
                    "controller": "GenericDungeonRoomController",
                    "spec_base": ROOM_66_SPEC.spec_id,
                    "alive_rule": "hp",
                    "via": "0x25 WEST 0x24 SOUTH 0x34 no 0x38 fight",
                },
                selected_trial={
                    "success": True,
                    "frames": fight["frames"],
                    "room": ROOM_34,
                    "live_gibdos": 0,
                    "doors_after": int(mid_snap.cur_opened_doors) & 0x0F,
                    "doorway_mask": int(mid_snap.open_doorway_mask) & 0x0F,
                    "bombs": int(mid_snap.bombs),
                    "keys": int(mid_snap.keys),
                    "rupees": int(mid_snap.rupees),
                    "whistle_0x065C": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
                },
                natural_entry=False,
            )
            saved = "Level5Cleared34"

        env.close()
        env = None
    finally:
        if env is not None:
            env.close()

    rescans = []
    for state, room, tag in RESCAN_ROOMS:
        try:
            rescans.append(rescan_cleared(state, room, tag))
        except Exception as exc:  # noqa: BLE001 — report and continue
            print("RESCAN_FAIL", state, exc, flush=True)
            rescans.append({"from_state": state, "ok": False, "error": str(exc)})

    whistle = 0
    if mid_dump:
        whistle = int(mid_dump.get("whistle_0x065C") or 0)
    for row in rescans:
        whistle = max(whistle, int(row.get("whistle_0x065C") or 0))
    if stairs and stairs.get("whistle_0x065C"):
        whistle = max(whistle, int(stairs["whistle_0x065C"]))

    candle_attempt = None
    if whistle == 0:
        print("NO_WHISTLE leave dungeon / candle attempt", flush=True)
        candle_attempt = leave_dungeon_from_66()

    fight_out = {k: v for k, v in (fight or {}).items() if k != "obs"} if fight else None
    report = {
        "ok": dead,
        "status_claim": None,
        "from_state": STATE,
        "pokes": False,
        "fought_0x38": bool(hop and hop.get("fought_0x38")),
        "commands": commands,
        "controller_reused": "GenericDungeonRoomController",
        "spec_reused": ROOM_66_SPEC.spec_id,
        "spec_id": "level5_room34_gibdos_reuse66",
        "rom34": rom34,
        "rom24": rom24,
        "hop": {
            "west_path": (hop or {}).get("west_path"),
            "changed": (hop or {}).get("changed"),
            "dest_room": (hop or {}).get("dest_room"),
            "fought_0x38": (hop or {}).get("fought_0x38"),
            "at24_room": ((hop or {}).get("at24") or {}).get("room_hex"),
            "at24_objects": [
                (o.get("type_hex"), o.get("hp"))
                for o in ((hop or {}).get("at24") or {}).get("objects") or []
            ],
        }
        if hop
        else None,
        "start_room": (start_dump or {}).get("room_hex"),
        "arrive": {
            "room": (arrive_dump or {}).get("room_hex"),
            "mode": (arrive_dump or {}).get("mode"),
            "xy": [(arrive_dump or {}).get("x"), (arrive_dump or {}).get("y")],
            "objects": (arrive_dump or {}).get("objects"),
            "doors": (arrive_dump or {}).get("doors"),
            "doorway_mask": (arrive_dump or {}).get("doorway_mask"),
            "gibdos": len(
                [
                    o
                    for o in (arrive_dump or {}).get("objects") or []
                    if o.get("type_id") == GIBDO_OBJECT_TYPE
                ]
            ),
            "room_item_id": (arrive_dump or {}).get("room_item_id"),
            "rupees": (arrive_dump or {}).get("rupees"),
        },
        "clear": fight_out,
        "post_clear_doors": post_clear_doors,
        "tiles": {
            "has_0x68": bool((scan or {}).get("has_0x68")),
            "has_stairs_tile": bool((scan or {}).get("has_stairs_tile")),
            "cellar": bool((scan or {}).get("cellar")),
            "tiles_seen": (scan or {}).get("tiles_seen"),
            "blocks_0x68": (scan or {}).get("blocks_0x68"),
            "stair_hits": (scan or {}).get("stair_hits"),
            "notes": (scan or {}).get("notes"),
            "dest": (scan or {}).get("dest"),
        }
        if scan
        else None,
        "stairs_dest": stairs,
        "rescans": [
            {
                "state": r.get("from_state"),
                "has_0x68": r.get("has_0x68"),
                "has_stairs_tile": r.get("has_stairs_tile"),
                "cellar": r.get("cellar"),
                "whistle_0x065C": r.get("whistle_0x065C"),
                "tiles_seen": ((r.get("scan") or {}).get("tiles_seen")),
                "blocks_0x68": ((r.get("scan") or {}).get("blocks_0x68")),
                "notes": ((r.get("scan") or {}).get("notes")),
            }
            for r in rescans
        ],
        "candle_attempt": candle_attempt,
        "checkpoint": saved,
        "checkpoint_reason": (
            "all 6 Gibdos dead in play mode 5"
            if saved
            else "not saved: enemies still alive or never arrived"
        ),
        "whistle_0x065C": whistle,
        "frames_total": total[0],
        "screenshot": str((RECORDINGS_DIR / "l5_34_clear.png").resolve()),
        "dump": mid_dump,
    }
    write_json_report(RECORDINGS_DIR / "l5_34_clear.json", report)
    return report


if __name__ == "__main__":
    r = main()
    print("CMD", r["commands"])
    print("OK", r.get("ok"), "FOUGHT_38", r.get("fought_0x38"), "POKES", r.get("pokes"))
    print("CONTROLLER", r.get("controller_reused"), "SPEC", r.get("spec_reused"))
    print("ROM34", r.get("rom34"))
    c = r.get("clear") or {}
    print("FRAMES", c.get("frames"), "KILLS", c.get("kills"), "END_N", c.get("end_n"))
    print("DOORS", r.get("post_clear_doors"))
    print("TILES", r.get("tiles"))
    print("STAIRS", r.get("stairs_dest"))
    print("RESCANS", r.get("rescans"))
    print("WHISTLE", r.get("whistle_0x065C"))
    print("CANDLE", r.get("candle_attempt") and {
        "on_ow": (r.get("candle_attempt") or {}).get("on_overworld"),
        "rupees": (r.get("candle_attempt") or {}).get("rupees"),
        "candle": (r.get("candle_attempt") or {}).get("candle_0x065B"),
        "blockers": (r.get("candle_attempt") or {}).get("blockers"),
    })
    print("CHECKPOINT", r.get("checkpoint"), r.get("checkpoint_reason"))
    print("STATUS_CLAIM", r.get("status_claim"))
