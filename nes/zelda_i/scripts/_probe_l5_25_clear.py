"""Clear L5 0x25 5x Pols Voice from Level5Cleared26, dump doors, probe exits.

Start: Level5Cleared26. Walk WEST y=141 into 0x25. Wait play mode 5.
Reuse Level5PolsVoiceController / ROOM_77 combat (same as 0x77 and 0x27).
5x Pols Voice 0x16 HP=160. Arrival @(224,141), doors live 0x00 mask 0x00,
item 0x03. Keys=4 bombs=7.

After 5/5 dead:
1. Dump doors
2. Check stairs/hole (push-block if secret-looking). Watch 0x065C whistle.
3. Walk every open exit (probe N/S/E/W). Dump new dests.
4. Level5Cleared25 only if 5/5 dead
5. If stairs open, go down and dump dest (whistle item room is the goal)

ROM 0x25: N=bomb S=wall W=key E=open secret=none.
No pokes, no candle invent, no Clean STATUS, no east67, no 0x65 bombs.
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
    RewardKind,
    RewardSpec,
)
from zelda_i.dungeon_ids import object_name
from zelda_i.dungeon_ops import idle, push_dir
from zelda_i.dungeon_trace import compact_snapshot, write_state_provenance
from zelda_i.level5_dungeon import (
    LEVEL_5,
    Level5PolsVoiceController,
    POLS_VOICE_OBJECT_TYPE,
    ROOM_25_SPEC,
    ROOM_77_SPEC,
    ROOM_ITEM_NONE,
    ROOM_L5_WEST_25,
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

STATE = "Level5Cleared26"
ROOM_26 = 0x26
ROOM_25 = ROOM_L5_WEST_25
MAX_FIGHT_FRAMES = 28000
PUSHABLE_BLOCK = 0x68
STAIR_TILE_LO = 0x70
STAIR_TILE_HI = 0x73
BLACK_MOUTH_TILE = 0x24
CELLAR_MODES = (9, 10, 11, 16)

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

STAIR_STANDS: tuple[tuple[int, int], ...] = (
    (120, 141),
    (128, 141),
    (112, 141),
    (96, 141),
    (144, 141),
    (80, 125),
    (160, 125),
    (80, 157),
    (160, 157),
    (120, 125),
    (120, 157),
    (208, 96),
    (48, 96),
    (120, 109),
    (120, 173),
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
    (64, 141, "LEFT"),
    (176, 141, "RIGHT"),
)


def make_25_spec() -> DungeonRoomSpec:
    """ROOM_77 combat / liveness, retargeted to 0x25 5x Pols Voice."""
    return replace(
        ROOM_77_SPEC,
        spec_id="level5_room25_pols_reuse77",
        source_room=ROOM_26,
        room_id=ROOM_25,
        entry=DoorRoute("LEFT", ((224, 141),)),
        expected_enemy_count=5,
        required_open_doors=0,
        reward=RewardSpec(kind=RewardKind.CLEAR_ONLY),
        room_item_id=ROOM_ITEM_NONE,
        exit_routes=(
            DoorRoute("RIGHT", ((208, 141),)),
            DoorRoute("LEFT", ((32, 141),)),
        ),
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


def five_dead(snap: ZeldaSnapshot) -> bool:
    return (
        snap.screen == ROOM_25
        and snap.mode == PLAY_MODE
        and not live_pols(snap)
    )


def stair_tile(tile: int) -> bool:
    return STAIR_TILE_LO <= int(tile) <= STAIR_TILE_HI or int(tile) == BLACK_MOUTH_TILE


def open_env(state: str = STATE):
    env = make_env(GAME, state, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True)
    obs, _ = reset_obs(env)
    obs, *_ = env.step(nes_idle_action())
    assist.apply_env(env, frame=0)
    return env, assist, obs


def open_from_bytes(state_data: bytes):
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True)
    obs, _ = reset_obs(env)
    env.em.set_state(state_data)
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


def walk_west_from_26(env, assist, total) -> dict:
    """Proven 0x26 leave: y=141 then west door. Fallbacks around moat/C-block."""
    log = []
    snap = read_snapshot(env.get_ram())
    keys0 = int(snap.keys)
    start_xy = [snap.link_x, snap.link_y]
    paths = (
        ("y141_west", (("y", 141), ("x", 32))),
        ("south189_west", (("y", 189), ("x", 32), ("y", 141))),
        ("north109_west", (("y", 109), ("x", 32), ("y", 141))),
        ("east208_south189_west", (("x", 208), ("y", 189), ("x", 32), ("y", 141))),
    )
    used = None
    for name, steps in paths:
        near = False
        for axis, tgt in steps:
            ok = walk_axis(env, assist, total, axis, tgt, max_f=450)
            snap = read_snapshot(env.get_ram())
            rec = {
                "path": name,
                "step": f"axis:{axis}:{tgt}",
                "ok": ok,
                "xy": [snap.link_x, snap.link_y],
                "tile": snap.colliding_tile,
                "room": f"0x{snap.screen:02x}",
            }
            log.append(rec)
            print("NAV26", rec, flush=True)
        at = align_door(env, assist, total, 32, 141)
        snap = read_snapshot(env.get_ram())
        near = abs(snap.link_x - 32) <= 6 and abs(snap.link_y - 141) <= 4
        if near:
            used = name
            break
    at = align_door(env, assist, total, 32, 141, frames=32)
    room0 = read_snapshot(env.get_ram()).screen
    keys_at = int(read_snapshot(env.get_ram()).keys)
    push_dir(env, assist, total, "LEFT", frames=220)
    idle(env, assist, total, 16)
    snap = read_snapshot(env.get_ram())
    if snap.screen != room0:
        wait_play(env, assist, total, snap.screen, max_f=240)
    idle(env, assist, total, 20)
    snap = read_snapshot(env.get_ram())
    return {
        "path": used,
        "log": log,
        "start_xy": start_xy,
        "at_mouth": at,
        "keys_in": keys0,
        "keys_at_door": keys_at,
        "keys_out": int(snap.keys),
        "key_spent": int(snap.keys) < keys0,
        "changed": snap.screen != room0,
        "dest": f"0x{snap.screen:02x}",
        "xy": [snap.link_x, snap.link_y],
        "mode": snap.mode,
    }


def fight_room(env, assist, total, spec: DungeonRoomSpec) -> dict:
    ctl = Level5PolsVoiceController(spec=spec)
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
        "controller_cls": "Level5PolsVoiceController",
        "reused": "Level5PolsVoiceController + ROOM_77_SPEC combat",
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


def check_stairs(env, assist, total) -> dict:
    """Look for hole/stairs and try push-block if a 0x68 is present."""
    ram = env.get_ram()
    snap = read_snapshot(ram)
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
    tile0 = int(snap.colliding_tile)
    mode0 = snap.mode
    room0 = snap.screen
    notes = []
    if blocks:
        notes.append(f"pushable_blocks_{len(blocks)}")
    else:
        notes.append("no_0x68_block")
    if stair_tile(tile0):
        notes.append(f"already_on_stair_tile_0x{tile0:02x}")

    pushes = []
    if blocks or True:
        # Always try a few center pushes — secret=none but user asked to check.
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
            if len(pushes) >= 10:
                break
            walk_axis(env, assist, total, "y", ty, max_f=220)
            walk_axis(env, assist, total, "x", tx, max_f=220)
            for _ in range(40):
                step(env, assist, total, nes_action(direction))
            idle(env, assist, total, 8)
            ram = env.get_ram()
            snap = read_snapshot(ram)
            rec = {
                "stand": [tx, ty],
                "dir": direction,
                "xy": [snap.link_x, snap.link_y],
                "tile": snap.colliding_tile,
                "mode": snap.mode,
                "room": f"0x{snap.screen:02x}",
                "whistle": int(read_u8(ram, ADDR_WHISTLE)),
                "stair_tile": stair_tile(snap.colliding_tile),
            }
            pushes.append(rec)
            print("PUSH", rec, flush=True)
            if snap.screen != room0 or snap.mode in CELLAR_MODES or stair_tile(snap.colliding_tile):
                notes.append("push_changed_room_or_stairs")
                break

    stands = []
    for tx, ty in STAIR_STANDS:
        snap = read_snapshot(env.get_ram())
        if snap.screen != room0 and snap.mode == PLAY_MODE:
            break
        walk_axis(env, assist, total, "y", ty, max_f=200)
        walk_axis(env, assist, total, "x", tx, max_f=200)
        idle(env, assist, total, 6)
        ram = env.get_ram()
        snap = read_snapshot(ram)
        rec = {
            "stand": [tx, ty],
            "xy": [snap.link_x, snap.link_y],
            "tile": snap.colliding_tile,
            "mode": snap.mode,
            "room": f"0x{snap.screen:02x}",
            "stair_tile": stair_tile(snap.colliding_tile),
        }
        stands.append(rec)
        if stair_tile(snap.colliding_tile) or snap.mode in CELLAR_MODES:
            notes.append("stood_on_stairs")
            # Try walking into the hole.
            for direction in ("UP", "DOWN", "LEFT", "RIGHT"):
                room_before = snap.screen
                mode_before = snap.mode
                push_dir(env, assist, total, direction, frames=80)
                idle(env, assist, total, 12)
                snap = read_snapshot(env.get_ram())
                if snap.screen != room_before or snap.mode != mode_before:
                    notes.append(f"stairs_took_{direction}_to_0x{snap.screen:02x}_m{snap.mode}")
                    break
            break

    ram = env.get_ram()
    snap = read_snapshot(ram)
    whistle1 = int(read_u8(ram, ADDR_WHISTLE))
    stairs_yes = (
        any(s.get("stair_tile") for s in stands)
        or any(p.get("stair_tile") for p in pushes)
        or snap.mode in CELLAR_MODES
        or snap.screen != room0
        or whistle1 > whistle0
    )
    dest = None
    if snap.screen != room0 or snap.mode in CELLAR_MODES:
        dest = {
            "room": f"0x{snap.screen:02x}",
            "mode": snap.mode,
            "xy": [snap.link_x, snap.link_y],
            "whistle_0x065C": whistle1,
            "dump": dump_live(snap, ram),
        }
    return {
        "looks_like_secret": bool(blocks) or stairs_yes,
        "blocks_0x68": blocks,
        "stairs": bool(stairs_yes),
        "whistle_in": whistle0,
        "whistle_out": whistle1,
        "whistle_got": whistle1 > whistle0,
        "start_tile": tile0,
        "end_tile": int(snap.colliding_tile),
        "end_mode": snap.mode,
        "end_room": f"0x{snap.screen:02x}",
        "notes": notes,
        "pushes": pushes,
        "stands": stands,
        "dest": dest,
        "pokes": False,
    }


def take_stairs_if_open(env, assist, total) -> dict | None:
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
    png = RECORDINGS_DIR / "l5_25_stairs.png"
    obs, *_ = env.step(nes_idle_action())
    total[0] += 1
    assist.apply_env(env, frame=total[0])
    save_rgb_png(obs, png)
    dump = dump_live(snap, ram)
    body = {
        "via": "0x25 stairs/hole",
        "ok": snap.screen != room0 or snap.mode != mode0,
        "dump": dump,
        "screenshot": str(png.resolve()),
        "whistle_0x065C": int(read_u8(ram, ADDR_WHISTLE)),
        "status_claim": None,
        "pokes": False,
    }
    write_json_report(RECORDINGS_DIR / "l5_25_stairs.json", body)
    return {
        "direction": "STAIRS",
        "success": snap.screen != room0 or snap.mode != mode0,
        "dest_room": f"0x{snap.screen:02x}",
        "mode": snap.mode,
        "xy": [snap.link_x, snap.link_y],
        "whistle_0x065C": int(read_u8(ram, ADDR_WHISTLE)),
        "dump": dump,
        "screenshot": str(png.resolve()),
    }


def probe_exit(state_data: bytes, total: list[int], direction: str, mouth: tuple[int, int], face: str, tag: str) -> dict:
    env = None
    try:
        env, assist, obs = open_from_bytes(state_data)
        idle(env, assist, total, 12)
        keys0 = int(read_snapshot(env.get_ram()).keys)
        bombs0 = int(read_snapshot(env.get_ram()).bombs)
        # Center then mouth. y=141 first for E/W like 0x26 moat.
        if face in ("LEFT", "RIGHT"):
            walk_axis(env, assist, total, "y", 141, max_f=300)
            walk_axis(env, assist, total, "x", mouth[0], max_f=500)
            walk_axis(env, assist, total, "y", mouth[1], max_f=250)
        else:
            walk_axis(env, assist, total, "x", 120, max_f=300)
            walk_axis(env, assist, total, "y", mouth[1], max_f=400)
            walk_axis(env, assist, total, "x", mouth[0], max_f=250)
        at = align_door(env, assist, total, mouth[0], mouth[1], frames=28)
        room0 = read_snapshot(env.get_ram()).screen
        keys_at = int(read_snapshot(env.get_ram()).keys)
        push_dir(env, assist, total, face, frames=200)
        idle(env, assist, total, 16)
        snap = read_snapshot(env.get_ram())
        if snap.screen != room0:
            wait_play(env, assist, total, snap.screen, max_f=240)
        idle(env, assist, total, 24)
        ram = env.get_ram()
        snap = read_snapshot(ram)
        dump = dump_live(snap, ram)
        dest = f"0x{snap.screen:02x}"
        changed = dest != "0x25" or snap.mode in CELLAR_MODES
        png = RECORDINGS_DIR / f"{tag}.png"
        obs, *_ = env.step(nes_idle_action())
        total[0] += 1
        assist.apply_env(env, frame=total[0])
        save_rgb_png(obs, png)
        body = {
            "via": f"0x25 {direction}",
            "ok": changed,
            "at_mouth": at,
            "keys_in": keys0,
            "keys_at_door": keys_at,
            "keys_out": int(snap.keys),
            "key_spent": int(snap.keys) < keys0,
            "bombs_in": bombs0,
            "bombs_out": int(snap.bombs),
            "dump": dump,
            "screenshot": str(png.resolve()),
            "status_claim": None,
            "pokes": False,
        }
        json_path = RECORDINGS_DIR / f"{tag}.json"
        write_json_report(json_path, body)
        print(
            "EXIT",
            direction,
            dest if changed else "sealed",
            "mode",
            snap.mode,
            "xy",
            [snap.link_x, snap.link_y],
            "keys",
            keys0,
            "->",
            int(snap.keys),
            "item",
            snap.room_item_id,
            flush=True,
        )
        return {
            "direction": direction,
            "success": changed,
            "sealed": not changed,
            "dest_room": dest if changed else None,
            "dest_room_id": int(snap.screen) if changed else None,
            "at_mouth": at,
            "xy": [snap.link_x, snap.link_y],
            "mode": snap.mode,
            "doors": dump.get("doors"),
            "doorway_mask": dump.get("doorway_mask"),
            "objects": dump.get("objects"),
            "room_item_id": snap.room_item_id,
            "room_item_name": dump.get("room_item_name"),
            "keys_in": keys0,
            "keys_out": int(snap.keys),
            "key_spent": int(snap.keys) < keys0,
            "bombs_in": bombs0,
            "bombs_out": int(snap.bombs),
            "whistle_0x065C": dump.get("whistle_0x065C"),
            "screenshot": str(png.resolve()),
            "dump_path": str(json_path.resolve()),
            "dump": dump,
        }
    finally:
        if env is not None:
            env.close()


def main() -> dict:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    commands = [
        "PYTHONPATH=nes:. uv run python nes/zelda_i/scripts/_probe_l5_25_clear.py  "
        "# Level5Cleared26 WEST 0x25, infinite-life, "
        "Level5PolsVoiceController+ROOM_77, 28000f"
    ]
    rom25 = rom_room(ROOM_25)
    rom26 = rom_room(ROOM_26)
    print("ROM25", rom25, flush=True)
    print("ROM26", rom26, flush=True)

    env = None
    env, assist, obs = open_env()
    total = [1]
    try:
        idle(env, assist, total, 30)
        ram = env.get_ram()
        start_snap = read_snapshot(ram)
        start_dump = dump_live(start_snap, ram)
        print(
            "START",
            hex(start_snap.screen),
            [start_snap.link_x, start_snap.link_y],
            "keys",
            start_snap.keys,
            "bombs",
            start_snap.bombs,
            "tile",
            start_snap.colliding_tile,
            flush=True,
        )

        hop = None
        walked = False
        if start_snap.screen != ROOM_25:
            hop = walk_west_from_26(env, assist, total)
            walked = bool(hop.get("changed"))
            print(
                "WEST_HOP",
                hop.get("changed"),
                hop.get("dest"),
                "keys",
                hop.get("keys_in"),
                "->",
                hop.get("keys_out"),
                "mouth",
                hop.get("at_mouth"),
                flush=True,
            )

        ready = wait_play(env, assist, total, ROOM_25, max_f=360)
        ram = env.get_ram()
        arrive_snap = read_snapshot(ram)
        arrive_dump = dump_live(arrive_snap, ram)
        arrive_png = RECORDINGS_DIR / "l5_25_arrive.png"
        obs, *_ = env.step(nes_idle_action())
        total[0] += 1
        assist.apply_env(env, frame=total[0])
        save_rgb_png(obs, arrive_png)
        n_pols = len(live_pols(arrive_snap))
        print(
            "READY",
            ready,
            "room",
            hex(arrive_snap.screen),
            "mode",
            arrive_snap.mode,
            "xy",
            (arrive_snap.link_x, arrive_snap.link_y),
            "pols",
            n_pols,
            "item",
            arrive_snap.room_item_id,
            "keys",
            arrive_snap.keys,
            "bombs",
            arrive_snap.bombs,
            "doors",
            hex(arrive_snap.cur_opened_doors),
            "mask",
            hex(arrive_snap.open_doorway_mask),
            flush=True,
        )

        if not ready or arrive_snap.screen != ROOM_25 or arrive_snap.mode != PLAY_MODE:
            report = {
                "ok": False,
                "reason": "never_play_mode_5_in_0x25",
                "status_claim": None,
                "pokes": False,
                "commands": commands,
                "controller_reused": "Level5PolsVoiceController + ROOM_77_SPEC",
                "rom25": rom25,
                "rom26": rom26,
                "start": start_dump,
                "arrive": arrive_dump,
                "walked_west": walked,
                "hop": hop,
                "ready": ready,
                "checkpoint": None,
                "whistle_0x065C": arrive_dump.get("whistle_0x065C"),
                "screenshot": str(arrive_png.resolve()),
            }
            write_json_report(RECORDINGS_DIR / "l5_25_clear.json", report)
            return report

        bombs_in = int(arrive_snap.bombs)
        keys_in = int(arrive_snap.keys)
        spec = make_25_spec()
        fight = fight_room(env, assist, total, spec)
        if fight.get("obs") is not None:
            obs = fight["obs"]
        idle(env, assist, total, 30)
        leftover = len(live_pols(read_snapshot(env.get_ram())))
        sweeps = [fight]
        if leftover > 0:
            sweep_spec = replace(spec, spec_id="level5_room25_pols_sweep", expected_enemy_count=max(1, leftover))
            sweep = fight_room(env, assist, total, sweep_spec)
            sweeps.append(sweep)
            leftover = len(live_pols(read_snapshot(env.get_ram())))
            print("POLS_SWEEP leftover", leftover, flush=True)

        extra = 0
        while extra < 90:
            snap = read_snapshot(env.get_ram())
            if snap.mode == PLAY_MODE and snap.screen == ROOM_25:
                break
            obs = step(env, assist, total, nes_idle_action())
            extra += 1
        idle(env, assist, total, 20)

        ram = env.get_ram()
        mid_snap = read_snapshot(ram)
        dead = five_dead(mid_snap)
        mid_dump = dump_live(mid_snap, ram)
        bombs_out = int(mid_snap.bombs)
        keys_out = int(mid_snap.keys)
        png = RECORDINGS_DIR / "l5_25_clear.png"
        obs, *_ = env.step(nes_idle_action())
        total[0] += 1
        assist.apply_env(env, frame=total[0])
        save_rgb_png(obs, png)

        doors_end = decode_doors(mid_snap.cur_opened_doors)
        mask_end = decode_doors(mid_snap.open_doorway_mask)
        # Capture in-room post-clear bytes BEFORE stairs/exit walks.
        state_bytes = env.em.get_state() if dead else None
        print(
            "MID_CLEAR dead",
            dead,
            "frames",
            sum(s.get("frames", 0) for s in sweeps),
            "kills",
            fight["kills"],
            "end_n",
            leftover,
            "keys",
            keys_in,
            "->",
            keys_out,
            "doors",
            doors_end,
            "mask",
            mask_end,
            flush=True,
        )

        stairs = None
        stairs_dest = None
        if dead:
            stairs = check_stairs(env, assist, total)
            print(
                "STAIRS",
                stairs.get("stairs"),
                "whistle",
                stairs.get("whistle_in"),
                "->",
                stairs.get("whistle_out"),
                "blocks",
                len(stairs.get("blocks_0x68") or []),
                "notes",
                stairs.get("notes"),
                flush=True,
            )
            taken = take_stairs_if_open(env, assist, total)
            if taken:
                stairs_dest = taken
                print("STAIRS_DEST", taken.get("dest_room"), "mode", taken.get("mode"), flush=True)

        env.close()
        env = None

        post_clear_doors = {
            "from_room": "0x25",
            "all_5_dead": dead,
            "doors": doors_end,
            "doorway_mask": mask_end,
            "keys": keys_out,
            "bombs": bombs_out,
            "room_item_id": mid_snap.room_item_id,
            "rom_N": rom25["N"],
            "rom_S": rom25["S"],
            "rom_E": rom25["E"],
            "rom_W": rom25["W"],
            "rom_secret": rom25["secret"],
        }

        saved = None
        if dead and state_bytes is not None:
            path = write_state_bytes(
                state_path(GAME_DIR, GAME, "Level5Cleared25"),
                state_bytes,
            )
            write_state_provenance(
                path,
                source_state_path=(
                    GAME_DIR / "custom_integrations" / GAME / f"{STATE}.state"
                ),
                request={
                    "segment": "Level5Cleared25",
                    "predecessor_entry": True,
                    "start_state": STATE,
                    "key_poke": False,
                    "door_poke": False,
                    "bomb_count_poke": False,
                    "selected_item_poke": False,
                    "controller": "Level5PolsVoiceController",
                    "spec_base": ROOM_25_SPEC.spec_id,
                    "combat_base": ROOM_77_SPEC.spec_id,
                    "alive_rule": "hp",
                },
                selected_trial={
                    "success": True,
                    "frames": sum(s.get("frames", 0) for s in sweeps),
                    "room": ROOM_25,
                    "live_pols": 0,
                    "doors_after": int(mid_snap.cur_opened_doors) & 0x0F,
                    "doorway_mask": int(mid_snap.open_doorway_mask) & 0x0F,
                    "bombs": bombs_out,
                    "keys": keys_out,
                },
                natural_entry=False,
            )
            saved = "Level5Cleared25"

        exits = []
        if dead and state_bytes is not None:
            probes = (
                ("NORTH", (120, 93), "UP", "l5_25_north"),
                ("SOUTH", (120, 205), "DOWN", "l5_25_south"),
                ("EAST", (208, 141), "RIGHT", "l5_25_east"),
                ("WEST", (32, 141), "LEFT", "l5_25_west"),
            )
            live_open = set(doors_end.get("open") or []) | set(mask_end.get("open") or [])
            # Always probe all four; mark which ROM/live consider open.
            for direction, mouth, face, tag in probes:
                rec = probe_exit(state_bytes, total, direction, mouth, face, tag)
                rec["rom"] = {
                    "NORTH": rom25["N"],
                    "SOUTH": rom25["S"],
                    "EAST": rom25["E"],
                    "WEST": rom25["W"],
                }[direction]
                rec["live_open"] = face in live_open or (
                    {"NORTH": "UP", "SOUTH": "DOWN", "EAST": "RIGHT", "WEST": "LEFT"}[direction]
                    in live_open
                )
                exits.append(rec)

        exits_body = {
            "from_room": "0x25",
            "all_5_dead": dead,
            "rom25": rom25,
            "post_clear_doors": post_clear_doors,
            "stairs": {k: v for k, v in (stairs or {}).items() if k not in ("stands",)} if stairs else None,
            "exits": [
                {k: v for k, v in e.items() if k != "dump"} for e in exits
            ],
            "status_claim": None,
            "pokes": False,
        }
        write_json_report(RECORDINGS_DIR / "l5_25_exits.json", exits_body)

        dest_after = None
        # Prefer stairs dest, else first new non-0x26 dest, else any success.
        if stairs_dest and stairs_dest.get("success"):
            dest_after = {
                "via": "stairs",
                "room": stairs_dest.get("dest_room"),
                "mode": stairs_dest.get("mode"),
                "xy": stairs_dest.get("xy"),
                "whistle_0x065C": stairs_dest.get("whistle_0x065C"),
            }
        else:
            for e in exits:
                if e.get("success") and e.get("dest_room") not in (None, "0x26"):
                    dest_after = {
                        "via": e.get("direction"),
                        "room": e.get("dest_room"),
                        "room_id": e.get("dest_room_id"),
                        "objects": e.get("objects"),
                        "doors": e.get("doors"),
                        "doorway_mask": e.get("doorway_mask"),
                        "item": e.get("room_item_id"),
                        "item_name": e.get("room_item_name"),
                        "xy": e.get("xy"),
                        "keys_in": e.get("keys_in"),
                        "keys_out": e.get("keys_out"),
                        "key_spent": e.get("key_spent"),
                        "whistle_0x065C": e.get("whistle_0x065C"),
                    }
                    break
            if dest_after is None:
                for e in exits:
                    if e.get("success"):
                        dest_after = {
                            "via": e.get("direction"),
                            "room": e.get("dest_room"),
                            "room_id": e.get("dest_room_id"),
                            "objects": e.get("objects"),
                            "doors": e.get("doors"),
                            "item": e.get("room_item_id"),
                            "xy": e.get("xy"),
                            "keys_in": e.get("keys_in"),
                            "keys_out": e.get("keys_out"),
                            "key_spent": e.get("key_spent"),
                        }
                        break

        whistle = (mid_dump.get("inventory") or {}).get("whistle_0x065C", 0)
        if stairs:
            whistle = stairs.get("whistle_out", whistle)
        fight_frames = sum(s.get("frames", 0) for s in sweeps)
        fight_out = {k: v for k, v in fight.items() if k != "obs"}
        report = {
            "ok": dead,
            "status_claim": None,
            "from_state": STATE,
            "pokes": False,
            "commands": commands,
            "controller_reused": "Level5PolsVoiceController",
            "spec_reused": ROOM_77_SPEC.spec_id,
            "spec_id": spec.spec_id,
            "rom25": rom25,
            "rom26": rom26,
            "walked_west": walked,
            "hop": hop,
            "start_room": f"0x{start_snap.screen:02x}",
            "arrive": {
                "room": arrive_dump.get("room_hex"),
                "mode": arrive_dump.get("mode"),
                "mode_name": arrive_dump.get("mode_name"),
                "xy": [arrive_dump.get("x"), arrive_dump.get("y")],
                "objects": arrive_dump.get("objects"),
                "doors": arrive_dump.get("doors"),
                "doorway_mask": arrive_dump.get("doorway_mask"),
                "bombs": bombs_in,
                "keys": keys_in,
                "room_item_id": arrive_snap.room_item_id,
                "pols": n_pols,
            },
            "clear": {
                **fight_out,
                "frames": fight_frames,
                "bombs_in": bombs_in,
                "bombs_out": bombs_out,
                "keys_in": keys_in,
                "keys_out": keys_out,
                "dead": dead,
                "sweeps": [{k: v for k, v in s.items() if k != "obs"} for s in sweeps],
            },
            "post_clear_doors": post_clear_doors,
            "doors_end": doors_end,
            "doorway_mask_end": mask_end,
            "stairs": {k: v for k, v in (stairs or {}).items() if k != "stands"} if stairs else None,
            "stairs_dest": stairs_dest,
            "exits": [{k: v for k, v in e.items() if k != "dump"} for e in exits],
            "dest_after_0x25": dest_after,
            "checkpoint": saved,
            "checkpoint_reason": (
                "all 5 Pols Voice dead in play mode 5"
                if saved
                else "not saved: enemies still alive or left room before capture"
            ),
            "whistle_0x065C": whistle,
            "frames_total": total[0],
            "screenshot": str(png.resolve()),
            "dump": mid_dump,
        }
        write_json_report(RECORDINGS_DIR / "l5_25_clear.json", report)
        return report
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    r = main()
    print("CMD", r["commands"])
    print("CONTROLLER", r.get("controller_reused"), "SPEC", r.get("spec_reused"))
    print("ROM25", r.get("rom25"))
    a = r.get("arrive") or {}
    print(
        "ARRIVE",
        a.get("room"),
        "mode",
        a.get("mode"),
        a.get("mode_name"),
        "xy",
        a.get("xy"),
        "pols",
        a.get("pols"),
        "keys",
        a.get("keys"),
        "bombs",
        a.get("bombs"),
        "item",
        a.get("room_item_id"),
    )
    c = r.get("clear") or {}
    print(
        "CLEAR frames",
        c.get("frames"),
        "kills",
        c.get("kills"),
        "start_n",
        c.get("start_n"),
        "end_n",
        c.get("end_n"),
        "dead",
        c.get("dead"),
        "keys",
        c.get("keys_in"),
        "->",
        c.get("keys_out"),
        "bombs",
        c.get("bombs_in"),
        "->",
        c.get("bombs_out"),
    )
    print("POST_CLEAR_DOORS", r.get("post_clear_doors"))
    print("STAIRS", r.get("stairs"))
    print("EXITS", r.get("exits"))
    print("DEST_AFTER_0x25", r.get("dest_after_0x25"))
    print("CKPT", r.get("checkpoint"), r.get("checkpoint_reason"))
    print("WHISTLE", r.get("whistle_0x065C"))
    print("FRAMES", r.get("frames_total"))
    print("status_claim", None)
