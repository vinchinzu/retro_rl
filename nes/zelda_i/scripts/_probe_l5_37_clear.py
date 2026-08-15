"""Clear L5 0x37 3x Darknut from Level5Cleared47, grab compass, dump exits.

Start: Level5Cleared47. Walk NORTH at x=120 (C-block pinch at x=128).
Reuse GenericDungeonRoomController + ROOM_5B_SPEC (3x Darknut) with
ROOM_59_SPEC combat (proven side/back Darknut patrol). No new fighter.
3x Darknut 0x0b HP=64; grab compass 0x16 if it drops (walk, no RAM poke).
No pokes, no candle invent, no Clean STATUS, no east67, no 0x65 bombs.
Level5Cleared37 only if all 3 Darknuts dead.
ROM N/S=open, E/W=wall, secret=foes_item — walk N/S; skip E/W unless live open.
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
from zelda_i.dungeon_ids import DARKNUT_OBJECT_TYPE, object_name
from zelda_i.dungeon_ops import goto, idle, push_dir
from zelda_i.dungeon_trace import compact_snapshot, write_state_provenance
from zelda_i.level3_dungeon import ROOM_5B_SPEC, ROOM_59_SPEC, ROOM_ITEM_COMPASS
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR, SHARED_ROM_ZIP
from zelda_i.ram import (
    ADDR_CANDLE,
    ADDR_COMPASS,
    ADDR_SELECTED_ITEM,
    ADDR_WHISTLE,
    PLAY_MODE,
    ZeldaSnapshot,
    read_snapshot,
    read_u8,
)

STATE = "Level5Cleared47"
ROOM_47 = 0x47
ROOM_37 = 0x37
MAX_FIGHT_FRAMES = 20000
LEVEL5_COMPASS_BIT = 0x10  # ADDR_COMPASS bit for dungeon level 5
KNOWN_ROOMS = {0x37, 0x46, 0x47, 0x55, 0x56, 0x57, 0x65, 0x66, 0x67, 0x76, 0x77}

# L1-6 Q1 tables (same 0x18700 dump used on 0x47).
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

COMPASS_WAYPOINTS: tuple[tuple[int, int], ...] = (
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
    (80, 134),
    (128, 163),
    (160, 147),
)


def make_37_spec() -> DungeonRoomSpec:
    """ROOM_5B_SPEC 3x Darknut + ROOM_59 combat, retargeted to L5 0x37."""
    return replace(
        ROOM_5B_SPEC,
        spec_id="level5_room37_darknuts_reuse5b59",
        source_room=ROOM_47,
        room_id=ROOM_37,
        entry=DoorRoute("UP", ((120, 205),)),
        enemy_types=(DARKNUT_OBJECT_TYPE,),
        expected_enemy_count=3,
        required_open_doors=0,
        reward=RewardSpec(kind=RewardKind.CLEAR_ONLY, settle_all_dead=0),
        room_item_id=ROOM_ITEM_COMPASS,
        combat=ROOM_59_SPEC.combat,
        exit_routes=(
            DoorRoute("UP", ((120, 93),)),
            DoorRoute("DOWN", ((120, 205),)),
        ),
        max_frames=MAX_FIGHT_FRAMES,
        level=5,
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
    compass = int(read_u8(ram, ADDR_COMPASS))
    return {
        "selected_item_0x0656": int(read_u8(ram, ADDR_SELECTED_ITEM)),
        "whistle_0x065C": int(read_u8(ram, ADDR_WHISTLE)),
        "candle_0x065B": int(read_u8(ram, ADDR_CANDLE)),
        "compass_0x0667": compass,
        "compass_l5_bit": bool(compass & LEVEL5_COMPASS_BIT),
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
    compact["compass_0x0667"] = int(read_u8(ram, ADDR_COMPASS))
    compact["compass_l5"] = bool(int(read_u8(ram, ADDR_COMPASS)) & LEVEL5_COMPASS_BIT)
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


def wait_play(env, assist, total, room: int, *, max_f: int = 360) -> bool:
    for _ in range(max_f):
        snap = read_snapshot(env.get_ram())
        if (
            snap.level == 5
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


def walk_north_from_47(env, assist, total) -> dict:
    """Align door at x=120. If pinched at x=128, use north band then slide."""
    snap = read_snapshot(env.get_ram())
    start_xy = [snap.link_x, snap.link_y]
    notes = []
    if abs(snap.link_x - 128) <= 6:
        # C-block pinch: climb the x=128 column to the north band, then slide.
        notes.append("pinch_x128_via_north_band")
        goto(env, assist, total, 128, 93, tol=3, max_f=500)
        goto(env, assist, total, 120, 93, tol=2, max_f=400)
    else:
        notes.append("center_x120")
        goto(env, assist, total, 120, snap.link_y, tol=3, max_f=400)
        goto(env, assist, total, 120, 141, tol=3, max_f=400)
        goto(env, assist, total, 120, 93, tol=2, max_f=500)
    # Fine-align door mouth x=120 (never push at x=128).
    for _ in range(24):
        snap = read_snapshot(env.get_ram())
        if abs(snap.link_x - 120) <= 1 and abs(snap.link_y - 93) <= 2:
            break
        if abs(snap.link_x - 120) > 1:
            step(
                env,
                assist,
                total,
                nes_action("RIGHT" if snap.link_x < 120 else "LEFT"),
            )
        else:
            step(
                env,
                assist,
                total,
                nes_action("DOWN" if snap.link_y < 93 else "UP"),
            )
    at = [read_snapshot(env.get_ram()).link_x, read_snapshot(env.get_ram()).link_y]
    room0 = read_snapshot(env.get_ram()).screen
    push_dir(env, assist, total, "UP", frames=150)
    idle(env, assist, total, 16)
    snap = read_snapshot(env.get_ram())
    changed = snap.screen != room0
    if changed:
        wait_play(env, assist, total, snap.screen, max_f=240)
    snap = read_snapshot(env.get_ram())
    return {
        "changed_room": changed,
        "start_xy": start_xy,
        "at_mouth": at,
        "result_room": f"0x{snap.screen:02x}",
        "result_xy": [snap.link_x, snap.link_y],
        "notes": notes,
        "align_x": 120,
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
                progress.append(
                    {"f": ctl.frames, "n": len(live), "hps": [o.hp for o in live]}
                )
            elif len(live) != last_n:
                last_n = len(live)
                progress.append(
                    {"f": ctl.frames, "n": len(live), "hps": [o.hp for o in live]}
                )
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
        "reused": (
            "GenericDungeonRoomController + ROOM_5B_SPEC (3x Darknut) "
            "+ ROOM_59_SPEC.combat (side/back patrol)"
        ),
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


def grab_compass(env, assist, total, compass0: int) -> dict:
    """Walk typical 0x16 drop tiles. No RAM poke. Success = L5 compass bit."""
    tried = []
    got = bool(compass0 & LEVEL5_COMPASS_BIT)
    for tx, ty in COMPASS_WAYPOINTS:
        snap = read_snapshot(env.get_ram())
        if snap.screen != ROOM_37 or snap.mode != PLAY_MODE:
            break
        compass = int(read_u8(env.get_ram(), ADDR_COMPASS))
        if compass & LEVEL5_COMPASS_BIT:
            got = True
            break
        goto(env, assist, total, tx, ty, tol=3, max_f=350)
        idle(env, assist, total, 12)
        snap = read_snapshot(env.get_ram())
        compass = int(read_u8(env.get_ram(), ADDR_COMPASS))
        tried.append(
            {
                "stand": [tx, ty],
                "xy": [snap.link_x, snap.link_y],
                "compass": compass,
                "l5": bool(compass & LEVEL5_COMPASS_BIT),
            }
        )
        if compass & LEVEL5_COMPASS_BIT:
            got = True
            break
    snap = read_snapshot(env.get_ram())
    compass1 = int(read_u8(env.get_ram(), ADDR_COMPASS))
    return {
        "compass_in": compass0,
        "compass_out": compass1,
        "grabbed": bool(compass1 & LEVEL5_COMPASS_BIT)
        and not bool(compass0 & LEVEL5_COMPASS_BIT),
        "have_l5": bool(compass1 & LEVEL5_COMPASS_BIT),
        "tried_n": len(tried),
        "xy": [snap.link_x, snap.link_y],
        "pokes": False,
    }


def walk_exit(state_data: bytes, direction: str, total: list[int]) -> dict:
    """Walk one exit from post-clear 0x37. Align door, push, dump dest."""
    mouths = {
        "UP": (120, 93),
        "DOWN": (120, 205),
        "LEFT": (32, 141),
        "RIGHT": (208, 141),
    }
    via = {
        "UP": ((120, 141), (120, 93)),
        "DOWN": ((120, 141), (120, 205)),
        "LEFT": ((120, 141), (32, 141)),
        "RIGHT": ((120, 141), (208, 141)),
    }
    env = None
    try:
        env, assist, obs = open_from_bytes(state_data)
        idle(env, assist, total, 12)
        for tx, ty in via[direction]:
            goto(env, assist, total, tx, ty, tol=3, max_f=400)
        mx, my = mouths[direction]
        goto(env, assist, total, mx, my, tol=2, max_f=400)
        for _ in range(20):
            snap = read_snapshot(env.get_ram())
            if abs(snap.link_x - mx) <= 1 and abs(snap.link_y - my) <= 1:
                break
            if abs(snap.link_x - mx) > 1:
                step(
                    env,
                    assist,
                    total,
                    nes_action("RIGHT" if snap.link_x < mx else "LEFT"),
                )
            else:
                step(
                    env,
                    assist,
                    total,
                    nes_action("DOWN" if snap.link_y < my else "UP"),
                )
        at = [read_snapshot(env.get_ram()).link_x, read_snapshot(env.get_ram()).link_y]
        room0 = read_snapshot(env.get_ram()).screen
        push_dir(env, assist, total, direction, frames=150)
        idle(env, assist, total, 16)
        snap = read_snapshot(env.get_ram())
        if snap.screen != room0:
            wait_play(env, assist, total, snap.screen, max_f=240)
        idle(env, assist, total, 24)
        ram = env.get_ram()
        snap = read_snapshot(ram)
        dump = dump_live(snap, ram)
        dest = f"0x{snap.screen:02x}"
        changed = dest != "0x37"
        tag = f"l5_{snap.screen:02x}_from37" if changed else f"l5_37_exit_{direction.lower()}"
        png = RECORDINGS_DIR / f"{tag}.png"
        obs, *_ = env.step(nes_idle_action())
        total[0] += 1
        assist.apply_env(env, frame=total[0])
        save_rgb_png(obs, png)
        json_path = RECORDINGS_DIR / f"{tag}.json"
        body = {
            "via": f"0x37 {direction}",
            "ok": changed,
            "at_mouth": at,
            "dump": dump,
            "screenshot": str(png.resolve()),
            "status_claim": None,
            "pokes": False,
        }
        write_json_report(json_path, body)
        return {
            "direction": direction,
            "success": changed,
            "sealed": not changed,
            "dest_room": dest if changed else None,
            "dest_room_id": int(snap.screen) if changed else None,
            "known": int(snap.screen) in KNOWN_ROOMS if changed else None,
            "at_mouth": at,
            "xy": [snap.link_x, snap.link_y],
            "mode": snap.mode,
            "doors": dump.get("doors"),
            "doorway_mask": dump.get("doorway_mask"),
            "objects": dump.get("objects"),
            "room_item_id": snap.room_item_id,
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
        "PYTHONPATH=nes:. uv run python nes/zelda_i/scripts/_probe_l5_37_clear.py  "
        "# Level5Cleared47 NORTH 0x37 x=120, infinite-life, "
        "GenericDungeonRoomController+ROOM_5B_SPEC+ROOM_59 combat, 20000f"
    ]
    rom37 = rom_room(ROOM_37)
    rom27 = rom_room(0x27)
    rom47 = rom_room(ROOM_47)
    print("ROM37", rom37, flush=True)
    print("ROM27", rom27, flush=True)

    env = None
    env, assist, obs = open_env()
    total = [1]
    try:
        idle(env, assist, total, 20)
        ram = env.get_ram()
        start_snap = read_snapshot(ram)
        start_dump = dump_live(start_snap, ram)

        hop = None
        walked = False
        if start_snap.screen != ROOM_37:
            hop = walk_north_from_47(env, assist, total)
            walked = bool(hop.get("changed_room"))
            print(
                "NORTH_HOP",
                hop.get("changed_room"),
                hop.get("result_room"),
                hop.get("at_mouth"),
                hop.get("notes"),
                flush=True,
            )

        ready = wait_play(env, assist, total, ROOM_37, max_f=360)
        ram = env.get_ram()
        arrive_snap = read_snapshot(ram)
        arrive_dump = dump_live(arrive_snap, ram)
        arrive_png = RECORDINGS_DIR / "l5_37_arrive.png"
        obs, *_ = env.step(nes_idle_action())
        total[0] += 1
        assist.apply_env(env, frame=total[0])
        save_rgb_png(obs, arrive_png)
        print(
            "READY",
            ready,
            "room",
            hex(arrive_snap.screen),
            "mode",
            arrive_snap.mode,
            "xy",
            (arrive_snap.link_x, arrive_snap.link_y),
            "darknuts",
            len(live_darknuts(arrive_snap)),
            "item",
            arrive_snap.room_item_id,
            "keys",
            arrive_snap.keys,
            "bombs",
            arrive_snap.bombs,
            "compass",
            arrive_dump.get("compass_0x0667"),
            flush=True,
        )

        if not ready or arrive_snap.screen != ROOM_37 or arrive_snap.mode != PLAY_MODE:
            report = {
                "ok": False,
                "reason": "never_play_mode_5_in_0x37",
                "status_claim": None,
                "pokes": False,
                "commands": commands,
                "controller_reused": (
                    "GenericDungeonRoomController + ROOM_5B_SPEC + ROOM_59_SPEC.combat"
                ),
                "rom37": rom37,
                "start": start_dump,
                "arrive": arrive_dump,
                "walked_north": walked,
                "hop": hop,
                "ready": ready,
                "checkpoint": None,
                "whistle_0x065C": arrive_dump.get("whistle_0x065C"),
                "screenshot": str(arrive_png.resolve()),
            }
            write_json_report(RECORDINGS_DIR / "l5_37_clear.json", report)
            return report

        bombs_in = int(arrive_snap.bombs)
        keys_in = int(arrive_snap.keys)
        compass_in = int(read_u8(env.get_ram(), ADDR_COMPASS))
        spec = make_37_spec()
        fight = fight_room(env, assist, total, spec)
        if fight.get("obs") is not None:
            obs = fight["obs"]
        idle(env, assist, total, 30)
        extra = 0
        while extra < 90:
            snap = read_snapshot(env.get_ram())
            if snap.mode == PLAY_MODE and snap.screen == ROOM_37:
                break
            obs = step(env, assist, total, nes_idle_action())
            extra += 1
        idle(env, assist, total, 20)

        ram = env.get_ram()
        mid_snap = read_snapshot(ram)
        dead = (
            mid_snap.screen == ROOM_37
            and mid_snap.mode == PLAY_MODE
            and not live_darknuts(mid_snap)
        )
        compass_pick = None
        if dead:
            compass_pick = grab_compass(env, assist, total, compass_in)
            idle(env, assist, total, 16)

        ram = env.get_ram()
        mid_snap = read_snapshot(ram)
        mid_dump = dump_live(mid_snap, ram)
        dead = (
            mid_snap.screen == ROOM_37
            and mid_snap.mode == PLAY_MODE
            and not live_darknuts(mid_snap)
        )
        bombs_out = int(mid_snap.bombs)
        keys_out = int(mid_snap.keys)
        compass_out = int(read_u8(ram, ADDR_COMPASS))
        png = RECORDINGS_DIR / "l5_37_clear.png"
        obs, *_ = env.step(nes_idle_action())
        total[0] += 1
        assist.apply_env(env, frame=total[0])
        save_rgb_png(obs, png)

        doors_end = decode_doors(mid_snap.cur_opened_doors)
        mask_end = decode_doors(mid_snap.open_doorway_mask)
        state_bytes = env.em.get_state()
        print(
            "MID_CLEAR dead",
            dead,
            "frames",
            fight["frames"],
            "kills",
            fight["kills"],
            "end_n",
            fight["end_n"],
            "keys",
            keys_in,
            "->",
            keys_out,
            "compass",
            hex(compass_in),
            "->",
            hex(compass_out),
            "l5",
            bool(compass_out & LEVEL5_COMPASS_BIT),
            "doors",
            doors_end,
            "mask",
            mask_end,
            flush=True,
        )
        env.close()
        env = None

        post_clear_doors = {
            "from_room": "0x37",
            "all_3_dead": dead,
            "doors": doors_end,
            "doorway_mask": mask_end,
            "N": bool(doors_end.get("north") or mask_end.get("north") or rom37["N"] == "open"),
            "S": bool(doors_end.get("south") or mask_end.get("south") or rom37["S"] == "open"),
            "E": bool(doors_end.get("east") or mask_end.get("east")),
            "W": bool(doors_end.get("west") or mask_end.get("west")),
            "rom_N": rom37["N"],
            "rom_S": rom37["S"],
            "rom_E": rom37["E"],
            "rom_W": rom37["W"],
            "keys": keys_out,
            "bombs": bombs_out,
            "room_item_id": mid_snap.room_item_id,
            "compass_l5": bool(compass_out & LEVEL5_COMPASS_BIT),
        }

        # ROM N/S open, E/W wall. Walk N/S; add E/W only if live looks open.
        walk_dirs = ["UP", "DOWN"]
        live_east = bool(doors_end.get("east") or mask_end.get("east"))
        live_west = bool(doors_end.get("west") or mask_end.get("west"))
        if live_east:
            walk_dirs.append("RIGHT")
        if live_west:
            walk_dirs.append("LEFT")
        probes = []
        if dead:
            for direction in walk_dirs:
                entered = walk_exit(state_bytes, direction, total)
                probes.append(entered)
                print(
                    "EXIT",
                    direction,
                    entered.get("dest_room") or "sealed",
                    flush=True,
                )

        saved = None
        if dead:
            path = write_state_bytes(
                state_path(GAME_DIR, GAME, "Level5Cleared37"),
                state_bytes,
            )
            write_state_provenance(
                path,
                source_state_path=(
                    GAME_DIR / "custom_integrations" / GAME / f"{STATE}.state"
                ),
                request={
                    "segment": "Level5Cleared37",
                    "predecessor_entry": True,
                    "start_state": STATE,
                    "key_poke": False,
                    "door_poke": False,
                    "bomb_count_poke": False,
                    "selected_item_poke": False,
                    "controller": "GenericDungeonRoomController",
                    "spec_base": ROOM_5B_SPEC.spec_id,
                    "combat_base": ROOM_59_SPEC.spec_id,
                    "alive_rule": "hp",
                },
                selected_trial={
                    "success": True,
                    "frames": fight["frames"],
                    "room": ROOM_37,
                    "live_darknuts": 0,
                    "doors_after": int(mid_snap.cur_opened_doors) & 0x0F,
                    "doorway_mask": int(mid_snap.open_doorway_mask) & 0x0F,
                    "bombs": bombs_out,
                    "keys": keys_out,
                    "compass_grabbed": bool(
                        compass_pick and compass_pick.get("grabbed")
                    ),
                    "compass_l5": bool(compass_out & LEVEL5_COMPASS_BIT),
                },
                natural_entry=False,
            )
            saved = "Level5Cleared37"

        new_dests = [
            p
            for p in probes
            if p.get("success") and int(p.get("dest_room_id") or -1) not in KNOWN_ROOMS
        ]
        all_known_or_sealed = bool(probes) and all(
            (not p.get("success")) or (int(p.get("dest_room_id") or -1) in KNOWN_ROOMS)
            for p in probes
        )
        pocket = bool(dead) and all_known_or_sealed

        exits_report = {
            "from_room": "0x37",
            "all_3_dead": dead,
            "rom": rom37,
            "rom27": rom27,
            "doors": doors_end,
            "doorway_mask": mask_end,
            "walked": walk_dirs,
            "skipped_east": not live_east,
            "skipped_west": not live_west,
            "probes": [
                {k: v for k, v in p.items() if k != "dump"} for p in probes
            ],
            "new_dests": [
                {k: v for k, v in p.items() if k != "dump"} for p in new_dests
            ],
            "pocket": pocket,
            "status_claim": None,
            "pokes": False,
        }
        write_json_report(RECORDINGS_DIR / "l5_37_exits.json", exits_report)

        whistle = (mid_dump.get("inventory") or {}).get("whistle_0x065C", 0)
        fight_out = {k: v for k, v in fight.items() if k != "obs"}
        next_dest = None
        if new_dests:
            next_dest = new_dests[0].get("dest_room")
        elif pocket:
            next_dest = "pocket"
        elif probes:
            next_dest = "known_or_sealed"

        report = {
            "ok": dead,
            "status_claim": None,
            "from_state": STATE,
            "pokes": False,
            "commands": commands,
            "controller_reused": "GenericDungeonRoomController",
            "spec_reused": ROOM_5B_SPEC.spec_id,
            "combat_reused": ROOM_59_SPEC.spec_id,
            "spec_id": spec.spec_id,
            "rom37": rom37,
            "rom27": rom27,
            "rom47": rom47,
            "walked_north": walked,
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
                "darknuts": len(live_darknuts(arrive_snap)),
                "compass": compass_in,
            },
            "clear": {
                **fight_out,
                "bombs_in": bombs_in,
                "bombs_out": bombs_out,
                "keys_in": keys_in,
                "keys_out": keys_out,
                "compass_in": compass_in,
                "compass_out": compass_out,
                "compass_l5": bool(compass_out & LEVEL5_COMPASS_BIT),
                "dead": dead,
                "compass_pick": compass_pick,
            },
            "post_clear_doors": post_clear_doors,
            "doors_end": doors_end,
            "doorway_mask_end": mask_end,
            "exits": [{k: v for k, v in p.items() if k != "dump"} for p in probes],
            "new_dests": [
                {k: v for k, v in p.items() if k != "dump"} for p in new_dests
            ],
            "pocket": pocket,
            "next_dest": next_dest,
            "checkpoint": saved,
            "checkpoint_reason": (
                "all 3 Darknuts dead in play mode 5"
                if saved
                else "not saved: enemies still alive"
            ),
            "whistle_0x065C": whistle,
            "frames_total": total[0],
            "screenshot": str(png.resolve()),
            "dump": mid_dump,
        }
        write_json_report(RECORDINGS_DIR / "l5_37_clear.json", report)
        return report
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    r = main()
    print("CMD", r["commands"])
    print("CONTROLLER", r.get("controller_reused"), "SPEC", r.get("spec_reused"),
          "COMBAT", r.get("combat_reused"))
    print("ROM37", r.get("rom37"))
    a = r.get("arrive") or {}
    print(
        "ARRIVE",
        a.get("room"),
        "mode",
        a.get("mode"),
        a.get("mode_name"),
        "xy",
        a.get("xy"),
        "darknuts",
        a.get("darknuts"),
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
        "compass",
        c.get("compass_in"),
        "->",
        c.get("compass_out"),
        "l5",
        c.get("compass_l5"),
    )
    print("COMPASS_PICK", c.get("compass_pick"))
    print("POST_CLEAR_DOORS", r.get("post_clear_doors"))
    print("EXITS")
    for p in r.get("exits") or []:
        print(
            " ",
            p.get("direction"),
            "dest" if p.get("success") else "sealed",
            p.get("dest_room"),
        )
    print("POCKET", r.get("pocket"), "NEXT_DEST", r.get("next_dest"))
    print("CKPT", r.get("checkpoint"), r.get("checkpoint_reason"))
    print("WHISTLE", r.get("whistle_0x065C"))
    print("FRAMES", r.get("frames_total"))
    print("status_claim", None)
