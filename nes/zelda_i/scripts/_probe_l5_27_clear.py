"""Clear L5 0x27 mixed room from Level5Cleared37, dump doors, spend west key.

Start: Level5Cleared37. Walk NORTH from ladder (120,141). Avoid x=56 pit
column; cross at y≈109 if needed. Wait play mode 5.

Reuse:
- Pols 0x16: Level5PolsVoiceController / ROOM_77 combat (L5 east key)
- Gibdo 0x30: ROOM_66_SPEC combat (same as 0x47/0x65)
- Keese 0x1b: generic TYPE_AND_HP + type_only (HP=0 while alive)

One mixed GenericDungeonRoomController (Level5PolsVoiceController subclass)
with type_only Keese; sequential leftover sweeps if needed.

After 6/6 dead: dump doors BEFORE key spend, grab floor key 0x19 if still
there (walk, no poke), walk WEST (ROM W=key), dump dest.
Level5Cleared27 only if six dead. No pokes, no candle invent, no Clean
STATUS, no east67, no 0x65 bombs.
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
    AliveRule,
    DoorRoute,
    DungeonPhase,
    DungeonRoomSpec,
    GenericDungeonRoomController,
    KEESE_OBJECT_TYPE,
    RewardKind,
    RewardSpec,
)
from zelda_i.dungeon_ids import object_name
from zelda_i.dungeon_ops import idle, push_dir
from zelda_i.dungeon_trace import compact_snapshot, write_state_provenance
from zelda_i.level5_dungeon import (
    GIBDO_OBJECT_TYPE,
    LEVEL_5,
    Level5PolsVoiceController,
    POLS_VOICE_OBJECT_TYPE,
    ROOM_27_SPEC,
    ROOM_66_SPEC,
    ROOM_77_SPEC,
    ROOM_ITEM_SMALL_KEY,
    ROOM_L5_NORTH_27,
)
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

STATE = "Level5Cleared37"
ROOM_37 = 0x37
ROOM_27 = ROOM_L5_NORTH_27
MAX_FIGHT_FRAMES = 28000
LEVEL5_COMPASS_BIT = 0x10

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

KEY_WAYPOINTS: tuple[tuple[int, int], ...] = (
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


def make_mixed_spec() -> DungeonRoomSpec:
    return replace(
        ROOM_27_SPEC,
        spec_id="level5_room27_mixed_reuse77",
        entry=DoorRoute("UP", ((120, 205),)),
        combat=ROOM_77_SPEC.combat,
        reward=RewardSpec(kind=RewardKind.CLEAR_ONLY, settle_all_dead=0),
        required_open_doors=0,
        max_frames=MAX_FIGHT_FRAMES,
    )


def make_pols_spec(n: int) -> DungeonRoomSpec:
    return replace(
        ROOM_77_SPEC,
        spec_id="level5_room27_pols_reuse77",
        source_room=ROOM_37,
        room_id=ROOM_27,
        entry=DoorRoute("UP", ((120, 205),)),
        enemy_types=(POLS_VOICE_OBJECT_TYPE,),
        expected_enemy_count=max(1, n),
        alive_rule=AliveRule.TYPE_AND_HP,
        type_only_enemy_types=(),
        reward=RewardSpec(kind=RewardKind.CLEAR_ONLY, settle_all_dead=0),
        required_open_doors=0,
        room_item_id=ROOM_ITEM_SMALL_KEY,
        max_frames=18000,
        level=LEVEL_5,
    )


def make_gibdo_spec(n: int) -> DungeonRoomSpec:
    return replace(
        ROOM_66_SPEC,
        spec_id="level5_room27_gibdo_reuse66",
        source_room=ROOM_37,
        room_id=ROOM_27,
        entry=DoorRoute("UP", ((120, 205),)),
        enemy_types=(GIBDO_OBJECT_TYPE,),
        expected_enemy_count=max(1, n),
        reward=RewardSpec(kind=RewardKind.CLEAR_ONLY, settle_all_dead=0),
        required_open_doors=0,
        max_frames=20000,
        level=LEVEL_5,
    )


def make_keese_spec(n: int) -> DungeonRoomSpec:
    return replace(
        ROOM_27_SPEC,
        spec_id="level5_room27_keese_type_hp",
        entry=DoorRoute("UP", ((120, 205),)),
        enemy_types=(KEESE_OBJECT_TYPE,),
        expected_enemy_count=max(1, n),
        alive_rule=AliveRule.TYPE_AND_HP,
        type_only_enemy_types=(KEESE_OBJECT_TYPE,),
        combat=ROOM_77_SPEC.combat,
        reward=RewardSpec(kind=RewardKind.CLEAR_ONLY, settle_all_dead=0),
        required_open_doors=0,
        max_frames=8000,
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


def live_by_type(snap: ZeldaSnapshot) -> dict[str, list]:
    if snap.mode != PLAY_MODE:
        return {"pols": [], "gibdo": [], "keese": []}
    pols, gibdo, keese = [], [], []
    for obj in snap.objects:
        if not (1 <= obj.slot <= 12):
            continue
        if obj.type_id == POLS_VOICE_OBJECT_TYPE and obj.hp > 0:
            pols.append(obj)
        elif obj.type_id == GIBDO_OBJECT_TYPE and obj.hp > 0:
            gibdo.append(obj)
        elif obj.type_id == KEESE_OBJECT_TYPE:
            keese.append(obj)
    return {"pols": pols, "gibdo": gibdo, "keese": keese}


def live_counts(snap: ZeldaSnapshot) -> dict[str, int]:
    groups = live_by_type(snap)
    return {k: len(v) for k, v in groups.items()}


def six_dead(snap: ZeldaSnapshot) -> bool:
    c = live_counts(snap)
    return (
        snap.screen == ROOM_27
        and snap.mode == PLAY_MODE
        and c["pols"] == 0
        and c["gibdo"] == 0
        and c["keese"] == 0
    )


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


def leave_pit_column(env, assist, total) -> dict:
    """If on x=56 pit/ladder column, cross at y≈109 then center x=120."""
    snap = read_snapshot(env.get_ram())
    start = [snap.link_x, snap.link_y]
    notes = []
    if abs(snap.link_x - 56) <= 10:
        notes.append("pit_x56_cross_y109")
        walk_axis(env, assist, total, "y", 109, max_f=300)
        walk_axis(env, assist, total, "x", 120, max_f=400)
        snap = read_snapshot(env.get_ram())
        if abs(snap.link_x - 120) > 3:
            for ty in (117, 93, 173, 189, 157):
                walk_axis(env, assist, total, "y", ty, max_f=250)
                walk_axis(env, assist, total, "x", 120, max_f=250)
                snap = read_snapshot(env.get_ram())
                notes.append(f"retry_y{ty}_x{snap.link_x}")
                if abs(snap.link_x - 120) <= 3:
                    break
    else:
        notes.append("already_off_pit")
        walk_axis(env, assist, total, "x", 120, max_f=300)
    walk_axis(env, assist, total, "y", 141, max_f=300)
    walk_axis(env, assist, total, "x", 120, max_f=200)
    snap = read_snapshot(env.get_ram())
    return {
        "start": start,
        "end": [snap.link_x, snap.link_y],
        "notes": notes,
        "centered": abs(snap.link_x - 120) <= 3,
    }


def walk_north_from_37(env, assist, total) -> dict:
    """From ladder (120,141) walk NORTH into 0x27. Avoid x=56 pit column."""
    snap = read_snapshot(env.get_ram())
    start_xy = [snap.link_x, snap.link_y]
    unstick = leave_pit_column(env, assist, total)
    walk_axis(env, assist, total, "x", 120, max_f=200)
    walk_axis(env, assist, total, "y", 141, max_f=250)
    walk_axis(env, assist, total, "x", 120, max_f=200)
    walk_axis(env, assist, total, "y", 93, max_f=350)
    walk_axis(env, assist, total, "x", 120, max_f=200)
    for _ in range(24):
        snap = read_snapshot(env.get_ram())
        if abs(snap.link_x - 120) <= 1 and abs(snap.link_y - 93) <= 2:
            break
        if abs(snap.link_x - 120) > 1:
            step(env, assist, total, nes_action("RIGHT" if snap.link_x < 120 else "LEFT"))
        else:
            step(env, assist, total, nes_action("DOWN" if snap.link_y < 93 else "UP"))
    at = [read_snapshot(env.get_ram()).link_x, read_snapshot(env.get_ram()).link_y]
    room0 = read_snapshot(env.get_ram()).screen
    push_dir(env, assist, total, "UP", frames=170)
    idle(env, assist, total, 16)
    snap = read_snapshot(env.get_ram())
    changed = snap.screen != room0
    if changed:
        wait_play(env, assist, total, snap.screen, max_f=240)
    snap = read_snapshot(env.get_ram())
    return {
        "changed_room": changed,
        "start_xy": start_xy,
        "unstick": unstick,
        "at_mouth": at,
        "result_room": f"0x{snap.screen:02x}",
        "result_xy": [snap.link_x, snap.link_y],
        "align_x": 120,
    }


def fight_with(env, assist, total, spec: DungeonRoomSpec, controller_cls) -> dict:
    ctl = controller_cls(spec=spec)
    obs = None
    start_counts = None
    last_counts = None
    progress = []
    for _ in range(spec.max_frames):
        snap = read_snapshot(env.get_ram())
        if snap.mode == PLAY_MODE and snap.screen == spec.room_id:
            counts = live_counts(snap)
            if start_counts is None:
                start_counts = dict(counts)
                last_counts = dict(counts)
                progress.append({"f": ctl.frames, "counts": dict(counts)})
            elif counts != last_counts:
                last_counts = dict(counts)
                progress.append({"f": ctl.frames, "counts": dict(counts)})
                print(f"KILL f={ctl.frames} {counts}", flush=True)
        action = ctl.step(snap)
        obs = step(env, assist, total, action.action)
        if ctl.success or ctl.phase is DungeonPhase.FAILED:
            break
    snap = read_snapshot(env.get_ram())
    end_counts = live_counts(snap) if snap.mode == PLAY_MODE else {}
    start_counts = start_counts or {"pols": 0, "gibdo": 0, "keese": 0}
    kills = {
        k: max(0, start_counts.get(k, 0) - end_counts.get(k, 0))
        for k in ("pols", "gibdo", "keese")
    }
    return {
        "obs": obs,
        "ok": bool(ctl.success),
        "frames": ctl.frames,
        "start_counts": start_counts,
        "end_counts": end_counts,
        "kills": kills,
        "progress": progress,
        "controller": ctl.report(),
        "spec_id": spec.spec_id,
        "controller_cls": controller_cls.__name__,
        "max_frames": spec.max_frames,
    }


def grab_floor_key(env, assist, total, keys0: int) -> dict:
    """Walk typical 0x19 drop tiles. No RAM poke."""
    tried = []
    got = int(read_snapshot(env.get_ram()).keys) > keys0
    for tx, ty in KEY_WAYPOINTS:
        snap = read_snapshot(env.get_ram())
        if snap.screen != ROOM_27 or snap.mode != PLAY_MODE:
            break
        if int(snap.keys) > keys0 or snap.room_item_id not in (ROOM_ITEM_SMALL_KEY, 0x19, 25):
            if int(snap.keys) > keys0:
                got = True
                break
        walk_axis(env, assist, total, "y", ty, max_f=220)
        walk_axis(env, assist, total, "x", tx, max_f=220)
        idle(env, assist, total, 10)
        snap = read_snapshot(env.get_ram())
        keys = int(snap.keys)
        tried.append(
            {
                "stand": [tx, ty],
                "xy": [snap.link_x, snap.link_y],
                "keys": keys,
                "item": snap.room_item_id,
            }
        )
        if keys > keys0:
            got = True
            break
    snap = read_snapshot(env.get_ram())
    return {
        "keys_in": keys0,
        "keys_out": int(snap.keys),
        "grabbed": int(snap.keys) > keys0,
        "item": snap.room_item_id,
        "tried_n": len(tried),
        "xy": [snap.link_x, snap.link_y],
        "pokes": False,
        "got": got,
    }


def walk_west(state_data: bytes, total: list[int]) -> dict:
    env = None
    try:
        env, assist, obs = open_from_bytes(state_data)
        idle(env, assist, total, 12)
        keys0 = int(read_snapshot(env.get_ram()).keys)
        # Center then west door. y-first to avoid any leftover pit pinch.
        leave_pit_column(env, assist, total)
        walk_axis(env, assist, total, "y", 141, max_f=300)
        walk_axis(env, assist, total, "x", 120, max_f=250)
        walk_axis(env, assist, total, "y", 141, max_f=250)
        walk_axis(env, assist, total, "x", 32, max_f=500)
        for _ in range(20):
            snap = read_snapshot(env.get_ram())
            if abs(snap.link_x - 32) <= 2 and abs(snap.link_y - 141) <= 2:
                break
            if abs(snap.link_y - 141) > 2:
                step(env, assist, total, nes_action("DOWN" if snap.link_y < 141 else "UP"))
            else:
                step(env, assist, total, nes_action("LEFT" if snap.link_x > 32 else "RIGHT"))
        at = [read_snapshot(env.get_ram()).link_x, read_snapshot(env.get_ram()).link_y]
        room0 = read_snapshot(env.get_ram()).screen
        keys_at = int(read_snapshot(env.get_ram()).keys)
        push_dir(env, assist, total, "LEFT", frames=200)
        idle(env, assist, total, 16)
        snap = read_snapshot(env.get_ram())
        if snap.screen != room0:
            wait_play(env, assist, total, snap.screen, max_f=240)
        idle(env, assist, total, 24)
        ram = env.get_ram()
        snap = read_snapshot(ram)
        dump = dump_live(snap, ram)
        dest = f"0x{snap.screen:02x}"
        changed = dest != "0x27"
        png = RECORDINGS_DIR / "l5_27_west.png"
        obs, *_ = env.step(nes_idle_action())
        total[0] += 1
        assist.apply_env(env, frame=total[0])
        save_rgb_png(obs, png)
        body = {
            "via": "0x27 LEFT key",
            "ok": changed,
            "at_mouth": at,
            "keys_in": keys0,
            "keys_at_door": keys_at,
            "keys_out": int(snap.keys),
            "key_spent": int(snap.keys) < keys0,
            "dump": dump,
            "screenshot": str(png.resolve()),
            "status_claim": None,
            "pokes": False,
        }
        write_json_report(RECORDINGS_DIR / "l5_27_west.json", body)
        return {
            "direction": "LEFT",
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
            "screenshot": str(png.resolve()),
            "dump": dump,
        }
    finally:
        if env is not None:
            env.close()


def main() -> dict:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    commands = [
        "PYTHONPATH=nes:. uv run python nes/zelda_i/scripts/_probe_l5_27_clear.py  "
        "# Level5Cleared37 NORTH 0x27, infinite-life, "
        "Level5PolsVoiceController+ROOM_77 / ROOM_66 Gibdo / Keese TYPE_AND_HP, 28000f"
    ]
    rom27 = rom_room(ROOM_27)
    rom37 = rom_room(ROOM_37)
    print("ROM27", rom27, flush=True)
    print("ROM37", rom37, flush=True)

    env = None
    env, assist, obs = open_env()
    total = [1]
    try:
        idle(env, assist, total, 20)
        ram = env.get_ram()
        start_snap = read_snapshot(ram)
        start_dump = dump_live(start_snap, ram)
        print(
            "START",
            start_dump.get("room_hex"),
            "xy",
            [start_dump.get("x"), start_dump.get("y")],
            "keys",
            start_dump.get("keys"),
            "mode",
            start_dump.get("mode"),
            flush=True,
        )

        hop = None
        walked = False
        if start_snap.screen != ROOM_27:
            hop = walk_north_from_37(env, assist, total)
            walked = bool(hop.get("changed_room"))
            print(
                "NORTH_HOP",
                hop.get("changed_room"),
                hop.get("result_room"),
                hop.get("at_mouth"),
                hop.get("unstick"),
                flush=True,
            )

        ready = wait_play(env, assist, total, ROOM_27, max_f=360)
        ram = env.get_ram()
        arrive_snap = read_snapshot(ram)
        arrive_dump = dump_live(arrive_snap, ram)
        arrive_counts = live_counts(arrive_snap)
        keys_in = int(arrive_snap.keys)
        print(
            "READY",
            ready,
            "room",
            hex(arrive_snap.screen),
            "mode",
            arrive_snap.mode,
            "xy",
            (arrive_snap.link_x, arrive_snap.link_y),
            "counts",
            arrive_counts,
            "item",
            arrive_snap.room_item_id,
            "keys",
            keys_in,
            flush=True,
        )

        if not ready or arrive_snap.screen != ROOM_27 or arrive_snap.mode != PLAY_MODE:
            png = RECORDINGS_DIR / "l5_27_clear.png"
            obs, *_ = env.step(nes_idle_action())
            save_rgb_png(obs, png)
            report = {
                "ok": False,
                "reason": "never_play_mode_5_in_0x27",
                "status_claim": None,
                "pokes": False,
                "commands": commands,
                "rom27": rom27,
                "start": start_dump,
                "arrive": arrive_dump,
                "walked_north": walked,
                "hop": hop,
                "ready": ready,
                "checkpoint": None,
                "whistle_0x065C": arrive_dump.get("whistle_0x065C"),
                "screenshot": str(png.resolve()),
            }
            write_json_report(RECORDINGS_DIR / "l5_27_clear.json", report)
            return report

        mixed = make_mixed_spec()
        fight = fight_with(env, assist, total, mixed, Level5PolsVoiceController)
        if fight.get("obs") is not None:
            obs = fight["obs"]
        idle(env, assist, total, 20)
        snap = read_snapshot(env.get_ram())
        leftover = live_counts(snap)
        sweeps = [fight]
        print("MIXED", fight.get("ok"), leftover, "f", fight.get("frames"), flush=True)

        if leftover.get("pols", 0) > 0:
            sweep = fight_with(env, assist, total, make_pols_spec(leftover["pols"]), Level5PolsVoiceController)
            sweeps.append(sweep)
            idle(env, assist, total, 12)
            leftover = live_counts(read_snapshot(env.get_ram()))
            print("POLS_SWEEP", leftover, flush=True)
        if leftover.get("gibdo", 0) > 0:
            sweep = fight_with(env, assist, total, make_gibdo_spec(leftover["gibdo"]), GenericDungeonRoomController)
            sweeps.append(sweep)
            idle(env, assist, total, 12)
            leftover = live_counts(read_snapshot(env.get_ram()))
            print("GIBDO_SWEEP", leftover, flush=True)
        if leftover.get("keese", 0) > 0:
            sweep = fight_with(env, assist, total, make_keese_spec(leftover["keese"]), GenericDungeonRoomController)
            sweeps.append(sweep)
            idle(env, assist, total, 12)
            leftover = live_counts(read_snapshot(env.get_ram()))
            print("KEESE_SWEEP", leftover, flush=True)

        extra = 0
        while extra < 90:
            snap = read_snapshot(env.get_ram())
            if snap.mode == PLAY_MODE and snap.screen == ROOM_27:
                break
            obs = step(env, assist, total, nes_idle_action())
            extra += 1
        idle(env, assist, total, 16)

        ram = env.get_ram()
        mid_snap = read_snapshot(ram)
        dead = six_dead(mid_snap)
        mid_dump = dump_live(mid_snap, ram)
        doors_before_spend = decode_doors(mid_snap.cur_opened_doors)
        mask_before_spend = decode_doors(mid_snap.open_doorway_mask)
        keys_after_fight = int(mid_snap.keys)
        print(
            "POST_CLEAR dead",
            dead,
            "counts",
            live_counts(mid_snap),
            "doors",
            doors_before_spend,
            "mask",
            mask_before_spend,
            "keys",
            keys_in,
            "->",
            keys_after_fight,
            "item",
            mid_snap.room_item_id,
            flush=True,
        )

        key_pick = None
        if dead:
            key_pick = grab_floor_key(env, assist, total, keys_after_fight)
            idle(env, assist, total, 12)
            print("KEY_PICK", key_pick, flush=True)

        ram = env.get_ram()
        mid_snap = read_snapshot(ram)
        mid_dump = dump_live(mid_snap, ram)
        dead = six_dead(mid_snap)
        keys_out = int(mid_snap.keys)
        png = RECORDINGS_DIR / "l5_27_clear.png"
        obs, *_ = env.step(nes_idle_action())
        total[0] += 1
        assist.apply_env(env, frame=total[0])
        save_rgb_png(obs, png)

        # Recapture doors from the post-clear / pre-west snapshot.
        doors_end = decode_doors(mid_snap.cur_opened_doors)
        mask_end = decode_doors(mid_snap.open_doorway_mask)
        state_bytes = env.em.get_state()
        env.close()
        env = None

        post_clear_doors = {
            "from_room": "0x27",
            "all_6_dead": dead,
            "before_key_spend": True,
            "doors": doors_before_spend,
            "doorway_mask": mask_before_spend,
            "doors_after_key_grab": doors_end,
            "N": bool(doors_before_spend.get("north") or mask_before_spend.get("north")),
            "S": bool(doors_before_spend.get("south") or mask_before_spend.get("south") or rom27["S"] == "open"),
            "E": bool(doors_before_spend.get("east") or mask_before_spend.get("east")),
            "W": bool(doors_before_spend.get("west") or mask_before_spend.get("west")),
            "rom_N": rom27["N"],
            "rom_S": rom27["S"],
            "rom_E": rom27["E"],
            "rom_W": rom27["W"],
            "keys": keys_after_fight,
            "keys_after_grab": keys_out,
            "room_item_id": mid_snap.room_item_id,
        }

        west = None
        if dead:
            west = walk_west(state_bytes, total)
            print(
                "WEST",
                west.get("dest_room") or "sealed",
                "keys",
                west.get("keys_in"),
                "->",
                west.get("keys_out"),
                "spent",
                west.get("key_spent"),
                flush=True,
            )

        saved = None
        if dead:
            path = write_state_bytes(
                state_path(GAME_DIR, GAME, "Level5Cleared27"),
                state_bytes,
            )
            write_state_provenance(
                path,
                source_state_path=(
                    GAME_DIR / "custom_integrations" / GAME / f"{STATE}.state"
                ),
                request={
                    "segment": "Level5Cleared27",
                    "predecessor_entry": True,
                    "start_state": STATE,
                    "key_poke": False,
                    "door_poke": False,
                    "bomb_count_poke": False,
                    "selected_item_poke": False,
                    "controller": "Level5PolsVoiceController",
                    "spec_base": ROOM_27_SPEC.spec_id,
                    "combat_base": ROOM_77_SPEC.spec_id,
                    "gibdo_combat": ROOM_66_SPEC.spec_id,
                    "alive_rule": "hp+type_only_keese",
                },
                selected_trial={
                    "success": True,
                    "frames": sum(s.get("frames", 0) for s in sweeps),
                    "room": ROOM_27,
                    "live_pols": 0,
                    "live_gibdo": 0,
                    "live_keese": 0,
                    "doors_after": int(mid_snap.cur_opened_doors) & 0x0F,
                    "doorway_mask": int(mid_snap.open_doorway_mask) & 0x0F,
                    "keys": keys_out,
                    "key_grabbed": bool(key_pick and key_pick.get("grabbed")),
                },
                natural_entry=False,
            )
            saved = "Level5Cleared27"

        kills_by_type = {
            "pols": max(0, arrive_counts.get("pols", 0) - leftover.get("pols", 0)),
            "gibdo": max(0, arrive_counts.get("gibdo", 0) - leftover.get("gibdo", 0)),
            "keese": max(0, arrive_counts.get("keese", 0) - leftover.get("keese", 0)),
        }
        fight_frames = sum(s.get("frames", 0) for s in sweeps)
        whistle = (mid_dump.get("inventory") or {}).get("whistle_0x065C", 0)
        dest = None
        if west and west.get("success"):
            dest = {
                "room": west.get("dest_room"),
                "room_id": west.get("dest_room_id"),
                "objects": west.get("objects"),
                "doors": west.get("doors"),
                "doorway_mask": west.get("doorway_mask"),
                "item": west.get("room_item_id"),
                "item_name": west.get("room_item_name"),
                "xy": west.get("xy"),
                "keys_in": west.get("keys_in"),
                "keys_out": west.get("keys_out"),
                "key_spent": west.get("key_spent"),
            }

        report = {
            "ok": dead,
            "status_claim": None,
            "from_state": STATE,
            "pokes": False,
            "commands": commands,
            "controller_reused": "Level5PolsVoiceController + GenericDungeonRoomController",
            "spec_reused": ROOM_27_SPEC.spec_id,
            "combat_reused": {
                "pols": ROOM_77_SPEC.spec_id,
                "gibdo": ROOM_66_SPEC.spec_id,
                "keese": "TYPE_AND_HP + type_only 0x1b",
            },
            "rom27": rom27,
            "rom37": rom37,
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
                "counts": arrive_counts,
                "keys": keys_in,
                "room_item_id": arrive_snap.room_item_id,
            },
            "clear": {
                "frames": fight_frames,
                "kills_by_type": kills_by_type,
                "arrive_counts": arrive_counts,
                "end_counts": leftover,
                "dead": dead,
                "keys_in": keys_in,
                "keys_after_fight": keys_after_fight,
                "keys_out": keys_out,
                "key_pick": key_pick,
                "sweeps": [
                    {k: v for k, v in s.items() if k != "obs"} for s in sweeps
                ],
            },
            "post_clear_doors": post_clear_doors,
            "dest_west": dest,
            "west": {k: v for k, v in (west or {}).items() if k != "dump"} if west else None,
            "checkpoint": saved,
            "checkpoint_reason": (
                "all 6 dead in play mode 5 (2 Pols + 2 Gibdo + 2 Keese)"
                if saved
                else "not saved: six not dead"
            ),
            "whistle_0x065C": whistle,
            "frames_total": total[0],
            "screenshot": str(png.resolve()),
            "dump": mid_dump,
        }
        write_json_report(RECORDINGS_DIR / "l5_27_clear.json", report)
        return report
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    r = main()
    print("CMD", r["commands"])
    print("CONTROLLER", r.get("controller_reused"))
    print("ROM27", r.get("rom27"))
    a = r.get("arrive") or {}
    print(
        "ARRIVE",
        a.get("room"),
        "mode",
        a.get("mode"),
        a.get("xy"),
        "counts",
        a.get("counts"),
        "keys",
        a.get("keys"),
        "item",
        a.get("room_item_id"),
    )
    c = r.get("clear") or {}
    print(
        "CLEAR frames",
        c.get("frames"),
        "kills",
        c.get("kills_by_type"),
        "dead",
        c.get("dead"),
        "keys",
        c.get("keys_in"),
        "->",
        c.get("keys_out"),
    )
    print("POST_CLEAR_DOORS", r.get("post_clear_doors"))
    print("DEST_WEST", r.get("dest_west"))
    print("CKPT", r.get("checkpoint"), r.get("checkpoint_reason"))
    print("WHISTLE", r.get("whistle_0x065C"))
    print("FRAMES", r.get("frames_total"))
    print("status_claim", None)
