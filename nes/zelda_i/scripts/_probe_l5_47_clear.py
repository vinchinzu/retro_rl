"""Clear L5 0x47 5x Gibdo from Level5Cleared57, dump doors, walk N/S/W.

Start: Level5Cleared57. Walk UP into 0x47. Wait for play mode 5.
Reuse GenericDungeonRoomController + ROOM_66_SPEC combat (AliveRule.TYPE_AND_HP).
5x Gibdo 0x30 HP=112; grab small key 0x19 if it drops (walk, no RAM poke).
No pokes, no candle invent, no Clean STATUS, no east67, no 0x65 bombs.
Level5Cleared47 only if all 5 Gibdos dead.
ROM N/S/W=open, E=wall — walk N/S/W; skip E unless live looks open.
If 0x47 is a pocket (all dests known/sealed), bomb 0x66 WEST once from
Level5Cleared56 (ROM W=bomb). One bomb. Success only if dest room changes.
"""
from __future__ import annotations

import zipfile
from dataclasses import replace
from pathlib import Path

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
from zelda_i.dungeon_lab import _drive_exit
from zelda_i.dungeon_ops import exit_door, goto, idle
from zelda_i.dungeon_trace import compact_snapshot, write_state_provenance
from zelda_i.level5_dungeon import GIBDO_OBJECT_TYPE, ROOM_66_SPEC, ROOM_ITEM_SMALL_KEY
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

STATE = "Level5Cleared57"
ROOM_57 = 0x57
ROOM_47 = 0x47
ROOM_66 = 0x66
MAX_FIGHT_FRAMES = 28000
KNOWN_ROOMS = {0x46, 0x47, 0x55, 0x56, 0x57, 0x65, 0x66, 0x67, 0x76, 0x77}

# L1-6 Q1 tables (backtrack: 0x66 W=bomb / E=key / N=shutter / S=open).
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

EXIT_ROUTES: dict[str, DoorRoute] = {
    "UP": DoorRoute("UP", ((120, 141), (120, 93))),
    "DOWN": DoorRoute("DOWN", ((120, 141), (120, 205))),
    "LEFT": DoorRoute("LEFT", ((120, 141), (32, 141))),
    "RIGHT": DoorRoute("RIGHT", ((120, 141), (208, 141))),
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
)


def make_47_spec() -> DungeonRoomSpec:
    """ROOM_66_SPEC combat / liveness, retargeted to 0x47 5x Gibdo."""
    return replace(
        ROOM_66_SPEC,
        spec_id="level5_room47_gibdos_reuse66",
        source_room=ROOM_57,
        room_id=ROOM_47,
        entry=DoorRoute("UP", ((120, 205),)),
        expected_enemy_count=5,
        required_open_doors=0,
        reward=RewardSpec(kind=RewardKind.CLEAR_ONLY),
        room_item_id=ROOM_ITEM_SMALL_KEY,
        exit_routes=(
            DoorRoute("UP", ((120, 93),)),
            DoorRoute("DOWN", ((120, 205),)),
            DoorRoute("LEFT", ((32, 141),)),
        ),
        max_frames=MAX_FIGHT_FRAMES,
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


def live_gibdos(snap: ZeldaSnapshot) -> list:
    if snap.mode != PLAY_MODE:
        return []
    return [
        obj
        for obj in snap.objects
        if 1 <= obj.slot <= 12
        and obj.type_id == GIBDO_OBJECT_TYPE
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


def grab_key(env, assist, total, keys0: int) -> dict:
    """Walk to typical 0x19 drop tiles. No RAM poke."""
    tried = []
    for tx, ty in KEY_WAYPOINTS:
        snap = read_snapshot(env.get_ram())
        if snap.screen != ROOM_47 or snap.mode != PLAY_MODE:
            break
        if int(snap.keys) > keys0:
            break
        goto(env, assist, total, tx, ty, tol=3, max_f=350)
        idle(env, assist, total, 12)
        snap = read_snapshot(env.get_ram())
        tried.append(
            {
                "stand": [tx, ty],
                "xy": [snap.link_x, snap.link_y],
                "keys": int(snap.keys),
            }
        )
        if int(snap.keys) > keys0:
            break
    snap = read_snapshot(env.get_ram())
    keys1 = int(snap.keys)
    return {
        "keys_in": keys0,
        "keys_out": keys1,
        "grabbed": keys1 > keys0,
        "delta": keys1 - keys0,
        "tried_n": len(tried),
        "xy": [snap.link_x, snap.link_y],
        "pokes": False,
    }


def probe_exits(state_data: bytes, room_id: int, directions: list[str]) -> list[dict]:
    results = []
    for direction in directions:
        route = EXIT_ROUTES[direction]
        png = RECORDINGS_DIR / f"l5_{room_id:02x}_exit_{direction.lower()}.png"
        raw = _drive_exit(
            state_data,
            spec_room=room_id,
            route=route,
            screenshot_path=png,
            max_frames=900,
        )
        dest = None
        sealed = not raw.get("success")
        if raw.get("success"):
            dest = raw.get("room_hex") or f"0x{raw.get('room', 0):02x}"
            if isinstance(dest, str):
                dest = dest.lower()
        results.append(
            {
                "direction": direction,
                "success": bool(raw.get("success")),
                "sealed": sealed,
                "dest_room": dest if not sealed else None,
                "dest_room_id": raw.get("room") if raw.get("success") else None,
                "frames": raw.get("frames"),
                "objects": raw.get("objects"),
                "room_item_id": raw.get("room_item_id"),
                "room_item_name": raw.get("room_item_name"),
                "x": raw.get("x"),
                "y": raw.get("y"),
                "mode": raw.get("mode"),
                "screenshot": raw.get("screenshot"),
            }
        )
    return results


def walk_and_dump(state_data: bytes, direction: str, dest_id: int, total: list[int]) -> dict:
    env = None
    try:
        env, assist, obs = open_from_bytes(state_data)
        keys0 = int(read_snapshot(env.get_ram()).keys)
        hop = exit_door(env, assist, total, direction)
        dest = hop.get("after", {}).get("screen")
        wait_play(env, assist, total, dest if dest is not None else dest_id, max_f=240)
        idle(env, assist, total, 20)
        ram = env.get_ram()
        snap = read_snapshot(ram)
        dump = dump_live(snap, ram)
        png = RECORDINGS_DIR / f"l5_{snap.screen:02x}_from47.png"
        obs, *_ = env.step(nes_idle_action())
        total[0] += 1
        assist.apply_env(env, frame=total[0])
        save_rgb_png(obs, png)
        json_path = RECORDINGS_DIR / f"l5_{snap.screen:02x}_from47.json"
        write_json_report(
            json_path,
            {
                "via": f"0x47 {direction}",
                "ok": hop.get("changed_room"),
                "keys_in": keys0,
                "keys_out": int(snap.keys),
                "dump": dump,
                "screenshot": str(png.resolve()),
                "status_claim": None,
                "pokes": False,
            },
        )
        return {
            "ok": hop.get("changed_room"),
            "via": f"0x47 {direction}",
            "room": f"0x{snap.screen:02x}",
            "mode": snap.mode,
            "xy": [snap.link_x, snap.link_y],
            "keys_in": keys0,
            "keys_out": int(snap.keys),
            "bombs": snap.bombs,
            "doors": dump.get("doors"),
            "doorway_mask": dump.get("doorway_mask"),
            "objects": dump.get("objects"),
            "room_item_id": snap.room_item_id,
            "known": int(snap.screen) in KNOWN_ROOMS,
            "screenshot": str(png.resolve()),
            "dump_path": str(json_path.resolve()),
        }
    finally:
        if env is not None:
            env.close()


def bomb_west_66() -> dict:
    """One west bomb from Level5Cleared56 → 0x66. No pokes. Success iff dest changes."""
    configure_headless()
    env, assist, obs = open_env("Level5Cleared56")
    total = [1]
    try:
        idle(env, assist, total, 20)
        start56 = dump_live(read_snapshot(env.get_ram()), env.get_ram())
        hop = exit_door(env, assist, total, "DOWN")
        wait_play(env, assist, total, ROOM_66, max_f=240)
        idle(env, assist, total, 20)
        at66 = dump_live(read_snapshot(env.get_ram()), env.get_ram())
        goto(env, assist, total, 40, 141, tol=3, max_f=500)
        for _ in range(8):
            step(env, assist, total, nes_action("LEFT"))
        idle(env, assist, total, 8)
        before = dump_live(read_snapshot(env.get_ram()), env.get_ram())
        obs, *_ = env.step(nes_idle_action())
        total[0] += 1
        assist.apply_env(env, frame=total[0])
        before_png = RECORDINGS_DIR / "l5_66_west_bomb_before.png"
        save_rgb_png(obs, before_png)

        selected0 = int(read_u8(env.get_ram(), ADDR_SELECTED_ITEM))
        bombs0 = int(read_snapshot(env.get_ram()).bombs)
        room0 = int(read_snapshot(env.get_ram()).screen)
        # Place one bomb with B. No selected-item / bomb-count poke.
        # ram.py: 0x0656 1=bombs. If not selected, open pause and step RIGHT to bombs.
        menu = None
        if selected0 != 1:
            step(env, assist, total, nes_action("START"))
            idle(env, assist, total, 20)
            for _ in range(3):
                step(env, assist, total, nes_action("RIGHT"))
                idle(env, assist, total, 6)
            step(env, assist, total, nes_action("START"))
            idle(env, assist, total, 24)
            menu = {
                "used": True,
                "selected_before": selected0,
                "selected_after": int(read_u8(env.get_ram(), ADDR_SELECTED_ITEM)),
            }
        else:
            menu = {"used": False, "selected": selected0, "reason": "already_bombs_1"}

        step(env, assist, total, nes_action("LEFT", "B"))
        idle(env, assist, total, 110)
        bombs1 = int(read_snapshot(env.get_ram()).bombs)
        for _ in range(160):
            snap = read_snapshot(env.get_ram())
            if snap.screen != room0 and snap.mode == PLAY_MODE:
                break
            step(env, assist, total, nes_action("LEFT"))
        idle(env, assist, total, 40)
        after = dump_live(read_snapshot(env.get_ram()), env.get_ram())
        obs, *_ = env.step(nes_idle_action())
        total[0] += 1
        assist.apply_env(env, frame=total[0])
        after_png = RECORDINGS_DIR / "l5_66_west_bomb.png"
        save_rgb_png(obs, after_png)
        dest_changed = after.get("room") != before.get("room")
        report = {
            "from_state": "Level5Cleared56",
            "pokes": False,
            "status_claim": None,
            "rom": rom_room(ROOM_66),
            "hop_down": {
                "changed": hop.get("changed_room"),
                "result": hop.get("result"),
            },
            "start56": {
                "room": start56.get("room_hex"),
                "keys": start56.get("keys"),
                "bombs": start56.get("bombs"),
            },
            "at66": at66,
            "before": before,
            "after": after,
            "menu": menu,
            "bombs_in": bombs0,
            "bombs_out": int(after.get("bombs") or bombs1),
            "bombs_spent": bombs0 - int(after.get("bombs") or bombs1),
            "one_bomb": True,
            "dest_changed": dest_changed,
            "dest_room": after.get("room_hex"),
            "success": dest_changed,
            "before_screenshot": str(before_png.resolve()),
            "screenshot": str(after_png.resolve()),
            "whistle_0x065C": after.get("whistle_0x065C"),
        }
        write_json_report(RECORDINGS_DIR / "l5_66_west_bomb.json", report)
        print(
            "BOMB66W dest_changed",
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
        return report
    finally:
        env.close()


def main() -> dict:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    commands = [
        "PYTHONPATH=nes:. uv run python nes/zelda_i/scripts/_probe_l5_47_clear.py  # Level5Cleared57 UP 0x47, infinite-life, GenericDungeonRoomController+ROOM_66_SPEC, 28000f"
    ]
    rom47 = rom_room(ROOM_47)
    rom66 = rom_room(ROOM_66)
    print("ROM47", rom47, flush=True)
    print("ROM66", rom66, flush=True)

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
        if start_snap.screen != ROOM_47:
            hop = exit_door(env, assist, total, "UP")
            walked = bool(hop.get("changed_room"))
            print("UP_HOP", hop.get("changed_room"), hop.get("result"), flush=True)

        ready = wait_play(env, assist, total, ROOM_47, max_f=360)
        ram = env.get_ram()
        arrive_snap = read_snapshot(ram)
        arrive_dump = dump_live(arrive_snap, ram)
        arrive_png = RECORDINGS_DIR / "l5_47_arrive.png"
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
            "gibdos",
            len(live_gibdos(arrive_snap)),
            "item",
            arrive_snap.room_item_id,
            "keys",
            arrive_snap.keys,
            "bombs",
            arrive_snap.bombs,
            flush=True,
        )

        if not ready or arrive_snap.screen != ROOM_47 or arrive_snap.mode != PLAY_MODE:
            report = {
                "ok": False,
                "reason": "never_play_mode_5_in_0x47",
                "status_claim": None,
                "pokes": False,
                "commands": commands,
                "controller_reused": "GenericDungeonRoomController + ROOM_66_SPEC",
                "rom47": rom47,
                "start": start_dump,
                "arrive": arrive_dump,
                "walked_up": walked,
                "hop": hop,
                "ready": ready,
                "checkpoint": None,
                "whistle_0x065C": arrive_dump.get("whistle_0x065C"),
                "screenshot": str(arrive_png.resolve()),
            }
            write_json_report(RECORDINGS_DIR / "l5_47_clear.json", report)
            return report

        bombs_in = int(arrive_snap.bombs)
        keys_in = int(arrive_snap.keys)
        spec = make_47_spec()
        fight = fight_room(env, assist, total, spec)
        if fight.get("obs") is not None:
            obs = fight["obs"]
        idle(env, assist, total, 30)
        extra = 0
        while extra < 90:
            snap = read_snapshot(env.get_ram())
            if snap.mode == PLAY_MODE and snap.screen == ROOM_47:
                break
            obs = step(env, assist, total, nes_idle_action())
            extra += 1
        idle(env, assist, total, 20)

        ram = env.get_ram()
        mid_snap = read_snapshot(ram)
        dead = (
            mid_snap.screen == ROOM_47
            and mid_snap.mode == PLAY_MODE
            and not live_gibdos(mid_snap)
        )
        key_pick = None
        if dead:
            key_pick = grab_key(env, assist, total, int(mid_snap.keys))
            idle(env, assist, total, 16)

        ram = env.get_ram()
        mid_snap = read_snapshot(ram)
        mid_dump = dump_live(mid_snap, ram)
        dead = (
            mid_snap.screen == ROOM_47
            and mid_snap.mode == PLAY_MODE
            and not live_gibdos(mid_snap)
        )
        bombs_out = int(mid_snap.bombs)
        keys_out = int(mid_snap.keys)
        png = RECORDINGS_DIR / "l5_47_clear.png"
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
            "doors",
            doors_end,
            "mask",
            mask_end,
            flush=True,
        )
        env.close()
        env = None

        post_clear_doors = {
            "from_room": "0x47",
            "all_5_dead": dead,
            "doors": doors_end,
            "doorway_mask": mask_end,
            "keys": keys_out,
            "bombs": bombs_out,
            "room_item_id": mid_snap.room_item_id,
        }

        walk_dirs = ["UP", "DOWN", "LEFT"]
        live_east = bool(doors_end.get("east") or mask_end.get("east"))
        if live_east:
            walk_dirs.append("RIGHT")
        probes = []
        if dead:
            probes = probe_exits(state_bytes, ROOM_47, walk_dirs)
        print(
            "PROBES",
            [(p.get("direction"), p.get("dest_room") or "sealed") for p in probes],
            flush=True,
        )

        saved = None
        if dead:
            path = write_state_bytes(
                state_path(GAME_DIR, GAME, "Level5Cleared47"),
                state_bytes,
            )
            write_state_provenance(
                path,
                source_state_path=(
                    GAME_DIR / "custom_integrations" / GAME / f"{STATE}.state"
                ),
                request={
                    "segment": "Level5Cleared47",
                    "predecessor_entry": True,
                    "start_state": STATE,
                    "key_poke": False,
                    "door_poke": False,
                    "bomb_count_poke": False,
                    "selected_item_poke": False,
                    "controller": "GenericDungeonRoomController",
                    "spec_base": ROOM_66_SPEC.spec_id,
                    "alive_rule": "hp",
                },
                selected_trial={
                    "success": True,
                    "frames": fight["frames"],
                    "room": ROOM_47,
                    "live_gibdos": 0,
                    "doors_after": int(mid_snap.cur_opened_doors) & 0x0F,
                    "doorway_mask": int(mid_snap.open_doorway_mask) & 0x0F,
                    "bombs": bombs_out,
                    "keys": keys_out,
                    "key_grabbed": bool(key_pick and key_pick.get("grabbed")),
                },
                natural_entry=False,
            )
            saved = "Level5Cleared47"

        next_rooms = []
        if dead:
            for p in probes:
                if not p.get("success"):
                    continue
                dest_id = p.get("dest_room_id")
                direction = p["direction"]
                entered = walk_and_dump(state_bytes, direction, dest_id, total)
                next_rooms.append(entered)
                print("ENTERED", entered.get("via"), entered.get("room"), flush=True)

        dest_ids = []
        for p in probes:
            if p.get("success") and p.get("dest_room_id") is not None:
                dest_ids.append(int(p["dest_room_id"]))
        all_known_or_sealed = bool(probes) and all(
            (not p.get("success")) or (int(p.get("dest_room_id") or -1) in KNOWN_ROOMS)
            for p in probes
        )
        pocket = bool(dead) and all_known_or_sealed
        new_dests = [
            p
            for p in probes
            if p.get("success") and int(p.get("dest_room_id") or -1) not in KNOWN_ROOMS
        ]

        bomb66 = None
        if pocket:
            print("POCKET — bombing 0x66 WEST once", flush=True)
            bomb66 = bomb_west_66()

        exits_report = {
            "from_room": "0x47",
            "all_5_dead": dead,
            "rom": rom47,
            "doors": doors_end,
            "doorway_mask": mask_end,
            "walked": walk_dirs,
            "skipped_east": not live_east,
            "probes": probes,
            "next_rooms": next_rooms,
            "pocket": pocket,
            "new_dests": new_dests,
            "status_claim": None,
            "pokes": False,
        }
        write_json_report(RECORDINGS_DIR / "l5_47_exits.json", exits_report)

        whistle = (mid_dump.get("inventory") or {}).get("whistle_0x065C", 0)
        fight_out = {k: v for k, v in fight.items() if k != "obs"}
        next_dest = None
        if new_dests:
            next_dest = new_dests[0].get("dest_room")
        elif pocket:
            if bomb66 and bomb66.get("dest_changed"):
                next_dest = bomb66.get("dest_room")
            else:
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
            "spec_reused": ROOM_66_SPEC.spec_id,
            "spec_id": spec.spec_id,
            "rom47": rom47,
            "rom66": rom66,
            "walked_up": walked,
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
                "gibdos": len(live_gibdos(arrive_snap)),
            },
            "clear": {
                **fight_out,
                "bombs_in": bombs_in,
                "bombs_out": bombs_out,
                "keys_in": keys_in,
                "keys_out": keys_out,
                "dead": dead,
                "key_pick": key_pick,
            },
            "post_clear_doors": post_clear_doors,
            "doors_end": doors_end,
            "doorway_mask_end": mask_end,
            "exits": probes,
            "next_rooms": next_rooms,
            "pocket": pocket,
            "new_dests": new_dests,
            "next_dest": next_dest,
            "bomb66_west": bomb66,
            "checkpoint": saved,
            "checkpoint_reason": (
                "all 5 Gibdos dead in play mode 5"
                if saved
                else "not saved: enemies still alive"
            ),
            "whistle_0x065C": whistle,
            "frames_total": total[0],
            "screenshot": str(png.resolve()),
            "dump": mid_dump,
        }
        write_json_report(RECORDINGS_DIR / "l5_47_clear.json", report)
        return report
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    r = main()
    print("CMD", r["commands"])
    print("CONTROLLER", r.get("controller_reused"), "SPEC", r.get("spec_reused"))
    print("ROM47", r.get("rom47"))
    a = r.get("arrive") or {}
    print(
        "ARRIVE",
        a.get("room"),
        "mode",
        a.get("mode"),
        a.get("mode_name"),
        "xy",
        a.get("xy"),
        "gibdos",
        a.get("gibdos"),
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
    print("KEY_PICK", c.get("key_pick"))
    print("POST_CLEAR_DOORS", r.get("post_clear_doors"))
    print("EXITS")
    for p in r.get("exits") or []:
        print(" ", p.get("direction"), "dest" if p.get("success") else "sealed", p.get("dest_room"))
    print("NEXT_ROOMS", [(n.get("via"), n.get("room"), n.get("known")) for n in (r.get("next_rooms") or [])])
    print("POCKET", r.get("pocket"), "NEXT_DEST", r.get("next_dest"))
    print("BOMB66W", None if not r.get("bomb66_west") else {
        "success": r["bomb66_west"].get("success"),
        "dest": r["bomb66_west"].get("dest_room"),
        "bombs": (r["bomb66_west"].get("bombs_in"), r["bomb66_west"].get("bombs_out")),
    })
    print("CKPT", r.get("checkpoint"), r.get("checkpoint_reason"))
    print("WHISTLE", r.get("whistle_0x065C"))
    print("FRAMES", r.get("frames_total"))
    print("status_claim", None)
