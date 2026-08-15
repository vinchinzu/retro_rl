"""Clear L5 0x26 5x Gibdo from Level5Cleared27, dump doors, walk west.

Start: Level5Cleared27. Walk WEST (key spend; dest 0x26). Wait play mode 5.
Reuse GenericDungeonRoomController + ROOM_66_SPEC combat (AliveRule.TYPE_AND_HP).
5x Gibdo 0x30 HP=112; grab small key 0x19 if it drops (walk, no RAM poke).
No pokes, no candle invent, no Clean STATUS, no east67, no 0x65 bombs.
Level5Cleared26 only if all 5 Gibdos dead.
ROM N/S=wall W=open E=key secret=foes_item — walk WEST after clear.

0x27 leave: east wall x=208, south y=189 around x=160 ladder pinch, west door y=141.
0x26 leave: align door y=141; detour south/north if moat/C-block pinches.
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
from zelda_i.dungeon_ops import idle
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

STATE = "Level5Cleared27"
ROOM_27 = 0x27
ROOM_26 = 0x26
MAX_FIGHT_FRAMES = 28000

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
    (192, 141),
    (48, 141),
    (120, 189),
    (120, 109),
)


def make_26_spec() -> DungeonRoomSpec:
    """ROOM_66_SPEC combat / liveness, retargeted to 0x26 5x Gibdo."""
    return replace(
        ROOM_66_SPEC,
        spec_id="level5_room26_gibdos_reuse66",
        source_room=ROOM_27,
        room_id=ROOM_26,
        entry=DoorRoute("LEFT", ((224, 141),)),
        expected_enemy_count=5,
        required_open_doors=0,
        reward=RewardSpec(kind=RewardKind.CLEAR_ONLY),
        room_item_id=ROOM_ITEM_SMALL_KEY,
        exit_routes=(
            DoorRoute("LEFT", ((32, 141),)),
            DoorRoute("RIGHT", ((208, 141),)),
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


def push_left(env, assist, total, frames: int = 220) -> None:
    room0 = read_snapshot(env.get_ram()).screen
    for _ in range(frames):
        snap = read_snapshot(env.get_ram())
        if snap.screen != room0 and snap.mode in (PLAY_MODE, 9, 10):
            break
        step(env, assist, total, nes_action("LEFT"))
    idle(env, assist, total, 16)
    snap = read_snapshot(env.get_ram())
    if snap.screen != room0:
        wait_play(env, assist, total, snap.screen, max_f=240)
    idle(env, assist, total, 20)


def walk_west_from_27(env, assist, total) -> dict:
    """Proven 0x27 leave: east wall, south band y=189 around x=160 pinch, west door."""
    log = []
    snap = read_snapshot(env.get_ram())
    keys0 = int(snap.keys)
    log.append({"step": "start", "xy": [snap.link_x, snap.link_y], "tile": snap.colliding_tile})
    steps = (
        ("x", 208),
        ("y", 189),
        ("x", 32),
        ("y", 141),
        ("x", 32),
    )
    for axis, tgt in steps:
        ok = walk_axis(env, assist, total, axis, tgt, max_f=500)
        snap = read_snapshot(env.get_ram())
        rec = {
            "step": f"axis:{axis}:{tgt}",
            "ok": ok,
            "xy": [snap.link_x, snap.link_y],
            "tile": snap.colliding_tile,
            "room": f"0x{snap.screen:02x}",
        }
        log.append(rec)
        print("NAV27", rec, flush=True)
    at = align_door(env, assist, total, 32, 141)
    room0 = read_snapshot(env.get_ram()).screen
    keys_at = int(read_snapshot(env.get_ram()).keys)
    push_left(env, assist, total, frames=220)
    snap = read_snapshot(env.get_ram())
    return {
        "path": "east_wall_south189_west_door",
        "log": log,
        "at_mouth": at,
        "keys_in": keys0,
        "keys_at_door": keys_at,
        "keys_out": int(snap.keys),
        "key_spent": int(snap.keys) < keys0,
        "changed": snap.screen != room0,
        "dest": f"0x{snap.screen:02x}",
        "xy": [snap.link_x, snap.link_y],
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
    """Walk typical 0x19 drop tiles. No RAM poke. South/north bands if pinched."""
    tried = []
    for tx, ty in KEY_WAYPOINTS:
        snap = read_snapshot(env.get_ram())
        if snap.screen != ROOM_26 or snap.mode != PLAY_MODE:
            break
        if int(snap.keys) > keys0:
            break
        # y then x — avoid mid-room C/moat pinch when possible
        walk_axis(env, assist, total, "y", ty, max_f=280)
        walk_axis(env, assist, total, "x", tx, max_f=280)
        idle(env, assist, total, 10)
        snap = read_snapshot(env.get_ram())
        tried.append(
            {
                "stand": [tx, ty],
                "xy": [snap.link_x, snap.link_y],
                "keys": int(snap.keys),
                "item": snap.room_item_id,
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
        "item": snap.room_item_id,
        "pokes": False,
    }


def walk_west_from_26(env, assist, total) -> dict:
    """ROM W=open. Align y=141; detour south/north around moat/C-block pinches."""
    snap = read_snapshot(env.get_ram())
    keys0 = int(snap.keys)
    room0 = snap.screen
    start_xy = [snap.link_x, snap.link_y]
    attempts = []

    def try_path(name: str, steps: list[tuple[str, int]]) -> bool:
        log = []
        for axis, tgt in steps:
            ok = walk_axis(env, assist, total, axis, tgt, max_f=450)
            snap = read_snapshot(env.get_ram())
            log.append(
                {
                    "step": f"axis:{axis}:{tgt}",
                    "ok": ok,
                    "xy": [snap.link_x, snap.link_y],
                    "tile": snap.colliding_tile,
                }
            )
        at = align_door(env, assist, total, 32, 141)
        snap = read_snapshot(env.get_ram())
        near = abs(snap.link_x - 32) <= 6 and abs(snap.link_y - 141) <= 4
        attempts.append({"name": name, "log": log, "at_mouth": at, "near": near})
        print("NAV26", name, "near", near, "mouth", at, flush=True)
        return near

    paths = (
        ("y141_west", (("y", 141), ("x", 32))),
        ("south189_west", (("y", 189), ("x", 32), ("y", 141))),
        ("north109_west", (("y", 109), ("x", 32), ("y", 141))),
        ("east208_south189_west", (("x", 208), ("y", 189), ("x", 32), ("y", 141))),
        ("east208_north109_west", (("x", 208), ("y", 109), ("x", 32), ("y", 141))),
    )
    used = None
    for name, steps in paths:
        if try_path(name, steps):
            used = name
            break
    at = align_door(env, assist, total, 32, 141, frames=32)
    push_left(env, assist, total, frames=220)
    ram = env.get_ram()
    snap = read_snapshot(ram)
    dump = dump_live(snap, ram)
    dest = f"0x{snap.screen:02x}"
    changed = snap.screen != room0
    png = RECORDINGS_DIR / "l5_26_west.png"
    obs, *_ = env.step(nes_idle_action())
    total[0] += 1
    assist.apply_env(env, frame=total[0])
    save_rgb_png(obs, png)
    body = {
        "via": "0x26 LEFT open",
        "ok": changed,
        "path": used,
        "start_xy": start_xy,
        "at_mouth": at,
        "attempts": attempts,
        "keys_in": keys0,
        "keys_out": int(snap.keys),
        "key_spent": int(snap.keys) < keys0,
        "dump": dump,
        "screenshot": str(png.resolve()),
        "status_claim": None,
        "pokes": False,
    }
    write_json_report(RECORDINGS_DIR / "l5_26_west.json", body)
    return {
        "direction": "LEFT",
        "success": changed,
        "sealed": not changed,
        "path": used,
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
        "attempts": attempts,
    }


def main() -> dict:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    commands = [
        "PYTHONPATH=nes:. uv run python nes/zelda_i/scripts/_probe_l5_26_clear.py  "
        "# Level5Cleared27 WEST 0x26, infinite-life, "
        "GenericDungeonRoomController+ROOM_66_SPEC, 28000f"
    ]
    rom26 = rom_room(ROOM_26)
    rom27 = rom_room(ROOM_27)
    print("ROM26", rom26, flush=True)
    print("ROM27", rom27, flush=True)

    env = None
    env, assist, obs = open_env()
    total = [1]
    try:
        idle(env, assist, total, 40)  # item-get freeze on Cleared27
        ram = env.get_ram()
        start_snap = read_snapshot(ram)
        start_dump = dump_live(start_snap, ram)
        print(
            "START",
            hex(start_snap.screen),
            [start_snap.link_x, start_snap.link_y],
            "keys",
            start_snap.keys,
            "tile",
            start_snap.colliding_tile,
            flush=True,
        )

        hop = None
        walked = False
        if start_snap.screen != ROOM_26:
            hop = walk_west_from_27(env, assist, total)
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

        ready = wait_play(env, assist, total, ROOM_26, max_f=360)
        ram = env.get_ram()
        arrive_snap = read_snapshot(ram)
        arrive_dump = dump_live(arrive_snap, ram)
        arrive_png = RECORDINGS_DIR / "l5_26_arrive.png"
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
            "doors",
            hex(arrive_snap.cur_opened_doors),
            "mask",
            hex(arrive_snap.open_doorway_mask),
            flush=True,
        )

        if not ready or arrive_snap.screen != ROOM_26 or arrive_snap.mode != PLAY_MODE:
            report = {
                "ok": False,
                "reason": "never_play_mode_5_in_0x26",
                "status_claim": None,
                "pokes": False,
                "commands": commands,
                "controller_reused": "GenericDungeonRoomController + ROOM_66_SPEC",
                "rom26": rom26,
                "rom27": rom27,
                "start": start_dump,
                "arrive": arrive_dump,
                "walked_west": walked,
                "hop": hop,
                "ready": ready,
                "checkpoint": None,
                "whistle_0x065C": arrive_dump.get("whistle_0x065C"),
                "screenshot": str(arrive_png.resolve()),
            }
            write_json_report(RECORDINGS_DIR / "l5_26_clear.json", report)
            return report

        bombs_in = int(arrive_snap.bombs)
        keys_in = int(arrive_snap.keys)
        spec = make_26_spec()
        fight = fight_room(env, assist, total, spec)
        if fight.get("obs") is not None:
            obs = fight["obs"]
        idle(env, assist, total, 30)
        extra = 0
        while extra < 90:
            snap = read_snapshot(env.get_ram())
            if snap.mode == PLAY_MODE and snap.screen == ROOM_26:
                break
            obs = step(env, assist, total, nes_idle_action())
            extra += 1
        idle(env, assist, total, 20)

        ram = env.get_ram()
        mid_snap = read_snapshot(ram)
        dead = (
            mid_snap.screen == ROOM_26
            and mid_snap.mode == PLAY_MODE
            and not live_gibdos(mid_snap)
        )
        key_pick = None
        if dead:
            key_pick = grab_key(env, assist, total, int(mid_snap.keys))
            idle(env, assist, total, 16)
            print("KEY_PICK", key_pick, flush=True)

        ram = env.get_ram()
        mid_snap = read_snapshot(ram)
        mid_dump = dump_live(mid_snap, ram)
        dead = (
            mid_snap.screen == ROOM_26
            and mid_snap.mode == PLAY_MODE
            and not live_gibdos(mid_snap)
        )
        bombs_out = int(mid_snap.bombs)
        keys_out = int(mid_snap.keys)
        png = RECORDINGS_DIR / "l5_26_clear.png"
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

        post_clear_doors = {
            "from_room": "0x26",
            "all_5_dead": dead,
            "doors": doors_end,
            "doorway_mask": mask_end,
            "keys": keys_out,
            "bombs": bombs_out,
            "room_item_id": mid_snap.room_item_id,
            "rom_N": rom26["N"],
            "rom_S": rom26["S"],
            "rom_E": rom26["E"],
            "rom_W": rom26["W"],
        }

        west = None
        if dead:
            west = walk_west_from_26(env, assist, total)
            print(
                "WEST",
                west.get("dest_room") or "sealed",
                "path",
                west.get("path"),
                "keys",
                west.get("keys_in"),
                "->",
                west.get("keys_out"),
                "spent",
                west.get("key_spent"),
                flush=True,
            )

        env.close()
        env = None

        saved = None
        if dead:
            path = write_state_bytes(
                state_path(GAME_DIR, GAME, "Level5Cleared26"),
                state_bytes,
            )
            write_state_provenance(
                path,
                source_state_path=(
                    GAME_DIR / "custom_integrations" / GAME / f"{STATE}.state"
                ),
                request={
                    "segment": "Level5Cleared26",
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
                    "room": ROOM_26,
                    "live_gibdos": 0,
                    "doors_after": int(mid_snap.cur_opened_doors) & 0x0F,
                    "doorway_mask": int(mid_snap.open_doorway_mask) & 0x0F,
                    "bombs": bombs_out,
                    "keys": keys_out,
                    "key_grabbed": bool(key_pick and key_pick.get("grabbed")),
                },
                natural_entry=False,
            )
            saved = "Level5Cleared26"

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
                "path": west.get("path"),
            }

        whistle = (mid_dump.get("inventory") or {}).get("whistle_0x065C", 0)
        fight_out = {k: v for k, v in fight.items() if k != "obs"}
        report = {
            "ok": dead,
            "status_claim": None,
            "from_state": STATE,
            "pokes": False,
            "commands": commands,
            "controller_reused": "GenericDungeonRoomController",
            "spec_reused": ROOM_66_SPEC.spec_id,
            "spec_id": spec.spec_id,
            "rom26": rom26,
            "rom27": rom27,
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
            "dest_west": dest,
            "west": {k: v for k, v in (west or {}).items() if k != "dump"} if west else None,
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
        write_json_report(RECORDINGS_DIR / "l5_26_clear.json", report)
        return report
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    r = main()
    print("CMD", r["commands"])
    print("CONTROLLER", r.get("controller_reused"), "SPEC", r.get("spec_reused"))
    print("ROM26", r.get("rom26"))
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
    print("DEST_WEST", r.get("dest_west"))
    print("CKPT", r.get("checkpoint"), r.get("checkpoint_reason"))
    print("WHISTLE", r.get("whistle_0x065C"))
    print("FRAMES", r.get("frames_total"))
    print("status_claim", None)
