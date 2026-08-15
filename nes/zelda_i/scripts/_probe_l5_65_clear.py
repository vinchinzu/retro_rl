"""Clear L5 0x65 5x Gibdo from Level5Cleared55, dump doors, walk new exits.

Start: Level5Cleared55. Walk DOWN into 0x65. Wait for play mode 5.
Reuse GenericDungeonRoomController + ROOM_66_SPEC combat (no new fighter).
Prior 3/5 timeout at 14000f — budget 28000f (Gibdo HP=112).
No pokes, no candle, no bomb walls, no east67, not Clean STATUS.
Level5Cleared65 only if all 5 Gibdos dead.
Bombs start at 4: do not enter Dodongo 0x31 (need 6).
Key door: dump FIRST then spend a key.
"""
from __future__ import annotations

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
)
from zelda_i.dungeon_ids import object_name
from zelda_i.dungeon_lab import _drive_exit
from zelda_i.dungeon_ops import exit_door, idle
from zelda_i.dungeon_trace import compact_snapshot, write_state_provenance
from zelda_i.level5_dungeon import GIBDO_OBJECT_TYPE, ROOM_66_SPEC
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_SELECTED_ITEM,
    ADDR_WHISTLE,
    PLAY_MODE,
    ZeldaSnapshot,
    read_snapshot,
    read_u8,
)

STATE = "Level5Cleared55"
ROOM_55 = 0x55
ROOM_65 = 0x65
DODONGO_TYPES = (0x31, 0x32)
# Prior attempt 3/5 @ 14000f. 5x HP=112 needs more time than 3x on 0x66.
MAX_FIGHT_FRAMES = 28000

EXIT_ROUTES: dict[str, DoorRoute] = {
    "UP": DoorRoute("UP", ((120, 141), (120, 93))),
    "DOWN": DoorRoute("DOWN", ((120, 141), (120, 205))),
    "LEFT": DoorRoute("LEFT", ((120, 141), (32, 141))),
    "RIGHT": DoorRoute("RIGHT", ((120, 141), (208, 141))),
}


def make_65_spec() -> DungeonRoomSpec:
    """ROOM_66_SPEC combat / liveness, retargeted to 0x65 5x Gibdo."""
    return replace(
        ROOM_66_SPEC,
        spec_id="level5_room65_gibdos_reuse66",
        source_room=ROOM_55,
        room_id=ROOM_65,
        entry=DoorRoute("DOWN", ((120, 93),)),
        expected_enemy_count=5,
        required_open_doors=0,
        exit_routes=(
            DoorRoute("UP", ((120, 93),)),
            DoorRoute("DOWN", ((120, 205),)),
            DoorRoute("LEFT", ((32, 141),)),
            DoorRoute("RIGHT", ((208, 141),)),
        ),
        max_frames=MAX_FIGHT_FRAMES,
    )


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


def is_dodongo(dump_or_objs) -> bool:
    objs = dump_or_objs
    if isinstance(dump_or_objs, dict) and "objects" in dump_or_objs:
        objs = dump_or_objs.get("objects")
    if isinstance(objs, dict):
        return any(k.lower() in ("0x31", "0x32") for k in objs)
    for obj in objs or []:
        tid = obj.get("type_id") if isinstance(obj, dict) else None
        if tid in DODONGO_TYPES and (obj.get("hp") or 0) > 0:
            return True
    return False


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


def walk_down_65(env, assist, total) -> dict:
    """Align x=120 then hold DOWN from 0x55 into 0x65."""
    snap0 = read_snapshot(env.get_ram())
    before = (snap0.screen, snap0.mode, snap0.link_x, snap0.link_y)
    hop = exit_door(env, assist, total, "DOWN")
    idle(env, assist, total, 16)
    snap = read_snapshot(env.get_ram())
    changed = snap.screen == ROOM_65
    return {
        "before": before,
        "after": (snap.screen, snap.mode, snap.link_x, snap.link_y),
        "changed_room": changed,
        "result": "room_change" if changed else hop.get("result", "blocked"),
        "hop": {
            "changed_room": hop.get("changed_room"),
            "result": hop.get("result"),
        },
    }


def wait_play(env, assist, total, room: int, *, max_f: int = 360) -> bool:
    """Idle until play mode 5 in *room* (not settle mode 4)."""
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
                    {
                        "f": ctl.frames,
                        "n": len(live),
                        "hps": [o.hp for o in live],
                    }
                )
            elif len(live) != last_n:
                last_n = len(live)
                progress.append(
                    {
                        "f": ctl.frames,
                        "n": len(live),
                        "hps": [o.hp for o in live],
                    }
                )
                print(
                    f"KILL n={len(live)} f={ctl.frames} "
                    f"hps={[o.hp for o in live]}",
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


def probe_all_exits(state_data: bytes, room_id: int, keys0: int) -> list[dict]:
    results = []
    for direction in ("UP", "DOWN", "LEFT", "RIGHT"):
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
        png = RECORDINGS_DIR / f"l5_{snap.screen:02x}_from65.png"
        obs, *_ = env.step(nes_idle_action())
        total[0] += 1
        assist.apply_env(env, frame=total[0])
        save_rgb_png(obs, png)
        keys1 = int(snap.keys)
        write_json_report(
            RECORDINGS_DIR / f"l5_{snap.screen:02x}_from65.json",
            {
                "via": f"0x65 {direction}",
                "ok": hop.get("changed_room"),
                "key_spent": keys1 < keys0,
                "keys_in": keys0,
                "keys_out": keys1,
                "dump": dump,
                "screenshot": str(png.resolve()),
                "status_claim": None,
                "pokes": False,
            },
        )
        return {
            "ok": hop.get("changed_room"),
            "via": f"0x65 {direction}",
            "room": f"0x{snap.screen:02x}",
            "mode": snap.mode,
            "xy": [snap.link_x, snap.link_y],
            "keys_in": keys0,
            "keys_out": keys1,
            "key_spent": keys1 < keys0,
            "bombs": snap.bombs,
            "doors": dump.get("doors"),
            "doorway_mask": dump.get("doorway_mask"),
            "objects": dump.get("objects"),
            "room_item_id": snap.room_item_id,
            "dodongo": is_dodongo(dump),
            "screenshot": str(png.resolve()),
            "dump_path": str((RECORDINGS_DIR / f"l5_{snap.screen:02x}_from65.json").resolve()),
        }
    finally:
        if env is not None:
            env.close()


def main() -> dict:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    commands = [
        "PYTHONPATH=nes:. uv run python nes/zelda_i/scripts/_probe_l5_65_clear.py  # Level5Cleared55, infinite-life, GenericDungeonRoomController+ROOM_66_SPEC, 28000f"
    ]
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
        if start_snap.screen != ROOM_65:
            hop = walk_down_65(env, assist, total)
            walked = bool(hop.get("changed_room"))
            print("DOWN_HOP", hop, flush=True)

        ready = wait_play(env, assist, total, ROOM_65, max_f=360)
        ram = env.get_ram()
        arrive_snap = read_snapshot(ram)
        arrive_dump = dump_live(arrive_snap, ram)
        arrive_png = RECORDINGS_DIR / "l5_65_arrive.png"
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
            "doors",
            hex(arrive_snap.cur_opened_doors),
            "keys",
            arrive_snap.keys,
            "bombs",
            arrive_snap.bombs,
            flush=True,
        )

        if not ready or arrive_snap.screen != ROOM_65 or arrive_snap.mode != PLAY_MODE:
            report = {
                "ok": False,
                "reason": "never_play_mode_5_in_0x65",
                "status_claim": None,
                "pokes": False,
                "commands": commands,
                "controller_reused": "GenericDungeonRoomController + ROOM_66_SPEC",
                "start": start_dump,
                "arrive": arrive_dump,
                "walked_down": walked,
                "hop": hop,
                "ready": ready,
                "checkpoint": None,
                "whistle_0x065C": arrive_dump.get("inventory", {}).get("whistle_0x065C"),
                "screenshot": str(arrive_png.resolve()),
            }
            write_json_report(RECORDINGS_DIR / "l5_65_clear.json", report)
            return report

        bombs_in = int(arrive_snap.bombs)
        keys_in = int(arrive_snap.keys)
        spec = make_65_spec()
        fight = fight_room(env, assist, total, spec)
        if fight.get("obs") is not None:
            obs = fight["obs"]
        idle(env, assist, total, 30)
        extra = 0
        while extra < 90:
            snap = read_snapshot(env.get_ram())
            if snap.mode == PLAY_MODE and snap.screen == ROOM_65:
                break
            obs = step(env, assist, total, nes_idle_action())
            extra += 1
        idle(env, assist, total, 20)

        ram = env.get_ram()
        mid_snap = read_snapshot(ram)
        mid_dump = dump_live(mid_snap, ram)
        dead = (
            mid_snap.screen == ROOM_65
            and mid_snap.mode == PLAY_MODE
            and not live_gibdos(mid_snap)
        )
        bombs_out = int(mid_snap.bombs)
        keys_out = int(mid_snap.keys)
        png = RECORDINGS_DIR / "l5_65_clear.png"
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
            "end_hps",
            fight.get("end_hps"),
            "doors",
            doors_end,
            "mask",
            mask_end,
            "keys",
            keys_out,
            "bombs",
            bombs_out,
            flush=True,
        )
        env.close()
        env = None

        # Dump doors FIRST (before walking / spending a key).
        post_clear_doors = {
            "from_room": "0x65",
            "all_5_dead": dead,
            "doors": doors_end,
            "doorway_mask": mask_end,
            "keys": keys_out,
            "bombs": bombs_out,
            "room_item_id": mid_snap.room_item_id,
            "dumped_before_key_spend": True,
        }

        probes = []
        if dead:
            probes = probe_all_exits(state_bytes, ROOM_65, keys_out)
        print(
            "PROBES",
            [(p.get("direction"), p.get("dest_room") or "sealed") for p in probes],
            flush=True,
        )

        saved = None
        if dead:
            path = write_state_bytes(
                state_path(GAME_DIR, GAME, "Level5Cleared65"),
                state_bytes,
            )
            write_state_provenance(
                path,
                source_state_path=(
                    GAME_DIR / "custom_integrations" / GAME / f"{STATE}.state"
                ),
                request={
                    "segment": "Level5Cleared65",
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
                    "room": ROOM_65,
                    "live_gibdos": 0,
                    "doors_after": int(mid_snap.cur_opened_doors) & 0x0F,
                    "doorway_mask": int(mid_snap.open_doorway_mask) & 0x0F,
                    "bombs": bombs_out,
                    "keys": keys_out,
                },
                natural_entry=False,
            )
            saved = "Level5Cleared65"

        next_rooms = []
        key_door = None
        if dead:
            for p in probes:
                if not p.get("success"):
                    continue
                dest_id = p.get("dest_room_id")
                dest = p.get("dest_room")
                direction = p["direction"]
                if dest_id in (None, ROOM_55) or (dest or "").lower() == "0x55":
                    p["skipped"] = "back_to_0x55"
                    continue
                if is_dodongo(p.get("objects")) and bombs_out < 6:
                    p["skipped"] = f"dodongo_bombs={bombs_out}<6"
                    continue
                # Key door: dump already done; now spend a key if needed.
                entered = walk_and_dump(state_bytes, direction, dest_id, total)
                next_rooms.append(entered)
                if entered.get("key_spent"):
                    key_door = {
                        "direction": direction,
                        "dest": entered.get("room"),
                        "keys_in": entered.get("keys_in"),
                        "keys_out": entered.get("keys_out"),
                        "dumped_first": True,
                    }
                print("ENTERED", entered.get("via"), entered.get("room"), flush=True)

        exits_report = {
            "from_room": "0x65",
            "all_5_dead": dead,
            "doors": doors_end,
            "doorway_mask": mask_end,
            "dumped_before_key_spend": True,
            "probes": probes,
            "next_rooms": next_rooms,
            "key_door": key_door,
            "status_claim": None,
            "pokes": False,
        }
        write_json_report(RECORDINGS_DIR / "l5_65_exits.json", exits_report)

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
            "walked_down": walked,
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
                "gibdos": len(live_gibdos(arrive_snap)),
            },
            "clear": {
                **fight_out,
                "bombs_in": bombs_in,
                "bombs_out": bombs_out,
                "keys_in": keys_in,
                "keys_out": keys_out,
                "dead": dead,
            },
            "post_clear_doors": post_clear_doors,
            "doors_end": doors_end,
            "doorway_mask_end": mask_end,
            "exits": probes,
            "next_rooms": next_rooms,
            "key_door": key_door,
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
        write_json_report(RECORDINGS_DIR / "l5_65_clear.json", report)
        return report
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    r = main()
    print("CMD", r["commands"])
    print("CONTROLLER", r.get("controller_reused"), "SPEC", r.get("spec_reused"))
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
    )
    print("ARRIVE_OBJECTS", a.get("objects"))
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
        "end_hps",
        c.get("end_hps"),
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
    print("CLEAR_CTRL", c.get("controller"))
    print("POST_CLEAR_DOORS", r.get("post_clear_doors"))
    print("EXITS")
    for p in r.get("exits") or []:
        print(
            " ",
            p.get("direction"),
            "dest" if p.get("success") else "sealed",
            p.get("dest_room"),
            p.get("objects"),
            p.get("skipped"),
        )
    print("NEXT_ROOMS", r.get("next_rooms"))
    print("KEY_DOOR", r.get("key_door"))
    print("CKPT", r.get("checkpoint"), r.get("checkpoint_reason"))
    print("WHISTLE", r.get("whistle_0x065C"))
    print("FRAMES", r.get("frames_total"))
    print("status_claim", None)
