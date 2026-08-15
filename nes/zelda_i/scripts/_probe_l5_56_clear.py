"""Clear L5 0x56 3x type 0x31 from Level5North56 via L2 fight_dodongo eat-cycle.

Reuses zelda_i.level2_boss_combat.fight_dodongo (mouth_target + B-in-mouth).
No bomb-count / key / door / selected-item pokes. No wall bombs. No Whistle.
No candle. No east67. Not Clean STATUS. Keeps Level5Entered46 / Level5Cleared57.
Writes Level5Cleared56 only if all 3 are dead in play mode AND door bits change.
"""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs, save_state
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.door_graph.core import DoorDir, dirs_from_mask
from zelda_i.dungeon_ops import exit_door
from zelda_i.dungeon_trace import compact_snapshot, write_state_provenance
from zelda_i.level2_boss_combat import fight_dodongo
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_SELECTED_ITEM,
    ADDR_WHISTLE,
    PLAY_MODE,
    ZeldaSnapshot,
    read_snapshot,
    read_u8,
)

STATE = "Level5North56"
ROOM_56 = 0x56
TYPE_31 = 0x31
NEED_BOMBS = 6
REUSED = "zelda_i.level2_boss_combat.fight_dodongo"


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
    compact["whistle_0x065C"] = int(read_u8(ram, ADDR_WHISTLE))
    compact["objects"] = [
        {
            "slot": obj.slot,
            "type_id": obj.type_id,
            "type_hex": f"0x{obj.type_id:02x}",
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


def live_31(snap: ZeldaSnapshot) -> list:
    return [
        obj
        for obj in snap.objects
        if 1 <= obj.slot <= 12 and obj.type_id == TYPE_31 and obj.hp > 0
    ]


def play_dead(snap: ZeldaSnapshot) -> bool:
    return (
        snap.mode == PLAY_MODE
        and snap.screen == ROOM_56
        and snap.level == 5
        and not live_31(snap)
    )


def open_env():
    env = make_env(GAME, STATE, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True)
    obs, _ = reset_obs(env)
    obs, *_ = env.step(nes_idle_action())
    assist.apply_env(env, frame=0)
    return env, assist, obs


def idle(env, assist, total, frames: int = 40):
    obs = None
    for _ in range(frames):
        obs, *_ = env.step(nes_idle_action())
        total[0] += 1
        assist.apply_env(env, frame=total[0])
    return obs


def main() -> dict:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    commands = [
        (
            "uv run python nes/zelda_i/scripts/_probe_l5_56_clear.py "
            f"--from-state {STATE} --infinite-life"
        )
    ]
    env, assist, obs = open_env()
    total = [1]
    try:
        obs = idle(env, assist, total, 24)
        ram = env.get_ram()
        start_snap = read_snapshot(ram)
        start_dump = dump_live(start_snap, ram)
        start_doors = int(start_snap.cur_opened_doors) & 0x0F
        start_mask = int(start_snap.open_doorway_mask) & 0x0F
        bombs_in = int(start_snap.bombs)
        start_live = len(live_31(start_snap))

        if bombs_in < NEED_BOMBS:
            env.close()
            env, assist, obs = open_env()
            total = [1]
            obs = idle(env, assist, total, 24)
            ram = env.get_ram()
            start_snap = read_snapshot(ram)
            start_dump = dump_live(start_snap, ram)
            start_doors = int(start_snap.cur_opened_doors) & 0x0F
            start_mask = int(start_snap.open_doorway_mask) & 0x0F
            bombs_in = int(start_snap.bombs)
            start_live = len(live_31(start_snap))

        if bombs_in < NEED_BOMBS:
            png = RECORDINGS_DIR / "l5_56_clear.png"
            save_rgb_png(obs, png)
            report = {
                "ok": False,
                "status_claim": None,
                "from_state": STATE,
                "pokes": False,
                "reused": REUSED,
                "commands": commands,
                "bombs_in": bombs_in,
                "bombs_out": bombs_in,
                "kills": 0,
                "reason": (
                    f"bombs={bombs_in} < {NEED_BOMBS} after reload {STATE}; "
                    "no OW bomb-bag helper; did not enter combat short"
                ),
                "checkpoint": None,
                "checkpoint_reason": "not entered: short bombs",
                "whistle_0x065C": start_dump.get("whistle_0x065C"),
                "screenshot": str(png.resolve()),
                "dump": start_dump,
            }
            write_json_report(RECORDINGS_DIR / "l5_56_clear.json", report)
            return report

        fight = fight_dodongo(
            env,
            assist,
            dodongo_type=TYPE_31,
            poke=False,
            check_tf=False,
            select_bomb=False,
            approach_mouth=True,
            clamp_x=(40, 208),
            clamp_y=(101, 189),
            mouth_tol=5,
            mouth_offset=18,
            strict_mouth=True,
            stable_face_frames=8,
        )
        total[0] += int(fight.get("frames") or 0)
        obs = idle(env, assist, total, 40)
        extra = 0
        while extra < 90:
            snap = read_snapshot(env.get_ram())
            if snap.mode == PLAY_MODE:
                break
            obs, *_ = env.step(nes_idle_action())
            total[0] += 1
            assist.apply_env(env, frame=total[0])
            extra += 1
        obs = idle(env, assist, total, 25)

        ram = env.get_ram()
        end_snap = read_snapshot(ram)
        end_dump = dump_live(end_snap, ram)
        end_doors = int(end_snap.cur_opened_doors) & 0x0F
        end_mask = int(end_snap.open_doorway_mask) & 0x0F
        bombs_out = int(end_snap.bombs)
        dead = play_dead(end_snap)
        end_live = len(live_31(end_snap))
        kills = max(0, start_live - end_live)
        doors_changed = end_doors != start_doors
        png = RECORDINGS_DIR / "l5_56_clear.png"
        save_rgb_png(obs, png)

        saved = None
        if dead and doors_changed:
            path = save_state(env, GAME_DIR, GAME, "Level5Cleared56")
            write_state_provenance(
                path,
                source_state_path=(
                    GAME_DIR / "custom_integrations" / GAME / f"{STATE}.state"
                ),
                request={
                    "segment": "Level5Cleared56",
                    "predecessor_entry": True,
                    "start_state": STATE,
                    "key_poke": False,
                    "door_poke": False,
                    "bomb_count_poke": False,
                    "selected_item_poke": False,
                    "reused": REUSED,
                },
                selected_trial={
                    "success": True,
                    "frames": total[0],
                    "room": ROOM_56,
                    "live_31": 0,
                    "kills": kills,
                    "doors_before": start_doors,
                    "doors_after": end_doors,
                    "bombs_in": bombs_in,
                    "bombs_out": bombs_out,
                    "bombs_used_est": fight.get("bombs_used_est"),
                    "hits_est": fight.get("hits_est"),
                },
                natural_entry=False,
            )
            saved = "Level5Cleared56"

        west = None
        west_opens = bool(end_doors & DoorDir.LEFT) or bool(end_mask & DoorDir.LEFT)
        if dead and west_opens:
            # y=141 channel then hold LEFT. Mid-room / y=133 sticks at the jamb.
            for _ in range(800):
                snap = read_snapshot(env.get_ram())
                if snap.mode == PLAY_MODE and snap.screen != ROOM_56:
                    break
                if snap.mode != PLAY_MODE:
                    obs, *_ = env.step(nes_action("LEFT"))
                    total[0] += 1
                    assist.apply_env(env, frame=total[0])
                    continue
                if abs(snap.link_y - 141) > 2 and snap.link_x > 40:
                    btn = "DOWN" if snap.link_y < 141 else "UP"
                    obs, *_ = env.step(nes_action(btn))
                else:
                    obs, *_ = env.step(nes_action("LEFT"))
                total[0] += 1
                assist.apply_env(env, frame=total[0])
            for _ in range(80):
                snap = read_snapshot(env.get_ram())
                if snap.mode == PLAY_MODE and snap.screen != ROOM_56:
                    break
                obs, *_ = env.step(nes_action("LEFT"))
                total[0] += 1
                assist.apply_env(env, frame=total[0])
            obs = idle(env, assist, total, 50)
            ram = env.get_ram()
            dest = read_snapshot(ram)
            dest_dump = dump_live(dest, ram)
            west_png = RECORDINGS_DIR / "l5_56_west.png"
            save_rgb_png(obs, west_png)
            changed = dest.screen != ROOM_56
            west = {
                "ok": changed,
                "result": "room_change" if changed else "blocked",
                "dest_room": dest_dump.get("room_hex"),
                "dump": dest_dump,
                "screenshot": str(west_png.resolve()),
            }
            write_json_report(RECORDINGS_DIR / "l5_56_west.json", west)

        report = {
            "ok": dead and doors_changed,
            "status_claim": None,
            "from_state": STATE,
            "pokes": False,
            "reused": REUSED,
            "reused_file": "nes/zelda_i/level2_boss_combat.py",
            "reused_function": "fight_dodongo",
            "commands": commands,
            "bombs_in": bombs_in,
            "bombs_out": bombs_out,
            "kills": kills,
            "start": {
                "xy": [start_dump.get("x"), start_dump.get("y")],
                "keys": start_dump.get("keys"),
                "bombs": bombs_in,
                "selected": start_dump.get("inventory", {}).get("selected_item_0x0656"),
                "objects": start_dump.get("objects"),
                "doors": start_dump.get("doors"),
            },
            "fight": {k: v for k, v in fight.items() if k != "log"},
            "fight_log": fight.get("log"),
            "died": dead,
            "live_31_end": end_live,
            "end_mode": end_snap.mode,
            "end_mode_name": end_dump.get("mode_name"),
            "doors_start": decode_doors(start_doors),
            "doors_end": decode_doors(end_doors),
            "doorway_mask_start": decode_doors(start_mask),
            "doorway_mask_end": decode_doors(end_mask),
            "doors_changed": doors_changed,
            "west_opens": west_opens,
            "west": None if west is None else {k: v for k, v in west.items() if k != "dump"},
            "west_dump": None if west is None else west.get("dump"),
            "next_room": (
                west["dest_room"]
                if west and west.get("ok")
                else "west still sealed"
            ),
            "checkpoint": saved,
            "checkpoint_reason": (
                "3 dead in play mode and door bits changed"
                if saved
                else (
                    "not saved: "
                    + ("enemies still alive" if not dead else "doors unchanged")
                )
            ),
            "whistle_0x065C": end_dump.get("whistle_0x065C"),
            "frames_total": total[0],
            "screenshot": str(png.resolve()),
            "dump": end_dump,
        }
        write_json_report(RECORDINGS_DIR / "l5_56_clear.json", report)
        return report
    finally:
        env.close()


if __name__ == "__main__":
    r = main()
    print("CMD", r["commands"])
    print("REUSED", r.get("reused"))
    print("BOMBS", r.get("bombs_in"), "->", r.get("bombs_out"))
    print("KILLS", r.get("kills"))
    print("FIGHT", {k: r.get("fight", {}).get(k) for k in (
        "success", "frames", "bombs_used_est", "hits_est", "dodongo_type", "poke"
    )})
    print("FIGHT_LOG", r.get("fight_log"))
    print("DIED", r.get("died"), "live", r.get("live_31_end"), "mode", r.get("end_mode"))
    print("DOORS_START", r.get("doors_start"))
    print("DOORS_END", r.get("doors_end"))
    print("DOORWAY_END", r.get("doorway_mask_end"))
    print("WEST_OPENS", r.get("west_opens"), "NEXT", r.get("next_room"))
    print("WEST", r.get("west"))
    if r.get("west_dump"):
        wd = r["west_dump"]
        print("WEST_ROOM", wd.get("room_hex"), "xy", wd.get("x"), wd.get("y"))
        print("WEST_DOORS", wd.get("doors"), wd.get("doorway_mask"))
        print("WEST_OBJECTS", wd.get("objects"))
    print("CKPT", r.get("checkpoint"), r.get("checkpoint_reason"))
    print("WHISTLE", r.get("whistle_0x065C"))
    print("FRAMES", r.get("frames_total"))
    print("status_claim", None)
