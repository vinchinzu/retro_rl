"""Re-bomb 0x65 west, settle on WEST mouth, probe center-stairs from the west. No pokes."""
from retro_harness.env import make_env, reset_obs, state_path, write_state_bytes
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ops import goto, idle, push_dir
from zelda_i.dungeon_trace import write_state_provenance
from zelda_i.level9_stairs import on_stair_tile, stair_transition_modes
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_SELECTED_ITEM, ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8
from zelda_i.scripts._probe_l5_whistle_path import (
    ROOM_64,
    ROOM_65,
    dump_live,
    rom_room,
    select_bombs_menu,
    shot,
    step,
    walk_axis,
    wait_play,
    write_dump,
)

STATE = "Level5Cleared65"


def open_env(state=STATE):
    env = make_env(GAME, state, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True)
    obs, _ = reset_obs(env)
    obs, *_ = env.step(nes_idle_action())
    assist.apply_env(env, frame=0)
    return env, assist, obs


def rec(snap, **extra):
    d = {
        "xy": [snap.link_x, snap.link_y],
        "tile": int(snap.colliding_tile),
        "stair": bool(on_stair_tile(snap)),
        "mode": snap.mode,
        "room": f"0x{snap.screen:02x}",
    }
    d.update(extra)
    return d


def main():
    configure_headless()
    env, assist, obs = open_env()
    total = [1]
    try:
        idle(env, assist, total, 16)
        start = dump_live(read_snapshot(env.get_ram()), env.get_ram())
        print("START", start.get("room_hex"), "bombs", start.get("bombs"), "xy", [start.get("x"), start.get("y")], flush=True)
        walk_axis(env, assist, total, "y", 141, max_f=400)
        walk_axis(env, assist, total, "x", 40, max_f=400)
        for _ in range(8):
            step(env, assist, total, nes_action("LEFT"))
        menu = select_bombs_menu(env, assist, total)
        bombs0 = int(read_snapshot(env.get_ram()).bombs)
        room0 = int(read_snapshot(env.get_ram()).screen)
        step(env, assist, total, nes_action("LEFT", "B"))
        for _ in range(20):
            step(env, assist, total, nes_action("RIGHT"))
        idle(env, assist, total, 100)
        for _ in range(200):
            snap = read_snapshot(env.get_ram())
            if snap.screen != room0 and snap.mode in (PLAY_MODE, 6, 7, 4):
                break
            step(env, assist, total, nes_action("LEFT"))
        idle(env, assist, total, 12)
        wait_play(env, assist, total, ROOM_64, max_f=280)
        idle(env, assist, total, 24)
        snap = read_snapshot(env.get_ram())
        arrive = dump_live(snap, env.get_ram())
        png = shot(env, assist, total, "l5_64_west_settle")
        print("SETTLE", arrive.get("room_hex"), "mode", snap.mode, "xy", [snap.link_x, snap.link_y], "tile", snap.colliding_tile, "bombs", snap.bombs, flush=True)
        if snap.screen != ROOM_64 or snap.mode != PLAY_MODE:
            write_dump("l5_64_from_west", {"ok": False, "reason": "did_not_settle_0x64", "arrive": arrive, "menu": menu})
            return
        # Save honest west-mouth checkpoint.
        path = write_state_bytes(state_path(GAME_DIR, GAME, "Level5Entered64"), env.em.get_state())
        write_state_provenance(
            path,
            source_state_path=GAME_DIR / "custom_integrations" / GAME / f"{STATE}.state",
            request={
                "segment": "Level5Entered64",
                "predecessor_entry": True,
                "start_state": STATE,
                "via": "0x65 WEST bomb settle west mouth",
                "key_poke": False,
                "door_poke": False,
                "bomb_count_poke": False,
                "selected_item_poke": False,
            },
            selected_trial={
                "success": True,
                "room": 0x64,
                "xy": [snap.link_x, snap.link_y],
                "bombs": int(snap.bombs),
                "keys": int(snap.keys),
                "whistle_0x065C": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
            },
            natural_entry=False,
        )
        # Probe west-side channels toward center stairs.
        probes = []
        paths = [
            ("w141_x120", (("y", 141), ("x", 80), ("x", 120))),
            ("w109_x120", (("y", 109), ("x", 120), ("y", 141))),
            ("w117_x120", (("y", 117), ("x", 120), ("y", 141))),
            ("w125_x120", (("y", 125), ("x", 120), ("y", 141))),
            ("w157_x120", (("y", 157), ("x", 120), ("y", 141))),
            ("w173_x120", (("y", 173), ("x", 120), ("y", 141))),
            ("w109_x96", (("y", 109), ("x", 96), ("y", 141))),
            ("w109_x112", (("y", 109), ("x", 112), ("y", 141))),
            ("w109_x128", (("y", 109), ("x", 128), ("y", 141))),
            ("w141_x96_up", (("y", 141), ("x", 96), ("y", 125))),
        ]
        # Do probes in THIS env sequentially? Darknuts will move. Better one env, try paths in order until stairs.
        # First record current, then try each path; if we leave 0x64 to cellar, stop.
        for name, steps in paths:
            if read_snapshot(env.get_ram()).screen != ROOM_64:
                break
            log = []
            for axis, tgt in steps:
                ok = walk_axis(env, assist, total, axis, tgt, max_f=350)
                snap = read_snapshot(env.get_ram())
                log.append(rec(snap, axis=axis, tgt=tgt, ok=ok))
                if stair_transition_modes(snap.mode) or snap.screen != ROOM_64:
                    break
            for direction in ("UP", "DOWN"):
                for _ in range(40):
                    snap = read_snapshot(env.get_ram())
                    if stair_transition_modes(snap.mode) or snap.screen != ROOM_64:
                        break
                    if snap.link_x > 170:
                        step(env, assist, total, nes_action("LEFT"))
                        continue
                    step(env, assist, total, nes_action(direction))
                snap = read_snapshot(env.get_ram())
                log.append(rec(snap, nudge=direction))
                if stair_transition_modes(snap.mode) or snap.screen != ROOM_64:
                    break
            snap = read_snapshot(env.get_ram())
            item = {
                "name": name,
                "log": log,
                "end": rec(snap),
                "took": stair_transition_modes(snap.mode) or (snap.screen not in (ROOM_64, ROOM_65)),
            }
            probes.append(item)
            print(name, "took", item["took"], "end", item["end"], flush=True)
            if item["took"]:
                break
        snap = read_snapshot(env.get_ram())
        final = dump_live(snap, env.get_ram())
        png2 = shot(env, assist, total, "l5_64_from_west_end")
        write_dump(
            "l5_64_from_west",
            {
                "pokes": False,
                "status_claim": None,
                "start": start,
                "menu": menu,
                "bombs_in": bombs0,
                "arrive": arrive,
                "arrive_png": png,
                "probes": probes,
                "final": final,
                "screenshot": png2,
                "whistle_0x065C": final.get("whistle_0x065C"),
                "rom64": rom_room(0x64),
            },
        )
        print("FINAL", final.get("room_hex"), "mode", snap.mode, "xy", [snap.link_x, snap.link_y], flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
