"""0x64: fight 5x 0x0C, keep pushing 0x68 right, take center stairs."""
from retro_harness.env import make_env, reset_obs, state_path, write_state_bytes
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ops import idle, push_dir
from zelda_i.dungeon_trace import write_state_provenance
from zelda_i.level9_stairs import on_stair_tile, stair_transition_modes
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8
from zelda_i.scripts._probe_l5_whistle_path import (
    ROOM_64,
    ROOM_65,
    dump_and_save_room,
    dump_live,
    fight_darknuts,
    live_darknuts,
    rom_room,
    shot,
    step,
    walk_axis,
    wait_play,
    write_dump,
)

STATE = "Level5Entered64"


def open_env():
    env = make_env(GAME, STATE, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True)
    obs, _ = reset_obs(env)
    obs, *_ = env.step(nes_idle_action())
    assist.apply_env(env, frame=0)
    return env, assist, obs


def block_xy(snap):
    for o in snap.objects:
        if 1 <= o.slot <= 12 and o.type_id == 0x68:
            return (o.x, o.y)
    return None


def main():
    configure_headless()
    env, assist, obs = open_env()
    total = [1]
    try:
        idle(env, assist, total, 16)
        start = dump_live(read_snapshot(env.get_ram()), env.get_ram())
        n = len(live_darknuts(read_snapshot(env.get_ram())))
        print("START64", [start.get("x"), start.get("y")], "dn", n, flush=True)
        fight = fight_darknuts(env, assist, total, ROOM_64, expected=max(5, n), source=ROOM_65)
        idle(env, assist, total, 20)
        print("FIGHT64", fight.get("ok"), fight.get("end_n"), "f", fight.get("frames"), flush=True)
        dump_and_save_room(env, assist, total, "l5_64_cleared", "Level5Cleared64", STATE, "0x64 5/5 blue darknuts")
        # North wall to west of block, keep pushing RIGHT.
        for axis, tgt in (("y", 93), ("x", 80), ("y", 144)):
            walk_axis(env, assist, total, axis, tgt, max_f=400)
        pushes = []
        last = block_xy(read_snapshot(env.get_ram()))
        for i in range(6):
            walk_axis(env, assist, total, "y", 144, max_f=200)
            # stand just left of current block
            bx = last[0] if last else 96
            walk_axis(env, assist, total, "x", max(32, bx - 16), max_f=250)
            push_dir(env, assist, total, "RIGHT", frames=120)
            idle(env, assist, total, 8)
            snap = read_snapshot(env.get_ram())
            now = block_xy(snap)
            rec = {
                "i": i,
                "xy": [snap.link_x, snap.link_y],
                "tile": int(snap.colliding_tile),
                "stair": bool(on_stair_tile(snap)),
                "mode": snap.mode,
                "room": f"0x{snap.screen:02x}",
                "block": list(now) if now else None,
            }
            pushes.append(rec)
            print("PUSH_R", rec, flush=True)
            if stair_transition_modes(snap.mode) or (snap.screen != ROOM_64 and snap.mode == PLAY_MODE):
                break
            if now == last:
                break
            last = now
        # Walk onto stairs.
        for axis, tgt in (("y", 141), ("x", 120), ("y", 125), ("x", 120), ("y", 157), ("x", 128), ("x", 112)):
            if read_snapshot(env.get_ram()).screen != ROOM_64:
                break
            walk_axis(env, assist, total, axis, tgt, max_f=300)
            snap = read_snapshot(env.get_ram())
            print("WALK", axis, tgt, [snap.link_x, snap.link_y], "tile", snap.colliding_tile, "stair", on_stair_tile(snap), flush=True)
            if stair_transition_modes(snap.mode) or on_stair_tile(snap):
                break
        for direction in ("UP", "DOWN", "RIGHT", "LEFT"):
            for _ in range(50):
                snap = read_snapshot(env.get_ram())
                if stair_transition_modes(snap.mode) or (snap.screen != ROOM_64 and snap.mode == PLAY_MODE):
                    break
                if snap.link_x > 180:
                    step(env, assist, total, nes_action("LEFT"))
                    continue
                step(env, assist, total, nes_action(direction))
            if stair_transition_modes(read_snapshot(env.get_ram()).mode) or read_snapshot(env.get_ram()).screen != ROOM_64:
                break
        wait_play(env, assist, total, max_f=240)
        idle(env, assist, total, 16)
        snap = read_snapshot(env.get_ram())
        dump = dump_live(snap, env.get_ram())
        png = shot(env, assist, total, "l5_64_stairs")
        ok = stair_transition_modes(snap.mode) or (snap.screen not in (0x64, 0x65))
        write_dump(
            "l5_64_clear_stairs",
            {
                "pokes": False,
                "status_claim": None,
                "fight": {k: fight[k] for k in fight if k != "controller"},
                "pushes": pushes,
                "ok": ok,
                "dump": dump,
                "screenshot": png,
                "whistle_0x065C": dump.get("whistle_0x065C"),
                "rom": rom_room(int(snap.screen)),
            },
        )
        print("FINAL", dump.get("room_hex"), "mode", snap.mode, "xy", [snap.link_x, snap.link_y], "ok", ok, flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
