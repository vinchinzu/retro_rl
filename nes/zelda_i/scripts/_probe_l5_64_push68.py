"""From Entered64: north-wall to (80,141), push 0x68, take center stairs. No pokes."""
from retro_harness.env import make_env, reset_obs, state_path, write_state_bytes
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ops import idle, push_dir
from zelda_i.dungeon_trace import write_state_provenance
from zelda_i.level9_stairs import on_stair_tile, stair_transition_modes
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8
from zelda_i.scripts._probe_l5_whistle_path import (
    ROOM_64,
    dump_live,
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


def main():
    configure_headless()
    env, assist, obs = open_env()
    total = [1]
    try:
        idle(env, assist, total, 12)
        start = dump_live(read_snapshot(env.get_ram()), env.get_ram())
        print("START", [start.get("x"), start.get("y")], "blocks", start.get("blocks_0x68"), flush=True)
        # North wall cross to west-center.
        for axis, tgt in (("y", 93), ("x", 80), ("y", 141)):
            ok = walk_axis(env, assist, total, axis, tgt, max_f=400)
            snap = read_snapshot(env.get_ram())
            print("NAV", axis, tgt, ok, [snap.link_x, snap.link_y], flush=True)
        walk_axis(env, assist, total, "y", 144, max_f=200)
        walk_axis(env, assist, total, "x", 80, max_f=200)
        at = dump_live(read_snapshot(env.get_ram()), env.get_ram())
        print("AT_BLOCK", [at.get("x"), at.get("y")], "blocks", at.get("blocks_0x68"), flush=True)
        png0 = shot(env, assist, total, "l5_64_at68")
        # Push the 0x68.
        log = []
        for direction in ("RIGHT", "UP", "DOWN", "LEFT"):
            push_dir(env, assist, total, direction, frames=100)
            idle(env, assist, total, 8)
            snap = read_snapshot(env.get_ram())
            rec = {
                "dir": direction,
                "xy": [snap.link_x, snap.link_y],
                "tile": int(snap.colliding_tile),
                "stair": bool(on_stair_tile(snap)),
                "mode": snap.mode,
                "room": f"0x{snap.screen:02x}",
                "blocks": [
                    {"x": o.x, "y": o.y}
                    for o in snap.objects
                    if 1 <= o.slot <= 12 and o.type_id == 0x68
                ],
            }
            log.append(rec)
            print("PUSH", rec, flush=True)
            if stair_transition_modes(snap.mode) or (snap.screen != ROOM_64 and snap.mode == PLAY_MODE):
                break
        # After push, walk toward center stairs.
        for axis, tgt in (("y", 141), ("x", 120), ("y", 141), ("x", 112), ("y", 125), ("x", 120)):
            if read_snapshot(env.get_ram()).screen != ROOM_64:
                break
            walk_axis(env, assist, total, axis, tgt, max_f=300)
            snap = read_snapshot(env.get_ram())
            print("AFTER", axis, tgt, [snap.link_x, snap.link_y], "tile", snap.colliding_tile, "stair", on_stair_tile(snap), flush=True)
            if stair_transition_modes(snap.mode) or on_stair_tile(snap):
                break
        for direction in ("UP", "DOWN", "RIGHT", "LEFT"):
            for _ in range(40):
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
            "l5_64_push68",
            {
                "pokes": False,
                "status_claim": None,
                "start": start,
                "at_block": at,
                "pushes": log,
                "ok": ok,
                "dump": dump,
                "screenshot": png,
                "at68_png": png0,
                "whistle_0x065C": dump.get("whistle_0x065C"),
                "rom": rom_room(int(snap.screen)),
            },
        )
        print("FINAL", dump.get("room_hex"), "mode", snap.mode, "xy", [snap.link_x, snap.link_y], "ok", ok, flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
