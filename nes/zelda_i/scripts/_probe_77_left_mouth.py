
"""From a laid-out 0x77 cellar, walk down to the floor then out each mouth.

Also try walking onto the visible stair tiles in play-mode 0x77.
"""
from retro_harness.env import make_env
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.level9_stairs import dest_report, landed_final_patra, walk_to_step
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_LINK_X, ADDR_LINK_Y, ADDR_MODE, read_snapshot
from zelda_i.scripts.run_level9_ganon import FIXTURE_SOURCE, _assign, _idle, _step
from zelda_i.scripts.run_level9_stairs import materialize_stair_room

ADDR_UPDATING = 0x0011
ADDR_UW_EXIT = 0x005A
FLOOR_Y = 0x9D
LEFT_X = 0x50
RIGHT_X = 0xB0
EXIT_Y = 0x3D


def _enter_cellar(env, total):
    obs, loader, loaded = materialize_stair_room(env, 0x77, total=total)
    _assign(env, ADDR_MODE, 9)
    _assign(env, 0x0013, 0)
    _assign(env, ADDR_UPDATING, 0)
    _assign(env, ADDR_UW_EXIT, 0)
    for i in range(400):
        obs = _step(env, nes_idle_action(), assist=None, total=total)
        snap = read_snapshot(env.get_ram())
        if snap.mode == 9 and int(env.get_ram()[ADDR_UPDATING]) != 0 and i > 20:
            break
    return obs, loader, loaded


def _walk(env, total, x, y, frames=400):
    obs = None
    for _ in range(frames):
        snap = read_snapshot(env.get_ram())
        frame = walk_to_step(snap, x, y, y_first=True)
        obs = _step(env, frame.action, assist=None, total=total)
        if frame.reason == "walk_arrived":
            return obs, snap
    return obs, read_snapshot(env.get_ram())


def exit_mouth(env, total, side: str):
    target_x = LEFT_X if side == "left" else RIGHT_X
    # 1. Down the current stairs onto the brick floor.
    obs, snap = _walk(env, total, snap_x_or(env, target_x), FLOOR_Y, frames=500)
    # 2. Across the floor to the chosen mouth.
    obs, snap = _walk(env, total, target_x, FLOOR_Y, frames=400)
    # 3. Up the mouth until dest or timeout.
    for i in range(500):
        snap = read_snapshot(env.get_ram())
        if snap.mode == 5 and snap.screen != 0x77:
            return obs, snap
        if snap.mode not in (5, 9, 10, 16) and snap.screen != 0x77:
            return obs, snap
        if abs(snap.link_x - target_x) > 3:
            obs = _step(
                env,
                nes_action("RIGHT" if snap.link_x < target_x else "LEFT"),
                assist=None,
                total=total,
            )
        else:
            obs = _step(env, nes_action("UP"), assist=None, total=total)
    return obs, read_snapshot(env.get_ram())


def snap_x_or(env, fallback):
    return read_snapshot(env.get_ram()).link_x


def run_cellar_side(side: str) -> dict:
    env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
    total = [0]
    try:
        obs, loader, loaded = _enter_cellar(env, total)
        save_rgb_png(obs, RECORDINGS_DIR / f"l9_77_cellar_ready_{side}.png")
        after = dest_report(read_snapshot(env.get_ram()))
        print(f"cellar ready {side}", after["mode"], after["link"], [o["type_name"] for o in after["objects"][:6]])
        obs, snap = exit_mouth(env, total, side)
        dest = dest_report(snap)
        save_rgb_png(obs, RECORDINGS_DIR / f"l9_77_cellar_dest_{side}.png")
        print(
            f"MOUTH {side} -> 0x{dest['screen']:02X} NEXT=0x{dest['next_screen']:02X} "
            f"mode={dest['mode']} patra={landed_final_patra(snap)} eyes={dest['patra_eyes']} "
            f"xy=({dest['link']['x']},{dest['link']['y']}) "
            f"objs={[o['type_name'] for o in dest['objects'][:8]]}"
        )
        return {
            "side": side,
            "loaded": loaded,
            "after_init": after,
            "dest": dest,
            "patra": landed_final_patra(snap),
        }
    finally:
        env.close()


def try_play_stairs() -> dict:
    """Walk from the open left floor onto the visible stair row in 0x77."""
    env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
    total = [0]
    hits = []
    try:
        obs, loader, loaded = materialize_stair_room(env, 0x77, total=total)
        # Place on open left floor, then walk toward candidate stair stands.
        stands = [
            (0xA0, 0x8D),
            (0xB0, 0x8D),
            (0x90, 0x8D),
            (0xC0, 0x8D),
            (0xA0, 0x9D),
            (0xB0, 0x9D),
            (0xA0, 0x7D),
            (0x80, 0x8D),
            (0xD0, 0x60),
            (0xD0, 0x5D),
        ]
        for x, y in stands:
            _assign(env, ADDR_LINK_X, 0x40)
            _assign(env, ADDR_LINK_Y, 0x8D)
            _idle(env, 4, assist=None, total=total)
            for _ in range(300):
                snap = read_snapshot(env.get_ram())
                if snap.mode != 5 or snap.screen != 0x77:
                    dest = dest_report(snap)
                    hits.append({"stand": [x, y], "dest": dest})
                    print(f"PLAY STAIRS hit stand=({x},{y}) mode={snap.mode} room=0x{snap.screen:02X}")
                    save_rgb_png(obs, RECORDINGS_DIR / f"l9_77_play_stairs_{x}_{y}.png")
                    return {"loaded": loaded, "hits": hits, "winner_stand": [x, y], "dest": dest}
                frame = walk_to_step(snap, x, y, y_first=False)
                obs = _step(env, frame.action, assist=None, total=total)
                if frame.reason == "walk_arrived":
                    for d in ("UP", "DOWN", "LEFT", "RIGHT"):
                        obs = _step(env, nes_action(d), assist=None, total=total)
                        snap = read_snapshot(env.get_ram())
                        if snap.mode != 5 or snap.screen != 0x77:
                            dest = dest_report(snap)
                            hits.append({"stand": [x, y], "nudge": d, "dest": dest})
                            print(f"PLAY STAIRS nudge {d} stand=({x},{y}) mode={snap.mode} room=0x{snap.screen:02X}")
                            save_rgb_png(obs, RECORDINGS_DIR / f"l9_77_play_stairs_{x}_{y}_{d}.png")
                            return {"loaded": loaded, "hits": hits, "winner_stand": [x, y], "dest": dest}
                    break
        save_rgb_png(obs, RECORDINGS_DIR / "l9_77_play_stairs_miss.png")
        return {"loaded": loaded, "hits": hits}
    finally:
        env.close()


def main():
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    rows = [run_cellar_side("left"), run_cellar_side("right")]
    play = try_play_stairs()
    winner = next((r for r in rows if r["patra"]), None)
    write_json_report(
        RECORDINGS_DIR / "l9_77_left_mouth.json",
        {"ok": winner is not None, "winner": winner, "cellar": rows, "play_stairs": play},
    )
    print("WINNER", winner)
    print("PLAY", play)


if __name__ == "__main__":
    main()
