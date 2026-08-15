
"""Place Link on the 0x77 cellar floor and walk each mouth (dest confirmation)."""
from retro_harness.env import make_env
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.level9_stairs import dest_report, landed_final_patra
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_LINK_X, ADDR_LINK_Y, ADDR_MODE, read_snapshot
from zelda_i.scripts.run_level9_ganon import FIXTURE_SOURCE, _assign, _idle, _step
from zelda_i.scripts.run_level9_stairs import materialize_stair_room

ADDR_UPDATING = 0x0011
ADDR_UW_EXIT = 0x005A


def enter_cellar(env, total):
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


def run(side: str, x: int, y: int) -> dict:
    env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
    total = [0]
    try:
        obs, loader, loaded = enter_cellar(env, total)
        _assign(env, ADDR_LINK_X, x)
        _assign(env, ADDR_LINK_Y, y)
        obs = _idle(env, 8, assist=None, total=total)
        save_rgb_png(obs, RECORDINGS_DIR / f"l9_77_placed_{side}_{x}_{y}.png")
        placed = dest_report(read_snapshot(env.get_ram()))
        print(f"placed {side} ({x},{y})", placed["mode"], placed["link"])
        for i in range(400):
            snap = read_snapshot(env.get_ram())
            if snap.mode == 5 and snap.screen != 0x77:
                break
            if snap.screen != 0x77 and snap.mode not in (6, 7, 9, 10, 16):
                break
            obs = _step(env, nes_action("UP"), assist=None, total=total)
        dest = dest_report(read_snapshot(env.get_ram()))
        save_rgb_png(obs, RECORDINGS_DIR / f"l9_77_placed_dest_{side}.png")
        print(
            f"UP {side} ({x},{y}) -> 0x{dest['screen']:02X} NEXT=0x{dest['next_screen']:02X} "
            f"mode={dest['mode']} patra={landed_final_patra(read_snapshot(env.get_ram()))} "
            f"eyes={dest['patra_eyes']} xy=({dest['link']['x']},{dest['link']['y']}) "
            f"objs={[o['type_name'] for o in dest['objects'][:8]]}"
        )
        return {
            "side": side,
            "stand": [x, y],
            "placed": placed,
            "dest": dest,
            "patra": landed_final_patra(read_snapshot(env.get_ram())),
        }
    finally:
        env.close()


def main():
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    stands = [
        ("left", 0x50, 0x9D),
        ("left", 0x50, 0x7D),
        ("left", 0x50, 0x5D),
        ("left", 0x50, 0x3D),
        ("right", 0xB0, 0x9D),
        ("right", 0xB0, 0x7D),
        ("right", 0xB0, 0x5D),
        ("right", 0xB0, 0x3D),
        ("mid", 0x80, 0x9D),
    ]
    rows = [run(side, x, y) for side, x, y in stands]
    winner = next((r for r in rows if r["patra"] or r["dest"]["screen"] == 0x52), None)
    write_json_report(
        RECORDINGS_DIR / "l9_77_place_mouth.json",
        {"ok": winner is not None, "winner": winner, "stands": rows},
    )
    print("WINNER", winner)


if __name__ == "__main__":
    main()
