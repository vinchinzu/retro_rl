
"""Properly InitMode9 in 0x77 (IsUpdatingMode=0) and walk left cellar mouth."""
from retro_harness.env import make_env
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.level9_stairs import dest_report, landed_final_patra
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from retro_harness.nes import nes_idle_action
from zelda_i.ram import ADDR_LINK_X, ADDR_LINK_Y, ADDR_MODE, read_snapshot
from zelda_i.scripts.run_level9_ganon import FIXTURE_SOURCE, _assign, _idle, _step
from zelda_i.scripts.run_level9_stairs import materialize_stair_room, _exit_cellar

# ADDR_IS_UPDATING_MODE may not be exported; 0x0011
ADDR_UPDATING = 0x0011
ADDR_UW_EXIT = 0x005A


def run_side(side: str) -> dict:
    env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
    total = [0]
    try:
        obs, loader, loaded = materialize_stair_room(env, 0x77, total=total)
        settle = dest_report(read_snapshot(env.get_ram()))
        save_rgb_png(obs, RECORDINGS_DIR / f"l9_77_before_mode9_{side}.png")
        # Let the engine initialize mode 9 (cellar layout), do not skip InitMode9.
        _assign(env, ADDR_MODE, 9)
        _assign(env, 0x0013, 0)
        _assign(env, ADDR_UPDATING, 0)
        _assign(env, ADDR_UW_EXIT, 0)  # not leaving underground
        # Idle through fade + layout.
        for i in range(400):
            obs = _step(env, nes_idle_action(), assist=None, total=total)
            snap = read_snapshot(env.get_ram())
            if snap.mode == 9 and int(env.get_ram()[ADDR_UPDATING]) != 0 and i > 20:
                break
        after_init = dest_report(read_snapshot(env.get_ram()))
        save_rgb_png(obs, RECORDINGS_DIR / f"l9_77_mode9_init_{side}.png")
        print(f"after InitMode9 {side}", after_init["mode"], after_init["screen"], after_init["link"])
        obs, snap = _exit_cellar(env, total=total, side=side)
        dest = dest_report(snap)
        save_rgb_png(obs, RECORDINGS_DIR / f"l9_77_mode9_exit_{side}.png")
        print(
            f"exit {side} -> 0x{dest['screen']:02X} mode={dest['mode']} "
            f"patra={landed_final_patra(snap)} eyes={dest['patra_eyes']} "
            f"xy=({dest['link']['x']},{dest['link']['y']}) "
            f"objs={[o['type_name'] for o in dest['objects'][:8]]}"
        )
        return {
            "side": side,
            "loaded": loaded,
            "loader": loader.label,
            "settle": settle,
            "after_init": after_init,
            "dest": dest,
            "patra": landed_final_patra(snap),
        }
    finally:
        env.close()


def main():
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    rows = [run_side("left"), run_side("right")]
    winner = next((r for r in rows if r["patra"]), None)
    write_json_report(
        RECORDINGS_DIR / "l9_77_init_mode9.json",
        {"ok": winner is not None, "winner": winner, "sides": rows},
    )
    print("WINNER", None if winner is None else winner["side"])


if __name__ == "__main__":
    main()
