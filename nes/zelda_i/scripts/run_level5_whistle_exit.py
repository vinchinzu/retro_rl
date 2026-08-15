"""Exit L5 Whistle cellar 0x04 onto the floor map (room 0x05).

From Level5Whistle (mode 9, recorder alcove). No door/key pokes. Survival OK.
Does not start Digdogger.

    PYTHONPATH=nes:. uv run python nes/zelda_i/scripts/run_level5_whistle_exit.py \
        --from-state Level5Whistle --infinite-life --save-state
"""
from __future__ import annotations

import argparse

from retro_harness.env import make_env, reset_obs, state_path, write_state_bytes
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ops import idle
from zelda_i.dungeon_trace import write_state_provenance
from zelda_i.level5_path import (
    PLAY_MODE,
    ROOM_L5_WHISTLE_05,
    ROOM_L5_WHISTLE_ITEM,
    exit_whistle_04,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_WHISTLE, read_snapshot, read_u8


def _inv(ram) -> dict:
    snap = read_snapshot(ram)
    return {
        "whistle_0x065C": int(read_u8(ram, ADDR_WHISTLE)),
        "keys": int(snap.keys),
        "bombs": int(snap.bombs),
        "room": snap.screen,
        "mode": snap.mode,
        "xy": [snap.link_x, snap.link_y],
        "tile": int(snap.colliding_tile),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-state", default="Level5Whistle")
    ap.add_argument("--infinite-life", action="store_true")
    ap.add_argument("--save-state", action="store_true")
    ap.add_argument("--tag", default="l5_whistle_floor")
    args = ap.parse_args()

    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    env = make_env(GAME, args.from_state, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True) if args.infinite_life else None
    total = [1]
    ckpt = None
    try:
        reset_obs(env)
        env.step(nes_idle_action())
        if assist is not None:
            assist.apply_env(env, frame=0)
        idle(env, assist, total, 8)
        start = _inv(env.get_ram())
        print("START", start, flush=True)
        obs, *_ = env.step(nes_idle_action())
        total[0] += 1
        save_rgb_png(obs, RECORDINGS_DIR / f"{args.tag}_start.png")

        walk = exit_whistle_04(env, assist, total)
        end = _inv(env.get_ram())
        print("EXIT", walk.get("success"), walk.get("left_cellar"), "dest", hex(walk.get("dest", 0)), walk.get("xy"), flush=True)
        print("LOG", walk.get("log"), flush=True)

        obs, *_ = env.step(nes_idle_action())
        total[0] += 1
        save_rgb_png(obs, RECORDINGS_DIR / f"{args.tag}_final.png")

        left = bool(walk.get("left_cellar")) and end["mode"] == PLAY_MODE and end["room"] != ROOM_L5_WHISTLE_ITEM
        if args.save_state and left and end["whistle_0x065C"] >= 1:
            name = "Level5WhistleFloor"
            path = write_state_bytes(state_path(GAME_DIR, GAME, name), env.em.get_state())
            write_state_provenance(
                path,
                source_state_path=GAME_DIR / "custom_integrations" / GAME / f"{args.from_state}.state",
                request={
                    "segment": name,
                    "predecessor_entry": True,
                    "start_state": args.from_state,
                    "via": "0x04 short ladder x=176 DOWN, pit y=189, left mouth x=48 UP",
                    "key_poke": False,
                    "door_poke": False,
                    "bomb_count_poke": False,
                    "selected_item_poke": False,
                },
                selected_trial={
                    "success": True,
                    "room": end["room"],
                    "mode": end["mode"],
                    "whistle_0x065C": end["whistle_0x065C"],
                    "xy": end["xy"],
                },
                natural_entry=False,
            )
            ckpt = name
            print("PINNED", name, "room", hex(end["room"]), "mode", end["mode"], flush=True)

        body = {
            "ok": left and end["whistle_0x065C"] >= 1,
            "status_claim": None,
            "pokes": False,
            "track": "assisted" if args.infinite_life else "clean",
            "start_state": args.from_state,
            "start": start,
            "exit": {k: walk[k] for k in walk if k != "log"},
            "log": walk.get("log"),
            "final": end,
            "checkpoint": ckpt,
            "landing_room": f"0x{end['room']:02x}",
            "landing_name": "six-Darknut whistle room" if end["room"] == ROOM_L5_WHISTLE_05 else None,
            "stand": "alcove (136,141) → ladder x=176 DOWN → pit (176,189) → left mouth x=48 UP",
        }
        write_json_report(RECORDINGS_DIR / f"{args.tag}.json", body)
        print("OK", body["ok"], "room", body["landing_room"], "ckpt", ckpt, flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
