"""Recon: drive the green prefix to 0x6B, clear the goriya, scan its exits.

Chains north / east69 / east6a, then in 0x6B: fight the live goriya(s) with
the shared goriya micro, then probe each cardinal for a walkable doorway
(walk into the wall band, see if the screen changes).  Do not poke.

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/sweep_l7_room6b.py
"""

from __future__ import annotations

import argparse
import json

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.combat import nearest_enemy
from zelda_i.dungeon.behaviors import EnemyKind, engagement_hint
from zelda_i.level7.path import (
    ROOM_6A,
    EntryNorthDoorController,
    Room6AEastController,
    Room69EastController,
    live_goriyas,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import PLAY_MODE, read_snapshot
from zelda_i.runner import add_common_args, make_assist

ROOM_6B = 0x6B


def _drive(env, assist):
    chain = [EntryNorthDoorController(), Room69EastController(), Room6AEastController()]
    idx = 0
    ctl = chain[idx]
    for frame in range(sum(c.max_frames for c in chain)):
        snap = read_snapshot(env.get_ram())
        env.step(ctl.step(snap).action)
        if assist is not None:
            assist.apply_env(env, frame=frame)
        s = read_snapshot(env.get_ram())
        if s.screen == ROOM_6B and s.mode == PLAY_MODE and not s.transitioning:
            return True, frame
        if ctl.success and idx < len(chain) - 1:
            idx += 1
            ctl = chain[idx]
        elif ctl.failed:
            return False, frame
    return False, -1


def _fight(env, assist, base, max_frames=3000):
    saw = False
    for i in range(max_frames):
        s = read_snapshot(env.get_ram())
        if s.screen != ROOM_6B:
            return base + i, f"left_to_0x{s.screen:02x}"
        live = live_goriyas(s)
        if live:
            saw = True
        if not live:
            if not saw or i < 90:
                env.step(nes_action("RIGHT"))
                if assist is not None:
                    assist.apply_env(env, frame=base + i)
                continue
            return base + i, "clear"
        tgt = nearest_enemy(s.link_x, s.link_y, live)
        hint = engagement_hint(EnemyKind.GORIYA, s, tgt)
        btn = hint.face
        act = nes_action(btn, "A") if (i % 8) < 4 else nes_action(btn)
        env.step(act)
        if assist is not None:
            assist.apply_env(env, frame=base + i)
    return base + max_frames, "timeout"


def _probe_dir(env, assist, base, button, want_y=None, want_x=None):
    # move to a sensible approach point then hold the button
    for _ in range(160):
        s = read_snapshot(env.get_ram())
        if s.screen != ROOM_6B:
            break
        if want_y is not None and abs(int(s.link_y) - want_y) > 3:
            env.step(nes_action("UP" if int(s.link_y) > want_y else "DOWN"))
        elif want_x is not None and abs(int(s.link_x) - want_x) > 3:
            env.step(nes_action("LEFT" if int(s.link_x) > want_x else "RIGHT"))
        else:
            break
        if assist is not None:
            assist.apply_env(env, frame=base)
        base += 1
    start = read_snapshot(env.get_ram())
    for _ in range(150):
        s = read_snapshot(env.get_ram())
        if s.screen != ROOM_6B:
            return base, {
                "dir": button,
                "result": f"reached_0x{s.screen:02x}",
                "xy": [int(s.link_x), int(s.link_y)],
            }
        env.step(nes_action(button))
        if assist is not None:
            assist.apply_env(env, frame=base)
        base += 1
    end = read_snapshot(env.get_ram())
    return base, {
        "dir": button,
        "result": "blocked",
        "from": [int(start.link_x), int(start.link_y)],
        "to": [int(end.link_x), int(end.link_y)],
        "tile": int(end.colliding_tile),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_state="Level7Entrance", default_tag="sweep6b")
    args = parser.parse_args()
    configure_headless()
    assist = make_assist(args.infinite_life)
    env = make_env(GAME, args.from_state, GAME_DIR, render_mode="rgb_array")
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    out: dict = {}
    try:
        reset_obs(env)
        ok, f = _drive(env, assist)
        s = read_snapshot(env.get_ram())
        print(f"reached_6b={ok} frame={f} xy=({s.link_x},{s.link_y}) obj={[hex(int(o.type_id)) for o in s.objects if 1<=o.slot<=12 and int(o.type_id) not in (0,0xff)]}")
        out["reached"] = ok
        if not ok:
            return
        base = f
        base, clear = _fight(env, assist, base)
        s = read_snapshot(env.get_ram())
        print(f"fight -> {clear} at frame {base} xy=({s.link_x},{s.link_y}) screen=0x{s.screen:02x}")
        out["fight"] = clear
        save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_cleared.png")
        probes = []
        if s.screen == ROOM_6B:
            for button, wy, wx in [
                ("RIGHT", 141, None), ("UP", None, 128),
                ("DOWN", None, 128), ("LEFT", 141, None),
            ]:
                base, r = _probe_dir(env, assist, base, button, want_y=wy, want_x=wx)
                print(r)
                probes.append(r)
                s2 = read_snapshot(env.get_ram())
                if s2.screen != ROOM_6B:
                    # step back into 6b
                    for _ in range(140):
                        ss = read_snapshot(env.get_ram())
                        if ss.screen == ROOM_6B and ss.mode == PLAY_MODE and not ss.transitioning:
                            break
                        opp = {"RIGHT": "LEFT", "LEFT": "RIGHT", "UP": "DOWN", "DOWN": "UP"}[button]
                        env.step(nes_action(opp))
                        if assist is not None:
                            assist.apply_env(env, frame=base)
                        base += 1
        out["probes"] = probes
        save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_final.png")
        (RECORDINGS_DIR / f"{args.tag}.json").write_text(json.dumps(out, indent=1))
        print("wrote", RECORDINGS_DIR / f"{args.tag}.json")
    finally:
        env.close()


if __name__ == "__main__":
    main()
