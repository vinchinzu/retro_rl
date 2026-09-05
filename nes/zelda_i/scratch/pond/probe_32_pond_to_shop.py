"""Recon: L6 -> bait shop 0x34 via the SOUTH approach through the pond screen.

Route under test: 0x22 -> 0x32 (DOWN) -> 0x42 (DOWN, = L7 pond screen) ->
0x43 (RIGHT) -> 0x44 (RIGHT) -> 0x34 (UP, shop from the south, per source).

Prefix walks to 0x32 with OverworldToBaitShopController; then a column-swept
DOWN probe to 0x42, screenshots the pond, then chains RIGHT/RIGHT/UP.  Every
screen change screenshotted; hard per-column give-up.

    uv run python nes/zelda_i/scratch/pond/probe_32_pond_to_shop.py --tag l7_32pond
"""

from __future__ import annotations

import argparse

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.level7.entry import POST_L6_EXIT_STATE
from zelda_i.level7.overworld import POST_L6_TO_BAIT_HOPS, OverworldToBaitShopController
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import read_snapshot
from zelda_i.runner import add_common_args, make_assist, write_report
from zelda_i.screen_glance import leftover_from_snapshot

PREFIX_HOPS = POST_L6_TO_BAIT_HOPS[:1]  # 0x32 DOWN -> stop on 0x32
WP_TOL = 7
OFF_MODES = (9, 11, 16)


def walk_to(env, obs, assist, tx, ty, *, screen, max_steps=1600):
    last = (-1, -1)
    stuck = 0
    for i in range(max_steps):
        snap = read_snapshot(env.get_ram())
        if snap.screen != screen or snap.mode in OFF_MODES:
            return obs, snap, "off"
        dx, dy = tx - snap.link_x, ty - snap.link_y
        if abs(dx) <= WP_TOL and abs(dy) <= WP_TOL:
            return obs, snap, "reached"
        cur = (snap.link_x, snap.link_y)
        stuck = stuck + 1 if cur == last else 0
        last = cur
        if stuck > 70:
            return obs, snap, "stuck"
        btn = ("RIGHT" if dx > 0 else "LEFT") if abs(dx) >= abs(dy) else ("DOWN" if dy > 0 else "UP")
        obs, *_ = env.step(nes_action(btn))
        if assist is not None:
            assist.apply_env(env, frame=i)
    return obs, read_snapshot(env.get_ram()), "timeout"


def push(env, obs, assist, btn, *, screen, max_steps=500):
    last = (-1, -1)
    stuck = 0
    for i in range(max_steps):
        snap = read_snapshot(env.get_ram())
        if snap.screen != screen or snap.mode in OFF_MODES:
            return obs, snap, "off"
        cur = (snap.link_x, snap.link_y)
        stuck = stuck + 1 if cur == last else 0
        last = cur
        if stuck > 70:
            return obs, snap, "stuck"
        obs, *_ = env.step(nes_action(btn))
        if assist is not None:
            assist.apply_env(env, frame=i)
    return obs, read_snapshot(env.get_ram()), "timeout"


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_state=POST_L6_EXIT_STATE, default_tag="l7_32pond")
    parser.add_argument("--down-cols", type=int, nargs="+",
                        default=[40, 72, 104, 128, 152, 184, 216])
    args = parser.parse_args()
    configure_headless()

    prefix = OverworldToBaitShopController(hops=PREFIX_HOPS)
    assist = make_assist(args.infinite_life)
    env = make_env(GAME, args.from_state, GAME_DIR, render_mode="rgb_array")
    events: list[str] = []

    def shot(obs, tag):
        s = read_snapshot(env.get_ram())
        save_rgb_png(obs, RECORDINGS_DIR / f"{args.tag}_{tag}_s{s.screen:02x}_m{s.mode}.png")
        return s

    def chain(obs, frm, to, btn, ys):
        """Try to cross `frm`->`to` via btn at a spread of align values `ys`."""
        for v in ys:
            if btn in ("RIGHT", "LEFT"):
                obs2, s, w = walk_to(env, obs, assist, 230 if btn == "RIGHT" else 8, v, screen=frm)
            else:
                obs2, s, w = walk_to(env, obs, assist, v, 40 if btn == "UP" else 200, screen=frm)
            obs = obs2
            if s.screen != frm:
                events.append(f"{frm:#x}->{s.screen:#x} via {btn} align~{v} (walk) ({s.link_x},{s.link_y})")
                return obs, s, True
            obs, s, w = push(env, obs, assist, btn, screen=frm)
            events.append(f"{frm:#x} {btn} align~{v}: {w} -> s{s.screen:#x} ({s.link_x},{s.link_y})")
            if s.screen != frm:
                return obs, s, s.screen == to
        return obs, read_snapshot(env.get_ram()), False

    try:
        obs, _ = reset_obs(env)
        obs, *_ = env.step(nes_idle_action())
        for frame in range(8000):
            snap = read_snapshot(env.get_ram())
            if prefix.success and snap.screen == 0x32:
                break
            if prefix.failed:
                events.append(f"prefix_failed {prefix.report().get('notes', [])[-3:]}")
                shot(obs, "prefixfail")
                raise SystemExit(1)
            obs, *_ = env.step(prefix.step(snap).action)
            if assist is not None:
                assist.apply_env(env, frame=frame)
        s = shot(obs, "arrive32")
        events.append(f"arrive 0x32 ({s.link_x},{s.link_y})")

        # Clear the 0x32 north wall: x=120 top is tile 216 (walled); shift to
        # x~112 then descend into the open interior before the column sweep.
        obs, s, w = push(env, obs, assist, "LEFT", screen=0x32, max_steps=40)
        obs, s, w = walk_to(env, obs, assist, 112, 130, screen=0x32)
        events.append(f"0x32 clear-north: {w} ({s.link_x},{s.link_y})")

        # --- 0x32 DOWN -> 0x42 ---
        on42 = False
        for cx in args.down_cols:
            obs, s, w = walk_to(env, obs, assist, cx, 170, screen=0x32)
            if s.screen != 0x32:
                events.append(f"0x32 col x{cx}: left to s{s.screen:#x} while approaching")
                shot(obs, f"x{cx}_off32")
                if s.screen == 0x42:
                    on42 = True
                break
            if w != "reached":
                events.append(f"0x32 col x{cx}: approach {w} ({s.link_x},{s.link_y})")
                continue
            obs, s, w = push(env, obs, assist, "DOWN", screen=0x32)
            events.append(f"0x32 col x{cx}: DOWN {w} -> s{s.screen:#x} ({s.link_x},{s.link_y})")
            shot(obs, f"x{cx}_down")
            if s.screen == 0x42:
                on42 = True
                break
            if s.screen != 0x32:
                break

        if not on42:
            events.append("did not reach 0x42 from 0x32")
        else:
            events.append("=== on 0x42 (pond screen) ===")
            for _ in range(150):
                obs, *_ = env.step(nes_idle_action())
            for k in range(3):
                shot(obs, f"pond42_k{k}")
                for _ in range(60):
                    obs, *_ = env.step(nes_idle_action())
            # 0x42 -> 0x43 RIGHT
            obs, s, ok = chain(obs, 0x42, 0x43, "RIGHT", (141, 110, 170, 80))
            if s.screen == 0x43:
                shot(obs, "on43")
                for _ in range(120):
                    obs, *_ = env.step(nes_idle_action())
                obs, s, ok = chain(obs, 0x43, 0x44, "RIGHT", (141, 110, 170, 80))
                if s.screen == 0x44:
                    shot(obs, "on44")
                    for _ in range(120):
                        obs, *_ = env.step(nes_idle_action())
                    obs, s, ok = chain(obs, 0x44, 0x34, "UP", (112, 128, 96, 144, 80))
                    if s.screen == 0x34:
                        events.append("*** REACHED 0x34 (bait shop screen) ***")
                        for k in range(6):
                            shot(obs, f"on34_k{k}")
                            for _ in range(50):
                                obs, *_ = env.step(nes_idle_action())

        s = read_snapshot(env.get_ram())
        leftover = leftover_from_snapshot(s)
        shot(obs, "final")
        payload = {"leftover": leftover, "events": events, "on42": on42}
        out = write_report("l7_32pond", payload, tag=args.tag)
        print(out)
        for e in events:
            print(e)
        print(f"final s{s.screen:#x} m{s.mode} ({s.link_x},{s.link_y})")
    finally:
        env.close()


if __name__ == "__main__":
    main()
