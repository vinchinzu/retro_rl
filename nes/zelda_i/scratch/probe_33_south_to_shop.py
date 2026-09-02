"""Recon: bait shop 0x34 is entered from the SOUTH (source: start U L3 U3, i.e.
arrive 0x34 walking north from 0x44).  From L6 the fixture zigzagged
0x22->0x32->0x33->0x23->0x24->0x25 and dead-ended (0x25 south = solid mountain).

Try instead the south dip: 0x33 -> 0x43 (DOWN) -> 0x44 (RIGHT) -> 0x34 (UP).
Prefix stops at 0x33; then a column-swept DOWN probe; on reaching 0x43 chain
RIGHT then UP.  Every screen change screenshotted; hard give-up per column.

    uv run python nes/zelda_i/scratch/probe_33_south_to_shop.py --tag l7_33s \
        --down-cols 40 72 104 128 152 184 216
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

PREFIX_HOPS = POST_L6_TO_BAIT_HOPS[:2]  # 0x32 DOWN, 0x33 RIGHT -> stop on 0x33
WP_TOL = 7


def _walk_to(env, obs, assist, tx, ty, *, screen, max_steps, tag_prefix):
    """Bang toward (tx,ty) while staying on `screen`.  Returns (obs, snap, why)."""
    last = (-1, -1)
    stuck = 0
    for i in range(max_steps):
        snap = read_snapshot(env.get_ram())
        if snap.screen != screen or snap.mode in (9, 11, 16):
            return obs, snap, "screen_or_mode_change"
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


def _push(env, obs, assist, btn, *, screen, max_steps):
    last = (-1, -1)
    stuck = 0
    for i in range(max_steps):
        snap = read_snapshot(env.get_ram())
        if snap.screen != screen or snap.mode in (9, 11, 16):
            return obs, snap, "screen_or_mode_change"
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
    add_common_args(parser, default_state=POST_L6_EXIT_STATE, default_tag="l7_33s")
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

    try:
        obs, _ = reset_obs(env)
        obs, *_ = env.step(nes_idle_action())
        # --- prefix to 0x33 ---
        for frame in range(8000):
            snap = read_snapshot(env.get_ram())
            if prefix.success and snap.screen == 0x33:
                break
            if prefix.failed:
                events.append(f"prefix_failed {prefix.report().get('notes', [])[-3:]}")
                shot(obs, "prefixfail")
                raise SystemExit(1)
            obs, *_ = env.step(prefix.step(snap).action)
            if assist is not None:
                assist.apply_env(env, frame=frame)
        s = shot(obs, "arrive33")
        events.append(f"arrive 0x33 ({s.link_x},{s.link_y})")

        reached_43 = False
        for cx in args.down_cols:
            obs, s, why1 = _walk_to(env, obs, assist, cx, 150, screen=0x33,
                                    max_steps=1500, tag_prefix=f"to_x{cx}")
            if why1 not in ("reached",):
                events.append(f"col x{cx}: approach {why1} at ({s.link_x},{s.link_y}) s{s.screen:02x}")
                if s.screen != 0x33:
                    events.append(f"  -> left 0x33 to s{s.screen:02x}! capturing")
                    shot(obs, f"x{cx}_left33")
                    if s.screen == 0x43:
                        reached_43 = True
                        break
                continue
            obs, s, why2 = _push(env, obs, assist, "DOWN", screen=0x33, max_steps=600)
            events.append(f"col x{cx}: DOWN -> {why2} at ({s.link_x},{s.link_y}) s{s.screen:02x} m{s.mode}")
            shot(obs, f"x{cx}_down")
            if s.screen == 0x43:
                reached_43 = True
                break
            if s.screen != 0x33:
                shot(obs, f"x{cx}_off33_s{s.screen:02x}")
                break

        if reached_43:
            events.append("=== on 0x43, chaining RIGHT to 0x44 ===")
            shot(obs, "on43")
            # settle
            for _ in range(120):
                obs, *_ = env.step(nes_idle_action())
            for cy in (110, 140, 80, 170):
                obs, s, w = _walk_to(env, obs, assist, 230, cy, screen=0x43,
                                     max_steps=1500, tag_prefix="43R")
                if s.screen != 0x43:
                    events.append(f"0x43 RIGHT@y~{cy} -> s{s.screen:02x} ({s.link_x},{s.link_y})")
                    shot(obs, f"43R_y{cy}_s{s.screen:02x}")
                    break
                obs, s, w = _push(env, obs, assist, "RIGHT", screen=0x43, max_steps=400)
                events.append(f"0x43 RIGHT@y~{cy}: {w} s{s.screen:02x} ({s.link_x},{s.link_y})")
                if s.screen != 0x43:
                    shot(obs, f"43R_y{cy}_s{s.screen:02x}")
                    break
            if s.screen == 0x44:
                events.append("=== on 0x44, chaining UP to 0x34 ===")
                for _ in range(120):
                    obs, *_ = env.step(nes_idle_action())
                for cx in (112, 128, 96, 144, 80):
                    obs, s, w = _walk_to(env, obs, assist, cx, 60, screen=0x44,
                                         max_steps=1500, tag_prefix="44U")
                    if s.screen != 0x44:
                        break
                    obs, s, w = _push(env, obs, assist, "UP", screen=0x44, max_steps=400)
                    events.append(f"0x44 UP@x~{cx}: {w} s{s.screen:02x} ({s.link_x},{s.link_y})")
                    if s.screen != 0x44:
                        break
                shot(obs, f"after44U_s{s.screen:02x}")
                if s.screen == 0x34:
                    events.append("*** REACHED 0x34 (bait shop screen) ***")
                    for k in range(240):
                        if k % 40 == 0:
                            shot(obs, f"on34_k{k}")
                        obs, *_ = env.step(nes_idle_action())
                    shot(obs, "on34_settled")

        s = read_snapshot(env.get_ram())
        leftover = leftover_from_snapshot(s)
        shot(obs, "final")
        payload = {"leftover": leftover, "events": events, "reached_43": reached_43}
        out = write_report("l7_33s", payload, tag=args.tag)
        print(out)
        for e in events:
            print(e)
        print(f"final s{s.screen:02x} m{s.mode} ({s.link_x},{s.link_y})")
    finally:
        env.close()


if __name__ == "__main__":
    main()
