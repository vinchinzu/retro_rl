"""Print Wizzrobe / magic-shot RAM each frame while a controller plays. Scratch.

    QT_QPA_PLATFORM=offscreen uv run python nes/zelda_i/scratch/wiz_trace.py \
        L9S5_s17_level9_stairs_05 zelda_i.level9.prefix:make_stairs_05_controller --frames 400
"""
from __future__ import annotations

import argparse
import importlib

from retro_harness.env import make_env, read_state_bytes, state_path
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import configure_headless
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import hearts_held, read_snapshot
from zelda_i.route.chain import bind_controller_env

TYPES = {0x23: "B", 0x24: "O", 0x58: "b*", 0x59: "o*", 0x47: "P", 0x48: "P", 0x25: "e", 0x26: "e"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("state")
    ap.add_argument("target", nargs="?", default=None)
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--idle", type=int, default=0)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--guard", action="store_true")
    a = ap.parse_args()
    configure_headless()
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    env.reset()
    env.em.set_state(read_state_bytes(state_path(GAME_DIR, GAME, a.state)))
    for w in a.set:
        addr, _, val = w.partition("=")
        env.unwrapped.data.memory.assign(int(addr, 0), "|u1", int(val, 0) & 0xFF)
    for _ in range(a.idle):
        env.step(nes_idle_action())
    ctl = None
    if a.target:
        mod, _, name = a.target.partition(":")
        ctl = getattr(importlib.import_module(mod), name)()
        if a.guard:
            from zelda_i.dungeon.shot_guard import GuardedController

            ctl = GuardedController(ctl)
        bind_controller_env(ctl, env)
    last_h = None
    for f in range(a.frames):
        ram = env.get_ram()
        snap = read_snapshot(ram)
        h = hearts_held(snap)
        cols = []
        for s in range(1, 12):
            t = int(ram[0x34F + s])
            if t in TYPES:
                cols.append(
                    f"{s}:{TYPES[t]}({ram[0x70+s]},{ram[0x84+s]}) d{ram[0x98+s]:x} st{ram[0xAC+s]:02x}"
                    f" tm{ram[0x28+s]:02x} rd{ram[0x394+s]:02x} hp{ram[0x485+s]}"
                )
        act = ctl.step(snap) if ctl else None
        mark = "" if last_h is None or h >= last_h else f"  HIT -{last_h - h:.2f}"
        last_h = h
        print(
            f"{f:4d} fc{ram[0x15]:02x} L({snap.link_x},{snap.link_y}) d{snap.facing:x} m{snap.mode}"
            f" {(act.reason if act else '')[:22]:22s} " + " ".join(cols) + mark
        )
        env.step(act.action if act else nes_idle_action())


if __name__ == "__main__":
    main()
