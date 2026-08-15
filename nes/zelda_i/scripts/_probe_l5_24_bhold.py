"""Dump inv, hold B from a safe stand, watch mode/objects."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless
from zelda_i.dungeon_ops import idle
from zelda_i.level5_path import select_b_item_menu, walk_axis
from zelda_i.ram import ADDR_SELECTED_ITEM, ADDR_WHISTLE, read_snapshot, read_u8

HERE = Path(__file__).resolve()
_spec = importlib.util.spec_from_file_location("l5whistle", HERE.parent / "_probe_l5_whistle_path.py")
w = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(w)


def inv(ram):
    return {f"0x{a:04X}": int(read_u8(ram, a)) for a in range(0x0656, 0x0667)}


def objs(snap):
    return [(hex(o.type_id), o.hp) for o in snap.objects if 1 <= o.slot <= 12 and o.type_id not in (0, 0xFF, 0x55)]


def main():
    configure_headless()
    env, assist, _ = w.open_env("Level5Whistle24")
    total = [1]
    try:
        idle(env, assist, total, 10)
        print("INV", inv(env.get_ram()), flush=True)
        walk_axis(env, assist, total, "y", 181, max_f=400)
        walk_axis(env, assist, total, "x", 160, max_f=400)
        idle(env, assist, total, 8)
        print("STAND", [read_snapshot(env.get_ram()).link_x, read_snapshot(env.get_ram()).link_y], "mode", read_snapshot(env.get_ram()).mode, flush=True)
        menu = select_b_item_menu(env, assist, total, 5)
        print("MENU", menu, "INV", inv(env.get_ram()), flush=True)
        # Hold B 12 frames
        for _ in range(12):
            w.step(env, assist, total, nes_action("B"))
        print("HELD", "mode", read_snapshot(env.get_ram()).mode, "objs", objs(read_snapshot(env.get_ram())), flush=True)
        for n in range(16):
            idle(env, assist, total, 16)
            snap = read_snapshot(env.get_ram())
            print(f"W{n}", "mode", snap.mode, "sub", snap.submode, "xy", [snap.link_x, snap.link_y], "objs", objs(snap), flush=True)
            if any(t == "0x18" for t, _ in objs(snap)):
                print("SHRANK", flush=True)
                break
        w.shot(env, assist, total, "l5_24_bhold")
        print("END INV", inv(env.get_ram()), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
