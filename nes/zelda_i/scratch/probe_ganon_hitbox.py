"""Map Ganon's blade and contact windows from the ROM. Scratch; what-if writes.

Pins Ganon (slot 1) and Link at a grid of offsets (Link minus Ganon's
ObjX/ObjY), faces Link one way, presses A once, and records per cell whether
Ganon's HP dropped (blade lands) and whether Link was harmed ($04F0 rising or
hearts drop). A measurement of collision geometry, never a route result.

    QT_QPA_PLATFORM=offscreen uv run python nes/zelda_i/scratch/probe_ganon_hitbox.py
"""
import argparse

from retro_harness.env import make_env, read_state_bytes, state_path
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import ADDR_HEALTH, ADDR_HEART_PARTIAL, ADDR_LINK_FACING, ADDR_LINK_IFRAMES, ADDR_OBJ_HP

FACING = {"UP": 0x08, "DOWN": 0x04, "RIGHT": 0x01, "LEFT": 0x02}

ap = argparse.ArgumentParser()
ap.add_argument("--state", default="BlueRingFull14_level9_ganon")
ap.add_argument("--idle", type=int, default=30)
ap.add_argument("--gx", type=int, default=112)
ap.add_argument("--gy", type=int, default=125)
ap.add_argument("--frames", type=int, default=24)
ap.add_argument("--step", type=int, default=4)
ap.add_argument("--span", type=int, default=44)
a = ap.parse_args()

configure_headless()
env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
env.reset()
env.em.set_state(read_state_bytes(state_path(GAME_DIR, GAME, a.state)))
for _ in range(a.idle):
    env.step(nes_idle_action())
base = env.em.get_state()
mem = env.unwrapped.data.memory


def put(addr: int, v: int) -> None:
    mem.assign(addr, "|u1", int(v) & 0xFF)


def cell(dx: int, dy: int, face: str) -> tuple[bool, bool]:
    env.em.set_state(base)
    lx, ly = a.gx + dx, a.gy + dy
    hp0 = int(env.get_ram()[ADDR_OBJ_HP + 1])
    harmed = False
    for f in range(a.frames):
        put(0x70, lx); put(0x84, ly); put(ADDR_LINK_FACING, FACING[face])
        put(0x71, a.gx); put(0x85, a.gy)
        put(ADDR_HEALTH, 0xDD); put(ADDR_HEART_PARTIAL, 0xFF)
        before = int(env.get_ram()[ADDR_LINK_IFRAMES])
        env.step(nes_action("A") if f == 1 else nes_idle_action())
        ram = env.get_ram()
        if int(ram[ADDR_LINK_IFRAMES]) > 0 and before == 0 or (int(ram[ADDR_HEALTH]) & 0x0F) < 0x0D:
            harmed = True
    return int(env.get_ram()[ADDR_OBJ_HP + 1]) < hp0, harmed


rng = range(-a.span, a.span + 1, a.step)
for face in ("UP", "DOWN", "LEFT", "RIGHT"):
    print(f"== facing {face}: rows dy, cols dx (Link - Ganon ObjXY). H=blade lands, x=harmed, *=both, .=neither")
    print("      " + "".join(f"{dx:>4d}" for dx in rng))
    for dy in rng:
        row = []
        for dx in rng:
            hit, harm = cell(dx, dy, face)
            row.append("*" if hit and harm else "H" if hit else "x" if harm else ".")
        print(f"{dy:>5d} " + "".join(f"{c:>4s}" for c in row), flush=True)
