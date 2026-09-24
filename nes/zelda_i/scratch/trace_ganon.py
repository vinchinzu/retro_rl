"""Trace the L9 Ganon fight from a pin: hits with the frames before them. Scratch.

    QT_QPA_PLATFORM=offscreen uv run python nes/zelda_i/scratch/trace_ganon.py \
        BlueRingFull14_level9_ganon --idle 7 --before 12
"""
import argparse

from retro_harness.env import make_env, read_state_bytes, state_path
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import configure_headless
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.level9.natural_path import NaturalGanonController
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot
from zelda_i.route.chain import bind_controller_env

ap = argparse.ArgumentParser()
ap.add_argument("state")
ap.add_argument("--idle", type=int, default=7)
ap.add_argument("--before", type=int, default=10)
ap.add_argument("--all", action="store_true")
a = ap.parse_args()

configure_headless()
env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
env.reset()
env.em.set_state(read_state_bytes(state_path(GAME_DIR, GAME, a.state)))
for _ in range(a.idle):
    env.step(nes_idle_action())
ctl = NaturalGanonController()
bind_controller_env(ctl, env)
assist = UnlimitedHealthAssist()
tail = []
for f in range(1, ctl.max_frames + 1):
    snap = read_snapshot(env.get_ram())
    act = ctl.step(snap)
    objs = [(hex(o.type_id), o.x, o.y, o.state, o.hp) for o in snap.objects if 1 <= o.slot <= 12]
    ram = env.get_ram()
    line = (f"f{f} L({snap.link_x},{snap.link_y}) face={snap.facing:#x} {act.reason} "
            f"rupees={snap.rupees} w=[{','.join(f'{int(ram[0x70+k])},{int(ram[0x84+k])},{int(ram[0xAC+k])}' for k in (13, 14, 15, 16, 18))}] objs={objs}")
    tail.append(line)
    del tail[:-a.before]
    before = assist.telemetry.total_damage
    iframes_before = int(env.get_ram()[0x04F0])
    env.step(act.action)
    iframes_after = int(env.get_ram()[0x04F0])
    assist.apply_env(env, frame=f)
    hit = assist.telemetry.total_damage > before or (iframes_after > 0 and iframes_before == 0)
    if a.all:
        print(line)
    if hit:
        print(f"--- HIT at f{f} (+{assist.telemetry.total_damage - before}h)")
        print("\n".join(tail))
    if ctl.success or ctl.failed:
        break
print("success", ctl.success, "frames", f, "damage", assist.telemetry.total_damage, ctl.report().get("reason"))
