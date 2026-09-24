"""Trace one stage controller from a pin: each hit with the frames before it,
and the reason census. A hit is a hearts drop or $04F0 rising. Scratch.

    QT_QPA_PLATFORM=offscreen uv run python nes/zelda_i/scratch/trace_stage.py \
        BlueRingFull14_level9_ganon zelda_i.level9.natural_path:NaturalGanonController \
        --idle 7 --before 12
"""
import argparse
import collections
import importlib

from retro_harness.env import make_env, read_state_bytes, state_path
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import configure_headless
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot
from zelda_i.route.chain import bind_controller_env

ap = argparse.ArgumentParser()
ap.add_argument("state")
ap.add_argument("target", help="module:factory")
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
mod, _, name = a.target.partition(":")
ctl = getattr(importlib.import_module(mod), name)()
reasons = collections.Counter()
bind_controller_env(ctl, env)
assist = UnlimitedHealthAssist()
tail = []
for f in range(1, int(getattr(ctl, "max_frames", 8000)) + 1):
    snap = read_snapshot(env.get_ram())
    act = ctl.step(snap)
    reasons[act.reason] += 1
    objs = [(hex(o.type_id), o.x, o.y, o.state, o.hp) for o in snap.objects if 1 <= o.slot <= 12]
    objs = [o for o in objs if o[0] != "0x0"]
    line = f"f{f} L({snap.link_x},{snap.link_y}) face={snap.facing:#x} {act.reason} objs={objs}"
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
    if ctl.success or getattr(ctl, "failed", False):
        break
print("success", ctl.success, "frames", f, "whole-heart damage", assist.telemetry.total_damage)
print("reasons:", " ".join(f"{r}={n}" for r, n in reasons.most_common(14)))
