"""Cut one save point per room the L9 Patra join fights in. Scratch.

Plays ``NaturalPatraJoinController`` from a ``*_level9_natural_patra_join``
pin under the Survival refill and writes ``<prefix>_<PHASE>`` on the first
frame of each phase in ``--phases``; prints frames and damage per phase.

    QT_QPA_PLATFORM=offscreen uv run python nes/zelda_i/scratch/cut_l9_join_pins.py \
        BlueRingFull14_level9_natural_patra_join --prefix L9Join14
"""
import argparse
import collections

from retro_harness.env import make_env, read_state_bytes, save_state, state_path
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import configure_headless
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.level9.natural_path import make_natural_patra_join_controller
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import hearts_held, read_snapshot
from zelda_i.route.chain import bind_controller_env

ap = argparse.ArgumentParser()
ap.add_argument("state")
ap.add_argument("--prefix", default="L9Join")
ap.add_argument("--phases", default="CLEAR_20,CLEAR_41,CLEAR_31,CLEAR_30,CLEAR_04,CLEAR_03,STAIRS_03")
ap.add_argument("--idle", type=int, default=0)
ap.add_argument("--frames", type=int, default=24000)
a = ap.parse_args()
want = set(a.phases.split(","))

configure_headless()
env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
env.reset()
env.em.set_state(read_state_bytes(state_path(GAME_DIR, GAME, a.state)))
for _ in range(a.idle):
    env.step(nes_idle_action())
ctl = make_natural_patra_join_controller()
bind_controller_env(ctl, env)
assist = UnlimitedHealthAssist()
frames = collections.Counter()
dmg = collections.Counter()
prev = None
for f in range(1, a.frames + 1):
    snap = read_snapshot(env.get_ram())
    phase = ctl.phase.name
    if phase != prev:
        if phase in want:
            print("saved", save_state(env, GAME_DIR, GAME, f"{a.prefix}_{phase}"), f"f{f}",
                  f"({snap.link_x},{snap.link_y}) 0x{snap.screen:02x}", flush=True)
        prev = phase
    before = assist.telemetry.total_damage
    act = ctl.step(snap)
    env.step(act.action)
    assist.apply_env(env, frame=f)
    frames[phase] += 1
    dmg[phase] += assist.telemetry.total_damage - before
    if ctl.success or ctl.failed:
        break
end = read_snapshot(env.get_ram())
print(f"success={ctl.success} failed={ctl.failed} frames={f} phase={ctl.phase.name} 0x{end.screen:02x}")
for p, n in frames.items():
    print(f"  {p:14s} {n:6d}f  dmg={dmg[p]}")
print("total_damage", assist.telemetry.total_damage, "(units: ", assist.report().get("health"), ")")
