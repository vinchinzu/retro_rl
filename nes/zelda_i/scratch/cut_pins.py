"""Cut save points inside one stage, which has none of its own. Scratch.

Plays ``TARGET`` (``module:factory``) from a pin under the Survival refill
and writes ``<prefix>_<key>`` the first frame each ``--phases`` name is the
controller's phase, or the first playable frame in each ``--screens`` room.
Prints frames and whole-heart damage per phase (or per room).

    QT_QPA_PLATFORM=offscreen uv run python nes/zelda_i/scratch/cut_pins.py \
        BlueRingFull14_level9_natural_patra_join \
        zelda_i.level9.natural_path:make_natural_patra_join_controller \
        --prefix L9Join14 --phases CLEAR_20,CLEAR_41
    ... --screens 0x61,0x10 --prefix L9Arrows14
"""
import argparse
import collections
import importlib

from retro_harness.env import make_env, read_state_bytes, save_state, state_path
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import configure_headless
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import PLAY_MODE, read_snapshot
from zelda_i.route.chain import bind_controller_env

ap = argparse.ArgumentParser()
ap.add_argument("state")
ap.add_argument("target")
ap.add_argument("--prefix", required=True)
ap.add_argument("--phases", default="")
ap.add_argument("--screens", default="")
ap.add_argument("--idle", type=int, default=0)
ap.add_argument("--frames", type=int, default=30000)
a = ap.parse_args()
phases = {p for p in a.phases.split(",") if p}
screens = {int(s, 0) for s in a.screens.split(",") if s}

configure_headless()
env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
env.reset()
env.em.set_state(read_state_bytes(state_path(GAME_DIR, GAME, a.state)))
for _ in range(a.idle):
    env.step(nes_idle_action())
mod, _, name = a.target.partition(":")
ctl = getattr(importlib.import_module(mod), name)()
bind_controller_env(ctl, env)
assist = UnlimitedHealthAssist()
frames = collections.Counter()
dmg = collections.Counter()
saved = set()
for f in range(1, a.frames + 1):
    snap = read_snapshot(env.get_ram())
    phase = getattr(getattr(ctl, "phase", None), "name", None)
    key = phase if phases else f"0x{snap.screen:02x}"
    todo = phase if phase in phases else (
        f"0x{snap.screen:02x}" if snap.screen in screens and snap.mode == PLAY_MODE
        and not snap.transitioning else None
    )
    if todo and todo not in saved:
        saved.add(todo)
        print("saved", save_state(env, GAME_DIR, GAME, f"{a.prefix}_{todo}"), f"f{f}",
              f"({snap.link_x},{snap.link_y}) 0x{snap.screen:02x}", flush=True)
    before = assist.telemetry.total_damage
    act = ctl.step(snap)
    env.step(act.action)
    assist.apply_env(env, frame=f)
    frames[key] += 1
    dmg[key] += assist.telemetry.total_damage - before
    if getattr(ctl, "success", False) or getattr(ctl, "failed", False):
        break
print(f"success={getattr(ctl, 'success', None)} failed={getattr(ctl, 'failed', None)} frames={f}")
for k, n in frames.items():
    print(f"  {k:14s} {n:6d}f  dmg={dmg[k]}")
