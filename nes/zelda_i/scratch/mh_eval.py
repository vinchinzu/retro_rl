"""L3 Manhandla fight from a ``*_level3_manhandla`` pin: outcome, hearts, hits. Scratch.

    uv run python nes/zelda_i/scratch/mh_eval.py L3o5_level3_manhandla
"""
import collections
import sys

from zelda_i.dungeon.postmortem import DamageLog
from zelda_i.dungeon.tracking import ObjectTracker
from zelda_i.level3.boss_path import Level3BossPathController
from zelda_i.ram import ram_hearts, read_snapshot
from zelda_i.runner import open_env

env = open_env(from_state=sys.argv[1])
log, tracker = DamageLog(), ObjectTracker()
step = env.step


def traced(action):
    out = step(action)
    snap = read_snapshot(env.get_ram())
    log.observe(snap, tracker.observe(snap))
    return out


env.step = traced
hearts_in = ram_hearts(env.get_ram())
ctl = Level3BossPathController(poke_bombs=None, tag="scratch_l3", continuous_mode=True)
total = [0]
fight = ctl.fight_manhandla(env, None, total, max_frames=16000)
snap = read_snapshot(env.get_ram())
causes = collections.Counter(
    f"0x{e.type_id:02x}" if e.type_id is not None else "?" for e in log.hits
)
print(sys.argv[1], "tf", fight.get("tf04"), "err", fight.get("error"), "frames", total[0],
      "hearts", hearts_in, "->", ram_hearts(env.get_ram()), "bombs", snap.bombs, "hits", dict(causes))
