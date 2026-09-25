"""Offset copies of one save point for resume evals. Scratch.

    uv run python nes/zelda_i/scratch/offset_pins.py CL64_enter_level3 L3o enter_level3 8

writes ``L3o<n>_enter_level3`` after ``n`` idle frames, for ``run_survival_spine.py
--resume enter_level3 --save-points L3o<n>`` (one RNG offset per prefix).
"""
import sys

from retro_harness.env import save_state
from retro_harness.nes import nes_idle_action
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.runner import open_env

state, prefix, stage, count = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
env = open_env(from_state=state)
base = env.em.get_state()
for n in range(count):
    env.em.set_state(base)
    for _ in range(n):
        env.step(nes_idle_action())
    print(save_state(env, GAME_DIR, GAME, f"{prefix}{n}_{stage}"))
