"""Clean gather stages from a save point under an RNG offset. Scratch.

Runs ``spine.survival.gather_stages()`` from ``--start`` through ``--stop``
with ``run_controller_stage`` (potion guard + hit census, no assist), after
``--offset`` idle frames at the pin. One JSON line per run: stages reached,
hearts in/out, hits by cause.

    uv run python nes/zelda_i/scratch/eval_gather_clean.py --state CL61_walk_2c \
        --start walk_2c --stop white --offset 3 --out /tmp/g3.json
"""
import argparse
import json
from pathlib import Path

from retro_harness.nes import nes_idle_action
from zelda_i.ram import read_snapshot
from zelda_i.route.chain import run_controller_stage
from zelda_i.runner import open_env
from zelda_i.spine.survival import gather_stages

ap = argparse.ArgumentParser()
ap.add_argument("--state", required=True)
ap.add_argument("--start", required=True)
ap.add_argument("--stop", required=True)
ap.add_argument("--offset", type=int, default=0)
ap.add_argument("--out")
ap.add_argument("--save-at", help="write Ev_<stage> at that stage's start")
a = ap.parse_args()

stages = gather_stages()
names = [n for n, _, _ in stages]
run = stages[names.index(a.start) : names.index(a.stop) + 1]
env = open_env(from_state=a.state)
obs = None
for _ in range(a.offset):
    obs, *_ = env.step(nes_idle_action())
rows, frames, died = [], 0, None
for name, ctl, limit in run:
    if a.save_at == name:
        from retro_harness.env import save_state
        from zelda_i.paths import GAME, GAME_DIR

        print("saved", save_state(env, GAME_DIR, GAME, f"Ev{a.offset}_{name}"), flush=True)
    obs, res = run_controller_stage(env, obs, name=name, controller=ctl, max_frames=limit, frame_base=frames)
    frames += res.frames
    snap = read_snapshot(env.get_ram())
    rep = res.report()
    hits = rep.get("hits") or {}
    rows.append(
        {
            "stage": name,
            "ok": bool(res.success),
            "frames": res.frames,
            "in": rep.get("hearts", {}).get("in"),
            "out": rep.get("hearts", {}).get("out"),
            "drank": (rep.get("potion") or {}).get("drinks", 0),
            "hits": hits.get("hits_by_cause", {}),
            "labels": [e["label"] for e in hits.get("events", [])],
            "trails": [e.get("trail") for e in hits.get("events", [])],
            "notes": (rep.get("controller") or {}).get("notes", [])[-12:],
            "reasons": (rep.get("controller") or {}).get("reason_by_screen"),
        }
    )
    if snap.mode == 17:
        died = name
        break
    if not res.success:
        break
out = {
    "offset": a.offset,
    "ok": died is None and len(rows) == len(run) and rows[-1]["ok"],
    "last": rows[-1]["stage"],
    "died": died,
    "frames": frames,
    "hearts": rows[-1]["out"],
    "potion": int(read_snapshot(env.get_ram()).potion),
    "rows": rows,
}
if a.out:
    Path(a.out).write_text(json.dumps(out, indent=1))
print(json.dumps({k: out[k] for k in ("offset", "ok", "last", "died", "frames", "hearts", "potion")}), flush=True)
