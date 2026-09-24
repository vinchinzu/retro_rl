"""Score one L9 Patra-join room over RNG offsets. Scratch.

Loads ``<prefix>_<PHASE>`` (cut by ``cut_pins.py``), latches the
join at that phase, plays N idle frames, then runs under the Survival refill
until the join reaches the room's stop phase (Link is in the next room).
One line per offset plus the mean: frames, damage (whole hearts), and the
frames the clear phase itself held.

    QT_QPA_PLATFORM=offscreen uv run python nes/zelda_i/scratch/eval_l9_join_offsets.py \
        CLEAR_20 --offsets 12 --out /tmp/c20_base.json
"""
import argparse
import json
import statistics
from pathlib import Path

from retro_harness.env import make_env, read_state_bytes, state_path
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import configure_headless
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.level9.natural_path import PatraJoinPhase, make_natural_patra_join_controller
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot
from zelda_i.route.chain import bind_controller_env

STOP = {
    "CLEAR_20": "CELLAR_75",
    "CLEAR_41": "CLEAR_31",
    "CLEAR_31": "CLEAR_30",
    "CLEAR_30": "CELLAR_67",
    "CLEAR_04": "CLEAR_03",
    "CLEAR_03": "CELLAR_77",
}

ap = argparse.ArgumentParser()
ap.add_argument("phase")
ap.add_argument("--prefix", default="L9Join14")
ap.add_argument("--offsets", type=int, default=12)
ap.add_argument("--step", type=int, default=7, help="idle frames between offsets")
ap.add_argument("--frames", type=int, default=12000)
ap.add_argument("--out")
ap.add_argument("--events", action="store_true", help="print each engine hit")
ap.add_argument("--tune", default=None, help='JSON CombatTuning overrides, e.g. {"inland_dash": 12}')
a = ap.parse_args()

configure_headless()
env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
env.reset()
raw = read_state_bytes(state_path(GAME_DIR, GAME, f"{a.prefix}_{a.phase}"))
stop = PatraJoinPhase[STOP[a.phase]]
if a.tune:
    from dataclasses import replace
    import zelda_i.level9.natural_path as npath
    over = {k: (tuple(map(tuple, v)) if isinstance(v, list) else v) for k, v in json.loads(a.tune).items()}
    spec = npath.JOIN_CLEAR_SPECS[PatraJoinPhase[a.phase]]
    npath.JOIN_CLEAR_SPECS[PatraJoinPhase[a.phase]] = replace(spec, combat=replace(spec.combat, **over))
    print("tune", over)
rows = []
for k in range(a.offsets):
    env.em.set_state(raw)
    for _ in range(k * a.step):
        env.step(nes_idle_action())
    ctl = make_natural_patra_join_controller()
    ctl.start_checked = True
    ctl._set_phase(PatraJoinPhase[a.phase])
    bind_controller_env(ctl, env)
    assist = UnlimitedHealthAssist()
    clear_frames = 0
    reentries = 0
    last_screen = None
    for f in range(1, a.frames + 1):
        snap = read_snapshot(env.get_ram())
        if ctl.phase is PatraJoinPhase[a.phase]:
            clear_frames += 1
        act = ctl.step(snap)
        env.step(act.action)
        assist.apply_env(env, frame=f)
        if ctl.phase is stop or ctl.success or ctl.failed:
            break
    end = read_snapshot(env.get_ram())
    row = {
        "offset": k * a.step,
        "ok": ctl.phase is stop,
        "frames": f,
        "clear_frames": clear_frames,
        "damage": assist.telemetry.total_damage,
        "phase": ctl.phase.name,
        "end": f"0x{end.screen:02x} ({end.link_x},{end.link_y})",
        "re10": ctl.reentries_10,
        "re31": ctl.reentries_31,
    }
    fight = getattr(ctl, "_fights", {}).get(PatraJoinPhase[a.phase])
    if fight is not None and fight._ctl is not None:
        rep = fight._ctl.report()
        row["hits"] = rep["damage"].get("hits_by_cause")
        row["top"] = dict(list(rep["reason_counts"].items())[:5])
        if a.events:
            for e in rep["damage"]["events"]:
                print("   hit", {k: e.get(k) for k in ("frame", "link_xy", "phase", "action", "type_id", "cause_xy", "cause_v", "bearing", "distance")})
    rows.append(row)
    print(json.dumps(row), flush=True)
ok = [r for r in rows if r["ok"]]
summary = {
    "phase": a.phase,
    "ok": f"{len(ok)}/{len(rows)}",
    "mean_frames": round(statistics.mean(r["frames"] for r in rows)),
    "mean_clear_frames": round(statistics.mean(r["clear_frames"] for r in rows)),
    "mean_damage": round(statistics.mean(r["damage"] for r in rows), 2),
    "max_damage": max(r["damage"] for r in rows),
}
print("SUMMARY", json.dumps(summary))
if a.out:
    Path(a.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
