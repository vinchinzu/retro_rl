"""Run Level 9 one hop at a time from a pin, Clean by default. Scratch.

The spine runs the whole Silver Arrows prefix as one stage, so a save point
exists only at 0x76 and the stage report cannot say which hop spent the
hearts. This plays the same controllers hop by hop (entry chapter, the 17
prefix hops, the Patra join, the credits chapter), prints hearts in/out and
damage by room per hop, and with ``--pins P`` writes ``P_<step>`` at every
hop start for ``--from``.
The dungeon shot guard defaults on, matching the spine; ``--no-guard``
selects the unguarded baseline. ``--guard`` remains accepted.

    QT_QPA_PLATFORM=offscreen uv run python nes/zelda_i/scratch/l9_probe.py \
        C11Evalo0_level9_post_l8_overworld --pins L9P0 --out /tmp/l9p0.json
    ... l9_probe.py L9P0_s09_level9_stairs_05 --from s09 --idle 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from retro_harness.env import make_env, read_state_bytes, state_path, write_state_bytes
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import configure_headless
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon.shot_guard import GuardedController, ShotGuard
from zelda_i.level9.dungeon import MEASURED_POST_L8_HANDOFF
from zelda_i.level9.hops import level9_credits_chapter, level9_entry_chapter
from zelda_i.level9.natural_path import (
    NaturalSilverArrowsController,
    make_natural_patra_join_controller,
)
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import hearts_held, read_snapshot
from zelda_i.rollout import PolicyGuard
from zelda_i.route.chain import run_controller_stage


def steps(*, guard=True, red_ring=False):
    out = []
    for name, ctl, cap in level9_entry_chapter(handoff=MEASURED_POST_L8_HANDOFF):
        out.append((name, ctl, cap))
    silver = NaturalSilverArrowsController(handoff=MEASURED_POST_L8_HANDOFF, red_ring=red_ring)
    # Keep prefix hysteresis/history across hops, as the chapter wrapper does.
    prefix_guard = ShotGuard()
    for hop in silver._hops:
        name = getattr(hop, "spec_id", type(hop).__name__)
        out.append((name, GuardedController(hop, prefix_guard) if guard else hop, int(hop.max_frames)))
    join = make_natural_patra_join_controller()
    join.start_checked = True
    out.append((
        "level9_natural_patra_join",
        PolicyGuard(GuardedController(join) if guard else join),
        join.max_frames,
    ))
    for name, ctl, cap in level9_credits_chapter():
        if not guard and isinstance(ctl, GuardedController):
            ctl = ctl.inner
        elif not guard and isinstance(ctl, PolicyGuard) and isinstance(ctl.inner, GuardedController):
            ctl = PolicyGuard(ctl.inner.inner)
        out.append((name, ctl, cap))
    # Keep the s00-s33 pins valid: the 0x16 bomb-item kill and the Red Ring
    # detour follow s14 (East15) under their own stable labels.
    aliases = {"level9_patra_16": "s14b", "level9_red_ring": "s14r"}
    labelled = []
    i = 0
    for name, ctl, cap in out:
        sid = aliases.get(name)
        if sid is None:
            sid = f"s{i:02d}"
            i += 1
        labelled.append((f"{sid}_{name}", ctl, cap))
    return labelled


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("state")
    ap.add_argument("--from", dest="start", default="s00", help="step id prefix, e.g. s09")
    ap.add_argument("--to", default=None, help="stop after this step id prefix")
    ap.add_argument("--idle", type=int, default=0)
    ap.add_argument("--pins", default=None, help="write <P>_<step> at every step start")
    ap.add_argument("--assist", action="store_true", help="Survival refill (measure damage)")
    ap.add_argument("--set", action="append", default=[], metavar="ADDR=VAL", help="what-if write at load")
    ap.add_argument("--out", default=None)
    ap.add_argument("--red-ring", action="store_true",
                    help="insert the Red Ring detour (s14r) after the 0x16 kill")
    ap.add_argument("--guard", action=argparse.BooleanOptionalAction, default=True,
                    help="Level 9 dungeon shot guard (default on, as in the spine)")
    a = ap.parse_args()

    configure_headless()
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    env.reset()
    p = Path(a.state)
    if not (p.suffix == ".state" and p.exists()):
        p = state_path(GAME_DIR, GAME, a.state)
    env.em.set_state(read_state_bytes(p))
    for w in a.set:
        addr, _, val = w.partition("=")
        env.unwrapped.data.memory.assign(int(addr, 0), "|u1", int(val, 0) & 0xFF)
    for _ in range(a.idle):
        env.step(nes_idle_action())
    assist = UnlimitedHealthAssist() if a.assist else None
    obs = None
    rows = []
    started = False
    frame = 0
    for sid, ctl, cap in steps(guard=a.guard, red_ring=a.red_ring):
        if not started:
            if not sid.startswith(a.start):
                continue
            started = True
        if a.pins:
            write_state_bytes(
                state_path(GAME_DIR, GAME, f"{a.pins}_{sid}"), env.em.get_state()
            )
        prior_overrides = ctl.guard.overrides if isinstance(ctl, GuardedController) else 0
        obs, res = run_controller_stage(
            env, obs, name=sid, controller=ctl, max_frames=cap, assist=assist, frame_base=frame
        )
        frame = res.end_frame
        snap = read_snapshot(env.get_ram())
        rep = res.report()
        hits = rep.get("hits") or {}
        row = {
            "step": sid,
            "ok": bool(res.success),
            "frames": res.frames,
            "hearts_in": rep.get("hearts", {}).get("in"),
            "hearts_out": round(hearts_held(snap), 2),
            "damage": rep.get("hearts", {}).get("damage"),
            "healed": rep.get("hearts", {}).get("healed"),
            "rooms": rep.get("hearts", {}).get("damage_by_room"),
            "bombs": int(snap.bombs),
            "ring": int(snap.ring),
            "rupees": int(snap.rupees),
            "end": f"L{snap.level}:0x{snap.screen:02x} ({snap.link_x},{snap.link_y}) m{snap.mode}",
            "by_cause": hits.get("hits_by_cause") if isinstance(hits, dict) else None,
        }
        c0 = rep.get("controller") or {}
        if isinstance(c0, dict) and c0.get("shot_guard"):
            row["guard"] = c0["shot_guard"]["overrides"] - prior_overrides
        if not res.success:
            c = rep.get("controller") or {}
            row["notes"] = (c.get("notes") or [])[-3:] if isinstance(c, dict) else None
        rows.append(row)
        print(json.dumps(row), flush=True)
        if not res.success or snap.mode == 17:
            break
        if a.to and sid.startswith(a.to):
            break
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
