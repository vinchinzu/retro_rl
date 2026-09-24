"""Replay one stage controller from a save point, without the spine around it.

A spine resume replays every later stage too; fixing one room wants that
room alone, in seconds, with the frames that matter printed::

    uv run python nes/zelda_i/scripts/stage_replay.py BlueRingFull3_level6_clear_0x29 \\
        zelda_i.level6.dungeon:ROOM_29_SPEC --frames 15000
    uv run python nes/zelda_i/scripts/stage_replay.py BlueRingFull5_collect_tf \\
        zelda_i.level2.tf_spine:Level2TfCollectController --frames 4000
    uv run python nes/zelda_i/scripts/stage_replay.py BlueRingFull3_level6_clear_0x29 \\
        zelda_i.level6.dungeon:ROOM_29_SPEC --window 14960-14964 --save-end /tmp/end.state

``TARGET`` is ``module:NAME``: a ``DungeonRoomSpec`` (run by the generic
engine) or a zero-argument controller factory/class. ``--set ADDR=VAL`` is a what-if RAM write at load (a measurement,
never a route result). ``--idle N`` plays N idle frames first (an RNG offset: score a
combat change over several). ``--assist`` adds the Survival refill and reports the
damage it absorbed; ``--last-heart`` the guarded last-heart one (replay a
last-heart stall with it: the full refill keeps the beam firing). ``--window A-B`` prints
each frame's pose, press, reason, the pre-filter press, the stepladder and
the live bodies. ``--save-end`` writes the end state for ``pin_probe.py``.
Same runner as the spine (``route.chain.run_controller_stage``), no assist.
"""

from __future__ import annotations

import argparse
import collections
import importlib
import sys
import time
from pathlib import Path

from retro_harness.env import make_env, read_state_bytes, state_path
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import configure_headless
from zelda_i.assist import LastHeartAssist, UnlimitedHealthAssist
from zelda_i.dungeon.engine import DungeonRoomSpec, GenericDungeonRoomController
from zelda_i.dungeon.ids import STEPLADDER_OBJECT_TYPE
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot
from zelda_i.route.chain import run_controller_stage

_BUTTONS = ("B", "", "s", "S", "U", "D", "L", "R", "A")


def _pressed(action) -> str:
    return "".join(n for n, b in zip(_BUTTONS, action) if b) or "-"


def build(target: str):
    module, _, name = target.partition(":")
    obj = getattr(importlib.import_module(module), name)
    if isinstance(obj, DungeonRoomSpec):
        return GenericDungeonRoomController(spec=obj), obj
    return obj(), None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("state", help="save-state name or a .state path")
    parser.add_argument("target", help="module:SPEC_OR_FACTORY")
    parser.add_argument("--frames", type=int, default=None, help="default: controller max_frames")
    parser.add_argument("--window", default=None, metavar="A-B", help="print frames A..B")
    parser.add_argument("--save-end", default=None, help="write the end state here")
    parser.add_argument(
        "--assist", action="store_true", help="Survival health refill, as the spine plays"
    )
    parser.add_argument(
        "--last-heart", action="store_true",
        help="the guarded last-heart refill (--engage-hearts 1 --observed-damage-guard)",
    )
    parser.add_argument(
        "--idle", type=int, default=0, help="idle frames first: an RNG offset for combat evals"
    )
    parser.add_argument(
        "--set", action="append", default=[], metavar="ADDR=VAL",
        help="what-if RAM write at load (e.g. 0x0676=1 Magical Shield); never a route result",
    )
    args = parser.parse_args(argv)

    configure_headless()
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    env.reset()
    path = Path(args.state)
    if not (path.suffix == ".state" and path.exists()):
        path = state_path(GAME_DIR, GAME, args.state)
    env.em.set_state(read_state_bytes(path))
    for item in args.set:
        addr, _, value = item.partition("=")
        env.unwrapped.data.memory.assign(int(addr, 0), "|u1", int(value, 0) & 0xFF)
    for _ in range(max(0, args.idle)):
        env.step(nes_idle_action())
    ctl, spec = build(args.target)
    frames = args.frames or int(getattr(ctl, "max_frames", 0) or 6000)
    lo, hi = (int(v) for v in args.window.split("-")) if args.window else (1, 0)

    reasons: collections.Counter[str] = collections.Counter()
    seen = {"frame": 0, "raw": None}
    guard = getattr(ctl, "_ladder_guard", None)
    if guard is not None:  # the engine's pre-filter press, for the window
        def spy(snap, action, _guard=guard):
            seen["raw"] = action
            return _guard(snap, action)

        ctl._ladder_guard = spy
    inner_step = ctl.step

    def step(snap, *a, **kw):
        seen["frame"] += 1
        act = inner_step(snap, *a, **kw)
        reasons[act.reason] += 1
        f = seen["frame"]
        if lo <= f <= hi:
            live = spec.live_enemies(snap) if spec is not None else ()
            ladder = [(o.x, o.y, hex(o.facing)) for o in snap.objects if o.type_id == STEPLADDER_OBJECT_TYPE]
            raw = seen["raw"]
            print(
                f"f{f} ({snap.link_x},{snap.link_y}) {_pressed(act.action)} {act.reason}"
                + (f" raw={_pressed(raw.action)}:{raw.reason}" if raw is not None else "")
                + (f" ladder={ladder}" if ladder else "")
                + (f" live={[(hex(o.type_id), o.x, o.y) for o in live]}" if live else "")
            )
        return act

    ctl.step = step
    before = read_snapshot(env.get_ram())
    start = time.time()
    assist = (
        LastHeartAssist(observed_damage_guard=True) if args.last_heart
        else UnlimitedHealthAssist() if args.assist else None
    )
    _, result = run_controller_stage(
        env, None, name="replay", controller=ctl, max_frames=frames, assist=assist
    )
    end = read_snapshot(env.get_ram())
    print(
        f"frames={result.frames}/{frames} success={result.success} {time.time() - start:.1f}s "
        f"L{end.level}:0x{end.screen:02x} ({end.link_x},{end.link_y}) mode={end.mode} "
        f"containers {before.heart_containers}->{end.heart_containers} "
        f"tf 0x{before.triforce:02x}->0x{end.triforce:02x} keys {before.keys}->{end.keys}"
    )
    print(
        f"damage {sum(result.damage_by_room.values()):.2f}h"
        + (f" (assist hits {assist.report().get('damage_events')})" if assist else "")
        + (
            f" refills {assist.report().get('target_refills')}+{assist.report().get('safety_refills')} safety"
            if args.last_heart else ""
        )
    )
    print("reasons:", " ".join(f"{r}={n}" for r, n in reasons.most_common(14)))
    report = ctl.report() if hasattr(ctl, "report") else {}
    if report.get("notes"):
        print("notes:", report["notes"][-12:])
    damage = report.get("damage")
    if isinstance(damage, dict) and damage.get("hits_by_cause"):
        print("hits by cause:", damage["hits_by_cause"])
    if args.save_end:
        Path(args.save_end).write_bytes(env.em.get_state())
        print(f"saved {args.save_end}")
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
