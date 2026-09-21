"""A/B the two evaders from one pin: the straight line vs. the ROM.

``dungeon.threat.ReactiveEvader`` decides a dodge by extending a measured
velocity; ``rollout.RolloutEvader`` decides it by asking the ROM. Both live in
the tree, both sit on ``path``'s threat ladder, and one flag picks which.
This probe runs the same walk twice -- once per arm -- and prints the only
numbers that can price the difference.

    QT_QPA_PLATFORM=offscreen uv run python nes/zelda_i/scratch/probe_evader_ab.py --arm reactive --tag ab_reactive
    QT_QPA_PLATFORM=offscreen uv run python nes/zelda_i/scratch/probe_evader_ab.py --arm rollout  --tag ab_rollout
    uv run python nes/zelda_i/scratch/probe_evader_ab.py --compare ab_reactive ab_rollout

One arm per process, on purpose: **one emulator per process**
(``AGENTS.md``), and ``run_survival_spine`` opens its own. ``--compare`` reads
the two JSON reports back and touches no emulator at all.

**What is a measurement here and what is not.** The emulator is
deterministic: a config has exactly one outcome, so a single run *is* the
measurement and a rupee count is not one -- money is a function of which
drops the walk happened to stand on. The two things that price a dodge are

* **hits by cause**, per screen and per stage. Damage is read off
  ``combat.heart_value`` (1/256 of a heart, so a wooden chip is 128 and not
  zero) and every ``$04F0`` arming is named by the body that was nearest one
  frame *before* the knockback moved Link away from it (the
  ``probe_contact.py`` ring-buffer rule).
* **the reason and rung censuses**, which say where the frames went. A dodge
  that avoids a hit by standing on a screen for 4000 frames has not won.

Both are printed per arm and diffed by ``--compare``. Rupees are carried in
``final`` for the record and are deliberately absent from the comparison
table.

The env is **not** wrapped in ``AuditedEnv`` here. A rollout is a
``set_state`` and the audit would count it as a mid-run load, which is true
but useless -- the walk's own tape is bit-identical either side of a fan. The
honest ledger for it is ``controller.report()["rollout"]``
(``rollouts`` / ``frames_rolled`` / ``replans``), which this probe prints per
stage and totals per arm. This is a measurement probe, not a STATUS claim.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from retro_harness.env import make_env, reset_obs
from retro_harness.segment_runner import configure_headless, write_json_report
from zelda_i.combat import chebyshev, heart_value
from zelda_i.dungeon.ids import OBJECT_NAMES
from zelda_i.overworld.path import OverworldPathController
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import PLAY_MODE, read_snapshot
from zelda_i.spine.survival import run_survival_spine, spine_final_fields

OUT_DIR = Path(__file__).resolve().parent
ARMS = ("reactive", "rollout")
# Keys pulled out of each stage's controller report. These are the ones that
# can price a behaviour; everything else is in the JSON.
_STAGE_KEYS = (
    "reason_by_screen",
    "rung_census",
    "rung_frames",
    "threat_census",
    "threat_frames",
    "evade_reasons",
    "evades",
    "parries",
    "spit_ducks",
    "stall_escapes",
    "rollout",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, default="reactive")
    parser.add_argument("--through", default="pre-l1")
    parser.add_argument("--tag", default=None)
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("A", "B"),
        default=None,
        help="two tags (or paths) written by earlier runs; prints the diff",
    )
    # Budget knobs, forwarded to ``RolloutEvader.__init__``. They have to be
    # constructor arguments: a ``@dataclass`` copies its defaults into
    # ``__init__`` when the class is created, so writing them onto the class
    # afterwards is a no-op on every instance and silently measures the
    # unablated arm (``AGENTS.md``; every ablation before 2026-09-16).
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--replan-frames", type=int, default=None)
    parser.add_argument("--trigger-radius", type=int, default=None)
    parser.add_argument("--min-gain", type=int, default=None)
    parser.add_argument("--screen-budget", type=int, default=None)
    args = parser.parse_args(argv)

    if args.compare:
        return _compare(*(_load(t) for t in args.compare))

    knobs = {
        k: v
        for k, v in {
            "frames": args.horizon,
            "replan_frames": args.replan_frames,
            "trigger_radius": args.trigger_radius,
            "min_gain": args.min_gain,
            "screen_budget": args.screen_budget,
        }.items()
        if v is not None
    }
    if knobs and args.arm != "rollout":
        parser.error("budget knobs only mean anything on --arm rollout")

    configure_headless()
    tag = args.tag or f"evader_ab_{args.arm}"
    payload = _run(args.arm, args.through, knobs)
    payload["tag"] = tag
    out = OUT_DIR / f"{tag}.json"
    write_json_report(out, payload)
    write_json_report(RECORDINGS_DIR / f"{tag}.json", payload)
    _print_arm(payload)
    print(f"wrote {out}")
    return 0 if payload["ok"] else 1


# --------------------------------------------------------------- one arm ---

def _run(arm: str, through: str, knobs: dict[str, int]) -> dict:
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")

    if arm == "rollout":
        # The seam is ``step``, not ``__init__``. Every walk the spine builds
        # is a ``@dataclass`` *subclass* of this controller
        # (``ShopP7WalkController``), and a dataclass subclass generates its
        # own ``__init__`` -- so patching the base constructor the way
        # ``probe_contact.py`` patches ``ScreenHunter.__init__`` would attach
        # the arm to nothing and silently measure the reactive one. That is
        # the ``AGENTS.md`` dataclass trap in a new dress. ``step`` is an
        # ordinary inherited method, so this reaches every subclass, and
        # ``attach_rollout`` needs the *live* env, which is why it is a
        # closure rather than a flag.
        _orig_step = OverworldPathController.step

        def _step(self, snap, **kw):  # type: ignore[no-untyped-def]
            if self._rollout is None:
                self.attach_rollout(env, **knobs)
            return _orig_step(self, snap, **kw)

        OverworldPathController.step = _step  # type: ignore[assignment]

    damage = {"units": 0, "prev_hp": None}
    prev = {"iframes": -1, "near": None}
    contacts: list[dict] = []

    def on_frame(env_, _obs, _action, frame: int) -> None:
        snap = read_snapshot(env_.get_ram())
        if int(snap.level) != 0 or int(snap.mode) not in (PLAY_MODE, 17):
            return
        hp = heart_value(snap)
        if damage["prev_hp"] is not None and hp < damage["prev_hp"]:
            # 1/256 of a heart: a wooden chip is 128, and a report in whole
            # hearts cannot see it at all.
            damage["units"] += damage["prev_hp"] - hp
        damage["prev_hp"] = hp
        iframes = int(snap.link_iframes)
        if prev["iframes"] == 0 and iframes > 0:
            # The collider is the body that was nearest one frame BEFORE the
            # knockback moved Link away from it.
            contacts.append(
                {
                    "frame": frame,
                    "screen": f"{int(snap.screen):#04x}",
                    "cause": prev["near"] or "unknown",
                    "xy": [int(snap.link_x), int(snap.link_y)],
                }
            )
        prev["iframes"] = iframes
        # Keep the pre-contact object for the ring-buffer attribution. A
        # fatal collision enters mode 17 on this same post-step callback; its
        # sprites have already moved/started despawning, so the last playable
        # frame is the cause, not the death frame's fresh census.
        if int(snap.mode) == PLAY_MODE:
            prev["near"] = _nearest_name(snap)

    try:
        obs, _ = reset_obs(env)
        run = run_survival_spine(
            env, obs, assist=None, on_frame=on_frame, through=through,
            allow_pokes=False,
        )
        report = run.report()
        final = spine_final_fields(read_snapshot(env.get_ram()), env.get_ram())
    finally:
        env.close()

    stages = [_stage(s) for s in report.get("stages", [])]
    return {
        "arm": arm,
        "through": through,
        "knobs": knobs,
        "ok": bool(report.get("ok")),
        "failed_stage": report.get("failed_stage"),
        "frames": report.get("end_frame"),
        "final": final,
        "damage_units": damage["units"],
        "hearts_lost": round(damage["units"] / 256.0, 3),
        "hits": len(contacts),
        "hits_by_cause": _tally(c["cause"] for c in contacts),
        "hits_by_screen": _nested(contacts, "screen", "cause"),
        "hits_by_stage": _by_stage(contacts, stages),
        "contacts": contacts,
        "rollout": _total_rollout(stages),
        "stages": stages,
    }


def _stage(stage: dict) -> dict:
    ctl = stage.get("controller") or {}
    out = {
        "name": stage.get("name"),
        "frames": stage.get("frames"),
        "success": stage.get("success"),
        "frame_base": stage.get("frame_base"),
        "end_frame": stage.get("end_frame"),
    }
    for key in _STAGE_KEYS:
        if key in ctl:
            out[key] = ctl[key]
    hunt = ctl.get("hunt")
    if isinstance(hunt, dict):
        out["hunt"] = {
            k: hunt.get(k)
            for k in (
                "kills", "damage_taken", "damage_units", "hurt_events",
                "hits_by_cause", "screens", "streak_best", "streak_resets",
                "hunt_frames_by_screen",
            )
        }
    return out


def _nearest_name(snap) -> str | None:
    """Name of the closest live slot. Nothing is filtered on ``hp``.

    ``AGENTS.md``: an octorok rock is slot 11 with hp 0, and a census
    filtered on ``hp > 0`` cannot see the thing that is hitting Link.
    """
    best = None
    best_gap = 10**9
    lx, ly = int(snap.link_x), int(snap.link_y)
    for obj in snap.objects:
        if int(obj.slot) < 1 or int(obj.type_id) in (0, 0xFF, 0x64):
            continue
        if int(obj.hp) >= 200:  # a door / wall slot, not a body
            continue
        gap = chebyshev(lx, ly, int(obj.x), int(obj.y))
        if gap < best_gap:
            best_gap = gap
            best = OBJECT_NAMES.get(int(obj.type_id), f"unk_{int(obj.type_id):#04x}")
    return best


def _tally(values) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for value in values:
        out[value] += 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _nested(rows: list[dict], outer: str, inner: str) -> dict:
    out: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        out[row[outer]][row[inner]] += 1
    return {k: dict(sorted(v.items(), key=lambda kv: -kv[1])) for k, v in out.items()}


def _by_stage(contacts: list[dict], stages: list[dict]) -> dict:
    """Hits by cause, bucketed into the stage that owned the frame.

    Per project memory: read per-stage hits-by-cause before tuning a red
    room — the health is usually spent in the stages that *pass*.
    """
    out: dict[str, dict[str, int]] = {}
    for stage in stages:
        lo, hi = stage.get("frame_base"), stage.get("end_frame")
        if lo is None or hi is None:
            continue
        rows = [c for c in contacts if lo <= c["frame"] <= hi]
        if rows:
            out[str(stage.get("name"))] = _tally(c["cause"] for c in rows)
    return out


def _total_rollout(stages: list[dict]) -> dict:
    """The ledger a report must not hide, summed over every stage."""
    total = {"rollouts": 0, "frames_rolled": 0, "replans": 0, "claims": 0,
             "held_frames": 0}
    declines: dict[str, int] = defaultdict(int)
    budget = None
    for stage in stages:
        row = stage.get("rollout")
        if not isinstance(row, dict):
            continue
        for key in total:
            total[key] += int(row.get(key, 0))
        for name, count in (row.get("declines") or {}).items():
            declines[name] += int(count)
        budget = row.get("budget", budget)
    total["declines"] = dict(sorted(declines.items(), key=lambda kv: -kv[1]))
    total["budget"] = budget
    return total


# ------------------------------------------------------------- reporting ---

def _print_arm(p: dict) -> None:
    print(
        f"arm={p['arm']} ok={p['ok']} failed={p['failed_stage']} "
        f"frames={p['frames']} knobs={p['knobs']}"
    )
    print(
        f"  damage_units={p['damage_units']} ({p['hearts_lost']} hearts) "
        f"hits={p['hits']}"
    )
    print(f"  hits_by_cause:  {p['hits_by_cause']}")
    print(f"  hits_by_screen: {p['hits_by_screen']}")
    print(f"  hits_by_stage:  {p['hits_by_stage']}")
    roll = p["rollout"]
    if roll["rollouts"]:
        print(
            f"  rollout: replans={roll['replans']} rollouts={roll['rollouts']} "
            f"frames_rolled={roll['frames_rolled']} claims={roll['claims']} "
            f"held={roll['held_frames']}"
        )
        print(f"    declines={roll['declines']}")
        print(f"    budget={roll['budget']}")
    for stage in p["stages"]:
        print(f"  STAGE {stage['name']} {stage['frames']}f")
        if stage.get("threat_census"):
            print(f"    threat: {stage['threat_census']}")
        if stage.get("rung_census"):
            top = sorted(stage["rung_census"].items(), key=lambda kv: -kv[1])[:8]
            print(f"    rungs:  {top}")
        if stage.get("evade_reasons"):
            print(f"    evade:  {stage['evade_reasons']}")
        for screen, rows in (stage.get("reason_by_screen") or {}).items():
            top = sorted(rows.items(), key=lambda kv: -kv[1])[:6]
            print(f"    {screen}: {top}")


def _load(tag: str) -> dict:
    path = Path(tag)
    if not path.is_file():
        path = OUT_DIR / f"{tag}.json"
    return json.loads(path.read_text())


def _compare(a: dict, b: dict) -> int:
    """Side by side. Rupees are absent on purpose -- see the module docstring."""
    print(f"{'':<22}{a['arm']:>14}{b['arm']:>14}   delta")
    for label, key in (
        ("ok", "ok"),
        ("frames", "frames"),
        ("damage_units", "damage_units"),
        ("hits", "hits"),
    ):
        left, right = a.get(key), b.get(key)
        delta = ""
        if isinstance(left, int) and isinstance(right, int) and not (
            isinstance(left, bool) or isinstance(right, bool)
        ):
            delta = f"{right - left:+d}"
        print(f"{label:<22}{str(left):>14}{str(right):>14}   {delta}")
    print("\nhits by cause")
    for name in sorted(set(a["hits_by_cause"]) | set(b["hits_by_cause"])):
        left = a["hits_by_cause"].get(name, 0)
        right = b["hits_by_cause"].get(name, 0)
        print(f"{name:<22}{left:>14}{right:>14}   {right - left:+d}")
    print("\nhits by screen")
    for screen in sorted(set(a["hits_by_screen"]) | set(b["hits_by_screen"])):
        left = sum(a["hits_by_screen"].get(screen, {}).values())
        right = sum(b["hits_by_screen"].get(screen, {}).values())
        print(f"{screen:<22}{left:>14}{right:>14}   {right - left:+d}")
    print("\nhits by stage")
    for stage in sorted(set(a["hits_by_stage"]) | set(b["hits_by_stage"])):
        left = sum(a["hits_by_stage"].get(stage, {}).values())
        right = sum(b["hits_by_stage"].get(stage, {}).values())
        print(f"{stage:<22}{left:>14}{right:>14}   {right - left:+d}")
    print("\nframes by stage")
    a_stage = {s["name"]: s["frames"] for s in a["stages"]}
    b_stage = {s["name"]: s["frames"] for s in b["stages"]}
    for stage in sorted(set(a_stage) | set(b_stage)):
        left, right = a_stage.get(stage, 0) or 0, b_stage.get(stage, 0) or 0
        print(f"{stage:<22}{left:>14}{right:>14}   {right - left:+d}")
    print(f"\nrollout ledger  {a['arm']}: {a['rollout']['rollouts']} rollouts / "
          f"{a['rollout']['frames_rolled']} frames")
    print(f"rollout ledger  {b['arm']}: {b['rollout']['rollouts']} rollouts / "
          f"{b['rollout']['frames_rolled']} frames")
    print(
        "\nrupees are not in this table. The emulator is deterministic, so a "
        "config has exactly one outcome and money is a function of which "
        "drops the walk stood on, not of the dodge."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
