"""Scratch probe: can a turn and a swing share one frame?

``hunt._a_edge`` presses ``nes_action(face, "A")`` — the direction Link
*should* face and the A edge on the same frame. Every contact strike where
the body is off Link's current facing therefore depends on the ROM turning
him before the blade comes out. The zhit tapes say it often does not: 11 of
11 ``dir+A`` presses with ``face_before != dir`` left ``$0098`` unchanged.
That population is biased (frames next to a hit), so measure the rule.

One trial: stand Link on 0x77, set his facing with a held direction, then
press ``target + A`` on one frame (``combined``) or spend a frame on
``target`` alone first (``turn_then_a``). Log facing, Link's object state
and the sword slot ($0D) until the animation clears, with the 8 px grid
residual of both coordinates, because a perpendicular turn in this ROM is
gated on alignment.

    QT_QPA_PLATFORM=offscreen uv run python nes/zelda_i/scratch/probe_turn_swing.py --tag turn2
"""

from __future__ import annotations

import argparse
import json

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless
from zelda_i.overworld.sword_cave import SEGMENT_MAX_FRAMES as SWORD_MAX
from zelda_i.overworld.sword_cave import SwordCaveController
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import PLAY_MODE, read_snapshot
from zelda_i.route.chain import boot_to_ready, run_controller_stage

OUT = RECORDINGS_DIR / "scratch_turn_swing"
ADDR_OBJ_STATE = 0x00AC
SWORD_SLOT = 0x0D
LINK_SLOT = 0x00
DIRS = ("RIGHT", "LEFT", "UP", "DOWN")
OPPOSITE = {"RIGHT": "LEFT", "LEFT": "RIGHT", "UP": "DOWN", "DOWN": "UP"}
# Middle of 0x77, clear of the cave mouth and every edge.
HOME = (128, 141)
WATCH = 24


def _sample(env) -> dict:
    ram = env.get_ram()
    snap = read_snapshot(ram)
    return {
        "facing": int(snap.facing),
        "x": int(snap.link_x),
        "y": int(snap.link_y),
        "screen": int(snap.screen),
        "mode": int(snap.mode),
        "link_state": int(ram[ADDR_OBJ_STATE + LINK_SLOT]),
        "sword_state": int(ram[ADDR_OBJ_STATE + SWORD_SLOT]),
    }


def _hold(env, obs, action, frames: int):
    for _ in range(frames):
        obs, *_ = env.step(action)
    return obs


def _recenter(env, obs):
    """Walk back to ``HOME`` so every trial starts from the same tile."""
    for _ in range(400):
        row = _sample(env)
        dx, dy = HOME[0] - row["x"], HOME[1] - row["y"]
        if abs(dx) <= 1 and abs(dy) <= 1:
            break
        if abs(dx) >= abs(dy):
            direction = "RIGHT" if dx > 0 else "LEFT"
        else:
            direction = "DOWN" if dy > 0 else "UP"
        obs, *_ = env.step(nes_action(direction))
    return _hold(env, obs, nes_idle_action(), 6), _sample(env)


def _trial(env, obs, base: str, target: str, *, turn_first: bool, nudge: int):
    """``nudge`` frames of ``base`` past the facing hold, to vary alignment."""
    obs, _ = _recenter(env, obs)
    obs = _hold(env, obs, nes_action(OPPOSITE[base]), 8)
    # No idle gap before the press: the ROM parks a stopped Link on the grid,
    # and a stopped Link is not what the contact rung ever sees.
    obs = _hold(env, obs, nes_action(base), 8 + nudge)
    before = _sample(env)
    after_turn = None
    if turn_first:
        obs, *_ = env.step(nes_action(target))
        after_turn = _sample(env)
    obs, *_ = env.step(nes_action(target, "A"))
    frames = []
    for _ in range(WATCH):
        frames.append(_sample(env))
        obs, *_ = env.step(nes_idle_action())
    busy = sum(1 for f in frames if f["link_state"] != 0)
    return obs, {
        "base": base,
        "target": target,
        "variant": "turn_then_a" if turn_first else "combined",
        "nudge": nudge,
        "x": before["x"],
        "y": before["y"],
        "x_mod8": before["x"] % 8,
        "y_mod8": before["y"] % 8,
        "screen": before["screen"],
        "mode": before["mode"],
        "on_screen": before["mode"] == PLAY_MODE,
        "leaving_axis_mod8": (
            before["y"] % 8 if base in ("UP", "DOWN") else before["x"] % 8
        ),
        "facing_before": before["facing"],
        "facing_after_turn": None if after_turn is None else after_turn["facing"],
        "facing_after_press": frames[0]["facing"],
        "turned": frames[0]["facing"] != before["facing"],
        "swung": any(f["sword_state"] != 0 for f in frames),
        "busy_frames": busy,
    }


def _turn_cost(env, obs, base: str, target: str, *, nudge: int):
    """Hold ``target`` from a *walking* Link: how many frames to the turn?

    This is the bill the fix pays. A contact strike that waits for the facing
    spends these frames walking instead of pinned, and only then swings.
    """
    obs, _ = _recenter(env, obs)
    obs = _hold(env, obs, nes_action(OPPOSITE[base]), 8)
    obs = _hold(env, obs, nes_action(base), 8 + nudge)
    before = _sample(env)
    rows = []
    turned_at = None
    for i in range(12):
        obs, *_ = env.step(nes_action(target))
        row = _sample(env)
        rows.append(row)
        if turned_at is None and row["facing"] != before["facing"]:
            turned_at = i + 1
    return obs, {
        "phase": "turn_cost",
        "base": base,
        "target": target,
        "nudge": nudge,
        "mode": before["mode"],
        "screen": before["screen"],
        "x": before["x"],
        "y": before["y"],
        "leaving_axis_mod8": (
            before["y"] % 8 if base in ("UP", "DOWN") else before["x"] % 8
        ),
        "frames_to_turn": turned_at,
        "moved": (rows[-1]["x"] - before["x"], rows[-1]["y"] - before["y"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="turn2")
    args = parser.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    configure_headless()

    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    try:
        obs, _ = reset_obs(env)
        obs, boot = boot_to_ready(env, first_playthrough=True, assist=None)
        sword = SwordCaveController()
        obs, res = run_controller_stage(
            env, obs, name="sword_cave", controller=sword,
            max_frames=SWORD_MAX, assist=None, frame_base=boot,
        )
        if not sword.success:
            raise SystemExit("sword cave failed")
        costs = []
        for base in DIRS:
            for target in DIRS:
                if target in (base, OPPOSITE[base]):
                    continue
                for nudge in range(8):
                    obs, row = _turn_cost(env, obs, base, target, nudge=nudge)
                    costs.append(row)
        trials = []
        for base in DIRS:
            for target in DIRS:
                if target == base or target == OPPOSITE[base]:
                    continue  # a 180 needs no alignment; perpendicular is the question
                for nudge in range(8):
                    for turn_first in (False, True):
                        obs, row = _trial(
                            env, obs, base, target,
                            turn_first=turn_first, nudge=nudge,
                        )
                        trials.append(row)
    finally:
        env.close()

    good = [t for t in trials if t["on_screen"] and t["swung"]]
    summary = {"trials": len(trials), "usable": len(good)}
    for variant in ("combined", "turn_then_a"):
        rows = [t for t in good if t["variant"] == variant]
        summary[variant] = {
            "n": len(rows),
            "turned": sum(int(t["turned"]) for t in rows),
            "busy_max": max((t["busy_frames"] for t in rows), default=0),
        }
        # Alignment is on the axis Link is *leaving*: a vertical facing turns
        # horizontal only on an aligned y, and the other way round.
        # ``ALIGNED`` per axis: the ROM parks a stopped Link on x%8==0,
        # y%8==5, so those residuals are the grid, not an arbitrary choice.
        aligned = [
            t for t in rows
            if t["leaving_axis_mod8"] == (5 if t["base"] in ("UP", "DOWN") else 0)
        ]
        summary[variant]["aligned_n"] = len(aligned)
        summary[variant]["aligned_turned"] = sum(int(t["turned"]) for t in aligned)
        off = [t for t in rows if t not in aligned]  # noqa: PLR6201
        summary[variant]["offgrid_n"] = len(off)
        summary[variant]["offgrid_turned"] = sum(int(t["turned"]) for t in off)
    live = [c for c in costs if c["mode"] == PLAY_MODE]
    summary["turn_cost"] = {
        "n": len(live),
        "never_turned": sum(1 for c in live if c["frames_to_turn"] is None),
        "by_frames": {
            str(k): sum(1 for c in live if c["frames_to_turn"] == k)
            for k in sorted({c["frames_to_turn"] for c in live if c["frames_to_turn"]})
        },
        "aligned_first_frame": sum(
            1 for c in live
            if c["frames_to_turn"] == 1
            and c["leaving_axis_mod8"] == (5 if c["base"] in ("UP", "DOWN") else 0)
        ),
    }
    path = OUT / f"{args.tag}.json"
    path.write_text(json.dumps(
        {"tag": args.tag, "summary": summary, "trials": trials, "turn_cost": costs}
    ))
    print(json.dumps({"tag": args.tag, "summary": summary, "out": str(path)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
