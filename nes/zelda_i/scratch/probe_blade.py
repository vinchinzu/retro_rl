"""Which swings land: a ledger of every A press on the pre-L1 walk.

``in_sword_hitbox`` is the model the whole overworld ladder decides with —
``SWORD_REACH`` 20, ``SWORD_HALF_WIDTH`` 12, no minimum — and it has never
been measured against the ROM. The contact windows say it is wrong at the
near end: ``zhit6`` f=4981 swings DOWN twice at a leever 4-5 px below and the
leever walks in anyway, f=6633 swings at one **1 px** away and takes the hit
through ``slash_recover``.

So ledger it. Every play frame records Link, his facing, whether A went out
and every live body; afterwards each *A edge* is matched against the hp drops
in the next ``DAMAGE_WINDOW`` frames, and the body's offset at the press is
written in Link's own frame: ``fwd`` along the facing, ``lat`` across it.

    QT_QPA_PLATFORM=offscreen uv run python nes/zelda_i/scratch/probe_blade.py --tag blade1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from retro_harness.controls import NES_BUTTON_NAME_TO_INDEX
from retro_harness.env import make_env, reset_obs
from retro_harness.segment_runner import configure_headless
from zelda_i.overworld.path import OverworldPathController
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import PLAY_MODE, read_snapshot
from zelda_i.spine.survival import run_survival_spine

OUT_DIR = Path(__file__).resolve().parent
# Sword damage is applied while the blade is out; the animation is 13 frames
# (``probe_turn_swing``), so a drop later than this is someone else's.
DAMAGE_WINDOW = 16
_FACE = {0x08: "UP", 0x04: "DOWN", 0x01: "RIGHT", 0x02: "LEFT"}


def _forward_lateral(face: str, dx: int, dy: int) -> tuple[int, int]:
    """Body offset in Link's frame: ``fwd`` along the facing, ``lat`` across."""
    if face == "RIGHT":
        return dx, dy
    if face == "LEFT":
        return -dx, -dy
    if face == "DOWN":
        return dy, dx
    return -dy, -dx  # UP


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="blade")
    args = parser.parse_args(argv)
    configure_headless()

    a_index = NES_BUTTON_NAME_TO_INDEX["A"]
    pressed = {"a": False, "reason": None}
    _orig_step = OverworldPathController.step

    def _step(self, snap):  # type: ignore[no-untyped-def]
        act = _orig_step(self, snap)
        raw = list(getattr(act, "action", ()) or ())
        pressed["a"] = bool(a_index is not None and a_index < len(raw) and raw[a_index])
        pressed["reason"] = getattr(act, "reason", None)
        return act

    OverworldPathController.step = _step  # type: ignore[assignment]

    frames: list[dict] = []

    def on_frame(env, _obs, _action, frame: int) -> None:
        snap = read_snapshot(env.get_ram())
        if int(snap.level) != 0 or int(snap.mode) != PLAY_MODE:
            return
        frames.append(
            {
                "f": frame,
                "scr": int(snap.screen),
                "x": int(snap.link_x),
                "y": int(snap.link_y),
                "face": _FACE.get(int(snap.facing)),
                "a": pressed["a"],
                "why": pressed["reason"],
                "bodies": [
                    [int(o.slot), int(o.type_id), int(o.hp), int(o.x), int(o.y)]
                    for o in snap.objects
                    if int(o.slot) >= 1
                    and int(o.type_id) not in (0, 0xFF, 0x64)
                    and int(o.hp) not in (0, 240)
                    and int(o.hp) < 200
                ],
            }
        )

    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    try:
        obs, _ = reset_obs(env)
        run = run_survival_spine(
            env, obs, assist=None, on_frame=on_frame, through="pre-l1",
            allow_pokes=False,
        )
        ok = bool(run.report().get("ok"))
    finally:
        env.close()
    OverworldPathController.step = _orig_step  # type: ignore[assignment]

    presses = []
    for i, row in enumerate(frames):
        if not row["a"] or (i and frames[i - 1]["a"]):
            continue  # the edge only; A held is one swing
        if row["face"] is None:
            continue
        hp_at_press = {b[0]: b[2] for b in row["bodies"]}
        hurt: set[int] = set()
        for later in frames[i + 1 : i + 1 + DAMAGE_WINDOW]:
            for slot, _t, hp, _x, _y in later["bodies"]:
                if slot in hp_at_press and hp < hp_at_press[slot]:
                    hurt.add(slot)
        bodies = []
        for slot, type_id, hp, bx, by in row["bodies"]:
            fwd, lat = _forward_lateral(row["face"], bx - row["x"], by - row["y"])
            bodies.append(
                {
                    "slot": slot,
                    "type": type_id,
                    "hp": hp,
                    "fwd": fwd,
                    "lat": lat,
                    "cheb": max(abs(bx - row["x"]), abs(by - row["y"])),
                    "hurt": slot in hurt,
                }
            )
        presses.append(
            {
                "f": row["f"],
                "scr": row["scr"],
                "why": row["why"],
                "face": row["face"],
                "landed": bool(hurt),
                "bodies": bodies,
            }
        )

    landed = [p for p in presses if p["landed"]]
    # Only the body that was actually hurt tells us where the blade reaches.
    hits = [b for p in landed for b in p["bodies"] if b["hurt"]]
    # A body inside the model's box on a press that hurt nothing is a miss the
    # model did not predict.
    model_misses = [
        b
        for p in presses
        if not p["landed"]
        for b in p["bodies"]
        if 0 < b["fwd"] <= 20 and abs(b["lat"]) <= 12
    ]
    summary = {
        "ok": ok,
        "play_frames": len(frames),
        "presses": len(presses),
        "landed": len(landed),
        "hit_fwd_min": min((b["fwd"] for b in hits), default=None),
        "hit_fwd_max": max((b["fwd"] for b in hits), default=None),
        "hit_lat_max": max((abs(b["lat"]) for b in hits), default=None),
        "hit_cheb_min": min((b["cheb"] for b in hits), default=None),
        "misses_inside_model": len(model_misses),
        "miss_fwd_hist": {},
        "hit_fwd_hist": {},
    }
    for b in hits:
        key = str(b["fwd"] // 4 * 4)
        summary["hit_fwd_hist"][key] = summary["hit_fwd_hist"].get(key, 0) + 1
    for b in model_misses:
        key = str(b["fwd"] // 4 * 4)
        summary["miss_fwd_hist"][key] = summary["miss_fwd_hist"].get(key, 0) + 1
    by_reason: dict[str, list[int]] = {}
    for p in presses:
        name = (p["why"] or "?").split("_slash")[0]
        tally = by_reason.setdefault(name, [0, 0])
        tally[0] += 1
        tally[1] += int(p["landed"])
    summary["by_reason"] = {k: {"presses": v[0], "landed": v[1]} for k, v in by_reason.items()}

    path = OUT_DIR / f"{args.tag}.json"
    path.write_text(json.dumps({"tag": args.tag, "summary": summary, "presses": presses}))
    (RECORDINGS_DIR / f"{args.tag}.json").write_text(json.dumps(summary))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
