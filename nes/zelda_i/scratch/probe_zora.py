"""What the Zora actually does in RAM, frame by frame, on the pre-L1 walk.

``tables1`` billed ``0x59`` the whole heart and the whole streak, and one of
its two hits is ``fireball_or_statue_projectile_E`` -- the Zora's spit. The
policy has no answer for it: ``behaviors.shield_blocks`` says ``0x55`` needs
the Magical Shield, so ``ShieldPolicy`` passes, and ``threat.off_line_step``
only looks at bodies, so nothing steps off the Zora's row *before* it fires.

Before writing a dodge, measure the thing being dodged. This logs every frame
a Zora (``0x11``) or a fireball (``0x55``) occupies a slot and answers four
questions the policy needs and cannot guess:

1. Which ``$00AC`` ObjState values a Zora holds, and which one the shot is
   born on -- the rising edge a 1 px/frame walker can still act on.
2. How many frames of warning that edge buys (surface -> shot).
3. Which axis the shot travels, against the Zora's ``$0098`` facing byte, so
   ``in_firing_line`` can be pointed at a Zora the same way it is at an
   octorok.
4. The shot's measured speed, which is what ``contact_frames`` extrapolates.

    QT_QPA_PLATFORM=offscreen uv run python nes/zelda_i/scratch/probe_zora.py --tag zora1
"""

from __future__ import annotations

import argparse
from pathlib import Path

from retro_harness.audit import AuditCapabilities, AuditedEnv
from retro_harness.env import make_env, reset_obs
from retro_harness.segment_runner import configure_headless, write_json_report
from zelda_i.combat import chebyshev
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import PLAY_MODE, read_snapshot
from zelda_i.spine.survival import run_survival_spine, spine_final_fields

OUT_DIR = Path(__file__).resolve().parent
ZORA = 0x11
FIREBALL = 0x55
# $0098-style facing byte -> compass letter, as ``hunt._FACING_SIDE`` reads it.
FACE = {0x08: "N", 0x04: "S", 0x01: "E", 0x02: "W"}


def _row(frame: int, snap, obj) -> dict:
    return {
        "f": frame,
        "screen": f"{int(snap.screen):#04x}",
        "slot": int(obj.slot),
        "type": int(obj.type_id),
        "state": int(obj.state),
        "hp": int(obj.hp),
        "x": int(obj.x),
        "y": int(obj.y),
        "facing": int(obj.facing),
        "face": FACE.get(int(obj.facing) & 0x0F, f"{int(obj.facing):#04x}"),
        "lx": int(snap.link_x),
        "ly": int(snap.link_y),
        "dx": int(obj.x) - int(snap.link_x),
        "dy": int(obj.y) - int(snap.link_y),
        "d": chebyshev(int(snap.link_x), int(snap.link_y), int(obj.x), int(obj.y)),
        "iframes": int(snap.link_iframes),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="zora")
    args = parser.parse_args(argv)
    configure_headless()

    frames: list[dict] = []
    # slot -> last state, so a transition is an event rather than 700 rows.
    last: dict[int, dict] = {}
    events: list[dict] = []

    def on_frame(env, _obs, _action, frame: int) -> None:
        snap = read_snapshot(env.get_ram())
        if int(snap.level) != 0 or int(snap.mode) != PLAY_MODE:
            return
        seen: set[int] = set()
        for obj in snap.objects:
            if int(obj.slot) < 1:
                continue
            type_id = int(obj.type_id) & 0xFF
            if type_id not in (ZORA, FIREBALL):
                continue
            seen.add(int(obj.slot))
            row = _row(frame, snap, obj)
            frames.append(row)
            prev = last.get(int(obj.slot))
            if prev is None:
                events.append({"kind": "spawn", **row})
            elif (
                prev["type"] != row["type"]
                or prev["state"] != row["state"]
                or prev["facing"] != row["facing"]
            ):
                events.append(
                    {
                        "kind": "change",
                        "from": {
                            "type": prev["type"],
                            "state": prev["state"],
                            "facing": prev["facing"],
                        },
                        **row,
                    }
                )
            last[int(obj.slot)] = row
        for gone in set(last) - seen:
            events.append({"kind": "gone", **last.pop(gone)})

    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    payload: dict | None = None
    try:
        obs, _ = reset_obs(env)
        env = AuditedEnv(env, capabilities=AuditCapabilities.all("zelda_i.zora"))
        run = run_survival_spine(
            env, obs, assist=None, through="pre-l1", allow_pokes=False, on_frame=on_frame
        )
        snap = read_snapshot(env.get_ram())
        report = run.report()
        payload = {
            "tag": args.tag,
            "ok": report.get("ok"),
            "failed": report.get("failed_stage"),
            "final": spine_final_fields(snap, env.get_ram()),
            "frames": frames,
            "events": events,
        }
    finally:
        env.close()

    write_json_report(OUT_DIR / f"{args.tag}.json", payload)
    write_json_report(RECORDINGS_DIR / f"{args.tag}.json", payload)
    print(f"{len(frames)} zora/fireball frames, {len(events)} events")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
