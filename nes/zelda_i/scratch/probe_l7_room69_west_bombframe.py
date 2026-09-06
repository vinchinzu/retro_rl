"""Frame-level diagnostic of the room69_west_bomb PLACE/WAIT window.

Chains pond drain -> entry north door -> Room69WestBombController from the
OW_L7PondNatural pin, then logs every frame from the moment the kill-clear
finishes ("cleared" note) through PLACE/WAIT/PUSH: snap.bombs, mode,
transitioning, and the full object census.  Read-only diagnostic.

    QT_QPA_PLATFORM=offscreen PYTHONPATH=nes:. uv run python \\
        nes/zelda_i/scratch/probe_l7_room69_west_bombframe.py --tag bombframe_v1
"""

from __future__ import annotations

import argparse
import json

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.dungeon.bomb_wall import BombWallPhase
from zelda_i.dungeon.ids import object_name
from zelda_i.level7.hops import (
    make_entry_first_door_controller,
    make_pond_entry_controller,
    make_room69_west_bomb_controller,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_SELECTED_ITEM, read_snapshot, read_u8
from zelda_i.runner import make_assist


def _objects(snap) -> list[dict]:
    out = []
    for o in snap.objects:
        t = int(o.type_id) & 0xFF
        if t in (0, 0xFF) or not (1 <= int(o.slot) <= 12):
            continue
        out.append(
            {
                "slot": int(o.slot),
                "type": f"0x{t:02x}:{object_name(t)}",
                "hp": int(o.hp),
                "state": int(o.state),
                "x": int(o.x),
                "y": int(o.y),
            }
        )
    return out


def _run_to_done(env, ctl, assist) -> None:
    bind = getattr(ctl, "bind_env", None)
    if bind is not None:
        bind(env)
    f = 0
    while f < ctl.max_frames + 10:
        snap = read_snapshot(env.get_ram())
        action = ctl.step(snap)
        env.step(action.action)
        assist.apply_env(env, frame=f)
        f += 1
        failed = getattr(ctl, "failed", None)
        if failed is None:
            failed = getattr(ctl, "phase", None) is BombWallPhase.FAILED
        if ctl.success or failed:
            return


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="bombframe_v1")
    ap.add_argument("--from-state", default="OW_L7PondNatural")
    args = ap.parse_args()
    configure_headless()
    a = make_assist(True)
    env = make_env(GAME, args.from_state, GAME_DIR, render_mode="rgb_array")
    log: list[dict] = []
    try:
        reset_obs(env)
        for _ in range(2):
            env.step(nes_idle_action())

        _run_to_done(env, make_pond_entry_controller(), a)
        _run_to_done(env, make_entry_first_door_controller(), a)

        ctl = make_room69_west_bomb_controller()
        bind = getattr(ctl, "bind_env", None)
        if bind is not None:
            bind(env)

        f = 0
        cleared_at = None
        last_phase = None
        while f < ctl.max_frames + 10:
            snap = read_snapshot(env.get_ram())
            action = ctl.step(snap)
            phase = getattr(ctl.bomb, "phase", None)
            phase_name = phase.name if phase is not None else None
            if cleared_at is None and "cleared" in ctl.notes:
                cleared_at = f
            if cleared_at is not None:
                if phase_name != last_phase or phase_name in (
                    "FACE",
                    "PLACE",
                    "WAIT",
                    "PUSH",
                ):
                    log.append(
                        {
                            "f": f,
                            "phase": phase_name,
                            "reason": action.reason,
                            "bombs": int(snap.bombs),
                            "mode": int(snap.mode),
                            "submode": int(snap.submode),
                            "is_updating_mode": int(snap.is_updating_mode),
                            "selected_item": int(read_u8(env.get_ram(), ADDR_SELECTED_ITEM)),
                            "transitioning": bool(snap.transitioning),
                            "xy": [int(snap.link_x), int(snap.link_y)],
                            "objects": _objects(snap),
                        }
                    )
                last_phase = phase_name
            env.step(action.action)
            a.apply_env(env, frame=f)
            f += 1
            if cleared_at is not None and phase_name in ("PLACE", "WAIT") and f in (
                1489,
                1490,
                1491,
                1492,
                1495,
                1500,
                1520,
                1550,
                1590,
            ):
                save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_f{f}.png")
            failed = getattr(ctl, "failed", None)
            if ctl.success or failed:
                log.append({"f": f, "FINAL_REPORT": ctl.report()})
                save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_final.png")
                break

        (RECORDINGS_DIR / f"{args.tag}.json").write_text(json.dumps(log, indent=1))
        print(json.dumps(log[-80:], indent=1))
    finally:
        env.close()


if __name__ == "__main__":
    main()
