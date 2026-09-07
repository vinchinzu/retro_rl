"""Sweep `room_item_id` across every room the L9 join walks, to find which
Level 9 room actually holds the Silver Arrows.

Room 0x10's premise ("arrows are a floor item at the statue-grid centre")
was falsified live: room_item_id==3 (no reward), no item object ever spawns,
and its 0x68 is neither self-moving nor pushable. So the arrows must be in
a different room. Runs `NaturalPatraJoinController` from the real power-on
pin L9Room10EntryReal and logs (screen, room_item_id, objects) on every
screen change.

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/probe_l9_join_item_sweep.py
"""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs
from retro_harness.segment_runner import configure_headless
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.level9.natural_path import make_natural_patra_join_controller
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot

STATE_NAME = "L9Room10EntryReal"


def main() -> int:
    configure_headless()
    env = make_env(GAME, STATE_NAME, GAME_DIR, render_mode="rgb_array")
    obs, _ = reset_obs(env)
    assist = UnlimitedHealthAssist(enabled=True)
    ctl = make_natural_patra_join_controller()
    seen: dict[int, int] = {}
    frame = 0
    while not ctl.success and not ctl.failed and frame < 40000:
        snap = read_snapshot(env.get_ram())
        if snap.mode == 5 and not snap.transitioning and snap.screen not in seen:
            seen[snap.screen] = snap.room_item_id
            types = sorted({hex(o.type_id) for o in snap.objects if o.type_id})
            print(
                f"f{frame} room=0x{snap.screen:02x} room_item_id=0x{snap.room_item_id:02x} "
                f"arrows={snap.arrows} objs={types}"
            )
        if frame % 4000 == 0:
            print(f"  ..f{frame} room=0x{snap.screen:02x} xy=({snap.link_x},{snap.link_y}) phase={ctl.phase}", flush=True)
        act = ctl.step(snap)
        obs, *_ = env.step(act.action)
        frame += 1
        assist.apply_env(env, frame=frame)
    snap = read_snapshot(env.get_ram())
    print(f"DONE success={ctl.success} failed={ctl.failed} frames={ctl.frames} "
          f"room=0x{snap.screen:02x} arrows={snap.arrows}")
    print("rooms seen:", {f"0x{k:02x}": f"0x{v:02x}" for k, v in seen.items()})
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
