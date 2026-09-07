"""Power-on sweep of `room_item_id` for every Level 9 room the prefix walks
(0x76 -> ... -> 0x10), to locate the room that actually holds the Silver
Arrows (canonical item id 0x09).

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/probe_l9_prefix_item_sweep.py
"""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs
from retro_harness.segment_runner import configure_headless
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot
from zelda_i.spine.survival import run_survival_spine

seen: dict[int, int] = {}


def on_frame(env, obs, action, frame):
    snap = read_snapshot(env.get_ram())
    if snap.level != 9 or snap.mode != 5 or snap.transitioning:
        return
    if snap.screen in seen:
        return
    seen[snap.screen] = snap.room_item_id
    types = sorted({hex(o.type_id) for o in snap.objects if o.type_id})
    print(
        f"f{frame} room=0x{snap.screen:02x} room_item_id=0x{snap.room_item_id:02x} "
        f"arrows={snap.arrows} objs={types}",
        flush=True,
    )


def main() -> int:
    configure_headless()
    assist = UnlimitedHealthAssist(enabled=True)
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    obs, _ = reset_obs(env)
    try:
        run_survival_spine(env, obs, assist=assist, through="level9-credits", on_frame=on_frame)
    finally:
        env.close()
    print("L9 rooms seen:", {f"0x{k:02x}": f"0x{v:02x}" for k, v in sorted(seen.items())}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
