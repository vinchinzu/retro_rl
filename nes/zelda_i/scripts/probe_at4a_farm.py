"""Probe Clean heart farming live on At4A pin (overworld 0x4A).

Runs HeartFarmController on the At4A save state without pokes or assists.

Examples::

    QT_QPA_PLATFORM=offscreen uv run python nes/zelda_i/scripts/probe_at4a_farm.py
    QT_QPA_PLATFORM=offscreen uv run python nes/zelda_i/scripts/probe_at4a_farm.py --min-filled 4
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from retro_harness.env import make_env, reset_obs
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.overworld.heart_farm import HeartFarmController, HeartFarmPhase
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import read_snapshot


def run_probe(
    *,
    min_filled: int = 3,
    max_frames: int = 3600,
    output_png: Path | str = "recordings/at4a_clean_farm.png",
) -> dict[str, Any]:
    configure_headless()
    out_path = Path(output_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env = make_env(GAME, "At4A", GAME_DIR, render_mode="rgb_array")
    obs, _ = reset_obs(env)
    snap0 = read_snapshot(env.get_ram())

    farm = HeartFarmController(
        min_filled=min_filled,
        max_frames=max_frames,
        farm_screen=0x4A,
    )

    drops_seen: list[dict[str, Any]] = []
    kills = 0
    prev_kill_count = snap0.world_kill_count

    print(
        f"Initial state: screen={snap0.screen:#02x} mode={snap0.mode} "
        f"pos=({snap0.link_x}, {snap0.link_y}) health={snap0.health:#02x} "
        f"filled_hearts={snap0.filled_hearts}/{snap0.heart_containers} "
        f"kill_count={snap0.world_kill_count}"
    )

    for f in range(1, max_frames + 1):
        snap = read_snapshot(env.get_ram())
        if snap.world_kill_count > prev_kill_count:
            kills += snap.world_kill_count - prev_kill_count
            prev_kill_count = snap.world_kill_count

        floor_drops = [
            o for o in snap.objects if o.type_id == 0x60 and o.slot >= 1 and o.y > 0
        ]
        for d in floor_drops:
            entry = {"slot": d.slot, "state": hex(d.state), "x": d.x, "y": d.y}
            if not any(e["slot"] == d.slot and e["state"] == entry["state"] for e in drops_seen):
                drops_seen.append(entry)

        act = farm.step(snap)
        obs, *_ = env.step(act.action)

        if f % 20 == 0 or act.reason.startswith("farm_heart") or snap.filled_hearts != snap0.filled_hearts:
            print(
                f"f={f:04d} link=({snap.link_x}, {snap.link_y}) hp={snap.health:#02x} "
                f"filled={snap.filled_hearts}/{snap.heart_containers} act={act.reason} "
                f"phase={farm.phase.name} drops={len(floor_drops)}"
            )

        if farm.phase in (HeartFarmPhase.DONE, HeartFarmPhase.FAILED):
            break

    snap_final = read_snapshot(env.get_ram())
    save_rgb_png(obs, out_path)
    # Also save in in-game recordings dir per repo convention
    in_game_png = RECORDINGS_DIR / "at4a_clean_farm.png"
    in_game_png.parent.mkdir(parents=True, exist_ok=True)
    save_rgb_png(obs, in_game_png)

    env.close()

    rep = farm.report()
    summary = {
        "success": farm.success,
        "phase": farm.phase.name,
        "frames": farm.frames,
        "min_filled": min_filled,
        "initial_health": hex(snap0.health),
        "initial_filled": snap0.filled_hearts,
        "final_health": hex(snap_final.health),
        "final_filled": snap_final.filled_hearts,
        "containers": snap_final.heart_containers,
        "kills": kills,
        "drops_seen": drops_seen,
        "occupancy_misses": rep.get("occupancy_misses", 0),
        "notes": rep.get("notes", []),
        "screenshot": str(out_path),
    }

    print("\n--- Summary ---")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-filled", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=3600)
    parser.add_argument("--output-png", default="recordings/at4a_clean_farm.png")
    args = parser.parse_args()

    run_probe(
        min_filled=args.min_filled,
        max_frames=args.max_frames,
        output_png=args.output_png,
    )


if __name__ == "__main__":
    main()
