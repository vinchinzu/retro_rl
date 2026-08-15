
"""Fast settle-only dump of each L9 stair source. Not a dest claim."""
from __future__ import annotations

from retro_harness.env import make_env
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.level9_room62 import LEVEL9_STAIR_SOURCES
from zelda_i.level9_stairs import dest_report, paired_stair_dest, stair_loader_for
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import read_snapshot
from zelda_i.scripts.run_level9_ganon import FIXTURE_SOURCE
from zelda_i.scripts.run_level9_stairs import materialize_stair_room

def main() -> None:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
    try:
        for room in LEVEL9_STAIR_SOURCES:
            env.close()
            env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
            total = [0]
            obs, loader, loaded = materialize_stair_room(env, room, total=total)
            snap = read_snapshot(env.get_ram())
            row = {
                "source": f"0x{room:02X}",
                "paired": None if paired_stair_dest(room) is None else f"0x{paired_stair_dest(room):02X}",
                "loader": loader.label,
                "loaded": loaded,
                "frames": total[0],
                "settled": dest_report(snap),
            }
            png = RECORDINGS_DIR / f"l9_stair_0x{room:02x}_settle.png"
            if obs is not None:
                save_rgb_png(obs, png)
                row["png"] = str(png)
            rows.append(row)
            objs = [o["type_name"] for o in row["settled"]["objects"]]
            print(
                f"0x{room:02X} loaded={loaded} room=0x{snap.screen:02X} "
                f"mode={snap.mode} xy=({snap.link_x},{snap.link_y}) "
                f"tile=0x{snap.colliding_tile:02x} objs={objs}"
            )
    finally:
        env.close()
    write_json_report(RECORDINGS_DIR / "l9_stair_settle_probe.json", {"sources": rows})

if __name__ == "__main__":
    main()
