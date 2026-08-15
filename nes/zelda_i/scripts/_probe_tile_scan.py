
"""Scan each stair-source room for stair tiles 0x70-0x73 by fixture-placing Link."""
from collections import Counter
from retro_harness.env import make_env
from retro_harness.segment_runner import configure_headless, write_json_report
from zelda_i.level9_room62 import LEVEL9_STAIR_SOURCES
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_LINK_X, ADDR_LINK_Y, read_snapshot
from zelda_i.scripts.run_level9_ganon import FIXTURE_SOURCE, _assign, _step
from zelda_i.scripts.run_level9_stairs import materialize_stair_room
from retro_harness.nes import nes_idle_action

def scan_room(room: int) -> dict:
    env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
    total = [0]
    try:
        obs, loader, loaded = materialize_stair_room(env, room, total=total)
        if not loaded:
            return {"source": f"0x{room:02X}", "loaded": False}
        hits = []
        counts = Counter()
        for y in range(0x4D, 0xDE, 8):
            for x in range(0x20, 0xE1, 8):
                _assign(env, ADDR_LINK_X, x)
                _assign(env, ADDR_LINK_Y, y)
                _step(env, nes_idle_action(), assist=None, total=total)
                snap = read_snapshot(env.get_ram())
                tile = int(snap.colliding_tile)
                counts[tile] += 1
                if 0x70 <= tile <= 0x73:
                    hits.append({"x": snap.link_x, "y": snap.link_y, "tile": tile})
        top = counts.most_common(8)
        print(
            f"0x{room:02X} stair_hits={len(hits)} "
            f"tiles={[(hex(t), n) for t, n in top]} "
            f"hits={hits[:8]}"
        )
        return {
            "source": f"0x{room:02X}",
            "loaded": True,
            "loader": loader.label,
            "stair_hits": hits,
            "tile_counts": {f"0x{t:02X}": n for t, n in top},
        }
    finally:
        env.close()


def main():
    configure_headless()
    rows = [scan_room(room) for room in LEVEL9_STAIR_SOURCES]
    write_json_report(RECORDINGS_DIR / "l9_stair_tile_scan.json", {"sources": rows})


if __name__ == "__main__":
    main()
