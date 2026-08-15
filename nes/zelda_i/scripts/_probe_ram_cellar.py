
"""Dump live L9 LevelInfo cellar array + PlayAreaTiles stair cells."""
from retro_harness.env import make_env
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.level9_room62 import LEVEL9_STAIR_SOURCES
from zelda_i.level9_stairs import dest_report, landed_final_patra, stair_transition_modes
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_LINK_X, ADDR_LINK_Y, read_snapshot
from zelda_i.scripts.run_level9_ganon import FIXTURE_SOURCE, _assign, _idle, _step
from zelda_i.scripts.run_level9_stairs import materialize_stair_room, _exit_cellar

PLAY_AREA = 0x6530
ATTRS_A = 0x687E
ATTRS_B = 0x68FE
ATTRS_C = 0x697E
ATTRS_D = 0x69FE
CELLAR_ARRAY = 0x6BB2
SHORTCUT_POS = 0x6BA7
START_ROOM = 0x6BAD
BOSS_ROOM = 0x6BBC
CELLAR_SRC = 0x0527
UW_EXIT = 0x005A
UW_TILE = 0x0065


def dump_levelinfo(ram) -> dict:
    cellar = [int(ram[CELLAR_ARRAY + i]) for i in range(10)]
    return {
        "cellar_array_6": [f"0x{v:02X}" for v in cellar[:6]],
        "cellar_array_10": [f"0x{v:02X}" for v in cellar],
        "shortcut_or_item_pos": [int(ram[SHORTCUT_POS + i]) for i in range(4)],
        "start_room": f"0x{int(ram[START_ROOM]):02X}",
        "boss_room": f"0x{int(ram[BOSS_ROOM]):02X}",
        "cellar_source_room": f"0x{int(ram[CELLAR_SRC]):02X}",
        "uw_exit_type": int(ram[UW_EXIT]),
        "uw_entrance_tile": int(ram[UW_TILE]),
        "attrs": {
            f"0x{room:02X}": {
                "A": f"0x{int(ram[ATTRS_A + room]):02X}",
                "B": f"0x{int(ram[ATTRS_B + room]):02X}",
                "C": f"0x{int(ram[ATTRS_C + room]):02X}",
                "D": f"0x{int(ram[ATTRS_D + room]):02X}",
            }
            for room in list(LEVEL9_STAIR_SOURCES) + [0x52, 0x42, 0x32, 0x76]
        },
    }


def stair_cells(ram) -> list[dict]:
    tiles = bytes(ram[PLAY_AREA:PLAY_AREA + 0x2C0])
    hits = []
    # UW play area: 24 tile-columns * 22 tile-rows, column-major.
    cols, rows = 24, 22
    for col in range(cols):
        for row in range(rows):
            idx = col * rows + row
            if idx >= len(tiles):
                continue
            tile = tiles[idx]
            if 0x70 <= tile <= 0x73 or tile == 0x24:
                hits.append({
                    "col": col,
                    "row": row,
                    "tile": f"0x{tile:02X}",
                    "guess_x": col * 8,
                    "guess_y": 0x40 + row * 8,
                })
    return hits


def try_walk_stairs(env, room: int, cells: list[dict], total: list[int]):
    results = []
    for cell in cells:
        gx, gy = cell["guess_x"], cell["guess_y"]
        # Link Y is typically square-aligned with low nibble 0xD.
        stands = [
            (gx, gy),
            (gx, (gy & 0xF0) | 0x0D),
            (max(0x20, gx - 8), (gy & 0xF0) | 0x0D),
            (min(0xD0, gx + 8), (gy & 0xF0) | 0x0D),
            (gx, max(0x4D, gy - 16)),
            (gx, min(0xBD, gy + 16)),
        ]
        for x, y in stands:
            _assign(env, ADDR_LINK_X, x)
            _assign(env, ADDR_LINK_Y, y)
            _idle(env, 2, assist=None, total=total)
            hit = False
            for direction in ("UP", "DOWN", "LEFT", "RIGHT"):
                obs = _step(env, nes_action(direction), assist=None, total=total)
                snap = read_snapshot(env.get_ram())
                if stair_transition_modes(snap.mode) or (
                    snap.screen != room and snap.mode not in (4, 6, 7)
                ):
                    hit = True
                    dest = dest_report(snap)
                    if snap.mode in (9, 10, 11, 16):
                        obs, snap = _exit_cellar(env, total=total, side="left")
                        dest = dest_report(snap)
                    entry = {
                        "cell": cell,
                        "stand": [x, y],
                        "nudge": direction,
                        "dest": dest,
                        "patra": landed_final_patra(snap),
                    }
                    results.append(entry)
                    print(
                        f"HIT 0x{room:02X} cell={cell} stand=({x},{y}) {direction} "
                        f"-> 0x{dest['screen']:02X} mode={dest['mode']} patra={entry['patra']}"
                    )
                    if entry["patra"]:
                        save_rgb_png(obs, RECORDINGS_DIR / f"l9_stair_0x{room:02x}_patra_hit.png")
                        return results, entry
                    # reload room
                    env.close()
                    return results, entry
            if hit:
                break
    return results, None


def main():
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
    total = [0]
    try:
        obs, loader, loaded = materialize_stair_room(env, 0x77, total=total)
        ram = env.get_ram()
        info = dump_levelinfo(ram)
        print("LEVELINFO", info)
        cells77 = stair_cells(ram)
        print("0x77 stair/cave tiles", cells77)
        info["room_0x77"] = {
            "loaded": loaded,
            "loader": loader.label,
            "settle": dest_report(read_snapshot(ram)),
            "stair_cells": cells77,
        }
        save_rgb_png(obs, RECORDINGS_DIR / "l9_stair_0x77_ramdump_settle.png")
        hits, winner = try_walk_stairs(env, 0x77, cells77, total)
        info["room_0x77"]["walk_hits"] = hits
        info["winner"] = winner
        # Also dump tiles for every source if 0x77 missed.
        if winner is None or not (winner or {}).get("patra"):
            other = {}
            for room in LEVEL9_STAIR_SOURCES:
                env.close()
                env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
                total = [0]
                obs, loader, loaded = materialize_stair_room(env, room, total=total)
                ram = env.get_ram()
                cells = stair_cells(ram)
                print(f"0x{room:02X} loaded={loaded} stair_cells={cells}")
                other[f"0x{room:02X}"] = {
                    "loaded": loaded,
                    "loader": loader.label,
                    "settle": dest_report(read_snapshot(ram)),
                    "stair_cells": cells,
                }
                if cells:
                    save_rgb_png(obs, RECORDINGS_DIR / f"l9_stair_0x{room:02x}_tiles.png")
                    hits, win = try_walk_stairs(env, room, cells, total)
                    other[f"0x{room:02X}"]["walk_hits"] = hits
                    if win and win.get("patra"):
                        info["winner"] = {"source": f"0x{room:02X}", **win}
                        break
            info["all_sources"] = other
        write_json_report(RECORDINGS_DIR / "l9_stair_ram_cellar.json", info)
        print("WINNER", info.get("winner"))
    finally:
        env.close()


if __name__ == "__main__":
    main()
