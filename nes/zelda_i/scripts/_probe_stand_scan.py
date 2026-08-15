
"""Stand Link on a grid; record cells that trigger a stair/passage mode change."""
from retro_harness.env import make_env
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, write_json_report
from zelda_i.level9_room62 import LEVEL9_STAIR_SOURCES
from zelda_i.level9_stairs import dest_report, stair_transition_modes
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_LINK_X, ADDR_LINK_Y, read_snapshot
from zelda_i.scripts.run_level9_ganon import FIXTURE_SOURCE, _assign, _step
from zelda_i.scripts.run_level9_stairs import materialize_stair_room

def scan_room(room: int) -> dict:
    env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
    total = [0]
    try:
        _, loader, loaded = materialize_stair_room(env, room, total=total)
        if not loaded:
            return {"source": f"0x{room:02X}", "loaded": False}
        triggers = []
        for y in range(0x55, 0xC6, 16):
            for x in range(0x30, 0xD1, 16):
                _assign(env, ADDR_LINK_X, x)
                _assign(env, ADDR_LINK_Y, y)
                _step(env, nes_idle_action(), assist=None, total=total)
                _step(env, nes_action("DOWN"), assist=None, total=total)
                _step(env, nes_idle_action(), assist=None, total=total)
                snap = read_snapshot(env.get_ram())
                if stair_transition_modes(snap.mode) or (snap.screen != room and snap.mode not in (4, 6, 7)):
                    triggers.append({
                        "stand": [x, y],
                        "now": dest_report(snap),
                    })
                    print(
                        f"0x{room:02X} TRIGGER stand=({x},{y}) "
                        f"room=0x{snap.screen:02x} mode={snap.mode} "
                        f"xy=({snap.link_x},{snap.link_y}) tile=0x{snap.colliding_tile:02x}"
                    )
                    # Reload room for the next stand.
                    env.close()
                    env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
                    total = [0]
                    _, loader, loaded = materialize_stair_room(env, room, total=total)
                    if not loaded:
                        break
        print(f"0x{room:02X} triggers={len(triggers)}")
        return {
            "source": f"0x{room:02X}",
            "loaded": True,
            "loader": loader.label,
            "triggers": triggers,
        }
    finally:
        env.close()


def main():
    configure_headless()
    rows = []
    for room in LEVEL9_STAIR_SOURCES:
        rows.append(scan_room(room))
    write_json_report(RECORDINGS_DIR / "l9_stair_stand_scan.json", {"sources": rows})


if __name__ == "__main__":
    main()
