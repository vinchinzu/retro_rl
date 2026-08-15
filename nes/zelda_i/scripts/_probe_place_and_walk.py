
"""Fixture-place Link on walkable floor, then controller-walk onto stairs.

Position writes are fixture-build only (same class as ganon final_link_position).
NEXT_SCREEN is never poked to 0x52. Dest comes from the game stair loader.
"""
from retro_harness.env import make_env
from retro_harness.nes import nes_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.level9_room62 import LEVEL9_STAIR_SOURCES
from zelda_i.level9_stairs import (
    dest_report,
    landed_final_patra,
    paired_stair_dest,
    stair_transition_modes,
    walk_to_step,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_LINK_X, ADDR_LINK_Y, PLAY_MODE, read_snapshot
from zelda_i.scripts.run_level9_ganon import FIXTURE_SOURCE, _assign, _idle, _step
from zelda_i.scripts.run_level9_stairs import materialize_stair_room, _exit_cellar

# Walkable-looking floor stands from settle screenshots (right-side stair wells).
PLACE = {
    0x60: (0xB0, 0x90),
    0x70: (0xB0, 0x90),
    0x72: (0x78, 0xB0),
    0x75: (0xB0, 0x90),
    0x67: (0x50, 0x90),
    0x77: (0x90, 0x90),
    0x00: (0x50, 0x90),
    0x4F: (0x50, 0x90),
}
STAIR_WALK = {
    0x60: ((0xC0, 0x90), (0xB0, 0x90), (0xA0, 0x90), (0xD0, 0x60), (0xB8, 0x88)),
    0x70: ((0xC0, 0x90), (0xB0, 0x90), (0xA0, 0x80), (0xD0, 0x60)),
    0x72: ((0xD0, 0x60), (0x88, 0x90), (0x78, 0x90), (0x40, 0x90)),
    0x75: ((0xC0, 0x90), (0xB0, 0x90), (0xD0, 0x60)),
    0x67: ((0xB0, 0x90), (0xA0, 0x80), (0xC0, 0x80), (0x50, 0x90)),
    0x77: ((0xB0, 0x90), (0xA0, 0x90), (0xC0, 0x90)),
    0x00: ((0xD0, 0x60), (0x88, 0x90), (0x40, 0xA0)),
    0x4F: ((0xD0, 0x60), (0x88, 0x90), (0xB0, 0x90)),
}


def try_room(room: int) -> dict:
    env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
    total = [0]
    try:
        obs, loader, loaded = materialize_stair_room(env, room, total=total)
        row = {
            "source": f"0x{room:02X}",
            "paired": None if paired_stair_dest(room) is None else f"0x{paired_stair_dest(room):02X}",
            "loaded": loaded,
            "loader": loader.label,
        }
        if not loaded:
            row["error"] = "settle failed"
            row["dest"] = dest_report(read_snapshot(env.get_ram()))
            return row
        px, py = PLACE[room]
        _assign(env, ADDR_LINK_X, px)
        _assign(env, ADDR_LINK_Y, py)
        obs = _idle(env, 8, assist=None, total=total)
        snap = read_snapshot(env.get_ram())
        row["placed"] = dest_report(snap)
        save_rgb_png(obs, RECORDINGS_DIR / f"l9_stair_0x{room:02x}_placed.png")
        print(f"0x{room:02X} placed ({snap.link_x},{snap.link_y}) tile=0x{snap.colliding_tile:02x}")

        # Push LEFT from the place stand (in case a block hides stairs).
        for _ in range(35):
            snap = read_snapshot(env.get_ram())
            if stair_transition_modes(snap.mode) or snap.screen != room:
                break
            obs = _step(env, nes_action("LEFT"), assist=None, total=total)

        for tx, ty in STAIR_WALK[room]:
            for _ in range(280):
                snap = read_snapshot(env.get_ram())
                if stair_transition_modes(snap.mode) or (
                    snap.screen != room and snap.mode not in (6, 7)
                ):
                    break
                frame = walk_to_step(snap, tx, ty, y_first=False)
                if frame.reason == "walk_arrived":
                    obs = _idle(env, 18, assist=None, total=total)
                    break
                obs = _step(env, frame.action, assist=None, total=total)
            snap = read_snapshot(env.get_ram())
            print(f"  walk ({tx},{ty}) now=({snap.link_x},{snap.link_y}) tile=0x{snap.colliding_tile:02x} room=0x{snap.screen:02x} mode={snap.mode}")
            if stair_transition_modes(snap.mode) or snap.screen != room:
                break

        snap = read_snapshot(env.get_ram())
        if stair_transition_modes(snap.mode) or snap.mode in (9, 10, 11, 16):
            save_rgb_png(obs, RECORDINGS_DIR / f"l9_stair_0x{room:02x}_passage.png")
            obs, snap = _exit_cellar(env, total=total, side="left")
            print(f"  cellar-left -> 0x{snap.screen:02x} mode={snap.mode} patra={landed_final_patra(snap)}")
            if not landed_final_patra(snap) and snap.screen != 0x52:
                # Rematerialize and try the right mouth.
                env.close()
                env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
                total = [0]
                obs, loader, loaded = materialize_stair_room(env, room, total=total)
                _assign(env, ADDR_LINK_X, px)
                _assign(env, ADDR_LINK_Y, py)
                _idle(env, 8, assist=None, total=total)
                for tx, ty in STAIR_WALK[room]:
                    for _ in range(280):
                        snap = read_snapshot(env.get_ram())
                        if stair_transition_modes(snap.mode) or snap.screen != room:
                            break
                        frame = walk_to_step(snap, tx, ty, y_first=False)
                        if frame.reason == "walk_arrived":
                            _idle(env, 18, assist=None, total=total)
                            break
                        _step(env, frame.action, assist=None, total=total)
                    snap = read_snapshot(env.get_ram())
                    if stair_transition_modes(snap.mode) or snap.screen != room:
                        break
                if stair_transition_modes(snap.mode) or snap.mode in (9, 10, 11, 16):
                    obs, snap = _exit_cellar(env, total=total, side="right")
                    print(f"  cellar-right -> 0x{snap.screen:02x} mode={snap.mode} patra={landed_final_patra(snap)}")

        if snap.mode == PLAY_MODE:
            obs = _idle(env, 18, assist=None, total=total)
            snap = read_snapshot(env.get_ram())
        row["dest"] = dest_report(snap)
        row["landed_final_patra"] = bool(landed_final_patra(snap))
        png = RECORDINGS_DIR / f"l9_stair_0x{room:02x}_dest.png"
        save_rgb_png(obs if obs is not None else env.render(), png)
        row["dest_png"] = str(png)
        d = row["dest"]
        print(
            f"0x{room:02X} DEST SCREEN=0x{d['screen']:02X} NEXT=0x{d['next_screen']:02X} "
            f"mode={d['mode']} patra={d['final_patra_live']} eyes={d['patra_eyes']} "
            f"objs={[o['type_name'] for o in d['objects'][:6]]}"
        )
        return row
    finally:
        env.close()


def main():
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    winner = None
    for room in LEVEL9_STAIR_SOURCES:
        row = try_room(room)
        rows.append(row)
        if row.get("landed_final_patra"):
            winner = row
            break
    write_json_report(
        RECORDINGS_DIR / "l9_stair_place_walk_probe.json",
        {"ok": winner is not None, "winner": winner, "sources": rows},
    )
    print("WINNER", None if winner is None else winner["source"])


if __name__ == "__main__":
    main()
