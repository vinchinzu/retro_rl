
"""Walk onto visible stairs in each L9 stair source; record live dest."""
from __future__ import annotations

from retro_harness.env import make_env
from retro_harness.nes import nes_action, nes_idle_action
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
from zelda_i.ram import PLAY_MODE, read_snapshot
from zelda_i.scripts.run_level9_ganon import FIXTURE_SOURCE, _idle, _step
from zelda_i.scripts.run_level9_stairs import materialize_stair_room, _exit_cellar

# Per-room visible stair stands from settle screenshots (px).
STAIR_TARGETS = {
    0x60: ((0x78, 0x90), (0x70, 0x90), (0x80, 0x90), (0x60, 0x90), (0x90, 0x90)),
    0x70: ((0xB0, 0x90), (0xA0, 0x90), (0xC0, 0x90), (0xD0, 0x90), (0xB0, 0x80)),
    0x72: ((0xD0, 0x60), (0x88, 0x90), (0x78, 0x90)),
    0x75: ((0xB0, 0x90), (0xA0, 0x90), (0xC0, 0x90), (0xD0, 0x60)),
    0x67: ((0xB0, 0x90), (0xA0, 0x90), (0xC0, 0x80), (0xD0, 0x60)),
    0x77: ((0xB0, 0x90), (0xA0, 0x90), (0xC0, 0x90), (0x90, 0x90)),
    0x00: ((0xD0, 0x60), (0x88, 0x90), (0x40, 0x90), (0x78, 0x90)),
    0x4F: ((0xD0, 0x60), (0x88, 0x90), (0x78, 0x90), (0xB0, 0x90)),
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
            return row
        snap = read_snapshot(env.get_ram())
        row["settled"] = dest_report(snap)
        save_rgb_png(obs, RECORDINGS_DIR / f"l9_stair_0x{room:02x}_pre.png")

        dest_snap = snap
        took = False
        for tx, ty in STAIR_TARGETS[room]:
            for _ in range(500):
                snap = read_snapshot(env.get_ram())
                if stair_transition_modes(snap.mode) or (
                    snap.screen != room and snap.mode not in (6, 7)
                ):
                    took = True
                    dest_snap = snap
                    break
                frame = walk_to_step(snap, tx, ty, y_first=True)
                if frame.reason == "walk_arrived":
                    obs = _step(env, nes_idle_action(), assist=None, total=total)
                    obs = _idle(env, 25, assist=None, total=total)
                    snap = read_snapshot(env.get_ram())
                    if stair_transition_modes(snap.mode) or snap.screen != room:
                        took = True
                        dest_snap = snap
                    break
                obs = _step(env, frame.action, assist=None, total=total)
            if took:
                break
            # Nudge UP/DOWN on the stand in case we are one tile off.
            for direction in ("UP", "DOWN", "LEFT", "RIGHT"):
                for _ in range(20):
                    snap = read_snapshot(env.get_ram())
                    if stair_transition_modes(snap.mode) or (
                        snap.screen != room and snap.mode not in (6, 7)
                    ):
                        took = True
                        dest_snap = snap
                        break
                    obs = _step(env, nes_action(direction), assist=None, total=total)
                if took:
                    break
            if took:
                break

        if stair_transition_modes(dest_snap.mode) or dest_snap.mode in (9, 10, 11, 16):
            for side in ("left", "right"):
                # If already leaving, just finish this side.
                obs, dest_snap = _exit_cellar(env, total=total, side=side)
                if dest_snap.mode == PLAY_MODE and dest_snap.screen != room:
                    row["cellar_side"] = side
                    break
                # Re-enter would require rematerialize; record and stop this env.

        dest_snap = read_snapshot(env.get_ram())
        if dest_snap.mode == PLAY_MODE:
            obs = _idle(env, 20, assist=None, total=total)
            dest_snap = read_snapshot(env.get_ram())
        row["took_stairs"] = took or dest_snap.screen != room or dest_snap.mode != PLAY_MODE
        row["dest"] = dest_report(dest_snap)
        row["landed_final_patra"] = bool(landed_final_patra(dest_snap))
        row["frames"] = total[0]
        png = RECORDINGS_DIR / f"l9_stair_0x{room:02x}_dest.png"
        save_rgb_png(obs if obs is not None else env.render(), png)
        row["dest_png"] = str(png)
        d = row["dest"]
        print(
            f"0x{room:02X} took={row['took_stairs']} -> "
            f"SCREEN=0x{d['screen']:02X} NEXT=0x{d['next_screen']:02X} "
            f"mode={d['mode']} patra={d['final_patra_live']} eyes={d['patra_eyes']} "
            f"xy=({d['link']['x']},{d['link']['y']}) "
            f"objs={[o['type_name'] for o in d['objects'][:8]]}"
        )
        return row
    finally:
        env.close()


def main() -> None:
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
    report = {"ok": winner is not None, "winner": winner, "sources": rows}
    write_json_report(RECORDINGS_DIR / "l9_stair_take_probe.json", report)
    print("WINNER", winner["source"] if winner else None)


if __name__ == "__main__":
    main()
