"""One-tile push + stair-stand scan in play 0x30. Fixture-only."""
from __future__ import annotations

from retro_harness.env import make_env
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.level9_ganon import LEVEL9
from zelda_i.level9_stairs import (
    ROOM30,
    ROOM30_STAIR_X,
    ROOM30_STAIR_Y,
    chase_sword_step,
    dest_report,
    in_room_30,
    live_combat_objects,
    on_warp_tile,
    pushable_block,
    stair_transition_modes,
    walk_to_step,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import PLAY_MODE, read_snapshot
from zelda_i.scripts.run_level9_ganon import _idle, _step
from zelda_i.scripts.run_level9_stairs import dump_room_tiles, materialize_stair_room

TAG = "l9_room30_push"


def main() -> int:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    assist = UnlimitedHealthAssist(enabled=True)
    env = make_env(GAME, "Level9EntranceReconFixture", GAME_DIR, render_mode="rgb_array")
    total = [0]
    report: dict = {"ok": False, "init_mode9": False, "route_eligible": False}
    try:
        obs, loader, loaded = materialize_stair_room(env, ROOM30, total=total)
        report["loaded"] = loaded
        report["loader"] = loader.label
        if not loaded:
            report["error"] = "no settle"
            write_json_report(RECORDINGS_DIR / f"{TAG}.json", report)
            return 1
        save_rgb_png(obs, RECORDINGS_DIR / f"{TAG}_settle.png")
        report["settled"] = dest_report(read_snapshot(env.get_ram()))

        cooldown = 0
        for _ in range(2500):
            snap = read_snapshot(env.get_ram())
            if in_room_30(snap) and not live_combat_objects(snap):
                break
            frame, cooldown = chase_sword_step(snap, cooldown)
            obs = _step(env, frame.action, assist=assist, total=total)
        obs = _idle(env, 20, assist=assist, total=total)
        snap = read_snapshot(env.get_ram())
        block = pushable_block(snap)
        report["after_clear"] = dest_report(snap)
        report["block_after_clear"] = None if block is None else {"x": block.x, "y": block.y}
        save_rgb_png(obs, RECORDINGS_DIR / f"{TAG}_after_clear.png")

        tiles = dump_room_tiles(env, total=total)
        report["tiles_after_clear"] = {
            "stair_hits": tiles["stair_hits"],
            "mouth_hits": tiles["mouth_hits"],
            "tile_counts": tiles["tile_counts"],
        }

        # Rematerialize + reclear so tile pokes are not on the push path.
        env.close()
        env = make_env(GAME, "Level9EntranceReconFixture", GAME_DIR, render_mode="rgb_array")
        total = [0]
        obs, _, loaded = materialize_stair_room(env, ROOM30, total=total)
        cooldown = 0
        for _ in range(2500):
            snap = read_snapshot(env.get_ram())
            if in_room_30(snap) and not live_combat_objects(snap):
                break
            frame, cooldown = chase_sword_step(snap, cooldown)
            _step(env, frame.action, assist=assist, total=total)

        # Walk south of the west block and push UP one tile (0x03 style).
        for _ in range(400):
            snap = read_snapshot(env.get_ram())
            frame = walk_to_step(snap, 96, 189, y_first=True)
            if frame.reason == "walk_arrived":
                break
            _step(env, frame.action, assist=assist, total=total)
        for _ in range(200):
            snap = read_snapshot(env.get_ram())
            frame = walk_to_step(snap, 96, 170, y_first=True, tol=0)
            if abs(int(snap.link_x) - 96) <= 1 and abs(int(snap.link_y) - 170) <= 1:
                break
            _step(env, frame.action, assist=assist, total=total)
        report["at_push_stand"] = dest_report(read_snapshot(env.get_ram()))
        for i in range(80):
            snap = read_snapshot(env.get_ram())
            block = pushable_block(snap)
            if block is not None and int(block.y) <= 0x80:
                report["push_up_frames"] = i
                break
            _step(env, nes_action("UP"), assist=assist, total=total)
        snap = read_snapshot(env.get_ram())
        block = pushable_block(snap)
        report["block_after_up"] = None if block is None else {"x": block.x, "y": block.y}
        idle = _idle(env, 8, assist=assist, total=total)
        save_rgb_png(idle, RECORDINGS_DIR / f"{TAG}_after_push_up.png")

        tiles = dump_room_tiles(env, total=total)
        report["tiles_after_push_up"] = {
            "stair_hits": tiles["stair_hits"],
            "mouth_hits": tiles["mouth_hits"],
        }

        # If no stairs, rematerialize and try LEFT one tile from east of block.
        if not tiles["stair_hits"]:
            env.close()
            env = make_env(GAME, "Level9EntranceReconFixture", GAME_DIR, render_mode="rgb_array")
            total = [0]
            materialize_stair_room(env, ROOM30, total=total)
            cooldown = 0
            for _ in range(2500):
                snap = read_snapshot(env.get_ram())
                if in_room_30(snap) and not live_combat_objects(snap):
                    break
                frame, cooldown = chase_sword_step(snap, cooldown)
                _step(env, frame.action, assist=assist, total=total)
            for _ in range(400):
                snap = read_snapshot(env.get_ram())
                frame = walk_to_step(snap, 120, 189, y_first=True)
                if frame.reason == "walk_arrived":
                    break
                _step(env, frame.action, assist=assist, total=total)
            for _ in range(300):
                snap = read_snapshot(env.get_ram())
                frame = walk_to_step(snap, 112, 144, y_first=True, tol=0)
                if abs(int(snap.link_x) - 112) <= 1 and abs(int(snap.link_y) - 144) <= 1:
                    break
                _step(env, frame.action, assist=assist, total=total)
            report["at_left_push"] = dest_report(read_snapshot(env.get_ram()))
            start_block = pushable_block(read_snapshot(env.get_ram()))
            start_x = None if start_block is None else int(start_block.x)
            for i in range(80):
                snap = read_snapshot(env.get_ram())
                block = pushable_block(snap)
                if block is not None and start_x is not None and int(block.x) <= start_x - 16:
                    report["push_left_frames"] = i
                    break
                _step(env, nes_action("LEFT"), assist=assist, total=total)
            snap = read_snapshot(env.get_ram())
            block = pushable_block(snap)
            report["block_after_left"] = None if block is None else {"x": block.x, "y": block.y}
            idle = _idle(env, 8, assist=assist, total=total)
            save_rgb_png(idle, RECORDINGS_DIR / f"{TAG}_after_push_left.png")
            tiles = dump_room_tiles(env, total=total)
            report["tiles_after_push_left"] = {
                "stair_hits": tiles["stair_hits"],
                "mouth_hits": tiles["mouth_hits"],
            }

        hits = (report.get("tiles_after_push_up") or {}).get("stair_hits") or (
            report.get("tiles_after_push_left") or {}
        ).get("stair_hits") or []
        report["stair_hits"] = hits

        # Exact-stand candidates around engine block-stairs + any tile hits.
        stands = [(ROOM30_STAIR_X, ROOM30_STAIR_Y), (208, 93), (200, 96), (208, 101)]
        stands.extend((h["x"], h["y"]) for h in hits[:8])
        report["stand_tries"] = []
        for x, y in stands:
            _assign = __import__(
                "zelda_i.scripts.run_level9_ganon", fromlist=["_assign"]
            )._assign
            from zelda_i.ram import ADDR_LINK_X, ADDR_LINK_Y

            _assign(env, ADDR_LINK_X, x)
            _assign(env, ADDR_LINK_Y, y)
            _idle(env, 12, assist=None, total=total)
            snap = read_snapshot(env.get_ram())
            row = {
                "try": [x, y],
                "link": {"x": snap.link_x, "y": snap.link_y},
                "tile": int(snap.colliding_tile),
                "mode": int(snap.mode),
                "screen": int(snap.screen),
                "on_warp": on_warp_tile(snap),
                "transition": stair_transition_modes(snap.mode),
            }
            report["stand_tries"].append(row)
            if stair_transition_modes(snap.mode) or snap.mode != PLAY_MODE:
                idle = _idle(env, 1, assist=None, total=total)
                save_rgb_png(idle, RECORDINGS_DIR / f"{TAG}_stairs_enter.png")
                report["entered"] = row
                break
        report["ok"] = True
        write_json_report(RECORDINGS_DIR / f"{TAG}.json", report)
        print("PUSH", {
            "block_clear": report.get("block_after_clear"),
            "block_up": report.get("block_after_up"),
            "block_left": report.get("block_after_left"),
            "stairs_up": (report.get("tiles_after_push_up") or {}).get("stair_hits"),
            "stairs_left": (report.get("tiles_after_push_left") or {}).get("stair_hits"),
            "stands": report.get("stand_tries"),
            "entered": report.get("entered"),
        })
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
