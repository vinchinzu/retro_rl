"""Resume Level5Entered07: other mouth 0x06, key west 0x05, block-stairs 0x04 whistle."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from retro_harness.segment_runner import configure_headless
from zelda_i.dungeon_ops import idle, push_dir
from zelda_i.level9_stairs import dest_report, on_stair_tile, walk_to_step
from zelda_i.paths import RECORDINGS_DIR
from zelda_i.ram import ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8

HERE = Path(__file__).resolve()
_spec = importlib.util.spec_from_file_location("l5whistle", HERE.parent / "_probe_l5_whistle_path.py")
w = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(w)

CELLAR = (9, 10, 11, 16)


def cellar(snap) -> bool:
    return snap.mode in CELLAR


def main() -> dict:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    hops = []
    checkpoints = []
    env, assist, _ = w.open_env("Level5Entered07")
    total = [1]
    try:
        idle(env, assist, total, 16)
        snap = read_snapshot(env.get_ram())
        start = w.dump_live(snap, env.get_ram())
        print("START07", start.get("room_hex"), "mode", start.get("mode"), [start.get("x"), start.get("y")], flush=True)
        hops.append({"hop": "0x64_stairs", "dest": "0x07", "ok": True, "resumed": True})

        cellar_ex = w.exit_cellar_other_mouth(env, assist, total)
        snap = read_snapshot(env.get_ram())
        hops.append({"hop": "0x07_other_mouth", "dest": f"0x{snap.screen:02x}", "mode": snap.mode, "ok": snap.screen == 0x06})
        print("CELLAR_OUT", hops[-1], "xy", [snap.link_x, snap.link_y], flush=True)
        if snap.screen != 0x06:
            rec = {"ok": False, "failed_room": "0x07", "reason": "other_mouth_not_0x06", "cellar": cellar_ex, "hops": hops, "pokes": False, "status_claim": None}
            w.write_dump("l5_whistle_path", rec)
            return rec
        d06 = w.dump_and_save_room(env, assist, total, "l5_06_arrive", "Level5Entered06", "Level5Entered07", "cellar 0x07 other mouth")
        checkpoints.append(d06["checkpoint"])

        west = w.key_west(env, assist, total, 0x05)
        hops.append({"hop": "0x06_key_west", "dest": west.get("dest"), "key_spent": west.get("key_spent"), "ok": west.get("ok")})
        print("KEYWEST", west, flush=True)
        if not west.get("ok"):
            rec = {"ok": False, "failed_room": "0x06", "reason": "key_west_not_0x05", "west": west, "hops": hops, "pokes": False, "status_claim": None}
            w.write_dump("l5_whistle_path", rec)
            return rec
        d05 = w.dump_and_save_room(env, assist, total, "l5_05_arrive", "Level5Entered05", "Level5Entered07", "0x06 WEST key")
        checkpoints.append(d05["checkpoint"])

        snap = read_snapshot(env.get_ram())
        n_dn = len(w.live_darknuts(snap))
        fight05 = w.fight_darknuts(env, assist, total, 0x05, expected=max(6, n_dn), source=0x06)
        idle(env, assist, total, 20)
        print("FIGHT05", fight05.get("ok"), "end", fight05.get("end_n"), "start", fight05.get("start_n"), flush=True)
        snap = read_snapshot(env.get_ram())
        if snap.screen != 0x05 or w.live_darknuts(snap):
            rec = {"ok": False, "failed_room": "0x05", "reason": "darknuts_not_cleared", "fight": {k: fight05[k] for k in fight05 if k != "controller"}, "now": w.dump_live(snap, env.get_ram()), "hops": hops, "pokes": False, "status_claim": None}
            w.write_dump("l5_whistle_path", rec)
            return rec
        cleared05 = w.dump_and_save_room(env, assist, total, "l5_05_cleared", "Level5Cleared05", "Level5Entered07", "0x05 6/6 darknuts")
        checkpoints.append(cleared05["checkpoint"])

        # Push the single block (secret=block_stairs), then walk onto stairs.
        pushed = w.push_blocks(env, assist, total, 0x05)
        print("PUSH05 took", pushed.get("took"), "blocks", pushed.get("blocks_seen"), flush=True)
        snap = read_snapshot(env.get_ram())
        if not (cellar(snap) or snap.screen == 0x04 or on_stair_tile(snap)):
            # Extra: stand east of block and push LEFT, then hunt center.
            blocks = [o for o in snap.objects if 1 <= o.slot <= 12 and o.type_id == 0x68]
            if blocks:
                bx, by = blocks[0].x, blocks[0].y
                w.walk_axis(env, assist, total, "y", by, max_f=240)
                w.walk_axis(env, assist, total, "x", bx + 16, max_f=240)
                push_dir(env, assist, total, "LEFT", frames=120)
                idle(env, assist, total, 10)
            for tx, ty in ((120, 141), (128, 141), (112, 141), (120, 144), (208, 96), (120, 125)):
                for _ in range(200):
                    snap = read_snapshot(env.get_ram())
                    if cellar(snap) or snap.screen == 0x04:
                        break
                    frame = walk_to_step(snap, tx, ty, y_first=True, tol=2)
                    if frame.reason == "walk_arrived":
                        idle(env, assist, total, 6)
                        break
                    w.step(env, assist, total, frame.action)
                snap = read_snapshot(env.get_ram())
                print("STAND05", [tx, ty], [snap.link_x, snap.link_y], "tile", snap.colliding_tile, "stair", on_stair_tile(snap), "mode", snap.mode, flush=True)
                if cellar(snap) or snap.screen == 0x04 or (on_stair_tile(snap) and snap.colliding_tile != 0x24):
                    for d in ("UP", "DOWN", "LEFT", "RIGHT"):
                        push_dir(env, assist, total, d, frames=70)
                        if cellar(read_snapshot(env.get_ram())) or read_snapshot(env.get_ram()).screen == 0x04:
                            break
                    break
        w.wait_play(env, assist, total, max_f=280)
        idle(env, assist, total, 16)
        snap = read_snapshot(env.get_ram())
        hops.append({"hop": "0x05_block_stairs", "dest": f"0x{snap.screen:02x}", "mode": snap.mode, "ok": snap.screen == 0x04 or cellar(snap)})
        print("STAIRS05", hops[-1], flush=True)
        if snap.screen != 0x04 and not cellar(snap):
            rec = {"ok": False, "failed_room": "0x05", "reason": "block_stairs_not_taken", "push": {k: v for k, v in pushed.items() if k != "log"}, "now": w.dump_live(snap, env.get_ram()), "hops": hops, "pokes": False, "status_claim": None}
            w.write_dump("l5_whistle_path", rec)
            return rec
        d04 = w.dump_and_save_room(env, assist, total, "l5_04_whistle", "Level5Entered04", "Level5Entered07", "0x05 block stairs")
        checkpoints.append(d04["checkpoint"])

        whistle_walk = w.hunt_item(env, assist, total, ADDR_WHISTLE)
        idle(env, assist, total, 12)
        whistle = int(read_u8(env.get_ram(), ADDR_WHISTLE))
        print("WHISTLE_WALK", whistle_walk.get("in"), "->", whistle_walk.get("out"), "now", whistle, flush=True)
        if whistle < 1:
            snap = read_snapshot(env.get_ram())
            w.walk_stands(env, assist, total, w.ITEM_WAYPOINTS, snap.screen, snap.mode)
            whistle = int(read_u8(env.get_ram(), ADDR_WHISTLE))
        snap = read_snapshot(env.get_ram())
        final04 = w.dump_live(snap, env.get_ram())
        png = w.shot(env, assist, total, "l5_04_whistle")
        w.write_dump("l5_04_whistle", {"via": "0x05 block stairs", "pokes": False, "status_claim": None, "arrive": d04["dump"], "walk": {k: v for k, v in whistle_walk.items() if k != "hits"}, "final": final04, "screenshot": png, "whistle_0x065C": whistle, "rom": w.rom_room(int(snap.screen))})
        if whistle < 1:
            rec = {"ok": False, "failed_room": "0x04", "reason": "whistle_still_0", "final04": final04, "hops": hops, "pokes": False, "status_claim": None}
            w.write_dump("l5_whistle_path", rec)
            return rec
        ckpt_w = w.save_ckpt(env, "Level5Whistle", "Level5Entered07", {"segment": "Level5Whistle", "via": "0x07->0x06 key->0x05 block->0x04", "key_poke": False, "door_poke": False, "bomb_count_poke": False, "selected_item_poke": False}, {"success": True, "room": int(snap.screen), "whistle_0x065C": whistle, "bombs": int(snap.bombs), "keys": int(snap.keys)})
        checkpoints.append(ckpt_w)
        hops.append({"hop": "0x04_whistle", "dest": f"0x{snap.screen:02x}", "whistle_0x065C": whistle, "ok": True})
    finally:
        env.close()

    boss = None
    if whistle >= 1:
        boss = w.digdogger_and_tf()
    report = {
        "ok": whistle >= 1,
        "failed_room": None,
        "status_claim": None,
        "pokes": False,
        "commands": ["PYTHONPATH=nes:. uv run python nes/zelda_i/scripts/_probe_l5_07_to_whistle.py"],
        "hops": hops,
        "checkpoints": checkpoints,
        "whistle_0x065C": whistle,
        "digdogger": None if boss is None else {"ok": boss.get("ok"), "tf_room": boss.get("tf_room"), "tf_l5": boss.get("tf_l5"), "triforce_0x0671": boss.get("triforce_0x0671"), "whistle_0x065C": boss.get("whistle_0x065C")},
    }
    w.write_dump("l5_whistle_path", report)
    return report


if __name__ == "__main__":
    r = main()
    print("OK", r.get("ok"), "HOPS", r.get("hops"), "WHISTLE", r.get("whistle_0x065C"), "DIG", r.get("digdogger"), "CKPT", r.get("checkpoints"), "status_claim", None)
