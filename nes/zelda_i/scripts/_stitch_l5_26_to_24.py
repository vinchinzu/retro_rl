"""Cleared26 west open → clear 0x25, then 0x25 west key → 0x24 door only.

One fceumm session. Do not fight Digdogger (type 0x38). Do not overwrite
Level5Cleared25 / Level5Cleared24 pins. Assisted (UnlimitedHealthAssist).
"""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon import DungeonPhase
from zelda_i.dungeon_ids import object_name
from zelda_i.level5_dungeon import (
    LEVEL_5,
    ROOM_25_SPEC,
    Level5PolsVoiceController,
    level5_in_room_24,
    level5_in_room_25,
    level5_room_25_cleared,
)
from zelda_i.level5_path import walk_west_from_25, walk_west_from_26
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import PLAY_MODE, read_snapshot, read_u8, ADDR_WHISTLE

TAG25 = "l5_cleared26_to_25_stitch"
TAG24 = "l5_cleared25_to_24_door_stitch"
DIGDOGGER = 0x38


def pin(env):
    s = read_snapshot(env.get_ram())
    return {
        "mode": s.mode,
        "level": s.level,
        "screen": s.screen,
        "screen_hex": f"0x{s.screen:02x}",
        "x": s.link_x,
        "y": s.link_y,
        "keys": s.keys,
        "bombs": s.bombs,
        "doors": s.cur_opened_doors,
        "mask": s.open_doorway_mask,
        "whistle": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
    }


def settle_room(env, assist, n, room: int, max_f: int = 240) -> bool:
    for _ in range(max_f):
        s = read_snapshot(env.get_ram())
        if s.level == LEVEL_5 and s.screen == room and s.mode == PLAY_MODE and not s.transitioning:
            return True
        env.step(nes_idle_action())
        n[0] += 1
        assist.apply_env(env, frame=n[0])
    return False


def live_objects(env):
    s = read_snapshot(env.get_ram())
    out = []
    for o in s.objects:
        if 1 <= o.slot <= 12 and o.type_id not in (0, 0xFF):
            out.append(
                {
                    "slot": o.slot,
                    "type": o.type_id,
                    "type_hex": f"0x{o.type_id:02x}",
                    "name": object_name(o.type_id),
                    "hp": o.hp,
                    "x": o.x,
                    "y": o.y,
                }
            )
    return out


def main():
    configure_headless()
    out = RECORDINGS_DIR / "stitches"
    movie = out / "bk2_26_to_25"
    movie.mkdir(parents=True, exist_ok=True)
    assist = UnlimitedHealthAssist(enabled=True)
    n = [0]
    hop26 = None
    fight = None
    hop25 = None
    start = None
    after25 = None
    after24 = None
    objs24 = None
    frames_at_25 = None
    cleared25 = False
    entered25 = False
    door24 = False
    fought_digdogger = False
    blocker = None
    obs = None
    shot25 = None
    shot24 = None

    env = make_env(GAME, "Level5Cleared26", GAME_DIR, render_mode="rgb_array", record=str(movie))
    try:
        obs, _ = reset_obs(env)
        obs, *_ = env.step(nes_idle_action())
        n[0] += 1
        assist.apply_env(env, frame=0)
        start = pin(env)
        print("start", start, flush=True)

        hop26 = walk_west_from_26(env, assist, n)
        print("hop26", hop26, pin(env), flush=True)
        settle_room(env, assist, n, 0x25)
        entered25 = level5_in_room_25(env.get_ram()) or hop26.get("dest") == 0x25
        print("after settle25", entered25, pin(env), flush=True)
        if not entered25:
            blocker = "fail_hop_26_to_25"
        else:
            ctl = Level5PolsVoiceController(spec=ROOM_25_SPEC)
            for _ in range(ROOM_25_SPEC.max_frames):
                assist.apply_env(env, frame=n[0])
                obs, *_ = env.step(ctl.step(read_snapshot(env.get_ram())).action)
                n[0] += 1
                if ctl.success or ctl.phase is DungeonPhase.FAILED:
                    break
            fight = ctl.report()
            cleared25 = level5_room_25_cleared(env.get_ram())
            after25 = pin(env)
            frames_at_25 = n[0]
            print("clear25", cleared25, fight.get("phase"), after25, flush=True)
            shot25 = out / f"{TAG25}_final.png"
            save_rgb_png(obs, shot25)
            if not cleared25:
                blocker = "fail_clear_25"
            else:
                hop25 = walk_west_from_25(env, assist, n)
                print("hop25", hop25, pin(env), flush=True)
                settle_room(env, assist, n, 0x24)
                door24 = level5_in_room_24(env.get_ram()) or hop25.get("dest") == 0x24
                after24 = pin(env)
                objs24 = live_objects(env)
                fought_digdogger = any(
                    o["type"] == DIGDOGGER and o["hp"] < 240 for o in objs24
                )
                print("door24", door24, "fought", fought_digdogger, after24, objs24, flush=True)
                shot24 = out / f"{TAG24}_final.png"
                save_rgb_png(obs, shot24)
                if not door24:
                    blocker = "fail_hop_25_to_24"

        if hasattr(env, "stop_record"):
            env.stop_record()
    finally:
        try:
            if hasattr(env, "stop_record"):
                env.stop_record()
        except Exception:
            pass
        env.close()

    bk2s = sorted(movie.glob("*.bk2"), key=lambda p: p.stat().st_mtime)
    bk2 = str(bk2s[-1]) if bk2s else None

    report25 = {
        "ok": bool(cleared25),
        "segment": TAG25,
        "continuous_emulator_session": True,
        "track": "assisted",
        "status_claim": False,
        "start_state": "Level5Cleared26",
        "end_claim": "Level5Cleared25" if cleared25 else None,
        "room_25": "cleared" if cleared25 else ("entered" if entered25 else "missed"),
        "total_frames": frames_at_25 if frames_at_25 is not None else n[0],
        "session_frames": n[0],
        "start": start,
        "hop": hop26,
        "fight": fight,
        "after25": after25,
        "png": str(shot25) if shot25 else None,
        "bk2": bk2,
        "blocker": blocker if blocker in ("fail_hop_26_to_25", "fail_clear_25") else None,
    }
    path25 = out / f"{TAG25}.json"
    write_json_report(path25, report25)
    print(f"wrote {path25}", flush=True)

    report24 = {
        "ok": bool(door24) and not fought_digdogger,
        "segment": TAG24,
        "continuous_emulator_session": True,
        "same_session_as": TAG25,
        "track": "assisted",
        "status_claim": False,
        "start_state": "Level5Cleared26 (chained after 26→25)",
        "end_claim": "entered_0x24_door_only" if door24 else None,
        "did_not_fight_digdogger": not fought_digdogger,
        "fought_digdogger": fought_digdogger,
        "hop": hop25,
        "after24": after24,
        "objects_at_24": objs24,
        "png": str(shot24) if shot24 else None,
        "bk2": bk2,
        "session_frames": n[0],
        "blocker": blocker if blocker == "fail_hop_25_to_24" else None,
    }
    path24 = out / f"{TAG24}.json"
    write_json_report(path24, report24)
    print(
        f"wrote {path24} frames={n[0]} bk2={bk2} "
        f"cleared25={cleared25} door24={door24} fought={fought_digdogger} blocker={blocker}",
        flush=True,
    )
    if blocker == "fail_hop_26_to_25":
        return 2
    if blocker == "fail_clear_25":
        return 3
    if blocker == "fail_hop_25_to_24":
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
