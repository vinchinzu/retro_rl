"""One-session Level5Whistle (0x04 cellar) -> Digdogger -> L5 TF bit 0x10.

Longest honest prefix. Leave 0x04, 0x05 east, 0x06 stairs, 0x07 cellar,
0x64 east, 0x65 bomb-east, clear 0x66, UP 0x56, walk 0x57 (no Zol clear),
0x47, 0x37, 0x27, west 0x26/0x25/0x24, whistle-shrink Digdogger, TF 0x10.
Skip 0x65 north. Assisted. One env + BK2. No Level5Complete / STATUS.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)
from retro_harness.video import VideoCaptureConfig, VideoRecorder
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon import DungeonPhase, GenericDungeonRoomController
from zelda_i.dungeon_ops import idle
from zelda_i.level5_dungeon import GIBDO_OBJECT_TYPE, LEVEL_5, ROOM_66_SPEC
from zelda_i.level5_path import (
    bomb_east_from_65,
    cellar_07_to_64,
    cellar_to_64,
    exit_whistle_04,
    take_block_stairs_06,
    take_center_stairs_06,
    walk_axis,
    walk_east_from_05,
    walk_east_from_64,
    walk_west_from_25,
    walk_west_from_26,
    walk_west_from_27,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_TRIFORCE, ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8
from zelda_i.scripts.run_level5_whistle_tf import (
    _digdogger_here,
    door,
    north_pinch,
    wait_play,
)

START = "Level5Whistle"
TAG = "l5_whistle04_to_tf_stitch"
ROOM_NAMES = {
    0x04: "Recorder / Whistle cellar",
    0x05: "six-Darknut",
    0x06: "empty passage east of 0x05",
    0x07: "cellar stairs (to Digdogger side)",
    0x14: "L5 triforce",
    0x24: "Digdogger",
    0x25: "west Pols Voice",
    0x26: "west Gibdos",
    0x27: "mixed Pols/Gibdo/Keese",
    0x37: "Darknuts + compass",
    0x47: "north Gibdos",
    0x56: "north Dodongos",
    0x57: "east Zols",
    0x64: "Blue Darknut stairs",
    0x65: "west Gibdo pocket",
    0x66: "3x Gibdo first key",
}


class _VideoEnv:
    """Record every controller step without changing the route helpers."""

    def __init__(self, env: Any, writer: VideoRecorder) -> None:
        self._env = env
        self._writer = writer
        self.video_frames = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def step(self, action: Any) -> Any:
        result = self._env.step(action)
        obs = result[0]
        self.video_frames += 1
        snap = read_snapshot(self._env.get_ram())
        self._writer.write_from_env(
            self._env,
            obs,
            action=action,
            frame_index=self.video_frames,
            room_id=int(snap.screen),
        )
        return result


def room_name(screen: int) -> str:
    return ROOM_NAMES.get(int(screen), f"room 0x{int(screen):02x}")


def pin(env) -> dict:
    s = read_snapshot(env.get_ram())
    tf = int(read_u8(env.get_ram(), ADDR_TRIFORCE))
    return {
        "mode": s.mode,
        "level": s.level,
        "screen": s.screen,
        "screen_hex": f"0x{s.screen:02x}",
        "name": room_name(s.screen) if s.level == 5 else None,
        "x": s.link_x,
        "y": s.link_y,
        "keys": int(s.keys),
        "bombs": int(s.bombs),
        "doors": int(s.cur_opened_doors),
        "mask": int(s.open_doorway_mask),
        "whistle": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
        "triforce": tf,
        "tf_hex": hex(tf),
        "tf_l5_bit": bool(tf & 0x10),
    }


def slim(rec):
    if not rec:
        return rec
    return {
        k: rec[k]
        for k in rec
        if k not in ("log", "steps", "progress", "menu", "before", "at_door", "after")
    }


def wait_room(env, assist, n, expect: int, max_f=240) -> bool:
    wait_play(env, assist, n, max_f=max_f)
    s = read_snapshot(env.get_ram())
    return s.level == LEVEL_5 and s.screen == expect and s.mode == PLAY_MODE


def fight_66(env, assist, n) -> dict:
    snap = read_snapshot(env.get_ram())
    live = [
        o
        for o in snap.objects
        if 1 <= o.slot <= 12 and o.type_id == GIBDO_OBJECT_TYPE and o.hp > 0
    ]
    if not live:
        return {"ok": True, "skipped": True, "end_n": 0}
    spec = replace(
        ROOM_66_SPEC,
        spec_id="level5_whistle04_66_gibdos",
        source_room=0x65,
        room_id=0x66,
        expected_enemy_count=len(live),
        required_open_doors=0,
        max_frames=20000,
        level=LEVEL_5,
    )
    ctl = GenericDungeonRoomController(spec)
    for _ in range(spec.max_frames):
        action = ctl.step(read_snapshot(env.get_ram()))
        env.step(action.action)
        n[0] += 1
        assist.apply_env(env, frame=n[0])
        if ctl.success or ctl.phase is DungeonPhase.FAILED:
            break
    leftover = [
        o
        for o in read_snapshot(env.get_ram()).objects
        if 1 <= o.slot <= 12 and o.type_id == GIBDO_OBJECT_TYPE and o.hp > 0
    ]
    return {
        "ok": bool(ctl.success) and not leftover,
        "frames": ctl.frames,
        "end_n": len(leftover),
        "phase": str(ctl.phase),
    }


def west_hop(fn, env, assist, n, expect: int) -> dict:
    rec = fn(env, assist, n)
    for _ in range(240):
        s = read_snapshot(env.get_ram())
        if s.mode == PLAY_MODE and s.screen == expect and not s.transitioning:
            break
        env.step(nes_idle_action())
        n[0] += 1
        assist.apply_env(env, frame=n[0])
    idle(env, assist, n, 8)
    s = read_snapshot(env.get_ram())
    rec = slim(rec)
    rec["dest"] = s.screen
    rec["mode"] = s.mode
    rec["xy"] = [s.link_x, s.link_y]
    rec["success"] = s.level == LEVEL_5 and s.screen == expect and s.mode == PLAY_MODE
    return rec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Optional MP4 path; captures the same one-session run as the BK2.",
    )
    parser.add_argument("--no-audio", action="store_true")
    args = parser.parse_args(argv)

    configure_headless()
    out = RECORDINGS_DIR / "stitches"
    movie = out / "bk2_whistle04_to_tf"
    movie.mkdir(parents=True, exist_ok=True)
    assist = UnlimitedHealthAssist(enabled=True)
    n = [0]
    hops = []
    rooms = []
    blocker = None
    boss = None
    start = None
    after = None
    writer: VideoRecorder | None = None
    video_frames = 0

    env = make_env(GAME, START, GAME_DIR, render_mode="rgb_array", record=str(movie))
    try:
        first_obs, _ = reset_obs(env)
        if args.video is not None:
            audio_rate: int | None = None
            if not args.no_audio and hasattr(env.em, "get_audio_rate"):
                audio_rate = int(env.em.get_audio_rate())
            writer = VideoRecorder(
                args.video,
                width=int(first_obs.shape[1]),
                height=int(first_obs.shape[0]),
                config=VideoCaptureConfig(
                    audio=audio_rate is not None,
                    footer=True,
                    scale=2,
                    crf=17,
                    preset="medium",
                ),
                audio_rate=audio_rate,
            )
            env = _VideoEnv(env, writer)
        env.step(nes_idle_action())
        n[0] += 1
        assist.apply_env(env, frame=0)
        idle(env, assist, n, 12)
        start = pin(env)
        after = start
        rooms.append(f"{start['screen_hex']} {start['name']}")
        print("START", start, flush=True)
        if not (
            start["whistle"] == 1
            and start["level"] == LEVEL_5
            and start["screen"] == 0x04
        ):
            blocker = "start_not_whistle04"
        else:
            leave = exit_whistle_04(env, assist, n)
            hops.append({"hop": "04_leave", **slim(leave)})
            print("LEAVE04", leave.get("success"), pin(env), flush=True)
            if not wait_room(env, assist, n, 0x05):
                blocker = "fail_leave_04_to_05"
            else:
                rooms.append("0x05 six-Darknut")
                rec = walk_east_from_05(env, assist, n)
                hops.append({"hop": "05_east", **slim(rec)})
                print("EAST05", rec.get("success"), pin(env), flush=True)
                if not wait_room(env, assist, n, 0x06):
                    blocker = "fail_hop_05_to_06"
                else:
                    rooms.append("0x06 empty passage east of 0x05")
                    stairs = take_block_stairs_06(env, assist, n)
                    if not stairs.get("success"):
                        stairs = take_center_stairs_06(env, assist, n)
                    hops.append({"hop": "06_stairs", **slim(stairs)})
                    print("STAIRS06", stairs.get("success"), pin(env), flush=True)
                    snap = read_snapshot(env.get_ram())
                    if not (
                        stairs.get("success")
                        or snap.mode in (9, 10, 11, 16)
                        or snap.screen == 0x07
                    ):
                        blocker = "fail_06_stairs"
                    else:
                        rooms.append("0x07 cellar stairs (to Digdogger side)")
                        cellar = cellar_07_to_64(env, assist, n)
                        hops.append({"hop": "07_to_64", **slim(cellar)})
                        print("CELLAR", cellar.get("success"), pin(env), flush=True)
                        if not wait_room(env, assist, n, 0x64):
                            cellar = cellar_to_64(env, assist, n)
                            hops.append({"hop": "07_to_64_retry", **slim(cellar)})
                            if not wait_room(env, assist, n, 0x64):
                                blocker = "fail_cellar_to_64"
                        if blocker is None:
                            rooms.append("0x64 Blue Darknut stairs")
                            rec = walk_east_from_64(env, assist, n)
                            hops.append({"hop": "64_east", **slim(rec)})
                            print("EAST64", rec.get("success"), pin(env), flush=True)
                            if not wait_room(env, assist, n, 0x65):
                                blocker = "fail_hop_64_to_65"
                            else:
                                rooms.append("0x65 west Gibdo pocket")
                                be = bomb_east_from_65(env, assist, n)
                                hops.append({"hop": "65_bomb_east", **slim(be)})
                                after = pin(env)
                                print("BOMB65", hops[-1], after, flush=True)
                                if after["screen"] != 0x66:
                                    blocker = "bomb_east_not_0x66"
                                else:
                                    rooms.append("0x66 3x Gibdo first key")
                                    clr = fight_66(env, assist, n)
                                    hops.append({"hop": "66_clear", **clr})
                                    after = pin(env)
                                    print("CLEAR66", clr, after, flush=True)
                                    if not clr.get("ok"):
                                        blocker = "66_gibdos_not_cleared"
                                    else:
                                        rec = door(env, assist, n, "UP", 0x56)
                                        hops.append({"hop": "66_up", **slim(rec)})
                                        after = pin(env)
                                        print("UP66", rec.get("ok"), after, flush=True)
                                        if not rec.get("ok"):
                                            idle(env, assist, n, 80)
                                            walk_axis(
                                                env, assist, n, "y", 173, max_f=300
                                            )
                                            walk_axis(
                                                env, assist, n, "x", 120, max_f=400
                                            )
                                            rec = door(
                                                env,
                                                assist,
                                                n,
                                                "UP",
                                                0x56,
                                                x_force=120,
                                                y_force=93,
                                                push=280,
                                            )
                                            hops.append(
                                                {"hop": "66_up_retry", **slim(rec)}
                                            )
                                            after = pin(env)
                                            print(
                                                "UP66_RETRY",
                                                rec.get("ok"),
                                                after,
                                                flush=True,
                                            )
                                        if not rec.get("ok"):
                                            blocker = "north66_not_0x56"
                                        else:
                                            rooms.append("0x56 north Dodongos")
                                            rec = door(env, assist, n, "RIGHT", 0x57)
                                            hops.append({"hop": "56_east", **slim(rec)})
                                            after = pin(env)
                                            print(
                                                "EAST56",
                                                rec.get("ok"),
                                                after,
                                                flush=True,
                                            )
                                            if not rec.get("ok"):
                                                blocker = "east56_not_0x57"
                                            else:
                                                rooms.append("0x57 east Zols")
                                                rec = door(env, assist, n, "UP", 0x47)
                                                hops.append(
                                                    {"hop": "57_up", **slim(rec)}
                                                )
                                                after = pin(env)
                                                print(
                                                    "UP57",
                                                    rec.get("ok"),
                                                    after,
                                                    flush=True,
                                                )
                                                if not rec.get("ok"):
                                                    walk_axis(
                                                        env,
                                                        assist,
                                                        n,
                                                        "x",
                                                        32,
                                                        max_f=400,
                                                    )
                                                    walk_axis(
                                                        env,
                                                        assist,
                                                        n,
                                                        "y",
                                                        141,
                                                        max_f=200,
                                                    )
                                                    walk_axis(
                                                        env,
                                                        assist,
                                                        n,
                                                        "x",
                                                        120,
                                                        max_f=400,
                                                    )
                                                    rec = door(
                                                        env,
                                                        assist,
                                                        n,
                                                        "UP",
                                                        0x47,
                                                        x_force=120,
                                                        y_force=93,
                                                        push=280,
                                                    )
                                                    hops.append(
                                                        {
                                                            "hop": "57_up_bank",
                                                            **slim(rec),
                                                        }
                                                    )
                                                    after = pin(env)
                                                    print(
                                                        "UP57_BANK",
                                                        rec.get("ok"),
                                                        after,
                                                        flush=True,
                                                    )
                                                if not rec.get("ok"):
                                                    blocker = "fail_hop_57_to_47"
                                                else:
                                                    rooms.append("0x47 north Gibdos")
                                                    rec = north_pinch(
                                                        env, assist, n, 0x37
                                                    )
                                                    hops.append(
                                                        {"hop": "47_up", **slim(rec)}
                                                    )
                                                    after = pin(env)
                                                    print(
                                                        "UP47",
                                                        rec.get("ok"),
                                                        after,
                                                        flush=True,
                                                    )
                                                    if not rec.get("ok"):
                                                        blocker = "fail_hop_47_to_37"
                                                    else:
                                                        rooms.append(
                                                            "0x37 Darknuts + compass"
                                                        )
                                                        rec = north_pinch(
                                                            env, assist, n, 0x27
                                                        )
                                                        hops.append(
                                                            {
                                                                "hop": "37_up",
                                                                **slim(rec),
                                                            }
                                                        )
                                                        after = pin(env)
                                                        print(
                                                            "UP37",
                                                            rec.get("ok"),
                                                            after,
                                                            flush=True,
                                                        )
                                                        if not rec.get("ok"):
                                                            blocker = (
                                                                "fail_hop_37_to_27"
                                                            )
                                                        else:
                                                            rooms.append(
                                                                "0x27 mixed Pols/Gibdo/Keese"
                                                            )
                                                            rec = west_hop(
                                                                walk_west_from_27,
                                                                env,
                                                                assist,
                                                                n,
                                                                0x26,
                                                            )
                                                            hops.append(
                                                                {
                                                                    "hop": "27_west",
                                                                    **rec,
                                                                }
                                                            )
                                                            after = pin(env)
                                                            print(
                                                                "WEST27",
                                                                rec.get("success"),
                                                                after,
                                                                flush=True,
                                                            )
                                                            if not rec.get("success"):
                                                                blocker = (
                                                                    "fail_hop_27_to_26"
                                                                )
                                                            else:
                                                                rooms.append(
                                                                    "0x26 west Gibdos"
                                                                )
                                                                rec = west_hop(
                                                                    walk_west_from_26,
                                                                    env,
                                                                    assist,
                                                                    n,
                                                                    0x25,
                                                                )
                                                                hops.append(
                                                                    {
                                                                        "hop": "26_west",
                                                                        **rec,
                                                                    }
                                                                )
                                                                after = pin(env)
                                                                print(
                                                                    "WEST26",
                                                                    rec.get("success"),
                                                                    after,
                                                                    flush=True,
                                                                )
                                                                if not rec.get(
                                                                    "success"
                                                                ):
                                                                    blocker = "fail_hop_26_to_25"
                                                                else:
                                                                    rooms.append(
                                                                        "0x25 west Pols Voice"
                                                                    )
                                                                    rec = west_hop(
                                                                        walk_west_from_25,
                                                                        env,
                                                                        assist,
                                                                        n,
                                                                        0x24,
                                                                    )
                                                                    hops.append(
                                                                        {
                                                                            "hop": "25_west",
                                                                            **rec,
                                                                        }
                                                                    )
                                                                    after = pin(env)
                                                                    print(
                                                                        "WEST25",
                                                                        rec.get(
                                                                            "success"
                                                                        ),
                                                                        after,
                                                                        flush=True,
                                                                    )
                                                                    if not rec.get(
                                                                        "success"
                                                                    ):
                                                                        blocker = "fail_hop_25_to_24"
                                                                    else:
                                                                        rooms.append(
                                                                            "0x24 Digdogger"
                                                                        )
                                                                        boss = _digdogger_here(
                                                                            env,
                                                                            assist,
                                                                            n,
                                                                        )
                                                                        hops.append(
                                                                            {
                                                                                "hop": "digdogger",
                                                                                "ok": boss.get(
                                                                                    "ok"
                                                                                ),
                                                                                "killed": boss.get(
                                                                                    "killed"
                                                                                ),
                                                                                "fight_ok": (
                                                                                    boss.get(
                                                                                        "fight"
                                                                                    )
                                                                                    or {}
                                                                                ).get(
                                                                                    "ok"
                                                                                ),
                                                                                "tf_l5": boss.get(
                                                                                    "tf_l5"
                                                                                ),
                                                                                "tf_out": boss.get(
                                                                                    "tf_out"
                                                                                ),
                                                                                "room": boss.get(
                                                                                    "room"
                                                                                ),
                                                                            }
                                                                        )
                                                                        after = pin(env)
                                                                        print(
                                                                            "DIGDOGGER",
                                                                            hops[-1],
                                                                            after,
                                                                            flush=True,
                                                                        )
                                                                        if after.get(
                                                                            "tf_l5_bit"
                                                                        ) or boss.get(
                                                                            "tf_l5"
                                                                        ):
                                                                            rooms.append(
                                                                                f"{after['screen_hex']} {after['name']}"
                                                                            )
                                                                        else:
                                                                            blocker = "tf_bit_0x10_not_set"

        obs, *_ = env.step(nes_idle_action())
        n[0] += 1
        png = out / f"{TAG}_final.png"
        save_rgb_png(obs, png)
        if hasattr(env, "stop_record"):
            env.stop_record()
    finally:
        video_frames = int(getattr(env, "video_frames", 0))
        try:
            if hasattr(env, "stop_record"):
                env.stop_record()
        except Exception:
            pass
        env.close()
        if writer is not None:
            writer.close()

    bk2s = sorted(movie.glob("*.bk2"), key=lambda p: p.stat().st_mtime)
    tf = after.get("triforce") if after else None
    tf_l5 = bool(after and after.get("tf_l5_bit"))
    fight_ok = bool(boss and ((boss.get("fight") or {}).get("ok") or boss.get("ok")))
    report = {
        "ok": tf_l5 and blocker is None,
        "segment": TAG,
        "continuous_emulator_session": True,
        "track": "assisted",
        "status_claim": False,
        "level5_complete_claim": False,
        "start_state": START,
        "start": start,
        "final": after,
        "hops": hops,
        "room_sequence": rooms,
        "whistle_0x065C": None if after is None else after.get("whistle"),
        "triforce_0x0671": tf,
        "tf_hex": None if tf is None else hex(tf),
        "tf_l5_bit_0x10_real": tf_l5,
        "digdogger_dead": fight_ok and tf_l5,
        "boss": None
        if boss is None
        else {k: boss[k] for k in boss if k not in ("after_whistle_objs",)},
        "total_frames": n[0],
        "blocker": blocker,
        "bk2": str(bk2s[-1]) if bk2s else None,
        "png": str(png),
        "video": (
            None
            if args.video is None
            else {
                "path": str(args.video.resolve()),
                "encoded_frames": video_frames,
                "audio": not args.no_audio,
            }
        ),
        "assist": assist.report(),
        "health_drop_note": {
            "total_damage": assist.report().get("total_damage", 0),
            "damage_by_location": assist.report().get("damage_by_location", {}),
            "fix_later": True,
        },
        "pokes": False,
        "resource_pokes": [],
        "path_note": (
            "Longest prefix from Level5Whistle 0x04. Skip 0x65 north. "
            "Walk 0x57 (no Zol clear — y=125 ladder lock). Whistle-shrink Digdogger. "
            "No Level5Complete / STATUS claim."
        ),
    }
    path = out / f"{TAG}.json"
    write_json_report(path, report)
    print(
        f"wrote {path} frames={n[0]} dest={after and after.get('screen_hex')} "
        f"tf={tf} tf_l5={tf_l5} blocker={blocker} bk2={report['bk2']}",
        flush=True,
    )
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
