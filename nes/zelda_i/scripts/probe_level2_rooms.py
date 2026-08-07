"""Live recon of Level 2 rooms 0x7d (entry) → 0x6d (Ropes north).

Assisted-friendly (``--infinite-life``). Records object type IDs, HP vs
type-liveness, spawn delay, door bits before/after clear, RoomItemId.

Examples::

    uv run python zelda_i/scripts/probe_level2_rooms.py --infinite-life
    uv run python zelda_i/scripts/probe_level2_rooms.py --from-state Level2Entrance \\
        --infinite-life --tag l2_recon
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from retro_harness.env import make_env, save_state
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ids import object_name, room_item_name
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import PLAY_MODE, ZeldaObject, ZeldaSnapshot, read_snapshot

ROOM_ENTRY = 0x7D
ROOM_ROPES = 0x6D
ROPE_TYPE = 0x28  # walkthrough-correlated; confirmed by this probe when live
LEVEL_2 = 2

# Door bit layout (ADDR_CUR_OPENED_DOORS / 0x00EE): bit0=R bit1=L bit2=D bit3=U
DOOR_RIGHT = 0x01
DOOR_LEFT = 0x02
DOOR_DOWN = 0x04
DOOR_UP = 0x08


def _objs(snap: ZeldaSnapshot, *, slots_only: bool = True) -> list[dict]:
    out: list[dict] = []
    for o in snap.objects:
        if slots_only and not (1 <= o.slot <= 10):
            continue
        if o.type_id in (0, 0xFF) and o.y == 0:
            continue
        if o.type_id in (0, 0xFF):
            continue
        out.append(
            {
                "slot": o.slot,
                "type": o.type_id,
                "type_name": object_name(o.type_id),
                "x": o.x,
                "y": o.y,
                "hp": o.hp,
                "facing": o.facing,
                "state": o.state,
            }
        )
    return out


def _room_fields(snap: ZeldaSnapshot) -> dict:
    return {
        "mode": snap.mode,
        "level": snap.level,
        "screen": snap.screen,
        "x": snap.link_x,
        "y": snap.link_y,
        "facing": snap.facing,
        "health": snap.health,
        "keys": snap.keys,
        "bombs": snap.bombs,
        "sword": snap.sword,
        "triforce": snap.triforce,
        "room_item_id": snap.room_item_id,
        "room_item_name": room_item_name(snap.room_item_id),
        "room_all_dead": snap.room_all_dead,
        "room_obj_count": snap.room_obj_count,
        "cur_opened_doors": snap.cur_opened_doors,
        "cur_opened_doors_bits": {
            "R": bool(snap.cur_opened_doors & DOOR_RIGHT),
            "L": bool(snap.cur_opened_doors & DOOR_LEFT),
            "D": bool(snap.cur_opened_doors & DOOR_DOWN),
            "U": bool(snap.cur_opened_doors & DOOR_UP),
            "raw": snap.cur_opened_doors,
        },
        "open_doorway_mask": snap.open_doorway_mask,
        "objects": _objs(snap),
        "type_counts": dict(
            Counter(o["type"] for o in _objs(snap))
        ),
    }


def _live_typed(
    snap: ZeldaSnapshot, types: set[int], *, hp_required: bool
) -> tuple[ZeldaObject, ...]:
    enemies = tuple(
        o
        for o in snap.objects
        if 1 <= o.slot <= 10 and o.type_id in types
    )
    if hp_required:
        return tuple(o for o in enemies if o.hp > 0)
    return enemies


def _swing(frames: int, direction: str, *, period: int = 8, hold: int = 3):
    if frames % period < hold:
        return nes_action(direction, "A")
    return nes_action(direction)


def _move_toward(snap: ZeldaSnapshot, tx: int, ty: int, *, tol: int = 4) -> str | None:
    dx = tx - snap.link_x
    dy = ty - snap.link_y
    if abs(dx) <= tol and abs(dy) <= tol:
        return None
    if abs(dx) > tol and abs(dx) >= abs(dy):
        return "RIGHT" if dx > 0 else "LEFT"
    return "DOWN" if dy > 0 else "UP"


def run_recon(
    *,
    start_state: str,
    infinite_life: bool,
    idle_entry_frames: int,
    max_north_frames: int,
    max_fight_frames: int,
    try_left: bool,
    save_checkpoint: bool,
    tag: str,
) -> dict:
    configure_headless()
    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True) if infinite_life else None
    track = "assisted" if infinite_life else "clean"
    try:
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        obs, *_ = env.step(nes_idle_action())
        if assist is not None:
            assist.apply_env(env, frame=0)

        snap = read_snapshot(env.get_ram())
        entry0 = _room_fields(snap)
        if not (
            snap.level == LEVEL_2
            and snap.mode == PLAY_MODE
            and snap.screen == ROOM_ENTRY
        ):
            png = RECORDINGS_DIR / f"{tag}_not_entry.png"
            save_rgb_png(obs, png)
            return {
                "ok": False,
                "track": track,
                "error": "not_in_level2_entry_0x7d",
                "entry": entry0,
                "screenshot": str(png),
            }

        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_7d_t0.png")

        # --- Phase A: idle on 0x7d, sample spawn / doors ---
        entry_timeline: list[dict] = []
        max_types_7d: Counter[int] = Counter()
        for f in range(idle_entry_frames):
            snap = read_snapshot(env.get_ram())
            types = Counter(o.type_id for o in snap.objects if 1 <= o.slot <= 10 and o.type_id not in (0, 0xFF))
            if types:
                max_types_7d |= types
                if len(entry_timeline) < 8 or f % 30 == 0:
                    entry_timeline.append({"f": f, **_room_fields(snap)})
            obs, *_ = env.step(nes_idle_action())
            if assist is not None:
                assist.apply_env(env, frame=f + 1)

        snap = read_snapshot(env.get_ram())
        entry_after_idle = _room_fields(snap)
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_7d_idle.png")

        # --- Phase B: walk north door into 0x6d; sample spawn during scroll ---
        north_log: list[dict] = []
        arrived_6d_frame: int | None = None
        first_screen_6d_frame: int | None = None
        spawn_first_frame: int | None = None  # frames after first screen==0x6d
        spawn_type_peak: Counter[int] = Counter()
        spawn_timeline: list[dict] = []
        hp_samples: list[dict] = []
        for f in range(max_north_frames):
            snap = read_snapshot(env.get_ram())
            if snap.mode == 17:
                break
            if snap.screen == ROOM_ROPES and first_screen_6d_frame is None:
                first_screen_6d_frame = f
                north_log.append(
                    {"f": f, "event": "screen_6d", **_room_fields(snap)}
                )
            if snap.screen == ROOM_ROPES and first_screen_6d_frame is not None:
                enemies = [
                    o
                    for o in snap.objects
                    if 1 <= o.slot <= 10 and o.type_id not in (0, 0xFF)
                ]
                type_c = Counter(o.type_id for o in enemies)
                if type_c:
                    spawn_type_peak |= type_c
                    if spawn_first_frame is None:
                        spawn_first_frame = f - first_screen_6d_frame
                        spawn_timeline.append(
                            {
                                "f": f,
                                "f_after_screen": spawn_first_frame,
                                "event": "first_spawn",
                                **_room_fields(snap),
                            }
                        )
                        for o in enemies:
                            hp_samples.append(
                                {
                                    "f_after_screen": spawn_first_frame,
                                    "slot": o.slot,
                                    "type": o.type_id,
                                    "hp": o.hp,
                                    "x": o.x,
                                    "y": o.y,
                                    "mode": snap.mode,
                                }
                            )
            if (
                snap.level == LEVEL_2
                and snap.mode == PLAY_MODE
                and snap.screen == ROOM_ROPES
                and arrived_6d_frame is None
            ):
                arrived_6d_frame = f
                north_log.append({"f": f, "event": "arrived_6d", **_room_fields(snap)})
                save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_6d_enter.png")
                # continue a short settle sample after play mode
                break
            if snap.transitioning or snap.mode != PLAY_MODE:
                act = nes_action("UP") if snap.mode in (4, 6, 7, 16) else nes_idle_action()
            else:
                # Center then press north doorway (~y 93)
                d = _move_toward(snap, 120, 93, tol=6)
                if d is None or snap.link_y <= 100:
                    act = nes_action("UP")
                else:
                    act = nes_action(d)
            obs, *_ = env.step(act)
            if assist is not None:
                assist.apply_env(env, frame=idle_entry_frames + f + 1)
            if f % 40 == 0:
                north_log.append({"f": f, **_room_fields(snap)})

        # Extra settle frames on 0x6d play mode to let HP activate
        for f in range(120):
            snap = read_snapshot(env.get_ram())
            enemies = [
                o
                for o in snap.objects
                if 1 <= o.slot <= 10 and o.type_id not in (0, 0xFF)
            ]
            type_c = Counter(o.type_id for o in enemies)
            if type_c:
                spawn_type_peak |= type_c
                if spawn_first_frame is None and first_screen_6d_frame is not None:
                    spawn_first_frame = (arrived_6d_frame or 0) + f - (
                        first_screen_6d_frame or 0
                    )
            if f in (0, 20, 40, 80) or (
                enemies and all(o.hp > 0 for o in enemies)
            ):
                spawn_timeline.append(
                    {
                        "f_settle": f,
                        "event": "settle_sample",
                        **_room_fields(snap),
                    }
                )
            if enemies and all(o.hp > 0 for o in enemies) and f >= 20:
                break
            d = _move_toward(snap, 120, 150, tol=12)
            act = nes_action(d) if d else nes_idle_action()
            obs, *_ = env.step(act)
            if assist is not None:
                assist.apply_env(env, frame=idle_entry_frames + max_north_frames + f)

        snap = read_snapshot(env.get_ram())
        if arrived_6d_frame is None:
            png = RECORDINGS_DIR / f"{tag}_north_fail.png"
            save_rgb_png(obs, png)
            return {
                "ok": False,
                "track": track,
                "error": "failed_north_to_0x6d",
                "entry_t0": entry0,
                "entry_after_idle": entry_after_idle,
                "entry_timeline": entry_timeline,
                "max_types_7d": {f"0x{k:02x}": v for k, v in max_types_7d.items()},
                "north_log": north_log,
                "final": _room_fields(snap),
                "screenshot": str(png),
            }

        doors_before_clear = snap.cur_opened_doors
        pre_clear = _room_fields(snap)
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_6d_spawned.png")

        # Liveness rule probe: compare type-only vs type+hp counts at peak
        type_only = len(
            _live_typed(
                snap,
                set(spawn_type_peak.keys()) or {ROPE_TYPE},
                hp_required=False,
            )
        )
        type_and_hp = len(
            _live_typed(
                snap,
                set(spawn_type_peak.keys()) or {ROPE_TYPE},
                hp_required=True,
            )
        )
        # Ropes show type during mode-4 settle with HP=0, then HP=0x10 in play.
        # Prefer type_and_hp once any live HP is observed; else type-only.
        alive_rule = (
            "type_and_hp"
            if type_and_hp > 0
            else ("type" if type_only > 0 else "unknown")
        )

        # --- Phase D: clear room with simple chase+swing ---
        enemy_types = set(spawn_type_peak.keys()) or {ROPE_TYPE}
        max_live = 0
        fight_log: list[dict] = []
        clear_frame: int | None = None
        doors_at_clear: int | None = None
        all_dead_at_clear: int | None = None
        for f in range(max_fight_frames):
            snap = read_snapshot(env.get_ram())
            if snap.mode == 17:
                fight_log.append({"f": f, "event": "death", **_room_fields(snap)})
                break
            if snap.screen != ROOM_ROPES:
                fight_log.append(
                    {"f": f, "event": "left_room", **_room_fields(snap)}
                )
                break
            live = _live_typed(snap, enemy_types, hp_required=(alive_rule == "type_and_hp"))
            max_live = max(max_live, len(live))
            if (
                not live
                and max_live >= 1
                and snap.room_all_dead >= 20
            ):
                clear_frame = f
                doors_at_clear = snap.cur_opened_doors
                all_dead_at_clear = snap.room_all_dead
                fight_log.append({"f": f, "event": "cleared", **_room_fields(snap)})
                break
            if snap.transitioning or snap.mode != PLAY_MODE:
                act = nes_idle_action()
            elif live:
                target = min(
                    live,
                    key=lambda o: abs(o.x - snap.link_x) + abs(o.y - snap.link_y),
                )
                dx = target.x - snap.link_x
                dy = target.y - snap.link_y
                if abs(dx) > 10:
                    d = "RIGHT" if dx > 0 else "LEFT"
                elif abs(dy) > 10:
                    d = "DOWN" if dy > 0 else "UP"
                else:
                    d = "RIGHT" if dx >= 0 else "LEFT"
                act = _swing(f, d)
            else:
                # roam while waiting for spawn / all_dead settle
                d = _move_toward(snap, 120 + (f // 40 % 3 - 1) * 40, 141, tol=6)
                act = _swing(f, d or "UP", period=12, hold=2)
            obs, *_ = env.step(act)
            if assist is not None:
                assist.apply_env(
                    env,
                    frame=idle_entry_frames + max_north_frames + 400 + f,
                )
            if f % 60 == 0:
                fight_log.append(
                    {
                        "f": f,
                        "live": len(live),
                        "all_dead": snap.room_all_dead,
                        "doors": snap.cur_opened_doors,
                        **{k: _room_fields(snap)[k] for k in (
                            "x", "y", "type_counts", "objects"
                        )},
                    }
                )

        snap = read_snapshot(env.get_ram())
        post_clear = _room_fields(snap)
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_6d_clear.png")

        # Settle a bit more for door bits / item
        for f in range(90):
            obs, *_ = env.step(nes_idle_action())
            if assist is not None:
                assist.apply_env(env, frame=9000 + f)
        snap = read_snapshot(env.get_ram())
        post_settle = _room_fields(snap)
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_6d_settle.png")

        checkpoint = None
        if save_checkpoint and clear_frame is not None and snap.screen == ROOM_ROPES:
            path = save_state(env, GAME_DIR, GAME, "Level2RopesCleared")
            checkpoint = str(path)

        left_try = None
        if try_left and snap.screen == ROOM_ROPES and snap.mode == PLAY_MODE:
            left_log: list[dict] = []
            left_room: int | None = None
            # Door center is mid-height (~141). Approach x then hold LEFT.
            for f in range(700):
                snap = read_snapshot(env.get_ram())
                if snap.screen != ROOM_ROPES and snap.level == LEVEL_2:
                    left_room = snap.screen
                    left_log.append(
                        {"f": f, "event": "left_screen", **_room_fields(snap)}
                    )
                    if snap.mode == PLAY_MODE:
                        left_log.append(
                            {"f": f, "event": "entered", **_room_fields(snap)}
                        )
                        break
                if snap.transitioning or snap.mode in (4, 6, 7):
                    act = nes_action("LEFT")
                else:
                    # Align y to mid-door first; avoid hugging wall too early.
                    if abs(snap.link_y - 141) > 4:
                        act = nes_action("DOWN" if snap.link_y < 141 else "UP")
                    elif snap.link_x > 40:
                        act = nes_action("LEFT")
                    else:
                        act = nes_action("LEFT")
                obs, *_ = env.step(act)
                if assist is not None:
                    assist.apply_env(env, frame=9500 + f)
                if f % 50 == 0 or snap.transitioning:
                    left_log.append({"f": f, **_room_fields(snap)})
            save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_left_try.png")
            left_try = {
                "reached_room": left_room,
                "final": _room_fields(read_snapshot(env.get_ram())),
                "log": left_log[:16],
            }

        ok = (
            arrived_6d_frame is not None
            and spawn_first_frame is not None
            and clear_frame is not None
        )
        report = {
            "ok": ok,
            "track": track,
            "start_state": start_state,
            "infinite_life": infinite_life,
            "assist": assist.report() if assist else None,
            "entry_t0": entry0,
            "entry_after_idle": entry_after_idle,
            "entry_timeline": entry_timeline,
            "max_types_7d": {f"0x{k:02x}": v for k, v in max_types_7d.items()},
            "north": {
                "arrived_frame": arrived_6d_frame,
                "log_head": north_log[:8],
            },
            "room_6d": {
                "first_screen_6d_frame": first_screen_6d_frame,
                "arrived_play_frame": arrived_6d_frame,
                "spawn_first_frame_after_screen": spawn_first_frame,
                "spawn_type_peak": {
                    f"0x{k:02x}": v for k, v in spawn_type_peak.items()
                },
                "spawn_type_names": {
                    f"0x{k:02x}": object_name(k) for k in spawn_type_peak
                },
                "alive_rule": alive_rule,
                "type_only_count_at_sample": type_only,
                "type_and_hp_count_at_sample": type_and_hp,
                "hp_samples_first_spawn": hp_samples[:12],
                "doors_before_clear": doors_before_clear,
                "doors_at_clear": doors_at_clear,
                "all_dead_at_clear": all_dead_at_clear,
                "max_live": max_live,
                "clear_frame": clear_frame,
                "pre_clear": pre_clear,
                "post_clear": post_clear,
                "post_settle": post_settle,
                "spawn_timeline": spawn_timeline[:20],
                "fight_log_tail": fight_log[-8:],
            },
            "left_try": left_try,
            "checkpoint": checkpoint,
            "screenshots": {
                "7d_t0": str(RECORDINGS_DIR / f"{tag}_7d_t0.png"),
                "7d_idle": str(RECORDINGS_DIR / f"{tag}_7d_idle.png"),
                "6d_enter": str(RECORDINGS_DIR / f"{tag}_6d_enter.png"),
                "6d_spawned": str(RECORDINGS_DIR / f"{tag}_6d_spawned.png"),
                "6d_clear": str(RECORDINGS_DIR / f"{tag}_6d_clear.png"),
                "6d_settle": str(RECORDINGS_DIR / f"{tag}_6d_settle.png"),
            },
            "final": _room_fields(read_snapshot(env.get_ram())),
        }
        return report
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-state",
        default="Level2Entrance",
        help="stable-retro state name (default Level2Entrance room-ready 0x7d)",
    )
    parser.add_argument("--infinite-life", action="store_true")
    parser.add_argument("--idle-entry-frames", type=int, default=120)
    parser.add_argument("--max-north-frames", type=int, default=600)
    parser.add_argument("--max-fight-frames", type=int, default=5000)
    parser.add_argument("--try-left", action="store_true", default=True)
    parser.add_argument("--no-try-left", action="store_false", dest="try_left")
    parser.add_argument("--save-state", action="store_true")
    parser.add_argument("--tag", default="l2_recon")
    parser.add_argument("--trials", type=int, default=1)
    args = parser.parse_args(argv)

    reports = []
    for trial in range(args.trials):
        tag = f"{args.tag}_t{trial}" if args.trials > 1 else args.tag
        report = run_recon(
            start_state=args.from_state,
            infinite_life=args.infinite_life,
            idle_entry_frames=args.idle_entry_frames,
            max_north_frames=args.max_north_frames,
            max_fight_frames=args.max_fight_frames,
            try_left=args.try_left,
            save_checkpoint=args.save_state,
            tag=tag,
        )
        reports.append(report)
        r6 = report.get("room_6d") or {}
        print(
            f"trial={trial} ok={report.get('ok')} "
            f"entry_room={report.get('entry_t0', {}).get('screen')} "
            f"spawn_after_sc={r6.get('spawn_first_frame_after_screen')} "
            f"types={r6.get('spawn_type_peak')} "
            f"alive={r6.get('alive_rule')} "
            f"doors_before={r6.get('doors_before_clear')} "
            f"doors_clear={r6.get('doors_at_clear')} "
            f"max_live={r6.get('max_live')} "
            f"clear_f={r6.get('clear_frame')} "
            f"left={ (report.get('left_try') or {}).get('reached_room') }"
        )

    out = RECORDINGS_DIR / f"{args.tag}_probe.json"
    write_json_report(
        out,
        {
            "segment": "level2_room_recon_7d_6d",
            "track": "assisted" if args.infinite_life else "clean",
            "intervention_class": "survival" if args.infinite_life else "clean",
            "start_state": args.from_state,
            "trials": args.trials,
            "successes": sum(1 for r in reports if r.get("ok")),
            "reports": reports,
        },
    )
    print(f"wrote {out}")
    return 0 if all(r.get("ok") for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
