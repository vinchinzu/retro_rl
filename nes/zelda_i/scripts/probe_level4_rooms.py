"""Live recon of Level 4 Snake entry room from ``Level4Entrance``.

Assisted-friendly (``--infinite-life``). Discovers object type IDs, doors,
exits, and key drops. **No walkthrough room-id hardcode** beyond the LIVE
entry room ``0x71`` from rr-0fx.

Examples::

    uv run python nes/zelda_i/scripts/probe_level4_rooms.py --infinite-life
    uv run python nes/zelda_i/scripts/probe_level4_rooms.py --from-state Level4Entrance \\
        --infinite-life --tag l4_recon --save-state
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_NES = Path(__file__).resolve().parents[2]
for p in (_REPO_ROOT, _NES):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from retro_harness.env import make_env, save_state
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.combat import should_swing_at
from zelda_i.dungeon_ids import object_name, room_item_name
from zelda_i.dungeon_trace import write_state_provenance
from zelda_i.level4_overworld import LEVEL4, LEVEL4_ENTRY_ROOM
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import PLAY_MODE, ZeldaObject, ZeldaSnapshot, read_snapshot

# Door bit layout (ADDR_CUR_OPENED_DOORS / 0x00EE): bit0=R bit1=L bit2=D bit3=U
DOOR_RIGHT = 0x01
DOOR_LEFT = 0x02
DOOR_DOWN = 0x04
DOOR_UP = 0x08

# Non-enemy noise (drops / projectiles / empty).
_IGNORE_TYPES = frozenset({0, 0xFF, 0x60, 0x55, 0x56})

_PATROL: tuple[tuple[int, int], ...] = (
    (64, 109),
    (120, 109),
    (176, 109),
    (176, 141),
    (176, 173),
    (120, 173),
    (64, 173),
    (64, 141),
    (120, 141),
)

PROBE_DIRS = ("LEFT", "UP", "RIGHT", "DOWN")


def _objs(snap: ZeldaSnapshot, *, slots_only: bool = True) -> list[dict]:
    out: list[dict] = []
    for o in snap.objects:
        if slots_only and not (1 <= o.slot <= 12):
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
    objs = _objs(snap)
    return {
        "mode": snap.mode,
        "level": snap.level,
        "screen": snap.screen,
        "screen_hex": f"0x{snap.screen:02x}",
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
        "objects": objs,
        "type_counts": dict(Counter(o["type"] for o in objs)),
        "type_names": {
            f"0x{t:02x}": object_name(t) for t in Counter(o["type"] for o in objs)
        },
    }


def _enemy_objs(snap: ZeldaSnapshot) -> tuple[ZeldaObject, ...]:
    return tuple(
        o
        for o in snap.objects
        if 1 <= o.slot <= 12 and o.type_id not in _IGNORE_TYPES
    )


def _move_toward(snap: ZeldaSnapshot, tx: int, ty: int, *, tol: int = 4) -> str | None:
    dx = tx - snap.link_x
    dy = ty - snap.link_y
    if abs(dx) <= tol and abs(dy) <= tol:
        return None
    if abs(dx) > tol and abs(dx) >= abs(dy):
        return "RIGHT" if dx > 0 else "LEFT"
    return "DOWN" if dy > 0 else "UP"


def _fight_step(
    snap: ZeldaSnapshot,
    *,
    patrol_i: int,
    frame: int,
) -> tuple[object, int]:
    enemies = _enemy_objs(snap)
    # Prefer nearest living-looking target (hp>0 or type-only like Keese).
    targets = [o for o in enemies if o.hp > 0] or list(enemies)
    if targets:
        tgt = min(
            targets,
            key=lambda o: abs(o.x - snap.link_x) + abs(o.y - snap.link_y),
        )
        d = _move_toward(snap, tgt.x, tgt.y, tol=8)
        direction = d or "UP"
        if should_swing_at(
            snap.link_x,
            snap.link_y,
            direction,
            enemies,
        ) or (abs(tgt.x - snap.link_x) + abs(tgt.y - snap.link_y) < 20):
            if frame % 8 < 4:
                return nes_action(direction, "A"), patrol_i
        return nes_action(direction), patrol_i

    # Clear: patrol / hunt room item.
    if snap.room_item_id == 0x19:
        # Fixed key hunt — walk toward common mid-room spots if coords unknown.
        tx, ty = 128, 141
        d = _move_toward(snap, tx, ty, tol=6)
        if d is None:
            return nes_action("UP"), patrol_i
        return nes_action(d), patrol_i

    tx, ty = _PATROL[patrol_i % len(_PATROL)]
    d = _move_toward(snap, tx, ty, tol=6)
    if d is None:
        patrol_i = (patrol_i + 1) % len(_PATROL)
        return nes_idle_action(), patrol_i
    return nes_action(d), patrol_i


def _probe_exit(
    env,
    obs,
    *,
    assist,
    direction: str,
    entry_room: int,
    tag: str,
    base_frame: int,
) -> tuple[object, dict]:
    """Push one doorway from entry room; sample destination."""
    # Return to entry first (best-effort).
    for rf in range(500):
        snap = read_snapshot(env.get_ram())
        if (
            snap.level == LEVEL4
            and snap.mode == PLAY_MODE
            and snap.screen == entry_room
            and not snap.transitioning
        ):
            break
        if snap.level == 0:
            btn = "UP"
        elif snap.screen != entry_room:
            # reverse last known push
            rev = {"LEFT": "RIGHT", "RIGHT": "LEFT", "UP": "DOWN", "DOWN": "UP"}
            btn = rev.get(direction, "DOWN")
        else:
            btn = "DOWN" if snap.link_y < 140 else "UP"
        obs, *_ = env.step(nes_action(btn))
        if assist is not None:
            assist.apply_env(env, frame=base_frame + rf)

    start = read_snapshot(env.get_ram()).screen
    arrived = None
    log: list[dict] = []
    for f in range(700):
        snap = read_snapshot(env.get_ram())
        if (
            snap.level == LEVEL4
            and snap.mode == PLAY_MODE
            and snap.screen != start
            and not snap.transitioning
        ):
            for _ in range(180):
                obs, *_ = env.step(nes_idle_action())
                if assist is not None:
                    assist.apply_env(env, frame=base_frame + f)
            snap = read_snapshot(env.get_ram())
            arrived = _room_fields(snap)
            save_rgb_png(
                obs, RECORDINGS_DIR / f"{tag}_{direction}_0x{snap.screen:02x}.png"
            )
            break
        if snap.level == 0:
            # Dropped out south mouth.
            arrived = {"level": 0, "screen": snap.screen, "event": "overworld_exit"}
            save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_{direction}_ow.png")
            break
        if direction in ("UP", "DOWN") and abs(snap.link_x - 120) > 10:
            btn = "LEFT" if snap.link_x > 120 else "RIGHT"
        elif direction in ("LEFT", "RIGHT") and abs(snap.link_y - 141) > 10:
            btn = "UP" if snap.link_y > 141 else "DOWN"
        else:
            btn = direction
        # South mouth is at y~205; don't leave while probing other dirs.
        if direction != "DOWN" and snap.link_y > 190:
            btn = "UP"
        act = nes_action(btn, "A") if f % 12 < 3 else nes_action(btn)
        obs, *_ = env.step(act)
        if assist is not None:
            assist.apply_env(env, frame=base_frame + f)
        if f % 80 == 0:
            log.append({"f": f, **_room_fields(snap)})

    final = _room_fields(read_snapshot(env.get_ram()))
    dest = None
    if arrived and "screen" in arrived and arrived.get("level") == LEVEL4:
        dest = f"0x{arrived['screen']:02x}"
    elif arrived and arrived.get("event") == "overworld_exit":
        dest = f"ow_0x{arrived.get('screen', 0):02x}"
    return obs, {
        "direction": direction,
        "from_screen": f"0x{start:02x}",
        "destination": dest,
        "ok": arrived is not None,
        "arrived": arrived,
        "final": final,
        "log_head": log[:6],
    }


def run_recon(
    *,
    start_state: str,
    infinite_life: bool,
    idle_frames: int,
    max_fight_frames: int,
    probe_exits: bool,
    save_checkpoint: bool,
    tag: str,
) -> dict:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
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
            snap.level == LEVEL4
            and snap.mode == PLAY_MODE
            and snap.screen == LEVEL4_ENTRY_ROOM
        ):
            png = RECORDINGS_DIR / f"{tag}_not_entry.png"
            save_rgb_png(obs, png)
            return {
                "ok": False,
                "bead": "rr-5lu",
                "track": track,
                "error": "not_in_level4_entry_0x71",
                "entry": entry0,
                "screenshot": str(png),
            }

        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_71_t0.png")
        keys0 = snap.keys

        # Phase A: idle sample spawn / doors.
        timeline: list[dict] = []
        max_types: Counter[int] = Counter()
        spawn_first: int | None = None
        for f in range(idle_frames):
            snap = read_snapshot(env.get_ram())
            enemies = _enemy_objs(snap)
            types = Counter(o.type_id for o in enemies)
            if types:
                max_types |= types
                if spawn_first is None:
                    spawn_first = f
                if len(timeline) < 12 or f % 30 == 0:
                    timeline.append({"f": f, **_room_fields(snap)})
            obs, *_ = env.step(nes_idle_action())
            if assist is not None:
                assist.apply_env(env, frame=f + 1)

        snap = read_snapshot(env.get_ram())
        after_idle = _room_fields(snap)
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_71_idle.png")
        doors_before = snap.cur_opened_doors

        # Phase B: fight until clear / key gain / timeout.
        fight_log: list[dict] = []
        patrol_i = 0
        clear_frame: int | None = None
        key_frame: int | None = None
        max_live = 0
        for f in range(max_fight_frames):
            snap = read_snapshot(env.get_ram())
            if snap.mode == 17:
                break
            enemies = _enemy_objs(snap)
            live_hp = [o for o in enemies if o.hp > 0]
            live_type = list(enemies)
            # Keese-style: type-only liveness when hp stays 0.
            live_n = len(live_hp) if live_hp else len(live_type)
            max_live = max(max_live, live_n)
            if snap.keys > keys0 and key_frame is None:
                key_frame = f
            if (
                clear_frame is None
                and live_n == 0
                and snap.room_all_dead >= 20
            ):
                clear_frame = f
                # Keep going briefly for key pickup.
            if clear_frame is not None and f > clear_frame + 400:
                if snap.keys > keys0 or snap.room_item_id in (0x03, 0x00):
                    break
            if clear_frame is not None and key_frame is not None and f > key_frame + 30:
                break

            act, patrol_i = _fight_step(snap, patrol_i=patrol_i, frame=f)
            # After clear, wander more aggressively for key drop.
            if clear_frame is not None and snap.keys <= keys0:
                # diamond walk
                wps = ((80, 120), (160, 120), (160, 160), (80, 160), (120, 140))
                tx, ty = wps[(f // 40) % len(wps)]
                d = _move_toward(snap, tx, ty, tol=4)
                act = nes_action(d) if d else nes_idle_action()
            obs, *_ = env.step(act)
            if assist is not None:
                assist.apply_env(env, frame=idle_frames + f + 1)
            if f % 60 == 0 or (clear_frame is not None and f == clear_frame):
                fight_log.append(
                    {
                        "f": f,
                        "live": live_n,
                        "keys": snap.keys,
                        "all_dead": snap.room_all_dead,
                        "doors": snap.cur_opened_doors,
                        "type_counts": dict(Counter(o.type_id for o in enemies)),
                        "x": snap.link_x,
                        "y": snap.link_y,
                    }
                )

        snap = read_snapshot(env.get_ram())
        post_fight = _room_fields(snap)
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_71_post_fight.png")
        doors_after = snap.cur_opened_doors
        keys1 = snap.keys

        # Extra settle.
        for f in range(90):
            obs, *_ = env.step(nes_idle_action())
            if assist is not None:
                assist.apply_env(env, frame=idle_frames + max_fight_frames + f)
        snap = read_snapshot(env.get_ram())
        post_settle = _room_fields(snap)
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_71_settle.png")

        checkpoint = None
        if save_checkpoint and snap.level == LEVEL4 and snap.screen == LEVEL4_ENTRY_ROOM:
            name = "Level4EntryCleared" if clear_frame is not None else "Level4Entrance"
            path = save_state(env, GAME_DIR, GAME, name)
            checkpoint = str(path)
            write_state_provenance(
                path,
                source_state_path=GAME_DIR
                / "custom_integrations"
                / GAME
                / f"{start_state}.state",
                request={
                    "bead": "rr-5lu",
                    "segment": "l4_entry_room_recon",
                    "track": track,
                    "intervention_class": "survival" if infinite_life else "clean",
                },
                selected_trial={
                    "ok": clear_frame is not None,
                    "entry_room": f"0x{LEVEL4_ENTRY_ROOM:02x}",
                    "clear_frame": clear_frame,
                    "keys": keys1,
                    "doors_after": doors_after,
                    "max_types": {f"0x{k:02x}": v for k, v in max_types.items()},
                    "final": _room_fields(snap),
                },
            )

        # Phase C: physical exit probes (from entry after clear if possible).
        exit_probes: list[dict] = []
        if probe_exits and snap.level == LEVEL4:
            for i, d in enumerate(PROBE_DIRS):
                obs, pr = _probe_exit(
                    env,
                    obs,
                    assist=assist,
                    direction=d,
                    entry_room=LEVEL4_ENTRY_ROOM,
                    tag=tag,
                    base_frame=20000 + i * 1000,
                )
                exit_probes.append(pr)

        ok = (
            snap.level == LEVEL4
            and max_types
            and (clear_frame is not None or keys1 > keys0)
        )
        # Partial success: at least spawn + idle snapshot is useful recon.
        recon_ok = bool(max_types) or bool(timeline)
        report = {
            "ok": ok or recon_ok,
            "bead": "rr-5lu",
            "track": track,
            "intervention_class": "survival" if infinite_life else "clean",
            "start_state": start_state,
            "infinite_life": infinite_life,
            "entry_room": f"0x{LEVEL4_ENTRY_ROOM:02x}",
            "assist": assist.report() if assist else None,
            "entry_t0": entry0,
            "after_idle": after_idle,
            "spawn": {
                "first_frame": spawn_first,
                "max_types": {f"0x{k:02x}": v for k, v in max_types.items()},
                "max_type_names": {
                    f"0x{k:02x}": object_name(k) for k in max_types
                },
                "timeline": timeline[:16],
            },
            "fight": {
                "clear_frame": clear_frame,
                "key_frame": key_frame,
                "keys_before": keys0,
                "keys_after": keys1,
                "key_gained": keys1 > keys0,
                "max_live": max_live,
                "doors_before": doors_before,
                "doors_after": doors_after,
                "doors_bits_after": {
                    "R": bool(doors_after & DOOR_RIGHT),
                    "L": bool(doors_after & DOOR_LEFT),
                    "D": bool(doors_after & DOOR_DOWN),
                    "U": bool(doors_after & DOOR_UP),
                    "raw": doors_after,
                },
                "post_fight": post_fight,
                "post_settle": post_settle,
                "log_tail": fight_log[-12:],
            },
            "exit_probes": exit_probes,
            "live_exits": {
                p["direction"]: p["destination"]
                for p in exit_probes
                if p.get("ok") and p.get("destination")
            },
            "checkpoint": checkpoint,
            "screenshots": {
                "t0": str(RECORDINGS_DIR / f"{tag}_71_t0.png"),
                "idle": str(RECORDINGS_DIR / f"{tag}_71_idle.png"),
                "post_fight": str(RECORDINGS_DIR / f"{tag}_71_post_fight.png"),
                "settle": str(RECORDINGS_DIR / f"{tag}_71_settle.png"),
            },
            "final": _room_fields(read_snapshot(env.get_ram())),
        }
        return report
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-state", default="Level4Entrance")
    parser.add_argument("--infinite-life", action="store_true")
    parser.add_argument("--idle-frames", type=int, default=200)
    parser.add_argument("--max-fight-frames", type=int, default=6000)
    parser.add_argument("--probe-exits", action="store_true", default=True)
    parser.add_argument("--no-probe-exits", action="store_false", dest="probe_exits")
    parser.add_argument("--save-state", action="store_true")
    parser.add_argument("--tag", default="l4_recon")
    parser.add_argument("--trials", type=int, default=1)
    args = parser.parse_args(argv)

    trials: list[dict] = []
    for i in range(args.trials):
        tag = args.tag if args.trials == 1 else f"{args.tag}_t{i}"
        r = run_recon(
            start_state=args.from_state,
            infinite_life=args.infinite_life,
            idle_frames=args.idle_frames,
            max_fight_frames=args.max_fight_frames,
            probe_exits=args.probe_exits,
            save_checkpoint=args.save_state,
            tag=tag,
        )
        trials.append(r)
        print(
            f"trial {i}: ok={r.get('ok')} clear={r.get('fight', {}).get('clear_frame')} "
            f"keys={r.get('fight', {}).get('keys_before')}→{r.get('fight', {}).get('keys_after')} "
            f"types={r.get('spawn', {}).get('max_types')} exits={r.get('live_exits')}"
        )

    summary = {
        "bead": "rr-5lu",
        "track": "assisted" if args.infinite_life else "clean",
        "start_state": args.from_state,
        "trials": len(trials),
        "ok_count": sum(1 for t in trials if t.get("ok")),
        "results": trials,
        "live_exits_union": {
            k: v
            for t in trials
            for k, v in (t.get("live_exits") or {}).items()
        },
    }
    out = RECORDINGS_DIR / f"{args.tag}.json"
    write_json_report(out, summary)
    print(f"wrote {out} ok={summary['ok_count']}/{summary['trials']}")
    return 0 if summary["ok_count"] == summary["trials"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
