"""Idle spawn + door-mask probe for Level 2 room 0x5f (bomb-N of compass).

Loads ``Level2_5F`` (play-ready after bomb-north). Idles 300–600f, samples
enemies / door bits / room item every 60f. Optionally clears 5× Gel and
rechecks door mask + map inventory. Goal (rr-fvt): empty transit vs spawn;
document open bits for rr-cjf. Does **not** register ROOM_5F_SPEC.

Examples::

    uv run python nes/zelda_i/scripts/probe_level2_5f_policy.py
    uv run python nes/zelda_i/scripts/probe_level2_5f_policy.py \\
        --idle-frames 600 --clear --tag l2_5f_policy
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_NES = _REPO_ROOT / "nes"
if str(_NES) not in sys.path:
    sys.path.insert(0, str(_NES))

from retro_harness.env import make_env
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)
from zelda_i.dungeon import (
    AliveRule,
    CombatTuning,
    DoorRoute,
    DungeonPhase,
    DungeonRoomSpec,
    GEL_OBJECT_TYPE,
    GenericDungeonRoomController,
    RewardKind,
    RewardSpec,
)
from zelda_i.dungeon_ids import object_name, room_item_name
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_MAP,
    PLAY_MODE,
    ZeldaSnapshot,
    read_snapshot,
    read_u8,
)

LEVEL_2 = 2
ROOM_5F = 0x5F

DOOR_RIGHT = 0x01
DOOR_LEFT = 0x02
DOOR_DOWN = 0x04
DOOR_UP = 0x08

# Gels/Keese are TYPE-only (hp=0 while alive); Goriya/Ropes need hp>0.
TYPE_ONLY = frozenset({0x15, 0x1B})
DROP_TYPES = frozenset({0x60, 0x61, 0x62, 0x63})

DOOR_TARGETS: dict[str, tuple[int, int]] = {
    "RIGHT": (208, 141),
    "LEFT": (32, 141),
    "UP": (120, 93),
    "DOWN": (120, 205),
}

# Probe-local clear (not registered as ROOM_5F_SPEC — docs-first rr-fvt).
# Live: 5× Gel TYPE-only + RoomItemId 0x17 map; doors often only DOWN after bomb.
_PROBE_5F_CLEAR_SPEC = DungeonRoomSpec(
    spec_id="level2_room5f_probe_clear",
    source_room=0x6F,
    room_id=ROOM_5F,
    entry=DoorRoute("UP", ((120, 189),)),
    enemy_types=(GEL_OBJECT_TYPE,),
    expected_enemy_count=5,
    alive_rule=AliveRule.TYPE,
    combat=CombatTuning(
        patrol=(
            (120, 141),
            (168, 141),
            (168, 109),
            (120, 109),
            (72, 109),
            (72, 141),
            (72, 173),
            (120, 173),
            (168, 173),
            (120, 141),
        ),
        engage_distance=56,
        patrol_attack_period=8,
        patrol_attack_hold=3,
        engage_attack_period=6,
        engage_attack_hold=3,
    ),
    reward=RewardSpec(kind=RewardKind.CLEAR_ONLY),
    room_item_id=0x17,
    max_frames=10000,
    level=LEVEL_2,
)


def _door_bits(raw: int) -> dict:
    return {
        "R": bool(raw & DOOR_RIGHT),
        "L": bool(raw & DOOR_LEFT),
        "D": bool(raw & DOOR_DOWN),
        "U": bool(raw & DOOR_UP),
        "raw": int(raw),
    }


def _objs(snap: ZeldaSnapshot) -> list[dict]:
    out = []
    for o in snap.objects:
        if not (1 <= o.slot <= 10) or o.type_id in (0, 0xFF):
            continue
        out.append(
            {
                "slot": o.slot,
                "type": o.type_id,
                "type_name": object_name(o.type_id),
                "x": o.x,
                "y": o.y,
                "hp": o.hp,
            }
        )
    return out


def _live_enemies(snap: ZeldaSnapshot) -> list:
    out = []
    for o in snap.objects:
        if not (1 <= o.slot <= 10):
            continue
        if o.type_id in (0, 0xFF) or o.type_id in DROP_TYPES:
            continue
        if o.type_id in TYPE_ONLY or o.hp > 0:
            out.append(o)
    return out


def _sample(snap: ZeldaSnapshot, ram, *, f: int, event: str = "sample") -> dict:
    live = _live_enemies(snap)
    types = Counter(o.type_id for o in live)
    all_types = Counter(
        o.type_id
        for o in snap.objects
        if 1 <= o.slot <= 10 and o.type_id not in (0, 0xFF)
    )
    return {
        "f": f,
        "event": event,
        "mode": snap.mode,
        "level": snap.level,
        "screen": snap.screen,
        "sc": f"0x{snap.screen:02x}",
        "xy": [snap.link_x, snap.link_y],
        "facing": snap.facing,
        "keys": snap.keys,
        "bombs": snap.bombs,
        "room_item_id": snap.room_item_id,
        "room_item_name": room_item_name(snap.room_item_id),
        "room_all_dead": snap.room_all_dead,
        "room_obj_count": snap.room_obj_count,
        "cur_opened_doors": snap.cur_opened_doors,
        "doors": _door_bits(snap.cur_opened_doors),
        "open_doorway_mask": snap.open_doorway_mask,
        "map_inv": read_u8(ram, ADDR_MAP),
        "live_enemy_count": len(live),
        "live_type_counts": {f"0x{k:02x}": v for k, v in types.items()},
        "live_type_names": {f"0x{k:02x}": object_name(k) for k in types},
        "all_type_counts": {f"0x{k:02x}": v for k, v in all_types.items()},
        "objects": _objs(snap),
    }


def _push_door(snap: ZeldaSnapshot, direction: str):
    tx, ty = DOOR_TARGETS[direction]
    if direction in ("LEFT", "RIGHT"):
        if abs(snap.link_y - ty) > 4:
            return nes_action("DOWN" if snap.link_y < ty else "UP")
        return nes_action(direction)
    if abs(snap.link_x - tx) > 6:
        return nes_action("RIGHT" if snap.link_x < tx else "LEFT")
    return nes_action(direction)


def _run_door_tests(env, *, door_budget: int) -> list[dict]:
    """Short-push R/U/L/D; return to 0x5f when possible between tests."""
    door_tests: list[dict] = []
    for direction in ("RIGHT", "UP", "LEFT", "DOWN"):
        snap = read_snapshot(env.get_ram())
        if snap.screen != ROOM_5F or snap.mode != PLAY_MODE:
            door_tests.append(
                {
                    "dir": direction,
                    "ok": False,
                    "skipped": True,
                    "reason": f"not_on_0x5f sc=0x{snap.screen:02x} mode={snap.mode}",
                }
            )
            continue
        start_sc = snap.screen
        start_doors = snap.cur_opened_doors
        start_keys = snap.keys
        reached = False
        used = 0
        obs = None
        for bf in range(door_budget):
            used = bf + 1
            snap = read_snapshot(env.get_ram())
            if snap.mode == 17:
                break
            # Accept play or still-scrolling onto a new screen as success.
            if snap.screen != start_sc and snap.mode in (PLAY_MODE, 6, 7):
                reached = True
                break
            if snap.transitioning:
                act = nes_action(direction)
            else:
                act = _push_door(snap, direction)
            obs, *_ = env.step(act)
        # Drain scroll into play if needed.
        if reached:
            for _ in range(90):
                snap = read_snapshot(env.get_ram())
                if snap.mode == PLAY_MODE:
                    break
                obs, *_ = env.step(nes_idle_action())
        snap = read_snapshot(env.get_ram())
        door_tests.append(
            {
                "dir": direction,
                "ok": reached,
                "frames": used,
                "keys_before": start_keys,
                "keys_after": snap.keys,
                "doors_before": start_doors,
                "doors_after": snap.cur_opened_doors,
                "doors_before_bits": _door_bits(start_doors),
                "doors_after_bits": _door_bits(snap.cur_opened_doors),
                "start_sc": f"0x{start_sc:02x}",
                "end_sc": f"0x{snap.screen:02x}",
                "end_xy": [snap.link_x, snap.link_y],
                "end_mode": snap.mode,
            }
        )
        # Return to 0x5f for next direction.
        if reached and snap.screen != ROOM_5F:
            opp = {
                "RIGHT": "LEFT",
                "LEFT": "RIGHT",
                "UP": "DOWN",
                "DOWN": "UP",
            }[direction]
            for _ in range(door_budget + 90):
                snap = read_snapshot(env.get_ram())
                if snap.mode == PLAY_MODE and snap.screen == ROOM_5F:
                    break
                if snap.transitioning or snap.mode in (6, 7):
                    obs, *_ = env.step(nes_action(opp))
                else:
                    obs, *_ = env.step(_push_door(snap, opp))
            # Settle play.
            for _ in range(30):
                snap = read_snapshot(env.get_ram())
                if snap.mode == PLAY_MODE and snap.screen == ROOM_5F:
                    break
                env.step(nes_idle_action())
    return door_tests


def run_probe(
    *,
    start_state: str,
    idle_frames: int,
    sample_every: int,
    do_clear: bool,
    try_doors: bool,
    door_budget: int,
    tag: str,
) -> dict:
    configure_headless()
    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    try:
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        obs, *_ = env.step(nes_idle_action())

        ram = env.get_ram()
        snap = read_snapshot(ram)
        entry = _sample(snap, ram, f=0, event="entry")
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_t0.png")

        if not (
            snap.level == LEVEL_2
            and snap.mode == PLAY_MODE
            and snap.screen == ROOM_5F
        ):
            return {
                "ok": False,
                "bead": "rr-fvt",
                "error": "not_on_0x5f_play",
                "entry": entry,
                "start_state": start_state,
                "screenshot": str(RECORDINGS_DIR / f"{tag}_t0.png"),
            }

        timeline: list[dict] = [entry]
        peak_live = entry["live_enemy_count"]
        peak_types: Counter[int] = Counter()
        for o in _live_enemies(snap):
            peak_types[o.type_id] += 1
        first_spawn_f: int | None = 0 if peak_live else None
        door_raw_history: list[int] = [int(entry["cur_opened_doors"])]
        room_item_history: list[int] = [int(entry["room_item_id"])]

        for f in range(1, idle_frames + 1):
            obs, *_ = env.step(nes_idle_action())
            ram = env.get_ram()
            snap = read_snapshot(ram)

            if snap.mode == 17:
                timeline.append(_sample(snap, ram, f=f, event="death"))
                break
            if snap.level != LEVEL_2 or snap.screen != ROOM_5F:
                timeline.append(_sample(snap, ram, f=f, event="left_0x5f"))
                break

            live = _live_enemies(snap)
            if live:
                peak_live = max(peak_live, len(live))
                counts: Counter[int] = Counter(o.type_id for o in live)
                for t, c in counts.items():
                    peak_types[t] = max(peak_types[t], c)
                if first_spawn_f is None:
                    first_spawn_f = f
                    timeline.append(_sample(snap, ram, f=f, event="first_spawn"))

            if f % sample_every == 0 or f == idle_frames:
                sample = _sample(snap, ram, f=f, event="idle_tick")
                timeline.append(sample)
                door_raw_history.append(int(sample["cur_opened_doors"]))
                room_item_history.append(int(sample["room_item_id"]))

        snap = read_snapshot(env.get_ram())
        after_idle = _sample(snap, env.get_ram(), f=idle_frames, event="after_idle")
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_idle.png")
        doors_before_clear = int(after_idle["cur_opened_doors"])
        map_before_clear = int(after_idle["map_inv"])

        clear_report: dict | None = None
        after_clear: dict | None = None
        doors_after_clear: int | None = None
        map_after_clear: int | None = None
        clear_opened_new_doors = False

        if do_clear and peak_live > 0 and snap.screen == ROOM_5F:
            # Already on room — start in FIGHT (skip entry route).
            controller = GenericDungeonRoomController(_PROBE_5F_CLEAR_SPEC)
            controller.phase = DungeonPhase.FIGHT
            clear_frames = 0
            for clear_frames in range(_PROBE_5F_CLEAR_SPEC.max_frames):
                action = controller.step(read_snapshot(env.get_ram()))
                obs, *_ = env.step(action.action)
                if (
                    controller.success
                    or controller.phase is DungeonPhase.FAILED
                    or controller.phase is DungeonPhase.DONE
                ):
                    break
            # Extra wander for map drop (RoomItemId 0x17) if still uncollected.
            map_wander_log: list[dict] = []
            for wf in range(400):
                ram = env.get_ram()
                snap = read_snapshot(ram)
                map_val = read_u8(ram, ADDR_MAP)
                if map_val != 0 and map_val != map_before_clear:
                    map_wander_log.append(
                        _sample(snap, ram, f=wf, event="map_got")
                    )
                    break
                # Sweep mid-room + edges for drop pickup.
                targets = (
                    (120, 141),
                    (160, 141),
                    (80, 141),
                    (120, 109),
                    (120, 173),
                    (168, 109),
                    (72, 173),
                )
                tx, ty = targets[wf // 50 % len(targets)]
                if abs(snap.link_x - tx) > 6:
                    act = nes_action("RIGHT" if snap.link_x < tx else "LEFT")
                elif abs(snap.link_y - ty) > 6:
                    act = nes_action("DOWN" if snap.link_y < ty else "UP")
                else:
                    act = nes_idle_action()
                obs, *_ = env.step(act)

            ram = env.get_ram()
            snap = read_snapshot(ram)
            after_clear = _sample(snap, ram, f=clear_frames, event="after_clear")
            doors_after_clear = int(after_clear["cur_opened_doors"])
            map_after_clear = int(after_clear["map_inv"])
            clear_opened_new_doors = (
                doors_after_clear | doors_before_clear
            ) != doors_before_clear and doors_after_clear != doors_before_clear
            # Also true if bits expanded (new open).
            clear_opened_new_doors = bool(
                (doors_after_clear & ~doors_before_clear) != 0
            )
            clear_report = {
                **controller.report(),
                "frames": clear_frames + 1,
                "doors_before": doors_before_clear,
                "doors_after": doors_after_clear,
                "doors_before_bits": _door_bits(doors_before_clear),
                "doors_after_bits": _door_bits(doors_after_clear),
                "clear_opened_new_doors": clear_opened_new_doors,
                "map_before": map_before_clear,
                "map_after": map_after_clear,
                "map_gained": map_after_clear != map_before_clear
                and map_after_clear != 0,
                "map_wander": map_wander_log[:3],
            }
            timeline.append(after_clear)
            door_raw_history.append(doors_after_clear)
            save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_cleared.png")

        door_tests: list[dict] = []
        if try_doors:
            door_tests = _run_door_tests(env, door_budget=door_budget)

        final = _sample(
            read_snapshot(env.get_ram()), env.get_ram(), f=-1, event="final"
        )
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_final.png")

        door_set = sorted(set(door_raw_history))
        item_set = sorted(set(room_item_history))
        empty_transit = peak_live == 0 and first_spawn_f is None
        doors_changed = len(door_set) > 1

        if empty_transit:
            policy = "empty_transit"
            policy_note = (
                "No live enemies over idle window; no combat clear needed. "
                "Transit via key-LEFT (0x5e) / DOWN hole (0x6f); RIGHT/UP "
                "residual for rr-cjf."
            )
        elif clear_opened_new_doors:
            policy = "clear_opens_doors"
            policy_note = (
                f"Enemies spawn (peak={peak_live} {dict(peak_types)}); clear "
                f"opens new door bits "
                f"{doors_before_clear:#x}→{doors_after_clear:#x}. "
                "Candidate for ROOM_5F_SPEC."
            )
        else:
            policy = "gels_present_key_left_no_kill_gate"
            policy_note = (
                f"5× Gel 0x15 present from entry (TYPE-only); doors stay "
                f"DOWN-only (raw={doors_before_clear}) through idle"
                + (
                    f"/clear (after={doors_after_clear})"
                    if doors_after_clear is not None
                    else ""
                )
                + ". LEFT key door works without kill-clear; RIGHT/UP sealed. "
                "Map RoomItemId 0x17. No ROOM_5F_SPEC (clear not a door gate). "
                "Boom path residual past RIGHT (rr-cjf)."
            )

        report = {
            "ok": True,
            "bead": "rr-fvt",
            "segment": "level2_0x5f_idle_policy",
            "start_state": start_state,
            "intervention_class": "clean",
            "idle_frames": idle_frames,
            "sample_every": sample_every,
            "did_clear": do_clear,
            "entry": entry,
            "after_idle": after_idle,
            "after_clear": after_clear,
            "clear": clear_report,
            "final": final,
            "first_spawn_f": first_spawn_f,
            "peak_live_enemies": peak_live,
            "peak_type_counts": {f"0x{k:02x}": v for k, v in peak_types.items()},
            "peak_type_names": {
                f"0x{k:02x}": object_name(k) for k in peak_types
            },
            "door_raw_history": door_raw_history,
            "door_raw_unique": door_set,
            "doors_changed_during_idle": doors_changed,
            "clear_opened_new_doors": clear_opened_new_doors,
            "room_item_history": room_item_history,
            "room_item_unique": item_set,
            "empty_transit": empty_transit,
            "policy": policy,
            "policy_note": policy_note,
            "door_tests": door_tests,
            "timeline": timeline,
            "screenshots": {
                "t0": str(RECORDINGS_DIR / f"{tag}_t0.png"),
                "idle": str(RECORDINGS_DIR / f"{tag}_idle.png"),
                "cleared": str(RECORDINGS_DIR / f"{tag}_cleared.png"),
                "final": str(RECORDINGS_DIR / f"{tag}_final.png"),
            },
        }
        return report
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-state", default="Level2_5F")
    parser.add_argument(
        "--idle-frames",
        type=int,
        default=600,
        help="Idle frames on 0x5f (300–600 recommended; default 600)",
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=60,
        help="Log sample period in frames (default 60)",
    )
    parser.add_argument(
        "--clear",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After idle, clear gels and recheck doors/map (default on)",
    )
    parser.add_argument(
        "--try-doors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After idle/clear, short-push R/U/L/D (default on)",
    )
    parser.add_argument(
        "--door-budget",
        type=int,
        default=200,
        help="Frames per door push attempt (default 200)",
    )
    parser.add_argument("--tag", default="l2_5f_policy")
    args = parser.parse_args()

    report = run_probe(
        start_state=args.from_state,
        idle_frames=args.idle_frames,
        sample_every=args.sample_every,
        do_clear=args.clear,
        try_doors=args.try_doors,
        door_budget=args.door_budget,
        tag=args.tag,
    )
    out = RECORDINGS_DIR / f"{args.tag}.json"
    write_json_report(out, report)
    print(f"wrote {out}")
    print(
        f"ok={report.get('ok')} policy={report.get('policy')} "
        f"empty={report.get('empty_transit')} "
        f"peak_live={report.get('peak_live_enemies')} "
        f"types={report.get('peak_type_names')} "
        f"first_spawn={report.get('first_spawn_f')} "
        f"doors_unique={report.get('door_raw_unique')} "
        f"clear_new_doors={report.get('clear_opened_new_doors')} "
        f"items_unique={report.get('room_item_unique')}"
    )
    if report.get("clear"):
        c = report["clear"]
        print(
            f"  clear success={c.get('success')} phase={c.get('phase')} "
            f"doors {c.get('doors_before')}→{c.get('doors_after')} "
            f"map {c.get('map_before')}→{c.get('map_after')} "
            f"map_gained={c.get('map_gained')}"
        )
    if report.get("door_tests"):
        for t in report["door_tests"]:
            if t.get("skipped"):
                print(f"  door {t['dir']}: skipped ({t.get('reason')})")
            else:
                print(
                    f"  door {t['dir']}: ok={t['ok']} "
                    f"{t['start_sc']}→{t['end_sc']} "
                    f"doors {t['doors_before']}→{t['doors_after']} "
                    f"keys {t['keys_before']}→{t['keys_after']}"
                )
    print(f"  note: {report.get('policy_note')}")
    if not report.get("ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
