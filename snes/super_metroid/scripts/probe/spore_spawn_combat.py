#!/usr/bin/env python3
"""Probe the no-assist Spore Spawn left-ledge missile policy.

Default start is the human-tape room-enter pin from full_start_v1.
Assist is **off** unless ``--assist`` is passed.

```bash
# No-assist fight from the human enter pin
uv run python snes/super_metroid/scripts/probe/spore_spawn_combat.py strategy

# Dump entry RAM / pickup slots
uv run python snes/super_metroid/scripts/probe/spore_spawn_combat.py dump
```
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from super_metroid.assist import UnlimitedResourcesAssist
from super_metroid.combat.features import spore_spawn_catalog
from super_metroid.combat.primitives import ensure_weapon
from super_metroid.combat.probe import (
    ProbeSession,
    open_state_env,
    resolve_named_state,
    write_json_report,
)
from super_metroid.combat.spore_spawn import (
    LEDGE_Y_MIN,
    ROOM_SPORE_SPAWN,
    WEAPON_MISSILES,
    SporeSpawnStrategy,
    _fire_window,
    _go_to_seat,
    list_pickups,
    mouth_open,
    play_spore_spawn_fight,
    seated,
)
from super_metroid.routes.controller_common import is_morph, unmorph
from super_metroid.paths import GAME_DIR, SCRATCH_STATE_DIR
from super_metroid.ram import read_bank7e_wram

DEFAULT_ENTRY = (
    GAME_DIR / "tasks" / "full_start_v1_anchors" / "f015374_enter_0x9DC7_0x9DC7.state"
)
DEFAULT_OUT = SCRATCH_STATE_DIR / "post_spore_spawn_defeated.state"

_NAMED_STATES: dict[str, Path] = {
    "human": DEFAULT_ENTRY,
    "human-enter": DEFAULT_ENTRY,
    "entry": DEFAULT_ENTRY,
}


def _resolve_state(name: str) -> Path:
    return resolve_named_state(name, _NAMED_STATES, extra_dirs=(DEFAULT_ENTRY.parent,))


def _open_env(state_path: Path):
    return open_state_env(state_path, missing_hint="Spore Spawn entry state missing")


def _snapshot(session: ProbeSession) -> dict[str, object]:
    st = session.state
    return {
        "room_id_hex": f"0x{st.room_id:04X}",
        "samus_x": st.samus_x,
        "samus_y": st.samus_y,
        "pose": st.pose,
        "health": st.health,
        "missiles": st.missiles,
        "max_missiles": st.max_missiles,
        "selected_item": st.selected_item,
        "enemy0_hp": st.enemy0_hp,
        "enemy0_x": st.enemy0_x,
        "enemy0_y": st.enemy0_y,
        "enemy0_spritemap": f"0x{st.enemy0_spritemap:04X}",
        "mouth_open": mouth_open(st),
        "seated": seated(st),
        "phase": st.phase.value,
        "pickups": [p.__dict__ for p in list_pickups(session.env)],
    }


def _wait_open_window(session: ProbeSession, *, timeout: int) -> bool:
    """Leave the seat on the first right-side open so we can close during the windup."""
    for _ in range(timeout):
        st = session.state
        if mouth_open(st) and st.enemy0_x >= 120:
            return True
        if int(st.health) == 0:
            return False
        session.step([0] * 12, "spore_wait_eye")
    return mouth_open(session.state) and session.state.enemy0_x >= 120


def _wait_window_closed(session: ProbeSession, *, timeout: int = 400) -> None:
    """Idle until the just-fired open has left (avoid re-entering the same window)."""
    for _ in range(timeout):
        st = session.state
        if (not mouth_open(st) and st.enemy0_x < 120) or int(st.health) == 0:
            return
        session.step([0] * 12, "spore_wait_close")


def _row(session: ProbeSession, reason: str) -> dict[str, object]:
    st = session.state
    return {
        "frame": session.frame,
        "reason": reason,
        "samus_x": st.samus_x,
        "samus_y": st.samus_y,
        "pose": st.pose,
        "missiles": st.missiles,
        "enemy0_hp": st.enemy0_hp,
        "enemy0_x": st.enemy0_x,
        "enemy0_y": st.enemy0_y,
        "enemy0_spritemap": f"0x{st.enemy0_spritemap:04X}",
        "mouth_open": mouth_open(st),
    }


def _fire_trace(session: ProbeSession) -> dict[str, object]:
    """Wrap session.step for one window: min y, missile spends, HP chips."""
    min_y = session.state.samus_y
    min_y_xy = (session.state.samus_x, session.state.samus_y)
    climbed = False
    log: list[dict[str, object]] = []
    spends: list[dict[str, object]] = []
    prev_ms = session.state.missiles
    prev_hp = session.state.enemy0_hp
    orig = session.step

    def tracked(action, reason: str):
        nonlocal min_y, min_y_xy, climbed, prev_ms, prev_hp
        st = orig(action, reason)
        if st.samus_y < min_y:
            min_y = st.samus_y
            min_y_xy = (st.samus_x, st.samus_y)
        if st.samus_y < 500:
            climbed = True
        periodic = session.frame % 10 == 0 and reason.startswith("spore_")
        spent = st.missiles < prev_ms
        if spent or st.enemy0_hp != prev_hp or st.samus_y < 500 or periodic:
            row = _row(session, reason)
            log.append(row)
            if spent:
                spends.append(row)
            prev_ms = st.missiles
            prev_hp = st.enemy0_hp
        return st

    session.step = tracked  # type: ignore[method-assign]
    try:
        shots = _fire_window(session, SporeSpawnStrategy())
    finally:
        session.step = orig  # type: ignore[method-assign]
    return {
        "shots": shots,
        "min_y": min_y,
        "min_y_xy": list(min_y_xy),
        "climbed": climbed,
        "events": log,
        "spends": spends,
    }


def _one_window(session: ProbeSession, *, wait: int) -> dict[str, object]:
    """Seat if needed, wait for the right-side open, fire, return the report."""
    strategy = SporeSpawnStrategy()
    if not seated(session.state):
        for _ in range(3):
            if seated(session.state) or int(session.state.health) == 0:
                break
            _go_to_seat(session, strategy)
    seated_snap = _snapshot(session)
    opened = False
    if seated(session.state) and int(session.state.health) > 0:
        if session.state.missiles > 0:
            try:
                ensure_weapon(session, WEAPON_MISSILES)
            except RuntimeError:
                pass
        opened = _wait_open_window(session, timeout=wait)
        if opened and is_morph(session.state.pose):
            unmorph(session)
    pre = _snapshot(session)
    if opened:
        fire = _fire_trace(session)
    else:
        fire = {
            "shots": 0,
            "min_y": session.state.samus_y,
            "min_y_xy": [session.state.samus_x, session.state.samus_y],
            "climbed": session.state.samus_y < 500,
            "events": [],
            "spends": [],
            "skipped": "window_not_open",
        }
    post = _snapshot(session)
    spent = int(pre["missiles"]) - int(post["missiles"])  # type: ignore[arg-type]
    hp_drop = int(pre["enemy0_hp"]) - int(post["enemy0_hp"])  # type: ignore[arg-type]
    min_y = int(fire["min_y"])
    final_seated = seated(session.state)
    killed = int(post["enemy0_hp"]) == 0  # type: ignore[arg-type]
    ok = (
        opened
        and min_y >= 500
        and not bool(fire["climbed"])
        and (
            killed
            or (
                spent >= 1
                and hp_drop == 100 * spent
                and final_seated
            )
        )
    )
    return {
        "success": ok,
        "opened": opened,
        "missiles_spent": spent,
        "hp_drop": hp_drop,
        "min_y": min_y,
        "min_y_xy": fire["min_y_xy"],
        "climbed": fire["climbed"],
        "shots_counted": fire["shots"],
        "returned_seated": final_seated,
        "final_xy": [session.state.samus_x, session.state.samus_y],
        "health": session.state.health,
        "seated": seated_snap,
        "pre_fire": pre,
        "post_fire": post,
        "spends": fire["spends"],
        "events": fire["events"],
    }


def cmd_window(args: argparse.Namespace) -> int:
    """Seat, wait for mouth_open + enemy x>=120, fire N windows, report."""
    state_path = _resolve_state(args.state)
    env, loaded = _open_env(state_path)
    assist = UnlimitedResourcesAssist(unlimited_energy=False, unlimited_ammo=False)
    try:
        session = ProbeSession(env, assist)
        entry = _snapshot(session)
        for _ in range(3):
            if seated(session.state):
                break
            _go_to_seat(session, SporeSpawnStrategy())
        seated_snap = _snapshot(session)
        windows: list[dict[str, object]] = []
        saved_pin: str | None = None
        count = max(1, int(args.windows))
        for index in range(count):
            if index > 0:
                _wait_window_closed(session)
            result = _one_window(session, wait=args.wait)
            windows.append(result)
            if (
                index == 0
                and args.save_pin is not None
                and result["success"]
                and int(result["post_fire"]["enemy0_hp"]) == 760  # type: ignore[index]
            ):
                pin = Path(args.save_pin)
                pin.parent.mkdir(parents=True, exist_ok=True)
                from super_metroid.dev.common import save_dev_state

                save_dev_state(env, pin)
                saved_pin = str(pin)
            if int(session.state.health) == 0 or bool(result["climbed"]):
                break
        first = windows[0]
        success = all(bool(w["success"]) for w in windows) and len(windows) == count
        compact = [
            {
                "i": i + 1,
                "ok": w["success"],
                "spent": w["missiles_spent"],
                "hp_drop": w["hp_drop"],
                "pre_hp": w["pre_fire"]["enemy0_hp"],  # type: ignore[index]
                "post_hp": w["post_fire"]["enemy0_hp"],  # type: ignore[index]
                "min_y": w["min_y"],
                "seated": w["returned_seated"],
                "spends": [
                    {
                        "xy": [s["samus_x"], s["samus_y"]],
                        "pose": s["pose"],
                        "eye": [s["enemy0_x"], s["enemy0_y"]],
                        "hp": s["enemy0_hp"],
                        "ms": s["missiles"],
                    }
                    for s in w["spends"]  # type: ignore[union-attr]
                ],
            }
            for i, w in enumerate(windows)
        ]
        report = {
            "command": "window",
            "state": loaded,
            "success": success,
            "window_count": count,
            "windows_ok": sum(1 for w in windows if w["success"]),
            "saved_pin": saved_pin,
            "summary": compact,
            "opened": first["opened"],
            "missiles_spent": first["missiles_spent"],
            "hp_drop": first["hp_drop"],
            "min_y": first["min_y"],
            "min_y_xy": first["min_y_xy"],
            "climbed": first["climbed"],
            "shots_counted": first["shots_counted"],
            "returned_seated": first["returned_seated"],
            "final_xy": windows[-1]["final_xy"],
            "entry": entry,
            "seated": seated_snap,
            "pre_fire": first["pre_fire"],
            "post_fire": first["post_fire"],
            "spends": first["spends"],
            "events": first["events"] if count == 1 else [],
            "windows": windows if count > 1 else [],
        }
        text = json.dumps(report, indent=2)
        print(json.dumps({k: report[k] for k in (
            "command", "success", "window_count", "windows_ok", "saved_pin", "summary"
        )}, indent=2))
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(text + "\n", encoding="utf-8")
        elif count == 1:
            print(text)
        return 0 if success else 1
    finally:
        env.close()


def _proj_table_dump(env: object) -> list[dict[str, object]]:
    """Raw $7E:1997–$1B20 words plus nonzero bytes (drop hunt)."""
    ram = read_bank7e_wram(env)
    start, end = 0x1997, 0x1B20
    words: list[dict[str, object]] = []
    for addr in range(start, end, 2):
        val = int(ram[addr]) | (int(ram[addr + 1]) << 8)
        if val:
            words.append({"addr": f"0x{addr:04X}", "u16": val, "u16_hex": f"0x{val:04X}"})
    return words


def cmd_scan_drops(args: argparse.Namespace) -> int:
    """One fire window, then idle until missiles increase; dump drop RAM."""
    state_path = _resolve_state(args.state)
    env, loaded = _open_env(state_path)
    assist = UnlimitedResourcesAssist(unlimited_energy=False, unlimited_ammo=False)
    try:
        session = ProbeSession(env, assist)
        _go_to_seat(session, SporeSpawnStrategy())
        from super_metroid.routes.controller_common import is_morph, unmorph
        from super_metroid.routes.runtime import hold

        opened = _wait_open_window(session, timeout=args.wait)
        fire = _fire_trace(session) if opened else {"shots": 0, "skipped": True}
        baseline_ms = session.state.missiles
        hits: list[dict[str, object]] = []
        seen_ids: set[int] = set()
        if is_morph(session.state.pose):
            try:
                unmorph(session)
            except Exception:
                pass
        from super_metroid.routes.controller_common import select_weapon as _sel

        try:
            _sel(session, 0)
        except RuntimeError:
            pass
        walk_right = True
        for i in range(args.idle):
            prev = session.state.missiles
            st = session.state
            names: list[str] = []
            if st.samus_y >= 680 and 40 <= st.samus_x <= 220:
                if st.samus_x >= 200:
                    walk_right = False
                elif st.samus_x <= 50:
                    walk_right = True
                names.append("RIGHT" if walk_right else "LEFT")
            if i % 8 < 2:
                names.append("X")
            if names:
                hold(session, 1, *names, reason="drop_sweep")
            else:
                session.step([0] * 12, "drop_idle")
            pickups = list_pickups(session.env)
            ram = read_bank7e_wram(session.env)
            for slot in range(18):
                seen_ids.add(int(ram[0x1997 + slot * 2]) | (int(ram[0x1998 + slot * 2]) << 8))
            grew = session.state.missiles > prev
            if grew or pickups:
                hits.append(
                    {
                        "idle_frame": i,
                        "frame": session.frame,
                        "missiles_before": prev,
                        "missiles_after": session.state.missiles,
                        "collected": grew,
                        "samus_x": session.state.samus_x,
                        "samus_y": session.state.samus_y,
                        "health": session.state.health,
                        "pickups_fn": [p.__dict__ for p in pickups],
                        "wram_1997_1B20": _proj_table_dump(session.env) if grew else [],
                    }
                )
                if grew and len([h for h in hits if h["collected"]]) >= args.max_hits:
                    break
            if int(session.state.health) == 0:
                break
        report = {
            "command": "scan-drops",
            "state": loaded,
            "opened": opened,
            "fire": {k: fire[k] for k in fire if k != "events"},
            "baseline_missiles": baseline_ms,
            "final_missiles": session.state.missiles,
            "hits": hits,
            "hit_count": len(hits),
            "collected": sum(1 for h in hits if h.get("collected")),
            "seen_proj_ids": sorted(f"0x{i:04X}" for i in seen_ids if i),
            "final": _snapshot(session),
        }
        text = json.dumps(report, indent=2)
        print(text)
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(text + "\n", encoding="utf-8")
        return 0 if hits else 1
    finally:
        env.close()


def _seat_once(session: ProbeSession, *, hop_a: int) -> dict[str, object]:
    """Walk to the hop band, short-hop, drift; return path + seated flag."""
    from super_metroid.routes.controller_common import ensure_morph, is_morph
    from super_metroid.routes.runtime import hold

    path: list[list[int]] = []
    strategy = SporeSpawnStrategy()

    def snap() -> None:
        st = session.state
        path.append([session.frame, st.samus_x, st.samus_y, st.pose, st.health])

    snap()
    for _ in range(80):
        st = session.state
        if st.pose not in (81, 164) and st.samus_y >= LEDGE_Y_MIN:
            break
        hold(session, 1, reason="spore_land")
    snap()
    for _ in range(90):
        st = session.state
        if st.samus_y >= 710 and 62 <= st.samus_x <= 78:
            break
        if int(st.health) == 0:
            break
        if st.samus_y < LEDGE_Y_MIN:
            hold(session, 1, reason="spore_fall_in")
        elif st.samus_x < 62:
            hold(session, 1, "RIGHT", reason="spore_off_wall")
        else:
            hold(session, 1, "LEFT", "B", reason="spore_floor_left")
    snap()
    hold(session, hop_a, "LEFT", "A", reason="spore_ledge_hop")
    snap()
    for _ in range(28):
        if session.state.samus_y <= 705 and session.state.samus_x <= 80:
            break
        if session.state.samus_x < 50:
            hold(session, 1, reason="spore_hop_idle")
        else:
            hold(session, 1, "LEFT", reason="spore_ledge_left")
    snap()
    on_ledge = LEDGE_Y_MIN <= session.state.samus_y <= 705 and session.state.samus_x <= 80
    if on_ledge and session.state.samus_x > strategy.seat_x_max:
        for _ in range(40):
            if session.state.samus_x <= strategy.seat_x_max:
                break
            hold(session, 1, "LEFT", reason="spore_ledge_left")
        snap()
    morphed = False
    if on_ledge and session.state.samus_x <= strategy.seat_x_max:
        try:
            ensure_morph(session)
            morphed = is_morph(session.state.pose)
        except TimeoutError:
            morphed = False
        snap()
    st = session.state
    return {
        "hop_a": hop_a,
        "on_ledge": on_ledge,
        "seated": seated(st, strategy),
        "morphed": morphed,
        "final_xy": [st.samus_x, st.samus_y],
        "pose": st.pose,
        "health": st.health,
        "min_y": min(p[2] for p in path),
        "path": path,
    }


def cmd_tune_hop(args: argparse.Namespace) -> int:
    """Reset the enter pin and try several LEFT+A hop lengths."""
    state_path = _resolve_state(args.state)
    hops = [int(x) for x in args.hops.split(",") if x.strip()]
    env, loaded = _open_env(state_path)
    from retro_harness.env import read_state_bytes

    pin = read_state_bytes(state_path)
    assist = UnlimitedResourcesAssist(unlimited_energy=False, unlimited_ammo=False)
    try:
        trials = []
        for hop_a in hops:
            env.em.set_state(pin)
            for _ in range(4):
                env.step([0] * 12)
            session = ProbeSession(env, assist)
            trials.append(_seat_once(session, hop_a=hop_a))
        report = {
            "command": "tune-hop",
            "state": loaded,
            "trials": trials,
            "landed": [t["hop_a"] for t in trials if t["on_ledge"]],
            "seated": [t["hop_a"] for t in trials if t["seated"]],
        }
        print(json.dumps(report, indent=2))
        return 0 if report["seated"] else 1
    finally:
        env.close()


def cmd_dump(args: argparse.Namespace) -> int:
    state_path = _resolve_state(args.state)
    env, loaded = _open_env(state_path)
    assist = UnlimitedResourcesAssist(unlimited_energy=False, unlimited_ammo=False)
    try:
        session = ProbeSession(env, assist)
        report = {
            "command": "dump",
            "state": loaded,
            "frames": args.frames,
            "samples": [_snapshot(session)],
        }
        for i in range(args.frames):
            session.step([0] * 12, "idle")
            if i % max(1, args.stride) == 0:
                report["samples"].append(_snapshot(session))
        print(json.dumps(report, indent=2))
        return 0
    finally:
        env.close()


def cmd_strategy(args: argparse.Namespace) -> int:
    catalog = spore_spawn_catalog()
    state_path = _resolve_state(args.state)
    env, loaded = _open_env(state_path)
    assist = UnlimitedResourcesAssist(
        unlimited_energy=args.assist,
        unlimited_ammo=args.assist,
    )
    try:
        session = ProbeSession(env, assist)
        if session.state.room_id != ROOM_SPORE_SPAWN:
            print(
                json.dumps(
                    {
                        "command": "strategy",
                        "state": loaded,
                        "success": False,
                        "outcome": "wrong_room",
                        "room_id_hex": f"0x{session.state.room_id:04X}",
                    },
                    indent=2,
                )
            )
            return 1

        entry = _snapshot(session)
        evidence = play_spore_spawn_fight(
            session,
            strategy=SporeSpawnStrategy(max_fight_frames=args.max_frames),
            require_boss_bit=not args.body_only,
        )
        success = evidence.outcome == "spore_spawn_defeated" or (
            args.body_only and evidence.defeat_frame is not None
        )
        out_path: Path | None = None
        if success and args.save_state:
            out = Path(args.save_state) if args.save_state is not True else DEFAULT_OUT
            for _ in range(30):
                session.step([0] * 12, "settle")
            out.parent.mkdir(parents=True, exist_ok=True)
            from super_metroid.dev.common import save_dev_state

            save_dev_state(env, out)
            out_path = out

        tel = assist.telemetry
        report = {
            "command": "strategy",
            "state": loaded,
            "success": success,
            "assist_enabled": bool(args.assist),
            "entry": entry,
            "fight": evidence.to_dict(),
            "reasons": dict(session.action_reasons),
            "assist": {
                "energy_restored": tel.energy.restored,
                "energy_writes": tel.energy.writes,
                "missile_writes": tel.ammo["missiles"].writes,
                "maximum_single_frame_damage": tel.maximum_single_frame_damage,
                "deaths": tel.deaths,
            },
            "final": _snapshot(session),
            "saved_state": str(out_path) if out_path is not None else None,
            "catalog": {"name": catalog.name, "max_hp": catalog.max_hp},
            "method": "left_ledge_two_missile",
            "notes": (
                "Human-tape seat (x=21 y=697 morph) + 2 missiles per open eye. "
                "No resource writes unless --assist."
            ),
        }
        text = json.dumps(report, indent=2)
        print(text)
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(text + "\n", encoding="utf-8")
        return 0 if success else 1
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("strategy", help="Run left-ledge two-missile policy")
    p.add_argument("--state", default="human", help="human|path (default: human enter pin)")
    p.add_argument("--max-frames", type=int, default=24_000)
    p.add_argument("--assist", action="store_true", help="Enable energy+ammo refill")
    p.add_argument("--body-only", action="store_true", help="Stop at HP 0 (skip boss bit)")
    p.add_argument("--save-state", nargs="?", const=True, default=False)
    p.add_argument("--report", type=Path, default=None)
    p.set_defaults(func=cmd_strategy)

    d = sub.add_parser("dump", help="Idle-dump RAM / pickup slots from entry")
    d.add_argument("--state", default="human")
    d.add_argument("--frames", type=int, default=180)
    d.add_argument("--stride", type=int, default=30)
    d.set_defaults(func=cmd_dump)

    w = sub.add_parser("window", help="Seat + N fire windows (no 20k fight)")
    w.add_argument("--state", default="human")
    w.add_argument("--wait", type=int, default=2400, help="Frames to wait for mouth_open")
    w.add_argument("--windows", type=int, default=1, help="Seated windows to fire in a row")
    w.add_argument("--save-pin", type=Path, default=None, help="Save 760 HP pin after window 1")
    w.add_argument("--report", type=Path, default=None)
    w.set_defaults(func=cmd_window)

    s = sub.add_parser("scan-drops", help="One window then idle-scan drop RAM")
    s.add_argument("--state", default="human")
    s.add_argument("--wait", type=int, default=2400)
    s.add_argument("--idle", type=int, default=1800, help="Idle frames after the window")
    s.add_argument("--max-hits", type=int, default=4)
    s.add_argument("--report", type=Path, default=None)
    s.set_defaults(func=cmd_scan_drops)

    t = sub.add_parser("tune-hop", help="Try LEFT+A hop lengths onto the left ledge")
    t.add_argument("--state", default="human")
    t.add_argument("--hops", default="2,3,4,5,6,8,10,12")
    t.set_defaults(func=cmd_tune_hop)

    args = parser.parse_args()
    if not getattr(args, "command", None):
        parser.print_help()
        return 2
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
