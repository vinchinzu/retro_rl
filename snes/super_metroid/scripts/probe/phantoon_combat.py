#!/usr/bin/env python3
"""Probe the no-assist Phantoon left-corner charge/missile policy.

Default start is the natural Basement→room pin (rr-cjpp). Assist is **off**
unless ``--assist`` is passed. Super-spray is not a hit.

```bash
# Idle dump: room, seat xy, enemy0 x/y/hp/spritemap, pickups
uv run python snes/super_metroid/scripts/probe/phantoon_combat.py dump --frames 400

# One measured window, halt at first miss
uv run python snes/super_metroid/scripts/probe/phantoon_combat.py window --windows 1
```
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from super_metroid.assist import UnlimitedResourcesAssist
from super_metroid.combat.features import phantoon_catalog
from super_metroid.combat.phantoon import (
    ROOM_PHANTOON,
    WEAPON_BEAM,
    WEAPON_MISSILES,
    PhantoonStrategy,
    _fire_window,
    _go_to_seat,
    beam_charge,
    enemy_extra,
    eye_open,
    list_pickups,
    play_phantoon_fight,
    seated,
)
from super_metroid.combat.probe import (
    ProbeSession,
    open_state_env,
    resolve_named_state,
    write_json_report,
)
from super_metroid.dev.phantoon_dev import (
    PHANTOON_ENTRY_STATE,
    phantoon_defeated,
    wrecked_ship_boss_bits,
)
from super_metroid.paths import SCRATCH_STATE_DIR


DEFAULT_ENTRY = SCRATCH_STATE_DIR / "post_ws_basement_to_phantoon.state"
DEFAULT_OUT = SCRATCH_STATE_DIR / "post_phantoon_poweron.state"
HUMAN_END = SCRATCH_STATE_DIR / "ws_ship_human_end.state"

_NAMED_STATES: dict[str, Path] = {
    "natural": DEFAULT_ENTRY,
    "post_ws_basement_to_phantoon": DEFAULT_ENTRY,
    "ws_ship_human_end": HUMAN_END,
    "human": HUMAN_END,
    "human-end": HUMAN_END,
    "entry": PHANTOON_ENTRY_STATE,
    "dev_entry": PHANTOON_ENTRY_STATE,
    "dev_phantoon_entry": PHANTOON_ENTRY_STATE,
    "human-enter": SCRATCH_STATE_DIR / "full_start_v1_phantoon.state",
    "human-mid": SCRATCH_STATE_DIR / "full_start_v1_phantoon_mid.state",
}


def _resolve_state(name: str) -> Path:
    return resolve_named_state(name, _NAMED_STATES)


def _open_env(state_path: Path):
    return open_state_env(
        state_path,
        missing_hint=(
            "Need scratch/post_ws_basement_to_phantoon.state "
            "(rr-cjpp Basement→room leave)."
        ),
    )


def _snapshot(session: ProbeSession) -> dict[str, object]:
    st = session.state
    extra = enemy_extra(session.env)
    return {
        "room_id_hex": f"0x{st.room_id:04X}",
        "samus_x": st.samus_x,
        "samus_y": st.samus_y,
        "pose": st.pose,
        "facing": st.facing,
        "health": st.health,
        "missiles": st.missiles,
        "super_missiles": st.super_missiles,
        "max_missiles": st.max_missiles,
        "selected_item": st.selected_item,
        "equipped_beams": f"0x{st.equipped_beams:04X}",
        "charge": beam_charge(session.env),
        "enemy0_hp": st.enemy0_hp,
        "enemy0_x": st.enemy0_x,
        "enemy0_y": st.enemy0_y,
        "enemy0_spritemap": f"0x{st.enemy0_spritemap:04X}",
        "enemy_ilist": extra.get("ilist"),
        "enemy_timer": extra.get("timer"),
        "enemy_palette": extra.get("palette"),
        "enemy_ai0": extra.get("ai0"),
        "enemy_ai1": extra.get("ai1"),
        "enemy_ai2": extra.get("ai2"),
        "func": extra.get("func"),
        "eye_ilist": extra.get("eye_ilist"),
        "eye_xy": [extra.get("eye_x"), extra.get("eye_y")],
        "eye_open": eye_open(st, session.env),
        "seated": seated(st),
        "phase": st.phase.value,
        "pickups": [p.__dict__ for p in list_pickups(session.env)],
    }


def _row(session: ProbeSession, reason: str) -> dict[str, object]:
    st = session.state
    return {
        "frame": session.frame,
        "reason": reason,
        "samus_x": st.samus_x,
        "samus_y": st.samus_y,
        "pose": st.pose,
        "missiles": st.missiles,
        "supers": st.super_missiles,
        "selected": st.selected_item,
        "charge": beam_charge(session.env),
        "health": st.health,
        "enemy0_hp": st.enemy0_hp,
        "enemy0_x": st.enemy0_x,
        "enemy0_y": st.enemy0_y,
        "enemy0_spritemap": f"0x{st.enemy0_spritemap:04X}",
        "ilist": enemy_extra(session.env).get("ilist"),
        "func": enemy_extra(session.env).get("func"),
        "eye_ilist": enemy_extra(session.env).get("eye_ilist"),
        "eye_open": eye_open(st, session.env),
        "hittable": bool(enemy_extra(session.env).get("func_vuln") or enemy_extra(session.env).get("eye_il_open")),
    }


def cmd_dump(args: argparse.Namespace) -> int:
    state_path = _resolve_state(args.state)
    env, loaded = _open_env(state_path)
    assist = UnlimitedResourcesAssist(unlimited_energy=False, unlimited_ammo=False)
    try:
        session = ProbeSession(env, assist)
        maps: dict[str, dict[str, object]] = {}
        samples = [_snapshot(session)]
        for i in range(args.frames):
            session.step([0] * 12, "idle")
            st = session.state
            extra = enemy_extra(session.env)
            key = (
                f"{extra.get('func')}|{extra.get('eye_ilist')}|"
                f"{extra.get('ilist')}|0x{st.enemy0_spritemap:04X}"
            )
            if key not in maps:
                maps[key] = {
                    "first_frame": session.frame,
                    "enemy0_x": st.enemy0_x,
                    "enemy0_y": st.enemy0_y,
                    "enemy0_hp": st.enemy0_hp,
                    "ilist": extra.get("ilist"),
                    "func": extra.get("func"),
                    "eye_ilist": extra.get("eye_ilist"),
                    "eye_xy": [extra.get("eye_x"), extra.get("eye_y")],
                    "eye_sm": extra.get("eye_spritemap"),
                    "func_vuln": extra.get("func_vuln"),
                    "eye_il_open": extra.get("eye_il_open"),
                    "timer": extra.get("timer"),
                    "samus_xy": [st.samus_x, st.samus_y],
                    "pose": st.pose,
                    "health": st.health,
                }
            if i % max(1, args.stride) == 0:
                samples.append(_snapshot(session))
        report = {
            "command": "dump",
            "state": loaded,
            "frames": args.frames,
            "distinct_spritemaps": maps,
            "samples": samples,
            "final": _snapshot(session),
        }
        print(json.dumps(report, indent=2))
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return 0
    finally:
        env.close()


def _left_side_open(session: ProbeSession) -> bool:
    st = session.state
    return eye_open(st, session.env) and int(st.enemy0_x) < PhantoonStrategy().skip_enemy_x


def _wait_open_window(session: ProbeSession, *, timeout: int) -> bool:
    from retro_harness.actions import buttons, idle_action

    for _ in range(timeout):
        st = session.state
        if _left_side_open(session):
            return True
        if int(st.health) == 0:
            return False
        if st.selected_item == WEAPON_BEAM:
            session.step(buttons("X"), "phan_wait_eye")
        else:
            session.step(idle_action(), "phan_wait_eye")
    return _left_side_open(session)


def _wait_window_closed(session: ProbeSession, *, timeout: int = 400) -> None:
    for _ in range(timeout):
        st = session.state
        if (not _left_side_open(session)) or int(st.health) == 0:
            return
        session.step([0] * 12, "phan_wait_close")


def _fire_trace(
    session: ProbeSession, strategy: PhantoonStrategy | None = None
) -> dict[str, object]:
    min_y = session.state.samus_y
    min_y_xy = (session.state.samus_x, session.state.samus_y)
    log: list[dict[str, object]] = []
    spends: list[dict[str, object]] = []
    prev_ms = session.state.missiles
    prev_charge = beam_charge(session.env)
    prev_hp = session.state.enemy0_hp
    orig = session.step

    def tracked(action, reason: str):
        nonlocal min_y, min_y_xy, prev_ms, prev_charge, prev_hp
        st = orig(action, reason)
        charge = beam_charge(session.env)
        if st.samus_y < min_y:
            min_y = st.samus_y
            min_y_xy = (st.samus_x, st.samus_y)
        periodic = str(reason).startswith("phan") and (
            session.frame % 5 == 0 or "fire" in str(reason) or "shot" in str(reason)
        )
        spent_ms = st.missiles < prev_ms
        spent_ch = charge < prev_charge and prev_charge >= 60
        hp_chip = st.enemy0_hp != prev_hp
        if spent_ms or spent_ch or hp_chip or periodic:
            row = _row(session, reason)
            log.append(row)
            if spent_ms or spent_ch or hp_chip:
                spends.append(row)
            prev_ms = st.missiles
            prev_charge = charge
            prev_hp = st.enemy0_hp
        return st

    session.step = tracked  # type: ignore[method-assign]
    try:
        shots = _fire_window(session, strategy or PhantoonStrategy())
    finally:
        session.step = orig  # type: ignore[method-assign]
    return {
        "shots": shots,
        "min_y": min_y,
        "min_y_xy": list(min_y_xy),
        "events": log,
        "spends": spends,
    }


def _one_window(
    session: ProbeSession, *, wait: int, strategy: PhantoonStrategy | None = None
) -> dict[str, object]:
    strategy = strategy or PhantoonStrategy()
    if not seated(session.state):
        for _ in range(3):
            if seated(session.state) or int(session.state.health) == 0:
                break
            _go_to_seat(session, strategy)
    seated_snap = _snapshot(session)
    opened = False
    if seated(session.state) and int(session.state.health) > 0:
        opened = _wait_open_window(session, timeout=wait)
    pre = _snapshot(session)
    if opened:
        fire = _fire_trace(session, strategy)
    else:
        fire = {
            "shots": 0,
            "min_y": session.state.samus_y,
            "min_y_xy": [session.state.samus_x, session.state.samus_y],
            "events": [],
            "spends": [],
            "skipped": "window_not_open",
        }
    post = _snapshot(session)
    spent_ms = int(pre["missiles"]) - int(post["missiles"])  # type: ignore[arg-type]
    hp_drop = int(pre["enemy0_hp"]) - int(post["enemy0_hp"])  # type: ignore[arg-type]
    shots = int(fire["shots"])
    ok = opened and shots >= 1 and hp_drop > 0
    return {
        "success": ok,
        "opened": opened,
        "missiles_spent": spent_ms,
        "hp_drop": hp_drop,
        "min_y": fire["min_y"],
        "min_y_xy": fire["min_y_xy"],
        "shots_counted": shots,
        "returned_seated": seated(session.state),
        "final_xy": [session.state.samus_x, session.state.samus_y],
        "health": session.state.health,
        "seated": seated_snap,
        "pre_fire": pre,
        "post_fire": post,
        "spends": fire["spends"],
        "events": fire["events"],
    }


def cmd_window(args: argparse.Namespace) -> int:
    state_path = _resolve_state(args.state)
    env, loaded = _open_env(state_path)
    assist = UnlimitedResourcesAssist(unlimited_energy=False, unlimited_ammo=False)
    try:
        session = ProbeSession(env, assist)
        entry = _snapshot(session)
        weapon = WEAPON_MISSILES if args.weapon == "missiles" else WEAPON_BEAM
        strategy = PhantoonStrategy(weapon=weapon)
        for _ in range(3):
            if seated(session.state):
                break
            _go_to_seat(session, strategy)
        seated_snap = _snapshot(session)
        windows: list[dict[str, object]] = []
        count = max(1, int(args.windows))
        for index in range(count):
            if index > 0:
                _wait_window_closed(session)
            result = _one_window(session, wait=args.wait, strategy=strategy)
            windows.append(result)
            if int(session.state.health) == 0:
                break
            if not result["success"]:
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
                "shots": w["shots_counted"],
                "seated": w["returned_seated"],
                "spends": [
                    {
                        "xy": [s["samus_x"], s["samus_y"]],
                        "pose": s["pose"],
                        "eye": [s["enemy0_x"], s["enemy0_y"]],
                        "hp": s["enemy0_hp"],
                        "ms": s["missiles"],
                        "charge": s.get("charge"),
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
            "summary": compact,
            "opened": first["opened"],
            "missiles_spent": first["missiles_spent"],
            "hp_drop": first["hp_drop"],
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
        print(json.dumps({k: report[k] for k in (
            "command", "success", "window_count", "windows_ok", "summary"
        )}, indent=2))
        text = json.dumps(report, indent=2)
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(text + "\n", encoding="utf-8")
        elif count == 1:
            print(text)
        return 0 if success else 1
    finally:
        env.close()


def cmd_strategy(args: argparse.Namespace) -> int:
    catalog = phantoon_catalog()
    state_path = _resolve_state(args.state)
    env, loaded = _open_env(state_path)
    assist = UnlimitedResourcesAssist(
        unlimited_energy=args.assist,
        unlimited_ammo=args.assist,
    )
    try:
        session = ProbeSession(env, assist)
        if session.state.room_id != ROOM_PHANTOON:
            report = {
                "command": "strategy",
                "state": loaded,
                "success": False,
                "outcome": "wrong_room",
                "room_id_hex": f"0x{session.state.room_id:04X}",
                "notes": "Load a Phantoon-room entry state (0xCD13).",
            }
            write_json_report(report, args.report)
            return 1

        entry = _snapshot(session)
        strategy = PhantoonStrategy(max_fight_frames=args.max_frames)
        evidence = play_phantoon_fight(
            session,
            strategy=strategy,
            require_boss_bit=not args.body_only,
        )
        success = evidence.outcome == "phantoon_defeated" or (
            args.body_only and evidence.body_zero_frame is not None
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
            "boss_bits_wrecked_ship": wrecked_ship_boss_bits(env),
            "phantoon_defeated": phantoon_defeated(env),
            "method": "left_corner_charge_missiles",
            "notes": (
                "KPDR beginner: seat left, charge when the eye opens, two more, "
                "repeat. No Super spray. No resource writes unless --assist."
            ),
        }
        write_json_report(report, args.report)
        return 0 if success else 1
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("strategy", help="Run left-corner charge/missile policy")
    p.add_argument(
        "--state",
        default="natural",
        help="natural|human|entry|path (default: natural Basement→room pin)",
    )
    p.add_argument("--max-frames", type=int, default=20_000)
    p.add_argument("--assist", action="store_true", help="Enable energy+ammo refill")
    p.add_argument("--body-only", action="store_true", help="Stop at HP 0 (skip boss bit)")
    p.add_argument("--save-state", nargs="?", const=True, default=False)
    p.add_argument("--report", type=Path, default=None)
    p.set_defaults(func=cmd_strategy)

    d = sub.add_parser("dump", help="Idle-dump RAM / distinct spritemaps from entry")
    d.add_argument("--state", default="natural")
    d.add_argument("--frames", type=int, default=400)
    d.add_argument("--stride", type=int, default=30)
    d.add_argument("--report", type=Path, default=None)
    d.set_defaults(func=cmd_dump)

    w = sub.add_parser("window", help="Seat + N fire windows (no 12k fight)")
    w.add_argument("--state", default="natural")
    w.add_argument("--wait", type=int, default=2400, help="Frames to wait for eye_open")
    w.add_argument("--windows", type=int, default=1, help="Seated windows to fire in a row")
    w.add_argument(
        "--weapon",
        choices=("beam", "missiles"),
        default="missiles",
        help="missiles = counted ammo chips; beam = charge shots",
    )
    w.add_argument("--report", type=Path, default=None)
    w.set_defaults(func=cmd_window)

    args = parser.parse_args()
    if not getattr(args, "command", None):
        parser.print_help()
        return 2
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
