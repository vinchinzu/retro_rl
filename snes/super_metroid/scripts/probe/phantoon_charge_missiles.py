#!/usr/bin/env python3
"""Probe the wiki KPDR Charge Plus Missiles Phantoon policy (MassHesteria).

Pin: scratch/post_ws_basement_to_phantoon.state. Assist ON for strategy/bench.
This is not the product charge-only body (20537f ×2). Do not wire into spine.

```bash
# One-window first: 2-missile opener, halt unless HP chips
uv run python snes/super_metroid/scripts/probe/phantoon_charge_missiles.py window --assist

# Full fight (only after the opener chips)
uv run python snes/super_metroid/scripts/probe/phantoon_charge_missiles.py strategy --assist \
  --report snes/super_metroid/scratch/phantoon_wiki_charge_missiles.json

# Dual-run vs 20537f (only if first fight is green)
uv run python snes/super_metroid/scripts/probe/phantoon_charge_missiles.py bench --assist
```
"""

from __future__ import annotations

import argparse
from pathlib import Path

from super_metroid.assist import UnlimitedResourcesAssist
from super_metroid.combat.features import phantoon_catalog
from super_metroid.combat.phantoon import (
    ROOM_PHANTOON,
    beam_charge,
    enemy_extra,
    eye_open,
    seated,
)
from super_metroid.combat.phantoon_charge_missiles import (
    PRODUCT_BENCH_FRAMES,
    WIKI_URL,
    ChargeMissilesStrategy,
    play_charge_missiles_fight,
    play_first_missile_window,
)
from super_metroid.combat.probe import (
    ProbeSession,
    open_state_env,
    resolve_named_state,
    write_json_report,
)
from super_metroid.dev.phantoon_dev import phantoon_defeated, wrecked_ship_boss_bits
from super_metroid.paths import GAME_DIR, SCRATCH_STATE_DIR
from super_metroid.room_timer import format_segment_time

DEFAULT_ENTRY = SCRATCH_STATE_DIR / "post_ws_basement_to_phantoon.state"
DEFAULT_REPORT = GAME_DIR / "scratch" / "phantoon_wiki_charge_missiles.json"
MISSION_PIN = GAME_DIR / "scratch" / "post_ws_basement_to_phantoon.state"

_NAMED_STATES: dict[str, Path] = {
    "natural": DEFAULT_ENTRY,
    "post_ws_basement_to_phantoon": DEFAULT_ENTRY,
    "mission": MISSION_PIN if MISSION_PIN.exists() else DEFAULT_ENTRY,
}


def _resolve_state(name: str) -> Path:
    return resolve_named_state(name, _NAMED_STATES)


def _open_env(state_path: Path):
    return open_state_env(
        state_path,
        missing_hint="Need scratch/post_ws_basement_to_phantoon.state (rr-cjpp).",
    )


def _assist(enabled: bool) -> UnlimitedResourcesAssist:
    # Energy keeps $D82A from eating the tank. Ammo stays natural so missile
    # spends are visible (20 missiles cover 4×2+2 rounds). Super stays unused.
    return UnlimitedResourcesAssist(unlimited_energy=enabled, unlimited_ammo=False)


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
        "game_state": st.game_state,
        "missiles": st.missiles,
        "super_missiles": st.super_missiles,
        "selected_item": st.selected_item,
        "equipped_beams": f"0x{st.equipped_beams:04X}",
        "charge": beam_charge(session.env),
        "enemy0_hp": st.enemy0_hp,
        "enemy0_x": st.enemy0_x,
        "enemy0_y": st.enemy0_y,
        "enemy0_spritemap": f"0x{st.enemy0_spritemap:04X}",
        "func": extra.get("func"),
        "eye_ilist": extra.get("eye_ilist"),
        "eye_open": eye_open(st, session.env),
        "seated": seated(st),
        "time": format_segment_time(session.frame),
    }


def _assist_tel(assist: UnlimitedResourcesAssist) -> dict[str, object]:
    tel = assist.telemetry
    return {
        "energy_restored": tel.energy.restored,
        "energy_writes": tel.energy.writes,
        "missile_writes": tel.ammo["missiles"].writes,
        "super_writes": tel.ammo["super_missiles"].writes,
        "maximum_single_frame_damage": tel.maximum_single_frame_damage,
        "deaths": tel.deaths,
    }


def _strategy(args: argparse.Namespace) -> ChargeMissilesStrategy:
    return ChargeMissilesStrategy(
        allow_super=bool(args.allow_super),
        max_fight_frames=int(args.max_frames),
    )


def cmd_window(args: argparse.Namespace) -> int:
    state_path = _resolve_state(args.state)
    env, loaded = _open_env(state_path)
    assist = _assist(bool(args.assist))
    try:
        session = ProbeSession(env, assist)
        entry = _snapshot(session)
        window = play_first_missile_window(session, _strategy(args))
        success = bool(window.get("success"))
        report = {
            "command": "window",
            "state": loaded,
            "success": success,
            "assist_enabled": bool(args.assist),
            "wiki": WIKI_URL,
            "recipe": "2 missiles (10f) opener only",
            "entry": entry,
            "window": window,
            "final": _snapshot(session),
            "assist": _assist_tel(assist),
            "time": format_segment_time(session.frame),
            "notes": (
                "One-window first. Do not start a 20k fight until this chips HP. "
                "Hits counted by ammo/HP delta, not X."
            ),
        }
        write_json_report(report, args.report)
        return 0 if success else 1
    finally:
        env.close()


def _run_fight(args: argparse.Namespace) -> tuple[dict[str, object], bool]:
    state_path = _resolve_state(args.state)
    env, loaded = _open_env(state_path)
    assist = _assist(bool(args.assist))
    try:
        session = ProbeSession(env, assist)
        if session.state.room_id != ROOM_PHANTOON:
            report = {
                "command": "strategy",
                "state": loaded,
                "success": False,
                "outcome": "wrong_room",
                "room_id_hex": f"0x{session.state.room_id:04X}",
                "time": format_segment_time(session.frame),
            }
            return report, False
        entry = _snapshot(session)
        evidence = play_charge_missiles_fight(
            session, strategy=_strategy(args), require_boss_bit=not args.body_only
        )
        fight = evidence.to_dict()
        success = evidence.outcome == "phantoon_defeated" or (
            args.body_only and evidence.body_zero_frame is not None
        )
        tel = _assist_tel(assist)
        report = {
            "command": "strategy",
            "state": loaded,
            "success": success,
            "assist_enabled": bool(args.assist),
            "wiki": WIKI_URL,
            "method": "kpdr_charge_plus_missiles",
            "allow_super": bool(args.allow_super),
            "entry": entry,
            "fight": fight,
            "shots": fight["shots"],
            "rounds": evidence.rounds,
            "windows": evidence.windows,
            "hp": evidence.final_body_hp,
            "boss_bit_set": evidence.boss_bit_set,
            "boss_bits_wrecked_ship": wrecked_ship_boss_bits(env),
            "phantoon_defeated": phantoon_defeated(env),
            "catalog": {"name": phantoon_catalog().name, "max_hp": phantoon_catalog().max_hp},
            "product_bench_frames": PRODUCT_BENCH_FRAMES,
            "delta_vs_product": evidence.action_frames - PRODUCT_BENCH_FRAMES,
            "reasons": dict(session.action_reasons),
            "assist": tel,
            "final": _snapshot(session),
            "time": format_segment_time(evidence.action_frames),
            "notes": (
                "Wiki 2+2+charge. Super default off. Do not wire into spine. "
                "Compare to product charge-only 20537f."
            ),
        }
        return report, success
    finally:
        env.close()


def cmd_strategy(args: argparse.Namespace) -> int:
    report, success = _run_fight(args)
    write_json_report(report, args.report)
    return 0 if success else 1


def cmd_bench(args: argparse.Namespace) -> int:
    rows: list[dict[str, object]] = []
    for index in (1, 2):
        report, success = _run_fight(args)
        timed = report.get("time") if isinstance(report.get("time"), dict) else format_segment_time(0)
        rows.append(
            {
                "run": index,
                "success": success,
                "frames": timed.get("frames") if isinstance(timed, dict) else 0,
                "seconds": timed.get("seconds") if isinstance(timed, dict) else 0,
                "clock": timed.get("clock") if isinstance(timed, dict) else "",
                "time": timed,
                "hp": report.get("hp"),
                "boss_bit_set": report.get("boss_bit_set"),
                "rounds": report.get("rounds"),
                "shots": report.get("shots"),
                "outcome": (report.get("fight") or {}).get("outcome") if isinstance(report.get("fight"), dict) else report.get("outcome"),
                "assist": report.get("assist"),
                "delta_vs_product": report.get("delta_vs_product"),
            }
        )
        if index == 1 and not success:
            break
    both_green = len(rows) == 2 and all(bool(r["success"]) for r in rows)
    frames = [int(r["frames"] or 0) for r in rows]
    report = {
        "command": "bench",
        "wiki": WIKI_URL,
        "product_bench_frames": PRODUCT_BENCH_FRAMES,
        "product_time": format_segment_time(PRODUCT_BENCH_FRAMES),
        "runs": rows,
        "success": both_green,
        "dual": both_green and len(set(frames)) == 1,
        "time": format_segment_time(frames[0] if frames else 0),
        "notes": (
            "Dual-run only if first fight is green (HP 0 + boss bit). "
            "Do not replace product 20537f charge-only."
        ),
    }
    out = args.report if args.report is not None else DEFAULT_REPORT
    write_json_report(report, out)
    return 0 if both_green else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    def _common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--state", default="natural")
        p.add_argument("--max-frames", type=int, default=40_000)
        p.add_argument("--assist", action="store_true")
        p.add_argument("--allow-super", action="store_true", help="Arm Super only at HP≤600")
        p.add_argument("--body-only", action="store_true")
        p.add_argument("--report", type=Path, default=None)

    w = sub.add_parser("window", help="Seat + first 2-missile opener (no 20k fight)")
    _common(w)
    w.set_defaults(func=cmd_window)

    s = sub.add_parser("strategy", help="Full 2+2+charge fight")
    _common(s)
    s.set_defaults(func=cmd_strategy)

    b = sub.add_parser("bench", help="Dual-run vs product 20537f (green first only)")
    _common(b)
    b.set_defaults(func=cmd_bench)

    args = parser.parse_args()
    if not getattr(args, "command", None):
        parser.print_help()
        return 2
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
