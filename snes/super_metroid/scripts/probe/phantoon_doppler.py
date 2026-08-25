#!/usr/bin/env python3
"""Probe wiki missile-doppler Phantoon (PRKD 2-2-N, KPDR inventory).

Assist ON so drop RNG is not the limiter. One-window first — prove missiles
spend AND HP chips before any 20k fight. Halt at first miss.

```bash
QT_QPA_PLATFORM=offscreen uv run python snes/super_metroid/scripts/probe/phantoon_doppler.py window --assist --report snes/super_metroid/scratch/phantoon_wiki_doppler_window.json
QT_QPA_PLATFORM=offscreen uv run python snes/super_metroid/scripts/probe/phantoon_doppler.py strategy --assist --report snes/super_metroid/scratch/phantoon_wiki_doppler.json
```
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from super_metroid.assist import UnlimitedResourcesAssist
from super_metroid.combat.phantoon import ROOM_PHANTOON, beam_charge, enemy_extra, eye_open, seated
from super_metroid.combat.phantoon_doppler import (
    DopplerStrategy,
    play_phantoon_doppler_fight,
    play_phantoon_doppler_window,
)
from super_metroid.combat.probe import (
    ProbeSession,
    open_state_env,
    resolve_named_state,
    write_json_report,
)
from super_metroid.dev.phantoon_dev import phantoon_defeated, wrecked_ship_boss_bits
from super_metroid.paths import SCRATCH_STATE_DIR
from super_metroid.room_timer import format_segment_time

DEFAULT_ENTRY = SCRATCH_STATE_DIR / "post_ws_basement_to_phantoon.state"
BASELINE_FRAMES = 20_537

_NAMED_STATES: dict[str, Path] = {
    "natural": DEFAULT_ENTRY,
    "post_ws_basement_to_phantoon": DEFAULT_ENTRY,
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
        "game_state": st.game_state,
        "missiles": st.missiles,
        "super_missiles": st.super_missiles,
        "max_missiles": st.max_missiles,
        "selected_item": st.selected_item,
        "equipped_beams": f"0x{st.equipped_beams:04X}",
        "collected_items": f"0x{st.collected_items:04X}",
        "charge": beam_charge(session.env),
        "enemy0_hp": st.enemy0_hp,
        "enemy0_x": st.enemy0_x,
        "enemy0_y": st.enemy0_y,
        "enemy0_spritemap": f"0x{st.enemy0_spritemap:04X}",
        "func": extra.get("func"),
        "eye_ilist": extra.get("eye_ilist"),
        "eye_open": eye_open(st, session.env),
        "seated": seated(st),
        "phase": st.phase.value,
    }


def _assist_block(assist: UnlimitedResourcesAssist) -> dict[str, object]:
    tel = assist.telemetry
    return {
        "energy_restored": tel.energy.restored,
        "energy_writes": tel.energy.writes,
        "missile_writes": tel.ammo["missiles"].writes,
        "missile_restored": tel.ammo["missiles"].restored,
        "super_writes": tel.ammo["super_missiles"].writes,
        "maximum_single_frame_damage": tel.maximum_single_frame_damage,
        "deaths": tel.deaths,
    }


def cmd_window(args: argparse.Namespace) -> int:
    state_path = _resolve_state(args.state)
    env, loaded = _open_env(state_path)
    assist = UnlimitedResourcesAssist(
        unlimited_energy=bool(args.assist),
        unlimited_ammo=bool(args.assist),
    )
    try:
        session = ProbeSession(env, assist)
        entry = _snapshot(session)
        got = play_phantoon_doppler_window(
            session, strategy=DopplerStrategy(), wait=args.wait
        )
        payload = got.to_dict()
        # Proof: ammo actually decreased and HP chipped. halt_miss means we
        # stopped after a later extra failed to land — still a green window.
        success = got.missiles_spent >= 1 and got.hp_drop > 0
        timed = format_segment_time(session.frame)
        report = {
            "command": "window",
            "state": loaded,
            "success": success,
            "halt_miss": got.halt_miss,
            "missiles_spent": got.missiles_spent,
            "super_spent": got.super_spent,
            "hp_drop": got.hp_drop,
            "recipe": payload["recipe"],
            "close_eye_extra": got.close_eye_extra,
            "pair1": got.pair1,
            "pair2": got.pair2,
            "extra": got.extra,
            "timing": timed,
            "assist_enabled": bool(args.assist),
            "assist": _assist_block(assist),
            "entry": entry,
            "final": _snapshot(session),
            "window": payload,
            "notes": (
                "One barrage. Halt at first miss (ammo spend without HP chip). "
                "Assist ON refills ammo so snapshot missile delta may be 0."
            ),
        }
        write_json_report(report, args.report)
        return 0 if success else 1
    finally:
        env.close()


def _run_strategy(args: argparse.Namespace) -> dict[str, object]:
    state_path = _resolve_state(args.state)
    env, loaded = _open_env(state_path)
    assist = UnlimitedResourcesAssist(
        unlimited_energy=bool(args.assist),
        unlimited_ammo=bool(args.assist),
    )
    try:
        session = ProbeSession(env, assist)
        if session.state.room_id != ROOM_PHANTOON:
            return {
                "command": "strategy",
                "state": loaded,
                "success": False,
                "outcome": "wrong_room",
                "room_id_hex": f"0x{session.state.room_id:04X}",
            }
        entry = _snapshot(session)
        evidence = play_phantoon_doppler_fight(
            session,
            strategy=DopplerStrategy(max_fight_frames=args.max_frames),
            require_boss_bit=not args.body_only,
        )
        success = evidence.outcome == "phantoon_defeated" or (
            args.body_only and evidence.body_zero_frame is not None
        )
        timed = format_segment_time(evidence.action_frames)
        fight = evidence.to_dict()
        return {
            "command": "strategy",
            "state": loaded,
            "success": success,
            "assist_enabled": bool(args.assist),
            "frames": evidence.action_frames,
            "seconds": timed["seconds"],
            "clock": timed["clock"],
            "timing": timed,
            "baseline_frames": BASELINE_FRAMES,
            "vs_baseline": evidence.action_frames - BASELINE_FRAMES,
            "missiles_spent": evidence.missiles_spent,
            "max_missiles_in_one_barrage": evidence.max_barrage,
            "rounds": evidence.rounds,
            "super_used": evidence.super_spent > 0,
            "super_spent": evidence.super_spent,
            "close_eye_extra": evidence.close_eye_extra,
            "hp": evidence.final_body_hp,
            "boss_bit": evidence.boss_bit_set,
            "outcome": evidence.outcome,
            "entry": entry,
            "fight": fight,
            "reasons": dict(session.action_reasons),
            "assist": _assist_block(assist),
            "final": _snapshot(session),
            "boss_bits_wrecked_ship": wrecked_ship_boss_bits(env),
            "phantoon_defeated": phantoon_defeated(env),
            "method": "wiki_missile_doppler_2_2_n",
            "notes": (
                "Wiki 2-2-N doppler, 10f spacing, Super-only-if-kill (HP≤600). "
                "Assist ON. Spine product (rr-asyg). Charge-only 20537f is research."
            ),
        }
    finally:
        env.close()


def cmd_strategy(args: argparse.Namespace) -> int:
    report = _run_strategy(args)
    write_json_report(report, args.report)
    if not (args.dual and report.get("success")):
        return 0 if report.get("success") else 1
    report2 = _run_strategy(args)
    dual_exact = (
        report.get("success")
        and report2.get("success")
        and report.get("frames") == report2.get("frames")
    )
    dual = {
        "success": dual_exact,
        "dual_exact": dual_exact,
        "frames": [report.get("frames"), report2.get("frames")],
        "runs": [report, report2],
    }
    dual_path = None
    if args.report is not None:
        dual_path = args.report.with_name(args.report.stem + "_dual" + args.report.suffix)
        dual_path.parent.mkdir(parents=True, exist_ok=True)
        dual_path.write_text(json.dumps(dual, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {dual_path}")
    print(json.dumps({"dual_exact": dual_exact, "frames": dual["frames"]}, indent=2))
    return 0 if dual_exact else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    w = sub.add_parser("window", help="Seat + one 2-2-N barrage (halt at first miss)")
    w.add_argument("--state", default="natural")
    w.add_argument("--wait", type=int, default=2400)
    w.add_argument("--assist", action="store_true")
    w.add_argument("--report", type=Path, default=None)
    w.set_defaults(func=cmd_window)

    p = sub.add_parser("strategy", help="Full wiki-doppler fight")
    p.add_argument("--state", default="natural")
    p.add_argument("--max-frames", type=int, default=40_000)
    p.add_argument("--assist", action="store_true")
    p.add_argument("--body-only", action="store_true")
    p.add_argument("--dual", action="store_true")
    p.add_argument("--report", type=Path, default=None)
    p.set_defaults(func=cmd_strategy)

    args = parser.parse_args()
    if not getattr(args, "command", None):
        parser.print_help()
        return 2
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
