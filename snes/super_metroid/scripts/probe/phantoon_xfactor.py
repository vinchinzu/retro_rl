#!/usr/bin/env python3
"""Probe Ice-on X-Factor / popTOON Phantoon (wiki 2-round).

Window first. If Ice-on Wave Shield does not chip ``enemy0_hp``, halt.
Do not Super-spray, do not pause-menu unequip Ice, do not replace 20537f.

```bash
QT_QPA_PLATFORM=offscreen uv run python \\
  snes/super_metroid/scripts/probe/phantoon_xfactor.py window --assist \\
  --report snes/super_metroid/scratch/phantoon_wiki_xfactor_window.json
```
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from super_metroid.assist import UnlimitedResourcesAssist
from super_metroid.combat.phantoon import (
    ROOM_PHANTOON,
    WEAPON_MISSILES,
    WEAPON_SUPERS,
    PhantoonStrategy,
    _go_to_seat,
    seated,
)
from super_metroid.combat.phantoon_xfactor import (
    MISSILE_COOLDOWN,
    PIN_BEAMS,
    PIN_ITEMS,
    PoptoonProgress,
    PoptoonStep,
    WEAPON_POWER_BOMBS,
    attempt_xfactor,
    ice_equipped,
    next_poptoon_step,
    super_ok,
    true_wave_shield,
    wait_charge_window,
    xfactor_snapshot,
)
from super_metroid.combat.primitives import ensure_weapon
from super_metroid.routes.runtime import hold
from super_metroid.combat.probe import (
    ProbeSession,
    open_state_env,
    resolve_named_state,
    write_json_report,
)
from super_metroid.paths import GAME_DIR, SCRATCH_STATE_DIR
from super_metroid.room_timer import format_segment_time

DEFAULT_ENTRY = SCRATCH_STATE_DIR / "post_ws_basement_to_phantoon.state"
REPORT_DIR = GAME_DIR / "scratch"
WINDOW_REPORT = REPORT_DIR / "phantoon_wiki_xfactor_window.json"
FIGHT_REPORT = REPORT_DIR / "phantoon_wiki_xfactor.json"
PRODUCT_BASELINE_FRAMES = 20537

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


def _assist(args: argparse.Namespace) -> UnlimitedResourcesAssist:
    on = bool(getattr(args, "assist", False))
    return UnlimitedResourcesAssist(unlimited_energy=on, unlimited_ammo=on)


def cmd_dump(args: argparse.Namespace) -> int:
    state_path = _resolve_state(args.state)
    env, loaded = _open_env(state_path)
    try:
        session = ProbeSession(env, _assist(args))
        entry = xfactor_snapshot(session)
        func_log: list[dict[str, object]] = []
        opened = False
        if args.idle:
            samples = [entry]
            for i in range(args.frames):
                session.step([0] * 12, "idle")
                if i % max(1, args.stride) == 0:
                    samples.append(xfactor_snapshot(session))
            final = samples[-1]
        else:
            _go_to_seat(session, PhantoonStrategy())
            opened = wait_charge_window(
                session, timeout=args.wait, func_log=func_log
            )
            samples = [entry, xfactor_snapshot(session)]
            final = samples[-1]
        beams = int(session.state.equipped_beams)
        report = {
            "command": "dump",
            "state": loaded,
            "opened": opened,
            "idle": bool(args.idle),
            "pin_beams": f"0x{PIN_BEAMS:04X}",
            "pin_items": f"0x{PIN_ITEMS:04X}",
            "measured_beams": f"0x{beams:04X}",
            "measured_items": f"0x{session.state.equipped_items:04X}",
            "ice_equipped": ice_equipped(beams),
            "true_wave_shield": true_wave_shield(beams),
            "spazer_in_word": bool(beams & 0x0004),
            "entry": entry,
            "samples": samples,
            "func_log": func_log,
            "final": final,
            "time": format_segment_time(session.frame),
        }
        write_json_report(report, args.report)
        return 0
    finally:
        env.close()


def _window_attempt(
    session: ProbeSession, *, wait: int
) -> tuple[dict[str, object], object]:
    entry = xfactor_snapshot(session)
    _go_to_seat(session, PhantoonStrategy())
    seated_snap = xfactor_snapshot(session)
    func_log: list[dict[str, object]] = []
    opened = wait_charge_window(session, timeout=wait, func_log=func_log)
    pre = xfactor_snapshot(session)
    evidence = attempt_xfactor(session)
    post = xfactor_snapshot(session)
    miss_dump: list[dict[str, object]] = []
    if not evidence.chips:
        for _ in range(40):
            miss_dump.append(xfactor_snapshot(session))
            if int(session.state.health) == 0:
                break
            session.step([0] * 12, "xf_miss_dump")
    payload = {
        "opened": opened,
        "chips": evidence.chips,
        "success": bool(opened and evidence.chips),
        "entry": entry,
        "seated": seated_snap,
        "pre_fire": pre,
        "post_fire": post,
        "evidence": evidence.to_dict(),
        "func_log": func_log,
        "miss_dump": miss_dump,
        "final": xfactor_snapshot(session),
        "time": format_segment_time(session.frame),
        "seated_after": seated(session.state),
    }
    return payload, evidence


def _tap_missiles(session: ProbeSession, n: int) -> int:
    try:
        ensure_weapon(session, WEAPON_MISSILES)
    except RuntimeError:
        return 0
    fired = 0
    for _ in range(n * (MISSILE_COOLDOWN + 6) + 24):
        if fired >= n or int(session.state.health) == 0:
            break
        if int(session.state.missiles) <= 0:
            break
        ms = int(session.state.missiles)
        hold(session, 2, "X", reason="poptoon_ms")
        if int(session.state.missiles) < ms:
            fired += 1
            hold(session, MISSILE_COOLDOWN, reason="poptoon_ms_cd")
    return fired


def _kill_super(session: ProbeSession) -> bool:
    if not super_ok(int(session.state.enemy0_hp)):
        return False
    try:
        ensure_weapon(session, WEAPON_SUPERS)
    except RuntimeError:
        return False
    hold(session, 2, "X", reason="poptoon_super")
    return True


def play_poptoon_fight(session: ProbeSession, *, combo_chips: bool, max_frames: int) -> dict[str, object]:
    """Live 2+2+XF / 2+2+S. Only after a measured chip. Super iff kill."""
    start = session.frame
    if not combo_chips:
        return {"outcome": "blocked_ice", "success": False, "action_frames": 0}
    _go_to_seat(session, PhantoonStrategy())
    progress = PoptoonProgress()
    ice_on = ice_equipped(int(session.state.equipped_beams))
    peak = min_hp = int(session.state.enemy0_hp)
    shots = windows = xf_attempts = 0
    blocked = (PoptoonStep.DONE, PoptoonStep.BLOCKED_ICE, PoptoonStep.BLOCKED_NO_PB)
    while session.frame - start < max_frames:
        st = session.state
        if int(st.health) == 0 or int(st.enemy0_hp) == 0:
            break
        peak, min_hp = max(peak, int(st.enemy0_hp)), min(min_hp, int(st.enemy0_hp))
        step = next_poptoon_step(
            progress,
            hp=int(st.enemy0_hp),
            power_bombs=int(st.power_bombs),
            ice_on=ice_on,
            combo_chips=combo_chips,
        )
        if step in blocked:
            break
        if not wait_charge_window(session, timeout=2400) or int(session.state.health) == 0:
            break
        windows += 1
        if step is PoptoonStep.FIRE_MISSILE:
            got = _tap_missiles(session, min(2, 4 - progress.missiles_this_round))
            shots += got
            progress.missiles_this_round += got
        elif step is PoptoonStep.CHARGE_XFACTOR:
            ev = attempt_xfactor(session)
            xf_attempts += 1
            progress.xfactor_fired = True
            min_hp = min(min_hp, ev.hp_after)
            if progress.round_index == 1:
                progress = PoptoonProgress(round_index=2)
        elif step is PoptoonStep.FIRE_SUPER:
            if not _kill_super(session):
                break
            progress.super_fired = True
        else:
            session.step([0] * 12, "poptoon_idle")
    hp = int(session.state.enemy0_hp)
    boss = bool(xfactor_snapshot(session).get("boss_bit"))
    if int(session.state.health) == 0:
        outcome = "died"
    elif boss and hp == 0:
        outcome = "phantoon_defeated"
    elif hp == 0:
        outcome = "phantoon_body_zero_no_boss_bit"
    else:
        outcome = "timeout"
    return {
        "outcome": outcome,
        "success": outcome == "phantoon_defeated",
        "action_frames": session.frame - start,
        "peak_body_hp": peak,
        "min_body_hp": min_hp,
        "final_body_hp": hp,
        "boss_bit_set": boss,
        "shots_fired": shots,
        "windows": windows,
        "xf_attempts": xf_attempts,
    }


def cmd_window(args: argparse.Namespace) -> int:
    state_path = _resolve_state(args.state)
    env, loaded = _open_env(state_path)
    try:
        session = ProbeSession(env, _assist(args))
        payload, evidence = _window_attempt(session, wait=args.wait)
        beams = int(session.state.equipped_beams)
        report = {
            "command": "window",
            "state": loaded,
            "assist_enabled": bool(args.assist),
            "success": payload["success"],
            "chips": evidence.chips,
            "outcome": evidence.outcome,
            "hp_drop": evidence.hp_drop,
            "hp_before": evidence.hp_before,
            "hp_after": evidence.hp_after,
            "pb_spent": evidence.pb_spent,
            "charge_peak": evidence.charge_peak,
            "combo_class": evidence.combo_class,
            "projectile_types": [f"0x{t:04X}" for t in evidence.projectile_types],
            "ice_equipped": ice_equipped(beams),
            "true_wave_shield": true_wave_shield(beams),
            "measured_beams": f"0x{beams:04X}",
            "measured_items": f"0x{session.state.equipped_items:04X}",
            "spazer_in_word": bool(beams & 0x0004),
            "product_baseline_frames": PRODUCT_BASELINE_FRAMES,
            "wire": False,
            "notes": evidence.notes,
            **payload,
        }
        out = args.report if args.report is not None else WINDOW_REPORT
        write_json_report(report, out)
        return 0 if evidence.chips else 1
    finally:
        env.close()


def cmd_strategy(args: argparse.Namespace) -> int:
    state_path = _resolve_state(args.state)
    env, loaded = _open_env(state_path)
    try:
        session = ProbeSession(env, _assist(args))
        if session.state.room_id != ROOM_PHANTOON:
            report = {
                "command": "strategy",
                "state": loaded,
                "success": False,
                "outcome": "wrong_room",
                "room_id_hex": f"0x{session.state.room_id:04X}",
            }
            write_json_report(report, args.report)
            return 1
        payload, evidence = _window_attempt(session, wait=args.wait)
        window_path = WINDOW_REPORT
        window_path.parent.mkdir(parents=True, exist_ok=True)
        window_path.write_text(
            json.dumps(
                {
                    "command": "window",
                    "state": loaded,
                    "success": payload["success"],
                    "chips": evidence.chips,
                    "outcome": evidence.outcome,
                    "hp_drop": evidence.hp_drop,
                    "evidence": evidence.to_dict(),
                    "pre_fire": payload["pre_fire"],
                    "post_fire": payload["post_fire"],
                    "miss_dump": payload["miss_dump"],
                    "time": payload["time"],
                    "notes": evidence.notes,
                    "wire": False,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if not evidence.chips:
            report = {
                "command": "strategy",
                "state": loaded,
                "success": False,
                "outcome": "ice_on_xfactor_miss",
                "fight_ran": False,
                "window": payload,
                "product_baseline_frames": PRODUCT_BASELINE_FRAMES,
                "wire": False,
                "notes": evidence.notes,
                "time": format_segment_time(session.frame),
            }
            write_json_report(report, args.report)
            return 1
        fight = play_poptoon_fight(
            session,
            combo_chips=True,
            max_frames=args.max_frames,
        )
        success = bool(fight.get("success"))
        report = {
            "command": "strategy",
            "state": loaded,
            "success": success,
            "assist_enabled": bool(args.assist),
            "fight_ran": True,
            "window": payload,
            "fight": fight,
            "product_baseline_frames": PRODUCT_BASELINE_FRAMES,
            "delta_vs_20537f": int(fight.get("action_frames") or 0)
            - PRODUCT_BASELINE_FRAMES,
            "wire": success,
            "final": xfactor_snapshot(session),
            "time": format_segment_time(session.frame),
            "selected": WEAPON_POWER_BOMBS,
        }
        out = args.report if args.report is not None else FIGHT_REPORT
        write_json_report(report, out)
        return 0 if success else 1
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    d = sub.add_parser("dump", help="Idle or seat+wait first open")
    d.add_argument("--state", default="natural")
    d.add_argument("--frames", type=int, default=400)
    d.add_argument("--stride", type=int, default=30)
    d.add_argument("--wait", type=int, default=2400)
    d.add_argument("--idle", action="store_true")
    d.add_argument("--assist", action="store_true")
    d.add_argument("--report", type=Path, default=None)
    d.set_defaults(func=cmd_dump)

    w = sub.add_parser("window", help="One X-Factor attempt; halt on miss")
    w.add_argument("--state", default="natural")
    w.add_argument("--wait", type=int, default=2400)
    w.add_argument("--assist", action="store_true")
    w.add_argument("--report", type=Path, default=WINDOW_REPORT)
    w.set_defaults(func=cmd_window)

    s = sub.add_parser("strategy", help="Full fight only if window chipped")
    s.add_argument("--state", default="natural")
    s.add_argument("--wait", type=int, default=2400)
    s.add_argument("--max-frames", type=int, default=8_000)
    s.add_argument("--assist", action="store_true")
    s.add_argument("--report", type=Path, default=None)
    s.set_defaults(func=cmd_strategy)

    args = parser.parse_args()
    if not getattr(args, "command", None):
        parser.print_help()
        return 2
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
