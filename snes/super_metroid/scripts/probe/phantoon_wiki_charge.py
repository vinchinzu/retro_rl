#!/usr/bin/env python3
"""Bench wiki beginner Ice/Wave/Spazer charge-only Phantoon (assist).

Public policy (https://wiki.supermetroid.run/Phantoon#Any.25_KPDR_.28Ice.2FWave.2FSpazer_Charge_Only.29):
Charge Beam + Wave + Spazer (+ Ice is equipped 0x1007). HP 2500. Charge
shot when the eye opens, then two more so he disappears (~300 dmg/charge,
3 per round). Repeat ~4 rounds. Left corner. Crouch/snipe rain. Never
Super except a last-round finisher that actually kills (do not Super-spray;
enrage is 8 flame waves). Eye opens after 1 / 6 / 11 s (fast/mid/slow),
left or right. Round-1 has 6 positions; later rounds another 6. 300+ dmg
in one barrage makes him disappear; space shots if you want 3 in one window.

Does **not** rewrite ``combat/phantoon.py``. Benches the product body:

- ``probe_default`` — ``PhantoonStrategy(weapon=beam, shots_per_window=1)``
- ``spine_three`` — spine hop ``shots_per_window=3``

```bash
QT_QPA_PLATFORM=offscreen uv run python \\
  snes/super_metroid/scripts/probe/phantoon_wiki_charge.py bench
```
"""

from __future__ import annotations

import argparse
from pathlib import Path

from super_metroid.assist import UnlimitedResourcesAssist
from super_metroid.combat.phantoon import (
    CHARGE_FULL,
    ROOM_PHANTOON,
    WEAPON_BEAM,
    PhantoonStrategy,
    beam_charge,
    play_phantoon_fight,
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

WIKI_URL = (
    "https://wiki.supermetroid.run/Phantoon"
    "#Any.25_KPDR_.28Ice.2FWave.2FSpazer_Charge_Only.29"
)
BASELINE_FRAMES = 20_537
DEFAULT_ENTRY = SCRATCH_STATE_DIR / "post_ws_basement_to_phantoon.state"
DEFAULT_REPORT = GAME_DIR / "scratch" / "phantoon_wiki_charge_only.json"
CHIP_WINDOW_GAP = 240

POLICIES: dict[str, int] = {
    "probe_default": 1,
    "spine_three": 3,
}

_NAMED_STATES: dict[str, Path] = {
    "natural": DEFAULT_ENTRY,
    "post_ws_basement_to_phantoon": DEFAULT_ENTRY,
    "pin": DEFAULT_ENTRY,
}


def make_strategy(policy: str, max_frames: int) -> PhantoonStrategy:
    """Product-body strategy for a named wiki-charge bench policy."""
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; expected {tuple(POLICIES)}")
    return PhantoonStrategy(
        weapon=WEAPON_BEAM,
        shots_per_window=POLICIES[policy],
        max_fight_frames=max_frames,
    )


def group_window_chips(
    chips: list[dict[str, object]], *, gap: int = CHIP_WINDOW_GAP
) -> list[list[dict[str, object]]]:
    """Cluster HP chips whose frames are within *gap* into one barrage."""
    groups: list[list[dict[str, object]]] = []
    current: list[dict[str, object]] = []
    last: int | None = None
    for chip in chips:
        drop = int(chip.get("drop") or 0)
        if drop <= 0:
            continue
        frame = int(chip["frame"])
        if last is not None and frame - last > gap:
            groups.append(current)
            current = []
        current.append(chip)
        last = frame
    if current:
        groups.append(current)
    return groups


def summarize_window_chips(groups: list[list[dict[str, object]]]) -> dict[str, object]:
    """Did shots_per_window=3 land 3 chips, or does 300 dmg close the eye?"""
    sizes = [len(group) for group in groups]
    drops = [[int(chip["drop"]) for chip in group] for group in groups]
    early = sizes[:-1] if len(sizes) > 1 else sizes
    return {
        "window_count": len(groups),
        "shots_per_window": sizes,
        "mean": round(sum(sizes) / len(sizes), 3) if sizes else 0.0,
        "max": max(sizes) if sizes else 0,
        "drops": drops,
        "three_chips_landed": any(size >= 3 for size in sizes),
        "disappear_after_300": bool(early) and all(size == 1 for size in early),
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
    return {
        "room_id_hex": f"0x{st.room_id:04X}",
        "samus_x": st.samus_x,
        "samus_y": st.samus_y,
        "pose": st.pose,
        "health": st.health,
        "game_state": st.game_state,
        "missiles": st.missiles,
        "super_missiles": st.super_missiles,
        "selected_item": st.selected_item,
        "equipped_beams": f"0x{st.equipped_beams:04X}",
        "collected_items": f"0x{st.collected_items:04X}",
        "enemy0_hp": st.enemy0_hp,
        "enemy0_x": st.enemy0_x,
        "enemy0_y": st.enemy0_y,
    }


def _attach_chip_trace(session: ProbeSession) -> list[dict[str, object]]:
    chips: list[dict[str, object]] = []
    orig = session.step
    prev_hp = int(session.state.enemy0_hp)
    prev_charge = beam_charge(session.env)

    def tracked(action, reason: str):
        nonlocal prev_hp, prev_charge
        st = orig(action, reason)
        charge = beam_charge(session.env)
        hp = int(st.enemy0_hp)
        drop = prev_hp - hp if hp < prev_hp else 0
        spent = charge < prev_charge and prev_charge >= CHARGE_FULL
        if drop or spent:
            chips.append(
                {
                    "frame": session.frame,
                    "reason": str(reason),
                    "hp_before": prev_hp,
                    "hp_after": hp,
                    "drop": drop,
                    "charge_before": prev_charge,
                    "charge_after": charge,
                }
            )
        prev_hp = hp
        prev_charge = charge
        return st

    session.step = tracked  # type: ignore[method-assign]
    return chips


def _assist_block(assist: UnlimitedResourcesAssist) -> dict[str, object]:
    tel = assist.telemetry
    return {
        "energy_restored": tel.energy.restored,
        "energy_writes": tel.energy.writes,
        "missile_writes": tel.ammo["missiles"].writes,
        "maximum_single_frame_damage": tel.maximum_single_frame_damage,
        "deaths": tel.deaths,
    }


def _compact_fight(evidence) -> dict[str, object]:
    payload = evidence.to_dict()
    payload.pop("phase_transitions", None)
    payload.pop("vulnerable_spritemaps", None)
    return payload


def _delta_vs_baseline(frames: int) -> dict[str, object]:
    delta = int(frames) - BASELINE_FRAMES
    timed = format_segment_time(abs(delta))
    sign = "-" if delta < 0 else "+"
    return {
        "frames": delta,
        "seconds": round(delta / float(timed["ntsc_fps"]), 3),
        "clock": sign + str(timed["clock"]),
        "ntsc_fps": timed["ntsc_fps"],
        "baseline_frames": BASELINE_FRAMES,
    }


def run_policy(
    env,
    policy: str,
    *,
    max_frames: int,
    loaded: str,
) -> dict[str, object]:
    assist = UnlimitedResourcesAssist(unlimited_energy=True, unlimited_ammo=True)
    session = ProbeSession(env, assist)
    if session.state.room_id != ROOM_PHANTOON:
        timing = format_segment_time(0)
        return {
            "policy": policy,
            "shots_per_window": POLICIES.get(policy),
            "success": False,
            "outcome": "wrong_room",
            "frames": 0,
            "seconds": timing["seconds"],
            "clock": timing["clock"],
            "timing": timing,
            "shots_fired": 0,
            "windows": 0,
            "final_hp": int(session.state.enemy0_hp),
            "boss_bit": False,
            "assist": _assist_block(assist),
            "method": "left_corner_charge_beam",
            "state": loaded,
        }

    entry = _snapshot(session)
    chips = _attach_chip_trace(session)
    strategy = make_strategy(policy, max_frames)
    evidence = play_phantoon_fight(session, strategy=strategy, require_boss_bit=True)
    groups = group_window_chips(chips)
    chip_summary = summarize_window_chips(groups)
    success = evidence.outcome == "phantoon_defeated"
    frames = int(evidence.action_frames)
    timing = format_segment_time(frames)
    reasons = dict(session.action_reasons)
    return {
        "policy": policy,
        "shots_per_window": POLICIES[policy],
        "success": success,
        "outcome": evidence.outcome,
        "frames": frames,
        "seconds": timing["seconds"],
        "clock": timing["clock"],
        "timing": timing,
        "shots_fired": evidence.shots_fired,
        "windows": evidence.windows,
        "shots_per_window_observed": chip_summary["mean"],
        "final_hp": evidence.final_body_hp,
        "boss_bit": evidence.boss_bit_set,
        "boss_bits_wrecked_ship": wrecked_ship_boss_bits(env),
        "phantoon_defeated": phantoon_defeated(env),
        "body_zero_frame": evidence.body_zero_frame,
        "boss_bit_frame": evidence.boss_bit_frame,
        "assist": _assist_block(assist),
        "method": "left_corner_charge_beam",
        "state": loaded,
        "entry": entry,
        "final": _snapshot(session),
        "fight": _compact_fight(evidence),
        "chips": chips,
        "chip_windows": chip_summary,
        "reasons": {
            "phan_farm_snipe": int(reasons.get("phan_farm_snipe", 0)),
            "phan_charge": int(reasons.get("phan_charge", 0)),
            "phan_fire": int(reasons.get("phan_fire", 0)),
            "phantoon_death_anim": int(reasons.get("phantoon_death_anim", 0)),
        },
        "delta_vs_20537f": _delta_vs_baseline(frames),
    }


def cmd_strategy(args: argparse.Namespace) -> int:
    env, loaded = _open_env(_resolve_state(args.state))
    try:
        row = run_policy(env, args.policy, max_frames=args.max_frames, loaded=loaded)
        report = {
            "command": "strategy",
            "wiki": WIKI_URL,
            "assist_enabled": True,
            **row,
        }
        write_json_report(report, args.report)
        return 0 if row["success"] else 1
    finally:
        env.close()


def _pick_winner(rows: dict[str, dict[str, object]]) -> str | None:
    ok = [name for name, row in rows.items() if row.get("success")]
    if not ok:
        return None
    return min(ok, key=lambda name: (int(rows[name]["frames"]), POLICIES[name]))


def cmd_bench(args: argparse.Namespace) -> int:
    state_path = _resolve_state(args.state)
    rows: dict[str, dict[str, object]] = {}
    for policy in POLICIES:
        env, loaded = _open_env(state_path)
        try:
            rows[policy] = run_policy(
                env, policy, max_frames=args.max_frames, loaded=loaded
            )
        finally:
            env.close()

    winner = _pick_winner(rows)
    dual_runs: list[dict[str, object]] = []
    if winner is not None:
        dual_runs.append(rows[winner])
        env, loaded = _open_env(state_path)
        try:
            dual_runs.append(
                run_policy(env, winner, max_frames=args.max_frames, loaded=loaded)
            )
        finally:
            env.close()

    dual_match = (
        len(dual_runs) == 2
        and bool(dual_runs[0].get("success"))
        and bool(dual_runs[1].get("success"))
        and int(dual_runs[0]["frames"]) == int(dual_runs[1]["frames"])
    )
    dual_block: dict[str, object] | None = None
    if dual_runs:
        dual_block = {
            "policy": winner,
            "runs": dual_runs,
            "match": dual_match,
            "frames": [int(run["frames"]) for run in dual_runs],
        }

    def _vs(name: str) -> dict[str, object]:
        other = "spine_three" if name == "probe_default" else "probe_default"
        a = int(rows[name]["frames"])
        b = int(rows[other]["frames"])
        delta = a - b
        timed = format_segment_time(abs(delta))
        sign = "-" if delta < 0 else "+"
        return {
            "vs": other,
            "frames": delta,
            "seconds": round(delta / float(timed["ntsc_fps"]), 3),
            "clock": sign + str(timed["clock"]),
            "ntsc_fps": timed["ntsc_fps"],
        }

    report = {
        "command": "bench",
        "wiki": WIKI_URL,
        "state": str(state_path),
        "assist_enabled": True,
        "baseline_frames": BASELINE_FRAMES,
        "policies": rows,
        "winner": winner,
        "delta_vs_20537f": {
            name: row.get("delta_vs_20537f") for name, row in rows.items()
        },
        "delta_policies": {name: _vs(name) for name in rows},
        "dual": dual_block,
        "method": "left_corner_charge_beam",
        "notes": (
            "Assist ON. Product body unchanged. Rain/right-park still skip "
            "except rain (48,96). Never Super. Reload pin between rows."
        ),
    }
    out = args.report if args.report is not None else DEFAULT_REPORT
    write_json_report(report, out)
    if dual_block is not None:
        dual_path = out.with_name(out.stem + "_dual.json")
        write_json_report(
            {
                "command": "dual",
                "wiki": WIKI_URL,
                "policy": winner,
                "assist_enabled": True,
                "baseline_frames": BASELINE_FRAMES,
                **dual_block,
            },
            dual_path,
        )
    ok = all(bool(row.get("success")) for row in rows.values())
    if dual_runs:
        ok = ok and all(bool(run.get("success")) for run in dual_runs)
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("strategy", help="Run one wiki-charge policy from the pin")
    p.add_argument("--state", default="natural")
    p.add_argument("--policy", choices=tuple(POLICIES), default="probe_default")
    p.add_argument("--max-frames", type=int, default=40_000)
    p.add_argument("--report", type=Path, default=None)
    p.set_defaults(func=cmd_strategy)

    b = sub.add_parser("bench", help="Reload pin between probe_default and spine_three")
    b.add_argument("--state", default="natural")
    b.add_argument("--max-frames", type=int, default=40_000)
    b.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    b.set_defaults(func=cmd_bench)

    args = parser.parse_args()
    if not getattr(args, "command", None):
        parser.print_help()
        return 2
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
