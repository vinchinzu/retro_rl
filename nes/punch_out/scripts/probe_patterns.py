"""Probe Glass Joe pattern / damage windows from Match1.

Logs opp_pattern_set / action / timer transitions, health deltas, and mode.
Used to map post-KD2 counter windows for M3 offense.

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
  uv run python nes/punch_out/scripts/probe_patterns.py --max-frames 15000
```
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from punch_out.paths import GAME, GAME_DIR, RECORDINGS_DIR
from punch_out.policy import GlassJoePolicy
from punch_out.ram import (
    ADDR_FIGHT_FLAG,
    ADDR_HEALTH,
    ADDR_KNOCKDOWN,
    ADDR_OPP_ACTION,
    ADDR_OPP_HEALTH,
    ADDR_OPP_PATTERN_SET,
    ADDR_OPP_PATTERN_TIMER,
    ADDR_ROUND,
    ADDR_CLOCK_MIN,
    ADDR_CLOCK_SEC,
    ADDR_CLOCK_TENTHS,
    hearts,
    is_match_live,
    stars,
)
from retro_harness.env import make_env
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import configure_headless, write_json_report

def _clock_str(ram) -> str:
    m = int(ram[ADDR_CLOCK_MIN])
    s = int(ram[ADDR_CLOCK_SEC])
    t = int(ram[ADDR_CLOCK_TENTHS])
    return f"{m}:{s:01d}{t:01d}"

def run_probe(
    *,
    max_frames: int = 15000,
    out_dir: Path | None = None,
    state_name: str = "Match1",
) -> dict[str, Any]:
    configure_headless()
    out = out_dir or (RECORDINGS_DIR / "probe_patterns")
    out.mkdir(parents=True, exist_ok=True)

    env = make_env(GAME, state_name, GAME_DIR, render_mode="rgb_array")
    try:
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        for _ in range(2000):
            if is_match_live(env.get_ram()):
                break
            step = env.step(nes_idle_action())
            obs = step[0] if isinstance(step, tuple) else step

        policy = GlassJoePolicy()
        rows: list[dict[str, Any]] = []
        transitions: list[dict[str, Any]] = []
        pattern_hits: dict[tuple[int, int], list[int]] = defaultdict(list)
        pattern_mac_dmg: dict[tuple[int, int], list[int]] = defaultdict(list)
        pattern_duration: Counter[tuple[int, int]] = Counter()
        prev_pset = prev_act = prev_timer = -1
        prev_opp = 96
        prev_mac = 96
        window_start = 0
        window_key: tuple[int, int] | None = None
        window_hits = 0
        window_mac_dmg = 0

        for frame in range(1, max_frames + 1):
            ram = env.get_ram()
            fa = policy.tick(ram)
            step = env.step(fa.action)
            obs = step[0] if isinstance(step, tuple) else step
            ram = env.get_ram()

            pset = int(ram[ADDR_OPP_PATTERN_SET])
            act = int(ram[ADDR_OPP_ACTION])
            timer = int(ram[ADDR_OPP_PATTERN_TIMER])
            opp = int(ram[ADDR_OPP_HEALTH])
            mac = int(ram[ADDR_HEALTH])
            kd = int(ram[ADDR_KNOCKDOWN])
            fight = int(ram[ADDR_FIGHT_FLAG])
            rnd = int(ram[ADDR_ROUND])
            key = (pset, act)

            if key != window_key:
                if window_key is not None:
                    pattern_duration[window_key] += frame - window_start
                    if window_hits:
                        pattern_hits[window_key].append(window_hits)
                    if window_mac_dmg:
                        pattern_mac_dmg[window_key].append(window_mac_dmg)
                    transitions.append(
                        {
                            "frame": frame,
                            "clock": _clock_str(ram),
                            "round": rnd,
                            "from": list(window_key),
                            "to": list(key),
                            "opp": opp,
                            "mac": mac,
                            "opp_kd": policy.opp_kd,
                            "mac_kd": policy.mac_kd,
                            "dur": frame - window_start,
                            "hits": window_hits,
                            "mac_dmg": window_mac_dmg,
                            "mode": policy.mode.name,
                        }
                    )
                window_key = key
                window_start = frame
                window_hits = 0
                window_mac_dmg = 0

            if opp < prev_opp:
                dmg = prev_opp - opp
                window_hits += dmg
            if mac < prev_mac and mac > 0:
                window_mac_dmg += prev_mac - mac
            if mac == 0 and prev_mac > 0:
                window_mac_dmg += prev_mac

            # Sample every frame around interesting events, else every 10
            interesting = (
                opp != prev_opp
                or mac != prev_mac
                or pset != prev_pset
                or act != prev_act
                or kd != 0
                or policy.mode.name in ("PUNCH_TAUNT", "GETUP", "WATCH_KD")
            )
            if interesting or frame % 10 == 0:
                rows.append(
                    {
                        "frame": frame,
                        "clock": _clock_str(ram),
                        "round": rnd,
                        "fight": fight,
                        "pset": pset,
                        "act": act,
                        "timer": timer,
                        "opp": opp,
                        "mac": mac,
                        "kd": kd,
                        "opp_kd": policy.opp_kd,
                        "mac_kd": policy.mac_kd,
                        "hearts": hearts(ram),
                        "stars": stars(ram),
                        "mode": policy.mode.name,
                        "reason": fa.reason,
                        "action": fa.reason,
                    }
                )

            prev_opp, prev_mac = opp, mac
            prev_pset, prev_act, prev_timer = pset, act, timer

            if policy.mac_kd >= 3:
                break
            if policy.opp_kd >= 3 and fight != 0xFF:
                break
        else:
            frame = max_frames

        # Flush last window
        if window_key is not None:
            pattern_duration[window_key] += frame - window_start
            if window_hits:
                pattern_hits[window_key].append(window_hits)
            if window_mac_dmg:
                pattern_mac_dmg[window_key].append(window_mac_dmg)

        # Summarize patterns by total frames and damage
        summary_rows = []
        for key, dur in pattern_duration.most_common():
            hits_list = pattern_hits.get(key, [])
            mac_list = pattern_mac_dmg.get(key, [])
            summary_rows.append(
                {
                    "pset": key[0],
                    "act": key[1],
                    "frames": dur,
                    "hit_events": len(hits_list),
                    "total_opp_dmg": sum(hits_list),
                    "mac_hit_events": len(mac_list),
                    "total_mac_dmg": sum(mac_list),
                }
            )

        csv_path = out / "frames.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            if rows:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)

        trans_path = out / "transitions.json"
        write_json_report(trans_path, {"transitions": transitions})

        report = {
            "frames": frame,
            "opp_kd": policy.opp_kd,
            "mac_kd": policy.mac_kd,
            "final_mac": int(env.get_ram()[ADDR_HEALTH]),
            "final_opp": int(env.get_ram()[ADDR_OPP_HEALTH]),
            "final_round": int(env.get_ram()[ADDR_ROUND]),
            "mode": policy.mode.name,
            "pattern_summary": summary_rows[:40],
            "reasons": dict(
                sorted(policy.reasons.items(), key=lambda kv: -kv[1])[:25]
            ),
            "n_transitions": len(transitions),
            "n_frame_samples": len(rows),
            "csv": str(csv_path.name),
            "transitions_file": str(trans_path.name),
        }
        write_json_report(out / "report.json", report)
        print(
            f"PROBE frames={frame} opp_kd={policy.opp_kd} mac_kd={policy.mac_kd} "
            f"mac={report['final_mac']} opp={report['final_opp']} "
            f"patterns={len(summary_rows)}"
        )
        print("Top patterns by duration (pset,act frames opp_dmg mac_dmg):")
        for r in summary_rows[:20]:
            print(
                f"  {r['pset']:3d},{r['act']:3d}  f={r['frames']:5d}  "
                f"opp={r['total_opp_dmg']:3d}  mac={r['total_mac_dmg']:3d}  "
                f"hits={r['hit_events']} mac_hits={r['mac_hit_events']}"
            )
        print("Damage-dealing patterns:")
        for r in summary_rows:
            if r["total_opp_dmg"] > 0:
                print(
                    f"  {r['pset']:3d},{r['act']:3d}  opp_dmg={r['total_opp_dmg']}  "
                    f"events={r['hit_events']} frames={r['frames']}"
                )
        print("Mac-damaging patterns:")
        for r in summary_rows:
            if r["total_mac_dmg"] > 0:
                print(
                    f"  {r['pset']:3d},{r['act']:3d}  mac_dmg={r['total_mac_dmg']}  "
                    f"events={r['mac_hit_events']} frames={r['frames']}"
                )
        return report
    finally:
        env.close()

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-frames", type=int, default=15000)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--state", default="Match1")
    args = p.parse_args()
    run_probe(max_frames=args.max_frames, out_dir=args.out_dir, state_name=args.state)

if __name__ == "__main__":
    main()
