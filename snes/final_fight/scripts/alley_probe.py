"""Instrument room-1 alley fights: enemy XY, attack hits, throw distance."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from final_fight.paths import GAME, GAME_DIR, RECORDINGS_DIR
from final_fight.policy import Stage1Policy
from final_fight.ram import (
    BOSS_BASE,
    ENEMY_BASES,
    OFF_HP,
    OFF_STATUS,
    OFF_X,
    OFF_Y,
    parse_game_state,
    read_u8,
    read_u16le,
)
from retro_harness.env import get_available_states, make_env, reset_obs
from retro_harness.actions import idle_action
from retro_harness.ram_state import GameMode, GameState
from retro_harness.segment_runner import configure_headless, write_json_report


def _raw_slots(ram: np.ndarray) -> list[dict[str, int]]:
    slots: list[dict[str, int]] = []
    for i, base in enumerate(ENEMY_BASES):
        slots.append(
            {
                "slot": i,
                "status": read_u8(ram, base + OFF_STATUS),
                "x": read_u16le(ram, base + OFF_X),
                "y": read_u16le(ram, base + OFF_Y),
                "hp": read_u8(ram, base + OFF_HP),
            }
        )
    slots.append(
        {
            "slot": 3,
            "status": read_u8(ram, BOSS_BASE + OFF_STATUS),
            "x": read_u16le(ram, BOSS_BASE + OFF_X),
            "y": read_u16le(ram, BOSS_BASE + OFF_Y),
            "hp": read_u8(ram, BOSS_BASE + OFF_HP),
        }
    )
    return slots

def _enemy_hp_map(state: GameState) -> dict[int, int]:
    return {e.slot: e.health for e in state.living_enemies}

def run_alley_probe(
    *,
    state_name: str = "Stage1_Clear_w5_cam1536",
    max_frames: int = 4000,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Run Stage1 policy and log hit/miss evidence for alley combat."""
    configure_headless()
    available = get_available_states(GAME, GAME_DIR)
    if state_name not in available:
        raise SystemExit(f"missing state {state_name}; have {available[:8]}")
    out = out_dir or (RECORDINGS_DIR / "alley_probe")
    out.mkdir(parents=True, exist_ok=True)

    env = make_env(GAME, state_name, GAME_DIR, render_mode="rgb_array")
    policy = Stage1Policy()
    hit_events: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    reason_counts: dict[str, int] = {}
    prev_hp = dict[int, int]()
    # Pending strikes: attribute HP drops within a few frames of the press.
    pending: list[dict[str, Any]] = []
    hp_drops: list[dict[str, Any]] = []
    boss_peak = 0
    dual_frames = 0
    right_edge_frames = 0
    hit_window = 8
    try:
        reset_obs(env)
        ram = env.get_ram()
        state = parse_game_state(ram, frame=0)
        prev_hp = _enemy_hp_map(state)

        for frame_i in range(1, max_frames + 1):
            tick = policy.tick(state)
            reason = (
                tick.action.reason
                if tick.action is not None
                else (tick.reason or "none")
            )
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            action = (
                tick.action.action
                if tick.action is not None
                else idle_action()
            )
            enemy = state.nearest_enemy()
            target_snap: dict[str, Any] | None = None
            if enemy is not None:
                target_snap = {
                    "slot": enemy.slot,
                    "x": enemy.x,
                    "y": enemy.y,
                    "hp": enemy.health,
                    "dx": enemy.x - state.player_x,
                    "dy": enemy.y - state.player_y,
                    "screen_x": enemy.x - state.camera_x,
                }
            is_strike = reason in {
                "attack",
                "throw_left",
                "throw_right",
            }
            pre_px = state.player_x
            pre_py = state.player_y
            pre_cam = state.camera_x
            if is_strike and target_snap is not None:
                pending.append(
                    {
                        "frame": frame_i,
                        "reason": reason,
                        "expire": frame_i + hit_window,
                        "player_x": pre_px,
                        "player_y": pre_py,
                        "player_sx": pre_px - pre_cam,
                        "cam": pre_cam,
                        "target": target_snap,
                        "hit": False,
                        "hp_delta": 0,
                    }
                )
            env.step(action)
            ram = env.get_ram()
            state = parse_game_state(ram, frame=frame_i)
            boss_peak = max(
                boss_peak, int(state.extras.get("boss_status", 0))
            )
            living = state.living_enemies
            if len(living) >= 2:
                dual_frames += 1
            player_sx = state.player_x - state.camera_x
            if player_sx >= 150:
                right_edge_frames += 1

            cur_hp = _enemy_hp_map(state)
            # Attribute any slot HP drop to the oldest pending strike.
            for slot, before in prev_hp.items():
                after = cur_hp.get(slot)
                if after is None:
                    delta = before
                else:
                    delta = before - after
                if delta <= 0:
                    continue
                attributed = False
                for ev in pending:
                    if ev["hit"]:
                        continue
                    if frame_i > int(ev["expire"]):
                        continue
                    if int(ev["target"]["slot"]) != slot:
                        continue
                    ev["hit"] = True
                    ev["hp_delta"] = delta
                    attributed = True
                    break
                hp_drops.append(
                    {
                        "frame": frame_i,
                        "slot": slot,
                        "delta": delta,
                        "attributed": attributed,
                        "px": state.player_x,
                        "py": state.player_y,
                        "living": len(living),
                    }
                )

            # Resolve expired pending into hit_events.
            keep: list[dict[str, Any]] = []
            for ev in pending:
                if ev["hit"] or frame_i >= int(ev["expire"]):
                    hit_events.append(ev)
                else:
                    keep.append(ev)
            pending = keep

            # Sparse geometry samples while locked / fighting.
            if living and (
                frame_i % 30 == 0
                or reason
                in {
                    "edge_wait",
                    "edge_weave_up",
                    "edge_weave_down",
                    "edge_recenter",
                    "edge_mid",
                }
            ):
                samples.append(
                    {
                        "frame": frame_i,
                        "reason": reason,
                        "px": state.player_x,
                        "py": state.player_y,
                        "psx": player_sx,
                        "cam": state.camera_x,
                        "hp": state.health,
                        "lives": state.lives,
                        "enemies": [
                            {
                                "slot": e.slot,
                                "x": e.x,
                                "y": e.y,
                                "hp": e.health,
                                "sx": e.x - state.camera_x,
                                "dx": e.x - state.player_x,
                                "dy": e.y - state.player_y,
                            }
                            for e in living
                        ],
                        "boss": int(state.extras.get("boss_status", 0)),
                    }
                )

            prev_hp = cur_hp

            if state.player_dead or state.mode is GameMode.CONTINUE:
                break
            if boss_peak >= 1:
                break
        else:
            frame_i = max_frames

        hit_events.extend(pending)
        hits = [e for e in hit_events if e["hit"]]
        misses = [e for e in hit_events if not e["hit"]]
        throw_hits = [
            e for e in hits if str(e["reason"]).startswith("throw")
        ]
        punch_hits = [
            e for e in hits if str(e["reason"]) == "attack"
        ]
        unattributed = [d for d in hp_drops if not d["attributed"]]
        report: dict[str, Any] = {
            "start_state": state_name,
            "frames": frame_i,
            "end_hp": state.health,
            "end_lives": state.lives,
            "end_cam": state.camera_x,
            "end_room": state.room,
            "boss_peak": boss_peak,
            "dual_frames": dual_frames,
            "right_edge_frames": right_edge_frames,
            "strikes": len(hit_events),
            "hits": len(hits),
            "misses": len(misses),
            "punch_hits": len(punch_hits),
            "throw_hits": len(throw_hits),
            "hit_rate": (
                round(len(hits) / len(hit_events), 3) if hit_events else 0.0
            ),
            "total_hp_delta": sum(int(e["hp_delta"]) for e in hits),
            "unattributed_drops": len(unattributed),
            "unattributed_hp": sum(int(d["delta"]) for d in unattributed),
            "avg_hit_dx": (
                round(
                    sum(abs(int(e["target"]["dx"])) for e in hits)
                    / len(hits),
                    1,
                )
                if hits
                else None
            ),
            "avg_miss_dx": (
                round(
                    sum(abs(int(e["target"]["dx"])) for e in misses)
                    / len(misses),
                    1,
                )
                if misses
                else None
            ),
            "avg_hit_dy": (
                round(
                    sum(abs(int(e["target"]["dy"])) for e in hits)
                    / len(hits),
                    1,
                )
                if hits
                else None
            ),
            "avg_miss_dy": (
                round(
                    sum(abs(int(e["target"]["dy"])) for e in misses)
                    / len(misses),
                    1,
                )
                if misses
                else None
            ),
            "reason_counts": dict(
                sorted(reason_counts.items(), key=lambda kv: -kv[1])
            ),
            "sample_hits": hits[:40],
            "sample_misses": misses[:40],
            "hp_drops": hp_drops[:60],
            "geometry_samples": samples[-80:],
        }
        path = write_json_report(out / "alley_probe.json", report)
        report["report_path"] = str(path)
        return report
    finally:
        env.close()

def main(argv: list[str] | None = None) -> int:
    """CLI for alley combat instrumentation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state", default="Stage1_Clear_w5_cam1536"
    )
    parser.add_argument("--max-frames", type=int, default=4000)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    report = run_alley_probe(
        state_name=args.state,
        max_frames=args.max_frames,
        out_dir=args.out_dir,
    )
    print(
        f"frames={report['frames']} hits={report['hits']}/"
        f"{report['strikes']} rate={report['hit_rate']} "
        f"hp_delta={report['total_hp_delta']} "
        f"punch_hits={report['punch_hits']} "
        f"throw_hits={report['throw_hits']} "
        f"avg_hit_dx={report['avg_hit_dx']} "
        f"avg_miss_dx={report['avg_miss_dx']} "
        f"avg_hit_dy={report['avg_hit_dy']} "
        f"avg_miss_dy={report['avg_miss_dy']} "
        f"dual_frames={report['dual_frames']} "
        f"boss={report['boss_peak']} "
        f"end_hp={report['end_hp']} lives={report['end_lives']}"
    )
    top = list(report["reason_counts"].items())[:10]
    print("reasons: " + ", ".join(f"{k}={v}" for k, v in top))
    print(f"report={report['report_path']}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
