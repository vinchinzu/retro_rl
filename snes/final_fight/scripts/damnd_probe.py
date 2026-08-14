"""Instrument / fight Damnd door from Boss.state.

Boss.state lights ``0x11E0=01`` with Damnd undrawn (HP 0 at ``0x11F4``).
Door thugs spawn in regular enemy slots first (observed peaks ~82, 60, 95)
before status ``03``. Idle/bait inside kick dx≈40–95 chips hard; prefer
LEFT-flank patient punches at dx≈28–35.

Evidence: HP deltas in ``damnd_probe.json`` + PNGs under ``--out-dir``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from final_fight.paths import GAME, GAME_DIR, RECORDINGS_DIR
from final_fight.policy import Stage1Policy
from final_fight.ram import (
    ADDR_GAME_STATUS,
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
from retro_harness.env import get_available_states, make_env, reset_obs, save_state
from retro_harness.actions import idle_action
from retro_harness.ram_state import GameMode
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)


def _boss_raw(ram: np.ndarray) -> dict[str, int]:
    return {
        "status": read_u8(ram, BOSS_BASE + OFF_STATUS),
        "x": read_u16le(ram, BOSS_BASE + OFF_X),
        "y": read_u16le(ram, BOSS_BASE + OFF_Y),
        "hp": read_u8(ram, BOSS_BASE + OFF_HP),
    }

def _enemy_slots(ram: np.ndarray) -> list[dict[str, int]]:
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
    boss = _boss_raw(ram)
    slots.append({"slot": 3, **boss})
    return slots

def run_damnd_probe(
    *,
    state_name: str = "Boss",
    max_frames: int = 12000,
    out_dir: Path | None = None,
    save_clear_state: bool = True,
) -> dict[str, Any]:
    """Load Boss*.state and fight with Stage1Policy; log boss/thug HP."""
    configure_headless()
    available = get_available_states(GAME, GAME_DIR)
    if state_name not in available:
        raise SystemExit(f"missing state {state_name}; have {available[:8]}")
    out = out_dir or (RECORDINGS_DIR / "damnd_probe")
    out.mkdir(parents=True, exist_ok=True)

    env = make_env(GAME, state_name, GAME_DIR, render_mode="rgb_array")
    policy = Stage1Policy()
    samples: list[dict[str, Any]] = []
    hp_deltas: list[dict[str, Any]] = []
    enemy_hp_deltas: list[dict[str, Any]] = []
    status_changes: list[dict[str, Any]] = []
    reason_counts: dict[str, int] = {}
    screenshots: list[str] = []
    saved_states: list[str] = []
    peak_enemy_hp: dict[int, int] = {}
    thug_kills = 0

    prev_boss_hp = -1
    prev_boss_status = -1
    prev_enemy_hp: dict[int, int] = {}
    drawn_frame: int | None = None
    peak_boss_hp = 0
    min_boss_hp_after_draw: int | None = None
    player_dmg = 0
    start_player_hp = 0
    outcome = "timeout"
    frame_i = 0

    try:
        obs, _ = reset_obs(env)
        ram = env.get_ram()
        state = parse_game_state(ram, frame=0)
        start_player_hp = state.health
        boss0 = _boss_raw(ram)
        prev_boss_hp = boss0["hp"]
        prev_boss_status = boss0["status"]
        peak_boss_hp = boss0["hp"]
        png = save_rgb_png(obs, out / "damnd_0000_start.png")
        screenshots.append(png.name)
        samples.append(
            {
                "frame": 0,
                "tag": "start",
                "player_hp": state.health,
                "lives": state.lives,
                "cam": state.camera_x,
                "px": state.player_x,
                "py": state.player_y,
                "boss": boss0,
                "game_status": read_u8(ram, ADDR_GAME_STATUS),
                "slots": _enemy_slots(ram),
            }
        )

        for frame_i in range(1, max_frames + 1):
            tick = policy.tick(state)
            if tick.action is not None:
                action = tick.action.action
                reason = tick.action.reason
            else:
                action = idle_action()
                reason = tick.reason or "no_action"
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            obs, _r, _t, _tr, _i = env.step(action)
            ram = env.get_ram()
            state = parse_game_state(ram, frame=frame_i)
            boss = _boss_raw(ram)

            living_hp = {
                e.slot: e.health for e in state.living_enemies
            }
            for slot, hp in living_hp.items():
                peak_enemy_hp[slot] = max(
                    peak_enemy_hp.get(slot, 0), hp
                )
                if slot in prev_enemy_hp and hp < prev_enemy_hp[slot]:
                    enemy = next(
                        e
                        for e in state.living_enemies
                        if e.slot == slot
                    )
                    enemy_hp_deltas.append(
                        {
                            "frame": frame_i,
                            "slot": slot,
                            "from": prev_enemy_hp[slot],
                            "to": hp,
                            "delta": hp - prev_enemy_hp[slot],
                            "dx": abs(state.player_x - enemy.x),
                            "dy": abs(state.player_y - enemy.y),
                            "player_hp": state.health,
                            "reason": reason,
                            "peak": peak_enemy_hp.get(slot),
                        }
                    )
            for slot, hp in list(prev_enemy_hp.items()):
                if (
                    slot not in living_hp
                    and hp > 0
                    and 0 < state.health <= 128
                ):
                    thug_kills += 1
                    enemy_hp_deltas.append(
                        {
                            "frame": frame_i,
                            "slot": slot,
                            "from": hp,
                            "to": 0,
                            "delta": -hp,
                            "player_hp": state.health,
                            "reason": reason,
                            "peak": peak_enemy_hp.get(slot),
                            "killed": True,
                        }
                    )
                    png = save_rgb_png(
                        obs,
                        out / f"damnd_{frame_i:04d}_kill{thug_kills}.png",
                    )
                    screenshots.append(png.name)
                    path = save_state(
                        env,
                        GAME_DIR,
                        GAME,
                        f"Boss_PostThug{thug_kills}",
                    )
                    saved_states.append(path.name)
            prev_enemy_hp = dict(living_hp)

            if boss["status"] != prev_boss_status:
                status_changes.append(
                    {
                        "frame": frame_i,
                        "from": prev_boss_status,
                        "to": boss["status"],
                        "hp": boss["hp"],
                        "px": state.player_x,
                        "bx": boss["x"],
                        "cam": state.camera_x,
                        "reason": reason,
                    }
                )
                prev_boss_status = boss["status"]
                if boss["status"] == 0x03 and drawn_frame is None:
                    drawn_frame = frame_i
                    png = save_rgb_png(
                        obs, out / f"damnd_{frame_i:04d}_drawn.png"
                    )
                    screenshots.append(png.name)
                    path = save_state(env, GAME_DIR, GAME, "Boss_Drawn")
                    saved_states.append(path.name)
                    samples.append(
                        {
                            "frame": frame_i,
                            "tag": "drawn",
                            "player_hp": state.health,
                            "lives": state.lives,
                            "cam": state.camera_x,
                            "px": state.player_x,
                            "boss": boss,
                            "game_status": read_u8(
                                ram, ADDR_GAME_STATUS
                            ),
                        }
                    )

            if boss["hp"] > peak_boss_hp:
                peak_boss_hp = boss["hp"]
            if drawn_frame is not None and boss["hp"] > 0:
                if min_boss_hp_after_draw is None:
                    min_boss_hp_after_draw = boss["hp"]
                else:
                    min_boss_hp_after_draw = min(
                        min_boss_hp_after_draw, boss["hp"]
                    )

            if boss["hp"] != prev_boss_hp:
                delta = boss["hp"] - prev_boss_hp
                hp_deltas.append(
                    {
                        "frame": frame_i,
                        "from": prev_boss_hp,
                        "to": boss["hp"],
                        "delta": delta,
                        "status": boss["status"],
                        "px": state.player_x,
                        "bx": boss["x"],
                        "dx": abs(state.player_x - boss["x"]),
                        "reason": reason,
                        "player_hp": state.health,
                    }
                )
                prev_boss_hp = boss["hp"]
                if delta < 0:
                    png = save_rgb_png(
                        obs,
                        out
                        / f"damnd_{frame_i:04d}_bosshit_{boss['hp']}.png",
                    )
                    screenshots.append(png.name)

            if (
                0 < start_player_hp <= 128
                and 0 < state.health <= 128
                and state.health < start_player_hp
            ):
                player_dmg = start_player_hp - state.health

            if state.level_complete:
                outcome = "stage_clear"
                png = save_rgb_png(
                    obs, out / f"damnd_{frame_i:04d}_clear.png"
                )
                screenshots.append(png.name)
                if save_clear_state:
                    path = save_state(
                        env, GAME_DIR, GAME, "Stage1_Clear"
                    )
                    saved_states.append(path.name)
                break

            if state.player_dead or (
                state.mode is GameMode.PLAYING
                and state.health <= 0
                and state.lives <= 0
            ):
                outcome = "death"
                png = save_rgb_png(
                    obs, out / f"damnd_{frame_i:04d}_death.png"
                )
                screenshots.append(png.name)
                break

            if (
                drawn_frame is not None
                and boss["status"] == 0
                and frame_i > drawn_frame + 30
            ):
                outcome = "boss_dead"
                png = save_rgb_png(
                    obs, out / f"damnd_{frame_i:04d}_boss_dead.png"
                )
                screenshots.append(png.name)
                if save_clear_state:
                    path = save_state(
                        env, GAME_DIR, GAME, "Stage1_Clear"
                    )
                    saved_states.append(path.name)
                break

            if frame_i % 400 == 0:
                samples.append(
                    {
                        "frame": frame_i,
                        "tag": "tick",
                        "player_hp": state.health,
                        "lives": state.lives,
                        "cam": state.camera_x,
                        "px": state.player_x,
                        "boss": boss,
                        "living": len(state.living_enemies),
                        "game_status": read_u8(ram, ADDR_GAME_STATUS),
                    }
                )
        else:
            outcome = "timeout"
            png = save_rgb_png(
                obs, out / f"damnd_{max_frames:04d}_timeout.png"
            )
            screenshots.append(png.name)

        end_boss = _boss_raw(ram)
        report: dict[str, Any] = {
            "outcome": outcome,
            "success": outcome in ("stage_clear", "boss_dead"),
            "frames": frame_i,
            "start_state": state_name,
            "start_player_hp": start_player_hp,
            "end_player_hp": state.health,
            "player_damage": player_dmg,
            "lives": state.lives,
            "drawn_frame": drawn_frame,
            "thug_kills": thug_kills,
            "peak_enemy_hp": peak_enemy_hp,
            "peak_boss_hp": peak_boss_hp,
            "min_boss_hp_after_draw": min_boss_hp_after_draw,
            "end_boss": end_boss,
            "boss_hp_dealt": (
                (peak_boss_hp - min_boss_hp_after_draw)
                if (
                    peak_boss_hp > 0
                    and min_boss_hp_after_draw is not None
                )
                else 0
            ),
            "hp_deltas": hp_deltas,
            "enemy_hp_deltas": enemy_hp_deltas,
            "status_changes": status_changes,
            "samples": samples,
            "reason_counts": dict(sorted(reason_counts.items())),
            "screenshots": screenshots,
            "saved_states": saved_states,
            "level_complete": state.level_complete,
            "camera_x": state.camera_x,
            "player_x": state.player_x,
            "room": state.room,
            "game_status": read_u8(ram, ADDR_GAME_STATUS),
            "end_slots": _enemy_slots(ram),
        }
        path = write_json_report(out / "damnd_probe.json", report)
        report["report_path"] = str(path)
        return report
    finally:
        env.close()

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", default="Boss")
    parser.add_argument("--max-frames", type=int, default=12000)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--no-save-clear",
        action="store_true",
        help="Do not write Stage1_Clear.state on success",
    )
    return parser

def main(argv: list[str] | None = None) -> int:
    """CLI for Damnd door/fight instrumentation."""
    args = _build_parser().parse_args(argv)
    report = run_damnd_probe(
        state_name=args.state,
        max_frames=args.max_frames,
        out_dir=args.out_dir,
        save_clear_state=not args.no_save_clear,
    )
    end = report.get("end_boss", {})
    print(
        f"outcome={report['outcome']} frames={report['frames']} "
        f"drawn={report['drawn_frame']} "
        f"thug_kills={report['thug_kills']} "
        f"peaks={report['peak_enemy_hp']} "
        f"boss_hp={report['peak_boss_hp']}->{end.get('hp')} "
        f"(dealt={report['boss_hp_dealt']}) "
        f"status={end.get('status')} "
        f"player={report['start_player_hp']}->"
        f"{report['end_player_hp']} lives={report['lives']}"
    )
    for d in report["enemy_hp_deltas"][:24]:
        tag = "KILL" if d.get("killed") else "hit"
        print(
            f"  {tag} e{d['slot']} {d['from']}->{d['to']} "
            f"peak={d.get('peak')} php={d['player_hp']} @{d['frame']}"
        )
    for d in report["hp_deltas"][:12]:
        print(
            f"  boss_hp@{d['frame']}: {d['from']}->{d['to']} "
            f"(d={d['delta']}) st={d['status']}"
        )
    print(f"report={report.get('report_path')}")
    return 0 if report.get("success") else 1

if __name__ == "__main__":
    raise SystemExit(main())
