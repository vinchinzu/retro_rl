"""Instrument room-1 waves 3–4: HP/life drops, geometry, food healing."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from final_fight.paths import GAME, GAME_DIR, RECORDINGS_DIR
from final_fight.policy import Stage1Policy
from final_fight.ram import (
    ENEMY_BASES,
    OFF_Y,
    PLAYER_BASE,
    parse_game_state,
    read_u16le,
)
from retro_harness.env import get_available_states, make_env, reset_obs
from retro_harness.actions import idle_action
from retro_harness.ram_state import GameMode
from retro_harness.segment_runner import (
    WaveChainTracker,
    configure_headless,
    save_rgb_png,
    write_json_report,
)

# Entity jump Y sits at base+0x0A (body Y at +0x0D); delta hints airborne.
_OFF_JUMP_Y = 0x0A


def _jump_y_snapshot(
    ram: np.ndarray, enemy_slot: int | None
) -> dict[str, int | None]:
    """Body vs jump Y for player and nearest enemy (airborne proxy)."""
    player_body = read_u16le(ram, PLAYER_BASE + OFF_Y)
    player_jump = read_u16le(ram, PLAYER_BASE + _OFF_JUMP_Y)
    enemy_body: int | None = None
    enemy_jump: int | None = None
    if enemy_slot is not None and 0 <= enemy_slot < len(ENEMY_BASES):
        base = ENEMY_BASES[enemy_slot]
        enemy_body = read_u16le(ram, base + OFF_Y)
        enemy_jump = read_u16le(ram, base + _OFF_JUMP_Y)
    return {
        "player_body_y": player_body,
        "player_jump_y": player_jump,
        "enemy_body_y": enemy_body,
        "enemy_jump_y": enemy_jump,
        "enemy_airborne": (
            None
            if enemy_body is None or enemy_jump is None
            else int(abs(enemy_jump - enemy_body) > 2)
        ),
    }

def _wave_hp_table(
    waves: list[dict[str, Any]],
    hp_events: list[dict[str, Any]],
    life_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Per-wave start/end HP, chips, jump-kick-correlated chips."""
    rows: list[dict[str, Any]] = []
    for w in waves:
        idx = int(w["index"])
        chips = [e for e in hp_events if e["wave"] == idx]
        lives = [e for e in life_events if e["wave"] == idx]
        kickish = [
            e
            for e in chips
            if e.get("dx") is not None
            and 45 <= int(e["dx"]) <= 95
            and e.get("reason")
            in {"edge_recenter", "edge_press", "edge_wait", "edge_mid"}
        ]
        airborne = [
            e for e in chips if e.get("enemy_airborne") == 1
        ]
        rows.append(
            {
                "wave": idx,
                "hp_start": w.get("start_health"),
                "hp_end": w.get("end_health"),
                "lives_start": w.get("start_lives"),
                "lives_end": w.get("end_lives"),
                "player_damage": w.get("player_damage"),
                "damage_dealt": w.get("damage_dealt"),
                "chip_count": len(chips),
                "kick_band_chips": len(kickish),
                "airborne_chips": len(airborne),
                "life_losses": len(lives),
                "chip_reasons": sorted(
                    {str(e["reason"]) for e in chips}
                ),
            }
        )
    return rows

def run_wave4_instrument(
    *,
    state_name: str = "Stage1_Room1_Healthy",
    max_frames: int = 5000,
    focus_waves: tuple[int, ...] = (3, 4),
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Run policy; log per-frame HP events around waves 3–4."""
    configure_headless()
    available = get_available_states(GAME, GAME_DIR)
    if state_name not in available:
        raise SystemExit(f"missing state {state_name}; have {available[:8]}")
    out = out_dir or (RECORDINGS_DIR / "wave4_instrument")
    out.mkdir(parents=True, exist_ok=True)

    env = make_env(GAME, state_name, GAME_DIR, render_mode="rgb_array")
    policy = Stage1Policy()
    tracker = WaveChainTracker(
        max_frames=max_frames,
        clear_hold_frames=30,
        target_waves=max(focus_waves) + 2,
        stop_on_boss=True,
    )
    hp_events: list[dict[str, Any]] = []
    life_events: list[dict[str, Any]] = []
    heal_events: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    focus_reasons: dict[str, int] = {}
    screenshots: list[str] = []
    try:
        reset_obs(env)
        state = parse_game_state(env.get_ram(), frame=0)
        tracker.begin(state)
        prev_hp = state.health
        prev_lives = state.lives
        prev_wave = 0
        png = save_rgb_png(env.render(), out / "w4i_0000_start.png")
        screenshots.append(png.name)

        for frame_i in range(1, max_frames + 1):
            tick = policy.tick(state)
            reason = (
                tick.action.reason
                if tick.action is not None
                else (tick.reason or "none")
            )
            action = (
                tick.action.action
                if tick.action is not None
                else idle_action()
            )
            tracker.note_reason(reason)
            obs, _r, _t, _tr, _i = env.step(action)
            ram = env.get_ram()
            state = parse_game_state(ram, frame=frame_i)
            stop = tracker.update(state)
            wave_idx = (
                tracker.waves[-1].index
                if tracker.waves and not tracker._in_wave
                else (
                    tracker._wave.index
                    if tracker._wave is not None
                    else 0
                )
            )
            # During an active wave, prefer the in-progress index.
            if tracker._wave is not None:
                wave_idx = tracker._wave.index
            in_focus = wave_idx in focus_waves

            if tracker.waves_cleared > prev_wave:
                prev_wave = tracker.waves_cleared
                tag = f"wave{prev_wave}_clear"
                png = save_rgb_png(
                    obs, out / f"w4i_{frame_i:04d}_{tag}.png"
                )
                screenshots.append(png.name)

            if in_focus:
                focus_reasons[reason] = focus_reasons.get(reason, 0) + 1
                enemy = state.nearest_enemy()
                if frame_i % 15 == 0 or reason.startswith(
                    ("attack", "throw", "space", "edge")
                ):
                    samples.append(
                        {
                            "frame": frame_i,
                            "wave": wave_idx,
                            "reason": reason,
                            "hp": state.health,
                            "lives": state.lives,
                            "px": state.player_x,
                            "py": state.player_y,
                            "psx": state.player_x - state.camera_x,
                            "cam": state.camera_x,
                            "ex": enemy.x if enemy else None,
                            "ey": enemy.y if enemy else None,
                            "ehp": enemy.health if enemy else None,
                            "esx": (
                                enemy.x - state.camera_x
                                if enemy
                                else None
                            ),
                            "dx": (
                                abs(enemy.x - state.player_x)
                                if enemy
                                else None
                            ),
                            "dy": (
                                abs(enemy.y - state.player_y)
                                if enemy
                                else None
                            ),
                        }
                    )

            hp = state.health
            lives = state.lives
            # Real chip while alive (ignore corpse underflow bytes).
            if (
                0 < prev_hp <= 128
                and 0 < hp <= 128
                and hp < prev_hp
                and lives == prev_lives
            ):
                enemy = state.nearest_enemy()
                jump = _jump_y_snapshot(
                    ram, enemy.slot if enemy is not None else None
                )
                event = {
                    "frame": frame_i,
                    "wave": wave_idx,
                    "kind": "chip",
                    "hp_before": prev_hp,
                    "hp_after": hp,
                    "delta": prev_hp - hp,
                    "lives": lives,
                    "reason": reason,
                    "psx": state.player_x - state.camera_x,
                    "esx": (
                        enemy.x - state.camera_x if enemy else None
                    ),
                    "dx": (
                        abs(enemy.x - state.player_x)
                        if enemy
                        else None
                    ),
                    "dy": (
                        abs(enemy.y - state.player_y)
                        if enemy
                        else None
                    ),
                    "ehp": enemy.health if enemy else None,
                    **jump,
                }
                hp_events.append(event)
                png = save_rgb_png(
                    obs,
                    out
                    / (
                        f"w4i_{frame_i:04d}_chip"
                        f"_w{wave_idx}_d{prev_hp - hp}.png"
                    ),
                )
                screenshots.append(png.name)
            # Heal without life change → food / item pickup.
            if (
                0 < prev_hp <= 128
                and 0 < hp <= 128
                and hp > prev_hp
                and lives == prev_lives
            ):
                heal_events.append(
                    {
                        "frame": frame_i,
                        "wave": wave_idx,
                        "kind": "heal",
                        "hp_before": prev_hp,
                        "hp_after": hp,
                        "delta": hp - prev_hp,
                        "lives": lives,
                        "reason": reason,
                        "px": state.player_x,
                        "py": state.player_y,
                    }
                )
                png = save_rgb_png(
                    obs,
                    out / f"w4i_{frame_i:04d}_heal_w{wave_idx}.png",
                )
                screenshots.append(png.name)
            if lives < prev_lives:
                enemy = state.nearest_enemy()
                life_events.append(
                    {
                        "frame": frame_i,
                        "wave": wave_idx,
                        "kind": "life_loss",
                        "hp_before": prev_hp,
                        "hp_after": hp,
                        "lives_before": prev_lives,
                        "lives_after": lives,
                        "reason": reason,
                        "psx": state.player_x - state.camera_x,
                        "esx": (
                            enemy.x - state.camera_x if enemy else None
                        ),
                        "dx": (
                            abs(enemy.x - state.player_x)
                            if enemy
                            else None
                        ),
                        "ehp": enemy.health if enemy else None,
                    }
                )
                png = save_rgb_png(
                    obs,
                    out
                    / (
                        f"w4i_{frame_i:04d}_life"
                        f"_w{wave_idx}_L{lives}.png"
                    ),
                )
                screenshots.append(png.name)

            prev_hp = hp
            prev_lives = lives
            if stop is not None:
                break
            if (
                tracker.waves_cleared >= max(focus_waves)
                and not tracker._in_wave
                and state.mode is GameMode.PLAYING
                and state.camera_x > 1600
            ):
                # Enough for unlock evidence after wave 4.
                break

        waves_dicts = [w.to_dict() for w in tracker.waves]
        report: dict[str, Any] = {
            "start_state": state_name,
            "frames": tracker.frames,
            "waves_cleared": tracker.waves_cleared,
            "waves": waves_dicts,
            "wave_hp_table": _wave_hp_table(
                waves_dicts, hp_events, life_events
            ),
            "hp_events": hp_events,
            "life_events": life_events,
            "heal_events": heal_events,
            "focus_reasons": dict(
                sorted(focus_reasons.items(), key=lambda kv: -kv[1])
            ),
            "samples": samples[-400:],
            "screenshots": screenshots,
            "end": {
                "hp": state.health,
                "lives": state.lives,
                "cam": state.camera_x,
                "boss_status": state.extras.get("boss_status"),
            },
        }
        write_json_report(out / "wave4_instrument.json", report)
        return report
    finally:
        env.close()

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", default="Stage1_Room1_Healthy")
    parser.add_argument("--max-frames", type=int, default=5000)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
    )
    return parser

def main(argv: list[str] | None = None) -> int:
    """CLI for wave 3–4 HP/life instrumentation."""
    args = _build_parser().parse_args(argv)
    report = run_wave4_instrument(
        state_name=args.state,
        max_frames=args.max_frames,
        out_dir=args.out_dir,
    )
    print(
        f"waves={report['waves_cleared']} frames={report['frames']} "
        f"chips={len(report['hp_events'])} "
        f"lives_lost={len(report['life_events'])} "
        f"heals={len(report['heal_events'])} "
        f"end_hp={report['end']['hp']} "
        f"end_lives={report['end']['lives']} "
        f"cam={report['end']['cam']}"
    )
    for row in report.get("wave_hp_table", []):
        print(
            f"  WAVE {row['wave']}: "
            f"hp {row['hp_start']}->{row['hp_end']} "
            f"lives {row['lives_start']}->{row['lives_end']} "
            f"chips={row['chip_count']} "
            f"kick_band={row['kick_band_chips']} "
            f"airborne={row['airborne_chips']} "
            f"life_loss={row['life_losses']} "
            f"reasons={row['chip_reasons']}"
        )
    for ev in report["life_events"]:
        print(
            f"  LIFE frame={ev['frame']} wave={ev['wave']} "
            f"hp={ev['hp_before']}->{ev['hp_after']} "
            f"lives={ev['lives_before']}->{ev['lives_after']} "
            f"dx={ev.get('dx')} ehp={ev.get('ehp')} "
            f"reason={ev['reason']}"
        )
    for ev in report["hp_events"][:20]:
        print(
            f"  CHIP frame={ev['frame']} wave={ev['wave']} "
            f"d={ev['delta']} hp={ev['hp_before']}->{ev['hp_after']} "
            f"dx={ev.get('dx')} dy={ev.get('dy')} "
            f"psx={ev.get('psx')} esx={ev.get('esx')} "
            f"air={ev.get('enemy_airborne')} "
            f"ej={ev.get('enemy_jump_y')} eb={ev.get('enemy_body_y')} "
            f"reason={ev['reason']}"
        )
    if report["heal_events"]:
        for ev in report["heal_events"]:
            print(
                f"  HEAL frame={ev['frame']} wave={ev['wave']} "
                f"+{ev['delta']} hp={ev['hp_before']}->{ev['hp_after']}"
            )
    else:
        print("  HEAL none (no food pickup observed)")
    print(f"report={args.out_dir or RECORDINGS_DIR / 'wave4_instrument'}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
