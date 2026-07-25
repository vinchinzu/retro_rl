"""Probe Stage2_Clear → West Side via CLEAR_AREA poke.

Sodom UF leaves ``0x0CD2=0`` (same as Damnd). Same softlock bridge:
``set_value(game_status, CLEAR_AREA)`` should run clear-area →
clear-round → open West Side (``round=02``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from final_fight.paths import GAME, GAME_DIR, RECORDINGS_DIR
from final_fight.ram import (
    ADDR_AREA,
    ADDR_BOSS_DEAD_FLAG,
    ADDR_GAME_STATUS,
    ADDR_ROUND,
    ADDR_ROUNDS_CLEARED,
    BOSS_BASE,
    ENEMY_BASES,
    GameStatus,
    OFF_HP,
    OFF_STATUS,
    OFF_X,
    RoundId,
    parse_game_state,
    read_u8,
    read_u16le,
)
from retro_harness.env import get_available_states, make_env, save_state
from snes_oneshot.actions import buttons, idle_action
from snes_oneshot.game_state import GameMode
from snes_oneshot.segment_runner import (
    configure_headless,
    save_rgb_png,
    snapshot_state,
    write_json_report,
)

WEST_SIDE = int(RoundId.WEST_SIDE)
ENGAGE_DX = 110


def _reset(env: Any) -> Any:
    result = env.reset()
    if isinstance(result, tuple):
        return result[0]
    return result


def _snap(ram: Any) -> dict[str, int]:
    return {
        "game_status": read_u8(ram, ADDR_GAME_STATUS),
        "round": read_u8(ram, ADDR_ROUND),
        "area": read_u8(ram, ADDR_AREA),
        "rounds_cleared": read_u8(ram, ADDR_ROUNDS_CLEARED),
        "boss_status": read_u8(ram, BOSS_BASE + OFF_STATUS),
        "boss_hp": read_u8(ram, BOSS_BASE + OFF_HP),
        "boss_dead_flag": read_u8(ram, ADDR_BOSS_DEAD_FLAG),
    }


def _living(ram: Any, cam: int, px: int) -> list[dict[str, int]]:
    out: list[dict[str, int]] = []
    for i, base in enumerate(ENEMY_BASES):
        status = read_u8(ram, base + OFF_STATUS)
        hp = read_u8(ram, base + OFF_HP)
        x = read_u16le(ram, base + OFF_X)
        if status == 3 and 0 < hp <= 192:
            out.append(
                {
                    "slot": i,
                    "hp": hp,
                    "x": x,
                    "sx": x - cam,
                    "dx": x - px,
                }
            )
    return out


def run_probe(
    *,
    state_name: str = "Stage2_Clear",
    max_frames: int = 3600,
    bridge: bool = True,
    save_stage3: bool = True,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Poke CLEAR_AREA from Sodom kill-frame; watch for West Side."""
    configure_headless()
    available = get_available_states(GAME, GAME_DIR)
    if state_name not in available:
        raise SystemExit(f"missing state {state_name}; have {available[:8]}")
    out = out_dir or (RECORDINGS_DIR / "stage3_bridge_probe")
    out.mkdir(parents=True, exist_ok=True)

    env = make_env(GAME, state_name, GAME_DIR, render_mode="rgb_array")
    obs = _reset(env)
    ram = env.get_ram()
    state = parse_game_state(ram, frame=0)
    start = {**snapshot_state(state), **_snap(ram)}
    transitions: list[dict[str, Any]] = [{"frame": 0, **start}]
    screenshots: list[str] = []
    saved: list[str] = []
    png = save_rgb_png(obs, out / "s3_0000_start.png")
    screenshots.append(png.name)
    print(
        f"start status=0x{start['game_status']:02X} "
        f"round={start['round']} area={start['area']} "
        f"cd2={start['boss_dead_flag']} "
        f"boss_st={start['boss_status']} boss_hp={start['boss_hp']} "
        f"hp={state.health} L={state.lives} cam={state.camera_x}"
    )

    if bridge:
        env.set_value("game_status", int(GameStatus.CLEAR_AREA))
        ram = env.get_ram()
        print(
            "bridge CLEAR_AREA "
            f"status now 0x{read_u8(ram, ADDR_GAME_STATUS):02X}"
        )

    west_frame: int | None = None
    clear_round_frame: int | None = None
    fight_ready_frame: int | None = None
    stage3_snap: dict[str, Any] | None = None
    last_key: tuple[Any, ...] | None = None
    outcome = "timeout"

    for frame_i in range(1, max_frames + 1):
        ram = env.get_ram()
        state = parse_game_state(ram, frame=frame_i)
        snap = {**snapshot_state(state), **_snap(ram)}
        enemies = _living(ram, state.camera_x, state.player_x)
        key = (
            snap["game_status"],
            snap["round"],
            snap["area"],
            snap["rounds_cleared"],
            len(enemies),
            state.camera_x // 64,
        )
        if key != last_key:
            last_key = key
            transitions.append(
                {"frame": frame_i, **snap, "enemies": enemies}
            )
            print(
                f"frame={frame_i} status=0x{snap['game_status']:02X} "
                f"round={snap['round']} area={snap['area']} "
                f"rc={snap['rounds_cleared']} "
                f"hp={state.health} L={state.lives} "
                f"cam={state.camera_x} enemies={len(enemies)} "
                f"cd2={snap['boss_dead_flag']}"
            )
            tag = (
                f"st{snap['game_status']:02X}"
                f"_r{snap['round']}_a{snap['area']}"
            )
            png = save_rgb_png(
                obs, out / f"s3_{frame_i:04d}_{tag}.png"
            )
            screenshots.append(png.name)

        if (
            clear_round_frame is None
            and snap["game_status"] == GameStatus.CLEAR_ROUND
        ):
            clear_round_frame = frame_i

        if west_frame is None and snap["round"] == WEST_SIDE:
            west_frame = frame_i
            png = save_rgb_png(
                obs, out / f"s3_{frame_i:04d}_west_side.png"
            )
            screenshots.append(png.name)

        status = snap["game_status"]
        engage = (
            snap["round"] == WEST_SIDE
            and state.mode is GameMode.PLAYING
            and 0 < state.health <= 128
            and state.lives > 0
            and enemies
            and abs(min(enemies, key=lambda e: abs(e["dx"]))["dx"])
            <= ENGAGE_DX
        )
        if engage and fight_ready_frame is None:
            fight_ready_frame = frame_i
            stage3_snap = {**snap, "enemies": enemies}
            outcome = "stage3_ready"
            if save_stage3:
                path = save_state(env, GAME_DIR, GAME, "Stage3")
                saved.append(path.name)
                print(
                    f"Stage3 saved frame={frame_i} "
                    f"hp={state.health} L={state.lives} "
                    f"cam={state.camera_x} "
                    f"dx={enemies[0]['dx']}"
                )
            png = save_rgb_png(
                obs, out / f"s3_{frame_i:04d}_fight_ready.png"
            )
            screenshots.append(png.name)
            break

        if status in (
            GameStatus.CLEAR_AREA,
            GameStatus.CLEAR_ROUND,
            GameStatus.OPEN_STAGE_A,
            GameStatus.OPEN_STAGE_B,
            GameStatus.CHARACTER_SELECT,
        ):
            action = buttons("START")
        elif (
            snap["round"] == WEST_SIDE
            and status == GameStatus.ACTIVE_GAMEPLAY
        ):
            if enemies:
                nearest = min(enemies, key=lambda e: abs(e["dx"]))
                sx = state.player_x - state.camera_x
                if nearest["dx"] > ENGAGE_DX and sx > 100:
                    action = buttons("LEFT")
                elif nearest["dx"] > 80:
                    action = idle_action()
                else:
                    action = buttons("RIGHT")
            else:
                action = buttons("RIGHT")
        else:
            action = idle_action()
        obs, _r, _t, _tr, _info = env.step(action)

        if state.player_dead or state.health > 128:
            outcome = "death"
            break

    env.close()
    report: dict[str, Any] = {
        "success": outcome == "stage3_ready",
        "outcome": outcome,
        "bridge_clear_area": bridge,
        "start_state": state_name,
        "start": start,
        "clear_round_frame": clear_round_frame,
        "west_side_frame": west_frame,
        "fight_ready_frame": fight_ready_frame,
        "stage3": stage3_snap,
        "transitions": transitions,
        "screenshots": screenshots,
        "saved_states": saved,
        "notes": (
            "Sodom UF leaves 0x0CD2=0; CLEAR_AREA poke bridges "
            "Stage2_Clear → West Side (round=02) like Damnd→subway."
        ),
    }
    write_json_report(out / "stage3_bridge_probe.json", report)
    print(f"outcome={outcome} west_frame={west_frame} saved={saved}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", default="Stage2_Clear")
    parser.add_argument("--max-frames", type=int, default=3600)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--no-bridge", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    run_probe(
        state_name=args.state,
        max_frames=args.max_frames,
        bridge=not args.no_bridge,
        save_stage3=not args.no_save,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
