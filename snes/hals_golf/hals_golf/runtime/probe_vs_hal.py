"""Probe Title → VS HAL menu path and early-match behavior."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import stable_retro as retro
from PIL import Image

from hals_golf.core.actions import idle, named_script
from hals_golf.core.ram import (
    WRAM_HOLE_INDEX,
    WRAM_LIE_TYPE,
    WRAM_REST_DISTANCE,
    WRAM_STROKE_COUNT,
    read_hole_number,
    read_u16_le,
    read_u8,
)
from hals_golf.core.scene import is_command_screen
from hals_golf.paths import DEBUG_FRAMES_DIR, GAME, PROJECT_DIR
from hals_golf.runtime.retro_setup import register_golf_integration
from retro_harness.env import make_env, save_state

OUT = DEBUG_FRAMES_DIR / "vs_hal"


def _mean_rgb(obs: np.ndarray) -> tuple[float, float, float]:
    return (
        float(np.mean(obs[:, :, 0])),
        float(np.mean(obs[:, :, 1])),
        float(np.mean(obs[:, :, 2])),
    )


def _dump(obs: np.ndarray, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    Image.fromarray(obs).save(OUT / name)


def _ram_info(ram: np.ndarray, obs: np.ndarray | None = None) -> dict:
    return {
        "stroke": read_u8(ram, WRAM_STROKE_COUNT),
        "hole_idx": read_u8(ram, WRAM_HOLE_INDEX),
        "hole": read_hole_number(ram),
        "rest": read_u16_le(ram, WRAM_REST_DISTANCE),
        "lie": read_u8(ram, WRAM_LIE_TYPE),
        "cmd": bool(is_command_screen(obs)),
    }


def probe_cursor_positions() -> list[dict]:
    """Dump mode-select frames after 0..6 DOWN taps from Title."""
    register_golf_integration(retro, quiet=True)
    rows: list[dict] = []
    for downs in range(0, 7):
        env = make_env(
            game=GAME,
            state="Title",
            game_dir=PROJECT_DIR,
            render_mode="rgb_array",
        )
        try:
            obs, _info = env.reset()
            steps: list[tuple[str, int]] = [("IDLE", 60)]
            for _ in range(downs):
                steps.extend([("DOWN", 2), ("IDLE", 8)])
            steps.append(("IDLE", 20))
            for action in named_script(steps):
                obs, *_ = env.step(action)
            name = f"cursor_down{downs}.png"
            _dump(obs, name)
            row = {
                "downs": downs,
                "frame": name,
                "rgb": list(_mean_rgb(obs)),
                "ram": _ram_info(env.get_ram(), obs),
            }
            rows.append(row)
            print(f"[PROBE] downs={downs} rgb={row['rgb']}")
        finally:
            env.close()
    return rows


def probe_confirm_path(downs: int, max_frames: int = 2500) -> dict:
    """Confirm VS HAL after ``downs`` and walk through menus with B/START."""
    register_golf_integration(retro, quiet=True)
    env = make_env(
        game=GAME,
        state="Title",
        game_dir=PROJECT_DIR,
        render_mode="rgb_array",
    )
    snapshots: list[dict] = []
    try:
        obs, info = env.reset()
        _dump(obs, f"path_d{downs}_0000_title.png")

        # Select mode
        steps: list[tuple[str, int]] = [("IDLE", 60)]
        for _ in range(downs):
            steps.extend([("DOWN", 2), ("IDLE", 8)])
        steps.extend([("B", 3), ("IDLE", 120)])
        # Then mash through likely menus similarly to stroke play, but
        # capture frames so we can trim timings later.
        # After mode: may go Name or Difficulty or Players.
        # Try: START name, START, clubs DOWN/RIGHT, START, skip flyover.
        downs_club = sum([(("DOWN", 2), ("IDLE", 5)) for _ in range(30)], ())
        rights_club = sum([(("RIGHT", 2), ("IDLE", 5)) for _ in range(10)], ())
        steps.extend(
            [
                ("B", 3),
                ("IDLE", 120),
                ("START", 4),
                ("IDLE", 180),
                ("START", 4),
                ("IDLE", 160),
                *downs_club,
                *rights_club,
                ("START", 4),
                ("IDLE", 516),
                ("B", 3),
                ("IDLE", 200),
            ]
        )
        script = named_script(steps)
        command_at: int | None = None
        for frame_i in range(max_frames):
            action = script[frame_i] if frame_i < len(script) else idle()
            obs, _r, terminated, truncated, info = env.step(action)
            ram = env.get_ram()
            info_row = _ram_info(ram, obs)
            if frame_i % 60 == 0 or info_row["cmd"] or info_row["hole"] == 1:
                name = f"path_d{downs}_{frame_i:04d}.png"
                if frame_i % 120 == 0 or info_row["cmd"]:
                    _dump(obs, name)
                    snapshots.append({"frame": frame_i, **info_row, "png": name})
                    print(
                        f"[PROBE] d={downs} f={frame_i} "
                        f"hole={info_row['hole']} stroke={info_row['stroke']} "
                        f"rest={info_row['rest']} cmd={info_row['cmd']} "
                        f"rgb={_mean_rgb(obs)}"
                    )
            if info_row["cmd"] and info_row["hole"] == 1 and command_at is None:
                command_at = frame_i
                save_state(env, PROJECT_DIR, GAME, "VsHal_Hole1_Command")
                _dump(obs, f"path_d{downs}_hole1_command.png")
                print(f"[PROBE] Hole1 command at frame={frame_i}")
                break
            if terminated or truncated:
                break
        report = {
            "downs": downs,
            "script_len": len(script),
            "command_at": command_at,
            "snapshots": snapshots[-40:],
        }
        (OUT / f"path_d{downs}_report.json").write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )
        return report
    finally:
        env.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cursors = probe_cursor_positions()
    (OUT / "cursor_report.json").write_text(
        json.dumps(cursors, indent=2),
        encoding="utf-8",
    )
    # Manual lists: Stroke, Match, Tournament, VS HAL → downs=3
    for downs in (3, 2, 4, 1):
        report = probe_confirm_path(downs)
        if report.get("command_at") is not None:
            print(f"[PROBE] success downs={downs}")
            break
    else:
        print("[PROBE] no Hole1 command found for downs 3/2/4/1")


if __name__ == "__main__":
    main()
