"""Cold-boot probing and initial save-state creation."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import stable_retro as retro
from PIL import Image

from hals_golf.core.actions import idle, press_named
from hals_golf.core.ram import WRAM_STROKE_COUNT, read_u8
from hals_golf.core.scene import mean_rgb
from hals_golf.paths import DEBUG_FRAMES_DIR, GAME, GAME_DIR, PROJECT_DIR
from hals_golf.runtime.retro_setup import register_golf_integration
from hals_golf.tasks.menus import cold_boot_from_none_frames
from retro_harness.env import make_env, save_state


def _save_frame(obs: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(obs).save(path)


def _ram_fingerprint(ram: np.ndarray, sample: int = 64) -> list[int]:
    idx = np.linspace(0, len(ram) - 1, sample, dtype=int)
    return [int(ram[i]) for i in idx]


def run_cold_boot_probe(*, frames: int = 1400, save_prefix: str = "probe") -> dict:
    """Boot from NONE, run menu script, save Title/Hole1 states + debug frames."""
    register_golf_integration(retro, quiet=True)
    DEBUG_FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    env = make_env(
        game=GAME,
        state="NONE",
        game_dir=PROJECT_DIR,
        render_mode="rgb_array",
    )
    try:
        obs, info = env.reset()
        save_state(env, PROJECT_DIR, GAME, "Boot")
        _save_frame(obs, DEBUG_FRAMES_DIR / f"{save_prefix}_000_boot.png")

        script = cold_boot_from_none_frames()
        stroke_hist: Counter[int] = Counter()
        hole_candidates: Counter[tuple[int, int]] = Counter()
        snapshots: list[dict] = []
        title_saved = False
        difficulty_saved = False

        for frame_i in range(frames):
            if frame_i < len(script):
                action = script[frame_i]
            else:
                action = press_named("A") if frame_i % 120 < 3 else idle()

            obs, _reward, terminated, truncated, info = env.step(action)
            ram = env.get_ram()
            stroke = read_u8(ram, WRAM_STROKE_COUNT)
            stroke_hist[stroke] += 1
            r, g, b = mean_rgb(obs)

            # Title / mode select: green field background.
            if not title_saved and g > 40 and g > r + 8 and frame_i > 300:
                save_state(env, PROJECT_DIR, GAME, "Title")
                _save_frame(obs, DEBUG_FRAMES_DIR / f"{save_prefix}_title.png")
                title_saved = True

            # Difficulty select often follows first confirm; capture mid-script.
            if (
                not difficulty_saved
                and title_saved
                and 480 <= frame_i <= 700
                and g > 40
            ):
                save_state(env, PROJECT_DIR, GAME, "Difficulty")
                _save_frame(obs, DEBUG_FRAMES_DIR / f"{save_prefix}_difficulty.png")
                difficulty_saved = True

            for off in range(0x1000, 0x1200):
                val = read_u8(ram, off)
                if 1 <= val <= 18:
                    hole_candidates[(off, val)] += 1

            checkpoint_frames = {
                120,
                240,
                420,
                520,
                650,
                800,
                1000,
                1200,
                frames - 1,
            }
            if frame_i in checkpoint_frames:
                name = f"{save_prefix}_{frame_i:04d}"
                save_state(env, PROJECT_DIR, GAME, name)
                _save_frame(obs, DEBUG_FRAMES_DIR / f"{name}.png")
                snapshots.append(
                    {
                        "frame": frame_i,
                        "stroke": stroke,
                        "mean_rgb": [r, g, b],
                        "info": {
                            k: int(v) if hasattr(v, "item") else v
                            for k, v in info.items()
                            if not isinstance(v, (bytes, bytearray, np.ndarray))
                        },
                        "fingerprint": _ram_fingerprint(ram),
                    }
                )

            # After the authored script, treat as in-round candidate.
            if frame_i == min(len(script) + 60, frames - 1):
                save_state(env, PROJECT_DIR, GAME, "Hole1_Command")
                save_state(env, PROJECT_DIR, GAME, "latest")
                _save_frame(obs, DEBUG_FRAMES_DIR / f"{save_prefix}_hole1.png")

            if terminated or truncated:
                obs, info = env.reset()

        ranked_holes = [
            {"offset": off, "value": val, "hits": hits}
            for (off, val), hits in hole_candidates.most_common(20)
        ]
        report = {
            "frames": frames,
            "script_len": len(script),
            "title_saved": title_saved,
            "difficulty_saved": difficulty_saved,
            "stroke_hist": dict(stroke_hist),
            "hole_candidates": ranked_holes,
            "snapshots": snapshots,
            "states_dir": str(GAME_DIR),
        }
        report_path = DEBUG_FRAMES_DIR / f"{save_prefix}_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[PROBE] Wrote {report_path}")
        print(f"[PROBE] Title/Hole1/latest states under {GAME_DIR}")
        print(f"[PROBE] title_saved={title_saved} difficulty_saved={difficulty_saved}")
        print(f"[PROBE] Top hole candidates: {ranked_holes[:5]}")
        return report
    finally:
        env.close()
