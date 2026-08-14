"""Record one continuous Great Waldo Search run from power-on to ending.

One emulator session, no mid-run save-state loads. Boots title → NORMAL,
clears Scenes 1–5 with the documented recipes, and holds the five-scrolls
ending screen. Soft Scene3–5 layouts abort the attempt so the caller can
retry (layout RNG).

Example:

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy uv run python \\
  -m great_waldo_search.scripts.record_full_run --dry-run

SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy uv run python \\
  -m great_waldo_search.scripts.record_full_run
```
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from great_waldo_search.paths import GAME, GAME_DIR, RECORDINGS_DIR
from great_waldo_search.scene_advance import (
    advance_scene,
    is_favorable_scroll_layout,
    probe_assist_landing,
)
from great_waldo_search.scene_recipe import SCENE_RECIPES, SceneRecipe, run_scene_recipe
from great_waldo_search.scripts.boot_probe import (
    build_boot_script,
    looks_like_search_scene,
)
from great_waldo_search.targets import (
    CONFIRMED_FIND_POINTS,
    CURSOR_X_ADDR,
    CURSOR_Y_ADDR,
    SCENE5_CLEAR_SCORE,
    SCORE_HI_ADDR,
    SCORE_LO_ADDR,
    WALDO_POINTS,
    score_u16,
)
from retro_harness.env import make_env, reset_obs
from retro_harness.actions import buttons_multi, idle_action_multi
from retro_harness.video import CaptureSession, FooterLabels, FrameVideoWriter
from retro_harness.video import FOOTER_HEIGHT, short_clock
from retro_harness.segment_runner import configure_headless
from retro_harness.showcase import title_card_with_footer

class SoftLayoutError(RuntimeError):
    """Scene3–5 assist landed on the soft/clock layout."""

class BootError(RuntimeError):
    """Title boot never reached a search HUD."""

def _metrics(env: object) -> dict[str, int]:
    ram = np.asarray(env.get_ram(), dtype=np.uint8)  # type: ignore[attr-defined]
    return {
        "score": score_u16(ram[SCORE_LO_ADDR], ram[SCORE_HI_ADDR]),
        "x": int(ram[CURSOR_X_ADDR]),
        "y": int(ram[CURSOR_Y_ADDR]),
    }

def _to_multi(action: list[int]) -> list[int]:
    """Pad a 12-button P1 action to a 24-button two-player vector."""
    if len(action) == 24:
        return list(action)
    if len(action) != 12:
        raise ValueError(f"unexpected action length: {len(action)}")
    return list(action) + [0] * 12

def _boot_script_multi() -> list[list[int]]:
    return [_to_multi(frame) for frame in build_boot_script()]

def _adapt_recipe(recipe: SceneRecipe, score_before: int) -> SceneRecipe:
    """Retarget absolute score gates to the live carry score."""
    # Scroll is usually +1000 (Scene5 can be +3000); Waldo +1500.
    scroll_floor = score_before + CONFIRMED_FIND_POINTS - 100
    clear_floor = score_before + CONFIRMED_FIND_POINTS + WALDO_POINTS - 200
    return replace(
        recipe,
        scroll_score=scroll_floor,
        clear_score=clear_floor,
    )

def _footer_for_scene(
    env: object,
    action: list[int],
    frame: int,
    fps: float,
    *,
    scene: int,
    name: str,
) -> FooterLabels:
    del action
    metrics = _metrics(env)
    return FooterLabels(
        upper_left=f"SCENE {scene:02d}/05 {name.upper()}",
        upper_right=short_clock(frame, fps),
        lower_left=(
            f"SCORE {metrics['score']:05d}  "
            f"CUR {metrics['x']:03d},{metrics['y']:03d}"
        ),
    )

def _run_boot(session: CaptureSession, env: object) -> dict[str, Any]:
    script = _boot_script_multi()
    for action in script:
        session.step(action)
    # Extra settle for late HUD appearance.
    for _ in range(120):
        obs = session.step(idle_action_multi(players=2))
        if looks_like_search_scene(np.asarray(obs)):
            break
    obs = session.idle(2)
    if not looks_like_search_scene(np.asarray(obs)):
        raise BootError("boot did not reach Scene1 search HUD")
    return {"boot_frames": session.frame, "metrics": _metrics(env)}

def _run_scene(
    session: CaptureSession,
    env: object,
    recipe: SceneRecipe,
    *,
    probe_layout: bool,
) -> dict[str, Any]:
    score_before = _metrics(env)["score"]
    adapted = _adapt_recipe(recipe, score_before)
    used_probe = 0
    if probe_layout and recipe.banner.number in (3, 4, 5):
        # Full assist seek before committing the scroll click.
        probe_frames = max(adapted.scroll_p2a, 1)
        land_x, land_y = probe_assist_landing(
            session, env, frames=probe_frames
        )
        used_probe = probe_frames
        if not is_favorable_scroll_layout(recipe.banner.number, land_x):
            raise SoftLayoutError(
                f"soft layout on Scene{recipe.banner.number}: "
                f"assist landed at ({land_x},{land_y})"
            )
        # Assist already completed; recipe should only drive/click.
        adapted = replace(adapted, scroll_p2a=0)
    summary = run_scene_recipe(session, env, adapted)
    summary["score_before"] = score_before
    summary["adapted_scroll_score"] = adapted.scroll_score
    summary["adapted_clear_score"] = adapted.clear_score
    summary["assist_probe_frames"] = used_probe
    return summary

def run_full_game(
    *,
    output: Path,
    dry_run: bool = False,
    frame_stride: int = 2,
    scale: int = 2,
    fps: int = 60,
    ending_hold_frames: int = 300,
    card_frames: int = 90,
    max_attempts: int = 8,
) -> dict[str, Any]:
    """Boot, clear five scenes, hold ending; optionally encode an MP4."""
    configure_headless()
    output.parent.mkdir(parents=True, exist_ok=True)
    attempts: list[dict[str, Any]] = []
    last_error: str | None = None

    for attempt in range(1, max_attempts + 1):
        writer: FrameVideoWriter | None = None
        env = make_env(
            game=GAME,
            state="NONE",
            game_dir=GAME_DIR,
            render_mode="rgb_array",
            players=2,
        )
        scene_reports: list[dict[str, Any]] = []
        current_scene = 0
        try:
            obs, _ = reset_obs(env)
            if not dry_run:
                writer = FrameVideoWriter(
                    output,
                    width=256,
                    height=224 + FOOTER_HEIGHT,
                    fps=fps,
                    scale=scale,
                )
                intro = title_card_with_footer(
                    [
                        "GREAT WALDO SEARCH",
                        "Continuous power-on to five-scrolls ending",
                        "One emulator session; no mid-run state loads",
                        "Live score/cursor footer + P1/P2 buttons",
                    ]
                )
                for _ in range(card_frames):
                    writer.write(intro)

            def footer(
                active_env: object,
                action: list[int],
                frame: int,
                active_fps: float,
            ) -> FooterLabels:
                recipe = SCENE_RECIPES[max(current_scene - 1, 0)]
                return _footer_for_scene(
                    active_env,
                    action,
                    frame,
                    active_fps,
                    scene=max(current_scene, 1),
                    name=recipe.banner.name,
                )

            sink = writer if writer is not None else _NullSink()
            session = CaptureSession(
                env,
                sink=sink,
                footer=footer,
                fps=float(fps),
                frame_stride=frame_stride,
                players=2,
                idle_action=lambda: idle_action_multi(players=2),
            )
            session.capture(np.asarray(obs), idle_action_multi(players=2))

            boot = _run_boot(session, env)
            current_scene = 1
            for index, recipe in enumerate(SCENE_RECIPES):
                current_scene = recipe.banner.number
                report = _run_scene(
                    session,
                    env,
                    recipe,
                    probe_layout=recipe.banner.number >= 3,
                )
                scene_reports.append(report)
                if recipe.banner.number < 5:
                    advance_scene(session, cleared_scene=recipe.banner.number)
                    # Brief settle so the next HUD is live before probing.
                    session.idle(30)

            # Ending hold — do not mash A (that advances into post-game).
            final_metrics = _metrics(env)
            if final_metrics["score"] < SCENE5_CLEAR_SCORE:
                raise RuntimeError(
                    f"ending score too low: {final_metrics['score']}"
                )
            current_scene = 5
            session.idle(ending_hold_frames)
            video_frames = 0
            if writer is not None:
                writer.close()
                video_frames = writer.frames_written
                writer = None

            manifest: dict[str, Any] = {
                "format": "great-waldo-search-full-credits-run",
                "game": "great_waldo_search",
                "continuous_run": True,
                "uses_development_checkpoints": False,
                "state_loads": 0,
                "ending_scope": (
                    "five-scrolls ending from continuous power-on run"
                ),
                "silent_capture": not dry_run,
                "dry_run": dry_run,
                "attempt": attempt,
                "attempts_failed": attempts,
                "boot": boot,
                "scenes": scene_reports,
                "final_metrics": final_metrics,
                "frame_stride": frame_stride,
                "video_fps": fps,
                "emulator_frames": session.frame,
                "video_frames": video_frames,
                "video": None if dry_run else output.name,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            }
            manifest_path = (
                RECORDINGS_DIR / "great_waldo_search_full_credits_dry.json"
                if dry_run
                else output.with_suffix(".json")
            )
            print(
                f"[full_run] SUCCESS attempt={attempt} "
                f"frames={session.frame} score={final_metrics['score']}"
            )
            manifest_path.write_text(
                json.dumps(manifest, indent=2) + "\n",
                encoding="utf-8",
            )
            manifest["manifest"] = str(manifest_path)
            return manifest
        except (SoftLayoutError, BootError, RuntimeError) as exc:
            last_error = str(exc)
            attempts.append(
                {
                    "attempt": attempt,
                    "error": last_error,
                    "scenes_cleared": len(scene_reports),
                }
            )
            print(f"[full_run] attempt {attempt}/{max_attempts} failed: {exc}")
            if writer is not None:
                try:
                    writer.close()
                except RuntimeError:
                    # Incomplete ffmpeg pipe on abort — delete partial.
                    pass
                if output.exists():
                    output.unlink(missing_ok=True)
        finally:
            env.close()

    raise RuntimeError(
        f"full run failed after {max_attempts} attempts: {last_error}"
    )

class _NullSink:
    """Discard frames during dry-run policy checks."""

    frames_written = 0

    def write(self, frame: np.ndarray) -> None:
        del frame
        self.frames_written += 1

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=RECORDINGS_DIR / "great_waldo_search_full_credits.mp4",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--ending-hold-frames", type=int, default=300)
    parser.add_argument("--max-attempts", type=int, default=8)
    return parser

def main(argv: list[str] | None = None) -> int:
    """CLI entry for continuous Waldo full-run capture."""
    args = _build_parser().parse_args(argv)
    manifest = run_full_game(
        output=args.output,
        dry_run=args.dry_run,
        frame_stride=args.frame_stride,
        scale=args.scale,
        fps=args.fps,
        ending_hold_frames=args.ending_hold_frames,
        max_attempts=args.max_attempts,
    )
    print(json.dumps({k: manifest[k] for k in manifest if k != "scenes"}, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
