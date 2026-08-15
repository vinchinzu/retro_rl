"""Shared segment-runner helpers for Zelda I scripts.

Thin CLIs should use this module instead of re-copying sys.path / make_env /
assist / JSON report boilerplate. Path logic stays in library controllers.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR

# Repo layout: nes/zelda_i/runner.py → parents[2] = repo root, parents[1] = nes/
_REPO_ROOT = Path(__file__).resolve().parents[2]
_NES_ROOT = Path(__file__).resolve().parents[1]


def ensure_import_paths() -> None:
    """Insert repo + nes roots on sys.path (idempotent)."""
    for p in (_REPO_ROOT, _NES_ROOT):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


def add_common_args(
    parser: argparse.ArgumentParser,
    *,
    default_state: str | None = None,
    default_tag: str = "run",
    default_trials: int = 1,
) -> argparse.ArgumentParser:
    """Standard flags shared by segment scripts."""
    parser.add_argument(
        "--from-state",
        default=default_state,
        help="Integration save-state name (None = natural boot where supported)",
    )
    parser.add_argument("--tag", default=default_tag)
    parser.add_argument("--trials", type=int, default=default_trials)
    parser.add_argument(
        "--infinite-life",
        action="store_true",
        help="Survival assist (not Clean STATUS)",
    )
    parser.add_argument(
        "--save-state",
        action="store_true",
        help="Write checkpoint on success when controller supports it",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser


def make_assist(enabled: bool):
    """Return UnlimitedHealthAssist or None."""
    if not enabled:
        return None
    from zelda_i.assist import UnlimitedHealthAssist

    return UnlimitedHealthAssist()


def open_env(
    *,
    from_state: str | None = None,
    seed: int = 0,
    headless: bool = True,
):
    """Create fceumm env for LegendOfZelda-Nes; optionally load a save state."""
    from retro_harness.env import load_state, make_env
    from retro_harness.segment_runner import configure_headless

    if headless:
        configure_headless()
    env = make_env(GAME, seed=seed)
    if from_state:
        load_state(env, GAME_DIR, GAME, from_state)
    return env


_STOP_PHASES = frozenset({"FAILED", "DONE"})


def controller_stopped(controller: Any) -> bool:
    """True when a controller reports success, fail, or a terminal phase.

    Accepts Enum phases (``.name``) and string phases (L3 raft ``"failed"``).
    """
    if getattr(controller, "success", False) or getattr(controller, "failed", False):
        return True
    phase = getattr(controller, "phase", None)
    if phase is None:
        return False
    name = getattr(phase, "name", None)
    token = str(name if name is not None else phase)
    return token.upper() in _STOP_PHASES


def run_controller(
    controller: Any,
    *,
    from_state: str | None,
    infinite_life: bool = False,
    max_frames: int | None = None,
    seed: int = 0,
    on_frame: Callable[[Any, Any, int], None] | None = None,
    step_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Step ``controller`` until success/fail/timeout; return report dict.

    Controller must expose ``.step(snap) -> FrameAction``, ``.success``,
    and preferably ``.report()``. Optional ``.phase`` / ``.failed`` / ``.max_frames``.
    """
    from zelda_i.ram import read_snapshot

    assist = make_assist(infinite_life)
    env = open_env(from_state=from_state, seed=seed)
    extra = step_kwargs or {}
    limit = max_frames
    if limit is None:
        limit = int(getattr(controller, "max_frames", 30000) or 30000)

    try:
        for frame in range(limit):
            snap = read_snapshot(env.get_ram())
            action = controller.step(snap, **extra)
            env.step(action.action)
            if assist is not None:
                assist.apply_env(env, frame=frame)
            if on_frame is not None:
                on_frame(env, controller, frame)
            if controller_stopped(controller):
                break
    finally:
        env.close()

    report: dict[str, Any] = {}
    if hasattr(controller, "report"):
        report = dict(controller.report())
    report.setdefault("success", bool(getattr(controller, "success", False)))
    report["from_state"] = from_state
    report["infinite_life"] = infinite_life
    if assist is not None and hasattr(assist, "report"):
        report["assist"] = assist.report()
    return report


def write_report(name: str, payload: dict[str, Any], *, tag: str = "") -> Path:
    """Write JSON under recordings/; return path."""
    from retro_harness.segment_runner import write_json_report

    stem = f"{name}_{tag}" if tag else name
    out = RECORDINGS_DIR / f"{stem}.json"
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    write_json_report(out, payload)
    return out


def add_video_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Optional MP4 flags shared by Survival spine runners."""
    parser.add_argument(
        "--video",
        nargs="?",
        const="AUTO",
        default=None,
        help="Record MP4 (ffmpeg). Omit the value for the runner default path.",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Disable emulator audio on the MP4",
    )
    parser.add_argument(
        "--no-footer",
        action="store_true",
        help="Disable button/frame footer on the MP4",
    )
    parser.add_argument(
        "--no-intro",
        action="store_true",
        help="Skip YouTube intro slide before gameplay",
    )
    parser.add_argument(
        "--intro-frames",
        type=int,
        default=None,
        help="Intro hold frames at 60fps (default from youtube_intro)",
    )
    parser.add_argument(
        "--hq",
        action="store_true",
        help="Higher quality encode (scale=3, crf=15, preset=slow)",
    )
    return parser


def resolve_video(
    args: argparse.Namespace,
    *,
    default_path: Path,
) -> tuple[Path | None, Any, int]:
    """Return (path, VideoCaptureConfig|None, intro_frames) from CLI args."""
    from retro_harness.video import VideoCaptureConfig
    from retro_harness.youtube_intro import DEFAULT_INTRO_FRAMES

    intro = 0 if getattr(args, "no_intro", False) else (
        DEFAULT_INTRO_FRAMES
        if getattr(args, "intro_frames", None) is None
        else max(0, int(args.intro_frames))
    )
    if getattr(args, "video", None) is None:
        return None, None, intro
    path = default_path if args.video == "AUTO" else Path(args.video)
    audio = not getattr(args, "no_audio", False)
    footer = not getattr(args, "no_footer", False)
    if getattr(args, "hq", False):
        config = VideoCaptureConfig.high_quality(audio=audio, footer=footer)
    else:
        config = VideoCaptureConfig(audio=audio, footer=footer)
    return path, config, intro


class VideoTap:
    """Attach MP4 capture + room-transition PNGs by wrapping ``env.step``."""

    def __init__(
        self,
        path: Path | None,
        config: Any | None,
        *,
        tag: str,
        intro_summary: str = "",
        intro_frames: int = 0,
        intervention: str = "Survival infinite-life",
    ) -> None:
        self.path = path
        self.config = config
        self.tag = tag
        self.intro_summary = intro_summary
        self.intro_frames = intro_frames
        self.intervention = intervention
        self.writer: Any = None
        self.intro_written = 0
        self.frame = 0
        self._last_room: tuple[int, int] | None = None
        self.transitions: list[dict[str, Any]] = []
        self._orig_step: Callable[..., Any] | None = None
        self._env: Any = None

    def attach(self, env: Any, obs: Any) -> None:
        self._env = env
        if self.path is None:
            return
        from retro_harness.video import VideoRecorder
        from retro_harness.youtube_intro import project_intro_lines, render_intro_card
        import numpy as np

        config = self.config
        audio_rate: int | None = None
        if config is not None and config.audio:
            em = getattr(env, "em", None)
            if em is not None and hasattr(em, "get_audio_rate"):
                audio_rate = int(em.get_audio_rate())
            else:
                from dataclasses import replace

                config = replace(config, audio=False)
        self.writer = VideoRecorder(
            self.path,
            width=int(obs.shape[1]),
            height=int(obs.shape[0]),
            config=config,
            audio_rate=audio_rate,
        )
        if self.intro_frames > 0:
            lines = project_intro_lines(
                game_title="The Legend of Zelda (NES)",
                run_summary=self.intro_summary or self.tag,
                intervention=self.intervention,
            )
            card = render_intro_card(
                lines,
                width=int(obs.shape[1]),
                height=int(obs.shape[0]),
                with_footer=False,
            )
            silent = None
            if self.writer.config.audio and audio_rate:
                n = max(1, int(round(audio_rate / float(self.writer.config.fps))))
                silent = np.zeros((n, 2), dtype=np.int16)
            for i in range(self.intro_frames):
                self.writer.write(
                    card,
                    audio=silent,
                    frame_index=-(self.intro_frames - i),
                )
            self.intro_written = self.intro_frames

        orig = env.step

        def _step(action, *args, **kwargs):
            result = orig(action, *args, **kwargs)
            self.observe(env, result[0], action)
            return result

        self._orig_step = orig
        env.step = _step

    def observe(self, env: Any, obs: Any, action: Any) -> None:
        from retro_harness.segment_runner import save_rgb_png
        from zelda_i.ram import read_snapshot

        if self.writer is not None:
            self.writer.write_from_env(
                env, obs, action=action, frame_index=self.frame
            )
        snap = read_snapshot(env.get_ram())
        room = (int(snap.level), int(snap.screen))
        if self._last_room is None:
            self._last_room = room
        elif room != self._last_room:
            entry = {
                "f": self.frame,
                "from": f"L{self._last_room[0]}:0x{self._last_room[1]:02x}",
                "to": f"L{room[0]}:0x{room[1]:02x}",
                "mode": snap.mode,
                "xy": [snap.link_x, snap.link_y],
            }
            self.transitions.append(entry)
            save_rgb_png(
                obs,
                RECORDINGS_DIR
                / f"{self.tag}_L{room[0]}_{room[1]:02x}_f{self.frame:05d}.png",
            )
            self._last_room = room
        self.frame += 1

    def close(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "path": None,
            "encoded_frames": 0,
            "intro_frames": self.intro_written,
            "gameplay_frames": self.frame,
            "transitions": list(self.transitions),
        }
        if self.writer is not None:
            encoded = self.writer.frames
            closed = self.writer.close()
            info["path"] = str(closed)
            info["encoded_frames"] = encoded
            info["gameplay_frames"] = max(0, encoded - self.intro_written)
            self.writer = None
        if self._env is not None and self._orig_step is not None:
            self._env.step = self._orig_step
            self._orig_step = None
        return info


__all__ = [
    "add_common_args",
    "add_video_args",
    "controller_stopped",
    "ensure_import_paths",
    "make_assist",
    "open_env",
    "resolve_video",
    "run_controller",
    "VideoTap",
    "write_report",
]
