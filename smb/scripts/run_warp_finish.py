"""Run the policy-driven eight-exit SMB warp route to the 8-4 ending.

Modes:

- ``poweron`` (preferred / M7): hard reset → fixed boot script frames → idle
  phase-align → one controller seed through all eight exits. **Clean** —
  zero emulator-state loads after ``env.reset()``.
- ``continuous``: published ``Level1_1`` + idle phase-align + same seed
  (no mid-attempt load; not power-on).
- ``suffix``: published ``Level1_2_WarpMid`` + ending suffix only.
- ``chain``: power-on natural 1-1 + one disclosed mid-1-2 splice + suffix.

```bash
# Clean power-on → 8-4 ending
uv run python -m smb.scripts.run_warp_finish --mode poweron --trials 3

# Record MP4
uv run python -m smb.scripts.run_warp_finish --mode poweron --record
```
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Any

import numpy as np

from retro_harness.env import make_env
from smb.full_run import read_state_bytes
from smb.menus import boot_to_level1_script
from smb.paths import (
    FULLGAME_REPLAYS_DIR,
    GAME_DIR,
    GAME_V0,
    INTEGRATION_V0_DIR,
    RECORDINGS_DIR,
)
from smb.policy import (
    CONTINUOUS_SETTLE_FRAMES,
    DEFAULT_1_1_SEED,
    DEFAULT_CONTINUOUS_SEED,
    DEFAULT_WARP_SUFFIX_SEED,
    POWERON_BOOT_FRAMES,
    POWERON_SETTLE_FRAMES,
    Level11ReplayPolicy,
    Nes9ReplayPolicy,
)
from smb.ram import read_snapshot, reached_ending, segment_1_1_success
from smb.scripts.run_warp_chain import (
    DEFAULT_MAX_FRAMES_11,
    NATURAL_SETTLE,
    _boot_to_ready,
    _idle,
)
from smb.timing import build_timing_block, summarize_comparisons
from snes_oneshot.recording_footer import (
    FOOTER_HEIGHT,
    frame_timestamp,
    render_footer_frame,
)
from snes_oneshot.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)

WARP_MID_STATE = INTEGRATION_V0_DIR / "Level1_2_WarpMid.state"
LEVEL1_1_STATE = INTEGRATION_V0_DIR / "Level1_1.state"
DEFAULT_MAX_SUFFIX_FRAMES = 22_000
DEFAULT_MAX_CONTINUOUS_FRAMES = 25_000
ENDING_SETTLE_FRAMES = 120

# Exit milestones for the mid-1-2 suffix (world/level after each exit).
_SUFFIX_MILESTONES: tuple[tuple[str, int, int], ...] = (
    ("1-2", 3, 0),
    ("4-1", 3, 1),
    ("4-2", 7, 0),
    ("8-1", 7, 1),
    ("8-2", 7, 2),
    ("8-3", 7, 3),
)

# Continuous Level1_1 path also records the 1-1 exit.
_CONTINUOUS_MILESTONES: tuple[tuple[str, int, int], ...] = (
    ("1-1", 0, 1),
    ("1-2", 3, 0),
    ("4-1", 3, 1),
    ("4-2", 7, 0),
    ("8-1", 7, 1),
    ("8-2", 7, 2),
    ("8-3", 7, 3),
)


def _snapshot_dict(snap) -> dict[str, int]:
    return {
        "player_x": snap.player_x,
        "player_y": snap.player_y,
        "world": snap.world,
        "level": snap.level,
        "level_id": snap.level_id,
        "area_pointer": snap.area_pointer,
        "lives": snap.lives,
        "player_state": snap.player_state,
        "oper_mode": snap.oper_mode,
    }


class _VideoWriter:
    """ffmpeg RGB (+ native PCM) capture with NES button / timestamp footer."""

    def __init__(
        self,
        path: Path,
        *,
        width: int,
        height: int,
        scale: int = 3,
        fps: int = 60,
        audio_rate: int | None = None,
        hud: bool = True,
        route_label: str = "SMB any%",
    ) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg is required for video recording")
        self.path = path
        self.scale = max(1, scale)
        self.fps = fps
        self.hud = hud
        self.route_label = route_label
        self.src_w = width
        self.src_h = height + (FOOTER_HEIGHT if hud else 0)
        self.out_w = self.src_w * self.scale
        self.out_h = self.src_h * self.scale
        self.frames = 0
        self.audio_samples = 0
        self.audio_rate = audio_rate
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._silent = path.with_suffix(".partial.video.mp4")
        self._wav = path.with_suffix(".partial.audio.wav")
        self._proc = subprocess.Popen(
            [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{self.out_w}x{self.out_h}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(self._silent),
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._audio: wave.Wave_write | None = None
        if audio_rate is not None and audio_rate > 0:
            self._audio = wave.open(str(self._wav), "wb")
            self._audio.setnchannels(2)
            self._audio.setsampwidth(2)
            self._audio.setframerate(audio_rate)

    def write(
        self,
        obs: np.ndarray,
        *,
        action: list[int] | np.ndarray | None = None,
        audio: np.ndarray | None = None,
        label: str = "",
        snap: Any | None = None,
    ) -> None:
        if self._proc.stdin is None:
            return
        rgb = np.asarray(obs, dtype=np.uint8)
        act_list: list[int] | None
        if action is None:
            act_list = None
        else:
            act_list = [int(v) for v in np.asarray(action).tolist()]

        if self.hud:
            level = ""
            lives = ""
            xpos = ""
            if snap is not None:
                level = f"{int(snap.world) + 1}-{int(snap.level) + 1}"
                lives = f"L{int(snap.lives)}"
                xpos = f"x={int(snap.player_x)}"
            upper_left = f"{self.route_label}  {level}  {lives}".strip()
            if label:
                upper_left = f"{upper_left}  {label}".strip()
            upper_right = frame_timestamp(self.frames, self.fps)
            lower_left = xpos or "---"
            rgb = render_footer_frame(
                rgb,
                upper_left=upper_left,
                upper_right=upper_right,
                lower_left=lower_left,
                action=act_list,
                players=1,
                layout="nes",
            )

        if rgb.shape != (self.src_h, self.src_w, 3):
            raise ValueError(
                f"expected frame {(self.src_h, self.src_w, 3)}, got {rgb.shape}"
            )
        if self.scale > 1:
            rgb = np.repeat(np.repeat(rgb, self.scale, axis=0), self.scale, axis=1)
        self._proc.stdin.write(rgb.tobytes())
        self.frames += 1

        if self._audio is None or audio is None:
            return
        pcm = np.asarray(audio, dtype=np.int16)
        if pcm.size == 0:
            return
        if pcm.ndim == 1:
            if pcm.size % 2:
                raise ValueError(f"odd stereo PCM sample count: {pcm.size}")
            pcm = pcm.reshape(-1, 2)
        if pcm.ndim != 2 or pcm.shape[1] != 2:
            raise ValueError(f"expected stereo PCM, got {pcm.shape}")
        self._audio.writeframesraw(pcm.astype("<i2", copy=False).tobytes())
        self.audio_samples += int(pcm.shape[0])

    def _close_streams(self) -> None:
        if self._audio is not None:
            self._audio.close()
            self._audio = None
        if self._proc.stdin is not None:
            try:
                self._proc.stdin.close()
            except BrokenPipeError:
                pass
        stderr = self._proc.stderr.read() if self._proc.stderr else b""
        code = self._proc.wait()
        if code != 0:
            raise RuntimeError(
                f"ffmpeg video encode failed ({code}): "
                f"{stderr.decode('utf-8', errors='replace')[-500:]}"
            )

    def close(self) -> None:
        self._close_streams()
        if self.audio_rate is None or not self._wav.exists():
            # Silent path: just promote the partial video.
            self._silent.replace(self.path)
            return
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(self._silent),
                "-i",
                str(self._wav),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-shortest",
                "-movflags",
                "+faststart",
                str(self.path),
            ],
            capture_output=True,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(
                result.stderr.decode("utf-8", errors="replace")[-500:]
            )
        self._silent.unlink(missing_ok=True)
        self._wav.unlink(missing_ok=True)


def _env_audio(env) -> np.ndarray | None:
    """Pull native stereo PCM from the emulator core when available."""
    em = getattr(env, "em", None)
    if em is None or not hasattr(em, "get_audio"):
        return None
    try:
        return np.asarray(em.get_audio(), dtype=np.int16)
    except Exception:  # noqa: BLE001 — audio is best-effort for recordings
        return None


def _env_audio_rate(env) -> int | None:
    em = getattr(env, "em", None)
    if em is None or not hasattr(em, "get_audio_rate"):
        return None
    try:
        rate = int(em.get_audio_rate())
    except Exception:  # noqa: BLE001
        return None
    return rate if rate > 0 else None


def _hud_action(action, snap) -> list[int] | np.ndarray | None:
    """Blank the footer buttons during automated flagpole / castle walk.

    The emulator still receives the real action; this only cleans the overlay
    so junk holds during player_state 3/4/5 do not clutter the recording.
    """
    if action is None:
        return None
    if snap is not None and int(getattr(snap, "player_state", -1)) in (3, 4, 5):
        return [0] * len(np.asarray(action).tolist())
    return action


def _write_video(
    video: _VideoWriter | None,
    obs,
    *,
    env=None,
    action=None,
    label: str = "",
    snap=None,
) -> None:
    if video is None or obs is None:
        return
    if snap is None and env is not None:
        snap = read_snapshot(env.get_ram())
    video.write(
        obs,
        action=_hud_action(action, snap),
        audio=_env_audio(env) if env is not None else None,
        label=label,
        snap=snap,
    )


def _run_policy_to_ending(
    env,
    *,
    seed_path: Path,
    max_frames: int,
    milestones: tuple[tuple[str, int, int], ...],
    ending_settle_frames: int = ENDING_SETTLE_FRAMES,
    video: _VideoWriter | None = None,
    label: str = "",
) -> tuple[dict[str, Any], object | None]:
    """Replay an RLE seed until ending / death / timeout; optional video."""
    policy = Nes9ReplayPolicy(
        seed_path=seed_path,
        action_size=int(env.action_space.shape[0]),
    )
    start = read_snapshot(env.get_ram())
    start_lives = start.lives
    reached: list[dict[str, Any]] = []
    next_milestone = 0
    max_x_by_level: dict[str, int] = {}
    outcome = "timeout"
    obs = None
    frame = 0

    while frame < max_frames:
        if policy.remaining == 0:
            outcome = "seed_exhausted"
            break
        tick = policy.step()
        obs, *_ = env.step(tick.action)
        frame += 1
        snap = read_snapshot(env.get_ram(), frame=frame)
        _write_video(
            video,
            obs,
            env=env,
            action=tick.action,
            label=label,
            snap=snap,
        )
        level_key = f"{snap.world + 1}-{snap.level + 1}"
        max_x_by_level[level_key] = max(
            max_x_by_level.get(level_key, 0), snap.player_x
        )

        if snap.lives < start_lives or snap.dying:
            outcome = "death"
            break

        if next_milestone < len(milestones):
            exit_id, world, level = milestones[next_milestone]
            if snap.world == world and snap.level == level:
                reached.append(
                    {
                        "exit_id": exit_id,
                        "frame": frame,
                        "world": snap.world,
                        "level": snap.level,
                        "lives": snap.lives,
                    }
                )
                next_milestone += 1

        if reached_ending(env.get_ram(), start_lives=start_lives):
            reached.append(
                {
                    "exit_id": "8-4",
                    "frame": frame,
                    "world": snap.world,
                    "level": snap.level,
                    "lives": snap.lives,
                }
            )
            outcome = "ending"
            break

    stable = 0
    if outcome == "ending":
        idle = np.zeros(int(env.action_space.shape[0]), dtype=np.int8)
        for _ in range(ending_settle_frames):
            obs, *_ = env.step(idle)
            snap = read_snapshot(env.get_ram(), frame=frame + stable + 1)
            _write_video(
                video,
                obs,
                env=env,
                action=idle,
                label="ending",
                snap=snap,
            )
            if reached_ending(env.get_ram(), start_lives=start_lives):
                stable += 1

    final = read_snapshot(env.get_ram(), frame=frame + stable)
    expected_ids = [row[0] for row in milestones] + ["8-4"]
    actual_ids = [row["exit_id"] for row in reached]
    success = (
        outcome == "ending"
        and actual_ids == expected_ids
        and stable == ending_settle_frames
    )
    return (
        {
            "success": success,
            "outcome": outcome,
            "label": label,
            "policy_frames": frame,
            "ending_settle_frames": stable,
            "start": _snapshot_dict(start),
            "final": _snapshot_dict(final),
            "milestones": reached,
            "max_x_by_level": max_x_by_level,
            "policy": policy.report(),
            "state_loads_during_policy": 0,
        },
        obs,
    )


def run_suffix_policy(
    env,
    *,
    seed_path: Path = DEFAULT_WARP_SUFFIX_SEED,
    max_frames: int = DEFAULT_MAX_SUFFIX_FRAMES,
    ending_settle_frames: int = ENDING_SETTLE_FRAMES,
    video: _VideoWriter | None = None,
) -> tuple[dict[str, Any], object | None]:
    """Run the no-reload mid-1-2-to-ending policy from the current state."""
    report, obs = _run_policy_to_ending(
        env,
        seed_path=seed_path,
        max_frames=max_frames,
        milestones=_SUFFIX_MILESTONES,
        ending_settle_frames=ending_settle_frames,
        video=video,
        label="continuous_suffix",
    )
    report["state_loads_during_suffix"] = 0
    return report, obs


def _run_natural_1_1(
    env,
    *,
    seed_path: Path,
    max_frames: int,
    video: _VideoWriter | None = None,
) -> tuple[dict[str, Any], object | None]:
    obs, boot_frames = _boot_to_ready(env)
    if obs is None:
        return {"success": False, "outcome": "boot_fail"}, obs
    if video is not None:
        # Boot frames were not recorded; capture from settle onward.
        pass
    obs = _idle(env, NATURAL_SETTLE)
    policy = Level11ReplayPolicy(
        seed_path=seed_path,
        action_size=int(env.action_space.shape[0]),
    )
    start = read_snapshot(env.get_ram())
    max_x = start.player_x
    outcome = "timeout"
    frame = 0
    for frame in range(1, max_frames + 1):
        tick = policy.step()
        obs, *_ = env.step(tick.action)
        snap = read_snapshot(env.get_ram(), frame=frame)
        _write_video(
            video,
            obs,
            env=env,
            action=tick.action,
            label="1-1",
            snap=snap,
        )
        max_x = max(max_x, snap.player_x)
        if snap.lives < start.lives or snap.dying:
            outcome = "death"
            break
        if segment_1_1_success(
            env.get_ram(),
            start_lives=start.lives,
            max_player_x=max_x,
        ):
            outcome = "success"
            break
    return (
        {
            "success": outcome == "success",
            "outcome": outcome,
            "frames": frame,
            "boot_frames": boot_frames,
            "settle_frames": NATURAL_SETTLE,
            "max_player_x": max_x,
            "policy": policy.report(),
        },
        obs,
    )


def _fixed_boot(
    env,
    n_frames: int,
    *,
    video: _VideoWriter | None = None,
) -> tuple[object | None, int]:
    """Run exactly ``n_frames`` of the title boot script (pad with idle)."""
    idle = np.zeros(int(env.action_space.shape[0]), dtype=np.int8)
    obs = None
    frames = 0
    for scripted in boot_to_level1_script():
        obs, *_ = env.step(scripted.action)
        frames += 1
        snap = read_snapshot(env.get_ram(), frame=frames)
        _write_video(
            video,
            obs,
            env=env,
            action=scripted.action,
            label="boot",
            snap=snap,
        )
        if frames >= n_frames:
            return obs, frames
    while frames < n_frames:
        obs, *_ = env.step(idle)
        frames += 1
        snap = read_snapshot(env.get_ram(), frame=frames)
        _write_video(
            video,
            obs,
            env=env,
            action=idle,
            label="boot",
            snap=snap,
        )
    return obs, frames


def run_warp_finish(
    *,
    mode: str = "poweron",
    seed_11: Path = DEFAULT_1_1_SEED,
    seed_suffix: Path = DEFAULT_WARP_SUFFIX_SEED,
    seed_continuous: Path = DEFAULT_CONTINUOUS_SEED,
    settle_continuous: int = CONTINUOUS_SETTLE_FRAMES,
    boot_poweron: int = POWERON_BOOT_FRAMES,
    settle_poweron: int = POWERON_SETTLE_FRAMES,
    max_frames_11: int = DEFAULT_MAX_FRAMES_11,
    max_suffix_frames: int = DEFAULT_MAX_SUFFIX_FRAMES,
    max_continuous_frames: int = DEFAULT_MAX_CONTINUOUS_FRAMES,
    out_dir: Path | None = None,
    tag: str = "warp_finish",
    record_path: Path | None = None,
    record_scale: int = 3,
    record_hud: bool = True,
    record_audio: bool = True,
) -> dict[str, Any]:
    """Run poweron / continuous / suffix / legacy chain finish attempt."""
    configure_headless()
    out = out_dir or (RECORDINGS_DIR / "warp_finish")
    out.mkdir(parents=True, exist_ok=True)

    if mode in ("poweron", "continuous") and not seed_continuous.exists():
        raise SystemExit(
            f"missing continuous seed: {seed_continuous} "
            "(run: uv run python -m smb.scripts.fold_continuous_policy)"
        )

    if mode == "poweron":
        intervention = {
            "class": "Clean",
            "start": "power_on",
            "boot_frames": boot_poweron,
            "settle_frames": settle_poweron,
            "mid_attempt_state_loads": 0,
            "note": (
                "env.reset power-on; fixed boot script frames + idle "
                "phase-align; controller-only through 8-4 ending"
            ),
        }
        benchmark_eligible = True
    elif mode == "continuous":
        if not LEVEL1_1_STATE.exists():
            raise SystemExit(f"missing Level1_1 state: {LEVEL1_1_STATE}")
        intervention = {
            "class": "Clean",
            "initial_state": str(LEVEL1_1_STATE.relative_to(GAME_DIR.parent)),
            "settle_frames": settle_continuous,
            "mid_attempt_state_loads": 0,
            "note": (
                "Level1_1 start + fixed idle phase-align; controller-only "
                "through 8-4 ending"
            ),
        }
        benchmark_eligible = False
    elif mode == "suffix":
        if not WARP_MID_STATE.exists():
            raise SystemExit(f"missing suffix start state: {WARP_MID_STATE}")
        intervention = {
            "class": "Clean",
            "initial_state": str(WARP_MID_STATE.relative_to(GAME_DIR.parent)),
            "mid_attempt_state_loads": 0,
            "note": "published development start state; controller-only afterward",
        }
        benchmark_eligible = False
    elif mode == "chain":
        intervention = {
            "class": "development splice (not benchmark eligible)",
            "mid_attempt_state_loads": 1,
            "note": (
                "one mid-1-2 state splice follows natural 1-1; "
                "prefer --mode poweron"
            ),
        }
        benchmark_eligible = False
    else:
        raise SystemExit(f"unknown mode {mode!r}")

    env = make_env(GAME_V0, "NONE", GAME_DIR, render_mode="rgb_array")
    report: dict[str, Any] = {
        "mode": mode,
        "success": False,
        "route_id": "smb_warp_any_percent",
        "runtime_observation": "Bronze",
        "benchmark_eligible": benchmark_eligible,
        "intervention": intervention,
        "stages": {},
    }
    video: _VideoWriter | None = None
    obs = None
    try:
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result

        if record_path is not None:
            if obs is None:
                obs = env.render()
            h, w = int(obs.shape[0]), int(obs.shape[1])
            audio_rate = _env_audio_rate(env) if record_audio else None
            video = _VideoWriter(
                record_path,
                width=w,
                height=h,
                scale=record_scale,
                audio_rate=audio_rate,
                hud=record_hud,
                route_label="SMB any%",
            )
            if obs is not None:
                _write_video(
                    video,
                    obs,
                    env=env,
                    action=None,
                    label="reset",
                )
            report["recording"] = {
                "path": str(record_path),
                "scale": record_scale,
                "hud": record_hud,
                "audio": audio_rate is not None,
                "audio_rate": audio_rate,
            }

        if mode == "poweron":
            obs, boot_frames = _fixed_boot(
                env, boot_poweron, video=video
            )
            idle = np.zeros(int(env.action_space.shape[0]), dtype=np.int8)
            for i in range(settle_poweron):
                obs, *_ = env.step(idle)
                snap = read_snapshot(env.get_ram(), frame=boot_frames + i + 1)
                _write_video(
                    video,
                    obs,
                    env=env,
                    action=idle,
                    label="settle",
                    snap=snap,
                )
            report["stages"]["boot"] = {
                "frames": boot_frames,
                "method": "fixed_boot_script",
                "settle_frames": settle_poweron,
            }
            policy_report, obs = _run_policy_to_ending(
                env,
                seed_path=seed_continuous,
                max_frames=max_continuous_frames,
                milestones=_CONTINUOUS_MILESTONES,
                video=video,
                label="poweron_to_ending",
            )
            report["stages"]["continuous"] = policy_report
            report["success"] = bool(policy_report["success"])
            report["outcome"] = policy_report["outcome"]
            report["exits_completed"] = len(policy_report["milestones"])
            report["state_loads_during_attempt"] = 0

        elif mode == "continuous":
            env.em.set_state(read_state_bytes(LEVEL1_1_STATE))
            idle = np.zeros(int(env.action_space.shape[0]), dtype=np.int8)
            if video is not None:
                frame0 = env.render()
                if frame0 is not None:
                    _write_video(
                        video,
                        frame0,
                        env=env,
                        action=None,
                        label="Level1_1",
                    )
            for i in range(settle_continuous):
                obs, *_ = env.step(idle)
                snap = read_snapshot(env.get_ram(), frame=i + 1)
                _write_video(
                    video,
                    obs,
                    env=env,
                    action=idle,
                    label="settle",
                    snap=snap,
                )
            report["stages"]["settle"] = {
                "frames": settle_continuous,
                "start_state": "Level1_1",
            }
            policy_report, obs = _run_policy_to_ending(
                env,
                seed_path=seed_continuous,
                max_frames=max_continuous_frames,
                milestones=_CONTINUOUS_MILESTONES,
                video=video,
                label="continuous_1_1_to_ending",
            )
            report["stages"]["continuous"] = policy_report
            report["success"] = bool(policy_report["success"])
            report["outcome"] = policy_report["outcome"]
            report["exits_completed"] = len(policy_report["milestones"])
            report["state_loads_during_attempt"] = 0

        elif mode == "chain":
            stage_11, obs = _run_natural_1_1(
                env,
                seed_path=seed_11,
                max_frames=max_frames_11,
                video=video,
            )
            report["stages"]["1-1"] = stage_11
            if not stage_11["success"]:
                report["outcome"] = f"1-1_{stage_11['outcome']}"
                return report
            if not WARP_MID_STATE.exists():
                raise SystemExit(f"missing suffix start state: {WARP_MID_STATE}")
            env.em.set_state(read_state_bytes(WARP_MID_STATE))
            report["suffix_entry"] = {
                "state": str(WARP_MID_STATE),
                "start_exit": "1-2",
            }
            suffix, obs = run_suffix_policy(
                env,
                seed_path=seed_suffix,
                max_frames=max_suffix_frames,
                video=video,
            )
            report["stages"]["continuous_suffix"] = suffix
            report["success"] = bool(suffix["success"])
            report["outcome"] = suffix["outcome"]
            report["exits_completed"] = 1 + len(suffix["milestones"])
            report["state_loads_during_attempt"] = 1

        elif mode == "suffix":
            env.em.set_state(read_state_bytes(WARP_MID_STATE))
            report["suffix_entry"] = {
                "state": str(WARP_MID_STATE),
                "start_exit": "1-2",
            }
            suffix, obs = run_suffix_policy(
                env,
                seed_path=seed_suffix,
                max_frames=max_suffix_frames,
                video=video,
            )
            report["stages"]["continuous_suffix"] = suffix
            report["success"] = bool(suffix["success"])
            report["outcome"] = suffix["outcome"]
            report["exits_completed"] = len(suffix["milestones"])
            report["state_loads_during_attempt"] = 0
        else:
            raise SystemExit(f"unknown mode {mode!r}")

        if obs is not None:
            suffix_name = "ending" if report["success"] else "fail"
            png = save_rgb_png(obs, out / f"{tag}_{suffix_name}.png")
            report["screenshot"] = str(png)
        if video is not None:
            report["video"] = str(record_path)
            report["video_frames"] = video.frames
            report["video_audio_samples"] = video.audio_samples

        # Attach named TAS/RTA timing contracts when we have a continuous path.
        continuous = report.get("stages", {}).get("continuous")
        if continuous and continuous.get("policy_frames") is not None:
            boot = report.get("stages", {}).get("boot") or {}
            settle = report.get("stages", {}).get("settle") or {}
            boot_frames = boot.get("frames") if mode == "poweron" else None
            settle_frames = (
                boot.get("settle_frames")
                if mode == "poweron"
                else settle.get("frames")
            )
            report["timing"] = build_timing_block(
                mode=mode,
                boot_frames=boot_frames,
                settle_frames=settle_frames,
                policy_frames_to_ending=int(continuous["policy_frames"]),
                milestones=continuous.get("milestones") or [],
            )
        return report
    finally:
        if video is not None:
            try:
                video.close()
            except Exception as exc:  # noqa: BLE001 — still write report
                report["video_error"] = str(exc)
        report_path = out / f"{tag}_report.json"
        write_json_report(report_path, report)
        stage = (
            report.get("stages", {}).get("continuous")
            or report.get("stages", {}).get("continuous_suffix")
            or {}
        )
        timing_note = ""
        if report.get("timing"):
            timing_note = " | " + summarize_comparisons(report["timing"])
        print(
            f"warp_finish mode={mode} outcome={report.get('outcome')} "
            f"success={report.get('success')} "
            f"exits={report.get('exits_completed', 0)} "
            f"policy_frames={stage.get('policy_frames', 0)} "
            f"report={report_path}"
            + (f" video={record_path}" if record_path else "")
            + timing_note,
            flush=True,
        )
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("poweron", "continuous", "chain", "suffix"),
        default="poweron",
        help="poweron=Clean reset-to-ending (default); continuous=Level1_1",
    )
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--seed-11", type=Path, default=DEFAULT_1_1_SEED)
    parser.add_argument(
        "--seed-suffix",
        type=Path,
        default=DEFAULT_WARP_SUFFIX_SEED,
    )
    parser.add_argument(
        "--seed-continuous",
        type=Path,
        default=DEFAULT_CONTINUOUS_SEED,
    )
    parser.add_argument(
        "--settle-continuous",
        type=int,
        default=CONTINUOUS_SETTLE_FRAMES,
    )
    parser.add_argument(
        "--boot-poweron",
        type=int,
        default=POWERON_BOOT_FRAMES,
    )
    parser.add_argument(
        "--settle-poweron",
        type=int,
        default=POWERON_SETTLE_FRAMES,
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--record",
        action="store_true",
        help="Write MP4 under recordings/fullgame_replays/",
    )
    parser.add_argument(
        "--record-path",
        type=Path,
        default=None,
        help="Explicit MP4 path (implies recording)",
    )
    parser.add_argument("--record-scale", type=int, default=3)
    parser.add_argument(
        "--no-record-hud",
        action="store_true",
        help="Disable button / timestamp footer overlay on recordings",
    )
    parser.add_argument(
        "--no-record-audio",
        action="store_true",
        help="Disable native emulator audio mux into the MP4",
    )
    args = parser.parse_args()

    successes = 0
    trial_reports: list[dict[str, Any]] = []
    for trial in range(1, args.trials + 1):
        tag = (
            f"warp_finish_{args.mode}_t{trial}"
            if args.trials > 1
            else f"warp_finish_{args.mode}"
        )
        record_path = args.record_path
        if args.record and record_path is None:
            record_path = (
                FULLGAME_REPLAYS_DIR
                / f"smb_warp_any_percent_{args.mode}"
                f"{'' if args.trials == 1 else f'_t{trial}'}.mp4"
            )
        elif args.trials > 1 and record_path is not None:
            record_path = record_path.with_name(
                f"{record_path.stem}_t{trial}{record_path.suffix}"
            )

        report = run_warp_finish(
            mode=args.mode,
            seed_11=args.seed_11,
            seed_suffix=args.seed_suffix,
            seed_continuous=args.seed_continuous,
            settle_continuous=args.settle_continuous,
            boot_poweron=args.boot_poweron,
            settle_poweron=args.settle_poweron,
            out_dir=args.out_dir,
            tag=tag,
            record_path=record_path,
            record_scale=args.record_scale,
            record_hud=not args.no_record_hud,
            record_audio=not args.no_record_audio,
        )
        trial_reports.append(report)
        successes += int(bool(report.get("success")))

    if args.trials > 1:
        stage_key = (
            "continuous"
            if args.mode in ("poweron", "continuous")
            else "continuous_suffix"
        )
        first_stage = (
            trial_reports[0].get("stages", {}).get(stage_key, {})
            if trial_reports
            else {}
        )
        seed_path = (
            args.seed_continuous
            if args.mode in ("poweron", "continuous")
            else args.seed_suffix
        ).resolve()
        try:
            seed_label = str(seed_path.relative_to(GAME_DIR.parent.resolve()))
        except ValueError:
            seed_label = str(seed_path)
        summary = {
            "route_id": "smb_warp_any_percent",
            "mode": args.mode,
            "runtime_observation": "Bronze",
            "benchmark_eligible": bool(
                trial_reports[0].get("benchmark_eligible") if trial_reports else False
            ),
            "intervention": (
                trial_reports[0].get("intervention") if trial_reports else None
            ),
            "trials": args.trials,
            "successes": successes,
            "success_rate": successes / args.trials,
            "outcomes": [report.get("outcome") for report in trial_reports],
            "exits_completed": [
                report.get("exits_completed") for report in trial_reports
            ],
            "state_loads_during_attempt": [
                report.get("state_loads_during_attempt") for report in trial_reports
            ],
            "policy_seed": seed_label,
            "policy_frames": first_stage.get("policy_frames"),
            "ending_settle_frames": first_stage.get("ending_settle_frames"),
            "milestones": first_stage.get("milestones"),
            "final": first_stage.get("final"),
        }
        if args.mode == "chain":
            summary["prelude"] = {
                "exit_id": "1-1",
                "outcomes": [
                    report.get("stages", {}).get("1-1", {}).get("outcome")
                    for report in trial_reports
                ],
                "frames": [
                    report.get("stages", {}).get("1-1", {}).get("frames")
                    for report in trial_reports
                ],
            }
        if args.mode == "continuous":
            summary["settle_frames"] = args.settle_continuous
        if args.mode == "poweron":
            summary["boot_frames"] = args.boot_poweron
            summary["settle_frames"] = args.settle_poweron
        summary_dir = args.out_dir or (RECORDINGS_DIR / "warp_finish")
        write_json_report(
            summary_dir / f"warp_finish_{args.mode}_trials_report.json",
            summary,
        )
        print(f"trials {successes}/{args.trials} success", flush=True)
    raise SystemExit(0 if successes == args.trials else 1)


if __name__ == "__main__":
    main()
