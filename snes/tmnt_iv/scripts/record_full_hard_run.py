"""Record one low-assist TMNT IV Hard run from power-on through staff credits.

The run uses one emulator session and selects Hard through the real menus. It
never loads a save state, never writes stage/lives/boss RAM, and never presses
the HP-draining special. Damage is measured from natural HP drops.

Assists (disclosed, minimized vs the old every-hit restore-to-96; **default ON**):
1. Emergency HP top-up when about to die (HP <= threshold → 80).
2. Super Shredder form-2 iframe hold at 1 — his demutation projectile
   bypasses ordinary HP and is not yet reliably dodged.

Clean track (``--clean``): both assists off, default artifacts use the
``tmnt_iv_full_hard_clean`` stem (never overwrites assisted baselines), and
integrity fails if any assist counter is non-zero. Long forms:
``--no-emergency-hp`` / ``--no-iframe-hold`` (either alone is not full Clean).

Any life decrement aborts the run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import wave
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from retro_harness.env import make_env, save_state  # noqa: E402
from retro_harness.controls import SNES_START  # noqa: E402
from retro_harness.actions import buttons, idle_action  # noqa: E402
from retro_harness.ram_state import GameMode, GameState  # noqa: E402
from retro_harness.video import render_footer_frame  # noqa: E402
from retro_harness.segment_runner import configure_headless  # noqa: E402
from tmnt_iv.paths import (  # noqa: E402
    GAME,
    GAME_DIR,
    RECORDINGS_DIR,
    ROMS_DIR,
    default_full_run_paths,
)
from tmnt_iv.policy import Stage1Policy  # noqa: E402
from tmnt_iv.ram import parse_game_state  # noqa: E402

_FOOTER_HEIGHT = 16
_METRIC_HOLD_FRAMES = 600
_FINAL_SCENE_SETTLE_FRAMES = 1200
_HARD_VALUE = 2
_FINAL_CREDITS_EVENT = 0x1A
# Only write HP when the turtle is one solid hit from death. The assist
# contract retains its fixed 80 restore after the route switches to Raphael
# (natural max 48); every intervention remains explicit in the manifest.
_EMERGENCY_HP_THRESHOLD = 16
_EMERGENCY_HP_RESTORE = 80

_STAGE_NAMES = {
    0: "BIG APPLE",
    1: "ALLEYCAT BLUES",
    2: "SEWER SURFIN'",
    3: "TECHNODROME",
    4: "PREHISTORIC",
    5: "SKULL & CROSSBONES",
    6: "WOUNDED KNEE",
    7: "NEON NIGHT RIDERS",
    8: "STARBASE",
    9: "FINAL SHELL SHOCK",
}

# Frame-accurate real-menu boot plan. The two DOWN presses enter Options,
# RIGHT changes Level to Hard, the two UP presses return to 1 Player, and the
# three RIGHT presses select Raphael and the last START confirms him.
_BOOT_ACTIONS: dict[int, tuple[str, ...]] = {
    300: ("START",),
    700: ("DOWN",),
    720: ("DOWN",),
    750: ("START",),
    950: ("RIGHT",),
    1000: ("START",),
    1200: ("UP",),
    1220: ("UP",),
    1250: ("START",),
    1440: ("RIGHT",),
    1441: ("RIGHT",),
    1442: ("RIGHT",),
    1443: ("RIGHT",),
    1444: ("RIGHT",),
    1452: ("RIGHT",),
    1453: ("RIGHT",),
    1454: ("RIGHT",),
    1455: ("RIGHT",),
    1456: ("RIGHT",),
    1464: ("RIGHT",),
    1465: ("RIGHT",),
    1466: ("RIGHT",),
    1467: ("RIGHT",),
    1468: ("RIGHT",),
    1490: ("START",),
}


@dataclass
class StageSplit:
    """First playable frame for one stage byte."""

    stage: int
    name: str
    frame: int
    elapsed_seconds: float


@dataclass
class RunMetrics:
    """Integrity and outcome metrics accumulated during a run."""

    total_damage_taken: int = 0
    max_single_frame_damage: int = 0
    health_guard_interventions: int = 0
    final_boss_iframe_guard_frames: int = 0
    life_losses: int = 0
    lives_start: int | None = None
    lives_peak: int | None = None
    lives_end: int | None = None
    min_health_seen: int | None = None
    credits_start_frame: int | None = None
    credits_complete_frame: int | None = None
    final_scene_start_frame: int | None = None
    hard_credits_event_seen: bool = False
    stage_splits: list[StageSplit] = field(default_factory=list)
    action_reasons: Counter[str] = field(default_factory=Counter)
    damage_by_stage: dict[int, int] = field(default_factory=dict)


class CreditsTracker:
    """Recognize the complete Hard staff/cast roll and final Splinter scene."""

    def __init__(self) -> None:
        self._last_playing = False
        self._stage9_playing_entries = 0

    def update(
        self,
        state: GameState,
        *,
        frame: int,
        metrics: RunMetrics,
    ) -> None:
        """Update credits evidence and completion frames."""
        event = int(state.extras.get("event", -1))
        menu = int(state.extras.get("menu", -1))
        if state.stage >= 10 and metrics.credits_start_frame is None:
            metrics.credits_start_frame = frame
        if metrics.credits_start_frame is None:
            return
        if event == _FINAL_CREDITS_EVENT:
            metrics.hard_credits_event_seen = True

        playing = (
            event == _FINAL_CREDITS_EVENT
            and state.stage == 9
            and menu == 6
            and state.player_x > 0
            and state.health > 0
        )
        if playing and not self._last_playing:
            self._stage9_playing_entries += 1
        self._last_playing = playing

        final_transition = (
            self._stage9_playing_entries >= 2
            and event == _FINAL_CREDITS_EVENT
            and state.stage == 9
            and state.player_x == 0
        )
        if final_transition and metrics.final_scene_start_frame is None:
            metrics.final_scene_start_frame = frame
        if (
            metrics.final_scene_start_frame is not None
            and metrics.credits_complete_frame is None
            and frame
            >= metrics.final_scene_start_frame + _FINAL_SCENE_SETTLE_FRAMES
        ):
            metrics.credits_complete_frame = frame


class NativeCapture:
    """Stream RGB video and native emulator PCM, then mux an MP4."""

    def __init__(
        self,
        output: Path,
        *,
        width: int,
        height: int,
        fps: float,
        audio_rate: int,
        scale: int,
    ) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg is required for video recording")
        self.output = output
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.width = width
        self.height = height
        self.fps = fps
        self.audio_rate = audio_rate
        self.scale = scale
        self.frames_written = 0
        self.audio_samples = 0
        self._silent = output.with_suffix(".partial.video.mp4")
        self._wav = output.with_suffix(".partial.audio.wav")
        self._video = subprocess.Popen(
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
                f"{width}x{height}",
                "-r",
                f"{fps:.12f}",
                "-i",
                "-",
                "-an",
                "-vf",
                f"scale={width * scale}:{height * scale}:flags=neighbor",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(self._silent),
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._audio = wave.open(str(self._wav), "wb")
        self._audio.setnchannels(2)
        self._audio.setsampwidth(2)
        self._audio.setframerate(audio_rate)

    def write(self, frame: np.ndarray, audio: np.ndarray | None) -> None:
        """Append one decorated RGB frame and the latest native PCM block."""
        if self._video.stdin is None:
            raise RuntimeError("ffmpeg video input is closed")
        rgb = np.asarray(frame, dtype=np.uint8)
        expected = (self.height, self.width, 3)
        if rgb.shape != expected:
            raise ValueError(f"expected frame {expected}, got {rgb.shape}")
        self._video.stdin.write(rgb.tobytes())
        self.frames_written += 1
        if audio is None:
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
        little = pcm.astype("<i2", copy=False)
        self._audio.writeframesraw(little.tobytes())
        self.audio_samples += int(pcm.shape[0])

    def _close_streams(self) -> None:
        """Finalize the elementary video and WAV streams."""
        self._audio.close()
        if self._video.stdin is not None:
            self._video.stdin.close()
        stderr = (
            self._video.stderr.read()
            if self._video.stderr is not None
            else b""
        )
        code = self._video.wait()
        if code:
            raise RuntimeError(stderr.decode("utf-8", errors="replace"))

    def finish(self) -> Path:
        """Mux the completed streams and remove the two partial files."""
        self._close_streams()
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
                str(self.output),
            ],
            capture_output=True,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
        self._silent.unlink(missing_ok=True)
        self._wav.unlink(missing_ok=True)
        return self.output

    def abort(self) -> None:
        """Close a failed capture without publishing a final MP4."""
        try:
            self._close_streams()
        except Exception:
            if self._video.poll() is None:
                self._video.kill()


def _reset(env: Any) -> tuple[np.ndarray, dict[str, Any]]:
    """Normalize Gymnasium and classic Retro reset results."""
    result = env.reset()
    if isinstance(result, tuple) and len(result) == 2:
        return np.asarray(result[0]), result[1]
    return np.asarray(result), {}


def _format_duration(seconds: float) -> str:
    """Format an elapsed duration as HH:MM:SS.mmm."""
    millis = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(millis, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def _boot_action(frame: int) -> list[int]:
    """Return the scheduled real-menu input for one power-on frame."""
    names = _BOOT_ACTIONS.get(frame)
    return buttons(*names) if names else idle_action()


def _short_clock(frame: int, fps: float) -> str:
    """Return MM:SS for the persistent footer."""
    seconds = int(frame / fps)
    minutes, secs = divmod(seconds, 60)
    return f"{minutes:02d}:{secs:02d}"


def _render_frame(
    obs: np.ndarray,
    *,
    state: GameState,
    frame: int,
    fps: float,
    metrics: RunMetrics,
    action: list[int] | None = None,
) -> np.ndarray:
    """Add a compact live footer and, at the end, an honest metric card."""
    rgb = np.asarray(obs, dtype=np.uint8)
    height, width = rgb.shape[:2]
    stage_name = _STAGE_NAMES.get(state.stage, "HARD CREDITS")
    stage_no = min(max(state.stage + 1, 1), 10)
    display_lives = (
        metrics.lives_end
        if metrics.credits_start_frame is not None
        else state.lives
    )
    assist = (
        f"E-HEAL {metrics.health_guard_interventions}"
        if metrics.health_guard_interventions
        else "NO HEAL"
    )
    image = render_footer_frame(
        rgb,
        upper_left=f"HARD  {stage_no:02d}/10 {stage_name}",
        upper_right=_short_clock(frame, fps),
        lower_left=(
            f"DMG {metrics.total_damage_taken:05d}  "
            f"LIVES {display_lives}  LOSS {metrics.life_losses}  {assist}"
        ),
        action=action,
        players=1,
    )

    complete = metrics.credits_complete_frame
    if complete is None or frame < complete:
        return image

    pil_image = Image.fromarray(image, mode="RGB")
    overlay = Image.new("RGBA", pil_image.size, (0, 0, 0, 0))
    card = ImageDraw.Draw(overlay)
    card.rounded_rectangle(
        (13, 35, width - 13, height - 20),
        radius=7,
        fill=(4, 10, 22, 232),
        outline=(82, 224, 168, 255),
        width=1,
    )
    title = ImageFont.load_default(size=13)
    body = ImageFont.load_default(size=9)
    final_seconds = complete / fps
    lines = [
        ("RUN COMPLETE - HARD CREDITS", title, (128, 255, 196, 255)),
        (
            f"POWER-ON TO CREDITS  {_format_duration(final_seconds)}",
            body,
            (242, 247, 255, 255),
        ),
        (
            f"DAMAGE TAKEN         {metrics.total_damage_taken}",
            body,
            (242, 247, 255, 255),
        ),
        ("LIFE LOSSES           0", body, (242, 247, 255, 255)),
        (
            "MIN HP SEEN          "
            f"{metrics.min_health_seen if metrics.min_health_seen is not None else '-'}",
            body,
            (194, 207, 222, 255),
        ),
        (
            "EMERGENCY HEALS      "
            f"{metrics.health_guard_interventions} (hp<={_EMERGENCY_HP_THRESHOLD})",
            body,
            (194, 207, 222, 255),
        ),
        (
            "F2 I-FRAME GUARD     "
            f"{metrics.final_boss_iframe_guard_frames}f",
            body,
            (194, 207, 222, 255),
        ),
        ("STATE LOADS 0  |  NO FULL-HP SPAM", body, (194, 207, 222, 255)),
        ("ONE EMULATOR SESSION + NATIVE AUDIO", body, (194, 207, 222, 255)),
    ]
    y = 47
    for text, current_font, color in lines:
        card.text((23, y), text, font=current_font, fill=color)
        y += 17 if current_font is title else 13
    return np.asarray(Image.alpha_composite(pil_image.convert("RGBA"), overlay).convert("RGB"))


def _rom_sha256() -> tuple[str, str]:
    """Return the local ROM filename and digest for reproducibility."""
    roms = sorted(path for path in ROMS_DIR.iterdir() if path.is_file())
    if len(roms) != 1:
        raise RuntimeError(f"expected one TMNT IV ROM, found {len(roms)}")
    path = roms[0]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return path.name, digest


def _file_sha256(path: Path) -> str:
    """Hash an artifact without retaining it in memory."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _probe_video(path: Path) -> dict[str, Any]:
    """Return ffprobe stream/container data for the finished MP4."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration,size:stream=index,codec_name,codec_type,width,height,avg_frame_rate,sample_rate,channels",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def _metrics_dict(metrics: RunMetrics, *, fps: float) -> dict[str, Any]:
    """Convert metrics to JSON-friendly values with readable timestamps."""
    payload = asdict(metrics)
    payload["action_reasons"] = dict(metrics.action_reasons.most_common())
    complete = metrics.credits_complete_frame
    if complete is not None:
        payload["power_on_to_credits_seconds"] = complete / fps
        payload["power_on_to_credits"] = _format_duration(complete / fps)
    start = metrics.credits_start_frame
    if start is not None:
        payload["credits_start_seconds"] = start / fps
    return payload


def assist_integrity(
    metrics: RunMetrics,
    *,
    require_clean_assists: bool = False,
) -> dict[str, bool]:
    """Boolean integrity flags for continuous assist counters."""
    flags = {
        "emergency_hp_zero": metrics.health_guard_interventions == 0,
        "iframe_guard_zero": metrics.final_boss_iframe_guard_frames == 0,
        "life_losses_zero": metrics.life_losses == 0,
        "state_loads_zero": True,
        "stage_writes_zero": True,
        "lives_writes_zero": True,
    }
    if require_clean_assists:
        flags["clean_assists_zero"] = (
            flags["emergency_hp_zero"] and flags["iframe_guard_zero"]
        )
    return flags


def evaluate_clean_integrity(metrics: RunMetrics) -> tuple[bool, dict[str, bool]]:
    """Return (ok, flags) requiring zero e-HP and zero iframe frames."""
    flags = assist_integrity(metrics, require_clean_assists=True)
    return bool(flags.get("clean_assists_zero")), flags


def run_full_hard(
    *,
    output: Path,
    report_path: Path,
    max_frames: int = 400_000,
    scale: int = 3,
    dry_run: bool = False,
    entry_state_prefix: str | None = None,
    emergency_hp: bool = True,
    iframe_hold: bool = True,
    require_clean_assists: bool | None = None,
) -> dict[str, Any]:
    """Run from power-on through complete Hard credits and record artifacts.

    Defaults keep both production assists on. Clean track passes
    ``emergency_hp=False``, ``iframe_hold=False``, and (implicitly)
    ``require_clean_assists=True`` so zero-assist integrity fails closed.
    """
    if require_clean_assists is None:
        require_clean_assists = not emergency_hp and not iframe_hold
    clean_mode = not emergency_hp and not iframe_hold

    configure_headless()
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    policy = Stage1Policy()
    metrics = RunMetrics()
    credits = CreditsTracker()
    capture: NativeCapture | None = None
    succeeded = False
    obs, _info = _reset(env)
    fps = float(env.em.get_screen_rate())
    audio_rate = int(env.em.get_audio_rate())
    height, width = obs.shape[:2]
    pending_audio: np.ndarray | None = None
    started = False
    previous_lives: int | None = None
    previous_health: int | None = None
    last_stage = -1
    split_stages: set[int] = set()
    hard_confirmed = False
    frame = 0
    final_state = parse_game_state(env.get_ram(), frame=0)

    try:
        if not dry_run:
            capture = NativeCapture(
                output,
                width=width,
                height=height + _FOOTER_HEIGHT,
                fps=fps,
                audio_rate=audio_rate,
                scale=scale,
            )

        for frame in range(0, max_frames + 1):
            state = parse_game_state(env.get_ram(), frame=frame)
            final_state = state
            menu = int(state.extras.get("menu", -1))
            event = int(state.extras.get("event", -1))
            active = (
                metrics.credits_start_frame is None
                and menu == 6
                and state.player_x > 0
                and state.stage <= 9
            )

            # Track natural damage from HP drops. Emergency heal only when
            # about to die — never the old every-hit restore-to-96.
            if (
                active
                and previous_health is not None
                and 0 <= state.health <= 0x60
                and previous_health <= 0x60
                and state.health < previous_health
            ):
                damage = previous_health - max(0, state.health)
                metrics.total_damage_taken += damage
                metrics.max_single_frame_damage = max(
                    metrics.max_single_frame_damage, damage
                )
                metrics.damage_by_stage[state.stage] = (
                    metrics.damage_by_stage.get(state.stage, 0) + damage
                )
            if active and 0 < state.health <= 0x60:
                if (
                    metrics.min_health_seen is None
                    or state.health < metrics.min_health_seen
                ):
                    metrics.min_health_seen = state.health
                if emergency_hp and state.health <= _EMERGENCY_HP_THRESHOLD:
                    env.set_value("player_hp", _EMERGENCY_HP_RESTORE)
                    metrics.health_guard_interventions += 1
                    state = parse_game_state(env.get_ram(), frame=frame)
                    final_state = state
                    previous_health = state.health
                else:
                    previous_health = state.health
            elif active and state.health == 0:
                # Last-chance revive before the life counter ticks (assisted).
                if previous_health is not None and previous_health > 0:
                    damage = previous_health
                    metrics.total_damage_taken += damage
                    metrics.max_single_frame_damage = max(
                        metrics.max_single_frame_damage, damage
                    )
                    metrics.damage_by_stage[state.stage] = (
                        metrics.damage_by_stage.get(state.stage, 0) + damage
                    )
                if emergency_hp:
                    env.set_value("player_hp", _EMERGENCY_HP_RESTORE)
                    metrics.health_guard_interventions += 1
                    state = parse_game_state(env.get_ram(), frame=frame)
                    final_state = state
                    previous_health = state.health
                else:
                    previous_health = 0
            elif not active:
                previous_health = None

            # Form-2 demutation bypasses HP; hold a 1-frame iframe while the
            # finale arena is live. Still far cheaper than the old full-bar spam.
            if (
                iframe_hold
                and active
                and state.stage == 9
                and event == 0x0A
            ):
                env.set_value("player_iframes", 1)
                metrics.final_boss_iframe_guard_frames += 1

            if active and not started:
                started = True
                previous_lives = state.lives
                metrics.lives_start = state.lives
                metrics.lives_peak = state.lives
                if 0 < state.health <= 0x60:
                    previous_health = state.health
                    metrics.min_health_seen = state.health

            if started and metrics.credits_start_frame is None:
                difficulty = int(env.get_ram()[0x1FEE])
                if difficulty == _HARD_VALUE:
                    hard_confirmed = True
                elif frame > 2500:
                    raise RuntimeError(
                        f"difficulty changed from Hard: {difficulty}"
                    )
                if menu == 0:
                    raise RuntimeError(
                        f"unexpected return to title at frame {frame}"
                    )
                if state.stage < last_stage:
                    raise RuntimeError(
                        f"stage regressed {last_stage}->{state.stage} "
                        f"at frame {frame}"
                    )
                last_stage = max(last_stage, state.stage)

            if (
                started
                and previous_lives is not None
                and metrics.credits_start_frame is None
            ):
                if state.lives < previous_lives:
                    metrics.life_losses += previous_lives - state.lives
                    raise RuntimeError(
                        f"life loss at frame {frame}: "
                        f"{previous_lives}->{state.lives} "
                        f"stage={state.stage} dmg={metrics.total_damage_taken}"
                    )
                previous_lives = state.lives
                metrics.lives_peak = max(
                    metrics.lives_peak or state.lives, state.lives
                )
                metrics.lives_end = state.lives

            if active and state.stage not in split_stages:
                split_stages.add(state.stage)
                # Stateful combat phases must not leak across natural stage
                # transitions. This also makes each continuous stage match its
                # independently verified checkpoint probe.
                policy.reset()
                previous_health = (
                    state.health if 0 < state.health <= 0x60 else None
                )
                metrics.stage_splits.append(
                    StageSplit(
                        stage=state.stage,
                        name=_STAGE_NAMES.get(state.stage, "UNKNOWN"),
                        frame=frame,
                        elapsed_seconds=frame / fps,
                    )
                )
                if entry_state_prefix:
                    save_state(
                        env,
                        GAME_DIR,
                        GAME,
                        f"{entry_state_prefix}Stage{state.stage + 1}",
                    )
                print(
                    f"stage {state.stage + 1:02d} "
                    f"{_STAGE_NAMES.get(state.stage, 'UNKNOWN')} "
                    f"at {_format_duration(frame / fps)} "
                    f"dmg={metrics.total_damage_taken}",
                    flush=True,
                )

            credits.update(state, frame=frame, metrics=metrics)

            if frame <= max(_BOOT_ACTIONS):
                action = _boot_action(frame)
                reason = "boot_menu" if any(action) else "boot_idle"
            elif metrics.credits_start_frame is not None:
                action = idle_action()
                reason = "credits_idle"
            elif started and (
                state.player_x == 0
                or state.mode in {GameMode.CUTSCENE, GameMode.CONTINUE}
            ):
                # Stage loads briefly look like CONTINUE because HP/X are zero.
                # Do not press START after gameplay begins.
                action = idle_action()
                reason = "transition_idle"
            else:
                tick = policy.tick(state)
                if tick.action is None:
                    action = idle_action()
                    reason = tick.reason or "policy_idle"
                else:
                    action = tick.action.action
                    reason = tick.action.reason
            if frame > max(_BOOT_ACTIONS) and action[SNES_START]:
                action = idle_action()
                reason = "suppressed_start"
            if action[8]:
                raise RuntimeError(f"forbidden A special at frame {frame}")
            metrics.action_reasons[reason] += 1

            if capture is not None:
                decorated = _render_frame(
                    obs,
                    state=state,
                    frame=frame,
                    fps=fps,
                    metrics=metrics,
                    action=action,
                )
                capture.write(decorated, pending_audio)

            complete = metrics.credits_complete_frame
            if complete is not None and frame >= complete + _METRIC_HOLD_FRAMES:
                succeeded = True
                break

            if frame and frame % 10_000 == 0:
                targets = [
                    (hex(enemy.kind), enemy.health, enemy.x, enemy.y)
                    for enemy in state.living_enemies
                ]
                print(
                    f"frame {frame}  stage={state.stage} event={event:#04x} "
                    f"damage={metrics.total_damage_taken} lives={state.lives} "
                    f"p=({state.player_x},{state.player_y}) "
                    f"targets={targets}",
                    flush=True,
                )

            obs, _reward, _terminated, _truncated, _step_info = env.step(
                action
            )
            pending_audio = np.asarray(env.em.get_audio(), dtype=np.int16)
        else:
            raise RuntimeError(f"run exceeded {max_frames} frames")

        if not hard_confirmed:
            raise RuntimeError("Hard difficulty was never confirmed in WRAM")
        if not metrics.hard_credits_event_seen:
            raise RuntimeError("Hard staff/cast credits event was not observed")
        if metrics.credits_complete_frame is None:
            raise RuntimeError("final Splinter credits scene did not complete")
        if metrics.life_losses:
            raise RuntimeError(f"run had {metrics.life_losses} life losses")

        video_path: Path | None = None
        if capture is not None:
            video_path = capture.finish()
            capture = None

        integrity_flags = assist_integrity(
            metrics, require_clean_assists=require_clean_assists
        )
        clean_ok = (not require_clean_assists) or bool(
            integrity_flags.get("clean_assists_zero", False)
        )
        if require_clean_assists and not clean_ok:
            raise RuntimeError(
                "clean integrity failed: "
                f"e-heals={metrics.health_guard_interventions} "
                f"iframe_frames={metrics.final_boss_iframe_guard_frames}"
            )

        if clean_mode:
            intervention_class = "Clean"
        elif emergency_hp and iframe_hold:
            intervention_class = "Resource-assisted + Protection-assisted"
        elif emergency_hp:
            intervention_class = "Resource-assisted"
        else:
            intervention_class = "Protection-assisted"

        rom_name, rom_digest = _rom_sha256()
        command = "uv run python -m tmnt_iv.scripts.record_full_hard_run"
        if clean_mode:
            command += " --clean"
        if dry_run:
            command += " --dry-run"
        report: dict[str, Any] = {
            "schema_version": 1,
            "status": "success",
            "created_at": datetime.now().astimezone().isoformat(),
            "game": GAME,
            "run": {
                "difficulty": "HARD",
                "difficulty_wram_value": _HARD_VALUE,
                "continuous_emulator_session": True,
                "power_on_start": True,
                "start_state": "NONE",
                "save_state_loads": 0,
                "stage_writes": 0,
                "lives_writes": 0,
                "native_audio": not dry_run,
                "assisted": not clean_mode,
                "intervention_class": intervention_class,
                "clean_track": clean_mode,
                "assists": {
                    "health_restore_to_96": False,
                    "emergency_hp_enabled": emergency_hp,
                    "iframe_hold_enabled": iframe_hold,
                    "emergency_hp_threshold": _EMERGENCY_HP_THRESHOLD,
                    "emergency_hp_restore": _EMERGENCY_HP_RESTORE,
                    "super_shredder_form2_iframe_value": 1,
                    "require_clean_assists": require_clean_assists,
                },
                "forbidden_a_special_uses": 0,
                "post_boot_start_presses": 0,
            },
            "metrics": _metrics_dict(metrics, fps=fps),
            "integrity": integrity_flags,
            "emulator": {
                "screen_rate": fps,
                "audio_rate": audio_rate,
                "native_width": width,
                "native_height": height,
                "frames_executed": frame,
            },
            "reproducibility": {
                "rom_filename": rom_name,
                "rom_sha256": rom_digest,
                "command": command,
            },
            "final_state": {
                "frame": frame,
                "stage": final_state.stage,
                "event": int(final_state.extras.get("event", -1)),
                "menu": int(final_state.extras.get("menu", -1)),
                "lives": final_state.lives,
            },
            "artifact": None,
        }
        if video_path is not None and capture is None:
            report["artifact"] = {
                "path": str(video_path),
                "sha256": _file_sha256(video_path),
                "ffprobe": _probe_video(video_path),
            }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            "complete: "
            f"{_format_duration(metrics.credits_complete_frame / fps)}  "
            f"damage={metrics.total_damage_taken}  "
            f"life_losses={metrics.life_losses}  "
            f"e-heals={metrics.health_guard_interventions}  "
            f"iframe={metrics.final_boss_iframe_guard_frames}  "
            f"class={intervention_class}",
            flush=True,
        )
        return report
    finally:
        if capture is not None:
            capture.abort()
        env.close()
        if not succeeded and not dry_run:
            print("capture did not reach a verified completion", flush=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Video path (default: recordings/tmnt_iv_full_hard_credits.mp4; "
            "with --clean: .../tmnt_iv_full_hard_clean.mp4)"
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help=(
            "JSON report path (default assisted credits/dry_run stems; "
            "with --clean: tmnt_iv_full_hard_clean[_dry_run].json)"
        ),
    )
    parser.add_argument("--max-frames", type=int, default=400_000)
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run all integrity checks without encoding video/audio",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help=(
            "Clean track: disable emergency HP and form-2 iframe hold, "
            "use *_clean default artifact stems, and require zero assist "
            "counters. Does not change assisted defaults when omitted."
        ),
    )
    parser.add_argument(
        "--no-emergency-hp",
        action="store_true",
        help="Disable emergency HP restore only (not full Clean alone)",
    )
    parser.add_argument(
        "--no-iframe-hold",
        action="store_true",
        help="Disable form-2 iframe hold only (not full Clean alone)",
    )
    parser.add_argument(
        "--entry-state-prefix",
        default=None,
        help=(
            "save development checkpoints at natural stage entries "
            "(for example LiveHard -> LiveHardStage5)"
        ),
    )
    return parser


def resolve_cli_paths(
    *,
    output: Path | None,
    report: Path | None,
    dry_run: bool,
    clean_artifacts: bool,
) -> tuple[Path, Path]:
    """Resolve video/report paths; explicit CLI paths always win."""
    default_video, default_report = default_full_run_paths(
        clean=clean_artifacts, dry_run=dry_run
    )
    return (
        output if output is not None else default_video,
        report if report is not None else default_report,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _build_parser().parse_args(argv)
    emergency_hp = not (args.clean or args.no_emergency_hp)
    iframe_hold = not (args.clean or args.no_iframe_hold)
    # Full Clean: both assists off.
    clean = not emergency_hp and not iframe_hold
    # Any assist-off run uses *_clean stems so assisted baselines stay safe.
    clean_artifacts = not emergency_hp or not iframe_hold

    output, report = resolve_cli_paths(
        output=args.output,
        report=args.report,
        dry_run=args.dry_run,
        clean_artifacts=clean_artifacts,
    )
    run_full_hard(
        output=output,
        report_path=report,
        max_frames=args.max_frames,
        scale=args.scale,
        dry_run=args.dry_run,
        entry_state_prefix=args.entry_state_prefix,
        emergency_hp=emergency_hp,
        iframe_hold=iframe_hold,
        require_clean_assists=clean,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
