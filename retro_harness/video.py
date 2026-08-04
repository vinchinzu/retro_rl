"""High-level emulator video capture shared across games.

One recorder surface for continuous / showcase MP4s:

- nearest-neighbor scale, x264 quality knobs (``crf`` / ``preset`` / ``fps``)
- optional stereo s16le audio mux from ``env.em.get_audio()``
- optional bottom footer with frame clock + pressed buttons
- optional start gate (skip until frame N, or until a room id latches)

Game packages should only supply presets (paths, room ids, default cutoffs),
not their own ffmpeg pipes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from typing import Any, BinaryIO, Protocol, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from retro_harness.controls import pressed_nes_buttons, pressed_snes_buttons

FOOTER_HEIGHT = 16


@dataclass(frozen=True)
class VideoCaptureConfig:
    """Capture settings (encode quality, audio, footer, start gate).

    Start gate is game-agnostic:

    - ``start_room_id``: write nothing until ``room_id`` matches, then latch
    - ``start_frame``: write nothing until ``frame >= start_frame``
    - both unset: write immediately

    When both are set, room latch wins (frame gate is ignored after room opens).
    """

    fps: int = 60
    scale: int = 2
    crf: int = 17
    preset: str = "medium"
    audio: bool = True
    audio_bitrate: str = "192k"
    footer: bool = True
    start_frame: int | None = None
    start_room_id: int | None = None

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.scale < 1:
            raise ValueError("scale must be >= 1")
        if not 0 <= self.crf <= 51:
            raise ValueError("crf must be in 0..51")
        if self.start_frame is not None and self.start_frame < 0:
            raise ValueError("start_frame must be >= 0")

    @classmethod
    def high_quality(cls, **overrides: Any) -> VideoCaptureConfig:
        """Slower encode + lower CRF + 3× nearest-neighbor upscale."""
        base: dict[str, Any] = dict(
            fps=60,
            scale=3,
            crf=15,
            preset="slow",
            audio=True,
            footer=True,
        )
        base.update(overrides)
        return cls(**base)

    def to_dict(self) -> dict[str, object]:
        return {
            "fps": self.fps,
            "scale": self.scale,
            "crf": self.crf,
            "preset": self.preset,
            "audio": self.audio,
            "audio_bitrate": self.audio_bitrate,
            "footer": self.footer,
            "start_frame": self.start_frame,
            "start_room_id": self.start_room_id,
        }


def should_capture_frame(
    *,
    frame: int,
    room_id: int | None,
    config: VideoCaptureConfig,
    recording_started: bool,
) -> tuple[bool, bool]:
    """Return ``(write_this_frame, recording_started_after)``.

    ``recording_started`` latches once the gate opens so room-based starts
    keep capturing after leaving the trigger room.
    """
    if recording_started:
        return True, True

    if config.start_room_id is not None:
        if room_id is not None and room_id == config.start_room_id:
            return True, True
        return False, False

    if config.start_frame is not None:
        if frame >= config.start_frame:
            return True, True
        return False, False

    return True, True


def format_snes_buttons(action: np.ndarray | Sequence[int] | None) -> str:
    """Compact pressed-button label (e.g. ``A+B+RIGHT``)."""
    if action is None:
        return "---"
    names = pressed_snes_buttons([int(v) for v in action])
    if not names:
        return "---"
    order = (
        "UP",
        "DOWN",
        "LEFT",
        "RIGHT",
        "A",
        "B",
        "X",
        "Y",
        "L",
        "R",
        "START",
        "SELECT",
    )
    ordered = [name for name in order if name in names]
    return "+".join(ordered) if ordered else "---"


def short_clock(frame: int, fps: float) -> str:
    """Return ``MM:SS`` for a live footer clock."""
    seconds = int(frame / fps) if fps > 0 else 0
    minutes, secs = divmod(seconds, 60)
    return f"{minutes:02d}:{secs:02d}"


def render_button_footer(
    obs: np.ndarray,
    *,
    action: np.ndarray | Sequence[int] | None,
    frame: int,
    fps: int,
    footer_bg: tuple[int, int, int] = (5, 10, 18),
    clock_color: tuple[int, int, int] = (103, 232, 164),
    frame_color: tuple[int, int, int] = (150, 170, 190),
    button_color: tuple[int, int, int] = (255, 214, 102),
) -> np.ndarray:
    """Extend one RGB frame with a bottom bar: frame · clock · buttons."""
    rgb = np.asarray(obs, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB frame, got {rgb.shape}")
    height, width = rgb.shape[:2]
    canvas = np.zeros((height + FOOTER_HEIGHT, width, 3), dtype=np.uint8)
    canvas[:height] = rgb
    canvas[height:] = footer_bg
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=8)

    left = f"F{frame:06d}"
    clock = short_clock(frame, float(fps))
    buttons = format_snes_buttons(action)

    draw.text((4, height + 4), left, fill=frame_color, font=font)
    left_w = draw.textbbox((0, 0), left, font=font)[2]
    draw.text((4 + left_w + 8, height + 4), clock, fill=clock_color, font=font)
    button_w = draw.textbbox((0, 0), buttons, font=font)[2]
    draw.text(
        (width - button_w - 4, height + 4),
        buttons,
        fill=button_color,
        font=font,
    )
    return np.asarray(image)


def concat_videos(
    parts: Sequence[str | Path],
    output: str | Path,
    *,
    reencode: bool = False,
    crf: int = 17,
    preset: str = "medium",
) -> dict[str, object]:
    """Concatenate H.264 clips with ffmpeg (stream-copy by default)."""
    paths = [Path(p).resolve() for p in parts]
    if not paths:
        raise ValueError("concat_videos requires at least one part")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"missing video part: {path}")
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".txt",
        delete=False,
        encoding="utf-8",
    ) as list_file:
        for path in paths:
            escaped = str(path).replace("'", r"'\''")
            list_file.write(f"file '{escaped}'\n")
        list_path = Path(list_file.name)

    try:
        command = [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
        ]
        if reencode:
            command.extend(
                [
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    preset,
                    "-crf",
                    str(crf),
                    "-pix_fmt",
                    "yuv420p",
                ]
            )
        else:
            command.extend(["-c", "copy"])
        command.append(str(out))
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            raise RuntimeError(
                f"ffmpeg concat failed ({result.returncode}): {stderr or 'no stderr'}"
            )
    finally:
        list_path.unlink(missing_ok=True)

    return {
        "path": str(out),
        "parts": [str(p) for p in paths],
        "reencode": reencode,
        "bytes": out.stat().st_size if out.is_file() else 0,
    }


class FrameVideoWriter:
    """Low-level ffmpeg RGB pipe with optional audio mux and button footer."""

    def __init__(
        self,
        path: str | Path,
        *,
        width: int,
        height: int,
        fps: int = 60,
        scale: int = 2,
        crf: int = 17,
        preset: str = "medium",
        audio_rate: int | float | None = None,
        audio_bitrate: str = "192k",
        footer: bool = False,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("frame dimensions must be positive")
        if fps <= 0:
            raise ValueError("fps must be positive")
        if scale < 1:
            raise ValueError("scale must be >= 1")
        if audio_rate is not None and audio_rate <= 0:
            raise ValueError("audio_rate must be positive")

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg is required to record video")

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.width = width
        self.height = height
        self.fps = fps
        self.scale = scale
        self.crf = crf
        self.preset = preset
        self.footer = footer
        self.audio_rate = int(audio_rate) if audio_rate is not None else None
        self.audio_bitrate = audio_bitrate
        self.audio_bytes_written = 0
        self.frames = 0
        self.frames_written = 0  # alias used by showcase / golf callers
        self._src_height = height + FOOTER_HEIGHT if footer else height
        self._video_path = self.path
        self._audio_handle = None
        self._audio_path: Path | None = None

        if self.audio_rate is not None:
            video_temp = tempfile.NamedTemporaryFile(
                dir=self.path.parent,
                prefix=f".{self.path.stem}.",
                suffix=f".video{self.path.suffix}",
                delete=False,
            )
            video_temp.close()
            self._video_path = Path(video_temp.name)
            audio_temp = tempfile.NamedTemporaryFile(
                dir=self.path.parent,
                prefix=f".{self.path.stem}.",
                suffix=".s16le",
                delete=False,
            )
            self._audio_handle = audio_temp
            self._audio_path = Path(audio_temp.name)

        command = [
            ffmpeg,
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{width}x{self._src_height}",
            "-framerate",
            str(fps),
            "-i",
            "-",
            "-vf",
            f"scale=iw*{scale}:ih*{scale}:flags=neighbor",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(self._video_path),
        ]
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if self._process.stdin is None:
            raise RuntimeError("ffmpeg did not expose stdin")
        self._stdin: BinaryIO = self._process.stdin

    def write(
        self,
        frame: np.ndarray,
        *,
        audio: Any = None,
        action: np.ndarray | Sequence[int] | None = None,
        frame_index: int | None = None,
    ) -> None:
        """Append one HxWx3 RGB frame (optional stereo s16le audio + footer)."""
        rgb = np.asarray(frame, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"expected HxWx3 RGB frame, got {rgb.shape}")
        if rgb.shape[0] != self.height or rgb.shape[1] != self.width:
            raise ValueError(
                f"frame size {rgb.shape[1]}x{rgb.shape[0]} != "
                f"{self.width}x{self.height}"
            )
        if self.footer:
            idx = self.frames if frame_index is None else frame_index
            rgb = render_button_footer(
                rgb,
                action=action,
                frame=idx,
                fps=self.fps,
            )
        if audio is not None:
            if self._audio_handle is None:
                raise RuntimeError("audio supplied without an audio_rate")
            audio_bytes = np.ascontiguousarray(audio, dtype=np.int16).tobytes()
            self._audio_handle.write(audio_bytes)
            self.audio_bytes_written += len(audio_bytes)
        self._stdin.write(np.ascontiguousarray(rgb, dtype=np.uint8).tobytes())
        self.frames += 1
        self.frames_written = self.frames

    def close(self) -> Path:
        """Finalize the MP4 (mux audio if present) and return the path."""
        if self._stdin.closed:
            return self.path
        self._stdin.close()
        stderr = b""
        if self._process.stderr is not None:
            stderr = self._process.stderr.read()
        result = self._process.wait()
        if result:
            detail = stderr.decode("utf-8", errors="replace").strip()
            self._cleanup_temporary_files()
            raise RuntimeError(
                f"ffmpeg exited with status {result}"
                + (f": {detail}" if detail else "")
            )

        if self._audio_handle is not None:
            self._audio_handle.close()
            self._audio_handle = None
            try:
                if self.audio_bytes_written:
                    self._mux_audio()
                else:
                    shutil.move(str(self._video_path), str(self.path))
            finally:
                self._cleanup_temporary_files()
        return self.path

    def _mux_audio(self) -> None:
        assert self.audio_rate is not None
        assert self._audio_path is not None
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(self._video_path),
                "-f",
                "s16le",
                "-ar",
                str(self.audio_rate),
                "-ac",
                "2",
                "-i",
                str(self._audio_path),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                self.audio_bitrate,
                "-shortest",
                "-movflags",
                "+faststart",
                str(self.path),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            detail = result.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"ffmpeg failed ({result.returncode}) muxing {self.path}: {detail}"
            )

    def _cleanup_temporary_files(self) -> None:
        for path in (self._video_path, self._audio_path):
            if path is not None and path != self.path:
                path.unlink(missing_ok=True)

    def __enter__(self) -> FrameVideoWriter:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


class VideoRecorder:
    """High-level continuous/showcase recorder used by game runtimes.

    Wraps :class:`FrameVideoWriter` with a start gate and ``env.em`` audio pull.
    Prefer this over constructing a bare writer in game code.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        width: int,
        height: int,
        config: VideoCaptureConfig | None = None,
        audio_rate: int | float | None = None,
    ) -> None:
        self.config = config or VideoCaptureConfig()
        self._recording_started = (
            self.config.start_room_id is None and self.config.start_frame is None
        )
        resolved_rate = audio_rate if self.config.audio else None
        if self.config.audio and resolved_rate is None:
            # Caller may fill audio per-frame without a pre-known rate only if
            # they pass audio_rate explicitly; require it when audio is on.
            raise ValueError("audio_rate is required when config.audio is True")
        self._writer = FrameVideoWriter(
            path,
            width=width,
            height=height,
            fps=self.config.fps,
            scale=self.config.scale,
            crf=self.config.crf,
            preset=self.config.preset,
            audio_rate=resolved_rate if self.config.audio else None,
            audio_bitrate=self.config.audio_bitrate,
            footer=self.config.footer,
        )

    @property
    def path(self) -> Path:
        return self._writer.path

    @property
    def frames(self) -> int:
        return self._writer.frames

    @property
    def frames_written(self) -> int:
        return self._writer.frames_written

    @property
    def audio_bytes_written(self) -> int:
        return self._writer.audio_bytes_written

    def write(
        self,
        frame: np.ndarray,
        *,
        action: np.ndarray | Sequence[int] | None = None,
        audio: Any = None,
        frame_index: int | None = None,
        room_id: int | None = None,
    ) -> bool:
        """Write one frame if the start gate is open. Returns whether written."""
        idx = self.frames if frame_index is None else frame_index
        write, self._recording_started = should_capture_frame(
            frame=idx,
            room_id=room_id,
            config=self.config,
            recording_started=self._recording_started,
        )
        if not write:
            return False
        self._writer.write(
            frame,
            audio=audio,
            action=action,
            frame_index=idx,
        )
        return True

    def write_from_env(
        self,
        env: object,
        frame: np.ndarray,
        *,
        action: np.ndarray | Sequence[int] | None = None,
        frame_index: int | None = None,
        room_id: int | None = None,
    ) -> bool:
        """Write one frame, pulling stereo audio from ``env.em`` when enabled."""
        audio = None
        if self.config.audio:
            em = getattr(env, "em", None)
            if em is not None and hasattr(em, "get_audio"):
                audio = em.get_audio()
        return self.write(
            frame,
            action=action,
            audio=audio,
            frame_index=frame_index,
            room_id=room_id,
        )

    def close(self) -> Path:
        return self._writer.close()

    def __enter__(self) -> VideoRecorder:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


def probe_video_evidence(path: Path, expected_frames: int) -> dict[str, object]:
    """ffprobe a finished MP4 and compare frame count (video + optional audio)."""
    import hashlib
    import json

    command = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-show_entries",
        "stream=index,codec_type,codec_name,width,height,r_frame_rate,"
        "nb_read_frames,duration,sample_rate,channels",
        "-of",
        "json",
        str(path),
    ]
    payload = json.loads(subprocess.check_output(command, text=True))
    streams = payload.get("streams") or []
    video_stream = next(s for s in streams if s.get("codec_type") == "video")
    audio_stream = next(
        (s for s in streams if s.get("codec_type") == "audio"),
        None,
    )
    actual_frames = int(video_stream["nb_read_frames"])
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    evidence: dict[str, object] = {
        "path": str(path.resolve()),
        "sha256": digest.hexdigest(),
        "codec": video_stream["codec_name"],
        "width": int(video_stream["width"]),
        "height": int(video_stream["height"]),
        "frame_rate": video_stream["r_frame_rate"],
        "duration_seconds": float(video_stream.get("duration") or 0.0),
        "frames": actual_frames,
        "expected_frames": expected_frames,
        "frame_count_matches": actual_frames == expected_frames,
        "has_audio": audio_stream is not None,
    }
    if audio_stream is not None:
        evidence["audio"] = {
            "codec": audio_stream.get("codec_name"),
            "sample_rate": int(audio_stream.get("sample_rate") or 0),
            "channels": int(audio_stream.get("channels") or 0),
        }
    return evidence


# -- Multi-player / NES showcase footer (from retro_harness.recording_footer) ----

SNES_PLAYER_STRIDE = 12
NES_PLAYER_STRIDE = 9


def frame_timestamp(frame: int, fps: float) -> str:
    """Return ``F#####  MM:SS.cc`` (centiseconds) for recording overlays."""
    total_cs = int(round(frame * 100 / fps)) if fps > 0 else 0
    minutes, rem = divmod(total_cs, 60 * 100)
    secs, cs = divmod(rem, 100)
    return f"F{frame:05d}  {minutes:02d}:{secs:02d}.{cs:02d}"


def format_player_buttons(
    action: list[int] | None,
    *,
    players: int = 1,
    stride: int = SNES_PLAYER_STRIDE,
    layout: str = "snes",
) -> str:
    """Render pressed buttons for one or more players.

    ``layout`` is ``"snes"`` (default) or ``"nes"`` (9-button fceumm).
    """
    if action is None:
        return "P1:---"
    if layout == "nes":
        stride = NES_PLAYER_STRIDE
        press_fn = pressed_nes_buttons
    else:
        press_fn = pressed_snes_buttons
    parts: list[str] = []
    for player in range(players):
        start = player * stride
        end = start + stride
        if end > len(action) and start >= len(action):
            break
        chunk = action[start:end]
        names = press_fn(chunk)
        label = "+".join(sorted(names)) if names else "---"
        parts.append(f"P{player + 1}:{label}")
    return "  ".join(parts) if parts else "P1:---"


def render_footer_frame(
    obs: np.ndarray,
    *,
    upper_left: str,
    upper_right: str,
    lower_left: str,
    action: list[int] | None = None,
    players: int = 1,
    layout: str = "snes",
    footer_bg: tuple[int, int, int] = (5, 10, 18),
    upper_left_color: tuple[int, int, int] = (219, 234, 246),
    upper_right_color: tuple[int, int, int] = (103, 232, 164),
    lower_left_color: tuple[int, int, int] = (150, 170, 190),
    button_color: tuple[int, int, int] = (255, 214, 102),
) -> np.ndarray:
    """Extend the frame with a multi-line footer and live button labels."""
    rgb = np.asarray(obs, dtype=np.uint8)
    height, width = rgb.shape[:2]
    canvas = np.zeros((height + FOOTER_HEIGHT, width, 3), dtype=np.uint8)
    canvas[:height] = rgb
    canvas[height:] = footer_bg
    image = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=8)

    draw.text((4, height), upper_left, fill=upper_left_color, font=font)
    clock_width = draw.textbbox((0, 0), upper_right, font=font)[2]
    draw.text(
        (width - clock_width - 4, height),
        upper_right,
        fill=upper_right_color,
        font=font,
    )

    draw.text((4, height + 8), lower_left, fill=lower_left_color, font=font)
    buttons = format_player_buttons(action, players=players, layout=layout)
    button_width = draw.textbbox((0, 0), buttons, font=font)[2]
    draw.text(
        (width - button_width - 4, height + 8),
        buttons,
        fill=button_color,
        font=font,
    )
    return np.asarray(image)


class FrameSink(Protocol):
    """Accept decorated RGB frames during a replay."""

    def write(self, frame: np.ndarray) -> None: ...


@dataclass(frozen=True)
class FooterLabels:
    """Three text fields rendered on the live recording banner."""

    upper_left: str
    upper_right: str
    lower_left: str


FooterProvider = Callable[[object, list[int], int, float], FooterLabels]


class RecordingSession:
    """Wrap env stepping with footer decoration and optional frame stride.

    Distinct from :class:`retro_harness.recorder.RecordingSession` (labeled
    human save-state recording). This type drives showcase/continuous capture.
    """

    def __init__(
        self,
        env: object,
        *,
        sink: FrameSink,
        footer: FooterProvider,
        fps: float,
        frame_stride: int = 1,
        players: int = 1,
        idle_action: Callable[[], list[int]] | None = None,
    ) -> None:
        self._env = env
        self._sink = sink
        self._footer = footer
        self._fps = fps
        self._frame_stride = max(1, frame_stride)
        self._players = players
        self._idle_action = idle_action
        self.frame = 0

    def capture(self, obs: np.ndarray, action: list[int]) -> None:
        """Decorate one emulator frame and optionally write it to the sink."""
        if self.frame % self._frame_stride != 0:
            return
        labels = self._footer(self._env, action, self.frame, self._fps)
        decorated = render_footer_frame(
            obs,
            upper_left=labels.upper_left,
            upper_right=labels.upper_right,
            lower_left=labels.lower_left,
            action=action,
            players=self._players,
        )
        self._sink.write(decorated)

    def step(self, action: list[int]) -> np.ndarray:
        """Step the emulator and capture the resulting frame."""
        obs, *_rest = self._env.step(action)  # type: ignore[attr-defined]
        rgb = np.asarray(obs)
        self.capture(rgb, action)
        self.frame += 1
        return rgb

    def _default_idle(self) -> list[int]:
        if self._idle_action is not None:
            return self._idle_action()
        from retro_harness.actions import idle_action

        return idle_action()

    def idle(self, frames: int) -> np.ndarray:
        """Hold idle input for several frames."""
        obs = np.zeros((1, 1, 3), dtype=np.uint8)
        for _ in range(max(frames, 1)):
            obs = self.step(self._default_idle())
        return obs

    def hold(self, action: list[int], frames: int) -> np.ndarray:
        """Repeat one action for several frames."""
        obs = np.zeros((1, 1, 3), dtype=np.uint8)
        for _ in range(max(frames, 1)):
            obs = self.step(action)
        return obs

