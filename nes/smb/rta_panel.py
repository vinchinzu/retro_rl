"""In-video RTA split panel + HUD capture for SMB recordings.

Tracks exit-detect cum times (first frame of each post-exit stage, then axe)
and freezes both the panel and the speedrun clock on ``oper_mode=2`` (Axe).

Labels match the warps route: 1-1, 1-2, 4-1, 4-2, 8-1…8-4 / AXE.
``VideoWriter`` is the shared ffmpeg pipe (warp finish, HappyLee, warpless).
"""

from __future__ import annotations

import shutil
import subprocess
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from retro_harness.video import FOOTER_HEIGHT, frame_timestamp, render_footer_frame
from retro_harness.youtube_intro import DEFAULT_INTRO_FRAMES, render_intro_card
from smb.ram import read_snapshot
from smb.timing import NTSC_FPS, format_time_mmss

# Zero-indexed (world, level) → short HUD label for warps any%.
_WARP_STAGE_LABELS: dict[tuple[int, int], str] = {
    (0, 0): "1-1",
    (0, 1): "1-2",
    (3, 0): "4-1",
    (3, 1): "4-2",
    (7, 0): "8-1",
    (7, 1): "8-2",
    (7, 2): "8-3",
    (7, 3): "8-4",
}

# Ordered expected stages for the warps showcase panel (fixed row order).
WARP_PANEL_ORDER: tuple[str, ...] = (
    "1-1",
    "1-2",
    "4-1",
    "4-2",
    "8-1",
    "8-2",
    "8-3",
    "8-4",
)

# Legal next labels on the warps route. Intermediate load flicker (1-3 after
# 1-2, 4-3 after 4-2, blackout garbage) is ignored so the panel only locks
# real route exits.
WARP_SUCCESSORS: dict[str, frozenset[str]] = {
    "1-1": frozenset({"1-2"}),
    "1-2": frozenset({"4-1"}),  # underground warp → World 4
    "4-1": frozenset({"4-2"}),
    "4-2": frozenset({"8-1"}),  # vine/top-clip warp → World 8
    "8-1": frozenset({"8-2"}),
    "8-2": frozenset({"8-3"}),
    "8-3": frozenset({"8-4"}),
    "8-4": frozenset(),  # ends on Axe, not a level change
}


def stage_label(world: int, level: int) -> str:
    """Human stage label; falls back to ``W-L`` for non-warp stages."""
    key = (int(world), int(level))
    if key in _WARP_STAGE_LABELS:
        return _WARP_STAGE_LABELS[key]
    return f"{int(world) + 1}-{int(level) + 1}"


def _format_hud_time(frames: int, fps: float) -> str:
    """Compact ``M:SS.cc`` (centiseconds) for the panel."""
    if frames < 0:
        frames = 0
    total_cs = int(round(frames * 100 / fps)) if fps > 0 else 0
    minutes, rem = divmod(total_cs, 60 * 100)
    secs, cs = divmod(rem, 100)
    if minutes >= 10:
        return f"{minutes:02d}:{secs:02d}.{cs:02d}"
    return f"{minutes}:{secs:02d}.{cs:02d}"


@dataclass
class RtaSplitTracker:
    """Live exit-detect splits for the recording HUD.

    Clock is caller-supplied (usually ``video.timer_frames``). On Axe
    (8-4 + ``oper_mode=2``) the tracker freezes: further ``observe`` calls
    leave completed splits and final time unchanged.
    """

    fps: float = NTSC_FPS
    panel_order: tuple[str, ...] = WARP_PANEL_ORDER
    completed: list[dict[str, Any]] = field(default_factory=list)
    current_label: str | None = None
    frozen: bool = False
    freeze_frame: int | None = None
    _seen_playing: bool = False
    _prev_key: tuple[int, int] | None = None

    def observe(self, snap: Any, *, clock_frames: int) -> bool:
        """Update from a post-step snapshot. Returns True if a split just locked."""
        if self.frozen:
            return False

        world = int(getattr(snap, "world"))
        level = int(getattr(snap, "level"))
        oper = int(getattr(snap, "oper_mode"))
        key = (world, level)

        # Axe / ending: lock final 8-4 split and freeze.
        if world == 7 and level == 3 and oper == 2:
            return self._freeze_ending(clock_frames=clock_frames)

        if oper != 1:
            return False

        # Ignore garbage pre-play / blackout until first real stage.
        if not self._seen_playing:
            if key not in _WARP_STAGE_LABELS and stage_label(world, level) not in self.panel_order:
                # Still allow non-warp full runs: any world 0–7 level 0–3.
                if not (0 <= world <= 7 and 0 <= level <= 3):
                    return False
            self._seen_playing = True
            self._prev_key = key
            self.current_label = stage_label(world, level)
            return False

        assert self._prev_key is not None
        if key == self._prev_key:
            return False

        # Level change → previous stage exit only if successor is on-route.
        # Skip load-screen flicker (e.g. 1-2→1-3 before warp settles on 4-1).
        prev_label = stage_label(*self._prev_key)
        new_label = stage_label(world, level)
        allowed = WARP_SUCCESSORS.get(prev_label)
        if allowed is not None and new_label not in allowed:
            return False

        self._lock_split(prev_label, clock_frames=clock_frames, kind="exit")
        self._prev_key = key
        self.current_label = new_label
        return True

    def _freeze_ending(self, *, clock_frames: int) -> bool:
        label = self.current_label or "8-4"
        # Prefer 8-4 as the final segment name even if current was blank.
        if self._prev_key == (7, 3) or label == "8-4":
            label = "8-4"
        already = {row["label"] for row in self.completed}
        if label not in already:
            self._lock_split(label, clock_frames=clock_frames, kind="axe")
        self.frozen = True
        self.freeze_frame = int(clock_frames)
        self.current_label = None
        return True

    def _lock_split(self, label: str, *, clock_frames: int, kind: str) -> None:
        prev_cum = int(self.completed[-1]["cum_frames"]) if self.completed else 0
        self.completed.append(
            {
                "label": label,
                "cum_frames": int(clock_frames),
                "seg_frames": int(clock_frames) - prev_cum,
                "cum_time": format_time_mmss(int(clock_frames), self.fps),
                "seg_time": format_time_mmss(int(clock_frames) - prev_cum, self.fps),
                "kind": kind,
            }
        )

    def lines(
        self,
        *,
        clock_frames: int,
        max_rows: int | None = None,
    ) -> list[str]:
        """Text rows for the overlay (header + stage rows + total)."""
        done = {row["label"]: row for row in self.completed}
        order = list(self.panel_order)
        # Include any unexpected completed labels after known order.
        for row in self.completed:
            if row["label"] not in order:
                order.append(row["label"])

        rows: list[str] = ["RTA"]
        for label in order:
            if label in done:
                t = _format_hud_time(int(done[label]["cum_frames"]), self.fps)
                mark = " *" if done[label].get("kind") == "axe" else ""
                rows.append(f"{label}  {t}{mark}")
            elif self.current_label == label and not self.frozen:
                t = _format_hud_time(int(clock_frames), self.fps)
                rows.append(f"{label}  {t}  ")
            else:
                rows.append(f"{label}  --:--.--")

        display_clock = (
            int(self.freeze_frame)
            if self.frozen and self.freeze_frame is not None
            else int(clock_frames)
        )
        total = _format_hud_time(display_clock, self.fps)
        if self.frozen:
            rows.append(f"AXE {total}")
        else:
            rows.append(f"TOT {total}")

        if max_rows is not None and max_rows > 0:
            # Keep header + last (max_rows-1) body lines.
            if len(rows) > max_rows:
                rows = [rows[0]] + rows[-(max_rows - 1) :]
        return rows

    def report(self) -> dict[str, Any]:
        return {
            "frozen": self.frozen,
            "freeze_frame": self.freeze_frame,
            "fps": self.fps,
            "completed": list(self.completed),
            "current_label": self.current_label,
        }


def draw_rta_split_panel(
    obs: np.ndarray,
    lines: Sequence[str],
    *,
    x: int = 2,
    y: int = 2,
    pad: int = 2,
    bg: tuple[int, int, int] = (0, 0, 0),
    bg_alpha: float = 0.55,
    fg: tuple[int, int, int] = (240, 248, 255),
    header_fg: tuple[int, int, int] = (103, 232, 164),
    axe_fg: tuple[int, int, int] = (255, 214, 102),
    current_fg: tuple[int, int, int] = (255, 255, 180),
) -> np.ndarray:
    """Composite a semi-transparent top-left text panel onto ``obs`` (RGB)."""
    if not lines:
        return np.asarray(obs, dtype=np.uint8)

    rgb = np.asarray(obs, dtype=np.uint8).copy()
    h, w = rgb.shape[:2]
    image = Image.fromarray(rgb).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default(size=8)

    line_sizes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    text_w = max((b[2] - b[0] for b in line_sizes), default=0)
    line_h = max((b[3] - b[1] for b in line_sizes), default=8) + 1
    box_w = text_w + pad * 2
    box_h = line_h * len(lines) + pad * 2
    x1 = max(0, min(x, w - 1))
    y1 = max(0, min(y, h - 1))
    x2 = max(x1 + 1, min(x1 + box_w, w))
    y2 = max(y1 + 1, min(y1 + box_h, h))

    alpha = int(round(255 * max(0.0, min(1.0, bg_alpha))))
    draw.rectangle((x1, y1, x2, y2), fill=(*bg, alpha))

    cy = y1 + pad
    for i, line in enumerate(lines):
        if i == 0:
            color = header_fg
        elif line.startswith("AXE"):
            color = axe_fg
        elif line.rstrip().endswith("  ") and "--:--.--" not in line and "*" not in line:
            # Live current row (trailing spaces marker from lines()).
            color = current_fg
        else:
            color = fg
        draw.text((x1 + pad, cy), line.rstrip(), fill=color, font=font)
        cy += line_h

    return np.asarray(image.convert("RGB"), dtype=np.uint8)


class VideoWriter:
    """ffmpeg RGB (+ native PCM) capture with NES button / timestamp footer.

    When ``splits_panel`` is on (default with HUD), a top-left RTA level list
    tracks exit-detect cum times and **freezes on Axe** (``oper_mode=2`` on
    8-4). The footer speedrun clock freezes on the same frame so Peach hold
    does not inflate the on-screen RTA time.
    """

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
        splits_panel: bool = True,
        splits_fps: float = NTSC_FPS,
    ) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg is required for video recording")
        self.path = path
        self.scale = max(1, scale)
        self.fps = fps
        self.hud = hud
        self.route_label = route_label
        self.splits_panel = bool(splits_panel and hud)
        self.src_w = width
        self.src_h = height + (FOOTER_HEIGHT if hud else 0)
        self.out_w = self.src_w * self.scale
        self.out_h = self.src_h * self.scale
        self.frames = 0
        self.timer_frames = 0
        self.intro_frames = 0
        self.audio_samples = 0
        self.audio_rate = audio_rate
        self.game_w = width
        self.game_h = height
        self._timer_frozen = False
        self.splits = (
            RtaSplitTracker(fps=splits_fps) if self.splits_panel else None
        )
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

        if not self._timer_frozen:
            self.timer_frames += 1
        display_clock = self.timer_frames

        if snap is not None and self.splits is not None:
            self.splits.observe(snap, clock_frames=display_clock)
            if self.splits.frozen:
                self._timer_frozen = True
                if self.splits.freeze_frame is not None:
                    self.timer_frames = int(self.splits.freeze_frame)
                    display_clock = self.timer_frames

        if self.hud:
            if self.splits is not None:
                panel_lines = self.splits.lines(clock_frames=display_clock)
                rgb = draw_rta_split_panel(rgb, panel_lines)

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
            if self._timer_frozen:
                upper_left = f"{upper_left}  AXE".strip()
            upper_right = frame_timestamp(display_clock, self.fps)
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

        self._emit_rgb(rgb, audio=audio, count_timer=False)

    def write_intro(
        self,
        lines: list[str],
        *,
        hold_frames: int = DEFAULT_INTRO_FRAMES,
    ) -> None:
        """Write a project intro slide before gameplay (same encode session)."""
        if hold_frames <= 0:
            return
        card = render_intro_card(
            lines,
            width=self.game_w,
            height=self.game_h,
            with_footer=self.hud,
        )
        silent = self._silent_audio_frame()
        for _ in range(hold_frames):
            self._emit_rgb(card, audio=silent, count_timer=False)
            self.intro_frames += 1

    def _silent_audio_frame(self) -> np.ndarray | None:
        if self.audio_rate is None or self.audio_rate <= 0:
            return None
        n = max(1, int(round(self.audio_rate / float(self.fps))))
        return np.zeros((n, 2), dtype=np.int16)

    def _emit_rgb(
        self,
        rgb: np.ndarray,
        *,
        audio: np.ndarray | None,
        count_timer: bool,
    ) -> None:
        if self._proc.stdin is None:
            return
        frame = np.asarray(rgb, dtype=np.uint8)
        if frame.shape != (self.src_h, self.src_w, 3):
            raise ValueError(
                f"expected frame {(self.src_h, self.src_w, 3)}, got {frame.shape}"
            )
        if self.scale > 1:
            frame = np.repeat(
                np.repeat(frame, self.scale, axis=0), self.scale, axis=1
            )
        self._proc.stdin.write(frame.tobytes())
        self.frames += 1
        if count_timer:
            self.timer_frames += 1

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
        if (
            self._audio is not None
            and self.audio_rate is not None
            and self.audio_rate > 0
            and self.frames > 0
        ):
            expected = int(round(self.frames * self.audio_rate / float(self.fps)))
            missing = expected - self.audio_samples
            if missing > 0:
                pad = np.zeros((missing, 2), dtype=np.int16)
                self._audio.writeframesraw(pad.astype("<i2", copy=False).tobytes())
                self.audio_samples += missing
        self._close_streams()
        if self.audio_rate is None or not self._wav.exists():
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


def env_audio(env: Any) -> np.ndarray | None:
    """Pull native stereo PCM from the emulator core when available."""
    em = getattr(env, "em", None)
    if em is None or not hasattr(em, "get_audio"):
        return None
    try:
        return np.asarray(em.get_audio(), dtype=np.int16)
    except Exception:  # noqa: BLE001 — audio is best-effort for recordings
        return None


def env_audio_rate(env: Any) -> int | None:
    em = getattr(env, "em", None)
    if em is None or not hasattr(em, "get_audio_rate"):
        return None
    try:
        rate = int(em.get_audio_rate())
    except Exception:  # noqa: BLE001
        return None
    return rate if rate > 0 else None


def hud_action(action: Any, snap: Any) -> list[int] | np.ndarray | None:
    """Blank the footer buttons during automated flagpole / castle walk."""
    if action is None:
        return None
    if snap is not None and int(getattr(snap, "player_state", -1)) in (3, 4, 5):
        return [0] * len(np.asarray(action).tolist())
    return action


def write_video(
    video: VideoWriter | None,
    obs: Any,
    *,
    env: Any = None,
    action: Any = None,
    label: str = "",
    snap: Any = None,
) -> None:
    if video is None or obs is None:
        return
    if snap is None and env is not None:
        snap = read_snapshot(env.get_ram())
    video.write(
        obs,
        action=hud_action(action, snap),
        audio=env_audio(env) if env is not None else None,
        label=label,
        snap=snap,
    )
