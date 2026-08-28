"""Per-tile hop recording for the room-grid demo."""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Sequence

from super_metroid.demo.grid_mosaic import label_frame
from super_metroid.demo.room_grid import (
    DEFAULT_SECONDS,
    GridTile,
    NTSC_FPS,
    probe_parallel,
)
from super_metroid.routes.kpdr.registry import KPDR_SEGMENTS
from super_metroid.source_states import get_source


class FrameBudget(Exception):
    """Demo cap: hop still running when the clip is long enough."""

    def __init__(self, frames: int) -> None:
        super().__init__(f"demo frame budget {frames}")
        self.frames = frames


def record_play_flags(exc: BaseException | None) -> tuple[bool, bool]:
    """Return ``(hopOk, capped)``. A frame cap is a demo clip, not hop success."""

    if exc is None:
        return True, False
    if isinstance(exc, FrameBudget):
        return False, True
    raise exc


class _GridSession:
    """ControllerSession that records RGB and stops at ``max_frames``."""

    def __init__(
        self,
        env: Any,
        assist: Any,
        writer: Any,
        *,
        label: str,
        max_frames: int,
    ) -> None:
        from super_metroid.ram import parse_env_state

        self.env = env
        self.assist = assist
        self.writer = writer
        self.label = label
        self.max_frames = max_frames
        self.frame = 0
        self.state = parse_env_state(env, mode="nav")

    def step(self, action: Any, reason: str = "") -> Any:
        from super_metroid.ram import parse_env_state

        del reason
        if self.frame >= self.max_frames:
            raise FrameBudget(self.max_frames)
        obs, *_ = self.env.step(action)
        self.frame += 1
        self.state = parse_env_state(self.env, frame=self.frame, mode="nav")
        self.assist.apply(self.env.data, self.state)
        if self.writer is not None:
            self.writer.write_from_env(
                self.env,
                label_frame(obs, self.label),
                action=action,
                frame_index=self.frame,
                room_id=int(self.state.room_id),
            )
        return self.state


def record_tile(
    tile: GridTile,
    video_path: Path,
    *,
    max_frames: int = DEFAULT_SECONDS * NTSC_FPS,
    scale: int = 1,
    crf: int = 23,
) -> dict[str, object]:
    """Boot one pin, play one hop, write a silent mp4 (capped)."""

    from retro_harness.actions import idle_action
    from retro_harness.video import VideoCaptureConfig, VideoRecorder
    from super_metroid.assist import UnlimitedResourcesAssist
    from super_metroid.dev.common import boot_from_state, make_dev_env
    from super_metroid.ram import parse_env_state, probe_pin

    source = get_source(tile.source_id)
    play = KPDR_SEGMENTS.get(tile.segment)
    report: dict[str, object] = {
        "segment": tile.segment,
        "sourceId": tile.source_id,
        "label": tile.label,
        "video": str(video_path),
        "ok": False,
        "hopOk": False,
        "capped": False,
        "frames": 0,
        "error": None,
    }
    if play is None:
        report["error"] = f"unknown segment {tile.segment!r}"
        return report
    if not source.path.is_file():
        report["error"] = f"missing pin {source.path}"
        return report

    video_path.parent.mkdir(parents=True, exist_ok=True)
    env = make_dev_env()
    assist = UnlimitedResourcesAssist()
    writer: VideoRecorder | None = None
    try:
        boot_from_state(env, source.path, settle_frames=5)
        obs = env.render()
        if obs is None:
            obs, *_ = env.step(idle_action())
            assist.apply(env.data, parse_env_state(env, mode="nav"))
            obs = env.render()
        if obs is None:
            report["error"] = "no rgb frame after boot"
            return report
        config = VideoCaptureConfig(
            fps=NTSC_FPS,
            scale=scale,
            crf=crf,
            preset="ultrafast",
            audio=False,
            footer=False,
        )
        writer = VideoRecorder(
            video_path,
            width=int(obs.shape[1]),
            height=int(obs.shape[0]),
            config=config,
        )
        session = _GridSession(
            env,
            assist,
            writer,
            label=tile.label,
            max_frames=max_frames,
        )
        writer.write(
            label_frame(obs, tile.label),
            action=None,
            frame_index=0,
            room_id=int(session.state.room_id),
        )
        try:
            play(session)
        except FrameBudget as exc:
            hop_ok, capped = record_play_flags(exc)
        else:
            hop_ok, capped = record_play_flags(None)
        report["hopOk"] = hop_ok
        report["capped"] = capped
        st = session.state
        report["frames"] = session.frame
        report["roomIdHex"] = f"0x{int(st.room_id):04X}"
        report["samusX"] = int(st.samus_x)
        report["samusY"] = int(st.samus_y)
        report["pose"] = int(st.pose)
        report["probePin"] = probe_pin(st)
        report["ok"] = session.frame > 0 and video_path.is_file()
    except Exception as exc:  # noqa: BLE001 — demo surface
        report["error"] = str(exc)
        report["ok"] = video_path.is_file() and int(report["frames"] or 0) > 0
    finally:
        if writer is not None:
            writer.close()
        env.close()
    return report


def _record_job(payload: dict[str, object]) -> dict[str, object]:
    """Spawn-safe worker: reconstruct a tile and record."""

    tile = GridTile(
        str(payload["segment"]),
        str(payload["sourceId"]),
        str(payload["label"]),
    )
    return record_tile(
        tile,
        Path(str(payload["video"])),
        max_frames=int(payload["maxFrames"]),
        scale=int(payload["scale"]),
        crf=int(payload["crf"]),
    )


def record_tiles(
    tiles: Sequence[GridTile],
    tile_dir: Path,
    *,
    workers: int = 1,
    max_frames: int = DEFAULT_SECONDS * NTSC_FPS,
    scale: int = 1,
    crf: int = 23,
    force: bool = False,
) -> list[dict[str, object]]:
    """Record every tile; ``workers>1`` uses spawned processes."""

    tile_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        {
            "segment": tile.segment,
            "sourceId": tile.source_id,
            "label": tile.label,
            "video": str(tile_dir / f"{index:02d}_{tile.segment}.mp4"),
            "maxFrames": max_frames,
            "scale": scale,
            "crf": crf,
        }
        for index, tile in enumerate(tiles)
    ]
    if workers <= 1:
        return [_record_job(job) for job in jobs]
    verdict = probe_parallel(workers)
    if not verdict.ok and not force:
        raise RuntimeError(verdict.reason)
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        return list(pool.map(_record_job, jobs))
