"""Replay a verified-tip button tape from power-on. No controller.

    SDL_VIDEODRIVER=dummy uv run python alttp/scripts/replay_tape.py --record-verified-tip
    SDL_VIDEODRIVER=dummy uv run python alttp/scripts/replay_tape.py alttp/recordings/verified_tip.tape.npz
    SDL_VIDEODRIVER=dummy uv run python alttp/scripts/replay_tape.py <tape> --video

The sync window is the full ``get_ram()`` image (no excluded bytes; see
``docs/ram_map.md``). ``frame_crc`` is that CRC after every committed
``env.step``. Bush-lift lookahead calls ``em.set_state`` back to the
approach and would desync a linear replay, so those frames are omitted.
Blank SRAM is applied once after ``reset``, before the audited attempt:
the boot inject in ``startup.py`` is the same 4 checksum bytes, and
replaying it mid-run is a load. Does not write ``verified_tip_run.json``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from alttp.paths import BOOT_STATE, INTEGRATION_DIR, RECORDINGS_DIR
from alttp.spine.tape import (
    TAPE_SUFFIX,
    LiveRun,
    load_tape,
    pack_buttons,
    ram_crc,
    sha1_file,
    tape_path,
)
from alttp.startup import (
    build_blank_sram,
    build_boot_env,
    inject_sram_into_env,
    inject_sram_into_state,
)
from retro_harness.controls import SNES_BUTTON_NAMES
from retro_harness.segment_runner import configure_headless
from retro_harness.video import FrameVideoWriter

VERIFIED_TIP_TAG = "verified_tip"
_BANNED_REPORT = "verified_tip_run.json"


class _EmProxy:
    def __init__(self, em: Any, get_state: Any, set_state: Any) -> None:
        self._em = em
        self._get = get_state
        self._set = set_state

    def get_state(self) -> Any:
        return self._get()

    def set_state(self, state: Any) -> Any:
        return self._set(state)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._em, name)


class _EnvProxy:
    """Forward the boot env, but expose the wrapped emulator."""

    def __init__(self, env: Any, em: _EmProxy) -> None:
        self._env = env
        self._em = em

    @property
    def em(self) -> _EmProxy:
        return self._em

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)


class _CommittedRecorder:
    """Button + sync-CRC tape with lookahead restores truncated out.

    A ``set_state`` of bytes previously returned by ``get_state`` is a
    restore: drop steps since that checkpoint. Any other ``set_state``
    (the boot SRAM inject) is not played back, so it is not applied.
    """

    def __init__(self, raw: Any) -> None:
        self.raw = raw
        self.buttons: list[np.ndarray] = []
        self.crcs: list[int] = []
        self.checkpoints: dict[bytes, int] = {}
        self.restores = 0
        self.swallowed = 0
        self.omitted = 0
        self._orig_step = raw.step
        self._orig_reset = raw.reset
        self._orig_get = raw.em.get_state
        self._orig_set = raw.em.set_state

    def attach(self) -> _EnvProxy:
        self.raw.step = self._step
        self.raw.reset = self._reset
        em = _EmProxy(self.raw.em, self._get_state, self._set_state)
        return _EnvProxy(self.raw, em)

    def _step(self, action: Any, *args: Any, **kwargs: Any) -> Any:
        self.buttons.append(pack_buttons(action))
        out = self._orig_step(action, *args, **kwargs)
        self.crcs.append(ram_crc(self.raw.get_ram()))
        return out

    def _reset(self, *args: Any, **kwargs: Any) -> Any:
        out = self._orig_reset(*args, **kwargs)
        # Hit the emulator directly. The proxy swallows non-checkpoint loads,
        # and this blank has to stick before the first taped frame.
        state = self._orig_get()
        patched = inject_sram_into_state(bytes(state), build_blank_sram())
        self._orig_set(patched)
        return out

    def _get_state(self) -> Any:
        state = self._orig_get()
        self.checkpoints[bytes(state)] = len(self.buttons)
        return state

    def _set_state(self, state: Any) -> Any:
        key = bytes(state)
        if key in self.checkpoints:
            keep = self.checkpoints[key]
            self.omitted += len(self.buttons) - keep
            del self.buttons[keep:]
            del self.crcs[keep:]
            self.restores += 1
            return self._orig_set(state)
        self.swallowed += 1
        return None


def _refuse_verified_tip_run(path: Path) -> None:
    if path.name == _BANNED_REPORT:
        raise RuntimeError(
            "refusing to overwrite recordings/verified_tip_run.json"
        )


def _save_tape(
    path: Path,
    buttons: np.ndarray,
    crcs: np.ndarray,
    *,
    rom_sha1: str,
    source_phase: str,
    source_ok: bool,
    source_blocker: str,
    restores: int,
    swallowed: int,
    omitted: int,
) -> Path:
    _refuse_verified_tip_run(path)
    final_crc = int(crcs[-1]) if len(crcs) else 0
    meta = {
        "frames": int(buttons.shape[0]),
        "boot_state": BOOT_STATE,
        "rom_sha1": rom_sha1,
        "final_ram_crc": final_crc,
        "button_order": list(SNES_BUTTON_NAMES),
        "tag": path.name[: -len(TAPE_SUFFIX)],
        "sync_window": "full get_ram()",
        "excluded_bytes": [],
        "sram_blank_at_boot": True,
        "committed": True,
        "lookahead_frames_omitted": omitted,
        "restores": restores,
        "swallowed_loads": swallowed,
        "source_phase": source_phase,
        "source_ok": source_ok,
        "source_blocker": source_blocker,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        buttons=buttons,
        meta=np.asarray(json.dumps(meta)),
        final_ram_crc=np.asarray(final_crc, dtype=np.uint32),
        frame_crc=np.asarray(crcs, dtype=np.uint32),
    )
    loaded = load_tape(path)
    if loaded.final_ram_crc != final_crc or len(loaded.buttons) != len(crcs):
        raise RuntimeError(f"tape round-trip failed for {path}")
    return path


def record_verified_tip(path: Path | None = None) -> Path:
    """Run the verified tip once and write the committed power-on tape."""
    from alttp.opening_route.full_tip import run_to_verified_tip

    out = tape_path(VERIFIED_TIP_TAG) if path is None else Path(path)
    raw = build_boot_env()
    rec = _CommittedRecorder(raw)
    env = rec.attach()
    try:
        result = run_to_verified_tip(env, close=False)
    finally:
        raw.close()
    if not rec.buttons or len(rec.buttons) != len(rec.crcs):
        raise RuntimeError("verified tip recorded no committed frames")
    saved = _save_tape(
        out,
        np.stack(rec.buttons),
        np.asarray(rec.crcs, dtype=np.uint32),
        rom_sha1=sha1_file(INTEGRATION_DIR / "rom.sfc"),
        source_phase=result.phase,
        source_ok=bool(result.ok),
        source_blocker=result.blocker,
        restores=rec.restores,
        swallowed=rec.swallowed,
        omitted=rec.omitted,
    )
    print(
        f"recorded {saved.name}: frames={len(rec.buttons)} "
        f"omitted={rec.omitted} restores={rec.restores} "
        f"swallowed={rec.swallowed} phase={result.phase} "
        f"ok={result.ok} blocker={result.blocker!r}"
    )
    return saved


def _load_frame_crc(path: Path) -> np.ndarray | None:
    with np.load(path, allow_pickle=False) as data:
        if "frame_crc" not in data.files:
            return None
        return np.asarray(data["frame_crc"], dtype=np.uint32)


def _report_path(tag: str) -> Path:
    path = RECORDINGS_DIR / f"{tag}.json"
    _refuse_verified_tip_run(path)
    return path


def _open_writer(path: Path, frame: np.ndarray) -> FrameVideoWriter:
    rgb = np.asarray(frame)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise RuntimeError(f"replay frame is not HxWx3 RGB, got {rgb.shape}")
    return FrameVideoWriter(
        path,
        width=int(rgb.shape[1]),
        height=int(rgb.shape[0]),
        fps=60,
        scale=2,
        preset="veryfast",
        audio_rate=None,
        footer=False,
    )


def replay_tape(
    path: Path,
    *,
    tag: str | None = None,
    video: Path | None = None,
) -> dict[str, Any]:
    """Play ``path`` from power-on. Returns the sync report."""
    tape = load_tape(path)
    crcs = _load_frame_crc(path)
    source = str(tape.meta.get("tag") or path.name.removesuffix(TAPE_SUFFIX))
    tag = tag or f"{source}_replay"
    if tape.boot_state != BOOT_STATE:
        raise RuntimeError(
            f"tape boot_state {tape.boot_state!r} is not power-on {BOOT_STATE}"
        )
    if tape.meta.get("resumed_from"):
        raise RuntimeError(
            f"tape resumed from {tape.meta['resumed_from']!r}; replay needs power-on"
        )
    report_path = _report_path(tag)
    frames = int(tape.buttons.shape[0])
    if crcs is None or len(crcs) != frames:
        report = {
            "ok": False,
            "desynced": None,
            "first_desync_frame": None,
            "loads": None,
            "writes": None,
            "frames": frames,
            "tape": str(path),
            "error": "tape has no per-frame sync CRC for the full get_ram() window",
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"replay {tag}: ok=False desynced=None ({report['error']})")
        return report

    blank = bool(tape.meta.get("sram_blank_at_boot"))
    env = build_boot_env()
    writer: FrameVideoWriter | None = None
    video_closed = False
    video_out: str | None = None
    try:
        env.reset()
        if blank:
            inject_sram_into_env(env, build_blank_sram())
        live = LiveRun(env, provider="alttp.replay_tape")
        wrapped = live.env
        first: int | None = None
        desynced = 0
        started = time.perf_counter()
        for index, vec in enumerate(tape.buttons):
            obs, *_ = wrapped.step(np.asarray(vec, dtype=np.int8))
            if video is not None and writer is None:
                writer = _open_writer(video, np.asarray(obs))
            if writer is not None:
                writer.write(np.asarray(obs), action=vec)
            if ram_crc(wrapped.get_ram()) != int(crcs[index]):
                desynced += 1
                if first is None:
                    first = index + 1
        wall_s = time.perf_counter() - started
        audit = live.report()
        if writer is not None:
            video_out = str(writer.close())
            video_closed = True
    finally:
        if writer is not None and not video_closed:
            writer.abort()
        env.close()

    loads = int(audit.set_state)
    writes = int(audit.writes)
    in_sync = desynced == 0 and len(crcs) == frames
    report = {
        "ok": bool(in_sync and loads == 0 and writes == 0),
        "tape": str(path),
        "tag": tag,
        "frames": frames,
        "desynced": desynced,
        "first_desync_frame": first,
        "loads": loads,
        "writes": writes,
        "set_state": loads,
        "sram_blank_at_boot": blank,
        "sync_window": "full get_ram()",
        "excluded_bytes": [],
        "wall_s": round(wall_s, 1),
        "x_realtime": round(frames / 60.0 / wall_s, 2) if wall_s else None,
        "video": video_out,
        "final_ram_crc_match": int(crcs[-1]) == int(tape.final_ram_crc),
        "source_phase": tape.meta.get("source_phase"),
        "source_ok": tape.meta.get("source_ok"),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"replay {tag}: ok={report['ok']} desynced={desynced} "
        f"first_desync={first} frames={frames} loads={loads} "
        f"writes={writes} wall={report['wall_s']}s "
        f"({report['x_realtime']}x real time) video={video_out}"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tape", nargs="?", type=Path)
    parser.add_argument(
        "--record-verified-tip",
        action="store_true",
        help="Write recordings/verified_tip.tape.npz from the current tip",
    )
    parser.add_argument("--tag", default=None, help="Report tag. Default: <tape>_replay")
    parser.add_argument(
        "--video",
        nargs="?",
        const="AUTO",
        default=None,
        help="Write an MP4 and print the realtime factor. Optional path.",
    )
    args = parser.parse_args(argv)
    configure_headless()
    if args.record_verified_tip:
        record_verified_tip()
        if args.tape is None:
            return 0
    if args.tape is None:
        parser.error("tape path is required unless --record-verified-tip")
    video: Path | None = None
    if args.video is not None:
        source = args.tape.name.removesuffix(TAPE_SUFFIX)
        tag = args.tag or f"{source}_replay"
        video = (
            RECORDINGS_DIR / f"{tag}.mp4"
            if args.video == "AUTO"
            else Path(args.video)
        )
        _refuse_verified_tip_run(video)
    report = replay_tape(args.tape, tag=args.tag, video=video)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
