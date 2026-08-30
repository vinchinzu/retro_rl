"""Replay TASVideos #1724M under fceumm and RAM-gate 1-1 checkpoints.

Source movie is FCEUX 0.98.28. This module replays it on stable-retro
fceumm and records honest core status: BizHawk is not invoked.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from smb.tas.fm2 import parse_fm2
from smb.tas.replay import to_action9
from smb2.paths import (
    CONTROL_PROOF_PATH,
    EVIDENCE_MANIFEST_PATH,
    GAME,
    GAME_DIR,
    REF_MOVIE_PATH,
    STATE_ARTIFACTS_DIR,
)
from smb2.ram import Smb2Snapshot, is_level1_control, is_level1_start, read_snapshot
from smb2.tas_manifest import (
    BizHawkValidationStatus,
    CheckpointEvidence,
    CheckpointStatus,
    TASEvidenceManifest,
)

CONTROL_SEARCH_MAX = 500
REPLAY_CORE = "fceumm"
SOURCE_CORE = "FCEUX 0.98.28 (movie header emuVersion 9828)"

_GATE_BY_NAME: dict[str, Callable[[Smb2Snapshot], bool]] = {
    "level1_start": is_level1_start,
    "level1_control": is_level1_control,
}


@dataclass(frozen=True, slots=True)
class CapturedCheckpoint:
    """One RAM-gated frame plus the emulator state bytes at that frame."""

    name: str
    frame: int
    snapshot: Smb2Snapshot
    state_bytes: bytes


def snapshot_dict(snap: Smb2Snapshot) -> dict[str, int | float | None]:
    """JSON-friendly RAM fingerprint for a captured checkpoint."""
    return {
        "frame": snap.frame,
        "player_x": snap.player_x,
        "player_y": snap.player_y,
        "x_page": snap.x_page,
        "y_page": snap.y_page,
        "abs_x": snap.abs_x,
        "abs_y": snap.abs_y,
        "x_speed": snap.x_speed,
        "jump_physics": snap.jump_physics,
        "character": snap.character,
        "hearts": snap.hearts,
        "lives": snap.lives,
        "transition": snap.transition,
        "area": snap.area,
        "subarea": snap.subarea,
        "level": snap.level,
        "world": snap.world,
        "obs_mean": None if snap.obs_mean is None else round(float(snap.obs_mean), 3),
    }


def first_matching_frame(
    snapshots: list[Smb2Snapshot],
    predicate: Callable[[Smb2Snapshot], bool],
) -> Smb2Snapshot | None:
    """Return the first snapshot that opens *predicate*, or None."""
    for snap in snapshots:
        if predicate(snap):
            return snap
    return None


def capture_named_checkpoints(
    env: Any,
    frames: list[list[int]],
    *,
    names: tuple[str, ...] = ("level1_start", "level1_control"),
    max_frames: int = CONTROL_SEARCH_MAX,
) -> dict[str, CapturedCheckpoint]:
    """Step TAS frames from the current env until each named gate opens."""
    pending = [name for name in names if name in _GATE_BY_NAME]
    if not pending:
        raise ValueError(f"no supported checkpoint names in {names}")
    captured: dict[str, CapturedCheckpoint] = {}
    limit = min(len(frames), max_frames)
    for index in range(limit):
        obs, *_ = env.step(to_action9(frames[index]))
        ram = env.get_ram()
        snap = read_snapshot(ram, frame=index, obs_mean=float(obs.mean()))
        for name in pending:
            if name in captured:
                continue
            if _GATE_BY_NAME[name](snap):
                captured[name] = CapturedCheckpoint(
                    name=name,
                    frame=index,
                    snapshot=snap,
                    state_bytes=bytes(env.em.get_state()),
                )
        if len(captured) == len(pending):
            return captured
    missing = [name for name in pending if name not in captured]
    raise RuntimeError(
        f"RAM gates did not open before frame {limit}: {', '.join(missing)}"
    )


def _describe(name: str, captured: CapturedCheckpoint) -> str:
    snap = captured.snapshot
    return (
        f"Materialized on {REPLAY_CORE} from {SOURCE_CORE} at movie frame "
        f"{captured.frame}; {name} x={snap.player_x} y={snap.player_y} "
        f"jump_physics={snap.jump_physics}. BizHawk not run."
    )


def apply_captures_to_manifest(
    manifest: TASEvidenceManifest,
    captures: dict[str, CapturedCheckpoint],
) -> TASEvidenceManifest:
    """Return a copy with captured slots marked materialized (BizHawk blocked)."""
    updated: list[CheckpointEvidence] = []
    for slot in manifest.checkpoints:
        captured = captures.get(slot.name)
        if captured is None:
            updated.append(slot)
            continue
        updated.append(
            replace(
                slot,
                frame=captured.frame,
                status=CheckpointStatus.MATERIALIZED,
                description=_describe(slot.name, captured),
            )
        )
    return replace(
        manifest,
        checkpoints=tuple(updated),
        bizhawk_validation_status=BizHawkValidationStatus.BLOCKED,
        source_core=SOURCE_CORE,
    )


def write_control_proof(
    captured: CapturedCheckpoint,
    path: Path = CONTROL_PROOF_PATH,
) -> Path:
    """Write a tracked RAM fingerprint for the 1-1 control gate."""
    payload = {
        "name": captured.name,
        "movie_frame": captured.frame,
        "replay_core": REPLAY_CORE,
        "source_core": SOURCE_CORE,
        "bizhawk_validation_status": BizHawkValidationStatus.BLOCKED.value,
        "gate": "is_level1_control",
        "snapshot": snapshot_dict(captured.snapshot),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def extract_level1_checkpoints(
    *,
    movie_path: Path = REF_MOVIE_PATH,
    manifest_path: Path = EVIDENCE_MANIFEST_PATH,
    max_frames: int = CONTROL_SEARCH_MAX,
    write_states: bool = True,
) -> TASEvidenceManifest:
    """Replay the TAS from power-on and materialize 1-1 start + control."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    from retro_harness.env import make_env, reset_obs, write_state_bytes

    movie = parse_fm2(movie_path)
    manifest = TASEvidenceManifest.from_json(manifest_path)
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    try:
        reset_obs(env)
        captures = capture_named_checkpoints(
            env,
            movie.frames,
            max_frames=max_frames,
        )
    finally:
        env.close()

    if write_states:
        STATE_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        for captured in captures.values():
            write_state_bytes(
                STATE_ARTIFACTS_DIR / f"{captured.name}.state",
                captured.state_bytes,
            )

    updated = apply_captures_to_manifest(manifest, captures)
    updated.write_json(manifest_path)
    if "level1_control" in captures:
        write_control_proof(captures["level1_control"])
    return updated


def main(argv: list[str] | None = None) -> int:
    """CLI: replay the vendored FM2 and write 1-1 checkpoint evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--movie",
        type=Path,
        default=REF_MOVIE_PATH,
        help="FM2 path (zip or unpacked)",
    )
    parser.add_argument("--max-frames", type=int, default=CONTROL_SEARCH_MAX)
    parser.add_argument(
        "--no-states",
        action="store_true",
        help="Update evidence only; do not write emulator states",
    )
    args = parser.parse_args(argv)
    updated = extract_level1_checkpoints(
        movie_path=args.movie,
        max_frames=args.max_frames,
        write_states=not args.no_states,
    )
    for slot in updated.checkpoints:
        print(f"{slot.name}: {slot.status.value} frame={slot.frame}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
