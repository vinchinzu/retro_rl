"""Pure RED auto-capture: frame dump + door/PLM-related RAM snapshot.

When a pure probe fails, write short visual + door/nav RAM context under
``super_metroid/debug/red_diag/`` so the next agent is not debugging dark.

PLM open-state / PLM-record WRAM offsets are **not** validated in this repo
(see ``kraid_door_plm_recon``). Snapshots include every source-confirmed door
and nav field we already trust, and explicitly mark PLM records as blocked.

Paths in artifacts and pin JSON are repo-relative (no absolute home prefixes).
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from super_metroid.paths import DEBUG_DIR, REPO_DIR
from super_metroid.ram import (
    ADDR_DOOR_DEF_PTR,
    ADDR_DOOR_TRANSITION,
    ADDR_GAME_STATE,
    ADDR_INVINCIBILITY_TIMER,
    ADDR_KNOCKBACK_TIMER,
    ADDR_TRANSITION_DIRECTION,
    SuperMetroidState,
    peek_wram,
    probe_pin,
)

# Short rolling window kept during pure play (RGB frames). ~45 @ 256×224×3 ≈ 7 MiB.
DEFAULT_RING_FRAMES = 45
FRAME_DUMP_SUBDIR = "frames"
SNAPSHOT_NAME = "door_plm_snapshot.json"
PIN_NAME = "pin.json"
MANIFEST_NAME = "red_diag_manifest.json"


def display_path(path: Path | str, *, root: Path | None = None) -> str:
    """Prefer repo-relative paths (no machine home prefixes)."""
    p = Path(path).resolve()
    base = (root or REPO_DIR).resolve()
    try:
        return str(p.relative_to(base))
    except ValueError:
        return str(p)


def default_red_diag_dir(
    *,
    segment: str = "",
    stamp: str | None = None,
    base: Path | None = None,
) -> Path:
    """``debug/red_diag/<stamp>[_segment]/`` under the game debug tree."""
    ts = stamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in segment)[:48]
    name = f"{ts}_{safe}" if safe else ts
    root = base if base is not None else DEBUG_DIR / "red_diag"
    return root / name


@dataclass
class RedDiagArtifacts:
    """Paths produced by a RED capture (all Path objects on disk)."""

    out_dir: Path
    snapshot_path: Path
    frame_paths: list[Path]
    pin_path: Path | None
    manifest_path: Path
    clip_path: Path | None = None  # reserved; frame dump is the default medium

    def as_report_dict(self, *, root: Path | None = None) -> dict[str, object]:
        """Schema attached to pure probe pin JSON / residual."""
        return {
            "outDir": display_path(self.out_dir, root=root),
            "snapshotPath": display_path(self.snapshot_path, root=root),
            "frameDumpDir": display_path(
                self.out_dir / FRAME_DUMP_SUBDIR, root=root
            ),
            "framePaths": [display_path(p, root=root) for p in self.frame_paths],
            "frameCount": len(self.frame_paths),
            "clipPath": (
                display_path(self.clip_path, root=root) if self.clip_path else None
            ),
            "pinPath": (
                display_path(self.pin_path, root=root) if self.pin_path else None
            ),
            "manifestPath": display_path(self.manifest_path, root=root),
            "medium": "frame_dump" if self.frame_paths else "snapshot_only",
        }


class FrameRing:
    """Fixed-size RGB frame ring for post-failure dump (no ffmpeg required)."""

    def __init__(self, maxlen: int = DEFAULT_RING_FRAMES) -> None:
        self._buf: deque[np.ndarray] = deque(maxlen=max(1, maxlen))

    def __len__(self) -> int:
        return len(self._buf)

    def push(self, frame: Any) -> None:
        if frame is None:
            return
        arr = np.asarray(frame)
        if arr.ndim < 2:
            return
        # Copy so later emulator buffers cannot mutate history.
        self._buf.append(np.array(arr, copy=True))

    def frames(self) -> list[np.ndarray]:
        return list(self._buf)


def build_door_plm_snapshot(
    env: Any | None,
    state: SuperMetroidState | None,
    *,
    error: str = "",
    segment: str = "",
    source: str | None = None,
    frames: int | None = None,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    """Door/nav RAM snapshot + explicit PLM-record blocked note.

    Safe without env (state-only pin). When env is present, peeks validated
    door definition / transition / timer fields via :func:`peek_wram`.
    """
    pin: dict[str, object] = probe_pin(state) if state is not None else {}
    door_fields: dict[str, object] = {}
    if state is not None:
        door_fields = {
            "door_transition": state.door_transition,
            "transition_direction": state.transition_direction,
            "game_state": state.game_state,
            "phase": (
                state.phase.name if hasattr(state.phase, "name") else str(state.phase)
            ),
            "room": f"0x{state.room_id:04X}",
            "pose": state.pose,
            "x": state.samus_x,
            "y": state.samus_y,
            "selected_item": state.selected_item,
            "enemy0_hp": state.enemy0_hp,
            "enemy0_x": state.enemy0_x,
            "enemy0_y": state.enemy0_y,
            "enemy0_spritemap": state.enemy0_spritemap,
        }
    peeks: dict[str, int] = {}
    if env is not None:
        try:
            peeks = peek_wram(
                env,
                {
                    "door_definition_ptr": ADDR_DOOR_DEF_PTR,
                    "door_transition": ADDR_DOOR_TRANSITION,
                    "transition_direction": ADDR_TRANSITION_DIRECTION,
                    "game_state": ADDR_GAME_STATE,
                    "invincibility_timer": ADDR_INVINCIBILITY_TIMER,
                    "knockback_timer": ADDR_KNOCKBACK_TIMER,
                },
            )
        except Exception as exc:  # noqa: BLE001 — diagnostic surface
            peeks = {}
            door_fields["peekError"] = str(exc)

    snapshot: dict[str, object] = {
        "kind": "pure_red_door_plm_snapshot",
        "segment": segment,
        "error": error,
        "source": source,
        "frames": frames,
        "probePin": pin,
        "doorNav": door_fields,
        "wramPeeks": peeks,
        "plmRecords": {
            "status": "blocked",
            "fields": [],
            "reason": (
                "No source-confirmed live WRAM offset for PLM records / "
                "PLM activation / blue-door open state (see kraid_door_plm_recon). "
                "Snapshot carries validated door_transition, door_def_ptr, and "
                "nav pin only."
            ),
        },
        "nonClaims": [
            "No blue-door open/closed determination from PLM records",
            "No PLM activation determination",
            "Not pure-green or continuous evidence",
            "No STATUS promotion",
        ],
    }
    if extra:
        snapshot["extra"] = extra
    return snapshot


def write_frame_dump(
    frames: Sequence[np.ndarray],
    out_dir: Path,
    *,
    prefix: str = "frame",
) -> list[Path]:
    """Write RGB frames as PNGs under ``out_dir``. Returns written paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if not frames:
        return []
    try:
        import cv2
    except ImportError:  # pragma: no cover — optional fallback
        return _write_frames_npy(frames, out_dir, prefix=prefix)

    written: list[Path] = []
    width = max(3, len(str(len(frames) - 1)))
    for i, frame in enumerate(frames):
        arr = np.asarray(frame)
        path = out_dir / f"{prefix}_{i:0{width}d}.png"
        if arr.ndim == 2:
            bgr = arr
        elif arr.shape[-1] == 3:
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        elif arr.shape[-1] == 4:
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        else:
            bgr = arr
        cv2.imwrite(str(path), bgr)
        written.append(path)
    return written


def _write_frames_npy(
    frames: Sequence[np.ndarray],
    out_dir: Path,
    *,
    prefix: str,
) -> list[Path]:
    written: list[Path] = []
    width = max(3, len(str(len(frames) - 1)))
    for i, frame in enumerate(frames):
        path = out_dir / f"{prefix}_{i:0{width}d}.npy"
        np.save(path, np.asarray(frame))
        written.append(path)
    return written


def write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def capture_red_artifacts(
    *,
    env: Any | None = None,
    state: SuperMetroidState | None = None,
    frames: Iterable[np.ndarray] | None = None,
    segment: str = "",
    error: str = "",
    source: str | None = None,
    probe_frames: int | None = None,
    out_dir: Path | None = None,
    pin_json: Path | None = None,
    report: dict[str, object] | None = None,
    write_pin: bool = True,
    extra: dict[str, object] | None = None,
    root: Path | None = None,
) -> RedDiagArtifacts:
    """Write frame dump + door/PLM snapshot under ``debug/red_diag/``.

    If ``report`` is provided and ``write_pin`` is true, also write pin JSON
    (either ``pin_json`` or ``out_dir/pin.json``) with redDiag paths attached.
    """
    dest = out_dir or default_red_diag_dir(segment=segment)
    dest.mkdir(parents=True, exist_ok=True)

    snapshot = build_door_plm_snapshot(
        env,
        state,
        error=error,
        segment=segment,
        source=source,
        frames=probe_frames,
        extra=extra,
    )
    snapshot_path = write_json(dest / SNAPSHOT_NAME, snapshot)

    frame_list = list(frames) if frames is not None else []
    frame_paths = write_frame_dump(frame_list, dest / FRAME_DUMP_SUBDIR)

    pin_path: Path | None = None
    artifacts = RedDiagArtifacts(
        out_dir=dest,
        snapshot_path=snapshot_path,
        frame_paths=frame_paths,
        pin_path=None,
        manifest_path=dest / MANIFEST_NAME,
        clip_path=None,
    )
    red_diag = artifacts.as_report_dict(root=root)

    if write_pin:
        pin_target = pin_json if pin_json is not None else dest / PIN_NAME
        pin_payload: dict[str, object]
        if report is not None:
            pin_payload = dict(report)
        else:
            pin_payload = {
                "success": False,
                "error": error,
                "segment": segment,
                "probePin": snapshot.get("probePin"),
            }
        pin_payload["redDiag"] = red_diag
        pin_payload["pinJson"] = display_path(pin_target, root=root)
        # Residual-friendly one-liners for PROCESS schema.
        pin_payload.setdefault(
            "residualArtifactLine",
            (
                f"snapshot={red_diag['snapshotPath']} "
                f"frames={red_diag['frameCount']} "
                f"frameDumpDir={red_diag['frameDumpDir']}"
            ),
        )
        write_json(pin_target, pin_payload)
        pin_path = pin_target
        artifacts.pin_path = pin_path
        red_diag = artifacts.as_report_dict(root=root)

    manifest = {
        "kind": "pure_red_diag_manifest",
        "segment": segment,
        "error": error,
        "redDiag": red_diag,
    }
    write_json(artifacts.manifest_path, manifest)
    return artifacts


def attach_red_diag(
    report: dict[str, object],
    artifacts: RedDiagArtifacts,
    *,
    root: Path | None = None,
) -> dict[str, object]:
    """Mutate *report* with redDiag paths and residual helper lines."""
    red = artifacts.as_report_dict(root=root)
    report["redDiag"] = red
    if artifacts.pin_path is not None:
        report["pinJson"] = display_path(artifacts.pin_path, root=root)
    report["residualArtifactLine"] = (
        f"snapshot={red['snapshotPath']} "
        f"frames={red['frameCount']} "
        f"frameDumpDir={red['frameDumpDir']}"
    )
    return report
