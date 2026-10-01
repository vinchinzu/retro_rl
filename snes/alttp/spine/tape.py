"""Live-run SNES button tapes.

Each ``env.step`` appends one 12-button vector in
``retro_harness.controls.SNES_BUTTON_NAMES`` order. ``save`` writes
``recordings/<tag>.tape.npz`` with the boot state name, ROM sha1, and
CRC32 of the final ``get_ram()`` buffer.

``.bk2`` import and ``LabeledRecorder`` do not store that triple, so this
follows the Zelda I ``.tape.npz`` layout with vectors instead of NES
bitmasks. Steps that bypass ``env.step`` (rollout lookahead) are not
recorded. The CRC is the whole RAM image; the replay sync window is later.

``LiveRun`` wraps the same env in ``AuditedEnv`` and reports ``set_state``,
``rollout_restores``, and RAM writes. ``memory.assign`` is read-only on the
stable-retro object, so the write count goes through a data proxy.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from alttp.paths import RECORDINGS_DIR, REPO_ROOT
from retro_harness.actions import SNES_ACTION_SIZE
from retro_harness.audit import AuditCapabilities, AuditedEnv
from retro_harness.controls import SNES_BUTTON_NAMES

TAPE_SUFFIX = ".tape.npz"

# Kinds a pin sidecar may name. ``poke`` / ``whatif`` / ``scan`` are not
# written by the spine; the set is closed so a later tool cannot invent one.
PIN_KINDS = frozenset({"natural", "resumed", "poke", "whatif", "scan"})

__all__ = [
    "PIN_KINDS",
    "TAPE_SUFFIX",
    "ButtonTape",
    "LiveRun",
    "RunAudit",
    "Tape",
    "load_tape",
    "pack_buttons",
    "provenance_path",
    "ram_crc",
    "read_pin_provenance",
    "sha1_file",
    "tape_path",
    "write_pin_provenance",
]


def pack_buttons(action: Any) -> np.ndarray:
    """One ``env.step`` action as a length-12 0/1 vector."""
    values = list(action)
    if len(values) < SNES_ACTION_SIZE:
        raise ValueError(
            f"SNES action needs {SNES_ACTION_SIZE} buttons, got {len(values)}"
        )
    vec = np.zeros(SNES_ACTION_SIZE, dtype=np.uint8)
    for index in range(SNES_ACTION_SIZE):
        if values[index]:
            vec[index] = 1
    return vec


def ram_crc(ram: Any) -> int:
    """CRC32 of the full RAM buffer passed in (not a sync window)."""
    raw = np.asarray(ram, dtype=np.uint8).tobytes()
    return zlib.crc32(raw) & 0xFFFFFFFF


def sha1_file(path: Path | str) -> str:
    """Lowercase SHA1 of a file. Does not boot the ROM."""
    digest = hashlib.sha1()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tape_path(tag: str, directory: Path | None = None) -> Path:
    """``recordings/<tag>.tape.npz``. ``tag`` is a single filename."""
    if not isinstance(tag, str) or not tag or tag in {".", ".."}:
        raise ValueError(f"tape tag must be a filename, got {tag!r}")
    if "/" in tag or "\\" in tag:
        raise ValueError(f"tape tag must be a single filename, got {tag!r}")
    stem = tag[: -len(TAPE_SUFFIX)] if tag.endswith(TAPE_SUFFIX) else tag
    if not stem or stem in {".", ".."}:
        raise ValueError(f"tape tag must be a filename, got {tag!r}")
    root = RECORDINGS_DIR if directory is None else Path(directory)
    return root / f"{stem}{TAPE_SUFFIX}"


@dataclass(frozen=True)
class Tape:
    """Loaded tape: one vector per step, plus the run identity fields."""

    buttons: Any
    boot_state: str
    rom_sha1: str
    final_ram_crc: int
    meta: dict[str, Any]


def load_tape(path: Path | str) -> Tape:
    """Load a tape written by :meth:`ButtonTape.save`."""
    with np.load(path, allow_pickle=False) as data:
        buttons = np.asarray(data["buttons"], dtype=np.uint8)
        meta = json.loads(str(data["meta"]))
        final_ram_crc = int(data["final_ram_crc"])
    if buttons.ndim != 2 or buttons.shape[1] != SNES_ACTION_SIZE:
        raise ValueError(
            f"buttons must be (N, {SNES_ACTION_SIZE}), got {buttons.shape}"
        )
    if int(meta["frames"]) != int(buttons.shape[0]):
        raise ValueError("meta frames does not match button rows")
    if int(meta["final_ram_crc"]) != final_ram_crc:
        raise ValueError("meta final_ram_crc does not match the array")
    return Tape(
        buttons=buttons,
        boot_state=str(meta["boot_state"]),
        rom_sha1=str(meta["rom_sha1"]),
        final_ram_crc=final_ram_crc,
        meta=meta,
    )


class ButtonTape:
    """Record every action ``env.step`` plays, then write the npz."""

    def __init__(self, *, boot_state: str, rom_sha1: str) -> None:
        if not boot_state or not isinstance(boot_state, str):
            raise ValueError("boot_state is required")
        if not _is_sha1(rom_sha1):
            raise ValueError("rom_sha1 must be 40 lowercase hex chars")
        self.boot_state = boot_state
        self.rom_sha1 = rom_sha1
        self._frames: list[np.ndarray] = []
        self._env: Any = None
        self._orig_step: Any = None

    def record(self, action: Any) -> None:
        self._frames.append(pack_buttons(action))

    def attach(self, env: Any) -> None:
        """Wrap ``env.step`` so each live step is packed onto the tape."""
        if self._orig_step is not None:
            raise RuntimeError("ButtonTape is already attached")
        orig = env.step

        def _step(action, *args, **kwargs):
            self.record(action)
            return orig(action, *args, **kwargs)

        self._env = env
        self._orig_step = orig
        env.step = _step

    def detach(self) -> None:
        if self._env is not None and self._orig_step is not None:
            self._env.step = self._orig_step
        self._env = None
        self._orig_step = None

    def save(
        self,
        tag: str,
        *,
        ram: Any,
        directory: Path | None = None,
    ) -> Path:
        """Write ``recordings/<tag>.tape.npz`` and return the path."""
        path = tape_path(tag, directory)
        if self._frames:
            buttons = np.stack(self._frames)
        else:
            buttons = np.zeros((0, SNES_ACTION_SIZE), dtype=np.uint8)
        crc = ram_crc(ram)
        meta = {
            "frames": int(buttons.shape[0]),
            "boot_state": self.boot_state,
            "rom_sha1": self.rom_sha1,
            "final_ram_crc": crc,
            "button_order": list(SNES_BUTTON_NAMES),
            "tag": path.name[: -len(TAPE_SUFFIX)],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            buttons=buttons,
            meta=np.asarray(json.dumps(meta)),
            final_ram_crc=np.asarray(crc, dtype=np.uint32),
        )
        return path


def _is_sha1(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 40:
        return False
    return all(ch in "0123456789abcdef" for ch in value)


def _caller_site() -> str:
    """First frame outside this module and ``retro_harness.audit``."""
    for info in inspect.stack()[1:]:
        path = info.filename.replace("\\", "/")
        if path.endswith("/tape.py") or path.endswith("/audit.py"):
            continue
        return f"{Path(path).name}:{info.lineno}:{info.function}"
    return "unknown"


class _MemoryProxy:
    """Count ``memory.assign``. The C method itself cannot be replaced."""

    def __init__(self, memory: Any, owner: "_AlttpAuditedEnv") -> None:
        self._memory = memory
        self._owner = owner

    def assign(self, *args: Any, **kwargs: Any) -> Any:
        return self._owner._count_write(self._memory.assign, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._memory, name)


class _DataProxy:
    def __init__(self, data: Any, owner: "_AlttpAuditedEnv") -> None:
        self._data = data
        self._owner = owner
        memory = getattr(data, "memory", None)
        self._memory = _MemoryProxy(memory, owner) if memory is not None else None

    @property
    def memory(self) -> Any:
        if self._memory is None:
            return self._data.memory
        return self._memory

    def set_value(self, key: str, value: Any) -> Any:
        return self._owner._count_write(self._data.set_value, key, value)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._data, name)


class _AlttpAuditedEnv(AuditedEnv):
    """Keep SNES write and load interfaces audited through ``unwrapped``."""

    def __init__(
        self,
        env: Any,
        *,
        capabilities: AuditCapabilities,
        sites: list[str],
    ) -> None:
        super().__init__(env, capabilities=capabilities)
        self._sites = sites
        self._in_write = False
        raw = getattr(env, "data", None)
        self._data_proxy = _DataProxy(raw, self) if raw is not None else None
        self._trace_set_state()

    @property
    def unwrapped(self) -> "_AlttpAuditedEnv":
        # RAM and emulator helpers commonly unwrap before touching the core.
        # Returning the raw env here would bypass both owned audit interfaces.
        return self

    @property
    def data(self) -> Any:
        if self._data_proxy is not None:
            return self._data_proxy
        return super().data

    def set_value(self, name: str, value: Any) -> Any:
        if self._data_proxy is None:
            raise AttributeError("wrapped environment has no data interface")
        return self._data_proxy.set_value(name, value)

    def _count_write(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        if self._in_write:
            return fn(*args, **kwargs)
        self._in_write = True
        try:
            self.record_ram_write()
            return fn(*args, **kwargs)
        finally:
            self._in_write = False

    def _trace_set_state(self) -> None:
        wrapper = self._emulator
        if wrapper is None:
            return
        orig = wrapper.set_state

        def set_state(state: bytes) -> Any:
            active = self._attempt_active
            site = _caller_site() if active else ""
            result = orig(state)
            if active:
                self._sites.append(site)
            return result

        wrapper.set_state = set_state


@dataclass(frozen=True)
class RunAudit:
    """One live run's load and write counts."""

    set_state: int
    rollout_restores: int
    writes: int
    sites: tuple[str, ...] = ()

    @property
    def mid_run_state_load(self) -> bool:
        """True when loads minus rollout restores is above 0."""
        return self.set_state - self.rollout_restores > 0

    def line(self) -> str:
        return (
            f"set_state={self.set_state} "
            f"rollout_restores={self.rollout_restores} "
            f"writes={self.writes}"
        )


class LiveRun:
    """Wrap a live env. Fail when non-rollout ``set_state`` calls remain.

    ``rollout_restores`` is disclosed by the caller. ``alttp.rollout`` records
    a frame budget only; it does not count restores yet.
    """

    def __init__(self, env: Any, *, provider: str = "alttp.live") -> None:
        self._sites: list[str] = []
        self.rollout_restores = 0
        self.env = _AlttpAuditedEnv(
            env,
            capabilities=AuditCapabilities.all(provider),
            sites=self._sites,
        )

    def note_rollout_restores(self, count: int) -> None:
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("rollout_restores must be a non-negative integer")
        self.rollout_restores = count

    def report(self) -> RunAudit:
        audit = self.env.audit()
        return RunAudit(
            set_state=int(audit.mid_run_loads or 0),
            rollout_restores=self.rollout_restores,
            writes=int(audit.ram_writes or 0),
            sites=tuple(self._sites),
        )

    def fail_mid_run_state_load(self) -> RunAudit:
        """Raise when ``set_state - rollout_restores`` is above 0."""
        report = self.report()
        other = report.set_state - report.rollout_restores
        if other < 0:
            raise ValueError(
                "rollout_restores exceeds set_state "
                f"set_state={report.set_state} "
                f"rollout_restores={report.rollout_restores}"
            )
        if other > 0:
            raise RuntimeError(
                "mid_run_state_load "
                f"set_state={report.set_state} "
                f"rollout_restores={report.rollout_restores} "
                f"writes={report.writes}"
            )
        return report


def provenance_path(state_path: Path | str) -> Path:
    """Sidecar next to a ``*.state`` pin."""
    return Path(state_path).with_suffix(".provenance.json")


def _git_identity() -> tuple[str, bool]:
    """HEAD sha and whether the worktree has anything ``git status`` would show."""
    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        text=True,
    )
    return sha, bool(status.strip())


def write_pin_provenance(
    state_path: Path | str,
    *,
    source_run: str,
    frame: int,
    kind: str,
) -> Path:
    """Write ``<pin>.provenance.json`` for one pin a tool just saved.

    ``kind`` is ``natural``, ``resumed``, ``poke``, ``whatif``, or ``scan``.
    ``frame`` is the step count on that run at the moment of the save.
    """
    if kind not in PIN_KINDS:
        raise ValueError(f"pin kind must be one of {sorted(PIN_KINDS)}, got {kind!r}")
    if not isinstance(source_run, str) or not source_run:
        raise ValueError(f"source run tag must be a non-empty string, got {source_run!r}")
    if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
        raise ValueError(f"frame must be a non-negative int, got {frame!r}")
    sha, dirty = _git_identity()
    payload = {
        "source_run": source_run,
        "frame": frame,
        "git_sha": sha,
        "dirty": dirty,
        "kind": kind,
    }
    sidecar = provenance_path(state_path)
    sidecar.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return sidecar


def read_pin_provenance(state_path: Path | str) -> dict[str, Any] | None:
    """Load a pin sidecar, or None when the pin has none."""
    sidecar = provenance_path(state_path)
    if not sidecar.is_file():
        return None
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"provenance is not an object: {sidecar}")
    return data
