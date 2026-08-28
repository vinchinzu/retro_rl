"""Process-group kill and Popen helpers for overnight PPO workers."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, IO

TERMINATE_GRACE_SECONDS = 2.0
KILL_REAP_SECONDS = 1.0


def spawn_worker(
    cmd: list[str], log_path: Path, *, cycle: int
) -> tuple[IO[bytes], subprocess.Popen[bytes]]:
    """Start one training subprocess in its own session; log stdout/stderr."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("ab")
    handle.write(
        f"\n--- cycle {cycle} {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n".encode()
    )
    handle.flush()
    proc = subprocess.Popen(
        cmd,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=os.name == "posix",
    )
    return handle, proc


def signal_worker_tree(proc: Any, sig: signal.Signals) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(proc.pid, sig)
        elif sig == signal.SIGTERM:
            proc.terminate()
        else:
            proc.kill()
    except ProcessLookupError:
        pass


def terminate_worker_trees(procs: list[Any]) -> None:
    """Stop all workers together, including their emulator descendants."""

    for proc in procs:
        signal_worker_tree(proc, signal.SIGTERM)

    grace_deadline = time.monotonic() + TERMINATE_GRACE_SECONDS
    for proc in procs:
        if proc.poll() is not None:
            continue
        try:
            proc.wait(timeout=max(0.0, grace_deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass

    for proc in procs:
        signal_worker_tree(proc, signal.SIGKILL)

    reap_deadline = time.monotonic() + KILL_REAP_SECONDS
    for proc in procs:
        if proc.poll() is not None:
            continue
        try:
            proc.wait(timeout=max(0.0, reap_deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass


__all__ = [
    "KILL_REAP_SECONDS",
    "TERMINATE_GRACE_SECONDS",
    "signal_worker_tree",
    "spawn_worker",
    "terminate_worker_trees",
]
