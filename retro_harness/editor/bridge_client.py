"""JSON-line stdio client for ``<package>.editor_bridge --stdio``."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, BinaryIO, Callable


def read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = int(size)
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError(f"expected {size} bytes, got {size - remaining}")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def bridge_python_command(*, project_root: Path) -> list[str]:
    from retro_harness.repo import monorepo_root

    repo = monorepo_root()
    for candidate in (
        repo / ".venv" / "bin" / "python",
        project_root / ".venv" / "bin" / "python",
        project_root.parent / ".venv" / "bin" / "python",
    ):
        if candidate.exists():
            return [str(candidate)]
    uv_path = shutil.which("uv")
    if uv_path is not None and (repo / "pyproject.toml").exists():
        return [uv_path, "run", "--project", str(repo), "python"]
    if os.environ.get("PYTHON"):
        return [os.environ["PYTHON"]]
    return [sys.executable]


class EditorBridgeClient:
    """Manage a long-lived ``editor_bridge --stdio`` subprocess."""

    def __init__(
        self,
        *,
        project_root: Path,
        bridge_module: str,
        on_disconnect: Callable[[str], None] | None = None,
    ) -> None:
        self._project_root = project_root
        self._bridge_module = bridge_module
        self._on_disconnect = on_disconnect
        self._process: subprocess.Popen[bytes] | None = None

    @property
    def process(self) -> subprocess.Popen[bytes] | None:
        return self._process

    def is_connected(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self) -> None:
        if self.is_connected():
            return
        env = dict(os.environ)
        root = str(self._project_root)
        env["PYTHONPATH"] = f"{root}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else root
        parent = str(self._project_root.parent)
        if parent not in env["PYTHONPATH"].split(os.pathsep):
            env["PYTHONPATH"] = f"{parent}{os.pathsep}{env['PYTHONPATH']}"
        self._process = subprocess.Popen(
            [
                *bridge_python_command(project_root=self._project_root),
                "-m",
                self._bridge_module,
                "--stdio",
            ],
            cwd=str(self._project_root),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        self.send_command("hello", includeFrame=False)
        self.send_command("discover", includeFrame=False)

    def stop(self) -> None:
        if self._process is None:
            return
        process = self._process
        try:
            self.send_command("close_session", includeFrame=False)
        except Exception:
            pass
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        for stream in (process.stdin, process.stdout):
            if stream is None:
                continue
            try:
                stream.close()
            except OSError:
                pass
        self._process = None

    def _disconnect(self, message: str) -> None:
        self._process = None
        if self._on_disconnect is not None:
            self._on_disconnect(message)

    def send_command(self, command: str, **kwargs: Any) -> dict[str, object] | None:
        if self._process is None or self._process.stdin is None or self._process.stdout is None:
            return None
        if self._process.poll() is not None:
            self._disconnect("Bridge exited")
            return None
        payload = {"id": str(uuid.uuid4())[:8], "command": command, **kwargs}
        try:
            stdin = self._process.stdin
            stdout = self._process.stdout
            assert stdin is not None
            assert stdout is not None
            stdin.write((json.dumps(payload) + "\n").encode("utf-8"))
            stdin.flush()
            line = stdout.readline()
        except (BrokenPipeError, OSError):
            self._disconnect("Bridge crashed")
            return None
        if not line:
            self._disconnect("Bridge closed")
            return None
        try:
            response: dict[str, object] = json.loads(line.decode("utf-8"))
            frame_len = response.get("frameBinaryLength")
            if frame_len:
                raw = read_exact(stdout, int(frame_len))
                snapshot = response.get("snapshot")
                if isinstance(snapshot, dict):
                    snapshot["frameRgb24Raw"] = raw
            wram_len = response.get("wramBinaryLength")
            if wram_len:
                raw = read_exact(stdout, int(wram_len))
                snapshot = response.get("snapshot")
                if isinstance(snapshot, dict):
                    snapshot["wramRaw"] = raw
            return response
        except (json.JSONDecodeError, EOFError, ValueError):
            return None
