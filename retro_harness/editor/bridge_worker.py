"""Background-thread bridge IPC so the Qt UI thread never blocks on stdio."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Callable

from PySide6.QtCore import QObject, QThread, Signal, Slot

from retro_harness.editor.bridge_client import EditorBridgeClient


@dataclass(frozen=True)
class BridgeRequest:
    """One bridge command posted to the worker thread."""

    request_id: str
    command: str
    kwargs: dict[str, object]


@dataclass(frozen=True)
class BridgeReply:
    """Worker-thread response for a posted command."""

    request_id: str
    command: str
    response: dict[str, object] | None


class BridgeWorker(QObject):
    """Owns ``EditorBridgeClient`` and executes commands on a worker thread."""

    request = Signal(object)
    reply = Signal(object)
    bridge_stopped = Signal(str)

    def __init__(
        self,
        *,
        project_root: object,
        bridge_module: str,
        on_disconnect: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__()
        self._client = EditorBridgeClient(
            project_root=project_root,  # type: ignore[arg-type]
            bridge_module=bridge_module,
            on_disconnect=lambda message: self.bridge_stopped.emit(message),
        )
        self._external_disconnect = on_disconnect
        self.request.connect(self._handle_request)
        self.bridge_stopped.connect(self._forward_disconnect)

    @Slot(str)
    def _forward_disconnect(self, message: str) -> None:
        if self._external_disconnect is not None:
            self._external_disconnect(message)

    @Slot()
    def connect_bridge(self) -> None:
        if self._client.is_connected():
            return
        self._client.start()

    @Slot()
    def disconnect_bridge(self) -> None:
        self._client.stop()

    @Slot(object)
    def _handle_request(self, payload: object) -> None:
        if not isinstance(payload, BridgeRequest):
            return
        response = self._client.send_command(payload.command, **payload.kwargs)
        self.reply.emit(
            BridgeReply(
                request_id=payload.request_id,
                command=payload.command,
                response=response,
            )
        )


class BridgeController(QObject):
    """Main-thread facade for async and blocking bridge commands."""

    reply = Signal(object)

    def __init__(
        self,
        *,
        project_root: object,
        bridge_module: str,
        on_disconnect: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__()
        self._thread = QThread()
        self._worker = BridgeWorker(
            project_root=project_root,
            bridge_module=bridge_module,
            on_disconnect=on_disconnect,
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.connect_bridge)
        self._worker.reply.connect(self.reply)
        self._sync_waiters: dict[str, Callable[[BridgeReply], None]] = {}
        self.reply.connect(self._dispatch_reply)

    def start(self) -> None:
        if self._thread.isRunning():
            return
        self._thread.start()

    def stop(self) -> None:
        if not self._thread.isRunning():
            return
        self._worker.disconnect_bridge()
        self._thread.quit()
        self._thread.wait(3000)
        self._sync_waiters.clear()

    def is_connected(self) -> bool:
        return self._thread.isRunning() and self._worker._client.is_connected()

    def post(self, command: str, **kwargs: object) -> str:
        """Queue a command on the worker thread; reply arrives via ``reply`` signal."""

        request_id = str(uuid.uuid4())[:8]
        self._worker.request.emit(
            BridgeRequest(request_id=request_id, command=command, kwargs=dict(kwargs))
        )
        return request_id

    def call(self, command: str, **kwargs: object) -> dict[str, object] | None:
        """Block until the worker returns a response (for start/stop/hot-save UI)."""

        from PySide6.QtCore import QEventLoop

        request_id = str(uuid.uuid4())[:8]
        loop = QEventLoop()
        holder: list[dict[str, object] | None] = [None]

        def accept(reply: BridgeReply) -> None:
            if reply.request_id != request_id:
                return
            holder[0] = reply.response
            loop.quit()

        self._sync_waiters[request_id] = accept
        self._worker.request.emit(
            BridgeRequest(request_id=request_id, command=command, kwargs=dict(kwargs))
        )
        loop.exec()
        self._sync_waiters.pop(request_id, None)
        return holder[0]

    @Slot(object)
    def _dispatch_reply(self, payload: object) -> None:
        if not isinstance(payload, BridgeReply):
            return
        waiter = self._sync_waiters.get(payload.request_id)
        if waiter is not None:
            waiter(payload)


__all__ = [
    "BridgeController",
    "BridgeReply",
    "BridgeRequest",
    "BridgeWorker",
]
