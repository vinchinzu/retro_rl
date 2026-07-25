"""Cursor SDK helpers for in-editor agent sessions."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
import json
import os
from typing import Any

from retro_harness.editor.snapshot import snapshot_without_frame


@dataclass(frozen=True)
class EditorAgentContext:
    """Structured editor state published to a Cursor agent prompt."""

    title: str
    summary: str
    details: dict[str, object] = field(default_factory=dict)


def compact_snapshot(snapshot: dict[str, object]) -> dict[str, object]:
    """Drop bulky frame bytes from bridge snapshots before agent prompts."""

    return snapshot_without_frame(snapshot)


def format_editor_context(context: EditorAgentContext) -> str:
    """Render editor context as markdown for agent prompts."""

    lines = [
        f"### {context.title}",
        "",
        context.summary.strip(),
        "",
        "```json",
        json.dumps(context.details, indent=2, sort_keys=True, default=str),
        "```",
    ]
    return "\n".join(lines)


def build_agent_prompt(
    user_message: str,
    *,
    instructions: tuple[str, ...] = (),
    context: EditorAgentContext | None = None,
    published_context: str | None = None,
) -> str:
    """Combine system instructions, editor context, and the user request."""

    parts: list[str] = []
    if instructions:
        parts.extend(instructions)
        parts.append("")
    if published_context and published_context.strip():
        parts.extend(["## Published Context", "", published_context.strip(), ""])
    if context is not None:
        parts.extend(["## Live Editor Context", "", format_editor_context(context), ""])
    parts.extend(["## User Request", "", user_message.strip()])
    return "\n".join(parts)


def format_sdk_message(message: object) -> list[str]:
    """Convert one SDK message into terminal-style lines."""

    message_type = getattr(message, "type", "")
    if message_type == "assistant":
        lines: list[str] = []
        content = getattr(getattr(message, "message", None), "content", ())
        for block in content:
            block_type = getattr(block, "type", "")
            if block_type == "text":
                text = getattr(block, "text", "")
                if text:
                    lines.append(text.rstrip())
            else:
                name = getattr(block, "name", "tool")
                lines.append(f"[tool request] {name}")
        return lines
    if message_type == "thinking":
        text = getattr(message, "text", "")
        if text:
            return [f"[thinking] {text.rstrip()}"]
        return []
    if message_type == "tool_call":
        name = getattr(message, "name", "tool")
        status = getattr(message, "status", "running")
        return [f"[tool {status}] {name}"]
    if message_type == "status":
        status = getattr(message, "status", "")
        detail = getattr(message, "message", "")
        label = f"[status {status}]"
        if detail:
            return [f"{label} {detail}"]
        return [label] if status else []
    if message_type == "task":
        status = getattr(message, "status", "")
        text = getattr(message, "text", "")
        prefix = f"[task {status}]" if status else "[task]"
        return [f"{prefix} {text}".rstrip()] if text else ([prefix] if status else [])
    return []


def stream_sdk_lines(messages: Iterator[object]) -> Iterator[str]:
    """Yield formatted terminal lines from an SDK message stream."""

    for message in messages:
        for line in format_sdk_message(message):
            yield line


def default_api_key() -> str:
    """Return the configured Cursor API key, if any."""

    return os.environ.get("CURSOR_API_KEY", "").strip()


def cursor_sdk_available() -> bool:
    """Return True when the optional cursor-sdk package is importable."""

    try:
        import cursor_sdk  # noqa: F401
    except ImportError:
        return False
    return True


def list_model_ids(api_key: str) -> list[str]:
    """Return model ids available to the caller, with safe fallbacks."""

    fallback = ["composer-2.5", "auto"]
    if not api_key.strip():
        return fallback
    try:
        from cursor_sdk import Cursor

        models = Cursor.models.list(api_key=api_key.strip())
    except Exception:
        return fallback
    ids = [str(getattr(model, "id", "") or "") for model in models]
    ids = [model_id for model_id in ids if model_id]
    return ids or fallback


def validate_api_key(api_key: str) -> None:
    """Raise when the API key cannot be validated."""

    from cursor_sdk import Cursor

    Cursor.me(api_key=api_key.strip())


@dataclass
class CursorAgentRunResult:
    """Terminal result from one agent run."""

    status: str
    result: str
    run_id: str
    agent_id: str


class CursorAgentSession:
    """Sync Cursor agent session used from a dedicated worker thread."""

    def __init__(
        self,
        *,
        api_key: str,
        workspace_cwd: str,
        model: str,
        name: str,
        instructions: tuple[str, ...] = (),
    ) -> None:
        from cursor_sdk import Agent, LocalAgentOptions

        self._instructions = instructions
        self._agent = Agent.create(
            api_key=api_key.strip(),
            model=model,
            name=name,
            local=LocalAgentOptions(cwd=workspace_cwd),
        )
        self._current_run: Any | None = None

    @property
    def agent_id(self) -> str:
        return str(self._agent.agent_id)

    def send(
        self,
        user_message: str,
        *,
        context: EditorAgentContext | None = None,
        published_context: str | None = None,
    ) -> Iterator[str]:
        """Start a run and stream terminal lines until it finishes."""

        prompt = build_agent_prompt(
            user_message,
            instructions=self._instructions,
            context=context,
            published_context=published_context,
        )
        run = self._agent.send(prompt)
        self._current_run = run
        yield f"[run started] {run.id} agent={self.agent_id}"
        try:
            for line in stream_sdk_lines(run.messages()):
                yield line
            result = run.wait()
            if result.status == "error":
                yield f"[run error] {result.id}"
            else:
                yield f"[run finished] {result.status}"
            if result.result:
                yield result.result.rstrip()
        finally:
            self._current_run = None

    def cancel(self) -> None:
        run = self._current_run
        if run is None:
            return
        if run.supports("cancel"):
            run.cancel()

    def close(self) -> None:
        self.cancel()
        self._agent.close()
