"""Thin Ollama HTTP client (chat, tools, optional vision)."""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib import error, request


@dataclass(frozen=True)
class OllamaConfig:
    """Connection settings for a local Ollama daemon."""

    host: str = "http://127.0.0.1:11434"
    model: str = "gemma4:12b"
    timeout: float = 180.0
    temperature: float = 0.2
    # gemma4 defaults to thinking; without this, content is often empty.
    think: bool = False
    num_predict: int = 512


@dataclass
class ChatMessage:
    """One Ollama chat message (user/assistant/tool/system)."""

    role: str
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_name: str | None = None
    images: list[str] = field(default_factory=list)

    def to_api(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            payload["tool_calls"] = self.tool_calls
        if self.tool_name:
            payload["tool_name"] = self.tool_name
        if self.images:
            payload["images"] = self.images
        return payload


@dataclass
class ChatTurn:
    """Parsed assistant turn from ``/api/chat``."""

    message: ChatMessage
    raw: dict[str, Any]


class OllamaError(RuntimeError):
    """Raised when the local model call fails."""


def chat_turn(
    *,
    config: OllamaConfig,
    messages: Sequence[ChatMessage | Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]] | None = None,
    format_json: bool = False,
) -> ChatTurn:
    """One multi-turn chat step; may return tool_calls and/or content."""
    api_messages = [
        m.to_api() if isinstance(m, ChatMessage) else dict(m) for m in messages
    ]
    payload: dict[str, Any] = {
        "model": config.model,
        "stream": False,
        "think": config.think,
        "options": {
            "temperature": config.temperature,
            "num_predict": config.num_predict,
        },
        "messages": api_messages,
    }
    if tools:
        payload["tools"] = list(tools)
    if format_json:
        payload["format"] = "json"
    response = _post_json(
        f"{config.host.rstrip('/')}/api/chat",
        payload,
        timeout=config.timeout,
    )
    message = _parse_message(response.get("message", {}))
    return ChatTurn(message=message, raw=response)


def chat_json(
    *,
    config: OllamaConfig,
    system: str,
    user: str,
    images: Sequence[Path] | None = None,
) -> dict[str, Any]:
    """Call ``/api/chat`` and parse a JSON object from the assistant text."""
    message = ChatMessage(role="user", content=user)
    if images:
        message.images = [_b64(path) for path in images]
    turn = chat_turn(
        config=config,
        messages=[
            ChatMessage(role="system", content=system),
            message,
        ],
        format_json=True,
    )
    return parse_json_object(turn.message.content or _assistant_fallback(turn))


def encode_images(paths: Sequence[Path]) -> list[str]:
    """Base64-encode image files for Ollama vision messages."""
    return [_b64(path) for path in paths]


def parse_json_object(text: str) -> dict[str, Any]:
    """Extract the first JSON object from model output."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
    if not cleaned.startswith("{"):
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise OllamaError(f"no JSON object in model output: {text[:240]!r}")
        cleaned = match.group(0)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise OllamaError(f"invalid JSON from model: {exc}") from exc
    if not isinstance(parsed, dict):
        raise OllamaError("model JSON was not an object")
    return parsed


def _parse_message(raw: Any) -> ChatMessage:
    if not isinstance(raw, Mapping):
        return ChatMessage(role="assistant", content="")
    content = str(raw.get("content", "") or "").strip()
    if not content:
        thinking = str(raw.get("thinking", "") or "").strip()
        content = thinking
    tool_calls = raw.get("tool_calls") or []
    if not isinstance(tool_calls, list):
        tool_calls = []
    return ChatMessage(
        role=str(raw.get("role", "assistant")),
        content=content,
        tool_calls=[dict(call) for call in tool_calls if isinstance(call, Mapping)],
        tool_name=(
            str(raw["tool_name"]) if raw.get("tool_name") is not None else None
        ),
    )


def _assistant_fallback(turn: ChatTurn) -> str:
    return turn.message.content


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _post_json(
    url: str,
    payload: Mapping[str, Any],
    *,
    timeout: float,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as response:
            data = response.read().decode("utf-8")
    except error.URLError as exc:
        raise OllamaError(f"ollama request failed: {exc}") from exc
    except TimeoutError as exc:
        raise OllamaError(f"ollama request timed out: {exc}") from exc
    parsed = json.loads(data)
    if not isinstance(parsed, dict):
        raise OllamaError("ollama response was not a JSON object")
    return parsed


__all__ = [
    "ChatMessage",
    "ChatTurn",
    "OllamaConfig",
    "OllamaError",
    "chat_json",
    "chat_turn",
    "encode_images",
    "parse_json_object",
]
