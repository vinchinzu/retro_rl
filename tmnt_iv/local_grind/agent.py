"""Multi-turn Ollama tool agent for TMNT grind experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from tmnt_iv.local_grind.ollama_client import (
    ChatMessage,
    OllamaConfig,
    OllamaError,
    chat_turn,
)
from tmnt_iv.local_grind.tools import TOOL_SPECS, GrindToolbox

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


@dataclass
class AgentConfig:
    """Runtime knobs for one agent session."""

    model: str = "gemma4:12b"
    host: str = "http://127.0.0.1:11434"
    focus: str = "slash"
    out_dir: Path = Path("tmnt_iv/recordings/local_grind_agent")
    max_trials: int = 3
    max_turns: int = 24
    min_rel_gain: float = 0.01
    temperature: float = 0.2
    timeout: float = 300.0
    num_predict: int = 768
    screenshot_every: int = 900
    max_screenshots: int = 3


@dataclass
class AgentResult:
    """Inspectable session outcome."""

    finished: bool
    summary: str
    turns: int
    tool_calls: int
    trials: int
    keeps: int
    out_dir: Path
    status: dict[str, Any] = field(default_factory=dict)


def run_grind_agent(
    config: AgentConfig,
    *,
    toolbox: GrindToolbox | None = None,
    chat_fn: Callable[..., Any] | None = None,
) -> AgentResult:
    """Run a tool-calling agent until finish / budgets / stuck."""
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    box = toolbox or GrindToolbox(
        out_dir=out_dir,
        focus=config.focus,
        max_trials=config.max_trials,
        min_rel_gain=config.min_rel_gain,
        screenshot_every=config.screenshot_every,
        max_screenshots=config.max_screenshots,
    )
    ollama = OllamaConfig(
        host=config.host,
        model=config.model,
        timeout=config.timeout,
        temperature=config.temperature,
        think=False,
        num_predict=config.num_predict,
    )
    chat = chat_fn or (
        lambda messages, tools: chat_turn(
            config=ollama,
            messages=messages,
            tools=tools,
        )
    )

    system = (PROMPTS_DIR / "agent_system.md").read_text(encoding="utf-8")
    messages: list[ChatMessage] = [
        ChatMessage(role="system", content=system),
        ChatMessage(
            role="user",
            content=(
                f"Focus target: {config.focus}. "
                f"Trial budget: {config.max_trials}. "
                "Use tools. Start by listing knobs and running baseline, "
                "then run useful trials, then finish."
            ),
        ),
    ]
    trace_path = out_dir / "agent_trace.jsonl"
    if trace_path.exists():
        trace_path.unlink()

    turns = 0
    tool_calls = 0
    idle_no_tool = 0

    while turns < config.max_turns and not box.finished:
        turns += 1
        try:
            turn = chat(messages, TOOL_SPECS)
        except OllamaError as exc:
            _append_trace(
                trace_path,
                {"turn": turns, "error": str(exc), "phase": "chat"},
            )
            messages.append(
                ChatMessage(
                    role="user",
                    content=(
                        f"Model/transport error: {exc}. "
                        "Call get_status or finish."
                    ),
                )
            )
            continue

        assistant = turn.message
        messages.append(assistant)
        _append_trace(
            trace_path,
            {
                "turn": turns,
                "role": "assistant",
                "content": assistant.content,
                "tool_calls": assistant.tool_calls,
            },
        )
        print(f"[agent turn {turns}] tools={len(assistant.tool_calls)}")

        if not assistant.tool_calls:
            idle_no_tool += 1
            if idle_no_tool >= 2 or "finish" in assistant.content.lower():
                box.finish(assistant.content or "stopped without tool finish")
                break
            messages.append(
                ChatMessage(
                    role="user",
                    content=(
                        "You must call tools (run_baseline / run_trial / "
                        "finish). Do not only chat."
                    ),
                )
            )
            continue
        idle_no_tool = 0

        for call in assistant.tool_calls:
            tool_calls += 1
            name, args, call_id = _parse_tool_call(call)
            print(f"  -> {name}({_short_args(args)})")
            result = box.dispatch(name, args)
            result_text = json.dumps(result, sort_keys=True)
            messages.append(
                ChatMessage(
                    role="tool",
                    tool_name=name,
                    content=result_text,
                )
            )
            _append_trace(
                trace_path,
                {
                    "turn": turns,
                    "role": "tool",
                    "tool": name,
                    "call_id": call_id,
                    "arguments": args,
                    "result": result,
                },
            )
            if name == "finish" or box.finished:
                break

        if box.finished:
            break
        if len(box.trials) >= config.max_trials:
            # Nudge one last chance to finish.
            messages.append(
                ChatMessage(
                    role="user",
                    content=(
                        "Trial budget exhausted. Call finish with a summary "
                        "of baseline vs trials and whether any KEEP landed."
                    ),
                )
            )

    if not box.finished:
        box.finish(
            box.finish_summary
            or "stopped: turn/trial budget without explicit finish"
        )

    status = box.get_status()
    result = AgentResult(
        finished=box.finished,
        summary=box.finish_summary,
        turns=turns,
        tool_calls=tool_calls,
        trials=len(box.trials),
        keeps=int(status.get("keeps", 0)),
        out_dir=out_dir,
        status=status,
    )
    _write_json(
        out_dir / "agent_result.json",
        {
            "finished": result.finished,
            "summary": result.summary,
            "turns": result.turns,
            "tool_calls": result.tool_calls,
            "trials": result.trials,
            "keeps": result.keeps,
            "status": result.status,
        },
    )
    return result


def _parse_tool_call(call: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
    call_id = str(call.get("id", ""))
    function = call.get("function") or {}
    if not isinstance(function, dict):
        return "unknown", {}, call_id
    name = str(function.get("name", "unknown"))
    raw_args = function.get("arguments", {})
    if isinstance(raw_args, str):
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            args = {}
    elif isinstance(raw_args, dict):
        args = raw_args
    else:
        args = {}
    if not isinstance(args, dict):
        args = {}
    return name, args, call_id


def _short_args(args: dict[str, Any]) -> str:
    text = json.dumps(args, sort_keys=True)
    return text if len(text) <= 120 else text[:117] + "..."


def _append_trace(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


__all__ = ["AgentConfig", "AgentResult", "run_grind_agent"]
