"""Unit tests for the tool-agent grind scaffold (no emulator / no Ollama)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tmnt_iv.local_grind.agent import AgentConfig, run_grind_agent
from tmnt_iv.local_grind.ollama_client import ChatMessage, ChatTurn
from tmnt_iv.local_grind.tools import GrindToolbox


def _fake_probe(**kwargs: Any) -> tuple[dict[str, Any], list[Path]]:
    knobs = kwargs.get("knobs") or {}
    # A specific patch "wins" so the scripted agent can KEEP something.
    winning = knobs.get("slash_approach_band") == 44
    metrics = {
        "state": kwargs["state_name"],
        "outcome": "stage_advance",
        "frames": 10_000 if winning else 13_000,
        "damage_taken": 400 if winning else 700,
        "heals": 5 if winning else 10,
        "min_hp": 20,
        "boss_hp": "160->0",
        "top_reasons": [("slash_attack", 100)],
    }
    return metrics, []


class _ScriptedChat:
    """Deterministic tool-call sequence standing in for Ollama."""

    def __init__(self) -> None:
        self.step = 0
        self.calls: list[str] = []

    def __call__(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]],
    ) -> ChatTurn:
        del tools
        self.step += 1
        plan = [
            [
                _tool("list_knobs", {"focus": "slash"}),
                _tool("run_baseline", {"target_label": "slash"}),
            ],
            [
                _tool(
                    "run_trial",
                    {
                        "target_label": "slash",
                        "hypothesis": "tighter approach",
                        "knobs": {"slash_approach_band": 44},
                        "rationale": "scripted keep",
                    },
                )
            ],
            [
                _tool("inspect_trial", {"trial_id": 1}),
                _tool(
                    "finish",
                    {"summary": "kept slash_approach_band=44"},
                ),
            ],
        ]
        idx = min(self.step - 1, len(plan) - 1)
        calls = plan[idx]
        for call in calls:
            self.calls.append(call["function"]["name"])
        return ChatTurn(
            message=ChatMessage(
                role="assistant",
                content="",
                tool_calls=calls,
            ),
            raw={},
        )


def _tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": f"call_{name}",
        "function": {"name": name, "arguments": arguments},
    }


def test_toolbox_run_trial_keep_and_discard(tmp_path: Path) -> None:
    box = GrindToolbox(
        out_dir=tmp_path,
        focus="slash",
        max_trials=3,
        probe_fn=_fake_probe,
    )
    base = box.run_baseline("slash")
    assert base["ok"] is True
    assert base["metrics"]["frames"] == 13_000

    bad = box.run_trial(
        target_label="slash",
        hypothesis="noop-ish",
        knobs={"slash_spin_dodge_adx": 52},
    )
    assert bad["ok"] is True
    assert bad["decision"] == "discard"

    good = box.run_trial(
        target_label="slash",
        hypothesis="tighter approach",
        knobs={"slash_approach_band": 44},
    )
    assert good["decision"] == "keep"
    assert box.best_knobs.slash_approach_band == 44


def test_agent_loop_with_scripted_tools(tmp_path: Path) -> None:
    chat = _ScriptedChat()
    box = GrindToolbox(
        out_dir=tmp_path,
        focus="slash",
        max_trials=2,
        probe_fn=_fake_probe,
    )
    result = run_grind_agent(
        AgentConfig(
            focus="slash",
            out_dir=tmp_path,
            max_trials=2,
            max_turns=8,
        ),
        toolbox=box,
        chat_fn=chat,
    )
    assert result.finished is True
    assert result.trials == 1
    assert result.keeps == 1
    assert "list_knobs" in chat.calls
    assert "run_baseline" in chat.calls
    assert "run_trial" in chat.calls
    assert "finish" in chat.calls
    assert (tmp_path / "agent_trace.jsonl").exists()
    assert (tmp_path / "agent_result.json").exists()
    assert (tmp_path / "summary.json").exists()


def test_run_trial_rejects_bare_knob_list(tmp_path: Path) -> None:
    box = GrindToolbox(
        out_dir=tmp_path,
        focus="slash",
        probe_fn=_fake_probe,
    )
    box.run_baseline("slash")
    result = box.run_trial(
        target_label="slash",
        hypothesis="bad shape",
        knobs=[44, 48],
    )
    assert result["ok"] is False
    assert "object" in result["error"]
