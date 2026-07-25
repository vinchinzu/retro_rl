"""Optional local-LLM adapter for day-plan advice.

The adapter is deliberately advisory: it can add notes and tomorrow-facing
deferrals, but it does not rewrite executable phases. That keeps the bot's
control loop deterministic unless a future caller explicitly validates and
applies phase changes.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any, Mapping, Optional
from urllib import request

from harvest.planner.day_plan_decision import (
    DayPlanDecision,
    DeferredPlan,
    deferred_from_phase_name,
)


@dataclass(frozen=True)
class LocalLLMPlanAdvisor:
    endpoint: str
    model: str = "llama3.1"
    api: str = "ollama"
    timeout: float = 10.0

    def advise_day_plan(self, decision: DayPlanDecision) -> DayPlanDecision:
        try:
            payload = self._request_payload(decision)
            response = _post_json(self.endpoint, payload, timeout=self.timeout)
            patch = self._extract_patch(response)
        except Exception as exc:  # pragma: no cover - defensive runtime path
            return decision.with_notes(
                [f"local_llm_advisor_unavailable: {exc}"],
                source="rules+local_llm_error",
            )
        return apply_advisor_patch(decision, patch, source="rules+local_llm")

    def _request_payload(self, decision: DayPlanDecision) -> dict[str, Any]:
        prompt = _advisor_prompt(decision)
        if self.api == "openai":
            return {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "Return only compact JSON. Do not include markdown.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
            }
        return {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
        }

    def _extract_patch(self, response: Mapping[str, Any]) -> dict[str, Any]:
        if self.api == "openai":
            choices = response.get("choices", [])
            content = choices[0].get("message", {}).get("content", "") if choices else ""
        else:
            content = str(response.get("response", ""))
        return _parse_json_object(content)


def build_local_llm_plan_advisor_from_env(
    environ: Optional[Mapping[str, str]] = None,
) -> Optional[LocalLLMPlanAdvisor]:
    env = environ if environ is not None else os.environ
    endpoint = env.get("HARVEST_PLAN_LLM_URL", "").strip()
    if not endpoint:
        return None
    timeout_text = env.get("HARVEST_PLAN_LLM_TIMEOUT", "10").strip()
    try:
        timeout = float(timeout_text)
    except ValueError:
        timeout = 10.0
    return LocalLLMPlanAdvisor(
        endpoint=endpoint,
        model=env.get("HARVEST_PLAN_LLM_MODEL", "llama3.1").strip() or "llama3.1",
        api=env.get("HARVEST_PLAN_LLM_API", "ollama").strip().lower() or "ollama",
        timeout=timeout,
    )


def apply_advisor_patch(
    decision: DayPlanDecision,
    patch: Mapping[str, Any],
    *,
    source: str,
) -> DayPlanDecision:
    notes = [str(note) for note in patch.get("notes", []) if str(note).strip()]
    deferred = _deferred_from_patch(patch)
    if patch.get("phase_names") or patch.get("phases"):
        notes.append("advisor_phase_changes_ignored")
    return decision.with_notes(notes, source=source).with_deferred(deferred, source=source)


def _deferred_from_patch(patch: Mapping[str, Any]) -> list[DeferredPlan]:
    raw_items = patch.get("deferred", patch.get("defer", []))
    if not isinstance(raw_items, list):
        return []
    deferred: list[DeferredPlan] = []
    for item in raw_items:
        if not isinstance(item, Mapping):
            continue
        phase = str(item.get("phase", "")).strip()
        if not phase:
            continue
        reason = str(item.get("reason", "advisor")).strip() or "advisor"
        retry = str(item.get("retry", "tomorrow")).strip() or "tomorrow"
        deferred.append(deferred_from_phase_name(phase, reason, retry=retry))
    return deferred


def _advisor_prompt(decision: DayPlanDecision) -> str:
    return (
        "You are advising a Harvest Moon SNES autonomous planner. "
        "Given this deterministic rule-based plan, return JSON with optional "
        '"notes" and optional "deferred" items. Do not invent executable phase '
        "changes. Deferred items use fields phase, reason, retry. "
        "Prefer deferring optional money/social work when late_day is true, "
        "and note when bedtime/cutscene recovery should take priority over "
        "new chores.\n"
        f"{json.dumps(decision.to_jsonable(), sort_keys=True)}"
    )


def _post_json(url: str, payload: Mapping[str, Any], *, timeout: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as response:
        data = response.read().decode("utf-8")
    parsed = json.loads(data)
    if not isinstance(parsed, dict):
        raise ValueError("local LLM response was not a JSON object")
    return parsed


def _parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end >= start:
            text = text[start : end + 1]
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("advisor content was not a JSON object")
    return parsed


__all__ = [
    "LocalLLMPlanAdvisor",
    "apply_advisor_patch",
    "build_local_llm_plan_advisor_from_env",
]
