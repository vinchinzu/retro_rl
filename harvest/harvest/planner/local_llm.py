"""Optional local-LLM adapter for day-plan advice.

The adapter is deliberately advisory by default: it can add notes and
tomorrow-facing deferrals, but it does not rewrite executable phases. That
keeps the bot's control loop deterministic.

Gated apply (``HARVEST_PLAN_LLM_APPLY=1`` or ``apply_validated=True``) allows a
schema-checked path that may reorder optional phases or append known deferred
work. Required phases cannot be deleted; unknown kinds are rejected.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any, Mapping, Optional, Sequence
from urllib import request

from harvest.planner.day_phase_types import PhaseKind, PhaseSpec, TaskContract, coerce_phase_kind
from harvest.planner.day_plan_decision import (
    DayPlanDecision,
    DeferredPlan,
    deferred_from_phase_name,
)

# Phases that advisors may reorder or inject when apply_validated is on.
# Required structural phases (exit house, sleep, etc.) stay fixed.
OPTIONAL_PHASE_NAMES = frozenset(
    {
        "COOP_CHORES",
        "COW_CHORES",
        "HARVEST_ROUTE",
        "CROP_WATER",
        "CLEAR_FIELD",
        "TOWN_EXPLORE",
        "GET_BERRIES_AND_SHIP",
        "HOT_SPRING_STAMINA",
        "EVE_TALK_LOOP",
        "BUY_SEEDS",
        "ENSURE_CROP_SEEDS",
        "ENSURE_WATERING_CAN",
        "NAV_CROP",
    }
)

KNOWN_PHASE_KINDS = frozenset(str(k) for k in PhaseKind)


@dataclass(frozen=True)
class LocalLLMPlanAdvisor:
    endpoint: str
    model: str = "llama3.1"
    api: str = "ollama"
    timeout: float = 10.0
    apply_validated: bool = False

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
        return apply_advisor_patch(
            decision,
            patch,
            source="rules+local_llm",
            apply_validated=self.apply_validated,
        )

    def _request_payload(self, decision: DayPlanDecision) -> dict[str, Any]:
        prompt = _advisor_prompt(decision, apply_validated=self.apply_validated)
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
    apply_flag = env.get("HARVEST_PLAN_LLM_APPLY", "").strip().lower()
    apply_validated = apply_flag in {"1", "true", "yes", "on"}
    return LocalLLMPlanAdvisor(
        endpoint=endpoint,
        model=env.get("HARVEST_PLAN_LLM_MODEL", "llama3.1").strip() or "llama3.1",
        api=env.get("HARVEST_PLAN_LLM_API", "ollama").strip().lower() or "ollama",
        timeout=timeout,
        apply_validated=apply_validated,
    )


def apply_advisor_patch(
    decision: DayPlanDecision,
    patch: Mapping[str, Any],
    *,
    source: str,
    apply_validated: bool = False,
) -> DayPlanDecision:
    notes = [str(note) for note in patch.get("notes", []) if str(note).strip()]
    deferred = _deferred_from_patch(patch)
    result = decision.with_notes(notes, source=source).with_deferred(deferred, source=source)

    has_phase_ops = bool(
        patch.get("phase_names")
        or patch.get("phases")
        or patch.get("append_phases")
        or patch.get("reorder_optional")
    )
    if not has_phase_ops:
        return result

    if not apply_validated:
        return result.with_notes(["advisor_phase_changes_ignored"], source=source)

    validated, reject_notes = validate_and_apply_phase_patch(decision, patch)
    if validated is None:
        return result.with_notes(
            reject_notes or ["advisor_phase_patch_rejected"],
            source=source,
        )
    combined_notes = tuple(
        n for n in (*result.notes, *validated.notes) if n
    )
    return DayPlanDecision(
        phases=validated.phases,
        facts=decision.facts,
        deferred=result.deferred,
        notes=combined_notes,
        source=source,
    )


def validate_and_apply_phase_patch(
    decision: DayPlanDecision,
    patch: Mapping[str, Any],
) -> tuple[Optional[DayPlanDecision], list[str]]:
    """Validate a tight JSON patch and return a new decision or reject notes.

    Supported ops (any combination):
    - ``reorder_optional``: list of optional phase names to front-load among
      already-scheduled optional work (order preserved for required phases).
    - ``append_phases``: list of phase dicts ``{phase, kind, params?, failure_policy?}``
      that must use known kinds and optional phase names only.
    - ``phases`` / ``phase_names``: full replacement is **not** allowed; rejected
      to keep required structural phases intact.
    """
    notes: list[str] = []
    if patch.get("phases") is not None or patch.get("phase_names") is not None:
        return None, ["advisor_full_phase_rewrite_rejected"]

    phases = list(decision.phases)

    reorder = patch.get("reorder_optional")
    if reorder is not None:
        if not isinstance(reorder, list):
            return None, ["advisor_reorder_optional_not_list"]
        reorder_names = [str(x).strip() for x in reorder if str(x).strip()]
        for name in reorder_names:
            if name not in OPTIONAL_PHASE_NAMES:
                return None, [f"advisor_reorder_not_optional:{name}"]
            if name not in {p.phase for p in phases}:
                notes.append(f"advisor_reorder_missing:{name}")
        phases = _reorder_optional_phases(phases, reorder_names)

    append_raw = patch.get("append_phases")
    if append_raw is not None:
        if not isinstance(append_raw, list):
            return None, ["advisor_append_phases_not_list"]
        append_specs: list[PhaseSpec] = []
        for item in append_raw:
            parsed, err = _parse_phase_spec_item(item)
            if err:
                return None, [err]
            assert parsed is not None
            if parsed.phase not in OPTIONAL_PHASE_NAMES:
                return None, [f"advisor_append_not_optional:{parsed.phase}"]
            if str(parsed.kind) not in KNOWN_PHASE_KINDS:
                return None, [f"advisor_append_unknown_kind:{parsed.kind}"]
            append_specs.append(parsed)
        # Insert optional work before end-day phases when present.
        phases = _append_before_end_day(phases, append_specs)

    notes.append("advisor_phase_patch_applied")
    return (
        DayPlanDecision(
            phases=tuple(phases),
            facts=decision.facts,
            deferred=decision.deferred,
            notes=tuple(notes),
            source="rules+local_llm_apply",
        ),
        notes,
    )


def _reorder_optional_phases(
    phases: Sequence[PhaseSpec],
    preferred: Sequence[str],
) -> list[PhaseSpec]:
    preferred_set = set(preferred)
    optional_by_name = {p.phase: p for p in phases if p.phase in preferred_set}
    ordered_optional = [optional_by_name[n] for n in preferred if n in optional_by_name]
    # Remaining optionals keep relative order.
    remaining_optional = [
        p for p in phases if p.phase in OPTIONAL_PHASE_NAMES and p.phase not in preferred_set
    ]
    result: list[PhaseSpec] = []
    optional_queue = ordered_optional + remaining_optional
    opt_i = 0
    for phase in phases:
        if phase.phase in OPTIONAL_PHASE_NAMES:
            if opt_i < len(optional_queue):
                result.append(optional_queue[opt_i])
                opt_i += 1
        else:
            result.append(phase)
    while opt_i < len(optional_queue):
        result.append(optional_queue[opt_i])
        opt_i += 1
    return result


def _append_before_end_day(
    phases: Sequence[PhaseSpec],
    extra: Sequence[PhaseSpec],
) -> list[PhaseSpec]:
    end_day = {"RETURN_HOME", "GO_TO_SLEEP", "READY_TO_GO_HOME"}
    insert_at = len(phases)
    for i, phase in enumerate(phases):
        if phase.phase in end_day:
            insert_at = i
            break
    existing = {p.phase for p in phases}
    to_add = [p for p in extra if p.phase not in existing]
    return list(phases[:insert_at]) + to_add + list(phases[insert_at:])


def _parse_phase_spec_item(item: Any) -> tuple[Optional[PhaseSpec], Optional[str]]:
    if not isinstance(item, Mapping):
        return None, "advisor_append_item_not_object"
    phase = str(item.get("phase", "")).strip()
    kind = str(item.get("kind", "")).strip()
    if not phase or not kind:
        return None, "advisor_append_missing_phase_or_kind"
    params = item.get("params") or {}
    if not isinstance(params, Mapping):
        return None, "advisor_append_params_not_object"
    failure_policy = str(item.get("failure_policy", "optional")).strip() or "optional"
    contract = item.get("contract")
    try:
        kind_c = coerce_phase_kind(kind)
        return (
            PhaseSpec(
                phase=phase,
                kind=kind_c,
                params=dict(params),
                failure_policy=failure_policy,
                contract=TaskContract.from_mapping(contract if isinstance(contract, dict) else None),
            ),
            None,
        )
    except Exception as exc:  # pragma: no cover
        return None, f"advisor_append_parse_error:{exc}"


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


def _advisor_prompt(decision: DayPlanDecision, *, apply_validated: bool) -> str:
    if apply_validated:
        schema = (
            'Return JSON with optional "notes", "deferred" '
            '[{phase, reason, retry}], optional "reorder_optional" '
            f"(subset of {sorted(OPTIONAL_PHASE_NAMES)}), and optional "
            '"append_phases" [{{phase, kind, params?, failure_policy?}}] '
            "for optional work only. Do not replace the full phase list. "
        )
    else:
        schema = (
            'Return JSON with optional "notes" and optional "deferred" items. '
            "Do not invent executable phase changes. Deferred items use fields "
            "phase, reason, retry. "
        )
    return (
        "You are advising a Harvest Moon SNES autonomous planner. "
        "Given this deterministic rule-based plan, "
        + schema
        + "Prefer deferring optional money/social work when late_day is true, "
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
    "KNOWN_PHASE_KINDS",
    "LocalLLMPlanAdvisor",
    "OPTIONAL_PHASE_NAMES",
    "apply_advisor_patch",
    "build_local_llm_plan_advisor_from_env",
    "validate_and_apply_phase_patch",
]
