"""Tool implementations for the TMNT local grind agent."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from tmnt_iv.grind_knobs import (
    KNOB_BOUNDS,
    GrindKnobs,
    clamp_knob_patch,
    focus_knob_names,
    knobs_as_dict,
    merge_knobs,
)
from tmnt_iv.local_grind.eval_probe import run_knob_probe
from tmnt_iv.local_grind.schema import (
    DEFAULT_TARGETS,
    ExperimentProposal,
    TrialDecision,
    TrialRecord,
    target_by_label,
)
from tmnt_iv.local_grind.scoring import is_improvement, score_metrics

ProbeFn = Callable[..., tuple[dict[str, Any], list[Path]]]


TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_targets",
            "description": "List cheap probe targets (state, budgets, labels).",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_knobs",
            "description": "List whitelist knobs and bounds for a focus target.",
            "parameters": {
                "type": "object",
                "properties": {
                    "focus": {
                        "type": "string",
                        "description": "Target label, e.g. slash or technodrome_tank",
                    }
                },
                "required": ["focus"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_baseline",
            "description": (
                "Run production knobs on a target. Call once before trials."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target_label": {
                        "type": "string",
                        "description": "Probe label from list_targets",
                    }
                },
                "required": ["target_label"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_trial",
            "description": (
                "Run one knob experiment vs current best. "
                "knobs must be an object of at most 3 name->int values."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target_label": {"type": "string"},
                    "hypothesis": {"type": "string"},
                    "knobs": {
                        "type": "object",
                        "description": "Whitelist knob name -> int",
                    },
                    "rationale": {"type": "string"},
                },
                "required": ["target_label", "hypothesis", "knobs"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_status",
            "description": "Budget remaining, best score/knobs, trial count.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_trial",
            "description": "Fetch metrics, decision, reasons, screenshot paths.",
            "parameters": {
                "type": "object",
                "properties": {
                    "trial_id": {
                        "type": "integer",
                        "description": "1-based trial id (0 = baseline)",
                    }
                },
                "required": ["trial_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "End the session with a short summary of findings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "What improved or failed and next step",
                    }
                },
                "required": ["summary"],
            },
        },
    },
]


@dataclass
class GrindToolbox:
    """Stateful tool backend shared across agent turns."""

    out_dir: Path
    focus: str = "slash"
    max_trials: int = 3
    min_rel_gain: float = 0.01
    screenshot_every: int = 900
    max_screenshots: int = 3
    probe_fn: ProbeFn = run_knob_probe
    best_knobs: GrindKnobs = field(default_factory=GrindKnobs)
    baselines: dict[str, dict[str, Any]] = field(default_factory=dict)
    best_scores: dict[str, float] = field(default_factory=dict)
    trials: list[TrialRecord] = field(default_factory=list)
    finished: bool = False
    finish_summary: str = ""

    def __post_init__(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "trials").mkdir(parents=True, exist_ok=True)

    def dispatch(self, name: str, arguments: MappingLike) -> dict[str, Any]:
        """Execute one tool call and return a JSON-serializable result."""
        args = _as_dict(arguments)
        if name == "list_targets":
            return self.list_targets()
        if name == "list_knobs":
            return self.list_knobs(str(args.get("focus", self.focus)))
        if name == "run_baseline":
            return self.run_baseline(str(args.get("target_label", self.focus)))
        if name == "run_trial":
            return self.run_trial(
                target_label=str(args.get("target_label", self.focus)),
                hypothesis=str(args.get("hypothesis", "")),
                knobs=args.get("knobs", {}),
                rationale=str(args.get("rationale", "")),
            )
        if name == "get_status":
            return self.get_status()
        if name == "inspect_trial":
            return self.inspect_trial(int(args.get("trial_id", -1)))
        if name == "finish":
            return self.finish(str(args.get("summary", "")))
        return {"ok": False, "error": f"unknown tool: {name}"}

    def list_targets(self) -> dict[str, Any]:
        return {
            "ok": True,
            "targets": [
                {
                    "label": t.label,
                    "state": t.state,
                    "max_frames": t.max_frames,
                    "stop_stage_gt": t.stop_stage_gt,
                }
                for t in DEFAULT_TARGETS
            ],
            "recommended_focus": self.focus,
        }

    def list_knobs(self, focus: str) -> dict[str, Any]:
        names = focus_knob_names(focus)
        current = knobs_as_dict(self.best_knobs)
        return {
            "ok": True,
            "focus": focus,
            "knobs": {
                name: {
                    "bounds": list(KNOB_BOUNDS[name]),
                    "current": current[name],
                }
                for name in names
                if name in KNOB_BOUNDS
            },
            "rules": {
                "max_knobs_per_trial": 3,
                "knobs_must_be_object": True,
                "lower_score_is_better": True,
            },
        }

    def run_baseline(self, target_label: str) -> dict[str, Any]:
        target = target_by_label(target_label)
        shot_dir = self.out_dir / "baseline" / target.label
        metrics, images = self.probe_fn(
            state_name=target.state,
            knobs={},
            max_frames=target.max_frames,
            stop_stage_gt=target.stop_stage_gt,
            screenshot_dir=shot_dir,
            screenshot_every=self.screenshot_every,
            max_screenshots=min(2, self.max_screenshots),
        )
        score = score_metrics(metrics)
        self.baselines[target.label] = metrics
        self.best_scores[target.label] = score
        payload = {
            "ok": True,
            "target_label": target.label,
            "score": score,
            "metrics": compact_metrics(metrics),
            "image_paths": [str(p) for p in images],
            "note": "Use these numbers as the keep/discard baseline.",
        }
        _write_json(self.out_dir / "baseline" / f"{target.label}.json", payload)
        self._persist_summary()
        return payload

    def run_trial(
        self,
        *,
        target_label: str,
        hypothesis: str,
        knobs: Any,
        rationale: str = "",
    ) -> dict[str, Any]:
        if len(self.trials) >= self.max_trials:
            return {
                "ok": False,
                "error": f"trial budget exhausted ({self.max_trials})",
                "status": self.get_status(),
            }
        if target_label not in self.baselines and target_label not in {
            t.label for t in DEFAULT_TARGETS
        }:
            return {"ok": False, "error": f"unknown target: {target_label}"}
        if target_label not in self.baselines:
            return {
                "ok": False,
                "error": f"run_baseline({target_label!r}) first",
            }
        try:
            proposal = ExperimentProposal.from_mapping(
                {
                    "hypothesis": hypothesis,
                    "target_label": target_label,
                    "knobs": knobs,
                    "rationale": rationale,
                }
            )
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}

        patch = clamp_knob_patch(proposal.knobs)
        if len(patch) > 3:
            # Keep the first three deterministic keys.
            patch = dict(list(patch.items())[:3])
        if not patch:
            return {"ok": False, "error": "no valid whitelist knobs after clamp"}

        trial_id = len(self.trials) + 1
        trial_dir = self.out_dir / "trials" / f"trial_{trial_id:03d}"
        target = target_by_label(target_label)
        metrics, images = self.probe_fn(
            state_name=target.state,
            knobs=patch,
            max_frames=target.max_frames,
            stop_stage_gt=target.stop_stage_gt,
            screenshot_dir=trial_dir / "frames",
            screenshot_every=self.screenshot_every,
            max_screenshots=self.max_screenshots,
        )
        score = score_metrics(metrics)
        baseline_score = self.best_scores[target.label]
        outcome = str(metrics.get("outcome"))
        failed = outcome in {"life_loss", "forbidden_a", "timeout"}
        keep = (not failed) and is_improvement(
            score,
            baseline_score,
            min_rel_gain=self.min_rel_gain,
        )
        decision = TrialDecision.KEEP if keep else TrialDecision.DISCARD
        proposal = ExperimentProposal(
            hypothesis=proposal.hypothesis,
            target_label=target.label,
            knobs=patch,
            rationale=proposal.rationale,
        )
        record = TrialRecord(
            trial_id=trial_id,
            decision=decision,
            proposal=proposal,
            metrics=metrics,
            score=score,
            baseline_score=baseline_score,
            delta_score=score - baseline_score,
            image_paths=[str(p) for p in images],
        )
        self.trials.append(record)
        if keep:
            self.best_knobs = merge_knobs(self.best_knobs, patch)
            self.best_scores[target.label] = score
            _write_json(
                self.out_dir / "best_knobs.json",
                knobs_as_dict(self.best_knobs),
            )
        _write_json(trial_dir / "result.json", record.to_jsonable())
        self._append_history(record)
        self._persist_summary()
        return {
            "ok": True,
            "trial_id": trial_id,
            "decision": decision.value,
            "score": score,
            "baseline_score": baseline_score,
            "delta_score": score - baseline_score,
            "metrics": compact_metrics(metrics),
            "knobs_applied": patch,
            "image_paths": [str(p) for p in images],
            "trials_remaining": self.max_trials - len(self.trials),
            "hint": (
                "KEEP beat best; continue from new knobs."
                if keep
                else "DISCARD; revert to best knobs and try a different axis."
            ),
        }

    def get_status(self) -> dict[str, Any]:
        return {
            "ok": True,
            "focus": self.focus,
            "trials_done": len(self.trials),
            "trials_remaining": max(0, self.max_trials - len(self.trials)),
            "baselines": {
                k: compact_metrics(v) for k, v in self.baselines.items()
            },
            "best_scores": self.best_scores,
            "best_knobs": knobs_as_dict(self.best_knobs),
            "keeps": sum(1 for t in self.trials if t.decision is TrialDecision.KEEP),
            "discards": sum(
                1 for t in self.trials if t.decision is TrialDecision.DISCARD
            ),
            "finished": self.finished,
        }

    def inspect_trial(self, trial_id: int) -> dict[str, Any]:
        if trial_id == 0:
            if self.focus not in self.baselines and len(self.baselines) == 1:
                label = next(iter(self.baselines))
            else:
                label = self.focus
            metrics = self.baselines.get(label)
            if metrics is None:
                return {"ok": False, "error": "no baseline yet"}
            return {
                "ok": True,
                "trial_id": 0,
                "kind": "baseline",
                "target_label": label,
                "metrics": metrics,
                "score": self.best_scores.get(label),
            }
        for record in self.trials:
            if record.trial_id == trial_id:
                return {
                    "ok": True,
                    "trial_id": trial_id,
                    "kind": "trial",
                    **record.to_jsonable(),
                }
        return {"ok": False, "error": f"unknown trial_id: {trial_id}"}

    def finish(self, summary: str) -> dict[str, Any]:
        self.finished = True
        self.finish_summary = summary.strip() or "finished"
        self._persist_summary()
        return {
            "ok": True,
            "finished": True,
            "summary": self.finish_summary,
            "status": self.get_status(),
        }

    def _append_history(self, record: TrialRecord) -> None:
        path = self.out_dir / "history.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_jsonable(), sort_keys=True) + "\n")

    def _persist_summary(self) -> None:
        _write_json(
            self.out_dir / "summary.json",
            {
                "focus": self.focus,
                "finished": self.finished,
                "finish_summary": self.finish_summary,
                "best_knobs": knobs_as_dict(self.best_knobs),
                "best_scores": self.best_scores,
                "baselines": self.baselines,
                "trials": [t.to_jsonable() for t in self.trials],
            },
        )


MappingLike = Any


def _as_dict(arguments: MappingLike) -> dict[str, Any]:
    if arguments is None:
        return {}
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "outcome",
        "frames",
        "damage_taken",
        "heals",
        "min_hp",
        "boss_hp",
        "top_reasons",
    )
    return {k: metrics[k] for k in keys if k in metrics}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


__all__ = ["TOOL_SPECS", "GrindToolbox", "compact_metrics"]
