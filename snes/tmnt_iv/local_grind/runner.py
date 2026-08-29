"""Propose → eval → keep/discard loop driven by a local Ollama model."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tmnt_iv.grind_knobs import (
    KNOB_BOUNDS,
    GrindKnobs,
    clamp_knob_patch,
    focus_knob_names,
    knobs_as_dict,
    merge_knobs,
)
from tmnt_iv.local_grind.eval_probe import run_knob_probe
from tmnt_iv.local_grind.tools import compact_metrics
from tmnt_iv.local_grind.ollama_client import OllamaConfig, OllamaError, chat_json
from tmnt_iv.local_grind.schema import (
    DEFAULT_TARGETS,
    ExperimentProposal,
    ProbeTarget,
    TrialDecision,
    TrialRecord,
    target_by_label,
)
from tmnt_iv.local_grind.scoring import is_improvement, score_metrics
from tmnt_iv.paths import GAME_DIR

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
DEFAULT_OUT_DIR = GAME_DIR / "recordings" / "local_grind"


@dataclass
class GrindLoopConfig:
    """CLI / programmatic loop settings."""

    model: str = "gemma4:12b"
    host: str = "http://127.0.0.1:11434"
    trials: int = 3
    focus_target: str = "slash"
    out_dir: Path = DEFAULT_OUT_DIR
    min_rel_gain: float = 0.01
    screenshot_every: int = 900
    max_screenshots: int = 3
    use_vision_review: bool = True
    temperature: float = 0.2
    timeout: float = 300.0
    skip_model: bool = False
    seed_proposals: list[ExperimentProposal] = field(default_factory=list)


def run_grind_loop(config: GrindLoopConfig) -> list[TrialRecord]:
    """Run baseline + N propose/eval cycles; persist JSONL history."""
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history_path = out_dir / "history.jsonl"
    best_path = out_dir / "best_knobs.json"

    ollama = OllamaConfig(
        host=config.host,
        model=config.model,
        timeout=config.timeout,
        temperature=config.temperature,
    )
    system = _read_prompt("system.md")
    propose_tmpl = _read_prompt("propose.md")
    review_tmpl = _read_prompt("review.md")

    focus = target_by_label(config.focus_target)
    baselines = _measure_baselines(
        targets=(focus,),
        out_dir=out_dir / "baseline",
        screenshot_every=config.screenshot_every,
        max_screenshots=1,
    )
    best_knobs = GrindKnobs()
    best_scores = {
        label: score_metrics(metrics) for label, metrics in baselines.items()
    }
    records: list[TrialRecord] = []

    for trial_id in range(1, config.trials + 1):
        trial_dir = out_dir / f"trial_{trial_id:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        try:
            proposal = _next_proposal(
                config=config,
                ollama=ollama,
                system=system,
                propose_tmpl=propose_tmpl,
                best_knobs=best_knobs,
                baselines=baselines,
                records=records,
                trial_id=trial_id,
            )
            target = target_by_label(proposal.target_label)
            patch = clamp_knob_patch(proposal.knobs)
            proposal = ExperimentProposal(
                hypothesis=proposal.hypothesis,
                target_label=target.label,
                knobs=patch,
                rationale=proposal.rationale,
            )
            metrics, images = run_knob_probe(
                state_name=target.state,
                knobs=patch,
                max_frames=target.max_frames,
                stop_stage_gt=target.stop_stage_gt,
                screenshot_dir=trial_dir / "frames",
                screenshot_every=config.screenshot_every,
                max_screenshots=config.max_screenshots,
            )
            score = score_metrics(metrics)
            baseline_score = best_scores.get(
                target.label,
                score_metrics(baselines.get(target.label, metrics)),
            )
            outcome = str(metrics.get("outcome"))
            failed = outcome in {"life_loss", "forbidden_a", "timeout"}
            keep = (not failed) and is_improvement(
                score,
                baseline_score,
                min_rel_gain=config.min_rel_gain,
            )
            decision = TrialDecision.KEEP if keep else TrialDecision.DISCARD
            notes = ""
            if config.use_vision_review and not config.skip_model:
                notes = _review_trial(
                    ollama=ollama,
                    system=system,
                    review_tmpl=review_tmpl,
                    proposal=proposal,
                    metrics=metrics,
                    score=score,
                    baseline_score=baseline_score,
                    decision=decision,
                    images=images if images else None,
                )
            if keep:
                best_knobs = merge_knobs(best_knobs, patch)
                best_scores[target.label] = score
                best_path.write_text(
                    json.dumps(knobs_as_dict(best_knobs), indent=2) + "\n",
                    encoding="utf-8",
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
                model_notes=notes,
            )
        except Exception as exc:  # noqa: BLE001 — loop must continue
            record = TrialRecord(
                trial_id=trial_id,
                decision=TrialDecision.ERROR,
                proposal=None,
                metrics={},
                score=0.0,
                baseline_score=0.0,
                delta_score=0.0,
                error=str(exc),
            )
        records.append(record)
        _append_jsonl(history_path, record.to_jsonable())
        (trial_dir / "result.json").write_text(
            json.dumps(record.to_jsonable(), indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"[trial {trial_id}] {record.decision.value} "
            f"score={record.score:.0f} "
            f"Δ={record.delta_score:+.0f} "
            f"err={record.error!r}"
        )
    summary = {
        "focus_target": focus.label,
        "best_knobs": knobs_as_dict(best_knobs),
        "best_scores": best_scores,
        "baselines": baselines,
        "trials": [r.to_jsonable() for r in records],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return records


def _next_proposal(
    *,
    config: GrindLoopConfig,
    ollama: OllamaConfig,
    system: str,
    propose_tmpl: str,
    best_knobs: GrindKnobs,
    baselines: dict[str, dict[str, Any]],
    records: list[TrialRecord],
    trial_id: int,
) -> ExperimentProposal:
    if config.seed_proposals and trial_id <= len(config.seed_proposals):
        return config.seed_proposals[trial_id - 1]
    if config.skip_model:
        return _heuristic_proposal(
            focus=config.focus_target,
            trial_id=trial_id,
            best_knobs=best_knobs,
        )
    focus_keys = focus_knob_names(config.focus_target)
    bounds = {k: KNOB_BOUNDS[k] for k in focus_keys if k in KNOB_BOUNDS}
    defaults = knobs_as_dict(GrindKnobs())
    best = knobs_as_dict(best_knobs)
    best_compact = {
        k: best[k]
        for k in focus_keys
        if k in best and (best[k] != defaults.get(k) or k in focus_keys[:6])
    }
    user = _fill_prompt(
        propose_tmpl,
        {
            "target_labels": ", ".join(t.label for t in DEFAULT_TARGETS),
            "knob_bounds": json.dumps(bounds, sort_keys=True),
            "best_knobs": json.dumps(best_compact, sort_keys=True),
            "baselines": json.dumps(
                {k: compact_metrics(v) for k, v in baselines.items()},
                sort_keys=True,
            ),
            "history": json.dumps(
                [_compact_trial(r) for r in records[-4:]],
                sort_keys=True,
            ),
            "focus_target": config.focus_target,
        },
    )
    try:
        raw = chat_json(config=ollama, system=system, user=user)
        return ExperimentProposal.from_mapping(raw)
    except (OllamaError, ValueError) as first_exc:
        repair_user = (
            "Your previous answer was invalid for this harness "
            f"({first_exc}). Reply with ONE corrected JSON object only. "
            'knobs MUST be an object like {"slash_approach_band":44}. '
            "Do not use arrays of bare numbers.\n\n"
            f"Original request:\n{user}"
        )
        raw = chat_json(config=ollama, system=system, user=repair_user)
        return ExperimentProposal.from_mapping(raw)


def _review_trial(
    *,
    ollama: OllamaConfig,
    system: str,
    review_tmpl: str,
    proposal: ExperimentProposal,
    metrics: dict[str, Any],
    score: float,
    baseline_score: float,
    decision: TrialDecision,
    images: list[Path] | None,
) -> str:
    user = _fill_prompt(
        review_tmpl,
        {
            "proposal": json.dumps(proposal.to_jsonable(), indent=2),
            "metrics": json.dumps(metrics, indent=2, sort_keys=True),
            "score": f"{score:.1f}",
            "baseline_score": f"{baseline_score:.1f}",
            "delta_score": f"{score - baseline_score:+.1f}",
            "decision": decision.value,
        },
    )
    try:
        raw = chat_json(
            config=ollama,
            system=system,
            user=user,
            images=images,
        )
    except OllamaError as exc:
        return f"review_unavailable: {exc}"
    notes = str(raw.get("notes", "")).strip()
    hint = str(raw.get("next_hint", "")).strip()
    parts = [p for p in (notes, hint) if p]
    return " | ".join(parts)


def _measure_baselines(
    *,
    targets: tuple[ProbeTarget, ...],
    out_dir: Path,
    screenshot_every: int,
    max_screenshots: int,
) -> dict[str, dict[str, Any]]:
    baselines: dict[str, dict[str, Any]] = {}
    for target in targets:
        metrics, _images = run_knob_probe(
            state_name=target.state,
            knobs={},
            max_frames=target.max_frames,
            stop_stage_gt=target.stop_stage_gt,
            screenshot_dir=out_dir / target.label,
            screenshot_every=screenshot_every,
            max_screenshots=max_screenshots,
        )
        baselines[target.label] = metrics
        print(
            f"[baseline {target.label}] {metrics['outcome']} "
            f"{metrics['frames']}f / {metrics['damage_taken']} dmg / "
            f"{metrics['heals']} heals"
        )
    return baselines


def _heuristic_proposal(
    *,
    focus: str,
    trial_id: int,
    best_knobs: GrindKnobs,
) -> ExperimentProposal:
    """Deterministic offline proposals for tests / no-Ollama dry runs."""
    current = knobs_as_dict(best_knobs)
    if focus == "technodrome_tank":
        patch = {
            "blocker_charge_min": current["blocker_charge_min"] + (-2 if trial_id % 2 else 2),
            "blocker_charge_dx": current["blocker_charge_dx"] + (1 if trial_id % 2 else -1),
        }
    else:
        patch = {
            "slash_approach_band": current["slash_approach_band"]
            + (-4 if trial_id % 2 else 4),
            "slash_spin_dodge_adx": current["slash_spin_dodge_adx"]
            + (4 if trial_id % 2 else -4),
        }
    return ExperimentProposal(
        hypothesis=f"heuristic sweep #{trial_id}",
        target_label=focus,
        knobs=clamp_knob_patch(patch),
        rationale="offline heuristic without local model",
    )


def _read_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


def _fill_prompt(template: str, values: dict[str, str]) -> str:
    """Replace ``{name}`` placeholders without interpreting other braces."""
    text = template
    for key, value in values.items():
        text = text.replace("{" + key + "}", value)
    return text


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _compact_trial(record: TrialRecord) -> dict[str, Any]:
    proposal = record.proposal
    return {
        "trial_id": record.trial_id,
        "decision": record.decision.value,
        "score": round(record.score, 1),
        "delta": round(record.delta_score, 1),
        "hypothesis": proposal.hypothesis if proposal else "",
        "knobs": proposal.knobs if proposal else {},
        "metrics": compact_metrics(record.metrics),
        "notes": record.model_notes[:160],
    }


__all__ = ["GrindLoopConfig", "run_grind_loop"]
