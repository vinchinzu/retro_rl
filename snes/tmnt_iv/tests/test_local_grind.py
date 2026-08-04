"""Unit tests for the local Ollama grind helpers (no emulator)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tmnt_iv.grind_knobs import (
    GrindKnobs,
    clamp_knob_patch,
    knobs_as_dict,
    merge_knobs,
    override_knobs,
    active_knobs,
)
from tmnt_iv.local_grind.ollama_client import parse_json_object
from tmnt_iv.local_grind.schema import (
    ExperimentProposal,
    normalize_knobs,
    target_by_label,
)
from tmnt_iv.local_grind.scoring import is_improvement, score_metrics
from tmnt_iv.local_grind.runner import _heuristic_proposal, _read_prompt


def test_clamp_knob_patch_drops_unknown_and_clamps() -> None:
    cleaned = clamp_knob_patch(
        {
            "slash_approach_band": 999,
            "not_a_knob": 1,
            "attack_gap": "3",
        }
    )
    assert cleaned == {
        "slash_approach_band": 80,
        "attack_gap": 3,
    }


def test_override_knobs_restores_previous() -> None:
    before = active_knobs()
    with override_knobs({"slash_approach_band": 40}):
        assert active_knobs().slash_approach_band == 40
    assert active_knobs() == before


def test_merge_knobs_preserves_untouched_fields() -> None:
    base = GrindKnobs()
    merged = merge_knobs(base, {"blocker_charge_min": 30})
    assert merged.blocker_charge_min == 30
    assert merged.slash_approach_band == base.slash_approach_band


def test_proposal_from_mapping_requires_fields() -> None:
    with pytest.raises(ValueError):
        ExperimentProposal.from_mapping({"knobs": {}})
    proposal = ExperimentProposal.from_mapping(
        {
            "hypothesis": "tighter spin dodge",
            "target_label": "slash",
            "knobs": {"slash_spin_dodge_adx": 48},
        }
    )
    assert proposal.knobs["slash_spin_dodge_adx"] == 48


def test_normalize_knobs_accepts_named_list() -> None:
    assert normalize_knobs(
        [{"name": "slash_approach_band", "value": 44}, ["attack_gap", 3]]
    ) == {"slash_approach_band": 44, "attack_gap": 3}
    with pytest.raises(ValueError):
        normalize_knobs([32, 18])


def test_score_prefers_clear_low_damage() -> None:
    clear = score_metrics(
        {"outcome": "cleared", "frames": 10_000, "damage_taken": 400, "heals": 5}
    )
    timeout = score_metrics(
        {"outcome": "timeout", "frames": 10_000, "damage_taken": 400, "heals": 5}
    )
    assert clear < timeout
    assert is_improvement(clear * 0.95, clear)


def test_target_by_label_accepts_state_name() -> None:
    target = target_by_label("RaphFullHardBoss5")
    assert target.label == "slash"
    leo = target_by_label("FullHardBoss5")
    assert leo.label == "slash_leo"


def test_parse_json_object_strips_fences() -> None:
    parsed = parse_json_object('```json\n{"a": 1}\n```')
    assert parsed == {"a": 1}


def test_parse_message_falls_back_to_thinking() -> None:
    from tmnt_iv.local_grind.ollama_client import _parse_message

    message = _parse_message(
        {"role": "assistant", "content": "", "thinking": '{"a": 2}'}
    )
    assert message.content == '{"a": 2}'


def test_saved_prompts_exist_and_have_placeholders() -> None:
    system = _read_prompt("system.md")
    propose = _read_prompt("propose.md")
    review = _read_prompt("review.md")
    assert "JSON" in system
    assert "{knob_bounds}" in propose
    assert "{metrics}" in review


def test_heuristic_proposal_is_clamped() -> None:
    proposal = _heuristic_proposal(
        focus="slash",
        trial_id=1,
        best_knobs=GrindKnobs(),
    )
    assert proposal.target_label == "slash"
    assert 1 <= len(proposal.knobs) <= 4
    for key, value in proposal.knobs.items():
        assert key in knobs_as_dict()
        assert isinstance(value, int)


def test_trial_record_roundtrip_shape(tmp_path: Path) -> None:
    from tmnt_iv.local_grind.schema import TrialDecision, TrialRecord

    record = TrialRecord(
        trial_id=1,
        decision=TrialDecision.DISCARD,
        proposal=ExperimentProposal(
            hypothesis="x",
            target_label="slash",
            knobs={"slash_approach_band": 44},
        ),
        metrics={"outcome": "timeout", "frames": 1},
        score=1.0,
        baseline_score=1.0,
        delta_score=0.0,
    )
    path = tmp_path / "r.json"
    path.write_text(json.dumps(record.to_jsonable()), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["decision"] == "discard"
    assert loaded["proposal"]["knobs"]["slash_approach_band"] == 44
