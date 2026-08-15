"""Mechanical lab-fixture vs route_eligible classifier (no live states)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from zelda_i.route_eligible import classify, require_route_pin


def _write_prov(tmp_path: Path, name: str, payload: dict) -> Path:
    path = tmp_path / f"{name}.provenance.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_l9_recon_fixture_name_is_lab_fixture() -> None:
    verdict = classify("Level9BeforeGanonReconFixture")
    assert verdict.eligible is False
    assert verdict.class_ == "lab_fixture"
    assert any("ReconFixture" in reason for reason in verdict.reasons)


def test_l9_recon_sidecar_cannot_promote(tmp_path: Path) -> None:
    path = _write_prov(
        tmp_path,
        "Level9Room30StairsReconFixture",
        {
            "natural_entry": False,
            "request": {
                "fixture_only": True,
                "route_eligible": False,
                "fixture_writes": [{"address": 1622, "value": 2}],
            },
        },
    )
    verdict = classify("Level9Room30StairsReconFixture", path)
    assert verdict.eligible is False
    assert verdict.class_ == "lab_fixture"
    assert "route_eligible=false" in verdict.reasons
    assert "fixture_only=true" in verdict.reasons
    assert "loader/fixture writes" in verdict.reasons


def test_old_level5_entrance_rejected_even_with_flags() -> None:
    verdict = classify(
        "Level5Entrance",
        {"route_eligible": True, "natural_entry": True},
    )
    assert verdict.eligible is False
    assert verdict.class_ == "lab_fixture"
    assert any("Raft/Stepladder" in reason for reason in verdict.reasons)


def test_l5_room_poke_states_are_fixtures() -> None:
    verdict = classify("L5_Room_67", {"natural_entry": False})
    assert verdict.eligible is False
    assert verdict.class_ == "lab_fixture"
    assert any("L5_Room_" in reason for reason in verdict.reasons)


def test_level5_from_l4_requires_honest_provenance(tmp_path: Path) -> None:
    name = "Level5EntranceFromL4"
    assert classify(name).eligible is False
    assert classify(name).class_ == "unknown"

    dishonest = _write_prov(
        tmp_path,
        name,
        {"natural_entry": False, "route_eligible": False, "development_only": True},
    )
    assert classify(name, dishonest).eligible is False

    poked = _write_prov(
        tmp_path,
        f"{name}_poked",
        {
            "route_eligible": True,
            "request": {"door_poke": True, "key_poke": False},
        },
    )
    poked_verdict = classify(name, poked)
    assert poked_verdict.eligible is False
    assert "door poke" in poked_verdict.reasons

    honest = _write_prov(
        tmp_path,
        f"{name}_honest",
        {
            "natural_entry": False,
            "route_eligible": True,
            "request": {
                "segment": "l4_complete_to_l5_entrance",
                "track": "assisted",
                "door_poke": False,
                "key_poke": False,
            },
        },
    )
    verdict = classify(name, honest)
    assert verdict.eligible is True
    assert verdict.class_ == "route_pin"
    assert verdict.name == name


def test_level5_complete_only_when_provenance_says_so(tmp_path: Path) -> None:
    name = "Level5Complete"
    assert classify(name, {"natural_entry": False, "development_only": True}).eligible is False

    path = _write_prov(
        tmp_path,
        name,
        {
            "route_eligible": True,
            "natural_entry": False,
            "request": {
                "door_poke": False,
                "key_poke": False,
                "bomb_count_poke": False,
            },
        },
    )
    verdict = classify(name, path)
    assert verdict.eligible is True
    assert verdict.class_ == "route_pin"


def test_level1_complete_clean_m5_natural_entry(tmp_path: Path) -> None:
    path = _write_prov(
        tmp_path,
        "Level1Complete",
        {
            "natural_entry": True,
            "development_only": False,
            "request": {"segment": "level1_complete", "natural_entry": True},
        },
    )
    verdict = classify("Level1Complete", path)
    assert verdict.eligible is True
    assert verdict.class_ == "route_pin"


def test_level1_exit_overworld_needs_natural_entry() -> None:
    assert classify("Level1ExitOverworld").eligible is False
    verdict = classify("Level1ExitOverworld", {"natural_entry": True})
    assert verdict.eligible is True
    assert verdict.class_ == "route_pin"


def test_default_unknown_name_is_ineligible() -> None:
    verdict = classify("SomeLabScratch")
    assert verdict.eligible is False
    assert verdict.class_ == "unknown"


def test_require_route_pin_fail_closed() -> None:
    with pytest.raises(ValueError, match="not route_eligible"):
        require_route_pin("Level9CreditsReconFixture")
    pin = require_route_pin(
        "Level1Complete",
        {"natural_entry": True, "route_eligible": True},
    )
    assert pin.eligible is True
    assert pin.to_dict()["class"] == "route_pin"
