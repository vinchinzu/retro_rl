"""Map Rando / sm-json-data tech catalog + builder registry (no ROM)."""

from __future__ import annotations

from super_metroid.rooms.tech_catalog import (
    MAPRANDO_LOGIC_URL,
    builder_coverage_summary,
    builder_priority_for,
    builder_targets,
    load_tech_catalog,
    parse_maprando_difficulties,
    parse_techs_from_sm_json_data,
    tech_by_id,
    tech_by_name,
    techs_at_difficulty,
    write_catalog,
)
from super_metroid.routes.skills.builders import (
    BUILDER_SKILLS,
    builder_gap_report,
    builder_skill,
    list_builder_skills,
    registered_tech_names,
)


def test_sm_json_tech_tree_has_hundreds() -> None:
    rows = parse_techs_from_sm_json_data()
    assert len(rows) >= 240
    names = {r["name"] for r in rows}
    assert "canWallJump" in names
    assert "canShinespark" in names
    assert "canStopOnADime" in names


def test_maprando_difficulty_table_covers_core() -> None:
    diffs = parse_maprando_difficulties()
    assert diffs["canStopOnADime"] == ("Implicit", 23)
    assert diffs["canWallJump"] == ("Basic", 76)
    assert diffs["canIBJ"] == ("Medium", 89)
    assert builder_priority_for("Basic") == "core"
    assert builder_priority_for("Medium") == "try"
    assert builder_priority_for("Expert") == "out_of_scope"


def test_tech_by_id_stop_on_a_dime() -> None:
    # Ensure catalog is present / loadable
    node = tech_by_id(23)
    assert node is not None
    assert node.name == "canStopOnADime"
    assert node.difficulty == "Implicit"
    assert node.logic_url == f"{MAPRANDO_LOGIC_URL}/tech/23"


def test_basic_and_medium_counts() -> None:
    basic = techs_at_difficulty("Basic")
    medium = techs_at_difficulty("Medium")
    implicit = techs_at_difficulty("Implicit")
    assert len(basic) == 5
    assert len(medium) == 17
    assert len(implicit) == 9


def test_builder_targets_are_core_and_try() -> None:
    targets = builder_targets()
    assert all(t.builder_priority in ("core", "try") for t in targets)
    assert len(targets) == 9 + 5 + 17  # Implicit + Basic + Medium


def test_builder_coverage_scores_core_try() -> None:
    cov = builder_coverage_summary()
    assert cov["total"] == 31
    assert "canWallJump" in cov["green"]
    assert "canShinespark" in cov["green"]
    assert "canUseFrozenEnemies" in cov["missing"]
    # At least half of core+try should be green or partial
    ready = cov["counts"]["green"] + cov["counts"]["partial"]
    assert ready >= 20


def test_builder_registry_resolves_callables() -> None:
    assert len(BUILDER_SKILLS) >= 12
    spark = builder_skill("canShinespark")
    assert spark is not None
    fn = spark.resolve()
    assert callable(fn)
    wj = builder_skill("canWallJump")
    assert wj is not None
    assert callable(wj.resolve())
    # Medium wrappers
    for name in ("canCrouchJump", "canDownGrab", "canSpeedyJump", "canStopOnADime"):
        skill = builder_skill(name)
        assert skill is not None, name
        assert callable(skill.resolve())


def test_builder_gap_report_lists_unregistered() -> None:
    report = builder_gap_report()
    assert report["registered_count"] == len(registered_tech_names())
    assert "canUseFrozenEnemies" in report["unregistered_targets"]
    greens = list_builder_skills(status="green")
    assert any(s.tech == "canShinespark" for s in greens)


def test_write_catalog_roundtrip(tmp_path) -> None:
    out = tmp_path / "tech.json"
    path, payload = write_catalog(catalog_path=out)
    assert path.is_file()
    assert payload["counts"]["total"] >= 240
    assert payload["source"]["mapRandoTechExample"].endswith("/tech/23")
    # load_tech_catalog uses default path; smoke-check payload fields
    wall = next(t for t in payload["techs"] if t["name"] == "canWallJump")
    assert wall["maprandoDifficulty"] == "Basic"
    assert wall["bot"]["status"] == "green"


def test_load_catalog_includes_extension_depth() -> None:
    nodes = load_tech_catalog()
    # Extension tech should exist under parent (depth > 0 for some)
    deep = [n for n in nodes if n.depth > 0]
    assert deep, "expected extensionTechs flattened with depth>0"
    # Known parent/child pair from sm-json-data tree
    midair = tech_by_name("canMidAirMorph")
    assert midair is not None
