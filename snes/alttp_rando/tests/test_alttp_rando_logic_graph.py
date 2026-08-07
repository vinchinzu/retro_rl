"""Offline capability path tests for ALTTP rando early graph."""

from __future__ import annotations

from alttp_rando.logic_graph import (
    N_EASTERN_BOW,
    N_LINKS_HOUSE,
    N_SANCTUARY,
    N_UNCLE,
    path_with_capabilities,
    plan_to_eastern_bow,
)


def test_house_to_uncle_open() -> None:
    path = path_with_capabilities(N_LINKS_HOUSE, N_UNCLE, frozenset())
    assert path is not None
    assert path[0].edge_id == "house_to_uncle"


def test_sanctuary_needs_lamp_scaffold() -> None:
    assert path_with_capabilities(
        N_LINKS_HOUSE, N_SANCTUARY, frozenset({"sword"})
    ) is None
    path = path_with_capabilities(
        N_LINKS_HOUSE, N_SANCTUARY, frozenset({"sword", "lamp"})
    )
    assert path is not None
    assert path[-1].target_id == N_SANCTUARY


def test_eastern_bow_tip() -> None:
    path = plan_to_eastern_bow(frozenset({"sword", "lamp"}))
    assert path is not None
    assert path[-1].target_id == N_EASTERN_BOW
