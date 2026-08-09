"""Contract and retained-manifest checks for the Landing entry corpus."""

from __future__ import annotations

import pytest
from importlib import metadata

from retro_harness.entry_states import EntryStateCorpus
from sm_rando.entry_corpus import (
    LANDING_CORPUS_MANIFEST,
    corpus_summary,
    harvest_landing_entry_corpus,
    landing_corpus_contracts,
)


def test_retained_landing_corpus_is_unique_split_and_contract_bound() -> None:
    corpus = EntryStateCorpus.load(LANDING_CORPUS_MANIFEST)
    summary = corpus_summary(corpus)
    assert summary["states"] >= 64
    assert summary["train"] > 0
    assert summary["eval"] > 0
    assert summary["frame_parities"] == [0, 1]
    assert (
        corpus.observation_schema_digest
        == landing_corpus_contracts().observation.identity_digest
    )
    try:
        metadata.version("stable-retro")
    except metadata.PackageNotFoundError:
        pass  # Core identity is intentionally unavailable under the pytest stub.
    else:
        assert corpus.contract_bundle_digest == landing_corpus_contracts().identity_digest
    assert len({record.state_digest for record in corpus.records}) == len(corpus.records)


@pytest.mark.rom
@pytest.mark.rom_smoke
def test_harvest_landing_entry_corpus_rom(tmp_path) -> None:
    import stable_retro as retro

    if not hasattr(retro.data.Integrations, "CUSTOM"):
        pytest.skip("stable_retro test stub cannot execute ROM smoke")
    corpus = harvest_landing_entry_corpus(
        count=8,
        output_path=tmp_path / "corpus.json",
    )
    assert len(corpus.records) == 8
