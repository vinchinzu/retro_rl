"""EntryStateCorpus integrity, filtering, and leakage tests."""

from __future__ import annotations

import gzip

import pytest

from retro_harness.entry_states import (
    EntryStateCorpus,
    EntryStateCorpusBuilder,
    EntryStateError,
    SplitStrategy,
)


def _corpus(tmp_path):
    builder = EntryStateCorpusBuilder(
        corpus_id="fixture",
        game_id="Fixture-Snes",
        contract_bundle_digest="contract-v1",
        observation_schema_digest="obs-v1",
    )
    for index in range(40):
        state = f"state-{index}".encode()
        path = tmp_path / "states" / f"{index}.state"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(gzip.compress(state))
        builder.add(
            state_bytes=state,
            ram_snapshot=f"ram-{index}".encode(),
            state_path=f"states/{index}.state",
            source_skill_id="predecessor",
            source_segment_id="door-entry",
            source_trajectory_digest=f"trajectory-{index // 10}",
            frame=index,
            metadata={"health": 99 - index, "parity": index % 2},
        )
    return builder.build()


def test_corpus_round_trip_filter_and_state_integrity(tmp_path) -> None:
    corpus = _corpus(tmp_path)
    loaded = EntryStateCorpus.load(corpus.write(tmp_path / "corpus.json"))
    assert loaded == corpus
    assert len(loaded.filter(source_segment_id="door-entry")) == 40
    first = loaded.records[0]
    assert loaded.state_bytes(first, root=tmp_path).startswith(b"state-")


def test_hash_split_is_deterministic_disjoint_and_nonempty(tmp_path) -> None:
    corpus = _corpus(tmp_path)
    first = corpus.split(train_fraction=0.75, salt="fixed")
    second = corpus.split(train_fraction=0.75, salt="fixed")
    assert first.to_record() == second.to_record()
    assert first.train and first.eval
    assert {value.state_digest for value in first.train}.isdisjoint(
        value.state_digest for value in first.eval
    )


def test_source_trajectory_split_prevents_trajectory_leakage(tmp_path) -> None:
    split = _corpus(tmp_path).split(
        strategy=SplitStrategy.SOURCE_TRAJECTORY,
        train_fraction=0.5,
        salt="trajectory-split",
    )
    assert {
        value.source_trajectory_digest for value in split.train
    }.isdisjoint(value.source_trajectory_digest for value in split.eval)


def test_builder_rejects_duplicate_emulator_states(tmp_path) -> None:
    builder = EntryStateCorpusBuilder(
        corpus_id="fixture",
        game_id="Fixture-Snes",
        contract_bundle_digest="contract-v1",
        observation_schema_digest="obs-v1",
    )
    values = dict(
        state_bytes=b"same",
        ram_snapshot=b"ram",
        state_path="states/a.state",
        source_skill_id="skill",
        source_segment_id="segment",
        source_trajectory_digest="trajectory",
        frame=1,
    )
    builder.add(**values)
    with pytest.raises(EntryStateError, match="duplicate"):
        builder.add(**values)
