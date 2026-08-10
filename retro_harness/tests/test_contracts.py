"""Golden identity and fail-closed model contract tests."""

from __future__ import annotations

from dataclasses import replace

import pytest

from retro_harness.model_artifacts import PolicyArtifactError
from retro_harness.contracts import (
    ActionContract,
    ContractBundle,
    ContractMismatchError,
    EnvironmentContract,
    ObservationContract,
    ObservationField,
    RewardComponent,
    RewardContract,
    WrapperContract,
    WrapperSpec,
)
from retro_harness.env import GameSpec
from retro_harness.model_artifacts import load_policy_artifact, write_policy_artifact


def _bundle() -> ContractBundle:
    return ContractBundle(
        environment=EnvironmentContract(
            game_id="Fixture-Snes",
            state_id="NaturalEntry",
            action_space_size=2,
            frame_skip=4,
            players=1,
            rom_identity_digest="rom-sha",
            state_identity_digest="state-sha",
            core_identity_digest="core-sha",
        ),
        observation=ObservationContract(
            (
                ObservationField("x", "float32", (1,), "player x"),
                ObservationField("y", "float32", (1,), "player y"),
            ),
            preprocessing={"normalize": True},
        ),
        action=ActionContract.from_button_rows(
            ((0, 0), (1, 0)),
            controller_buttons=("LEFT", "RIGHT"),
            labels=("idle", "left"),
        ),
        reward=RewardContract(
            (RewardComponent("progress", 1.0, "forward progress"),)
        ),
        wrappers=WrapperContract(
            (
                WrapperSpec("RetroEnv"),
                WrapperSpec("FrameSkip", {"n": 4}),
            )
        ),
    )


def test_contract_bundle_has_golden_digest_and_round_trips(tmp_path) -> None:
    bundle = _bundle()
    assert bundle.identity_digest == (
        "04ba38b43579c9a046caf864b9a8feb77f4cdaf70d7f04c610737f1026af42b7"
    )
    path = bundle.write(tmp_path / "contracts.json")
    assert ContractBundle.load(path) == bundle


def test_action_order_and_wrapper_order_are_identity_significant() -> None:
    bundle = _bundle()
    action_flip = replace(
        bundle,
        action=ActionContract(
            bundle.action.controller_buttons,
            tuple(reversed(bundle.action.entries)),
        ),
    )
    wrapper_flip = replace(
        bundle,
        wrappers=WrapperContract(tuple(reversed(bundle.wrappers.stack))),
    )
    with pytest.raises(ContractMismatchError, match="action"):
        bundle.assert_compatible(action_flip)
    with pytest.raises(ContractMismatchError, match="wrapper"):
        bundle.assert_compatible(wrapper_flip)


def test_policy_load_rejects_deliberate_action_schema_flip(tmp_path) -> None:
    checkpoint = tmp_path / "policy.zip"
    checkpoint.write_bytes(b"weights")
    lock = tmp_path / "uv.lock"
    lock.write_text("locked", encoding="utf-8")
    bundle = _bundle()
    write_policy_artifact(
        checkpoint,
        bundle,
        algorithm="fixture",
        hyperparameters={},
        training_seed=7,
        dependency_lock_path=lock,
        source_commit="deadbeef",
    )
    load_policy_artifact(checkpoint, bundle)

    flipped = replace(
        bundle,
        action=ActionContract(
            bundle.action.controller_buttons,
            tuple(reversed(bundle.action.entries)),
        ),
    )
    with pytest.raises(PolicyArtifactError, match="action schema"):
        load_policy_artifact(checkpoint, flipped)


def test_game_spec_carries_and_checks_full_contract(tmp_path) -> None:
    bundle = _bundle()
    spec = GameSpec("Fixture-Snes", tmp_path, contract=bundle)
    assert spec.contract_digest == bundle.identity_digest
    spec.require_compatible_contract(bundle)
    with pytest.raises(ValueError, match="no compatibility contract"):
        GameSpec("Fixture-Snes", tmp_path).require_compatible_contract(bundle)
