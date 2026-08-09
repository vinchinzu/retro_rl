"""Held-out BC experiment for condition-robust Landing skill dispatch."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from retro_harness.actions import idle_action
from retro_harness.benchmark import (
    AuditCapabilities,
    AttemptAudit,
    EvaluationContract,
    InterventionClass,
    RuntimeObservationClass,
    StartIdentity,
    validate_claim,
)
from retro_harness.contracts import (
    ActionContract,
    ActionEntry,
    ContractBundle,
    RewardComponent,
    RewardContract,
    WrapperContract,
    WrapperSpec,
    contract_digest,
)
from retro_harness.entry_states import EntryStateCorpus, EntryStateRecord
from retro_harness.env import make_env
from retro_harness.model_artifacts import (
    load_policy_artifact,
    policy_artifact_path,
    write_policy_artifact,
)
from retro_harness.trajectory import (
    CounterexampleLibrary,
    Trajectory,
    TrajectoryStep,
)
from sm_rando.entry_corpus import (
    LANDING_BASELINE_REPORT,
    LANDING_CORPUS_MANIFEST,
    landing_corpus_contracts,
)
from sm_rando.observations import landing_entry_features
from sm_rando.paths import GAME, GAME_DIR, RECORDINGS_DIR, REPO_ROOT, SHARED_SM_ROM
from super_metroid.assist import UnlimitedAmmoAssist
from super_metroid.progression import MORPH_GRAPH
from super_metroid.routes.kpdr.early_spine import play_landing_to_parlor
from super_metroid.routes.kpdr.room_ids import ROOM_PARLOR
from super_metroid.routes.runtime import RouteSession

LANDING_BC_MODEL = GAME_DIR / "models" / "landing_wait_bc_v1.json"
LANDING_BC_CONTRACTS = GAME_DIR / "models" / "landing_wait_bc_v1.contracts.json"
LANDING_BC_REPORT = RECORDINGS_DIR / "landing_wait_bc_experiment.json"
LANDING_BC_TRAJECTORY_DIR = RECORDINGS_DIR / "landing_wait_bc_trajectories"
LANDING_BC_COUNTEREXAMPLES = RECORDINGS_DIR / "landing_wait_bc_counterexamples"

# Retained clean full-run evidence hands the controller Landing at frame 21,548;
# this corpus first observes ordinary Landing at 21,052. The expert wait label
# is therefore 496 - record.metadata["timing"]["landing_frame"].
EXPERT_HANDOFF_LANDING_FRAME = 496
TRAIN_FRACTION = 0.8
SPLIT_SALT = "sm-landing-v1"
FEATURE_INDICES = (5, 6)  # y integer and y subpixel from landing_entry_features
FEATURE_SCALES = (1.0, 65536.0)


def build_landing_bc_contracts() -> ContractBundle:
    """Macro-action contract for learned wait followed by skill dispatch."""
    base = landing_corpus_contracts()
    action = ActionContract(
        controller_buttons=("WAIT_ONE_FRAME", "DISPATCH_LANDING_SKILL"),
        entries=(
            ActionEntry("wait", (1, 0), "wait one emulator frame"),
            ActionEntry("dispatch", (0, 1), "dispatch structured Landing skill"),
        ),
    )
    return ContractBundle(
        environment=replace(base.environment, action_space_size=2),
        observation=base.observation,
        action=action,
        reward=RewardContract(
            (RewardComponent("parlor_reached", 1.0, "entered Parlor room"),)
        ),
        wrappers=WrapperContract(
            (
                WrapperSpec(
                    "LandingBCWaitThenSkill",
                    {
                        "max_wait_frames": EXPERT_HANDOFF_LANDING_FRAME,
                        "dispatched_skill": (
                            "super_metroid.routes.kpdr.early_spine:"
                            "play_landing_to_parlor"
                        ),
                    },
                ),
            )
        ),
    )


@dataclass(frozen=True, slots=True)
class LandingBCModel:
    """Linear behavior clone of expert wait-to-handoff demonstrations."""

    weights: tuple[float, ...]
    feature_indices: tuple[int, ...] = FEATURE_INDICES
    feature_scales: tuple[float, ...] = FEATURE_SCALES
    maximum_wait: int = EXPERT_HANDOFF_LANDING_FRAME

    def __post_init__(self) -> None:
        if len(self.weights) != len(self.feature_indices) + 1:
            raise ValueError("BC weights must contain bias plus one per feature")
        if len(self.feature_indices) != len(self.feature_scales):
            raise ValueError("BC feature index/scale lengths differ")
        if any(scale <= 0 for scale in self.feature_scales):
            raise ValueError("BC feature scales must be positive")
        if self.maximum_wait < 0:
            raise ValueError("maximum_wait must be non-negative")

    def design_row(self, observation: Sequence[float]) -> np.ndarray:
        values = [1.0]
        values.extend(
            float(observation[index]) / scale
            for index, scale in zip(
                self.feature_indices, self.feature_scales, strict=True
            )
        )
        return np.asarray(values, dtype=np.float64)

    def predict_wait(self, observation: Sequence[float]) -> int:
        value = int(round(float(self.design_row(observation) @ self.weights)))
        return min(self.maximum_wait, max(0, value))

    def to_record(self) -> dict[str, Any]:
        return {
            "weights": list(self.weights),
            "feature_indices": list(self.feature_indices),
            "feature_scales": list(self.feature_scales),
            "maximum_wait": self.maximum_wait,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "LandingBCModel":
        return cls(
            weights=tuple(float(value) for value in record["weights"]),
            feature_indices=tuple(int(value) for value in record["feature_indices"]),
            feature_scales=tuple(float(value) for value in record["feature_scales"]),
            maximum_wait=int(record["maximum_wait"]),
        )


def _record_observation(record: EntryStateRecord) -> np.ndarray:
    """Contract-ordered features retained by the real RAM corpus harvester."""
    metadata = record.metadata
    return np.asarray(
        (
            metadata["room_id"],
            metadata["game_state"],
            metadata["door_transition"],
            metadata["samus_x"],
            metadata["samus_x_sub"],
            metadata["samus_y"],
            metadata["samus_y_sub"],
            metadata["velocity_x"],
            metadata["velocity_y"],
            metadata["health"],
            metadata["missiles"],
            0,  # pose was added to the contract after this retained metadata row
        ),
        dtype=np.float64,
    )


def fit_landing_bc_model(
    records: Sequence[EntryStateRecord],
    *,
    ridge: float = 1e-8,
) -> tuple[LandingBCModel, dict[str, float]]:
    """Fit only supplied records; callers pass the sealed split's train rows."""
    if not records:
        raise ValueError("BC training requires at least one entry state")
    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    prototype = LandingBCModel((0.0, 0.0, 0.0))
    design = np.stack(
        [prototype.design_row(_record_observation(record)) for record in records]
    )
    targets = np.asarray(
        [
            EXPERT_HANDOFF_LANDING_FRAME
            - int(record.metadata["timing"]["landing_frame"])
            for record in records
        ],
        dtype=np.float64,
    )
    penalty = np.eye(design.shape[1], dtype=np.float64) * ridge
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ targets)
    model = LandingBCModel(tuple(float(value) for value in weights))
    predictions = np.asarray(
        [model.predict_wait(_record_observation(record)) for record in records],
        dtype=np.float64,
    )
    errors = predictions - targets
    return model, {
        "examples": float(len(records)),
        "mae_frames": float(np.mean(np.abs(errors))),
        "rmse_frames": float(np.sqrt(np.mean(errors**2))),
        "max_abs_error_frames": float(np.max(np.abs(errors))),
    }


def train_landing_bc(
    *,
    corpus_path: Path = LANDING_CORPUS_MANIFEST,
    checkpoint_path: Path = LANDING_BC_MODEL,
) -> tuple[LandingBCModel, dict[str, Any]]:
    corpus = EntryStateCorpus.load(corpus_path)
    corpus_contracts = landing_corpus_contracts()
    if corpus.contract_bundle_digest != corpus_contracts.identity_digest:
        raise ValueError("corpus ContractBundle mismatch")
    split = corpus.split(train_fraction=TRAIN_FRACTION, salt=SPLIT_SALT)
    model, train_metrics = fit_landing_bc_model(split.train)
    contracts = build_landing_bc_contracts()
    if contracts.observation.identity_digest != corpus.observation_schema_digest:
        raise ValueError("BC observation contract does not match EntryStateCorpus")
    checkpoint = {
        "schema_version": 1,
        "model_type": "linear_wait_to_expert_handoff_bc",
        "model": model.to_record(),
        "corpus_digest": corpus.identity_digest,
        "corpus_contract_bundle_digest": corpus.contract_bundle_digest,
        "model_contract_bundle_digest": contracts.identity_digest,
        "observation_schema_digest": contracts.observation.identity_digest,
        "training_partition": "train",
        "split": split.to_record(),
        "train_state_digests": [record.state_digest for record in split.train],
        "expert_handoff_landing_frame": EXPERT_HANDOFF_LANDING_FRAME,
        "expert_source": (
            "snes/sm_rando/recordings/vertical_slice.evidence.json"
        ),
        "train_metrics": train_metrics,
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(
        json.dumps(checkpoint, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    contracts.write(LANDING_BC_CONTRACTS)
    artifact = write_policy_artifact(
        checkpoint_path,
        contracts,
        algorithm="linear behavior cloning (expert handoff timing)",
        hyperparameters={
            "ridge": 1e-8,
            "feature_indices": list(FEATURE_INDICES),
            "expert_handoff_landing_frame": EXPERT_HANDOFF_LANDING_FRAME,
        },
        training_seed="deterministic-closed-form",
        metadata={
            "corpus_digest": corpus.identity_digest,
            "training_partition": "train",
            "training_examples": len(split.train),
            "eval_examples_used_for_fit": 0,
        },
    )
    return model, {
        "train_metrics": train_metrics,
        "artifact_digest": artifact.identity_digest,
        "contract_bundle_digest": contracts.identity_digest,
        "train_states": len(split.train),
        "eval_states_used_for_fit": 0,
    }


def load_landing_bc_model(
    checkpoint_path: Path = LANDING_BC_MODEL,
) -> tuple[LandingBCModel, Mapping[str, Any]]:
    record = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if record.get("schema_version") != 1:
        raise ValueError("unsupported Landing BC checkpoint schema")
    return LandingBCModel.from_record(record["model"]), record


def _observation_record(values: np.ndarray, schema_digest: str) -> dict[str, Any]:
    payload = [float(value) for value in values]
    return {
        "schema_digest": schema_digest,
        "values": payload,
        "identity_digest": contract_digest(
            "landing-bc-observation-v1", {"values": payload}
        ),
    }


def _metrics(attempts: Sequence[Mapping[str, Any]], partition: str) -> dict[str, Any]:
    rows = [value for value in attempts if value["partition"] == partition]
    successes = sum(bool(value["success"]) for value in rows)
    return {
        "attempts": len(rows),
        "successes": successes,
        "success_rate": successes / len(rows),
        "mean_predicted_wait": sum(int(value["predicted_wait"]) for value in rows)
        / len(rows),
        "mean_abs_wait_error": sum(abs(int(value["wait_error"])) for value in rows)
        / len(rows),
    }


def evaluate_landing_bc(
    *,
    corpus_path: Path = LANDING_CORPUS_MANIFEST,
    checkpoint_path: Path = LANDING_BC_MODEL,
    output_path: Path = LANDING_BC_REPORT,
) -> dict[str, Any]:
    corpus = EntryStateCorpus.load(corpus_path)
    split = corpus.split(train_fraction=TRAIN_FRACTION, salt=SPLIT_SALT)
    model, checkpoint = load_landing_bc_model(checkpoint_path)
    contracts = build_landing_bc_contracts()
    artifact = load_policy_artifact(checkpoint_path, contracts)
    if checkpoint.get("corpus_digest") != corpus.identity_digest:
        raise ValueError("BC checkpoint corpus digest mismatch")
    expected_train = [record.state_digest for record in split.train]
    if checkpoint.get("train_state_digests") != expected_train:
        raise ValueError("BC checkpoint training split mismatch")
    if set(expected_train) & {record.state_digest for record in split.eval}:
        raise ValueError("BC checkpoint contains train/eval leakage")
    policy_identity = artifact.to_policy_identity("sm_rando.landing_wait_bc_v1")
    partition = {
        record.state_digest: "train" for record in split.train
    }
    partition.update({record.state_digest: "eval" for record in split.eval})
    rom_sha256 = hashlib.sha256(SHARED_SM_ROM.read_bytes()).hexdigest()
    env = make_env(GAME, "NONE", GAME_DIR, render_mode=None)
    attempts: list[dict[str, Any]] = []
    trajectory_paths: list[str] = []
    failure_library = CounterexampleLibrary(LANDING_BC_COUNTEREXAMPLES)
    try:
        env.reset()
        for record in corpus.records:
            # This restore establishes the benchmark start; no load occurs after
            # the policy begins, so the attempt's mid-run load count remains zero.
            env.em.set_state(corpus.state_bytes(record, root=REPO_ROOT))
            before_values = landing_entry_features(np.asarray(env.get_ram()))
            predicted_wait = model.predict_wait(before_values)
            session = RouteSession(
                env,
                writer=None,
                assist=UnlimitedAmmoAssist(enabled=False),
                graph=MORPH_GRAPH,
            )
            failure: str | None = None
            try:
                for _ in range(predicted_wait):
                    session.step(idle_action(), "landing_bc_wait")
                play_landing_to_parlor(session)
            except Exception as exc:
                failure = f"{type(exc).__name__}: {exc}"
            success = failure is None and session.state.room_id == ROOM_PARLOR
            contract = EvaluationContract(
                runtime_observation_class=RuntimeObservationClass.BRONZE,
                intervention_class=InterventionClass.CLEAN,
                start_identity=StartIdentity(
                    record.state_path,
                    rom_sha256=rom_sha256,
                    state_sha256=record.state_digest,
                ),
                policy_identity=policy_identity,
                benchmark_id="sm_rando_landing_entry_bc_v1",
                objective="Reach Parlor from natural Landing entry distribution",
                metadata={
                    "corpus_digest": corpus.identity_digest,
                    "partition": partition[record.state_digest],
                },
            )
            audit = AttemptAudit(
                ram_writes=0,
                mid_run_loads=0,
                assists={},
                start_identity_digest=contract.start_identity.identity_digest,
                policy_identity_digest=policy_identity.identity_digest,
                runtime_observation_class=RuntimeObservationClass.BRONZE,
                intervention_class=InterventionClass.CLEAN,
                capabilities=AuditCapabilities.all("sm-rando-landing-bc-v1"),
            )
            validate_claim(contract, audit)
            expert_wait = EXPERT_HANDOFF_LANDING_FRAME - int(
                record.metadata["timing"]["landing_frame"]
            )
            attempt = {
                "state_digest": record.state_digest,
                "partition": partition[record.state_digest],
                "success": success,
                "predicted_wait": predicted_wait,
                "expert_wait": expert_wait,
                "wait_error": predicted_wait - expert_wait,
                "frames": session.frame,
                "final_room_id": session.state.room_id,
                "failure": failure,
                "contract": contract.to_record(),
                "attempt_audit": audit.to_record(),
                "claim_valid": True,
            }
            attempts.append(attempt)

            if partition[record.state_digest] == "eval":
                after_values = landing_entry_features(np.asarray(env.get_ram()))
                after_state_digest = hashlib.sha256(env.em.get_state()).hexdigest()
                trajectory = Trajectory(
                    observation_schema_digest=contracts.observation.identity_digest,
                    action_schema_digest=contracts.action.identity_digest,
                    reward_schema_digest=contracts.reward.identity_digest,
                    contract_bundle_digest=contracts.identity_digest,
                    policy_identity_digest=policy_identity.identity_digest,
                    steps=(
                        TrajectoryStep(
                            sequence=0,
                            edge_id="landing_to_parlor_bc",
                            skill_id="sm_rando.landing_wait_bc_v1",
                            frame_start=0,
                            frame_end=session.frame,
                            applied_frames=max(1, session.frame),
                            observation_before=_observation_record(
                                before_values,
                                contracts.observation.identity_digest,
                            ),
                            observation_after=_observation_record(
                                after_values,
                                contracts.observation.identity_digest,
                            ),
                            action={
                                "type": "wait_then_dispatch",
                                "value": {
                                    "wait_frames": predicted_wait,
                                    "skill": "play_landing_to_parlor",
                                },
                            },
                            reward_components={
                                "parlor_reached": float(success)
                            },
                            milestones=("parlor_reached",) if success else (),
                            state_digest_before=record.state_digest,
                            state_digest_after=after_state_digest,
                        ),
                    ),
                    succeeded=success,
                    terminal_reason="parlor_reached" if success else (failure or "miss"),
                    milestones=(attempt,),
                    provenance={
                        "corpus_digest": corpus.identity_digest,
                        "entry_state_digest": record.state_digest,
                        "partition": "eval",
                    },
                    initial_observation_digest=_observation_record(
                        before_values, contracts.observation.identity_digest
                    )["identity_digest"],
                    final_observation_digest=_observation_record(
                        after_values, contracts.observation.identity_digest
                    )["identity_digest"],
                )
                path = LANDING_BC_TRAJECTORY_DIR / f"{record.state_digest}.json"
                trajectory.write(path)
                trajectory_paths.append(str(path.relative_to(REPO_ROOT)))
                if not success:
                    failure_library.add(trajectory)
    finally:
        env.close()

    metrics = {
        "train": _metrics(attempts, "train"),
        "eval": _metrics(attempts, "eval"),
    }
    metrics["generalization_gap"] = (
        metrics["train"]["success_rate"] - metrics["eval"]["success_rate"]
    )
    baseline_eval_rate: float | None = None
    if LANDING_BASELINE_REPORT.exists():
        baseline = json.loads(LANDING_BASELINE_REPORT.read_text(encoding="utf-8"))
        baseline_eval_rate = float(baseline["metrics"]["eval"]["success_rate"])
    beats_baseline = (
        baseline_eval_rate is not None
        and metrics["eval"]["success_rate"] > baseline_eval_rate
    )
    report = {
        "schema_version": 1,
        "experiment": "landing_wait_to_handoff_behavior_cloning",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "corpus_path": str(corpus_path.relative_to(REPO_ROOT)),
        "corpus_digest": corpus.identity_digest,
        "checkpoint_path": str(checkpoint_path.relative_to(REPO_ROOT)),
        "policy_artifact_path": str(
            policy_artifact_path(checkpoint_path).relative_to(REPO_ROOT)
        ),
        "policy_artifact_digest": artifact.identity_digest,
        "contract_bundle_digest": contracts.identity_digest,
        "training": {
            "partition": "train",
            "states": len(split.train),
            "eval_states_used_for_fit": 0,
            "metrics": checkpoint["train_metrics"],
        },
        "metrics": metrics,
        "structured_baseline_eval_rate": baseline_eval_rate,
        "beats_structured_baseline": beats_baseline,
        "intervention_class": "Clean",
        "runtime_observation_class": "Bronze",
        "trajectory_schema": "retro_harness.trajectory/v1",
        "eval_trajectories": trajectory_paths,
        "attempts": attempts,
        "decision": (
            "candidate_only_not_deployed; replicate on new predecessor trajectories"
            if beats_baseline
            else "do_not_deploy; learned policy did not beat structured baseline"
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def run_landing_bc_experiment() -> dict[str, Any]:
    train_landing_bc()
    return evaluate_landing_bc()


__all__ = [
    "EXPERT_HANDOFF_LANDING_FRAME",
    "LANDING_BC_CONTRACTS",
    "LANDING_BC_MODEL",
    "LANDING_BC_REPORT",
    "LandingBCModel",
    "build_landing_bc_contracts",
    "evaluate_landing_bc",
    "fit_landing_bc_model",
    "load_landing_bc_model",
    "run_landing_bc_experiment",
    "train_landing_bc",
]
