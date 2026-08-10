"""Held-out BC experiment for condition-robust Landing skill dispatch.

Ownership boundaries (rr-3f3e):
  train    — fit_landing_bc_model / train_landing_bc / load_landing_bc_model
  evaluate — run_landing_bc_rom_evaluation (ROM + audited claims only)
  report   — export_landing_bc_trajectories / package_landing_bc_report /
             write_landing_bc_report
Orchestrators: evaluate_landing_bc, run_landing_bc_experiment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from retro_harness.actions import idle_action
from retro_harness.audit import (
    AuditCapabilities,
    AuditedEnv,
    InterventionClass,
    RuntimeObservationClass,
)
from retro_harness.benchmark_claims import (
    EvaluationContract,
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
from retro_harness.identity import sha256_bytes, sha256_file
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
from sm_rando.observations import (
    landing_entry_features,
    landing_entry_features_from_metadata,
)
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


# ---------------------------------------------------------------------------
# Contracts + model (shared; train owns fitting, evaluate owns ROM use)
# ---------------------------------------------------------------------------


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
    """Reconstruct features only through the versioned metadata mapping."""
    return landing_entry_features_from_metadata(record.metadata)


# ---------------------------------------------------------------------------
# TRAIN ownership — fitting and checkpoint packaging only
# ---------------------------------------------------------------------------


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
    """Train-only: sealed train split fit + checkpoint/artifact write."""
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


# ---------------------------------------------------------------------------
# EVALUATE ownership — ROM rollouts + owned Clean/Bronze audit gates
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LandingBCTrajectoryCapture:
    """ROM-side capture needed for later trajectory export (eval partition)."""

    state_digest: str
    before_values: tuple[float, ...]
    after_values: tuple[float, ...]
    after_state_digest: str
    predicted_wait: int
    frames: int
    success: bool
    failure: str | None
    attempt: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class LandingBCEvalResult:
    """Structured ROM evaluation result; no report packaging or trajectory IO."""

    corpus_path: Path
    checkpoint_path: Path
    corpus_digest: str
    contract_bundle_digest: str
    observation_schema_digest: str
    action_schema_digest: str
    reward_schema_digest: str
    policy_identity_digest: str
    policy_artifact_digest: str
    train_state_count: int
    train_metrics: Mapping[str, Any]
    attempts: tuple[dict[str, Any], ...]
    eval_captures: tuple[LandingBCTrajectoryCapture, ...]


def run_landing_bc_rom_evaluation(
    *,
    corpus_path: Path = LANDING_CORPUS_MANIFEST,
    checkpoint_path: Path = LANDING_BC_MODEL,
) -> LandingBCEvalResult:
    """ROM evaluation only: load states, predict wait, play, audit claims.

    Does not write trajectories or the experiment report. Preserves
    metadata-v2 feature reconstruction and backend-owned Clean/Bronze gates.
    """
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
    rom_sha256 = sha256_file(SHARED_SM_ROM)
    env = AuditedEnv(
        make_env(GAME, "NONE", GAME_DIR, render_mode=None),
        capabilities=AuditCapabilities.all("sm-rando-landing-bc-v2"),
    )
    attempts: list[dict[str, Any]] = []
    eval_captures: list[LandingBCTrajectoryCapture] = []
    try:
        env.reset()
        for record in corpus.records:
            # metadata-v2 gate: corpus rows must expose versioned observation
            # metadata so train-time fitting and eval-time checks stay aligned.
            meta_version = int(
                record.metadata.get("observation_metadata_version", 0)
            )
            if meta_version < 2:
                raise ValueError(
                    f"entry state {record.state_digest} lacks metadata-v2 "
                    f"(observation_metadata_version={meta_version})"
                )
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
            env.load_start_state(
                corpus.state_bytes(record, root=REPO_ROOT),
                start_identity_digest=contract.start_identity.identity_digest,
                policy_identity_digest=policy_identity.identity_digest,
                runtime_observation_class=RuntimeObservationClass.BRONZE,
                intervention_class=InterventionClass.CLEAN,
            )
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
            audit = env.audit()
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
                after_state_digest = sha256_bytes(env.em.get_state())
                eval_captures.append(
                    LandingBCTrajectoryCapture(
                        state_digest=record.state_digest,
                        before_values=tuple(float(v) for v in before_values),
                        after_values=tuple(float(v) for v in after_values),
                        after_state_digest=after_state_digest,
                        predicted_wait=predicted_wait,
                        frames=session.frame,
                        success=success,
                        failure=failure,
                        attempt=attempt,
                    )
                )
    finally:
        env.close()

    return LandingBCEvalResult(
        corpus_path=corpus_path,
        checkpoint_path=checkpoint_path,
        corpus_digest=corpus.identity_digest,
        contract_bundle_digest=contracts.identity_digest,
        observation_schema_digest=contracts.observation.identity_digest,
        action_schema_digest=contracts.action.identity_digest,
        reward_schema_digest=contracts.reward.identity_digest,
        policy_identity_digest=policy_identity.identity_digest,
        policy_artifact_digest=artifact.identity_digest,
        train_state_count=len(split.train),
        train_metrics=dict(checkpoint["train_metrics"]),
        attempts=tuple(attempts),
        eval_captures=tuple(eval_captures),
    )


# ---------------------------------------------------------------------------
# REPORT ownership — trajectories + experiment report packaging
# ---------------------------------------------------------------------------


def _observation_record(values: Sequence[float], schema_digest: str) -> dict[str, Any]:
    payload = [float(value) for value in values]
    return {
        "schema_digest": schema_digest,
        "values": payload,
        "identity_digest": contract_digest(
            "landing-bc-observation-v1", {"values": payload}
        ),
    }


def partition_metrics(
    attempts: Sequence[Mapping[str, Any]], partition: str
) -> dict[str, Any]:
    """Aggregate attempt rows for one sealed split partition."""
    rows = [value for value in attempts if value["partition"] == partition]
    if not rows:
        return {
            "attempts": 0,
            "successes": 0,
            "success_rate": 0.0,
            "mean_predicted_wait": 0.0,
            "mean_abs_wait_error": 0.0,
        }
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


def build_landing_bc_trajectory(
    capture: LandingBCTrajectoryCapture,
    *,
    eval_result: LandingBCEvalResult,
) -> Trajectory:
    """Build one eval Trajectory object from a ROM capture (no disk IO)."""
    obs_digest = eval_result.observation_schema_digest
    before = _observation_record(capture.before_values, obs_digest)
    after = _observation_record(capture.after_values, obs_digest)
    return Trajectory(
        observation_schema_digest=obs_digest,
        action_schema_digest=eval_result.action_schema_digest,
        reward_schema_digest=eval_result.reward_schema_digest,
        contract_bundle_digest=eval_result.contract_bundle_digest,
        policy_identity_digest=eval_result.policy_identity_digest,
        steps=(
            TrajectoryStep(
                sequence=0,
                edge_id="landing_to_parlor_bc",
                skill_id="sm_rando.landing_wait_bc_v1",
                frame_start=0,
                frame_end=capture.frames,
                applied_frames=max(1, capture.frames),
                observation_before=before,
                observation_after=after,
                action={
                    "type": "wait_then_dispatch",
                    "value": {
                        "wait_frames": capture.predicted_wait,
                        "skill": "play_landing_to_parlor",
                    },
                },
                reward_components={"parlor_reached": float(capture.success)},
                milestones=("parlor_reached",) if capture.success else (),
                state_digest_before=capture.state_digest,
                state_digest_after=capture.after_state_digest,
            ),
        ),
        succeeded=capture.success,
        terminal_reason=(
            "parlor_reached" if capture.success else (capture.failure or "miss")
        ),
        milestones=(dict(capture.attempt),),
        provenance={
            "corpus_digest": eval_result.corpus_digest,
            "entry_state_digest": capture.state_digest,
            "partition": "eval",
        },
        initial_observation_digest=before["identity_digest"],
        final_observation_digest=after["identity_digest"],
    )


def export_landing_bc_trajectories(
    eval_result: LandingBCEvalResult,
    *,
    trajectory_dir: Path = LANDING_BC_TRAJECTORY_DIR,
    counterexample_dir: Path = LANDING_BC_COUNTEREXAMPLES,
) -> list[str]:
    """Write eval trajectories and failure library; returns repo-relative paths."""
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    failure_library = CounterexampleLibrary(counterexample_dir)
    trajectory_paths: list[str] = []
    for capture in eval_result.eval_captures:
        trajectory = build_landing_bc_trajectory(capture, eval_result=eval_result)
        path = trajectory_dir / f"{capture.state_digest}.json"
        trajectory.write(path)
        try:
            trajectory_paths.append(str(path.resolve().relative_to(REPO_ROOT)))
        except ValueError:
            # Allow tmp-dir unit tests and off-repo dry runs.
            trajectory_paths.append(str(path))
        if not capture.success:
            failure_library.add(trajectory)
    return trajectory_paths


def package_landing_bc_report(
    eval_result: LandingBCEvalResult,
    *,
    eval_trajectories: Sequence[str],
    baseline_report_path: Path = LANDING_BASELINE_REPORT,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Pure report packaging from an eval result + trajectory path list."""
    metrics = {
        "train": partition_metrics(eval_result.attempts, "train"),
        "eval": partition_metrics(eval_result.attempts, "eval"),
    }
    metrics["generalization_gap"] = (
        metrics["train"]["success_rate"] - metrics["eval"]["success_rate"]
    )
    baseline_eval_rate: float | None = None
    if baseline_report_path.exists():
        baseline = json.loads(baseline_report_path.read_text(encoding="utf-8"))
        baseline_eval_rate = float(baseline["metrics"]["eval"]["success_rate"])
    beats_baseline = (
        baseline_eval_rate is not None
        and metrics["eval"]["success_rate"] > baseline_eval_rate
    )
    stamp = generated_at or datetime.now(timezone.utc).isoformat()
    return {
        "schema_version": 1,
        "experiment": "landing_wait_to_handoff_behavior_cloning",
        "generated_at": stamp,
        "corpus_path": str(eval_result.corpus_path.relative_to(REPO_ROOT)),
        "corpus_digest": eval_result.corpus_digest,
        "checkpoint_path": str(eval_result.checkpoint_path.relative_to(REPO_ROOT)),
        "policy_artifact_path": str(
            policy_artifact_path(eval_result.checkpoint_path).relative_to(REPO_ROOT)
        ),
        "policy_artifact_digest": eval_result.policy_artifact_digest,
        "contract_bundle_digest": eval_result.contract_bundle_digest,
        "training": {
            "partition": "train",
            "states": eval_result.train_state_count,
            "eval_states_used_for_fit": 0,
            "metrics": dict(eval_result.train_metrics),
        },
        "metrics": metrics,
        "structured_baseline_eval_rate": baseline_eval_rate,
        "beats_structured_baseline": beats_baseline,
        "intervention_class": "Clean",
        "runtime_observation_class": "Bronze",
        "trajectory_schema": "retro_harness.trajectory/v1",
        "eval_trajectories": list(eval_trajectories),
        "attempts": list(eval_result.attempts),
        "decision": (
            "candidate_only_not_deployed; replicate on new predecessor trajectories"
            if beats_baseline
            else "do_not_deploy; learned policy did not beat structured baseline"
        ),
    }


def write_landing_bc_report(
    report: Mapping[str, Any],
    *,
    output_path: Path = LANDING_BC_REPORT,
) -> Path:
    """Persist a packaged report JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


# ---------------------------------------------------------------------------
# Orchestrators
# ---------------------------------------------------------------------------


def evaluate_landing_bc(
    *,
    corpus_path: Path = LANDING_CORPUS_MANIFEST,
    checkpoint_path: Path = LANDING_BC_MODEL,
    output_path: Path = LANDING_BC_REPORT,
) -> dict[str, Any]:
    """Evaluate → export trajectories → package + write report."""
    eval_result = run_landing_bc_rom_evaluation(
        corpus_path=corpus_path,
        checkpoint_path=checkpoint_path,
    )
    trajectory_paths = export_landing_bc_trajectories(eval_result)
    report = package_landing_bc_report(
        eval_result, eval_trajectories=trajectory_paths
    )
    write_landing_bc_report(report, output_path=output_path)
    return report


def run_landing_bc_experiment() -> dict[str, Any]:
    """Train ownership then evaluate/report ownership, end-to-end."""
    train_landing_bc()
    return evaluate_landing_bc()


__all__ = [
    "EXPERT_HANDOFF_LANDING_FRAME",
    "LANDING_BC_CONTRACTS",
    "LANDING_BC_MODEL",
    "LANDING_BC_REPORT",
    "LandingBCEvalResult",
    "LandingBCModel",
    "LandingBCTrajectoryCapture",
    "build_landing_bc_contracts",
    "build_landing_bc_trajectory",
    "evaluate_landing_bc",
    "export_landing_bc_trajectories",
    "fit_landing_bc_model",
    "load_landing_bc_model",
    "package_landing_bc_report",
    "partition_metrics",
    "run_landing_bc_experiment",
    "run_landing_bc_rom_evaluation",
    "train_landing_bc",
    "write_landing_bc_report",
]
