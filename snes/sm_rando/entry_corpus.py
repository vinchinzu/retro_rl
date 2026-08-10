"""Real predecessor-state harvest for the natural Landing Site distribution."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from retro_harness.actions import idle_action
from retro_harness.entry_states import (
    EntryStateCorpus,
    EntryStateCorpusBuilder,
    EntryStateRecord,
)
from retro_harness.env import make_env, write_state_bytes
from retro_harness.identity import sha256_bytes, sha256_file
from retro_harness.platformer.contracts import build_platformer_contracts
from retro_harness.platformer.level_config import get_level_config
from retro_harness.platformer.neuro.net import SMB_OUTPUT_BUTTONS
from sm_rando.paths import (
    GAME,
    GAME_DIR,
    INTEGRATION_DIR,
    RECORDINGS_DIR,
    REPO_ROOT,
    SHARED_SM_ROM,
)
from sm_rando.observations import (
    LANDING_ENTRY_METADATA_VERSION,
    landing_entry_features,
)
from super_metroid.assist import UnlimitedAmmoAssist
import super_metroid.platformer_levels  # noqa: F401 - registers SM levels
import sm_rando.platformer_levels  # noqa: F401 - registers corpus training level
from super_metroid.progression import MORPH_GRAPH
from super_metroid.routes.kpdr.early_spine import MORPH_SPINE
from super_metroid.routes.kpdr.early_spine import play_landing_to_parlor
from super_metroid.routes.kpdr.room_ids import ROOM_LANDING_SITE
from super_metroid.routes.kpdr.room_ids import ROOM_PARLOR
from super_metroid.routes.runtime import RouteSession, Split
from super_metroid.routes.tips import play_hops

LANDING_CORPUS_MANIFEST = RECORDINGS_DIR / "landing_entry_corpus.json"
LANDING_CORPUS_STATE_DIR = INTEGRATION_DIR / "entry_corpus" / "landing_v1"
LANDING_CORPUS_SIZE = 64
LANDING_BASELINE_REPORT = RECORDINGS_DIR / "landing_entry_baseline.json"


def landing_corpus_contracts():
    config = get_level_config("sm_rando_landing_entry")
    return build_platformer_contracts(
        config,
        n_inputs=12,
        read_inputs_fn=landing_entry_features,
        output_buttons=SMB_OUTPUT_BUTTONS,
    )


def _trajectory_digest() -> str:
    source = (
        sha256_file(SHARED_SM_ROM)
        + ":power_on:ceres_to_landing:MORPH_SPINE-v1"
    )
    return sha256_bytes(source.encode("utf-8"))


def harvest_landing_entry_corpus(
    *,
    count: int = LANDING_CORPUS_SIZE,
    output_path: Path = LANDING_CORPUS_MANIFEST,
) -> EntryStateCorpus:
    """Capture unique consecutive states after natural Ceres→Landing play."""
    if isinstance(count, bool) or not isinstance(count, int) or count < 2:
        raise ValueError("count must be an integer >= 2")
    contracts = landing_corpus_contracts()
    builder = EntryStateCorpusBuilder(
        corpus_id="sm-rando-landing-natural-v1",
        game_id=GAME,
        contract_bundle_digest=contracts.identity_digest,
        observation_schema_digest=contracts.observation.identity_digest,
        metadata={
            "observation_metadata_version": LANDING_ENTRY_METADATA_VERSION,
            "requested_count": count,
            "source": "power-on MORPH_SPINE[:3] predecessor",
            "intervention_class": "Clean",
            "state_loads": 0,
            "progression_writes": 0,
            "rng": {
                "available": False,
                "reason": "SM general RNG address is not yet contract-mapped",
            },
        },
    )
    env = make_env(GAME, "NONE", GAME_DIR, render_mode=None)
    try:
        env.reset()
        trajectory_digest = _trajectory_digest()
        LANDING_CORPUS_STATE_DIR.mkdir(parents=True, exist_ok=True)
        landing_frames_seen = 0

        def capture(state: Any) -> None:
            nonlocal landing_frames_seen
            if (
                state.room_id != ROOM_LANDING_SITE
                or state.game_state != 8
                or len(builder) >= count
            ):
                return
            landing_frames_seen += 1
            # Spread 64 samples over ~192 natural transition/settle frames;
            # odd stride intentionally covers both emulator frame parities.
            if (landing_frames_seen - 1) % 3:
                return
            index = len(builder)
            state_bytes = env.em.get_state()
            ram = np.asarray(env.get_ram(), dtype=np.uint8)
            state_path = LANDING_CORPUS_STATE_DIR / f"landing_{index:03d}.state"
            write_state_bytes(state_path, state_bytes)
            builder.add(
                state_bytes=state_bytes,
                ram_snapshot=ram.tobytes(),
                state_path=str(state_path.relative_to(REPO_ROOT)),
                source_skill_id="super_metroid.ceres_escape",
                source_segment_id="ceres_to_landing",
                source_trajectory_digest=trajectory_digest,
                frame=state.frame,
                metadata={
                    "observation_metadata_version": LANDING_ENTRY_METADATA_VERSION,
                    "sample_index": index,
                    "frame_parity": state.frame % 2,
                    "room_id": state.room_id,
                    "game_state": state.game_state,
                    "door_transition": state.door_transition,
                    "samus_x": state.samus_x,
                    "samus_x_sub": state.samus_x_sub,
                    "samus_y": state.samus_y,
                    "samus_y_sub": state.samus_y_sub,
                    "velocity_x": state.velocity_x,
                    "velocity_x_sub": state.velocity_x_sub,
                    "velocity_y": state.velocity_y,
                    "velocity_y_sub": state.velocity_y_sub,
                    "health": state.health,
                    "missiles": state.missiles,
                    "pose": state.pose,
                    "enemy0_hp": state.enemy0_hp,
                    "timing": {
                        "source_frame": state.frame,
                        "landing_frame": landing_frames_seen,
                        "transition_direction": state.transition_direction,
                    },
                },
            )

        session = RouteSession(
            env,
            writer=None,
            assist=UnlimitedAmmoAssist(enabled=False),
            graph=MORPH_GRAPH,
            frame_observer=capture,
        )
        splits: list[Split] = []
        play_hops(session, splits, MORPH_SPINE[:3])
        if session.state.room_id != ROOM_LANDING_SITE:
            raise RuntimeError("natural predecessor did not reach Landing Site")
        while len(builder) < count:
            session.step(idle_action(), "landing_entry_corpus_idle")
        corpus = builder.build()
        split = corpus.split(train_fraction=0.8, salt="sm-landing-v1")
        corpus = replace(
            corpus,
            metadata={
                **dict(corpus.metadata),
                "actual_count": len(corpus.records),
                "default_split": split.to_record(),
            },
        )
        corpus.write(output_path)
        return corpus
    finally:
        env.close()


def corpus_summary(corpus: EntryStateCorpus) -> dict[str, Any]:
    split = corpus.split(train_fraction=0.8, salt="sm-landing-v1")
    parities = {int(record.metadata["frame_parity"]) for record in corpus.records}
    game_states = {int(record.metadata["game_state"]) for record in corpus.records}
    return {
        "states": len(corpus.records),
        "train": len(split.train),
        "eval": len(split.eval),
        "frame_parities": sorted(parities),
        "game_states": sorted(game_states),
        "identity_digest": corpus.identity_digest,
    }


def migrate_landing_entry_metadata_v1_to_v2(
    path: Path = LANDING_CORPUS_MANIFEST,
) -> EntryStateCorpus:
    """Reconstruct the added pose field from each retained emulator state."""
    corpus = EntryStateCorpus.load(path)
    if all(
        record.metadata.get("observation_metadata_version")
        == LANDING_ENTRY_METADATA_VERSION
        for record in corpus.records
    ):
        return corpus
    env = make_env(GAME, "NONE", GAME_DIR, render_mode=None)
    try:
        env.reset()
        migrated: list[EntryStateRecord] = []
        for record in corpus.records:
            env.em.set_state(corpus.state_bytes(record, root=REPO_ROOT))
            features = landing_entry_features(np.asarray(env.get_ram()))
            migrated.append(
                replace(
                    record,
                    metadata={
                        **dict(record.metadata),
                        "observation_metadata_version": (
                            LANDING_ENTRY_METADATA_VERSION
                        ),
                        "pose": int(features[11]),
                    },
                )
            )
        result = replace(
            corpus,
            records=tuple(migrated),
            metadata={
                **dict(corpus.metadata),
                "observation_metadata_version": LANDING_ENTRY_METADATA_VERSION,
                "metadata_migration": (
                    "v1_to_v2_pose_reconstructed_from_retained_state"
                ),
            },
        )
        result.write(path)
        return result
    finally:
        env.close()


def evaluate_structured_landing_baseline(
    *,
    corpus_path: Path = LANDING_CORPUS_MANIFEST,
    output_path: Path = LANDING_BASELINE_REPORT,
) -> dict[str, Any]:
    """Measure the existing vanilla skill on train and held-out entry states."""
    corpus = EntryStateCorpus.load(corpus_path)
    contracts = landing_corpus_contracts()
    if corpus.contract_bundle_digest != contracts.identity_digest:
        raise ValueError("corpus contract does not match Landing baseline")
    split = corpus.split(train_fraction=0.8, salt="sm-landing-v1")
    partition_by_digest = {
        record.state_digest: "train" for record in split.train
    }
    partition_by_digest.update(
        {record.state_digest: "eval" for record in split.eval}
    )
    env = make_env(GAME, "NONE", GAME_DIR, render_mode=None)
    attempts: list[dict[str, Any]] = []
    try:
        env.reset()
        for record in corpus.records:
            env.em.set_state(corpus.state_bytes(record, root=REPO_ROOT))
            session = RouteSession(
                env,
                writer=None,
                assist=UnlimitedAmmoAssist(enabled=False),
                graph=MORPH_GRAPH,
            )
            failure: str | None = None
            try:
                play_landing_to_parlor(session)
            except Exception as exc:  # retain counterexamples, continue cohort
                failure = f"{type(exc).__name__}: {exc}"
            success = failure is None and session.state.room_id == ROOM_PARLOR
            attempts.append(
                {
                    "state_digest": record.state_digest,
                    "partition": partition_by_digest[record.state_digest],
                    "success": success,
                    "frames": session.frame,
                    "failure": failure,
                    "final_room_id": session.state.room_id,
                    "entry_metadata": dict(record.metadata),
                }
            )
    finally:
        env.close()

    metrics: dict[str, Any] = {}
    for name in ("train", "eval"):
        values = [attempt for attempt in attempts if attempt["partition"] == name]
        successes = sum(bool(attempt["success"]) for attempt in values)
        metrics[name] = {
            "attempts": len(values),
            "successes": successes,
            "success_rate": successes / len(values),
            "mean_frames": sum(int(value["frames"]) for value in values) / len(values),
        }
    metrics["generalization_gap"] = (
        metrics["train"]["success_rate"] - metrics["eval"]["success_rate"]
    )
    report = {
        "schema_version": 1,
        "experiment": "structured_landing_policy_on_entry_corpus",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "corpus_path": str(corpus_path.relative_to(REPO_ROOT)),
        "corpus_digest": corpus.identity_digest,
        "contract_bundle_digest": contracts.identity_digest,
        "policy": (
            "super_metroid.routes.kpdr.early_spine:play_landing_to_parlor"
        ),
        "intervention_class": "Clean",
        "metrics": metrics,
        "attempts": attempts,
        "claim": (
            "measurement only; no claim of learned or structured superiority"
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


__all__ = [
    "LANDING_CORPUS_MANIFEST",
    "LANDING_CORPUS_SIZE",
    "LANDING_CORPUS_STATE_DIR",
    "LANDING_BASELINE_REPORT",
    "corpus_summary",
    "harvest_landing_entry_corpus",
    "evaluate_structured_landing_baseline",
    "landing_corpus_contracts",
    "landing_entry_features",
    "migrate_landing_entry_metadata_v1_to_v2",
]
