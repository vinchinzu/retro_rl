"""V3 training run spec and artifact naming (no torch / sb3 / gym)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mortal_kombat.roster import v3_filename


@dataclass(frozen=True)
class V3Run:
    state: str
    stage: str
    steps: int
    n_envs: int = 2
    load: str | None = None
    candidate: str | None = None  # required when load is set; must differ from stage
    learning_rate: float = 3e-4
    ent_coef_start: float = 0.01
    ent_coef_end: float = 0.0002
    max_seconds: float = 0
    randomize_state: bool = False

    def __post_init__(self) -> None:
        if self.load and (not self.candidate or self.candidate == self.stage):
            raise ValueError(
                "continuations require a distinct --output-prefix "
                "so the incumbent is not overwritten"
            )

    @property
    def output_stage(self) -> str:
        return self.candidate or self.stage


@dataclass(frozen=True)
class TrainResult:
    path: Path
    wall_stopped: bool
    timesteps: int


def v3_final_name(stage: str) -> str:
    return v3_filename(stage)


def v3_steps_name(stage: str, timesteps: int) -> str:
    return f"mk1_v3_{stage}_ppo_{int(timesteps)}_steps.zip"


def v3_artifact_name(stage: str, *, wall_stopped: bool, timesteps: int) -> str:
    if wall_stopped:
        return v3_steps_name(stage, timesteps)
    return v3_final_name(stage)
