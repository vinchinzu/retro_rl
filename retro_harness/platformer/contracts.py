"""Contract builder for RAM-observation platformer neural policies."""

from __future__ import annotations

from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Sequence

from retro_harness.contracts import (
    ActionContract,
    ContractBundle,
    EnvironmentContract,
    ObservationContract,
    ObservationField,
    RewardComponent,
    RewardContract,
    SNES_BUTTONS,
    WrapperContract,
    WrapperSpec,
    identity_digest,
    sha256_file,
)
from retro_harness.platformer.level_config import LevelConfig


def callable_identity(value: Callable[..., Any]) -> str:
    module = getattr(value, "__module__", type(value).__module__)
    name = getattr(value, "__qualname__", type(value).__qualname__)
    return f"{module}:{name}"


def _core_identity() -> str:
    try:
        version = metadata.version("stable-retro")
    except metadata.PackageNotFoundError:
        version = "unavailable"
    return identity_digest("stable-retro-core", version)


def _integration_dir(config: LevelConfig) -> Path:
    return config.game_dir / "custom_integrations" / config.game_name


def _rom_identity(config: LevelConfig) -> str:
    integration = _integration_dir(config)
    for name in ("rom.sfc", "rom.smc", "rom.nes", "rom.bin"):
        candidate = integration / name
        if candidate.is_file():
            return sha256_file(candidate)
    raise FileNotFoundError(f"no integration ROM found under {integration}")


def _state_identity(config: LevelConfig) -> str:
    if config.start_state == "NONE":
        return identity_digest("stable-retro-state", "power-on")
    path = _integration_dir(config) / f"{config.start_state}.state"
    if path.is_file():
        return sha256_file(path)
    return identity_digest("state-name", config.start_state)


def _button_rows(output_buttons: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    rows: list[tuple[int, ...]] = []
    for combo in output_buttons:
        row = [0] * len(SNES_BUTTONS)
        for index in combo:
            if not 0 <= int(index) < len(row):
                raise ValueError("platformer output button is out of range")
            row[int(index)] = 1
        rows.append(tuple(row))
    return tuple(rows)


def build_platformer_contracts(
    config: LevelConfig,
    *,
    n_inputs: int,
    read_inputs_fn: Callable[..., Any],
    output_buttons: Sequence[Sequence[int]],
) -> ContractBundle:
    action = ActionContract.from_button_rows(
        _button_rows(output_buttons),
        controller_buttons=SNES_BUTTONS,
    )
    reader_id = callable_identity(read_inputs_fn)
    observation = ObservationContract(
        fields=(
            ObservationField(
                "policy_features",
                "float32",
                (n_inputs,),
                f"ordered output of {reader_id}",
            ),
        ),
        preprocessing={
            "reader": reader_id,
            "ram_schema": config.ram_schema.to_dict(),
        },
    )
    reward = RewardContract(
        components=(
            RewardComponent("progress", config.progress_weight, "max progress"),
            RewardComponent("death", -config.death_penalty, "death penalty"),
            RewardComponent(
                "completion", config.completion_bonus, "level completion"
            ),
            RewardComponent(
                "time", -config.time_bonus_weight, "elapsed frame cost"
            ),
        )
    )
    wrappers = WrapperContract(
        (
            WrapperSpec("stable_retro.RetroEnv"),
            WrapperSpec(
                "RAMSchema",
                {"schema": config.ram_schema.to_dict()},
            ),
            WrapperSpec("ObservationReader", {"callable": reader_id}),
            WrapperSpec("NeuralNetPolicy", {"output_count": action.action_count}),
        )
    )
    environment = EnvironmentContract(
        game_id=config.game_name,
        state_id=config.start_state,
        action_space_size=action.action_count,
        frame_skip=1,
        rom_identity_digest=_rom_identity(config),
        state_identity_digest=_state_identity(config),
        core_identity_digest=_core_identity(),
        metadata={
            "level_id": config.level_id,
            "target_level_id": config.target_level_id,
        },
    )
    return ContractBundle(environment, observation, action, reward, wrappers)


__all__ = ["build_platformer_contracts", "callable_identity"]
