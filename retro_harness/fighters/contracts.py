"""Canonical contracts for the shared fighting-game PPO stack."""

from __future__ import annotations

from importlib import metadata
from pathlib import Path
from typing import Any, Mapping, Sequence

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
    action_rows_from_sparse_maps,
    identity_digest,
    sha256_file,
)


def _core_identity() -> str:
    try:
        version = metadata.version("stable-retro")
    except metadata.PackageNotFoundError:
        version = "unavailable"
    return identity_digest("stable-retro-core", version)


def _integration_dir(game_dir: Path, game_id: str) -> Path:
    return game_dir / "custom_integrations" / game_id


def _rom_identity(game_dir: Path, game_id: str) -> str:
    integration = _integration_dir(game_dir, game_id)
    for name in ("rom.sfc", "rom.smc", "rom.nes", "rom.bin"):
        candidate = integration / name
        if candidate.is_file():
            return sha256_file(candidate)
    raise FileNotFoundError(f"no integration ROM found under {integration}")


def _state_identity(game_dir: Path, game_id: str, state: str) -> str:
    if state == "NONE":
        return identity_digest("stable-retro-state", "power-on")
    path = _integration_dir(game_dir, game_id) / f"{state}.state"
    return sha256_file(path) if path.is_file() else identity_digest("state-name", state)


def build_fighter_contracts(
    *,
    game_id: str,
    game_dir: str | Path,
    state: str,
    action_maps: Sequence[Mapping[int, int]],
    reward_weights: Mapping[str, float],
    frame_skip: int,
    frame_stack: int,
    practice: bool = False,
    direct_ram: bool = False,
    monitor: bool = False,
    randomize_state: bool = False,
) -> ContractBundle:
    game_path = Path(game_dir).resolve()
    action = ActionContract.from_button_rows(
        action_rows_from_sparse_maps(action_maps),
        controller_buttons=SNES_BUTTONS,
    )
    observation = ObservationContract(
        fields=(
            ObservationField(
                "grayscale_frames",
                "uint8",
                (frame_stack, 84, 84),
                "ordered newest-last grayscale emulator frames",
            ),
        ),
        preprocessing={
            "colorspace": "RGB_TO_GRAY",
            "resize": [84, 84],
            "resize_interpolation": "INTER_AREA",
            "frame_stack": frame_stack,
        },
    )
    reward = RewardContract(
        components=tuple(
            RewardComponent(name, weight, f"FightingEnv.{name}")
            for name, weight in reward_weights.items()
        )
    )
    wrappers: list[WrapperSpec] = [WrapperSpec("stable_retro.RetroEnv")]
    if practice:
        wrappers.append(WrapperSpec("NullP2Wrapper", {"players": 2}))
    if direct_ram:
        wrappers.append(WrapperSpec("DirectRAMReader"))
    if frame_skip > 1:
        wrappers.append(WrapperSpec("FrameSkip", {"n_skip": frame_skip}))
    wrappers.extend(
        (
            WrapperSpec("GrayscaleResize", {"width": 84, "height": 84}),
            WrapperSpec("FightingEnv", {"randomize_state": randomize_state}),
            WrapperSpec("DiscreteAction", {"action_count": action.action_count}),
        )
    )
    if frame_stack > 1:
        wrappers.append(WrapperSpec("FrameStack", {"n_frames": frame_stack}))
    if monitor:
        wrappers.append(WrapperSpec("stable_baselines3.Monitor"))
    environment = EnvironmentContract(
        game_id=game_id,
        state_id=state,
        action_space_size=action.action_count,
        frame_skip=frame_skip,
        players=2 if practice else 1,
        rom_identity_digest=_rom_identity(game_path, game_id),
        state_identity_digest=_state_identity(game_path, game_id, state),
        core_identity_digest=_core_identity(),
        metadata={"randomize_state": randomize_state},
    )
    return ContractBundle(
        environment=environment,
        observation=observation,
        action=action,
        reward=reward,
        wrappers=WrapperContract(tuple(wrappers)),
    )


def fighting_reward_weights(fighting_env_type: type[Any]) -> dict[str, float]:
    return {
        "damage_dealt": fighting_env_type.REWARD_DAMAGE_DEALT,
        "damage_taken": fighting_env_type.REWARD_DAMAGE_TAKEN,
        "round_win": fighting_env_type.REWARD_ROUND_WIN,
        "round_loss": fighting_env_type.REWARD_ROUND_LOSS,
        "double_ko": fighting_env_type.REWARD_DOUBLE_KO,
        "match_win": fighting_env_type.REWARD_MATCH_WIN,
        "time_penalty": fighting_env_type.REWARD_TIME_PENALTY,
        "timeout_round": fighting_env_type.REWARD_TIMEOUT_ROUND,
    }


__all__ = ["build_fighter_contracts", "fighting_reward_weights"]
