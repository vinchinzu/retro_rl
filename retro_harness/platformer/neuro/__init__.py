"""Neuroevolution for platformer speedruns.

Inspired by MarI/O (SethBling's NEAT for Super Mario). Instead of evolving
raw button sequences (which break on crossover), evolve neural networks
that read game state and output button presses.

Architecture options (``arch``):
- ``mlp``: flat observation → 1–2 hidden layers (ReLU) → sigmoid combo outputs
- ``cnn_mlp``: 13×13 grid → 3×3 conv + pool → concat state → MLP head

Observations are **injected** via ``obs_fn`` / ``read_inputs_fn``. For SMB the
default is ``smb.obs.read_smb_inputs`` (210 dims; use ``read_smb_inputs_legacy``
for 189-dim checkpoints). Shared code holds **no** per-game RAM maps.

Public import path (back-compat)::

    from retro_harness.platformer.neuro import NeuralNet, run_neuro_ga, ...
"""

from __future__ import annotations

from retro_harness.platformer.neuro.checkpoint import (
    net_from_dict,
    net_to_dict,
    save_neuro_checkpoint,
    _save_neuro_checkpoint,
)
from retro_harness.platformer.neuro.net import (
    GRID_SIZE,
    N_GRID,
    ArchName,
    NeuralNet,
    SMB_OUTPUT_BUTTONS,
    buttons_to_output_target,
    grid_as_image,
    outputs_to_buttons,
)
from retro_harness.platformer.neuro.obs import (
    N_SMB_INPUTS,
    N_SMB_INPUTS_LEGACY,
    default_obs_fn,
    read_smb_inputs,
    read_smb_inputs_legacy,
    resolve_obs_fn,
)
from retro_harness.platformer.neuro.train import (
    NeuroIndividual,
    behavior_clone_init,
    collect_bc_dataset,
    crossover_net,
    evaluate_network,
    mutate_net,
    run_neuro_ga,
    warm_start_from_seed,
)

__all__ = [
    "ArchName",
    "GRID_SIZE",
    "N_GRID",
    "N_SMB_INPUTS",
    "N_SMB_INPUTS_LEGACY",
    "NeuralNet",
    "NeuroIndividual",
    "SMB_OUTPUT_BUTTONS",
    "behavior_clone_init",
    "buttons_to_output_target",
    "collect_bc_dataset",
    "crossover_net",
    "default_obs_fn",
    "evaluate_network",
    "grid_as_image",
    "mutate_net",
    "net_from_dict",
    "net_to_dict",
    "outputs_to_buttons",
    "read_smb_inputs",
    "read_smb_inputs_legacy",
    "resolve_obs_fn",
    "run_neuro_ga",
    "save_neuro_checkpoint",
    "warm_start_from_seed",
    "_save_neuro_checkpoint",
]
