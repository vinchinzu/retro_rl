"""Observation builders for neuroevolution — no per-game RAM maps here.

SMB observation vectors live in ``smb.obs``. Shared training code accepts an
injected ``obs_fn: Callable[[ram], np.ndarray]`` (optionally with
``prev_action=``). Thin wrappers re-export SMB builders when the game package
is importable; otherwise they fail with a clear error (no inlined RAM fallback).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

# Architecture-level grid size (also in net.py); not a RAM map.
from retro_harness.platformer.neuro.net import GRID_SIZE, N_GRID, grid_as_image

ObsFn = Callable[..., np.ndarray]

_SMB_OBS_ERR = (
    "SMB observation builder unavailable (could not import smb.obs). "
    "Ensure the smb package is on PYTHONPATH (nes/), or pass an explicit "
    "obs_fn / read_inputs_fn to neuro training."
)


def _import_smb_obs():
    """Import smb.obs or raise ImportError with a clear message."""
    try:
        import smb.obs as obs  # type: ignore
        return obs
    except Exception as e:  # pragma: no cover
        raise ImportError(_SMB_OBS_ERR) from e


def read_smb_inputs(
    ram: np.ndarray,
    prev_action: Sequence[int] | None = None,
) -> np.ndarray:
    """Rich SMB observation (210 dims). Delegates to ``smb.obs`` — no RAM here."""
    obs = _import_smb_obs()
    try:
        return obs.read_smb_inputs(ram, prev_action=prev_action)
    except TypeError:
        return obs.read_smb_inputs(ram)


def read_smb_inputs_legacy(ram: np.ndarray) -> np.ndarray:
    """189-dim legacy SMB observation. Delegates to ``smb.obs``."""
    obs = _import_smb_obs()
    return obs.read_smb_inputs_legacy(ram)


def default_obs_fn() -> ObsFn:
    """Return the default SMB rich obs builder, or raise if smb is missing."""
    _import_smb_obs()  # fail early with clear message
    return read_smb_inputs


def resolve_obs_fn(obs_fn: ObsFn | None) -> ObsFn:
    """Use *obs_fn* if given; else default SMB builder (must be importable)."""
    if obs_fn is not None:
        return obs_fn
    return default_obs_fn()


def _smb_constants() -> dict[str, Any]:
    """Load dim constants from smb.obs when available."""
    obs = _import_smb_obs()
    return {
        "N_SMB_INPUTS": int(obs.N_SMB_INPUTS),
        "N_SMB_INPUTS_LEGACY": int(obs.N_SMB_INPUTS_LEGACY),
        "GRID_SIZE": int(getattr(obs, "GRID_SIZE", GRID_SIZE)),
        "N_GRID": int(getattr(obs, "N_GRID", N_GRID)),
    }


# Eager constants for tests / arch sizing when smb is on the path (normal monorepo).
try:
    _c = _smb_constants()
    N_SMB_INPUTS: int | None = _c["N_SMB_INPUTS"]
    N_SMB_INPUTS_LEGACY: int | None = _c["N_SMB_INPUTS_LEGACY"]
except Exception:  # pragma: no cover
    N_SMB_INPUTS = None
    N_SMB_INPUTS_LEGACY = None


__all__ = [
    "GRID_SIZE",
    "N_GRID",
    "N_SMB_INPUTS",
    "N_SMB_INPUTS_LEGACY",
    "ObsFn",
    "default_obs_fn",
    "grid_as_image",
    "read_smb_inputs",
    "read_smb_inputs_legacy",
    "resolve_obs_fn",
]
