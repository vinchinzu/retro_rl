"""Save / load neuroevolution checkpoints and button replays."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from retro_harness.platformer.neuro.net import NeuralNet

if TYPE_CHECKING:
    from retro_harness.platformer.evaluator import Evaluator
    from retro_harness.platformer.neuro.train import NeuroIndividual


def net_to_dict(net: NeuralNet, *, generation: int = 0, fitness: float = 0.0,
                completed: bool = False, total_frames: int = 0,
                max_progress: float = 0.0) -> dict:
    """Serialize network + training meta for JSON."""
    return {
        "weights": net.weights.tolist(),
        "n_inputs": net.n_inputs,
        "n_hidden": net.n_hidden,
        "n_outputs": net.n_outputs,
        "arch": net.arch,
        "hidden_layers": list(net.hidden_layers),
        "n_conv": net.n_conv,
        "use_recurrent": net.use_recurrent,
        "generation": generation,
        "fitness": fitness,
        "completed": completed,
        "total_frames": total_frames,
        "max_progress": max_progress,
    }


def net_from_dict(data: dict) -> NeuralNet:
    """Load a NeuralNet from a checkpoint dict."""
    return NeuralNet(
        n_inputs=int(data["n_inputs"]),
        n_hidden=int(data["n_hidden"]),
        n_outputs=int(data["n_outputs"]),
        weights=np.array(data["weights"], dtype=np.float32),
        arch=data.get("arch", "mlp"),
        hidden_layers=tuple(data.get("hidden_layers", (data["n_hidden"],))),
        n_conv=int(data.get("n_conv", 8)),
        use_recurrent=bool(data.get("use_recurrent", False)),
    )


def save_neuro_checkpoint(
    best: NeuroIndividual,
    gen: int,
    output_dir: Path,
    evaluator: Evaluator,
    max_frames: int,
    read_inputs_fn: Callable,
) -> None:
    """Save best network weights + replay its buttons."""
    from retro_harness.platformer.neuro.train import evaluate_network

    net_path = output_dir / "neuro_best.json"
    data = net_to_dict(
        best.net,
        generation=gen,
        fitness=best.fitness,
        completed=best.result.completed if best.result else False,
        total_frames=best.result.total_frames if best.result else 0,
        max_progress=best.result.max_progress if best.result else 0,
    )
    net_path.write_text(json.dumps(data, indent=2))

    buttons, _ = evaluate_network(best.net, evaluator, max_frames, read_inputs_fn)
    btn_path = output_dir / "neuro_best_buttons.json"
    btn_data = {
        "raw_buttons": buttons,
        "num_frames": len(buttons),
        "fitness": best.fitness,
        "completed": best.result.completed if best.result else False,
        "max_progress": best.result.max_progress if best.result else 0,
    }
    btn_path.write_text(json.dumps(btn_data, indent=2))


# Back-compat private name used by older call sites / docs.
_save_neuro_checkpoint = save_neuro_checkpoint
