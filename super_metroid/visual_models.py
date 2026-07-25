"""Exact observation/action adapters for the imported legacy BC models."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np

from super_metroid.models import LegacyModel, load_model_registry


@dataclass(frozen=True)
class ModelContract:
    model_id: str
    input_channels: int
    height: int
    width: int
    output_buttons: int
    frame_stack: int
    color_mode: str


@dataclass(frozen=True)
class ModelPrediction:
    buttons: tuple[int, ...]
    probabilities: tuple[float, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class LegacyBCPolicy:
    """Hash-checked PyTorch inference with the original resize/stack contract."""

    def __init__(self, model_id: str, *, threshold: float = 0.5) -> None:
        try:
            import torch
            from torch import nn
        except ImportError as exc:  # pragma: no cover - optional ML dependency
            raise RuntimeError("install the 'ml' extra to use legacy models") from exc

        registry = load_model_registry()
        if model_id not in {"legacy_navigation_bc", "legacy_bomb_torizo_bc"}:
            raise ValueError(f"{model_id!r} is not a supported BC model")
        model = registry[model_id]
        self._verify_checkpoint(model)
        state_dict = torch.load(model.path, map_location="cpu", weights_only=True)
        input_channels = int(state_dict["features.0.weight"].shape[1])
        output_buttons = int(state_dict["fc.2.weight"].shape[0])
        if input_channels == 1:
            frame_stack = 1
            color_mode = "grayscale"
        elif input_channels % 3 == 0:
            frame_stack = input_channels // 3
            color_mode = "rgb"
        else:
            raise ValueError(f"unsupported input channel count {input_channels}")
        if output_buttons != 12:
            raise ValueError(f"expected 12 SNES outputs, got {output_buttons}")

        class _Policy(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=4, stride=2),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1),
                    nn.ReLU(),
                    nn.Flatten(),
                )
                self.fc = nn.Sequential(
                    nn.Linear(7680, 512),
                    nn.ReLU(),
                    nn.Linear(512, output_buttons),
                )

            def forward(self, value):
                return self.fc(self.features(value.float() / 255.0))

        self.model_info = model
        self.contract = ModelContract(
            model_id=model_id,
            input_channels=input_channels,
            height=112,
            width=128,
            output_buttons=output_buttons,
            frame_stack=frame_stack,
            color_mode=color_mode,
        )
        self.threshold = float(threshold)
        self._torch = torch
        self._network = _Policy()
        self._network.load_state_dict(state_dict)
        self._network.eval()
        self._frames: deque[np.ndarray] = deque(maxlen=frame_stack)

    @staticmethod
    def _verify_checkpoint(model: LegacyModel) -> None:
        if not model.path.is_file():
            raise FileNotFoundError(model.path)
        actual = _sha256(model.path)
        if actual != model.sha256:
            raise ValueError(
                f"checkpoint hash mismatch for {model.model_id}: {actual}"
            )

    def reset(self) -> None:
        self._frames.clear()

    def _preprocess(self, observation: np.ndarray) -> np.ndarray:
        if observation.shape != (224, 256, 3):
            raise ValueError(
                f"expected 224x256 RGB observation, got {observation.shape}"
            )
        reduced = np.asarray(observation[::2, ::2, :], dtype=np.uint8)
        if self.contract.color_mode == "grayscale":
            grayscale = np.dot(reduced[..., :3], (0.299, 0.587, 0.114))
            return np.asarray(grayscale, dtype=np.uint8)[None, ...]
        return reduced.transpose(2, 0, 1)

    def predict(self, observation: np.ndarray) -> ModelPrediction:
        frame = self._preprocess(observation)
        if not self._frames:
            self._frames.extend(frame.copy() for _ in range(self.contract.frame_stack))
        else:
            self._frames.append(frame)
        stacked = np.concatenate(tuple(self._frames), axis=0)
        tensor = self._torch.from_numpy(stacked).unsqueeze(0)
        with self._torch.no_grad():
            probabilities = self._torch.sigmoid(self._network(tensor))[0]
        action = (probabilities >= self.threshold).to(self._torch.int8).cpu().numpy()
        # Preserve the original sanitizer: never press opposite directions.
        if action[4] and action[5]:
            action[4] = 0
            action[5] = 0
        if action[6] and action[7]:
            action[6] = 0
            action[7] = 0
        return ModelPrediction(
            buttons=tuple(int(value) for value in action),
            probabilities=tuple(float(value) for value in probabilities.cpu().numpy()),
        )
