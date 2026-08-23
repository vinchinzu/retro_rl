"""Load pixel CNN or RAM v3 MLP specialists and emit SNES buttons."""

from __future__ import annotations

from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from stable_baselines3 import PPO

from retro_harness.fighters.fighting_env import MK_FIGHTING_ACTIONS
from mortal_kombat.compat import install_fighters_common_alias
from mortal_kombat.ram import snapshot_features, parse_ram
from mortal_kombat.roster import KIND_PIXEL, KIND_RAM_V3

install_fighters_common_alias()

FRAME_SKIP = 4
FRAME_STACK = 4


def _device(kind: str) -> torch.device:
    if kind == KIND_RAM_V3:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def decode_action(action_idx: int) -> np.ndarray:
    buttons = np.zeros(12, dtype=np.int8)
    for btn, val in MK_FIGHTING_ACTIONS[int(action_idx)].items():
        buttons[btn] = val
    return buttons


class PixelPolicy:
    """84×84×4 grayscale stack — old CNN checkpoints."""

    kind = KIND_PIXEL

    def __init__(self, path: Path):
        self.name = path.name
        self.model = PPO.load(str(path), device=_device(KIND_PIXEL))
        self._stack: deque[np.ndarray] = deque(maxlen=FRAME_STACK)

    def reset(self) -> None:
        self._stack.clear()

    def observe_rgb(self, rgb: np.ndarray) -> None:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        self._stack.append(resized)
        while len(self._stack) < FRAME_STACK:
            self._stack.append(resized)

    def act(self, ram: np.ndarray, rgb: np.ndarray | None, *, deterministic: bool = False) -> np.ndarray:
        del ram
        if rgb is not None:
            self.observe_rgb(rgb)
        if len(self._stack) < FRAME_STACK:
            return np.zeros(12, dtype=np.int8)
        obs = np.stack(self._stack, axis=0)
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return decode_action(int(action))


class RamV3Policy:
    """20-dim hitbox RAM vector — overnight specialists."""

    kind = KIND_RAM_V3

    def __init__(self, path: Path):
        self.name = path.name
        self.model = PPO.load(str(path), device=_device(KIND_RAM_V3))
        self._prev = (0, 0)

    def reset(self) -> None:
        self._prev = (0, 0)

    def act(self, ram: np.ndarray, rgb: np.ndarray | None, *, deterministic: bool = False) -> np.ndarray:
        del rgb
        snap = parse_ram(ram)
        obs, self._prev = snapshot_features(snap, self._prev)
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return decode_action(int(action))


def load_policy(path: Path, kind: str):
    install_fighters_common_alias()
    if kind == KIND_RAM_V3:
        return RamV3Policy(path)
    return PixelPolicy(path)
