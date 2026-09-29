"""lean_step must not change emulation: same RAM after the same inputs."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from harvest.runtime import retro_setup

STATE = "Y1_Summer_D13_Morning"


def _ram_digest(lean: bool, frames: int = 240) -> tuple[str, object]:
    env = retro_setup.make_harvest_env(STATE, lean_step=lean)
    env.reset()
    rng = np.random.default_rng(7)
    last = None
    for _ in range(frames):
        action = np.zeros(12, np.uint8)
        action[rng.integers(0, 12)] = rng.integers(0, 2)
        last = env.step(action)
    digest = hashlib.sha256(np.asarray(env.get_ram()).tobytes()).hexdigest()
    frame = env.render()
    env.close()
    return digest, (last, frame)


def test_lean_step_ram_identical_and_render_still_works() -> None:
    if retro_setup.find_harvest_rom() is None or not (
        retro_setup.STATES_DIR / f"{STATE}.state"
    ).exists():
        pytest.skip("needs the Harvest ROM and the pin")
    full, _ = _ram_digest(False)
    lean, (step, frame) = _ram_digest(True)
    assert lean == full
    assert step[0] is None and step[2] is False and step[4] == {}
    assert frame is not None and frame.shape[2] == 3
