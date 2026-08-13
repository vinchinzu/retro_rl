from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from super_metroid.legacy.visual_models import LegacyBCPolicy


@pytest.mark.parametrize(
    ("model_id", "channels", "stack", "mode"),
    (
        ("legacy_navigation_bc", 1, 1, "grayscale"),
        ("legacy_bomb_torizo_bc", 12, 4, "rgb"),
    ),
)
def test_legacy_bc_checkpoint_contract(
    model_id: str,
    channels: int,
    stack: int,
    mode: str,
) -> None:
    policy = LegacyBCPolicy(model_id)

    prediction = policy.predict(np.zeros((224, 256, 3), dtype=np.uint8))

    assert policy.contract.input_channels == channels
    assert policy.contract.frame_stack == stack
    assert policy.contract.color_mode == mode
    assert len(prediction.buttons) == 12
    assert len(prediction.probabilities) == 12
    assert not (prediction.buttons[4] and prediction.buttons[5])
    assert not (prediction.buttons[6] and prediction.buttons[7])


def test_legacy_bc_rejects_wrong_observation_shape() -> None:
    policy = LegacyBCPolicy("legacy_navigation_bc")

    with pytest.raises(ValueError, match="224x256"):
        policy.predict(np.zeros((112, 128, 3), dtype=np.uint8))
