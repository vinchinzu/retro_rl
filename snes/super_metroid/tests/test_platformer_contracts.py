"""Super Metroid platformer checkpoint contract consumer tests."""

from __future__ import annotations

import numpy as np
import pytest

from retro_harness.contracts import ContractMismatchError
from retro_harness.platformer.contracts import build_platformer_contracts
from retro_harness.platformer.level_config import get_level_config
from retro_harness.platformer.neuro.checkpoint import net_from_dict, net_to_dict
from retro_harness.platformer.neuro.net import NeuralNet, SMB_OUTPUT_BUTTONS
import super_metroid.platformer_levels  # noqa: F401 - registers SM levels


def _sm_features(ram, *, prev_action=None):
    del ram, prev_action
    return np.zeros(5, dtype=np.float32)


def test_super_metroid_neuro_checkpoint_hard_fails_on_feature_flip() -> None:
    config = get_level_config("sm_landing_site")
    contracts = build_platformer_contracts(
        config,
        n_inputs=5,
        read_inputs_fn=_sm_features,
        output_buttons=SMB_OUTPUT_BUTTONS,
    )
    net = NeuralNet(5, 2, len(SMB_OUTPUT_BUTTONS))
    record = net_to_dict(net)
    record["contracts"] = contracts.to_record()
    loaded = net_from_dict(record, expected_contracts=contracts)
    assert loaded.n_inputs == 5

    flipped = build_platformer_contracts(
        config,
        n_inputs=6,
        read_inputs_fn=_sm_features,
        output_buttons=SMB_OUTPUT_BUTTONS,
    )
    with pytest.raises(ContractMismatchError, match="observation"):
        net_from_dict(record, expected_contracts=flipped)
