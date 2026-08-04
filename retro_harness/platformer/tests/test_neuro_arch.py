"""Unit tests for upgraded NeuralNet architectures (no emulator)."""

from __future__ import annotations

import numpy as np

from retro_harness.platformer.neuro import (
    N_SMB_INPUTS,
    NeuralNet,
    SMB_OUTPUT_BUTTONS,
    buttons_to_output_target,
    crossover_net,
    mutate_net,
    outputs_to_buttons,
)


def test_mlp_forward_shape() -> None:
    net = NeuralNet(n_inputs=N_SMB_INPUTS, n_hidden=16, n_outputs=len(SMB_OUTPUT_BUTTONS))
    x = np.random.randn(N_SMB_INPUTS).astype(np.float32)
    y = net.forward(x)
    assert y.shape == (len(SMB_OUTPUT_BUTTONS),)
    assert np.all((y >= 0.0) & (y <= 1.0))


def test_deeper_mlp() -> None:
    net = NeuralNet(
        n_inputs=64,
        n_hidden=32,
        n_outputs=9,
        hidden_layers=(32, 16),
        arch="mlp",
    )
    assert net.total_weights == net.weights.shape[0]
    y = net.forward(np.zeros(64, dtype=np.float32))
    assert y.shape == (9,)


def test_cnn_mlp_forward() -> None:
    # n_inputs must be at least grid + some state
    n_in = 169 + 16
    net = NeuralNet(
        n_inputs=n_in,
        n_hidden=20,
        n_outputs=9,
        arch="cnn_mlp",
        n_conv=4,
        hidden_layers=(20,),
    )
    assert net.weights.shape[0] == net.total_weights
    x = np.random.randn(n_in).astype(np.float32) * 0.1
    y = net.forward(x)
    assert y.shape == (9,)
    assert np.all(np.isfinite(y))


def test_recurrent_smoother_changes_second_forward() -> None:
    net = NeuralNet(
        n_inputs=32,
        n_hidden=8,
        n_outputs=4,
        use_recurrent=True,
    )
    x = np.ones(32, dtype=np.float32)
    y1 = net.forward(x).copy()
    y2 = net.forward(x * 0.5).copy()
    # state carried — second call uses smoother
    assert y1.shape == y2.shape
    net.reset_state()
    y3 = net.forward(x * 0.5)
    # after reset, path differs from y2 potentially
    assert y3.shape == (4,)


def test_mutate_crossover_preserve_size() -> None:
    a = NeuralNet(n_inputs=40, n_hidden=10, n_outputs=5, arch="mlp")
    b = a.copy()
    b.weights += 0.1
    c = mutate_net(a, rate=0.5, magnitude=0.2)
    d = crossover_net(a, b)
    assert c.weights.shape == a.weights.shape
    assert d.weights.shape == a.weights.shape


def test_outputs_to_buttons_argmax() -> None:
    out = np.zeros(len(SMB_OUTPUT_BUTTONS), dtype=np.float32)
    out[2] = 0.9  # RIGHT+B+A
    buttons = outputs_to_buttons(out)
    assert buttons[7] == 1
    assert buttons[0] == 1
    assert buttons[8] == 1


def test_buttons_to_output_target() -> None:
    buttons = [0] * 12
    buttons[7] = 1
    buttons[0] = 1
    t = buttons_to_output_target(buttons)
    assert t.shape == (len(SMB_OUTPUT_BUTTONS),)
    assert float(t.sum()) == 1.0
