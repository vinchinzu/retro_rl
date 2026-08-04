"""Neural network policy: MLP / tiny-CNN, activations, output combos."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

# Local tile window size for CNN arch (MarI/O-style 13×13). Not a RAM address —
# observation builders supply the grid; this only sizes the conv head.
GRID_SIZE = 13
N_GRID = GRID_SIZE * GRID_SIZE  # 169

ArchName = Literal["mlp", "cnn_mlp"]


def grid_as_image(inputs: np.ndarray) -> np.ndarray:
    """Reshape the leading grid portion to (1, GRID_SIZE, GRID_SIZE) for CNN."""
    return np.asarray(inputs[:N_GRID], dtype=np.float32).reshape(1, GRID_SIZE, GRID_SIZE)


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def _mlp_weight_count(n_in: int, hidden: Sequence[int], n_out: int) -> int:
    dims = [n_in, *list(hidden), n_out]
    total = 0
    for a, b in zip(dims, dims[1:]):
        total += a * b + b
    return total


def _mlp_forward(x: np.ndarray, w: np.ndarray, hidden: Sequence[int], n_out: int) -> np.ndarray:
    dims = [int(x.shape[-1]), *list(hidden), n_out]
    idx = 0
    h = x.astype(np.float32, copy=False)
    for li, (a, b) in enumerate(zip(dims, dims[1:])):
        mat = w[idx : idx + a * b].reshape(a, b)
        idx += a * b
        bias = w[idx : idx + b]
        idx += b
        h = h @ mat + bias
        if li < len(dims) - 2:
            h = _relu(h)
        else:
            h = _sigmoid(h)
    return h


def _conv3x3_weight_count(in_ch: int, out_ch: int) -> int:
    return in_ch * out_ch * 9 + out_ch


def _cnn_feature_dim(n_conv: int = 8) -> int:
    # 13×13 valid 3×3 → 11×11; 2×2 avg pool → 5×5
    return n_conv * 5 * 5


def _conv3x3(img: np.ndarray, w: np.ndarray, in_ch: int, out_ch: int) -> tuple[np.ndarray, int]:
    k = w[: in_ch * out_ch * 9].reshape(out_ch, in_ch, 3, 3)
    b = w[in_ch * out_ch * 9 : in_ch * out_ch * 9 + out_ch]
    _, h, ww = img.shape
    out_h, out_w = h - 2, ww - 2
    out = np.zeros((out_ch, out_h, out_w), dtype=np.float32)
    for oc in range(out_ch):
        acc = np.zeros((out_h, out_w), dtype=np.float32)
        for ic in range(in_ch):
            kernel = k[oc, ic]
            src = img[ic]
            for dy in range(3):
                for dx in range(3):
                    acc += src[dy : dy + out_h, dx : dx + out_w] * float(kernel[dy, dx])
        out[oc] = _relu(acc + float(b[oc]))
    return out, in_ch * out_ch * 9 + out_ch


@dataclass
class NeuralNet:
    """Feedforward or tiny-CNN policy with a flat weight vector for GA.

    ``arch='mlp'``: flat ``n_inputs`` → hidden layers → outputs.
    ``arch='cnn_mlp'``: grid CNN + concat non-grid features → MLP head.
    ``hidden_layers``: empty → single layer of size ``n_hidden``.
    ``use_recurrent``: exponential output smoother with evolved alpha (timing).
    """

    n_inputs: int
    n_hidden: int
    n_outputs: int
    weights: np.ndarray = field(default_factory=lambda: np.array([]))
    arch: ArchName = "mlp"
    hidden_layers: tuple[int, ...] = ()
    n_conv: int = 8
    use_recurrent: bool = False

    def __post_init__(self) -> None:
        if not self.hidden_layers:
            object.__setattr__(self, "hidden_layers", (int(self.n_hidden),))
        else:
            object.__setattr__(self, "n_hidden", int(self.hidden_layers[0]))
        if len(self.weights) == 0:
            self.weights = np.random.randn(self.total_weights).astype(np.float32) * 0.5
        self._prev_out: np.ndarray | None = None

    def reset_state(self) -> None:
        self._prev_out = None

    @property
    def total_weights(self) -> int:
        if self.arch == "mlp":
            n = _mlp_weight_count(self.n_inputs, self.hidden_layers, self.n_outputs)
        else:
            conv_n = _conv3x3_weight_count(1, self.n_conv)
            feat = _cnn_feature_dim(self.n_conv)
            rest = max(0, self.n_inputs - N_GRID)
            n = conv_n + _mlp_weight_count(feat + rest, self.hidden_layers, self.n_outputs)
        if self.use_recurrent:
            n += 1  # single alpha logit
        return n

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        x = np.asarray(inputs, dtype=np.float32).reshape(-1)
        if x.shape[0] != self.n_inputs:
            if x.shape[0] < self.n_inputs:
                pad = np.zeros(self.n_inputs, dtype=np.float32)
                pad[: x.shape[0]] = x
                x = pad
            else:
                x = x[: self.n_inputs]

        if self.arch == "mlp":
            out = _mlp_forward(x, self.weights, self.hidden_layers, self.n_outputs)
        else:
            img = grid_as_image(x)
            conv_out, idx = _conv3x3(img, self.weights, 1, self.n_conv)
            c, h, ww = conv_out.shape
            ph, pw = h // 2, ww // 2
            pooled = (
                conv_out[:, : ph * 2, : pw * 2]
                .reshape(c, ph, 2, pw, 2)
                .mean(axis=(2, 4))
                .reshape(-1)
            )
            rest = x[N_GRID:]
            feats = np.concatenate([pooled, rest])
            out = _mlp_forward(
                feats, self.weights[idx:], self.hidden_layers, self.n_outputs
            )

        if self.use_recurrent:
            alpha = float(_sigmoid(np.array([self.weights[-1]], dtype=np.float32))[0])
            if self._prev_out is None:
                self._prev_out = out
            out = alpha * out + (1.0 - alpha) * self._prev_out
            self._prev_out = out.copy()
        return out

    def copy(self) -> NeuralNet:
        child = NeuralNet(
            n_inputs=self.n_inputs,
            n_hidden=self.n_hidden,
            n_outputs=self.n_outputs,
            weights=self.weights.copy(),
            arch=self.arch,
            hidden_layers=self.hidden_layers,
            n_conv=self.n_conv,
            use_recurrent=self.use_recurrent,
        )
        return child


# Discrete action combos for NES-style platformer policies (B=0, LEFT=6, RIGHT=7, A=8).
# Named historically for SMB; usable for any game sharing that button layout.
SMB_OUTPUT_BUTTONS = [
    [7],  # RIGHT
    [7, 0],  # RIGHT + B
    [7, 0, 8],  # RIGHT + B + A
    [7, 8],  # RIGHT + A
    [8],  # JUMP
    [6],  # LEFT
    [0],  # B
    [6, 0],  # LEFT + B
    [6, 0, 8],  # LEFT + B + A
]


def outputs_to_buttons(outputs: np.ndarray) -> list[int]:
    """Convert network outputs to 12-element button array.

    Softmax-style: if any output > 0.5, take the argmax combo only (avoids
    conflicting multi-sigmoid fires). Falls back to multi-threshold if all
    outputs are mid-range weak.
    """
    buttons = [0] * 12
    if len(outputs) == 0:
        return buttons
    best = int(np.argmax(outputs))
    if float(outputs[best]) > 0.5 and best < len(SMB_OUTPUT_BUTTONS):
        for btn_idx in SMB_OUTPUT_BUTTONS[best]:
            if btn_idx < 12:
                buttons[btn_idx] = 1
        return buttons
    for i, val in enumerate(outputs):
        if val > 0.5 and i < len(SMB_OUTPUT_BUTTONS):
            for btn_idx in SMB_OUTPUT_BUTTONS[i]:
                if btn_idx < 12:
                    buttons[btn_idx] = 1
    return buttons


def buttons_to_output_target(buttons: Sequence[int]) -> np.ndarray:
    """Map a button frame to a soft target over SMB_OUTPUT_BUTTONS (for BC)."""
    target = np.zeros(len(SMB_OUTPUT_BUTTONS), dtype=np.float32)
    bset = {i for i, v in enumerate(buttons) if int(v)}
    best_i, best_overlap = 0, -1
    for i, combo in enumerate(SMB_OUTPUT_BUTTONS):
        cset = set(combo)
        if cset == bset & cset and cset <= bset and len(cset) == len(bset):
            target[i] = 1.0
            return target
        overlap = len(cset & bset) - 0.25 * len(cset - bset)
        if overlap > best_overlap:
            best_overlap = overlap
            best_i = i
    target[best_i] = 1.0
    return target
