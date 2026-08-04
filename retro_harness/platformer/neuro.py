"""Neuroevolution for platformer speedruns.

Inspired by MarI/O (SethBling's NEAT for Super Mario). Instead of evolving
raw button sequences (which break on crossover), evolve neural networks
that read game state and output button presses.

Architecture options (``arch``):
- ``mlp``: flat observation → 1–2 hidden layers (ReLU) → sigmoid combo outputs
- ``cnn_mlp``: 13×13 grid → 3×3 conv + pool → concat state → MLP head

Observations default to the richer builder in ``smb.obs`` (210 dims: velocities,
grounded, timer, camera). Use ``read_smb_inputs_legacy`` (189) for old
checkpoints. Optional behavior-clone warm-start from an RLE seed.
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np

from retro_harness.platformer.evaluator import Evaluator, EvalResult

# -- Observation builder (prefer smb.obs) --------------------------------------

try:
    from smb.obs import (  # type: ignore
        GRID_SIZE,
        N_GRID,
        N_SMB_INPUTS as _OBS_N,
        N_SMB_INPUTS_LEGACY,
        grid_as_image,
        read_smb_inputs as _obs_read,
        read_smb_inputs_legacy,
    )

    N_SMB_INPUTS = int(_OBS_N)
except Exception:  # pragma: no cover
    N_SMB_INPUTS = 189
    N_SMB_INPUTS_LEGACY = 189
    N_GRID = 169
    GRID_SIZE = 13
    read_smb_inputs_legacy = None  # type: ignore
    _obs_read = None  # type: ignore

    def grid_as_image(inputs: np.ndarray) -> np.ndarray:
        return np.asarray(inputs[:N_GRID], dtype=np.float32).reshape(1, GRID_SIZE, GRID_SIZE)


def _legacy_inline(ram: np.ndarray) -> np.ndarray:
    """189-dim observation without smb.obs (in-air from Y speed)."""
    inputs = np.zeros(189, dtype=np.float32)
    x_page = int(ram[0x006D])
    x_offset = int(ram[0x0086])
    mario_y = int(ram[0x03B8]) + 16
    mario_abs_x = x_page * 256 + x_offset
    gi = 0
    for dy in range(-6, 7):
        for dx in range(-6, 7):
            tile_x = x_offset + dx
            tile_page = x_page
            if tile_x < 0:
                tile_x += 256
                tile_page -= 1
            elif tile_x >= 256:
                tile_x -= 256
                tile_page += 1
            tile_y = mario_y // 16 + dy
            if 0 <= tile_y < 13 and tile_page >= 0:
                addr = 0x0500 + (tile_page % 2) * 13 * 16 + tile_y * 16 + (tile_x // 16)
                if addr < len(ram):
                    inputs[gi] = 1.0 if int(ram[addr]) != 0 else 0.0
            gi += 1
    for slot in range(5):
        if int(ram[0x000F + slot]) == 0:
            continue
        ex = int(ram[0x006E + slot]) * 256 + int(ram[0x0087 + slot])
        ey = int(ram[0x00CF + slot]) + 24
        gx = (ex - mario_abs_x) // 16 + 6
        gy = (ey - mario_y) // 16 + 6
        if 0 <= gx < 13 and 0 <= gy < 13:
            inputs[gy * 13 + gx] = -1.0
    inputs[169] = mario_y / 240.0
    inputs[170] = x_offset / 256.0
    inputs[171] = min(int(ram[0x000E]) / 2.0, 1.0)
    ys = int(ram[0x009F])
    if ys >= 128:
        ys -= 256
    inputs[172] = 1.0 if ys != 0 else 0.0
    inputs[173] = 1.0
    for slot in range(5):
        base = 174 + slot * 3
        et = int(ram[0x000F + slot])
        if et == 0:
            continue
        ex = int(ram[0x006E + slot]) * 256 + int(ram[0x0087 + slot])
        ey = int(ram[0x00CF + slot])
        inputs[base] = (ex - mario_abs_x) / 256.0
        inputs[base + 1] = (ey - mario_y) / 240.0
        inputs[base + 2] = et / 64.0
    return inputs


def read_smb_inputs(
    ram: np.ndarray,
    prev_action: Sequence[int] | None = None,
) -> np.ndarray:
    """Default rich SMB observation (210 dims when ``smb.obs`` is available)."""
    if _obs_read is not None:
        try:
            return _obs_read(ram, prev_action=prev_action)
        except TypeError:
            return _obs_read(ram)
    return _legacy_inline(ram)


if read_smb_inputs_legacy is None:  # type: ignore[truthy-function]

    def read_smb_inputs_legacy(ram: np.ndarray) -> np.ndarray:  # type: ignore[no-redef]
        return _legacy_inline(ram)


# -- Activations / small CNN ---------------------------------------------------

ArchName = Literal["mlp", "cnn_mlp"]


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


# -- Neural Network ------------------------------------------------------------


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


# -- Output-to-buttons mapping ------------------------------------------------

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
    # Prefer single discrete combo (cleaner than multi-sigmoid OR)
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
        # prefer exact match, else max Jaccard
        if cset == bset & cset and cset <= bset and len(cset) == len(bset):
            target[i] = 1.0
            return target
        overlap = len(cset & bset) - 0.25 * len(cset - bset)
        if overlap > best_overlap:
            best_overlap = overlap
            best_i = i
    target[best_i] = 1.0
    return target


# -- Neuroevolution GA ---------------------------------------------------------


@dataclass
class NeuroIndividual:
    """A candidate: neural network + fitness."""

    net: NeuralNet
    fitness: float = float("-inf")
    result: Optional[EvalResult] = None


def mutate_net(net: NeuralNet, rate: float = 0.05, magnitude: float = 0.3) -> NeuralNet:
    """Mutate network weights with Gaussian noise."""
    child = net.copy()
    mask = np.random.random(len(child.weights)) < rate
    noise = np.random.randn(len(child.weights)).astype(np.float32) * magnitude
    child.weights += mask * noise
    reset_mask = np.random.random(len(child.weights)) < (rate * 0.05)
    n_reset = int(reset_mask.sum())
    if n_reset:
        child.weights[reset_mask] = np.random.randn(n_reset).astype(np.float32) * 0.5
    return child


def crossover_net(p1: NeuralNet, p2: NeuralNet) -> NeuralNet:
    """Uniform crossover on network weights."""
    child = p1.copy()
    mask = np.random.random(len(child.weights)) < 0.5
    child.weights[mask] = p2.weights[mask]
    return child


def evaluate_network(
    net: NeuralNet,
    evaluator: Evaluator,
    max_frames: int = 6000,
    read_inputs_fn: Callable = read_smb_inputs,
    render_surface=None,
    render_scale: int = 3,
    gen_info: str = "",
) -> tuple[list[list[int]], EvalResult]:
    """Play the game with a neural network, return (buttons, result)."""
    evaluator._ensure_env()
    env = evaluator._env
    assert env is not None and evaluator._cached_state is not None
    env.em.set_state(evaluator._cached_state)
    net.reset_state()

    config = evaluator.config
    all_buttons: list[list[int]] = []
    action_size = env.action_space.shape[0]

    initial_values = evaluator._read_ram(env.get_ram())
    initial_lives = initial_values.get("lives", 0)
    prev_level_id = initial_values.get("level_id", 0)
    max_progress = 0.0
    frames_since_progress = 0
    completed = False
    died = False
    total_frames = 0
    prev_action: list[int] | None = None

    pygame_mod = None
    if render_surface is not None:
        import pygame

        pygame_mod = pygame

    for frame_idx in range(max_frames):
        ram = env.get_ram()
        try:
            inputs = read_inputs_fn(ram, prev_action=prev_action)
        except TypeError:
            inputs = read_inputs_fn(ram)

        outputs = net.forward(inputs)
        buttons = outputs_to_buttons(outputs)
        prev_action = buttons

        action = np.array(buttons[:action_size], dtype=np.int8)
        obs, reward, terminated, truncated, info = env.step(action)
        all_buttons.append(buttons)

        if render_surface is not None and pygame_mod is not None:
            frame_rgb = env.render()
            if frame_rgb is not None:
                surf = pygame_mod.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
                scaled = pygame_mod.transform.scale(
                    surf,
                    (surf.get_width() * render_scale, surf.get_height() * render_scale),
                )
                render_surface.blit(scaled, (0, 0))
                font = pygame_mod.font.SysFont("monospace", 16)
                hud_lines = [f"F:{frame_idx:5d} prog:{max_progress:7.1f} {gen_info}"]
                for li, line in enumerate(hud_lines):
                    txt = font.render(line, True, (255, 255, 0), (0, 0, 0))
                    render_surface.blit(txt, (4, 4 + li * 18))
                pygame_mod.display.flip()
            for ev in pygame_mod.event.get():
                if ev.type == pygame_mod.QUIT:
                    return all_buttons, EvalResult(
                        completed=False,
                        died=False,
                        total_frames=frame_idx + 1,
                        max_progress=max_progress,
                    )
                if ev.type == pygame_mod.KEYDOWN and ev.key == pygame_mod.K_ESCAPE:
                    return all_buttons, EvalResult(
                        completed=False,
                        died=False,
                        total_frames=frame_idx + 1,
                        max_progress=max_progress,
                    )

        ram = env.get_ram()
        values = evaluator._read_ram(ram)

        progress = float(values.get("player_x", 0))
        if progress > max_progress:
            max_progress = progress
            frames_since_progress = 0
        else:
            frames_since_progress += 1

        level_id = values.get("level_id", prev_level_id)
        main_level_ids = {config.target_level_id, *config.level_id_aliases}
        if config.completion_signal == "ram_flag":
            ram_key = getattr(config, "completion_ram_key", None)
            ram_val = getattr(config, "completion_ram_value", None)
            if (
                ram_key
                and values.get(ram_key) == ram_val
                and max_progress >= config.completion_min_progress
            ):
                completed = True
                total_frames = frame_idx + 1
                break
        else:
            if (
                level_id not in main_level_ids
                and level_id != 0
                and max_progress >= config.completion_min_progress
            ):
                allowed = config.completion_level_ids
                excluded = config.completion_exclude_ids
                if (not allowed or level_id in allowed) and level_id not in excluded:
                    completed = True
                    total_frames = frame_idx + 1
                    break

        lives = values.get("lives", initial_lives)
        if lives < initial_lives:
            died = True
            total_frames = frame_idx + 1
            break

        if frames_since_progress > config.max_stall_frames:
            total_frames = frame_idx + 1
            break

        prev_level_id = level_id

    if total_frames == 0:
        total_frames = max_frames

    result = EvalResult(
        completed=completed,
        died=died,
        total_frames=total_frames,
        max_progress=max_progress,
    )
    result.fitness = max_progress * config.progress_weight - total_frames * 0.01
    if completed:
        result.fitness = config.completion_bonus - total_frames

    return all_buttons, result


def collect_bc_dataset(
    evaluator: Evaluator,
    seed_frames: Sequence[Sequence[int]],
    *,
    read_inputs_fn: Callable = read_smb_inputs,
    max_frames: int | None = None,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Replay a seed while recording (observation, action-target) pairs.

    Returns ``(X, Y)`` with shapes ``(N, n_inputs)`` and ``(N, n_outputs)``.
    """
    evaluator._ensure_env()
    env = evaluator._env
    assert env is not None and evaluator._cached_state is not None
    env.em.set_state(evaluator._cached_state)
    action_size = int(env.action_space.shape[0])

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    prev_action: list[int] | None = None
    limit = len(seed_frames) if max_frames is None else min(len(seed_frames), max_frames)

    for i in range(limit):
        ram = env.get_ram()
        try:
            obs = read_inputs_fn(ram, prev_action=prev_action)
        except TypeError:
            obs = read_inputs_fn(ram)
        buttons = list(seed_frames[i][:action_size])
        if len(buttons) < action_size:
            buttons.extend([0] * (action_size - len(buttons)))
        if i % stride == 0:
            xs.append(np.asarray(obs, dtype=np.float32))
            ys.append(buttons_to_output_target(buttons))
        env.step(np.array(buttons, dtype=np.int8))
        prev_action = buttons

    if not xs:
        return np.zeros((0, N_SMB_INPUTS), dtype=np.float32), np.zeros(
            (0, len(SMB_OUTPUT_BUTTONS)), dtype=np.float32
        )
    return np.stack(xs), np.stack(ys)


def behavior_clone_init(
    net: NeuralNet,
    X: np.ndarray,
    Y: np.ndarray,
    *,
    steps: int = 200,
    lr: float = 0.05,
    batch_size: int = 64,
) -> NeuralNet:
    """Lightweight supervised warm-start via finite-difference SGD on weights.

    Not a full autograd trainer — enough to bias a random net toward the seed's
    action distribution before neuroevolution. O(steps * batch * weights).
    """
    if len(X) == 0:
        return net
    child = net.copy()
    n = len(child.weights)
    eps = 1e-2
    rng = np.random.default_rng(0)

    def batch_loss(w: np.ndarray, idx: np.ndarray) -> float:
        child.weights = w
        child.reset_state()
        loss = 0.0
        for j in idx:
            pred = child.forward(X[j])
            # binary cross-entropy
            p = np.clip(pred, 1e-6, 1 - 1e-6)
            y = Y[j]
            loss += float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        return loss / max(1, len(idx))

    w = child.weights.copy()
    for _ in range(steps):
        idx = rng.choice(len(X), size=min(batch_size, len(X)), replace=False)
        # coordinate-wise random subset gradient estimate
        coords = rng.choice(n, size=min(64, n), replace=False)
        grad = np.zeros(n, dtype=np.float32)
        base = batch_loss(w, idx)
        for c in coords:
            w2 = w.copy()
            w2[c] += eps
            grad[c] = (batch_loss(w2, idx) - base) / eps
        w = w - lr * grad
    child.weights = w.astype(np.float32)
    child.reset_state()
    return child


def warm_start_from_seed(
    evaluator: Evaluator,
    seed_frames: Sequence[Sequence[int]],
    *,
    n_hidden: int = 32,
    arch: ArchName = "mlp",
    hidden_layers: tuple[int, ...] = (),
    use_recurrent: bool = False,
    read_inputs_fn: Callable = read_smb_inputs,
    bc_steps: int = 150,
    max_frames: int | None = 4000,
    stride: int = 2,
) -> NeuralNet:
    """Build a net and behavior-clone it from an RLE/raw seed replay."""
    # Probe input size
    evaluator._ensure_env()
    env = evaluator._env
    assert env is not None and evaluator._cached_state is not None
    env.em.set_state(evaluator._cached_state)
    ram = env.get_ram()
    try:
        probe = read_inputs_fn(ram, prev_action=None)
    except TypeError:
        probe = read_inputs_fn(ram)
    n_in = int(np.asarray(probe).shape[0])
    n_out = len(SMB_OUTPUT_BUTTONS)
    hl = hidden_layers or (n_hidden, max(16, n_hidden // 2))
    net = NeuralNet(
        n_inputs=n_in,
        n_hidden=hl[0],
        n_outputs=n_out,
        arch=arch,
        hidden_layers=hl,
        use_recurrent=use_recurrent,
    )
    X, Y = collect_bc_dataset(
        evaluator,
        seed_frames,
        read_inputs_fn=read_inputs_fn,
        max_frames=max_frames,
        stride=stride,
    )
    return behavior_clone_init(net, X, Y, steps=bc_steps)


def run_neuro_ga(
    evaluator: Evaluator,
    population_size: int = 100,
    num_generations: int = 300,
    n_hidden: int = 20,
    elite_count: int = 10,
    output_dir: Path | None = None,
    verbose: bool = True,
    max_frames: int = 6000,
    read_inputs_fn: Callable = read_smb_inputs,
    render: bool = False,
    render_scale: int = 3,
    arch: ArchName = "mlp",
    hidden_layers: tuple[int, ...] | None = None,
    use_recurrent: bool = False,
    seed_frames: Sequence[Sequence[int]] | None = None,
    bc_steps: int = 100,
) -> NeuroIndividual:
    """Evolve neural networks to play the level.

    Pass ``seed_frames`` to warm-start the population around a behavior-cloned
    seed (recommended: current best RLE continuous policy).
    """
    config = evaluator.config
    if output_dir is None:
        output_dir = config.runs_dir / "neuro"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Probe observation size from the chosen reader
    evaluator._ensure_env()
    env = evaluator._env
    assert env is not None and evaluator._cached_state is not None
    env.em.set_state(evaluator._cached_state)
    try:
        n_inputs = int(np.asarray(read_inputs_fn(env.get_ram())).shape[0])
    except Exception:
        n_inputs = N_SMB_INPUTS
    n_outputs = len(SMB_OUTPUT_BUTTONS)
    hl = tuple(hidden_layers) if hidden_layers else (n_hidden,)

    checkpoint_path = output_dir / "neuro_best.json"
    resumed_net = None
    start_gen = 0
    if checkpoint_path.exists():
        try:
            ckpt = json.loads(checkpoint_path.read_text())
            if ckpt.get("n_inputs") == n_inputs and ckpt.get("n_outputs") == n_outputs:
                resumed_net = NeuralNet(
                    n_inputs=ckpt["n_inputs"],
                    n_hidden=ckpt["n_hidden"],
                    n_outputs=ckpt["n_outputs"],
                    weights=np.array(ckpt["weights"], dtype=np.float32),
                    arch=ckpt.get("arch", "mlp"),
                    hidden_layers=tuple(ckpt.get("hidden_layers", (ckpt["n_hidden"],))),
                    n_conv=int(ckpt.get("n_conv", 8)),
                    use_recurrent=bool(ckpt.get("use_recurrent", False)),
                )
                start_gen = ckpt.get("generation", 0)
                if verbose:
                    print(
                        f"[NEURO] Resumed from checkpoint gen={start_gen} "
                        f"fitness={ckpt.get('fitness', '?')} "
                        f"progress={ckpt.get('max_progress', '?')}"
                    )
        except Exception as e:
            if verbose:
                print(f"[NEURO] Could not load checkpoint: {e}, starting fresh")

    population: list[NeuroIndividual] = []
    if resumed_net is not None:
        population.append(NeuroIndividual(net=resumed_net.copy()))
        for _ in range(population_size - 1):
            population.append(
                NeuroIndividual(net=mutate_net(resumed_net, rate=0.10, magnitude=0.5))
            )
    elif seed_frames is not None:
        if verbose:
            print(f"[NEURO] BC warm-start from seed ({len(seed_frames)} frames)...")
        base = warm_start_from_seed(
            evaluator,
            seed_frames,
            n_hidden=n_hidden,
            arch=arch,
            hidden_layers=hl,
            use_recurrent=use_recurrent,
            read_inputs_fn=read_inputs_fn,
            bc_steps=bc_steps,
            max_frames=min(max_frames, len(seed_frames)),
        )
        population.append(NeuroIndividual(net=base.copy()))
        for _ in range(population_size - 1):
            population.append(
                NeuroIndividual(net=mutate_net(base, rate=0.08, magnitude=0.4))
            )
    else:
        for _ in range(population_size):
            net = NeuralNet(
                n_inputs=n_inputs,
                n_hidden=n_hidden,
                n_outputs=n_outputs,
                arch=arch,
                hidden_layers=hl,
                use_recurrent=use_recurrent,
            )
            population.append(NeuroIndividual(net=net))

    if verbose:
        total_weights = population[0].net.total_weights
        print(
            f"[NEURO] Population: {population_size}, arch={arch}, "
            f"hidden={hl}, weights: {total_weights}, inputs: {n_inputs}"
        )
        btn_names = {0: "B", 4: "UP", 5: "DN", 6: "L", 7: "R", 8: "A"}
        labels = [
            "+".join(btn_names.get(b, str(b)) for b in btns) for btns in SMB_OUTPUT_BUTTONS
        ]
        print(f"[NEURO] Outputs: {n_outputs} ({labels})")
        print(f"[NEURO] Max frames per eval: {max_frames}")

    render_surface = None
    if render:
        import os

        os.environ.setdefault("SDL_VIDEO_WAYLAND_WMCLASS", "neuroevo")
        os.environ.setdefault("SDL_VIDEO_X11_WMCLASS", "neuroevo")
        import pygame

        pygame.init()
        pygame.font.init()
        evaluator._ensure_env()
        env = evaluator._env
        assert env is not None
        env.em.set_state(evaluator._cached_state)
        frame = env.render()
        if frame is not None:
            w, h = frame.shape[1] * render_scale, frame.shape[0] * render_scale
        else:
            w, h = 256 * render_scale, 240 * render_scale
        render_surface = pygame.display.set_mode((w, h), pygame.SCALED | pygame.RESIZABLE)
        pygame.display.set_caption("Neuroevolution - Training Live")

    if verbose:
        print("[NEURO] Evaluating initial population...")

    best_init_fitness = float("-inf")
    for i, ind in enumerate(population):
        gen_info = f"init {i + 1}/{population_size}"
        _, result = evaluate_network(
            ind.net,
            evaluator,
            max_frames,
            read_inputs_fn,
            render_surface=render_surface,
            render_scale=render_scale,
            gen_info=gen_info,
        )
        ind.result = result
        ind.fitness = result.fitness
        if ind.fitness > best_init_fitness:
            best_init_fitness = ind.fitness
            if verbose:
                print(
                    f"  [{i + 1}/{population_size}] new best: fitness={ind.fitness:.1f} "
                    f"progress={result.max_progress:.1f} frames={result.total_frames}"
                )

    best_ever = max(population, key=lambda x: x.fitness)
    best_ever = NeuroIndividual(
        net=best_ever.net.copy(),
        fitness=best_ever.fitness,
        result=best_ever.result,
    )
    stall_gens = 0
    start_time = time.time()
    user_quit = False

    for gen in range(start_gen, start_gen + num_generations):
        if user_quit:
            break

        population.sort(key=lambda x: x.fitness, reverse=True)

        gen_best = population[0]
        if gen_best.fitness > best_ever.fitness:
            best_ever = NeuroIndividual(
                net=gen_best.net.copy(),
                fitness=gen_best.fitness,
                result=gen_best.result,
            )
            stall_gens = 0
        else:
            stall_gens += 1

        # Adaptive mutation scale when stalled
        mut_rate = 0.05 if stall_gens < 20 else min(0.15, 0.05 + stall_gens * 0.002)
        mut_mag = 0.3 if stall_gens < 20 else min(0.8, 0.3 + stall_gens * 0.01)

        if verbose and (gen % 5 == 0 or gen == start_gen + num_generations - 1):
            elapsed = time.time() - start_time
            status = (
                "COMPLETE"
                if best_ever.result and best_ever.result.completed
                else "incomplete"
            )
            frames = best_ever.result.total_frames if best_ever.result else 0
            progress = best_ever.result.max_progress if best_ever.result else 0
            print(
                f"[NEURO] gen={gen:4d} best_fitness={best_ever.fitness:10.1f} "
                f"frames={frames:5d} progress={progress:7.1f} "
                f"status={status} stall={stall_gens} elapsed={elapsed:.1f}s"
            )

        if gen % 10 == 0 and gen > 0:
            _save_neuro_checkpoint(
                best_ever, gen, output_dir, evaluator, max_frames, read_inputs_fn
            )

        next_gen: list[NeuroIndividual] = []
        for i in range(min(elite_count, len(population))):
            next_gen.append(NeuroIndividual(net=population[i].net.copy()))

        while len(next_gen) < population_size:
            candidates = random.sample(population, min(5, len(population)))
            p1 = max(candidates, key=lambda x: x.fitness)
            candidates = random.sample(population, min(5, len(population)))
            p2 = max(candidates, key=lambda x: x.fitness)

            if random.random() < 0.7:
                child_net = crossover_net(p1.net, p2.net)
            else:
                child_net = p1.net.copy()

            child_net = mutate_net(child_net, rate=mut_rate, magnitude=mut_mag)
            next_gen.append(NeuroIndividual(net=child_net))

        for ci, ind in enumerate(next_gen):
            if user_quit:
                break
            if ind.fitness == float("-inf"):
                gen_info_child = (
                    f"gen={gen} eval {ci + 1}/{len(next_gen)} best={best_ever.fitness:.0f}"
                )
                _, result = evaluate_network(
                    ind.net,
                    evaluator,
                    max_frames,
                    read_inputs_fn,
                    render_surface=render_surface,
                    render_scale=render_scale,
                    gen_info=gen_info_child,
                )
                ind.result = result
                ind.fitness = result.fitness

        population = next_gen

    _save_neuro_checkpoint(
        best_ever,
        start_gen + num_generations,
        output_dir,
        evaluator,
        max_frames,
        read_inputs_fn,
    )

    if render_surface is not None:
        import pygame

        pygame.quit()

    if verbose:
        elapsed = time.time() - start_time
        print(
            f"\n[NEURO] Done! gens {start_gen}-{start_gen + num_generations} "
            f"in {elapsed:.1f}s"
        )
        print(f"[NEURO] Best fitness: {best_ever.fitness:.1f}")
        if best_ever.result:
            print(f"[NEURO] Completed: {best_ever.result.completed}")
            print(f"[NEURO] Frames: {best_ever.result.total_frames}")
            print(f"[NEURO] Progress: {best_ever.result.max_progress:.1f}")

    return best_ever


def _save_neuro_checkpoint(
    best: NeuroIndividual,
    gen: int,
    output_dir: Path,
    evaluator: Evaluator,
    max_frames: int,
    read_inputs_fn,
) -> None:
    """Save best network weights + replay its buttons."""
    net_path = output_dir / "neuro_best.json"
    data = {
        "weights": best.net.weights.tolist(),
        "n_inputs": best.net.n_inputs,
        "n_hidden": best.net.n_hidden,
        "n_outputs": best.net.n_outputs,
        "arch": best.net.arch,
        "hidden_layers": list(best.net.hidden_layers),
        "n_conv": best.net.n_conv,
        "use_recurrent": best.net.use_recurrent,
        "generation": gen,
        "fitness": best.fitness,
        "completed": best.result.completed if best.result else False,
        "total_frames": best.result.total_frames if best.result else 0,
        "max_progress": best.result.max_progress if best.result else 0,
    }
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
