"""Neuroevolution for platformer speedruns.

Inspired by MarI/O (SethBling's NEAT for Super Mario). Instead of evolving
raw button sequences (which break on crossover), evolve neural networks
that read game state and output button presses.

Key advantages over raw-button GA:
- Crossover on network weights is smooth (no frame-alignment issues)
- Networks generalize across different enemy timings/positions
- Much smaller search space (~500 weights vs 3000+ frames × 12 buttons)

Architecture:
- Input: tile grid around player (13×13=169) + player state (4) + bias (1) = 174
- Hidden: single layer, configurable size (default 20)
- Output: button decisions (RIGHT, RIGHT+B, RIGHT+B+A, RIGHT+A, A, LEFT, DOWN, NOTHING)
- Activation: tanh hidden, sigmoid output (>0.5 = pressed)

Tiles are read from NES/SNES RAM nametables. Enemies appear as -1 in the grid.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from platformer_common.evaluator import Evaluator, EvalResult
from platformer_common.level_config import LevelConfig


# -- SMB RAM tile/sprite reading -----------------------------------------------

N_SMB_INPUTS = 189  # 169 grid + 5 player state + 15 enemy slots


def read_smb_inputs(ram: np.ndarray) -> np.ndarray:
    """Read 13×13 tile grid + player state + enemy positions from NES SMB RAM.

    Returns float array of 189 values:
      [0:169]  = 13×13 tile grid centered on Mario (1=solid, -1=enemy, 0=empty)
      [169]    = player_y normalized (0-1)
      [170]    = speed proxy (x_offset / 256)
      [171]    = player_status (0=small, 1=big, 2=fire)
      [172]    = in_air (1 if jumping/falling)
      [173]    = bias (always 1.0)
      [174:189] = 5 enemy slots × 3 (relative_x, relative_y, type) normalized
    """
    inputs = np.zeros(N_SMB_INPUTS, dtype=np.float32)

    # Mario position
    x_page = int(ram[0x006D])
    x_offset = int(ram[0x0086])
    mario_y = int(ram[0x03B8]) + 16  # sprite Y + offset

    # Read 13×13 tile grid centered on Mario
    # SMB tile data is at 0x0500, organized by columns
    grid_idx = 0
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

            if tile_y < 0 or tile_y >= 13 or tile_page < 0:
                inputs[grid_idx] = 0.0
            else:
                # SMB nametable tile address
                addr = 0x0500 + (tile_page % 2) * 13 * 16 + tile_y * 16 + (tile_x // 16)
                if addr < len(ram):
                    tile_val = int(ram[addr])
                    inputs[grid_idx] = 1.0 if tile_val != 0 else 0.0
                else:
                    inputs[grid_idx] = 0.0
            grid_idx += 1

    # Check enemy sprites (5 slots)
    for slot in range(5):
        enemy_type = int(ram[0x000F + slot]) if (0x000F + slot) < len(ram) else 0
        if enemy_type == 0:
            continue
        enemy_x_page = int(ram[0x006E + slot]) if (0x006E + slot) < len(ram) else 0
        enemy_x_off = int(ram[0x0087 + slot]) if (0x0087 + slot) < len(ram) else 0
        enemy_y = int(ram[0x00CF + slot]) + 24 if (0x00CF + slot) < len(ram) else 0

        # Convert to grid coords relative to Mario
        ex = (enemy_x_page * 256 + enemy_x_off) - (x_page * 256 + x_offset)
        ey = (enemy_y - mario_y) // 16

        gx = ex // 16 + 6  # center of 13×13 grid
        gy = ey + 6

        if 0 <= gx < 13 and 0 <= gy < 13:
            inputs[gy * 13 + gx] = -1.0  # enemy marker

    # Player state inputs
    inputs[169] = mario_y / 240.0  # normalized Y
    inputs[170] = x_offset / 256.0  # position within page
    player_status = int(ram[0x000E]) if 0x000E < len(ram) else 0
    inputs[171] = min(player_status / 2.0, 1.0)
    # In-air detection: check if Mario's Y velocity is nonzero
    inputs[172] = 0.0  # simplified - could read velocity RAM
    inputs[173] = 1.0  # bias

    # Direct enemy positions (5 slots × 3: relative_x, relative_y, type)
    # Gives the network visibility beyond the 13×13 grid (hammers, fireballs)
    mario_abs_x = x_page * 256 + x_offset
    for slot in range(5):
        base = 174 + slot * 3
        enemy_type = int(ram[0x000F + slot]) if (0x000F + slot) < len(ram) else 0
        if enemy_type == 0:
            continue
        enemy_x = int(ram[0x006E + slot]) * 256 + int(ram[0x0087 + slot])
        enemy_y = int(ram[0x00CF + slot]) if (0x00CF + slot) < len(ram) else 0
        inputs[base] = (enemy_x - mario_abs_x) / 256.0  # relative X (screen-widths)
        inputs[base + 1] = (enemy_y - mario_y) / 240.0   # relative Y
        inputs[base + 2] = enemy_type / 64.0              # type normalized

    return inputs


# -- Neural Network ------------------------------------------------------------

@dataclass
class NeuralNet:
    """Simple feedforward neural network with one hidden layer.

    Weights stored as flat arrays for easy GA manipulation.
    """
    n_inputs: int
    n_hidden: int
    n_outputs: int
    # Weights: input->hidden (n_inputs * n_hidden) + hidden->output (n_hidden * n_outputs)
    # Biases: hidden (n_hidden) + output (n_outputs)
    weights: np.ndarray = field(default_factory=lambda: np.array([]))

    def __post_init__(self):
        if len(self.weights) == 0:
            total = self.n_inputs * self.n_hidden + self.n_hidden + self.n_hidden * self.n_outputs + self.n_outputs
            self.weights = np.random.randn(total).astype(np.float32) * 0.5

    @property
    def total_weights(self) -> int:
        return self.n_inputs * self.n_hidden + self.n_hidden + self.n_hidden * self.n_outputs + self.n_outputs

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Forward pass. Returns output activations (sigmoid, 0-1)."""
        ni, nh, no = self.n_inputs, self.n_hidden, self.n_outputs
        w = self.weights

        # Unpack weights
        idx = 0
        w_ih = w[idx:idx + ni * nh].reshape(ni, nh)
        idx += ni * nh
        b_h = w[idx:idx + nh]
        idx += nh
        w_ho = w[idx:idx + nh * no].reshape(nh, no)
        idx += nh * no
        b_o = w[idx:idx + no]

        # Hidden layer (tanh)
        hidden = np.tanh(inputs @ w_ih + b_h)
        # Output layer (sigmoid)
        output = 1.0 / (1.0 + np.exp(-(hidden @ w_ho + b_o)))
        return output

    def copy(self) -> NeuralNet:
        return NeuralNet(
            n_inputs=self.n_inputs,
            n_hidden=self.n_hidden,
            n_outputs=self.n_outputs,
            weights=self.weights.copy(),
        )


# -- Output-to-buttons mapping ------------------------------------------------

# 8 outputs → SMB button combos (matching SMB_ACTIONS indices)
# [RIGHT, B(run), A(jump), LEFT, DOWN, UP, RIGHT+B+A, nothing_bias]
# We use a simpler scheme: 6 outputs that map to common button combos

SMB_OUTPUT_BUTTONS = [
    # Each entry: list of NES button indices to press
    # NES: [B=0, null=1, SELECT=2, START=3, UP=4, DOWN=5, LEFT=6, RIGHT=7, A=8]
    [7],        # 0: RIGHT
    [7, 0],     # 1: RIGHT + B (run)
    [7, 0, 8],  # 2: RIGHT + B + A (run + jump)
    [7, 8],     # 3: RIGHT + A (walk + jump)
    [8],        # 4: JUMP (standing)
    [6],        # 5: LEFT
    [0],        # 6: B (run in place / fireball)
    [6, 0],     # 7: LEFT + B (run left)
    [6, 0, 8],  # 8: LEFT + B + A (run left + jump)
]

def outputs_to_buttons(outputs: np.ndarray) -> list[int]:
    """Convert network outputs to 12-element button array.

    Each output > 0.5 means that action's buttons are pressed.
    Multiple outputs can fire simultaneously.
    """
    buttons = [0] * 12
    for i, val in enumerate(outputs):
        if val > 0.5 and i < len(SMB_OUTPUT_BUTTONS):
            for btn_idx in SMB_OUTPUT_BUTTONS[i]:
                if btn_idx < 12:
                    buttons[btn_idx] = 1
    return buttons


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

    # Occasionally reset a weight to random (5% of mutations)
    reset_mask = np.random.random(len(child.weights)) < (rate * 0.05)
    child.weights[reset_mask] = np.random.randn(int(reset_mask.sum())).astype(np.float32) * 0.5

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
    read_inputs_fn=read_smb_inputs,
    render_surface=None,
    render_scale: int = 3,
    gen_info: str = "",
) -> tuple[list[list[int]], EvalResult]:
    """Play the game with a neural network, return (buttons, result).

    Runs the network frame-by-frame, reading RAM state as input and
    outputting button presses. Collects the full button sequence.

    If render_surface is provided (a pygame Surface), renders each frame
    to it for live viewing.
    """
    evaluator._ensure_env()
    env = evaluator._env
    assert env is not None and evaluator._cached_state is not None
    env.em.set_state(evaluator._cached_state)

    config = evaluator.config
    ram_schema = config.ram.to_schema()

    all_buttons: list[list[int]] = []
    action_size = env.action_space.shape[0]

    # Initialize tracking (mirrors evaluator logic)
    initial_values = evaluator._read_ram(env.get_ram())
    initial_lives = initial_values.get("lives", 0)
    prev_level_id = initial_values.get("level_id", 0)
    max_progress = 0.0
    frames_since_progress = 0
    completed = False
    died = False
    total_frames = 0

    pygame_mod = None
    if render_surface is not None:
        import pygame
        pygame_mod = pygame

    for frame_idx in range(max_frames):
        # Read inputs from RAM
        ram = env.get_ram()
        inputs = read_inputs_fn(ram)

        # Forward pass
        outputs = net.forward(inputs)
        buttons = outputs_to_buttons(outputs)

        # Step environment
        action = np.array(buttons[:action_size], dtype=np.int8)
        obs, reward, terminated, truncated, info = env.step(action)
        all_buttons.append(buttons)

        # Render if surface provided
        if render_surface is not None and pygame_mod is not None:
            # Draw game frame
            frame_rgb = env.render()
            if frame_rgb is not None:
                surf = pygame_mod.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
                scaled = pygame_mod.transform.scale(
                    surf,
                    (surf.get_width() * render_scale, surf.get_height() * render_scale),
                )
                render_surface.blit(scaled, (0, 0))

                # HUD overlay
                font = pygame_mod.font.SysFont("monospace", 16)
                hud_lines = [
                    f"F:{frame_idx:5d} prog:{max_progress:7.1f} {gen_info}",
                ]
                for li, line in enumerate(hud_lines):
                    txt = font.render(line, True, (255, 255, 0), (0, 0, 0))
                    render_surface.blit(txt, (4, 4 + li * 18))

                pygame_mod.display.flip()

            # Pump events (allow window close / ESC to abort)
            for ev in pygame_mod.event.get():
                if ev.type == pygame_mod.QUIT:
                    return all_buttons, EvalResult(
                        completed=False, died=False,
                        total_frames=frame_idx + 1, max_progress=max_progress,
                    )
                if ev.type == pygame_mod.KEYDOWN and ev.key == pygame_mod.K_ESCAPE:
                    return all_buttons, EvalResult(
                        completed=False, died=False,
                        total_frames=frame_idx + 1, max_progress=max_progress,
                    )

        # Read state for tracking
        ram = env.get_ram()
        values = evaluator._read_ram(ram)

        # Progress
        progress = float(values.get("player_x", 0))
        if progress > max_progress:
            max_progress = progress
            frames_since_progress = 0
        else:
            frames_since_progress += 1

        # Completion check — match Evaluator (aliases + completion_level_ids).
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
            # level_id_change: leave main level set for a real completion id.
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

        # Death check
        lives = values.get("lives", initial_lives)
        if lives < initial_lives:
            died = True
            total_frames = frame_idx + 1
            break

        # Stall check
        if frames_since_progress > config.max_stall_frames:
            total_frames = frame_idx + 1
            break

        prev_level_id = level_id

    if total_frames == 0:
        total_frames = max_frames

    # Build result
    result = EvalResult(
        completed=completed,
        died=died,
        total_frames=total_frames,
        max_progress=max_progress,
    )
    # Fitness: progress-based (same as GA smooth fitness)
    result.fitness = max_progress * config.progress_weight - total_frames * 0.01
    if completed:
        result.fitness = config.completion_bonus - total_frames

    return all_buttons, result


def run_neuro_ga(
    evaluator: Evaluator,
    population_size: int = 100,
    num_generations: int = 300,
    n_hidden: int = 20,
    elite_count: int = 10,
    output_dir: Path | None = None,
    verbose: bool = True,
    max_frames: int = 6000,
    read_inputs_fn=read_smb_inputs,
    render: bool = False,
    render_scale: int = 3,
) -> NeuroIndividual:
    """Evolve neural networks to play the level.

    Args:
        evaluator: Headless evaluator (used for env + state management).
        population_size: Number of networks per generation.
        num_generations: Generations to run.
        n_hidden: Hidden layer size.
        elite_count: Top networks preserved each generation.
        output_dir: Where to save checkpoints.
        verbose: Print progress.
        max_frames: Max frames per evaluation.
        read_inputs_fn: Function to read RAM inputs (default: SMB).
        render: If True, render best network each generation live.
        render_scale: Pixel scale for render window.

    Returns:
        Best NeuroIndividual found.
    """
    config = evaluator.config
    if output_dir is None:
        output_dir = config.runs_dir / "neuro"
    output_dir.mkdir(parents=True, exist_ok=True)

    n_inputs = N_SMB_INPUTS  # 13×13 grid + 5 player state + 15 enemy slots
    n_outputs = len(SMB_OUTPUT_BUTTONS)

    # Try to resume from checkpoint
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
                )
                start_gen = ckpt.get("generation", 0)
                if verbose:
                    print(f"[NEURO] Resumed from checkpoint gen={start_gen} "
                          f"fitness={ckpt.get('fitness', '?')} "
                          f"progress={ckpt.get('max_progress', '?')}")
        except Exception as e:
            if verbose:
                print(f"[NEURO] Could not load checkpoint: {e}, starting fresh")

    # Initialize population (seed from checkpoint if available)
    population: list[NeuroIndividual] = []
    if resumed_net is not None:
        # Elite: exact copy of best
        population.append(NeuroIndividual(net=resumed_net.copy()))
        # Fill rest with mutations of the best (diverse exploration around it)
        for _ in range(population_size - 1):
            child = mutate_net(resumed_net, rate=0.10, magnitude=0.5)
            population.append(NeuroIndividual(net=child))
    else:
        for _ in range(population_size):
            net = NeuralNet(n_inputs=n_inputs, n_hidden=n_hidden, n_outputs=n_outputs)
            population.append(NeuroIndividual(net=net))

    if verbose:
        total_weights = population[0].net.total_weights
        print(f"[NEURO] Population: {population_size}, hidden: {n_hidden}, weights: {total_weights}")
        print(f"[NEURO] Inputs: {n_inputs} (13×13 grid + 4 state + bias)")
        btn_names = {0: 'B', 4: 'UP', 5: 'DN', 6: 'L', 7: 'R', 8: 'A'}
        labels = ['+'.join(btn_names.get(b, str(b)) for b in btns) for btns in SMB_OUTPUT_BUTTONS]
        print(f"[NEURO] Outputs: {n_outputs} ({labels})")
        print(f"[NEURO] Max frames per eval: {max_frames}")

    # Set up render window if requested
    render_surface = None
    if render:
        import os
        # Set app_id so Hyprland can match with windowrulev2 float,class:neuroevo
        os.environ.setdefault("SDL_VIDEO_WAYLAND_WMCLASS", "neuroevo")
        os.environ.setdefault("SDL_VIDEO_X11_WMCLASS", "neuroevo")
        import pygame
        pygame.init()
        pygame.font.init()
        # Get frame size from a quick env render
        evaluator._ensure_env()
        env = evaluator._env
        assert env is not None
        env.em.set_state(evaluator._cached_state)
        frame = env.render()
        if frame is not None:
            w, h = frame.shape[1] * render_scale, frame.shape[0] * render_scale
        else:
            w, h = 256 * render_scale, 240 * render_scale
        # SCALED keeps aspect ratio even if tiling WM resizes the window
        render_surface = pygame.display.set_mode((w, h), pygame.SCALED | pygame.RESIZABLE)
        pygame.display.set_caption("Neuroevolution - Training Live")

    # Evaluate initial population — render each one live so window isn't black
    if verbose:
        print(f"[NEURO] Evaluating initial population...")

    best_init_fitness = float("-inf")
    for i, ind in enumerate(population):
        # Render every network during initial eval so user sees something immediately
        gen_info = f"init {i+1}/{population_size}"
        _, result = evaluate_network(
            ind.net, evaluator, max_frames, read_inputs_fn,
            render_surface=render_surface,
            render_scale=render_scale,
            gen_info=gen_info,
        )
        ind.result = result
        ind.fitness = result.fitness
        if ind.fitness > best_init_fitness:
            best_init_fitness = ind.fitness
            if verbose:
                print(f"  [{i+1}/{population_size}] new best: fitness={ind.fitness:.1f} "
                      f"progress={result.max_progress:.1f} frames={result.total_frames}")

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
        improved = False
        if gen_best.fitness > best_ever.fitness:
            best_ever = NeuroIndividual(
                net=gen_best.net.copy(),
                fitness=gen_best.fitness,
                result=gen_best.result,
            )
            stall_gens = 0
            improved = True
        else:
            stall_gens += 1

        if verbose and (gen % 5 == 0 or gen == start_gen + num_generations - 1):
            elapsed = time.time() - start_time
            status = "COMPLETE" if best_ever.result and best_ever.result.completed else "incomplete"
            frames = best_ever.result.total_frames if best_ever.result else 0
            progress = best_ever.result.max_progress if best_ever.result else 0
            print(
                f"[NEURO] gen={gen:4d} best_fitness={best_ever.fitness:10.1f} "
                f"frames={frames:5d} progress={progress:7.1f} "
                f"status={status} stall={stall_gens} elapsed={elapsed:.1f}s"
            )

        # Checkpoint every 10 gens
        if gen % 10 == 0 and gen > 0:
            _save_neuro_checkpoint(best_ever, gen, output_dir, evaluator, max_frames, read_inputs_fn)

        # Build next generation
        next_gen: list[NeuroIndividual] = []

        # Elitism
        for i in range(min(elite_count, len(population))):
            next_gen.append(NeuroIndividual(net=population[i].net.copy()))

        # Fill with crossover + mutation
        while len(next_gen) < population_size:
            # Tournament selection (k=5)
            candidates = random.sample(population, min(5, len(population)))
            p1 = max(candidates, key=lambda x: x.fitness)
            candidates = random.sample(population, min(5, len(population)))
            p2 = max(candidates, key=lambda x: x.fitness)

            if random.random() < 0.7:
                child_net = crossover_net(p1.net, p2.net)
            else:
                child_net = p1.net.copy()

            child_net = mutate_net(child_net, rate=0.05, magnitude=0.3)
            next_gen.append(NeuroIndividual(net=child_net))

        # Evaluate new individuals (skip elites that already have fitness)
        for ci, ind in enumerate(next_gen):
            if user_quit:
                break
            if ind.fitness == float("-inf"):
                gen_info_child = f"gen={gen} eval {ci+1}/{len(next_gen)} best={best_ever.fitness:.0f}"
                _, result = evaluate_network(
                    ind.net, evaluator, max_frames, read_inputs_fn,
                    render_surface=render_surface,
                    render_scale=render_scale,
                    gen_info=gen_info_child,
                )
                ind.result = result
                ind.fitness = result.fitness

        population = next_gen

    # Final save
    _save_neuro_checkpoint(best_ever, start_gen + num_generations, output_dir, evaluator, max_frames, read_inputs_fn)

    if render_surface is not None:
        import pygame
        pygame.quit()

    if verbose:
        elapsed = time.time() - start_time
        print(f"\n[NEURO] Done! gens {start_gen}-{start_gen + num_generations} in {elapsed:.1f}s")
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
    # Save network
    net_path = output_dir / "neuro_best.json"
    data = {
        "weights": best.net.weights.tolist(),
        "n_inputs": best.net.n_inputs,
        "n_hidden": best.net.n_hidden,
        "n_outputs": best.net.n_outputs,
        "generation": gen,
        "fitness": best.fitness,
        "completed": best.result.completed if best.result else False,
        "total_frames": best.result.total_frames if best.result else 0,
        "max_progress": best.result.max_progress if best.result else 0,
    }
    net_path.write_text(json.dumps(data, indent=2))

    # Also save the button sequence for compatibility with watch/verify
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
