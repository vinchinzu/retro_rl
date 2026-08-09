"""Neuroevolution GA: evaluate, mutate, crossover, BC warm-start, run_neuro_ga."""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from retro_harness.platformer.evaluator import Evaluator, EvalResult
from retro_harness.platformer.neuro.checkpoint import (
    net_from_dict,
    save_neuro_checkpoint,
)
from retro_harness.platformer.neuro.net import (
    ArchName,
    NeuralNet,
    SMB_OUTPUT_BUTTONS,
    buttons_to_output_target,
    outputs_to_buttons,
)
from retro_harness.platformer.neuro.obs import (
    N_SMB_INPUTS,
    resolve_obs_fn,
)
from retro_harness.contracts import ContractMismatchError
from retro_harness.platformer.contracts import build_platformer_contracts


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
    read_inputs_fn: Callable | None = None,
    render_surface=None,
    render_scale: int = 3,
    gen_info: str = "",
    *,
    obs_fn: Callable | None = None,
) -> tuple[list[list[int]], EvalResult]:
    """Play the game with a neural network, return (buttons, result).

    Provide ``read_inputs_fn`` / ``obs_fn`` (callable ``ram -> ndarray``, optionally
    accepting ``prev_action=``). Defaults to SMB obs via ``smb.obs`` when available.
    """
    read_inputs_fn = resolve_obs_fn(obs_fn if obs_fn is not None else read_inputs_fn)
    evaluator._ensure_env()
    env = evaluator._env
    assert env is not None and evaluator._cached_state is not None
    evaluator.restore_start_state()
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
    read_inputs_fn: Callable | None = None,
    obs_fn: Callable | None = None,
    max_frames: int | None = None,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Replay a seed while recording (observation, action-target) pairs.

    Returns ``(X, Y)`` with shapes ``(N, n_inputs)`` and ``(N, n_outputs)``.
    """
    read_inputs_fn = resolve_obs_fn(obs_fn if obs_fn is not None else read_inputs_fn)
    evaluator._ensure_env()
    env = evaluator._env
    assert env is not None and evaluator._cached_state is not None
    evaluator.restore_start_state()
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
        n_in = N_SMB_INPUTS if N_SMB_INPUTS is not None else 0
        return np.zeros((0, n_in), dtype=np.float32), np.zeros(
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
            p = np.clip(pred, 1e-6, 1 - 1e-6)
            y = Y[j]
            loss += float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        return loss / max(1, len(idx))

    w = child.weights.copy()
    for _ in range(steps):
        idx = rng.choice(len(X), size=min(batch_size, len(X)), replace=False)
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
    read_inputs_fn: Callable | None = None,
    obs_fn: Callable | None = None,
    bc_steps: int = 150,
    max_frames: int | None = 4000,
    stride: int = 2,
) -> NeuralNet:
    """Build a net and behavior-clone it from an RLE/raw seed replay."""
    read_inputs_fn = resolve_obs_fn(obs_fn if obs_fn is not None else read_inputs_fn)
    evaluator._ensure_env()
    env = evaluator._env
    assert env is not None and evaluator._cached_state is not None
    evaluator.restore_start_state()
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
    read_inputs_fn: Callable | None = None,
    render: bool = False,
    render_scale: int = 3,
    arch: ArchName = "mlp",
    hidden_layers: tuple[int, ...] | None = None,
    use_recurrent: bool = False,
    seed_frames: Sequence[Sequence[int]] | None = None,
    bc_steps: int = 100,
    entry_corpus_path: Path | None = None,
    entry_corpus_root: Path | None = None,
    *,
    obs_fn: Callable | None = None,
) -> NeuroIndividual:
    """Evolve neural networks to play the level.

    Pass ``seed_frames`` to warm-start the population around a behavior-cloned
    seed (recommended: current best RLE continuous policy).

    Observation builder: pass ``obs_fn`` / ``read_inputs_fn``
    (``Callable[[ram], ndarray]``). When omitted, defaults to SMB via
    ``smb.obs`` and fails clearly if that package is unavailable.
    """
    read_inputs_fn = resolve_obs_fn(obs_fn if obs_fn is not None else read_inputs_fn)
    config = evaluator.config
    if output_dir is None:
        output_dir = config.runs_dir / "neuro"
    output_dir.mkdir(parents=True, exist_ok=True)

    if entry_corpus_path is not None:
        from retro_harness.entry_states import EntryStateCorpus
        from retro_harness.repo import monorepo_root

        corpus = EntryStateCorpus.load(entry_corpus_path)
        selected = corpus.split(train_fraction=0.8, salt="sm-landing-v1").train
        root = entry_corpus_root or monorepo_root()
        evaluator.configure_entry_states(
            [corpus.state_bytes(record, root=root) for record in selected],
            corpus_digest=corpus.identity_digest,
            split="train",
            state_digests=[record.state_digest for record in selected],
        )

    # Probe observation size from the chosen reader
    evaluator._ensure_env()
    env = evaluator._env
    assert env is not None and evaluator._cached_state is not None
    evaluator.restore_start_state()
    try:
        n_inputs = int(np.asarray(read_inputs_fn(env.get_ram())).shape[0])
    except Exception:
        if N_SMB_INPUTS is None:
            raise RuntimeError(
                "Could not probe observation size and no default N_SMB_INPUTS; "
                "pass a working obs_fn / read_inputs_fn."
            ) from None
        n_inputs = N_SMB_INPUTS
    n_outputs = len(SMB_OUTPUT_BUTTONS)
    hl = tuple(hidden_layers) if hidden_layers else (n_hidden,)
    active_contracts = build_platformer_contracts(
        config,
        n_inputs=n_inputs,
        read_inputs_fn=read_inputs_fn,
        output_buttons=SMB_OUTPUT_BUTTONS,
    )
    if entry_corpus_path is not None:
        if corpus.contract_bundle_digest != active_contracts.identity_digest:
            raise ContractMismatchError(
                "entry-state corpus contract does not match training environment"
            )

    checkpoint_path = output_dir / "neuro_best.json"
    resumed_net = None
    start_gen = 0
    if checkpoint_path.exists():
        try:
            ckpt = json.loads(checkpoint_path.read_text())
            if ckpt.get("n_inputs") != n_inputs or ckpt.get("n_outputs") != n_outputs:
                raise ContractMismatchError(
                    "neuro checkpoint tensor dimensions disagree with active contracts"
                )
            resumed_net = net_from_dict(
                ckpt,
                expected_contracts=active_contracts,
            )
            start_gen = ckpt.get("generation", 0)
            if verbose:
                print(
                    f"[NEURO] Resumed from checkpoint gen={start_gen} "
                    f"fitness={ckpt.get('fitness', '?')} "
                    f"progress={ckpt.get('max_progress', '?')}"
                )
        except ContractMismatchError:
            raise
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
        evaluator.restore_start_state()
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
            save_neuro_checkpoint(
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

    save_neuro_checkpoint(
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
