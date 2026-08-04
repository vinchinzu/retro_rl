"""Watch, verify, extract, list, prepare, auto-state, chain, and trace commands."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from retro_harness.platformer.actions import (
    DEFAULT_PLATFORMER_ACTIONS,
    action_index_to_buttons,
    buttons_to_action_index,
)
from retro_harness.platformer.bk2_extract import (
    extract_action_indices_from_bk2,
    extract_raw_actions_from_bk2,
    load_actions,
    save_actions,
)
from retro_harness.platformer.cli.helpers import _resolve_config, _get_action_table
from retro_harness.platformer.evaluator import Evaluator
from retro_harness.platformer.level_config import list_levels
from retro_harness.platformer.replay_hud import _replay_with_hud


def cmd_list_levels(args: argparse.Namespace) -> None:
    """List all registered levels."""
    from retro_harness.platformer.level_config import LEVEL_REGISTRY

    levels = list_levels()
    if not levels:
        print("No levels registered.")
        return

    print(f"{'ID':<30s} {'Display Name':<30s} {'Game':<30s} {'State'}")
    print("-" * 120)
    for cfg in levels:
        print(f"{cfg.level_id:<30s} {cfg.display_name:<30s} {cfg.game_name:<30s} {cfg.start_state}")

    # Show aliases
    print(f"\nAliases:")
    for alias, cfg in sorted(LEVEL_REGISTRY.items()):
        if alias != cfg.level_id:
            print(f"  {alias} -> {cfg.level_id}")


def cmd_extract(args: argparse.Namespace) -> None:
    """Extract action sequence from a bk2 recording."""
    config = _resolve_config(args)
    bk2_path = Path(args.bk2)
    if not bk2_path.exists():
        print(f"Error: bk2 file not found: {bk2_path}")
        return

    action_table = _get_action_table(config)
    print(f"Extracting from: {bk2_path}")

    raw = extract_raw_actions_from_bk2(bk2_path, bk2_to_env=config.bk2_to_env)
    print(f"Total raw frames: {len(raw)}")

    if args.raw_preview:
        print("\nFirst 10 raw frames (env button order: B Y Sel Sta U D L R A X L R):")
        for i, frame in enumerate(raw[:10]):
            print(f"  {i:4d}: {frame}")

    actions = extract_action_indices_from_bk2(
        bk2_path, action_table=action_table, bk2_to_env=config.bk2_to_env
    )
    print(f"Action indices: {len(actions)} frames")

    # Action distribution
    from collections import Counter

    dist = Counter(actions)
    num_actions = len(action_table)
    print(f"\nAction distribution ({num_actions} actions):")
    for idx in sorted(dist.keys()):
        print(f"  {idx:2d}: {dist[idx]:5d} frames ({dist[idx]/len(actions)*100:.1f}%)")

    output = Path(args.output) if args.output else config.runs_dir / f"{bk2_path.parent.name}_extracted.json"
    metadata = {"source_bk2": str(bk2_path), "raw_frames": len(raw), "level": config.level_id}
    save_actions(actions, output, metadata=metadata)




def _recording_start_state(path: Path) -> str | None:
    """Return an unambiguous start state embedded in a recording JSON."""
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    metadata = payload.get("metadata")
    candidates = [
        payload.get("state"),
        payload.get("start_state"),
        metadata.get("state") if isinstance(metadata, dict) else None,
    ]
    states = {state for state in candidates if isinstance(state, str) and state}
    return next(iter(states)) if len(states) == 1 else None


def cmd_verify(args: argparse.Namespace) -> None:
    """Verify an action sequence by replaying it headlessly."""
    from retro_harness.platformer.bk2_extract import load_raw_buttons

    config = _resolve_config(args)
    explicit_state = getattr(args, "state", None)
    actions_path = Path(args.actions)
    if not actions_path.exists():
        print(f"Error: actions file not found: {actions_path}")
        return
    metadata_state = _recording_start_state(actions_path)
    start_state = explicit_state or metadata_state

    raw = load_raw_buttons(actions_path)
    if raw is not None:
        actions: list[int] | list[list[int]] = raw
        print(f"Loaded {len(actions)} frames (raw buttons) from {actions_path}")
    else:
        actions = load_actions(actions_path)
        print(f"Loaded {len(actions)} frames (action indices) from {actions_path}")
    print(f"Level: {config.display_name}")
    if explicit_state:
        print(f"State override: {start_state}")
    elif metadata_state:
        print(f"State from recording metadata: {start_state}")

    evaluator = Evaluator(config, start_state=start_state)

    if getattr(args, "trace", False):
        print("Tracing level_id changes (no early termination)...")
        start = time.time()
        result = evaluator.evaluate_trace(actions)
        elapsed = time.time() - start
    else:
        print("Evaluating (no early termination)...")
        start = time.time()
        result = evaluator.evaluate(actions, early_terminate=False)
        elapsed = time.time() - start

    gameplay_frames = result.total_frames - result.gameplay_start_frame
    print(f"\nResult:")
    print(f"  Completed:      {result.completed}")
    print(f"  Died:           {result.died}")
    print(f"  Total frames:   {result.total_frames}")
    print(f"  Gameplay start: frame {result.gameplay_start_frame}")
    print(f"  Gameplay frames:{gameplay_frames}")
    print(f"  Gameplay secs:  {gameplay_frames / 60:.2f}s")
    print(f"  Timer frames:   {result.timer_frames}")
    print(f"  Timer secs:     {result.timer_frames / 60:.2f}s")
    print(f"  Max X:          {result.max_x:.1f}")
    print(f"  Max progress:   {result.max_progress:.1f}")
    print(f"  Final pos:      ({result.final_x:.1f}, {result.final_y:.1f})")
    print(f"  Level ID end:   0x{result.level_id_at_end:02X} ({result.level_id_at_end})")
    print(f"  Bonus frames:   {result.bonus_frames}")
    print(f"  Fitness:        {result.fitness:.1f}")
    print(f"  Eval time:      {elapsed:.2f}s")

    evaluator.close()




def cmd_watch(args: argparse.Namespace) -> None:
    """Watch an action sequence play out visually using pygame."""
    from retro_harness.platformer.bk2_extract import load_raw_buttons

    config = _resolve_config(args)
    start_state = getattr(args, "state", None)
    actions_path = Path(args.actions)
    if not actions_path.exists():
        print(f"Error: actions file not found: {actions_path}")
        return

    raw = load_raw_buttons(actions_path)
    if raw is not None:
        actions: list[int] | list[list[int]] = raw
        print(f"Loaded {len(actions)} frames (raw buttons) from {actions_path}")
    else:
        actions = load_actions(actions_path)
        print(f"Loaded {len(actions)} frames (action indices) from {actions_path}")
    print(f"Level: {config.display_name}")
    if start_state:
        print(f"State override: {start_state}")
    print("Controls: SPACE=pause  [/]=speed  N=note  1-5=tag  LEFT/RIGHT=step  ESC=quit")

    _replay_with_hud(config, actions, scale=args.scale, start_state=start_state, actions_path=actions_path)
    print("Done.")


def cmd_watch_bk2(args: argparse.Namespace) -> None:
    """Replay a bk2 recording visually using its embedded state."""
    import numpy as np
    import pygame
    import stable_retro as retro
    from retro_harness.env import add_custom_integrations

    config = _resolve_config(args)
    bk2_path = Path(args.bk2)
    if not bk2_path.exists():
        print(f"Error: bk2 file not found: {bk2_path}")
        return

    raw_actions = extract_raw_actions_from_bk2(bk2_path, bk2_to_env=config.bk2_to_env)
    print(f"Extracted {len(raw_actions)} frames from {bk2_path}")

    add_custom_integrations(config.game_dir)
    movie = retro.Movie(str(bk2_path))
    game = movie.get_game()
    env = retro.make(
        game=game,
        state=retro.State.NONE,
        render_mode="rgb_array",
        inttype=retro.data.Integrations.CUSTOM_ONLY,
    )
    env.initial_state = movie.get_state()
    obs, _ = env.reset()

    pygame.init()
    scale = args.scale
    width, height = obs.shape[1], obs.shape[0]
    screen = pygame.display.set_mode(
        (width * scale, height * scale), pygame.SWSURFACE
    )
    pygame.display.set_caption(f"BK2 Replay: {bk2_path.parent.name}")
    clock = pygame.time.Clock()

    print("Playing... (close window or press ESC to stop)")
    running = True
    for frame_idx, buttons in enumerate(raw_actions):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
        if not running:
            break

        action_size = env.action_space.shape[0]
        if len(buttons) < action_size:
            buttons = buttons + [0] * (action_size - len(buttons))

        obs, reward, terminated, truncated, info = env.step(
            np.array(buttons, dtype=np.int8)
        )

        surf = pygame.surfarray.make_surface(obs.swapaxes(0, 1))
        screen.blit(pygame.transform.scale(surf, screen.get_size()), (0, 0))
        pygame.display.flip()
        clock.tick(60)

        if terminated or truncated:
            print(f"Episode ended at frame {frame_idx}")
            break

    if running:
        for _ in range(120):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    break
            env.step(np.zeros(env.action_space.shape[0], dtype=np.int8))
            pygame.display.flip()
            clock.tick(60)

    pygame.quit()
    env.close()
    print("Done.")


def cmd_extract_all(args: argparse.Namespace) -> None:
    """Extract and evaluate all bk2 recordings in the recordings directory."""
    config = _resolve_config(args)
    recordings_dir = Path(args.recordings_dir) if args.recordings_dir else config.game_dir / "recordings"
    if not recordings_dir.exists():
        print(f"Error: recordings directory not found: {recordings_dir}")
        return

    bk2_files = sorted(recordings_dir.rglob("*.bk2"))
    if not bk2_files:
        print("No bk2 files found.")
        return

    action_table = _get_action_table(config)
    print(f"Found {len(bk2_files)} bk2 files")
    print(f"Level: {config.display_name}")

    evaluator = Evaluator(config)
    results = []

    for bk2_path in bk2_files:
        folder = bk2_path.parent.name
        print(f"\n--- {folder}/{bk2_path.name} ---")

        actions = extract_action_indices_from_bk2(
            bk2_path, action_table=action_table, bk2_to_env=config.bk2_to_env
        )
        print(f"  Frames: {len(actions)}")

        result = evaluator.evaluate(actions, early_terminate=False)
        print(f"  Completed: {result.completed}")
        print(f"  Fitness: {result.fitness:.1f}")
        print(f"  Max X: {result.max_x:.1f}")
        if result.completed:
            print(f"  Total frames: {result.total_frames}")
            print(f"  Timer: {result.timer_frames / 60:.2f}s")

        results.append({
            "bk2": str(bk2_path),
            "folder": folder,
            "num_frames": len(actions),
            "completed": result.completed,
            "fitness": result.fitness,
            "total_frames": result.total_frames,
            "max_x": result.max_x,
            "timer_seconds": result.timer_frames / 60 if result.completed else None,
        })

        output = config.runs_dir / f"{folder}_extracted.json"
        metadata = {"source_bk2": str(bk2_path), "level": config.level_id}
        save_actions(actions, output, metadata=metadata)

    evaluator.close()

    print("\n\n=== SUMMARY (sorted by fitness) ===")
    results.sort(key=lambda r: r["fitness"], reverse=True)
    for r in results:
        timer_str = f"{r['timer_seconds']:.2f}s" if r["timer_seconds"] else "N/A"
        status = "DONE" if r["completed"] else "FAIL"
        print(
            f"  {r['folder']:15s} {status:4s} "
            f"fitness={r['fitness']:10.1f} "
            f"frames={r['total_frames']:5d} "
            f"timer={timer_str:8s} "
            f"max_x={r['max_x']:7.1f}"
        )

    best = results[0] if results else None
    if best and best["completed"]:
        print(f"\nBest completed run: {best['folder']} ({best['timer_seconds']:.2f}s)")
        print(f"  Seed file: {config.runs_dir / (best['folder'] + '_extracted.json')}")


def cmd_prepare_seeds(args: argparse.Namespace) -> None:
    """Batch-process recordings: extract all BK2s, evaluate, save top N as seeds."""
    config = _resolve_config(args)
    recordings_dir = Path(args.recordings_dir) if args.recordings_dir else config.game_dir / "recordings"
    if not recordings_dir.exists():
        print(f"Error: recordings directory not found: {recordings_dir}")
        return

    bk2_files = sorted(recordings_dir.rglob("*.bk2"))
    if not bk2_files:
        print("No bk2 files found.")
        return

    action_table = _get_action_table(config)
    top_n = args.top
    print(f"Found {len(bk2_files)} bk2 files, selecting top {top_n}")
    print(f"Level: {config.display_name}")

    evaluator = Evaluator(config)
    candidates: list[tuple[float, str, list[int]]] = []

    for bk2_path in bk2_files:
        actions = extract_action_indices_from_bk2(
            bk2_path, action_table=action_table, bk2_to_env=config.bk2_to_env
        )
        result = evaluator.evaluate(actions, early_terminate=False)
        candidates.append((result.fitness, str(bk2_path), actions))
        status = "COMPLETE" if result.completed else "incomplete"
        print(f"  {bk2_path.name}: fitness={result.fitness:.1f} {status}")

    evaluator.close()

    # Sort by fitness descending, take top N
    candidates.sort(key=lambda c: c[0], reverse=True)
    seeds_dir = config.runs_dir / "seeds"
    seeds_dir.mkdir(parents=True, exist_ok=True)

    for i, (fitness, source, actions) in enumerate(candidates[:top_n]):
        output_path = seeds_dir / f"seed_{i:02d}.json"
        metadata = {"source_bk2": source, "fitness": fitness, "rank": i, "level": config.level_id}
        save_actions(actions, output_path, metadata=metadata)

    print(f"\nSaved {min(top_n, len(candidates))} seeds to {seeds_dir}")
    if candidates:
        print(f"Best: fitness={candidates[0][0]:.1f} from {Path(candidates[0][1]).name}")


def cmd_auto_state(args: argparse.Namespace) -> None:
    """Create a save state by navigating from an existing state."""
    from retro_harness.platformer.auto_state import parse_nav_string, navigate_and_save

    config = _resolve_config(args)
    steps = parse_nav_string(args.nav)

    result = navigate_and_save(
        game_name=config.game_name,
        game_dir=config.game_dir,
        from_state=args.from_state,
        save_name=config.start_state,
        steps=steps,
        ram=config.ram,
        expected_level_id=config.target_level_id if config.target_level_id != 0 else None,
        settle_frames=args.settle,
        save_screenshot=args.screenshot,
    )

    if not result.success:
        sys.exit(1)




def cmd_trace_map(args: argparse.Namespace) -> None:
    """Render a position trace overlaid on an area map PNG."""
    config = _resolve_config(args)

    # Resolve trace path
    trace_path = None
    if getattr(args, "trace", None):
        trace_path = Path(args.trace)
    elif getattr(args, "actions", None):
        trace_path = Path(args.actions).parent / f"{Path(args.actions).stem}_trace.json"
    else:
        # Look in runs dir for most recent trace
        traces = sorted(config.runs_dir.glob("*_trace.json"))
        if traces:
            trace_path = traces[-1]

    if trace_path is None or not trace_path.exists():
        print(f"Error: trace file not found: {trace_path}")
        print("Run 'watch' first to generate a trace, or specify --trace path.")
        return

    output = Path(args.output) if getattr(args, "output", None) else trace_path.with_suffix(".png")
    map_dir = Path(args.map_dir) if getattr(args, "map_dir", None) else None

    # Game-owned renderers only (shared harness must not import game map code).
    # Games may expose ``render_trace_map(trace_path=, level_id=, output_path=, map_dir=, **kw)``.
    renderer = None
    for module_name in (
        f"{config.game_name.split('-')[0].lower()}.trace_renderer",
        "smb.trace_renderer",
        "super_metroid.trace_renderer",
    ):
        try:
            mod = __import__(module_name, fromlist=["render_trace_map"])
        except ImportError:
            continue
        renderer = getattr(mod, "render_trace_map", None) or getattr(
            mod, "render_smb_trace", None
        )
        if renderer is not None:
            break

    if renderer is None:
        print(
            "Error: no game-owned trace renderer found. Implement "
            "`render_trace_map` under the game package (e.g. smb.trace_renderer "
            "or super_metroid.trace_renderer). Shared harness does not embed "
            "game map renderers."
        )
        sys.exit(2)

    renderer(
        trace_path=trace_path,
        level_id=config.level_id,
        output_path=output,
        map_dir=map_dir,
        area=getattr(args, "area", None),
    )


# -- Route commands ----------------------------------------------------------


def cmd_list_routes(args: argparse.Namespace) -> None:
    """List all registered routes."""
    from retro_harness.platformer.route import list_routes

    routes = list_routes()
    if not routes:
        print("No routes registered.")
        return

    print(f"{'ID':<25s} {'Display Name':<40s} {'Segments':>8s}")
    print("-" * 75)
    for r in routes:
        print(f"{r.route_id:<25s} {r.display_name:<40s} {len(r.segments):>8d}")


def cmd_chain(args: argparse.Namespace) -> None:
    """Evaluate a full speedrun route (all segments independently)."""
    from retro_harness.platformer.route import get_route, evaluate_route

    route = get_route(args.route)
    result = evaluate_route(route, verbose=True)

    if result.all_completed:
        print(f"\nAll segments completed! Total: {result.total_frames}f "
              f"({result.total_frames / 60:.1f}s)")
    else:
        sys.exit(1)


def cmd_chain_live(args: argparse.Namespace) -> None:
    """Run a true end-to-end chain on a single emulator (no state reloads)."""
    from retro_harness.platformer.route import get_route, chain_live

    route = get_route(args.route)
    result = chain_live(
        route,
        save_states=args.save_states,
        verbose=True,
        video_path=args.video,
        video_scale=args.scale,
    )

    if result.all_completed:
        print(f"\nFull chain completed! {result.total_frames}f ({result.total_frames / 60:.1f}s)")
    else:
        sys.exit(1)


def cmd_chain_optimize(args: argparse.Namespace) -> None:
    """Iteratively hill-climb each segment from chained states."""
    from retro_harness.platformer.route import get_route, chain_optimize

    route = get_route(args.route)
    result = chain_optimize(
        route,
        iterations=args.iterations,
        verbose=True,
    )

    if result.all_completed:
        print(f"\nFull chain optimized! {result.total_frames}f ({result.total_frames / 60:.1f}s)")
    else:
        sys.exit(1)


def cmd_chain_video(args: argparse.Namespace) -> None:
    """Render a full speedrun route to a single MP4."""
    from retro_harness.platformer.route import get_route, record_route_video

    route = get_route(args.route)
    output = args.output or f"{route.route_id}.mp4"
    record_route_video(route, output, scale=args.scale)


# -- Main CLI ----------------------------------------------------------------
