"""CLI entry point for platformer speedrun optimizer.

All commands take a --level flag to select the level to optimize.
Level configs are registered by importing platformer_common.levels.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Trigger level registration on import
import platformer_common.levels  # noqa: F401

from platformer_common.level_config import LevelConfig, get_level_config, list_levels
from platformer_common.actions import (
    DEFAULT_PLATFORMER_ACTIONS,
    action_index_to_buttons,
    buttons_to_action_index,
)
from platformer_common.bk2_extract import (
    extract_action_indices_from_bk2,
    extract_raw_actions_from_bk2,
    save_actions,
    load_actions,
)
from platformer_common.evaluator import Evaluator


def _resolve_config(args: argparse.Namespace) -> LevelConfig:
    """Get level config from --level arg."""
    return get_level_config(args.level)


def _get_action_table(config: LevelConfig) -> list[list[int]]:
    return config.action_table or DEFAULT_PLATFORMER_ACTIONS


# -- Commands ----------------------------------------------------------------


def cmd_list_levels(args: argparse.Namespace) -> None:
    """List all registered levels."""
    from platformer_common.level_config import LEVEL_REGISTRY

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


def cmd_verify(args: argparse.Namespace) -> None:
    """Verify an action sequence by replaying it headlessly."""
    from platformer_common.bk2_extract import load_raw_buttons

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


def cmd_optimize(args: argparse.Namespace) -> None:
    """Run GA optimization on a seed action sequence."""
    from platformer_common.genetic import run_ga

    config = _resolve_config(args)
    start_state = getattr(args, "state", None)
    seed_path = Path(args.seed)
    if not seed_path.exists():
        print(f"Error: seed file not found: {seed_path}")
        return

    seed_actions = load_actions(seed_path)
    print(f"Seed: {len(seed_actions)} frames from {seed_path}")
    print(f"Level: {config.display_name}")

    output_dir = Path(args.output_dir) if args.output_dir else config.runs_dir
    evaluator = Evaluator(config, start_state=start_state)

    resume_from = Path(args.resume) if args.resume else None

    best = run_ga(
        seed_actions=seed_actions,
        evaluator=evaluator,
        population_size=args.population,
        num_generations=args.generations,
        output_dir=output_dir,
        num_workers=args.workers,
        resume_from=resume_from,
        render_interval=args.render,
    )

    final_path = output_dir / "ga_best_final.json"
    data = {
        "actions": best.actions,
        "num_frames": len(best.actions),
        "fitness": best.fitness,
        "completed": best.result.completed if best.result else False,
        "total_frames": best.result.total_frames if best.result else 0,
        "max_progress": best.result.max_progress if best.result else 0,
        "level": config.level_id,
    }
    final_path.write_text(json.dumps(data, indent=2))
    print(f"\nSaved best to {final_path}")

    evaluator.close()


def cmd_hillclimb_raw(args: argparse.Namespace) -> None:
    """Hill climb with raw button mutation (no lossy action-index conversion)."""
    from platformer_common.hillclimb_raw import hillclimb_raw
    from platformer_common.bk2_extract import load_raw_buttons

    config = _resolve_config(args)
    start_state = getattr(args, "state", None)
    seed_path = Path(args.seed)
    if not seed_path.exists():
        print(f"Error: seed file not found: {seed_path}")
        return

    raw = load_raw_buttons(seed_path)
    if raw is None:
        print(f"Error: no raw_buttons in {seed_path}")
        return

    print(f"Seed: {len(raw)} raw frames from {seed_path}")
    print(f"Level: {config.display_name}")
    if start_state:
        print(f"State override: {start_state}")

    output_dir = Path(args.output_dir) if args.output_dir else config.runs_dir
    evaluator = Evaluator(config, start_state=start_state)

    best_raw, best_result = hillclimb_raw(
        raw_buttons=raw,
        evaluator=evaluator,
        max_iterations=args.iterations,
        output_dir=output_dir,
    )

    print(f"\nSaved best to {output_dir / 'hillclimb_raw_best.json'}")
    evaluator.close()


def cmd_hillclimb(args: argparse.Namespace) -> None:
    """Run hill climbing refinement on an action sequence."""
    from platformer_common.hillclimb import hillclimb
    from platformer_common.bk2_extract import load_raw_buttons

    config = _resolve_config(args)
    start_state = getattr(args, "state", None)
    seed_path = Path(args.seed)
    if not seed_path.exists():
        print(f"Error: seed file not found: {seed_path}")
        return

    # Prefer raw buttons if available (faithful replay)
    raw = load_raw_buttons(seed_path)
    if raw is not None:
        # Convert raw buttons to action indices for hill climbing
        # (hill climber mutates action indices, not raw buttons)
        seed_actions = [
            buttons_to_action_index(frame, action_table=_get_action_table(config))
            for frame in raw
        ]
        print(f"Seed: {len(seed_actions)} frames (from raw buttons) from {seed_path}")
    else:
        seed_actions = load_actions(seed_path)
        print(f"Seed: {len(seed_actions)} frames from {seed_path}")
    print(f"Level: {config.display_name}")
    if start_state:
        print(f"State override: {start_state}")

    output_dir = Path(args.output_dir) if args.output_dir else config.runs_dir
    evaluator = Evaluator(config, start_state=start_state)

    best_actions, best_result = hillclimb(
        actions=seed_actions,
        evaluator=evaluator,
        max_iterations=args.iterations,
        output_dir=output_dir,
        render_interval=args.render,
        render_scale=args.scale,
    )

    final_path = output_dir / "hillclimb_best_final.json"
    data = {
        "actions": best_actions,
        "num_frames": len(best_actions),
        "fitness": best_result.fitness,
        "completed": best_result.completed,
        "total_frames": best_result.total_frames,
        "max_x": best_result.max_x,
        "max_progress": best_result.max_progress,
        "bonus_frames": best_result.bonus_frames,
        "level": config.level_id,
    }
    final_path.write_text(json.dumps(data, indent=2))
    print(f"\nSaved best to {final_path}")

    evaluator.close()


def _button_names(buttons: list[int]) -> str:
    """Format a 12-element button array as compact pressed-button string."""
    names = ["B", "Y", "Sel", "Sta", "U", "D", "L", "R", "A", "X", "L1", "R1"]
    pressed = [names[i] for i in range(min(len(buttons), len(names))) if buttons[i]]
    return "+".join(pressed) if pressed else "-"


def _replay_with_hud(
    config,
    actions: list[int] | list[list[int]],
    scale: int = 3,
    title: str | None = None,
    start_state: str | None = None,
    actions_path: Path | None = None,
) -> None:
    """Shared replay logic with HUD overlay for watch/hillclimb render.

    Controls:
      SPACE       Pause/resume
      RIGHT/LEFT  Step forward/backward one frame (while paused)
      N           Add note at current frame (while paused)
      1-5         Toggle issue tags on current frame (while paused)
      [ / ]       Decrease/increase playback speed
      ESC         Quit (saves annotations and trace if any)
    """
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "x11")

    import numpy as np
    import pygame

    action_table = _get_action_table(config)

    from retro_harness.env import make_env

    env = make_env(
        game=config.game_name,
        state=start_state or config.start_state,
        game_dir=config.game_dir,
        render_mode="rgb_array",
    )
    obs, _ = env.reset()
    initial_obs = obs.copy()

    from platformer_common.progress import make_progress_tracker

    # Save emulator state for rewind
    initial_emu_state = env.em.get_state()

    schema = config.ram_schema
    tracker = make_progress_tracker(config)
    ram = env.get_ram()
    initial_values = schema.read(ram)
    config.apply_computed(initial_values)
    initial_lives = initial_values.get("lives")
    initial_cam_x = float(initial_values.get("camera_x", 0))
    # Seed tracker with initial values (same as evaluator)
    tracker.reset()
    tracker.update(initial_values)
    # For camera-based games, gate death detection on camera scroll;
    # for player-position games (SM), start immediately.
    gameplay_started = initial_cam_x == 0

    # -- Annotation constants --
    TAG_KEYS = {
        pygame.K_1: "ledge_hit",
        pygame.K_2: "bad_path",
        pygame.K_3: "slow",
        pygame.K_4: "good",
        pygame.K_5: "other",
    }
    TAG_COLORS = {
        "ledge_hit": (255, 80, 80),
        "bad_path": (255, 165, 0),
        "slow": (255, 255, 0),
        "good": (80, 255, 80),
        "other": (160, 160, 160),
    }
    SPEEDS = [0.25, 0.5, 1.0, 2.0, 4.0]
    speed_idx = 2  # 1.0x

    # -- Load existing annotations --
    annotations: dict[int, dict] = {}
    annotations_changed = False
    annotations_file = None
    if actions_path is not None:
        annotations_file = actions_path.parent / f"{actions_path.stem}_annotations.json"
        if annotations_file.exists():
            try:
                data = json.loads(annotations_file.read_text())
                for entry in data.get("annotations", []):
                    annotations[entry["frame"]] = {
                        "tags": list(entry.get("tags", [])),
                        "note": entry.get("note", ""),
                    }
                print(f"Loaded {len(annotations)} annotations from {annotations_file}")
            except Exception as e:
                print(f"Warning: could not load annotations: {e}")

    # -- Trace collection --
    trace: list[dict] = []
    trace_rooms: dict[int, dict] = {}  # room_id -> {enter_frame, last_frame}

    pygame.init()
    width, height = obs.shape[1], obs.shape[0]
    timeline_h = 8
    screen = pygame.display.set_mode(
        (width * scale, height * scale + timeline_h), pygame.SWSURFACE
    )
    pygame.display.set_caption(title or f"Replay: {config.display_name}")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 16)

    running = True
    paused = False
    text_input_mode = False
    text_input_buf = ""
    max_progress = 0.0
    raw_mode = len(actions) > 0 and isinstance(actions[0], list)
    total_frames = len(actions)
    current_frame = -1  # most recently executed frame (-1 = initial state)
    current_values = dict(initial_values)
    current_buttons: list[int] | None = None
    current_in_sub = False

    def _get_buttons(frame_i: int) -> list[int]:
        """Convert actions[frame_i] to padded button array."""
        act = actions[frame_i]
        if raw_mode:
            btns = list(act)  # type: ignore[arg-type]
        else:
            btns = action_index_to_buttons(act, action_table)  # type: ignore[arg-type]
        action_size = env.action_space.shape[0]
        if len(btns) < action_size:
            btns = btns + [0] * (action_size - len(btns))
        elif len(btns) > action_size:
            btns = btns[:action_size]
        return btns

    def _simulate_to(target: int) -> None:
        """Re-simulate from initial state through actions[0..target].

        Updates nonlocal: obs, tracker, max_progress, gameplay_started,
        current_values, current_buttons, current_in_sub, current_frame.
        """
        nonlocal obs, tracker, max_progress, gameplay_started
        nonlocal current_values, current_buttons, current_in_sub, current_frame

        env.em.set_state(initial_emu_state)
        tracker = make_progress_tracker(config)
        _ram = env.get_ram()
        _vals = schema.read(_ram)
        config.apply_computed(_vals)
        _init_cam = float(_vals.get("camera_x", 0))
        _gs = _init_cam == 0
        tracker.reset()
        tracker.update(_vals)
        _mp = 0.0

        if target < 0:
            obs = initial_obs.copy()
            current_values = dict(initial_values)
            current_buttons = None
            current_in_sub = False
            current_frame = -1
            max_progress = 0.0
            gameplay_started = _gs
            return

        _in_sub = False
        for i in range(target + 1):
            btns = _get_buttons(i)
            obs, _, _, _, _ = env.step(np.array(btns, dtype=np.int8))
            _ram = env.get_ram()
            _vals = schema.read(_ram)
            config.apply_computed(_vals)
            _cam = float(_vals.get("camera_x", 0))
            _lid = _vals.get("level_id", config.target_level_id)
            _mids = {config.target_level_id} | set(config.level_id_aliases)
            _in_sub = _lid != 0 and _lid not in _mids
            if not _in_sub:
                _p = tracker.update(_vals)
                if _p > _mp:
                    _mp = _p
            if not _gs and _cam > _init_cam:
                _gs = True

        current_frame = target
        current_values = _vals
        current_buttons = _get_buttons(target)
        current_in_sub = _in_sub
        max_progress = _mp
        gameplay_started = _gs

    def _step_one() -> bool:
        """Execute one frame forward. Returns True if a frame was played."""
        nonlocal obs, max_progress, gameplay_started, running
        nonlocal current_values, current_buttons, current_in_sub, current_frame

        next_f = current_frame + 1
        if next_f >= total_frames:
            return False

        btns = _get_buttons(next_f)
        obs, reward, terminated, truncated, info = env.step(
            np.array(btns, dtype=np.int8)
        )

        ram = env.get_ram()
        values = schema.read(ram)
        config.apply_computed(values)
        cam_x = float(values.get("camera_x", 0))
        level_id = values.get("level_id", config.target_level_id)

        _main_ids = {config.target_level_id} | set(config.level_id_aliases)
        in_sub = level_id != 0 and level_id not in _main_ids

        if not in_sub:
            progress = tracker.update(values)
            if progress > max_progress:
                max_progress = progress

        if not gameplay_started and cam_x > initial_cam_x:
            gameplay_started = True

        current_frame = next_f
        current_values = values
        current_buttons = btns
        current_in_sub = in_sub

        # Collect trace point
        btn_str = _button_names(btns) if btns else "-"
        trace_pt: dict = {
            "frame": current_frame,
            "room_id": level_id,
            "x": int(values.get("player_x", 0)),
            "y": int(values.get("player_y", 0)),
            "buttons": btn_str,
        }
        health_val = values.get("health")
        if health_val is not None:
            trace_pt["health"] = int(health_val)
        trace.append(trace_pt)

        # Track room transitions
        if level_id not in trace_rooms:
            trace_rooms[level_id] = {"enter_frame": current_frame, "last_frame": current_frame}
        else:
            trace_rooms[level_id]["last_frame"] = current_frame

        # Check completion
        if config.completion_signal == "level_id_change":
            if level_id not in _main_ids and level_id != 0:
                is_real = (
                    max_progress >= config.completion_min_progress
                    and (not config.completion_level_ids
                         or level_id in config.completion_level_ids)
                    and level_id not in config.completion_exclude_ids
                )
                if is_real:
                    print(f"  COMPLETED at frame {current_frame}: level_id=0x{level_id:04X}, progress={max_progress:.0f}")
                    running = False
                    return True

        # Check death
        if gameplay_started:
            for signal in config.death_signals:
                if signal == "lives_drop":
                    lives = values.get("lives")
                    if initial_lives is not None and lives is not None and lives < initial_lives:
                        print(f"  DIED at frame {current_frame}: lives {initial_lives}->{lives}, progress={max_progress:.0f}")
                        running = False
                        return True
                elif signal == "health_zero":
                    health = values.get("health", 1)
                    if health <= 0:
                        print(f"  DIED at frame {current_frame}: health=0, progress={max_progress:.0f}")
                        running = False
                        return True
                elif signal == "camera_reset":
                    if in_sub:
                        continue
                    if initial_cam_x > config.camera_reset_threshold and cam_x < initial_cam_x - config.camera_reset_threshold:
                        print(f"  DIED at frame {current_frame}: camera reset ({cam_x:.0f} << {initial_cam_x:.0f})")
                        running = False
                        return True

        if terminated or truncated:
            print(f"Episode ended at frame {current_frame}")
            running = False
            return True

        return True

    def _draw() -> None:
        """Render current frame + HUD + timeline."""
        surf = pygame.surfarray.make_surface(obs.swapaxes(0, 1))
        screen.blit(
            pygame.transform.scale(surf, (width * scale, height * scale)),
            (0, 0),
        )

        # HUD text
        btn_str = _button_names(current_buttons) if current_buttons else "-"
        lives_val = current_values.get("lives", "?")
        health_val = current_values.get("health")
        bonus_tag = " | BONUS" if current_in_sub else ""
        speed_str = f" | {SPEEDS[speed_idx]}x" if speed_idx != 2 else ""
        frame_display = max(current_frame, 0)

        lines: list[str] = []
        if paused:
            lines.append("PAUSED  [N]note [1-5]tag [LEFT/RIGHT]step [[ ]]speed")
        lines.append(
            f"F{frame_display}/{total_frames} | {btn_str}{bonus_tag}{speed_str}"
        )
        lines.append(
            f"progress={max_progress:.0f} | "
            + (f"hp={health_val}" if health_val is not None else f"lives={lives_val}")
            + f" | cam={float(current_values.get('camera_x', 0)):.0f}"
        )

        # Show annotations for current frame
        if current_frame in annotations:
            ann = annotations[current_frame]
            tags = ann.get("tags", [])
            note = ann.get("note", "")
            if tags:
                lines.append(f"TAGS: {', '.join(tags)}")
            if note:
                lines.append(f"NOTE: {note}")

        for i, line in enumerate(lines):
            text = font.render(line, True, (255, 255, 0))
            screen.blit(text, (4, 4 + i * 18))

        # Text input bar
        if text_input_mode:
            bar_y = height * scale - 28
            pygame.draw.rect(screen, (0, 0, 100), (0, bar_y, width * scale, 28))
            prompt = font.render(f"Note: {text_input_buf}_", True, (255, 255, 255))
            screen.blit(prompt, (4, bar_y + 5))

        # Timeline bar
        bar_y = height * scale
        bar_w = width * scale
        pygame.draw.rect(screen, (30, 30, 30), (0, bar_y, bar_w, timeline_h))
        if total_frames > 0:
            for ann_f, ann_data in annotations.items():
                if 0 <= ann_f < total_frames:
                    x = int(ann_f / total_frames * bar_w)
                    tags = ann_data.get("tags", [])
                    color = TAG_COLORS.get(tags[0], (160, 160, 160)) if tags else (160, 160, 160)
                    pygame.draw.rect(screen, color, (x - 1, bar_y, 3, timeline_h))
            px = int(max(current_frame, 0) / max(total_frames, 1) * bar_w)
            pygame.draw.rect(screen, (255, 255, 255), (px - 1, bar_y, 3, timeline_h))

        pygame.display.flip()

    # -- Main loop --
    while running:
        if text_input_mode:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    text_input_mode = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        text_input_mode = False
                        text_input_buf = ""
                    elif event.key == pygame.K_RETURN:
                        if current_frame not in annotations:
                            annotations[current_frame] = {"tags": [], "note": ""}
                        annotations[current_frame]["note"] = text_input_buf
                        annotations_changed = True
                        text_input_mode = False
                        text_input_buf = ""
                    elif event.key == pygame.K_BACKSPACE:
                        text_input_buf = text_input_buf[:-1]
                    elif event.unicode and event.unicode.isprintable():
                        text_input_buf += event.unicode
            _draw()
            clock.tick(30)

        elif paused:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = False
                    elif event.key == pygame.K_RIGHT:
                        _step_one()
                    elif event.key == pygame.K_LEFT:
                        if current_frame > 0:
                            _simulate_to(current_frame - 1)
                    elif event.key == pygame.K_n:
                        existing = annotations.get(current_frame, {}).get("note", "")
                        text_input_buf = existing
                        text_input_mode = True
                    elif event.key in TAG_KEYS:
                        tag = TAG_KEYS[event.key]
                        if current_frame not in annotations:
                            annotations[current_frame] = {"tags": [], "note": ""}
                        tags = annotations[current_frame]["tags"]
                        if tag in tags:
                            tags.remove(tag)
                        else:
                            tags.append(tag)
                        if not tags and not annotations[current_frame]["note"]:
                            del annotations[current_frame]
                        annotations_changed = True
                    elif event.key == pygame.K_LEFTBRACKET:
                        speed_idx = max(0, speed_idx - 1)
                    elif event.key == pygame.K_RIGHTBRACKET:
                        speed_idx = min(len(SPEEDS) - 1, speed_idx + 1)
            _draw()
            clock.tick(30)

        else:
            # Normal playback
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = True
                    elif event.key == pygame.K_LEFTBRACKET:
                        speed_idx = max(0, speed_idx - 1)
                    elif event.key == pygame.K_RIGHTBRACKET:
                        speed_idx = min(len(SPEEDS) - 1, speed_idx + 1)

            if not running or paused:
                if paused:
                    _draw()
                continue

            if current_frame + 1 >= total_frames:
                break

            _step_one()
            if not running:
                break

            _draw()
            clock.tick(int(60 * SPEEDS[speed_idx]))

    # Post-playback idle (2 seconds at last frame)
    if running and current_frame + 1 >= total_frames:
        for _ in range(120):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    break
            env.step(np.zeros(env.action_space.shape[0], dtype=np.int8))
            pygame.display.flip()
            clock.tick(60)

    # Save annotations on exit
    if annotations and annotations_file is not None:
        ann_list = []
        for frame_num in sorted(annotations):
            entry: dict = {"frame": frame_num}
            ann = annotations[frame_num]
            if ann.get("tags"):
                entry["tags"] = ann["tags"]
            if ann.get("note"):
                entry["note"] = ann["note"]
            ann_list.append(entry)
        out = {
            "actions_file": actions_path.name if actions_path else "",
            "annotations": ann_list,
        }
        annotations_file.write_text(json.dumps(out, indent=2))
        print(f"Saved {len(annotations)} annotations to {annotations_file}")

    # Save trace JSON on exit
    if trace and actions_path is not None:
        trace_file = actions_path.parent / f"{actions_path.stem}_trace.json"

        # Build rooms_visited summary
        rooms_visited = []
        for rid, info in sorted(trace_rooms.items(), key=lambda kv: kv[1]["enter_frame"]):
            rooms_visited.append({
                "room_id": rid,
                "enter_frame": info["enter_frame"],
                "exit_frame": info["last_frame"],
                "frames": info["last_frame"] - info["enter_frame"] + 1,
            })

        # Compute center of gravity (most-visited room, mean x/y)
        cog: dict = {}
        if rooms_visited:
            most_visited = max(rooms_visited, key=lambda r: r["frames"])
            rid = most_visited["room_id"]
            room_pts = [(pt["x"], pt["y"]) for pt in trace if pt["room_id"] == rid]
            if room_pts:
                mean_x = sum(p[0] for p in room_pts) / len(room_pts)
                mean_y = sum(p[1] for p in room_pts) / len(room_pts)
                cog = {"x": round(mean_x, 1), "y": round(mean_y, 1), "room_id": rid}

        # Build annotation list for trace
        ann_list_for_trace = []
        for frame_num in sorted(annotations):
            entry_t: dict = {"frame": frame_num}
            ann_t = annotations[frame_num]
            if ann_t.get("tags"):
                entry_t["tags"] = ann_t["tags"]
            if ann_t.get("note"):
                entry_t["note"] = ann_t["note"]
            ann_list_for_trace.append(entry_t)

        trace_out = {
            "level": config.level_id,
            "total_frames": max(current_frame, 0) + 1,
            "trace": trace,
            "rooms_visited": rooms_visited,
            "center_of_gravity": cog,
            "annotations": ann_list_for_trace,
        }
        trace_file.write_text(json.dumps(trace_out))
        print(f"Saved trace ({len(trace)} points) to {trace_file}")

    pygame.quit()
    env.close()


def cmd_watch(args: argparse.Namespace) -> None:
    """Watch an action sequence play out visually using pygame."""
    from platformer_common.bk2_extract import load_raw_buttons

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
    from platformer_common.auto_state import parse_nav_string, navigate_and_save

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


def cmd_play(args: argparse.Namespace) -> None:
    """Play a level manually while recording inputs as action indices."""
    import numpy as np

    config = _resolve_config(args)
    action_table = _get_action_table(config)
    start_state = getattr(args, "state", None) or config.start_state

    from retro_harness.env import make_env

    env = make_env(
        game=config.game_name,
        state=start_state,
        game_dir=config.game_dir,
        render_mode="rgb_array",
    )

    from retro_harness.play_session import PlaySession
    from platformer_common.progress import make_progress_tracker

    schema = config.ram_schema
    tracker = make_progress_tracker(config)
    tracker.reset()
    recorded_actions: list[int] = []
    recorded_raw: list[list[int]] = []
    best_progress = 0.0
    tracker_seeded = False

    # Workaround: stable-retro ignores SNES Select for SM weapon toggle.
    # Track toggle state and force via RAM on rising edge of Select.
    _select_state = [0, False]  # [current_item, was_pressed_last_frame]

    def on_step(obs, reward, done, info):
        nonlocal best_progress, tracker_seeded
        # Record the action index and raw buttons for this frame
        raw = list(last_raw_action)
        idx = buttons_to_action_index(raw, action_table=action_table)
        recorded_actions.append(idx)
        recorded_raw.append(raw)

        # Workaround: force weapon toggle via RAM on Select press edge
        select_pressed = bool(raw[2])  # SNES_SELECT
        if select_pressed and not _select_state[1]:
            try:
                _select_state[0] ^= 1
                env.unwrapped.data.set_value(
                    "selected_item", _select_state[0]
                )
            except Exception:
                pass
        _select_state[1] = select_pressed

        ram = env.get_ram()
        values = schema.read(ram)
        config.apply_computed(values)
        if not tracker_seeded:
            tracker.update(values)
            tracker_seeded = True
        progress = tracker.update(values)
        if progress > best_progress:
            best_progress = progress

    _ROOM_NAMES = {
        0x91F8: 'Landing Site', 0x92FD: 'Parlor', 0x9879: 'Flyway',
        0x9804: 'Bomb Torizo', 0x96BA: 'Climb', 0x975C: 'Pit Room',
        0x9AD9: 'BB Elevator', 0x9E9F: 'Morph Ball Room',
        0x9F11: 'Construction Zone', 0x9E52: 'First Missile Room',
        0xA011: 'BB E-Tank', 0x962A: 'Terminator Room',
        0x99BD: 'Green Pirates Shaft', 0x9BC8: 'Lower Mushrooms',
        0x9938: 'Green Brinstar Elev', 0x9F64: 'GB Main Shaft',
        0x9FBA: 'GB Fireflea', 0x9FC5: 'GB Missile Refill',
    }

    def on_hud(info):
        ram = env.get_ram()
        values = schema.read(ram)
        config.apply_computed(values)
        level_id = values.get("level_id", 0)
        health = values.get("health")
        lives = values.get("lives", "?")
        stat = f"hp={health}" if health is not None else f"lives={lives}"
        lines = [
            f"REC {len(recorded_actions)} frames | progress={best_progress:.0f}",
            f"{stat} | {_ROOM_NAMES.get(level_id, f'0x{level_id:04X}')}",
        ]
        # Show weapon status if SM (has selected_item in data.json)
        try:
            sel = env.unwrapped.data.lookup_value("selected_item")
            missiles = env.unwrapped.data.lookup_value("missiles")
            weapon = "MISSILES" if sel else "BEAM"
            lines.append(f"weapon={weapon} | missiles={missiles}")
        except Exception:
            pass
        return lines

    # Intercept raw actions before they go to the env
    last_raw_action = [0] * 12
    _orig_gather = PlaySession._gather_action

    def patched_gather(self, pg, keyboard_action, controller_action, sanitize_action):
        nonlocal last_raw_action
        action = _orig_gather(self, pg, keyboard_action, controller_action, sanitize_action)
        last_raw_action = list(action) if hasattr(action, '__iter__') else [0] * 12
        return action

    PlaySession._gather_action = patched_gather

    # Checkpoint recording state: map slot -> frame count at save time
    _checkpoint_frames: dict[int, int] = {}

    def on_key_down(key):
        import pygame as pg
        nonlocal best_progress, tracker_seeded

        _SLOT_KEYS = {pg.K_F1: 1, pg.K_F2: 2, pg.K_F3: 3, pg.K_F4: 4}

        if key in _SLOT_KEYS:
            slot = _SLOT_KEYS[key]
            mods = pg.key.get_mods()
            if mods & pg.KMOD_SHIFT:
                # Load checkpoint → truncate recording to that frame
                frame = session.load_checkpoint(slot)
                if frame is not None and slot in _checkpoint_frames:
                    rec_frame = _checkpoint_frames[slot]
                    del recorded_actions[rec_frame:]
                    del recorded_raw[rec_frame:]
                    # Reset progress tracker from truncated recording
                    tracker.reset()
                    tracker_seeded = False
                    best_progress = 0.0
                    print(f"  Recording truncated to {rec_frame} frames")
            else:
                # Save checkpoint → record frame position
                _checkpoint_frames[slot] = len(recorded_actions)
                session.save_checkpoint(slot)
            return True

        if key == pg.K_r:
            # Restart: reset env and clear recording
            recorded_actions.clear()
            recorded_raw.clear()
            tracker.reset()
            tracker_seeded = False
            best_progress = 0.0
            print("[RESTART] recording cleared")
            return False  # let PlaySession handle the env.reset()

        return False

    session = PlaySession(
        env,
        game_dir=str(config.game_dir),
        game=config.game_name,
        scale=args.scale,
        title=f"RECORD: {config.display_name}",
    )
    session.on_step = on_step
    session.on_hud = on_hud
    session.on_key_down = on_key_down

    print(f"Recording: {config.display_name}")
    print(f"State: {start_state}")
    print(f"Action table: {len(action_table)} actions")
    print(f"\nControls:")
    print(f"  Arrow keys = D-pad")
    print(f"  Z = B    X = A    A = Y    S = X")
    print(f"  F1-F4 = save checkpoint    Shift+F1-F4 = load checkpoint")
    print(f"  F5 = export state to disk  R = restart & clear recording")
    print(f"  TAB = turbo    ESC = stop & save")
    print(f"  Controller also supported\n")

    try:
        session.run()
    finally:
        # Restore original method
        PlaySession._gather_action = _orig_gather

    if not recorded_actions:
        print("No frames recorded.")
        return

    # Save recording
    output_dir = config.runs_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-increment filename
    existing = sorted(output_dir.glob("recording_*.json"))
    next_idx = 0
    for p in existing:
        try:
            n = int(p.stem.split("_")[1])
            next_idx = max(next_idx, n + 1)
        except (IndexError, ValueError):
            pass

    output_path = output_dir / f"recording_{next_idx:03d}.json"
    metadata = {
        "level": config.level_id,
        "source": "manual_play",
        "best_progress": best_progress,
        "total_frames": len(recorded_actions),
    }
    save_actions(recorded_actions, output_path, metadata=metadata)

    # Also save raw 12-button arrays for faithful replay
    raw_path = output_path.with_name(output_path.stem + "_raw.json")
    import json as _json
    with open(raw_path, "w") as _f:
        _json.dump({
            "raw_buttons": recorded_raw,
            "metadata": metadata,
        }, _f)
    print(f"Raw buttons: {raw_path}")

    print(f"\nRecorded {len(recorded_actions)} frames")
    print(f"Best progress: {best_progress:.0f}")
    print(f"Saved to: {output_path}")
    print(f"\nNext steps:")
    print(f"  Verify:    uv run python -m platformer_common -l {config.level_id} verify --actions {output_path}")
    print(f"  Hillclimb: uv run python -m platformer_common -l {config.level_id} hillclimb --seed {output_path}")


def cmd_selftest(args: argparse.Namespace) -> None:
    """Self-test: verify death detection and level-change guards work correctly."""
    import numpy as np

    config = _resolve_config(args)
    print(f"=== Platformer Optimizer Self-Test: {config.display_name} ===\n")
    failures = 0

    evaluator = Evaluator(config)
    evaluator._ensure_env()

    initial_cam = evaluator._initial_camera_x
    initial_values = evaluator._initial_values
    print(f"State: {config.start_state}")
    print(f"  initial_camera_x = {initial_cam:.0f}")
    print(f"  initial_lives    = {initial_values.get('lives', 'N/A')}")

    # Check level_id is correct
    level_id = initial_values.get("level_id", -1)
    if level_id != config.target_level_id:
        print(f"  FAIL: level_id=0x{level_id:04X}, expected 0x{config.target_level_id:04X}")
        failures += 1
    else:
        print(f"  OK: level_id=0x{level_id:04X}")

    # Test 1: sequence that dies must be flagged as died, NOT completed
    print(f"\n[Test 1] Sprint-jump dies -> detected as death, never completion")
    death_seq = ([2] * 40 + [3] * 15 + [2] * 5 + [5] * 10) * 28
    result = evaluator.evaluate(death_seq[:2000], early_terminate=False)
    if not result.died:
        print(f"  FAIL: died={result.died}, expected True")
        failures += 1
    elif result.completed:
        print(f"  FAIL: completed={result.completed}, should be False when died")
        failures += 1
    else:
        print(f"  OK: died=True, completed=False, frame={result.total_frames}, progress={result.max_progress:.0f}")

    # Test 2: fitness for dead < alive at same progress
    print(f"\n[Test 2] Death fitness < alive fitness at same progress")
    dead_fitness_at_100 = 100 * config.progress_weight - config.death_penalty
    alive_fitness_at_100 = 100 * config.progress_weight
    if dead_fitness_at_100 >= alive_fitness_at_100:
        print(f"  FAIL: dead_fitness ({dead_fitness_at_100}) >= alive_fitness ({alive_fitness_at_100})")
        failures += 1
    else:
        print(f"  OK: dead@100={dead_fitness_at_100} < alive@100={alive_fitness_at_100}")

    # Test 3: short alive sequence stays in level
    print(f"\n[Test 3] Short alive sequence stays in level")
    short_result = evaluator.evaluate([0] * 60, early_terminate=False)
    if short_result.completed:
        print(f"  FAIL: 60 frames of nothing showed completed=True!")
        failures += 1
    elif short_result.died:
        print(f"  FAIL: 60 frames of nothing showed died=True!")
        failures += 1
    else:
        print(f"  OK: alive, not completed, level_id=0x{short_result.level_id_at_end:04X}")

    # Test 4: determinism
    print(f"\n[Test 4] Determinism check")
    r1 = evaluator.evaluate(death_seq[:500], early_terminate=False)
    r2 = evaluator.evaluate(death_seq[:500], early_terminate=False)
    if r1.fitness != r2.fitness or r1.total_frames != r2.total_frames:
        print(f"  FAIL: run1 fitness={r1.fitness:.0f}/frames={r1.total_frames} != run2")
        failures += 1
    else:
        print(f"  OK: both runs -> fitness={r1.fitness:.0f}, frames={r1.total_frames}")

    # Test 5: first-frame button stability
    print(f"\n[Test 5] First-frame button stability")
    action_table = _get_action_table(config)
    for idx in range(len(action_table)):
        r = evaluator.evaluate([idx], early_terminate=False)
        if r.completed:
            print(f"  FAIL: action {idx} on first frame triggered completion!")
            failures += 1
            break
    else:
        print(f"  OK: all {len(action_table)} actions stable on first frame")

    evaluator.close()

    print(f"\n{'=' * 40}")
    if failures == 0:
        print("ALL TESTS PASSED")
    else:
        print(f"{failures} TEST(S) FAILED")
    return failures


def cmd_trace_map(args: argparse.Namespace) -> None:
    """Render a position trace overlaid on an area map PNG."""
    from super_metroid_rl.navigation.trace_renderer import (
        render_trace_on_map,
        detect_area,
        _load_nodes,
        DEFAULT_EXPORT_DIR,
    )

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

    # Auto-detect or use specified area
    area = getattr(args, "area", None)
    if not area:
        trace_data = json.loads(trace_path.read_text())
        export_dir = Path(getattr(args, "map_dir", None) or DEFAULT_EXPORT_DIR)
        nodes = _load_nodes(export_dir)
        area = detect_area(trace_data, nodes)
        if not area:
            print("Error: could not auto-detect area. Specify --area.")
            return
        print(f"Auto-detected area: {area}")

    output = Path(args.output) if getattr(args, "output", None) else trace_path.with_suffix(".png")
    map_dir = Path(args.map_dir) if getattr(args, "map_dir", None) else None

    render_trace_on_map(
        trace_path=trace_path,
        area_name=area,
        output_path=output,
        map_dir=map_dir,
    )


# -- Main CLI ----------------------------------------------------------------


def main(default_level: str | None = None) -> None:
    """Build and run the CLI parser.

    Args:
        default_level: If set, use this level when --level is omitted.
            Used by game-specific wrappers (e.g., DKC optimizer).
    """
    parser = argparse.ArgumentParser(
        description="Platformer Speedrun Optimizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Global --level flag
    level_required = default_level is None
    parser.add_argument(
        "--level", "-l",
        default=default_level,
        required=False,
        help=f"Level ID or alias (default: {default_level or 'required'})",
    )

    sub = parser.add_subparsers(dest="command")

    # list-levels
    sub.add_parser("list-levels", help="List all registered levels")

    # extract
    p_extract = sub.add_parser("extract", help="Extract actions from a bk2 recording")
    p_extract.add_argument("--bk2", required=True, help="Path to bk2 file")
    p_extract.add_argument("--output", "-o", help="Output JSON path")
    p_extract.add_argument("--raw-preview", action="store_true")

    # extract-all
    p_extract_all = sub.add_parser("extract-all", help="Extract and evaluate all bk2 recordings")
    p_extract_all.add_argument("--recordings-dir", help="Recordings directory")

    # verify
    p_verify = sub.add_parser("verify", help="Verify action sequence via headless replay")
    p_verify.add_argument("--actions", required=True, help="Path to actions JSON")
    p_verify.add_argument("--trace", action="store_true", help="Log all level_id changes")
    p_verify.add_argument("--state", help="Override start state")

    # optimize
    p_optimize = sub.add_parser("optimize", help="Run GA optimization")
    p_optimize.add_argument("--seed", required=True, help="Path to seed actions JSON")
    p_optimize.add_argument("--generations", type=int, default=None)
    p_optimize.add_argument("--population", type=int, default=None)
    p_optimize.add_argument("--output-dir", help="Output directory")
    p_optimize.add_argument("--workers", type=int, default=1, help="Parallel workers")
    p_optimize.add_argument("--resume", help="Resume from checkpoint JSON")
    p_optimize.add_argument("--render", type=int, nargs="?", const=1, default=0,
                            metavar="N", help="Render best every N gens (default: every gen)")
    p_optimize.add_argument("--state", help="Override start state")

    # hillclimb
    p_hill = sub.add_parser("hillclimb", help="Run hill climbing refinement")
    p_hill.add_argument("--seed", required=True, help="Path to seed actions JSON")
    p_hill.add_argument("--iterations", type=int, default=5000)
    p_hill.add_argument("--output-dir", help="Output directory")
    p_hill.add_argument("--render", type=int, nargs="?", const=100, default=0,
                        metavar="N", help="Render best every N iterations (default: every 100)")
    p_hill.add_argument("--scale", type=int, default=3, help="Render scale")
    p_hill.add_argument("--state", help="Override start state")

    # hillclimb-raw (raw button mutation, no lossy action-index conversion)
    p_hraw = sub.add_parser("hillclimb-raw", help="Hill climb with raw button mutation")
    p_hraw.add_argument("--seed", required=True, help="Path to seed JSON with raw_buttons")
    p_hraw.add_argument("--iterations", type=int, default=1000)
    p_hraw.add_argument("--output-dir", help="Output directory")
    p_hraw.add_argument("--state", help="Override start state")

    # watch
    p_watch = sub.add_parser("watch", help="Watch action sequence visually")
    p_watch.add_argument("--actions", required=True, help="Path to actions JSON")
    p_watch.add_argument("--scale", type=int, default=3)
    p_watch.add_argument("--state", help="Override start state")

    # watch-bk2
    p_watch_bk2 = sub.add_parser("watch-bk2", help="Replay a bk2 recording visually")
    p_watch_bk2.add_argument("--bk2", required=True, help="Path to bk2 file")
    p_watch_bk2.add_argument("--scale", type=int, default=3)

    # prepare-seeds
    p_seeds = sub.add_parser("prepare-seeds", help="Extract and rank recordings, save top N as seeds")
    p_seeds.add_argument("--recordings-dir", help="Recordings directory")
    p_seeds.add_argument("--top", type=int, default=5, help="Number of top seeds to save")

    # auto-state
    p_auto = sub.add_parser("auto-state", help="Create save state via scripted navigation")
    p_auto.add_argument("--from-state", required=True, help="Starting state name")
    p_auto.add_argument("--nav", required=True, help="Navigation steps: 'BUTTON:hold:wait ...'")
    p_auto.add_argument("--settle", type=int, default=30, help="Extra NOOP frames after nav (default: 30)")
    p_auto.add_argument("--screenshot", action="store_true", help="Save screenshot for verification")

    # play (record)
    p_play = sub.add_parser("play", help="Play a level manually and record inputs")
    p_play.add_argument("--scale", type=int, default=3)
    p_play.add_argument("--state", help="Override start state (e.g. ResumeRun)")

    # selftest
    sub.add_parser("selftest", help="Run self-tests")

    # trace-map
    p_trace = sub.add_parser("trace-map", help="Render position trace on area map")
    p_trace.add_argument("--trace", help="Path to trace JSON (auto-detected if omitted)")
    p_trace.add_argument("--actions", help="Actions file (to find trace alongside it)")
    p_trace.add_argument("--area", help="Area name: crateria, brinstar, norfair, etc.")
    p_trace.add_argument("-o", "--output", help="Output PNG path")
    p_trace.add_argument("--map-dir", help="Override map PNG directory")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Validate --level is provided for commands that need it
    needs_level = args.command not in ("list-levels",)
    if needs_level and not args.level:
        parser.error(f"--level is required for '{args.command}'. Use 'list-levels' to see available levels.")

    commands = {
        "list-levels": cmd_list_levels,
        "extract": cmd_extract,
        "extract-all": cmd_extract_all,
        "verify": cmd_verify,
        "optimize": cmd_optimize,
        "hillclimb": cmd_hillclimb,
        "hillclimb-raw": cmd_hillclimb_raw,
        "watch": cmd_watch,
        "watch-bk2": cmd_watch_bk2,
        "prepare-seeds": cmd_prepare_seeds,
        "auto-state": cmd_auto_state,
        "play": cmd_play,
        "selftest": cmd_selftest,
        "trace-map": cmd_trace_map,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
