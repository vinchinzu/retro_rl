"""Visual replay with HUD overlay for watch / hillclimb render."""

from __future__ import annotations

import json
from pathlib import Path

from retro_harness.platformer.actions import action_index_to_buttons
from retro_harness.platformer.cli.helpers import _get_action_table


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

    from retro_harness.platformer.progress import make_progress_tracker

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
        px = int(values.get("player_x", 0))
        py = int(values.get("player_y", 0))
        trace_pt: dict = {
            "frame": current_frame,
            "room_id": level_id,
            "x": px,
            "y": py,
            "buttons": btn_str,
        }
        health_val = values.get("health")
        if health_val is not None:
            trace_pt["health"] = int(health_val)
        # Speed: x delta from previous frame (pixels/frame)
        if trace:
            prev_x = trace[-1].get("x", px)
            prev_room = trace[-1].get("room_id", level_id)
            # Only compute speed within same room (avoids door transition spikes)
            if prev_room == level_id:
                trace_pt["speed_x"] = px - prev_x
            else:
                trace_pt["speed_x"] = 0
        else:
            trace_pt["speed_x"] = 0
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
        elif config.completion_signal == "ram_flag":
            flag_val = values.get(config.completion_ram_key, None)
            if (flag_val is not None
                    and flag_val == config.completion_ram_value
                    and max_progress >= config.completion_min_progress):
                print(f"  COMPLETED at frame {current_frame}: {config.completion_ram_key}={flag_val}, progress={max_progress:.0f}")
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


