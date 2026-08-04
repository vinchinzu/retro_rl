"""Practice mode: auto-reset on death, save all attempts."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from retro_harness.platformer.actions import buttons_to_action_index
from retro_harness.platformer.bk2_extract import save_actions
from retro_harness.platformer.cli.helpers import (
    _get_action_table,
    _parse_room_id_arg,
    _resolve_config,
)
from retro_harness.platformer.level_config import LevelConfig


def _practice_completion_token(
    config: LevelConfig,
    values: dict[str, int],
    progress: float,
) -> tuple[str, int] | None:
    """Return a stable token while the configured completion signal is active."""
    if progress < config.completion_min_progress:
        return None
    if config.completion_signal == "ram_flag":
        key = config.completion_ram_key
        value = values.get(key) if key else None
        if value == config.completion_ram_value:
            return ("ram_flag", int(value))
        return None

    level_id = int(values.get("level_id", 0) or 0)
    main_level_ids = {config.target_level_id, *config.level_id_aliases}
    if level_id == 0 or level_id in main_level_ids:
        return None
    if config.completion_level_ids and level_id not in config.completion_level_ids:
        return None
    if level_id in config.completion_exclude_ids:
        return None
    return ("level_id", level_id)


def _load_practice_pb_frames(practice_dir: Path) -> int | None:
    """Load the fastest completed attempt from an existing practice directory."""
    best: int | None = None
    for path in practice_dir.glob("attempt_*.json"):
        if path.stem.endswith("_raw"):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            metadata = data.get("metadata", {})
            if not metadata.get("completed"):
                continue
            frames = int(metadata.get("total_frames", data.get("num_frames", 0)))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        if frames > 0 and (best is None or frames < best):
            best = frames
    return best


def _best_practice_attempt(attempts: list[dict]) -> int:
    """Choose fastest completion, falling back to furthest partial attempt."""
    completed = [attempt for attempt in attempts if attempt["completed"]]
    if completed:
        return min(completed, key=lambda attempt: attempt["frames"])["attempt"]
    if attempts:
        return max(attempts, key=lambda attempt: attempt["max_progress"])["attempt"]
    return -1



def cmd_practice(args: argparse.Namespace) -> None:
    """Practice a level with auto-reset on death, saving all attempts."""
    # Prefer runner-module bindings so tests can monkeypatch runner._resolve_config.
    try:
        from retro_harness.platformer import runner as _runner_mod

        resolve_config = _runner_mod._resolve_config
        get_action_table = _runner_mod._get_action_table
    except Exception:  # pragma: no cover - import cycle / partial init
        resolve_config = _resolve_config
        get_action_table = _get_action_table

    config = resolve_config(args)
    action_table = get_action_table(config)
    start_state = getattr(args, "state", None) or config.start_state
    keep_playing = bool(getattr(args, "keep_playing", False))
    until_room_arg = getattr(args, "until_room", None)
    until_room = (
        _parse_room_id_arg(until_room_arg) if until_room_arg is not None else None
    )
    until_playable = bool(getattr(args, "until_playable", False))
    until_label = getattr(args, "until_label", None)
    keep_playing = keep_playing or until_room is not None
    room_debounce = max(1, int(getattr(args, "room_debounce", 3)))

    from retro_harness.env import make_env
    from retro_harness.play_session import PlaySession
    from retro_harness.platformer.progress import make_progress_tracker

    env = make_env(
        game=config.game_name,
        state=start_state,
        game_dir=config.game_dir,
        render_mode="rgb_array",
    )

    schema = config.ram_schema

    output_dir = getattr(args, "output_dir", None)
    practice_dir = Path(output_dir).expanduser() if output_dir else config.runs_dir / "practice"
    practice_dir.mkdir(parents=True, exist_ok=True)
    session_label = getattr(args, "session_label", None) or config.level_id

    # Auto-number from highest existing attempt
    existing = sorted(practice_dir.glob("attempt_*.json"))
    next_attempt = 0
    for p in existing:
        try:
            stem = p.stem
            if "_raw" in stem:
                continue
            n = int(stem.split("_")[1])
            next_attempt = max(next_attempt, n + 1)
        except (IndexError, ValueError):
            pass

    # Per-attempt state
    tracker = make_progress_tracker(config)
    tracker.reset()
    recorded_actions: list[int] = []
    recorded_raw: list[list[int]] = []
    best_progress = 0.0
    tracker_seeded = False

    # Session-wide stats
    attempt_num = next_attempt
    all_attempts: list[dict] = []
    session_best_progress = 0.0
    completion_pb_frames = _load_practice_pb_frames(practice_dir)
    result_flash = 0
    result_message = ""
    completion_candidate: tuple[str, int] | None = None
    completion_candidate_frames = 0
    discard_current = False
    split_message = ""
    split_flash = 0

    # Cache emulator state for instant reset
    env.reset()
    cached_emu_state = env.em.get_state()
    ram = env.get_ram()
    initial_values = schema.read(ram)
    config.apply_computed(initial_values)
    initial_lives = initial_values.get("lives")
    current_values = dict(initial_values)
    initial_room_id = int(initial_values.get("level_id", 0) or 0)

    # Continuous recording state. Split frame values are action counts, so they
    # are directly usable as exclusive slice boundaries.
    room_splits: list[dict] = []
    stable_room_id = initial_room_id
    segment_start_frame = 0
    room_candidate_id: int | None = None
    room_candidate_frame = 0
    room_candidate_count = 0
    room_candidate_values: dict[str, int] = {}
    attempt_has_input = False
    recording_checkpoints: dict[int, dict] = {}

    # Select toggle workaround
    _select_state = [0, False]

    def _save_attempt(
        completed: bool,
        terminal_reason: str,
        *,
        discard_trivial: bool = False,
    ) -> bool:
        nonlocal attempt_num, completion_pb_frames
        if not recorded_actions:
            return False
        if (
            discard_trivial
            and not attempt_has_input
            and best_progress <= 0
            and not room_splits
        ):
            print(f"  [DISCARD] empty tail attempt {attempt_num}")
            return False

        frame_count = len(recorded_actions)
        previous_pb = completion_pb_frames
        is_pb = completed and (previous_pb is None or frame_count < previous_pb)
        metadata = {
            "level": config.level_id,
            "source": "practice",
            "session_label": session_label,
            "attempt": attempt_num,
            "best_progress": best_progress,
            "total_frames": frame_count,
            "completed": completed,
            "state": start_state,
            "terminal_reason": terminal_reason,
            "start_room_id": initial_room_id,
            "end_room_id": int(current_values.get("level_id", 0) or 0),
            "until_room_id": until_room,
            "until_playable": until_playable,
            "until_label": until_label,
            "room_splits": [dict(split) for split in room_splits],
        }
        # Save action indices
        out_path = practice_dir / f"attempt_{attempt_num:03d}.json"
        save_actions(recorded_actions, out_path, metadata=metadata)

        # Save raw buttons
        raw_path = practice_dir / f"attempt_{attempt_num:03d}_raw.json"
        import json as _json
        with open(raw_path, "w") as _f:
            _json.dump({"raw_buttons": recorded_raw, "metadata": metadata}, _f)

        all_attempts.append({
            "attempt": attempt_num,
            "frames": frame_count,
            "max_progress": best_progress,
            "completed": completed,
            "is_pb": is_pb,
            "terminal_reason": terminal_reason,
            "room_splits": len(room_splits),
        })
        if is_pb:
            completion_pb_frames = frame_count
        attempt_num += 1
        return True

    def _observe_room(values: dict[str, int]) -> int | None:
        nonlocal stable_room_id, segment_start_frame, room_candidate_id
        nonlocal room_candidate_frame, room_candidate_count, room_candidate_values

        room_id = int(values.get("level_id", 0) or 0)
        if room_id == 0:
            return None
        if room_id == stable_room_id:
            room_candidate_id = None
            room_candidate_count = 0
            room_candidate_values = {}
            return None
        if room_id != room_candidate_id:
            room_candidate_id = room_id
            room_candidate_frame = len(recorded_raw)
            room_candidate_count = 1
            room_candidate_values = dict(values)
        else:
            room_candidate_count += 1
        if room_candidate_count < room_debounce:
            return None

        split = {
            "from_room_id": stable_room_id,
            "room_id": room_id,
            "frame": room_candidate_frame,
            "segment_start_frame": segment_start_frame,
            "segment_frames": room_candidate_frame - segment_start_frame,
        }
        for source, dest in (
            ("player_x", "x"),
            ("player_y", "y"),
            ("health", "health"),
            ("max_health", "max_health"),
            ("missiles", "missiles"),
            ("max_missiles", "max_missiles"),
            ("super_missiles", "super_missiles"),
            ("max_super_missiles", "max_super_missiles"),
            ("game_state", "game_state"),
            ("door_transition", "door_transition"),
        ):
            if source in room_candidate_values:
                split[dest] = int(room_candidate_values[source])
        split["configured_completion"] = (
            _practice_completion_token(config, room_candidate_values, best_progress)
            is not None
        )
        room_splits.append(split)
        stable_room_id = room_id
        segment_start_frame = room_candidate_frame
        room_candidate_id = None
        room_candidate_count = 0
        room_candidate_values = {}
        return room_id

    def _reset_attempt() -> None:
        nonlocal best_progress, tracker_seeded, tracker, result_flash
        nonlocal result_message, completion_candidate, completion_candidate_frames
        nonlocal discard_current, current_values
        nonlocal stable_room_id, segment_start_frame, room_candidate_id
        nonlocal room_candidate_frame, room_candidate_count, room_candidate_values
        nonlocal attempt_has_input, split_message, split_flash
        env.em.set_state(cached_emu_state)
        tracker = make_progress_tracker(config)
        tracker.reset()
        tracker_seeded = False
        recorded_actions.clear()
        recorded_raw.clear()
        best_progress = 0.0
        result_flash = 0
        result_message = ""
        completion_candidate = None
        completion_candidate_frames = 0
        discard_current = False
        current_values = dict(initial_values)
        _select_state[:] = [0, False]
        room_splits.clear()
        stable_room_id = initial_room_id
        segment_start_frame = 0
        room_candidate_id = None
        room_candidate_frame = 0
        room_candidate_count = 0
        room_candidate_values = {}
        attempt_has_input = False
        split_message = ""
        split_flash = 0
        recording_checkpoints.clear()

    def on_step(obs, reward, done, info):
        nonlocal best_progress, tracker_seeded, result_flash, result_message
        nonlocal session_best_progress, completion_candidate
        nonlocal completion_candidate_frames, current_values
        nonlocal attempt_has_input, split_message, split_flash

        if result_flash > 0:
            result_flash -= 1
            if result_flash == 0:
                _reset_attempt()
            return
        if split_flash > 0:
            split_flash -= 1

        # Record
        raw = session.last_action_post_sanitize
        idx = buttons_to_action_index(raw, action_table=action_table)
        recorded_actions.append(idx)
        recorded_raw.append(raw)
        attempt_has_input = attempt_has_input or any(raw)

        # Select workaround
        select_pressed = bool(raw[2])
        if select_pressed and not _select_state[1]:
            try:
                _select_state[0] ^= 1
                env.unwrapped.data.set_value("selected_item", _select_state[0])
            except Exception:
                pass
        _select_state[1] = select_pressed

        # Track progress
        ram = env.get_ram()
        values = schema.read(ram)
        config.apply_computed(values)
        current_values = values
        if not tracker_seeded:
            tracker.update(values)
            tracker_seeded = True
        progress = tracker.update(values)
        if progress > best_progress:
            best_progress = progress
        if best_progress > session_best_progress:
            session_best_progress = best_progress

        confirmed_room = _observe_room(values)
        if keep_playing:
            if confirmed_room is not None:
                latest_split = room_splits[-1]
                split_message = (
                    f"SPLIT 0x{latest_split['from_room_id']:04X} -> "
                    f"0x{confirmed_room:04X} @ {latest_split['frame']}f"
                )
                split_flash = 120
                print(f"  {split_message}")
            target_split_confirmed = bool(
                until_room is not None
                and room_splits
                and room_splits[-1]["room_id"] == until_room
            )
            game_state = values.get("game_state", info.get("game_state"))
            door_transition = values.get(
                "door_transition",
                info.get("door_transition", 0),
            )
            target_playable = (
                not until_playable
                or (game_state == 8 and not bool(door_transition))
            )
            if target_split_confirmed and target_playable:
                frames = len(recorded_actions)
                previous_pb = completion_pb_frames
                _save_attempt(completed=True, terminal_reason="until_room")
                if previous_pb is None or frames < previous_pb:
                    result_message = f">>> TARGET {frames}f | NEW PB <<<"
                else:
                    result_message = (
                        f">>> TARGET {frames}f | PB +{frames - previous_pb}f <<<"
                    )
                print(f"  {result_message.strip('> <')} attempt {attempt_num - 1}")
                result_flash = 60
                return
        else:
            completion_token = _practice_completion_token(config, values, best_progress)
            if completion_token is None:
                completion_candidate = None
                completion_candidate_frames = 0
            elif completion_token == completion_candidate:
                completion_candidate_frames += 1
            else:
                completion_candidate = completion_token
                completion_candidate_frames = 1

            required_completion_frames = max(1, config.completion_debounce_frames + 1)
            if completion_candidate_frames >= required_completion_frames:
                frames = len(recorded_actions)
                previous_pb = completion_pb_frames
                _save_attempt(completed=True, terminal_reason="configured_completion")
                if previous_pb is None or frames < previous_pb:
                    result_message = f">>> SUCCESS {frames}f | NEW PB <<<"
                else:
                    result_message = (
                        f">>> SUCCESS {frames}f | PB +{frames - previous_pb}f <<<"
                    )
                print(f"  {result_message.strip('> <')} attempt {attempt_num - 1}")
                result_flash = 60
                return

        # Death detection
        lives = values.get("lives")
        gameplay_started = len(recorded_actions) > 30  # small grace period
        if gameplay_started:
            is_dead = False
            for signal in config.death_signals:
                if signal == "lives_drop":
                    if initial_lives is not None and lives is not None and lives < initial_lives:
                        is_dead = True
                elif signal == "health_zero":
                    health = values.get("health", 1)
                    if health <= 0:
                        is_dead = True

            if is_dead:
                print(f"  DIED attempt {attempt_num}: {len(recorded_actions)}f, progress={best_progress:.0f}")
                _save_attempt(completed=False, terminal_reason="death")
                result_message = ">>> DIED <<<  (auto-resetting...)"
                result_flash = 60
                return

        if done:
            _save_attempt(completed=False, terminal_reason="env_done")
            result_message = ">>> EPISODE ENDED <<<  (auto-resetting...)"
            result_flash = 60

    def on_hud(info):
        lines = [
            f"{session_label} | attempt #{attempt_num} | {len(recorded_actions)}f",
            (
                f"start progress={best_progress:.0f} | best={session_best_progress:.0f}"
                if keep_playing
                else f"progress={best_progress:.0f} | best={session_best_progress:.0f}"
            ),
            f"saved: {len(all_attempts)} attempts | splits: {len(room_splits)}",
        ]
        telemetry = []
        room_id = current_values.get("level_id")
        if room_id is not None:
            telemetry.append(f"room=0x{room_id:04X}")
        if "player_x" in current_values and "player_y" in current_values:
            telemetry.append(f"pos=({current_values['player_x']},{current_values['player_y']})")
        if "health" in current_values:
            telemetry.append(f"hp={current_values['health']}")
        if "missiles" in current_values:
            maximum = current_values.get("max_missiles")
            value = current_values["missiles"]
            telemetry.append(
                f"missiles={value}/{maximum}"
                if maximum is not None
                else f"missiles={value}"
            )
        if "super_missiles" in current_values:
            maximum = current_values.get("max_super_missiles")
            value = current_values["super_missiles"]
            telemetry.append(
                f"supers={value}/{maximum}"
                if maximum is not None
                else f"supers={value}"
            )
        if telemetry:
            lines.append(" | ".join(telemetry))
        if completion_pb_frames is not None:
            delta = len(recorded_actions) - completion_pb_frames
            lines.append(f"completed PB={completion_pb_frames}f | delta={delta:+d}f")
        if until_room is not None:
            suffix = " playable" if until_playable else ""
            label = f" ({until_label})" if until_label else ""
            lines.append(f"target room=0x{until_room:04X}{label}{suffix}")
        if split_flash > 0:
            lines.insert(0, split_message)
        if saved_state_path[0]:
            lines.append(f"F5 state: {save_name} (progress={best_progress:.0f})")
        if result_flash > 0:
            lines.insert(0, result_message)
        return lines

    # Name for F5 save state (user can override via --save-name)
    save_name = getattr(args, "save_name", None) or f"Chained_{config.level_id}_practice"
    saved_state_path = [None]  # mutable ref for HUD

    def on_key_down(key):
        nonlocal discard_current
        import pygame as pg
        if key == pg.K_F5:
            # Save persistent .state at current position
            from retro_harness.env import save_state as _save_state
            path = _save_state(env, str(config.game_dir), config.game_name, save_name)
            saved_state_path[0] = path
            print(f"  [STATE SAVED] {save_name} at progress={best_progress:.0f} ({len(recorded_actions)}f)")
            print(f"  -> {path}")
            print(f"  Practice from here: uv run python -m retro_harness.platformer -l {config.level_id} practice --state {save_name}")
            return True
        if key == pg.K_r:
            # Discard current attempt and restart
            print(f"  [DISCARD] attempt {attempt_num}")
            _reset_attempt()
            discard_current = True
            return False  # let PlaySession handle env.reset()
        return False

    def trigger_save(slot: int) -> None:
        recording_checkpoints[slot] = {
            "frame": len(recorded_actions),
            "room_splits": [dict(split) for split in room_splits],
            "stable_room_id": stable_room_id,
            "segment_start_frame": segment_start_frame,
            "best_progress": best_progress,
            "attempt_has_input": attempt_has_input,
            "current_values": dict(current_values),
            "select_state": list(_select_state),
        }
        session.save_checkpoint(slot)

    def trigger_load(slot: int) -> None:
        nonlocal tracker, tracker_seeded, best_progress, stable_room_id
        nonlocal segment_start_frame, room_candidate_id, room_candidate_count
        nonlocal room_candidate_values, attempt_has_input, current_values
        nonlocal completion_candidate, completion_candidate_frames
        nonlocal split_message, split_flash, result_message, result_flash
        checkpoint = recording_checkpoints.get(slot)
        if checkpoint is None:
            print(f"[CHECKPOINT {slot}] no recording checkpoint")
            return
        if session.load_checkpoint(slot) is None:
            return
        frame = checkpoint["frame"]
        del recorded_actions[frame:]
        del recorded_raw[frame:]
        room_splits[:] = [dict(split) for split in checkpoint["room_splits"]]
        stable_room_id = checkpoint["stable_room_id"]
        segment_start_frame = checkpoint["segment_start_frame"]
        best_progress = checkpoint["best_progress"]
        attempt_has_input = checkpoint["attempt_has_input"]
        current_values = dict(checkpoint["current_values"])
        _select_state[:] = checkpoint["select_state"]
        room_candidate_id = None
        room_candidate_count = 0
        room_candidate_values = {}
        completion_candidate = None
        completion_candidate_frames = 0
        split_message = ""
        split_flash = 0
        result_message = ""
        result_flash = 0
        tracker = make_progress_tracker(config)
        tracker.reset()
        tracker_seeded = False
        print(f"  Recording truncated to {frame} frames and {len(room_splits)} splits")

    # Override PlaySession reset so R key uses our cached state
    def on_reset():
        if discard_current:
            env.em.set_state(cached_emu_state)

    session = PlaySession(
        env,
        game_dir=str(config.game_dir),
        game=config.game_name,
        scale=args.scale,
        title=f"PRACTICE: {session_label} | {config.display_name}",
    )
    session.on_step = on_step
    session.on_hud = on_hud
    session.on_key_down = on_key_down
    session.on_reset = on_reset
    session.on_trigger_save = trigger_save
    session.on_trigger_load = trigger_load

    print(f"Practice mode: {config.display_name}")
    print(f"Session: {session_label}")
    print(f"State: {start_state}")
    print(f"Output: {practice_dir}")
    if keep_playing:
        target = f"0x{until_room:04X}" if until_room is not None else "manual stop"
        print(f"Continuous rooms: on (target: {target}, debounce: {room_debounce})")
    print("\nControls:")
    print("  Arrow keys = D-pad    Z = B    X = A    A = Y    S = X")
    print("  F5 = save .state at current position (for later practice)")
    print("  TAB = turbo    R = discard & restart    ESC = save & quit")
    print("  On death or target: auto-saves attempt, resets after 1s\n")

    session.run()

    # Save final attempt if there are unsaved frames
    if recorded_actions and result_flash == 0:
        _save_attempt(
            completed=False,
            terminal_reason="user_exit",
            discard_trivial=True,
        )

    # Write summary
    summary = {
        "level": config.level_id,
        "session_label": session_label,
        "state": start_state,
        "total_attempts": len(all_attempts),
        "best_progress": session_best_progress,
        "best_completion_frames": completion_pb_frames,
        "until_room_id": until_room,
        "until_playable": until_playable,
        "until_label": until_label,
        "keep_playing": keep_playing,
        "attempts": all_attempts,
    }
    summary["best_attempt"] = _best_practice_attempt(all_attempts)
    summary_path = practice_dir / "practice_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n=== Practice Summary ===")
    print(f"Attempts: {len(all_attempts)}")
    print(f"Best progress: {session_best_progress:.0f}")
    if summary["best_attempt"] >= 0:
        best = next(
            attempt
            for attempt in all_attempts
            if attempt["attempt"] == summary["best_attempt"]
        )
        print(f"Best attempt: #{best['attempt']} ({best['frames']}f)")
    print(f"Saved to: {practice_dir}")


