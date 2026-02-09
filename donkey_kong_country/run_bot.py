#!/usr/bin/env python3
"""Play Donkey Kong Country (SNES) with keyboard/controller via stable-retro."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
import time
import statistics
import re

# Ensure retro_harness is importable
SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import stable_retro as retro
from retro_harness import (
    make_env,
    keyboard_action,
    controller_action,
    sanitize_action,
    sanitize_action_multi,
    SNES_X,
    init_controller,
    init_controllers,
    save_state,
    ensure_gzip_state,
    append_jsonl,
    iter_jsonl,
    find_latest_recording,
    find_latest_recording_from_manifest,
)
from autosplit import LevelStartDetector, read_level_timer_frames

DEFAULT_GAME = "DonkeyKongCountry-Snes"
DEFAULT_STATE = "QuickSave"
DEFAULT_START_OVERRIDE = os.environ.get("RETRO_START_OVERRIDE", "")
DEFAULT_SELECT_OVERRIDE = os.environ.get("RETRO_SELECT_OVERRIDE", "")
DEFAULT_SWAP_XY = os.environ.get("RETRO_SWAP_XY", "")
DEFAULT_PLAYERS = os.environ.get("RETRO_PLAYERS", "")
BEST_TIMES_PATH = SCRIPT_DIR / "best_times.json"
SPLIT_LOG_PATH = SCRIPT_DIR / "split_runs.jsonl"
LEVEL_NAMES_PATH = SCRIPT_DIR / "level_names.json"
DEFAULT_RECORDINGS_DIR = SCRIPT_DIR / "recordings"
DEFAULT_RECORDINGS_MANIFEST = "recordings_manifest.jsonl"
DEFAULT_LEVEL_ID_OFFSET = os.environ.get("RETRO_LEVEL_ID_OFFSET", "")
DEFAULT_LEVEL_ID_CANDIDATES = os.environ.get(
    "RETRO_LEVEL_ID_CANDIDATES",
    "0x3E,0x76,0x256,0x27E,0x286",
)
DEFAULT_LEVEL_ID_VERBOSE = os.environ.get("RETRO_LEVEL_ID_VERBOSE", "")
DEFAULT_INPUT_DEBUG = os.environ.get("RETRO_INPUT_DEBUG", "")
DEFAULT_FORCE_START = os.environ.get("RETRO_FORCE_START", "")
DEFAULT_FORCE_START_FRAMES = os.environ.get("RETRO_FORCE_START_FRAMES", "")

# RAM offsets (7E0000 base)
RAM_LEVEL_ID = 0x0076
RAM_LEVEL_TIMER_FRAMES = 0x0046
RAM_LEVEL_TIMER_MINUTES = 0x0048
RAM_BONUS_TIMER = 0x13F3
RAM_POS_X = 0x00B4
RAM_POS_Y = 0x00B6
RAM_ACTION_DK = 0x10D3
RAM_ACTION_DIDDY = 0x10D5


def _parse_start_overrides(count: int) -> list[Optional[int]]:
    """Parse per-controller Start button overrides.

    Env: RETRO_START_OVERRIDE="0" (all) or "0,7" (per controller).
    """
    if not DEFAULT_START_OVERRIDE:
        return [None for _ in range(count)]
    parts = [p.strip() for p in DEFAULT_START_OVERRIDE.split(",") if p.strip()]
    if not parts:
        return [None for _ in range(count)]
    if len(parts) == 1:
        try:
            btn = int(parts[0])
        except ValueError:
            return [None for _ in range(count)]
        return [btn for _ in range(count)]
    overrides: list[Optional[int]] = [None for _ in range(count)]
    for i in range(min(count, len(parts))):
        try:
            overrides[i] = int(parts[i])
        except ValueError:
            overrides[i] = None
    return overrides


def _parse_select_overrides(count: int) -> list[Optional[int]]:
    """Parse per-controller Select button overrides.

    Env: RETRO_SELECT_OVERRIDE="0" (all) or "0,7" (per controller).
    """
    if not DEFAULT_SELECT_OVERRIDE:
        return [None for _ in range(count)]
    parts = [p.strip() for p in DEFAULT_SELECT_OVERRIDE.split(",") if p.strip()]
    if not parts:
        return [None for _ in range(count)]
    if len(parts) == 1:
        try:
            btn = int(parts[0])
        except ValueError:
            return [None for _ in range(count)]
        return [btn for _ in range(count)]
    overrides: list[Optional[int]] = [None for _ in range(count)]
    for i in range(min(count, len(parts))):
        try:
            overrides[i] = int(parts[i])
        except ValueError:
            overrides[i] = None
    return overrides


def _parse_swap_xy(count: int) -> list[bool]:
    """Parse per-controller X/Y swap flags.

    Env: RETRO_SWAP_XY="1" (all) or "0,1" (per controller).
    """
    if not DEFAULT_SWAP_XY:
        return [False] + [True for _ in range(max(0, count - 1))]
    parts = [p.strip() for p in DEFAULT_SWAP_XY.split(",") if p.strip()]
    if not parts:
        return [False for _ in range(count)]
    if len(parts) == 1:
        return [parts[0] in ("1", "true", "yes", "on") for _ in range(count)]
    flags = [False for _ in range(count)]
    for i in range(min(count, len(parts))):
        flags[i] = parts[i] in ("1", "true", "yes", "on")
    return flags


def _parse_players(default_players: int) -> int:
    if not DEFAULT_PLAYERS:
        return default_players
    text = DEFAULT_PLAYERS.strip().lower()
    if not text:
        return default_players
    try:
        value = int(text, 10)
    except ValueError:
        return default_players
    return max(1, min(2, value))


def _parse_flag(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    text = value.strip().lower()
    if not text:
        return default
    return text in ("1", "true", "yes", "on")


def _env_button_index(env, name: str) -> Optional[int]:
    buttons = getattr(env, "buttons", None)
    if not buttons:
        return None
    target = name.strip().upper()
    for idx, btn in enumerate(buttons):
        if str(btn).upper() == target:
            return idx
    return None


def _parse_int(value: str, default: int) -> int:
    if value is None:
        return default
    text = value.strip().lower()
    if not text:
        return default
    try:
        return int(text, 10)
    except ValueError:
        return default


def _parse_level_id_offset() -> int:
    if not DEFAULT_LEVEL_ID_OFFSET:
        return RAM_LEVEL_ID
    value = DEFAULT_LEVEL_ID_OFFSET.strip().lower()
    try:
        return int(value, 16) if value.startswith("0x") else int(value, 10)
    except ValueError:
        return RAM_LEVEL_ID


def _parse_level_id_candidates(primary_offset: int) -> list[int]:
    parts = [p.strip() for p in DEFAULT_LEVEL_ID_CANDIDATES.split(",") if p.strip()]
    offsets: list[int] = []
    for part in parts:
        try:
            offsets.append(int(part, 16) if part.lower().startswith("0x") else int(part, 10))
        except ValueError:
            continue
    if primary_offset not in offsets:
        offsets.insert(0, primary_offset)
    seen = set()
    unique: list[int] = []
    for off in offsets:
        if off in seen:
            continue
        seen.add(off)
        unique.append(off)
    return unique


def _swap_xy_map() -> dict[int, int]:
    """Return controller map with X/Y swapped."""
    return {
        0: 0,   # A -> B
        1: 8,   # B -> A
        2: 9,   # X -> SNES X (swap)
        3: 1,   # Y -> SNES Y (swap)
        4: 10,  # LB -> L
        5: 11,  # RB -> R
        6: 2,   # Back -> Select
        7: 3,   # Start -> Start
    }


def _axis_map_for_controller(index: int) -> Optional[dict[int, int]]:
    """Default axis-to-button mapping for misreported controllers."""
    if index == 1:
        # Some Xbox pads report Y on triggers; map triggers to SNES X (after swap).
        return {2: SNES_X, 5: SNES_X}
    return None


def _ensure_pygame():
    try:
        import pygame
    except ImportError:
        print("Error: pygame not installed")
        print("Run: cd .. && ./setup.sh")
        sys.exit(1)
    return pygame


def _load_best_times(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {str(k): float(v) for k, v in data.items()}
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    return {}


def _save_best_times(path: Path, data: dict[str, float]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _format_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    sec = seconds - minutes * 60
    return f"{minutes:02d}:{sec:05.2f}"


def _append_run_log(
    path: Path,
    *,
    event: str,
    session_id: str,
    game: str,
    state: str,
    level_id: int | None,
    level_name: str | None,
    elapsed_seconds: float | None,
    elapsed_frames: int | None,
    best_seconds: float | None,
    timer_start: int | None,
    timer_end: int | None,
    frame_start: int | None,
    frame_end: int | None,
    pos_x: int | None,
    pos_y: int | None,
    reason: str | None = None,
) -> None:
    entry = {
        "event": event,
        "run_id": session_id,
        "session_id": session_id,
        "timestamp": time.time(),
        "game": game,
        "state": state,
        "level_id": level_id,
        "level_name": level_name,
        "elapsed_seconds": elapsed_seconds,
        "elapsed_frames": elapsed_frames,
        "best_seconds": best_seconds,
        "timer_start": timer_start,
        "timer_end": timer_end,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "pos_x": pos_x,
        "pos_y": pos_y,
        "reason": reason,
    }
    append_jsonl(path, entry)


def _infer_level_name_from_state(state: str) -> str | None:
    if "JungleHijinks" in state or "JungleHijinx" in state:
        return "Jungle Hijinks"
    return None


def _split_camel_case(value: str) -> str:
    spaced = re.sub(r"(?<!^)(?=[A-Z])", " ", value)
    return spaced.replace("_", " ").strip()


def _extract_level_name_from_state_name(state: str) -> str | None:
    parts = state.replace("\\", "/").split("/")[-1].split(".")
    if not parts:
        return None
    level_name = None
    for part in parts:
        if part.lower().startswith("level"):
            break
        level_name = part
    if not level_name:
        return None
    return _split_camel_case(level_name)


def _load_level_names(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    names: dict[int, str] = {}
    for key, value in data.items():
        if not isinstance(value, str):
            continue
        level_id: int | None = None
        if isinstance(key, int):
            level_id = key
        elif isinstance(key, str):
            try:
                level_id = int(key, 16) if key.lower().startswith("0x") else int(key, 10)
            except ValueError:
                level_id = None
        if level_id is not None:
            names[level_id] = value
    return names


def _save_level_names(path: Path, names: dict[int, str]) -> None:
    data = {f"0x{level_id:02X}": name for level_id, name in sorted(names.items())}
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _bootstrap_level_names_from_states(
    *,
    game: str,
    game_dir: Path,
    level_names_path: Path,
) -> dict[int, str]:
    from retro_harness import get_available_states

    states = get_available_states(game, game_dir)
    if not states:
        return _load_level_names(level_names_path)
    names = _load_level_names(level_names_path)
    for state in states:
        if state.lower() == "quicksave":
            continue
        name = _extract_level_name_from_state_name(state)
        if not name:
            continue
        try:
            env = make_env(game=game, state=state, game_dir=game_dir)
            obs, info = env.reset()
            ram = env.get_ram()
            level_id = int(ram[RAM_LEVEL_ID]) if len(ram) > RAM_LEVEL_ID else None
            env.close()
        except Exception:
            continue
        if level_id is None:
            continue
        if level_id not in names:
            names[level_id] = name
    if names:
        _save_level_names(level_names_path, names)
    return names


def play_game(
    game: str,
    state: str,
    scale: int = 3,
    autosplit: bool = False,
    record_dir: str | None = None,
    level_log_every: int = 0,
    level_id_offset: int | None = None,
    practice: bool = False,
    label: str | None = None,
):
    pygame = _ensure_pygame()
    pygame.init()

    print(f"Loading {game} from state: {state}")

    controllers = init_controllers(pygame)
    if controllers:
        for idx, joy in enumerate(controllers):
            print(
                f"Controller {idx}: {joy.get_name()} "
                f"(buttons={joy.get_numbuttons()} axes={joy.get_numaxes()} hats={joy.get_numhats()})"
            )
    else:
        print("Controller: none detected")
    joystick = controllers[0] if controllers else None
    players = _parse_players(2 if len(controllers) >= 2 else 1)
    start_overrides = _parse_start_overrides(len(controllers))
    select_overrides = _parse_select_overrides(len(controllers))
    swap_xy = _parse_swap_xy(len(controllers))
    last_pressed = [-1, -1]
    level_id_verbose = _parse_flag(DEFAULT_LEVEL_ID_VERBOSE, default=False)

    session_id = str(int(time.time()))
    ensure_gzip_state(SCRIPT_DIR, game, state)
    record_path = None
    if record_dir:
        record_path = Path(record_dir)
        if not record_path.is_absolute():
            record_path = SCRIPT_DIR / record_path
        record_path = record_path / session_id
        record_path.mkdir(parents=True, exist_ok=True)

    env_kwargs = {"players": players}
    if record_path:
        env_kwargs["record"] = str(record_path)
    env_kwargs["use_restricted_actions"] = retro.Actions.ALL

    env = make_env(
        game=game,
        state=state,
        game_dir=SCRIPT_DIR,
        **env_kwargs,
    )
    obs, info = env.reset()
    print(f"Action space size: {env.action_space.shape[0]}")
    input_debug = _parse_flag(DEFAULT_INPUT_DEBUG, default=False)
    force_start = _parse_flag(DEFAULT_FORCE_START, default=False)
    force_start_frames = max(1, _parse_int(DEFAULT_FORCE_START_FRAMES, 30))
    start_action_index = _env_button_index(env, "START")
    select_action_index = _env_button_index(env, "SELECT")
    if input_debug:
        buttons = getattr(env, "buttons", None)
        if buttons:
            print(f"[INPUT] buttons: {buttons}")
        print(f"[INPUT] start_index={start_action_index} select_index={select_action_index}")
    if len(controllers) >= 2 and env.action_space.shape[0] < 24:
        print("[WARN] Detected 2 controllers, but action space is not 24. P2 inputs may map to P1.")

    width, height = obs.shape[1], obs.shape[0]
    screen = pygame.display.set_mode((width * scale, height * scale), pygame.SWSURFACE)
    pygame.display.set_caption(f"Stable-Retro: {game}")
    font = pygame.font.SysFont("monospace", 14)

    clock = pygame.time.Clock()
    running = True
    total_reward = 0.0
    speed_levels = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
    speed_idx = speed_levels.index(1.0)
    fast_forward_factor = 2.0
    frame_idx = 0
    best_times = _load_best_times(BEST_TIMES_PATH) if autosplit else {}
    current_level_id: Optional[int] = None
    segment_level_id: Optional[int] = None
    level_start_frame = 0
    segment_start_frame = 0
    segment_start_timer: Optional[int] = None
    segment_elapsed_offset = 0
    level_active = False
    last_level_timer_frames: Optional[int] = None
    last_level_id: Optional[int] = None
    last_level_log_frame = -1
    start_detector = LevelStartDetector(min_moving_frames=2)
    last_pos: Optional[tuple[int, int]] = None
    last_bonus_active = False
    level_names: dict[int, str] = _load_level_names(LEVEL_NAMES_PATH)
    record_start_level_id: Optional[int] = None
    record_start_level_name: Optional[str] = None
    record_start_time = time.time()
    record_last_mtime: Optional[float] = None
    manifest_path = record_path / DEFAULT_RECORDINGS_MANIFEST if record_path else None
    level_id_offset = level_id_offset if level_id_offset is not None else _parse_level_id_offset()
    level_id_candidates = _parse_level_id_candidates(level_id_offset)
    last_level_id_candidates: dict[int, int] = {}

    # Practice mode: working state for quick reload
    working_state: bytes | None = None
    working_state_level: int = 0
    working_state_name: str = ""

    # Labeled checkpoints for practice mode
    practice_recorder = None
    if practice:
        from retro_harness import LabeledRecorder
        ram = env.get_ram()
        init_level = int(ram[level_id_offset]) if len(ram) > level_id_offset else 0
        practice_label = label or f"level_{init_level}"
        practice_recorder = LabeledRecorder(
            game=game,
            game_dir=SCRIPT_DIR,
            label=practice_label,
        )
        practice_recorder.start(env)
        print(f"\n[PRACTICE] Label: {practice_label}")
        print("  F5: Save working state | F7/F8/Backspace: Reload | L3+LB: Reload")
        print("  F6: Labeled checkpoint")

    def reload_working_state():
        """Reload the F5 working state."""
        nonlocal obs, info, frame_idx, working_state, working_state_level, working_state_name
        if working_state:
            env.em.set_state(working_state)
            obs, info = env.reset()
            frame_idx = 0
            print(f"\n[RELOAD] {working_state_name} (level {working_state_level})")
            return True
        else:
            print("\n[RELOAD] No working state (use F5 first)")
            return False

    last_start_debug = False
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_m:
                    _save_state(env, SCRIPT_DIR, game, "ManualSave")
                if event.key == pygame.K_F5:
                    _save_state(env, SCRIPT_DIR, game, "QuickSave")
                    if practice:
                        # Also store in memory for quick reload
                        working_state = env.em.get_state()
                        ram = env.get_ram()
                        working_state_level = int(ram[level_id_offset]) if len(ram) > level_id_offset else 0
                        working_state_name = level_names.get(working_state_level, "???")
                        print(f"[PRACTICE] Working state saved (level {working_state_level}: {working_state_name})")
                if event.key == pygame.K_F6:
                    if practice and practice_recorder:
                        # Labeled checkpoint
                        ram = env.get_ram()
                        lvl = int(ram[level_id_offset]) if len(ram) > level_id_offset else 0
                        practice_recorder.session.checkpoint(env, {
                            "level_id": lvl,
                            "level_name": level_names.get(lvl, ""),
                            "frame": frame_idx,
                        })
                        print(f"[PRACTICE] Checkpoint #{practice_recorder.current_index - 1} saved")
                    elif len(controllers) > 0:
                        btn = last_pressed[0]
                        if btn >= 0:
                            start_overrides[0] = btn
                            print(f"[MAP] P1 Start override -> button {btn}")
                if event.key in (pygame.K_F7, pygame.K_F8, pygame.K_BACKSPACE):
                    if practice:
                        reload_working_state()
                    elif event.key == pygame.K_F8 and len(controllers) > 0:
                        btn = last_pressed[0]
                        if btn >= 0:
                            select_overrides[0] = btn
                            print(f"[MAP] P1 Select override -> button {btn}")
                    elif event.key == pygame.K_F7 and len(controllers) > 1:
                        btn = last_pressed[1]
                        if btn >= 0:
                            start_overrides[1] = btn
                            print(f"[MAP] P2 Start override -> button {btn}")
                if event.key == pygame.K_F9:
                    if len(controllers) > 1:
                        btn = last_pressed[1]
                        if btn >= 0:
                            select_overrides[1] = btn
                            print(f"[MAP] P2 Select override -> button {btn}")
                if event.key == pygame.K_LEFTBRACKET:
                    speed_idx = max(0, speed_idx - 1)
                    print(f"[SPEED] {speed_levels[speed_idx]}x")
                if event.key == pygame.K_RIGHTBRACKET:
                    speed_idx = min(len(speed_levels) - 1, speed_idx + 1)
                    print(f"[SPEED] {speed_levels[speed_idx]}x")

        keys = pygame.key.get_pressed()
        action_size = env.action_space.shape[0]
        action = [0] * action_size

        keyboard_action(
            keys,
            action,
            pygame,
            start_action_index=start_action_index,
            select_action_index=select_action_index,
        )
        if force_start and start_action_index is not None and frame_idx < force_start_frames:
            action[start_action_index] = 1
        if controllers:
            player_stride = 12
            if action_size >= player_stride * 2:
                for i, joy in enumerate(controllers[:2]):
                    controller_action(
                        joy,
                        action,
                        offset=i * player_stride,
                        start_override=start_overrides[i],
                        select_override=select_overrides[i],
                        start_action_index=start_action_index,
                        select_action_index=select_action_index,
                        controller_map=_swap_xy_map() if swap_xy[i] else None,
                        axis_map=_axis_map_for_controller(i),
                    )
            else:
                for i, joy in enumerate(controllers[:2]):
                    controller_action(
                        joy,
                        action,
                        offset=0,
                        start_override=start_overrides[i],
                        select_override=select_overrides[i],
                        start_action_index=start_action_index,
                        select_action_index=select_action_index,
                        controller_map=_swap_xy_map() if swap_xy[i] else None,
                        axis_map=_axis_map_for_controller(i),
                    )
        else:
            controller_action(joystick, action)
        if action_size >= 24:
            sanitize_action_multi(action, players=2)
        else:
            sanitize_action(action)

        if controllers:
            for idx, joy in enumerate(controllers[:2]):
                pressed = [i for i in range(joy.get_numbuttons()) if joy.get_button(i)]
                if pressed:
                    last_pressed[idx] = pressed[-1]

        # Practice mode: L3+LB controller combo to reload working state
        if practice and controllers:
            joy = controllers[0]
            num_btns = joy.get_numbuttons()
            # L3: button 9 (Xbox) or 10 (PS), LB: button 4
            l3_pressed = (9 < num_btns and joy.get_button(9)) or (10 < num_btns and joy.get_button(10))
            lb_pressed = 4 < num_btns and joy.get_button(4)
            if l3_pressed and lb_pressed:
                reload_working_state()
                continue  # Skip frame after reload

        if input_debug and start_action_index is not None:
            start_now = bool(action[start_action_index])
            if start_now != last_start_debug:
                state = "on" if start_now else "off"
                print(f"[INPUT] start={state} frame={frame_idx}")
                last_start_debug = start_now

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += float(reward)

        ram = env.get_ram()
        level_id = ram[level_id_offset] if len(ram) > level_id_offset else None
        level_timer_frames = read_level_timer_frames(
            ram,
            frames_offset=RAM_LEVEL_TIMER_FRAMES,
            minutes_offset=RAM_LEVEL_TIMER_MINUTES,
        )
        pos_x = ram[RAM_POS_X] if len(ram) > RAM_POS_X else None
        pos_y = ram[RAM_POS_Y] if len(ram) > RAM_POS_Y else None
        action_dk = ram[RAM_ACTION_DK] if len(ram) > RAM_ACTION_DK else None
        action_diddy = ram[RAM_ACTION_DIDDY] if len(ram) > RAM_ACTION_DIDDY else None
        bonus_timer = ram[RAM_BONUS_TIMER] if len(ram) > RAM_BONUS_TIMER else None
        if level_id is not None:
            bonus_active = bonus_timer is not None and int(bonus_timer) > 0
            bonus_recent = bonus_active or last_bonus_active
            timer_reset = (
                level_timer_frames is not None
                and last_level_timer_frames is not None
                and level_timer_frames < last_level_timer_frames
            )
            if current_level_id is None:
                current_level_id = int(level_id)
                segment_level_id = int(level_id)
                level_start_frame = frame_idx
                segment_start_frame = frame_idx
                segment_start_timer = level_timer_frames
                segment_elapsed_offset = 0
                level_active = False
                start_detector.reset()
                inferred = _infer_level_name_from_state(state)
                if inferred and int(level_id) not in level_names:
                    level_names[int(level_id)] = inferred
                    _save_level_names(LEVEL_NAMES_PATH, level_names)
                _append_run_log(
                    SPLIT_LOG_PATH,
                    event="level_seen",
                    session_id=session_id,
                    game=game,
                    state=state,
                    level_id=current_level_id,
                    level_name=level_names.get(current_level_id),
                    elapsed_seconds=0.0,
                    elapsed_frames=0,
                    best_seconds=None,
                    timer_start=level_timer_frames,
                    timer_end=None,
                    frame_start=frame_idx,
                    frame_end=None,
                    pos_x=int(pos_x) if pos_x is not None else None,
                    pos_y=int(pos_y) if pos_y is not None else None,
                    reason=None,
                )
            else:
                current_level_id = int(level_id)

            if timer_reset:
                if bonus_recent:
                    elapsed_chunk = max(0, (last_level_timer_frames or 0) - (segment_start_timer or 0))
                    segment_elapsed_offset += elapsed_chunk
                    segment_start_timer = level_timer_frames
                    level_start_frame = frame_idx
                    segment_start_frame = frame_idx
                    level_active = False
                    start_detector.reset()
                else:
                    if level_active and segment_level_id is not None and current_level_id != segment_level_id:
                        elapsed_frames = max(0, (last_level_timer_frames or 0) - (segment_start_timer or 0))
                        elapsed_frames += segment_elapsed_offset
                        elapsed = elapsed_frames / 60.0
                        level_key = f"{segment_level_id:02X}"
                        best = best_times.get(level_key) if autosplit else None
                        if autosplit:
                            if best is None or elapsed < best:
                                best_times[level_key] = elapsed
                                _save_best_times(BEST_TIMES_PATH, best_times)
                                print(f"[SPLIT] Level {level_key} { _format_time(elapsed) } (NEW BEST)")
                            else:
                                print(f"[SPLIT] Level {level_key} { _format_time(elapsed) } (Best { _format_time(best) })")
                        _append_run_log(
                            SPLIT_LOG_PATH,
                            event="level_end",
                            session_id=session_id,
                            game=game,
                            state=state,
                            level_id=segment_level_id,
                            level_name=level_names.get(segment_level_id),
                            elapsed_seconds=elapsed,
                            elapsed_frames=elapsed_frames,
                            best_seconds=best_times.get(level_key) if autosplit else None,
                            timer_start=segment_start_timer,
                            timer_end=last_level_timer_frames,
                            frame_start=segment_start_frame,
                            frame_end=frame_idx,
                            pos_x=int(pos_x) if pos_x is not None else None,
                            pos_y=int(pos_y) if pos_y is not None else None,
                            reason="split",
                        )
                    elif level_active and segment_level_id is not None and current_level_id == segment_level_id:
                        elapsed_frames = max(0, (last_level_timer_frames or 0) - (segment_start_timer or 0))
                        elapsed_frames += segment_elapsed_offset
                        elapsed = elapsed_frames / 60.0
                        _append_run_log(
                            SPLIT_LOG_PATH,
                            event="level_end",
                            session_id=session_id,
                            game=game,
                            state=state,
                            level_id=segment_level_id,
                            level_name=level_names.get(segment_level_id),
                            elapsed_seconds=elapsed,
                            elapsed_frames=elapsed_frames,
                            best_seconds=best_times.get(f"{segment_level_id:02X}") if autosplit else None,
                            timer_start=segment_start_timer,
                            timer_end=last_level_timer_frames,
                            frame_start=segment_start_frame,
                            frame_end=frame_idx,
                            pos_x=int(pos_x) if pos_x is not None else None,
                            pos_y=int(pos_y) if pos_y is not None else None,
                            reason="death",
                        )
                    segment_level_id = current_level_id
                    segment_start_timer = level_timer_frames
                    segment_start_frame = frame_idx
                    segment_elapsed_offset = 0
                    level_start_frame = frame_idx
                    level_active = False
                    start_detector.reset()

            moved_this_frame = False
            if pos_x is not None and pos_y is not None:
                cur_pos = (int(pos_x), int(pos_y))
                if last_pos is not None and cur_pos != last_pos:
                    moved_this_frame = True
                last_pos = cur_pos

            if not level_active and start_detector.update(level_timer_frames, moved_this_frame):
                level_active = True
                level_start_frame = frame_idx
                segment_start_frame = frame_idx
                level_label = level_names.get(current_level_id or -1)
                label = f"{level_label} " if level_label else ""
                print(f"[LEVEL] start: {label}timer running + movement (state={state})")
                _append_run_log(
                    SPLIT_LOG_PATH,
                    event="level_start",
                    session_id=session_id,
                    game=game,
                    state=state,
                    level_id=current_level_id,
                    level_name=level_label,
                    elapsed_seconds=0.0,
                    elapsed_frames=0,
                    best_seconds=None,
                    timer_start=level_timer_frames,
                    timer_end=None,
                    frame_start=frame_idx,
                    frame_end=None,
                    pos_x=int(pos_x) if pos_x is not None else None,
                    pos_y=int(pos_y) if pos_y is not None else None,
                    reason=None,
                )

            should_log = False
            if level_log_every > 0:
                if last_level_id != level_id:
                    should_log = True
                elif (
                    level_timer_frames is not None
                    and last_level_timer_frames is not None
                    and level_timer_frames < last_level_timer_frames
                ):
                    should_log = True
                elif frame_idx - last_level_log_frame >= level_log_every:
                    should_log = True

            if should_log:
                timer_text = "NA" if level_timer_frames is None else str(int(level_timer_frames))
                seg_text = (
                    f"seg=0x{segment_level_id:02X}({segment_level_id}) "
                    if segment_level_id is not None
                    else "seg=NA "
                )
                print(
                    f"[LEVEL] id=0x{int(level_id):02X}({int(level_id)}) "
                    f"timer={timer_text} "
                    f"pos=({int(pos_x) if pos_x is not None else 'NA'},{int(pos_y) if pos_y is not None else 'NA'}) "
                    f"act=({int(action_dk) if action_dk is not None else 'NA'},{int(action_diddy) if action_diddy is not None else 'NA'}) "
                    f"{seg_text}"
                    f"active={level_active}"
                )
                last_level_id = int(level_id)
                last_level_log_frame = frame_idx
            last_bonus_active = bonus_active

        if record_path and record_start_level_id is None and level_id is not None:
            record_start_level_id = int(level_id)
            record_start_level_name = level_names.get(record_start_level_id)
            name_text = f"{record_start_level_name} " if record_start_level_name else ""
            print(f"[RECORD] start: {name_text}0x{record_start_level_id:02X}({record_start_level_id})")

        surf = pygame.surfarray.make_surface(obs.swapaxes(0, 1))
        fast_forward = keys[pygame.K_TAB]
        screen.blit(pygame.transform.scale(surf, screen.get_size()), (0, 0))
        lines = [
            f"Speed: {speed_levels[speed_idx]}x",
        ]
        if autosplit and current_level_id is not None:
            elapsed_frames = 0
            if level_active:
                if level_timer_frames is not None and segment_start_timer is not None:
                    elapsed_frames = max(0, level_timer_frames - segment_start_timer) + segment_elapsed_offset
                else:
                    elapsed_frames = max(0, frame_idx - level_start_frame)
            elapsed = elapsed_frames / 60.0
            display_level_id = segment_level_id if segment_level_id is not None else current_level_id
            level_key = f"{display_level_id:02X}"
            best = best_times.get(level_key)
            best_text = _format_time(best) if best is not None else "--:--.--"
            timer_text = _format_time(elapsed)
            level_name = level_names.get(display_level_id)
            level_label = (
                f"{level_name} 0x{level_key}({display_level_id})"
                if level_name
                else f"Level 0x{level_key}({display_level_id})"
            )
            if level_timer_frames is not None:
                status = "RUN" if level_active else "WAIT"
                lines.append(
                    f"{level_label} {status} {timer_text} | "
                    f"Best {best_text} | RAM {int(level_timer_frames)}"
                )
            else:
                status = "RUN" if level_active else "WAIT"
                lines.append(
                    f"{level_label} {status} {timer_text} | Best {best_text}"
                )
        if fast_forward:
            lines[0] += f" FAST({fast_forward_factor}x)"
        if record_path:
            lines[0] += " REC"
        btn_names = ["B", "Y", "Sel", "St", "Up", "Dn", "Lt", "Rt", "A", "X", "L", "R"]
        active_btns = [btn_names[i] for i, v in enumerate(action[:12]) if v > 0]
        if active_btns:
            lines.append(f"P1 Buttons: {' '.join(active_btns)}")
        if len(action) >= 24:
            active_btns_p2 = [btn_names[i] for i, v in enumerate(action[12:24]) if v > 0]
            if active_btns_p2:
                lines.append(f"P2 Buttons: {' '.join(active_btns_p2)}")
        for i, line in enumerate(lines):
            text = font.render(line, True, (255, 255, 255))
            screen.blit(text, (5, 5 + i * 16))

        pygame.display.flip()
        tick_rate = 60 * speed_levels[speed_idx] * (fast_forward_factor if fast_forward else 1.0)
        clock.tick(int(min(tick_rate, 180)))

        if done:
            if record_path and manifest_path:
                latest = find_latest_recording(record_path, game=game)
                if latest:
                    try:
                        stat = latest.stat()
                        mtime = stat.st_mtime
                    except OSError:
                        mtime = None
                        stat = None
                    if mtime is None or record_last_mtime is None or mtime > record_last_mtime:
                        record_last_mtime = mtime if mtime is not None else record_last_mtime
                        entry = {
                            "event": "episode_end",
                            "timestamp": time.time(),
                            "session_id": session_id,
                            "game": game,
                            "state": state,
                            "recording": latest.name,
                            "recording_mtime": mtime,
                            "recording_bytes": stat.st_size if stat is not None else None,
                            "start_level_id": record_start_level_id,
                            "start_level_name": record_start_level_name,
                            "end_level_id": int(level_id) if level_id is not None else None,
                            "end_level_name": level_names.get(int(level_id)) if level_id is not None else None,
                            "frames": frame_idx,
                            "started_at": record_start_time,
                        }
                        with manifest_path.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(entry) + "\n")
                        end_level_id = int(level_id) if level_id is not None else None
                        end_level_name = level_names.get(end_level_id) if end_level_id is not None else None
                        if end_level_id is not None:
                            name_text = f"{end_level_name} " if end_level_name else ""
                            print(f"[RECORD] end: {name_text}0x{end_level_id:02X}({end_level_id})")
            if level_active and segment_level_id is not None:
                elapsed_frames = 0
                if level_timer_frames is not None and segment_start_timer is not None:
                    elapsed_frames = max(0, level_timer_frames - segment_start_timer) + segment_elapsed_offset
                else:
                    elapsed_frames = max(0, frame_idx - level_start_frame)
                elapsed = elapsed_frames / 60.0
                _append_run_log(
                    SPLIT_LOG_PATH,
                    event="level_end",
                    session_id=session_id,
                    game=game,
                    state=state,
                    level_id=segment_level_id,
                    level_name=level_names.get(segment_level_id),
                    elapsed_seconds=elapsed,
                    elapsed_frames=elapsed_frames,
                    best_seconds=best_times.get(f"{segment_level_id:02X}") if autosplit else None,
                    timer_start=segment_start_timer,
                    timer_end=level_timer_frames,
                    frame_start=segment_start_frame,
                    frame_end=frame_idx,
                    pos_x=int(pos_x) if pos_x is not None else None,
                    pos_y=int(pos_y) if pos_y is not None else None,
                    reason="run_end",
                )
            _append_run_log(
                SPLIT_LOG_PATH,
                event="episode_end",
                session_id=session_id,
                game=game,
                state=state,
                level_id=int(level_id) if level_id is not None else None,
                level_name=level_names.get(int(level_id)) if level_id is not None else None,
                elapsed_seconds=None,
                elapsed_frames=None,
                best_seconds=None,
                timer_start=None,
                timer_end=level_timer_frames,
                frame_start=None,
                frame_end=frame_idx,
                pos_x=int(pos_x) if pos_x is not None else None,
                pos_y=int(pos_y) if pos_y is not None else None,
                reason="episode_end",
            )
            print(f"\nEpisode ended! Total reward: {total_reward}")
            pygame.time.wait(1500)
            obs, info = env.reset()
            total_reward = 0.0
            record_start_level_id = None
            record_start_level_name = None
            record_start_time = time.time()
        frame_idx += 1
        last_level_timer_frames = level_timer_frames
        if level_id_candidates:
            changed = False
            snapshot: dict[int, int] = {}
            for off in level_id_candidates:
                if len(ram) <= off:
                    continue
                val = int(ram[off])
                snapshot[off] = val
                if last_level_id_candidates.get(off) != val:
                    changed = True
            if changed and snapshot:
                if level_id_verbose:
                    parts = [f"0x{off:04X}=0x{val:02X}" for off, val in sorted(snapshot.items())]
                    print(f"[LEVEL_ID] candidates: {' '.join(parts)}")
                last_level_id_candidates = snapshot

    env.close()
    pygame.quit()
    if segment_level_id is not None:
        elapsed_frames = 0
        if last_level_timer_frames is not None and segment_start_timer is not None:
            elapsed_frames = max(0, last_level_timer_frames - segment_start_timer) + segment_elapsed_offset
        else:
            elapsed_frames = max(0, frame_idx - level_start_frame)
        elapsed = elapsed_frames / 60.0
        _append_run_log(
            SPLIT_LOG_PATH,
            event="level_end",
            session_id=session_id,
            game=game,
            state=state,
            level_id=segment_level_id,
            level_name=level_names.get(segment_level_id),
            elapsed_seconds=elapsed,
            elapsed_frames=elapsed_frames,
            best_seconds=best_times.get(f"{segment_level_id:02X}") if autosplit else None,
            timer_start=segment_start_timer,
            timer_end=last_level_timer_frames,
            frame_start=segment_start_frame,
            frame_end=frame_idx,
            pos_x=int(pos_x) if pos_x is not None else None,
            pos_y=int(pos_y) if pos_y is not None else None,
            reason="quit",
        )


def _save_state(env, game_dir: Path, game: str, name: str):
    """Save current emulator state."""
    try:
        print(f"\nSaving state to '{name}.state'...")
        save_path = save_state(env, game_dir, game, name)
        print("Saved!")
        print(f"Also saved to {save_path}")
    except Exception as e:
        print(f"Could not save state: {e}")


def list_states(game: str):
    """List available save states for the game."""
    from retro_harness import get_available_states

    states = get_available_states(game, SCRIPT_DIR)
    if not states:
        print(f"No states found for {game}")
        return

    print(f"Available states for {game}:")
    for state in states:
        print(f"  {state}")


def sync_level_names(game: str) -> None:
    names = _bootstrap_level_names_from_states(
        game=game,
        game_dir=SCRIPT_DIR,
        level_names_path=LEVEL_NAMES_PATH,
    )
    if not names:
        print("No level names found or ROM missing.")
        return
    print("Level names:")
    for level_id in sorted(names.keys()):
        print(f"  0x{level_id:02X}({level_id}) {names[level_id]}")


def run_headless_tests(test_name: str) -> None:
    """Run headless unit tests."""
    import unittest
    import sys
    from pathlib import Path

    # Add tests directory to path
    tests_dir = SCRIPT_DIR / "tests"
    if str(tests_dir) not in sys.path:
        sys.path.insert(0, str(tests_dir))

    # Import and run tests
    if test_name:
        # Run specific test
        suite = unittest.TestLoader().loadTestsFromName(f"test_headless.TestHeadless.{test_name}")
    else:
        # Run all tests in test_headless
        import test_headless
        suite = unittest.TestLoader().loadTestsFromModule(test_headless)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if not result.wasSuccessful():
        sys.exit(1)


def replay_movie(
    bk2_path: Path,
    *,
    output_path: Path,
    game_dir: Path,
    max_frames: int | None = None,
    dump_ram: bool = True,
    sample_every: int = 1,
    level_id_offset: int | None = None,
) -> None:
    if not bk2_path.exists():
        print(f"Recording not found: {bk2_path}")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)

    movie = retro.Movie(str(bk2_path))
    game = movie.get_game()
    print(f"[REPLAY] game={game} file={bk2_path.name}")

    # Configure env for playback
    try:
        env = make_env(game=game, state="NONE", game_dir=game_dir)
    except Exception as exc:
        print(f"[REPLAY] error creating env: {exc}")
        return
    try:
        movie.configure(env)
    except Exception:
        pass
    try:
        env.initial_state = movie.get_state()
    except Exception as exc:
        print(f"[REPLAY] warning: failed to set initial state: {exc}")
    obs, info = env.reset()

    num_buttons = len(getattr(env, "buttons", []))
    action_size = env.action_space.shape[0]
    players = max(1, action_size // max(1, num_buttons))
    level_id_offset = level_id_offset if level_id_offset is not None else _parse_level_id_offset()

    frame_idx = 0
    first_level_id: int | None = None
    last_level_id: int | None = None

    with output_path.open("w", encoding="utf-8") as f:
        while True:
            if max_frames is not None and frame_idx >= max_frames:
                break
            if not movie.step():
                break
            action = [0] * action_size
            if num_buttons > 0:
                for p in range(players):
                    for b in range(num_buttons):
                        idx = p * num_buttons + b
                        if idx >= action_size:
                            continue
                        action[idx] = 1 if movie.get_key(p, b) else 0

            obs, reward, terminated, truncated, info = env.step(action)
            if frame_idx % max(1, sample_every) == 0:
                ram = env.get_ram()
                level_id = int(ram[level_id_offset]) if len(ram) > level_id_offset else None
                if level_id is not None:
                    if first_level_id is None:
                        first_level_id = level_id
                    last_level_id = level_id
                entry = {
                    "frame": frame_idx,
                    "timestamp": time.time(),
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "action": action,
                    "level_id": level_id,
                    "level_timer_frames": read_level_timer_frames(
                        ram,
                        frames_offset=RAM_LEVEL_TIMER_FRAMES,
                        minutes_offset=RAM_LEVEL_TIMER_MINUTES,
                    ),
                    "pos_x": int(ram[RAM_POS_X]) if len(ram) > RAM_POS_X else None,
                    "pos_y": int(ram[RAM_POS_Y]) if len(ram) > RAM_POS_Y else None,
                }
                if dump_ram:
                    entry["ram"] = list(int(v) for v in ram)
                f.write(json.dumps(entry) + "\n")

            frame_idx += 1
            if terminated or truncated:
                break

    env.close()
    print(f"[REPLAY] frames={frame_idx} output={output_path}")
    if first_level_id is not None:
        print(f"[REPLAY] level_start=0x{first_level_id:02X}({first_level_id})")
    if last_level_id is not None:
        print(f"[REPLAY] level_last=0x{last_level_id:02X}({last_level_id})")


def replay_splits(
    replay_path: Path,
    *,
    out_log: Path,
    run_id: str | None = None,
    game: str = DEFAULT_GAME,
    state: str = DEFAULT_STATE,
) -> None:
    entries = iter_jsonl(replay_path)
    if not entries:
        print(f"No replay entries found at {replay_path}")
        return
    level_names = _load_level_names(LEVEL_NAMES_PATH)
    session_id = run_id or f"replay:{replay_path.stem}"

    last_level = None
    last_timer = None
    segment_start_timer = None
    segment_start_frame = None
    segment_start_pos = None

    for entry in entries:
        level_id = entry.get("level_id")
        timer = entry.get("level_timer_frames")
        frame = entry.get("frame")
        pos_x = entry.get("pos_x")
        pos_y = entry.get("pos_y")
        if not isinstance(level_id, int) or not isinstance(frame, int):
            last_timer = timer if isinstance(timer, int) else last_timer
            continue

        if last_level is None:
            last_level = level_id
            segment_start_timer = timer if isinstance(timer, int) else None
            segment_start_frame = frame
            segment_start_pos = (pos_x, pos_y)
            _append_run_log(
                out_log,
                event="level_start",
                session_id=session_id,
                game=game,
                state=state,
                level_id=level_id,
                level_name=level_names.get(level_id),
                elapsed_seconds=0.0,
                elapsed_frames=0,
                best_seconds=None,
                timer_start=segment_start_timer,
                timer_end=None,
                frame_start=segment_start_frame,
                frame_end=None,
                pos_x=pos_x if isinstance(pos_x, int) else None,
                pos_y=pos_y if isinstance(pos_y, int) else None,
                reason="replay_start",
            )
            last_timer = timer if isinstance(timer, int) else last_timer
            continue

        timer_reset = (
            isinstance(timer, int)
            and isinstance(last_timer, int)
            and timer < last_timer
        )
        level_change = level_id != last_level

        if timer_reset or level_change:
            elapsed_frames = None
            elapsed_seconds = None
            if isinstance(segment_start_timer, int) and isinstance(last_timer, int):
                elapsed_frames = max(0, last_timer - segment_start_timer)
                elapsed_seconds = elapsed_frames / 60.0
            elif isinstance(segment_start_frame, int) and isinstance(frame, int):
                elapsed_frames = max(0, frame - segment_start_frame)
                elapsed_seconds = elapsed_frames / 60.0

            _append_run_log(
                out_log,
                event="level_end",
                session_id=session_id,
                game=game,
                state=state,
                level_id=last_level,
                level_name=level_names.get(last_level),
                elapsed_seconds=elapsed_seconds,
                elapsed_frames=elapsed_frames,
                best_seconds=None,
                timer_start=segment_start_timer,
                timer_end=last_timer if isinstance(last_timer, int) else None,
                frame_start=segment_start_frame,
                frame_end=frame,
                pos_x=segment_start_pos[0] if segment_start_pos else None,
                pos_y=segment_start_pos[1] if segment_start_pos else None,
                reason="replay_timer_reset" if timer_reset else "replay_level_change",
            )

            last_level = level_id
            segment_start_timer = timer if isinstance(timer, int) else None
            segment_start_frame = frame
            segment_start_pos = (pos_x, pos_y)
            _append_run_log(
                out_log,
                event="level_start",
                session_id=session_id,
                game=game,
                state=state,
                level_id=level_id,
                level_name=level_names.get(level_id),
                elapsed_seconds=0.0,
                elapsed_frames=0,
                best_seconds=None,
                timer_start=segment_start_timer,
                timer_end=None,
                frame_start=segment_start_frame,
                frame_end=None,
                pos_x=pos_x if isinstance(pos_x, int) else None,
                pos_y=pos_y if isinstance(pos_y, int) else None,
                reason="replay_start",
            )

        last_timer = timer if isinstance(timer, int) else last_timer

    if last_level is not None:
        elapsed_frames = None
        elapsed_seconds = None
        if isinstance(segment_start_timer, int) and isinstance(last_timer, int):
            elapsed_frames = max(0, last_timer - segment_start_timer)
            elapsed_seconds = elapsed_frames / 60.0
        elif isinstance(segment_start_frame, int) and isinstance(frame, int):
            elapsed_frames = max(0, frame - segment_start_frame)
            elapsed_seconds = elapsed_frames / 60.0
        _append_run_log(
            out_log,
            event="level_end",
            session_id=session_id,
            game=game,
            state=state,
            level_id=last_level,
            level_name=level_names.get(last_level),
            elapsed_seconds=elapsed_seconds,
            elapsed_frames=elapsed_frames,
            best_seconds=None,
            timer_start=segment_start_timer,
            timer_end=last_timer if isinstance(last_timer, int) else None,
            frame_start=segment_start_frame,
            frame_end=frame,
            pos_x=segment_start_pos[0] if segment_start_pos else None,
            pos_y=segment_start_pos[1] if segment_start_pos else None,
            reason="replay_end",
        )
    print(f"[REPLAY] split log appended to {out_log}")


def replay_splits_all(
    recordings_dir: Path,
    *,
    out_log: Path,
    generate_missing: bool = True,
    sample_every: int = 1,
) -> None:
    recordings_dir = recordings_dir.resolve()
    replay_paths: list[Path] = []
    for replay_path in recordings_dir.rglob("*.replay.jsonl"):
        replay_paths.append(replay_path)

    if generate_missing:
        for bk2_path in recordings_dir.rglob("*.bk2"):
            replay_path = bk2_path.with_suffix(".replay.jsonl")
            if replay_path.exists():
                continue
            replay_movie(
                bk2_path,
                output_path=replay_path,
                game_dir=SCRIPT_DIR,
                max_frames=None,
                dump_ram=False,
                sample_every=max(1, sample_every),
            )
            replay_paths.append(replay_path)

    if not replay_paths:
        print(f"No replay files found under {recordings_dir}")
        return

    for replay_path in sorted(set(replay_paths)):
        run_id = f"replay:{replay_path.parent.name}/{replay_path.stem}"
        replay_splits(
            replay_path,
            out_log=out_log,
            run_id=run_id,
        )


def summarize_splits(path: Path) -> None:
    entries = iter_jsonl(path)
    if not entries:
        print(f"No split log entries found at {path}")
        return
    level_names = _load_level_names(LEVEL_NAMES_PATH)
    level_times: dict[int, list[float]] = {}
    level_name_override: dict[int, str] = {}
    for entry in entries:
        event = entry.get("event", "split")
        reason = entry.get("reason")
        if event not in ("split", "finish", "level_end", "run_end"):
            continue
        if reason == "death":
            continue
        level_id = entry.get("level_id")
        elapsed = entry.get("elapsed_seconds")
        entry_name = entry.get("level_name")
        if not isinstance(level_id, int) or not isinstance(elapsed, (int, float)):
            continue
        if isinstance(entry_name, str) and level_id not in level_name_override:
            level_name_override[level_id] = entry_name
        level_times.setdefault(level_id, []).append(float(elapsed))
    if not level_times:
        print("No completed split entries found (event=split/finish).")
        return
    print("Split stats (seconds):")
    for level_id in sorted(level_times.keys()):
        times = level_times[level_id]
        name = level_names.get(level_id) or level_name_override.get(level_id)
        avg = statistics.mean(times)
        stdev = statistics.pstdev(times) if len(times) > 1 else 0.0
        spread = max(times) - min(times)
        name_text = f"{name} " if name else ""
        print(
            f"  {name_text}0x{level_id:02X}({level_id}) "
            f"n={len(times)} avg={avg:.2f} std={stdev:.2f} spread={spread:.2f}"
        )
    all_times = [t for times in level_times.values() for t in times]
    if all_times:
        avg = statistics.mean(all_times)
        stdev = statistics.pstdev(all_times) if len(all_times) > 1 else 0.0
        spread = max(all_times) - min(all_times)
        print(f"  ALL n={len(all_times)} avg={avg:.2f} std={stdev:.2f} spread={spread:.2f}")


def refresh_best_times_from_log(log_path: Path, *, best_path: Path) -> None:
    entries = iter_jsonl(log_path)
    if not entries:
        print(f"No split log entries found at {log_path}")
        return
    best_times: dict[str, float] = {}
    for entry in entries:
        event = entry.get("event")
        reason = entry.get("reason")
        if event not in ("level_end", "split", "finish", "run_end"):
            continue
        if reason == "death":
            continue
        level_id = entry.get("level_id")
        elapsed = entry.get("elapsed_seconds")
        if not isinstance(level_id, int) or not isinstance(elapsed, (int, float)):
            continue
        level_key = f"{level_id:02X}"
        elapsed = float(elapsed)
        best = best_times.get(level_key)
        if best is None or elapsed < best:
            best_times[level_key] = elapsed
    if not best_times:
        print("No completed split entries found (event=level_end/split/finish).")
        return
    _save_best_times(best_path, best_times)
    print(f"Wrote best times to {best_path}")


def refresh_best_times_from_log_with_min(
    log_path: Path,
    *,
    best_path: Path,
    min_seconds: float,
    dedupe_run: bool,
) -> None:
    entries = iter_jsonl(log_path)
    if not entries:
        print(f"No split log entries found at {log_path}")
        return
    best_times: dict[str, float] = {}
    per_run: dict[tuple[str, int], float] = {}
    for entry in entries:
        event = entry.get("event")
        reason = entry.get("reason")
        if event not in ("level_end", "split", "finish", "run_end"):
            continue
        if reason == "death":
            continue
        level_id = entry.get("level_id")
        elapsed = entry.get("elapsed_seconds")
        run_id = entry.get("run_id") or entry.get("session_id")
        if not isinstance(level_id, int) or not isinstance(elapsed, (int, float)):
            continue
        elapsed = float(elapsed)
        if elapsed < min_seconds:
            continue
        if dedupe_run and isinstance(run_id, str):
            key = (run_id, level_id)
            best = per_run.get(key)
            if best is None or elapsed < best:
                per_run[key] = elapsed
            continue
        level_key = f"{level_id:02X}"
        best = best_times.get(level_key)
        if best is None or elapsed < best:
            best_times[level_key] = elapsed
    if dedupe_run:
        for (run_id, level_id), elapsed in per_run.items():
            level_key = f"{level_id:02X}"
            best = best_times.get(level_key)
            if best is None or elapsed < best:
                best_times[level_key] = elapsed
    if not best_times:
        print("No completed split entries found (event=level_end/split/finish).")
        return
    _save_best_times(best_path, best_times)
    print(f"Wrote best times to {best_path}")


def print_split_table(
    log_path: Path,
    *,
    level_id: int | None = None,
    level_name: str | None = None,
    min_seconds: float = 0.0,
    dedupe_run: bool = True,
) -> None:
    entries = iter_jsonl(log_path)
    if not entries:
        print(f"No split log entries found at {log_path}")
        return
    level_names = _load_level_names(LEVEL_NAMES_PATH)
    rows = []
    seen: set[tuple[str, int]] = set()
    norm_name = level_name.lower() if level_name else None
    for entry in entries:
        event = entry.get("event")
        reason = entry.get("reason")
        if event not in ("level_end", "split", "finish", "run_end"):
            continue
        if reason == "death":
            continue
        lid = entry.get("level_id")
        elapsed = entry.get("elapsed_seconds")
        if not isinstance(lid, int) or not isinstance(elapsed, (int, float)):
            continue
        elapsed = float(elapsed)
        if elapsed < min_seconds:
            continue
        name = entry.get("level_name") or level_names.get(lid)
        if level_id is not None and lid != level_id:
            continue
        if norm_name is not None:
            if not isinstance(name, str) or name.lower() != norm_name:
                continue
        run_id = entry.get("run_id") or entry.get("session_id")
        if isinstance(run_id, str) and dedupe_run:
            key = (run_id, lid)
            if key in seen:
                continue
            seen.add(key)
        rows.append({
            "run_id": run_id,
            "level_id": lid,
            "level_name": name,
            "elapsed": elapsed,
            "reason": reason or "",
        })
    if not rows:
        print("No matching split rows found.")
        return
    rows.sort(key=lambda r: r["elapsed"])
    print("Split Times:")
    for row in rows:
        name = row["level_name"]
        name_text = f"{name} " if name else ""
        level_key = f"0x{row['level_id']:02X}({row['level_id']})"
        print(f"  {name_text}{level_key} {row['elapsed']:.2f}s run={row['run_id']}")


def main():
    from retro_harness import add_custom_integrations
    add_custom_integrations(SCRIPT_DIR)

    parser = argparse.ArgumentParser(description="Play Donkey Kong Country")
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Play command
    play_parser = subparsers.add_parser("play", help="Play the game")
    play_parser.add_argument("--game", default=DEFAULT_GAME)
    play_parser.add_argument("--state", default=DEFAULT_STATE)
    play_parser.add_argument("--scale", type=int, default=3, help="Window scale factor")
    play_parser.add_argument("--autosplit", action="store_true", help="Track level splits + best times")
    play_parser.add_argument("--record", action="store_true", help="Record .bk2 gameplay to recordings/")
    play_parser.add_argument("--record-dir", default="recordings", help="Recording output directory")
    play_parser.add_argument(
        "--level-id-offset",
        default="",
        help="Override RAM offset for level id (hex like 0x3E or decimal)",
    )
    play_parser.add_argument(
        "--level-log-every",
        type=int,
        default=0,
        help="Log [LEVEL] lines every N frames (0 = disable periodic [LEVEL] logs)",
    )
    play_parser.add_argument(
        "--practice",
        action="store_true",
        help="Practice mode: F5=save working state, F7/F8=reload, L3+LB=reload, F6=checkpoint",
    )
    play_parser.add_argument(
        "--label",
        default=None,
        help="Label for practice checkpoints (auto-detects from level if not set)",
    )

    # List command
    list_parser = subparsers.add_parser("list", help="List available states")
    list_parser.add_argument("--game", default=DEFAULT_GAME)

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Summarize split log stats")
    stats_parser.add_argument("--path", default=str(SPLIT_LOG_PATH), help="Path to split_runs.jsonl")

    # Refresh best times from logs
    refresh_parser = subparsers.add_parser(
        "refresh-best",
        help="Rebuild best_times.json from split log entries",
    )
    refresh_parser.add_argument("--log", default=str(SPLIT_LOG_PATH))
    refresh_parser.add_argument("--out", default=str(BEST_TIMES_PATH))
    refresh_parser.add_argument(
        "--min-seconds",
        type=float,
        default=5.0,
        help="Ignore splits shorter than this (filters partial runs)",
    )
    refresh_parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Do not dedupe multiple entries from the same run",
    )

    # Table command
    table_parser = subparsers.add_parser("table", help="Print split times table from logs")
    table_parser.add_argument("--log", default=str(SPLIT_LOG_PATH))
    table_parser.add_argument("--level-id", default="", help="Filter by level id (hex like 0xD9 or decimal)")
    table_parser.add_argument("--level-name", default="", help="Filter by level name (exact match)")
    table_parser.add_argument(
        "--min-seconds",
        type=float,
        default=5.0,
        help="Ignore splits shorter than this (filters partial runs)",
    )
    table_parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Do not dedupe multiple entries from the same run",
    )

    # Replay command
    replay_parser = subparsers.add_parser("replay", help="Replay a .bk2 recording headless and dump data")
    replay_parser.add_argument("--bk2", default="", help="Path to .bk2 file (overrides --latest)")
    replay_parser.add_argument("--latest", action="store_true", help="Use latest .bk2 in recordings/")
    replay_parser.add_argument("--recordings-dir", default=str(DEFAULT_RECORDINGS_DIR))
    replay_parser.add_argument("--out", default="", help="Output JSONL path")
    replay_parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = no limit)")
    replay_parser.add_argument("--no-ram", action="store_true", help="Do not include full RAM dump")
    replay_parser.add_argument("--every", type=int, default=1, help="Sample every N frames")
    replay_parser.add_argument("--level-id", default="", help="Filter by start level id (hex like 0x2E or decimal)")
    replay_parser.add_argument("--level-name", default="", help="Filter by start level name (exact match)")
    replay_parser.add_argument(
        "--level-id-offset",
        default="",
        help="Override RAM offset for level id (hex like 0x3E or decimal)",
    )

    # Replay split extraction
    replay_splits_parser = subparsers.add_parser(
        "replay-splits",
        help="Extract split timings from a replay JSONL and append to split log",
    )
    replay_splits_parser.add_argument("--replay", required=True, help="Path to .replay.jsonl")
    replay_splits_parser.add_argument(
        "--out-log",
        default=str(SPLIT_LOG_PATH),
        help="Output log path (defaults to split_runs.jsonl)",
    )
    replay_splits_parser.add_argument("--run-id", default="", help="Override run id for log entries")
    replay_splits_parser.add_argument("--game", default=DEFAULT_GAME)
    replay_splits_parser.add_argument("--state", default=DEFAULT_STATE)

    replay_all_parser = subparsers.add_parser(
        "replay-splits-all",
        help="Extract split timings from all replays (and optionally bk2s) in recordings/",
    )
    replay_all_parser.add_argument("--recordings-dir", default=str(DEFAULT_RECORDINGS_DIR))
    replay_all_parser.add_argument(
        "--out-log",
        default=str(SPLIT_LOG_PATH),
        help="Output log path (defaults to split_runs.jsonl)",
    )
    replay_all_parser.add_argument(
        "--no-generate",
        action="store_true",
        help="Do not generate missing .replay.jsonl from .bk2",
    )
    replay_all_parser.add_argument("--every", type=int, default=1, help="Sample every N frames when generating")

    # Level name sync
    level_parser = subparsers.add_parser("level-names", help="Build level_names.json from available states")
    level_parser.add_argument("--game", default=DEFAULT_GAME)

    # Test command
    test_parser = subparsers.add_parser("test", help="Run headless unit tests")
    test_parser.add_argument("--test-name", default="", help="Specific test to run (empty = all)")

    args = parser.parse_args()

    if args.command == "list":
        list_states(args.game)
    elif args.command == "stats":
        summarize_splits(Path(args.path))
    elif args.command == "refresh-best":
        log_path = Path(args.log)
        if not log_path.is_absolute():
            log_path = SCRIPT_DIR / log_path
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = SCRIPT_DIR / out_path
        refresh_best_times_from_log_with_min(
            log_path,
            best_path=out_path,
            min_seconds=max(0.0, float(args.min_seconds)),
            dedupe_run=not args.no_dedupe,
        )
    elif args.command == "table":
        log_path = Path(args.log)
        if not log_path.is_absolute():
            log_path = SCRIPT_DIR / log_path
        level_id = None
        if args.level_id:
            try:
                level_id = int(args.level_id, 16) if args.level_id.lower().startswith("0x") else int(args.level_id)
            except ValueError:
                level_id = None
        level_name = args.level_name.strip() or None
        print_split_table(
            log_path,
            level_id=level_id,
            level_name=level_name,
            min_seconds=max(0.0, float(args.min_seconds)),
            dedupe_run=not args.no_dedupe,
        )
    elif args.command == "replay":
        recordings_dir = Path(args.recordings_dir)
        bk2_path = Path(args.bk2) if args.bk2 else None
        if bk2_path and not bk2_path.is_absolute():
            bk2_path = SCRIPT_DIR / bk2_path
        level_id = None
        if args.level_id:
            try:
                level_id = int(args.level_id, 16) if args.level_id.lower().startswith("0x") else int(args.level_id)
            except ValueError:
                level_id = None
        level_name = args.level_name.strip() or None
        if (level_id is not None or level_name is not None) and not bk2_path:
            manifest_path = recordings_dir / DEFAULT_RECORDINGS_MANIFEST
            bk2_path = find_latest_recording_from_manifest(
                manifest_path,
                recordings_dir,
                level_id=level_id,
                level_name=level_name,
            )
        if not bk2_path or args.latest:
            bk2_path = find_latest_recording(recordings_dir, game=DEFAULT_GAME)
        if bk2_path is None:
            print(f"No recordings found in {recordings_dir}")
            return
        out_path = Path(args.out) if args.out else bk2_path.with_suffix(".replay.jsonl")
        if not out_path.is_absolute():
            out_path = SCRIPT_DIR / out_path
        max_frames = args.max_frames if args.max_frames > 0 else None
        level_id_offset = None
        if args.level_id_offset:
            try:
                level_id_offset = int(args.level_id_offset, 16) if args.level_id_offset.lower().startswith("0x") else int(args.level_id_offset)
            except ValueError:
                level_id_offset = None
        replay_movie(
            bk2_path,
            output_path=out_path,
            game_dir=SCRIPT_DIR,
            max_frames=max_frames,
            dump_ram=not args.no_ram,
            sample_every=max(1, args.every),
            level_id_offset=level_id_offset,
        )
    elif args.command == "replay-splits":
        replay_path = Path(args.replay)
        if not replay_path.is_absolute():
            replay_path = SCRIPT_DIR / replay_path
        out_log = Path(args.out_log)
        if not out_log.is_absolute():
            out_log = SCRIPT_DIR / out_log
        run_id = args.run_id.strip() or None
        replay_splits(
            replay_path,
            out_log=out_log,
            run_id=run_id,
            game=args.game,
            state=args.state,
        )
    elif args.command == "replay-splits-all":
        recordings_dir = Path(args.recordings_dir)
        if not recordings_dir.is_absolute():
            recordings_dir = SCRIPT_DIR / recordings_dir
        out_log = Path(args.out_log)
        if not out_log.is_absolute():
            out_log = SCRIPT_DIR / out_log
        replay_splits_all(
            recordings_dir,
            out_log=out_log,
            generate_missing=not args.no_generate,
            sample_every=max(1, args.every),
        )
    elif args.command == "level-names":
        sync_level_names(args.game)
    elif args.command == "test":
        run_headless_tests(args.test_name)
    else:
        # Default to play
        game = getattr(args, "game", DEFAULT_GAME)
        state = getattr(args, "state", DEFAULT_STATE)
        scale = getattr(args, "scale", 3)
        autosplit = getattr(args, "autosplit", False)
        record_dir = getattr(args, "record_dir", "recordings") if getattr(args, "record", False) else None
        level_log_every = getattr(args, "level_log_every", 0)
        level_id_offset = None
        if getattr(args, "level_id_offset", ""):
            try:
                level_id_offset = int(args.level_id_offset, 16) if args.level_id_offset.lower().startswith("0x") else int(args.level_id_offset)
            except ValueError:
                level_id_offset = None
        play_game(
            game,
            state,
            scale,
            autosplit=autosplit,
            record_dir=record_dir,
            level_log_every=level_log_every,
            level_id_offset=level_id_offset,
            practice=getattr(args, "practice", False),
            label=getattr(args, "label", None),
        )


if __name__ == "__main__":
    main()
