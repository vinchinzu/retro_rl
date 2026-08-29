"""Interactive emulator session with autoplay, HUD, and diagnostics."""

from __future__ import annotations

import gzip
import json
import os
import sys
from datetime import datetime
from typing import List, Optional

import numpy as np
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame

from harvest.paths import PROJECT_DIR, SAVES_DIR as PROJECT_SAVES_DIR, TASKS_DIR as PROJECT_TASKS_DIR
from harvest.core.ram_catalog import LiveRamEditor, RamPatch, read_ram_value
from harvest.core.scene import classify_scene_from_ram
from harvest.core.task_progress import task_progress_snapshot
from harvest.planner.day_plan import is_farm_tilemap, read_world_day_time
from harvest.runtime.autoplay_bot import AutoClearBot
from harvest.runtime.bot_input import (
    check_hotswap_chord,
    env_flag,
    get_controller_action,
    get_keyboard_action,
    init_controller,
)
from harvest.runtime.game_state import GameState
from harvest.runtime.probe_utils import (
    DEFAULT_WATCH_FIELDS,
    event_row,
    print_ram_narrow_hits,
    print_ram_search_hits,
    ram_narrow_hits,
    ram_search_hits,
    snapshot_from_ram,
    watch_values,
)
from harvest.runtime.recording_trace import recording_trace_entry, write_task_recording
from harvest.runtime.retro_setup import backup_mutable_start_state, make_harvest_env
from harvest.runtime.watch_display import (
    SPEED_LEVELS,
    bot_speed_timing,
    build_session_hud_lines,
    configure_headed,
    default_speed_index,
    display_set_mode,
    draw_session_hud,
    gym_env_step,
    pace_present,
    preview_interval,
)
from harvest.tasks.crop_planter import CropWaterTask
from harvest.core.tile_catalog import (
    ADDR_INPUT_LOCK,
    ADDR_TILEMAP,
    Tool,
)
from harvest.tasks.nav import (
    TILE_SIZE,
    get_pos_from_ram,
    make_action,
    Point,
)

from harvest.tasks.harvest_task import HarvestTask
from retro_harness import WorldState, sanitize_action

WATCHDOG_POSITION_PROGRESS_PIXELS = 64
SCRIPT_DIR = os.fspath(PROJECT_DIR)
STATES_DIR = os.path.join(os.fspath(PROJECT_DIR), "custom_integrations", "HarvestMoon-Snes")
SAVES_DIR = os.fspath(PROJECT_SAVES_DIR)
TASKS_DIR = os.fspath(PROJECT_TASKS_DIR)

class PlaySession:
    """Interactive play session."""

    def __init__(
        self,
        state: Optional[str] = None,
        scale: int = 2,
        bot: Optional[AutoClearBot] = None,
        autoplay: bool = False,
        max_frames: Optional[int] = None,
        record_name: Optional[str] = None,
        save_end: bool = False,
        money_hack: Optional[int] = None,
        ram_patches: Optional[List[RamPatch]] = None,
        hud_width: int = 176,
        diagnostics_dir: Optional[str] = None,
        watchdog_frames: Optional[int] = None,
        exit_on_bot_disable: bool = True,
    ):
        self.initial_state = state
        self.scale = scale
        self.autoplay = autoplay
        self.mode = 'bot' if autoplay else 'human'
        self.bot = bot or AutoClearBot()
        self.bot.enabled = autoplay
        self.frame_count = 0
        self.max_frames = max_frames
        self.hotswap_cooldown = 0
        self.hotswap_cancel_frames = 0
        self.hotswap_cancel_until_clear = False
        self.record_name = record_name
        self.recorded_frames: List[list] = []
        self.recorded_trace: List[dict] = []
        self.save_end = save_end
        self._end_saved = False
        self.money_hack = money_hack
        self.ram_patches = list(ram_patches or [])
        self.hud_width = hud_width
        self._hud_counts_frame = -10_000
        self._hud_crop_counts = (None, None)
        self._was_harvesting_for_hud = False
        self.diagnostics_dir = diagnostics_dir or os.path.join(SCRIPT_DIR, "logs", "long_runs")
        if watchdog_frames is None:
            watchdog_frames = int(os.getenv("AUTOPLAY_STALL_FRAMES", "5400"))
        self.watchdog_frames = watchdog_frames
        self.exit_on_bot_disable = exit_on_bot_disable
        self._diagnostic_run_dir: Optional[str] = None
        self._event_log_path: Optional[str] = None
        self._last_progress_signature = None
        self._last_progress_pos: Optional[tuple[int, int]] = None
        self._last_progress_frame = 0
        self._last_watchdog_frame = -10_000_000
        self._terminal_disable_captured_reason: Optional[str] = None

    @staticmethod
    def _safe_slug(text: str) -> str:
        keep = []
        for ch in text:
            if ch.isalnum() or ch in {"-", "_"}:
                keep.append(ch)
            else:
                keep.append("_")
        slug = "".join(keep).strip("_")
        return slug[:80] or "session"

    def _artifact_dir(self) -> str:
        if self._diagnostic_run_dir is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            state = self._safe_slug(self.initial_state or "nostate")
            self._diagnostic_run_dir = os.path.join(self.diagnostics_dir, f"{stamp}_{state}")
            os.makedirs(self._diagnostic_run_dir, exist_ok=True)
        return self._diagnostic_run_dir

    def _event_log(self) -> str:
        if self._event_log_path is None:
            self._event_log_path = os.path.join(self._artifact_dir(), "events.jsonl")
        return self._event_log_path

    def _append_event(self, row: dict) -> None:
        try:
            with open(self._event_log(), "a") as f:
                json.dump(row, f, sort_keys=True)
                f.write("\n")
        except Exception as exc:
            print(f"[DIAG] event log write failed: {exc}")

    def _active_task_key(self, task, *, depth: int = 0):
        if task is None or depth > 4:
            return None
        snap = task_progress_snapshot(task)
        if snap is not None:
            return snap.signature()
        return (task.__class__.__name__,)

    def _progress_signature(self, env, game_state: GameState):
        ram = env.get_ram()
        tilemap = int(ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(ram) else 0
        scene = classify_scene_from_ram(ram)
        day, _hour, _minute = read_world_day_time(ram)
        task_key = None
        if self.bot.day_plan_enabled:
            task_key = self._active_task_key(self.bot.day_plan_task)
        elif self.bot.crop_enabled:
            task_key = self._active_task_key(self.bot.crop_task)
        elif self.bot.grass_enabled:
            task_key = self._active_task_key(self.bot.grass_task)
        return (
            bool(self.bot.enabled),
            self.bot.disable_reason,
            int(getattr(game_state, "season", 0)),
            int(day),
            int(tilemap),
            scene.mode.value,
            task_key,
        )

    def _watchdog_progress_changed(self, signature, pos: Point) -> bool:
        if signature != self._last_progress_signature:
            return True
        if self._last_progress_pos is None:
            return True
        last_x, last_y = self._last_progress_pos
        return max(abs(int(pos.x) - last_x), abs(int(pos.y) - last_y)) >= WATCHDOG_POSITION_PROGRESS_PIXELS

    def _mark_watchdog_progress(self, signature, pos: Point) -> None:
        self._last_progress_signature = signature
        self._last_progress_pos = (int(pos.x), int(pos.y))
        self._last_progress_frame = self.frame_count

    def _capture_observation(self, env, obs: np.ndarray) -> np.ndarray:
        try:
            return env._update_obs()
        except Exception:
            return obs

    def _write_diagnostic_artifacts(self, env, game_state: GameState, action: np.ndarray, obs: np.ndarray, *, event: str, reason: str) -> None:
        artifact_dir = self._artifact_dir()
        base = os.path.join(
            artifact_dir,
            f"{self.frame_count:08d}_{self._safe_slug(event)}_{self._safe_slug(reason)}",
        )
        ram = env.get_ram()
        scene = classify_scene_from_ram(ram)
        paths: dict[str, str] = {}

        try:
            state_path = f"{base}.state"
            with gzip.open(state_path, "wb") as f:
                f.write(env.em.get_state())
            paths["state"] = state_path
        except Exception as exc:
            paths["state_error"] = str(exc)

        try:
            image_obs = self._capture_observation(env, obs)
            image_path = f"{base}.png"
            surf = pygame.surfarray.make_surface(image_obs.swapaxes(0, 1))
            pygame.image.save(surf, image_path)
            paths["screenshot"] = image_path
        except Exception as exc:
            paths["screenshot_error"] = str(exc)

        try:
            watches = watch_values(ram, DEFAULT_WATCH_FIELDS)
        except Exception:
            watches = {}
        snapshot = snapshot_from_ram(ram, frame=self.frame_count, action=action)
        row = event_row(
            event,
            snapshot,
            watches=watches,
            day_plan=self.bot.day_plan_task if self.bot.day_plan_enabled else None,
            note=reason,
        )
        row.update(
            {
                "date": game_state.date_str,
                "time": game_state.time_str,
                "scene": scene.summary(),
                "bot_enabled": bool(self.bot.enabled),
                "disable_reason": self.bot.disable_reason,
                "artifacts": paths,
            }
        )

        json_path = f"{base}.json"
        try:
            with open(json_path, "w") as f:
                json.dump(row, f, indent=2, sort_keys=True)
            paths["json"] = json_path
        except Exception as exc:
            paths["json_error"] = str(exc)
        self._append_event(row)
        print(f"[DIAG] {event}: {reason} -> {json_path}")

    def _check_autoplay_watchdog(self, env, game_state: GameState, action: np.ndarray, obs: np.ndarray) -> bool:
        if not self.autoplay or self.watchdog_frames <= 0:
            return False
        ram = env.get_ram()
        pos = get_pos_from_ram(ram)
        signature = self._progress_signature(env, game_state)
        if self._watchdog_progress_changed(signature, pos):
            self._mark_watchdog_progress(signature, pos)
            return False
        stalled_for = self.frame_count - self._last_progress_frame
        if stalled_for < self.watchdog_frames:
            return False
        if self.frame_count - self._last_watchdog_frame < self.watchdog_frames:
            return False

        reason = f"no progress for {stalled_for} frames"
        self._last_watchdog_frame = self.frame_count
        self._write_diagnostic_artifacts(env, game_state, action, obs, event="stall_watchdog", reason=reason)
        world = WorldState(frame=self.frame_count, ram=env.get_ram(), info={}, obs=obs)
        if self.bot.force_end_day(reason, world):
            self._last_progress_signature = None
            self._last_progress_pos = None
            self._last_progress_frame = self.frame_count
            return False
        self.bot.disable(f"watchdog stall: {reason}")
        return True

    @staticmethod
    def _disable_reason_is_failure(reason: str) -> bool:
        lower = reason.lower()
        failure_terms = ("stopped", "failed", "blocked", "timeout", "watchdog", "map changed", "error")
        return any(term in lower for term in failure_terms)

    def _handle_autoplay_terminal_disable(self, env, game_state: GameState, action: np.ndarray, obs: np.ndarray) -> bool:
        if not self.autoplay or not self.exit_on_bot_disable or self.bot.enabled:
            return False
        reason = self.bot.disable_reason or "bot disabled"
        if self._disable_reason_is_failure(reason) and self._terminal_disable_captured_reason != reason:
            self._write_diagnostic_artifacts(env, game_state, action, obs, event="bot_disabled", reason=reason)
            self._terminal_disable_captured_reason = reason
        print(f"[STOP] bot disabled: {reason}")
        return True

    def _skip_hotswap_cancel_warmup(self) -> bool:
        """Recorded-transition openings need frame-accurate starts."""
        if not self.bot.day_plan_enabled or self.bot.day_plan_started:
            return False
        phases = self.bot.day_plan_task.phases
        if not phases:
            return False
        return phases[0].kind == "recorded_transition"

    def _active_crop_task(self) -> Optional[CropWaterTask]:
        if self.bot.crop_enabled and self.bot.crop_task_started and not self.bot.crop_task_done:
            return self.bot.crop_task
        if self.bot.day_plan_enabled and self.bot.day_plan_started and not self.bot.day_plan_done:
            task = self.bot.day_plan_task.current_task
            if isinstance(task, CropWaterTask):
                return task
        return None

    def _active_harvest_task(self) -> Optional[HarvestTask]:
        if self.bot.day_plan_enabled and self.bot.day_plan_started and not self.bot.day_plan_done:
            task = self.bot.day_plan_task.current_task
            if isinstance(task, HarvestTask):
                return task
        return None

    def _invalidate_hud_counts(self) -> None:
        self._hud_counts_frame = -10_000
        self._hud_crop_counts = (None, None)

    def _note_task_state_for_hud(self) -> None:
        harvest_active = self._active_harvest_task() is not None
        if self._was_harvesting_for_hud and not harvest_active:
            self._invalidate_hud_counts()
        self._was_harvesting_for_hud = harvest_active

    def _sync_active_item(self, env) -> None:
        """Optional live shortcut for forcing the crop item selection."""
        crop_task = self._active_crop_task()
        if crop_task is None:
            return

        desired_item: Optional[int] = None
        if self.bot.crop_seed_hack and crop_task._plot_phase in ("water", "refill"):
            desired_item = int(Tool.WATERING_CAN)

        if desired_item is not None:
            self._set_live_value(env, "item_in_hand", desired_item, 0x4921)

    def _build_hud_lines(self, env, game_state: GameState, action: np.ndarray) -> List[str]:
        return build_session_hud_lines(self, env, game_state, action)

    def _draw_hud(self, screen, font, env, game_state: GameState, action: np.ndarray, height: int) -> None:
        draw_session_hud(self, screen, font, env, game_state, action, height)

    def _start_hotswap_cancel(self) -> None:
        """Clear any modal/input-lock state introduced by the hotswap chord."""
        self.hotswap_cancel_until_clear = True
        self.hotswap_cancel_frames = 0 if self._skip_hotswap_cancel_warmup() else 90

    def _bot_mode_action(self, env, game_state: GameState, obs: np.ndarray) -> np.ndarray:
        """Run hotswap cleanup before handing control to the bot."""
        ram = env.get_ram()
        input_lock = ram[ADDR_INPUT_LOCK] if ADDR_INPUT_LOCK < len(ram) else 1
        if self.hotswap_cancel_until_clear:
            if input_lock != 1:
                return make_action(b=self.frame_count % 2 == 0, a=self.frame_count % 2 == 1)
            self.hotswap_cancel_until_clear = False

        if self.hotswap_cancel_frames > 0:
            action = make_action(b=self.frame_count % 2 == 0)
            self.hotswap_cancel_frames -= 1
            if self.hotswap_cancel_frames == 0:
                self.hotswap_cancel_until_clear = False
            return action

        return self.bot.get_action(game_state, obs)

    def run(self):
        headless_boot = env_flag("HEADLESS") or os.getenv("SDL_VIDEODRIVER", "").lower() == "dummy"
        if not headless_boot:
            configure_headed()
        pygame.init()

        if self.record_name:
            stable_state = backup_mutable_start_state(self.initial_state, self.record_name)
            if stable_state != self.initial_state:
                print(f"[REC] Backed up start state {self.initial_state} -> {stable_state}")
                self.initial_state = stable_state

        try:
            env = make_harvest_env(self.initial_state)
        except Exception as e:
            print(f"Error: {e}")
            return

        obs, info = env.reset()
        self.bot.set_env(env)
        if self.bot.enabled:
            self.bot.prepare_for_enable()

        h, w = obs.shape[0], obs.shape[1]
        screen = None
        headless = env_flag("HEADLESS") or os.getenv("SDL_VIDEODRIVER", "").lower() == "dummy"
        if not headless:
            try:
                screen = display_set_mode(
                    pygame,
                    (self.hud_width + w * self.scale, h * self.scale),
                    caption=f"Harvest Moon [{self.mode.upper()}]",
                )
            except pygame.error:
                self.mode = 'bot'
                self.bot.enabled = True
        else:
            self.mode = 'bot'
            self.bot.enabled = True

        clock = pygame.time.Clock()
        font = pygame.font.SysFont('monospace', 10)
        joystick = init_controller()

        print(f"\n[HARVEST BOT] mode={self.mode.upper()}", end="")
        if self.initial_state is None and getattr(self.bot, "power_on_enabled", False):
            print(" power-on", end="")
        elif self.initial_state:
            print(f" state={self.initial_state}", end="")
        if self.record_name:
            print(f" record={self.record_name}", end="")
        if self.ram_patches:
            patch_text = ", ".join(f"{patch.field}={patch.value}" for patch in self.ram_patches)
            print(f" ram_set=[{patch_text}]", end="")
        print()
        print("[CONTROLS] [ ] = speed down/up | TAB = turbo | L+R+SELECT = human/bot | ESC = quit")

        running = True
        game_state = GameState(info, env.get_ram())
        last_day = game_state.day

        # Debug: show initial state from RAM
        ram = env.get_ram()
        money_raw = read_ram_value(ram, "money", raw=True)
        print(f"[INIT] {game_state.date_str} {game_state.time_str} | ${game_state.money:,} (raw={money_raw})")
        quick_save = None
        ram_search_candidates = None  # set of addresses from last search
        speed_levels = list(SPEED_LEVELS)
        # Start a notch above realtime so power-on boot is watchable without waiting.
        speed_idx = default_speed_index()
        self._display_speed = speed_levels[speed_idx]
        self._next_present = 0.0

        while running:
            if screen:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        elif event.key == pygame.K_F5:
                            if self.record_name:
                                self._save_recording(env, game_state)
                                running = False
                            else:
                                quick_save = env.em.get_state()
                                # Save as "latest" for easy resume
                                latest_path = os.path.join(STATES_DIR, "latest.state")
                                with gzip.open(latest_path, 'wb') as f:
                                    f.write(quick_save)
                                print(f"[SAVED] latest")
                        elif event.key == pygame.K_F9 and quick_save:
                            env.em.set_state(quick_save)
                            self._invalidate_hud_counts()
                            print("[LOADED]")
                        elif event.key == pygame.K_F2:
                            print("[RAM SEARCH] Enter value to find (decimal): ", end="", flush=True)
                            try:
                                search_val = int(input().strip())
                            except (ValueError, EOFError):
                                print("[RAM SEARCH] invalid input")
                                search_val = None
                            if search_val is not None:
                                ram = np.array(env.get_ram(), dtype=np.uint8)
                                ram_search_candidates = ram_search_hits(ram, search_val)
                                print_ram_search_hits(ram_search_candidates, search_val)
                        elif event.key == pygame.K_F3:
                            if ram_search_candidates is None:
                                print("[RAM NARROW] no previous search — press F2 first")
                            else:
                                print(
                                    f"[RAM NARROW] Enter NEW value to narrow {len(ram_search_candidates)} candidates: ",
                                    end="",
                                    flush=True,
                                )
                                try:
                                    search_val = int(input().strip())
                                except (ValueError, EOFError):
                                    print("[RAM NARROW] invalid input")
                                    search_val = None
                                if search_val is not None:
                                    ram = np.array(env.get_ram(), dtype=np.uint8)
                                    ram_search_candidates = ram_narrow_hits(
                                        ram, ram_search_candidates, search_val
                                    )
                                    print_ram_narrow_hits(ram, ram_search_candidates)
                        elif event.key in (pygame.K_LEFTBRACKET, pygame.K_COMMA, pygame.K_MINUS):
                            # [ , - all slow down (comma/minus as layout fallbacks)
                            speed_idx = max(0, speed_idx - 1)
                            self._display_speed = speed_levels[speed_idx]
                            print(f"[SPEED] {speed_levels[speed_idx]}x", flush=True)
                        elif event.key in (pygame.K_RIGHTBRACKET, pygame.K_PERIOD, pygame.K_EQUALS, pygame.K_PLUS):
                            # ] . = + all speed up
                            speed_idx = min(len(speed_levels) - 1, speed_idx + 1)
                            self._display_speed = speed_levels[speed_idx]
                            print(f"[SPEED] {speed_levels[speed_idx]}x", flush=True)
                        elif event.key == pygame.K_F1 and self.record_name:
                            print("[REC] F1 save is supported as an alias; use F5 for new recordings.")
                            self._save_recording(env, game_state)
                            running = False
                        elif event.key == pygame.K_p:
                            pos = get_pos_from_ram(env.get_ram())
                            tx, ty = pos.x // TILE_SIZE, pos.y // TILE_SIZE
                            self.bot.clearer.pathfinder.no_go_tiles.add((tx, ty))
                            print(f"[NO_GO] ({tx},{ty})")

            keys = pygame.key.get_pressed() if screen else pygame.key.ScancodeWrapper([0] * 512)

            if self.hotswap_cooldown > 0:
                self.hotswap_cooldown -= 1
            elif check_hotswap_chord(joystick, keys):
                self.mode = 'bot' if self.mode == 'human' else 'human'
                self.bot.enabled = (self.mode == 'bot')
                self._invalidate_hud_counts()
                if screen:
                    pygame.display.set_caption(f"Harvest Moon [{self.mode.upper()}]")
                print(f"[HOTSWAP] {self.mode.upper()}")
                self.hotswap_cooldown = 30
                if self.mode == 'bot':
                    self.bot.prepare_for_enable()
                    self._start_hotswap_cancel()

            speed = speed_levels[speed_idx]
            self._display_speed = speed
            tab_turbo = bool(screen and keys[pygame.K_TAB])
            if screen:
                repeat, tick_fps, skip_presents = bot_speed_timing(
                    speed, turbo=tab_turbo, bot=self.mode == "bot"
                )
            else:
                repeat, tick_fps, skip_presents = 1, 0, True

            stop = False
            action = np.zeros(12, dtype=np.int32)
            terminated = truncated = False
            for sub in range(repeat):
                if self.mode == 'human':
                    action = np.zeros(12, dtype=np.int32)
                    get_keyboard_action(keys, action)
                    get_controller_action(joystick, action)
                    sanitize_action(action)
                else:
                    action = self._bot_mode_action(env, game_state, obs)
                self._note_task_state_for_hud()

                render_this_frame = bool(
                    screen
                    and sub == repeat - 1
                    and (
                        not skip_presents
                        or self.frame_count % preview_interval(speed) == 0
                    )
                )

                obs, reward, terminated, truncated, info = gym_env_step(
                    env,
                    action,
                    obs,
                    update_obs=render_this_frame,
                )

                record_frame = None
                if self.record_name:
                    record_frame = len(self.recorded_frames)
                    self.recorded_frames.append(action.tolist())

                # Keep optional cheats topped up for autoplay. Stamina is real by
                # default so hot-spring refill / clear exhaustion can be verified;
                # re-enable with INFINITE_STAMINA=1.
                try:
                    if os.getenv("INFINITE_STAMINA", "").lower() in ("1", "true", "yes"):
                        self._set_live_value(env, "stamina", 100, 0x4918)
                    if self.bot.grass_seed_hack:
                        self._set_live_value(env, "grass_seeds", 99, 0x4927)
                    if self.bot.crop_seed_hack:
                        if os.getenv("FULL_WATER_CAN_HACK", "").lower() in ("1", "true", "yes"):
                            self._set_live_value(env, "water_can", 20, 0x4926)
                        self._sync_active_item(env)
                    if self.money_hack is not None:
                        self._apply_money_hack(env)
                    if self.ram_patches:
                        self._apply_ram_patches(env)
                except Exception as e:
                    if self.frame_count <= 5:
                        import traceback
                        print(f"[HACK ERR] {e}")
                        traceback.print_exc()

                if self.record_name and record_frame is not None:
                    self.recorded_trace.append(
                        recording_trace_entry(
                            env.get_ram(),
                            frame=record_frame,
                            action=action,
                        )
                    )

                self.frame_count += 1
                game_state = GameState(info, env.get_ram())

                if game_state.day != last_day:
                    print(f"[DAY] {game_state.date_str}")
                    if self.autoplay:
                        self._append_event(
                            {
                                "event": "day_change",
                                "frame": self.frame_count,
                                "date": game_state.date_str,
                                "time": game_state.time_str,
                            }
                        )
                    last_day = game_state.day

                if self.frame_count % 300 == 0:
                    if getattr(self.bot, "power_on_enabled", False) and not getattr(self.bot, "power_on_done", True):
                        task = getattr(self.bot, "power_on_task", None)
                        phase = task.phase_text if task is not None else "power-on"
                        print(f"[BOT] f={self.frame_count} {phase} speed={speed:g}x")
                    elif getattr(self.bot, "d1_handoff_enabled", False) and not getattr(self.bot, "d1_handoff_done", True):
                        task = getattr(self.bot, "d1_handoff_task", None)
                        phase = (
                            task.progress_snapshot().phase_text
                            if task is not None
                            else "d1-handoff"
                        )
                        print(f"[BOT] f={self.frame_count} D1 handoff {phase} speed={speed:g}x")
                    elif self.bot.day_plan_enabled and not self.bot.day_plan_done:
                        dp = self.bot.day_plan_task
                        print(f"[BOT] f={self.frame_count} day_plan {dp.phase_text} {dp.progress_text}")
                    elif self.bot.crop_enabled and not self.bot.crop_task_done:
                        ct = self.bot.crop_task
                        print(f"[BOT] f={self.frame_count} {ct.phase_text} {ct.progress_text}")
                    elif self.bot.grass_enabled and not self.bot.grass_task_done:
                        gt = self.bot.grass_task
                        print(f"[BOT] f={self.frame_count} {gt.phase_text} {gt.progress_text}")
                    else:
                        print(f"[BOT] f={self.frame_count} {self.bot.clearer.state}")
                    sys.stdout.flush()

                # Auto-save state when day plan or crop task completes
                if self.save_end and not self._end_saved and (self.bot.day_plan_done or self.bot.crop_task_done):
                    suffix = "day_plan_end" if self.bot.day_plan_done else "crop_end"
                    save_name = f"{self.initial_state}_{suffix}" if self.initial_state else suffix
                    save_path = os.path.join(STATES_DIR, f"{save_name}.state")
                    state_data = env.em.get_state()
                    with gzip.open(save_path, 'wb') as f:
                        f.write(state_data)
                    self._end_saved = True
                    print(f"[SAVED] {save_name} -> {save_path}")

                if screen and render_this_frame:
                    surf = pygame.surfarray.make_surface(obs.swapaxes(0, 1))
                    scaled = pygame.transform.scale(surf, (w * self.scale, h * self.scale))
                    screen.fill((0, 0, 0))
                    self._draw_hud(screen, font, env, game_state, action, h * self.scale)
                    screen.blit(scaled, (self.hud_width, 0))

                    color = (0, 255, 0) if self.mode == 'human' else (255, 100, 100)
                    mode_text = font.render(f"[{self.mode.upper()}]", True, color)
                    screen.blit(mode_text, (self.hud_width + w * self.scale - 70, 5))

                    pygame.display.flip()
                elif screen:
                    pygame.event.pump()

                if self._handle_autoplay_terminal_disable(env, game_state, action, obs):
                    stop = True
                    break
                if self._check_autoplay_watchdog(env, game_state, action, obs):
                    stop = True
                    break
                if terminated or truncated:
                    stop = True
                    break
                if self.max_frames is not None and self.frame_count >= self.max_frames:
                    print(f"[STOP] max_frames={self.max_frames}")
                    stop = True
                    break

            if screen:
                pace_present(tick_fps, self)
            if stop:
                break

        env.close()
        pygame.quit()
        print(f"\nFrames: {self.frame_count}")

    def _set_live_value(self, env, key: str, value: int, memory_addr: Optional[int] = None):
        """Write a live integration value with a raw-memory fallback."""
        try:
            LiveRamEditor(env).set_field(key, value)
            return
        except KeyError:
            pass
        try:
            env.data.set_value(key, value)
            return
        except Exception:
            pass
        if memory_addr is None:
            raise RuntimeError(f"Could not set live RAM field {key!r}")
        env.data.memory.assign(memory_addr, "|u1", value)

    def _apply_money_hack(self, env):
        """Write displayed money to live RAM."""
        sram_val = LiveRamEditor(env).set_field("money", self.money_hack)

        if self.frame_count <= 3:
            actual = read_ram_value(env.get_ram(), "money", raw=True)
            print(f"[MONEY DBG] wrote {sram_val} -> money_raw={actual} (={actual*10}g)")

    def _apply_ram_patches(self, env):
        LiveRamEditor(env).apply(self.ram_patches)

    def _save_recording(self, env, game_state: GameState):
        del game_state
        write_task_recording(
            name=self.record_name,
            frames=self.recorded_frames,
            trace=self.recorded_trace,
            start_state=self.initial_state,
            tasks_dir=TASKS_DIR,
            states_dir=STATES_DIR,
            end_state=env.em.get_state(),
        )
