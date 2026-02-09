#!/usr/bin/env python3
"""
Harvest Moon SNES Bot

Usage:
    python harvest_bot.py play --autoplay --state Y1_Spring_Day01_06h00m
    python harvest_bot.py play --priority "rock,stump"
    python harvest_bot.py list
"""

import os
import sys
import glob
import gzip
import argparse
from datetime import datetime
from typing import Optional, Dict, List

# Add parent directory for retro_harness import
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
import stable_retro as retro

from retro_harness import (
    init_controller as _init_controller,
    controller_action,
    keyboard_action,
    sanitize_action,
    TaskStatus,
    WorldState,
    SNES_L, SNES_R, SNES_SELECT,
)
from farm_clearer import (
    FarmClearer, DebrisType, Tool, Point,
    parse_priority_list, make_action,
    ADDR_TOOL, ADDR_TILEMAP, ADDR_INPUT_LOCK, TILE_SIZE, get_pos_from_ram,
)
from fence_flow import FenceClearLoopTask
from grass_planter import GrassPlantTask, DEFAULT_BOUNDS as GRASS_DEFAULT_BOUNDS, DEFAULT_NO_GO_RECTS as GRASS_DEFAULT_NO_GO
from crop_planter import CropWaterTask, SEED_DATA_KEY, SEED_ITEM, DEFAULT_CROP_BOUNDS
from day_plan import DayPlanTask


# Wrapper for init_controller (retro_harness version takes pygame arg)
def init_controller():
    return _init_controller(pygame)


# Wrapper for controller input (retro_harness uses different name/signature)
def get_controller_action(joystick, action):
    controller_action(joystick, action)


# Wrapper for keyboard input (retro_harness uses different signature)
def get_keyboard_action(keys, action):
    keyboard_action(keys, action, pygame)


# Harvest-specific: Hotswap chord detection (L+R+SELECT)
HOTSWAP_KEYS = {pygame.K_a, pygame.K_s, pygame.K_TAB}

def check_hotswap_chord(joystick, keys):
    """Check if hotswap chord (L+R+SELECT) is pressed."""
    if all(keys[k] for k in HOTSWAP_KEYS):
        return True
    if joystick is not None:
        try:
            l_pressed = joystick.get_button(4) if joystick.get_numbuttons() > 4 else False
            r_pressed = joystick.get_button(5) if joystick.get_numbuttons() > 5 else False
            sel_pressed = joystick.get_button(6) if joystick.get_numbuttons() > 6 else False
            if l_pressed and r_pressed and sel_pressed:
                return True
        except:
            pass
    return False


def print_controls(joystick=None):
    """Print Harvest Moon control scheme."""
    print("\nControls:")
    if joystick:
        print(f"  Controller: {joystick.get_name()}")
        print("    D-Pad/Stick: Movement")
        print("    A: Confirm | B: Cancel | X: Menu | Y: Use Item")
        print("    LB/RB: Cycle Items")
        print("    LB+RB+SELECT: Toggle Human/Bot Mode")
    print("  Keyboard:")
    print("    Arrows: D-Pad")
    print("    Z: Cancel (B) | C: Confirm (A) | V: Menu (X) | X: Use Item (Y)")
    print("    A/S: Cycle Items (L/R)")
    print("    A+S+TAB: Toggle Human/Bot Mode")
    print("    P: Mark current tile as no-go (debug)")

# Paths (SCRIPT_DIR defined above for retro_harness import)
INTEGRATION_PATH = os.path.join(SCRIPT_DIR, "custom_integrations")
STATES_DIR = os.path.join(INTEGRATION_PATH, "HarvestMoon-Snes")
SAVES_DIR = os.path.join(SCRIPT_DIR, "saves")
TASKS_DIR = os.path.join(SCRIPT_DIR, "tasks")

os.makedirs(SAVES_DIR, exist_ok=True)
retro.data.Integrations.add_custom_path(INTEGRATION_PATH)

# Game constants
SEASON_NAMES = {0: "Spring", 1: "Summer", 2: "Fall", 3: "Winter"}
DAY_NAMES = {0: "Sun", 1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat"}
ITEM_NAMES = {
    0x00: "Empty",
    0x01: "Sickle", 0x02: "Hoe", 0x03: "Hammer", 0x04: "Axe",
    0x10: "Watering Can",
    0x11: "Gold Sickle", 0x12: "Gold Hoe", 0x13: "Gold Hammer", 0x14: "Gold Axe",
}


class GameState:
    """Parse game state from info dict."""

    def __init__(self, info: Dict):
        self.year = info.get('year', 1)
        self.season = info.get('season', 0)
        self.day_of_week = info.get('day_of_week', 0)
        self.day = info.get('day', 1)
        self.hour = info.get('hour', 6)
        self.minute = info.get('minute', 0)
        self.stamina = info.get('stamina', 0)
        self.item = info.get('item_in_hand', 0)

        if 'money_bcd_lo' in info and 'money_bcd_hi' in info:
            lo, hi = info.get('money_bcd_lo', 0), info.get('money_bcd_hi', 0)
            self.money = (lo & 0x0F) + ((lo >> 4) & 0x0F) * 10 + (hi & 0x0F) * 100 + ((hi >> 4) & 0x0F) * 1000
        else:
            self.money = info.get('money_lo', 0) + (info.get('money_mid', 0) << 8) + (info.get('money_hi', 0) << 16)

    @property
    def season_name(self) -> str:
        return SEASON_NAMES.get(self.season, "?")

    @property
    def day_name(self) -> str:
        return DAY_NAMES.get(self.day_of_week, "?")

    @property
    def item_name(self) -> str:
        return ITEM_NAMES.get(self.item, f"0x{self.item:02X}")

    @property
    def date_str(self) -> str:
        return f"Y{self.year} {self.season_name} {self.day} ({self.day_name})"

    @property
    def time_str(self) -> str:
        return f"{self.hour}:00 AM" if self.hour <= 12 else f"{self.hour-12}:00 PM"

    def state_name(self) -> str:
        return f"Y{self.year}_{self.season_name}_Day{self.day:02d}_{self.hour:02d}h{self.minute:02d}m"


class AutoClearBot:
    """Farm clearing bot."""

    def __init__(
        self,
        priority: Optional[List[DebrisType]] = None,
        clear_fences_first: Optional[bool] = None,
        clear_fences_only: bool = False,
        grass_enabled: bool = False,
        till_only: bool = False,
        grass_bounds: Optional[tuple] = None,
        grass_no_go: Optional[List[tuple]] = None,
        grass_seed_hack: bool = False,
        crop_enabled: bool = False,
        crop_seed_type: str = "potato",
        day_plan_enabled: bool = False,
    ):
        self.clearer = FarmClearer(priority=priority)
        self.clearer.tasks_dir = TASKS_DIR
        self.clearer.configure(
            prefer_lift_for_weeds=True,
            prefer_lift_for_stones=False,
        )

        self.env = None
        self.enabled = False
        self.disable_reason: Optional[str] = None
        self.frame_count = 0
        self.initial_tilemap: Optional[int] = None
        self.map_locked = False
        self.map_hist: Dict[int, int] = {}
        self.fence_task = FenceClearLoopTask(max_fences=None)
        self.fence_task_started = False
        self.fence_task_done = False
        if clear_fences_first is None:
            clear_fences_first = not os.getenv("SKIP_FENCE_TOSS", "").lower() in ("1", "true", "yes")
        self.fence_task_enabled = clear_fences_first
        self.fence_task_only = clear_fences_only

        # Grass planting
        self.grass_enabled = grass_enabled
        self.grass_seed_hack = grass_seed_hack or grass_enabled
        self.grass_task = GrassPlantTask(
            bounds=grass_bounds or GRASS_DEFAULT_BOUNDS,
            no_go_rects=grass_no_go if grass_no_go is not None else list(GRASS_DEFAULT_NO_GO),
            till_only=till_only,
        )
        self.grass_task_started = False
        self.grass_task_done = False

        # Crop planting + watering
        self.crop_enabled = crop_enabled
        self.crop_seed_type = crop_seed_type
        self.crop_seed_hack = crop_enabled
        self.crop_task = CropWaterTask(
            seed_type=crop_seed_type,
            bounds=DEFAULT_CROP_BOUNDS,
        )
        self.crop_task_started = False
        self.crop_task_done = False

        # Day plan mode
        self.day_plan_enabled = day_plan_enabled
        self.day_plan_task = DayPlanTask(seed_type=crop_seed_type)
        self.day_plan_started = False
        self.day_plan_done = False

        # Skip startup tasks in day plan mode
        if day_plan_enabled:
            self.crop_seed_hack = True  # ensure seeds available for CropWaterTask
            self.grass_seed_hack = False
            self.fence_task_enabled = False
            self.fence_task_done = True

        # Skip startup tasks in crop mode (no hammer/axe needed)
        if crop_enabled:
            self.fence_task_enabled = False
            self.fence_task_done = True
        else:
            # Add startup tasks
            if not os.getenv("SKIP_HAMMER", "").lower() in ("1", "true", "yes"):
                get_hammer_path = os.path.join(TASKS_DIR, "get_hammer.json")
                shed_grab_path = os.path.join(TASKS_DIR, "shed_grab_hammer_smash_rock.json")
                if os.path.exists(get_hammer_path):
                    self.clearer.add_startup_task("task", name="get_hammer")
                elif os.path.exists(shed_grab_path):
                    self.clearer.add_startup_task("nav", name="go_shed", target=Point(342, 489), radius=12, timeout=1800)
                    self.clearer.add_startup_task("task", name="shed_grab_hammer_smash_rock")

            if not os.getenv("SKIP_AXE", "").lower() in ("1", "true", "yes"):
                get_axe_path = os.path.join(TASKS_DIR, "get_axe.json")
                if os.path.exists(get_axe_path):
                    self.clearer.add_startup_task("task", name="get_axe")

    def set_env(self, env):
        self.env = env

    def disable(self, reason: str):
        self.disable_reason = reason
        self.enabled = False
        print(f"[BOT] Disabled: {reason}")

    def get_goal_text(self) -> str:
        if self.day_plan_enabled and not self.day_plan_done:
            if self.day_plan_started:
                return f"Goal: day plan {self.day_plan_task.phase_text} ({self.day_plan_task.progress_text})"
            return "Goal: day plan (waiting)"
        if self.fence_task_enabled and not self.fence_task_done:
            return "Goal: clear fences"
        if self.crop_enabled and not self.crop_task_done:
            if self.crop_task_started:
                return f"Goal: crop {self.crop_task.phase_text} ({self.crop_task.progress_text})"
            return "Goal: crop (waiting)"
        if self.grass_enabled and not self.grass_task_done:
            if self.grass_task_started:
                return f"Goal: grass {self.grass_task.phase_text} ({self.grass_task.progress_text})"
            return "Goal: grass (waiting)"
        if not self.clearer.startup_done:
            idx = self.clearer.startup_index
            if idx < len(self.clearer.startup_tasks):
                return f"Goal: {self.clearer.startup_tasks[idx].get('name', 'startup')}"
            return "Goal: startup"
        if self.clearer.current_target:
            t = self.clearer.current_target
            return f"Goal: {t.debris_type.name} at {t.tile}"
        return f"Goal: {self.clearer.state}"

    def get_action(self, game_state: GameState, obs: np.ndarray) -> np.ndarray:
        if not self.enabled or self.env is None:
            return np.zeros(12, dtype=np.int32)

        self.frame_count += 1
        ram = self.env.get_ram()
        world = WorldState(frame=self.frame_count, ram=ram, info={}, obs=obs)

        if self.fence_task_enabled and not self.fence_task_done:
            if not self.fence_task_started:
                if self.fence_task.can_start(world):
                    self.fence_task.reset(world)
                    self.fence_task_started = True
                    print("[BOT] Fence clear: start")
                else:
                    self.fence_task_done = True
                    print("[BOT] Fence clear: missing recording")

            if self.fence_task_started and not self.fence_task_done:
                result = self.fence_task.step(world)
                if result.action is not None:
                    return result.action.action
                if result.status == TaskStatus.SUCCESS:
                    print(f"[BOT] Fence clear: complete ({self.fence_task.cleared_count})")
                    self.fence_task_done = True
                elif result.status in (TaskStatus.FAILURE, TaskStatus.BLOCKED):
                    reason = result.reason or result.status.value
                    print(f"[BOT] Fence clear: stopped ({reason})")
                    self.fence_task_done = True
                if self.fence_task_only and self.fence_task_done:
                    self.disable("Fence-only complete")
                return np.zeros(12, dtype=np.int32)
        elif self.fence_task_only:
            self.disable("Fence-only complete")
            return np.zeros(12, dtype=np.int32)

        # Day plan mode: runs before map lock since it traverses multiple tilemaps
        if self.day_plan_enabled and not self.day_plan_done:
            if not self.day_plan_started:
                if self.day_plan_task.can_start(world):
                    self.day_plan_task.reset(world)
                    self.day_plan_started = True
                    print("[BOT] Day plan: start")
                else:
                    self.day_plan_done = True
                    print("[BOT] Day plan: cannot start")

            if self.day_plan_started and not self.day_plan_done:
                result = self.day_plan_task.step(world)
                if result.action is not None:
                    return result.action.action
                if result.status == TaskStatus.SUCCESS:
                    print(f"[BOT] Day plan: complete ({self.day_plan_task.progress_text})")
                    self.day_plan_done = True
                elif result.status in (TaskStatus.FAILURE, TaskStatus.BLOCKED):
                    reason = result.reason or result.status.value
                    print(f"[BOT] Day plan: stopped ({reason})")
                    self.day_plan_done = True
                if self.day_plan_done:
                    self.disable(f"Day plan complete ({self.day_plan_task.progress_text})")
                return np.zeros(12, dtype=np.int32)

        # Lock to initial map after warmup
        tilemap = ram[ADDR_TILEMAP] if ADDR_TILEMAP < len(ram) else 0
        if not self.map_locked:
            self.map_hist[tilemap] = self.map_hist.get(tilemap, 0) + 1
            if self.frame_count >= 180:
                nonzero = {k: v for k, v in self.map_hist.items() if k != 0}
                self.initial_tilemap = max((nonzero or self.map_hist).items(), key=lambda kv: kv[1])[0]
                self.map_locked = True
                print(f"[BOT] Map locked: 0x{self.initial_tilemap:02X}")

        if self.map_locked and tilemap != self.initial_tilemap:
            self.disable(f"Map changed to 0x{tilemap:02X}")
            return np.zeros(12, dtype=np.int32)

        # Crop mode: detect plots, plant + water
        if self.crop_enabled and not self.crop_task_done:
            if not self.crop_task_started:
                if self.crop_task.can_start(world):
                    self.crop_task.reset(world)
                    self.crop_task_started = True
                    print("[BOT] Crop task: start")
                else:
                    self.crop_task_done = True
                    print("[BOT] Crop task: cannot start")

            if self.crop_task_started and not self.crop_task_done:
                result = self.crop_task.step(world)
                if result.action is not None:
                    return result.action.action
                if result.status == TaskStatus.SUCCESS:
                    print(f"[BOT] Crop task: complete ({self.crop_task.progress_text})")
                    self.crop_task_done = True
                elif result.status in (TaskStatus.FAILURE, TaskStatus.BLOCKED):
                    reason = result.reason or result.status.value
                    print(f"[BOT] Crop task: stopped ({reason})")
                    self.crop_task_done = True
                if self.crop_task_done:
                    self.disable(f"Crop complete ({self.crop_task.progress_text})")
                return np.zeros(12, dtype=np.int32)

        # If grass mode, run grass task instead of (or after) clearing
        if self.grass_enabled and not self.grass_task_done:
            if not self.grass_task_started:
                if self.grass_task.can_start(world):
                    self.grass_task.reset(world)
                    self.grass_task_started = True
                    print("[BOT] Grass planter: start")
                else:
                    self.grass_task_done = True
                    print("[BOT] Grass planter: cannot start")

            if self.grass_task_started and not self.grass_task_done:
                result = self.grass_task.step(world)
                if result.action is not None:
                    return result.action.action
                if result.status == TaskStatus.SUCCESS:
                    print(f"[BOT] Grass planter: complete ({self.grass_task.progress_text})")
                    self.grass_task_done = True
                elif result.status in (TaskStatus.FAILURE, TaskStatus.BLOCKED):
                    reason = result.reason or result.status.value
                    print(f"[BOT] Grass planter: stopped ({reason})")
                    self.grass_task_done = True
                if self.grass_task_done:
                    self.disable(f"Grass complete ({self.grass_task.progress_text})")
                return np.zeros(12, dtype=np.int32)

        action = self.clearer.tick(ram)
        if action is None:
            self.disable(f"Complete ({self.clearer.cleared_count} cleared)")
            return np.zeros(12, dtype=np.int32)

        return action


class PlaySession:
    """Interactive play session."""

    def __init__(
        self,
        state: Optional[str] = None,
        scale: int = 3,
        bot: Optional[AutoClearBot] = None,
        autoplay: bool = False,
        max_frames: Optional[int] = None,
        record_name: Optional[str] = None,
        save_end: bool = False,
    ):
        self.initial_state = state
        self.scale = scale
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
        self.save_end = save_end
        self._end_saved = False

    def run(self):
        pygame.init()

        env_kwargs = {
            "game": "HarvestMoon-Snes",
            "inttype": retro.data.Integrations.ALL,
            "use_restricted_actions": retro.Actions.ALL,
            "render_mode": "rgb_array"
        }
        if self.initial_state:
            env_kwargs["state"] = self.initial_state

        try:
            env = retro.make(**env_kwargs)
        except Exception as e:
            print(f"Error: {e}")
            return

        obs, info = env.reset()
        self.bot.set_env(env)

        h, w = obs.shape[0], obs.shape[1]
        screen = None
        try:
            screen = pygame.display.set_mode((w * self.scale, h * self.scale))
            pygame.display.set_caption(f"Harvest Moon [{self.mode.upper()}]")
        except pygame.error:
            self.mode = 'bot'
            self.bot.enabled = True

        clock = pygame.time.Clock()
        font = pygame.font.SysFont('monospace', 11)
        joystick = init_controller()

        print(f"\n[HARVEST BOT] mode={self.mode.upper()}", end="")
        if self.record_name:
            print(f" record={self.record_name}", end="")
        print()

        running = True
        game_state = GameState(info)
        last_day = game_state.day
        quick_save = None
        speed_levels = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]
        speed_idx = 2

        while running:
            if screen:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        elif event.key == pygame.K_F5:
                            quick_save = env.em.get_state()
                            save_path = os.path.join(STATES_DIR, f"{game_state.state_name()}.state")
                            with gzip.open(save_path, 'wb') as f:
                                f.write(quick_save)
                            print(f"[SAVED] {game_state.state_name()}")
                        elif event.key == pygame.K_F9 and quick_save:
                            env.em.set_state(quick_save)
                            print("[LOADED]")
                        elif event.key == pygame.K_LEFTBRACKET:
                            speed_idx = max(0, speed_idx - 1)
                            print(f"[SPEED] {speed_levels[speed_idx]}x")
                        elif event.key == pygame.K_RIGHTBRACKET:
                            speed_idx = min(len(speed_levels) - 1, speed_idx + 1)
                            print(f"[SPEED] {speed_levels[speed_idx]}x")
                        elif event.key == pygame.K_F1 and self.record_name:
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
                if screen:
                    pygame.display.set_caption(f"Harvest Moon [{self.mode.upper()}]")
                print(f"[HOTSWAP] {self.mode.upper()}")
                self.hotswap_cooldown = 30
                if self.mode == 'bot':
                    self.hotswap_cancel_frames = 90
                    self.hotswap_cancel_until_clear = True

            if self.mode == 'human':
                action = np.zeros(12, dtype=np.int32)
                get_keyboard_action(keys, action)
                get_controller_action(joystick, action)
                sanitize_action(action)
            else:
                input_lock = env.get_ram()[ADDR_INPUT_LOCK] if ADDR_INPUT_LOCK < len(env.get_ram()) else 1
                if self.hotswap_cancel_until_clear and input_lock != 1:
                    action = make_action(b=self.frame_count % 2 == 0, a=self.frame_count % 2 == 1)
                elif self.hotswap_cancel_frames > 0:
                    action = make_action(b=self.frame_count % 2 == 0)
                    self.hotswap_cancel_frames -= 1
                    if self.hotswap_cancel_frames == 0:
                        self.hotswap_cancel_until_clear = False
                else:
                    action = self.bot.get_action(game_state, obs)
                if not self.bot.enabled and self.bot.disable_reason:
                    self.mode = 'human'
                    if screen:
                        pygame.display.set_caption(f"Harvest Moon [{self.mode.upper()}]")

            obs, reward, terminated, truncated, info = env.step(action)

            if self.record_name:
                self.recorded_frames.append(action.tolist())
            
            try:
                # Direct data set is most reliable in retro
                env.data.set_value("stamina", 100)
                if self.bot.grass_seed_hack:
                    env.data.set_value("grass_seeds", 99)
                if self.bot.crop_seed_hack:
                    seed_key = SEED_DATA_KEY.get(self.bot.crop_seed_type, "potato_seeds")
                    env.data.set_value(seed_key, 99)
                    # Restore tool slots if recording corrupted them
                    ram = env.get_ram()
                    if int(ram[ADDR_TOOL]) == 0x00:
                        env.data.set_value("item_in_hand", SEED_ITEM.get(self.bot.crop_seed_type, 0x07))
                    if int(ram[0x0923]) != 0x10:  # item_in_hand_alt not watering can
                        env.data.set_value("item_in_hand_alt", 0x10)
                    env.data.set_value("water_can", 20)  # ensure watering can ownership
            except Exception:
                # Fallbacks
                try:
                    if hasattr(env.unwrapped, 'set_ram'):
                        env.unwrapped.set_ram(0x0918, 100)
                    elif hasattr(env.unwrapped, 'memory'):
                        env.unwrapped.memory.set_u8(0x0918, 100)
                except Exception:
                    pass

            self.frame_count += 1
            game_state = GameState(info)

            if game_state.day != last_day:
                print(f"[DAY] {game_state.date_str}")
                last_day = game_state.day

            if self.frame_count % 300 == 0:
                if self.bot.day_plan_enabled and not self.bot.day_plan_done:
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

            fast_forward = keys[pygame.K_TAB] if screen else True

            if screen and not fast_forward:
                surf = pygame.surfarray.make_surface(obs.swapaxes(0, 1))
                scaled = pygame.transform.scale(surf, (w * self.scale, h * self.scale))
                screen.blit(scaled, (0, 0))

                lines = [
                    f"{game_state.date_str} {game_state.time_str}",
                    f"${game_state.money:,} | Stamina: {game_state.stamina}",
                    self.bot.get_goal_text(),
                ]
                # Player position + target info for debugging
                if self.bot.day_plan_enabled and self.bot.day_plan_started and not self.bot.day_plan_done:
                    pos = get_pos_from_ram(env.get_ram())
                    tx, ty = pos.x // TILE_SIZE, pos.y // TILE_SIZE
                    lines.append(f"Pos: ({tx},{ty}) px=({pos.x},{pos.y})")
                elif self.bot.crop_enabled and self.bot.crop_task_started and not self.bot.crop_task_done:
                    ct = self.bot.crop_task
                    player_tile = ct._navigator.current_tile
                    player_pos = ct._navigator.current_pos
                    lines.append(f"Pos: ({player_tile[0]},{player_tile[1]}) px=({player_pos.x},{player_pos.y})")
                    if ct._target_tile:
                        lines.append(f"Target: ({ct._target_tile[0]},{ct._target_tile[1]}) plot={ct._plot_index + 1}/{len(ct._plots)}")
                elif self.bot.grass_enabled and self.bot.grass_task_started and not self.bot.grass_task_done:
                    gt = self.bot.grass_task
                    player_tile = gt._navigator.current_tile
                    player_pos = gt._navigator.current_pos
                    lines.append(f"Pos: ({player_tile[0]},{player_tile[1]}) px=({player_pos.x},{player_pos.y})")
                    if gt._target_tile:
                        lines.append(f"Target: ({gt._target_tile[0]},{gt._target_tile[1]}) chunk={gt._chunk_origin}")
                else:
                    player_tile = self.bot.clearer.navigator.current_tile
                    player_pos = self.bot.clearer.navigator.current_pos
                    lines.append(f"Pos: ({player_tile[0]},{player_tile[1]}) px=({player_pos.x},{player_pos.y})")
                    if self.bot.clearer.current_target:
                        t = self.bot.clearer.current_target
                        lines.append(f"Target: {t.debris_type.name} @ ({t.tile[0]},{t.tile[1]}) id=0x{t.tile_id:02X}")
                # Add active buttons display
                btn_names = ["B", "Y", "Sel", "St", "Up", "Dn", "Lt", "Rt", "A", "X", "L", "R"]
                active_btns = [btn_names[i] for i, v in enumerate(action) if v > 0]
                if active_btns:
                    lines.append(f"Buttons: {' '.join(active_btns)}")
                
                for i, line in enumerate(lines):
                    text = font.render(line, True, (255, 255, 255))
                    screen.blit(text, (5, 5 + i * 14))

                color = (0, 255, 0) if self.mode == 'human' else (255, 100, 100)
                mode_text = font.render(f"[{self.mode.upper()}]", True, color)
                screen.blit(mode_text, (w * self.scale - 70, 5))

                pygame.display.flip()
                clock.tick(int(60 * speed_levels[speed_idx]))
            elif screen:
                pygame.event.pump()

            if terminated or truncated:
                break
            if self.max_frames is not None and self.frame_count >= self.max_frames:
                print(f"[STOP] max_frames={self.max_frames}")
                break

        env.close()
        pygame.quit()
        print(f"\nFrames: {self.frame_count}")

    def _save_recording(self, env, game_state: GameState):
        import json as _json

        os.makedirs(TASKS_DIR, exist_ok=True)
        task_data = {
            "name": self.record_name,
            "frames": self.recorded_frames,
            "start_state": self.initial_state,
            "metadata": {
                "frame_count": len(self.recorded_frames),
                "duration_seconds": len(self.recorded_frames) / 60.0,
            },
        }
        path = os.path.join(TASKS_DIR, f"{self.record_name}.json")
        with open(path, "w") as f:
            _json.dump(task_data, f, indent=2)
        print(f"[REC] Saved task: {path} ({len(self.recorded_frames)} frames)")

        end_state = env.em.get_state()
        state_path = os.path.join(STATES_DIR, f"{self.record_name}_end.state")
        with gzip.open(state_path, "wb") as f:
            f.write(end_state)
        print(f"[REC] Saved end state: {state_path}")


def list_states():
    states = sorted(glob.glob(os.path.join(STATES_DIR, "*.state")))
    print("\nSAVE STATES:")
    for path in states:
        name = os.path.basename(path).replace(".state", "")
        dt = datetime.fromtimestamp(os.path.getmtime(path)).strftime('%Y-%m-%d %H:%M')
        print(f"  {name} ({dt})")
    print(f"Total: {len(states)}")


def main():
    parser = argparse.ArgumentParser(description="Harvest Moon Bot")
    subparsers = parser.add_subparsers(dest='command')

    play = subparsers.add_parser('play')
    play.add_argument('--state', type=str)
    play.add_argument('--scale', type=int, default=3)
    play.add_argument('--autoplay', action='store_true')
    play.add_argument('--priority', type=str)
    play.add_argument('--priority-only', action='store_true')
    play.add_argument('--fence-only', action='store_true', help='Only clear fences then stop')
    play.add_argument('--grass', action='store_true', help='Enable grass planting (till + plant)')
    play.add_argument('--till-only', action='store_true', help='Only till, skip planting')
    play.add_argument('--grass-bounds', type=str, default=None, help='Custom bounds x1,y1,x2,y2 (default: right half)')
    play.add_argument('--grass-no-go', type=str, default=None, help='No-go rects: x1,y1,x2,y2;x1,y1,x2,y2 (areas to skip)')
    play.add_argument('--crop', action='store_true', help='Crop mode: detect plots, plant + water')
    play.add_argument('--seed', type=str, default='potato', help='Seed type (potato, turnip, corn, tomato)')
    play.add_argument('--no-day-plan', action='store_true', help='Disable day plan (default: on unless --crop/--grass/--fence-only)')
    play.add_argument('--save-end', action='store_true', help='Save state when task completes')
    play.add_argument('--max-frames', type=int, default=None, help='Stop after N frames (testing)')
    play.add_argument('--record', type=str, default=None, metavar='NAME', help='Record inputs as a task (F1 to save)')

    subparsers.add_parser('list')

    args = parser.parse_args()

    if args.command == 'play':
        priority = parse_priority_list(getattr(args, 'priority', None), getattr(args, 'priority_only', False))
        grass_bounds = None
        if args.grass_bounds:
            parts = [int(x.strip()) for x in args.grass_bounds.split(",")]
            if len(parts) == 4:
                grass_bounds = tuple(parts)
        grass_no_go = None
        if args.grass_no_go:
            grass_no_go = []
            for rect_str in args.grass_no_go.split(";"):
                rect_str = rect_str.strip()
                if not rect_str:
                    continue
                parts = [int(x.strip()) for x in rect_str.split(",")]
                if len(parts) == 4:
                    grass_no_go.append(tuple(parts))
        bot = AutoClearBot(
            priority=priority,
            clear_fences_only=bool(args.fence_only),
            grass_enabled=bool(args.grass) or bool(args.till_only),
            till_only=bool(args.till_only),
            grass_bounds=grass_bounds,
            grass_no_go=grass_no_go,
            crop_enabled=bool(args.crop),
            crop_seed_type=args.seed,
            day_plan_enabled=not (args.no_day_plan or args.crop or args.grass or args.till_only or args.fence_only),
        )
        PlaySession(
            state=args.state,
            scale=args.scale,
            bot=bot,
            autoplay=args.autoplay,
            max_frames=args.max_frames,
            record_name=args.record,
            save_end=bool(getattr(args, 'save_end', False)),
        ).run()
    elif args.command == 'list':
        list_states()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
