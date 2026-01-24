#!/usr/bin/env python3
"""
Harvest Moon SNES Bot

Usage:
    python harvest_bot.py play --autoplay --state Y1_Spring_Day01_06h00m
    python harvest_bot.py play --priority "rock,stump"
    python harvest_bot.py list
"""

import os
import glob
import gzip
import argparse
from datetime import datetime
from typing import Optional, Dict, List

import numpy as np
import pygame
import stable_retro as retro

from controls import (
    init_controller, get_controller_action, get_keyboard_action,
    sanitize_action, check_hotswap_chord, print_controls,
)
from farm_clearer import (
    FarmClearer, DebrisType, Tool, Point,
    parse_priority_list, make_action,
    ADDR_TILEMAP, ADDR_INPUT_LOCK, TILE_SIZE, get_pos_from_ram,
)
from fence_flow import FenceClearLoopTask
from harness import TaskStatus, WorldState

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
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
        if self.fence_task_enabled and not self.fence_task_done:
            return "Goal: clear fences"
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

        print("\n" + "=" * 60)
        print("HARVEST MOON BOT")
        print("=" * 60)
        print_controls(joystick)
        print("  F5: Save | F9: Load | TAB: Fast Forward | ESC: Exit")
        print("  L+R+SELECT: Toggle Human/Bot")
        print("=" * 60)

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
            
            try:
                # Direct data set is most reliable in retro
                env.data.set_value("stamina", 100)
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

            if self.frame_count % 60 == 0:
                print(f"[BOT] Frame: {self.frame_count} Stamina: {game_state.stamina} State: {self.bot.clearer.state}")
                import sys; sys.stdout.flush()

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
    play.add_argument('--max-frames', type=int, default=None, help='Stop after N frames (testing)')

    subparsers.add_parser('list')

    args = parser.parse_args()

    if args.command == 'play':
        priority = parse_priority_list(getattr(args, 'priority', None), getattr(args, 'priority_only', False))
        bot = AutoClearBot(priority=priority, clear_fences_only=bool(args.fence_only))
        PlaySession(
            state=args.state,
            scale=args.scale,
            bot=bot,
            autoplay=args.autoplay,
            max_frames=args.max_frames,
        ).run()
    elif args.command == 'list':
        list_states()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
