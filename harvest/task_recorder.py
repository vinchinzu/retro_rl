#!/usr/bin/env python3
"""
Task Recorder for Harvest Moon Bot

Records human gameplay as repeatable "tasks" that the bot can execute.

Tasks are sequences of inputs that accomplish specific goals like:
- Pick berry and ship it
- Water all crops
- Feed chickens
- Go to bed

Usage:
    # Record a new task
    python task_recorder.py record ship_berry --state Y1_Spring_Day01_06h

    # List recorded tasks
    python task_recorder.py list

    # Replay a task
    python task_recorder.py replay ship_berry --state Y1_Spring_Day01_06h

    # Test task reliability (run N times)
    python task_recorder.py test ship_berry --runs 5
"""

import os
import sys
import json
import gzip
import time
import argparse
from datetime import datetime
from typing import Optional, List, Dict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

INTEGRATION_PATH = os.path.join(SCRIPT_DIR, "custom_integrations")
STATES_DIR = os.path.join(INTEGRATION_PATH, "HarvestMoon-Snes")
TASKS_DIR = os.path.join(SCRIPT_DIR, "tasks")

os.makedirs(TASKS_DIR, exist_ok=True)

import numpy as np
import pygame
import stable_retro as retro

from retro_harness import (
    init_controller as _init_controller,
    controller_action,
    keyboard_action,
    sanitize_action,
)


# Wrappers for retro_harness (different signatures)
def init_controller():
    return _init_controller(pygame)


def get_controller_action(joystick, action):
    controller_action(joystick, action)


def get_keyboard_action(keys, action):
    keyboard_action(keys, action, pygame)


def print_controls(joystick=None):
    """Print Harvest Moon control scheme for recording."""
    print("\nRecording Controls:")
    if joystick:
        print(f"  Controller: {joystick.get_name()}")
        print("    D-Pad/Stick: Movement")
        print("    A: Confirm | B: Cancel | X: Menu | Y: Use Item")
        print("    LB/RB: Cycle Items")
    print("  Keyboard:")
    print("    Arrows: D-Pad")
    print("    Z: Cancel (B) | X: Use Item (Y) | C: Confirm (A) | V: Menu (X)")
    print("    A/S: Cycle Items (L/R)")
    print("  Recording:")
    print("    F5: Save | Ctrl+S: Stop Recording")


retro.data.Integrations.add_custom_path(INTEGRATION_PATH)


class Task:
    # ... (Task class remains the same) ...
    """A recorded sequence of inputs."""

    def __init__(self, name: str):
        self.name = name
        self.frames: List[List[int]] = []  # List of action arrays
        self.start_state: Optional[str] = None
        self.end_state_data: Optional[bytes] = None
        self.metadata: Dict = {}

    def save(self, path: str):
        """Save task to file."""
        data = {
            "name": self.name,
            "frames": self.frames,
            "start_state": self.start_state,
            "metadata": self.metadata,
            "recorded_at": datetime.now().isoformat(),
            "frame_count": len(self.frames)
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        # Save end state if available
        if self.end_state_data:
            state_path = path.replace('.json', '_end.state')
            with gzip.open(state_path, 'wb') as f:
                f.write(self.end_state_data)

    @classmethod
    def load(cls, path: str) -> 'Task':
        """Load task from file."""
        with open(path) as f:
            data = json.load(f)

        task = cls(data["name"])
        task.frames = data["frames"]
        task.start_state = data.get("start_state")
        task.metadata = data.get("metadata", {})

        # Load end state if available
        state_path = path.replace('.json', '_end.state')
        if os.path.exists(state_path):
            with gzip.open(state_path, 'rb') as f:
                task.end_state_data = f.read()

        return task


class TaskRecorder:
    # ... (TaskRecorder class remains the same) ...
    """Records human input as a task."""

    def __init__(self, task_name: str, start_state: Optional[str] = None, scale: int = 3):
        self.task_name = task_name
        self.start_state = start_state
        self.scale = scale
        self.task = Task(task_name)
        self.task.start_state = start_state

    def run(self) -> Optional[Task]:
        """Record a task. Returns the recorded task or None if cancelled."""
        pygame.init()

        env_kwargs = {
            "game": "HarvestMoon-Snes",
            "inttype": retro.data.Integrations.ALL,
            "use_restricted_actions": retro.Actions.ALL,
            "render_mode": "rgb_array"
        }
        if self.start_state:
            env_kwargs["state"] = self.start_state

        try:
            env = retro.make(**env_kwargs)
        except Exception as e:
            print(f"Error: {e}")
            return None

        obs, info = env.reset()
        h, w = obs.shape[0], obs.shape[1]

        screen = pygame.display.set_mode((w * self.scale, h * self.scale))
        pygame.display.set_caption(f"Recording: {self.task_name} [F1=Save | ESC=Cancel]")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont('monospace', 14)

        joystick = init_controller()

        print("\n" + "=" * 60)
        print(f"RECORDING TASK: {self.task_name}")
        print("=" * 60)
        print_controls(joystick)
        print("\n  F1: Save & Exit | ESC: Cancel")
        print("  TAB (hold): Fast Forward 64x | [ ]: Speed (0.25x-32x)")
        print("  Recording starts immediately!")
        print("=" * 60)

        running = True
        recording = True
        speed_levels = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
        speed_idx = 2  # Start at 1.0x
        speed = speed_levels[speed_idx]

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    recording = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        print("Recording cancelled.")
                        running = False
                        recording = False
                    elif event.key == pygame.K_F1:
                        print("Saving recording...")
                        running = False
                    elif event.key == pygame.K_LEFTBRACKET:
                        speed_idx = max(0, speed_idx - 1)
                        speed = speed_levels[speed_idx]
                        print(f"Speed: {speed}x")
                    elif event.key == pygame.K_RIGHTBRACKET:
                        speed_idx = min(len(speed_levels) - 1, speed_idx + 1)
                        speed = speed_levels[speed_idx]
                        print(f"Speed: {speed}x")

            # Capture input
            keys = pygame.key.get_pressed()
            action = np.zeros(12, dtype=np.int32)
            get_keyboard_action(keys, action)
            get_controller_action(joystick, action)
            sanitize_action(action)

            # Check if TAB held for fast forward (64x)
            fast_forward = keys[pygame.K_TAB]

            # Record frame
            self.task.frames.append(action.tolist())

            # Step
            obs, reward, term, trunc, info = env.step(action)

            # Render
            surf = pygame.surfarray.make_surface(obs.swapaxes(0, 1))
            scaled = pygame.transform.scale(surf, (w * self.scale, h * self.scale))
            screen.blit(scaled, (0, 0))

            # Render (skip if fast forward for max speed)
            frame_count = len(self.task.frames)
            if not fast_forward:
                # HUD
                speed_str = f"{speed}x"
                money = 0
                ram = env.get_ram()
                if len(ram) > 0x0D2:
                    money = (
                        (int(ram[0x0D1]) & 0x0F)
                        + ((int(ram[0x0D1]) >> 4) & 0x0F) * 10
                        + (int(ram[0x0D2]) & 0x0F) * 100
                        + ((int(ram[0x0D2]) >> 4) & 0x0F) * 1000
                    )
                hud_text = f"[REC] Frame: {frame_count} | {speed_str} | F1=Save"
                text = font.render(hud_text, True, (255, 0, 0))
                screen.blit(text, (5, 5))
                money_text = font.render(f"Money: ${money:,}", True, (255, 255, 255))
                screen.blit(money_text, (5, 25))

                # Button display
                btn_names = ['B', 'Y', 'Sel', 'Sta', 'U', 'D', 'L', 'R', 'A', 'X', 'L', 'R']
                pressed = [btn_names[i] for i in range(12) if action[i]]
                if pressed:
                    btn_text = font.render(' '.join(pressed), True, (255, 255, 0))
                    screen.blit(btn_text, (5, h * self.scale - 25))

                pygame.display.flip()
                clock.tick(int(60 * speed))
            else:
                # Fast forward - minimal render, no frame limit
                if frame_count % 60 == 0:  # Update display every 60 frames
                    pygame.display.set_caption(f"Recording: {self.task_name} [FF] Frame {frame_count}")
                    pygame.display.flip()

        # Save end state
        if recording:
            self.task.end_state_data = env.em.get_state()
            self.task.metadata = {
                "frame_count": len(self.task.frames),
                "duration_seconds": len(self.task.frames) / 60.0
            }

        env.close()
        pygame.quit()

        if recording and len(self.task.frames) > 0:
            return self.task
        return None


class TaskPlayer:
    # ... (TaskPlayer class remains the same) ...
    """Replays a recorded task."""

    def __init__(self, task: Task, start_state: Optional[str] = None, scale: int = 3):
        self.task = task
        self.start_state = start_state or task.start_state
        self.scale = scale

    def run(self, visualize: bool = True) -> bool:
        """Replay the task. Returns True if completed successfully."""
        if visualize:
            pygame.init()

        env_kwargs = {
            "game": "HarvestMoon-Snes",
            "inttype": retro.data.Integrations.ALL,
            "use_restricted_actions": retro.Actions.ALL,
            "render_mode": "rgb_array"
        }
        if self.start_state:
            env_kwargs["state"] = self.start_state

        try:
            env = retro.make(**env_kwargs)
        except Exception as e:
            print(f"Error: {e}")
            return False

        obs, info = env.reset()
        h, w = obs.shape[0], obs.shape[1]

        if visualize:
            screen = pygame.display.set_mode((w * self.scale, h * self.scale))
            pygame.display.set_caption(f"Replaying: {self.task.name}")
            clock = pygame.time.Clock()
            font = pygame.font.SysFont('monospace', 14)

        print(f"\nReplaying task: {self.task.name} ({len(self.task.frames)} frames)")
        print("  TAB (hold): Fast Forward 64x | [ ]: Speed | ESC: Cancel")

        speed_levels = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
        speed_idx = 2
        speed = speed_levels[speed_idx]

        for i, action in enumerate(self.task.frames):
            if visualize:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        pygame.quit()
                        return False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            env.close()
                            pygame.quit()
                            return False
                        elif event.key == pygame.K_LEFTBRACKET:
                            speed_idx = max(0, speed_idx - 1)
                            speed = speed_levels[speed_idx]
                        elif event.key == pygame.K_RIGHTBRACKET:
                            speed_idx = min(len(speed_levels) - 1, speed_idx + 1)
                            speed = speed_levels[speed_idx]

            obs, reward, term, trunc, info = env.step(action)

            # Check if TAB held for fast forward
            if visualize:
                keys = pygame.key.get_pressed()
                fast_forward = keys[pygame.K_TAB]
            else:
                fast_forward = True

            if visualize and not fast_forward:
                surf = pygame.surfarray.make_surface(obs.swapaxes(0, 1))
                scaled = pygame.transform.scale(surf, (w * self.scale, h * self.scale))
                screen.blit(scaled, (0, 0))

                progress = (i + 1) / len(self.task.frames) * 100
                speed_str = f"{speed}x"
                hud_text = f"[PLAY] {self.task.name} | {i+1}/{len(self.task.frames)} ({progress:.0f}%) | {speed_str}"
                text = font.render(hud_text, True, (0, 255, 0))
                screen.blit(text, (5, 5))

                pygame.display.flip()
                clock.tick(int(60 * speed))
            elif visualize and fast_forward:
                if i % 120 == 0:
                    pygame.display.set_caption(f"Replaying: {self.task.name} [FF] {i}/{len(self.task.frames)}")

        env.close()
        if visualize:
            pygame.quit()

        print(f"Task completed: {self.task.name}")
        return True


def list_tasks():
    """List all recorded tasks."""
    # ... (list_tasks remains the same) ...
    tasks = sorted([f for f in os.listdir(TASKS_DIR) if f.endswith('.json')])

    print("\n" + "=" * 60)
    print("RECORDED TASKS")
    print("=" * 60)

    if not tasks:
        print("  No tasks recorded yet.")
        print(f"  Tasks will be saved in: {TASKS_DIR}")
    else:
        for task_file in tasks:
            path = os.path.join(TASKS_DIR, task_file)
            task = Task.load(path)
            duration = task.metadata.get('duration_seconds', 0)
            print(f"  {task.name}")
            print(f"    Frames: {len(task.frames)} | Duration: {duration:.1f}s")
            print(f"    Start state: {task.start_state or 'None'}")

    print("=" * 60)


def record_task(name: str, state: Optional[str] = None):
    # ... (record_task remains the same) ...
    """Record a new task."""
    recorder = TaskRecorder(name, start_state=state)
    task = recorder.run()

    if task:
        path = os.path.join(TASKS_DIR, f"{name}.json")
        task.save(path)
        print(f"\nTask saved: {path}")
        print(f"  Frames: {len(task.frames)}")
        print(f"  Duration: {len(task.frames)/60:.1f}s")
    else:
        print("\nRecording cancelled or failed.")


def replay_task(name: str, state: Optional[str] = None):
    # ... (replay_task remains the same) ...
    """Replay a recorded task."""
    path = os.path.join(TASKS_DIR, f"{name}.json")
    if not os.path.exists(path):
        print(f"Task not found: {name}")
        list_tasks()
        return

    task = Task.load(path)
    player = TaskPlayer(task, start_state=state)
    player.run()


def test_task(name: str, runs: int = 5):
    # ... (test_task remains the same) ...
    """Test task reliability by running multiple times."""
    path = os.path.join(TASKS_DIR, f"{name}.json")
    if not os.path.exists(path):
        print(f"Task not found: {name}")
        return

    task = Task.load(path)
    successes = 0

    print(f"\nTesting task: {name} ({runs} runs)")
    for i in range(runs):
        print(f"\n--- Run {i+1}/{runs} ---")
        player = TaskPlayer(task)
        if player.run(visualize=True):
            successes += 1

    print(f"\n{'='*40}")
    print(f"Results: {successes}/{runs} successful ({successes/runs*100:.0f}%)")


def main():
    parser = argparse.ArgumentParser(description="Task Recorder for Harvest Moon Bot")
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Record
    rec = subparsers.add_parser('record', help='Record a new task')
    rec.add_argument('name_pos', type=str, nargs='?', help='Task name')
    rec.add_argument('--name', type=str, help='Task name (deprecated)')
    rec.add_argument('--state', type=str, default="Y1_Spring_Day01_06h", help='Starting save state')

    # Record batch
    batch = subparsers.add_parser('record-batch', help='Record multiple tasks (name[:state] ...)')
    batch.add_argument('items', nargs='+', help='Task entries as name or name:state')
    batch.add_argument('--state', type=str, default="Y1_Spring_Day01_06h", help='Default starting state')

    # List
    subparsers.add_parser('list', help='List recorded tasks')

    # Replay
    rep = subparsers.add_parser('replay', help='Replay a task')
    rep.add_argument('name_pos', type=str, nargs='?', help='Task name')
    rep.add_argument('--name', type=str, help='Task name (deprecated)')
    rep.add_argument('--state', type=str, help='Override starting state')

    # Test
    test = subparsers.add_parser('test', help='Test task reliability')
    test.add_argument('name_pos', type=str, nargs='?', help='Task name')
    test.add_argument('--name', type=str, help='Task name (deprecated)')
    test.add_argument('--runs', type=int, default=5, help='Number of test runs')

    args = parser.parse_args()

    def get_name(args):
        if args.name_pos:
            return args.name_pos
        if args.name:
            return args.name
        return None

    if args.command == 'record':
        name = get_name(args)
        if not name:
            parser.error("Task name required (provide as argument or --name)")
        record_task(name, args.state)
    elif args.command == 'list':
        list_tasks()
    elif args.command == 'replay':
        name = get_name(args)
        if not name:
            parser.error("Task name required (provide as argument or --name)")
        replay_task(name, args.state)
    elif args.command == 'record-batch':
        for item in args.items:
            if ':' in item:
                name, state = item.split(':', 1)
                name = name.strip()
                state = state.strip() or args.state
            else:
                name, state = item.strip(), args.state
            if not name:
                parser.error("Task name required in record-batch entry")
            record_task(name, state)
    elif args.command == 'test':
        name = get_name(args)
        if not name:
            parser.error("Task name required (provide as argument or --name)")
        test_task(name, args.runs)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
