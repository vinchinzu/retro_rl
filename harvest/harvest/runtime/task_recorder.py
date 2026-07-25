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
    # Record a new task (delegates to harvest_bot.py; F5 saves)
    uv run python task_recorder.py record ship_berry --state Y1_Spring_Day01_06h

    # List recorded tasks
    uv run python task_recorder.py list

    # Replay a task
    uv run python task_recorder.py replay ship_berry --state Y1_Spring_Day01_06h

    # Test task reliability (run N times)
    uv run python task_recorder.py test ship_berry --runs 5
"""

import os
import sys
import json
import gzip
import time
import argparse
import subprocess
from datetime import datetime
from typing import Optional, List, Dict

from harvest.paths import CUSTOM_INTEGRATIONS_DIR, PROJECT_DIR, TASKS_DIR as PROJECT_TASKS_DIR

SCRIPT_DIR = os.fspath(PROJECT_DIR)
INTEGRATION_PATH = os.fspath(CUSTOM_INTEGRATIONS_DIR)
STATES_DIR = os.path.join(INTEGRATION_PATH, "HarvestMoon-Snes")
TASKS_DIR = os.fspath(PROJECT_TASKS_DIR)

os.makedirs(TASKS_DIR, exist_ok=True)

import numpy as np
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
import pygame

from retro_harness import (
    init_controller as _init_controller,
    controller_action,
    keyboard_action,
    sanitize_action,
    describe_input_mapping,
    format_input_mapping,
    SNES_BUTTON_NAMES,
)
from harvest.runtime.recording_trace import recording_trace_entry, summarize_recording
from harvest.runtime.retro_setup import backup_mutable_start_state, make_harvest_env


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
        print(f"    Mapping: {format_input_mapping(describe_input_mapping(joystick=joystick))}")
        print("    D-Pad/Stick: Movement")
        print("    B: Run/Cancel | A: Confirm/Talk | Y: Use Item | X: Menu")
        print("    L/R: Cycle Items")
    print("  Keyboard:")
    print("    Arrows: D-Pad")
    print("    Z: Cancel (B) | X: Use Item (Y) | C: Confirm (A) | V: Menu (X)")
    print("    A/S: Cycle Items (L/R)")
    print("  Recording:")
    print("    F5: Save Recording")


class Task:
    # ... (Task class remains the same) ...
    """A recorded sequence of inputs."""

    def __init__(self, name: str):
        self.name = name
        self.frames: List[List[int]] = []  # List of action arrays
        self.trace: List[Dict] = []        # Per-frame {x, y, tm} snapshots
        self.start_state: Optional[str] = None
        self.end_state_data: Optional[bytes] = None
        self.metadata: Dict = {}

    def save(self, path: str):
        """Save task to file."""
        data = {
            "name": self.name,
            "frames": self.frames,
            "trace": self.trace,
            "start_state": self.start_state,
            "metadata": self.metadata,
            "recorded_at": datetime.now().isoformat(),
            "frame_count": len(self.frames)
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        # Save end state if available
        if self.end_state_data:
            for state_path in end_state_output_paths(path):
                os.makedirs(os.path.dirname(state_path), exist_ok=True)
                with gzip.open(state_path, 'wb') as f:
                    f.write(self.end_state_data)

    @classmethod
    def load(cls, path: str) -> 'Task':
        """Load task from file."""
        with open(path) as f:
            data = json.load(f)

        task = cls(data["name"])
        task.frames = data["frames"]
        task.trace = data.get("trace", [])
        task.start_state = data.get("start_state")
        task.metadata = data.get("metadata", {})

        # Load end state if available
        state_path = path.replace('.json', '_end.state')
        if os.path.exists(state_path):
            with gzip.open(state_path, 'rb') as f:
                task.end_state_data = f.read()

        return task


def end_state_output_paths(task_json_path: str) -> List[str]:
    """Return recording-local and emulator-loadable end-state paths."""
    task_state_path = task_json_path.replace(".json", "_end.state")
    integration_state_path = os.path.join(STATES_DIR, os.path.basename(task_state_path))
    paths = [task_state_path]
    if os.path.abspath(integration_state_path) != os.path.abspath(task_state_path):
        paths.append(integration_state_path)
    return paths


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

        stable_state = backup_mutable_start_state(self.start_state, self.task_name)
        if stable_state != self.start_state:
            print(f"[REC] Backed up start state {self.start_state} -> {stable_state}")
            self.start_state = stable_state
            self.task.start_state = stable_state

        try:
            env = make_harvest_env(self.start_state)
        except Exception as e:
            print(f"Error: {e}")
            return None

        obs, info = env.reset()
        h, w = obs.shape[0], obs.shape[1]

        screen = pygame.display.set_mode((w * self.scale, h * self.scale))
        pygame.display.set_caption(f"Recording: {self.task_name} [F5=Save | ESC=Cancel]")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont('monospace', 14)

        joystick = init_controller()

        print("\n" + "=" * 60)
        print(f"RECORDING TASK: {self.task_name}")
        print("=" * 60)
        print_controls(joystick)
        print("\n  F5: Save & Exit | ESC: Cancel")
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
                    elif event.key in {pygame.K_F5, pygame.K_F1}:
                        if event.key == pygame.K_F1:
                            print("F1 save is supported as an alias; use F5 for new recordings.")
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

            # Capture per-frame trace
            ram = env.get_ram()
            self.task.trace.append(
                recording_trace_entry(
                    ram,
                    frame=len(self.task.frames) - 1,
                    action=action,
                )
            )

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
                hud_text = f"[REC] Frame: {frame_count} | {speed_str} | F5=Save"
                text = font.render(hud_text, True, (255, 0, 0))
                screen.blit(text, (5, 5))
                money_text = font.render(f"Money: ${money:,}", True, (255, 255, 255))
                screen.blit(money_text, (5, 25))

                # Button display
                btn_names = list(SNES_BUTTON_NAMES)
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
            self.task.metadata = summarize_recording(
                frames=self.task.frames,
                trace=self.task.trace,
            )

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

        try:
            env = make_harvest_env(self.start_state)
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
    """Record a new task through the canonical harvest_bot recorder."""
    cmd = [sys.executable, "-m", "harvest.runtime.harvest_bot", "play", "--record", name]
    if state:
        cmd.extend(["--state", state])
    cmd.append("--no-day-plan")
    print("[REC] Using canonical recorder: " + " ".join(cmd))
    print("[REC] Press F5 to save task JSON, trace, and mirrored end state.")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


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


def refresh_task_trace(name: str, state: Optional[str] = None, *, force: bool = False) -> Task:
    """Replay a task headlessly and rewrite its trace/metadata."""
    path = os.path.join(TASKS_DIR, f"{name}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Task not found: {name}")

    with open(path) as f:
        data = json.load(f)

    frames = data["frames"]
    start_state = state or data.get("start_state") or "latest"
    if start_state == "latest" and state is None and not force:
        raise RuntimeError(
            "Refusing to refresh from mutable start_state=latest. "
            "Pass --state <stable_state_name> or --force-refresh if you really want current latest."
        )
    env = make_harvest_env(start_state)
    trace = []
    try:
        env.reset()
        for frame, action in enumerate(frames):
            action_array = np.asarray(action, dtype=np.int32)
            env.step(action_array)
            trace.append(recording_trace_entry(env.get_ram(), frame=frame, action=action_array))
    finally:
        env.close()

    old_metadata = data.get("metadata", {})
    new_metadata = summarize_recording(frames=frames, trace=trace)
    old_coop_frames = int(old_metadata.get("coop", {}).get("frame_count", 0) or 0)
    new_coop_frames = int(new_metadata.get("coop", {}).get("frame_count", 0) or 0)
    old_transitions = old_metadata.get("transitions", [])
    new_transitions = new_metadata.get("transitions", [])
    if not force and old_coop_frames > 0 and new_coop_frames == 0:
        raise RuntimeError(
            "Refusing to overwrite trace: existing metadata has coop frames, "
            "but refreshed replay has none. The start state probably drifted."
        )
    if not force and len(old_transitions) > 1 and len(new_transitions) <= 1:
        raise RuntimeError(
            "Refusing to overwrite trace: refreshed replay lost tilemap transitions. "
            "The start state probably drifted."
        )

    backup_path = path.replace(".json", f".pre_refresh_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(backup_path, "w") as f:
        json.dump(data, f, indent=2)

    data["start_state"] = start_state
    data["trace"] = trace
    data["metadata"] = new_metadata
    data["frame_count"] = len(frames)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Backed up previous task JSON: {backup_path}")

    task = Task(name)
    task.frames = frames
    task.trace = trace
    task.start_state = start_state
    task.metadata = new_metadata
    return task


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

    # Analyze
    ana = subparsers.add_parser('analyze', help='Analyze a recorded task trace')
    ana.add_argument('name_pos', type=str, nargs='?', help='Task name')
    ana.add_argument('--name', type=str, help='Task name (deprecated)')
    ana.add_argument('--refresh-trace', action='store_true', help='Replay and rebuild trace/metadata before analysis')
    ana.add_argument('--state', type=str, help='Override starting save state when refreshing trace')
    ana.add_argument('--force-refresh', action='store_true', help='Allow refresh even when start state is mutable or replay diverges')

    args = parser.parse_args()

    def get_name(args):
        if args.name_pos:
            return args.name_pos
        if args.name:
            return args.name
        return None

    if args.command == 'analyze':
        name = get_name(args)
        if not name:
            parser.error("Task name required")
        path = os.path.join(TASKS_DIR, f"{name}.json")
        if args.refresh_trace:
            print(f"Refreshing trace for {name}...")
            try:
                task = refresh_task_trace(name, state=args.state, force=args.force_refresh)
            except Exception as exc:
                print(f"Trace refresh failed: {exc}")
                return
        else:
            task = Task.load(path)
        if not task.trace:
            reason = task.metadata.get("invalid_reason")
            if reason:
                print(f"No valid trace data in {name}: {reason}")
            else:
                print(f"No trace data in {name}. Re-record to capture trace.")
        else:
            print(f"\nTask: {name} ({len(task.frames)} frames, {len(task.frames)/60:.1f}s)")
            # Show tilemap transitions
            transitions = task.metadata.get("transitions", [])
            if transitions:
                print(f"\nTilemap transitions ({len(transitions)}):")
                for i, t in enumerate(transitions):
                    end_frame = transitions[i + 1]["frame"] if i + 1 < len(transitions) else len(task.trace)
                    dur = end_frame - t["frame"]
                    print(f"  f={t['frame']:>5d} (+{dur:>4d})  tm=0x{t['tilemap']:02X}  pos=({t['x']},{t['y']})")
            # Show position range per tilemap segment
            print(f"\nPosition ranges per tilemap:")
            cur_tm = None
            seg_start = 0
            for i, snap in enumerate(task.trace + [{"tm": -1}]):
                if snap["tm"] != cur_tm:
                    if cur_tm is not None:
                        seg = task.trace[seg_start:i]
                        xs = [s["x"] for s in seg]
                        ys = [s["y"] for s in seg]
                        print(f"  tm=0x{cur_tm:02X}  x=[{min(xs)}-{max(xs)}]  y=[{min(ys)}-{max(ys)}]  "
                              f"entry=({seg[0]['x']},{seg[0]['y']})  exit=({seg[-1]['x']},{seg[-1]['y']})")
                    cur_tm = snap["tm"]
                    seg_start = i
            coop = task.metadata.get("coop", {})
            if coop:
                print(f"\nCoop summary:")
                print(f"  Frames: {coop.get('frame_count', 0)}")
                print(f"  Player tiles: {len(coop.get('player_tiles', []))}")
                print(f"  Chicken tiles: {len(coop.get('chicken_tiles', []))}")
                print(f"  Adult blockers: {len(coop.get('adult_chicken_tiles', []))}")
                print(f"  Walk-over chicks: {len(coop.get('chick_tiles', []))}")
                for key in (
                    "stored_grass_change_windows",
                    "fed_chickens_change_windows",
                    "egg_available_change_windows",
                    "held_item_change_windows",
                    "shipping_money_change_windows",
                ):
                    windows = coop.get(key, [])
                    if windows:
                        print(f"  {key}: {windows}")
            stasis = task.metadata.get("stasis_windows", [])
            if stasis:
                print(f"\nStasis windows ({len(stasis)}):")
                for window in stasis[:12]:
                    print(
                        f"  f={window['start']}-{window['end']} len={window['length']} "
                        f"tm=0x{window['tilemap']:02X} tile={tuple(window['tile'])} "
                        f"buttons={'+'.join(window['buttons'])}"
                    )
    elif args.command == 'record':
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
