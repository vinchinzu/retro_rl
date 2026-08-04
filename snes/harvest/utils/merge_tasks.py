#!/usr/bin/env python3
"""
Merge multiple task JSON files into a single task.
Usage: python merge_tasks.py task1 task2 task3 --output combined_task --start-state BaseState
"""
import os
import json
import argparse
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TASKS_DIR = os.path.join(SCRIPT_DIR, "tasks")


def load_task(name: str) -> dict:
    """Load task JSON."""
    path = os.path.join(TASKS_DIR, f"{name}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Task not found: {path}")
    with open(path) as f:
        return json.load(f)


def merge_tasks(task_names: list[str], output_name: str, start_state: str):
    """Merge multiple tasks into one."""
    all_frames = []
    total_duration = 0.0

    print(f"Merging {len(task_names)} tasks into '{output_name}'...")

    for task_name in task_names:
        task = load_task(task_name)
        frames = task["frames"]
        duration = task["metadata"]["duration_seconds"]

        print(f"  + {task_name}: {len(frames)} frames ({duration:.1f}s)")
        all_frames.extend(frames)
        total_duration += duration

    # Create merged task
    merged = {
        "name": output_name,
        "frames": all_frames,
        "start_state": start_state,
        "metadata": {
            "frame_count": len(all_frames),
            "duration_seconds": total_duration,
            "merged_from": task_names,
            "merged_at": datetime.now().isoformat()
        },
        "recorded_at": datetime.now().isoformat(),
        "frame_count": len(all_frames)
    }

    # Save
    output_path = os.path.join(TASKS_DIR, f"{output_name}.json")
    with open(output_path, 'w') as f:
        json.dump(merged, f, indent=2)

    print(f"\nMerged task saved: {output_path}")
    print(f"  Total frames: {len(all_frames)}")
    print(f"  Total duration: {total_duration:.1f}s")

    return merged


def main():
    parser = argparse.ArgumentParser(description="Merge multiple task files")
    parser.add_argument("tasks", nargs="+", help="Task names to merge (in order)")
    parser.add_argument("--output", "-o", required=True, help="Output task name")
    parser.add_argument("--start-state", "-s", required=True, help="Start state for merged task")
    args = parser.parse_args()

    merge_tasks(args.tasks, args.output, args.start_state)


if __name__ == "__main__":
    main()
