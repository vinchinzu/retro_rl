#!/usr/bin/env python3
"""
Comprehensive farm clearing test - runs in background and logs progress.
Tests ability to clear entire farm of liftable debris (bushes, stones).
"""
import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import harvest_bot as hb
import numpy as np
import stable_retro as retro


class FarmClearAnalyzer:
    """Analyzes farm clearing progress and detects stuck states."""

    def __init__(self, log_file: str):
        self.log_file = log_file
        self.start_time = time.time()
        self.frame_count = 0
        self.toss_count = 0
        self.stuck_events: List[Dict] = []
        self.position_history: List[Tuple[int, int]] = []
        self.max_history = 60
        self.last_target: Tuple[int, int, int] = None  # (x, y, tile_id)
        self.target_retry_count = 0
        self.debris_cleared: List[Tuple[int, int, int]] = []  # (tile_x, tile_y, tile_id)
        self.initial_debris_count = 0
        self.current_debris_count = 0

    def log(self, message: str):
        """Log message with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        elapsed = time.time() - self.start_time
        with open(self.log_file, 'a') as f:
            f.write(f"[{timestamp} +{elapsed:.1f}s F{self.frame_count}] {message}\n")
        print(f"[{timestamp}] {message}")

    def check_stuck(self, bot: hb.AutoClearBot) -> bool:
        """Check if bot is stuck and log details."""
        pos = bot.nav.current_pos
        self.position_history.append((pos.x, pos.y))
        if len(self.position_history) > self.max_history:
            self.position_history.pop(0)

        # Check for position stasis (stuck in small area)
        if len(self.position_history) >= self.max_history:
            xs = [p[0] for p in self.position_history]
            ys = [p[1] for p in self.position_history]
            x_range = max(xs) - min(xs)
            y_range = max(ys) - min(ys)

            if x_range < 32 and y_range < 32:  # Stuck in 2x2 tile area for 60 frames
                stuck_info = {
                    'frame': self.frame_count,
                    'position': (pos.x, pos.y),
                    'x_range': x_range,
                    'y_range': y_range,
                    'target': bot.brain.target_tile,
                    'carrying': bot.brain.carrying,
                    'current_debris': bot.brain.current_debris_type,
                    'tool_id': bot.tm.current_tool_id,
                }
                self.stuck_events.append(stuck_info)
                self.log(f"STUCK DETECTED: pos=({pos.x},{pos.y}) range=({x_range},{y_range}) "
                        f"target={bot.brain.target_tile} carry={bot.brain.carrying}")
                return True

        return False

    def update_debris_count(self, env, bot: hb.AutoClearBot):
        """Update debris count and log changes."""
        ram = env.get_ram()
        targets = bot.brain.tf.scan(ram)
        liftable_count = 0
        for point, debris_type in targets:
            tile_id = bot.brain.tf.get_tile_at(ram, point.x, point.y)
            if bot.brain._is_liftable(tile_id):
                liftable_count += 1

        if self.initial_debris_count == 0:
            self.initial_debris_count = liftable_count
            self.log(f"Initial liftable debris count: {liftable_count}")

        if liftable_count != self.current_debris_count:
            cleared = self.current_debris_count - liftable_count
            if cleared > 0:
                self.log(f"Progress: {liftable_count} liftable debris remaining "
                        f"({self.initial_debris_count - liftable_count} cleared)")
            self.current_debris_count = liftable_count

    def on_toss(self, bot: hb.AutoClearBot):
        """Called when debris is tossed."""
        self.toss_count += 1
        self.log(f"Toss #{self.toss_count} completed at pos=({bot.nav.current_pos.x},{bot.nav.current_pos.y})")

    def finalize(self):
        """Write final summary."""
        elapsed = time.time() - self.start_time
        self.log("\n" + "="*60)
        self.log("FINAL SUMMARY")
        self.log("="*60)
        self.log(f"Total frames: {self.frame_count}")
        self.log(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f}m)")
        self.log(f"Initial liftable debris: {self.initial_debris_count}")
        self.log(f"Final liftable debris: {self.current_debris_count}")
        self.log(f"Debris cleared: {self.initial_debris_count - self.current_debris_count}")
        self.log(f"Tosses completed: {self.toss_count}")
        self.log(f"Stuck events detected: {len(self.stuck_events)}")
        if self.stuck_events:
            self.log("\nStuck events:")
            for i, event in enumerate(self.stuck_events[:10]):  # First 10
                self.log(f"  {i+1}. Frame {event['frame']}: pos={event['position']} "
                        f"target={event['target']} carry={event['carrying']}")


def run_comprehensive_test(
    state: str = "Y1_L6_Mixed",
    max_frames: int = 10000,
    log_file: str = "/tmp/farm_clear_comprehensive.log"
):
    """Run comprehensive farm clearing test."""
    # Clear log file
    with open(log_file, 'w') as f:
        f.write(f"Comprehensive Farm Clear Test - {datetime.now()}\n")
        f.write(f"State: {state}\n")
        f.write(f"Max frames: {max_frames}\n")
        f.write("="*60 + "\n\n")

    analyzer = FarmClearAnalyzer(log_file)
    analyzer.log("Starting comprehensive farm clearing test...")

    # Create environment
    env = retro.make(
        game="HarvestMoon-Snes",
        state=state,
        inttype=retro.data.Integrations.ALL,
        use_restricted_actions=retro.Actions.ALL,
        render_mode="rgb_array"
    )

    # Create bot with liftable-only mode
    priority = [hb.DebrisType.STONE, hb.DebrisType.WEED]
    bot = hb.AutoClearBot(priority=priority)
    bot.pond_dispose_enabled = True
    bot.brain.only_liftable = True
    bot.enabled = True

    obs, info = env.reset()
    bot.set_env(env)

    analyzer.log(f"Bot initialized. Priority: {[d.name for d in priority]}")
    analyzer.log(f"Pond targets: {[(p.x, p.y) for p in bot.pond_targets]}")

    last_toss_count = 0
    last_debris_check = 0
    debris_check_interval = 300  # Check every 5 seconds at 60fps

    try:
        for frame in range(max_frames):
            analyzer.frame_count = frame
            game_state = hb.GameState(info)
            action = bot.get_action(game_state, obs)
            obs, reward, terminated, truncated, info = env.step(action)

            # Check if bot disabled
            if not bot.enabled:
                analyzer.log(f"Bot disabled: {bot.disable_reason}")
                break

            # Detect tosses
            if bot.brain.toss_count > last_toss_count:
                analyzer.on_toss(bot)
                last_toss_count = bot.brain.toss_count

            # Periodic debris count update
            if frame - last_debris_check >= debris_check_interval:
                analyzer.update_debris_count(env, bot)
                last_debris_check = frame

            # Check for stuck every 60 frames
            if frame % 60 == 0:
                analyzer.check_stuck(bot)

            # Log progress every 600 frames (10 seconds)
            if frame % 600 == 0 and frame > 0:
                analyzer.log(f"Progress: Frame {frame} pos=({bot.nav.current_pos.x},{bot.nav.current_pos.y}) "
                           f"tosses={bot.brain.toss_count}")

            if terminated or truncated:
                analyzer.log("Environment terminated")
                break

        # Final debris count
        analyzer.update_debris_count(env, bot)

    except KeyboardInterrupt:
        analyzer.log("Test interrupted by user")
    except Exception as e:
        analyzer.log(f"Test failed with error: {e}")
        import traceback
        with open(log_file, 'a') as f:
            f.write("\nException traceback:\n")
            f.write(traceback.format_exc())
    finally:
        env.close()
        analyzer.finalize()
        analyzer.log(f"Test complete. Log saved to: {log_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Comprehensive farm clearing test")
    parser.add_argument('--state', default='Y1_L6_Mixed', help='Starting state')
    parser.add_argument('--frames', type=int, default=10000, help='Max frames to run')
    parser.add_argument('--log', default='/tmp/farm_clear_comprehensive.log', help='Log file path')
    args = parser.parse_args()

    run_comprehensive_test(args.state, args.frames, args.log)
