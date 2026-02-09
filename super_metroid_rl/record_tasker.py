#!/usr/bin/env python3
"""
Demo Recording Tasker for Super Metroid RL

Provides a streamlined interface for recording demonstration runs
for training the hierarchical PPO pipeline.

Usage:
    # Record a full run from ZebesStart
    python record_tasker.py record --state ZebesStart

    # Record from a specific room
    python record_tasker.py record --state "Morph Ball Room"

    # List available states
    python record_tasker.py list-states

    # Analyze existing recordings
    python record_tasker.py analyze

    # Start loop recording session (auto-save and restart)
    python record_tasker.py loop --state ZebesStart
"""

import os
import sys
import json
import time
import argparse
import uuid
from datetime import datetime
from typing import Optional, List, Dict

# =============================================================================
# PATHS & IMPORTS
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

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
    """Print Super Metroid recording controls."""
    print("\nRecording Controls:")
    if joystick:
        print(f"  Controller: {joystick.get_name()}")
        print("    D-Pad/Stick: Movement")
        print("    A: Jump | B: Run | X: Shoot | Y: Item Cancel")
        print("    LB/RB: Aim Up/Down")
    print("  Keyboard:")
    print("    Arrows: D-Pad")
    print("    Z: Run (B) | C: Jump (A) | V: Shoot (X) | X: Item Cancel (Y)")
    print("    A/S: Aim Up/Down (L/R)")
    print("  Recording:")
    print("    F5: Save State | Ctrl+S: Stop Recording")


INTEGRATION_PATH = os.path.join(SCRIPT_DIR, "custom_integrations")
retro.data.Integrations.add_custom_path(INTEGRATION_PATH)

DEMO_DIR = os.path.join(SCRIPT_DIR, "demos")
STATES_DIR = os.path.join(INTEGRATION_PATH, "SuperMetroid-Snes")
MANIFEST_PATH = os.path.join(DEMO_DIR, "recordings.json")
WORLD_MAP_PATH = os.path.join(SCRIPT_DIR, "world_map.json")
BEST_TIMES_PATH = os.path.join(SCRIPT_DIR, "best_times.json")

os.makedirs(DEMO_DIR, exist_ok=True)

# =============================================================================
# KEYBOARD MAPPING
# =============================================================================
# SNES Button Order: B, Y, Select, Start, Up, Down, Left, Right, A, X, L, R
KEY_MAP = {
    pygame.K_z: 0,      # B (run/back)
    pygame.K_x: 1,      # Y (item cancel)
    pygame.K_TAB: 2,    # Select
    pygame.K_RETURN: 3, # Start
    pygame.K_UP: 4,     # Up
    pygame.K_DOWN: 5,   # Down
    pygame.K_LEFT: 6,   # Left
    pygame.K_RIGHT: 7,  # Right
    pygame.K_c: 8,      # A (jump)
    pygame.K_v: 9,      # X (shoot)
    pygame.K_a: 10,     # L (aim up)
    pygame.K_s: 11,     # R (aim down)
}

# Alternate WASD controls for movement
WASD_MAP = {
    pygame.K_w: 4,  # Up
    pygame.K_s: 5,  # Down (conflicts with R, so optional)
    pygame.K_a: 6,  # Left
    pygame.K_d: 7,  # Right
}

# =============================================================================
# WORLD MAP
# =============================================================================
def load_world_map() -> Dict[str, str]:
    """Load room name -> room ID mapping."""
    if os.path.exists(WORLD_MAP_PATH):
        with open(WORLD_MAP_PATH, 'r') as f:
            return json.load(f)
    return {}


def get_room_name(room_id: int, world_map: Dict[str, str]) -> str:
    """Get room name from ID."""
    room_hex = f"0x{room_id:x}"
    for name, rid in world_map.items():
        if rid.lower() == room_hex.lower():
            return name
    return f"Unknown (0x{room_id:04X})"


# =============================================================================
# MANIFEST MANAGEMENT
# =============================================================================
def load_manifest() -> List[Dict]:
    """Load recordings manifest."""
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, 'r') as f:
            return json.load(f)
    return []


def save_manifest(manifest: List[Dict]):
    """Save recordings manifest."""
    with open(MANIFEST_PATH, 'w') as f:
        json.dump(manifest, f, indent=4)


def add_to_manifest(
    filename: str,
    start_state: str,
    tags: List[str] = None,
    route: List[str] = None
):
    """Add a new recording to the manifest."""
    manifest = load_manifest()

    entry = {
        "id": str(uuid.uuid4()),
        "filename": filename,
        "timestamp": int(time.time()),
        "start_state": start_state,
        "tags": tags or ["manual_demo"],
        "analyzed": False,
        "route": route or []
    }

    manifest.append(entry)
    save_manifest(manifest)
    return entry


# =============================================================================
# RECORDING SESSION
# =============================================================================
class RecordingSession:
    """Interactive recording session with pygame display."""

    def __init__(
        self,
        state: str = "ZebesStart",
        record_dir: str = DEMO_DIR,
        scale: int = 2
    ):
        self.state = state
        self.record_dir = record_dir
        self.scale = scale

        # Game state tracking
        self.world_map = load_world_map()
        self.frame_count = 0
        self.rooms_visited = []
        self.current_room = None
        self.room_entry_frame = 0
        self.room_splits = {}

        # Recording state
        self.recording = True
        self.paused = False

    def run(self) -> Optional[str]:
        """Run the recording session. Returns path to saved bk2 file."""
        pygame.init()

        # Create environment with recording enabled
        timestamp = int(time.time())
        bk2_prefix = f"{self.state}-{timestamp}"

        env = retro.make(
            game="SuperMetroid-Snes",
            state=self.state,
            inttype=retro.data.Integrations.ALL,
            use_restricted_actions=retro.Actions.ALL,
            render_mode='rgb_array',
            record=self.record_dir
        )

        obs, info = env.reset()
        h, w = obs.shape[0], obs.shape[1]

        screen = pygame.display.set_mode((w * self.scale, h * self.scale))
        pygame.display.set_caption(f"Recording: {self.state} [F1=Save | ESC=Cancel | P=Pause]")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont('monospace', 12)

        # Initialize room tracking
        self.current_room = info.get('room_id', 0)
        self.rooms_visited = [self.current_room]

        running = True
        bk2_path = None

        # Initialize controller
        joystick = init_controller()

        print("\n" + "="*60)
        print("RECORDING SESSION STARTED")
        print("="*60)
        print(f"State: {self.state}")
        print_controls(joystick)
        print("  F1/START: Save & Exit | ESC/SELECT+START: Cancel | P: Pause")
        print("="*60)

        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    self.recording = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        print("Recording cancelled.")
                        running = False
                        self.recording = False

                    elif event.key == pygame.K_F1:
                        print("Saving recording...")
                        running = False

                    elif event.key == pygame.K_p:
                        self.paused = not self.paused
                        status = "PAUSED" if self.paused else "RESUMED"
                        print(f"[{status}]")

                elif event.type == pygame.JOYBUTTONDOWN:
                    # START (7) = save and exit
                    if event.button == 7:
                        if joystick and joystick.get_button(6):  # SELECT held = cancel
                            print("Recording cancelled.")
                            running = False
                            self.recording = False
                        else:
                            print("Saving recording...")
                            running = False

            if self.paused:
                # Draw pause overlay
                screen.fill((0, 0, 0))
                text = font.render("PAUSED - Press P to resume", True, (255, 255, 0))
                screen.blit(text, (10, h * self.scale // 2))
                pygame.display.flip()
                clock.tick(10)
                continue

            # Build action from keyboard + controller
            keys = pygame.key.get_pressed()
            action = np.zeros(12, dtype=np.int32)
            get_keyboard_action(keys, action)
            get_controller_action(joystick, action)
            sanitize_action(action)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            self.frame_count += 1

            # Track room transitions
            room_id = info.get('room_id', 0)
            if room_id != self.current_room:
                # Log split time
                room_frames = self.frame_count - self.room_entry_frame
                room_name = get_room_name(self.current_room, self.world_map)
                self.room_splits[room_name] = room_frames
                print(f"[ROOM] Left {room_name} in {room_frames} frames ({room_frames/60:.1f}s)")

                # Update tracking
                self.current_room = room_id
                self.room_entry_frame = self.frame_count
                if room_id not in self.rooms_visited:
                    self.rooms_visited.append(room_id)

            # Render
            surf = pygame.surfarray.make_surface(obs.swapaxes(0, 1))
            scaled = pygame.transform.scale(surf, (w * self.scale, h * self.scale))
            screen.blit(scaled, (0, 0))

            # HUD overlay
            room_name = get_room_name(room_id, self.world_map)
            hp = info.get('health', 0)
            missiles = info.get('missiles', 0)

            hud_lines = [
                f"Room: {room_name}",
                f"HP: {hp} | Missiles: {missiles}",
                f"Frame: {self.frame_count} | Rooms: {len(self.rooms_visited)}",
                "[REC]" if self.recording else ""
            ]

            for i, line in enumerate(hud_lines):
                text = font.render(line, True, (255, 0, 0) if "REC" in line else (0, 255, 0))
                screen.blit(text, (5, 5 + i * 14))

            # Button display
            btn_names = ['B', 'Y', 'Sel', 'Sta', 'U', 'D', 'L', 'R', 'A', 'X', 'L', 'R']
            pressed = [btn_names[i] for i in range(12) if action[i]]
            btn_text = font.render(' '.join(pressed), True, (255, 255, 0))
            screen.blit(btn_text, (5, h * self.scale - 20))

            pygame.display.flip()
            clock.tick(60)

            if terminated or truncated:
                print("Episode ended.")
                break

        # Close environment (this finalizes the bk2)
        env.close()
        pygame.quit()

        # Find the saved bk2 file
        if self.recording:
            import glob
            pattern = os.path.join(self.record_dir, f"SuperMetroid-Snes-{self.state}*.bk2")
            bk2_files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)

            if bk2_files:
                bk2_path = bk2_files[0]
                print(f"\nRecording saved: {bk2_path}")

                # Build route
                route = [get_room_name(rid, self.world_map) for rid in self.rooms_visited]

                # Add to manifest
                add_to_manifest(
                    filename=os.path.basename(bk2_path),
                    start_state=self.state,
                    tags=["manual_demo", "zebes_route"],
                    route=route
                )

                # Print summary
                print("\n" + "="*60)
                print("RECORDING SUMMARY")
                print("="*60)
                print(f"Total frames: {self.frame_count} ({self.frame_count/60:.1f}s)")
                print(f"Rooms visited: {len(self.rooms_visited)}")
                print("\nRoom splits:")
                for room, frames in self.room_splits.items():
                    print(f"  {room}: {frames} frames ({frames/60:.1f}s)")
                print("="*60)

        return bk2_path


# =============================================================================
# LIST STATES
# =============================================================================
def list_available_states():
    """List all available save states."""
    import glob

    pattern = os.path.join(STATES_DIR, "*.state")
    states = glob.glob(pattern)

    print("\n" + "="*60)
    print("AVAILABLE STATES")
    print("="*60)

    for state_path in sorted(states):
        name = os.path.basename(state_path).replace(".state", "")
        print(f"  {name}")

    print(f"\nTotal: {len(states)} states")
    print("="*60)


# =============================================================================
# ANALYZE RECORDINGS
# =============================================================================
def analyze_recordings():
    """Analyze existing recordings and show statistics."""
    manifest = load_manifest()

    print("\n" + "="*60)
    print("RECORDING ANALYSIS")
    print("="*60)

    if not manifest:
        print("No recordings found.")
        return

    print(f"Total recordings: {len(manifest)}")

    # Group by state
    by_state = {}
    for entry in manifest:
        state = entry.get('start_state', 'Unknown')
        if state not in by_state:
            by_state[state] = []
        by_state[state].append(entry)

    print("\nBy starting state:")
    for state, entries in sorted(by_state.items()):
        print(f"  {state}: {len(entries)} recordings")

    # Show recent recordings
    recent = sorted(manifest, key=lambda x: x.get('timestamp', 0), reverse=True)[:10]
    print("\nRecent recordings:")
    for entry in recent:
        ts = entry.get('timestamp', 0)
        dt = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M')
        state = entry.get('start_state', 'Unknown')
        fname = entry.get('filename', 'Unknown')
        print(f"  [{dt}] {state}: {fname}")

    # Check for unanalyzed
    unanalyzed = [e for e in manifest if not e.get('analyzed', False)]
    if unanalyzed:
        print(f"\nUnanalyzed recordings: {len(unanalyzed)}")

    print("="*60)


# =============================================================================
# LOOP RECORDING
# =============================================================================
def loop_recording(state: str = "ZebesStart", count: int = 0):
    """
    Loop recording mode - automatically restarts after each recording.
    Args:
        state: Starting state
        count: Number of recordings (0 = infinite)
    """
    print("\n" + "="*60)
    print("LOOP RECORDING MODE")
    print("="*60)
    print(f"State: {state}")
    print(f"Recordings: {'infinite' if count == 0 else count}")
    print("Press Ctrl+C to stop")
    print("="*60)

    recordings = 0
    try:
        while count == 0 or recordings < count:
            recordings += 1
            print(f"\n--- Recording #{recordings} ---")

            session = RecordingSession(state=state)
            bk2_path = session.run()

            if bk2_path:
                print(f"Saved: {bk2_path}")
            else:
                print("Recording cancelled or failed.")
                break

            # Brief pause between recordings
            print("\nStarting next recording in 3 seconds... (Ctrl+C to stop)")
            time.sleep(3)

    except KeyboardInterrupt:
        print("\n\nLoop recording stopped.")

    print(f"\nTotal recordings: {recordings}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Demo Recording Tasker")
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Record command
    rec = subparsers.add_parser('record', help='Start recording session')
    rec.add_argument('--state', type=str, default='ZebesStart',
                     help='Starting state (default: ZebesStart)')
    rec.add_argument('--scale', type=int, default=2,
                     help='Display scale (default: 2)')

    # List states command
    subparsers.add_parser('list-states', help='List available states')

    # Analyze command
    subparsers.add_parser('analyze', help='Analyze existing recordings')

    # Loop command
    loop = subparsers.add_parser('loop', help='Loop recording mode')
    loop.add_argument('--state', type=str, default='ZebesStart',
                      help='Starting state')
    loop.add_argument('--count', type=int, default=0,
                      help='Number of recordings (0 = infinite)')

    args = parser.parse_args()

    if args.command == 'record':
        session = RecordingSession(state=args.state, scale=args.scale)
        session.run()
    elif args.command == 'list-states':
        list_available_states()
    elif args.command == 'analyze':
        analyze_recordings()
    elif args.command == 'loop':
        loop_recording(state=args.state, count=args.count)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
