"""Interactive RAM explorer for discovering addresses in new games.

Play the game while the overlay highlights RAM addresses that are
changing, matching a search value, or correlated with input.

Usage::

    uv run python -m platformer_common.ram_explorer \\
        --game DonkeyKongCountry-Snes --state WinkysWalkway \\
        --game-dir donkey_kong_country

Controls:
    Normal gameplay: keyboard/controller (see retro_harness.controls)
    TAB: toggle turbo
    F1: cycle mode (diff / search / track)
    F2: in search mode, prompt for value to search
    F3: freeze/unfreeze display (keep exploring without overlay churn)
    ESC: quit
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "x11")

ROOT_DIR = Path(__file__).parent.parent.resolve()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from retro_harness.env import make_env
from retro_harness.play_session import PlaySession


class RAMExplorer:
    """RAM exploration overlay for PlaySession."""

    MODES = ["diff", "search", "track"]

    def __init__(self, top_n: int = 20, diff_window: int = 10) -> None:
        self.mode = "diff"
        self.top_n = top_n
        self.diff_window = diff_window
        self.frozen = False
        self.search_value: int | None = None

        # History for diff mode
        self._ram_history: list[np.ndarray] = []
        self._change_counts: np.ndarray | None = None

        # For track mode: correlate with input
        self._input_frames: list[bool] = []  # True if any button pressed
        self._ram_deltas: np.ndarray | None = None

        # Display cache
        self._display_lines: list[str] = []

    def on_step(self, obs, reward, done, info, env) -> None:
        if self.frozen:
            return

        ram = env.get_ram()

        if self.mode == "diff":
            self._update_diff(ram)
        elif self.mode == "search":
            self._update_search(ram)
        elif self.mode == "track":
            self._update_track(ram, env)

    def on_hud(self, info: dict) -> list[str]:
        mode_str = f"RAM: {self.mode.upper()}"
        if self.frozen:
            mode_str += " [FROZEN]"
        if self.mode == "search" and self.search_value is not None:
            mode_str += f" val={self.search_value}"
        lines = [mode_str, "F1:mode F2:search F3:freeze"]
        lines.extend(self._display_lines[:self.top_n])
        return lines

    def _update_diff(self, ram: np.ndarray) -> None:
        """Highlight addresses that changed recently."""
        self._ram_history.append(ram.copy())
        if len(self._ram_history) > self.diff_window + 1:
            self._ram_history.pop(0)

        if len(self._ram_history) < 2:
            return

        n = len(ram)
        if self._change_counts is None or len(self._change_counts) != n:
            self._change_counts = np.zeros(n, dtype=np.int32)
        else:
            self._change_counts[:] = 0

        prev = self._ram_history[0]
        for snap in self._ram_history[1:]:
            min_len = min(len(prev), len(snap), n)
            self._change_counts[:min_len] += (prev[:min_len] != snap[:min_len]).astype(np.int32)
            prev = snap

        # Top N most active addresses
        top_indices = np.argsort(self._change_counts)[::-1][:self.top_n]
        self._display_lines = []
        for idx in top_indices:
            count = self._change_counts[idx]
            if count == 0:
                break
            val = int(ram[idx])
            self._display_lines.append(f"  0x{idx:04X}={val:3d} (0x{val:02X}) chg={count}")

    def _update_search(self, ram: np.ndarray) -> None:
        """Find addresses containing search value."""
        if self.search_value is None:
            self._display_lines = ["  Press F2 to set search value"]
            return

        matches = []
        for i in range(len(ram)):
            if int(ram[i]) == self.search_value:
                matches.append(i)

        self._display_lines = [f"  Found {len(matches)} matches for {self.search_value}:"]
        for addr in matches[:self.top_n - 1]:
            self._display_lines.append(f"  0x{addr:04X}={self.search_value}")

    def _update_track(self, ram: np.ndarray, env) -> None:
        """Track addresses correlated with D-pad input."""
        if len(self._ram_history) == 0:
            self._ram_history.append(ram.copy())
            return

        prev = self._ram_history[-1]
        self._ram_history.append(ram.copy())
        if len(self._ram_history) > 60:
            self._ram_history.pop(0)

        n = min(len(ram), len(prev))
        if self._ram_deltas is None or len(self._ram_deltas) != n:
            self._ram_deltas = np.zeros(n, dtype=np.float64)

        # Accumulate absolute deltas
        delta = np.abs(ram[:n].astype(np.float64) - prev[:n].astype(np.float64))
        self._ram_deltas[:n] = self._ram_deltas[:n] * 0.95 + delta * 0.05

        top_indices = np.argsort(self._ram_deltas)[::-1][:self.top_n]
        self._display_lines = []
        for idx in top_indices:
            score = self._ram_deltas[idx]
            if score < 0.01:
                break
            val = int(ram[idx])
            self._display_lines.append(f"  0x{idx:04X}={val:3d} (0x{val:02X}) activity={score:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Interactive RAM Explorer")
    parser.add_argument("--game", required=True, help="Game name (e.g., DonkeyKongCountry-Snes)")
    parser.add_argument("--state", required=True, help="State name")
    parser.add_argument("--game-dir", required=True, help="Game directory name")
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--top", type=int, default=20, help="Number of addresses to show")
    args = parser.parse_args()

    game_dir = ROOT_DIR / args.game_dir

    env = make_env(
        game=args.game,
        state=args.state,
        game_dir=game_dir,
        render_mode="rgb_array",
    )

    explorer = RAMExplorer(top_n=args.top)

    session = PlaySession(
        env,
        game_dir=str(game_dir),
        game=args.game,
        scale=args.scale,
        title=f"RAM Explorer: {args.game} / {args.state}",
    )

    session.on_hud = explorer.on_hud
    session.on_step = lambda obs, rew, done, info: explorer.on_step(obs, rew, done, info, env)

    def on_key(key):
        import pygame

        if key == pygame.K_F1:
            idx = RAMExplorer.MODES.index(explorer.mode)
            explorer.mode = RAMExplorer.MODES[(idx + 1) % len(RAMExplorer.MODES)]
            explorer._ram_history.clear()
            explorer._display_lines.clear()
            explorer._change_counts = None
            explorer._ram_deltas = None
            print(f"[RAM] Mode: {explorer.mode}")
            return True
        elif key == pygame.K_F2:
            try:
                val = int(input("Search value (decimal): "))
                explorer.search_value = val
                print(f"[RAM] Searching for {val}")
            except (ValueError, EOFError):
                print("[RAM] Invalid value")
            return True
        elif key == pygame.K_F3:
            explorer.frozen = not explorer.frozen
            print(f"[RAM] {'FROZEN' if explorer.frozen else 'UNFROZEN'}")
            return True
        return False

    session.on_key_down = on_key
    session.run()


if __name__ == "__main__":
    main()
