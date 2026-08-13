#!/usr/bin/env python3
"""Headed pure-hop watcher — run a pure controller with a live pygame window.

Not continuous evidence. For visual debug of one pure segment.

```bash
# Double Chamber → Wave (gate open + Super door attempt)
uv run python snes/super_metroid/scripts/probe/watch_pure_hop.py double-chamber-to-wave \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_single_to_double_chamber_pure.state \
  --speed 0.5

# From post-gate pure pin (if present)
uv run python snes/super_metroid/scripts/probe/watch_pure_hop.py double-chamber-to-wave \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/dev_gate_open_pure.state \
  --speed 0.5
```

Keys: ESC/Q quit · [/] speed · TAB turbo (via pygame loop throttle only)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
_SNES = Path(__file__).resolve().parents[3]
for _p in (ROOT, _SNES):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# SDL before pygame (Hyprland / Wayland)
if "SDL_VIDEODRIVER" not in os.environ:
    if os.environ.get("WAYLAND_DISPLAY"):
        os.environ["SDL_VIDEODRIVER"] = "wayland"
    else:
        os.environ["SDL_VIDEODRIVER"] = "x11"

from retro_harness.actions import idle_action  # noqa: E402
from super_metroid.assist import UnlimitedResourcesAssist  # noqa: E402
from super_metroid.dev.common import boot_from_state, make_dev_env  # noqa: E402
from super_metroid.paths import GAME, GAME_DIR  # noqa: E402
from super_metroid.ram import parse_env_state  # noqa: E402
from super_metroid.routes.kpdr.wave import (  # noqa: E402
    play_bubble_to_single_chamber,
    play_double_chamber_to_wave,
    play_single_to_double_chamber,
)
from super_metroid.routes.kpdr.speed_return import play_speed_return_to_bubble  # noqa: E402

PLAYERS = {
    "double-chamber-to-wave": play_double_chamber_to_wave,
    "single-to-double-chamber": play_single_to_double_chamber,
    "bubble-to-single-chamber": play_bubble_to_single_chamber,
    "speed-return-to-bubble": play_speed_return_to_bubble,
}


class WatchSession:
    """ControllerSession that blits RGB every step to a pygame window."""

    def __init__(
        self,
        env,
        assist: UnlimitedResourcesAssist,
        *,
        scale: int = 3,
        speed: float = 0.5,
        title: str = "pure hop watch",
    ) -> None:
        import pygame

        self.env = env
        self.assist = assist
        self.frame = 0
        self.state = parse_env_state(env, mode="nav")
        self.reason = ""
        self.scale = max(1, scale)
        self.speed = max(0.1, speed)
        self._base_fps = 60.0
        self._pygame = pygame
        self._running = True
        pygame.init()
        # First obs size after boot
        obs = env.render() if hasattr(env, "render") else None
        if obs is None:
            obs, *_ = env.step(idle_action())
            env.step  # keep type quiet
        # Prefer last step obs later; init from render
        try:
            obs = env.render()
        except Exception:  # noqa: BLE001
            obs, *_ = env.step(idle_action())
        h, w = int(obs.shape[0]), int(obs.shape[1])
        self._screen = pygame.display.set_mode((w * self.scale, h * self.scale))
        pygame.display.set_caption(title)
        self._font = pygame.font.SysFont("monospace", 16)
        self._clock = pygame.time.Clock()
        self._last_obs = obs
        self._blit(obs)

    def _handle_events(self) -> None:
        pygame = self._pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    self._running = False
                elif event.key == pygame.K_LEFTBRACKET:
                    self.speed = max(0.1, self.speed / 1.5)
                elif event.key == pygame.K_RIGHTBRACKET:
                    self.speed = min(4.0, self.speed * 1.5)
                elif event.key == pygame.K_TAB:
                    self.speed = 4.0 if self.speed < 3.0 else 0.5

    def _blit(self, obs) -> None:
        import numpy as np

        pygame = self._pygame
        arr = np.asarray(obs)
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        # SNES RGB — pygame wants (w,h) surface from (h,w,3)
        surf = pygame.surfarray.make_surface(arr.swapaxes(0, 1))
        if self.scale != 1:
            surf = pygame.transform.scale(
                surf, (surf.get_width() * self.scale, surf.get_height() * self.scale)
            )
        self._screen.blit(surf, (0, 0))
        st = self.state
        lines = [
            f"f={self.frame} speed={self.speed:.2f}x  [ ] speed  TAB turbo  ESC quit",
            f"room=0x{st.room_id:04X} xy=({st.samus_x},{st.samus_y}) pose={st.pose}",
            f"beams=0x{int(getattr(st, 'collected_beams', 0) or 0):04X} "
            f"mis={st.missiles} sup={st.super_missiles} sel={st.selected_item}",
            f"reason={self.reason[:60]}",
        ]
        y = 4
        for line in lines:
            img = self._font.render(line, True, (255, 255, 0))
            self._screen.blit(img, (4, y))
            y += 18
        pygame.display.flip()

    def step(self, action, reason: str = ""):
        if not self._running:
            raise SystemExit("watch aborted")
        self._handle_events()
        if not self._running:
            raise SystemExit("watch aborted")
        self.reason = reason or ""
        obs, *_ = self.env.step(action)
        self.frame += 1
        self.state = parse_env_state(self.env, frame=self.frame, mode="nav")
        self.assist.apply(self.env.data, self.state)
        self._last_obs = obs
        self._blit(obs)
        # Throttle to wall-clock speed
        self._clock.tick(self._base_fps * self.speed)
        return self.state

    def close(self) -> None:
        try:
            self._pygame.quit()
        except Exception:  # noqa: BLE001
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("segment", choices=sorted(PLAYERS))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument(
        "--speed",
        type=float,
        default=0.5,
        help="Playback rate vs 60fps (0.5 = half speed)",
    )
    parser.add_argument("--no-assist", action="store_true")
    args = parser.parse_args()
    source = args.source
    if not source.is_file():
        print(f"missing source: {source}", file=sys.stderr)
        return 2

    play = PLAYERS[args.segment]
    env = make_dev_env()
    assist = UnlimitedResourcesAssist()
    if args.no_assist:
        # still construct but skip apply by using a no-op wrapper
        class _NoAssist:
            def apply(self, data, state) -> None:
                return None

        assist = _NoAssist()  # type: ignore[assignment]

    boot_from_state(env, source)
    for _ in range(5):
        env.step(idle_action())
        if not args.no_assist:
            assist.apply(env.data, parse_env_state(env, mode="nav"))

    session = WatchSession(
        env,
        assist,  # type: ignore[arg-type]
        scale=args.scale,
        speed=args.speed,
        title=f"pure: {args.segment}",
    )
    print(
        f"[watch] {args.segment} source={source} "
        f"start room=0x{session.state.room_id:04X} "
        f"xy=({session.state.samus_x},{session.state.samus_y}) speed={args.speed}x",
        flush=True,
    )
    try:
        play(session)  # type: ignore[arg-type]
        st = session.state
        beams = int(getattr(st, "collected_beams", 0) or 0)
        print(
            f"[watch] DONE frames={session.frame} room=0x{st.room_id:04X} "
            f"xy=({st.samus_x},{st.samus_y}) pose={st.pose} beams=0x{beams:04X}",
            flush=True,
        )
        # Hold last frame briefly so user can read HUD
        for _ in range(int(90 * args.speed) or 30):
            session._handle_events()
            if not session._running:
                break
            session._blit(session._last_obs)
            session._clock.tick(60)
        return 0
    except SystemExit as exc:
        print(f"[watch] abort: {exc}", flush=True)
        return 130
    except Exception as exc:  # noqa: BLE001
        st = session.state
        print(
            f"[watch] RED {exc}\n"
            f"  pin room=0x{st.room_id:04X} xy=({st.samus_x},{st.samus_y}) "
            f"pose={st.pose} frames={session.frame} reason={session.reason}",
            flush=True,
        )
        # Freeze on fail so user can inspect
        for _ in range(180):
            session._handle_events()
            if not session._running:
                break
            session._blit(session._last_obs)
            session._clock.tick(30)
        return 1
    finally:
        session.close()
        try:
            env.close()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    raise SystemExit(main())
