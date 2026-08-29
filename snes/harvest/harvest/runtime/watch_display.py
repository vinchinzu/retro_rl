"""Headed pygame watch + bot [ ] speed (2x/4x via frame-repeat).

Harvest's pygame window vsyncs on Wayland/X11. Stepping one emu frame then
``Clock.tick(60 * speed)`` cannot beat the display refresh, so 2x/4x looked
like 1x. The editor already uses ``speed_uses_frame_repeat``; bot play and
probe ``--watch`` do the same: N emu steps per 60 Hz present.
"""

from __future__ import annotations

import os
import time

import numpy as np

SPEED_LEVELS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)
DEFAULT_BOT_SPEED = 4.0
DISPLAY_FPS = 60
UNTHROTTLED_FROM = 8.0
TURBO_PREVIEW_INTERVAL = 8


def configure_headed() -> None:
    """Drop dummy/HEADLESS drivers and pick a real SDL video backend."""
    os.environ.pop("HEADLESS", None)
    driver = os.environ.get("SDL_VIDEODRIVER", "").lower()
    if driver == "dummy":
        os.environ.pop("SDL_VIDEODRIVER", None)
        driver = ""
    if "SDL_VIDEODRIVER" not in os.environ:
        if os.environ.get("WAYLAND_DISPLAY"):
            os.environ["SDL_VIDEODRIVER"] = "wayland"
        elif os.environ.get("DISPLAY"):
            os.environ["SDL_VIDEODRIVER"] = "x11"
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("SDL_HINT_RENDER_VSYNC", "0")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")


def configure_headless() -> None:
    os.environ.setdefault("HEADLESS", "1")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.pop("INFINITE_STAMINA", None)


def default_speed_index(speed: float = DEFAULT_BOT_SPEED) -> int:
    levels = SPEED_LEVELS
    return min(range(len(levels)), key=lambda idx: abs(levels[idx] - speed))


def pace_present(tick_fps: int, holder) -> None:
    """Sleep only the leftover time in the present slot.

    ``Clock.tick(60)`` after a vsync ``flip()`` double-waits and pins 2x/4x to
    1x. Subtract time already spent in emu steps + blit.
    """
    now = time.perf_counter()
    if tick_fps <= 0:
        holder._next_present = now
        return
    target_dt = 1.0 / float(tick_fps)
    next_t = float(getattr(holder, "_next_present", 0.0) or 0.0)
    target = next_t + target_dt
    if target < now - target_dt:
        target = now
    while target > now:
        time.sleep(min(0.002, target - now))
        now = time.perf_counter()
    holder._next_present = target


def bot_speed_timing(
    speed: float,
    *,
    turbo: bool = False,
    bot: bool = True,
) -> tuple[int, int, bool]:
    """Return ``(emu_repeat, clock_tick_fps, skip_most_presents)``.

    ``clock_tick_fps`` 0 means unthrottled (``Clock.tick(0)``).
    Bot 2x/4x repeats that many emu frames per 60 Hz present so [ ] is real
    even when the compositor vsyncs. Human play stays 1 emu step / loop.
    """
    speed = float(speed)
    if turbo or speed >= UNTHROTTLED_FROM:
        return 1, 0, True
    if bot and speed >= 2.0:
        return max(1, int(round(speed))), DISPLAY_FPS, False
    tick = max(1, int(round(DISPLAY_FPS * speed)))
    return 1, tick, False


def fast_env_step(env, action, *, update_obs: bool):
    """Step stable-retro; skip the per-frame info dict and optional blit obs."""
    for player, player_action in enumerate(env.action_to_array(action)):
        if env.movie:
            for button_idx in range(env.num_buttons):
                env.movie.set_key(button_idx, player_action[button_idx], player)
        env.em.set_button_mask(player_action, player)
    if env.movie:
        env.movie.step()
    env.em.step()
    env.data.update_ram()
    if update_obs:
        return env._update_obs()
    return None


def display_set_mode(pygame, size: tuple[int, int], *, caption: str | None = None):
    """Open a window with vsync off; fall back from Wayland to X11."""

    def _open():
        try:
            return pygame.display.set_mode(size, vsync=0)
        except TypeError:
            return pygame.display.set_mode(size)

    try:
        screen = _open()
    except pygame.error:
        if os.environ.get("SDL_VIDEODRIVER") == "wayland" and os.environ.get("DISPLAY"):
            pygame.display.quit()
            os.environ["SDL_VIDEODRIVER"] = "x11"
            pygame.display.init()
            screen = _open()
        else:
            raise
    if caption:
        pygame.display.set_caption(caption)
    return screen


class WatchDisplay:
    """Probe-side pygame window: [ ] speed, TAB turbo, ESC/close."""

    def __init__(
        self,
        *,
        scale: int = 3,
        title: str = "Harvest",
        speed: float = DEFAULT_BOT_SPEED,
    ) -> None:
        self.scale = max(1, int(scale))
        self.title = title
        self.speed_idx = default_speed_index(speed)
        self.speed = float(SPEED_LEVELS[self.speed_idx])
        self.closed = False
        self._pg = None
        self._screen = None
        self._obs = None
        self._next_present = 0.0

    def start(self, obs) -> bool:
        configure_headed()
        import pygame

        pygame.init()
        self._pg = pygame
        arr = np.asarray(obs)
        if arr.ndim < 2:
            raise ValueError(f"expected image obs, got shape={arr.shape}")
        h, w = int(arr.shape[0]), int(arr.shape[1])
        caption = f"{self.title}  {self.speed:g}x  [ ] speed  TAB turbo  ESC quit"
        try:
            self._screen = display_set_mode(
                pygame, (w * self.scale, h * self.scale), caption=caption
            )
        except pygame.error as exc:
            print(f"[WATCH] display failed: {exc}", flush=True)
            self.closed = True
            return False
        print(
            f"[WATCH] {self.title} {self.speed:g}x  "
            "[ ] = speed down/up | TAB = turbo | ESC = quit",
            flush=True,
        )
        return self.present(obs, emu_frame=0)

    def pump(self) -> bool:
        if self.closed or self._pg is None:
            return False
        pygame = self._pg
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.closed = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.closed = True
                elif event.key in (pygame.K_LEFTBRACKET, pygame.K_COMMA, pygame.K_MINUS):
                    self._nudge_speed(-1)
                elif event.key in (
                    pygame.K_RIGHTBRACKET,
                    pygame.K_PERIOD,
                    pygame.K_EQUALS,
                    pygame.K_PLUS,
                ):
                    self._nudge_speed(1)
        return not self.closed

    def tab_held(self) -> bool:
        if self._pg is None or self.closed:
            return False
        return bool(self._pg.key.get_pressed()[self._pg.K_TAB])

    def emu_repeat(self) -> int:
        repeat, _tick, _skip = bot_speed_timing(
            self.speed, turbo=self.tab_held(), bot=True
        )
        return repeat

    def present(self, obs, *, emu_frame: int) -> bool:
        if not self.pump():
            return False
        self._obs = obs
        pygame = self._pg
        _repeat, tick_fps, skip = bot_speed_timing(
            self.speed, turbo=self.tab_held(), bot=True
        )
        should_blit = (not skip) or (int(emu_frame) % TURBO_PREVIEW_INTERVAL == 0)
        if should_blit and obs is not None:
            self._blit(obs)
        elif self._screen is not None:
            pygame.event.pump()
        pace_present(tick_fps, self)
        return not self.closed

    def close(self) -> None:
        self.closed = True
        if self._pg is not None:
            try:
                self._pg.quit()
            except Exception:
                pass
            self._pg = None
            self._screen = None

    def _nudge_speed(self, delta: int) -> None:
        self.speed_idx = max(0, min(len(SPEED_LEVELS) - 1, self.speed_idx + delta))
        self.speed = float(SPEED_LEVELS[self.speed_idx])
        print(f"[SPEED] {self.speed:g}x", flush=True)
        if self._pg is not None:
            self._pg.display.set_caption(
                f"{self.title}  {self.speed:g}x  [ ] speed  TAB turbo  ESC quit"
            )

    def _blit(self, obs) -> None:
        pygame = self._pg
        screen = self._screen
        if pygame is None or screen is None:
            return
        arr = np.asarray(obs)
        if arr.ndim != 3 or arr.shape[-1] < 3:
            return
        frame = arr[..., :3]
        h, w = frame.shape[0], frame.shape[1]
        surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        size = (w * self.scale, h * self.scale)
        if screen.get_size() != size:
            screen = display_set_mode(pygame, size)
            self._screen = screen
        scaled = pygame.transform.scale(surf, size)
        screen.blit(scaled, (0, 0))
        pygame.display.flip()


def preview_interval(speed: float) -> int:
    """How often to blit when high-speed autoplay skips most presents."""
    if speed <= 4.0:
        return 1
    if speed <= 8.0:
        return 30
    if speed <= 32.0:
        return 45
    return 60


def gym_env_step(env, action, obs, *, update_obs: bool):
    """Gym-shaped ``(obs, reward, terminated, truncated, info)`` around :func:`fast_env_step`."""
    if env.img is None and env.ram is None:
        raise RuntimeError("Please call env.reset() before stepping")
    new_obs = fast_env_step(env, action, update_obs=update_obs)
    if update_obs and new_obs is not None:
        obs = new_obs
    try:
        terminated = bool(env.data.is_done())
    except Exception:
        terminated = False
    return obs, 0.0, terminated, False, {}


def _hud_count_text(value) -> str:
    return "--" if value is None else str(value)


def _location_text(ram) -> str:
    from harvest.core.tile_catalog import ADDR_TILEMAP
    from harvest.maps.map_config import get_map_name

    tilemap = int(ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(ram) else 0
    return f"{get_map_name(tilemap).replace('_', ' ')} (0x{tilemap:02X})"


def _crop_waterable_count(session, ram, skip_tiles=None) -> int:
    from harvest.tasks.crop_planter import DEFAULT_CROP_BOUNDS, tile_needs_watering
    from harvest.tasks.harvest_task import live_harvestable_crop_tiles
    from harvest.tasks.nav import get_tile_at

    left, top, right, bottom = DEFAULT_CROP_BOUNDS
    if skip_tiles is None:
        state_name = getattr(session.bot, "auto_day_plan_state_name", None)
        skip_tiles = set(live_harvestable_crop_tiles(ram, state_name)) if state_name else set()
    count = 0
    for y in range(top, bottom + 1):
        for x in range(left, right + 1):
            if (x, y) in skip_tiles:
                continue
            if tile_needs_watering(get_tile_at(ram, x, y)):
                count += 1
    return count


def _cached_hud_crop_counts(session, ram) -> tuple:
    from harvest.core.tile_catalog import ADDR_TILEMAP
    from harvest.planner.day_plan import is_farm_tilemap
    from harvest.tasks.harvest_task import live_harvestable_crop_tiles

    tilemap = int(ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(ram) else 0
    if not is_farm_tilemap(tilemap):
        return None, None
    interval = 60 if session._active_harvest_task() is not None else 15
    if session.frame_count - session._hud_counts_frame < interval:
        return session._hud_crop_counts
    state_name = getattr(session.bot, "auto_day_plan_state_name", None)
    ready_tiles = set(live_harvestable_crop_tiles(ram, state_name))
    skip_tiles = ready_tiles if state_name else set()
    session._hud_crop_counts = (len(ready_tiles), _crop_waterable_count(session, ram, skip_tiles=skip_tiles))
    session._hud_counts_frame = session.frame_count
    return session._hud_crop_counts


def _active_task_lines(bot, ram) -> list[str]:
    if getattr(bot, "power_on_enabled", False) and not getattr(bot, "power_on_done", True):
        task = getattr(bot, "power_on_task", None)
        if task is not None and getattr(bot, "power_on_started", False):
            return [f"Power-on: {task.phase_text}", task.progress_text]
        return ["Power-on: waiting"]
    if getattr(bot, "d1_handoff_enabled", False) and not getattr(bot, "d1_handoff_done", True):
        task = getattr(bot, "d1_handoff_task", None)
        if task is not None and getattr(bot, "d1_handoff_started", False):
            snap = task.progress_snapshot()
            return [f"D1 handoff: {snap.phase_text or 'running'}", f"step={snap.step_count}"]
        return ["D1 handoff: waiting"]
    if bot.day_plan_enabled and bot.day_plan_started and not bot.day_plan_done:
        dp = bot.day_plan_task
        lines = [f"Plan: {dp.phase_text}", dp.progress_text]
        task = dp.current_task
        if task is not None and hasattr(task, "progress_text"):
            lines.append(str(task.progress_text))
        return lines
    if bot.crop_enabled and bot.crop_task_started and not bot.crop_task_done:
        ct = bot.crop_task
        return [f"Crop: {ct.phase_text}", ct.progress_text]
    if bot.grass_enabled and bot.grass_task_started and not bot.grass_task_done:
        gt = bot.grass_task
        return [f"Grass: {gt.phase_text}", gt.progress_text]
    return [f"Clearer: {bot.clearer.state}"]


def _target_lines(bot) -> list[str]:
    if bot.day_plan_enabled and bot.day_plan_started and not bot.day_plan_done:
        task = bot.day_plan_task.current_task
        target = getattr(task, "_target_tile", None)
        approach = getattr(task, "_approach_tile", None)
        if target is not None:
            return [f"Target: {target}", f"Stand: {approach}"]
    if bot.crop_enabled and bot.crop_task_started and not bot.crop_task_done:
        ct = bot.crop_task
        if ct._target_tile:
            return [f"Target: {ct._target_tile}", f"Plot: {ct._plot_index + 1}/{len(ct._plots)}"]
    if bot.clearer.current_target:
        t = bot.clearer.current_target
        return [f"Target: {t.debris_type.name}", f"Tile: {t.tile} id=0x{t.tile_id:02X}"]
    return []


def build_session_hud_lines(session, env, game_state, action) -> list[str]:
    from harvest.core.ram_catalog import read_ram_value
    from harvest.planner.day_plan import count_chicken_slots
    from harvest.tasks.nav import TILE_SIZE, get_pos_from_ram
    from retro_harness import SNES_BUTTON_NAMES

    ram = env.get_ram()
    pos = get_pos_from_ram(ram)
    adults, chicks, eggs = count_chicken_slots(ram)
    active_btns = " ".join(SNES_BUTTON_NAMES[i] for i, v in enumerate(action) if v > 0)
    session._note_task_state_for_hud()
    ready_count, waterable_count = _cached_hud_crop_counts(session, ram)
    speed = getattr(session, "_display_speed", 1.0)
    lines = [
        "HARVEST",
        f"Mode {session.mode.upper()}",
        f"Speed {speed:g}x [ ]",
        f"{game_state.date_str}",
        f"{game_state.time_str}",
        f"Loc {_location_text(ram)}",
        f"$ {game_state.money:,}",
        f"Ship ${read_ram_value(ram, 'shipping_money'):,}",
        f"Can {read_ram_value(ram, 'water_can', raw=True)}/20",
        f"Item {game_state.item_name}",
        "",
        "Coop",
        f"A/C/E {adults}/{chicks}/{eggs}",
        f"Fed {read_ram_value(ram, 'fed_chickens_n', raw=True)}",
        f"Egg {read_ram_value(ram, 'egg_available', raw=True)}",
        "",
        "Crops",
        f"Ready {_hud_count_text(ready_count)}",
        f"Unwatered {_hud_count_text(waterable_count)}",
        "",
    ]
    lines.extend(_active_task_lines(session.bot, ram))
    lines.extend(["", f"Pos: ({pos.x // TILE_SIZE},{pos.y // TILE_SIZE})", f"Px: ({pos.x},{pos.y})"])
    lines.extend(_target_lines(session.bot))
    if active_btns:
        lines.append(f"Buttons: {active_btns}")
    if session.bot.disable_reason:
        lines.append(f"Disabled: {session.bot.disable_reason}")
    return lines


def draw_session_hud(session, screen, font, env, game_state, action, height) -> None:
    import pygame

    panel = pygame.Rect(0, 0, session.hud_width, height)
    pygame.draw.rect(screen, (18, 22, 25), panel)
    pygame.draw.line(screen, (72, 82, 88), (session.hud_width - 1, 0), (session.hud_width - 1, height))
    y = 8
    for line in build_session_hud_lines(session, env, game_state, action):
        if not line:
            y += 8
            continue
        color = (210, 232, 218)
        if line.isupper() or line in {"Coop", "Crops"}:
            color = (255, 238, 170)
        screen.blit(font.render(line[:24], True, color), (8, y))
        y += 13
        if y > height - 16:
            break
