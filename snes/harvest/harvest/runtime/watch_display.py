"""Harvest session HUD. Watch/speed live in ``retro_harness.headed``."""

from __future__ import annotations

from retro_harness.headed import (
    SPEED_LEVELS,
    WatchDisplay,
    bot_speed_timing,
    configure_headed,
    configure_headless as _configure_headless,
    default_speed_index as _default_speed_index,
    display_set_mode,
    fast_env_step,
    gym_env_step,
    pace_present,
    preview_interval,
)

DEFAULT_BOT_SPEED = 4.0
DISPLAY_FPS = 60
UNTHROTTLED_FROM = 8.0
TURBO_PREVIEW_INTERVAL = 8


def configure_headless() -> None:
    _configure_headless()
    import os

    os.environ.pop("INFINITE_STAMINA", None)


def default_speed_index(speed: float = DEFAULT_BOT_SPEED) -> int:
    return _default_speed_index(speed)


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
