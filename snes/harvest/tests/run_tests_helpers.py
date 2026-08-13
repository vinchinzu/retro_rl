#!/usr/bin/env python3
"""Shared helpers for the Harvest Moon script-style integration runner."""
from __future__ import annotations

import gzip
import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import stable_retro as retro

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from harvest.runtime import harvest_bot as hb
from harvest.core.tile_catalog import (
    ADDR_X,
    ADDR_Y,
    ADDR_TOOL,
    ADDR_TILEMAP,
    ADDR_MAP,
)
from harvest.tasks.nav import (
    Point,
    MAP_WIDTH,
)
from harvest.runtime.task_recorder import Task

STATES_DIR = hb.STATES_DIR
TASKS_DIR = os.path.join(SCRIPT_DIR, "tasks")


@dataclass
class TestResult:
    name: str
    status: str  # PASS, FAIL, SKIP
    detail: str = ""


def make_env(state: Optional[str] = None):
    kwargs = {
        "game": "HarvestMoon-Snes",
        "inttype": retro.data.Integrations.ALL,
        "use_restricted_actions": retro.Actions.ALL,
        "render_mode": "rgb_array",
    }
    if state:
        kwargs["state"] = state
    return retro.make(**kwargs)


def get_pos(env) -> Point:
    ram = env.get_ram()
    if ADDR_X + 1 >= len(ram) or ADDR_Y + 1 >= len(ram):
        return Point(0, 0)
    x = int(ram[ADDR_X]) + (int(ram[ADDR_X + 1]) << 8)
    y = int(ram[ADDR_Y]) + (int(ram[ADDR_Y + 1]) << 8)
    return Point(x, y)


def get_tool_id(env) -> int:
    ram = env.get_ram()
    return int(ram[ADDR_TOOL]) if ADDR_TOOL < len(ram) else 0


def get_tilemap(env) -> int:
    ram = env.get_ram()
    return int(ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(ram) else 0


def get_water_can_level(env) -> int:
    addr = 0x0926
    ram = env.get_ram()
    return int(ram[addr]) if addr < len(ram) else 0


def get_potato_seeds(env) -> Optional[int]:
    addr = 0x092A
    ram = env.get_ram()
    if addr >= len(ram):
        return None
    return int(ram[addr])


def bcd_to_int(bytes_seq) -> int:
    val = 0
    mult = 1
    for b in bytes_seq:
        b = int(b)
        low = b & 0x0F
        high = (b >> 4) & 0x0F
        val += low * mult
        mult *= 10
        val += high * mult
        mult *= 10
    return val


def get_money_values(env) -> dict:
    ram = env.get_ram()
    values: dict[str, Optional[int]] = {
        "money_bcd": None,
        "money_bcd_mirror": None,
    }
    if len(ram) > 0x0D2:
        values["money_bcd"] = bcd_to_int(ram[0x0D1:0x0D3])
    if len(ram) > 0x40D2:
        values["money_bcd_mirror"] = bcd_to_int(ram[0x40D1:0x40D3])
    return values


def get_money_from_info(info: dict) -> int:
    if "money_bcd_lo" in info and "money_bcd_hi" in info:
        lo_bcd = int(info.get("money_bcd_lo", 0))
        hi_bcd = int(info.get("money_bcd_hi", 0))
        return (
            (lo_bcd & 0x0F)
            + ((lo_bcd >> 4) & 0x0F) * 10
            + (hi_bcd & 0x0F) * 100
            + ((hi_bcd >> 4) & 0x0F) * 1000
        )
    lo = int(info.get("money_lo", 0))
    mid = int(info.get("money_mid", 0))
    hi = int(info.get("money_hi", 0))
    return lo + (mid << 8) + (hi << 16)


def get_potato_seeds_from_info(info: dict) -> int:
    return int(info.get("potato_seeds", 0))


def get_tile_at(env, tx: int, ty: int) -> int:
    ram = env.get_ram()
    if tx < 0 or ty < 0 or tx >= MAP_WIDTH or ty >= MAP_WIDTH:
        return 0
    idx = ty * MAP_WIDTH + tx
    addr = ADDR_MAP + idx
    return int(ram[addr]) if addr < len(ram) else 0


def count_tile_id(env, tile_id: int) -> int:
    ram = env.get_ram()
    end = min(ADDR_MAP + MAP_WIDTH * MAP_WIDTH, len(ram))
    if ADDR_MAP >= end:
        return 0
    data = ram[ADDR_MAP:end]
    return int(np.sum(data == tile_id))


def load_state_bytes(state_name: str) -> Optional[bytes]:
    candidates = [
        os.path.join(STATES_DIR, f"{state_name}.state"),
        os.path.join(TASKS_DIR, f"{state_name}.state"),
        os.path.join(TASKS_DIR, f"{state_name}_end.state"),
    ]
    for state_path in candidates:
        if os.path.exists(state_path):
            with gzip.open(state_path, "rb") as f:
                return f.read()
    return None


def make_env_from_state_bytes(state_bytes: bytes):
    env = make_env()
    obs, info = env.reset()
    env.em.set_state(state_bytes)
    obs, reward, terminated, truncated, info = env.step(np.zeros(12, dtype=np.int32))
    return env, obs, info


def run_task(env, task: Task):
    for frame in task.frames:
        action = np.array(frame, dtype=np.int32)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break


def run_bot(env, bot: hb.AutoClearBot, max_frames: int):
    obs, info = env.reset()
    bot.set_env(env)
    bot.enabled = True
    for _ in range(max_frames):
        game_state = hb.GameState(info)
        action = bot.get_action(game_state, obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    return bot


def require_task(task_name: str) -> Optional[Task]:
    task_path = os.path.join(TASKS_DIR, f"{task_name}.json")
    if not os.path.exists(task_path):
        return None
    return Task.load(task_path)
