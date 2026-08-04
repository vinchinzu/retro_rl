"""Shared CLI helpers for platformer runner commands."""

from __future__ import annotations

import argparse

from retro_harness.platformer.actions import DEFAULT_PLATFORMER_ACTIONS
from retro_harness.platformer.level_config import LevelConfig, get_level_config


def _resolve_config(args: argparse.Namespace) -> LevelConfig:
    """Get level config from --level arg."""
    return get_level_config(args.level)


def _get_action_table(config: LevelConfig) -> list[list[int]]:
    return config.action_table or DEFAULT_PLATFORMER_ACTIONS


def _parse_room_id_arg(value: int | str) -> int:
    """Parse a decimal or hexadecimal room ID from CLI or wrapper arguments."""
    if isinstance(value, int):
        return value
    text = value.strip()
    try:
        return int(text, 0)
    except ValueError:
        return int(text, 16)

