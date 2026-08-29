"""Live Fight_LiuKang health confirmation (real ROM + state)."""

from __future__ import annotations

import os

import pytest

from mortal_kombat_ii.eval_match import make_raw_eval_env, probe_health
from mortal_kombat_ii.ram import MAX_HEALTH

pytestmark = [pytest.mark.rom, pytest.mark.rom_smoke]


def test_fight_liukang_health_is_high_wram_not_020a() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    env = make_raw_eval_env("Fight_LiuKang")
    try:
        probe = probe_health(env)
    finally:
        env.close()
    assert probe["p1_health"] == MAX_HEALTH
    assert probe["p2_health"] == MAX_HEALTH
    assert probe["addr_p1"] == 0x4EFD
    assert probe["addr_p2"] == 0x50AB
    assert probe["decoy_020a"] != MAX_HEALTH
    assert probe["decoy_020e"] != MAX_HEALTH
