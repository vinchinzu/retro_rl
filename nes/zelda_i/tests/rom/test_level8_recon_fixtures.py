"""Level 8 interior fixture pins: boot the real ROM, assert live RAM.

``nes/zelda_i/level8/recon.py`` documents an approximate ``keys_in`` /
``bombs_in`` / ``entry_pose`` per fixture (from the *entry* frame of the
route, before the ``.state`` was actually saved). Loading each pin live and
reading two idle frames later shows the real save can drift by a spend or a
few walk pixels from that entry snapshot (see notes below) -- so this file
asserts the values actually measured live this session, not the recon
module's approximate ones. That is the point of a ROM tier: a stale doc
disagreeing with a live read is exactly the drift a fake-RAM unit test can
never see.

Each case: boot ``make_env`` on the named pin, step 2 idle frames (matching
the probe that produced these numbers), then assert ``level``, ``mode``,
``screen`` (the dest room id), ``triforce``, keys/bombs, ``room_item_id``,
Link's pose and the live enemy census (``(type_id, hp)`` for every hp>0,
type_id!=0 object slot 1-12). A change to any of these on a fixed pin is a
real regression in RAM decoding or a rotted fixture -- not a reason string.

    uv run pytest nes/zelda_i/tests/rom/test_level8_recon_fixtures.py -m rom -q
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_idle_action
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import health_byte_is_coherent, read_snapshot

from .conftest import ROM, pin_available

IDLE_SETTLE_FRAMES = 2


@dataclass(frozen=True)
class L8Pin:
    fixture: str
    level: int
    mode: int
    screen: int
    keys: int
    bombs: int
    room_item_id: int
    xy: tuple[int, int]
    triforce: int
    census: tuple[tuple[int, int], ...]  # sorted (type_id, hp)


# Measured live this session (configure_headless -> make_env -> reset_obs ->
# 2 idle frames -> read_snapshot). See module docstring for why these differ
# slightly from `zelda_i.level8.recon`'s documented entry-frame values.
PINS: tuple[L8Pin, ...] = (
    L8Pin(
        "Level8Interior3EReconFixture", 8, 5, 0x3E, 9, 7, 0x03, (120, 205), 0x7F,
        ((12, 128),) * 6,
    ),
    L8Pin(
        "Level8Interior2EReconFixture", 8, 5, 0x2E, 9, 6, 0x17, (120, 189), 0x7F,
        ((60, 64),) * 5 + ((86, 240),) * 4,
    ),
    L8Pin(
        "Level8Interior1EReconFixture", 8, 5, 0x1E, 8, 6, 0x03, (120, 205), 0x7F,
        ((51, 96), (85, 192), (85, 192), (86, 240)),
    ),
    L8Pin(
        "Level8Interior1FReconFixture", 8, 5, 0x1F, 8, 6, 0x03, (16, 141), 0x7F,
        ((11, 64), (11, 64), (12, 128), (12, 128), (22, 160), (22, 160), (104, 176)),
    ),
    L8Pin(
        "Level8InteriorMKReconFixture", 8, 9, 0x0F, 8, 6, 0x0B, (136, 141), 0x7F,
        (),
    ),
    L8Pin(
        "Level8Interior1EWestReconFixture", 8, 5, 0x1E, 8, 6, 0x03, (208, 141), 0x7F,
        ((85, 192),) * 4,
    ),
    L8Pin(
        "Level8Interior3FEastReconFixture", 8, 5, 0x3F, 8, 6, 0x00, (32, 141), 0x7F,
        ((11, 64), (11, 64), (12, 128), (43, 240), (43, 240), (48, 112), (48, 112), (48, 112)),
    ),
    L8Pin(
        "Level8Interior2FCellarReconFixture", 8, 9, 0x2F, 8, 6, 0x03, (192, 93), 0x7F,
        (),
    ),
    L8Pin(
        "Level8Interior4CWestReconFixture", 8, 5, 0x4C, 8, 6, 0x19, (112, 125), 0x7F,
        ((22, 160),) * 8,
    ),
    L8Pin(
        "Level8Interior3CNorthReconFixture", 8, 5, 0x3C, 8, 5, 0x1A, (120, 189), 0x7F,
        ((69, 160),),
    ),
    L8Pin(
        "Level8Interior2CTriforceReconFixture", 8, 5, 0x2C, 8, 5, 0x1B, (120, 205), 0x7F,
        (),
    ),
)


def _live_census(snap) -> tuple[tuple[int, int], ...]:
    return tuple(
        sorted(
            (int(o.type_id), int(o.hp))
            for o in snap.objects
            if 1 <= int(o.slot) <= 12 and int(o.hp) > 0 and int(o.type_id) != 0
        )
    )


@ROM
@pytest.mark.parametrize("pin", PINS, ids=[p.fixture for p in PINS])
def test_level8_fixture_live_ram(pin: L8Pin) -> None:
    if not pin_available(pin.fixture):
        pytest.skip(f"Zelda I ROM or {pin.fixture!r} fixture missing")
    env = make_env(GAME, pin.fixture, GAME_DIR, render_mode="rgb_array")
    try:
        reset_obs(env)
        for _ in range(IDLE_SETTLE_FRAMES):
            env.step(nes_idle_action())
        snap = read_snapshot(env.get_ram())
        assert health_byte_is_coherent(int(snap.health))
        assert int(snap.level) == pin.level
        assert int(snap.mode) == pin.mode
        assert int(snap.screen) == pin.screen
        assert int(snap.triforce) == pin.triforce
        assert int(snap.keys) == pin.keys
        assert int(snap.bombs) == pin.bombs
        assert int(snap.room_item_id) == pin.room_item_id
        assert (int(snap.link_x), int(snap.link_y)) == pin.xy
        assert _live_census(snap) == pin.census
    finally:
        env.close()
