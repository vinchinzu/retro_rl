"""Hidden rupee caves on the real ROM: open the secret, take the ROM payout.

Each test boots a ``BFS_<screen>`` overworld pin, makes one disclosed
what-if write so the pin holds the opener the route would (the blue candle,
or bombs; the pins predate both), and runs the production
``make_secret_rupee_controller``. The claim under test is the geometry in
``SECRET_RUPEE_CAVES`` (stand + facing + keeper) against the ROM, which
took three live tries per screen to get right.

    uv run pytest nes/zelda_i/tests/rom/test_secret_caves.py -m rom -q
"""

from __future__ import annotations

import pytest

from retro_harness.env import make_env, reset_obs
from zelda_i.overworld.gather_segments import make_secret_rupee_controller
from zelda_i.overworld.locations import SECRET_RUPEE_CAVES
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import ADDR_BOMBS, ADDR_CANDLE, ADDR_SELECTED_ITEM, read_snapshot
from zelda_i.route.chain import run_controller_stage

from .conftest import ROM, pin_available

B_BOMBS, B_CANDLE = 1, 4


def _take(pin: str, screen: int, writes: dict[int, int]) -> tuple[int, int, object]:
    env = make_env(GAME, pin, GAME_DIR, render_mode="rgb_array")
    try:
        obs, _ = reset_obs(env)
        for addr, value in writes.items():
            env.unwrapped.data.memory.assign(addr, "|u1", value)
        before = read_snapshot(env.get_ram())
        ctl = make_secret_rupee_controller(screen)
        obs, result = run_controller_stage(
            env, obs, name=f"rupees_{screen:02x}", controller=ctl, max_frames=3000,
            potions=False,
        )
        after = read_snapshot(env.get_ram())
        credited = int(after.rupees) + int(after.rupees_to_add)
        return int(before.rupees), credited, result
    finally:
        env.close()


@ROM
@pytest.mark.parametrize(
    ("screen", "pin", "writes"),
    [
        (0x28, "BFS_28", {ADDR_CANDLE: 1, ADDR_SELECTED_ITEM: B_CANDLE}),
        (0x2D, "BFS_2D", {ADDR_BOMBS: 4, ADDR_SELECTED_ITEM: B_BOMBS}),
        (0x6B, "BFS_6B", {ADDR_CANDLE: 1, ADDR_SELECTED_ITEM: B_CANDLE}),
    ],
)
def test_secret_cave_pays_its_rom_payout(screen: int, pin: str, writes: dict[int, int]) -> None:
    if not pin_available(pin):
        pytest.skip(f"Zelda I ROM or {pin!r} missing")
    before, credited, result = _take(pin, screen, writes)
    assert result.success
    assert credited - before == SECRET_RUPEE_CAVES[screen].rupees
