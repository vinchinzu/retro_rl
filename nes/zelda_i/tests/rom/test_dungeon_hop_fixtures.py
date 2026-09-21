"""Controller-driven fixture pins: boot the real ROM, play, assert live RAM.

Unlike ``test_level8_recon_fixtures.py`` (static pin content), these two
tests actually drive the production hop controller from its pin and let the
real ROM decide whether it arrives. Both are walk-only room crossings (no
combat, no bomb spend) picked because they are cheap (a few hundred frames)
and, per ``nes/zelda_i/AGENTS.md``, "OccupancyWalker is banned" / "No RAM
writes" in both source rooms -- these are exactly the small, well-understood
hops the fake-RAM unit tests for these modules already cover in isolation;
this file is the live half of that contract.

    uv run pytest nes/zelda_i/tests/rom/test_dungeon_hop_fixtures.py -m rom -q
"""

from __future__ import annotations

import pytest

from retro_harness.env import make_env, reset_obs
from zelda_i.level7.cellar import DEST_ROOM as L7_NOSE_DEST, make_nose_cellar_cross_controller
from zelda_i.level8.passage import DEST as L8_PASSAGE_DEST, DEST_POSE as L8_PASSAGE_POSE, make_passage_2f_controller
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot
from zelda_i.route.chain import run_controller_stage

from .conftest import ROM, pin_available

_L8_PASSAGE_FIXTURE = "Level8Interior2FCellarReconFixture"
_L7_NOSE_FIXTURE = "Level7Interior0DNoseCellarReconFixture"


@ROM
@pytest.mark.skipif(
    not pin_available(_L8_PASSAGE_FIXTURE),
    reason=f"Zelda I ROM or {_L8_PASSAGE_FIXTURE!r} fixture missing",
)
def test_live_passage_2f_reaches_0x4c() -> None:
    """L8 cellar 0x2F east-ladder spawn -> floor cross -> play 0x4C.

    Measured this session: end_frame 356, arrival exactly ``DEST_POSE``
    (112, 125), keys/bombs untouched (8/6 in, 8/6 out -- a floor walk, no
    bomb or key spend), triforce byte unchanged at 0x7F.
    """
    env = make_env(GAME, _L8_PASSAGE_FIXTURE, GAME_DIR, render_mode="rgb_array")
    try:
        obs, _ = reset_obs(env)
        ctl = make_passage_2f_controller()
        obs, result = run_controller_stage(
            env, obs, name="passage_2f", controller=ctl, max_frames=4000
        )
        snap = read_snapshot(env.get_ram())
        assert result.success, ctl.notes
        assert not ctl.failed
        assert int(snap.level) == 8
        assert int(snap.screen) == L8_PASSAGE_DEST == 0x4C
        assert (int(snap.link_x), int(snap.link_y)) == L8_PASSAGE_POSE == (112, 125)
        assert int(snap.keys) == 8
        assert int(snap.bombs) == 6
        assert int(snap.triforce) == 0x7F
        assert 100 <= result.end_frame <= 1500
    finally:
        env.close()


@ROM
@pytest.mark.skipif(
    not pin_available(_L7_NOSE_FIXTURE),
    reason=f"Zelda I ROM or {_L7_NOSE_FIXTURE!r} fixture missing",
)
def test_live_nose_cellar_cross_reaches_0x29() -> None:
    """L7 nose cellar 0x7B B-side spawn -> floor cross -> play 0x29.

    Measured this session: end_frame 370, arrival play 0x29 at (96, 157),
    keys=2 bombs=6 (pin's own count, untouched by this walk-only hop),
    triforce byte 0x00 (this fixture predates L1-L6 TF on this pin).
    """
    env = make_env(GAME, _L7_NOSE_FIXTURE, GAME_DIR, render_mode="rgb_array")
    try:
        obs, _ = reset_obs(env)
        ctl = make_nose_cellar_cross_controller()
        obs, result = run_controller_stage(
            env, obs, name="nose_cellar_cross", controller=ctl, max_frames=4000
        )
        snap = read_snapshot(env.get_ram())
        assert result.success, ctl.notes
        assert not ctl.failed
        assert int(snap.level) == 7
        assert int(snap.screen) == L7_NOSE_DEST == 0x29
        assert (int(snap.link_x), int(snap.link_y)) == (96, 157)
        assert int(snap.keys) == 2
        assert int(snap.bombs) == 6
        assert 100 <= result.end_frame <= 1500
    finally:
        env.close()
