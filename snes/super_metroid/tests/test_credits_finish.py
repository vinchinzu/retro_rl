"""Verify the escape finish and the final credits screen on the vanilla ROM."""

import pytest

from retro_harness.env import read_state_bytes
from super_metroid.combat.escape import (
    ROOM_ESCAPE_1,
    ROOM_ESCAPE_2,
    ROOM_ESCAPE_3,
    ROOM_ESCAPE_4,
    play_credits_tail,
)
from super_metroid.combat.mother_brain import ROOM_MOTHER_BRAIN
from super_metroid.dev.common import boot_from_state, make_dev_env
from super_metroid.paths import GAME_DIR, SHARED_ROM
from super_metroid.ram import credits_complete
from super_metroid.routes.kpdr.room_ids import ROOM_CLIMB, ROOM_LANDING_SITE, ROOM_PARLOR
from super_metroid.splice.assemble import _guard_writes
from super_metroid.splice.credits import _play_finish_splice, benchmark_finish_splice


@pytest.mark.rom
@pytest.mark.parametrize(("room", "delay"), [
    (ROOM_MOTHER_BRAIN, 0),
    (ROOM_ESCAPE_1, 17),
    (ROOM_ESCAPE_2, 3),
    (ROOM_ESCAPE_3, 43),
    (ROOM_ESCAPE_4, 17),
    (ROOM_CLIMB, 17),
    (ROOM_PARLOR, 3),
    (ROOM_LANDING_SITE, 43),
])
def test_escape_policy_reaches_credits_from_a_splice(room: int, delay: int) -> None:
    anchors = GAME_DIR / "tasks" / "g4_tourian_human_mb_anchors"
    source = (
        anchors / "f000000_boot_0xDD58.state" if room == ROOM_MOTHER_BRAIN
        else next(anchors.glob(f"*_enter_0x{room:04X}_*.state"), None)
    )
    if not SHARED_ROM.is_file() or source is None or not source.is_file():
        pytest.skip("vanilla ROM and escape checkpoint required")

    if room == ROOM_MOTHER_BRAIN:
        report = benchmark_finish_splice(source, delay=delay)
        assert report["replay_verified"]
        assert report["ending_entry_frame"] < 18_038  # Original live policy split from this pin.
    else:
        report, _ = _play_finish_splice(read_state_bytes(source), delay)
    assert report["credits_complete"]
    assert report["credits_tail_frames"] > 0
    assert report["state_loads_after_startup"] == report["ram_writes"] == report["deaths"] == 0
    assert report["rows"][0]["room"] == f"0x{room:04X}"
    assert report["rows"][-1]["escape_clock"] != "0:00.00"


@pytest.mark.rom
def test_credits_finish_requires_the_full_final_message() -> None:
    ending = GAME_DIR / "tasks" / "g4_tourian_human_mb_end.state"
    if not SHARED_ROM.is_file() or not ending.is_file():
        pytest.skip("vanilla ROM and human ending pin required")

    env = make_dev_env()
    state = boot_from_state(env, ending, settle_frames=0)
    restore_writes = _guard_writes(env)
    try:
        assert state.game_state == 39
        assert not credits_complete(env)

        # This is already item-percentage display, with the final message
        # still being drawn. Earlier idle cinematic functions also use E7BA.
        with pytest.raises(RuntimeError, match="final credits screen did not appear"):
            play_credits_tail(env, timeout_frames=9_926)
        assert not credits_complete(env)

        tail = play_credits_tail(env, timeout_frames=1)
        assert tail.shape == (1, 12)
        assert not tail.any()
        assert credits_complete(env)
    finally:
        restore_writes()
        env.close()
