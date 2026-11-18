"""Verify the post-launch input tail on the vanilla ROM."""

import pytest

from super_metroid.combat.escape import play_credits_tail
from super_metroid.dev.common import boot_from_state, make_dev_env
from super_metroid.paths import GAME_DIR, SHARED_ROM
from super_metroid.ram import credits_complete
from super_metroid.splice.assemble import _guard_writes


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
