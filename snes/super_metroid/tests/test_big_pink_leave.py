"""Clean Big Pink leave (board row 41): drop kinds and the master-leave ROM run."""

from __future__ import annotations

import pytest

from super_metroid.routes.kpdr.brinstar import big_pink


def test_drop_kind_follows_the_pickup_loop() -> None:
    """Zeb drops are projectile 0xF345; the settled loop names the item."""
    ram = bytearray(0x2000)

    def put(slot: int, loop: int) -> None:
        for base, value in ((big_pink._ADDR_PROJ_ID, big_pink._DROP_HEADER),
                            (big_pink._ADDR_PROJ_LIST, loop),
                            (big_pink._ADDR_PROJ_X, 640 + slot),
                            (big_pink._ADDR_PROJ_Y, 536)):
            ram[base + 2 * slot] = value & 0xFF
            ram[base + 2 * slot + 1] = value >> 8

    put(0, 0xEDBD)  # missile loop
    put(1, 0xED95)  # small energy loop
    put(2, 0xEDAB)  # big energy loop

    class Env:
        def get_ram(self) -> bytearray:
            return ram

    kinds = [kind for kind, _, _ in big_pink._drops(Env())]
    assert kinds == ["missile", "small", "big"]


def test_human_climb_has_no_shots_or_item_select() -> None:
    from super_metroid.paths import GAME_DIR

    if not (GAME_DIR / big_pink.HUMAN_TAPE).is_file():
        pytest.skip("human tape not present in clone")
    body, where = big_pink._human_climb()
    assert body.shape == (big_pink.HUMAN_SPAN[1] - big_pink.HUMAN_SPAN[0] + 1, 12)
    assert not body[:, big_pink._X].any() and not body[:, big_pink._SELECT].any()
    assert where.shape == (len(body), 2)


@pytest.mark.rom
def test_rom_leave_from_master_farms_opens_door_and_enters_kihunter_room() -> None:
    """From the master's Big Pink leave: no missiles in, Kihunter room out, no energy lost."""
    from super_metroid.dev.common import boot_from_state, make_dev_env
    from super_metroid.paths import SCRATCH_STATE_DIR
    from super_metroid.ram import parse_env_state

    pin = SCRATCH_STATE_DIR / "master_tape_clean.state"
    if not pin.is_file():
        pytest.skip("master tape endpoint state not present in clone")

    class Session:
        def __init__(self, env):
            self.env, self.frame = env, 0
            self.state = parse_env_state(env, mode="nav")

        def step(self, action, reason):
            self.env.step(action)
            self.frame += 1
            self.state = parse_env_state(self.env, frame=self.frame, mode="nav")
            return self.state

    env = make_dev_env()
    try:
        boot_from_state(env, pin, settle_frames=0)
        session = Session(env)
        before = session.state
        assert before.room_id == big_pink.ROOM_BIG_PINK and before.missiles == 0
        result = big_pink.play_clean_big_pink_leave(session)
        assert session.state.room_id == big_pink.ROOM_KIHUNTER
        assert session.state.game_state == 8 and not session.state.door_transition
        assert session.state.health >= before.health
        assert result.report["drops"] >= 3
        assert result.fight.rollout_restores > 0
    finally:
        env.close()
