from types import SimpleNamespace

from zelda_i.level5_overworld import (
    LEVEL5_PATH_HOPS,
    POST_L4_TO_LEVEL5_HOPS,
    Level5NavPhase,
    OverworldToLevel5Controller,
)


def test_post_l4_path_preserves_verified_level5_suffix() -> None:
    suffix = POST_L4_TO_LEVEL5_HOPS[-len(LEVEL5_PATH_HOPS) :]
    assert suffix == LEVEL5_PATH_HOPS


def test_post_l4_path_returns_from_raft_island_and_joins_4a() -> None:
    targets = [hop.target for hop in POST_L4_TO_LEVEL5_HOPS]
    assert targets[:7] == [0x55, 0x56, 0x57, 0x58, 0x59, 0x49, 0x4A]
    assert POST_L4_TO_LEVEL5_HOPS[0].direction == "DOWN"
    assert POST_L4_TO_LEVEL5_HOPS[0].align_x == 128


def test_post_l4_56_entry_uses_open_center_channel() -> None:
    hop = POST_L4_TO_LEVEL5_HOPS[2]
    assert hop.target == 0x57
    assert hop.direction == "RIGHT"
    assert hop.align_y == 141


def test_lost_hills_east_ledge_steps_left_before_down() -> None:
    controller = OverworldToLevel5Controller()
    controller.phase = Level5NavPhase.FREE_POCKET
    controller.phase_frames = 1

    action = controller._free_pocket(
        SimpleNamespace(screen=0x1B, link_x=240, link_y=141)
    )

    assert action.reason.startswith("pocket_unwedge")
