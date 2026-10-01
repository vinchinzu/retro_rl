"""Sanctuary arrival is weaker than completed Zelda delivery."""
from dataclasses import replace

from alttp.opening_route.escort_to_sanctuary import evaluate_acceptance
from alttp.ram import AlttpSnapshot


def test_sanctuary_rejects_arrival_and_partial_story() -> None:
    arrived = AlttpSnapshot(
        game_mode=7, submodule=0, room_id=0x12, indoors=True, screen_id=0,
        link_x=1272, link_y=568, link_direction=2, link_action=0,
        camera_x=1152, camera_y=512, dark_world=False, follower=1,
    )
    assert not evaluate_acceptance(arrived, progress=1, progress_flags=0x15)["sanctuary_completed"]
    delivered = replace(arrived, follower=0)
    assert not evaluate_acceptance(delivered, progress=1, progress_flags=0x15)["sanctuary_completed"]
    assert not evaluate_acceptance(delivered, progress=2, progress_flags=0x11)["sanctuary_completed"]
    speaking = replace(delivered, game_mode=14, submodule=2)
    assert not evaluate_acceptance(speaking, progress=2, progress_flags=0x15)["sanctuary_completed"]
    assert evaluate_acceptance(delivered, progress=2, progress_flags=0x15)["sanctuary_completed"]
