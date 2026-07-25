"""Unit tests for Stage3 West Side helpers."""

from __future__ import annotations

from types import SimpleNamespace

from final_fight.scripts.stage3_advance import (
    W5Tactic,
    _entity_status,
    _lane_vert,
    _w5_dual_action,
    _w5_setup_walk_frames,
    _west_pack_target,
)


def test_lane_vert_matches_memory_y_axis() -> None:
    """UP raises memory Y; DOWN lowers it — aim toward target depth."""
    assert _lane_vert(target_y=70, player_y=50) == "UP"
    assert _lane_vert(target_y=40, player_y=50) == "DOWN"


def test_west_pack_target_prefers_andore() -> None:
    living = [
        SimpleNamespace(health=142, x=800, y=79),
        SimpleNamespace(health=216, x=820, y=79),
    ]
    assert _west_pack_target(living, player_x=750).health == 216


def test_west_pack_target_dual_weak_first() -> None:
    living = [
        SimpleNamespace(health=142, x=829, y=79),
        SimpleNamespace(health=96, x=789, y=61),
    ]
    assert _west_pack_target(living, player_x=750).health == 96


def test_entity_status_defaults_to_combat() -> None:
    assert _entity_status(SimpleNamespace()) == 3
    assert _entity_status(SimpleNamespace(animation=1)) == 1


def test_w5_setup_walk_skipped_on_chip_resume() -> None:
    assert (
        _w5_setup_walk_frames(
            "Stage3_Mid_w5_chip_p31_w59_t142_cam640", "w5"
        )
        == 0
    )
    assert (
        _w5_setup_walk_frames("Stage3_Mid_w5_entry_p31_cam640", "w5")
        == 55
    )
    assert (
        _w5_setup_walk_frames("Stage3_Clear_w4_hp31_cam640", "w5")
        == 55
    )


def test_w5_throw_prefers_upy_in_grab_band() -> None:
    action, reason = _w5_dual_action(
        tactic=W5Tactic.THROW,
        sx=100,
        wdx=-16,
        wadx=16,
        tdx=50,
        dy=2,
        target_hp=22,
        player_hp=20,
        player_y=50,
        target_y=52,
        weak_st=3,
        tough_close=False,
        frame_j=0,
    )
    assert reason == "west_pack_throw"
    assert sum(action) >= 2  # UP+Y held


def test_w5_bait_spaces_when_tough_close() -> None:
    _action, reason = _w5_dual_action(
        tactic=W5Tactic.BAIT,
        sx=90,
        wdx=-40,
        wadx=40,
        tdx=30,
        dy=2,
        target_hp=59,
        player_hp=31,
        player_y=50,
        target_y=52,
        weak_st=3,
        tough_close=False,
        frame_j=4,
    )
    assert reason == "west_pack_space"


def test_w5_kick_uses_jd_in_kick_band() -> None:
    _action, reason = _w5_dual_action(
        tactic=W5Tactic.KICK,
        sx=100,
        wdx=-50,
        wadx=50,
        tdx=80,
        dy=2,
        target_hp=59,
        player_hp=31,
        player_y=50,
        target_y=52,
        weak_st=3,
        tough_close=False,
        frame_j=0,
    )
    assert reason == "west_pack_jd"


def test_w5_split_retargets_tough_while_weak_kd() -> None:
    living = [
        SimpleNamespace(health=142, x=800, y=40, animation=3),
        SimpleNamespace(health=59, x=650, y=50, animation=1),
    ]
    target = _west_pack_target(
        living, player_x=720, tactic=W5Tactic.SPLIT
    )
    assert target.health == 142


def test_w5_split_kd_window_fights_tough() -> None:
    _action, reason = _w5_dual_action(
        tactic=W5Tactic.SPLIT,
        sx=100,
        wdx=-50,
        wadx=50,
        tdx=-50,
        dy=2,
        target_hp=142,
        player_hp=49,
        player_y=40,
        target_y=40,
        weak_st=3,
        tough_close=False,
        frame_j=0,
    )
    assert reason == "west_pack_face"
