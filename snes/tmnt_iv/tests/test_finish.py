"""Finish-critical TMNT IV checks: RAM, credits, stalls, pizza, no A."""

from __future__ import annotations

from dataclasses import replace

from retro_harness.ram_state import GameMode, GameState
from tmnt_iv.policy import Stage1Policy
from tmnt_iv.tactics.neon import NeonLaneTactics
from tmnt_iv.ram import (
    ADDR_EVENT,
    ADDR_LIVES,
    ADDR_STAGE,
    BOSS_CHAR_IDS,
    NPC_CHAR_IDS,
    parse_game_state,
)
from tmnt_iv.scripts.record_full_hard_run import (
    CreditsTracker,
    RunMetrics,
    _FINAL_CREDITS_EVENT,
    _FINAL_SCENE_SETTLE_FRAMES,
)
from tmnt_iv.tests._state import A, B, Y, enemy, playing, ram, write_enemy


def test_parse_player_and_living_enemy() -> None:
    buf = ram()
    write_enemy(buf, x=200, y=170, health=16)
    state = parse_game_state(buf)
    assert state.mode is GameMode.PLAYING
    assert state.player_x == 64
    assert state.health == 80
    assert len(state.living_enemies) == 1
    assert state.living_enemies[0].x == 200


def test_non_combat_slots_are_not_living() -> None:
    buf = ram()
    write_enemy(buf, 0, x=208, y=176, health=48, char_id=0xC4)
    write_enemy(buf, 1, x=140, y=180, health=0, char_id=0x30)
    write_enemy(buf, 2, x=0, y=0, health=3, char_id=0x00)
    write_enemy(buf, 3, x=100, y=296, health=2, char_id=0x66)
    state = parse_game_state(buf)
    assert state.living_enemies == ()
    assert state.extras["pickups"] == ((140, 180, 0x30),)
    assert 0xC4 in NPC_CHAR_IDS


def test_boss_chars_stay_active_at_low_hp() -> None:
    for char_id in BOSS_CHAR_IDS:
        buf = ram()
        write_enemy(buf, x=180, y=160, health=24, char_id=char_id)
        state = parse_game_state(buf)
        assert state.boss_active, char_id
        assert state.extras["boss_hp"] == 24


def test_last_life_is_playing_and_ending_is_cutscene() -> None:
    buf = ram()
    buf[ADDR_LIVES] = 0
    buf[ADDR_EVENT] = 0x0A
    buf[ADDR_STAGE] = 1
    live = parse_game_state(buf)
    assert live.mode is GameMode.PLAYING
    assert live.lives == 0

    buf[ADDR_EVENT] = 0x19
    assert parse_game_state(buf).mode is GameMode.CUTSCENE

    buf[ADDR_STAGE] = 10
    buf[ADDR_EVENT] = 0x0A
    assert parse_game_state(buf).mode is GameMode.CUTSCENE


def test_neon_props_are_not_combat_and_jetpack_is_not_boss() -> None:
    buf = ram()
    buf[ADDR_STAGE] = 7
    write_enemy(buf, 0, x=120, y=160, health=2, char_id=0x36)
    write_enemy(buf, 1, x=160, y=160, health=2, char_id=0xAC)
    write_enemy(buf, 2, x=200, y=165, health=2, char_id=0x86)
    write_enemy(buf, 3, x=180, y=160, health=80, char_id=0x1E)
    state = parse_game_state(buf)
    assert {e.kind for e in state.living_enemies} == {0x86, 0x1E}
    assert state.boss_active is False


def test_credits_tracker_marks_complete_after_settle() -> None:
    metrics = RunMetrics()
    tracker = CreditsTracker()
    extras = {"event": _FINAL_CREDITS_EVENT, "menu": 6}

    start = GameState(
        frame=1, mode=GameMode.CUTSCENE, stage=10, health=80, player_x=0, extras=extras
    )
    tracker.update(start, frame=100, metrics=metrics)
    assert metrics.credits_start_frame == 100
    assert metrics.hard_credits_event_seen

    playing_roll = GameState(
        frame=1, mode=GameMode.PLAYING, stage=9, health=80, player_x=10, extras=extras
    )
    idle = replace(playing_roll, player_x=0)
    tracker.update(playing_roll, frame=200, metrics=metrics)
    tracker.update(idle, frame=201, metrics=metrics)
    tracker.update(playing_roll, frame=300, metrics=metrics)
    tracker.update(idle, frame=400, metrics=metrics)
    assert metrics.final_scene_start_frame == 400

    tracker.update(idle, frame=400 + _FINAL_SCENE_SETTLE_FRAMES, metrics=metrics)
    assert metrics.credits_complete_frame == 400 + _FINAL_SCENE_SETTLE_FRAMES


def test_empty_screen_walks_right_without_y_or_a() -> None:
    policy = Stage1Policy()
    for frame in range(1, 60):
        action = policy.tick(playing(frame=frame)).action
        assert action is not None
        assert action.reason == "walk_right"
        assert action.action[Y] == 0
        assert action.action[A] == 0


def test_pizza_seek_is_not_global() -> None:
    foot = enemy(120, 180)
    skull = replace(
        playing(player_x=80, player_y=180, health=24, enemies=(foot,), stage=5),
        extras={"pickups": ((240, 180, 0x30),)},
    )
    reason = Stage1Policy().tick(skull).action.reason
    assert reason not in {"pizza_seek", "pizza_pickup", "pizza_disengage"}

    alley_mid = replace(
        playing(player_x=80, player_y=180, health=24, enemies=(foot,), stage=1),
        extras={"pickups": ((240, 180, 0x30),)},
    )
    assert Stage1Policy().tick(alley_mid).action.reason not in {
        "pizza_seek",
        "pizza_pickup",
        "pizza_disengage",
    }


def test_production_does_not_hijack_for_hazards() -> None:
    state = replace(
        playing(player_x=120, player_y=180, health=28, stage=0),
        extras={"hazards": ((130, 90, 0x36),), "pickups": ()},
    )
    reason = Stage1Policy().tick(state).action.reason
    assert reason not in {"hazard_jump", "hazard_dodge"}


def test_dumpster_unstick_by_stage() -> None:
    policy = Stage1Policy()

    def reasons(stage: int, x: int, n: int = 100) -> list[str]:
        out: list[str] = []
        for frame in range(1, n):
            tick = policy.tick(
                playing(player_x=x, player_y=192, camera_x=frame * 2, frame=frame, stage=stage)
            )
            assert tick.action is not None
            out.append(tick.action.reason)
        return out

    alley = reasons(1, 109)
    assert "stall_down" in alley and "stall_jump_right" in alley

    sewer = reasons(2, 207, n=120)
    assert not any(r.startswith("stall_") for r in sewer)

    starbase = reasons(8, 128)
    assert "stall_down" in starbase and "stall_jump_right" in starbase


def test_starbase_x207_exhausts_right_x126_does_not() -> None:
    policy = Stage1Policy()

    def reasons(x: int) -> list[str]:
        out: list[str] = []
        for frame in range(1, 800):
            tick = policy.tick(
                playing(
                    player_x=x,
                    player_y=194,
                    camera_x=40_000 + frame,
                    frame=frame,
                    stage=8,
                )
            )
            assert tick.action is not None
            out.append(tick.action.reason)
        return out

    looped = reasons(207)
    assert "stall_down" in looped
    assert "starbase_unstick_right" in looped
    assert looped[-1] == "starbase_unstick_right"

    pin = reasons(126)
    assert "stall_down" in pin
    assert "starbase_unstick_right" not in pin


def test_starbase_launch_holds_right() -> None:
    tick = Stage1Policy().tick(playing(player_x=64, player_y=156, stage=8))
    assert tick.action is not None
    assert tick.action.reason == "starbase_launch_right"
    assert tick.action.action[7] == 1
    assert tick.action.action[A] == 0


def test_neon_empty_screen_holds_lane() -> None:
    policy = Stage1Policy()
    right = policy.tick(playing(player_x=50, player_y=180, stage=7)).action
    assert right is not None
    assert right.reason == "neon_drift_right"
    assert right.action[A] == 0

    left = policy.tick(playing(player_x=200, player_y=180, stage=7)).action
    assert left is not None
    assert left.reason == "neon_drift_left"

    wait = policy.tick(playing(player_x=128, player_y=180, stage=7)).action
    assert wait is not None
    assert wait.reason == "neon_wait"

    near = playing(player_x=50, player_y=180, stage=7, enemies=(enemy(120, 160),))
    assert NeonLaneTactics().next(near) is None
    fought = policy.tick(near).action
    assert fought is not None
    assert fought.reason not in {"neon_drift_right", "neon_drift_left", "neon_wait"}

    far = playing(player_x=50, player_y=180, stage=7, enemies=(enemy(120, 80),))
    assert NeonLaneTactics().next(far) is not None
    far_tick = policy.tick(far).action
    assert far_tick is not None
    assert far_tick.reason == "neon_drift_right"


def test_starbase_form1_rail_holds_right() -> None:
    policy = Stage1Policy()
    for frame in range(1, 40):
        tick = policy.tick(
            playing(player_x=229, player_y=156, camera_x=50582 + frame, frame=frame, stage=8)
        )
        assert tick.action is not None
        assert tick.action.reason == "starbase_rail_right"
        assert tick.action.action[7] == 1
        assert tick.action.action[B] == 0
        assert tick.action.action[A] == 0


def test_far_park_is_not_edge_wait() -> None:
    tick = Stage1Policy().tick(playing(player_x=128, player_y=214, enemies=(enemy(286, 214, 8),)))
    assert tick.action is not None
    assert tick.action.reason != "edge_wait"


def test_combat_stall_escape_never_presses_a() -> None:
    state = playing(player_x=54, player_y=151, enemies=(enemy(87, 192, 11),))
    policy = Stage1Policy()
    actions = [policy.tick(state).action for _ in range(242)]
    assert actions[-1] is not None
    assert actions[-1].reason == "combat_stall_escape"
    assert actions[-1].action[A] == 0


def test_alleycat_living_stall_uses_dumpster_not_hop_left() -> None:
    state = playing(
        player_x=109,
        player_y=192,
        stage=1,
        enemies=(enemy(180, 192, kind=0x60),),
    )
    policy = Stage1Policy()
    actions = [policy.tick(state).action for _ in range(241)]
    assert actions[-1] is not None
    assert actions[-1].reason == "stall_down"
    assert actions[-1].action[A] == 0
    assert actions[-1].action[6] == 0


def test_alleycat_fade_does_not_dumpster() -> None:
    policy = Stage1Policy()
    reasons: list[str] = []
    for frame in range(1, 80):
        tick = policy.tick(
            playing(
                player_x=135,
                player_y=186,
                camera_x=frame * 2,
                frame=frame,
                stage=1,
                extras={"event": 0x0B},
            )
        )
        assert tick.action is not None
        reasons.append(tick.action.reason)
        assert tick.action.action[A] == 0
    assert not any(r.startswith("stall_") for r in reasons)


def test_metalhead_stall_does_not_use_dumpster() -> None:
    state = playing(
        player_x=91,
        player_y=192,
        stage=1,
        boss_active=True,
        enemies=(enemy(120, 192, 96, kind=0x46),),
    )
    policy = Stage1Policy()
    actions = [policy.tick(state).action for _ in range(241)]
    assert actions[-1] is not None
    assert actions[-1].reason == "combat_stall_escape"
    assert not str(actions[-1].reason).startswith("stall_")
    assert actions[-1].action[A] == 0


def test_sewer_jumps_near_spike_columns_with_air_lock() -> None:
    def sewer(*, hazards, extras=None, **kwargs):
        payload = {"hazards": hazards, "anim": 0}
        if extras:
            payload.update(extras)
        return replace(
            playing(player_x=200, player_y=192, stage=2, **kwargs),
            extras=payload,
        )

    # hy is board-lane depth, not drop anim — still hop at adx 30.
    retracted = Stage1Policy().tick(sewer(hazards=((230, 80, 0x1C),))).action
    assert retracted is not None
    assert retracted.reason == "sewer_spike_jump"
    assert retracted.action[B] == 1
    assert retracted.action[A] == 0

    far = Stage1Policy().tick(sewer(hazards=((280, 202, 0x1C),))).action
    assert far is not None
    assert far.reason != "sewer_spike_jump"

    hopper = Stage1Policy()
    hop = hopper.tick(sewer(hazards=((230, 202, 0x1C),))).action
    assert hop is not None
    assert hop.reason == "sewer_spike_jump"
    locked = hopper.tick(sewer(hazards=((229, 202, 0x1C),))).action
    assert locked is not None
    assert locked.reason != "sewer_spike_jump"

    foot = Stage1Policy().tick(
        sewer(
            hazards=((230, 202, 0x1C),),
            enemies=(enemy(210, 192, kind=0x60),),
        )
    ).action
    assert foot is not None
    assert foot.reason != "sewer_spike_jump"

    rat = Stage1Policy().tick(
        sewer(
            hazards=((230, 202, 0x1C),),
            boss_active=True,
            enemies=(enemy(180, 192, 96, kind=0x4A),),
        )
    ).action
    assert rat is not None
    assert rat.reason != "sewer_spike_jump"
