"""Unit tests for TAS annotate + frame helpers (no emulator)."""

from __future__ import annotations

from super_metroid.ram import GameplayPhase, SuperMetroidState
from super_metroid.tas.annotate import Annotator, is_settled_control, pose_cluster
from super_metroid.tas.trace import action_array, frame_button_names, resolve_frames


def _state(
    *,
    frame: int = 0,
    game_state: int = 8,
    room_id: int = 0x91F8,
    door_transition: int = 0,
    pose: int = 1,
    x: int = 100,
    y: int = 200,
    items: int = 0,
    beams: int = 0,
    max_missiles: int = 0,
    max_health: int = 99,
    speed_counter: int = 0,
    shinespark_timer: int = 0,
    phase: GameplayPhase | None = None,
) -> SuperMetroidState:
    gs = game_state
    door = door_transition
    if phase is None:
        if gs == 8 and door == 0:
            phase = GameplayPhase.ORDINARY_GAMEPLAY
        elif gs in {0, 1, 2, 3, 4, 5, 6}:
            phase = GameplayPhase.BOOT_OR_MENU
        elif gs in {19, 20}:
            phase = GameplayPhase.DEATH_OR_GAME_OVER
        else:
            phase = GameplayPhase.ROOM_TRANSITION
    return SuperMetroidState(
        frame=frame,
        game_state=gs,
        phase=phase,
        room_id=room_id,
        area_index=0,
        door_transition=door,
        transition_direction=0,
        samus_x=x,
        samus_y=y,
        velocity_x=0,
        velocity_y=0,
        pose=pose,
        health=99,
        max_health=max_health,
        reserve_health=0,
        max_reserve_health=0,
        missiles=0,
        max_missiles=max_missiles,
        super_missiles=0,
        max_super_missiles=0,
        power_bombs=0,
        max_power_bombs=0,
        selected_item=0,
        equipped_items=items,
        collected_items=items,
        equipped_beams=beams,
        collected_beams=beams,
        timer_type=0,
        escape_timer_frames=0,
        escape_timer_seconds=0,
        escape_timer_minutes=0,
        num_enemies=0,
        enemies_killed=0,
        enemy0_x=0,
        enemy0_y=0,
        enemy0_hp=0,
        enemy0_spritemap=0,
        event_flags=(0,) * 8,
        boss_bits=(0,) * 8,
        speed_counter=speed_counter,
        shinespark_timer=shinespark_timer,
    )


def test_frame_button_names_preserves_lr() -> None:
    fr = [0] * 12
    fr[0] = 1  # B
    fr[6] = 1  # LEFT
    fr[7] = 1  # RIGHT
    names = frame_button_names(fr)
    assert "B" in names and "LEFT" in names and "RIGHT" in names
    act = action_array(fr)
    assert int(act[6]) == 1 and int(act[7]) == 1


def test_annotator_first_control_and_room_enter() -> None:
    ann = Annotator(stall_frames=30)
    # Boot
    ann.observe(_state(frame=1, game_state=1, room_id=0, phase=GameplayPhase.BOOT_OR_MENU))
    # First control
    new = ann.observe(_state(frame=100, room_id=0x91F8, x=40, y=180))
    kinds = {e.kind for e in new}
    assert "control" in kinds
    assert "room_enter" in kinds
    assert ann.summary()["first_control_frame"] == 100


def test_annotator_item_and_beam_gain() -> None:
    ann = Annotator()
    ann.observe(_state(frame=1, items=0, beams=0))
    new = ann.observe(_state(frame=2, items=0x0004, beams=0x1000))  # morph + charge
    kinds = {(e.kind, e.detail) for e in new}
    assert ("item_gain", "morph") in kinds
    assert ("beam_gain", "charge") in kinds


def test_annotator_capacity_gain() -> None:
    ann = Annotator()
    ann.observe(_state(frame=1, max_missiles=0, max_health=99))
    new = ann.observe(_state(frame=2, max_missiles=5, max_health=199))
    details = {e.detail for e in new if e.kind == "capacity_gain"}
    assert any("missiles" in d for d in details)
    assert any("energy" in d for d in details)


def test_annotator_speed_echo_and_shine() -> None:
    ann = Annotator()
    ann.observe(_state(frame=1, room_id=0x91F8, speed_counter=0, shinespark_timer=0))
    new = ann.observe(
        _state(frame=2, room_id=0x91F8, speed_counter=4, shinespark_timer=0)
    )
    assert any(e.kind == "speed_echo" and "4" in e.detail for e in new)
    new2 = ann.observe(
        _state(frame=3, room_id=0x91F8, speed_counter=4, shinespark_timer=60)
    )
    assert any(e.kind == "shine_arm" for e in new2)
    new3 = ann.observe(
        _state(frame=4, room_id=0x91F8, speed_counter=0, shinespark_timer=0)
    )
    kinds = {e.kind for e in new3}
    assert "shine_clear" in kinds
    assert "speed_echo_drop" in kinds


def test_annotator_ignores_boot_ram_noise() -> None:
    ann = Annotator()
    ann.observe(
        _state(
            frame=1,
            game_state=1,
            room_id=0,
            max_health=0,
            phase=GameplayPhase.BOOT_OR_MENU,
        )
    )
    new = ann.observe(
        _state(
            frame=2,
            game_state=1,
            room_id=0,
            max_health=99,
            max_missiles=900,
            speed_counter=4,
            shinespark_timer=60,
            phase=GameplayPhase.BOOT_OR_MENU,
        )
    )
    assert not any(
        e.kind in ("capacity_gain", "speed_echo", "shine_arm", "item_gain")
        for e in new
    )


def test_annotator_room_transition() -> None:
    ann = Annotator()
    ann.observe(_state(frame=1, room_id=0x91F8))
    # Leave ordinary
    leave = ann.observe(
        _state(
            frame=2,
            room_id=0x91F8,
            game_state=9,
            door_transition=1,
            phase=GameplayPhase.ROOM_TRANSITION,
        )
    )
    assert any(e.kind == "room_leave" for e in leave)
    # Enter new room
    enter = ann.observe(_state(frame=50, room_id=0x93AA, x=16, y=100))
    assert any(e.kind == "room_enter" and "93AA" in e.detail.upper() for e in enter)


def test_annotator_desync_suspect() -> None:
    ann = Annotator(stall_frames=5)
    ann.observe(_state(frame=1, pose=1, x=10, y=20))
    # Freeze with buttons held; threshold fires once after stall_frames same frames.
    saw = False
    for f in range(2, 12):
        new = ann.observe(
            _state(frame=f, pose=1, x=10, y=20),
            buttons=("RIGHT", "B"),
        )
        if any(e.kind == "desync_suspect" for e in new):
            saw = True
            break
    assert saw, "desync_suspect not emitted"


def test_pose_cluster_morph() -> None:
    assert pose_cluster(0x1D) == "morph"
    assert pose_cluster(0xC9) == "shinespark"
    assert pose_cluster(1) is None
    # spin/walljump off by default
    assert pose_cluster(0x19) is None
    assert pose_cluster(0x19, enabled=frozenset({"spinjump"})) == "spinjump"


def test_is_settled_control() -> None:
    assert is_settled_control(_state(room_id=0x91F8))
    assert not is_settled_control(_state(room_id=0))
    assert not is_settled_control(
        _state(door_transition=1, phase=GameplayPhase.ROOM_TRANSITION)
    )


def test_resolve_frames_slice_menu() -> None:
    from super_metroid.paths import GAME_DIR

    seed = GAME_DIR / "tas" / "slices" / "sniq_any_menu.json"
    if not seed.exists():
        import pytest

        pytest.skip("menu slice missing")
    frames, source = resolve_frames(slice_id="sniq_any_menu")
    assert len(frames) == 600
    assert "sniq_any_menu" in source
    assert len(frames[0]) == 12
