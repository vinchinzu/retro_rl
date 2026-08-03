from __future__ import annotations

import inspect
from collections.abc import Callable

from super_metroid.progression import START_TO_SPEED_GRAPH
from super_metroid.routes.kpdr import bubble_mountain, k4_norfair


CAPS = frozenset(
    {
        "morph_ball",
        "bombs",
        "missiles",
        "super_missiles",
        "hi_jump",
        "varia_suit",
    }
)


def test_k4_norfair_segment_callables_are_importable() -> None:
    segments: tuple[Callable[..., object], ...] = (
        k4_norfair.play_business_to_frog_save,
        k4_norfair.play_frog_save_to_business,
        k4_norfair.play_business_to_cathedral_entrance,
        k4_norfair.play_cathedral_entrance_to_cathedral,
        k4_norfair.play_cathedral_to_rising_tide,
        k4_norfair.play_rising_tide_to_bubble,
        k4_norfair.play_bubble_to_bat_cave,
        k4_norfair.play_frog_save_to_speedway,
        k4_norfair.play_speedway_to_farm,
        k4_norfair.play_farm_to_bubble,
    )

    assert all(callable(segment) for segment in segments)


def test_cathedral_climb_segments_are_registered() -> None:
    from super_metroid.routes.kpdr import get_segment

    assert (
        get_segment("business_to_cathedral_entrance")
        is k4_norfair.play_business_to_cathedral_entrance
    )
    assert get_segment("frog_save_to_business") is k4_norfair.play_frog_save_to_business


def test_business_to_frog_is_registered_for_pure_segment_use() -> None:
    from super_metroid.routes.kpdr import get_segment

    assert get_segment("business_to_frog_save") is k4_norfair.play_business_to_frog_save


def test_business_to_cathedral_entrance_is_registered_for_pure_segment_use() -> None:
    from super_metroid.routes.kpdr import get_segment

    assert (
        get_segment("business_to_cathedral_entrance")
        is k4_norfair.play_business_to_cathedral_entrance
    )


def test_cathedral_entrance_to_cathedral_is_registered_for_pure_segment_use() -> None:
    from super_metroid.routes.kpdr import get_segment

    assert (
        get_segment("cathedral_entrance_to_cathedral")
        is k4_norfair.play_cathedral_entrance_to_cathedral
    )


def test_cathedral_to_rising_tide_is_registered_for_pure_segment_use() -> None:
    from super_metroid.routes.kpdr import get_segment

    assert (
        get_segment("cathedral_to_rising_tide")
        is k4_norfair.play_cathedral_to_rising_tide
    )


def test_rising_tide_to_bubble_is_registered_for_pure_segment_use() -> None:
    from super_metroid.routes.kpdr import get_segment

    assert (
        get_segment("rising_tide_to_bubble")
        is k4_norfair.play_rising_tide_to_bubble
    )


def test_bubble_to_bat_cave_is_registered_for_pure_segment_use() -> None:
    from super_metroid.routes.kpdr import get_segment

    assert (
        get_segment("bubble_to_bat_cave")
        is k4_norfair.play_bubble_to_bat_cave
    )


def test_frog_save_to_speedway_is_registered_for_pure_segment_use() -> None:
    from super_metroid.routes.kpdr import get_segment

    assert (
        get_segment("frog_save_to_speedway")
        is k4_norfair.play_frog_save_to_speedway
    )


def test_speedway_to_farm_is_registered_for_pure_segment_use() -> None:
    from super_metroid.routes.kpdr import get_segment

    assert get_segment("speedway_to_farm") is k4_norfair.play_speedway_to_farm


def test_k4_norfair_constants_match_graph_path() -> None:
    """No-Speed first Bubble visit is Cathedral climb (not Frog Speedway)."""
    path = START_TO_SPEED_GRAPH.shortest_path(
        k4_norfair.ROOM_BUSINESS,
        k4_norfair.ROOM_BUBBLE,
        CAPS,
    )

    assert path is not None
    assert [(edge.source_room_id, edge.target_room_id) for edge in path] == [
        (k4_norfair.ROOM_BUSINESS, 0xA7B3),  # cathedral entrance
        (0xA7B3, 0xA788),  # cathedral
        (0xA788, 0xAFA3),  # rising tide
        (0xAFA3, k4_norfair.ROOM_BUBBLE),
    ]
    assert [edge.edge_id for edge in path] == [
        "business_to_cathedral_entrance",
        "cathedral_entrance_to_cathedral",
        "cathedral_to_rising_tide",
        "rising_tide_to_bubble",
    ]


def test_k4_norfair_key_rooms_match_route_contract() -> None:
    assert k4_norfair.ROOM_BUSINESS == 0xA7DE
    assert k4_norfair.ROOM_BUBBLE == 0xACB3
    assert k4_norfair.ROOM_BAT_CAVE == 0xB07A
    assert k4_norfair.ROOM_SPEED == 0xAD1B
    assert k4_norfair.ROOM_CATHEDRAL_ENTRANCE == 0xA7B3
    assert k4_norfair.ROOM_CATHEDRAL == 0xA788
    assert k4_norfair.ROOM_RISING_TIDE == 0xAFA3


class _FakeState:
    def __init__(
        self,
        *,
        room_id: int = k4_norfair.ROOM_BUBBLE,
        x: int = 0,
        y: int = 0,
        pose: int = 1,
        velocity_x: int = 0,
        velocity_y: int = 0,
    ) -> None:
        self.room_id = room_id
        self.samus_x = x
        self.samus_y = y
        self.pose = pose
        self.velocity_x = velocity_x
        self.velocity_y = velocity_y


def test_bubble_phase_c_usable_right_contact_band() -> None:
    """Phase C: right structure at height — not floor thrash max_x."""
    # Natural-ish first contact band.
    assert k4_norfair.bubble_phase_c_usable_right_contact(
        _FakeState(x=340, y=410)  # type: ignore[arg-type]
    )
    assert k4_norfair.bubble_phase_c_usable_right_contact(
        _FakeState(x=300, y=430)  # type: ignore[arg-type]
    )
    # Below usable altitude (cavity floor thrash).
    assert not k4_norfair.bubble_phase_c_usable_right_contact(
        _FakeState(x=340, y=484)  # type: ignore[arg-type]
    )
    # Too far left (lip peak class).
    assert not k4_norfair.bubble_phase_c_usable_right_contact(
        _FakeState(x=150, y=260)  # type: ignore[arg-type]
    )
    # Outside Bubble.
    assert not k4_norfair.bubble_phase_c_usable_right_contact(
        _FakeState(room_id=k4_norfair.ROOM_BAT_CAVE, x=340, y=300)  # type: ignore[arg-type]
    )


def test_bubble_phase_d_top_band() -> None:
    assert k4_norfair.bubble_phase_d_top_band(
        _FakeState(x=320, y=180)  # type: ignore[arg-type]
    )
    assert not k4_norfair.bubble_phase_d_top_band(
        _FakeState(x=320, y=260)  # type: ignore[arg-type]
    )
    assert not k4_norfair.bubble_phase_d_top_band(
        _FakeState(x=200, y=180)  # type: ignore[arg-type]
    )


def test_bubble_phase_stop_message_includes_pin() -> None:
    st = _FakeState(x=360, y=400, pose=25, velocity_x=3, velocity_y=5)
    exc = k4_norfair.BubblePhaseStop(
        "C",
        st,  # type: ignore[arg-type]
        metrics={"min_y": 260},
    )
    assert exc.phase == "C"
    assert "bubble_phase_stop:C" in str(exc)
    assert "xy=(360,400)" in str(exc)
    assert exc.metrics["min_y"] == 260


def test_play_bubble_to_bat_cave_product_api_has_no_recon_kwargs() -> None:
    """Product play accepts session only — recon flags live on dev helpers."""
    sig = inspect.signature(k4_norfair.play_bubble_to_bat_cave)
    assert list(sig.parameters) == ["session"]
    sig_bm = inspect.signature(bubble_mountain.play_bubble_to_bat_cave)
    assert list(sig_bm.parameters) == ["session"]
    # Dev helpers exist and are distinct callables.
    assert callable(bubble_mountain.play_bubble_climb_from_handoff)
    assert callable(bubble_mountain.play_bubble_from_top_door)
    assert callable(bubble_mountain.play_bubble_to_bat_cave_with_phase_capture)
    assert callable(bubble_mountain.capture_bubble_phase_c)


def test_bubble_phase_helpers_are_importable() -> None:
    helpers = (
        bubble_mountain.bubble_land_and_prepare,
        bubble_mountain.bubble_lower_to_mid_pin,
        bubble_mountain.bubble_mid_repin,
        bubble_mountain.bubble_run_mid,
        bubble_mountain.bubble_top_super_door,
        bubble_mountain.bubble_phase_d_near_top,
        bubble_mountain.bubble_is_true_ground,
        bubble_mountain.bubble_on_mid_iso_pin,
        bubble_mountain.bubble_on_launch_lip,
        bubble_mountain.bubble_on_right_shelf,
        bubble_mountain.bubble_on_save_runway,
    )
    assert all(callable(h) for h in helpers)


def test_bubble_true_ground_excludes_spin_apex() -> None:
    """R11: pose 25 + vy≈0 is not true ground (spin-apex false-land)."""
    spin_apex = _FakeState(x=150, y=300, pose=25, velocity_y=0)
    stand = _FakeState(x=150, y=300, pose=1, velocity_y=0)
    assert not bubble_mountain.bubble_is_true_ground(spin_apex)  # type: ignore[arg-type]
    assert bubble_mountain.bubble_is_true_ground(stand)  # type: ignore[arg-type]
    # Pose 2 / 9 / 10 also true ground when vy quiet.
    for pose in (2, 9, 10):
        assert bubble_mountain.bubble_is_true_ground(
            _FakeState(pose=pose, velocity_y=0)  # type: ignore[arg-type]
        )
    # Rising spin is not ground.
    assert not bubble_mountain.bubble_is_true_ground(
        _FakeState(pose=1, velocity_y=3)  # type: ignore[arg-type]
    )


def test_bubble_mid_iso_pin_accepts_pose_26() -> None:
    """Mid-iso handoff (pose 26 in pin band) still uses stand_pin class."""
    pin = _FakeState(x=98, y=374, pose=26, velocity_y=1)
    assert bubble_mountain.bubble_on_mid_iso_pin(pin)  # type: ignore[arg-type]
    # Outside x band is not mid-iso pin.
    assert not bubble_mountain.bubble_on_mid_iso_pin(
        _FakeState(x=200, y=374, pose=26, velocity_y=0)  # type: ignore[arg-type]
    )
    # True ground pose 1 in band also counts as pin class.
    assert bubble_mountain.bubble_on_mid_iso_pin(
        _FakeState(x=100, y=370, pose=1, velocity_y=0)  # type: ignore[arg-type]
    )


def test_bubble_launch_lip_uses_stand_pin_not_true_ground_only() -> None:
    """Lip seat detection must accept pose 25/26 (pre-extract HEAD / R6).

    Extract regression: true_ground-only on lip blocked natural pure launch
    (launched=False thrash, min_y~365, frames~56k). Climb reseat stays
    true_ground-only (R11 spin-apex fix).
    """
    lip_xy = dict(x=79, y=427, velocity_y=0)
    for pose in (1, 2, 9, 10, 25, 26, 27, 28):
        st = _FakeState(pose=pose, **lip_xy)
        assert bubble_mountain.bubble_on_launch_lip(st)  # type: ignore[arg-type]
        assert bubble_mountain.bubble_is_stand_pin_pose(st)  # type: ignore[arg-type]
    # Spin apex is stand_pin but NOT true ground.
    apex = _FakeState(pose=25, **lip_xy)
    assert bubble_mountain.bubble_is_stand_pin_pose(apex)  # type: ignore[arg-type]
    assert not bubble_mountain.bubble_is_true_ground(apex)  # type: ignore[arg-type]
    # Off lip x/y or rising: not launch seat.
    assert not bubble_mountain.bubble_on_launch_lip(
        _FakeState(x=50, y=427, pose=1, velocity_y=0)  # type: ignore[arg-type]
    )
    assert not bubble_mountain.bubble_on_launch_lip(
        _FakeState(x=79, y=500, pose=1, velocity_y=0)  # type: ignore[arg-type]
    )
    assert not bubble_mountain.bubble_on_launch_lip(
        _FakeState(x=79, y=427, pose=1, velocity_y=3)  # type: ignore[arg-type]
    )


def test_bubble_save_runway_seat() -> None:
    """R14/R15: save-door outer runway (max-left human pin) is stand-pin band."""
    for pose in (1, 2, 9, 10, 25, 26):
        assert bubble_mountain.bubble_on_save_runway(
            _FakeState(x=55, y=395, pose=pose, velocity_y=0)  # type: ignore[arg-type]
        )
    # R15 max-left fire seat (~human x27).
    assert bubble_mountain.bubble_on_save_runway(
        _FakeState(x=27, y=395, pose=2, velocity_y=0)  # type: ignore[arg-type]
    )
    # Too close to Save door shell / too far right / wrong height.
    assert not bubble_mountain.bubble_on_save_runway(
        _FakeState(x=20, y=395, pose=1, velocity_y=0)  # type: ignore[arg-type]
    )
    assert not bubble_mountain.bubble_on_save_runway(
        _FakeState(x=120, y=395, pose=1, velocity_y=0)  # type: ignore[arg-type]
    )
    assert not bubble_mountain.bubble_on_save_runway(
        _FakeState(x=55, y=500, pose=1, velocity_y=0)  # type: ignore[arg-type]
    )


def test_bubble_save_runway_r15_double_wj_params() -> None:
    """R15 double-WJ timings that clear Phase D on human runway pin."""
    from super_metroid.routes.kpdr import bubble_mountain_params as P

    assert P.SAVE_RUN_FRAMES == 21
    assert P.SAVE_SPIN_FRAMES == 83
    assert P.SAVE_WJ_LEFT_A == 20
    assert P.SAVE_WJ_RIGHT_A == 8
    assert P.SAVE_WJ2_LEFT_A == 24
    assert P.SAVE_WJ2_RIGHT_A == 14
    assert P.SAVE_RUNWAY_FIRE_X[0] <= 27 <= P.SAVE_RUNWAY_FIRE_X[1]


def test_bubble_r16_lower_shelves_end_on_fire_solid() -> None:
    """R16 lower path targets y395 fire solid, not mid-iso float (105,370)."""
    from super_metroid.routes.kpdr import bubble_mountain_params as P

    assert P.LOWER_SHELVES[-1] == (50, 395)
    assert P.LOWER_SHELVES[-2][1] == 420
    # Fire window still covers human pin and pure land x~50–56.
    assert P.SAVE_RUNWAY_FIRE_X[0] <= 50 <= P.SAVE_RUNWAY_FIRE_X[1]
    assert P.SAVE_CLEAR_X_FRAMES >= 8
    assert P.SAVE_EDGE_LEFT_FRAMES >= 20


def test_bubble_fire_or_mid_pin_accepts_save_runway() -> None:
    """R16 lower stops on fire solid, not mid-iso float or lip."""
    fire = _FakeState(x=50, y=395, pose=9, velocity_y=0)
    mid = _FakeState(x=107, y=365, pose=25, velocity_y=0)
    lip = _FakeState(x=79, y=427, pose=2, velocity_y=0)
    assert bubble_mountain._bubble_fire_or_mid_pin(fire)  # type: ignore[arg-type]
    # Mid-iso float must NOT end lower — that falls onto lip.
    assert not bubble_mountain._bubble_fire_or_mid_pin(mid)  # type: ignore[arg-type]
    assert not bubble_mountain._bubble_fire_or_mid_pin(lip)  # type: ignore[arg-type]


def test_bubble_track_has_no_mid_loop_control_fields() -> None:
    """Mid budget/phase live in run_bubble_mid_loop locals, not BubbleTrack."""
    fields = set(bubble_mountain.BubbleTrack.__dataclass_fields__)
    assert "mid_phase" not in fields
    assert "mid_done" not in fields
    assert "mid_frames_used" not in fields
    assert "mid_i" not in fields
    assert "height_class" not in fields


def test_bubble_phase_d_near_top_slack() -> None:
    assert bubble_mountain.bubble_phase_d_near_top(
        _FakeState(x=320, y=180)  # type: ignore[arg-type]
    )
    # Within slack of top band.
    assert bubble_mountain.bubble_phase_d_near_top(
        _FakeState(x=270, y=230), slack=40  # type: ignore[arg-type]
    )
    assert not bubble_mountain.bubble_phase_d_near_top(
        _FakeState(x=200, y=300), slack=40  # type: ignore[arg-type]
    )
