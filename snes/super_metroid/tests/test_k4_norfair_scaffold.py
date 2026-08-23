from __future__ import annotations

import inspect

from super_metroid.progression import SPEED_GRAPH
from super_metroid.routes.kpdr import get_segment, k4_norfair, to_bat_cave


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

# Mainline K4 segment_id → controller (registry identity lock).
_K4_SEGMENTS = (
    ("business_to_frog_save", k4_norfair.play_business_to_frog_save),
    ("frog_save_to_business", k4_norfair.play_frog_save_to_business),
    ("business_to_cathedral_entrance", k4_norfair.play_business_to_cathedral_entrance),
    ("cathedral_entrance_to_cathedral", k4_norfair.play_cathedral_entrance_to_cathedral),
    ("cathedral_to_rising_tide", k4_norfair.play_cathedral_to_rising_tide),
    ("rising_tide_to_bubble", k4_norfair.play_rising_tide_to_bubble),
    ("bubble_to_bat_cave", k4_norfair.play_bubble_to_bat_cave),
    ("frog_save_to_speedway", k4_norfair.play_frog_save_to_speedway),
    ("bat_cave_to_speed_hall", k4_norfair.play_bat_cave_to_speed_hall),
    ("speed_hall_to_speed", k4_norfair.play_speed_hall_to_speed),
    ("speed_return_to_bubble", k4_norfair.play_speed_return_to_bubble),
    ("bubble_to_single_chamber", k4_norfair.play_bubble_to_single_chamber),
    ("single_to_double_chamber", k4_norfair.play_single_to_double_chamber),
    ("double_chamber_to_wave", k4_norfair.play_double_chamber_to_wave),
    ("speedway_to_farm", k4_norfair.play_speedway_to_farm),
    ("farm_to_bubble", k4_norfair.play_farm_to_bubble),
)


def test_k4_norfair_segments_registered() -> None:
    for segment_id, controller in _K4_SEGMENTS:
        assert get_segment(segment_id) is controller


def test_k4_norfair_graph_and_room_contract() -> None:
    """No-Speed first Bubble visit is Cathedral climb (not Frog Speedway)."""
    assert k4_norfair.ROOM_BUSINESS == 0xA7DE
    assert k4_norfair.ROOM_BUBBLE == 0xACB3
    assert k4_norfair.ROOM_BAT_CAVE == 0xB07A
    assert k4_norfair.ROOM_SPEED_HALL == 0xACF0
    assert k4_norfair.ROOM_SPEED == 0xAD1B
    assert k4_norfair.ROOM_SINGLE_CHAMBER == 0xAD5E
    assert k4_norfair.ROOM_DOUBLE_CHAMBER == 0xADAD
    assert k4_norfair.ROOM_WAVE == 0xADDE
    assert k4_norfair.WAVE_BEAM_MASK == 0x0001
    assert k4_norfair.ROOM_CATHEDRAL_ENTRANCE == 0xA7B3
    assert k4_norfair.ROOM_CATHEDRAL == 0xA788
    assert k4_norfair.ROOM_RISING_TIDE == 0xAFA3

    path = SPEED_GRAPH.shortest_path(
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
    sig_hop = inspect.signature(to_bat_cave.play_bubble_to_bat_cave)
    assert list(sig_hop.parameters) == ["session"]
    assert callable(to_bat_cave.play_bubble_climb_from_handoff)
    assert callable(to_bat_cave.play_bubble_from_top_door)
    assert callable(to_bat_cave.play_bubble_to_bat_cave_with_phase_capture)
    assert callable(to_bat_cave.capture_bubble_phase_c)


def test_bubble_true_ground_excludes_spin_apex() -> None:
    """R11: pose 25 + vy≈0 is not true ground (spin-apex false-land)."""
    spin_apex = _FakeState(x=150, y=300, pose=25, velocity_y=0)
    stand = _FakeState(x=150, y=300, pose=1, velocity_y=0)
    assert not to_bat_cave.bubble_is_true_ground(spin_apex)  # type: ignore[arg-type]
    assert to_bat_cave.bubble_is_true_ground(stand)  # type: ignore[arg-type]
    # Pose 2 / 9 / 10 also true ground when vy quiet.
    for pose in (2, 9, 10):
        assert to_bat_cave.bubble_is_true_ground(
            _FakeState(pose=pose, velocity_y=0)  # type: ignore[arg-type]
        )
    # Rising spin is not ground.
    assert not to_bat_cave.bubble_is_true_ground(
        _FakeState(pose=1, velocity_y=3)  # type: ignore[arg-type]
    )


def test_bubble_mid_iso_pin_accepts_pose_26() -> None:
    """Mid-iso handoff (pose 26 in pin band) still uses stand_pin class."""
    pin = _FakeState(x=98, y=374, pose=26, velocity_y=1)
    assert to_bat_cave.bubble_on_mid_iso_pin(pin)  # type: ignore[arg-type]
    # Outside x band is not mid-iso pin.
    assert not to_bat_cave.bubble_on_mid_iso_pin(
        _FakeState(x=200, y=374, pose=26, velocity_y=0)  # type: ignore[arg-type]
    )
    # True ground pose 1 in band also counts as pin class.
    assert to_bat_cave.bubble_on_mid_iso_pin(
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
        assert to_bat_cave.bubble_on_launch_lip(st)  # type: ignore[arg-type]
        assert to_bat_cave.bubble_is_stand_pin_pose(st)  # type: ignore[arg-type]
    # Spin apex is stand_pin but NOT true ground.
    apex = _FakeState(pose=25, **lip_xy)
    assert to_bat_cave.bubble_is_stand_pin_pose(apex)  # type: ignore[arg-type]
    assert not to_bat_cave.bubble_is_true_ground(apex)  # type: ignore[arg-type]
    # Off lip x/y or rising: not launch seat.
    assert not to_bat_cave.bubble_on_launch_lip(
        _FakeState(x=50, y=427, pose=1, velocity_y=0)  # type: ignore[arg-type]
    )
    assert not to_bat_cave.bubble_on_launch_lip(
        _FakeState(x=79, y=500, pose=1, velocity_y=0)  # type: ignore[arg-type]
    )
    assert not to_bat_cave.bubble_on_launch_lip(
        _FakeState(x=79, y=427, pose=1, velocity_y=3)  # type: ignore[arg-type]
    )


def test_bubble_save_runway_seat() -> None:
    """R14/R15: save-door outer runway (max-left human pin) is stand-pin band."""
    for pose in (1, 2, 9, 10, 25, 26):
        assert to_bat_cave.bubble_on_save_runway(
            _FakeState(x=55, y=395, pose=pose, velocity_y=0)  # type: ignore[arg-type]
        )
    # R15 max-left fire seat (~human x27).
    assert to_bat_cave.bubble_on_save_runway(
        _FakeState(x=27, y=395, pose=2, velocity_y=0)  # type: ignore[arg-type]
    )
    # Too close to Save door shell / too far right / wrong height.
    assert not to_bat_cave.bubble_on_save_runway(
        _FakeState(x=20, y=395, pose=1, velocity_y=0)  # type: ignore[arg-type]
    )
    assert not to_bat_cave.bubble_on_save_runway(
        _FakeState(x=120, y=395, pose=1, velocity_y=0)  # type: ignore[arg-type]
    )
    assert not to_bat_cave.bubble_on_save_runway(
        _FakeState(x=55, y=500, pose=1, velocity_y=0)  # type: ignore[arg-type]
    )


def test_bubble_save_runway_r15_double_wj_params() -> None:
    """R15 double-WJ timings that clear Phase D on human runway pin."""
    from super_metroid.routes.skills.policies import bubble_to_bat as P

    assert P.SAVE_RUN_FRAMES == 21
    assert P.SAVE_SPIN_FRAMES == 83
    assert P.SAVE_WJ_LEFT_A == 20
    assert P.SAVE_WJ_RIGHT_A == 8
    # R18 live pure Phase D timings (R15 human isolation was 24/14).
    assert P.SAVE_WJ2_LEFT_A == 14
    assert P.SAVE_WJ2_RIGHT_A == 6
    assert P.SAVE_WJ_FOLLOW == 40
    assert P.SAVE_ARM_PUMP is False
    assert P.SAVE_RUNWAY_FIRE_X[0] <= 27 <= P.SAVE_RUNWAY_FIRE_X[1]


def test_bubble_r17_stationary_clear_and_human_seat_params() -> None:
    """R17: stationary X clear + human seat band (not LEFT+X walk)."""
    from super_metroid.routes.skills.policies import bubble_to_bat as P
    from super_metroid.routes import skills as prim

    assert P.SAVE_STATIONARY_X >= 16
    assert P.SAVE_STATIONARY_FACE >= 4
    human_lo, human_hi = P.SAVE_HUMAN_SEAT_X
    assert human_lo <= 27 <= human_hi
    assert human_hi <= P.SAVE_RUNWAY_FIRE_X[1]
    assert callable(prim.bubble_stationary_missile_clear)
    assert callable(prim.bubble_double_walljump_r15)
    assert callable(prim.bubble_save_runway_open_loop_r15)
    assert callable(prim.bubble_walk_brake_to_x)


def test_bubble_walljump_skill_library() -> None:
    """R18: reusable WJ + runway skills — double required; physics knobs."""
    from super_metroid.routes.skills.policies import bubble_to_bat as P
    from super_metroid.routes import skills as prim

    assert prim.POSE_WALL_LATCH == 132
    assert prim.bubble_is_wall_latch(
        _FakeState(x=264, y=297, pose=132)  # type: ignore[arg-type]
    )
    assert not prim.bubble_is_wall_latch(
        _FakeState(x=264, y=297, pose=25)  # type: ignore[arg-type]
    )
    assert prim.bubble_is_knockback(
        _FakeState(x=40, y=395, pose=138)  # type: ignore[arg-type]
    )
    assert prim.bubble_wall_approach_band(
        _FakeState(x=260, y=280, pose=25)  # type: ignore[arg-type]
    )
    assert not prim.bubble_wall_approach_band(
        _FakeState(x=100, y=280, pose=25)  # type: ignore[arg-type]
    )
    # R15 double chain is exactly two timings; single is insufficient product.
    assert len(prim.R15_DOUBLE) == 2
    assert prim.R15_WJ1.into_frames == P.SAVE_WJ_LEFT_A
    assert prim.R15_WJ2.into_frames == P.SAVE_WJ2_LEFT_A
    assert P.WJ_LATCH_TIMEOUT >= 20
    assert P.WJ_INTO_X >= 240
    assert P.SAVE_RUN_FRAMES == 21
    assert P.SAVE_DASH_MAX_FRAMES == 32
    assert P.SAVE_ARM_PUMP is False  # R18 pure product
    assert P.SAVE_ARM_PUMP_PERIOD >= 1
    assert P.SAVE_WJ2_LEFT_A == 14
    assert P.SAVE_WJ2_RIGHT_A == 6
    assert P.DMG_BOOST_HOLD_FRAMES >= 5
    # Vertical constants documented for skill comments / experiments.
    assert prim.HIJUMP_WALLJUMP_VY0 > prim.REGULAR_WALLJUMP_VY0
    assert callable(prim.bubble_walljump_once)
    assert callable(prim.bubble_consecutive_walljumps)
    assert callable(prim.bubble_wait_wall_ready)
    assert callable(prim.bubble_wait_wall_latch)
    assert callable(prim.bubble_prepare_fire_run)
    assert callable(prim.bubble_runway_dash)
    assert callable(prim.bubble_spin_glide)
    assert callable(prim.bubble_save_runway_fire_recipe)
    assert callable(prim.bubble_period_walljump_climb)
    assert callable(prim.bubble_walljump_approach_coast)
    assert callable(prim.bubble_damage_boost_hold)
    assert callable(prim.bubble_seat_max_left_fire)
    assert callable(prim.bubble_walljump_second_left_wall)
    assert P.WJ2_LEFT_X <= 230
    assert P.WJ2_LEFT_SEEK >= 16
    # WJ2_LEFT_SEEK named group (into/flip promoted off skill defaults).
    assert P.WJ2_LEFT_INTO >= 4
    assert P.WJ2_LEFT_FLIP >= 8
    assert P.WJ2_LEFT_Y <= 220
    assert P.WJ2_LEFT_Y >= 160


def test_bubble_r19_fire_phase_geometry() -> None:
    """R19: Geruta phase classes A/B gate Phase D; wait params seat-safe."""
    from super_metroid.routes.skills.policies import bubble_to_bat as P
    from super_metroid.routes import skills as prim

    assert P.FIRE_PHASE_MAX_WAIT >= 200
    assert P.FIRE_PHASE_MAX_WAIT <= 400
    assert P.FIRE_PHASE_SLOTS == (4, 6)
    # Class A (fullpure wait ~89–93)
    assert prim.bubble_fire_phase_geometry(120, 272, 196, 163)
    assert prim.bubble_fire_phase_geometry(118, 271, 197, 160)
    assert prim.bubble_fire_phase_geometry(124, 275, 191, 171)
    # Class B (fullpure wait ~233–235)
    assert prim.bubble_fire_phase_geometry(161, 274, 179, 187)
    assert prim.bubble_fire_phase_geometry(163, 275, 177, 186)
    # Known fails: wait 0, near-miss, live-adjacent pure false positive
    assert not prim.bubble_fire_phase_geometry(158, 155, 167, 126)
    assert not prim.bubble_fire_phase_geometry(127, 275, 188, 175)
    assert not prim.bubble_fire_phase_geometry(135, 275, 181, 184)
    assert not prim.bubble_fire_phase_geometry(112, 264, 203, 148)
    assert not prim.bubble_fire_phase_geometry(185, 105, 140, 157)
    assert not prim.bubble_fire_phase_geometry(179, 113, 146, 155)
    assert callable(prim.bubble_wait_fire_phase)
    assert callable(prim.bubble_fire_phase_clear)
    assert callable(prim.bubble_read_enemy_slot)
    # Recipe accepts phase_wait kw (default True for product).
    sig = inspect.signature(prim.bubble_save_runway_fire_recipe)
    assert "phase_wait" in sig.parameters
    assert sig.parameters["phase_wait"].default is True


def test_bubble_r19_super_door_params() -> None:
    """R19 Phase E sticky right WJ + Super band (from Phase D pin)."""
    from super_metroid.routes.skills.policies import bubble_to_bat as P

    assert P.DOOR_SUPER_X >= 400
    assert P.DOOR_SUPER_Y <= 180
    assert P.DOOR_WJ_PERIOD == 10
    assert P.DOOR_WJ_INTO == 3
    assert P.DOOR_WJ_BOUNCE == 2
    assert P.DOOR_X_CAP >= 470
    assert P.DOOR_OUTER_X <= 400  # recover continuous fall at ~(446,383)
    assert P.DOOR_FRAMES >= 700
    # Reactive Phase E must start immediately from the earned top band.
    assert P.DOOR_CROUCH_FRAMES == 0
    human_lo, human_hi = P.SAVE_HUMAN_SEAT_X
    assert human_lo <= 27 <= human_hi
    assert human_hi <= 30  # continuous fire rejects x=32 seat


def test_bubble_r16_lower_shelves_end_on_fire_solid() -> None:
    """R16 lower path targets y395 fire solid, not mid-iso float (105,370)."""
    from super_metroid.routes.skills.policies import bubble_to_bat as P

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
    assert to_bat_cave._bubble_fire_or_mid_pin(fire)  # type: ignore[arg-type]
    # Mid-iso float must NOT end lower — that falls onto lip.
    assert not to_bat_cave._bubble_fire_or_mid_pin(mid)  # type: ignore[arg-type]
    assert not to_bat_cave._bubble_fire_or_mid_pin(lip)  # type: ignore[arg-type]


def test_bubble_track_has_no_mid_loop_control_fields() -> None:
    """Mid budget/phase live in run_bubble_mid_loop locals, not BubbleTrack."""
    fields = set(to_bat_cave.BubbleTrack.__dataclass_fields__)
    assert "mid_phase" not in fields
    assert "mid_done" not in fields
    assert "mid_frames_used" not in fields
    assert "mid_i" not in fields
    assert "height_class" not in fields


def test_bubble_phase_d_near_top_slack() -> None:
    assert to_bat_cave.bubble_phase_d_near_top(
        _FakeState(x=320, y=180)  # type: ignore[arg-type]
    )
    # Within slack of top band.
    assert to_bat_cave.bubble_phase_d_near_top(
        _FakeState(x=270, y=230), slack=40  # type: ignore[arg-type]
    )
    assert not to_bat_cave.bubble_phase_d_near_top(
        _FakeState(x=200, y=300), slack=40  # type: ignore[arg-type]
    )
