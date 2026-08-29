from __future__ import annotations

from super_metroid.routes.kpdr import to_bat_cave
from super_metroid.routes.kpdr.rooms import ROOM_BAT_CAVE, ROOM_BUBBLE


class _FakeState:
    def __init__(
        self,
        *,
        room_id: int = ROOM_BUBBLE,
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
    assert to_bat_cave.bubble_phase_c_usable_right_contact(
        _FakeState(x=340, y=410)  # type: ignore[arg-type]
    )
    assert to_bat_cave.bubble_phase_c_usable_right_contact(
        _FakeState(x=300, y=430)  # type: ignore[arg-type]
    )
    # Below usable altitude (cavity floor thrash).
    assert not to_bat_cave.bubble_phase_c_usable_right_contact(
        _FakeState(x=340, y=484)  # type: ignore[arg-type]
    )
    # Too far left (lip peak class).
    assert not to_bat_cave.bubble_phase_c_usable_right_contact(
        _FakeState(x=150, y=260)  # type: ignore[arg-type]
    )
    # Outside Bubble.
    assert not to_bat_cave.bubble_phase_c_usable_right_contact(
        _FakeState(room_id=ROOM_BAT_CAVE, x=340, y=300)  # type: ignore[arg-type]
    )


def test_bubble_phase_d_top_band() -> None:
    assert to_bat_cave.bubble_phase_d_top_band(
        _FakeState(x=320, y=180)  # type: ignore[arg-type]
    )
    assert not to_bat_cave.bubble_phase_d_top_band(
        _FakeState(x=320, y=260)  # type: ignore[arg-type]
    )
    assert not to_bat_cave.bubble_phase_d_top_band(
        _FakeState(x=200, y=180)  # type: ignore[arg-type]
    )


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


def test_bubble_r19_fire_phase_geometry() -> None:
    """R19: Geruta phase classes A/B gate Phase D; known near-misses fail."""
    from super_metroid.routes import skills as prim

    # Class A (fullpure wait ~89–93)
    assert prim.bubble_fire_phase_geometry(120, 272, 196, 163)
    assert prim.bubble_fire_phase_geometry(118, 271, 197, 160)
    # Class B (fullpure wait ~233–235)
    assert prim.bubble_fire_phase_geometry(161, 274, 179, 187)
    # Known fails: wait 0, near-miss, live-adjacent pure false positive
    assert not prim.bubble_fire_phase_geometry(158, 155, 167, 126)
    assert not prim.bubble_fire_phase_geometry(127, 275, 188, 175)
    assert not prim.bubble_fire_phase_geometry(185, 105, 140, 157)


def test_bubble_fire_or_mid_pin_accepts_save_runway() -> None:
    """R16 lower stops on fire solid, not mid-iso float or lip."""
    fire = _FakeState(x=50, y=395, pose=9, velocity_y=0)
    mid = _FakeState(x=107, y=365, pose=25, velocity_y=0)
    lip = _FakeState(x=79, y=427, pose=2, velocity_y=0)
    assert to_bat_cave._bubble_fire_or_mid_pin(fire)  # type: ignore[arg-type]
    # Mid-iso float must NOT end lower — that falls onto lip.
    assert not to_bat_cave._bubble_fire_or_mid_pin(mid)  # type: ignore[arg-type]
    assert not to_bat_cave._bubble_fire_or_mid_pin(lip)  # type: ignore[arg-type]


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
