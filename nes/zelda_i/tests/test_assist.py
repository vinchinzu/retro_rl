"""Unit tests for survival assist (no emulator required)."""

from __future__ import annotations

from zelda_i.assist import UnlimitedHealthAssist, assist_phase_name
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot, full_health_byte


def _snap(
    *,
    mode: int = PLAY_MODE,
    level: int = 0,
    health: int = 0x20,
    screen: int = 0x4A,
) -> ZeldaSnapshot:
    return ZeldaSnapshot(
        mode=mode,
        level=level,
        screen=screen,
        next_screen=screen,
        link_x=120,
        link_y=141,
        facing=8,
        sword=1,
        bombs=0,
        rupees=0,
        keys=0,
        health=health,
        triforce=1,
        dialog_timer=0,
        colliding_tile=0x26,
        room_item_id=0,
        room_all_dead=0,
        room_obj_count=0,
        cur_opened_doors=0,
        open_doorway_mask=0,
        objects=(),
    )


class _FakeData:
    def __init__(self) -> None:
        self.values: dict[str, int] = {}

    def set_value(self, key: str, value: int) -> None:
        self.values[key] = int(value)


def test_full_health_byte_preserves_containers() -> None:
    assert full_health_byte(0x20) == 0x2F
    assert full_health_byte(0x31) == 0x3F
    assert full_health_byte(0x2F) == 0x2F


def test_phase_names() -> None:
    assert assist_phase_name(_snap(mode=PLAY_MODE)) == "ordinary_gameplay"
    assert assist_phase_name(_snap(mode=17)) == "death"
    assert assist_phase_name(_snap(mode=18)) == "triforce_fanfare"
    assert assist_phase_name(_snap(mode=7)) == "transition"
    assert assist_phase_name(_snap(mode=11)) == "ordinary_gameplay"


def test_assist_refills_on_ordinary_play() -> None:
    data = _FakeData()
    assist = UnlimitedHealthAssist(enabled=True)
    assist.apply_snapshot(data, _snap(health=0x20), frame=10)
    assert data.values["health"] == 0x2F
    assert assist.telemetry.health.writes == 1
    assert assist.telemetry.health.restored == 0xF - 0x0
    assert assist.telemetry.health.first_active_frame == 10
    assert assist.telemetry.progression_writes == 0
    assert assist.telemetry.capacity_writes == 0


def test_assist_skips_when_full() -> None:
    data = _FakeData()
    assist = UnlimitedHealthAssist(enabled=True)
    assist.apply_snapshot(data, _snap(health=0x2F), frame=1)
    assert data.values == {}
    assert assist.telemetry.health.writes == 0


def test_assist_suspends_on_death_and_counts() -> None:
    data = _FakeData()
    assist = UnlimitedHealthAssist(enabled=True)
    assist.apply_snapshot(data, _snap(mode=17, health=0x00), frame=5)
    assert data.values == {}
    assert assist.telemetry.deaths == 1
    assert assist.telemetry.suspended_phase_frames["death"] == 1


def test_assist_disabled_noop() -> None:
    data = _FakeData()
    assist = UnlimitedHealthAssist(enabled=False)
    assist.apply_snapshot(data, _snap(health=0x20), frame=1)
    assert data.values == {}
    rep = assist.report()
    assert rep["enabled"] is False


def test_damage_telemetry() -> None:
    data = _FakeData()
    assist = UnlimitedHealthAssist(enabled=True)
    # First frame establishes baseline filled=2 (0x22).
    assist.apply_snapshot(data, _snap(health=0x22), frame=1)
    # After refill, prev filled is 0xF; next damaged frame 0x20 → large "damage".
    data.values.clear()
    assist.apply_snapshot(data, _snap(health=0x20), frame=2)
    assert assist.telemetry.maximum_single_frame_damage >= 1
