"""Unit tests for survival assist (no emulator required)."""

from __future__ import annotations

from zelda_i.assist import (
    UnlimitedHealthAssist,
    assist_phase_name,
    location_key,
)
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot, full_health_byte


def _snap(
    *,
    mode: int = PLAY_MODE,
    level: int = 0,
    health: int = 0x20,
    screen: int = 0x4A,
    link_x: int = 120,
    link_y: int = 141,
    room_item_id: int = 0,
) -> ZeldaSnapshot:
    return ZeldaSnapshot(
        mode=mode,
        level=level,
        screen=screen,
        next_screen=screen,
        link_x=link_x,
        link_y=link_y,
        facing=8,
        sword=1,
        bombs=0,
        rupees=0,
        keys=0,
        health=health,
        triforce=1,
        compass=0,
        dialog_timer=0,
        colliding_tile=0x26,
        room_item_id=room_item_id,
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
    assert full_health_byte(0x20) == 0x22
    assert full_health_byte(0x31) == 0x33
    assert full_health_byte(0x2F) == 0x22


def test_phase_names() -> None:
    assert assist_phase_name(_snap(mode=PLAY_MODE)) == "ordinary_gameplay"
    assert assist_phase_name(_snap(mode=17)) == "death"
    assert assist_phase_name(_snap(mode=18)) == "triforce_fanfare"
    assert assist_phase_name(_snap(mode=7)) == "transition"
    assert assist_phase_name(_snap(mode=11)) == "ordinary_gameplay"
    assert assist_phase_name(_snap(mode=9)) == "ordinary_gameplay"


def test_location_key() -> None:
    assert location_key(_snap(level=2, screen=0x5f)) == "L2:0x5f"
    assert location_key(_snap(level=0, screen=0x4A)) == "L0:0x4a"


def test_assist_refills_on_ordinary_play() -> None:
    data = _FakeData()
    assist = UnlimitedHealthAssist(enabled=True)
    assist.apply_snapshot(data, _snap(health=0x20), frame=10)
    assert data.values["health"] == 0x22
    assert data.values["heart_partial"] == 0xFF
    assert assist.telemetry.health.writes == 1
    assert assist.telemetry.health.restored == 2
    assert assist.telemetry.health.first_active_frame == 10
    assert assist.telemetry.progression_writes == 0
    assert assist.telemetry.capacity_writes == 0


def test_assist_refills_in_controllable_underworld_passage() -> None:
    data = _FakeData()
    assist = UnlimitedHealthAssist(enabled=True)
    assist.apply_snapshot(data, _snap(mode=9, health=0x20), frame=10)
    assert data.values == {"health": 0x22, "heart_partial": 0xFF}


def test_assist_skips_when_full() -> None:
    data = _FakeData()
    assist = UnlimitedHealthAssist(enabled=True)
    assist.apply_snapshot(data, _snap(health=0x22), frame=1)
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


def test_damage_telemetry_cumulative_and_heatmap() -> None:
    data = _FakeData()
    assist = UnlimitedHealthAssist(enabled=True)
    # Baseline full on OW screen 0x4A (3 containers, 3 hearts = 0x22).
    assist.apply_snapshot(data, _snap(health=0x22, screen=0x4A, level=0), frame=1)
    assert assist.telemetry.total_damage == 0

    # Drop 0x22 → 0x20 (2 whole-heart units) on same screen.
    data.values.clear()
    assist.apply_snapshot(data, _snap(health=0x20, screen=0x4A, level=0), frame=2)
    assert assist.telemetry.total_damage == 2
    assert assist.telemetry.damage_events == 1
    assert assist.telemetry.maximum_single_frame_damage == 2
    assert assist.telemetry.damage_by_location["L0:0x4a"] == 2
    assert data.values["health"] == 0x22

    # Second hit on dungeon room 0x5f: full→0x21 (1 unit).
    data.values.clear()
    assist.apply_snapshot(
        data,
        _snap(health=0x21, screen=0x5F, level=2, link_x=100, link_y=120),
        frame=10,
    )
    assert assist.telemetry.total_damage == 3
    assert assist.telemetry.damage_events == 2
    assert assist.telemetry.maximum_single_frame_damage == 2
    assert assist.telemetry.damage_by_location["L2:0x5f"] == 1

    rep = assist.report()
    assert rep["total_damage"] == 3
    assert rep["damage_events"] == 2
    locs = list(rep["damage_by_location"].keys())
    assert locs[0] == "L0:0x4a"
    assert len(rep["damage_samples"]) == 2
    assert rep["damage_samples"][1]["location"] == "L2:0x5f"
    assert rep["damage_samples"][1]["amount"] == 1


def test_damage_samples_capped() -> None:
    data = _FakeData()
    assist = UnlimitedHealthAssist(enabled=True)
    assist.apply_snapshot(data, _snap(health=0x22), frame=0)
    # Force many events; samples stay bounded, totals do not.
    for i in range(80):
        data.values.clear()
        assist.apply_snapshot(data, _snap(health=0x21), frame=i + 1)
    assert assist.telemetry.damage_events == 80
    assert assist.telemetry.total_damage == 80
    assert len(assist.telemetry.damage_samples) == 64


def test_assist_clamps_transient_container_jump() -> None:
    """A mid-play high-nibble spike must not lock extra hearts."""
    data = _FakeData()
    assist = UnlimitedHealthAssist(enabled=True)
    assist.apply_snapshot(data, _snap(health=0x22), frame=1)
    assert assist.telemetry.accepted_containers == 3

    data.values.clear()
    # Tape bug: fill INC from a 0xF write, or a transient 0x6F (3 → 7).
    assist.apply_snapshot(data, _snap(health=0x6F), frame=2)
    assert data.values["health"] == 0x22
    assert assist.telemetry.accepted_containers == 3
    assert assist.telemetry.container_clamps >= 1
    assert assist.telemetry.capacity_writes == 0


def test_assist_accepts_heart_container_plus_one() -> None:
    data = _FakeData()
    assist = UnlimitedHealthAssist(enabled=True)
    assist.apply_snapshot(data, _snap(health=0x22), frame=1)
    data.values.clear()
    assist.apply_snapshot(
        data, _snap(health=0x33, room_item_id=0x1A), frame=2
    )
    assert assist.telemetry.accepted_containers == 4
    assert data.values == {}
    assert assist.telemetry.container_clamps == 0


def test_assist_triforce_grants_only_one_container() -> None:
    data = _FakeData()
    assist = UnlimitedHealthAssist(enabled=True)
    assist.apply_snapshot(data, _snap(health=0x33), frame=1)
    assert assist.telemetry.accepted_containers == 4
    assist.apply_snapshot(data, _snap(mode=18, health=0x33), frame=2)
    data.values.clear()
    # Fanfare return with a glitched 7-container byte (user saw 7 after TF1).
    assist.apply_snapshot(data, _snap(health=0x6F), frame=3)
    assert assist.telemetry.accepted_containers == 5
    assert data.values["health"] == 0x44
    assert assist.telemetry.capacity_writes == 0


def test_health_byte_for_containers_roundtrip() -> None:
    from zelda_i.ram import health_byte_for_containers

    assert health_byte_for_containers(3) == 0x22
    assert health_byte_for_containers(5) == 0x44
    assert health_byte_for_containers(7) == 0x66
